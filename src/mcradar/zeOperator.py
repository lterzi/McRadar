# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import subprocess
import numpy as np
import xarray as xr
from glob import glob
from scipy import constants
from mcradar.tableOperator import creatRadarCols
import warnings
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors

debugging = False
onlyInterp = False

# TODO: this function should deal with the LUTs

def radarScat(sp, wl, K2):
#TODO check if K2 is for ice or liquid!
    """
    Calculates the single scattering radar quantities from the matrix values
    Parameters
    ----------
    sp: dataArray [n] superparticles containing backscattering matrix 
            and forward amplitude matrix information needed to compute
            spectral radar quantities
    wl: wavelength [mm]
    K2: Rayleigh dielectric factor |(m^2-1)/(m^2+2)|^2

    Returns
    -------
    reflect_h: super particle horizontal reflectivity[mm^6/m^3] (array[n])
    reflect_v: super particle vertical reflectivity[mm^6/m^3] (array[n])
    kdp: calculated kdp from each particle (array[n])
    rho_hv: correlation coefficient (array[n])
    """
    prefactor = 2*np.pi*wl**4/(np.pi**5*K2)
    
    
    reflect_hh = prefactor*(sp['Z11']+sp['Z22']+sp['Z12']+sp['Z21'])
    reflect_vv = prefactor*(sp['Z11']+sp['Z22']-sp['Z12']-sp['Z21'])
    kdp = 1e-3*(180.0/np.pi)*wl*(sp['S22r'] - sp['S11r'])

    reflect_hv = prefactor*(sp['Z11'] - sp['Z12'] + sp['Z21'] - sp['Z22'])
    #reflect_vh = prefactor*(sp.Z11 + sp.Z12 - sp.Z21 - sp.Z22).values
               
    # delta_hv np.arctan2(Z[2,3] - Z[3,2], -Z[2,2] - Z[3,3])
    #a = (Z[2,2] + Z[3,3])**2 + (Z[3,2] - Z[2,3])**2
    #b = (Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1])
    #c = (Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1])
    #rho_hv np.sqrt(a / (b*c))
    rho_hv = np.nan*np.ones_like(reflect_hh) # disable rho_hv for now
    #Ah = 4.343e-3 * 2 * scatterer.wavelength * sp.S22i.values # attenuation horizontal polarization
    #Av = 4.343e-3 * 2 * scatterer.wavelength * sp.S11i.values # attenuation vertical polarization

    #- test: calculate extinction: TODO: test Cextx that is given in DDA with this calculation.
    k = 2 * np.pi / (wl)
    cext_hh = sp['S22i']*4.0*np.pi/k
    cext_vv = sp['S11i']*4.0*np.pi/k
    
    return reflect_hh, reflect_vv, reflect_hv, kdp, rho_hv, cext_hh, cext_vv


def calcParticleZe(wls, elvs, mcTable,scatSet,beta,beta_std):#zeOperator
    """
    Calculates the horizontal and vertical reflectivity of 
    each superparticle from a given distribution of super 
    particles,in this case I just quickly wanted to change the function to deal with Monomers with the DDA LUT and use Tmatrix for the aggregates
    
    Parameters
    ----------
    wls: wavelength [mm] (iterable)
    elv: elevation angle [°] # TODO: maybe also this can become iterable
    mcTable: McSnow table returned from getMcSnowTable()
    scatSet: type of scattering calculations to use, choose between full and DDA
    orientational_avg: boolean to choose if the scattering properties are averaged over multiple orientations
    beta: mean canting angle of particle
    beta_std= standard deviation of canting angle of particle
    Returns 
    -------
    mcTable including the horizontal and vertical reflectivity
    of each super particle calculated for X, Ka and W band. The
    calculation is made separetely for aspect ratio < 1 and >=1.
    Kdp is also included. TODO spectral ldr and rho_hv
    """
    
    #calling the function to create output columns

    
    if scatSet['mode'] == 'orientational_avg':
        """
        #-- this option uses the output of the DDA calculations. 
        We are reading in all data, then selecting the corresponding wl, elevation.
        Then, you can choose how you want your points selected out of the table. 
        We have the option to select the n closest neighbours and average over them, 
        to define a radius in which all values are taken and averaged,
        or you can choose a nearest neighbour regression which chooses n closest neighbours and wheights the average with the inverse distance of the points. 
        """
        scatPoints={}
        # different DDA LUT for monomers and Aggregates. 
        mcTableCry = mcTable.where(mcTable['sNmono']==1,drop=True) # select only cry
        #print(mcTableCry)
        #print(mcTableCry)
        #print(mcTableCry.dia,mcTableCry.mTot,mcTableCry.sPhi)
        #mcTablePlate = mcTableCry.where(mcTableCry['sPhi']<=1,drop=True) # select only plates
        #mcTableColumn = mcTableCry.where(mcTableCry['sPhi']>1,drop=True) # select only needle 
        mcTableAgg = mcTable.where(mcTable['sNmono']>1,drop=True) # select only aggregates
        rimed = False # TODO: make that dependent on rime mass fraction!
        betas = np.random.normal(loc=beta, scale=beta_std, size=len(mcTableCry.dia))

        if scatSet['orientational_avg'] == False:
            DDA_data_cry = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_crystals.nc')
            DDA_data_agg = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_aggregates.nc')
        else:
            DDA_data_cry = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_crystals_withbeta.nc') #all_crystals #only_beta2.0000e-01_gamma1.5849e-04_
            DDA_data_agg = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_aggregates.nc')
            if 'D_max' in DDA_data_agg:
                DDA_data_agg = DDA_data_agg.rename({'D_max':'Dmax'})
        
        DDA_data_cry = DDA_data_cry.to_dataframe()
        DDA_data_agg = DDA_data_agg.to_dataframe()
        #print(DDA_data)
        # generate points to look up in the DDA LUT
        #fig,ax = plt.subplots(ncols=2,figsize=(10,5),constrained_layout=True)
        for wl in wls:
            wl_close = DDA_data_agg.iloc[(DDA_data_agg['wavelength']-wl).abs().argsort()].wavelength.values[0] # get closest wavelength to select from LUT
            DDA_wl_agg = DDA_data_agg[DDA_data_agg.wavelength==wl_close]

            wl_close = DDA_data_cry.iloc[(DDA_data_cry['wavelength']-wl).abs().argsort()].wavelength.values[0] # get closest wavelength to select from LUT
            DDA_wl_cry = DDA_data_cry[DDA_data_cry.wavelength==wl_close]
            
            for elv in elvs:
                #print(DDA_wl)
                el_close = DDA_wl_agg.iloc[(DDA_wl_agg['elevation']-elv).abs().argsort()].elevation.values[0] # get closest elevation to select from LUT
                DDA_elv_agg = DDA_wl_agg[DDA_wl_agg.elevation==el_close]

                el_close = DDA_wl_cry.iloc[(DDA_wl_cry['elevation']-elv).abs().argsort()].elevation.values[0] # get closest elevation to select from LUT
                DDA_elv_cry = DDA_wl_cry[DDA_wl_cry.elevation==el_close]
                
                if scatSet['orientational_avg'] == False:

                    if len(mcTableCry.sPhi)>0: # only possible if we have plate-like particles
                        pointsCry = np.array(list(zip(np.log10(DDA_elv_cry.D_max), np.log10(DDA_elv_cry.mass), np.log10(DDA_elv_cry.ar))))
                        mcSnowPointsCry = np.array(list(zip(np.log10(mcTableCry.dia), np.log10(mcTableCry.mTot), np.log10(mcTableCry.sPhi))))
                        # select now the points according to the defined method
                        # Fit the KNeighborsRegressor
                        if scatSet['selmode'] == 'KNeighborsRegressor':
                            knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                            # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                            scatPoints = {'Z11':10**knn.fit(pointsCry, np.log10(DDA_elv_cry.Z11.values)).predict(mcSnowPointsCry),
                                            'Z12':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                            'Z21':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                            'Z22':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                            
                                            'S11i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                            'S22i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                            'S11r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                            'S22r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22r.values))-1}
                        
                        elif scatSet['selmode'] == 'radius':
                            neigh = NearestNeighbors(radius=scatSet['radius'])
                            neigh.fit(pointsCry)
                            distances, indices = neigh.radius_neighbors(mcSnowPointsCry)
                            for idx in indices:
                                if len(idx) == 0:
                                    #warnings.warn('No points found in radius, please increase radius!!!')
                                    raise ValueError('No points found in radius, please increase radius!!!')# if we do not have any points wihtin the radius, we cannot calculate the scattering properties

                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_cry.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22r.values))-1}

                            
                        
                        elif scatSet['selMode'] == 'NearestNeighbors':
                            neigh = NearestNeighbors(n_neighbors=scatSet['n_neighbors'])
                            neigh.fit(pointsCry)
                            distances, indices = neigh.kneighbors(mcSnowPointsCry)
                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_cry.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22r.values))-1}
                                        
                        # calculate scattering properties from Matrix entries                
                        reflect_h,  reflect_v, reflect_hv, kdp_M1, rho_hv, cext_hh, cext_vv = radarScat(scatPoints, wl,scatSet['K2']) # calculate scattering properties from Matrix entries
                        mcTable['sZeH'].loc[elv,wl,mcTableCry.index] = reflect_h#points.ZeH
                        mcTable['sZeV'].loc[elv,wl,mcTableCry.index] = reflect_v#points.ZeV
                        mcTable['sZeHV'].loc[elv,wl,mcTableCry.index] = reflect_hv
                        mcTable['sKDP'].loc[elv,wl,mcTableCry.index] = kdp_M1#points.KDP
                        mcTable['sCextH'].loc[elv,wl,mcTableCry.index] = cext_hh
                        mcTable['sCextV'].loc[elv,wl,mcTableCry.index] = cext_vv

                        #if elv == 30 and wl == wls[2]:
                        #    plt.plot(mcTableCry.dia,10*np.log10(reflect_h/reflect_v),marker='.',ls='None',label='cry, zdr, '+str(elv))
                        #    plt.legend()
                        #    plt.show()
                        #    ax[3].plot(mcTableCry.dia,scatPoints['kdp'],marker='.',ls='None',label='aggs, kdp, '+str(elv))
                        #    ax[3].legend()
                    #- now for aggregates
                    if len(mcTableAgg.mTot)>0: # only if aggregates are here
                        # only used rimed particles if riming is True. TODO: make that dependent on riming fraction
                        if rimed:
                            DDA_elv_agg = DDA_elv_agg[DDA_elv_agg.rimeFlag==1]
                        else:
                            DDA_elv_agg = DDA_elv_agg[DDA_elv_agg.rimeFlag==0]
                        
                        pointsAgg = np.array(list(zip(np.log10(DDA_elv_agg.Dmax), np.log10(DDA_elv_agg.mass)))) # we need to differentiate here because for aggregates we are only selecting with mass and Dmax
                        mcSnowPointsAgg = np.array(list(zip(np.log10(mcTableAgg.dia), np.log10(mcTableAgg.mTot)))) 

                        if scatSet['selmode'] == 'KNeighborsRegressor':
                            knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                            # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                            #print(points)
                            #print(np.log10(DDA_elv.Z11.values))
                            #print(mcSnowPoints)
                            #quit()
                            scatPoints = {'cbck_h':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.c_bck_h.values)).predict(mcSnowPointsAgg)),#'Z11':10**knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z11.values)).predict(mcSnowPointsAgg),
                                        #  'Z12':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                        #  'Z21':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                        #  'Z22':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                        #  'S11i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                        #  'S22i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                        #  'S11r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                        #  'S22r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S22r.values))-1,
                                            
                                            'cbck_v':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.c_bck_v.values)).predict(mcSnowPointsAgg)),
                                            'cbck_hv':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.c_bck_hv.values+abs(np.min(DDA_elv_agg.c_bck_hv.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.c_bck_hv.values))-1,
                                            'cext_h':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.cext_hh.values+abs(np.min(DDA_elv_agg.cext_hh.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.cext_hh.values))-1,
                                            'cext_v':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.cext_vv.values+abs(np.min(DDA_elv_agg.cext_vv.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.cext_vv.values))-1,
                                            'kdp':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.kdp.values+abs(np.min(DDA_elv_agg.kdp.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.kdp.values))-1}
                        
                        elif scatSet['selmode'] == 'radius':
                            neigh = NearestNeighbors(radius=scatSet['radius'])
                            neigh.fit(pointsAgg)
                            distances, indices = neigh.radius_neighbors(mcSnowPointsAgg)
                            for idx in indices:
                                if len(idx) == 0:
                                    #warnings.warn('No points found in radius, please increase radius!!!')
                                    raise ValueError('No points found in radius, please increase radius!!!')# if we do not have any points wihtin the radius, we cannot calculate the scattering properties

                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_agg.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22r.values))-1}

                            
                        
                        elif scatSet['selMode'] == 'NearestNeighbors':
                            neigh = NearestNeighbors(n_neighbors=scatSet['n_neighbors'])
                            neigh.fit(pointsAgg)
                            distances, indices = neigh.kneighbors(mcSnowPointsAgg)
                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_agg.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22r.values))-1}
                                        
                        
                        #reflect_h,  reflect_v, reflect_hv, kdp_M1, rho_hv, cext_hh, cext_vv = radarScat(scatPoints, wl,scatSet['K2']) # get scattering properties from Matrix entries
                        #if elv == 90:
                        #   plt.plot(mcTableAgg.dia,10*np.log10(reflect_h),marker='.',ls='None',label='wl: '+str(wl)+' elv: '+str(elv))
                        #mcTable['sZeH'].loc[elv,wl,mcTableAgg.index] = reflect_h
                        #mcTable['sCextH'].loc[elv,wl,mcTableAgg.index] = cext_hh
                        #mcTable['sCextV'].loc[elv,wl,mcTableAgg.index] = cext_vv
                        #mcTable['sZeV'].loc[elv,wl,mcTableAgg.index] = reflect_v
                        #mcTable['sZeHV'].loc[elv,wl,mcTableAgg.index] = reflect_hv
                        #mcTable['sKDP'].loc[elv,wl,mcTableAgg.index] = kdp_M1
                        mcTable['sZeH'].loc[elv,wl,mcTableAgg.index] = scatPoints['cbck_h']
                        mcTable['sCextH'].loc[elv,wl,mcTableAgg.index] = scatPoints['cext_h']
                        mcTable['sCextV'].loc[elv,wl,mcTableAgg.index] = scatPoints['cext_v']
                        mcTable['sZeV'].loc[elv,wl,mcTableAgg.index] = scatPoints['cbck_v']
                        mcTable['sZeHV'].loc[elv,wl,mcTableAgg.index] = scatPoints['cbck_hv']
                        mcTable['sKDP'].loc[elv,wl,mcTableAgg.index] = scatPoints['kdp']
                else:# if we do orientational averaging:
                
                
                    #print(betas)
                    #quit()
                    pointsCry = np.array(list(zip(np.log10(DDA_elv_cry.Dmax), np.log10(DDA_elv_cry.mass), np.log10(DDA_elv_cry.ar),DDA_elv_cry.beta)))
                    mcSnowPointsCry = np.array(list(zip(np.log10(mcTableCry.dia), np.log10(mcTableCry.mTot), np.log10(mcTableCry.sPhi),betas)))
                    #scatPoints={}
                    
                    if len(mcTableCry.sPhi)>0: # only possible if we have plate-like particles
                        # select now the points according to the defined method
                        # Fit the KNeighborsRegressor
                        if scatSet['selmode'] == 'KNeighborsRegressor':
                            knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                            #print(max(np.log10(DDA_elv_cry.c_bck_hv.values+2*abs(np.min(DDA_elv_cry.c_bck_hv.values)))))
                            #quit()
                            #print(np.isfinite(DDA_elv_cry.c_bck_hv.values+2*abs(np.min(DDA_elv_cry.c_bck_hv.values))).all())
                            # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                            scatPoints = {'cbck_h':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.c_bck_h.values)).predict(mcSnowPointsCry)),#'Z11':10**knn.fit(pointsCry, np.log10(DDA_elv_cry.Z11.values)).predict(mcSnowPointsCry),
                                        # 'Z12':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                            #'Z21':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                            #'Z22':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                            #'S11i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                            #'S22i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                            #'S11r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                            #'S22r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22r.values))-1,
                                            
                                            'cbck_v':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.c_bck_v.values)).predict(mcSnowPointsCry)),
                                            'cbck_hv':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.c_bck_hv.values+abs(np.min(DDA_elv_cry.c_bck_hv.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.c_bck_hv.values))-1,
                                            'cext_h':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_hh.values+2*abs(np.min(DDA_elv_cry.cext_hh.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.cext_hh.values)),
                                            'cext_v':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_vv.values+2*abs(np.min(DDA_elv_cry.cext_vv.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.cext_vv.values)),
                                            'kdp':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.kdp.values+2*abs(np.min(DDA_elv_cry.kdp.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.kdp.values)),
                                            }
                        
                        elif scatSet['selmode'] == 'radius':
                            neigh = NearestNeighbors(radius=scatSet['radius'])
                            neigh.fit(pointsCry)
                            distances, indices = neigh.radius_neighbors(mcSnowPointsCry)
                            for idx in indices:
                                if len(idx) == 0:
                                    #warnings.warn('No points found in radius, please increase radius!!!')
                                    raise ValueError('No points found in radius, please increase radius!!!')# if we do not have any points wihtin the radius, we cannot calculate the scattering properties

                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_cry.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22r.values))-1,
                                        'cbck_h':10**np.array([(np.log10(DDA_elv_cry.c_bck_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_v':10**np.array([(np.log10(DDA_elv_cry.c_bck_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_hv':10**np.array([(np.log10(DDA_elv_cry.c_bck_hv.values+abs(np.min(DDA_elv_cry.c_bck_hv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.c_bck_hv.values))-1,
                                        'cext_h':10**np.array([(np.log10(DDA_elv_cry.cext_hh.values+abs(np.min(DDA_elv_cry.cext_hh.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.cext_hh.values))-1,
                                        'cext_v':10**np.array([(np.log10(DDA_elv_cry.cext_vv.values+abs(np.min(DDA_elv_cry.cext_vv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.cext_vv.values))-1,
                                        'kdp':10**np.array([(np.log10(DDA_elv_cry.kdp.values+abs(np.min(DDA_elv_cry.kdp.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.kdp.values))-1}
                            

                            
                        
                        elif scatSet['selMode'] == 'NearestNeighbors':
                            neigh = NearestNeighbors(n_neighbors=scatSet['n_neighbors'])
                            neigh.fit(pointsCry)
                            distances, indices = neigh.kneighbors(mcSnowPointsCry)
                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_cry.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22r.values))-1,
                                        'cbck_h':10**np.array([(np.log10(DDA_elv_cry.c_bck_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_v':10**np.array([(np.log10(DDA_elv_cry.c_bck_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_hv':10**np.array([(np.log10(DDA_elv_cry.c_bck_hv.values+abs(np.min(DDA_elv_cry.c_bck_hv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.c_bck_hv.values))-1,
                                        'cext_h':10**np.array([(np.log10(DDA_elv_cry.cext_hh.values+abs(np.min(DDA_elv_cry.cext_hh.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.cext_hh.values))-1,
                                        'cext_v':10**np.array([(np.log10(DDA_elv_cry.cext_vv.values+abs(np.min(DDA_elv_cry.cext_vv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.cext_vv.values))-1,
                                        'kdp':10**np.array([(np.log10(DDA_elv_cry.kdp.values+abs(np.min(DDA_elv_cry.kdp.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.kdp.values))-1}
                                        
                        # calculate scattering properties from Matrix entries                
                        #reflect_h,  reflect_v, reflect_hv, kdp_M1, rho_hv, cext_hh, cext_vv = radarScat(scatPoints, wl,scatSet['K2']) # calculate scattering properties from Matrix entries
                        
                        #mcTable['sZeH'].loc[elv,wl,mcTableCry.index] = reflect_h#points.ZeH
                        mcTable['sZeH'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_h']
                        mcTable['sCextH'].loc[elv,wl,mcTableCry.index] = scatPoints['cext_h']
                        mcTable['sCextV'].loc[elv,wl,mcTableCry.index] = scatPoints['cext_v']
                        mcTable['sZeV'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_v']
                        mcTable['sZeHV'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_hv']
                        mcTable['sKDP'].loc[elv,wl,mcTableCry.index] = scatPoints['kdp']
                        
                        #if elv == 90:# and wl==wls[2]:
                            #ax[0].plot(mcTableCry.dia,10*np.log10(scatPoints['cbck_h']),marker='.',ls='None',label='cry, h, '+str(wl))
                        #    ax[0].plot(mcTableCry.dia,scatPoints.kdp,marker='.',ls='None',label='cry, kdp, '+str(wl))
                        #    ax[0].legend()
                            #plt.plot(mcTableCry.dia,10*np.log10(scatPoints['cbck_v']),marker='.',ls='None',label='cry, v, '+str(wl))
                        #if elv == 30 and wl == wls[2]:
                        #    ax[2].plot(mcTableCry.dia,10*np.log10(scatPoints['cbck_h']/scatPoints['cbck_v']),marker='.',ls='None',label='cry, zdr, '+str(elv))
                        #    ax[2].legend()
                        #    ax[3].plot(mcTableCry.dia,scatPoints['kdp'],marker='.',ls='None',label='aggs, kdp, '+str(elv))
                        #    ax[3].legend()
                    
                    #- now for aggregates
                    if len(mcTableAgg.mTot)>0: # only if aggregates are here
                        # only used rimed particles if riming is True. TODO: make that dependent on riming fraction
                        if rimed:
                                DDA_elv_agg = DDA_elv_agg[DDA_elv_agg.rimeFlag==1]
                        else:
                            DDA_elv_agg = DDA_elv_agg[DDA_elv_agg.rimeFlag==0]
                        scatPoints={}
                        pointsAgg = np.array(list(zip(np.log10(DDA_elv_agg.Dmax), np.log10(DDA_elv_agg.mass)))) # we need to differentiate here because for aggregates we are only selecting with mass and Dmax xr.stack DDA_elv_agg.get('Dmax','mass'), vorher schon log10 berechnen
                        mcSnowPointsAgg = np.array(list(zip(np.log10(mcTableAgg.dia), np.log10(mcTableAgg.mTot)))) 
                        if scatSet['selmode'] == 'KNeighborsRegressor':
                            knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                            # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                            #print(points)
                            #print(np.log10(DDA_elv.Z11.values))
                            #print(mcSnowPoints)
                            #quit()
                            scatPoints = {'cbck_h':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.c_bck_h.values)).predict(mcSnowPointsAgg)),#'Z11':10**knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z11.values)).predict(mcSnowPointsAgg),
                                        #  'Z12':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                        #  'Z21':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                        #  'Z22':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                        #  'S11i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                        #  'S22i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                        #  'S11r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                        #  'S22r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S22r.values))-1,
                                            
                                            'cbck_v':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.c_bck_v.values)).predict(mcSnowPointsAgg)),
                                            'cbck_hv':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.c_bck_hv.values+abs(np.min(DDA_elv_agg.c_bck_hv.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.c_bck_hv.values))-1,
                                            'cext_h':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.cext_hh.values+abs(np.min(DDA_elv_agg.cext_hh.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.cext_hh.values))-1,
                                            'cext_v':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.cext_vv.values+abs(np.min(DDA_elv_agg.cext_vv.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.cext_vv.values))-1,
                                            'kdp':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.kdp.values+abs(np.min(DDA_elv_agg.kdp.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.kdp.values))-1}
                        
                        elif scatSet['selmode'] == 'radius':
                            neigh = NearestNeighbors(radius=scatSet['radius'])
                            neigh.fit(pointsAgg)
                            distances, indices = neigh.radius_neighbors(mcSnowPointsAgg)
                            for idx in indices:
                                if len(idx) == 0:
                                    #warnings.warn('No points found in radius, please increase radius!!!')
                                    raise ValueError('No points found in radius, please increase radius!!!')# if we do not have any points wihtin the radius, we cannot calculate the scattering properties

                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_agg.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22r.values))-1,
                                        'cbck_h':10**np.array([(np.log10(DDA_elv_agg.c_bck_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_v':10**np.array([(np.log10(DDA_elv_agg.c_bck_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_hv':10**np.array([(np.log10(DDA_elv_agg.c_bck_hv.values+abs(np.min(DDA_elv_agg.c_bck_hv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.c_bck_hv.values))-1,
                                        'cext_h':10**np.array([(np.log10(DDA_elv_agg.cext_hh.values+abs(np.min(DDA_elv_agg.cext_hh.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.cext_hh.values))-1,
                                        'cext_v':10**np.array([(np.log10(DDA_elv_agg.cext_vv.values+abs(np.min(DDA_elv_agg.cext_vv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.cext_vv.values))-1,
                                        'kdp':10**np.array([(np.log10(DDA_elv_agg.kdp.values+abs(np.min(DDA_elv_agg.kdp.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.kdp.values))-1}

                            
                        
                        elif scatSet['selMode'] == 'NearestNeighbors':
                            neigh = NearestNeighbors(n_neighbors=scatSet['n_neighbors'])
                            neigh.fit(pointsAgg)
                            distances, indices = neigh.kneighbors(mcSnowPointsAgg)
                            scatPoints={'Z11':10**np.array([np.log10(DDA_elv_agg.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'Z12':10**np.array([(np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                        'Z21':10**np.array([(np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                        'Z22':10**np.array([(np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                        'S11i':10**np.array([(np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                        'S22i':10**np.array([(np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                        'S11r':10**np.array([(np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                        'S22r':10**np.array([(np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22r.values))-1,
                                        'cbck_h':10**np.array([(np.log10(DDA_elv_agg.c_bck_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_v':10**np.array([(np.log10(DDA_elv_agg.c_bck_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                        'cbck_hv':10**np.array([(np.log10(DDA_elv_agg.c_bck_hv.values+abs(np.min(DDA_elv_agg.c_bck_hv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.c_bck_hv.values))-1,
                                        'cext_h':10**np.array([(np.log10(DDA_elv_agg.cext_hh.values+abs(np.min(DDA_elv_agg.cext_hh.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.cext_hh.values))-1,
                                        'cext_v':10**np.array([(np.log10(DDA_elv_agg.cext_vv.values+abs(np.min(DDA_elv_agg.cext_vv.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.cext_vv.values))-1,
                                        'kdp':10**np.array([(np.log10(DDA_elv_agg.kdp.values+abs(np.min(DDA_elv_agg.kdp.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.kdp.values))-1}
                                        
                        
                        mcTable['sZeH'].loc[elv,wl,mcTableAgg.index] = scatPoints['cbck_h']
                        mcTable['sCextH'].loc[elv,wl,mcTableAgg.index] = scatPoints['cext_h']
                        mcTable['sCextV'].loc[elv,wl,mcTableAgg.index] = scatPoints['cext_v']
                        mcTable['sZeV'].loc[elv,wl,mcTableAgg.index] = scatPoints['cbck_v']
                        mcTable['sZeHV'].loc[elv,wl,mcTableAgg.index] = scatPoints['cbck_hv']
                        mcTable['sKDP'].loc[elv,wl,mcTableAgg.index] = scatPoints['kdp']
    elif scatSet['mode'] == 'stochastic_aggs': #TODO: differentiate between aggregates of needles and aggregates of dendrites
        # now we use the new stochastic aggregates generated by Axel and Fabian. Stochastic means that the aggregates do not follow a fixed mD but are generated stochastically using the aggregation model. 
        # For this implementation we have generated a new DDA file with aggregates and random orientation (random elevation and azimuth). In order to save computation time we have decided to generate a large number of aggregates, where each aggregate only has one orientation. 
        # in this code we need to select the aggregates according to their elevation, the azimuth is random. I would use the same method as in mode "azi_avg" because the selection of n neighbors is the same for now. In the future, if we want to include some "wobbling" effects,
        # we can try to also include a variability in elevation  
        scatPoints={}
        # different DDA LUT for monomers and Aggregates. 
        mcTableCry = mcTable.where(mcTable['sNmono']==1,drop=True)#.to_dataframe() # select only cry
        mcTableAgg = mcTable.where(mcTable['sNmono']>1,drop=True)#.to_dataframe() # select only aggregates
        mcTableAgg = mcTableAgg.where(~np.isnan(mcTableAgg.dia),drop=True) # remove NaN values
        mcTableAgg = mcTableAgg.where(~np.isnan(mcTableAgg.mTot),drop=True)
        mcTableAgg = mcTableAgg.where(~np.isnan(mcTableAgg.vel),drop=True)
        #print(mcTableAgg)
        #mcTableAgg = mcTableAgg.where(~np.isnan(mcTableAgg.dia))
        #print(mcTableAgg)
        #quit()
        DDA_data_cry = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_crystals.nc')
        DDA_data_agg = xr.open_dataset(scatSet['lutPath']+'stochastic_aggregates1.nc')
        
        if 'D_max' in DDA_data_agg:
            DDA_data_agg = DDA_data_agg.rename({'D_max':'Dmax'})
        if 'D_max' in DDA_data_cry:
            DDA_data_cry = DDA_data_cry.rename({'D_max':'Dmax'})
        if 'Nmonomers' in DDA_data_agg:
            DDA_data_agg = DDA_data_agg.rename({'Nmonomers':'Nmono'})
        DDA_data_cry = DDA_data_cry.to_dataframe()
        DDA_data_agg = DDA_data_agg.to_dataframe()
        for wl in wls:
            # select correct wavelength:
            wl_close = DDA_data_agg.iloc[(DDA_data_agg['wavelength']-wl).abs().argsort()].wavelength.values[0] # get closest wavelength to select from LUT
            DDA_wl_agg = DDA_data_agg[DDA_data_agg.wavelength==wl_close]
            

            wl_close = DDA_data_cry.iloc[(DDA_data_cry['wavelength']-wl).abs().argsort()].wavelength.values[0] # get closest wavelength to select from LUT
            DDA_wl_cry = DDA_data_cry[DDA_data_cry.wavelength==wl_close]
            
            for elv in elvs:
                #print(DDA_wl)
                # select correct elevation:
                el_close = DDA_wl_agg.iloc[(DDA_wl_agg['elevation']-elv).abs().argsort()].elevation.values[0] # get closest elevation to select from LUT
                DDA_elv_agg = DDA_wl_agg[DDA_wl_agg.elevation==el_close]

                el_close = DDA_wl_cry.iloc[(DDA_wl_cry['elevation']-elv).abs().argsort()].elevation.values[0] # get closest elevation to select from LUT
                DDA_elv_cry = DDA_wl_cry[DDA_wl_cry.elevation==el_close]
                
                # now lets select points for crystals
                if len(mcTableCry.mTot)>0: # only possible if we have crystals
                    pointsCry = np.array(list(zip(np.log10(DDA_elv_cry.Dmax), np.log10(DDA_elv_cry.mass), np.log10(DDA_elv_cry.ar)))) # these are the points from the DDA file
                    mcSnowPointsCry = np.array(list(zip(np.log10(mcTableCry.dia), np.log10(mcTableCry.mTot), np.log10(mcTableCry.sPhi)))) # we need to find the closest points in the DDA file to the points in the McSnow file
                    # select now the points according to the defined method
                    # Fit the KNeighborsRegressor
                    if scatSet['selmode'] == 'KNeighborsRegressor':
                        knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                        # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                        scatPoints = {'Z11':10**knn.fit(pointsCry, np.log10(DDA_elv_cry.Z11.values)).predict(mcSnowPointsCry),
                                        'Z12':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z12.values+2*abs(np.min(DDA_elv_cry.Z12.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.Z12.values)),
                                        'Z21':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z21.values+2*abs(np.min(DDA_elv_cry.Z21.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.Z21.values)),
                                        'Z22':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z22.values+2*abs(np.min(DDA_elv_cry.Z22.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.Z22.values)),
                                        
                                        'S11i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11i.values+2*abs(np.min(DDA_elv_cry.S11i.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.S11i.values)),
                                        'S22i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22i.values+2*abs(np.min(DDA_elv_cry.S22i.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.S22i.values)),
                                        'S11r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11r.values+2*abs(np.min(DDA_elv_cry.S11r.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.S11r.values)),
                                        'S22r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22r.values+2*abs(np.min(DDA_elv_cry.S22r.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.S22r.values))}
                    
                    # calculate scattering properties from Matrix entries                
                    reflect_h,  reflect_v, reflect_hv, kdp_M1, rho_hv, cext_hh, cext_vv = radarScat(scatPoints, wl,scatSet['K2']) # calculate scattering properties from Matrix entries
                    mcTable['sZeH'].loc[elv,wl,mcTableCry.index] = reflect_h#points.ZeH
                    mcTable['sZeV'].loc[elv,wl,mcTableCry.index] = reflect_v#points.ZeV
                    mcTable['sZeHV'].loc[elv,wl,mcTableCry.index] = reflect_hv
                    mcTable['sKDP'].loc[elv,wl,mcTableCry.index] = kdp_M1#points.KDP
                    mcTable['sCextH'].loc[elv,wl,mcTableCry.index] = cext_hh
                    mcTable['sCextV'].loc[elv,wl,mcTableCry.index] = cext_vv

                if len(mcTableAgg.mTot)>0: # only possible if we have crystals
                    
                    # select now the points according to the defined method
                    # Fit the KNeighborsRegressor
                    if scatSet['selmode'] == 'KNeighborsRegressor':
                        pointsAgg = np.array(list(zip(np.log10(DDA_elv_agg.Dmax), np.log10(DDA_elv_agg.mass)))) # these are the points from the DDA file
                        mcSnowPointsAgg = np.array(list(zip(np.log10(mcTableAgg.dia), np.log10(mcTableAgg.mTot)))) # we need to find the closest points in the DDA file to the points in the McSnow file
                        knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                        # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                        scatPoints = {'Z11':10**knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z11.values)).predict(mcSnowPointsAgg),
                                        'Z12':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z12.values+2*abs(np.min(DDA_elv_agg.Z12.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.Z12.values)),
                                        'Z21':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z21.values+2*abs(np.min(DDA_elv_agg.Z21.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.Z21.values)),
                                        'Z22':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z22.values+2*abs(np.min(DDA_elv_agg.Z22.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.Z22.values)),
                                        
                                        'S11i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S1i.values+2*abs(np.min(DDA_elv_agg.S1i.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.S1i.values)),
                                        'S22i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S2i.values+2*abs(np.min(DDA_elv_agg.S2i.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.S2i.values)),
                                        'S11r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S1r.values+2*abs(np.min(DDA_elv_agg.S1r.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.S1r.values)),
                                        'S22r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S2r.values+2*abs(np.min(DDA_elv_agg.S2r.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.S2r.values))}
                    
                    # calculate scattering properties from Matrix entries                
                    #plt.semilogx(mcTableAgg.mTot,10*np.log10(reflect_h),marker='.',ls='None')
                    #plt.semilogx(mcTableAgg.mTot,10*np.log10(reflect_v),marker='.',ls='None')
                    #plt.show()
                    #TODO do I need to make everything in mm2? Or is that already done?
                    #plt.savefig('test.png')
                    #quit()
                    #print(reflect_h)
                    #print(mcTable)
                    #print(mcTableAgg.index)
                    if scatSet['selmode'] == 'stochastic_sampling':
                        print(mcTableAgg.dia, mcTableAgg.sMult)

                        quit()
                        #knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                        # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                        #scatPoints = {'Z11':10**knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z11.values)).predict(mcSnowPointsAgg),
                        #                'Z12':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z12.values+2*abs(np.min(DDA_elv_agg.Z12.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.Z12.values)),
                        #                'Z21':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z21.values+2*abs(np.min(DDA_elv_agg.Z21.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.Z21.values)),
                        #                'Z22':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z22.values+2*abs(np.min(DDA_elv_agg.Z22.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.Z22.values)),
                                        
                        #                'S11i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S1i.values+2*abs(np.min(DDA_elv_agg.S1i.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.S1i.values)),
                        #                'S22i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S2i.values+2*abs(np.min(DDA_elv_agg.S2i.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.S2i.values)),
                        #                'S11r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S1r.values+2*abs(np.min(DDA_elv_agg.S1r.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.S1r.values)),
                        #                'S22r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S2r.values+2*abs(np.min(DDA_elv_agg.S2r.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.S2r.values))}
                    
                    # calculate scattering properties from Matrix entries                


                    reflect_h,  reflect_v, reflect_hv, kdp_M1, rho_hv, cext_hh, cext_vv = radarScat(scatPoints, wl,scatSet['K2']) # calculate scattering properties from Matrix entries
                    mcTable['sZeH'].loc[elv,wl,mcTableAgg.index] = reflect_h#points.ZeH
                    mcTable['sZeV'].loc[elv,wl,mcTableAgg.index] = reflect_v#points.ZeV
                    mcTable['sZeHV'].loc[elv,wl,mcTableAgg.index] = reflect_hv
                    mcTable['sKDP'].loc[elv,wl,mcTableAgg.index] = kdp_M1#points.KDP
                    mcTable['sCextH'].loc[elv,wl,mcTableAgg.index] = cext_hh
                    mcTable['sCextV'].loc[elv,wl,mcTableAgg.index] = cext_vv



    return mcTable


