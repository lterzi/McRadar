# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license

import numpy as np
import xarray as xr
from scipy import constants
from mcradar.tableOperator import creatRadarCols
import warnings
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors

debugging = False
onlyInterp = False


def radarScat(sp, wl, K2):
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
    prefactor = wl**4/(np.pi**5*K2)
    
    
    reflect_vv = prefactor*(sp['Z11']+sp['Z22']+sp['Z12']+sp['Z21'])
    reflect_hh = prefactor*(sp['Z11']+sp['Z22']-sp['Z12']-sp['Z21'])
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


def calcParticleZe(wls, elvs,mcTable, mcTableAgg,mcTableCry,scatSet,beta,beta_std):#zeOperator
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

    
    #if scatSet['mode'] == 'azimuthal_random_orientation':
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
    if scatSet['mode'] == 'fixed_orientation':
        DDA_data_cry = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_crystals.nc')
        DDA_data_agg = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_aggregates.nc')
        if 'D_max' in DDA_data_agg:
            DDA_data_agg = DDA_data_agg.rename({'D_max':'Dmax'})
        if 'D_max' in DDA_data_cry:
            DDA_data_cry = DDA_data_cry.rename({'D_max':'Dmax'})
    elif scatSet['mode']== 'azimuthal_random_orientation':
        betasCry = np.random.normal(loc=beta, scale=beta_std, size=len(mcTableCry.dia))
        betasAgg = np.random.normal(loc=beta, scale=beta_std, size=len(mcTableAgg.dia))
        DDA_data_cry = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_crystals_withbetanew_kdp1.nc') #all_crystals #only_beta2.0000e-01_gamma1.5849e-04_
        DDA_data_agg = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_aggregates_withbeta.nc') #scattering_properties_all_aggregates.nc
        if 'D_max' in DDA_data_agg:
            DDA_data_agg = DDA_data_agg.rename({'D_max':'Dmax'})
        if 'D_max' in DDA_data_cry:
            DDA_data_cry = DDA_data_cry.rename({'D_max':'Dmax'})
    else:
        raise ValueError('Unknown mode: '+scatSet['mode']+'! Please choose between "fixed_orientation" and "azimuthal_random_orientation"!')

    DDA_data_cry = DDA_data_cry.to_dataframe()
    DDA_data_agg = DDA_data_agg.to_dataframe()
    # generate points to look up in the DDA LUT
    for i,wl in enumerate(wls):
        wl_close = DDA_data_agg.iloc[(DDA_data_agg['wavelength']-wl).abs().argsort()].wavelength.values[0] # get closest wavelength to select from LUT
        DDA_wl_agg = DDA_data_agg[DDA_data_agg.wavelength==wl_close]

        wl_close = DDA_data_cry.iloc[(DDA_data_cry['wavelength']-wl).abs().argsort()].wavelength.values[0] # get closest wavelength to select from LUT
        DDA_wl_cry = DDA_data_cry[DDA_data_cry.wavelength==wl_close]
        
        for elv in elvs:
            el_close = DDA_wl_agg.iloc[(DDA_wl_agg['elevation']-elv).abs().argsort()].elevation.values[0] # get closest elevation to select from LUT
            DDA_elv_agg = DDA_wl_agg[DDA_wl_agg.elevation==el_close]
            
            el_close = DDA_wl_cry.iloc[(DDA_wl_cry['elevation']-elv).abs().argsort()].elevation.values[0] # get closest elevation to select from LUT
            DDA_elv_cry = DDA_wl_cry[DDA_wl_cry.elevation==el_close]
            DDA_elv_cry = DDA_elv_cry[DDA_elv_cry.kdp<1]
            #print(len(mcTableCry.sPhi),len(mcTableCry.sPhi)>0)
            #print(len(mcTableAgg.mTot))
            if len(mcTableCry.sPhi)>0: # only possible if we have plate-like particles
                if scatSet['mode'] == 'azimuthal_random_orientation':
                    pointsCry = np.array(list(zip(np.log10(DDA_elv_cry.Dmax), np.log10(DDA_elv_cry.mass), np.log10(DDA_elv_cry.ar),DDA_elv_cry.beta)))
                    mcSnowPointsCry = np.array(list(zip(np.log10(mcTableCry.dia), np.log10(mcTableCry.mTot), np.log10(mcTableCry.sPhi),betasCry)))
                elif scatSet['mode']== 'fixed_orientation':
                    pointsCry = np.array(list(zip(np.log10(DDA_elv_cry.Dmax), np.log10(DDA_elv_cry.mass), np.log10(DDA_elv_cry.ar))))
                    mcSnowPointsCry = np.array(list(zip(np.log10(mcTableCry.dia), np.log10(mcTableCry.mTot), np.log10(mcTableCry.sPhi))))
            
                # select now the points according to the defined method
                # Fit the KNeighborsRegressor
                if scatSet['selmode'] == 'KNeighborsRegressor':
                    knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                    #print(np.isnan(DDA_elv_cry).any())
                    # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                    scatPoints = {'reflect_hh':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_h.values)).predict(mcSnowPointsCry)),#'Z11':10**knn.fit(pointsCry, np.log10(DDA_elv_cry.Z11.values)).predict(mcSnowPointsCry),
                                    'reflect_vv':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_v.values)).predict(mcSnowPointsCry)),
                                    'reflect_hv':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_hv.values+abs(np.max(DDA_elv_cry.Ze_hv.values)))).predict(mcSnowPointsCry))-abs(np.max(DDA_elv_cry.Ze_hv.values)),
                                    'cext_h':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_hh.values+2*abs(np.min(DDA_elv_cry.cext_hh.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.cext_hh.values)),
                                    'cext_v':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_vv.values+2*abs(np.min(DDA_elv_cry.cext_vv.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.cext_vv.values)),
                                    'kdp':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.kdp.values+2*abs(np.min(DDA_elv_cry.kdp.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.kdp.values)),

                                    #'Z11':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z11.values+abs(np.min(DDA_elv_cry.Z11.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z11.values))-1,
                                    #'Z12':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                    #'Z21':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                    #'Z22':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                    #'S11i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                    #'S22i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                    #'S11r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                    #'S22r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22r.values))-1,
                                    
                                    }
                
                elif scatSet['selmode'] == 'radius':
                    neigh = NearestNeighbors(radius=scatSet['radius'])
                    neigh.fit(pointsCry)
                    distances, indices = neigh.radius_neighbors(mcSnowPointsCry)
                    for idx in indices:
                        if len(idx) == 0:
                            #warnings.warn('No points found in radius, please increase radius!!!')
                            raise ValueError('No points found in radius, please increase radius!!!')# if we do not have any points wihtin the radius, we cannot calculate the scattering properties

                    scatPoints={'reflect_hh':10**np.array([(np.log10(DDA_elv_cry.Ze_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                'reflect_vv':10**np.array([(np.log10(DDA_elv_cry.Ze_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                'reflect_hv':10**np.array([(np.log10(DDA_elv_cry.Ze_hv.values+abs(np.max(DDA_elv_cry.Ze_hv.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.max(DDA_elv_cry.Ze_hv.values)),
                                'cext_h':10**np.array([(np.log10(DDA_elv_cry.cext_hh.values+2*abs(np.min(DDA_elv_cry.cext_hh.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_cry.cext_hh.values)),
                                'cext_v':10**np.array([(np.log10(DDA_elv_cry.cext_vv.values+2*abs(np.min(DDA_elv_cry.cext_vv.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_cry.cext_vv.values)),
                                'kdp':10**np.array([(np.log10(DDA_elv_cry.kdp.values+2*abs(np.min(DDA_elv_cry.kdp.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_cry.kdp.values)),
                        
                                # 'Z11':10**np.array([np.log10(DDA_elv_cry.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                # 'Z12':10**np.array([(np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                # 'Z21':10**np.array([(np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                # 'Z22':10**np.array([(np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                # 'S11i':10**np.array([(np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                # 'S22i':10**np.array([(np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                # 'S11r':10**np.array([(np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                # 'S22r':10**np.array([(np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22r.values))-1,
                                }
                    

                    
                
                elif scatSet['selMode'] == 'NearestNeighbors':
                    neigh = NearestNeighbors(n_neighbors=scatSet['n_neighbors'])
                    neigh.fit(pointsCry)
                    distances, indices = neigh.kneighbors(mcSnowPointsCry)
                    scatPoints={'reflect_hh':10**np.array([(np.log10(DDA_elv_cry.Ze_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                'reflect_vv':10**np.array([(np.log10(DDA_elv_cry.Ze_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                'reflect_hv':10**np.array([(np.log10(DDA_elv_cry.Ze_hv.values+abs(np.max(DDA_elv_cry.Ze_hv.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.max(DDA_elv_cry.Ze_hv.values)),
                                'cext_h':10**np.array([(np.log10(DDA_elv_cry.cext_hh.values+2*abs(np.min(DDA_elv_cry.cext_hh.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_cry.cext_hh.values)),
                                'cext_v':10**np.array([(np.log10(DDA_elv_cry.cext_vv.values+2*abs(np.min(DDA_elv_cry.cext_vv.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_cry.cext_vv.values)),
                                'kdp':10**np.array([(np.log10(DDA_elv_cry.kdp.values+2*abs(np.min(DDA_elv_cry.kdp.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_cry.kdp.values)),

                                # 'Z11':10**np.array([np.log10(DDA_elv_cry.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                # 'Z12':10**np.array([(np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z12.values))-1,
                                # 'Z21':10**np.array([(np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z21.values))-1,
                                # 'Z22':10**np.array([(np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.Z22.values))-1,
                                # 'S11i':10**np.array([(np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11i.values))-1,
                                # 'S22i':10**np.array([(np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22i.values))-1,
                                # 'S11r':10**np.array([(np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S11r.values))-1,
                                # 'S22r':10**np.array([(np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_cry.S22r.values))-1,
                                }
                                
                # calculate scattering properties from Matrix entries                
                #reflect_h,  reflect_v, reflect_hv, kdp_M1, rho_hv, cext_hh, cext_vv = radarScat(scatPoints, wl,scatSet['K2']) # calculate scattering properties from Matrix entries
                
                #mcTable['sZeH'].loc[elv,wl,mcTableCry.index] = reflect_h#points.ZeH
                #print(mcTableCry.index,mcTable.index)
                #try:
                #.loglog(mcTableCry.dia, scatPoints['cbck_h'],'.',label='cbck_h')
                #plt.savefig('test_cbck_h.png')
                #quit()
                mcTable['sZeH'].loc[elv,wl,mcTableCry.index] = scatPoints['reflect_hh'] #reflect_h #scatPoints['cbck_h']
                mcTable['sZeV'].loc[elv,wl,mcTableCry.index] = scatPoints['reflect_vv'] #reflect_v #scatPoints['cbck_v']
                mcTable['sZeHV'].loc[elv,wl,mcTableCry.index] = scatPoints['reflect_hv'] #reflect_hv #scatPoints['cbck_hv']
                mcTable['sKDP'].loc[elv,wl,mcTableCry.index] = scatPoints['kdp'] #kdp_M1 #scatPoints['kdp']
                mcTable['sCextH'].loc[elv,wl,mcTableCry.index] = scatPoints['cext_h'] #cext_hh #scatPoints['cext_h']
                mcTable['sCextV'].loc[elv,wl,mcTableCry.index] = scatPoints['cext_v'] #cext_vv #scatPoints['cext_v']
                # if i == 2 and elv == 90:
                #     plt.semilogx(mcTableCry.dia, 10*np.log10(scatPoints['reflect_hh']),'.',label='cbck_h')
                #     plt.show()
                #except KeyError as e:
                    #warnings.warn('KeyError: '+str(e)+'! Please check if the keys are correct in the mcTable!')
                #    print('Keys in mcTableCry:', mcTableCry.index)
                #    print('Keys in mcTable:', mcTable.index)
                #    raise e
                
                
            
            #- now for aggregates
            if len(mcTableAgg.mTot)>0: # only if aggregates are here
                scatPoints={}
                if scatSet['mode'] == 'azimuthal_random_orientation':
                    pointsAgg = np.array(list(zip(np.log10(DDA_elv_agg.Dmax), np.log10(DDA_elv_agg.mass),DDA_elv_agg.beta)))
                    #print(np.isnan(DDA_elv_agg.Ze_h.values).any())
                    mcSnowPointsAgg = np.array(list(zip(np.log10(mcTableAgg.dia), np.log10(mcTableAgg.mTot), betasAgg)))
                    #print(mcSnowPointsAgg.shape, mcTableAgg.dia.shape, mcTableAgg.mTot.shape)
                elif scatSet['mode']== 'fixed_orientation':
                    pointsAgg = np.array(list(zip(np.log10(DDA_elv_agg.Dmax), np.log10(DDA_elv_agg.mass)))) # we need to differentiate here because for aggregates we are only selecting with mass and Dmax
                    mcSnowPointsAgg = np.array(list(zip(np.log10(mcTableAgg.dia), np.log10(mcTableAgg.mTot)))) 
                if scatSet['selmode'] == 'KNeighborsRegressor':
                    knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                    # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                    scatPoints = {'reflect_hh':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Ze_h.values)).predict(mcSnowPointsAgg)),#'Z11':10**knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z11.values)).predict(mcSnowPointsAgg),
                                    'reflect_vv':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Ze_v.values)).predict(mcSnowPointsAgg)),
                                    'reflect_hv':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Ze_hv.values+abs(np.max(DDA_elv_agg.Ze_hv.values)))).predict(mcSnowPointsAgg))-abs(np.max(DDA_elv_agg.Ze_hv.values)),
                                    'cext_h':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.cext_hh.values)).predict(mcSnowPointsAgg)),
                                    'cext_v':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.cext_vv.values+2*abs(np.min(DDA_elv_agg.cext_vv.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.cext_vv.values)),
                                    'kdp':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.kdp.values+2*abs(np.min(DDA_elv_agg.kdp.values)))).predict(mcSnowPointsAgg))-2*abs(np.min(DDA_elv_agg.kdp.values)),
                                 #'Z11':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z11.values+abs(np.min(DDA_elv_agg.Z11.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z11.values))-1,
                                 #'Z12':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                 #'Z21':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                 #'Z22':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                 #'S11i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                 #'S22i':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                 #'S11r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                 #'S22r':10**(knn.fit(pointsAgg, np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1)).predict(mcSnowPointsAgg))-abs(np.min(DDA_elv_agg.S22r.values))-1,
                                    }
                
                elif scatSet['selmode'] == 'radius':
                    neigh = NearestNeighbors(radius=scatSet['radius'])
                    neigh.fit(pointsAgg)
                    distances, indices = neigh.radius_neighbors(mcSnowPointsAgg)
                    for idx in indices:
                        if len(idx) == 0:
                            #warnings.warn('No points found in radius, please increase radius!!!')
                            raise ValueError('No points found in radius, please increase radius!!!')# if we do not have any points wihtin the radius, we cannot calculate the scattering properties

                    scatPoints={'reflect_hh':10**np.array([(np.log10(DDA_elv_agg.Ze_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                'reflect_vv':10**np.array([(np.log10(DDA_elv_agg.Ze_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                'reflect_hv':10**np.array([(np.log10(DDA_elv_agg.Ze_hv.values+2*abs(np.min(DDA_elv_agg.Ze_hv.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_agg.Ze_hv.values)),
                                'cext_h':10**np.array([(np.log10(DDA_elv_agg.cext_hh.values+2*abs(np.min(DDA_elv_agg.cext_hh.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_agg.cext_hh.values)),
                                'cext_v':10**np.array([(np.log10(DDA_elv_agg.cext_vv.values+2*abs(np.min(DDA_elv_agg.cext_vv.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_agg.cext_vv.values)),
                                'kdp':10**np.array([(np.log10(DDA_elv_agg.kdp.values+2*abs(np.min(DDA_elv_agg.kdp.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_agg.kdp.values)),

                                # 'Z11':10**np.array([np.log10(DDA_elv_agg.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                # 'Z12':10**np.array([(np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                # 'Z21':10**np.array([(np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                # 'Z22':10**np.array([(np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                # 'S11i':10**np.array([(np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                # 'S22i':10**np.array([(np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                # 'S11r':10**np.array([(np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                # 'S22r':10**np.array([(np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22r.values))-1,
                                }

                    
                
                elif scatSet['selMode'] == 'NearestNeighbors':
                    neigh = NearestNeighbors(n_neighbors=scatSet['n_neighbors'])
                    neigh.fit(pointsAgg)
                    distances, indices = neigh.kneighbors(mcSnowPointsAgg)
                    scatPoints={'reflect_hh':10**np.array([(np.log10(DDA_elv_agg.Ze_h.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                'reflect_vv':10**np.array([(np.log10(DDA_elv_agg.Ze_v.values))[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                'reflect_hv':10**np.array([(np.log10(DDA_elv_agg.Ze_hv.values+2*abs(np.min(DDA_elv_agg.Ze_hv.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_agg.Ze_hv.values)),
                                'cext_h':10**np.array([(np.log10(DDA_elv_agg.cext_hh.values+2*abs(np.min(DDA_elv_agg.cext_hh.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_agg.cext_hh.values)),
                                'cext_v':10**np.array([(np.log10(DDA_elv_agg.cext_vv.values+2*abs(np.min(DDA_elv_agg.cext_vv.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_agg.cext_vv.values)),
                                'kdp':10**np.array([(np.log10(DDA_elv_agg.kdp.values+2*abs(np.min(DDA_elv_agg.kdp.values))))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-2*abs(np.min(DDA_elv_agg.kdp.values)),

                                # 'Z11':10**np.array([np.log10(DDA_elv_agg.Z11.values)[idx].mean() if len(idx) > 0 else np.nan for idx in indices]),
                                # 'Z12':10**np.array([(np.log10(DDA_elv_agg.Z12.values+abs(np.min(DDA_elv_agg.Z12.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z12.values))-1,
                                # 'Z21':10**np.array([(np.log10(DDA_elv_agg.Z21.values+abs(np.min(DDA_elv_agg.Z21.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z21.values))-1,
                                # 'Z22':10**np.array([(np.log10(DDA_elv_agg.Z22.values+abs(np.min(DDA_elv_agg.Z22.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.Z22.values))-1,
                                # 'S11i':10**np.array([(np.log10(DDA_elv_agg.S11i.values+abs(np.min(DDA_elv_agg.S11i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11i.values))-1,
                                # 'S22i':10**np.array([(np.log10(DDA_elv_agg.S22i.values+abs(np.min(DDA_elv_agg.S22i.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22i.values))-1,
                                # 'S11r':10**np.array([(np.log10(DDA_elv_agg.S11r.values+abs(np.min(DDA_elv_agg.S11r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S11r.values))-1,
                                # 'S22r':10**np.array([(np.log10(DDA_elv_agg.S22r.values+abs(np.min(DDA_elv_agg.S22r.values))+1))[idx].mean() if len(idx) > 0 else np.nan for idx in indices])-abs(np.min(DDA_elv_agg.S22r.values))-1,
                                }
                                
                #reflect_h,  reflect_v, reflect_hv, kdp_M1, rho_hv, cext_hh, cext_vv = radarScat(scatPoints, wl,scatSet['K2'])
                #print(len(scatPoints['reflect_hh']))
                #print(len(mcTableAgg.index))
                mcTable['sZeH'].loc[elv,wl,mcTableAgg.index] = scatPoints['reflect_hh']
                mcTable['sZeV'].loc[elv,wl,mcTableAgg.index] = scatPoints['reflect_vv']
                mcTable['sZeHV'].loc[elv,wl,mcTableAgg.index] = scatPoints['reflect_hv']
                mcTable['sKDP'].loc[elv,wl,mcTableAgg.index] = scatPoints['kdp']
                mcTable['sCextH'].loc[elv,wl,mcTableAgg.index] = scatPoints['cext_h']
                mcTable['sCextV'].loc[elv,wl,mcTableAgg.index] = scatPoints['cext_v']



            


   
            

    return mcTable


