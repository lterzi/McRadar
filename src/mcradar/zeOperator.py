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
def search_ckdtree(tree, scaling, target):
	scaled_target = np.array(list(target.values())).T*scaling
	idx = tree.query_ball_point(scaled_target, r=1.0)
	return idx 

def calcParticleZe(wls, elvs,mcTable, mcTableAgg,mcTableCry,scatSet,beta,beta_std,tree,scaling,DDA_data_agg):#zeOperator
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
    
    if scatSet['mode']== 'wobbling':
        betas = np.random.normal(loc=beta, scale=beta_std, size=len(mcTableCry.dia))
        DDA_data_cry = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_crystals_withbetanew.nc') #all_crystals #only_beta2.0000e-01_gamma1.5849e-04_
        
        if 'D_max' in DDA_data_cry:
            DDA_data_cry = DDA_data_cry.rename({'D_max':'Dmax'})
    elif scatSet['mode'] == 'fixed_orientation':
        DDA_data_cry = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_crystals.nc') #all_crystals #only_beta2.0000e-01_gamma1.5849e-04_
        if 'D_max' in DDA_data_cry:
            DDA_data_cry = DDA_data_cry.rename({'D_max':'Dmax'})
    else:
        raise ValueError('Unknown mode: '+scatSet['mode']+'! Please choose between "fixed_orientation" and "wobbling"!')

    DDA_data_cry = DDA_data_cry.to_dataframe()
    # generate points to look up in the DDA LUT
    for i,wl in enumerate(wls):
        
        wl_close = DDA_data_cry.iloc[(DDA_data_cry['wavelength']-wl).abs().argsort()].wavelength.values[0] # get closest wavelength to select from LUT
        DDA_wl_cry = DDA_data_cry[DDA_data_cry.wavelength==wl_close]
        
        for elv in elvs:
            
            el_close = DDA_wl_cry.iloc[(DDA_wl_cry['elevation']-elv).abs().argsort()].elevation.values[0] # get closest elevation to select from LUT
            DDA_elv_cry = DDA_wl_cry[DDA_wl_cry.elevation==el_close]
            DDA_elv_cry = DDA_elv_cry[DDA_elv_cry.kdp<1]
            #print(len(mcTableCry.sPhi),len(mcTableCry.sPhi)>0)
            #print(len(mcTableAgg.mTot))
            if len(mcTableCry.sPhi)>0: # only possible if we have plate-like particles
                if scatSet['mode'] == 'wobbling':
                    pointsCry = np.array(list(zip(np.log10(DDA_elv_cry.Dmax), np.log10(DDA_elv_cry.mass), np.log10(DDA_elv_cry.ar),DDA_elv_cry.beta)))
                    mcSnowPointsCry = np.array(list(zip(np.log10(mcTableCry.dia), np.log10(mcTableCry.mTot), np.log10(mcTableCry.sPhi),betas)))
                elif scatSet['mode']== 'fixed_orientation':
                    pointsCry = np.array(list(zip(np.log10(DDA_elv_cry.Dmax), np.log10(DDA_elv_cry.mass), np.log10(DDA_elv_cry.ar))))
                    mcSnowPointsCry = np.array(list(zip(np.log10(mcTableCry.dia), np.log10(mcTableCry.mTot), np.log10(mcTableCry.sPhi))))
            
                # select now the points according to the defined method
                # Fit the KNeighborsRegressor
                if scatSet['selmode'] == 'KNeighborsRegressor':
                    knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'],weights='distance')
                    #print(np.isnan(DDA_elv_cry).any())
                    # in order to apply the log, all values need to be positive, so we are going to shift all values by the minimum value (except for Z11 because this is always positive)
                    # scatPoints = {'reflect_hh':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_h.values)).predict(mcSnowPointsCry)),#'Z11':10**knn.fit(pointsCry, np.log10(DDA_elv_cry.Z11.values)).predict(mcSnowPointsCry),
                    #                 'reflect_vv':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_v.values)).predict(mcSnowPointsCry)),
                    #                 'reflect_hv':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_hv.values+abs(np.max(DDA_elv_cry.Ze_hv.values)))).predict(mcSnowPointsCry))-abs(np.max(DDA_elv_cry.Ze_hv.values)),
                    #                 'cext_h':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_hh.values+2*abs(np.min(DDA_elv_cry.cext_hh.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.cext_hh.values)),
                    #                 'cext_v':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_vv.values+2*abs(np.min(DDA_elv_cry.cext_vv.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.cext_vv.values)),
                    #                 'kdp':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.kdp.values+2*abs(np.min(DDA_elv_cry.kdp.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.kdp.values)),

                    #                 'Z11':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z11.values+abs(np.min(DDA_elv_cry.Z11.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z11.values))-1,
                    #                 'Z12':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z12.values+abs(np.min(DDA_elv_cry.Z12.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z12.values))-1,
                    #                 'Z21':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z21.values+abs(np.min(DDA_elv_cry.Z21.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z21.values))-1,
                    #                 'Z22':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Z22.values+abs(np.min(DDA_elv_cry.Z22.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Z22.values))-1,
                    #                 'S11i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11i.values+abs(np.min(DDA_elv_cry.S11i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11i.values))-1,
                    #                 'S22i':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22i.values+abs(np.min(DDA_elv_cry.S22i.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22i.values))-1,
                    #                 'S11r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S11r.values+abs(np.min(DDA_elv_cry.S11r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S11r.values))-1,
                    #                 'S22r':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.S22r.values+abs(np.min(DDA_elv_cry.S22r.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.S22r.values))-1,
                                    
                    #                 }
                    #print(DDA_elv_cry)
                    scatPoints = {'cbck_h':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_h.values)).predict(mcSnowPointsCry)),#'Z11':10**knn.fit(pointsCry, np.log10(DDA_elv_cry.Z11.values)).predict(mcSnowPointsCry),
                                    'cbck_v':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_v.values)).predict(mcSnowPointsCry)),
                                    'cbck_hv':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_hv.values+abs(np.min(DDA_elv_cry.Ze_hv.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Ze_hv.values))-1,
                                    'cext_h':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_hh.values+2*abs(np.min(DDA_elv_cry.cext_hh.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.cext_hh.values)),
                                    'cext_v':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_vv.values+2*abs(np.min(DDA_elv_cry.cext_vv.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.cext_vv.values)),
                                    'kdp':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.kdp.values+2*abs(np.min(DDA_elv_cry.kdp.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.kdp.values)),
                                    }
                
                
                mcTable['sZeH'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_h']
                mcTable['sCextH'].loc[elv,wl,mcTableCry.index] = scatPoints['cext_h']
                mcTable['sCextV'].loc[elv,wl,mcTableCry.index] = scatPoints['cext_v']
                mcTable['sZeV'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_v']
                mcTable['sZeHV'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_hv']
                mcTable['sKDP'].loc[elv,wl,mcTableCry.index] = scatPoints['kdp']
                

            
            if len(mcTableAgg.mTot)>0:
                from tqdm import tqdm
                import time
                start = time.time()
                target = dict(
                        logmass     = np.log10(mcTableAgg.mTot.values),
                        logDmax     = np.log10(mcTableAgg.dia.values),
                        elevation 	= np.ones(len(mcTableAgg.mTot.values))*elv,
                        wavelength  = np.ones(len(mcTableAgg.mTot.values))*wl,
                    )
                search_idx = search_ckdtree(tree, scaling, target)

                if False:
                    for i_part, trgt, idx in tqdm(zip(range(mcTableAgg.index.size), zip(*target.values()), search_idx), total=mcTableAgg.index.size):
                        #print(f"for super particle {trgt=}, we have {len(idx)=} entries") 
                        scatPoints = DDA_data_agg.isel(index=idx)

                        superparticle = mcTableAgg.isel(index=i_part)
                        if len(scatPoints.index) > superparticle.sMult.values:
                            idx_sel = np.random.choice(np.arange(0,len(scatPoints.index)-1), size = int(superparticle.sMult.values),replace=False)
                        elif len(scatPoints.index)==0:
                            print('Warning: no scatPoints found for superparticle {0}, wl {1}, elv {2}'.format(superparticle.index.values, wl,elv))
                            print('dia',superparticle.dia.values, 'mass',superparticle.mTot.values)
                            continue
                        else:
                            print('Warning: not enough scatPoints, sMult {0}, scatPoints {1}'.format(superparticle.sMult.values, len(scatPoints.index)))
                            print(superparticle.dia.values, superparticle.mTot.values, superparticle.index.values, wl,elv)
                            idx_sel = np.random.choice(np.arange(0,len(scatPoints.index)-1), size = int(superparticle.sMult.values),replace=True)

                        # we can already sum the randomly selected points here, because otherwise we would have summed them for creating the Doppler spectra anyway.
                        mcTable['sZeH'].loc[elv,wl,superparticle.index] = scatPoints['ZeH'][idx_sel].values.sum()*2*np.pi 
                        mcTable['sCextH'].loc[elv,wl,superparticle.index] = scatPoints['CextH'][idx_sel].values.sum()*2*np.pi 
                        mcTable['sCextV'].loc[elv,wl,superparticle.index] = scatPoints['CextV'][idx_sel].values.sum()*2*np.pi 
                        mcTable['sZeV'].loc[elv,wl,superparticle.index] = scatPoints['ZeV'][idx_sel].values.sum()*2*np.pi 
                        mcTable['sZeHV'].loc[elv,wl,superparticle.index] = scatPoints['ZeHV'][idx_sel].values.sum()*2*np.pi 
                        mcTable['sKDP'].loc[elv,wl,superparticle.index] = scatPoints['KDP'][idx_sel].values.sum()*2*np.pi 
                        # mcTable['sMult'].loc[superparticle.index] = 1 # Note, cant do it here because it would change results next iteration
                    print(f"aggregate scattering lookup took {time.time() - start}s")
                else:
                    def scatLookup():
                        variables = ('ZeH', 'CextH', 'CextV', 'ZeV', 'ZeHV', 'KDP')
                        result = mcTableAgg.sel(elevation=elv, wavelength=wl).get([f"s{_}" for _ in variables]).copy() # local slice that gets updated
                        not_found_particles = []
                        for isp, idx in enumerate(tqdm(search_idx)):
                            if len(idx) < 1:
                                #raise ( ValueError(f'Could not find scattering props for aggregate: {elv=} {wl=} {mcTableAgg.isel(index=isp).get(["mTot","dia"]).to_dict()=}') )
                                not_found_particles.append(isp)
                                continue

                            xi = int(mcTableAgg.sMult[isp])

                            # for now we accept all acceptable scattering aggregates with equal weight
                            if True:
                                # randomly take xi many out of candidates
                                idx = np.random.choice(idx, min(len(idx), xi), replace=False)
                            else:
                                # we just take the first ones up to xi
                                idx = idx[:xi]
                            Ncandidates = len(idx)

                            wgt = np.ones(Ncandidates) * xi / Ncandidates

                            # normalize to make sure however many candidates we have, we end up with xi contribution
                            wgt *= 2 * np.pi * xi / np.sum(wgt)

                            for v in variables:
                                result[f"s{v}"][isp] = np.sum(DDA_data_agg[v].data[idx] * wgt)

                        return result, not_found_particles

                    scat_agg, not_found_particles = scatLookup()
                    print(f"aggregate scattering lookup took {time.time() - start}s")

                    if len(not_found_particles) > 1:
                        print(f"Warning, we have {len(not_found_particles)} of {len(search_idx)} sps that did not match the DDA scatter db")
                        print(f"Missing mass: {float(mcTableAgg.mTot.isel(index=not_found_particles).sum())} of {float(mcTableAgg.mTot.sum())} ",
                                f"({100*float(mcTableAgg.mTot.isel(index=not_found_particles).sum()) / float(mcTableAgg.mTot.sum())}%)")

                    for k,v in scat_agg.data_vars.items():
                        mcTable[k].loc[dict(elevation=elv, wavelength=wl, index=scat_agg.index)] = v


            #plt.semilogx(mcTable.dia.loc[mcTableAgg.index], 10*np.log10(mcTable.sZeH.loc[elv,wl,mcTableAgg.index]),'.',label='cbck_h',ls='None',c='C1')
    #plt.show()

                #plt.savefig('test_cbck_h.png')

    # We just need to make sure that now the multiplicity is one now, so that this particle is only taken once into account for spectrum.
    mcTable['sMult'].loc[dict(index=mcTableAgg.index)] = 1

    return mcTable


