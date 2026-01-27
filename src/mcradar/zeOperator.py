# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import xarray as xr
import warnings
import time
from sklearn import neighbors
from tqdm import tqdm
from scipy import constants
import pandas as pd
import matplotlib.pyplot as plt
                            
debugging = False
onlyInterp = False

# TODO: this function should deal with the LUTs
def calcScatTmatrix(wl, radii, as_ratio, 
                        rho, elv, ndgs=30,
                        canting=False, cantingStd=1, 
                        meanAngle=0, safeTmatrix=True):
    from pytmatrix.tmatrix import Scatterer
    from pytmatrix import psd, orientation, radar
    from pytmatrix import refractive, tmatrix_aux
    import subprocess

    """
    Calculates the Ze at H and V polarization, Kdp for one wavelength
    TODO: LDR???
    
    Parameters
    ----------
    wl: wavelength [mm] (single value)
    radii: radius [mm] of the particle (array[n])
    as_ratio: aspect ratio of the super particle (array[n])
    rho: density [g/mmˆ3] of the super particle (array[n])
    elv: elevation angle [°]
    ndgs: division points used to integrate over the particle surface
    canting: boolean (default = False)
    cantingStd: standard deviation of the canting angle [°] (default = 1)
    meanAngle: mean value of the canting angle [°] (default = 0)
    
    Returns
    -------
    reflect_h: super particle horizontal reflectivity[mm^6/m^3] (array[n])
    reflect_v: super particle vertical reflectivity[mm^6/m^3] (array[n])
    refIndex: refractive index from each super particle (array[n])
    kdp: calculated kdp from each particle (array[n])
    """
    
    #---pyTmatrix setup
    # initialize a scatterer object
    scatterer = Scatterer(wavelength=wl)
    scatterer.radius_type = Scatterer.RADIUS_MAXIMUM
    scatterer.ndgs = ndgs
    scatterer.ddelta = 1e-6

    if canting==True: 
        scatterer.or_pdf = orientation.gaussian_pdf(std=cantingStd, mean=meanAngle)  
#         scatterer.orient = orientation.orient_averaged_adaptive
        scatterer.orient = orientation.orient_averaged_fixed
    
    # geometric parameters - incident direction
    scatterer.thet0 = 90. - elv
    scatterer.phi0 = 0.
    
    # parameters for backscattering
    refIndex = np.ones_like(radii, np.complex128)*np.nan
    reflect_h = np.ones_like(radii)*np.nan
    reflect_v = np.ones_like(radii)*np.nan

    # S matrix for Kdp
    sMat = np.ones_like(radii)*np.nan
    Z11Mat = np.ones_like(radii)*np.nan
    Z12Mat = np.ones_like(radii)*np.nan
    Z21Mat = np.ones_like(radii)*np.nan
    Z22Mat = np.ones_like(radii)*np.nan
    Z33Mat = np.ones_like(radii)*np.nan
    Z44Mat = np.ones_like(radii)*np.nan
    S11iMat = np.ones_like(radii)*np.nan
    S22iMat = np.ones_like(radii)*np.nan
    for i, radius in enumerate(radii): #tqdm(zip(range(len(radii)), radii),total=len(radii)):
        # A quick function to save the distribution of values used in the test
        #with open('/home/dori/table_McRadar.txt', 'a') as f:
        #    f.write('{0:f} {1:f} {2:f} {3:f} {4:f} {5:f} {6:f}\n'.format(wl, elv,
        #                                                                 meanAngle,
        #                                                                 cantingStd,
        #                                                                 radius,
        #                                                                 rho[i],
        #                                                                 as_ratio[i]))
        # scattering geometry backward
        # radius = 100.0 # just a test to force nans

        scatterer.thet = 180. - scatterer.thet0
        scatterer.phi = (180. + scatterer.phi0) % 360.
        scatterer.radius = radius
        scatterer.axis_ratio = 1./as_ratio[i]
        scatterer.m = refractive.mi(wl, rho[i])
        refIndex[i] = refractive.mi(wl, rho[i])

        if safeTmatrix:
            inputs = [str(scatterer.radius),
                      str(scatterer.wavelength),
                      str(scatterer.m),
                      str(scatterer.axis_ratio),
                      str(int(canting)),
                      str(cantingStd),
                      str(meanAngle),
                      str(ndgs),
                      str(scatterer.thet0),
                      str(scatterer.phi0)]
            arguments = ' '.join(inputs)
            a = subprocess.run(['spheroidMcRadar'] + inputs, # this script should be installed by McRadar
                               capture_output=True)
            # print(str(a))
            try:
                back_hh, back_vv, sMatrix, Z11, Z12, Z21, Z22, Z33, Z44, S11i, S22i, _ = str(a.stdout).split('Results ')[-1].split()
                back_hh = float(back_hh)
                back_vv = float(back_vv)
                sMatrix = float(sMatrix)
                Z11 = float(Z11)
                Z12 = float(Z12)
                Z21 = float(Z21)
                Z22 = float(Z22)
                Z33 = float(Z33)
                Z44 = float(Z44)
                S11i = float(S11i)
                S22i = float(S22i)
            except:
                back_hh = np.nan
                back_vv = np.nan
                sMatrix = np.nan
                Z11 = np.nan
                Z12 = np.nan
                Z21 = np.nan
                Z22 = np.nan
                Z33 = np.nan
                Z44 = np.nan
                S11i = np.nan
                S22i = np.nan
            # print(back_hh, radar.radar_xsect(scatterer, True))
            # print(back_vv, radar.radar_xsect(scatterer, False))
            reflect_h[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * back_hh # radar.radar_xsect(scatterer, True)  # Kwsqrt is not correct by default at every frequency
            reflect_v[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * back_vv # radar.radar_xsect(scatterer, False)

            # scattering geometry forward
            # scatterer.thet = scatterer.thet0
            # scatterer.phi = (scatterer.phi0) % 360. #KDP geometry
            # S = scatterer.get_S()
            sMat[i] = sMatrix # (S[1,1]-S[0,0]).real
            Z11Mat[i] = Z11
            Z12Mat[i] = Z12
            Z21Mat[i] = Z21
            Z22Mat[i] = Z22
            Z33Mat[i] = Z33
            Z44Mat[i] = Z44
            S11iMat[i] = S11i
            S22iMat[i] = S22i
            # print(sMatrix, sMat[i])
            # print(sMatrix)
        else:

            reflect_h[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * radar.radar_xsect(scatterer, True)  # Kwsqrt is not correct by default at every frequency
            reflect_v[i] = scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * radar.radar_xsect(scatterer, False)

            # scattering geometry forward
            scatterer.thet = scatterer.thet0
            scatterer.phi = (scatterer.phi0) % 360. #KDP geometry
            S = scatterer.get_S()
            Z = scatterer.get_Z()
            sMat[i] = (S[1,1]-S[0,0]).real
            Z11Mat[i] = Z[0,0]
            Z12Mat[i] = Z[0,1]
            Z21Mat[i] = Z[1,0]
            Z22Mat[i] = Z[1,1]
            Z33Mat[i] = Z[2,2]
            Z44Mat[i] = Z[3,3]
            S11iMat[i] = S[0,0].imag
            S22iMat[i] = S[1,1].imag
            
    kdp = 1e-3* (180.0/np.pi)*scatterer.wavelength*sMat

    del scatterer # TODO: Evaluate the chance to have one Scatterer object already initiated instead of having it locally
    
    return reflect_h, reflect_v, refIndex, kdp, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat

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

def calcParticleZe(wls, elvs,mcTable, mcTableAgg,mcTableCry,mcTableFrozen,mcTableMelted,mcTableLiquid,scatSet,beta,beta_std,treeAgg,scalingAgg,treeCry,scalingCry,DDA_data_agg,DDA_data_cry,nmono_array, temperature=None, ice_core=True,height=None):#zeOperator
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
    if True:
        if scatSet['mode']== 'wobbling':
            betas = np.random.normal(loc=beta, scale=beta_std, size=len(mcTableCry.dia))
            DDA_data_cry = xr.open_dataset(scatSet['lutPath']+'scattering_properties_all_crystals_withbetanew_kdp1.nc') #all_crystals #only_beta2.0000e-01_gamma1.5849e-04_
            
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
        
        if True:
            wl_close = DDA_data_cry.iloc[(DDA_data_cry['wavelength']-wl).abs().argsort()].wavelength.values[0] # get closest wavelength to select from LUT
            DDA_wl_cry = DDA_data_cry[DDA_data_cry.wavelength==wl_close]
        
        for elv in elvs:
            if len(mcTableCry.sPhi)>0: # only possible if we have plate-like particles
                print('Calculating crystals with DDA LUT')
                if True:
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
                            
                            scatPoints = {'cbck_h':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_h.values)).predict(mcSnowPointsCry)),#'Z11':10**knn.fit(pointsCry, np.log10(DDA_elv_cry.Z11.values)).predict(mcSnowPointsCry),
                                            'cbck_v':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_v.values)).predict(mcSnowPointsCry)),
                                            'cbck_hv':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.Ze_hv.values+abs(np.min(DDA_elv_cry.Ze_hv.values))+1)).predict(mcSnowPointsCry))-abs(np.min(DDA_elv_cry.Ze_hv.values))-1,
                                            'cext_h':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_hh.values+2*abs(np.min(DDA_elv_cry.cext_hh.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.cext_hh.values)),
                                            'cext_v':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.cext_vv.values+2*abs(np.min(DDA_elv_cry.cext_vv.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.cext_vv.values)),
                                            'kdp':10**(knn.fit(pointsCry, np.log10(DDA_elv_cry.kdp.values+2*abs(np.min(DDA_elv_cry.kdp.values)))).predict(mcSnowPointsCry))-2*abs(np.min(DDA_elv_cry.kdp.values)),
                                            }
                        
                        
                        mcTable['sZeH'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_h']*2*np.pi*2*np.pi # multiply by 2pi to be consistent with Rayleigh
                        mcTable['sCextH'].loc[elv,wl,mcTableCry.index] = scatPoints['cext_h']
                        mcTable['sCextV'].loc[elv,wl,mcTableCry.index] = scatPoints['cext_v']
                        mcTable['sZeV'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_v']*2*np.pi*2*np.pi
                        mcTable['sZeHV'].loc[elv,wl,mcTableCry.index] = scatPoints['cbck_hv']
                        mcTable['sKDP'].loc[elv,wl,mcTableCry.index] = scatPoints['kdp']
                else:
                    start = time.time()
                    target = dict(
                            logmass     = np.log10(mcTableCry.mTot.values),
                            logDmax     = np.log10(mcTableCry.dia.values),
                            logar      = np.log10(mcTableCry.sPhi.values),
                            elevation 	= np.ones(len(mcTableCry.mTot.values))*elv,
                            wavelength  = np.ones(len(mcTableCry.mTot.values))*wl,
                        )
                    search_idx = search_ckdtree(treeCry, scalingCry, target)

                    def scatLookup():
                        variables = ('ZeH', 'CextH', 'CextV', 'ZeV', 'ZeHV', 'KDP')
                        result = mcTableCry.sel(elevation=elv, wavelength=wl).get([f"s{_}" for _ in variables]).copy() # local slice that gets updated
                        not_found_particles = []
                        for isp, idx in enumerate((search_idx)):
                        #for isp, idx in enumerate(tqdm(search_idx)):
                            if len(idx) < 1:
                                #raise ( ValueError(f'Could not find scattering props for aggregate: {elv=} {wl=} {mcTableAgg.isel(index=isp).get(["mTot","dia"]).to_dict()=}') )
                                not_found_particles.append(isp)
                                continue

                            xi = int(mcTableCry.sMult[isp])

                            # for now we accept all acceptable scattering aggregates with equal weight # what is happening here: randomly select particles out of candidates, maximal: xi, minimal: all candidates
                            if True:
                                # randomly take xi many out of candidates
                                idx = np.random.choice(idx, min(len(idx), xi), replace=False)
                            else:
                                # we just take the first ones up to xi
                                idx = idx[:xi]
                            Ncandidates = len(idx)

                            wgt = np.ones(Ncandidates) * xi / Ncandidates

                            # normalize to make sure however many candidates we have, we end up with xi contribution
                            wgt *= 2 * np.pi * xi / np.sum(wgt) #2 * np.pi * xi / np.sum(wgt)

                            for v in variables:
                                result[f"s{v}"][isp] = np.sum(mcTableCry[v].data[idx] * wgt)

                        return result, not_found_particles
                    
                    scat_cry, not_found_particles = scatLookup()
                    
                    if len(not_found_particles) > 1:
                        print(f"Warning, we have {len(not_found_particles)} of {len(search_idx)} crystals that did not match the DDA scatter db")
                        print(f"Missing mass: {float(mcTableCry.mTot.isel(index=not_found_particles).sum())} of {float(mcTableCry.mTot.sum())} ",
                                f"({100*float(mcTableCry.mTot.isel(index=not_found_particles).sum()) / float(mcTableCry.mTot.sum())}%)")
                    for k,v in scat_cry.data_vars.items():
                        mcTable[k].loc[dict(elevation=elv, wavelength=wl, index=scat_cry.index)] = v

            
            if len(mcTableAgg.mTot)>0:
                print('calculating aggregates with DDA LUT')
                start = time.time()
                target = dict(
                        logmass     = np.log10(mcTableAgg.mTot.values),
                        logDmax     = np.log10(mcTableAgg.dia.values),
                        elevation 	= np.ones(len(mcTableAgg.mTot.values))*elv,
                        wavelength  = np.ones(len(mcTableAgg.mTot.values))*wl,
                        #habit       = mcTableAgg.habit_code.values,
                        #Nmono       = mcTableAgg.sNmono.values,
                    )
                search_idx = search_ckdtree(treeAgg, scalingAgg, target)
                # if elv == 30:
                #    fig1,ax1 = plt.subplots(ncols=3,nrows=4,figsize=(20,15),constrained_layout=True)
                #fig2,ax2 = plt.subplots(ncols=3,nrows=3,figsize=(15,15),constrained_layout=True)            
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
                        particles_smaller_smult = []
                        particles_smaller_10 = []
                        particles_large_spread = []
                        
                        for isp, idx in enumerate((search_idx)):
                            
                        #for isp, idx in enumerate(tqdm(search_idx)):
                            if len(idx) < 1:
                                #raise ( ValueError(f'Could not find scattering props for aggregate: {elv=} {wl=} {mcTableAgg.isel(index=isp).get(["mTot","dia"]).to_dict()=}') )
                                #print(mcTableAgg.sNmono[isp].values)
                                not_found_particles.append(isp)
                                continue
                            
                            
                            idx = np.asarray(idx, dtype=np.int64)  # 
                            mask = nmono_array[idx] > 10  # only Nmono larger 10
                            idx = idx[mask]
                            if len(idx) < 1:
                                #raise ( ValueError(f'Could not find scattering props for aggregate: {elv=} {wl=} {mcTableAgg.isel(index=isp).get(["mTot","dia"]).to_dict()=}') )
                                not_found_particles.append(isp)
                                continue
                            # KDPdata = DDA_data_agg.KDP.data[idx]
                            # meanKDP = np.mean(KDPdata)
                            # stdKDP = np.std(KDPdata)
                            # #medianKDP = np.median(DDA_data_agg.KDP.data[idx])
                            # spread_around_mean = np.abs((KDPdata - meanKDP)/stdKDP)
                            # #print(spread_around_mean < 1.5)
                            # #print(idx)
                            # #print(spread_around_mean)
                            # #quit()
                            # idx = idx[spread_around_mean < 0.75] # only take values within 1.5 stddev around mean                            

                            #if stdKDP > 0.5*abs(meanKDP):
                            #    particles_large_spread.append(isp)
                                #print('large spread KDP for particle', isp, 'mean KDP:', meanKDP, 'std KDP:', stdKDP, 'median KDP:', medianKDP)
                                #continue
                            xi = int(mcTableAgg.sMult[isp])
                            
                            if len(idx) < xi:
                                particles_smaller_smult.append(xi)
                                #particles_smaller_idx.append(len(idx))
                                #print(len(idx),'<',xi,'for particle',isp,'sMult',mcTableAgg.sMult.isel(index=isp).values)
                            # for now we accept all acceptable scattering aggregates with equal weight # what is happening here: randomly select particles out of candidates, maximal: xi, minimal: all candidates
                            
                            if True:
                                # randomly take xi many out of candidates
                                idx = np.random.choice(idx, min(len(idx), xi), replace=False)
                            else:
                                # we just take the first ones up to xi
                                idx = idx[:xi]
                            #Ncandidates = len(idx)

                            if False:
                                wgt = np.ones(Ncandidates) * xi / Ncandidates

                                # normalize to make sure however many candidates we have, we end up with xi contribution
                                wgt *= xi / np.sum(wgt) #2 * np.pi * xi / np.sum(wgt) # multiply by 2pi, to make consistent with Rayleigh

                            for v in variables:
                                #print(f"s{v}min", np.min(DDA_data_agg[v].data[idx]), "max", np.max(DDA_data_agg[v].data[idx]))
                                if 'Z' in v:
                                    # add 2pi factor here for Ze to be consistent with Rayleigh
                                    result[f"s{v}"][isp] = np.mean(DDA_data_agg[v].data[idx])*xi*2*np.pi#np.sum(DDA_data_agg[v].data[idx] * wgt)*2*np.pi
                                else:
                                    result[f"s{v}"][isp] = np.mean(DDA_data_agg[v].data[idx])*xi#np.sum(DDA_data_agg[v].data[idx] * wgt)

                        return result, not_found_particles, particles_smaller_smult, particles_smaller_10#, p1

                    scat_agg, not_found_particles, particles_smaller_smult, particles_smaller_10 = scatLookup()
                    
                    if len(not_found_particles) > 1:
                        print(f"Warning, we have {len(not_found_particles)} of {len(search_idx)} aggs that did not match the DDA scatter db")
                        print(f"Missing mass: {float(mcTableAgg.mTot.isel(index=not_found_particles).sum())} of {float(mcTableAgg.mTot.sum())} ",
                                f"({100*float(mcTableAgg.mTot.isel(index=not_found_particles).sum()) / float(mcTableAgg.mTot.sum())}%)")
                    if len(particles_smaller_10) > 1:
                        print(f"Warning, we have {len(particles_smaller_10)} of {len(search_idx)} aggs that had less than 10 scatter db entries")
                        print(f"Missing mass: {float(mcTableAgg.mTot.isel(index=particles_smaller_10).sum())} of {float(mcTableAgg.mTot.sum())} ",
                                f"({100*float(mcTableAgg.mTot.isel(index=particles_smaller_10).sum()) / float(mcTableAgg.mTot.sum())}%)")
                    # #print(mcTable)    
                    # if len(particles_smaller_smult) > 1:
                    #     print(f"Warning, we have {len(particles_smaller_smult)} of {len(search_idx)} aggs that had less scatter db entries than sMult")
                    #     print(f"min(sMult): {min(particles_smaller_smult)}, len(idx(min(sMult))): {particles_smaller_idx[np.argmin(particles_smaller_smult)]}")
                    #     print(f"max(sMult): {max(particles_smaller_smult)}, len(idx(max(sMult))): {particles_smaller_idx[np.argmax(particles_smaller_smult)]}")
                    for k,v in scat_agg.data_vars.items():
                        #print(k,v)
                        if k == 'sZeV':
                            k = 'sZeH'
                        elif k == 'sZeH':
                            k = 'sZeV'
                        mcTable[k].loc[dict(elevation=elv, wavelength=wl, index=scat_agg.index)] = v
                    #quit()
                # if elv==30:
                #     #cbar = fig1.colorbar(p1,ax=ax1.ravel().tolist(),aspect=70, pad=0.01)
                #     #cbar.ax.tick_params(labelsize=12)
                #     #cbar.set_label('Number of candidates from LUT', fontsize=18)
                #     fig1.savefig('aggregate_lookup_stats_height{}_wl{:.2f}_elv{}.png'.format(height, wl,elv))
                #     plt.close()
                # #     #plt.close()
            if len(mcTableFrozen.mTot)>0:
                print('Calculating frozen particles with T-matrix')
                # now we use Tmatrix for the frozen particles...
                reflect_h, reflect_v, refIndex, kdp, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat = calcScatTmatrix(wl,
                                                                                                                                            mcTableFrozen.dia.values/2*1e3,
                                                                                                                                            mcTableFrozen.sPhi.values,
                                                                                                                                            mcTableFrozen.sRho_tot.values*1e3/1e9, #kg/m3 to g/mm3
                                                                                                                                            elv,
                                                                                                                                            ndgs=30,
                                                                                                                                            canting=False,
                                                                                                                                            cantingStd=beta_std,
                                                                                                                                            meanAngle=beta,
                                                                                                                                            safeTmatrix=True)
                mcTable['sZeH'].loc[elv,wl,mcTableFrozen.index] = reflect_h#*mcTableFrozen.sMult.values#
                mcTable['sZeV'].loc[elv,wl,mcTableFrozen.index] = reflect_v#*mcTableFrozen.sMult.values#
                mcTable['sKDP'].loc[elv,wl,mcTableFrozen.index] = kdp#*mcTableFrozen.sMult.values#
                mcTable['sCextH'].loc[elv,wl,mcTableFrozen.index] = (S22iMat*4*np.pi/(2*np.pi/wl))#*mcTableFrozen.sMult.values#
                mcTable['sCextV'].loc[elv,wl,mcTableFrozen.index] = (S11iMat*4*np.pi/(2*np.pi/wl))#*mcTableFrozen.sMult.values#
                #mcTable['sZeHV'].loc[elv,wl,mcTableFrozen.index] = reflect_hv#*mcTableFrozen.sMult.values#
            if len(mcTableMelted.mTot)>0:
                print('Calculating melted particles with scattnlay')
                # for now melted particles are spheres with ice core and water coating, if changed to water core and ice coating, need to change code here and in fullRadarOperator
                # we use scattnlay for that: 
                from scattnlay import scattnlay
                from pytmatrix import refractive
                
                if ice_core:
                    x_water_coating = scatt_param(mcTableMelted.dia/2*1e3, wl)
                    x_ice_core = scatt_param(mcTableMelted.dia_ice_core/2*1e3, wl)
                    m_water = m_water_wl(wl)
                    m_total = np.zeros((2),dtype =complex)
                    m_total[1] = m_water
                    for x_ice, x_water, dia, rho_ice, index in zip(x_ice_core, x_water_coating, mcTableMelted.dia, mcTableMelted.rho_ice_core, mcTableMelted.index):
                        #print(noParts.values)
                        m_ice = refractive.mi(wl, rho_ice)
                        
                        m_total[0] = m_ice
                        x_total = np.array([x_ice, x_water])

                        terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(x_total,m_total)#,theta=np.array([180]))

                        Cext, Csca, Cabs, Cbk  = Q2C(np.array([Qext, Qsca, Qabs, Qbk]), dia.values/2*1e3) 
                        mcTable['sZeH'].loc[elv,wl,index] = wl**4*Cbk/(np.pi**5*scatSet['K2']) #(refl(Cbk, wl, Kw2)) 
                        mcTable['sCextH'].loc[elv,wl,index] = Cext
                        mcTable['sZeV'].loc[elv,wl,index] = np.nan
                else:
                    x_water_coating = scatt_param(mcTableMelted.dia_water_core/2*1e3, wl)
                    x_ice_core = scatt_param(mcTableMelted.dia/2*1e3, wl)
                    m_water = m_water_wl(wl)
                    m_total = np.zeros((2),dtype =complex)
                    m_total[0] = m_water
                    for x_ice, x_water, dia, rho_ice, index in zip(x_ice_core, x_water_coating, mcTableMelted.dia, mcTableMelted.rho_ice_coat, mcTableMelted.index):
                        #print(noParts.values)
                        m_ice = refractive.mi(wl, rho_ice)
                        
                        m_total[1] = m_ice
                        x_total = np.array([x_water, x_ice])

                        terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(x_total,m_total)#,theta=np.array([180]))

                        Cext, Csca, Cabs, Cbk  = Q2C(np.array([Qext, Qsca, Qabs, Qbk]), dia.values/2*1e3) 
                        mcTable['sZeH'].loc[elv,wl,index] = wl**4*Cbk/(np.pi**5*scatSet['K2']) #(refl(Cbk, wl, Kw2)) 
                        mcTable['sCextH'].loc[elv,wl,index] = Cext
                    #data['cbck'].loc[elv,wl,index] = Cbk
            if len(mcTableLiquid.mTot)>0:
                print('Calculating liquid particles with pre-calculated LUT')
                freq = (constants.c / (wl*1e-3))*1e-9
                #print(freq)
                temperature='283.15'
                scatTable = pd.read_csv(scatSet['lutPath']+'liquid_{}_{:.1f}GHz_elv{}.csv'.format(temperature,freq,elv), skiprows=1)
                #print(scatTable)
                scatTable = scatTable.set_index('diameter[mm]').to_xarray().rename({'diameter[mm]':'Dmax','radarXSh[mm2]':'c_bck_h','radarXSv[mm2]':'c_bck_v','extxs[mm2]':'cext_h','sKdp[mm2]':'sKDP'})
                #print(scatTable)
                #quit()
                scatSel = scatTable.sel(Dmax=mcTableLiquid.dia*1e3, method='nearest', tolerance=0.1)
                prefactor = wl**4/(np.pi**5*scatSet['K2'])
                mcTable['sZeH'].loc[elv,wl,mcTableLiquid.index] = scatSel['c_bck_h'].values*prefactor#*mcTableLiquid.sMult.values#
                mcTable['sZeV'].loc[elv,wl,mcTableLiquid.index] = scatSel['c_bck_v'].values*prefactor#*mcTableLiquid.sMult.values#
                mcTable['sKDP'].loc[elv,wl,mcTableLiquid.index] = scatSel['sKDP'].values#*mcTableLiquid.sMult.values#
                mcTable['sCextH'].loc[elv,wl,mcTableLiquid.index] = scatSel['cext_h'].values#*mcTableLiquid.sMult.values#
    # We just need to make sure that now the multiplicity is one now, so that this particle is only taken once into account for spectrum.
    mcTable['sMult'].loc[dict(index=mcTableAgg.index)] = 1
    #import matplotlib.pyplot as plt
    #plt.plot(mcTable.sNmono,mcTable.sMult,'.',ls='None')
    #plt.show()
    #plt.plot(mcTable.dia,mcTable.sKDP.sel(elevation=wls[2], wavelength=elvs[0],method='nearest'),'.',ls='None')
    #plt.show()
    #mcTable['sMult'].loc[dict(index=mcTableCry.index)] = 1

    return mcTable


def scatt_param(r, wl, mm=1.0):
    return 2.0*np.pi*r*mm/wl

def Q2C(Q, r):
    return Q*np.pi*r**2
def refl(cbk, wl, Kw2):
    return wl**4*cbk/(np.pi**5*Kw2)
def m_water_wl(wl):
    m_c = 8.34 + 2.22j
    m_x = 7.20 + 2.84j
    m_ka = 4.05 + 2.42j
    m_w = 2.89 + 1.43j
    ms = np.array([m_c, m_x, m_ka, m_w])
    wls = np.array([53.5,31.2,8.4,3.2])
    closest = np.argmin(np.abs(wls - wl))
    return ms[closest]
