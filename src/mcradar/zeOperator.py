# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
#import xarray as xr
#import warnings
#import time
#from sklearn import neighbors
#from tqdm import tqdm
#from scipy import constants
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import truncnorm

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
    print(wl)
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
                print('did not find suitable results for',i,' with radius', radius, ' and aspect ratio', as_ratio[i], 'and density', rho[i])
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
    Calculates the single scattering radar quantities from the matrix values.

    Parameters
    ----------
    sp : xarray.DataArray or dict
        Superparticles containing backscattering and forward amplitude matrix information.
    wl : float
        Wavelength [mm].
    K2 : float
        Rayleigh dielectric factor $|(m^2-1)/(m^2+2)|^2$.

    Returns
    -------
    reflect_hh : array-like
        Superparticle horizontal reflectivity $[mm^6/m^3]$.
    reflect_vv : array-like
        Superparticle vertical reflectivity $[mm^6/m^3]$.
    reflect_hv : array-like
        Cross-polarized reflectivity $[mm^6/m^3]$.
    kdp : array-like
        Calculated Kdp from each particle.
    rho_hv : array-like
        Correlation coefficient (currently disabled, returns NaN array).
    cext_hh : array-like
        Extinction cross-section for horizontal polarization.
    cext_vv : array-like
        Extinction cross-section for vertical polarization.
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
    """
    Search a cKDTree for points within a scaled radius of the target.

    Parameters
    ----------
    tree : cKDTree
        The KDTree object to search.
    scaling : array-like
        Scaling factors for each dimension.
    target : dict
        Dictionary of target arrays (each key is a variable, values are arrays).

    Returns
    -------
    idx : list
        List of indices for each target point within radius 1.0.
    """
    scaled_target = np.array(list(target.values())).T * scaling
    idx = tree.query_ball_point(scaled_target, r=1.0)
    return idx 
def asinh_transform(x, x0=1.0):
    """
    Apply an inverse hyperbolic sine (asinh) transformation to the input.

    Parameters
    ----------
    x : array-like
        Input data.
    x0 : float, optional
        Scale parameter (default 1.0).

    Returns
    -------
    transformed : array-like
        Transformed data.
    """
    return np.arcsinh(x / x0)
def inv_asinh_transform(y, x0=1.0):
    """
    Inverse of the asinh transformation.

    Parameters
    ----------
    y : array-like
        Transformed data.
    x0 : float, optional
        Scale parameter (default 1.0).

    Returns
    -------
    x : array-like
        Original data before transformation.
    """
    return x0 * np.sinh(y)
def suggest_x0(x):
    """
    Suggest a scale parameter x0 for asinh transformation from data.

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    x0 : float
        Suggested scale parameter (10th percentile of nonzero values, or 1.0 if all zero).
    """
    a = np.abs(np.asarray(x))
    a = a[a > 0]
    if a.size == 0:
        return 1.0
    return np.quantile(a, 0.10)  # Use 10th percentile

def scatt_param(r, wl, mm=1.0):
    """
    Calculate the size parameter for scattering.

    Parameters
    ----------
    r : float or array-like
        Particle radius (same units as wl).
    wl : float
        Wavelength (same units as r).
    mm : float, optional
        Refractive index ratio (default 1.0).

    Returns
    -------
    param : float or array-like
        Size parameter (dimensionless).
    """
    return 2.0 * np.pi * r * mm / wl

def Q2C(Q, r):
    """
    Convert scattering efficiency Q to cross-section C.

    Parameters
    ----------
    Q : float or array-like
        Scattering efficiency.
    r : float or array-like
        Particle radius.

    Returns
    -------
    C : float or array-like
        Cross-section area.
    """
    return Q * np.pi * r ** 2
def refl(cbk, wl, Kw2):
    """
    Calculate radar reflectivity from backscattering cross-section.

    Parameters
    ----------
    cbk : float or array-like
        Backscattering cross-section.
    wl : float
        Wavelength.
    Kw2 : float
        Dielectric factor.

    Returns
    -------
    refl : float or array-like
        Radar reflectivity.
    """
    return wl ** 4 * cbk / (np.pi ** 5 * Kw2)
def m_water_wl(wl):
    """
    Return the refractive index of water for a given wavelength.

    Parameters
    ----------
    wl : float
        Wavelength (mm).

    Returns
    -------
    m : complex
        Refractive index for the closest wavelength.
    """
    m_c = 8.34 + 2.22j
    m_x = 7.20 + 2.84j
    m_ka = 4.05 + 2.42j
    m_w = 2.89 + 1.43j
    ms = np.array([m_c, m_x, m_ka, m_w])
    wls = np.array([53.5, 31.2, 8.4, 3.2])
    closest = np.argmin(np.abs(wls - wl))
    return ms[closest]

class ZeOperator:
    def __init__(self, settings, DDA_data_agg, DDA_data_cry, treeAgg, scalingAgg, treeCry=None, scalingCry=None, nmono_array=None):
        """
        Initialize the ZeOperator class for radar reflectivity calculations.

        Parameters
        ----------
        settings : dict or RadarSettings
            Radar and scattering settings.
        DDA_data_agg : DataFrame
            Lookup table for aggregates.
        DDA_data_cry : DataFrame
            Lookup table for crystals.
        treeAgg : cKDTree
            KDTree for aggregates.
        scalingAgg : array-like
            Scaling for aggregate KDTree.
        treeCry : cKDTree, optional
            KDTree for crystals.
        scalingCry : array-like, optional
            Scaling for crystal KDTree.
        nmono_array : array-like, optional
            Number of monomers for aggregates.
        """
        self.settings = settings
        self.DDA_data_agg = DDA_data_agg
        self.DDA_data_cry = DDA_data_cry
        self.treeAgg = treeAgg
        self.scalingAgg = scalingAgg
        self.treeCry = treeCry
        self.scalingCry = scalingCry
        self.nmono_array = nmono_array
    def compute(self, mcTableTmp, mcTableAggTmp, mcTableCryTmp, mcTableFrozenTmp, mcTableMeltedTmp, mcTableLiquidTmp, beta_std_use, height):
        """
        Main dispatcher: call each handler for the relevant particle type.

        Parameters
        ----------
        mcTableTmp : xarray.Dataset
            Output table to fill.
        mcTableAggTmp, mcTableCryTmp, mcTableFrozenTmp, mcTableMeltedTmp, mcTableLiquidTmp : xarray.Dataset
            Input tables for each particle type.
        beta_std_use : float
            Standard deviation for canting angle.
        height : float
            Height of the radar bin.

        Returns
        -------
        mcTableTmp : xarray.Dataset
            Output table with filled radar variables.
        """
        # Main dispatcher: call each handler for the relevant particle type
        if len(mcTableCryTmp.sPhi) > 0:
            self._handle_crystals(mcTableTmp, mcTableCryTmp, beta_std_use, height)
        if len(mcTableAggTmp.mTot) > 0:
            self._handle_aggregates(mcTableTmp, mcTableAggTmp, height)
        if len(mcTableFrozenTmp.mTot) > 0:
            self._handle_frozen(mcTableTmp, mcTableFrozenTmp, beta_std_use, height)
        if len(mcTableMeltedTmp.mTot) > 0:
            self._handle_melted(mcTableTmp, mcTableMeltedTmp, height)
        if len(mcTableLiquidTmp.mTot) > 0:
            self._handle_liquid(mcTableTmp, mcTableLiquidTmp, beta_std_use, height)
        return mcTableTmp

    def _handle_crystals(self, mcTable, mcTableCry, beta_std_use, height):
        """
        Crystal-specific scattering logic (modularized from calcParticleZe).

        Parameters
        ----------
        mcTable : xarray.Dataset
            Output table to fill.
        mcTableCry : xarray.Dataset
            Input table for crystals.
        beta_std_use : float
            Standard deviation for canting angle.
        height : float
            Height of the radar bin.
        """
        # Crystal-specific scattering logic (modularized from calcParticleZe)
        import numpy as np
        from sklearn import neighbors
        from scipy.stats import truncnorm
        print('Handling crystals at height', height, 'with', len(mcTableCry.mTot), 'particles')

        ## so far this is using the non-stochastic crystals, so our old LUT setup because it is faster
        # Example: get settings from self
        scatSet = self.settings['scatSet'] if isinstance(self.settings, dict) and 'scatSet' in self.settings else self.settings
        wls = self.settings['wl'] #scatSet['wls'] if 'wls' in scatSet else [scatSet['wl']]
        elvs = self.settings['elv'] #scatSet['elvs'] if 'elvs' in scatSet else [scatSet['elv']]
    
        # Truncated normal for betas
        lower, upper = 0, 90
        #print(scatSet)
        #quit()
        mu, sigma = self.settings.get('beta', 0), beta_std_use#self.settings.get('beta_std', 0)
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        betas = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=len(mcTableCry.dia))

        # Open LUT (assume already loaded in self.DDA_data_cry)
        DDA_data_cry = self.DDA_data_cry
        #print(DDA_data_cry)
        #for var in DDA_data_cry:
        #    print(var)
        #if hasattr(DDA_data_cry, 'to_dataframe'):
        #DDA_data_cry = DDA_data_cry.to_dataframe()

        for wl in wls:
            wl_close = DDA_data_cry.iloc[(DDA_data_cry['wavelength']-wl).abs().argsort()].wavelength.values[0]
            DDA_wl_cry = DDA_data_cry[DDA_data_cry.wavelength==wl_close]
            for elv in elvs:
                el_close = DDA_wl_cry.iloc[(DDA_wl_cry['elevation']-elv).abs().argsort()].elevation.values[0]
                DDA_elv_cry = DDA_wl_cry[DDA_wl_cry.elevation==el_close]
                #print(DDA_elv_cry)
                DDA_elv_cry = DDA_elv_cry[DDA_elv_cry.kdp<1]
                # KNN regression for ZeH, ZeV, etc.
                pointsCry = np.array(list(zip(DDA_elv_cry.Dmax, DDA_elv_cry.mass, DDA_elv_cry.ar, DDA_elv_cry.beta)))
                mcSnowPointsCry = np.array(list(zip(mcTableCry.dia, mcTableCry.mTot, mcTableCry.sPhi, betas)))
                knn = neighbors.KNeighborsRegressor(scatSet['n_neighbors'], weights='distance')
                # Utility functions
                def asinh_transform(x, x0=1.0):
                    return np.arcsinh(x/x0)
                def inv_asinh_transform(y, x0=1.0):
                    return x0 * np.sinh(y)
                def suggest_x0(x):
                    a = np.abs(np.asarray(x))
                    a = a[a > 0]
                    if a.size == 0:
                        return 1.0
                    return np.quantile(a, 0.10)
                # Find x0 for each variable
                x0_Ze_h = suggest_x0(DDA_elv_cry.Ze_h.values)
                x0_Ze_v = suggest_x0(DDA_elv_cry.Ze_v.values)
                x0_Ze_hv = suggest_x0(DDA_elv_cry.Ze_hv.values)
                x0_cext_hh = suggest_x0(DDA_elv_cry.cext_hh.values)
                x0_cext_vv = suggest_x0(DDA_elv_cry.cext_vv.values)
                x0_kdp = suggest_x0(DDA_elv_cry.kdp.values)
                scatPoints = {
                    'cbck_h': inv_asinh_transform(
                        knn.fit(pointsCry, asinh_transform(DDA_elv_cry.Ze_h.values, x0_Ze_h)).predict(mcSnowPointsCry),
                        x0_Ze_h
                    ),
                    'cbck_v': inv_asinh_transform(
                        knn.fit(pointsCry, asinh_transform(DDA_elv_cry.Ze_v.values, x0_Ze_v)).predict(mcSnowPointsCry),
                        x0_Ze_v
                    ),
                    'cbck_hv': inv_asinh_transform(
                        knn.fit(pointsCry, asinh_transform(DDA_elv_cry.Ze_hv.values, x0_Ze_hv)).predict(mcSnowPointsCry),
                        x0_Ze_hv
                    ),
                    'cext_h': inv_asinh_transform(
                        knn.fit(pointsCry, asinh_transform(DDA_elv_cry.cext_hh.values, x0_cext_hh)).predict(mcSnowPointsCry),
                        x0_cext_hh
                    ),
                    'cext_v': inv_asinh_transform(
                        knn.fit(pointsCry, asinh_transform(DDA_elv_cry.cext_vv.values, x0_cext_vv)).predict(mcSnowPointsCry),
                        x0_cext_vv
                    ),
                    'kdp': inv_asinh_transform(
                        knn.fit(pointsCry, asinh_transform(DDA_elv_cry.kdp.values, x0_kdp)).predict(mcSnowPointsCry),
                        x0_kdp
                    ),
                }
                # Assign results to mcTable
                mcTable['sZeH'].loc[elv, wl, mcTableCry.index] = scatPoints['cbck_h']
                mcTable['sCextH'].loc[elv, wl, mcTableCry.index] = scatPoints['cext_h']
                mcTable['sCextV'].loc[elv, wl, mcTableCry.index] = scatPoints['cext_v']
                mcTable['sZeV'].loc[elv, wl, mcTableCry.index] = scatPoints['cbck_v']
                mcTable['sZeHV'].loc[elv, wl, mcTableCry.index] = scatPoints['cbck_hv']
                mcTable['sKDP'].loc[elv, wl, mcTableCry.index] = scatPoints['kdp']

    def _handle_aggregates(self, mcTable, mcTableAgg, height):
        """
        Aggregate-specific scattering logic (modularized from calcParticleZe).

        Parameters
        ----------
        mcTable : xarray.Dataset
            Output table to fill.
        mcTableAgg : xarray.Dataset
            Input table for aggregates.
        height : float
            Height of the radar bin.
        """
        import numpy as np
        print('Handling aggregates at height', height, 'with', len(mcTableAgg.mTot), 'particles')
        wls = self.settings['wl'] #scatSet['wls'] if 'wls' in scatSet else [scatSet['wl']]
        elvs = self.settings['elv'] #scatSet['elvs'] if 'elvs' in scatSet else [scatSet['elv']]
        treeAgg = self.treeAgg
        scalingAgg = self.scalingAgg
        DDA_data_agg = self.DDA_data_agg
        nmono_array = self.nmono_array if self.nmono_array is not None else np.ones(len(mcTableAgg.mTot))

        def search_ckdtree(tree, scaling, target):
            scaled_target = np.array(list(target.values())).T * scaling
            idx = tree.query_ball_point(scaled_target, r=1.0)
            return idx

        for wl in wls:
            for elv in elvs:
                target = dict(
                    logmass=np.log10(mcTableAgg.mTot.values),
                    logDmax=np.log10(mcTableAgg.dia.values),
                    wavelength=np.ones(len(mcTableAgg.mTot.values)) * wl,
                )
                search_idx = search_ckdtree(treeAgg, scalingAgg, target)
                variables = ('ZeH', 'CextH', 'CextV', 'ZeV', 'ZeHV', 'KDP')
                result = mcTableAgg.sel(elevation=elv, wavelength=wl).get([f"s{_}" for _ in variables]).copy()
                not_found_particles = []
                for isp, idx in enumerate(search_idx):
                    idx = np.asarray(idx, dtype=np.int64)
                    mask = nmono_array[idx] > 10 if len(idx) > 0 else []
                    idx = idx[mask] if len(idx) > 0 else idx
                    if len(idx) < 1:
                        not_found_particles.append(isp)
                        continue
                    xi = int(mcTableAgg.sMult[isp]) if hasattr(mcTableAgg, 'sMult') else 1
                    for v in variables:
                        if 'Z' in v:
                            result[f"s{v}"][isp] = np.mean(DDA_data_agg[v].data[idx]) * xi
                        else:
                            result[f"s{v}"][isp] = np.mean(DDA_data_agg[v].data[idx]) * xi
                for k, v in result.data_vars.items():
                    # Swap ZeH/ZeV if needed (as in original code)
                    key = k
                    if k == 'sZeV':
                        key = 'sZeH'
                    elif k == 'sZeH':
                        key = 'sZeV'
                    mcTable[key].loc[dict(elevation=elv, wavelength=wl, index=result.index)] = v
        mcTable['sMult'].loc[dict(index=mcTableAgg.index)] = 1 # we need to make sMult 1 because we already multiplied the scattering variables by sMult when we assigned them from the DDA table, so we should not multiply by sMult again when we sum over particles in the same bin

    def _handle_frozen(self, mcTable, mcTableFrozen, beta_std_use, height):
        """
        Frozen-specific scattering logic (modularized from calcParticleZe).

        Parameters
        ----------
        mcTable : xarray.Dataset
            Output table to fill.
        mcTableFrozen : xarray.Dataset
            Input table for frozen particles.
        beta_std_use : float
            Standard deviation for canting angle.
        height : float
            Height of the radar bin.
        """
        # Frozen-specific scattering logic (modularized from calcParticleZe)
        import numpy as np
        #from pytmatrix import refractive
        print('Handling frozen particles at height', height, 'with', len(mcTableFrozen.mTot), 'particles')
        scatSet = self.settings['scatSet'] if isinstance(self.settings, dict) and 'scatSet' in self.settings else self.settings
        wls = self.settings['wl'] #scatSet['wls'] if 'wls' in scatSet else [scatSet['wl']]
        elvs = self.settings['elv'] #scatSet['elvs'] if 'elvs' in scatSet else [scatSet['elv']]
        for wl in wls:
            for elv in elvs:
                reflect_h, reflect_v, refIndex, kdp, Z11Mat, Z12Mat, Z21Mat, Z22Mat, Z33Mat, Z44Mat, S11iMat, S22iMat, sMat = calcScatTmatrix(
                    wl,
                    mcTableFrozen.dia.values / 2 * 1e3,
                    mcTableFrozen.sPhi.values,
                    mcTableFrozen.sRho_tot.values * 1e3 / 1e9,
                    elv,
                    ndgs=30,
                    canting=False,
                    cantingStd=beta_std_use,
                    meanAngle=scatSet.get('beta', 0),
                    safeTmatrix=True
                )
                mcTable['sZeH'].loc[elv, wl, mcTableFrozen.index] = reflect_h
                mcTable['sZeV'].loc[elv, wl, mcTableFrozen.index] = reflect_v
                mcTable['sKDP'].loc[elv, wl, mcTableFrozen.index] = kdp
                mcTable['sCextH'].loc[elv, wl, mcTableFrozen.index] = (S22iMat * 4 * np.pi / (2 * np.pi / wl))
                mcTable['sCextV'].loc[elv, wl, mcTableFrozen.index] = (S11iMat * 4 * np.pi / (2 * np.pi / wl))

    def _handle_melted(self, mcTable, mcTableMelted, height):
        """
        Melted-specific scattering logic (modularized from calcParticleZe).

        Parameters
        ----------
        mcTable : xarray.Dataset
            Output table to fill.
        mcTableMelted : xarray.Dataset
            Input table for melted particles.
        height : float
            Height of the radar bin.
        """
        # Melted-specific scattering logic (modularized from calcParticleZe)
        import numpy as np
        from scattnlay import scattnlay
        #from pytmatrix import refractive
        from mcradar import refractive
        print('Handling melted particles at height', height, 'with', len(mcTableMelted.mTot), 'particles')
        scatSet = self.settings['scatSet'] if isinstance(self.settings, dict) and 'scatSet' in self.settings else self.settings
        wls = self.settings['wl'] #scatSet['wls'] if 'wls' in scatSet else [scatSet['wl']]
        elvs = self.settings['elv'] #scatSet['elvs'] if 'elvs' in scatSet else [scatSet['elv']]
        ice_core = scatSet.get('ice_core', False) # if True, ice core with water coating; if False, water core with ice coating
        for wl in wls:
            for elv in elvs:
                if ice_core:
                    x_water_coating = scatt_param(mcTableMelted.dia / 2 * 1e3, wl)
                    x_ice_core = scatt_param(mcTableMelted.dia_ice_core / 2 * 1e3, wl)
                    m_water = m_water_wl(wl)
                    m_total = np.zeros((2), dtype=complex)
                    m_total[1] = m_water
                    for x_ice, x_water, dia, rho_ice, index in zip(x_ice_core, x_water_coating, mcTableMelted.dia, mcTableMelted.rho_ice_core, mcTableMelted.index):
                        m_ice = refractive.mi(wl, rho_ice)
                        m_total[0] = m_ice
                        x_total = np.array([x_ice, x_water])
                        terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(x_total, m_total)
                        Cext, Csca, Cabs, Cbk = Q2C(np.array([Qext, Qsca, Qabs, Qbk]), dia.values / 2 * 1e3)
                        mcTable['sZeH'].loc[elv, wl, index] = wl ** 4 * Cbk / (np.pi ** 5 * scatSet['K2'])
                        mcTable['sCextH'].loc[elv, wl, index] = Cext
                        mcTable['sZeV'].loc[elv, wl, index] = wl ** 4 * Cbk / (np.pi ** 5 * scatSet['K2'])
                else:
                    x_water_coating = scatt_param(mcTableMelted.dia_water_core / 2 * 1e3, wl)
                    x_ice_core = scatt_param(mcTableMelted.dia / 2 * 1e3, wl)
                    m_water = m_water_wl(wl)
                    m_total = np.zeros((2), dtype=complex)
                    m_total[0] = m_water
                    for x_ice, x_water, dia, rho_ice, index in zip(x_ice_core, x_water_coating, mcTableMelted.dia, mcTableMelted.rho_ice_coat, mcTableMelted.index):
                        m_ice = refractive.mi(wl, rho_ice)
                        m_total[1] = m_ice
                        x_total = np.array([x_water, x_ice])
                        terms, Qext, Qsca, Qabs, Qbk, Qpr, g, Albedo, S1, S2 = scattnlay(x_total, m_total)
                        Cext, Csca, Cabs, Cbk = Q2C(np.array([Qext, Qsca, Qabs, Qbk]), dia.values / 2 * 1e3)
                        mcTable['sZeH'].loc[elv, wl, index] = wl ** 4 * Cbk / (np.pi ** 5 * scatSet['K2'])
                        mcTable['sZeV'].loc[elv, wl, index] = wl ** 4 * Cbk / (np.pi ** 5 * scatSet['K2'])
                        mcTable['sCextH'].loc[elv, wl, index] = Cext

    def _handle_liquid(self, mcTable, mcTableLiquid, beta_std_use, height):
        """
        Liquid-specific scattering logic (modularized from calcParticleZe).

        Parameters
        ----------
        mcTable : xarray.Dataset
            Output table to fill.
        mcTableLiquid : xarray.Dataset
            Input table for liquid particles.
        beta_std_use : float
            Standard deviation for canting angle.
        height : float
            Height of the radar bin.
        """
        # Liquid-specific scattering logic (modularized from calcParticleZe)
        #import numpy as np
        #import pandas as pd
        import xarray as xr
        from scipy import constants
        print('Handling liquid particles at height', height, 'with', len(mcTableLiquid.mTot), 'particles')
        scatSet = self.settings['scatSet'] if isinstance(self.settings, dict) and 'scatSet' in self.settings else self.settings
        #print(scatSet)
        wls = self.settings['wl'] #scatSet['wls'] if 'wls' in scatSet else [scatSet['wl']]
        elvs = self.settings['elv'] #scatSet['elvs'] if 'elvs' in scatSet else [scatSet['elv']]
        for wl in wls:
            #print(wl)
            for elv in elvs:
                freq = (constants.c / (wl * 1e-3)) * 1e-9
                temperature = '283.15'
                scatTable = xr.open_dataset(scatSet['lutPath'] + f'liquid_{temperature}_{freq:.1f}GHz_elv{elv}_canting.nc')
                large = mcTableLiquid.dia.where(mcTableLiquid.dia * 1e3 > scatTable.Dmax.max())
                if large.count() > 0:
                    mcTableLiquid = mcTableLiquid.where(mcTableLiquid.dia * 1e3 < scatTable.Dmax.max(), drop=True)
                scatSel = scatTable.sel(Dmax=mcTableLiquid.dia * 1e3, method='nearest', tolerance=0.2)
                scatSel = scatSel.sel(cantingStd=beta_std_use, method='nearest', tolerance=10)
                mcTable['sZeH'].loc[elv, wl, mcTableLiquid.index] = scatSel['c_bck_h'].values
                mcTable['sZeV'].loc[elv, wl, mcTableLiquid.index] = scatSel['c_bck_v'].values
                mcTable['sKDP'].loc[elv, wl, mcTableLiquid.index] = scatSel['sKDP'].values
                mcTable['sCextH'].loc[elv, wl, mcTableLiquid.index] = scatSel['cext_h'].values
