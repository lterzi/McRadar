# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from glob import glob
import numpy as np
from scipy import constants
import pandas as pd

class RadarSettings:
    def __init__(self, PSD=False, dataPath=None, atmoFile=None, elv=None, nfft=512,
                 convolute=True, nave=None, noise_pow=None, theta=None, time_int=2.0, tau=143e-9,
                 uwind=10.0, eps_diss=1e-6, k_theta=None, k_phi=None, k_r=None, shear_height0=0, shear_height1=0,
                 maxVel=3, minVel=-3, velVec=None, freq=None, maxHeight=5500, minHeight=0,
                 heightRes=50, gridBaseArea=1, gridVolume=1, attenuation=False, onlyIce=True,
                 beta=0, beta_std=0, scatSet=None):
        
        """
        This function defines the settings for starting the 
        calculation.
        
        Parameters
        ----------
        dataPath: path to the output from McSnow (mandaroty)
        elv: radar elevation (default = 90) [°]
        nfft: number of fourier decomposition (default = 512) 
        maxVel: maximum fall velocity (default = 3) [m/s]
        minVel: minimum fall velocity (default = -3) [m/s]
        convolute: if True, the spectrum will be convoluted with turbulence and random noise (default = True)
        nave: number of spectral averages (default = 10, 20, 28, 90 for X-Band, Ka-Band, W-Band and pol. W-Band), needed only if convolute == True
        noise_pow: radar noise power [mm^6/m^3] (default = -40 dB), needed only if convolute == True
        theta: beamwidth of radar, in degree (will later be transformed into rad)
        time_int: integration time of radar in sec, needed only if convolute == True
        tau: pulse width of radar in seconds (default: 143ns, which is the one used in Ka-Band radar), needed only if convolute == True
        uwind: x component of wind velocity in m/s (horizontal wind), needed only if convolute == True
        eps_diss: eddy dissipation rate, m/s^2, needed only if convolute == True
        k_theta: wind shear in theta direction (when looking zenith this is in x direction). Needs to be provided with shear_height
        k_phi: wind shear in phi direction (when looking zenith this is in y direction) Needs to be provided with shear_height
        k_r: wind shear in r direction (when looking zenith this is in z direction) Needs to be provided with shear_height
        shear_height0: bottom height of wind shear zone
        shear_height1: top height of wind shear zone
        freq: radar frequency (default = 9.5e9, 35e9, 95e9) [Hz]
        maxHeight: maximum height (default = 5500) [m]
        minHeight: minimun height (default = 0) [m]
        heightRes: resolution of the height bins (default = 50) [m]
        gridBaseArea: area of the grid base (default = 1) [m^2]
        attenuation: get path integrated attenuation based on water vapour, O2 and H2O. This uses PAMTRA
        onlyIce: if the atmo file goes to warmer temperatures, only go until 0°C and remove everything that is warmer!
        beta: wobbling angle of particles, default: 0°
        beta_std: standard deviation of the wobbling angle of particles, default: 0°. For the wobbling of particles, a normal distribution with mean beta and std beta_std is used.
        scatSet: dictionary that defines the settings for the scattering calculations
        scatSet['mode']: string that defines the scattering mode. Valid values are
                            - Tmatrix -> pytmatrix calculations for each superparticle
                            - DDA -> this mode uses DDA table. Possible frequencies: 9.6GHz, 35.5GHz, 94.0GHz. Selection is based on mass, ar and size. 
                                    Columnar or plate-like scattering table will be chosen depending on the aspect ratio of the particles. You need to specify the path to the LUT.  
        scatSet['K2']: dielectric constant of the particles, default: 0.93
        scatSet['selmode']: string that defines the selection mode for the DDA database. Valid entries are 
                            - KNeighborsRegressor: this mode uses the KNeighborsRegressor from sklearn. The n_neighbors closest neighbors in Dmax, aspect ratio and mass are selected 
                                                    and the corresponding scattering properties are averaged based on the inverse distance of the neighbors.  
                            - NearestNeighbors: the n_neighbors closest neighbors in Dmax, aspect ratio and mass are selected and the scattering properties of these points are averaged.
                            - radius: this mode uses the radius_neighbors from sklearn. All neighbors (Dmax, aspect ratio, mass) within the predefined radius are selected and the scattering properties are averaged.
        scatSet['n_neighbors']: number of neighbors to use for the KNeighborsRegressor
        scatSet['radius']: radius in which the nearest neighbours are selected when scatSet['selmode'] is set to radius.
        scatSet['lutPath']: path to where the DDA calculations are stored. This is only needed when scatSet['mode'] is set to DDA.
        scatSet['ndgs']: number of division points used to integrate over the particle surface (default = 30) when using Tmatrix as scattering mode
        Returns
        -------
        dicSettings: dictionary with all parameters
        for starting the caculations
        """
        
        
        
        
        # Set defaults for mutable types
        self.nave = nave if nave is not None else np.array([20])
        self.noise_pow = noise_pow if noise_pow is not None else np.array([-50])
        self.theta = theta if theta is not None else np.array([0.6])
        self.k_theta = k_theta if k_theta is not None else np.array([0])
        self.k_phi = k_phi if k_phi is not None else np.array([0])
        self.k_r = k_r if k_r is not None else np.array([0])
        self.freq = freq if freq is not None else np.array([9.6e9])
        self.elv = elv if elv is not None else np.asarray([90])

        self.scatSet = scatSet if scatSet is not None else {
            'mode':'DDA', 'selmode':'KNeighborsRegressor', 'n_neighbors':5, 'radius':1e-10,
            'safeTmatrix':True, 'K2':0.93, 'orientational_avg':False, 'ice_core':True
        }
        # Store all other parameters as attributes
        self.eps_diss = eps_diss
        self.PSD = PSD
        self.dataPath = dataPath
        self.atmoFile = atmoFile
        self.nfft = nfft
        self.convolute = convolute
        self.time_int = time_int
        self.tau = tau
        self.uwind = uwind
        self.shear_height0 = shear_height0
        self.shear_height1 = shear_height1
        self.maxVel = maxVel
        self.minVel = minVel
        self.velVec = velVec
        self.maxHeight = maxHeight
        self.minHeight = minHeight
        self.heightRes = heightRes
        self.gridBaseArea = gridBaseArea
        self.gridVolume = gridVolume
        self.attenuation = attenuation
        self.onlyIce = onlyIce
        self.beta = beta
        self.beta_std = beta_std

        # Call a method to process and validate settings
        self._process_settings()

    def _process_settings(self):
        # Validate that freq, nave, theta, and noise_pow have the same length
        freq_len = len(np.atleast_1d(self.freq))
        for arr, name in [
            (self.nave, 'nave'),
            (self.theta, 'theta'),
            (self.noise_pow, 'noise_pow')
        ]:
            if len(np.atleast_1d(arr)) != freq_len:
                raise ValueError(f"Length mismatch: freq has {freq_len} elements but {name} has {len(np.atleast_1d(arr))} elements. All must match.")
        # Fill in missing scatSet keys
            for k, v in [('mode', 'DDA'), ('safeTmatrix', False), ('K2', 0.93), ('ndgs', 30),
                         ('n_neighbors', 5), ('radius', 1e-10), ('selmode', 'KNeighborsRegressor'),
                         ('orientational_avg', False), ('ice_core', True)]:
                if k not in self.scatSet:
                    self.scatSet[k] = v

            if self.dataPath is not None:
                del_v = (self.maxVel - self.minVel) / self.nfft
                self.settings = {
                    'dataPath': self.dataPath,
                    'elv': self.elv,
                    'freq': self.freq,
                    'wl': (constants.c / self.freq) * 1e3,
                    'maxHeight': self.maxHeight,
                    'minHeight': self.minHeight,
                    'heightRes': self.heightRes,
                    'heightRange': np.arange(self.minHeight, self.maxHeight, self.heightRes),
                    'gridBaseArea': self.gridBaseArea,
                    'gridVolume': self.gridVolume,
                    'scatSet': self.scatSet,
                    'convolute': self.convolute,
                    'attenuation': self.attenuation,
                    'nave': self.nave,
                    'eps_diss': self.eps_diss,
                    'theta': self.theta,
                    'time_int': self.time_int,
                    'uwind': self.uwind,
                    'tau': self.tau,
                    'k_theta': self.k_theta,
                    'k_phi': self.k_phi,
                    'k_r': self.k_r,
                    'shear_height0': self.shear_height0,
                    'shear_height1': self.shear_height1,
                    'onlyIce': self.onlyIce,
                    'beta': self.beta,
                    'beta_std': self.beta_std,
                }
                if self.velVec is not None and hasattr(self.velVec, 'any') and self.velVec.any():
                    self.settings['velBins'] = self.velVec
                    self.settings['velCenterBin'] = self.velVec[0:-1] + np.diff(self.velVec) / 2.
                    self.settings['velRes'] = np.diff(self.velVec)[0]
                    self.settings['nfft'] = len(self.velVec) - 1
                    self.settings['noise_pow'] = (10 ** (self.noise_pow / 10)) * (self.settings['nfft'] * self.settings['velRes'])
                else:
                    self.settings['velRes'] = (self.maxVel - self.minVel) / self.nfft
                    velBins = np.arange(self.minVel, self.maxVel, self.settings['velRes'])
                    velCenterBin = velBins[0:-1] + np.diff(velBins) / 2.
                    self.settings['velBins'] = velBins
                    self.settings['velCenterBin'] = velCenterBin
                    self.settings['nfft'] = self.nfft
                    self.settings['maxVel'] = self.maxVel
                    self.settings['minVel'] = self.minVel
                    self.settings['noise_pow'] = (10 ** (self.noise_pow / 10)) * (self.settings['nfft'] * self.settings['velRes'])
                if self.onlyIce:
                    print(self.atmoFile)
                    if self.atmoFile is not None:
                        atmo = np.loadtxt(self.atmoFile)
                        height = atmo[:, 0]
                        temp = atmo[:, 2]
                        atmoPD = pd.DataFrame(data=temp, index=height, columns=['temp'])
                        atmoPD.index.name = 'range'
                        atmoPD['press'] = atmo[:, 3]
                        atmoPD['relHum'] = atmo[:, 6]
                        atmoXR = atmoPD.to_xarray()
                        atmoReindex = atmoXR.reindex({'range': self.settings['heightRange'] + self.settings['heightRes'] / 2}, method='nearest')
                        self.settings['temp'] = atmoReindex.temp
                        self.settings['relHum'] = atmoReindex.relHum
                        self.settings['press'] = atmoReindex.press
                    else:
                        raise FileNotFoundError('since you want to check for melted particles (onlyIce==True) you need to give an atmoFile as input.')
            elif self.PSD:
                del_v = (self.maxVel - self.minVel) / self.nfft
                self.settings = {
                    'elv': self.elv,
                    'nfft': self.nfft,
                    'maxVel': self.maxVel,
                    'minVel': self.minVel,
                    'velRes': (self.maxVel - self.minVel) / self.nfft,
                    'freq': self.freq,
                    'wl': (constants.c / self.freq) * 1e3,
                    'maxHeight': self.maxHeight,
                    'minHeight': self.minHeight,
                    'heightRes': self.heightRes,
                    'heightRange': np.arange(self.minHeight, self.maxHeight, self.heightRes),
                    'gridBaseArea': self.gridBaseArea,
                    'scatSet': self.scatSet,
                    'convolute': self.convolute,
                    'nave': self.nave,
                    'noise_pow': (10 ** (self.noise_pow / 10)) * (self.nfft * del_v),
                    'eps_diss': self.eps_diss,
                    'theta': self.theta,
                    'time_int': self.time_int,
                    'uwind': self.uwind,
                    'tau': self.tau,
                    'k_theta': self.k_theta,
                    'k_phi': self.k_phi,
                    'k_r': self.k_r,
                    'shear_height0': self.shear_height0,
                    'shear_height1': self.shear_height1,
                    'attenuation': self.attenuation,
                    'onlyIce': self.onlyIce,
                    'beta': self.beta,
                    'beta_std': self.beta_std,
                }
                velBins = np.arange(self.minVel, self.maxVel, self.settings['velRes'])
                velCenterBin = velBins[0:-1] + np.diff(velBins) / 2.
                self.settings['velBins'] = velBins
                self.settings['velCenterBin'] = velCenterBin
            else:
                raise ValueError('No valid data are provided. Either give the path to the McSnow output (use the dataPath parameter e.g. RadarSettings(dataPath="/data/path/.") or set PSD=True')

            if self.attenuation:
                if not (('temp' in self.settings) and ('relHum' in self.settings) and ('press' in self.settings)):
                    if self.atmoFile is not None:
                        atmo = np.loadtxt(self.atmoFile)
                        height = atmo[:, 0]
                        temp = atmo[:, 2]
                        atmoPD = pd.DataFrame(data=temp, index=height, columns=['temp'])
                        atmoPD.index.name = 'range'
                        atmoPD['press'] = atmo[:, 3]
                        atmoPD['relHum'] = atmo[:, 6]
                        atmoXR = atmoPD.to_xarray()
                        atmoReindex = atmoXR.reindex({'range': self.settings['heightRange'] + self.settings['heightRes'] / 2}, method='nearest')
                        self.settings['temp'] = atmoReindex.temp
                        self.settings['relHum'] = atmoReindex.relHum
                        self.settings['press'] = atmoReindex.press
                    else:
                        raise FileNotFoundError(['since you want to do the attenuation correction you need to give an atmoFile as input.'])

            if self.scatSet['mode'] == 'DDA':
                print(self.scatSet)
                print(self.scatSet['selmode'])
                print('you selected DDA as scattering mode. The scattering properties are selected from a LUT by choosing the closest neighbors in Dmax, aspect ratio and mass using the method {}'.format(self.scatSet['selmode']))
                if 'lutPath' not in self.scatSet:
                    raise FileNotFoundError('with this scattering mode ', self.scatSet['mode'], 'a valid path to the scattering LUT is required.')
                elif not os.path.exists(self.scatSet['lutPath']):
                    raise FileNotFoundError(self.scatSet['lutPath'], 'is not valid, check your settings')

    # Optionally, add methods to export settings as dict, load LUTs, etc.
    def as_dict(self):
        return self.__dict__