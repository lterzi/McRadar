import numpy as np
from scipy import constants

class SpectraOperator:
    def __init__(self,wls, elvs, velBins, velCenterBins, centerHeight,
                convolute, nave, noise_pow, eps_diss, uwind, time_int, theta,
                k_theta, k_phi, k_r,tau):
        """
        Initialize the SpectraOperator with necessary parameters.
        Parameters
        ----------
        wls : array-like
            Wavelengths [mm].
        elvs : array-like
            Elevations [deg].
        
        velBins : array-like
            Velocity bin edges [m/s].
        velCenterBins : array-like
            Velocity bin centers [m/s].
        centerHeight : float or array
            Center height(s) of range gates [m].
        convolute : bool
            If True, apply turbulence and noise convolution.
        nave : array-like
            Number of spectral averages per wavelength.
        noise_pow : array-like
            Noise power per wavelength.
        eps_diss : float
            Eddy dissipation rate [m^2/s^3].
        uwind : array-like
            Wind velocity components [m/s].
        time_int : float
            Radar integration time [s].
        theta : array-like
            Radar beamwidth [rad].
        k_theta, k_phi, k_r : float
            Wind shear in theta, phi, r directions.
        tau : float
            Pulse width.
        """
        # Ensure wls, elvs, nave, noise_pow, theta are always 1D arrays of scalars
        self.wls = np.atleast_1d(wls).flatten()
        self.elvs = np.atleast_1d(elvs).flatten()
        self.nave = np.atleast_1d(nave).flatten()
        self.noise_pow = np.atleast_1d(noise_pow).flatten()
        self.theta = np.atleast_1d(theta).flatten()
        self.velBins = velBins
        self.velCenterBins = velCenterBins
        self.centerHeight = centerHeight
        self.convolute = convolute
        self.nave = nave
        self.noise_pow = noise_pow
        self.eps_diss = eps_diss
        self.uwind = uwind
        self.time_int = time_int   
        self.theta = theta
        self.k_theta = k_theta
        self.k_phi = k_phi
        self.k_r = k_r 
        self.tau = tau
    def compute(self, mcTable, diffAggMono=True, diffParticleTypes=True):
        """
        Main method for multi-frequency spectrogram calculation.
        Parameters
        ----------
        mcTable : xarray.Dataset
            Input table with particle properties and scattering results.
        diffAggMono : bool
            If True, compute separate spectra for monomers and aggregates.
        diffParticleTypes : bool
            If True, compute separate spectra for melted, liquid, ice, and rimed particles.
        Returns
        -------
        specTable : xarray.Dataset
            Dataset containing computed spectra for each category and convolution state.
        """
        import xarray as xr
        specTable = xr.Dataset()

        mcTable['sZeMultH'] = mcTable['sZeH'] * mcTable['sMult']
        mcTable['sZeMultV'] = mcTable['sZeV'] * mcTable['sMult']
        mcTable['sZeMultHV'] = mcTable['sZeHV'] * mcTable['sMult']
        group = mcTable.groupby_bins('vel', self.velBins, labels=self.velCenterBins).sum()

        specTable['spec_H'] = group['sZeMultH'].rename({'vel_bins': 'vel'})
        specTable['spec_V'] = group['sZeMultV'].rename({'vel_bins': 'vel'})
        specTable['spec_HV'] = group['sZeMultHV'].rename({'vel_bins': 'vel'})
        # print(self.wls)
        if self.convolute:
            for wl, th, nv, noise in zip(self.wls, self.theta, self.nave, self.noise_pow):
                #wl = float(wl)
                for elv in self.elvs:
                    #print(elv)
                    #if elv == 30:
                    #    nv = self.nave[3]
                    specTable['spec_H'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_H'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                            noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                    specTable['spec_V'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_V'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                            noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                    specTable['spec_HV'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_HV'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                            noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)

        if diffAggMono:
            mcTableMono = mcTable.where(mcTable['sNmono'] == 1, drop=True)
            mcTableAgg = mcTable.where(mcTable['sNmono'] > 1, drop=True)
            if len(mcTableMono.sPhi) > 0:
                mcTableMono['sZeMultH'] = mcTableMono['sZeH'] * mcTableMono['sMult']
                mcTableMono['sZeMultV'] = mcTableMono['sZeV'] * mcTableMono['sMult']
                mcTableMono['sZeMultHV'] = mcTableMono['sZeHV'] * mcTableMono['sMult']
                groupMono = mcTableMono.groupby_bins('vel', self.velBins, labels=self.velCenterBins).sum()
                specTable['spec_H_Mono'] = groupMono['sZeMultH'].rename({'vel_bins': 'vel'})
                specTable['spec_V_Mono'] = groupMono['sZeMultV'].rename({'vel_bins': 'vel'})
                specTable['spec_HV_Mono'] = groupMono['sZeMultHV'].rename({'vel_bins': 'vel'})
                if self.convolute:
                    for wl, th, nv, noise in zip(self.wls, self.theta, self.nave, self.noise_pow):
                        for elv in self.elvs:
                            #if elv == 30:
                            #    nv = self.nave[3]
                            specTable['spec_H_Mono'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_H_Mono'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_V_Mono'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_V_Mono'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_HV_Mono'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_HV_Mono'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)

            if len(mcTableAgg.sPhi) > 0:
                mcTableAgg['sZeMultH'] = mcTableAgg['sZeH'] * mcTableAgg['sMult']
                mcTableAgg['sZeMultV'] = mcTableAgg['sZeV'] * mcTableAgg['sMult']
                mcTableAgg['sZeMultHV'] = mcTableAgg['sZeHV'] * mcTableAgg['sMult']
                groupAgg = mcTableAgg.groupby_bins('vel', self.velBins, labels=self.velCenterBins).sum()
                specTable['spec_H_Agg'] = groupAgg['sZeMultH'].rename({'vel_bins': 'vel'})
                specTable['spec_V_Agg'] = groupAgg['sZeMultV'].rename({'vel_bins': 'vel'})
                specTable['spec_HV_Agg'] = groupAgg['sZeMultHV'].rename({'vel_bins': 'vel'})
                if self.convolute:
                    for wl, th, nv, noise in zip(self.wls, self.theta, self.nave, self.noise_pow):
                        for elv in self.elvs:
                            #if elv == 30:
                            #    nv = self.nave[3]
                            specTable['spec_H_Agg'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_H_Agg'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_V_Agg'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_V_Agg'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_HV_Agg'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_HV_Agg'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)

        if diffParticleTypes:
            melted_particle = (mcTable.m_w > 0) & (mcTable.mass_all_ice > 0)
            mcTableMelted = mcTable.where(melted_particle, drop=True)
            liquid_particles = (mcTable.m_w > 0) & (mcTable.mass_all_ice == 0)
            mcTableLiquid = mcTable.where(liquid_particles, drop=True)
            mcTableIce = mcTable.where(mcTable.m_w == 0, drop=True)
            mcTableIce['rimefraction'] = mcTableIce.m_r / mcTableIce.mass_all_ice
            rimedparticles = (mcTableIce.rimefraction > 0.5) & (mcTableIce.sPhi > 0.7) & (mcTableIce.sPhi < 1.3)
            mcTableRimed = mcTableIce.where(rimedparticles, drop=True)
            if len(mcTableMelted.sPhi) > 0:
                mcTableMelted['sZeMultH'] = mcTableMelted['sZeH'] * mcTableMelted['sMult']
                mcTableMelted['sZeMultV'] = mcTableMelted['sZeV'] * mcTableMelted['sMult']
                mcTableMelted['sZeMultHV'] = mcTableMelted['sZeHV'] * mcTableMelted['sMult']
                groupMelted = mcTableMelted.groupby_bins('vel', self.velBins, labels=self.velCenterBins).sum()
                specTable['spec_H_Melted'] = groupMelted['sZeMultH'].rename({'vel_bins': 'vel'})
                specTable['spec_V_Melted'] = groupMelted['sZeMultV'].rename({'vel_bins': 'vel'})
                specTable['spec_HV_Melted'] = groupMelted['sZeMultHV'].rename({'vel_bins': 'vel'})
                if self.convolute:
                    for wl, th, nv, noise in zip(self.wls, self.theta, self.nave, self.noise_pow):
                        for elv in self.elvs:
                            #if elv == 30:
                            #    nv = self.nave[3]
                            specTable['spec_H_Melted'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_H_Melted'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_V_Melted'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_V_Melted'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_HV_Melted'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_HV_Melted'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
            if len(mcTableLiquid.sPhi) > 0:
                mcTableLiquid['sZeMultH'] = mcTableLiquid['sZeH'] * mcTableLiquid['sMult']
                mcTableLiquid['sZeMultV'] = mcTableLiquid['sZeV'] * mcTableLiquid['sMult']
                mcTableLiquid['sZeMultHV'] = mcTableLiquid['sZeHV'] * mcTableLiquid['sMult']
                groupLiquid = mcTableLiquid.groupby_bins('vel', self.velBins, labels=self.velCenterBins).sum()
                specTable['spec_H_Liquid'] = groupLiquid['sZeMultH'].rename({'vel_bins': 'vel'})
                specTable['spec_V_Liquid'] = groupLiquid['sZeMultV'].rename({'vel_bins': 'vel'})
                specTable['spec_HV_Liquid'] = groupLiquid['sZeMultHV'].rename({'vel_bins': 'vel'})
                if self.convolute:
                    for wl, th, nv, noise in zip(self.wls, self.theta, self.nave, self.noise_pow):
                        for elv in self.elvs:
                            #if elv == 30:
                            #    nv = self.nave[3]
                            specTable['spec_H_Liquid'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_H_Liquid'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_V_Liquid'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_V_Liquid'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_HV_Liquid'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_HV_Liquid'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
            if len(mcTableIce.sPhi) > 0:
                mcTableIce['sZeMultH'] = mcTableIce['sZeH'] * mcTableIce['sMult']
                mcTableIce['sZeMultV'] = mcTableIce['sZeV'] * mcTableIce['sMult']
                mcTableIce['sZeMultHV'] = mcTableIce['sZeHV'] * mcTableIce['sMult']
                groupIce = mcTableIce.groupby_bins('vel', self.velBins, labels=self.velCenterBins).sum()
                specTable['spec_H_Ice'] = groupIce['sZeMultH'].rename({'vel_bins': 'vel'})
                specTable['spec_V_Ice'] = groupIce['sZeMultV'].rename({'vel_bins': 'vel'})
                specTable['spec_HV_Ice'] = groupIce['sZeMultHV'].rename({'vel_bins': 'vel'})
                if self.convolute:
                    for wl, th, nv, noise in zip(self.wls, self.theta, self.nave, self.noise_pow):
                        for elv in self.elvs:
                            #if elv == 30:
                            #    nv = self.nave[3]
                            specTable['spec_H_Ice'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_H_Ice'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_V_Ice'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_V_Ice'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_HV_Ice'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_HV_Ice'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
            if len(mcTableRimed.sPhi) > 0:
                mcTableRimed['sZeMultH'] = mcTableRimed['sZeH'] * mcTableRimed['sMult']
                mcTableRimed['sZeMultV'] = mcTableRimed['sZeV'] * mcTableRimed['sMult']
                mcTableRimed['sZeMultHV'] = mcTableRimed['sZeHV'] * mcTableRimed['sMult']
                groupRime = mcTableRimed.groupby_bins('vel', self.velBins, labels=self.velCenterBins).sum()
                specTable['spec_H_Rimed'] = groupRime['sZeMultH'].rename({'vel_bins': 'vel'})
                specTable['spec_V_Rimed'] = groupRime['sZeMultV'].rename({'vel_bins': 'vel'})
                specTable['spec_HV_Rimed'] = groupRime['sZeMultHV'].rename({'vel_bins': 'vel'})
                if self.convolute:
                    for wl, th, nv, noise in zip(self.wls, self.theta, self.nave, self.noise_pow):
                        for elv in self.elvs:
                            #if elv == 30:
                           #     nv = self.nave[3]
                            specTable['spec_H_Rimed'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_H_Rimed'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_V_Rimed'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_V_Rimed'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)
                            specTable['spec_HV_Rimed'].loc[:, elv, wl] = self.convoluteSpec(specTable['spec_HV_Rimed'].sel(wavelength=wl, elevation=elv).fillna(0), wl, self.velCenterBins, self.eps_diss,
                                                                                noise, nv, th, self.uwind, self.time_int, self.centerHeight, self.k_theta, self.k_phi, self.k_r, self.tau)

        specTable = specTable.expand_dims(dim='range').assign_coords(range=[self.centerHeight])
        return specTable

    @staticmethod
    def getVelIntSpec(mcTable, mcTable_binned, variable):
        """
        Calculates the integrated reflectivity for each velocity bin.
        Parameters        
        ----------
        mcTable : xarray.Dataset
            Input table with particle properties and scattering results.
        mcTable_binned : xarray.Dataset
            The same table but with an additional coordinate for velocity bins.
        variable : str
            The variable to integrate (e.g., 'sZeMultH').
        Returns
        -------
        mcTableVelIntegrated : xarray.Dataset
            Dataset containing the integrated variable for each velocity bin.
        """
        mcTableVelIntegrated = mcTable.groupby(mcTable_binned)[variable].agg(['sum'])
        return mcTableVelIntegrated

    @staticmethod
    def convoluteSpec(spec, wl, vel, eps, noise_pow, nave, theta, u_wind, time_int, height, k_theta, k_phi, k_r, tau, PSD=False):
        """
        Convolutes the spectrum with turbulence and adds random noise
        
        Parameters
        ----------
        spec: spectral data (xarray.dataarray) [mm^6/m^3]
        wl: wavelength in mm
        vel: Doppler velocity array (m/s)
        eps: eddy dissipation rate m/s²
        noise_pow: radar noise power [mm^6/m^3] 
        nave: number of spectral averages
        theta: beamwidth of radar in rad
        u_wind: horizontal wind velocity in m/s
        time_int: integration time of radar in sec 
        height: centre height of range gate    
        k_theta: wind shear in theta direction (when looking zenith this is in x direction)
        k_phi: wind shear in phi direction (when looking zenith this is in y direction)
        k_r: wind shear in r direction (when looking zenith this is in z direction)
        tau: pulse width
        
        Returns
        -------
        spectrum with added noise and turbulence broadening as np.array 
        """
        
        L_s = u_wind*time_int + 2*height*np.sin(theta)
        L_lam = (wl*1e-3)/2    
        sigma_t2 = 3/4*(eps/(2*np.pi))**(2/3)*( L_s**(2/3) - L_lam**(2/3) ) # turbulence broadening term (3*kolmogorov/2, where kolmogorov=0.5, therefore 3/4)
        
        #finite beamwidth broadening
        sigma_b2 = u_wind**2*theta**2/2.76 
        # windshear broadening according to Doviak and Zrnic 1993
        sigma_theta = theta/(4*np.sqrt(np.log(2)))
        sigma_stheta = height*sigma_theta*k_theta # x-component (in case of 90° elevation)
        sigma_sphi = height*sigma_theta*k_phi # y-component (in case of 90° elevation)
        sigma_sr = 0.35*constants.c*tau/2*k_r # z-component (in case of 90° elevation)
        sigma_s2 = sigma_stheta**2 + sigma_sphi**2 + sigma_sr**2
        
        # all broadening terms  
        specBroad = np.sqrt(sigma_t2 + sigma_b2 + sigma_s2)
        # Turbulence convolution
        spec_turb = SpectraOperator.convoluteBroadfft(spec, vel, specBroad)
        # Add noise
        spec_noise = SpectraOperator.convoluteNoise(spec_turb, vel, noise_pow, nave)
        return spec_noise

    @staticmethod
    def convoluteBroadfft(spec, vel, specBroad):
        """
        Use FFT to convolve the spectrum with a Gaussian broadening kernel (turbulence).
        Parameters        
        ----------
        spec: spectral data (xarray.dataarray) [mm^6/m^3]
        vel: Doppler velocity array (m/s)
        specBroad: standard deviation of the Gaussian broadening kernel (m/s)
        Returns
        -------
        spectrum convolved with turbulence broadening as np.array
        """
        import numpy as np
        import scipy.signal as sig
        #spec = spec.values
        prefactor_turb = 1.0 / (np.sqrt(2.0 * np.pi) * specBroad)
        turb = np.zeros(len(vel))
        dv = np.diff(vel)[0]
        for i in range(len(vel)):
            turb[i] = np.exp(-1 * (vel[i] - 0) ** 2.0 / (2.0 * specBroad ** 2.0)) * dv
        spec_turb = sig.fftconvolve(spec.values.flatten(), turb, mode='same')
        return spec_turb * prefactor_turb

    @staticmethod
    def convoluteNoise(spec, vel, noise_pow, nave):
        """
        Add random noise to the spectrum.
        Parameters        
        ----------
        spec: spectral data (xarray.dataarray) [mm^6/m^3]
        vel: Doppler velocity array (m/s)
        noise_pow: radar noise power [mm^6/m^3]
        nave: number of spectral averages
        Returns
        -------
        spectrum with added noise as np.array
        """
        import numpy as np
        dv = np.diff(vel)[0]
        Ni = noise_pow / (len(vel) * dv)
        random_numbers = np.random.uniform(size=len(vel) * nave)
        S_bin_noise = np.zeros(len(vel))
        for iave in range(nave):
            S_bin_noise = S_bin_noise + (-np.log(random_numbers[iave * (len(vel)) : ((iave + 1) * len(vel))]) * (spec + np.ones(len(vel)) * Ni))
        return S_bin_noise / nave
    
