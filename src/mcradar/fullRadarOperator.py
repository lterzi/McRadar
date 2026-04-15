# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Author: José Dias Neto

import numpy as np
import xarray as xr
from mcradar import *
import matplotlib.pyplot as plt
from mcradar.zeOperator import ZeOperator
from mcradar.spectraOperator import SpectraOperator
from mcradar.kdpOperator import KdpOperator

from scipy.spatial import cKDTree 
#if not sys.warnoptions: # bad form, would not recommend!
#	import warnings
#    warnings.simplefilter("ignore")
import warnings
warnings.filterwarnings('ignore')
debugging=True
reduce_ncores = True

def gen_ckdtree(aggdb, search_radii):
    import time
    start = time.time()
    scaling = np.array([1.0 / search_radii[dim] for dim in search_radii.keys()]) # scale euclidean space for search
    points = np.stack([aggdb[dim] for dim in search_radii.keys()], axis=-1) # sample points out of aggdb
    scaled_points = points * scaling
    tree = cKDTree(scaled_points)
    end = time.time()
    print(f"construction of ckdtree took {end - start}s")
    return tree, scaling

def creatRadarCols(mcTable, dicSettings):
	"""
	Create the Ze and KDP column

	Parameters
	----------
	mcTable: output from getMcSnowTable()
	wls: wavelenght (iterable) [mm]

	Returns
	-------
	mcTable with a empty columns 'sZe*_*' 'sKDP_*' for 
	storing Ze_H and Ze_V and sKDP of one particle of a 
	given wavelength
	"""
	#print(mcTable)
	mcTable['sZeH'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan#.assign_coords(elevation=dicSettings['elv'])
	mcTable['sZeV'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan
	mcTable['sKDP'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan
	mcTable['sZeHV'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan
	mcTable['sZeMultH'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan
	mcTable['sZeMultV'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan
	mcTable['sKDPMult'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan
	mcTable['sZeMultHV'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan
	mcTable['sCextH'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan
	mcTable['sCextV'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan
	mcTable['sCextHMult'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan
	mcTable['sCextVMult'] = mcTable.dia.expand_dims(dim={'elevation':dicSettings['elv'],'wavelength':dicSettings['wl']})*np.nan
	
	return mcTable

def prepare_mcTable(mcTable,dicSettings):
	"""
	Here I am outsourcing the calculations of the different particle species
	"""
	mcTable = creatRadarCols(mcTable, dicSettings)
	if 'm_f' not in mcTable:
		mcTable['m_f'] = mcTable.mTot.copy()*0
	mcTable['frozen_fraction'] = mcTable['m_f']/mcTable.mTot
	if 'm_w' not in mcTable and 'm_r' not in mcTable and 'm_i' not in mcTable:
		mcTable['m_w'] = mcTable.mTot.copy()*0
		mcTable['m_r'] = mcTable.mTot.copy()*0
		mcTable['m_i'] = mcTable.mTot.copy()*0
		mcTable['mass_all_ice'] = mcTable.mTot# mcTable.m_i + mcTable.m_f + mcTable.m_r
	else:
		mcTable['mass_all_ice'] = mcTable.m_i + mcTable.m_f + mcTable.m_r

	# define liquid particles
	liquid_particles = (mcTable.m_w > 0) & (mcTable.mass_all_ice == 0)
	mcTableLiquid = mcTable.where(liquid_particles, drop=True) # remove particles with liquid water present
	
	# define melted particles:
	# for now melted particles are spheres with ice core and water coating, if changed to water core and ice coating, need to change code here
	melted_particle = (mcTable.m_w > 0) & (mcTable.mass_all_ice > 0)
	mcTableMelted = mcTable.where(melted_particle, drop=True) # select only melted particles
	rho_w = 1000
	rho_ice = 917
	
	if dicSettings['scatSet']['ice_core']==True:
		density_ice_core = 3*mcTableMelted.mass_all_ice/(4*np.pi*((mcTableMelted.dia/2)**3-3*mcTableMelted.m_w/(4*np.pi*rho_w))) # calculate reqiured ice density to reach Dmax from ICON
		mcTableMelted['rho_ice_core'] = density_ice_core
		mcTableMelted['dia_ice_core'] = (3*mcTableMelted.mass_all_ice/(rho_ice*4*np.pi))**(1/3)*2
		mcTableMelted['dia_water_cover'] = mcTableMelted.dia - mcTableMelted.dia_ice_core
	else:
		density_ice_core = 3*mcTableMelted.mass_all_ice/(4*np.pi*((mcTableMelted.dia/2)**3-3*mcTableMelted.m_w/(4*np.pi*rho_w))) # calculate reqiured ice density to reach Dmax from ICON
		mcTableMelted['rho_ice_coat'] = density_ice_core
		mcTableMelted['dia_water_core'] = (3*mcTableMelted.m_w/(4*np.pi*rho_w))**(1/3)*2
		mcTableMelted['dia_ice_coat'] = mcTableMelted.dia - mcTableMelted.dia_water_core

	mcTableIce = mcTable.where(mcTable.m_w ==0, drop=True) # remove particles with liquid water present
	mcTableFrozen = mcTableIce.where((mcTableIce['frozen_fraction']>1) & (mcTableIce['sPhi']>0.8),drop=True)
	mcTableUnfrozen = mcTableIce#.where((mcTableIce['frozen_fraction']<=0.8) & (mcTableIce['sPhi']<=0.8),drop=True)
	mcTableCry = mcTableUnfrozen.where(mcTableIce['sNmono']==1,drop=True) # select only cry, only calculate that once!
	mcTableAgg = mcTableUnfrozen.where(mcTableIce['sNmono']>1,drop=True) # select only aggregates
	mcTableIce['rimefraction'] = mcTableIce.m_r/mcTableIce.mass_all_ice
	#rimedparticles = (mcTableIce.rimefraction > 0.8) & (mcTableIce.sPhi > 0.7) & (mcTableIce.sPhi < 1.3) & (mcTableIce.sNmono > 1)
	#mcTableRimed = mcTableIce.where(rimedparticles, drop=True)
	
	# define habit codes to be consistent with the codes of the DDA_data_agg database:
	if False:
		ratioPN = np.round(((mcTableAgg.sNmono - mcTableAgg['sp%pp'])/mcTableAgg.sNmono).values,1)*10+20 # ratio of plates and needles
		ratioPN = np.where(ratioPN==20, 21, ratioPN)
		ratioPN = np.where(ratioPN==22, 23, ratioPN)
		ratioPN = np.where(ratioPN==24, 25, ratioPN)
		ratioPN = np.where(ratioPN==26, 27, ratioPN)
		ratioPN = np.where(ratioPN==28, 29, ratioPN)
		ratioDN = np.round(((mcTableAgg.sNmono - mcTableAgg['sp%dd'])/mcTableAgg.sNmono).values,1)*10+30 # ratio of dendrites and needles
		ratioDN = np.where(ratioDN==30, 31, ratioDN)
		ratioDN = np.where(ratioDN==32, 33, ratioDN)
		ratioDN = np.where(ratioDN==34, 35, ratioDN)
		ratioDN = np.where(ratioDN==36, 37, ratioDN)
		ratioDN = np.where(ratioDN==38, 39, ratioDN)
		ratioPD = np.round(((mcTableAgg.sNmono - mcTableAgg['sp%pp'])/mcTableAgg.sNmono).values,1)*10+40 # ratio of plates and dendrites
		ratioPD = np.where(ratioPD==40, 41, ratioPD)
		ratioPD = np.where(ratioPD==42, 43, ratioPD)
		ratioPD = np.where(ratioPD==44, 45, ratioPD)
		ratioPD = np.where(ratioPD==46, 47, ratioPD)
		ratioPD = np.where(ratioPD==48, 49, ratioPD)

		# define condition for plates and dendrites:
		condPD = mcTableAgg['sp%pp'] + mcTableAgg['sp%dd'] == mcTableAgg.sNmono
		# define condition for plate and needle:
		condPN = (mcTableAgg['sp%dd'] == 0) & (mcTableAgg['sp%pp'] > 0) & (mcTableAgg.sNmono > mcTableAgg['sp%pp'])
		# define condition for dendrite and needle: 
		condDN = (mcTableAgg['sp%pp'] == 0) & (mcTableAgg['sp%dd'] > 0) & (mcTableAgg.sNmono > mcTableAgg['sp%dd'])
		# define condition for needle:
		condN = mcTableAgg['sp%pp'] + mcTableAgg['sp%dd'] == 0
		# define condition for plates:
		condP  = mcTableAgg['sp%pp'] == mcTableAgg.sNmono
		#plates = mcTableAgg.where(mcTableAgg['sp%pp'] == mcTableAgg.sNmono,drop=True)
		# define dendrites:
		condD = mcTableAgg['sp%dd'] == mcTableAgg.sNmono
	
		habit_code = mcTableAgg.habit_code
		habit_code = xr.where(condDN, ratioDN, habit_code)
		habit_code = xr.where(condPN, ratioPN, habit_code)
		habit_code = xr.where(condPD, ratioPD, habit_code)
		habit_code = xr.where(condP, 0, habit_code)
		habit_code = xr.where(condD, 2, habit_code)
		habit_code = xr.where(condN,1,habit_code)
		mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(condDN), ratioDN)#, mcTable['habit_code'])
		mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(condPN), ratioPN)#, mcTable['habit_code'])
		mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(mcTableAgg['sp%pp'] + mcTableAgg['sp%dd'] == mcTableAgg.sNmono), ratioPD)#, mcTable['habit_code'])
		mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(mcTableAgg['sp%pp'] == mcTableAgg.sNmono), 0)#, mcTable['habit_code'])
		mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(mcTableAgg['sp%dd'] == mcTableAgg.sNmono), 2)#, mcTable['habit_code'])
		mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(mcTableAgg['sp%pp'] + mcTableAgg['sp%dd'] == 0), 1)#, mcTable['habit_code'])
		
	mcTableAgg['habit_code'] = mcTableAgg.sNmono.copy()*0
	return mcTableAgg, mcTableCry, mcTableFrozen, mcTableMelted, mcTableLiquid#, mcTableRimed


class RadarSimulation:
    def __init__(self, settings):
        """
        Initialize the simulation with settings (dict or RadarSettings object).
        """
        self.settings = settings
        self.mcTable = None
        self.results = {}
        self.DDA_data_agg = None
        self.DDA_data_cry = None
        self.treeAgg = None
        self.scalingAgg = None
        self.treeCry = None
        self.scalingCry = None
        self.nmono_array = None

    def initialize(self, mcTable):
        """
        Load LUTs, build trees, and prepare the particle table.
        """
        self.mcTable = mcTable
        self.mcTableAgg, self.mcTableCry, self.mcTableFrozen, self.mcTableMelted, self.mcTableLiquid = prepare_mcTable(self.mcTable, self.settings)
        # Load LUTs
        self.DDA_data_agg = xr.open_dataset(self.settings['scatSet']['lutPath']+'all_aggregates_small_kdp.nc')
        self.DDA_data_agg['logmass'] = np.log10(self.DDA_data_agg.mass)
        self.DDA_data_agg['logDmax'] = np.log10(self.DDA_data_agg.Dmax)
        self.DDA_data_cry = xr.open_dataset(self.settings['scatSet']['lutPath']+'scattering_properties_all_crystals_withbetanew.nc') #all_crystals #only_beta2.0000e-01_gamma1.5849e-04_
            
        if 'D_max' in self.DDA_data_cry:
            self.DDA_data_cry = self.DDA_data_cry.rename({'D_max':'Dmax'})

        self.DDA_data_cry = self.DDA_data_cry.to_dataframe()
        #self.DDA_data_cry = xr.open_dataset(self.settings['scatSet']['lutPath']+'stochastic_crystals.nc')
        #self.DDA_data_cry['logmass'] = np.log10(self.DDA_data_cry.mass)
        #self.DDA_data_cry['logDmax'] = np.log10(self.DDA_data_cry.Dmax)
        #self.DDA_data_cry['logar'] = np.log10(self.DDA_data_cry.aspect_ratio)
        # Build trees
        search_radii = dict(
            logmass=abs(np.log10(1) - np.log10(1.05)),
            logDmax=abs(np.log10(1) - np.log10(1.05)),
            wavelength=0.1,
        )
        self.treeAgg, self.scalingAgg = gen_ckdtree(self.DDA_data_agg, search_radii)
        self.nmono_array = self.DDA_data_agg.Nmono.values
        # For crystals (if needed)
        self.treeCry = None
        self.scalingCry = None

    def run(self):
        specXR = xr.Dataset()
        for i, heightEdge0 in enumerate(self.settings['heightRange']):
            if len(self.settings['gridVolume']) > 1:
                vol = self.settings['gridVolume'][i]
            else:
                vol = self.settings['gridVolume']
            heightEdge1 = heightEdge0 + self.settings['heightRes']
            print(f"Processing height bin: {heightEdge0} - {heightEdge1} m, volume: {vol} m^3")
            # Prepare sub-tables for this height bin
            #print(self.mcTable.sHeight.min().values, self.mcTable.sHeight.max().values)
            mcTableTmp = self.mcTable.where((self.mcTable['sHeight']>heightEdge0) & (self.mcTable['sHeight']<=heightEdge1),drop=True)
            #print(mcTableTmp.sHeight.min().values, mcTableTmp.sHeight.max().values)
            #quit()
            mcTableAggTmp = self.mcTableAgg.where((self.mcTableAgg['sHeight']>heightEdge0) & (self.mcTableAgg['sHeight']<=heightEdge1),drop=True)
            mcTableCryTmp = self.mcTableCry.where((self.mcTableCry['sHeight']>heightEdge0) & (self.mcTableCry['sHeight']<=heightEdge1),drop=True)
            mcTableFrozenTmp = self.mcTableFrozen.where((self.mcTableFrozen['sHeight']>heightEdge0) & (self.mcTableFrozen['sHeight']<=heightEdge1),drop=True)
            mcTableMeltedTmp = self.mcTableMelted.where((self.mcTableMelted['sHeight']>heightEdge0) & (self.mcTableMelted['sHeight']<=heightEdge1),drop=True)
            mcTableLiquidTmp = self.mcTableLiquid.where((self.mcTableLiquid['sHeight']>heightEdge0) & (self.mcTableLiquid['sHeight']<=heightEdge1),drop=True)
            if mcTableTmp.vel.any():
                if mcTableTmp.vel.std() > 1:
                    beta_std_use = 90
                else:
                    beta_std_use = self.settings['beta_std']
                # Ze calculation
                mcTableTmp = self.run_ze_operator(mcTableTmp, mcTableAggTmp, mcTableCryTmp, mcTableFrozenTmp, mcTableMeltedTmp, mcTableLiquidTmp, beta_std_use, heightEdge0, heightEdge1)
                # Spectra calculation
                tmpSpecXR = self.run_spectra_operator(mcTableTmp, heightEdge0, heightEdge1, vol)
                # KDP calculation
                tmpKdpXR = self.run_kdp_operator(mcTableTmp, heightEdge0, heightEdge1, vol)
                # Attenuation calculation
                tmpSpecXR = self.run_attenuation_operator(tmpSpecXR)
                # Merge results
                specXR = xr.merge([specXR, tmpSpecXR, tmpKdpXR])
            else:
                print(f"No particles found in height bin: {heightEdge0} - {heightEdge1} m")
        self.results['spectra'] = specXR

    def run_ze_operator(self, mcTableTmp, mcTableAggTmp, mcTableCryTmp, mcTableFrozenTmp, mcTableMeltedTmp, mcTableLiquidTmp, beta_std_use, heightEdge0, heightEdge1):
        # Use ZeOperator class
        ze_operator = ZeOperator(
            self.settings,
            self.DDA_data_agg,
            self.DDA_data_cry,
            self.treeAgg,
            self.scalingAgg,
            self.treeCry,
            self.scalingCry,
            self.nmono_array
        )
        return ze_operator.compute(
            mcTableTmp,
            mcTableAggTmp,
            mcTableCryTmp,
            mcTableFrozenTmp,
            mcTableMeltedTmp,
            mcTableLiquidTmp,
            beta_std_use,
            (heightEdge1 + heightEdge0) / 2
        )

    def run_spectra_operator(self, mcTableTmp, heightEdge0, heightEdge1, vol):
        # Use SpectraOperator class
        k_theta, k_phi, k_r = 0, 0, 0
        spectra_operator = SpectraOperator(
            self.settings['wls'] if 'wls' in self.settings else [self.settings['wl']],
            self.settings['elvs'] if 'elvs' in self.settings else [self.settings['elv']],
            self.settings['velBins'],
            self.settings['velCenterBin'],
            (heightEdge1 + heightEdge0) / 2,
            self.settings['convolute'],
            self.settings['nave'],
            self.settings['noise_pow'],
            self.settings['eps_diss'],
            self.settings['uwind'],
            self.settings['time_int'],
            self.settings['theta'] / 2. / 180. * np.pi,
            k_theta,
            k_phi,
            k_r,
            self.settings['tau']
        )
        tmpSpecXR = spectra_operator.compute(mcTableTmp)
        return tmpSpecXR / vol

    def run_kdp_operator(self, mcTableTmp, heightEdge0, heightEdge1,vol):
        return KdpOperator().compute(mcTableTmp, (heightEdge1+heightEdge0)/2) / vol

    def run_attenuation_operator(self, tmpSpecXR):
        if self.settings['attenuation'] == True:
            tmpSpecXR['att_atm_ice_HH'] = 2*tmpSpecXR.att_ice_HH.cumsum(dim='range') + 2*tmpSpecXR.att_atmo.cumsum(dim='range')
            tmpSpecXR['att_atm_ice_VV'] = 2*tmpSpecXR.att_ice_VV.cumsum(dim='range') + 2*tmpSpecXR.att_atmo.cumsum(dim='range')
            tmpSpecXR.att_atm_ice_HH.attrs['long_name'] = '2 way attenuation at HH polarization'
            tmpSpecXR.att_atm_ice_HH.attrs['unit'] = 'dB'
            tmpSpecXR.att_atm_ice_HH.attrs['comment'] = '2 way attenuation for ice particles and atmospheric gases (N2,O2,H2O). The spectra are divided my this, so to get unattenuated spectra, multiply with this (in linear units)'
            tmpSpecXR.att_atm_ice_HH.attrs['long_name'] = '2 way attenuation at VV polarization'
            tmpSpecXR.att_atm_ice_HH.attrs['unit'] = 'dB'
            tmpSpecXR.att_atm_ice_HH.attrs['comment'] = '2 way attenuation for ice particles and atmospheric gases (N2,O2,H2O). The spectra are divided my this, so to get unattenuated spectra, multiply with this (in linear units)'
            tmpSpecXR['spec_H_att'] = tmpSpecXR.spec_H/(10**(tmpSpecXR.att_atm_ice_HH/10))
            tmpSpecXR['spec_V'] = tmpSpecXR.spec_V/(10**(tmpSpecXR.att_atm_ice_VV/10))
            tmpSpecXR['spec_HV'] = tmpSpecXR.spec_HV/(10**(tmpSpecXR.att_atm_ice_HH/10))
        return tmpSpecXR