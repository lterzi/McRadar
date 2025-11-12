# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Author: José Dias Neto

import numpy as np
import xarray as xr
from mcradar import *
import matplotlib.pyplot as plt
from mcradar.tableOperator import creatRadarCols
import time
import multiprocessing
from multiprocessing import Process, Queue
import sys
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

def getRadarParParallel(heightEdge0,mcTable,mcTableAgg,mcTableCry,mcTableFrozen,mcTableMelted,dicSettings,treeAgg,scalingAgg,treeCry,scalingCry,DDA_data_agg,DDA_data_cry):#heightRes,wl,elv,ndgsVal,scatSet,velBins,velCenterBin,convolute,nave,noise_pow,eps_diss,uwind,time_int,theta,tau):
	if len(dicSettings['gridBaseArea']) > 1:
			vol = dicSettings['gridBaseArea'][i] * dicSettings['heightRes']
	else:
		vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']
	heightEdge1 = heightEdge0 + dicSettings['heightRes']

	print('Range: from {0} to {1}'.format(heightEdge0, heightEdge1))
	mcTableTmp = mcTable.where((mcTable['sHeight']>heightEdge0) &
				 					(mcTable['sHeight']<=heightEdge1),drop=True)
	
	if mcTableTmp.vel.any():
		mcTableAggTmp = mcTableAgg.where((mcTableAgg['sHeight']>heightEdge0) &
							(mcTableAgg['sHeight']<=heightEdge1),drop=True)
		mcTableCryTmp = mcTableCry.where((mcTableCry['sHeight']>heightEdge0) &
								(mcTableCry['sHeight']<=heightEdge1),drop=True)
		mcTableFrozenTmp = mcTableFrozen.where((mcTableFrozen['sHeight']>heightEdge0) &
								(mcTableFrozen['sHeight']<=heightEdge1),drop=True)
		mcTableMeltedTmp = mcTableMelted.where((mcTableMelted['sHeight']>heightEdge0) &
								(mcTableMelted['sHeight']<=heightEdge1),drop=True)
		mcTableLiquidTmp = mcTableLiquid.where((mcTableLiquid['sHeight']>heightEdge0) &
								(mcTableLiquid['sHeight']<=heightEdge1),drop=True)
		#- get the scattering properties for each particle, we have separate tables for aggregates and crystals
		mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'], mcTableTmp,mcTableAggTmp,mcTableCryTmp,mcTableFrozenTmp,mcTableMeltedTmp,mcTableLiquidTmp,
							   dicSettings['scatSet'],dicSettings['beta'],dicSettings['beta_std'],treeAgg,scalingAgg,treeCry,scalingCry,DDA_data_agg,DDA_data_cry, ice_core=dicSettings['scatSet']['ice_core'])#,height=(heightEdge1+heightEdge0)/2)
		#- get the spectra, there is the possibility to add shear, but I have not implemented it yet
		k_theta, k_phi, k_r = 0,0,0
		tmpSpecXR = getMultFrecSpec(dicSettings['wl'], dicSettings['elv'],mcTableTmp, dicSettings['velBins'],
									dicSettings['velCenterBin'], (heightEdge1+heightEdge0)/2,dicSettings['convolute'],dicSettings['nave'],dicSettings['noise_pow'],
									dicSettings['eps_diss'], dicSettings['uwind'],dicSettings['time_int'], dicSettings['theta']/2./180.*np.pi,
									k_theta,k_phi,k_r, dicSettings['tau'])
		tmpSpecXR = tmpSpecXR/vol
		tmpKdpXR =  getIntKdp(mcTableTmp,(heightEdge1+heightEdge0)/2)
		tmpSpecXR = xr.merge([tmpSpecXR, tmpKdpXR/vol])
		
		if dicSettings['attenuation'] == True:
			
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
	else:
		print('empty dataset at this height range')


	

def fullRadarParallel(dicSettings, mcTable):
	"""
	Calculates the radar variables over the entire range

	Parameters
	----------
	dicSettings: a dictionary with all settings output from loadSettings()
	mcTable: McSnow data output from getMcSnowTable()

	Returns
	-------
	specXR: xarray dataset with the spectra(range, vel) and KDP(range)
	"""


	mcTable = creatRadarCols(mcTable, dicSettings)
	t0 = time.time()
	att_atm0 = 0.; att_ice_HH0=0.; att_ice_VV0=0.
	DDA_data_agg = xr.open_dataset(dicSettings['scatSet']['lutPath']+'all_aggregates_small_kdp.nc')
	DDA_data_agg['logmass'] = np.log10(DDA_data_agg.mass)
	DDA_data_agg['logDmax'] = np.log10(DDA_data_agg.Dmax)
	DDA_data_cry = xr.open_dataset(dicSettings['scatSet']['lutPath']+'all_crystals_allazi_withradar.nc')
	DDA_data_cry['logmass'] = np.log10(DDA_data_cry.mass)
	DDA_data_cry['logDmax'] = np.log10(DDA_data_cry.Dmax)
	DDA_data_cry['logar'] = np.log10(DDA_data_cry.aspect_ratio)

	# separate into species;
	mcTableAgg, mcTableCry, mcTableFrozen, mcTableMelted, mcTableLiquid = prepare_mcTable(mcTable,dicSettings)
	if dicSettings['beta_std'] == 0:
		elevation_radius = 1
	else:
		elevation_radius = dicSettings['beta']
	search_radii = dict(
						logmass=abs(np.log10(1) - np.log10(1.05)), # 2 %
						logDmax=abs(np.log10(1) - np.log10(1.05)), # 5 %
						elevation = elevation_radius,
						wavelength = 0.1,
						#habit = 7, # 10 % for habit code (which works because habit=0 for plates, so 0 tolerance, habit = 1 for dendrites, so 10% tolerance will not shift to other habit, only if habit = 20 or large, then 10% will be a int number)
						)
	print(DDA_data_agg, search_radii)
	treeAgg, scalingAgg = gen_ckdtree(DDA_data_agg, search_radii)
	
	search_radii = dict(
						logmass=abs(np.log10(1) - np.log10(1.1)), # 2 %
						logDmax=abs(np.log10(1) - np.log10(1.1)), # 5 %
						logar = abs(np.log10(1) - np.log10(1.1)), # 2 %
						elevation = elevation_radius,
						wavelength = 0.1,
						)
	treeCry, scalingCry = gen_ckdtree(DDA_data_cry, search_radii)

	t0 = time.time()
	n_cores = 8#multiprocessing.cpu_count()
	print(n_cores)
	pool = multiprocessing.Pool(n_cores)

	args = [(h, mcTable,mcTableAgg,mcTableCry,mcTableFrozen,mcTableMelted, dicSettings,treeAgg,scalingAgg,treeCry,scalingCry,DDA_data_agg,DDA_data_cry) for h in dicSettings['heightRange']]

	result =  pool.starmap(getRadarParParallel,args)
	result = [x for x in result if x is not None]
	
	specXR = xr.merge(result)
	
	if debugging:
		print('total time with parallelizing for all heights was', time.time()-t0)
	
	return specXR

def fullRadarParallelNew(dicSettings, mcTable):
	"""
	Calculates the radar variables over the entire range

	Parameters
	----------
	dicSettings: a dictionary with all settings output from loadSettings()
	mcTable: McSnow data output from getMcSnowTable()

	Returns
	-------
	specXR: xarray dataset with the spectra(range, vel) and KDP(range)
	"""

	import concurrent.futures
	mcTable = creatRadarCols(mcTable, dicSettings)
	t0 = time.time()
	att_atm0 = 0.; att_ice_HH0=0.; att_ice_VV0=0.
	DDA_data_agg = xr.open_dataset(dicSettings['scatSet']['lutPath']+'all_aggregates_small_kdp.nc')
	DDA_data_agg['logmass'] = np.log10(DDA_data_agg.mass)
	DDA_data_agg['logDmax'] = np.log10(DDA_data_agg.Dmax)
	DDA_data_cry = xr.open_dataset(dicSettings['scatSet']['lutPath']+'all_crystals_allazi_withradar.nc')
	DDA_data_cry['logmass'] = np.log10(DDA_data_cry.mass)
	DDA_data_cry['logDmax'] = np.log10(DDA_data_cry.Dmax)
	DDA_data_cry['logar'] = np.log10(DDA_data_cry.aspect_ratio)

	# separate into species;
	mcTableAgg, mcTableCry, mcTableFrozen, mcTableMelted, mcTableLiquid = prepare_mcTable(mcTable,dicSettings)
	if dicSettings['beta_std'] == 0:
		elevation_radius = 1
	else:
		elevation_radius = dicSettings['beta']
	search_radii = dict(
						logmass=abs(np.log10(1) - np.log10(1.05)), # 2 %
						logDmax=abs(np.log10(1) - np.log10(1.05)), # 5 %
						elevation = elevation_radius,
						wavelength = 0.1,
						#habit = 7, # 10 % for habit code (which works because habit=0 for plates, so 0 tolerance, habit = 1 for dendrites, so 10% tolerance will not shift to other habit, only if habit = 20 or large, then 10% will be a int number)
						)
	print(DDA_data_agg, search_radii)
	treeAgg, scalingAgg = gen_ckdtree(DDA_data_agg, search_radii)
	
	search_radii = dict(
						logmass=abs(np.log10(1) - np.log10(1.1)), # 2 %
						logDmax=abs(np.log10(1) - np.log10(1.1)), # 5 %
						logar = abs(np.log10(1) - np.log10(1.1)), # 2 %
						elevation = elevation_radius,
						wavelength = 0.1,
						)
	treeCry, scalingCry = gen_ckdtree(DDA_data_cry, search_radii)

	t0 = time.time()
	n_cores = 8#multiprocessing.cpu_count()
	print(n_cores)
	with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
		future = []
		for h in dicSettings['heightRange']:
			#print(Dx.values)
			future.append(executor.submit(getRadarParParallel, heightEdge0=h,mcTable=mcTable,mcTableAgg=mcTableAgg,
																mcTableCry=mcTableCry,mcTableFrozen=mcTableFrozen,
																mcTableMelted=mcTableMelted,dicSettings=dicSettings,
																treeAgg=treeAgg,scalingAgg=scalingAgg,treeCry=treeCry,
																scalingCry=scalingCry,DDA_data_agg=DDA_data_agg,
																DDA_data_cry=DDA_data_cry))#,
	print("multithreads done in {} seconds".format(time.time()-t0))
	results = [i.result() for i in future]
	specXR = xr.merge(results)
	
	if debugging:
		print('total time with parallelizing for all heights was', time.time()-t0)
	
	return specXR


def prepare_mcTable(mcTable,dicSettings):
	"""
	Here I am outsourcing the calculations of the different particle species
	"""
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
	mcTableFrozen = mcTableIce.where(mcTableIce['frozen_fraction']>1,drop=True)
	mcTableUnfrozen = mcTableIce.where(mcTableIce['frozen_fraction']<=1,drop=True)
	mcTableCry = mcTableUnfrozen.where(mcTableIce['sNmono']==1,drop=True) # select only cry, only calculate that once!
	mcTableAgg = mcTableUnfrozen.where(mcTableIce['sNmono']>1,drop=True) # select only aggregates
	
	# define habit codes to be consistent with the codes of the DDA_data_agg database:
	
	# ratioPN = np.round(((mcTableAgg.sNmono - mcTableAgg['sp%pp'])/mcTableAgg.sNmono).values,1)*10+20 # ratio of plates and needles
	# ratioPN = np.where(ratioPN==20, 21, ratioPN)
	# ratioPN = np.where(ratioPN==22, 23, ratioPN)
	# ratioPN = np.where(ratioPN==24, 25, ratioPN)
	# ratioPN = np.where(ratioPN==26, 27, ratioPN)
	# ratioPN = np.where(ratioPN==28, 29, ratioPN)
	# ratioDN = np.round(((mcTableAgg.sNmono - mcTableAgg['sp%dd'])/mcTableAgg.sNmono).values,1)*10+30 # ratio of dendrites and needles
	# ratioDN = np.where(ratioDN==30, 31, ratioDN)
	# ratioDN = np.where(ratioDN==32, 33, ratioDN)
	# ratioDN = np.where(ratioDN==34, 35, ratioDN)
	# ratioDN = np.where(ratioDN==36, 37, ratioDN)
	# ratioDN = np.where(ratioDN==38, 39, ratioDN)
	# ratioPD = np.round(((mcTableAgg.sNmono - mcTableAgg['sp%pp'])/mcTableAgg.sNmono).values,1)*10+40 # ratio of plates and dendrites
	# ratioPD = np.where(ratioPD==40, 41, ratioPD)
	# ratioPD = np.where(ratioPD==42, 43, ratioPD)
	# ratioPD = np.where(ratioPD==44, 45, ratioPD)
	# ratioPD = np.where(ratioPD==46, 47, ratioPD)
	# ratioPD = np.where(ratioPD==48, 49, ratioPD)

	# # define condition for plates and dendrites:
	# condPD = mcTableAgg['sp%pp'] + mcTableAgg['sp%dd'] == mcTableAgg.sNmono
	# # define condition for plate and needle:
	# condPN = (mcTableAgg['sp%dd'] == 0) & (mcTableAgg['sp%pp'] > 0) & (mcTableAgg.sNmono > mcTableAgg['sp%pp'])
	# # define condition for dendrite and needle: 
	# condDN = (mcTableAgg['sp%pp'] == 0) & (mcTableAgg['sp%dd'] > 0) & (mcTableAgg.sNmono > mcTableAgg['sp%dd'])
	# # define condition for needle:
	# condN = mcTableAgg['sp%pp'] + mcTableAgg['sp%dd'] == 0
	# # define condition for plates:
	# condP  = mcTableAgg['sp%pp'] == mcTableAgg.sNmono
	# #plates = mcTableAgg.where(mcTableAgg['sp%pp'] == mcTableAgg.sNmono,drop=True)
	# # define dendrites:
	# condD = mcTableAgg['sp%dd'] == mcTableAgg.sNmono
	
	mcTableAgg['habit_code'] = mcTableAgg.sNmono.copy()*0
	# habit_code = mcTableAgg.habit_code
	# habit_code = xr.where(condDN, ratioDN, habit_code)
	# habit_code = xr.where(condPN, ratioPN, habit_code)
	# habit_code = xr.where(condPD, ratioPD, habit_code)
	# habit_code = xr.where(condP, 0, habit_code)
	# habit_code = xr.where(condD, 2, habit_code)
	# habit_code = xr.where(condN,1,habit_code)
	# mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(condDN), ratioDN)#, mcTable['habit_code'])
	# mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(condPN), ratioPN)#, mcTable['habit_code'])
	# mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(mcTableAgg['sp%pp'] + mcTableAgg['sp%dd'] == mcTableAgg.sNmono), ratioPD)#, mcTable['habit_code'])
	# mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(mcTableAgg['sp%pp'] == mcTableAgg.sNmono), 0)#, mcTable['habit_code'])
	# mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(mcTableAgg['sp%dd'] == mcTableAgg.sNmono), 2)#, mcTable['habit_code'])
	# mcTableAgg['habit_code'] = mcTableAgg.habit_code.where(np.logical_not(mcTableAgg['sp%pp'] + mcTableAgg['sp%dd'] == 0), 1)#, mcTable['habit_code'])
	return mcTableAgg, mcTableCry, mcTableFrozen, mcTableMelted, mcTableLiquid

def fullRadar(dicSettings, mcTable):
	"""
	Calculates the radar variables over the entire range

	Parameters
	----------
	dicSettings: a dictionary with all settings output from loadSettings()
	mcTable: McSnow data output from getMcSnowTable()

	Returns
	-------
	specXR: xarray dataset with the spectra(range, vel) and KDP(range)
	"""


	specXR = xr.Dataset()
	#specXR_turb = xr.Dataset()
	
	mcTable = creatRadarCols(mcTable, dicSettings)
	t0 = time.time()
	att_atm0 = 0.; att_ice_HH0=0.; att_ice_VV0=0.
	DDA_data_agg = xr.open_dataset(dicSettings['scatSet']['lutPath']+'all_aggregates_small_kdp.nc')
	DDA_data_agg['logmass'] = np.log10(DDA_data_agg.mass)
	DDA_data_agg['logDmax'] = np.log10(DDA_data_agg.Dmax)
	DDA_data_cry = xr.open_dataset(dicSettings['scatSet']['lutPath']+'stochastic_crystals.nc')
	DDA_data_cry['logmass'] = np.log10(DDA_data_cry.mass)
	DDA_data_cry['logDmax'] = np.log10(DDA_data_cry.Dmax)
	DDA_data_cry['logar'] = np.log10(DDA_data_cry.aspect_ratio)

	# separate into species;
	mcTableAgg, mcTableCry, mcTableFrozen, mcTableMelted, mcTableLiquid = prepare_mcTable(mcTable,dicSettings)
	#mcDN = mcTableAgg.where(habit_code > 30)
	#mcDN = mcDN.where(habit_code < 40, drop=True)
	#print(len(mcDN.index), 'DN')
	#quit()
	# bins=np.arange(0.5,50.5)#[00.5,1.5,2.5,20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5,30.5,31.5,32.5,33.5,34.5,35.5,36.5,37.5,38.5,39.5,40.5,41.5,42.5,43.5,44.5,45.5,46.5,47.5,48.5,49.5]
	# plt.hist(habit_code,bins=bins)
	# plt.savefig('test_habit_code.png')
	# quit()
	# mc20 = mcTableAgg.where(mcTableAgg.habit_code == 40,drop=True)
	# for i,s in enumerate(mc20.index):
	# 	mcsel = mc20.sel(index=s)
	# 	print('plate',mcsel['sp%pp'].values)
	# 	print('dendrite',mcsel['sp%dd'].values)
	# 	print('sNmono',mcsel.sNmono.values)
	# 	if i == 10:
	# 		quit()
	# quit()
	#print(aggdb)
	if dicSettings['beta_std'] == 0:
		elevation_radius = 1
	else:
		elevation_radius = dicSettings['beta']
	search_radii = dict(
						logmass=abs(np.log10(1) - np.log10(1.02)), # 2 %
						logDmax=abs(np.log10(1) - np.log10(1.02)), # 5 %
						elevation = elevation_radius,
						wavelength = 0.1,
						habit = 0.5, # 10 % for habit code (which works because habit=0 for plates, so 0 tolerance, habit = 1 for dendrites, so 10% tolerance will not shift to other habit, only if habit = 20 or large, then 10% will be a int number)
						)
	print(DDA_data_agg, search_radii)
	treeAgg, scalingAgg = gen_ckdtree(DDA_data_agg, search_radii)
	
	search_radii = dict(
						logmass=abs(np.log10(1) - np.log10(1.05)), # 5 %
						logDmax=abs(np.log10(1) - np.log10(1.05)), # 5 %
						logar = abs(np.log10(1) - np.log10(1.05)), # 5 %
						elevation = elevation_radius,
						wavelength = 0.1,
						)
	#treeCry, scalingCry = gen_ckdtree(DDA_data_cry, search_radii)
	treeCry = None; scalingCry = None
	for i, heightEdge0 in enumerate(dicSettings['heightRange']):
		if len(dicSettings['gridBaseArea']) > 1:
			vol = dicSettings['gridBaseArea'][i] * dicSettings['heightRes']
		else:
			vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']
		heightEdge1 = heightEdge0 + dicSettings['heightRes']

		print('Range: from {0} to {1}'.format(heightEdge0, heightEdge1))
		mcTableTmp = mcTable.where((mcTable['sHeight']>heightEdge0) &
				 					(mcTable['sHeight']<=heightEdge1),drop=True)
		mcTableAggTmp = mcTableAgg.where((mcTableAgg['sHeight']>heightEdge0) &
			 					(mcTableAgg['sHeight']<=heightEdge1),drop=True)
		mcTableCryTmp = mcTableCry.where((mcTableCry['sHeight']>heightEdge0) &
			 					(mcTableCry['sHeight']<=heightEdge1),drop=True)
		mcTableFrozenTmp = mcTableFrozen.where((mcTableFrozen['sHeight']>heightEdge0) &
			 					(mcTableFrozen['sHeight']<=heightEdge1),drop=True)
		mcTableMeltedTmp = mcTableMelted.where((mcTableMelted['sHeight']>heightEdge0) &
			 					(mcTableMelted['sHeight']<=heightEdge1),drop=True)
		mcTableLiquidTmp = mcTableLiquid.where((mcTableLiquid['sHeight']>heightEdge0) &
			 					(mcTableLiquid['sHeight']<=heightEdge1),drop=True)
		#print(mcTableCryTmp)
		#print(mcTableAggTmp)
		#print(mcTableTmp.vel)
		if mcTableTmp.vel.any():
			mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'], mcTableTmp,mcTableAggTmp,mcTableCryTmp,mcTableFrozenTmp,mcTableMeltedTmp,mcTableLiquidTmp,
							   dicSettings['scatSet'],dicSettings['beta'],dicSettings['beta_std'],treeAgg,scalingAgg,treeCry,scalingCry,DDA_data_agg,DDA_data_cry, ice_core=dicSettings['scatSet']['ice_core'])#,height=(heightEdge1+heightEdge0)/2)
			#- get the spectra, there is the possibility to add shear, but I have not implemented it yet
			k_theta, k_phi, k_r = 0,0,0
			tmpSpecXR = getMultFrecSpec(dicSettings['wl'], dicSettings['elv'],mcTableTmp, dicSettings['velBins'],
										dicSettings['velCenterBin'], (heightEdge1+heightEdge0)/2,dicSettings['convolute'],dicSettings['nave'],dicSettings['noise_pow'],
										dicSettings['eps_diss'], dicSettings['uwind'],dicSettings['time_int'], dicSettings['theta']/2./180.*np.pi,
										k_theta,k_phi,k_r, dicSettings['tau'])
			print(vol)
			tmpSpecXR = tmpSpecXR/vol
			#print(tmpSpecXR)
			
			#quit()
			tmpKdpXR =  getIntKdp(mcTableTmp,(heightEdge1+heightEdge0)/2)
			#print(vol)
			specXR = xr.merge([specXR,tmpSpecXR, tmpKdpXR/vol])
			#print(specXR.KDP.min(), specXR.KDP.max())
			#print('Agg',specXR.KDPAgg.min(), specXR.KDPAgg.max())
			#print('Mono',specXR.KDPMono.min(), specXR.KDPMono.max())
			# plt.plot(specXR.range, specXR.KDP.sel(elevation=30,wavelength=3, method='nearest'))
			# plt.plot(specXR.range, specXR.KDPMono.sel(elevation=30,wavelength=3, method='nearest'),label='Mono')
			# plt.plot(specXR.range, specXR.KDPAgg.sel(elevation=30,wavelength=3, method='nearest'),label='Agg')
			# plt.legend()
			# plt.show()
			fig,ax = plt.subplots(ncols=3,figsize=(15,5),constrained_layout=True)
			Ze_H = specXR['spec_H'].sum(dim='vel')
			ZeKa = 10*np.log10(Ze_H.sel(wavelength=8,elevation=90,method='nearest'))
			ZeW = 10*np.log10(Ze_H.sel(wavelength=3,elevation=90,method='nearest'))

			ax[0].plot(ZeKa-ZeW,specXR.range)
			ax[0].set_ylabel('Range [m]',fontsize=16)
			ax[0].set_xlabel('DWR KaW [dB]',fontsize=16)
			
			ax[1].plot(specXR.KDP.sel(wavelength=3,elevation=30, method='nearest'),specXR.range)
			ax[1].set_ylabel('Range [m]',fontsize=16)
			ax[1].set_xlabel('KDP W [deg/km]',fontsize=16)

			p1 = ax[2].pcolormesh(specXR.vel,specXR.range, 10*np.log10(specXR.spec_H.sel(wavelength=8,elevation=90,method='nearest')),vmin=-30,vmax=10,cmap='turbo')
			cb = plt.colorbar(p1,ax=ax[2])
			cb.set_label('Spec_H Ka [dBZ/(m/s)]',fontsize=16)
			cb.ax.tick_params(labelsize=16)
			ax[2].set_xlabel('Velocity [m/s]',fontsize=16)
			ax[2].set_ylabel('Range [m]',fontsize=16)
			ax[2].set_xlim(-2,0)
			for a in ax:
				a.tick_params(labelsize=16)
				a.grid()
				#plt.show()
			plt.savefig('test_newMcRadar.png')
			plt.close()

			if dicSettings['attenuation'] == True:
				
				specXR['att_atm_ice_HH'] = 2*specXR.att_ice_HH.cumsum(dim='range') + 2*specXR.att_atmo.cumsum(dim='range')
				specXR['att_atm_ice_VV'] = 2*specXR.att_ice_VV.cumsum(dim='range') + 2*specXR.att_atmo.cumsum(dim='range')
				specXR.att_atm_ice_HH.attrs['long_name'] = '2 way attenuation at HH polarization'
				specXR.att_atm_ice_HH.attrs['unit'] = 'dB'
				specXR.att_atm_ice_HH.attrs['comment'] = '2 way attenuation for ice particles and atmospheric gases (N2,O2,H2O). The spectra are divided my this, so to get unattenuated spectra, multiply with this (in linear units)'
				
				specXR.att_atm_ice_HH.attrs['long_name'] = '2 way attenuation at VV polarization'
				specXR.att_atm_ice_HH.attrs['unit'] = 'dB'
				specXR.att_atm_ice_HH.attrs['comment'] = '2 way attenuation for ice particles and atmospheric gases (N2,O2,H2O). The spectra are divided my this, so to get unattenuated spectra, multiply with this (in linear units)'
				
				specXR['spec_H_att'] = specXR.spec_H/(10**(specXR.att_atm_ice_HH/10))
				specXR['spec_V'] = specXR.spec_V/(10**(specXR.att_atm_ice_VV/10))
				specXR['spec_HV'] = specXR.spec_HV/(10**(specXR.att_atm_ice_HH/10))
		else:
			print('empty dataset at this height range')
	return specXR

def singleParticleTrajectories(dicSettings, mcTable):
	"""
	Calculates the radar variables over the entire range

	Parameters
	----------
	dicSettings: a dictionary with all settings output from loadSettings()
	mcTable: McSnow data output from getMcSnowTable()

	Returns
	-------
	specXR: xarray dataset with the single particle scattering properties
	"""

	t0 = time.time()
	specXR = xr.Dataset()
	#specXR_turb = xr.Dataset()
	mcTable = creatRadarCols(mcTable, dicSettings)
	#counts = np.ones_like(dicSettings['heightRange'])*np.nan
	#vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']

	for i, pID in enumerate(np.unique(mcTable['sMult'].values)[::-1]):

		mcTableTmp = mcTable.where(mcTable.sMult==pID,drop=True)
		print(len(np.unique(mcTable['sMult'].values)),i)
		mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'], mcTableTmp, dicSettings['scatSet'],dicSettings['beta'],dicSettings['beta_std'])
		
		mcTableTmp = mcTableTmp.assign_coords(index=mcTableTmp.sHeight).rename({'index':'range'})#.set_index('sHeight')
		print(mcTableTmp.sHeight)
		#quit()
		#mcTableTmp = mcTableTmp.reindex(range=dicSettings['heightRange'],method='nearest',tolerance=dicSettings['heightRes'])
		
		vars2drop = ['sMult','sZeMultH','sZeMultV','sZeMultHV','sCextHMult','sCextHMult','sKDPMult']
		
		mcTableTmp = mcTableTmp.drop_vars(vars2drop)
		
		print(mcTableTmp)
		#plt.plot(mcTableTmp.sHeight,)#mcTableTmp.sZeH.sel(wavelength=3.189, elevation=90,method='nearest'))
		#plt.savefig('/project/meteo/work/L.Terzi/McSnow_habit/test_single_particle.png')
		#plt.show()
		#quit()
		mcTableTmp = mcTableTmp.expand_dims(dim='pID').assign_coords(pID=[pID])
		
		specXR = xr.merge([specXR, mcTableTmp])
		print(specXR)
		
	print('total time was ', time.time()-t0)
	return specXR

def calc_1_particle(mcTable,pID,dicSettings,i):
	mcTableTmp = mcTable.where(mcTable.sMult==pID,drop=True)
	print(len(np.unique(mcTable['sMult'].values)[::5]),i)

	mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'], mcTableTmp,dicSettings['scatSet'],dicSettings['beta'],dicSettings['beta_std'])
	
	mcTableTmp = mcTableTmp.assign_coords(index=mcTableTmp.sHeight).rename({'index':'range'})#.set_index('sHeight')
	
	#mcTableTmp = mcTableTmp.reindex(range=dicSettings['heightRange'],method='nearest',tolerance=dicSettings['heightRes'])
	
	vars2drop = ['sMult','sZeMultH','sZeMultV','sZeMultHV','sCextHMult','sCextHMult','sKDPMult']
	
	mcTableTmp = mcTableTmp.drop_vars(vars2drop)
	
	mcTableTmp = mcTableTmp.expand_dims(dim='pID').assign_coords(pID=[pID])
	return mcTableTmp

def singleParticleTrajParallel(dicSettings, mcTable,MaxWorkers=10):
	"""
	Calculates the radar variables over the entire range

	Parameters
	----------
	dicSettings: a dictionary with all settings output from loadSettings()
	mcTable: McSnow data output from getMcSnowTable()

	Returns
	-------
	specXR: xarray dataset with the single particle scattering properties
	"""

	t0 = time.time()
	specXR = xr.Dataset()
	#specXR_turb = xr.Dataset()
	mcTable = creatRadarCols(mcTable, dicSettings)
	#counts = np.ones_like(dicSettings['heightRange'])*np.nan
	#vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']

	pool = multiprocessing.Pool(MaxWorkers)
	
	args = [(mcTable, pID, dicSettings,i) for i,pID in enumerate(np.unique(mcTable['sMult'].values)[::5])]
	
	result =  pool.starmap(calc_1_particle,args)
	print('done with calcs, now need to merge')
	result = [x for x in result if x is not None]
	#print(result)
	specXR = xr.merge(result)
		
	print('total time was ', time.time()-t0)
	return specXR


def singleParticleScat(dicSettings, mcTable):
	"""
	Calculates the radar variables over the entire range

	Parameters
	----------
	dicSettings: a dictionary with all settings output from loadSettings()
	mcTable: McSnow data output from getMcSnowTable()

	Returns
	-------
	specXR: xarray dataset with the spectra(range, vel) and KDP(range)
	"""

	t0 = time.time()
	singlePart = xr.Dataset()
	#specXR_turb = xr.Dataset()
	vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']
	mcTable = creatRadarCols(mcTable, dicSettings)
	t0 = time.time()
	att_atm0 = 0.; att_ice_HH0=0.; att_ice_VV0=0.
	for i, heightEdge0 in enumerate(dicSettings['heightRange']):

		heightEdge1 = heightEdge0 + dicSettings['heightRes']

		print('Range: from {0} to {1}'.format(heightEdge0, heightEdge1))
		mcTableTmp = mcTable.where((mcTable['sHeight']>heightEdge0) &
				 					(mcTable['sHeight']<=heightEdge1),drop=True)
		
		if mcTableTmp.vel.any():
			mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
						       			mcTableTmp, ndgs=dicSettings['ndgsVal'],
						        		scatSet=dicSettings['scatSet'])#,height=(heightEdge1+heightEdge0)/2)
			#print(mcTableTmp.sZeH)
			singlePart = xr.merge([mcTableTmp,singlePart]) # TODO: do I need to normalize with Volume? I think so!!
			#print(singlePart.sZeH)
	print('total time with old method', time.time()-t0)
	return singlePart


'''
def singleParticleTrajectories(dicSettings, mcTable):
	"""
	Calculates the radar variables over the entire range

	Parameters
	----------
	dicSettings: a dictionary with all settings output from loadSettings()
	mcTable: McSnow data output from getMcSnowTable()

	Returns
	-------
	specXR: xarray dataset with the single particle scattering properties
	"""


	specXR = xr.Dataset()
	#specXR_turb = xr.Dataset()
	counts = np.ones_like(dicSettings['heightRange'])*np.nan
	vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']
	for i, heightEdge0 in enumerate(dicSettings['heightRange']):

		heightEdge1 = heightEdge0 + dicSettings['heightRes']

		print('Range: from {0} to {1}'.format(heightEdge0, heightEdge1))
		mcTableTmp = mcTable[(mcTable['sHeight']>heightEdge0) &
				             (mcTable['sHeight']<=heightEdge1)].copy()
		#for i, pID in enumerate(mcTable['sMult'].unique()):

		#    mcTableTmp = mcTable[(mcTable['sMult']==pID)].copy()

		#print(len(mcTable['sMult'].unique()),i)
		mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'],
				                    mcTableTmp, ndgs=dicSettings['ndgsVal'],
				                    scatSet=dicSettings['scatSet'])
		print(mcTableTmp)
		quit()
		mcTableTmp = mcTableTmp.set_index('sHeight')
		specTable = mcTableTmp.to_xarray()
		specTable = specTable.drop_vars('sMult')
		specTable = specTable.expand_dims(dim='sMult').assign_coords(sMult=[pID])
		print(specTable)

		#specTable = specTable.expand_dims(dim='range').assign_coords(range=[centerHeight])
		specXR = xr.merge([specXR, specTable])
		print(specXR)
		quit()

	return specXR
'''
