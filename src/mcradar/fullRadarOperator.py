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
#if not sys.warnoptions: # bad form, would not recommend!
#	import warnings
#    warnings.simplefilter("ignore")
import warnings
warnings.filterwarnings('ignore')
debugging=True
reduce_ncores = True
def getRadarParParallel(heightEdge0,mcTable,mcTableAgg,mcTableCry,dicSettings):#heightRes,wl,elv,ndgsVal,scatSet,velBins,velCenterBin,convolute,nave,noise_pow,eps_diss,uwind,time_int,theta,tau):
	vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']
	heightEdge1 = heightEdge0 + dicSettings['heightRes']

	print('Range: from {0} to {1}'.format(heightEdge0, heightEdge1))
	mcTableAggTmp = mcTableAgg.where((mcTableAgg['sHeight']>heightEdge0) &
			 					(mcTableAgg['sHeight']<=heightEdge1),drop=True)
	mcTableCryTmp = mcTableCry.where((mcTableCry['sHeight']>heightEdge0) &
			 					(mcTableCry['sHeight']<=heightEdge1),drop=True)
	mcTableTmp = mcTable.where((mcTable['sHeight']>heightEdge0) &
			 					(mcTable['sHeight']<=heightEdge1),drop=True)
	if mcTableTmp.vel.any():
		#- get the scattering properties for each particle, we have separate tables for aggregates and crystals
		mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'], mcTableTmp,mcTableAggTmp,mcTableCryTmp, dicSettings['scatSet'],dicSettings['beta'],dicSettings['beta_std'])#,height=(heightEdge1+heightEdge0)/2)
		#- get the spectra, there is the possibility to add shear, but I have not implemented it yet
		k_theta, k_phi, k_r = 0,0,0
		tmpSpecXR = getMultFrecSpec(dicSettings['wl'], dicSettings['elv'],mcTableTmp, dicSettings['velBins'],
					        		dicSettings['velCenterBin'], (heightEdge1+heightEdge0)/2,dicSettings['convolute'],dicSettings['nave'],dicSettings['noise_pow'],
					        		dicSettings['eps_diss'], dicSettings['uwind'],dicSettings['time_int'], dicSettings['theta']/2./180.*np.pi,
					        		k_theta,k_phi,k_r, dicSettings['tau'],
					        		scatSet=dicSettings['scatSet'])
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


	specXR = xr.Dataset()
	#specXR_turb = xr.Dataset()
	
	mcTable = creatRadarCols(mcTable, dicSettings)
	mcTableCry = mcTable.where(mcTable['sNmono']==1,drop=True) # select only cry, only calculate that once!
	mcTableAgg = mcTable.where(mcTable['sNmono']>1,drop=True) # select only aggregates
	t0 = time.time()
	n_cores = 8#multiprocessing.cpu_count()
	print(n_cores)
	pool = multiprocessing.Pool(n_cores)
	
	args = [(h, mcTable,mcTableAgg,mcTableCry, dicSettings) for h in dicSettings['heightRange']]
	
	result =  pool.starmap(getRadarParParallel,args)
	result = [x for x in result if x is not None]
	
	specXR = xr.merge(result)
	
	if debugging:
		print('total time with parallelizing for all heights was', time.time()-t0)
	
	return specXR



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
	vol = dicSettings['gridBaseArea'] * dicSettings['heightRes']
	mcTable = creatRadarCols(mcTable, dicSettings)
	t0 = time.time()
	att_atm0 = 0.; att_ice_HH0=0.; att_ice_VV0=0.
	mcTableCry = mcTable.where(mcTable['sNmono']==1,drop=True) # select only cry, only calculate that once!
	mcTableAgg = mcTable.where(mcTable['sNmono']>1,drop=True) # select only aggregates
	for i, heightEdge0 in enumerate(dicSettings['heightRange']):

		heightEdge1 = heightEdge0 + dicSettings['heightRes']

		print('Range: from {0} to {1}'.format(heightEdge0, heightEdge1))
		mcTableTmp = mcTable.where((mcTable['sHeight']>heightEdge0) &
				 					(mcTable['sHeight']<=heightEdge1),drop=True)
		mcTableAggTmp = mcTableAgg.where((mcTableAgg['sHeight']>heightEdge0) &
			 					(mcTableAgg['sHeight']<=heightEdge1),drop=True)
		mcTableCryTmp = mcTableCry.where((mcTableCry['sHeight']>heightEdge0) &
			 					(mcTableCry['sHeight']<=heightEdge1),drop=True)
		
		if mcTableTmp.vel.any():
			mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'], mcTableTmp,mcTableAggTmp,mcTableCryTmp, dicSettings['scatSet'],dicSettings['beta'],dicSettings['beta_std'])#,height=(heightEdge1+heightEdge0)/2)
			#- get the spectra, there is the possibility to add shear, but I have not implemented it yet
			k_theta, k_phi, k_r = 0,0,0
			tmpSpecXR = getMultFrecSpec(dicSettings['wl'], dicSettings['elv'],mcTableTmp, dicSettings['velBins'],
										dicSettings['velCenterBin'], (heightEdge1+heightEdge0)/2,dicSettings['convolute'],dicSettings['nave'],dicSettings['noise_pow'],
										dicSettings['eps_diss'], dicSettings['uwind'],dicSettings['time_int'], dicSettings['theta']/2./180.*np.pi,
										k_theta,k_phi,k_r, dicSettings['tau'],
										scatSet=dicSettings['scatSet'])
			tmpSpecXR = tmpSpecXR/vol
			#plt.plot(tmpSpecXR.vel,10*np.log10(tmpSpecXR.spec_H.sel(wavelength=3,elevation=90,range=(heightEdge1+heightEdge0)/2,method='nearest')))#range=(heightEdge1+heightEdge0)/2))
			#plt.savefig('test_spec.png')
			#plt.close()

			
			#quit()
			tmpKdpXR =  getIntKdp(mcTableTmp,(heightEdge1+heightEdge0)/2)
			specXR = xr.merge([specXR,tmpSpecXR, tmpKdpXR/vol])
			#print(specXR)
			# if i > 50:
			# 	plt.pcolormesh(specXR.vel,specXR.range, 10*np.log10(specXR.spec_H.sel(wavelength=8,elevation=90,method='nearest')),cmap='turbo',vmin=-30,vmax=10)
			# 	plt.colorbar()
			# 	plt.xlim([-3,1])
			# 	#plt.savefig('test_spec.png')
			# 	plt.show()


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

	for i, pID in enumerate(np.unique(mcTable['sMult'].values)):

		mcTableTmp = mcTable.where(mcTable.sMult==pID,drop=True)
		mcTableCry = mcTable.where(mcTable['sNmono']==1,drop=True) # select only cry, only calculate that once!
		mcTableAgg = mcTable.where(mcTable['sNmono']>1,drop=True)
		print(len(np.unique(mcTable['sMult'].values)),i)
		mcTableTmp = calcParticleZe(dicSettings['wl'], dicSettings['elv'], mcTableTmp,mcTableAggTmp,mcTableCryTmp, dicSettings['scatSet'],dicSettings['beta'],dicSettings['beta_std'])
		
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
