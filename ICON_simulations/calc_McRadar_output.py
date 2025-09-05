#-*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Author: Leonie von Terzi


# this calculates the polarimetric variables at Wband for McSnow output. 
# It is intended to test habit prediction, aggregation has not been implemented in this McSnow run.
# The McSnow data was produced by Jan-Niklas Welß

import numpy as np
import mcradar as mcr
from mcradar import *
import matplotlib.pyplot as plt
import concurrent.futures
from time import time
def str2bool(v):
  return v.lower() in ("yes", "True", "t", "1","true")

def closest(lst, K):
    
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return lst[idx]
def myround(x, base=5):
    return base * np.ceil(x/base)

#- get all variables necessary to calculate scattering from environment
convolute=True
elv = np.array([30,90])
freq = np.array([9.6e9,35.6e9,94.0e9]) # in Hz
ori_avg = True
beta = 0
beta_std = 0
selMode = 'KNeighborsRegressor'
n_neighbors = 10
scatMode = 'wobbling'
attenuation = False
lutPath = '/project/meteo/work/L.Terzi/McRadarTest/LUT/' #'/work/lvonterz/SSRGA/snowScatt/ssrga_LUT/' #'/data/optimice/McRadarLUTs/'
# define the velocity vector:
velVec = np.loadtxt('/project/meteo/work/L.Terzi/McSnow_depogrowth_paper/dopplerVelocities_Wband_CEL.txt')
#-- define range resolution 
heightRes = 36

inputPath = '/scratch/f/Fabian.Jakub/icon_mcsnow_build/experiments/mcsnow2d.nosani.sphabit3.rt3/'

mass2frname = 'particles00001800.000.nc' #'mass2fr.nc'
outName = '9.6_35.5_94.0GHz_output_{}'.format(mass2frname)

#domTop = inputPath.split('domtop')[1].split('_')[0].split('.')[0]
domTop = 11000.0 # in m, this is the height of the top of the domain
#box_area = 250#5.0
print('loading the settings')

#In order to avoid volume sampling problems, you have to insert the gridBaseArea as it was defined in the McSnow simulation
dicSettings = mcr.loadSettings(dataPath=inputPath+mass2frname,velVec=velVec, #atmoFile=inputPath+'atmo.dat',
							elv=elv, freq=freq,gridBaseArea=box_area,maxHeight=int(domTop),minHeight=0,
							heightRes=heightRes,convolute=convolute,attenuation=attenuation,beta=beta,beta_std=beta_std,onlyIce=False,
							scatSet={'mode':scatMode,'selmode':selMode,'n_neighbors':n_neighbors,'K2':0.93,'lutPath':lutPath,'orientational_avg':ori_avg})

print('loading the McSnow output')
#quit()
# now generate a table from the McSnow output.
mcTable = mcr.getMcSnowTable(dicSettings['dataPath'])
mcTable = mcTable.where(~np.isnan(mcTable.vel),drop=True)


#print(mcTable['habit_code'].min().values, mcTable['habit_code'].max().values)

#print('now here')
if 'time' in mcTable:
	times = mcTable['time']
	selTime = mcTable['time'].max()
	mcTableTmp = mcTable.where(times==selTime,drop=True)	#mcTable[times==selTime]#
else:
	mcTableTmp = mcTable
#print('after sel')

if dicSettings['onlyIce'] == True:
	coldT = dicSettings['temp'].where(dicSettings['temp'] < 273.15,drop=True)
	mcTableTmp = mcTableTmp.where((mcTableTmp['sHeight']>coldT.range.min().values) &
									(mcTableTmp['sHeight']<=coldT.range.max().values),drop=True)
#print(mcTable.sMult.min().values, mcTable.sMult.max().values)
#quit()
print('getting things done :) -> calculating radar variables for '+str(freq)+'Hz')
#output = mcr.fullRadarParallel(dicSettings, mcTableTmp)
#print(output)
output = mcr.fullRadar(dicSettings, mcTableTmp)
#quit()
#- calculate moments and noise from the spectra:	
output['Ze_H'] = output['spec_H'].sum(dim='vel')
output['Ze_V'] = output['spec_V'].sum(dim='vel')
if 'spec_H_Agg' in output:
	output['Ze_H_Agg'] = output['spec_H_Agg'].sum(dim='vel')
	output['Ze_V_Agg'] = output['spec_V_Agg'].sum(dim='vel')
	output['ZDR_Agg'] = mcr.lin2db(output['Ze_H_Agg']/output['Ze_H_Agg'])

if 'spec_H_Mono' in output:
	output['Ze_H_Mono'] = output['spec_H_Mono'].sum(dim='vel')
	output['Ze_V_Mono'] = output['spec_V_Mono'].sum(dim='vel')
	output['ZDR_Mono'] = mcr.lin2db(output['Ze_H_Mono']/output['Ze_H_Mono'])

output['ZDR'] = mcr.lin2db(output['Ze_H']/output['Ze_H'])
output['Ze_HV'] = output['spec_HV'].sum(dim='vel')
output['LDR'] = mcr.lin2db(output['Ze_HV']/output['Ze_H'])
output['MDV_H'] = (output['spec_H']*output['vel']).sum(dim='vel')/output['Ze_H']
output['MDV_V'] = (output['spec_V']*output['vel']).sum(dim='vel')/output['Ze_V']
#NoiseDens = dicSettings['noise_pow']/len(dicSettings['velCenterBin'])
NoisePow = dicSettings['noise_pow']/(dicSettings['nfft']*dicSettings['velRes'])
output['SNR_H'] = output['Ze_H']/NoisePow
output['SNR_V'] = output['Ze_V']/NoisePow
output['sSNR_H'] = output['spec_H']/dicSettings['noise_pow']
output['sSNR_V'] = output['spec_V']/dicSettings['noise_pow']
		
#-- now save it
output.to_netcdf(inputPath+outName)#inputPath+outName)
#singlePart.to_netcdf(inputPath+'test_singlescattering.nc')
