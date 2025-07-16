#-*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Author: Leonie von Terzi


# this calculates the radar variables at W-Band, Ka-Band and X-Band at 30 and 90° elevation for the McSnow case study


import numpy as np
import mcradar as mcr
from mcradar import *
import pandas as pd
import xarray as xr
import plotting_functions as plot


def str2bool(v):
  return v.lower() in ("yes", "True", "t", "1","true")

#- get all variables necessary to calculate scattering from environment
allTimes=False
single_particle=False 
convolute=True
freq = np.array([9.6e9,35.6e9,94e9])
elv = np.array([30,90])
ori_avg = True
beta = 0
beta_std = 0
#particle_name = os.environ['particle']
selMode = 'KNeighborsRegressor'
n_neighbors = 10
outName = 'McRadar_McSnow_case_study_n_neighbours{}.nc'.format(n_neighbors)
inputPath = '/project/meteo/work/L.Terzi/McRadarTest/examples/data/McSnow_case_study/'

scatMode = 'azimuthal_random_orientation'
attenuation = False
lutPath = '/project/meteo/work/L.Terzi/McRadarTest/LUT/'

domTop = 5500
box_area= 5
velVec = np.loadtxt('dopplerVelocities_example.txt')
heightRes = 36

#In order to avoid volume sampling problems, you have to insert the gridBaseArea as it was defined in the McSnow simulation
dicSettings = mcr.loadSettings(dataPath=inputPath+'mass2fr.nc',atmoFile=inputPath+'atmo.dat',velVec=velVec,
                               elv=elv, freq=freq,gridBaseArea=box_area,maxHeight=int(domTop),minHeight=0,
                               heightRes=heightRes,convolute=convolute,attenuation=attenuation,beta=beta,beta_std=beta_std,
                               scatSet={'mode':scatMode,'selmode':selMode,'n_neighbors':n_neighbors,'K2':0.93,'lutPath':lutPath,'orientational_avg':ori_avg})

print('loading the McSnow output')
#quit()
# now generate a table from the McSnow output.
mcTable = mcr.getMcSnowTable(dicSettings['dataPath'])
mcTable = mcTable.where(~np.isnan(mcTable.vel),drop=True)


#quit()
#print('now here')
# select the last time step of McSnow simulation
selTime = mcTable['time'].max()
mcTableTmp = mcTable.where(mcTable['time']==selTime,drop=True)	#mcTable[times==selTime]#

print('getting things done :) -> calculating radar variables for '+str(freq)+'Hz')

output = mcr.fullRadarParallel(dicSettings, mcTableTmp)
#- alternative if only one core is available:
#output = mcr.fullRadar(dicSettings, mcTableTmp)

print('done with the calculations, now calculating moments from the spectra')
	
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
print('saving the output file at: '+inputPath+outName)
output.to_netcdf(inputPath+outName)#inputPath+outName)

#-- now plot it
print('now plotting McRadar')
name2save=inputPath+outName.split('.nc')[0]+'.png'
atmoFile = np.loadtxt(inputPath+'atmo.dat')
height = atmoFile[:,0]
Temp = atmoFile[:,2] -273.15
atmoPD = pd.DataFrame(data=Temp,index=height,columns=['Temp'])
atmoPD.index.name='range'
atmoXR = atmoPD.to_xarray()
atmoReindex = atmoXR.reindex_like(output,method='nearest')
output = xr.merge([atmoReindex,output])
plot.plotOverview6Panels(output,name2save,allDWR=False)