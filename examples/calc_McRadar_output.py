#-*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Author: Leonie von Terzi


# this calculates the polarimetric variables at Wband for McSnow output. 
# It is intended to test habit prediction, aggregation has not been implemented in this McSnow run.
# The McSnow data was produced by Jan-Niklas Welß
# habit_codes = dict(
#         plate =0,
#         needle=1,
#         dendrite=2,
#         spheroid=3,
#         mixPN1=21,
#         mixPN2=22,
#         mixPN3=23,
#         mixPN4=24,
#         mixPN5=25,
#         mixPN6=26,
#         mixPN7=27,
#         mixPN8=28,
#         mixPN9=29,
#         mixDN1=31,
#         mixDN2=32,
#         mixDN3=33,
#         mixDN4=34,
#         mixDN5=35,
#         mixDN6=36,
#         mixDN7=37,
#         mixDN8=38,
#         mixDN9=39,
#         mixPD1=41,
#         mixPD2=42,
#         mixPD3=43,
#         mixPD4=44,
#         mixPD5=45,
#         mixPD6=46,
#         mixPD7=47,
#         mixPD8=48,
#         mixPD9=49,
#         mixPS1=111,
#         mixPS2=112,
#         mixPS3=113,
#         mixPS4=114,
#         mixPS5=115,
#         mixPS6=116,
#         mixPS7=117,
#         mixPS8=118,
#         mixPS9=119,
#         mixNS1=121,
#         mixNS2=122,
#         mixNS3=123,
#         mixNS4=124,
#         mixNS5=125,
#         mixNS6=126,
#         mixNS7=127,
#         mixNS8=128,
#         mixNS9=129,
#         mixDS1=131,
#         mixDS2=132,
#         mixDS3=133,
#         mixDS4=134,
#         mixDS5=135,
#         mixDS6=136,
#         mixDS7=137,
#         mixDS8=138,
#         mixDS9=139,
#         )


import numpy as np
import mcradar as mcr
from mcradar import *
import matplotlib.pyplot as plt
import concurrent.futures
from time import time
def str2bool(v):
  return v.lower() in ("yes", "True", "t", "1","true")
# def calculate_habit_code(sp,i):
# 	"""
# 	Calculate the habit code from the super particle data.
# 	Returns
# 	-------
# 	habit_code: habit code (int)
# 	"""
# 	print(i,'of total',len(mcTable.index))
# 	if sp['sp%pp'] == mcTablesel.sNmono:
# 		habit_code = 0
# 	elif sp['sp%dd'] == mcTablesel.sNmono:
# 		habit_code = 2
# 	elif sp['sp%pp'] + mcTablesel['sp%dd'] == 0:
# 		habit_code = 1
# 	elif sp['sp%pp'] + sp['sp%dd'] == sp.sNmono:
# 		#print('mix of plates and dendrites')
# 		ratio = (sp.sNmono - sp['sp%pp'])/sp.sNmono
# 		habit_code = '4{}'.format(int(round(100*ratio.values,-1)/10))
# 		#print('ratio plates/dendrites',ratio.values)
# 	elif sp['sp%pp'] > 0 and sp['sp%dd'] == 0 and sp.sNmono > sp['sp%pp']:
# 		#print('mix of plates and needles')
# 		ratio = (sp.sNmono - sp['sp%pp'])/sp.sNmono #/(mcTablesel.sNmono - mcTablesel['sp%pp'])
# 		habit_code = '2{}'.format(int(round(100*ratio.values,-1)/10))
# 		#print('ratio plates/needles',int(round(100*ratio.values,-1)),100*ratio.values)
# 		#print('habit_code','2{}'.format(int(round(100*ratio.values,-1)/10)))
# 	elif sp['sp%dd'] > 0 and sp['sp%pp'] == 0 and sp.sNmono > sp['sp%dd']:
# 		#print('mix of dendrites and needles')
# 		habit_code = '3{}'.format(int(round(100*ratio.values,-1)/10))
# 		ratio = (sp.sNmono - sp['sp%dd']) / sp.sNmono#mcTablesel['sp%dd']/(mcTablesel.sNmono - mcTablesel['sp%dd'])
# 		#print('ratio dendrites/needles',ratio.values)
	
# 	sp['habit_code'] = habit_code
# 	return sp
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
outName = '9.6_35.5_94.0GHz_output_DDA_kdtree_with_habitcode_crystals_KDTree_new_30_90_oriavgTrue_beta0_beta_std0_convoluteTrue_attenuationFalse.nc'

#inputPath = '/project/meteo/work/L.Terzi/McSnowoutput/habit/case_studies/20220206/NewAggs//1d_habit_habit1_IGF2_xi100_nz200_dtc5_fpm2_0_mult1_frag1_Dmode75_timeend36000_nh12000_nh26000_ncl75_nclmass4.8_nuclType1_at2_stick2_agggeo5_spkernsig0_ba500_domtop6000._atmo1_radiosondes_juelich_20220206_042141/'
allPaths = [#'1d_habit1_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg1_AR00/',
            #'1d_habit1_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg4_AR30/',
            '1d_habit1_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg5_AR30/',
            #'1d_habit2_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg1_AR00/',
            #'1d_habit2_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg4_AR30/',
            #'1d_habit2_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg5_AR30/'
			]

for inputPath in allPaths:
    
	inputPath = 'data/' + inputPath
	print(inputPath)
	#habit 1: first case,
	# habit 2: second case
	# agg1: mitchell, agg4: deterministic, agg5: stochastic
	#1d_habit1_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg1_AR00/
	#1d_habit1_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg4_AR30
	#1d_habit1_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg5_AR30
	#1d_habit2_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg1_AR00
	#1d_habit2_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg4_AR30
	#1d_habit2_xi016_nz250_lwc01_sat05_dt5_ncl42_rt2_habit1_agg5_AR30

	mass2frname = 'mass2fr.nc' #'mass2fr.nc'

	#inputPath = 'data/McSnow_stoch_aggs/'
	#domTop = inputPath.split('domtop')[1].split('_')[0].split('.')[0]
	domTop = 5000.0 # in m, this is the height of the top of the domain
	#box_area = inputPath.split('ba')[1].split('_')[0]
	#box_area=float(box_area)/100 #In order to avoid volume sampling problems, you have to insert the gridBaseArea as it was defined in the McSnow simulation
	box_area = 250#5.0
	print('loading the settings')
	#minmax=True
	#vmin= 180; vmax=350
	# define the velocity vector:

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
	
	# now determine the habit code: (listed at the top of the script)
	#habitcodes = [0,1,2,21,23,25,27,29,31,33,35,37,39,43,45,47]
	#ratioPD = myround((((mcTable.sNmono - mcTable['sp%pp'])/mcTable.sNmono).values)*10+40) #np.round(((mcTable.sNmono - mcTable['sp%pp'])/mcTable.sNmono).values,1)*10+40
	

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
