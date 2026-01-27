import numpy as np
import mcradar as mcr
from mcradar import *
from mcradar.tableOperator import creatRadarCols
import matplotlib.pyplot as plt 
import pandas as pd
import xarray as xr


g = 9.81 # gravitational acceleration [m/s^2]
rho_i = 917.6 # density of ice [kg/m^3]
def Nexp(D, lam):
    return np.exp(-lam*D)

def dB(x):
    return 10.0*np.log10(x)

def Bd(x):
    return 10.0**(0.1*x)
def gammadis(D, lam, mu, nu):
	return D**mu*np.exp(-lam*D**nu)


# define PSD and particle properties:
Dmax = np.linspace(0.01e-3, 10.0e-3, 10000) # list of sizes
lams = 1.0/np.linspace(0.1e-3, 4.0e-3, 2) # list of lambdas
am=0.02522677;bm=2.19978322 # mass size relationship parameters
mass = am*Dmax**bm
av=5.97000795;bv=0.45396479 # velocity size relationship parameters
vel = av*Dmax**bv#fall_velocity_HW(area,mass,Dmax,273.15,1000e2)
data = {'dia':Dmax,'mTot':mass,'sPhi':np.ones_like(mass),'sNmono':np.ones_like(mass)*3,'vel':vel}
dataTable = pd.DataFrame(data = data).to_xarray()
PSD = 30.0*np.array(Nexp(Dmax, 1/5e-3))

# define simulation settings
convolute=True
elv = np.array([30,90])
freq = np.array([9.6e9,35.6e9,94.0e9]) # in Hz
beta = 0
beta_std = 0
selMode = 'KNeighborsRegressor'
n_neighbors = 10
scatMode = 'azimuthal_random_orientation'
attenuation = False
lutPath = '/project/meteo/work/L.Terzi/McRadarTest/LUT/' #'/work/lvonterz/SSRGA/snowScatt/ssrga_LUT/' #'/data/optimice/McRadarLUTs/'
# define the velocity vector:
velVec = np.loadtxt('/project/meteo/work/L.Terzi/McSnow_depogrowth_paper/dopplerVelocities_Wband_CEL.txt') # the velocity vector used in your radar (Doppler coordinate)
#-- define range resolution 
heightRes = 36
outName = '9.6_35.5_94.0GHz_output_DDA_kdtree_with_habitcode_30_90_oriavgTrue_beta0_beta_std0_convoluteTrue_attenuationFalse_PSD.nc'

dicSettings = mcr.loadSettings(PSD=True,#'mass2fr.nc',#inputPath+'mass2fr.nc',
                               elv=np.array([90]), freq=freq,gridBaseArea=1,maxHeight=100,
                               heightRes=36,convolute=True,beta=beta,beta_std=beta_std,
                               scatSet={'mode':scatMode,'selmode':selMode,'n_neighbors':n_neighbors,'K2':0.93,'lutPath':lutPath})

# prepare the data table with all necessary columns
dataTable = creatRadarCols(dataTable, dicSettings)
dataTableAgg = dataTable.where(dataTable.sNmono>1,drop=True)# in McRadar we make the differentiation between aggregates and crystals, in this case all particles are aggregates.
dataTableCry = dataTable.where(dataTable.sNmono==1,drop=True)

# calculate single particle scattering
Zepart = calcParticleZe(dicSettings['wl'], dicSettings['elv'], dataTable,dataTableAgg,dataTableCry, dicSettings['scatSet'],dicSettings['beta'],dicSettings['beta_std']) # calculate Ze for each particle with your mass and size
print('calculated single particle Ze')
# now lets calculate correct Doppler spectrum:
specTable = xr.Dataset()
dataTable = dataTable.sortby('vel')
dataTable['sZePH'] = Zepart.sZeH*PSD*np.gradient(Dmax)/np.gradient(vel) # need to go from size to vel
dataTable['sZePV'] = Zepart.sZeV*PSD*np.gradient(Dmax)/np.gradient(vel) # need to go from size to vel
group = dataTable.groupby_bins('vel', dicSettings['velBins'],labels=dicSettings['velCenterBin']).mean() # get correct Doppler resolution
specTable['spec_H'] = group['sZePH'].rename({'vel_bins':'vel'})
specTable['spec_V'] = group['sZePV'].rename({'vel_bins':'vel'})

#- now lets convolute the spectra with noise!
centerHeight = 100 # this is important for noise power. select something close to ground
for wl,th,nv,noise in zip(dicSettings['wl'],dicSettings['theta']/2./180.*np.pi,dicSettings['nave'],dicSettings['noise_pow']):
	for elv in dicSettings['elv']:
		specTable['spec_H'].loc[:,elv,wl] = convoluteSpec(specTable['spec_H'].sel(wavelength=wl,elevation=elv).fillna(0),wl,dicSettings['velCenterBin'],dicSettings['eps_diss'],
					                                      noise,nv,th,dicSettings['uwind'],dicSettings['time_int'],centerHeight,0,0,0,dicSettings['tau'])
		specTable['spec_V'].loc[:,elv,wl] = convoluteSpec(specTable['spec_V'].sel(wavelength=wl,elevation=elv).fillna(0),wl,dicSettings['velCenterBin'],dicSettings['eps_diss'],
					                                      noise,nv,th,dicSettings['uwind'],dicSettings['time_int'],centerHeight,0,0,0,dicSettings['tau'])
print(specTable)
# lets plot the Doppler spectra
fig,ax=plt.subplots()
for wl,band in zip(dicSettings['wl'],['X','Ka','W']):
	ax.plot(specTable.vel,dB(specTable.spec_H.sel(wavelength=wl,elevation=90)),label=band+'-band')
ax.legend()#fontsize=16)
ax.grid()
ax.set_ylabel('Ze [dBz]',fontsize=18)
#ax.set_ylim([-50, 0])
ax.set_xlabel('velocity [m/s]',fontsize=18)
ax.tick_params(axis='both',labelsize=16)
plt.tight_layout()
plt.savefig('PSD_McRadar_one_height_exp.png')
plt.show()




















