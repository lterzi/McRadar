import numpy as np
import xarray as xr
import glob
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def lin2db(x):
    return 10*np.log10(x)
def readFiles(f,i):
    print(i,len(files))
    data = xr.open_dataset(f)
    n_dims = len(data.dims)
    if n_dims == 0:
        empty = True
    else:
        empty = False
    if not empty:
        #radarPos = int(f.split('radarPosX')[-1].split('.nc')[0])
        radarPos = int(f.split('radarPosX')[-1].split('_')[0])
        data = data.expand_dims('radar_position').assign_coords(radar_position=[radarPos])
        #print(data.range)
        # specH90 = lin2db(data['spec_H'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2)) # Ka-Band reflectivity
        # specH30 = lin2db(data['spec_H'].sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2))
        # specV30 = lin2db(data['spec_V'].sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2))
            
        # specH90 = specH90.where(specH90 > -50)
        # specH30 = specH30.where(specH30 > -50)
        # specV30 = specV30.where(specV30 > -50)
        data['spec_H'] = data['spec_H'].where(data['spec_H']>10**-50)
        data['spec_V'] = data['spec_V'].where(data['spec_V']>10**-50)
        sZDR = lin2db(data['spec_H']) - lin2db(data['spec_V'])
        ZDR = 10*np.log10(data.Ze_H/data.Ze_V)
        if 'sSNR_H' in data:
            sZDR = sZDR.where(10*np.log10(data.sSNR_H)>10) # TODO: change that back to 10!!!!
            sZDR = sZDR.where(10*np.log10(data.sSNR_V)>10)
        if 'SNR_H' in data:
            ZDR = ZDR.where(10*np.log10(data.SNR_H)>10)
            ZDR = ZDR.where(10*np.log10(data.SNR_V)>10)
            KDP = data.KDP.where(10*np.log10(data.SNR_H)>10)
            KDP = data.KDP.where(10*np.log10(data.SNR_V)>10)
        data['sZDRmax'] = sZDR.max(dim='vel')
        data['ZDR'] = ZDR
        data['KDP'] = KDP
        data = data[['Ze_H','Ze_V','MDV_H','MDV_V','KDP','KDPMono','KDPAgg','sZDRmax','ZDR']]
        return data
path = '/project/meteo/work/L.Terzi/ICON_McSnow_Axel/exp059_u15_q12_ccn7_xi3e7_rt1_habit3_agg4_rfrag0_cfrag0_ffrag0'
range = np.arange(0, 12000, 36)
#files = sorted(glob.glob(path+'/McRadar/particles00004200_water_core/9.6_35.5_94.0GHz_elv90_output_DDA_kdtree_with_habitcode_melted_water_core_nofrozen_separatePart_beta90ifvelstd1_rimed_oriavgTru_gridVolume_beta0_beta_std90_particles00004200.000_radarPosX*.nc'))
#files = [f for f in files if '9.6_35.5_94.0GHz_elv30_output_DDA_kdtree_with_habitcode_melted_water_core_frozen_rimed_oriavgTrue_beta0_beta_std0_convoluteTrue_attenuationFalse_particles00004200.000_radarPosX10000.nc' not in f]
files = sorted(glob.glob(path+'/McRadar/particles00004200/9.6GHz_elv90*McRadar.nc'))
#combined = xr.Dataset()
result = [readFiles(f,i) for i,f in enumerate(files)]
combined = xr.merge(result)
combined = combined.reindex(range=range,method='nearest',tolerance=36)
# files = sorted(glob.glob(path+'/McRadar/particles00004200_water_core/9.6_35.5_94.0GHz_elv30_output_DDA_kdtree_with_habitcode_melted_water_core_nofrozen_separatePart_beta90ifvelstd1_rimed_oriavgTru_gridVolume_beta0_beta_std90_particles00004200.000_radarPosX*.nc'))
# files = [f for f in files if '9.6_35.5_94.0GHz_elv30_output_DDA_kdtree_with_habitcode_melted_water_core_nofrozen_separatePart_beta90ifvelstd1_rimed_oriavgTru_gridVolume_beta0_beta_std90_particles00004200.000_radarPosX10000_testmaxRange.nc' not in f]
# #combined = xr.Dataset()
# result = [readFiles(f,i) for i,f in enumerate(files)]
files = sorted(glob.glob(path+'/McRadar/particles00004200/9.6GHz_elv30*McRadar.nc'))
#combined = xr.Dataset()
result = [readFiles(f,i) for i,f in enumerate(files)]
combined30 = xr.merge(result)
#combined30['range'] = combined30.range*0.5
combined30 = combined30.reindex(range=range,method='nearest',tolerance=36)
combined = xr.merge([combined,combined30])
#print(combined30.range.values)
#quit()
combined.to_netcdf(path+'/McRadar/Radar_Moments_particles00004200.000_newMcRadar.nc')

fontsize = 24
radarPos = np.arange(-20000, 55000, 500)
data = xr.open_dataset(path+'/McRadar/Radar_Moments_particles00004200.000_newMcRadar.nc')
print(data)
#quit()
data = data.reindex(radar_position=radarPos,method='nearest',tolerance=200)

DWRKaW = lin2db(data.Ze_H.sel(wavelength=8.44,elevation=90,method='nearest')) - lin2db(data.Ze_H.sel(wavelength=3.189,elevation=90,method='nearest'))
DWRXKa = lin2db(data.Ze_H.sel(wavelength=31,elevation=90,method='nearest')) - lin2db(data.Ze_H.sel(wavelength=8.44,elevation=90,method='nearest'))

#fig,ax = plt.subplots(nrows=5,figsize=(12,20),sharex=True,constrained_layout=True)
fig,ax = plt.subplots(nrows=4,figsize=(15,15),sharex=True,constrained_layout=True)

p1 = ax[0].pcolormesh(data.radar_position/1000,data.range/1000,lin2db(data.Ze_H.sel(wavelength=31,elevation=90,method='nearest').T),vmin=-30,vmax=60,cmap='turbo',shading='nearest')
cbar = fig.colorbar(p1,ax=ax[0],pad=0.01)#,aspect=40)
cbar.ax.tick_params(labelsize=fontsize)
cbar.set_label('Ze X [dBZ]',fontsize=fontsize+2)

p2 = ax[1].pcolormesh(data.radar_position/1000,data.range/1000,data.MDV_H.sel(wavelength=31,elevation=90,method='nearest').T,vmin=-10,vmax=10,cmap='turbo',shading='nearest')
cbar2 = fig.colorbar(p2,ax=ax[1],pad=0.01)#,aspect=40)
cbar2.ax.tick_params(labelsize=fontsize)
cbar2.set_label('MDV X [m/s]',fontsize=fontsize+2)

# p3 = ax[2].pcolormesh(data.radar_position/1000,data.range/1000,DWRKaW.T,vmin=-1,vmax=20,cmap='turbo',shading='nearest')
# cbar3 = fig.colorbar(p3,ax=ax[2],pad=0.01)#,aspect=40)
# cbar3.ax.tick_params(labelsize=fontsize)
# cbar3.set_label('DWR Ka-W [dB]',fontsize=fontsize+2)

p4 = ax[2].pcolormesh(data.radar_position/1000,data.range/1000,data.ZDR.sel(wavelength=31,elevation=30,method='nearest').T,vmin=0,vmax=3,cmap='turbo',shading='nearest')
cbar4 = fig.colorbar(p4,ax=ax[2],pad=0.01)#,aspect=40)
cbar4.ax.tick_params(labelsize=fontsize)
cbar4.set_label('ZDR X [dB]',fontsize=fontsize+2)

p5 = ax[3].pcolormesh(data.radar_position/1000,data.range/1000,data.KDP.sel(wavelength=31,elevation=30,method='nearest').T,vmin=0,vmax=50,cmap='turbo',shading='nearest')
cbar5 = fig.colorbar(p5,ax=ax[3],pad=0.01)#,aspect=40)
cbar5.ax.tick_params(labelsize=fontsize)
cbar5.set_label('KDP X [°/km]',fontsize=fontsize+2)

for a in ax:
    a.tick_params(labelsize=fontsize)
    a.grid()

fig.supylabel('Range [km]',fontsize=fontsize+2)
ax[3].set_xlabel('Radar Position [km]',fontsize=fontsize+2)
ax[0].set_title('Radar Moments from McRadar Simulation at t=4200s',fontsize=fontsize+2)

plt.savefig(path+'/McRadar/Radar_Moments_4200s_Xband_newMcRadar.png')
plt.close()
quit()
fig,ax = plt.subplots(nrows=5,figsize=(12,20),sharex=True,constrained_layout=True)

p1 = ax[0].pcolormesh(data.radar_position/1000,data.range/1000,lin2db(data.Ze_H.sel(wavelength=8.44,elevation=90,method='nearest').T),vmin=-40,vmax=60,cmap='turbo',shading='nearest')
cbar = fig.colorbar(p1,ax=ax[0],pad=0.01)#,aspect=40)
cbar.ax.tick_params(labelsize=fontsize)
cbar.set_label('Ze Ka [dBZ]',fontsize=fontsize+2)

p2 = ax[1].pcolormesh(data.radar_position/1000,data.range/1000,data.MDV_H.sel(wavelength=3.189,elevation=90,method='nearest').T,vmin=-10,vmax=10,cmap='turbo',shading='nearest')
cbar2 = fig.colorbar(p2,ax=ax[1],pad=0.01)#,aspect=40)
cbar2.ax.tick_params(labelsize=fontsize)
cbar2.set_label('MDV Ka [m/s]',fontsize=fontsize+2)

p3 = ax[2].pcolormesh(data.radar_position/1000,data.range/1000,DWRKaW.T,vmin=-1,vmax=20,cmap='turbo',shading='nearest')
cbar3 = fig.colorbar(p3,ax=ax[2],pad=0.01)#,aspect=40)
cbar3.ax.tick_params(labelsize=fontsize)
cbar3.set_label('DWR Ka-W [dB]',fontsize=fontsize+2)

p4 = ax[3].pcolormesh(data.radar_position/1000,data.range/1000,data.ZDR.sel(wavelength=3,elevation=30,method='nearest').T,vmin=-1,vmax=5,cmap='turbo',shading='nearest')
cbar4 = fig.colorbar(p4,ax=ax[3],pad=0.01)#,aspect=40)
cbar4.ax.tick_params(labelsize=fontsize)
cbar4.set_label('ZDR W [dB]',fontsize=fontsize+2)

p5 = ax[4].pcolormesh(data.radar_position/1000,data.range/1000,data.KDP.sel(wavelength=3,elevation=30,method='nearest').T,vmin=0,vmax=200,cmap='turbo',shading='nearest')
cbar5 = fig.colorbar(p5,ax=ax[4],pad=0.01)#,aspect=40)
cbar5.ax.tick_params(labelsize=fontsize)
cbar5.set_label('KDP W [°/km]',fontsize=fontsize+2)

for a in ax:
    a.tick_params(labelsize=fontsize)
    a.grid()

fig.supylabel('Range [km]',fontsize=fontsize+2)
ax[4].set_xlabel('Radar Position [km]',fontsize=fontsize+2)
ax[0].set_title('Radar Moments from McRadar Simulation at t=4200s',fontsize=fontsize+2)

plt.savefig(path+'/McRadar/Radar_Moments_4200s_Wband.png')
plt.close()