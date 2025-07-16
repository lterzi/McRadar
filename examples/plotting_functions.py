'''
This is the beginning of plotting routines for McSnow output
'''
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import warnings
from matplotlib.ticker import FormatStrFormatter
from string import ascii_lowercase

warnings.filterwarnings('ignore')

def setlabel(ax, label, loc=2, borderpad=0.6, **kwargs):
    legend = ax.get_legend()
    if legend:
        ax.add_artist(legend)
    line, = ax.plot(np.nan,np.nan,color='none',label=label)
    label_legend = ax.legend(handles=[line],loc=loc,handlelength=0,handleheight=0,handletextpad=0,borderaxespad=0,borderpad=borderpad,frameon=False,**kwargs)
    label_legend.remove()
    ax.add_artist(label_legend)
    line.remove()
    return ax
#import math
def lin2db(data):
	return 10*np.log10(data)

def plotOneRow(ax,output,atmoFile,i,ylim=None,legend=False,largeDWR=False,xlabel=False,xlabel1=False,ylabel=False,cmap='turbo',plotZDRsplit=False,plotTemp=True,move2zero=True):
	height = atmoFile[:,0]
	Temp = atmoFile[:,2] -273.15
	atmoPD = pd.DataFrame(data=Temp,index=height,columns=['Temp'])
	atmoPD.index.name='range'
	atmoXR = atmoPD.to_xarray()
	atmoReindex = atmoXR.reindex_like(output,method='nearest')
	output = xr.merge([atmoReindex,output])
	output['range'] = output.range/1e3
	specH90 = lin2db(output['spec_H'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2)) # Ka-Band reflectivity
	specH90W = lin2db(output['spec_H'].sel(wavelength=3,elevation=90,method='nearest',tolerance=2)) # Ka-Band reflectivity
	specH30 = lin2db(output['spec_H'].sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2))
	specV30 = lin2db(output['spec_V'].sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2))
		
	specH90 = specH90.where(specH90 > -60)
	specH90W = specH90W.where(specH90W > -60)
	specH30 = specH30.where(specH30 > -60)
	specV30 = specV30.where(specV30 > -60)
	sZDR = specH30 - specV30
	ZDR = lin2db(output.Ze_H/output.Ze_V)
	sZDR = sZDR.where(lin2db(output.sSNR_H.sel(wavelength=3.189,elevation=30,method='nearest'))>10)
	sZDR = sZDR.where(lin2db(output.sSNR_V.sel(wavelength=3.189,elevation=30,method='nearest'))>10)
	ZDR = ZDR.where(lin2db(output.SNR_H.sel(wavelength=3.189,elevation=30,method='nearest'))>10)
	ZDR = ZDR.where(lin2db(output.SNR_V.sel(wavelength=3.189,elevation=30,method='nearest'))>10)
	KDP = output.KDP.where(lin2db(output.SNR_H.sel(wavelength=3.189,elevation=30,method='nearest'))>10)
	KDP = output.KDP.where(lin2db(output.SNR_V.sel(wavelength=3.189,elevation=30,method='nearest'))>10)
     

	if plotTemp:
		ycoord = 'Temp'
		ylbl='Temperature [°C]'
	else:
		ycoord = 'range'
		ylbl='Height [km]'

	if 'Ze_H_Agg' in output:
		lns1=ax[i,0].plot(lin2db(output['Ze_H'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2)),output[ycoord],lw=2,c='C0',label='Ze')
		lns2=ax[i,0].plot(lin2db(output['Ze_H_Agg'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2)),output[ycoord],lw=2,c='C0',ls='-.',label=r'Ze$_{\rm Agg}$')
		lns3=ax[i,0].plot(lin2db(output['Ze_H_Mono'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2)),output[ycoord],lw=2,c='C0',ls=':',label=r'Ze$_{\rm Mono}$')
		
		lns1 = lns1+lns2+lns3
	else:
		lns1=ax[i,0].plot(lin2db(output['Ze_H'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2)),output[ycoord],lw=2,label='Ze')
	if xlabel:
		ax[i,0].set_xlabel('Ze [dBz]',fontsize=28)
	if not ylim:
		if ycoord=='Temp':
			ax[i,0].set_ylim([0,np.min(output[ycoord])-1])	
		else:
			ax[i,0].set_ylim([0,np.max(output[ycoord])])	
	#ax[i,0].set_xlim([-50,25])
	ax[i,0].set_xlim([-40,25])
	ax[i,0].set_xticks([-40,-20,0,20])
	
	if ylabel:
		ax[i,0].set_ylabel(ylbl,fontsize=28)
	ax1 = ax[i,0].twiny()
	lns2=ax1.plot(output.MDV_H.sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2),output[ycoord],lw=2,color='C1',label='MDV')
	if xlabel1:
		ax1.set_xlabel(r'MDV [ms$^{-1}$]',fontsize=28,color='C1')
	ax1.tick_params(axis='both', which='major', labelsize=24,labelcolor='C1',color='C1')
	ax1.set_xlim([-1.2,0])
	ax1.set_xticks([-1,-2/3,-1/3,0])
	#ax1.set_xlim([-2,0])
	#ax1.set_xticks([-2,-1,0])
	ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	ax1.tick_params(labelsize=24)
	if legend:
		lns = lns1 + lns2
		labs = [l.get_label() for l in lns]
		ax[i,0].legend(lns, labs,fontsize=18,loc='upper right')#ncol=2,
	#ax[0].legend(fontsize=15,ncol=4,bbox_to_anchor=(0,1,1.,0))
	  
	DWRKaW = lin2db(output['Ze_H'].where(lin2db(output.SNR_H)>10).sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2)) - lin2db(output['Ze_H'].where(lin2db(output.SNR_H)>10).sel(wavelength=3.189,elevation=90,method='nearest',tolerance=2))
	DWRKaW = DWRKaW.rolling(range=2).mean()
	Temp_dwr = output['Temp'].rolling(range=2).mean()
	ax[i,1].plot(DWRKaW,output[ycoord],lw=2,c='C0',label='DWR$_{KaW}$')
	#DWRKaW = lin2db(output['Ze_H'].where(lin2db(output.SNR_H)>10).sel(wavelength=8.44,elevation=30,method='nearest',tolerance=2)) - lin2db(output['Ze_H'].where(lin2db(output.SNR_H)>10).sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2))
	#ax[i,1].plot(DWRKaW,output['Temp'],lw=2,c='C0',ls=':',label='30° elv')
	
	DWRXKa = lin2db(output['Ze_H'].where(lin2db(output.SNR_H)>10).sel(wavelength=31.23,elevation=90,method='nearest',tolerance=2)) - lin2db(output['Ze_H'].where(lin2db(output.SNR_H)>10).sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2))
	DWRXKa = DWRXKa.rolling(range=2).mean()
	ax[i,1].plot(DWRXKa,output[ycoord],lw=2,c='C1',label='DWR$_{XKa}$')
	#DWRXKa = lin2db(output['Ze_H'].where(lin2db(output.SNR_H)>10).sel(wavelength=31.23,elevation=30,method='nearest',tolerance=2)) - lin2db(output['Ze_H'].where(lin2db(output.SNR_H)>10).sel(wavelength=8.44,elevation=30,method='nearest',tolerance=2))
	#ax[i,1].plot(DWRXKa,output['Temp'],lw=2,c='C1',ls=':')
	if xlabel:
		ax[i,1].set_xlabel('DWR [dB]',fontsize=28)
	if legend:
		ax[i,1].legend(fontsize=18,loc='upper right')
	ax[i,1].set_xlim([-0.1,8])
	ax[i,1].set_xticks([0,2,4,6,8])
	#ax[i,1].set_xlim([-0.2,15])
	#ax[i,1].set_xticks([0,5,10,15])
	sDWRKaW = specH90-specH90W
	#sDWRKaW1 = specH901-specH901W
	
	p1=ax[i,2].pcolormesh(output.vel,output[ycoord],specH90,vmin=-40,vmax=10,cmap=cmap,shading='auto')
	
	cb = plt.colorbar(p1,ax=ax[i,2],extend='both')
	cb.set_label('sZe [dBz]',fontsize=26)
	
	cb.ax.tick_params(labelsize=24)
	if xlabel:
		ax[i,2].set_xlabel(r'v$_{\rm Doppler}$ [ms$^{-1}$]',fontsize=32)
#	
	ax[i,2].set_xlim([-2,0.5])
	ax[i,2].set_xticks([-2,-1,0])
	
	if 'KDPAgg' in output: 
		
		ax[i,3].plot(KDP.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],c='C0',lw=2,label='KDP')
		ax[i,3].plot(output.KDPAgg.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],lw=2,c='C0',ls='-.',label=r'KDP$_{\rm Agg}$')
		ax[i,3].plot(output.KDPMono.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],lw=2,ls=':',c='C0',label=r'KDP$_{\rm Mono}$')      
		if legend:
			ax[i,3].legend(fontsize=18,loc='upper right')
	else:
		ax[i,3].plot(KDP.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],lw=2)
	if xlabel:
		ax[i,3].set_xlabel(r'KDP [°km$^{-1}$]',fontsize=28)
	ax[i,3].set_xlim([-0,3.2])
	ax[i,3].set_xticks([0,1,2,3])
	
	ZDR = ZDR.rolling(range=2).mean()
	if 'ZDR_Agg' in output and plotZDRsplit:
		
		ax[i,4].plot(ZDR.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],lw=2)
		ax[i,4].plot(output.ZDR_Agg.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],lw=2,c='C0',ls='-.',label='ZDR Agg')
		ax[i,4].plot(output.ZDR_Mono.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],lw=2,c='C0',ls=':',label='ZDR Mono')
	else:
		ax[i,4].plot(ZDR.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],c='C0',lw=2,label='ZDR')
	
	if xlabel:
		ax[i,4].set_xlabel(r'ZDR [dB]',fontsize=28)
	
	sZDR = sZDR.rolling(vel=5).mean()
	sZDRmax = sZDR.max(dim='vel')
	sZDRmax = sZDRmax.rolling(range=2).mean()
	
	ax[i,4].plot(sZDRmax,output[ycoord],lw=2,label=r'sZDR$_{\rm max}$',c='C1')
	if legend:
		ax[i,4].legend(loc='upper right',fontsize=18)#,ncol=2)	
	ax[i,4].set_xlim([0,5])
	ax[i,4].set_xticks([0,1,2,3,4,5]) 
	
	if move2zero:
		vel = output.vel*0.5 # spectrum is squished because of 30° elevation
		maxVel = vel.where(~np.isnan(sZDR)).max(dim='vel')+np.diff(vel)[0]
		vel2zero = vel-maxVel
	else:
		vel2zero = output.vel
	
	p1=ax[i,5].pcolormesh(vel2zero.fillna(0).values.T,output[ycoord].values.reshape(len(output.Temp),1),sZDR.values,vmin=0,vmax=5,cmap=cmap,shading='auto')
	cb = plt.colorbar(p1,ax=ax[i,5],extend='both')
	cb.set_label('sZDR [dB]',fontsize=26)
	cb.set_ticks([0,1,2,3,4,5])
	cb.ax.tick_params(labelsize=24)
	if xlabel:
		ax[i,5].set_xlabel(r'v$^*_{\rm Doppler}$ [ms$^{-1}$]',fontsize=28)
	ax[i,5].set_xlim([-1,0.05])
	ax[i,5].set_xticks([-1,-0.5,0])
	
	for ia,a in enumerate(ax[i,:]):
		#if ia == 0:
		#	a.set_ylabel('T [°C]',fontsize=24)
		if ylim:
			a.set_ylim(ylim)
			a.set_yticks(np.arange(ylim[1],ylim[0],10)[::-1])
		
		a.tick_params(which='both',labelsize=26)
		#a.set_ylim([0,-31])	
		a.grid()
	return ax

def plotOverview6Panels(output,name2save,allDWR=False, plotEdges=False,plotTemp=True, cmap='turbo'):
	#cmap = cmap.copy()
	#cmap.set_extremes(under='bisque', over='Grey') 

	nwl = len(output.wavelength)
	
	specH90 = lin2db(output['spec_H'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2)) # Ka-Band reflectivity
	specH30 = lin2db(output['spec_H'].sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2))
	specV30 = lin2db(output['spec_V'].sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2))
		
	specH90 = specH90.where(specH90 > -50)
	specH30 = specH30.where(specH30 > -50)
	specV30 = specV30.where(specV30 > -50)
	sZDR = specH30 - specV30
	ZDR = 10*np.log10(output.Ze_H/output.Ze_V)
	if 'sSNR_H' in output:
		sZDR = sZDR.where(10*np.log10(output.sSNR_H.sel(wavelength=3.189,elevation=30,method='nearest'))>10) # TODO: change that back to 10!!!!
		sZDR = sZDR.where(10*np.log10(output.sSNR_V.sel(wavelength=3.189,elevation=30,method='nearest'))>10)
	if 'SNR_H' in output:
		ZDR = ZDR.where(10*np.log10(output.SNR_H.sel(wavelength=3.189,elevation=30,method='nearest'))>10)
		ZDR = ZDR.where(10*np.log10(output.SNR_V.sel(wavelength=3.189,elevation=30,method='nearest'))>10)
		KDP = output.KDP.where(10*np.log10(output.SNR_H.sel(wavelength=3.189,elevation=30,method='nearest'))>10)
		KDP = output.KDP.where(10*np.log10(output.SNR_V.sel(wavelength=3.189,elevation=30,method='nearest'))>10)
	
	if plotTemp:
		ycoord='Temp'
		ylabel='T [°C]'
	else:
		ycoord='sHeight'
		ylabel='Height [m]'

	fig,ax = plt.subplots(ncols=3,nrows=2,figsize=(15,10),sharey=True)
	
	ax[0,0].plot(lin2db(output['Ze_H'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2)),output[ycoord],lw=2)
	ax[0,0].set_xlabel('Ze Ka-Band [dBz]',fontsize=24)
	ax[0,0].set_ylim([0,np.min(output.Temp)-1])	
	ax[0,0].set_ylabel(ylabel,fontsize=24)
	ax[0,0].set_xlim([-30,25])
	ax[0,0].set_xticks([-30,-20,-10,0,10,20])
	ax[0,0].text(ax[0,0].get_xlim()[0]+0.03*(ax[0,0].get_xlim()[1]-ax[0,0].get_xlim()[0]),ax[0,0].get_ylim()[1]-0.08*(ax[0,0].get_ylim()[1]-ax[0,0].get_ylim()[0]),'(a)',fontsize=20)
	
	ax1 = ax[0,0].twiny()
	ax1.plot(output['MDV_H'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2),output[ycoord],lw=2,c='C1')
	ax1.set_xlabel(r'MDV [ms$^{-1}$]',fontsize=22,color='C1')
	ax1.set_xlim([-1.5,0])
	ax1.tick_params(axis='both', which='major', labelsize=20,labelcolor='C1',color='C1')
	
	if allDWR:
		DWRKaW = lin2db(output['Ze_H'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2)) - lin2db(output['Ze_H'].sel(wavelength=3.189,elevation=90,method='nearest',tolerance=2))
		ax[0,1].plot(DWRKaW,output['Temp'],lw=2,c='C0',label='DWR$_{KaW}$, 90° elv')
		DWRKaW = lin2db(output['Ze_H'].sel(wavelength=8.44,elevation=30,method='nearest',tolerance=2)) - lin2db(output['Ze_H'].sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2))
		ax[0,1].plot(DWRKaW,output[ycoord],lw=2,c='C0',ls=':',label='30° elv')
		if nwl==3:
			DWRXKa = lin2db(output['Ze_H'].sel(wavelength=31.23,elevation=90,method='nearest',tolerance=2)) - lin2db(output['Ze_H'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2))
			ax[0,1].plot(DWRXKa,output[ycoord],lw=2,c='C1',label='DWR$_{XKa}$')
			DWRXKa = lin2db(output['Ze_H'].sel(wavelength=31.23,elevation=30,method='nearest',tolerance=2)) - lin2db(output['Ze_H'].sel(wavelength=8.44,elevation=30,method='nearest',tolerance=2))
			ax[0,1].plot(DWRXKa,output[ycoord],lw=2,c='C1',ls=':')
		ax[0,1].set_xlabel('DWR [dB]',fontsize=24)
		ax[0,1].legend(fontsize=12)
	else:
		DWRKaW = lin2db(output['Ze_H'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2)) - lin2db(output['Ze_H'].sel(wavelength=3.189,elevation=90,method='nearest',tolerance=2))
		ax[0,1].plot(DWRKaW,output[ycoord],lw=2,c='C0',label='DWR$_{KaW}$')
		DWRXKa = lin2db(output['Ze_H'].sel(wavelength=31.23,elevation=90,method='nearest',tolerance=2)) - lin2db(output['Ze_H'].sel(wavelength=8.44,elevation=90,method='nearest',tolerance=2))
		ax[0,1].plot(DWRXKa,output[ycoord],lw=2,c='C1',label='DWR$_{XKa}$')
		ax[0,1].set_xlabel(r'DWR [dB]',fontsize=24)
		ax[0,1].legend(fontsize=12)
		#ax[1].legend(fontsize=12)
		#ax[0,1].set_xlim([0,9.5])
	ax[0,1].text(ax[0,1].get_xlim()[0]+0.03*(ax[0,1].get_xlim()[1]-ax[0,1].get_xlim()[0]),ax[0,1].get_ylim()[1]-0.08*(ax[0,1].get_ylim()[1]-ax[0,1].get_ylim()[0]),'(b)',fontsize=20)
	
	p1=ax[0,2].pcolormesh(output.vel,output[ycoord],specH90,vmin=-30,vmax=10,cmap=cmap,shading='auto')
	if ('minVel40dB' in output) and (plotEdges==True):
		ax[0,2].plot(output.minVel40dB.sel(wavelength=8.44,method='nearest',tolerance=2),output.Temp,lw=2,c='r')
		ax[0,2].plot(output.maxVel40dB.sel(wavelength=8.44,method='nearest',tolerance=2),output.Temp,lw=2,c='r')
	cb = plt.colorbar(p1,ax=ax[0,2],extend='both')
	cb.set_label('sZeH Ka-Band [dBz]',fontsize=24)
	cb.set_ticks([-30,-20,-10,0,10])
	cb.ax.tick_params(labelsize=20)
	ax[0,2].set_xlabel(r'Doppler velocity [ms$^{-1}$]',fontsize=24)
	ax[0,2].set_xlim([-2,0])
	ax[0,2].text(ax[0,2].get_xlim()[0]+0.03*(ax[0,2].get_xlim()[1]-ax[0,2].get_xlim()[0]),ax[0,2].get_ylim()[1]-0.08*(ax[0,2].get_ylim()[1]-ax[0,2].get_ylim()[0]),'(c)',fontsize=20)
	
	if 'KDPAgg' in output: 
		ax[1,0].plot(KDP.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],lw=2,label='KDP')
		ax[1,0].plot(output.KDPAgg.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],lw=2,c='C0',ls='-.',label='Agg')
		ax[1,0].plot(output.KDPMono.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],lw=2,ls=':',c='C0',label='Mono')
		ax[1,0].legend(fontsize=14)
	else:
		ax[1,0].plot(KDP.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],lw=2)
	ax[1,0].set_xlabel(r'KDP [°km$^{-1}$]',fontsize=24)
	ax[1,0].text(ax[1,0].get_xlim()[0]+0.04*(ax[1,0].get_xlim()[1]-ax[1,0].get_xlim()[0]),ax[1,0].get_ylim()[1]-0.08*(ax[1,0].get_ylim()[1]-ax[1,0].get_ylim()[0]),'(d)',fontsize=20)
	ax[1,0].set_ylabel('T [°C]',fontsize=24)
	
	
	ax[1,1].plot(ZDR.sel(wavelength=3.189,elevation=30,method='nearest',tolerance=2),output[ycoord],lw=2,label='ZDR')
	ax[1,1].set_xlabel(r'ZDR [dB]',fontsize=24)
	sZDR = sZDR.rolling(vel=5).mean()
	sZDRmax = sZDR.max(dim='vel')
	ax[1,1].plot(sZDRmax,output[ycoord],lw=2,label=r'sZDR_${\mathrm{max}}$')
	#ax[1,0].set_xlabel(r'sZDR$_{\rm max}$ [dB]',fontsize=24)
	ax[1,1].legend(fontsize=12)
	ax[1,1].text(ax[1,1].get_xlim()[0]+0.03*(ax[1,1].get_xlim()[1]-ax[1,1].get_xlim()[0]),ax[1,1].get_ylim()[1]-0.08*(ax[1,1].get_ylim()[1]-ax[1,1].get_ylim()[0]),'(e)',fontsize=20)
	
	
	vel = output.vel*0.5 # spectrum is squished because of 30° elevation
	maxVel = vel.where(~np.isnan(sZDR)).max(dim='vel')
	vel2zero = vel-maxVel
	p1=ax[1,2].pcolormesh(vel2zero.fillna(0).values.T,output[ycoord].values.reshape(len(output.Temp),1),sZDR.values,vmin=0,vmax=5,cmap=cmap,shading='auto')
	
	#p1=ax[4].pcolormesh(output.vel,output.Temp,ZDR,vmin=-1,vmax=5,cmap=getNewNipySpectral(),shading='auto')
	cb = plt.colorbar(p1,ax=ax[1,2],extend='both')
	cb.set_label('sZDR [dB]',fontsize=24)
	cb.ax.tick_params(labelsize=20)
	cb.set_ticks([0,1,2,3,4,5])
	ax[1,2].set_xlabel(r'Doppler velocity [ms$^{-1}$]',fontsize=24)
	ax[1,2].set_xlim([-1,0.05])
	ax[1,2].text(ax[1,2].get_xlim()[0]+0.03*(ax[1,2].get_xlim()[1]-ax[1,2].get_xlim()[0]),ax[1,2].get_ylim()[1]-0.08*(ax[1,2].get_ylim()[1]-ax[1,2].get_ylim()[0]),'(f)',fontsize=20)
	
	for a in ax[1,:]:
		a.tick_params(which='both',labelsize=20)
		a.grid()
		if plotTemp:
			#a.set_ylim([0,-20])
			a.axhline(y=-15,c='grey',ls='--')
	for a in ax[0,:]:
		a.tick_params(which='both',labelsize=20)
		a.grid()
		if plotTemp:
			a.axhline(y=-15,c='grey',ls='--')
	plt.tight_layout()
	#if allDWR:
	#	name2save = 'Profiles_overview_testColumn.png'
	#else:
	#	name2save = 'Profiles_overview_singleDWR_new_McRadar.png'
	plt.tight_layout()
	plt.savefig(name2save)
	plt.close()	
	
