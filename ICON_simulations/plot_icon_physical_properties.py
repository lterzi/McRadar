#%%
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import mcradar as mcr
from scipy import constants
from mcradar.tableOperator import creatRadarCols
from scipy.spatial import cKDTree 
import glob
import os
from sys import argv

#%%
def grid_info(gridfile='Torus_Triangles_1024x4_150m.nc'):
    print(f"Trying to load grid info from {gridfile=}")
    with xr.open_dataset(gridfile) as grid:
       dx = dy = dz = float(grid.edge_length.isel(edge=0))
       vol = dx * dz * grid.domain_height
       domain_length = grid.domain_length
       domain_length_y = (grid.cartesian_y_vertices.max().item() - grid.cartesian_y_vertices.min().item())/np.sin(np.deg2rad(60))
       print(f"{dx=} {dz=} {domain_length=} {vol=}")
    return dict(dx=dx, dz=dz, vol=vol, domain_length=domain_length, domain_length_y=domain_length_y)
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

varentry = {
        'm_w'       : ( 1, 'liq. mass [kg]'),
        'm_i'       : ( 2, 'ice mass [kg]'),
        'm_r'       : ( 3, 'rimed mass [kg]'),
        'v_r'       : ( 4, 'volume rime [m3]'),
        'd'         : ( 5, 'diameter [m]'),
        'A'         : ( 6, 'projected area [m2]'),
        'xi'        : ( 7, 'multiplicity'),
        'mm'        : ( 8, 'monomer multiplicity'),
        'statusb'   : ( 9, 'status bit'),
        'vt'        : (10, 'vt'),
        'gblCellId' : (11, 'gblCellId'),
        'jk'        : (12, 'jk'),
        'm_f'       : (13, 'frozen mass [kg]'),
        'T'         : (14, 'particle Temperature [K]'),
        'atmoT'     : (15, 'atmospheric temperature'),
        'dQdt'      : (16, 'dQdt'),
        'V_i'       : (17, 'volume of ice part [m3]'),
        'phi'       : (18, 'aspect ratio []'),
        'pp'        : (19, 'number of prolate particles (monomers) [#]'),
        'dd'        : (20, 'number of dendrite monomers [#]'),
        }
def ds_get_var(ds, varname, multiplicity=True):
    get_var = lambda vname: ds[f'addVar{varentry[vname][0]:04d}']
    if varname == 'm_tot':
        vardata = xr.concat([ get_var(vname) for vname in ('m_f', 'm_w', 'm_i', 'm_r')], dim='tracer').sum('tracer')
    else:
        vardata = get_var(varname)

    if multiplicity:
        vardata *= get_var('xi')
    return vardata


#%%
fileName , pfile = argv

#filePath = '/scratch/f/Fabian.Jakub/icon_mcsnow_build/experiments/mcsnow2d.nosani.sphabit3.rt3/'
#filePath = 'ICON/experiments/mcs.wk82.rt1.sph3.agg5.coll3.frag0/'
#filePath = 'ICON/experiments/mcs.wk82.rt1.sph3.agg5.coll3.frag1.iceicebreak10/'
#filePath = 'ICON_output/'
#filePath = '/scratch/l/L.Terzi/icon/experiments/mcs.wk82.rt1.sph3.agg5.coll3.frag1.iceicebreak10'
filePath = pfile.rsplit('/',1)[0]+'/'

if not os.path.exists('{}evolution_plots'.format(filePath)):
    os.makedirs('{}evolution_plots'.format(filePath))
#if not os.path.exists('{}nc4mcradar'.format(filePath)):
#    os.makedirs('{}nc4mcradar'.format(filePath))

ginfo = grid_info(filePath+'Torus_Triangles_1024x4_150m.nc') #gridfile
#files = sorted(glob.glob(filePath+'ICON_output*.nc'))
#pfile=filePath+'particles00002760.000.nc' # 'particles00006960.000.nc'#'particles00001800.000.nc'
#for pfile in files:
#pfile = '{}ICON_output_for_McRadar_particles0000{}.000.nc'.format(filePath,time)

print(pfile)
fname = pfile.split('/')[-1].split('.nc')[0]
timestep = float(fname.split('particles')[-1])
if not os.path.isfile('{}evolution_plots/w_vel_x_{}1.png'.format(filePath,fname)):
    print(fname)
    ds = xr.open_dataset(pfile)
    #ds = ds.rename({'m_tot':'mTot','vel':'vt','w_vel':'vel'})

    #quit()
    ds['x'] = ds.longitude / (2*np.pi) * ginfo['domain_length']
    ds['y'] = ds.latitude              * ginfo['domain_length_y']

    # ds = ds.rename({'addVar0001':'m_w','addVar0002':'m_i','addVar0003':'m_r',
    #                 'addVar0004':'v_r','addVar0005':'dia','addVar0006':'A',
    #                 'addVar0007':'xi','addVar0008':'sNmono','addVar0009':'statusb',
    #                 'addVar0010':'vt','addVar0011':'gblCellId','addVar0012':'jk',
    #                 'addVar0013':'m_f','addVar0014':'T','addVar0015':'atmoT',
    #                 'addVar0016':'dQdt','addVar0017':'V_i','addVar0018':'sPhi',
    #                 'addVar0019':'sp%pp','addVar0020':'sp%dd'})
    # #dss = ds
    #quit()
    #print(ds)
    #quit()
    ds['mTot'] = ds.m_f+ ds.m_w+ ds.m_i+ ds.m_r # funktioniert nicht weil m_f,.. nicht defined
    #ds.to_netcdf('{}/nc4mcradar/ICON_output_for_McRadar_{}.nc'.format(filePath,fname))
    #ds.to_netcdf('{}/nc4mcradar/{}.nc'.format(filePath,fname)) 
    #ds['prolate_ratio'] = ds['sp%pp'] / ds.sNmono # funktioniert nicht weil pp,... nicht defined
    idx = np.arange(ds.noParts.size); np.random.shuffle(idx); dss=ds.isel(noParts=idx[:int(1e6)])
    #dss = ds
    # print(dss)
    # for var in dss.data_vars:
    #     print(var)

    
    #%%
    dss['frozen_frac'] = dss.m_f / (dss.mTot)
    dss['ice_fraction'] = (dss.m_f + dss.m_i + dss.m_r) / (dss.mTot)
    dss['liquid_frac'] = dss.m_w / (dss.mTot)
    dss['rime_frac'] = dss.m_r / (dss.mTot)
    dss['ice_mass'] = dss.m_f + dss.m_i + dss.m_r
    dss['x'] = dss.x*1e-3
    dss['altitude'] = dss.altitude*1e-3
    # TODO: separate different masses and fractions into two plots
    #dss = xr.open_dataset('ICON_output_for_McRadar_particles00002760.000.nc')

    fig,ax = plt.subplots(nrows=4,figsize=(20,20),constrained_layout=True,sharey=True,sharex=True)
    #p1 = dss.plot.scatter(ax=ax,x='x', y='altitude', hue='m_tot', s=10, cmap='turbo',norm=matplotlib.colors.LogNorm(1e-12,1e-4),add_colorbar=False)
    p1 = ax[0].scatter(x=dss.x, y=dss.altitude, c=dss.ice_mass, s=0.01, cmap='Spectral_r',norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    cbar = fig.colorbar(p1,ax=ax[0],pad=0.001,aspect=20)
    cbar.set_label('total ice mass [kg]',fontsize=22)
    cbar.ax.tick_params(labelsize=20)
    ax[0].set_title('Timestep: {}'.format(int(timestep)),fontsize=22)

    p1 = ax[1].scatter(x=dss.x, y=dss.altitude, c=dss.m_w, s=0.01, cmap='Spectral_r',norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    cbar = fig.colorbar(p1,ax=ax[1],pad=0.001,aspect=20)
    cbar.set_label('liquid mass [kg]',fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    p1 = ax[2].scatter(x=dss.x, y=dss.altitude, c=dss.m_f, s=0.01, cmap='Spectral_r',norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    cbar = fig.colorbar(p1,ax=ax[2],pad=0.001,aspect=20)
    cbar.set_label('frozen mass [kg]',fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    #p1 = ax[3].scatter(x=dss.x, y=dss.altitude, c=dss.m_i, s=0.01, cmap='Spectral_r',norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    #cbar = fig.colorbar(p1,ax=ax[3],pad=0.001,aspect=20)
    #cbar.set_label('ice mass [kg]',fontsize=22)
    #cbar.ax.tick_params(labelsize=20)

    p1 = ax[3].scatter(x=dss.x, y=dss.altitude, c=dss.m_r, s=0.01, cmap='Spectral_r',norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    cbar = fig.colorbar(p1,ax=ax[3],pad=0.001,aspect=20)
    cbar.set_label('rime mass [kg]',fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    for a in ax:
        a.tick_params(labelsize=20)
        a.set_ylabel('Altitude [km]',fontsize=22)
        a.set_ylim([0,12])
        a.set_xlim([-60,60])
        
    ax[3].set_xticks([-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60])
    ax[3].set_xlabel('x [km]',fontsize=18)
        
    plt.savefig('{}evolution_plots/different_masses_{}.png'.format(filePath,fname))
    plt.close()
    
    
    fig,ax = plt.subplots(nrows=4,figsize=(20,20),constrained_layout=True,sharey=True,sharex=True)

    p1 = ax[0].scatter(x=dss.x, y=dss.altitude, c=dss.ice_fraction, s=0.01, cmap='Spectral_r',vmin=0,vmax=1)#,norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    cbar = fig.colorbar(p1,ax=ax[0],pad=0.001,aspect=20)
    cbar.set_label('ice fraction',fontsize=22) # ((m_f + m_i + m_r)/m_tot)
    cbar.ax.tick_params(labelsize=20)
    ax[0].set_title('Timestep: {}'.format(int(timestep)),fontsize=22)

    p1 = ax[1].scatter(x=dss.x, y=dss.altitude, c=dss.frozen_frac, s=0.01, cmap='Spectral_r',vmin=0,vmax=1)#,norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    cbar = fig.colorbar(p1,ax=ax[1],pad=0.001,aspect=20)
    cbar.set_label('frozen fraction',fontsize=22) # (m_f/m_tot)
    cbar.ax.tick_params(labelsize=20)

    p1 = ax[2].scatter(x=dss.x, y=dss.altitude, c=dss.rime_frac, s=0.01, cmap='Spectral_r',vmin=0,vmax=1)#,norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    cbar = fig.colorbar(p1,ax=ax[2],pad=0.001,aspect=20)
    cbar.set_label('rime fraction',fontsize=22) # (m_r/m_tot)
    cbar.ax.tick_params(labelsize=20)

    p1 = ax[3].scatter(x=dss.x, y=dss.altitude, c=dss.liquid_frac, s=0.01, cmap='Spectral_r',vmin=0,vmax=1)#,norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    cbar = fig.colorbar(p1,ax=ax[3],pad=0.001,aspect=20)
    cbar.set_label('liquid fraction',fontsize=22) # (m_w/m_tot)
    cbar.ax.tick_params(labelsize=20)
    
    
    for a in ax:
        a.tick_params(labelsize=20)
        a.set_ylabel('Altitude [km]',fontsize=22)
        a.set_ylim([0,12])
        a.set_xlim([-60,60])
        
    ax[3].set_xticks([-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60])
    ax[3].set_xlabel('x [km]',fontsize=18)
    plt.savefig('{}evolution_plots/mass_fractions_{}.png'.format(filePath,fname))
    plt.close()
    
    
    # fig,ax = plt.subplots(figsize=(20,5),constrained_layout=True)
    # #p1 = dss.plot.scatter(ax=ax,x='x', y='altitude', hue='m_tot', s=10, cmap='turbo',norm=matplotlib.colors.LogNorm(1e-12,1e-4),add_colorbar=False)
    # p1 = plt.scatter(x=dss.x, y=dss.altitude, c=dss.m_tot, s=0.01, cmap='turbo',norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    # cbar = fig.colorbar(p1,ax=ax,pad=0.001,aspect=20)
    # cbar.set_label('Total mass [kg]',fontsize=18)
    # cbar.ax.tick_params(labelsize=16)
    # ax.tick_params(labelsize=16)
    # ax.set_ylabel('Altitude [km]',fontsize=18)
    # ax.set_xlabel('x [km]',fontsize=18)
    # ax.set_ylim([0,12])
    # ax.set_xlim([-60,60])
    # ax.set_xticks([-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60])
    # ax.set_title('Timestep: {}'.format(int(timestep)),fontsize=18)
    # plt.savefig('evolution_plots/m_tot_x_{}_turbo.png'.format(fname))
    # plt.close()
    # # fig,ax = plt.subplots(figsize=(15,6),constrained_layout=True)
    # # p1 = dss.plot.scatter(ax=ax,x='x', y='altitude', hue='m_tot', s=10, cmap='turbo',norm=matplotlib.colors.LogNorm(1e-12,1e-4),add_colorbar=False)
    # # #p1 = plt.scatter(x=dss.x, y=dss.altitude, c=dss.m_tot, s=0.1, cmap='turbo',norm=matplotlib.colors.LogNorm(1e-12,1e-4),alpha=0.5)
    # # cbar = fig.colorbar(p1,ax=ax,pad=0.01,aspect=30)
    # # cbar.set_label('total mass [kg]',fontsize=16)
    # # cbar.ax.tick_params(labelsize=14)
    # # ax.tick_params(labelsize=14)
    # # ax.set_ylabel('Altitude [km]',fontsize=16)
    # # ax.set_xlabel('x [km]',fontsize=16)
    # # ax.set_ylim([0,12])
    # # ax.set_xlim([-60,60])
    # # plt.savefig('m_tot_x_{}_turbo_hue.png'.format(fname))
    # # plt.show()


    fig,ax = plt.subplots(figsize=(20,5),constrained_layout=True)
    p1 = plt.scatter(x=dss.x, y=dss.altitude, c=dss.mTot, s=0.01, cmap='Spectral_r',norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    cbar = fig.colorbar(p1,ax=ax,pad=0.001,aspect=20)
    cbar.set_label('Total mass [kg]',fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    ax.tick_params(labelsize=16)
    ax.set_ylabel('Altitude [km]',fontsize=18)
    ax.set_xlabel('x [km]',fontsize=18)
    ax.set_ylim([0,12])
    ax.set_xlim([-60,60])
    ax.set_xticks([-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60])
    ax.set_title('Timestep: {}'.format(int(timestep)),fontsize=18)
    plt.savefig('{}evolution_plots/m_tot_x_{}.png'.format(filePath,fname))
    plt.close()

    fig,ax = plt.subplots(figsize=(20,5),constrained_layout=True)
    p1 = plt.scatter(x=dss.x, y=dss.altitude, c=dss.w_vel, s=0.01, cmap='Spectral_r',vmin=-20,vmax=20)#,norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    cbar = fig.colorbar(p1,ax=ax,pad=0.001,aspect=20)
    cbar.set_label('vertical velocity [m/s]',fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    ax.tick_params(labelsize=16)
    ax.set_ylabel('Altitude [km]',fontsize=18)
    ax.set_xlabel('x [km]',fontsize=18)
    ax.set_ylim([0,12])
    ax.set_xlim([-60,60])
    ax.set_xticks([-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60])
    ax.set_title('Timestep: {}'.format(int(timestep)),fontsize=18)
    plt.savefig('{}evolution_plots/w_vel_x_{}.png'.format(filePath,fname))
    plt.close()

    print('finished plotting')

    # fig,ax = plt.subplots(figsize=(20,5),constrained_layout=True)
    # p1 = plt.scatter(x=dss.x, y=dss.altitude, c=dss.w_vel, s=0.01, cmap='turbo',vmin=-20,vmax=20)#,norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
    # cbar = fig.colorbar(p1,ax=ax,pad=0.001,aspect=20)
    # cbar.set_label('vertical velocity [m/s]',fontsize=18)
    # cbar.ax.tick_params(labelsize=16)
    # ax.tick_params(labelsize=16)
    # ax.set_ylabel('Altitude [km]',fontsize=18)
    # ax.set_xlabel('x [km]',fontsize=18)
    # ax.set_ylim([0,12])
    # ax.set_xlim([-60,60])
    # ax.set_xticks([-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60])
    # ax.set_title('Timestep: {}'.format(int(timestep)),fontsize=18)
    # plt.savefig('evolution_plots/w_vel_x_{}_turbo.png'.format(fname))
    # plt.close()

        # ds.to_netcdf('ICON_output_for_McRadar_{}.nc'.format(fname))
else:
    print('plots for {} already existing, skipping...'.format(fname))
# dss.plot.scatter(x='x', y='altitude', hue='prolate_ratio', s=10, cmap='viridis')
# plt.tight_layout()
# plt.savefig('prolate_ratio_x_particles00002760.000.png')
# plt.close()
quit()

# try to get plot of particles within one beam of radar. Lets assume radar is at x=-40000

radarPosX = -40000; radarPosY = -100
beamWidth = 0.6 # in degree
beamWidthRad = np.deg2rad(beamWidth)
maxRange = 11000 # in m
range = np.arange(0,maxRange,36)

# zenith view, cut beam out of data
maxWidth = maxRange*np.tan(beamWidthRad/2)
widthRange = range*np.tan(beamWidthRad/2)

cutRegion1X = radarPosX + widthRange
cutRegion2X = radarPosX - widthRange
cutRegion1Y = radarPosY + widthRange
cutRegion2Y = radarPosY - widthRange

# fig,ax = plt.subplots(figsize=(15,6))
# dss.plot.scatter(ax=ax,x='x', y='altitude', hue='m_tot', s=10, cmap='Spectral_r',norm=matplotlib.colors.LogNorm(1e-12,1e-4))
# plt.plot(cutRegion1X,range)
# plt.plot(cutRegion2X,range)
# plt.tight_layout()
# plt.savefig('m_tot_x_radarBeam.png')
# plt.show()

# %%
print(dss.altitude)
cutRegion1X = xr.DataArray(cutRegion1X, coords={'altitude':range}, dims='altitude')
cutRegion2X = xr.DataArray(cutRegion2X, coords={'altitude':range}, dims='altitude')
cutRegion1Y = xr.DataArray(cutRegion1Y, coords={'altitude':range}, dims='altitude')
cutRegion2Y = xr.DataArray(cutRegion2Y, coords={'altitude':range}, dims='altitude')
print(dss)
radarBeamX = dss.where( ( (dss.x < cutRegion1X.interp(coords={'altitude':dss.altitude}, method='nearest')) &
                            (dss.x > cutRegion2X.interp(coords={'altitude':dss.altitude}, method='nearest')) ), drop=True)
print(radarBeamX)
radarBeamY = radarBeamX.where( ( (radarBeamX.y < cutRegion1Y.interp(coords={'altitude':radarBeamX.altitude}, method='nearest')) &
                            (radarBeamX.y > cutRegion2Y.interp(coords={'altitude':radarBeamX.altitude}, method='nearest')) ), drop=True)
print(radarBeamY)
# %%

fig,ax = plt.subplots(figsize=(15,6))
dss.plot.scatter(ax=ax,x='x', y='altitude', hue='m_tot', s=10, cmap='Greys',norm=matplotlib.colors.LogNorm(1e-12,1e-4),alpha=0.1,add_colorbar=False)
radarBeamY.plot.scatter(ax=ax,x='x', y='altitude', hue='m_tot', s=10, cmap='Spectral_r',norm=matplotlib.colors.LogNorm(1e-12,1e-4))
#plt.plot(cutRegion1X,range)
#plt.plot(cutRegion2X,range)
plt.tight_layout()
#ax.set_xlim([-41000,-39000])
plt.savefig('m_tot_x_radarBeam_cut.png')
plt.show()

# %%
# calculate McRadar now: 
freq = np.asarray([9.6e9,35.6e9,94.0e9]) # in Hz
wavelength = (constants.c / freq) * 1e3, #[mm]
elv = [30,90]
selMode = 'KNeighborsRegressor'
n_neighbors = 10
scatMode = 'wobbling'
lutPath = '/project/meteo/work/L.Terzi/McRadarTest/LUT/' #'/work/lvonterz/SSRGA/snowScatt/ssrga_LUT/' #'/data/optimice/McRadarLUTs/'
# define the velocity vector:
velVec = np.loadtxt('/project/meteo/work/L.Terzi/McSnow_depogrowth_paper/dopplerVelocities_Wband_CEL.txt')
radarBeamY['vel'] = -1. * radarBeamY['vel']

dicSettings = mcr.loadSettings(dataPath='',velVec=velVec, #atmoFile=inputPath+'atmo.dat',
								elv=elv, freq=freq,maxHeight=10000,minHeight=0,
								heightRes=36,convolute=True,beta=0,beta_std=10,onlyIce=False,
								scatSet={'mode':scatMode,'selmode':selMode,'n_neighbors':n_neighbors,'K2':0.93,'lutPath':lutPath,'orientational_avg':True})
# %%
import mcradar as mcr
output = mcr.fullRadar(dicSettings, radarBeamY)

# mcTable = creatRadarCols(radarBeamY, dicSettings)
# print(mcTable)
# # %%
# # now lets get the scattering 
# mcTableCry = mcTable.where(mcTable['sNmono']==1,drop=True) # select only cry, only calculate that once!
# mcTableAgg = mcTable.where(mcTable['sNmono']>1,drop=True) # select only aggregates
# DDA_data_agg = xr.open_dataset(dicSettings['scatSet']['lutPath']+'stochastic_aggregates.nc')
# DDA_data_agg['logmass'] = np.log10(DDA_data_agg.mass)
# DDA_data_agg['logDmax'] = np.log10(DDA_data_agg.Dmax)
# DDA_data_cry = xr.open_dataset(dicSettings['scatSet']['lutPath']+'all_crystals_allazi_withradar.nc')
# DDA_data_cry['logmass'] = np.log10(DDA_data_cry.mass)
# DDA_data_cry['logDmax'] = np.log10(DDA_data_cry.Dmax)
# DDA_data_cry['logar'] = np.log10(DDA_data_cry.aspect_ratio)

# # %%
# # define habit codes to be consistent with the codes of the DDA_data_agg database:
# ratioPN = np.round(((mcTableAgg.sNmono - mcTableAgg['pp'])/mcTableAgg.sNmono).values,1)*10+20 # ratio of plates and needles
# ratioPN = np.where(ratioPN==20, 21, ratioPN)
# ratioPN = np.where(ratioPN==22, 23, ratioPN)
# ratioPN = np.where(ratioPN==24, 25, ratioPN)
# ratioPN = np.where(ratioPN==26, 27, ratioPN)
# ratioPN = np.where(ratioPN==28, 29, ratioPN)
# ratioDN = np.round(((mcTableAgg.sNmono - mcTableAgg['dd'])/mcTableAgg.sNmono).values,1)*10+30 # ratio of dendrites and needles
# ratioDN = np.where(ratioDN==30, 31, ratioDN)
# ratioDN = np.where(ratioDN==32, 33, ratioDN)
# ratioDN = np.where(ratioDN==34, 35, ratioDN)
# ratioDN = np.where(ratioDN==36, 37, ratioDN)
# ratioDN = np.where(ratioDN==38, 39, ratioDN)
# ratioPD = np.round(((mcTableAgg.sNmono - mcTableAgg['pp'])/mcTableAgg.sNmono).values,1)*10+40 # ratio of plates and dendrites
# ratioPD = np.where(ratioPD==40, 41, ratioPD)
# ratioPD = np.where(ratioPD==42, 43, ratioPD)
# ratioPD = np.where(ratioPD==44, 45, ratioPD)
# ratioPD = np.where(ratioPD==46, 47, ratioPD)
# ratioPD = np.where(ratioPD==48, 49, ratioPD)
# #ratioPND = 

# #TODO: add mix of plate, dendrite and needle! OR: look at the 25% that were not defined! So filter by that 25%!!
# # define condition for plates and dendrites:
# condPD = (mcTableAgg['pp'] + mcTableAgg['dd']) == mcTableAgg.sNmono
# # define condition for plate and needle:
# condPN = (mcTableAgg['dd'] == 0) & (mcTableAgg['pp'] > 0) & (mcTableAgg.sNmono > mcTableAgg['pp'])
# # define condition for dendrite and needle: 
# condDN = (mcTableAgg['pp'] == 0) & (mcTableAgg['dd'] > 0) & (mcTableAgg.sNmono > mcTableAgg['dd'])
# # define condition for needle:
# condN = (mcTableAgg['pp'] + mcTableAgg['dd']) == 0
# # define condition for plates:
# condP  = mcTableAgg['pp'] == mcTableAgg.sNmono
# # define dendrites:
# condD = mcTableAgg['dd'] == mcTableAgg.sNmono
# # define PND aggregate:
# condPND = (mcTableAgg['dd'] > 0) & (mcTableAgg['pp'] > 0) & (mcTableAgg.sNmono > (mcTableAgg['pp'] + mcTableAgg['dd']))

# mcTableAgg['habit_code'] = mcTableAgg.sNmono.copy()*0
# habit_code = mcTableAgg.habit_code
# habit_code = xr.where(condDN, ratioDN, habit_code)
# habit_code = xr.where(condPN, ratioPN, habit_code)
# habit_code = xr.where(condPD, ratioPD, habit_code)
# habit_code = xr.where(condP, 40, habit_code) # todo: put that back into 2!!!, for now only because we have many plate-dendrite aggregates and barely any for forward simulation!
# habit_code = xr.where(condD, 2, habit_code)
# habit_code = xr.where(condN,1,habit_code)
# habit_code = xr.where(condPND, 45, habit_code) # for now have plate, needle dendrite aggregate be described as plate dendrite aggregate with 50,50
# mcTableAgg['habit_code'] = habit_code

# DDA_data_agg['habit'] = xr.where(DDA_data_agg.habit == 0, 40, DDA_data_agg.habit) # TODO: return that to 0!
# # %%
# if dicSettings['beta_std'] == 0:
# 		elevation_radius = 1
# else:
#     elevation_radius = dicSettings['beta']
# search_radii_agg = dict(
#                     logmass=abs(np.log10(1) - np.log10(1.05)), # 2 %
#                     logDmax=abs(np.log10(1) - np.log10(1.05)), # 5 %
#                     elevation = elevation_radius,
#                     wavelength = 0.1,
#                     habit = 7, # 10 % for habit code (which works because habit=0 for plates, so 0 tolerance, habit = 1 for dendrites, so 10% tolerance will not shift to other habit, only if habit = 20 or large, then 10% will be a int number)
#                     )
# print(DDA_data_agg)
# treeAgg, scalingAgg = gen_ckdtree(DDA_data_agg, search_radii_agg)

# search_radii_cry = dict(
#                     logmass=abs(np.log10(1) - np.log10(1.05)), # 2 %
#                     logDmax=abs(np.log10(1) - np.log10(1.05)), # 5 %
#                     logar = abs(np.log10(1) - np.log10(1.05)), # 2 %
#                     elevation = elevation_radius,
#                     wavelength = 0.1,
#                     )

# treeCry, scalingCry = gen_ckdtree(DDA_data_cry, search_radii_cry)
# # %%

# %%
