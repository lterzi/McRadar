import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import mcradar as mcr
from scipy import constants
from mcradar.tableOperator import creatRadarCols
from scipy.spatial import cKDTree 
from sys import argv
import os
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
def calcRho(mcTable):
    """
    Calculate the density of each super particles [g/cm^3].
    
    Parameters
    ----------
    mcTable: output from getMcSnowTable()
    
    Returns
    -------
    mcTable with an additional column for the density.
    The density is calculated separately for aspect ratio < 1
    and for aspect ratio >= 1.
    """
    
    # density calculation for different AR ranges
    mcTable['sRho_tot'] = mcTable.mTot.copy()*np.nan

    #calculaiton for AR < 1
    tmpTable = mcTable.where(mcTable['sPhi']<1,drop=True)
    tmpVol = (np.pi/6.) * (tmpTable['dia'])**3 * tmpTable['sPhi']
    Rho = tmpTable['mTot']/tmpVol
    #mcTable['sRho_tot'] = mcTable.sRho_tot.where(mcTable.sPhi < 1, tmpRho.values, mcTable.sRho_tot.values)
    
    # calculation for AR >= 1
    tmpTable1 = mcTable.where(mcTable['sPhi']>=1,drop=True)
    tmpVol = (np.pi/6.) * (tmpTable1['dia'])**3 / (tmpTable1['sPhi'])**2
    Rho1 = (tmpTable1['mTot'])/tmpVol
    #print(tmpTable1,tmpTable)
    #mcTable = xr.merge([tmpTable,tmpTable1])
    #mcTable['sRho_tot'] = mcTable.sRho_tot.where(mcTable.sPhi >= 1, tmpRho, mcTable.sRho_tot)
    mcTable['sRho_tot'].loc[tmpTable1.index] = Rho1
    mcTable['sRho_tot'].loc[tmpTable.index] = Rho
    mcTable['sRho_tot'].attrs['units'] = 'kg/m^3'
    mcTable['sRho_tot'].attrs['long_name'] = 'particle density'
    return mcTable

def select_radar_beam_data(model_data, radar_x, radar_y, radar_z, azimuth, elevation, beam_width, max_range):
    """
    Select model data within a radar beam's conical area.
    
    Parameters
    ----------
    model_data : xarray.Dataset
        Model output data with x, y, z as data variables
    radar_x, radar_y, radar_z : float
        Radar position coordinates
    azimuth : float
        Beam azimuth angle in degrees (0-360, 0=North, 90=East)
    elevation : float
        Beam elevation angle in degrees (-90 to 90, 0=horizontal)
    beam_width : float
        Half-power beam width in degrees
    max_range : float
        Maximum range to consider
    
    Returns
    -------
    beam_data : xarray.Dataset
        Data points within the radar beam
    """
    
    # Convert angles to radians
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    beam_width_rad = np.radians(beam_width / 2)  # Half beam width
    
    # Get particle positions from data variables
    x_pos = model_data['x']
    y_pos = model_data['y'] 
    z_pos = model_data['sHeight']  # Using sHeight as z coordinate based on your code
    
    # Calculate relative positions from radar
    dx = x_pos - radar_x
    dy = y_pos - radar_y
    dz = z_pos - radar_z
    
    # Calculate range (distance from radar)
    range_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Calculate beam direction vector
    beam_x = np.cos(el_rad) * np.sin(az_rad)
    beam_y = np.cos(el_rad) * np.cos(az_rad)
    beam_z = np.sin(el_rad)
    
    # Calculate unit vectors from radar to each grid point
    unit_x = dx / range_3d
    unit_y = dy / range_3d
    unit_z = dz / range_3d
    
    # Handle division by zero (radar position)
    unit_x = xr.where(range_3d == 0, 0, unit_x)
    unit_y = xr.where(range_3d == 0, 0, unit_y)
    unit_z = xr.where(range_3d == 0, 0, unit_z)
    
    # Calculate dot product (cosine of angle between beam and point direction)
    cos_angle = unit_x * beam_x + unit_y * beam_y + unit_z * beam_z
    
    # Calculate angle from beam center
    angle_from_beam = np.arccos(np.clip(cos_angle, -1, 1))
    
    # Create mask for points within beam
    within_beam = (angle_from_beam <= beam_width_rad) & (range_3d <= max_range) & (range_3d > 0)
    
    # Apply mask to select data
    beam_data = model_data.where(within_beam, drop=True)
    
    # Add beam geometry information as new data variables
    # beam_data = beam_data.assign({
    #     'range': range_3d,
    #     'beam_angle': np.degrees(angle_from_beam)
    # })
    
    return beam_data
def calc_beam_area(range_vec, beam_width_deg, elevation_deg):
    """
    Calculate beam cross-sectional area for any elevation angle.
    """
    beam_width_rad = np.radians(beam_width_deg / 2)
    elevation_rad = np.radians(elevation_deg)
    
    # For tilted beams, the effective "height" in the horizontal plane
    # is range * cos(elevation)
    if elevation_deg == 90:
        # Vertical beam - simple case
        radius = range_vec * np.tan(beam_width_rad)
    else:
        # Tilted beam - more complex
        radius = range_vec * np.tan(beam_width_rad)
        # Area might need projection correction depending on your needs
    
    return np.pi * radius**2
def plot_beam_verification(dss, beam_data, radar_x, radar_y, radar_z, azimuth, elevation, beam_width, max_range):
    """
    Create comprehensive plots to verify radar beam selection.
    """
    
    # 1. 3D scatter plot showing the conical beam
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Side view (x-z or y-z depending on azimuth)
    ax1 = fig.add_subplot(131)
    
    # Plot all data in gray
    plt.scatter(dss.x*1e-3, dss.sHeight*1e-3, c='gray', s=0.1, alpha=0.3, label='All data')
    
    # Plot selected beam data in color
    if len(beam_data.sHeight) > 0:
        plt.scatter(beam_data.x*1e-3, beam_data.sHeight*1e-3, 
                   c=beam_data.mTot, s=1, cmap='viridis', 
                   norm=matplotlib.colors.LogNorm(1e-12,1e-4), label='Beam data')
    

    plt.plot(radar_x*1e-3, radar_z*1e-3, 'ro', markersize=8, label='Radar')
    plt.xlabel('x [km]')
    plt.ylabel('Height [km]')
    plt.title('Side View (x-z)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    # Plot 2: Top view (x-y)
    ax2 = fig.add_subplot(132)
    plt.scatter(dss.x*1e-3, dss.y*1e-3, c='gray', s=0.1, alpha=0.3)
    if len(beam_data.sHeight) > 0:
        plt.scatter(beam_data.x*1e-3, beam_data.y*1e-3, 
                   c=beam_data.sHeight*1e-3, s=1, cmap='plasma')
    
    
    #plt.plot(radar_x*1e-3, radar_y*1e-3, 'ro', markersize=8, label='Radar')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.xlim([3,7])
    plt.ylim([-2,2])
    #plt.ylim([dss.y.min().item()*1e-3, dss.y.max().item()*1e-3])
    plt.title('Top View (x-y)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    #plt.axis('equal')
    
    
    plt.tight_layout()
    return fig

def plot_beam_cross_sections(dss, beam_data, radar_x, radar_y, radar_z, max_range):
    """
    Plot cross-sections at different heights to show beam footprint.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    heights = [2000, 4000, 6000, 8000]  # Heights in meters
    
    for i, height in enumerate(heights):
        ax = axes[i//2, i%2]
        
        # Select data near this height (±200m)
        height_tolerance = 200
        data_at_height = dss.where(
            (dss.sHeight >= height - height_tolerance) & 
            (dss.sHeight <= height + height_tolerance), 
            drop=True
        )
        beam_at_height = beam_data.where(
            (beam_data.sHeight >= height - height_tolerance) & 
            (beam_data.sHeight <= height + height_tolerance), 
            drop=True
        )
        
        if len(data_at_height.sHeight) > 0:
            ax.scatter(data_at_height.x*1e-3, data_at_height.y*1e-3, 
                      c='gray', s=0.5, alpha=0.3)
        
        if len(beam_at_height.sHeight) > 0:
            ax.scatter(beam_at_height.x*1e-3, beam_at_height.y*1e-3, 
                      c=beam_at_height.mTot, s=2, cmap='viridis',
                      norm=matplotlib.colors.LogNorm(1e-12,1e-4))
        
        # Draw theoretical beam circle
        beam_width_rad = np.radians(0.6/2)  # Your beam width
        radius = height * np.tan(beam_width_rad)
        circle = plt.Circle((radar_x*1e-3, radar_y*1e-3), radius*1e-3, 
                          fill=False, color='red', linewidth=2)
        ax.add_patch(circle)
        
        ax.plot(radar_x*1e-3, radar_y*1e-3, 'ro', markersize=6)
        ax.set_xlim([3,7])
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.set_title(f'Height: {height/1000:.1f} km')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig
######################################################################################################################################
#%%
# TODO: investigate particles which can not be found in the aggregate LUT
# TODO: use actual u wind from ICON simulation for McRadar simulation
# TODO: correct K2 for ice, 0.93 is for water! No, I think for Ze you need to use the one for water, since it is the water equivalent reflectivity. I think? ice: 0.27


# use 5460 output, because less turbulence, so maybe less frozen particles and less rimed particles?
fileName, radarPosX, number_of_beams,time = argv
print(radarPosX, number_of_beams,time)
radarPosX1 = float(radarPosX)
outFolder = 'McRadar_output/particles0000{}_water_core/'.format(time)
if not os.path.exists(outFolder):
    os.makedirs(outFolder)
#quit()
#time=5460
radarPosX = float(radarPosX)
number_of_beams = int(number_of_beams)
# calculate McRadar now: 
convolute=True
elv = np.array([30]) # with this setup only one elevation angle is possible
freq = np.array([9.6e9,35.6e9,94.0e9]) # in Hz
ori_avg = True
beta = 0
beta_std = 0
selMode = 'KNeighborsRegressor'
n_neighbors = 10
scatMode = 'wobbling'
attenuation = False
ice_core = False
lutPath = '/project/meteo/work/L.Terzi/McRadarTest/LUT/' #'/work/lvonterz/SSRGA/snowScatt/ssrga_LUT/' #'/data/optimice/McRadarLUTs/'
# define the velocity vector:
velVec = np.loadtxt('doppler_vel_Xband.csv')
#-- define range resolution 
heightRes = 36
outName = '9.6_35.5_94.0GHz_elv{}_output_DDA_kdtree_with_habitcode_melted_water_core_nofrozen_smallkdp_rimed_30_90_oriavgTrue_beta0_beta_std0_convoluteTrue_attenuationFalse_particles0000{}.000_radarPosX{}.nc'.format(elv[0],time,int(radarPosX1))
    
dss = xr.open_dataset('ICON_output/ICON_output_for_McRadar_particles0000{}.000.nc'.format(time))

dss = dss.rename({'altitude':'sHeight','noParts':'index','xi':'sMult','m_tot':'mTot','vel':'vt','w_vel':'vel'})
dss = dss.where(~np.isnan(dss.sHeight),drop=True)
dss = dss.where(np.isfinite(dss.mTot),drop=True)
dss = dss.where(np.isfinite(dss.dia),drop=True)
dss = dss.where(np.isfinite(dss.sNmono),drop=True)
dss = dss.where(np.isfinite(dss.sPhi),drop=True)
dss = dss.where(np.isfinite(dss.vel),drop=True)
dss = calcRho(dss)
dss['sRho_tot'] = dss.sRho_tot.where(dss.sRho_tot<918,918)

#%%

radarPosY = -100
beamWidth = 0.6 # in degree#
if elv == 90:
    maxRange = 12000
else:
    maxRange = 12000/np.sin(np.deg2rad(elv))
print(maxRange)
rangeVec = np.arange(0,maxRange,heightRes)
dssnew = xr.Dataset()
j = 0

beam_data = select_radar_beam_data(dss,radarPosX,radarPosY,0,270,elv,beamWidth,maxRange)
#to get better Doppler Spectra we average over multiple beams next to each other
# for i in range(number_of_beams):
#     radarPosX = radarPosX + i*10
#     beam_data = select_radar_beam_data(dss,radarPosX,radarPosY,0,270,elv,beamWidth,maxRange)
#     if len(beam_data.sHeight) > 0:
#         if i == 0 or j==0:
#             beam_data = beam_data.assign_coords(index=beam_data.index)
#         else:
#             beam_data = beam_data.assign_coords(index=beam_data.index + dssnew.index.max().item()+1)
#         dssnew = xr.merge([beam_data,dssnew])
#         j += 1


# fig1 = plot_beam_verification(dss, beam_data, radarPosX1, radarPosY, 0, 0, 90, beamWidth, maxRange)
# plt.savefig('beam_verification_comprehensive_90°.png', dpi=150, bbox_inches='tight')
# plt.close()

# fig2 = plot_beam_cross_sections(dss, beam_data, radarPosX1, radarPosY, 0, maxRange)
# plt.savefig('beam_cross_sections_90°.png', dpi=150, bbox_inches='tight')
# plt.close()

# quit()

# # Also add a simple statistics check
# print(f"Total particles in domain: {len(dss.sHeight)}")
# print(f"Particles selected by beam: {len(beam_data.sHeight)}")
# print(f"Selection ratio: {len(beam_data.sHeight)/len(dss.sHeight)*100:.2f}%")

# # Check range distribution
# if len(beam_data.sHeight) > 0:
#     dx = beam_data.x - radarPosX1
#     dy = beam_data.y - radarPosY
#     dz = beam_data.sHeight - 0
#     ranges = np.sqrt(dx**2 + dy**2 + dz**2)
#     print(f"Range statistics: min={ranges.min().values/1000:.1f}km, max={ranges.max().values/1000:.1f}km, mean={ranges.mean().values/1000:.1f}km")

fig,ax = plt.subplots(figsize=(20,5),constrained_layout=True)
p1 = plt.scatter(x=dss.x*1e-3, y=dss.sHeight*1e-3, c=dss.mTot, s=0.01, cmap='Greys',norm=matplotlib.colors.LogNorm(1e-12,1e-4))#,alpha=0.5)
p1 = plt.scatter(x=beam_data.x*1e-3, y=beam_data.sHeight*1e-3, c='red', s=0.01)#,alpha=0.5)
ax.annotate('', xy=(radarPosX*1e-3, 1), xytext=(radarPosX*1e-3,0.1),
            arrowprops=dict(facecolor='red', edgecolor='red', width=2, headwidth=9))#, arrowstyle='->'))#, width=3, headwidth=8))
#cbar = fig.colorbar(p1,ax=ax,pad=0.001,aspect=20)
#cbar.set_label('Total mass [kg]',fontsize=18)
#cbar.ax.tick_params(labelsize=16)
ax.tick_params(labelsize=16)
ax.set_ylabel('Altitude [km]',fontsize=18)
ax.set_xlabel('x [km]',fontsize=18)
ax.set_ylim([0,12])
ax.set_xlim([-40,35])
ax.set_xticks([-40,-30,-20,-10,0,10,20,30])
plt.savefig('test_beam_elv30_arrow_pos{}.png'.format(int(radarPosX1)))
plt.close()

quit()

if j > 0:

    #gridBaseArea = np.pi*(rangeVec*np.tan(np.deg2rad(beamWidth)/2))**2*number_of_beams# works only for vertical beam
    gridBaseArea = calc_beam_area(rangeVec, beamWidth, elv)*number_of_beams
    #gridBaseArea = (rangeVec*np.tan(beamWidthRad))**2*number_of_beams
    dicSettings = mcr.loadSettings(dataPath='',velVec=velVec, #atmoFile=inputPath+'atmo.dat',
                                    elv=elv, freq=freq,gridBaseArea=gridBaseArea,maxHeight=12000,minHeight=0,
                                    heightRes=heightRes,convolute=convolute,attenuation=attenuation,beta=beta,beta_std=beta_std,onlyIce=False,
                                    scatSet={'mode':scatMode,'selmode':selMode,'n_neighbors':n_neighbors,'K2':0.93,'lutPath':lutPath,'orientational_avg':ori_avg,'ice_core':ice_core})
    # %%
    output = mcr.fullRadar(dicSettings, dssnew)
    print('done with output, saving at',outName)
    #- calculate moments and noise from the spectra:	
    output['Ze_H'] = output['spec_H'].sum(dim='vel')
    output['Ze_V'] = output['spec_V'].sum(dim='vel')
    if 'spec_H_Agg' in output:
        output['Ze_H_Agg'] = output['spec_H_Agg'].sum(dim='vel')
        output['Ze_V_Agg'] = output['spec_V_Agg'].sum(dim='vel')
        output['ZDR_Agg'] = mcr.lin2db(output['Ze_H_Agg']/output['Ze_V_Agg'])

    if 'spec_H_Mono' in output:
        output['Ze_H_Mono'] = output['spec_H_Mono'].sum(dim='vel')
        output['Ze_V_Mono'] = output['spec_V_Mono'].sum(dim='vel')
        output['ZDR_Mono'] = mcr.lin2db(output['Ze_H_Mono']/output['Ze_V_Mono'])

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
    output.to_netcdf(outFolder+outName)#inputPath+outName)
else:
    print('no particles in any beam, no McRadar simulation done')
