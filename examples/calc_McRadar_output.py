import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import mcradar as mcr
from mcradar.settings import RadarSettings
from mcradar.fullRadarOperator import RadarSimulation
from scipy import constants
from scipy.spatial import cKDTree 
from sys import argv
import os
#%%
def grid_info(gridfile='Torus_Triangles_1024x4_150m.nc'):
    """
    Load grid information from a NetCDF file and compute grid metrics.

    Parameters
    ----------
    gridfile : str, optional
        Path to the grid NetCDF file (default: 'Torus_Triangles_1024x4_150m.nc').

    Returns
    -------
    dict
        Dictionary with dx, dz, vol, domain_length, and domain_length_y.
    """
    print(f"Trying to load grid info from {gridfile=}")
    with xr.open_dataset(gridfile) as grid:
       dx = dy = dz = float(grid.edge_length.isel(edge=0))
       vol = dx * dz * grid.domain_height
       domain_length = grid.domain_length
       domain_length_y = (grid.cartesian_y_vertices.max().item() - grid.cartesian_y_vertices.min().item())/np.sin(np.deg2rad(60))
       print(f"{dx=} {dz=} {domain_length=} {vol=}")
    return dict(dx=dx, dz=dz, vol=vol, domain_length=domain_length, domain_length_y=domain_length_y)
def gen_ckdtree(aggdb, search_radii):
    """
    Construct a cKDTree for fast nearest-neighbor search in aggregate database.

    Parameters
    ----------
    aggdb : dict or structured array
        Aggregate database with dimensions as keys.
    search_radii : dict
        Dictionary of search radii for each dimension.

    Returns
    -------
    tree : cKDTree
        Constructed KDTree object.
    scaling : np.ndarray
        Scaling factors for each dimension.
    """
    import time
    start = time.time()
    scaling = np.array([1.0 / search_radii[dim] for dim in search_radii.keys()]) # scale euclidean space for search
    points = np.stack([aggdb[dim] for dim in search_radii.keys()], axis=-1) # sample points out of aggdb
    scaled_points = points * scaling
    tree = cKDTree(scaled_points)
    end = time.time()
    print(f"construction of ckdtree took {end - start}s")
    return tree, scaling
def calcRho(mcTable):
    """
    Calculate the density of each superparticle [kg/m^3].

    Parameters
    ----------
    mcTable : xarray.Dataset
        Output from getMcSnowTable().

    Returns
    -------
    mcTable : xarray.Dataset
        Table with an additional column for the density ('sRho_tot').
        The density is calculated separately for aspect ratio < 1 and >= 1.
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

def calculate_radial_velocity(model_data,azimuth, elevation):
    """
    Calculate the radial velocity component along the radar beam direction.

    Parameters
    ----------
    model_data : xarray.Dataset
        Model output data with u_vel, v_vel, w_vel, vt.
    azimuth : float
        Beam azimuth angle in degrees (0-360, 0=North, 90=East).
    elevation : float
        Beam elevation angle in degrees (-90 to 90, 0=horizontal).

    Returns
    -------
    radial_velocity : xarray.DataArray
        Velocity component along radar beam direction.
    """
    
    # Convert angles to radians
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    
    # Calculate radar beam direction unit vector
    # (pointing from radar towards target)
    beam_x = np.cos(el_rad) * np.sin(az_rad)
    beam_y = np.cos(el_rad) * np.cos(az_rad)
    beam_z = np.sin(el_rad)
    
    # Get velocity components
    u = model_data['u_vel']  # east-west
    v = model_data['v_vel']  # north-south
    w = model_data['w_vel']  # vertical
    vt = model_data['vt']     # terminal fall velocity
    
    # Total vertical velocity is w + vt
    w_total = w + vt
    
    # Project velocity vector onto beam direction (dot product)
    # Positive radial velocity = away from radar
    radial_vel = u * beam_x + v * beam_y + w_total * beam_z
    
    return radial_vel


def select_radar_beam_data(model_data, radar_x, radar_y, radar_z, azimuth, elevation, beam_width, max_range):
    """
    Select model data within a radar beam's conical area.

    Parameters
    ----------
    model_data : xarray.Dataset
        Model output data with x, y, sHeight as data variables.
    radar_x, radar_y, radar_z : float
        Radar position coordinates.
    azimuth : float
        Beam azimuth angle in degrees (0-360, 0=North, 90=East).
    elevation : float
        Beam elevation angle in degrees (-90 to 90, 0=horizontal).
    beam_width : float
        Half-power beam width in degrees.
    max_range : float
        Maximum range to consider.

    Returns
    -------
    beam_data : xarray.Dataset
        Data points within the radar beam.
    """
    
    # Convert angles to radians
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    beam_width_rad = np.radians(beam_width / 2)  # Half beam width
    
    # Get particle positions from data variables
    x_pos = model_data['x']
    y_pos = model_data['y'] 
    z_pos = model_data['sHeight']  
    
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

    Parameters
    ----------
    range_vec : array-like
        Range(s) from radar [m].
    beam_width_deg : float
        Half-power beam width [degrees].
    elevation_deg : float
        Elevation angle [degrees].

    Returns
    -------
    area : array-like
        Cross-sectional area(s) [m^2].
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
def calc_beam_volume(range_vec, range_res, beam_width_deg, elevation_deg=None):
    """
    Calculate the volume of each range bin for a radar beam.

    Parameters
    ----------
    range_vec : array-like
        Range bin centers [m].
    range_res : float
        Range resolution (bin width) [m].
    beam_width_deg : float
        Half-power beam width [degrees].
    elevation_deg : float, optional
        Elevation angle [degrees]. Not needed for volume calculation.

    Returns
    -------
    volumes : array-like
        Volume of each range bin [m³].
    """
    
    # Convert beam width to radians (half angle)
    theta = np.radians(beam_width_deg / 2)
    
    # Range bin edges
    r1 = range_vec - range_res / 2  # inner edge
    r2 = range_vec + range_res / 2  # outer edge
    
    # Volume of a cone frustum: V = (π/3) * h * (R1² + R1*R2 + R2²)
    # where h is the height (range_res) and R1, R2 are the radii at each end
    
    # Radii at inner and outer edges
    R1 = r1 * np.tan(theta)
    R2 = r2 * np.tan(theta)
    
    # Volume of frustum
    volumes = (np.pi / 3) * range_res * (R1**2 + R1*R2 + R2**2)
    
    return volumes


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
######################################################################################################################################

# TODO: use actual u wind from ICON simulation for McRadar simulation
# TODO: get frozen mass back in, but maybe make threshold with frozen mass and sphericity, size of particle? Righ now KDP is again ridiculously high at cloud top, so my particles probably don't work..
# TODO: what if we increase the wobbling in areas where wind is high?

fileName, radarPosX, number_of_beams,time,path, elv = argv
print(radarPosX, number_of_beams,time,path)
radarPosX1 = float(radarPosX)
outFolder = '/project/meteo/work/L.Terzi/ICON_McSnow_Axel/{}/McRadar/particles0000{}/'.format(path,time)
if not os.path.exists(outFolder):
    os.makedirs(outFolder)
radarPosX = float(radarPosX)
number_of_beams = int(number_of_beams)

# Simulation parameters: 
convolute=True
elv = np.array([int(elv)]) # with this setup only one elevation angle is possible
freq = np.array([9.6e9]) # in Hz np.array([5.6e9,9.6e9,35.5e9])#
ori_avg = True
beta = 0
beta_std = 30
selMode = 'KNeighborsRegressor'
n_neighbors = 10
scatMode = 'wobbling'
attenuation = False
ice_core = False
lutPath = '/project/meteo/work/L.Terzi/McRadarTest/LUT/' #'/work/lvonterz/SSRGA/snowScatt/ssrga_LUT/' #'/data/optimice/McRadarLUTs/'
# define the velocity vector:
velVec = np.loadtxt('doppler_vel_Xband.csv')
#-- define output Name: 
outName = '{:.1f}GHz_elv{}_output_DDA_kdtree_melted_water_core_oriavgTru_gridVolume_beta{}_beta_std{}_particles0000{}.000_radarPosX{}_newsRange.nc'.format(freq[0]*1e-9,elv[0],beta,beta_std,time,int(radarPosX1))

# now lets open the dataset and convert it to the format needed for McRadar.
ginfo = grid_info('Torus_Triangles_1024x4_150m.nc') #gridfile
file = '/project/meteo/work/L.Terzi/ICON_McSnow_Axel/{}/particles0000{}.000.nc'.format(path, time)
dss = xr.open_dataset(file)
dss['x'] = dss.longitude / (2*np.pi) * ginfo['domain_length']
dss['y'] = dss.latitude              * ginfo['domain_length_y']

dss['mTot'] = dss.m_f+ dss.m_w+ dss.m_i+ dss.m_r # mTot is not yet calculated, but we need that for McRadar
dss = dss.rename({'altitude':'sHeight','noParts':'index','xi':'sMult','d':'dia','mm':'sNmono','phi':'sPhi'})


# now we need to select the particles that are within the radar beam, and calculate the radial velocity for those particles, which is needed for the Doppler spectra calculation in McRadar. 
# We will do this for multiple beams next to each other, and then average the spectra over those
radarPosY = -100
beamWidth = 0.6 # in degree#
if elv == 90:
    maxRange = 12000
    heightRes = 36
else:
    maxRange = 12000/np.sin(np.deg2rad(elv))
    heightRes = 36/np.sin(np.deg2rad(elv))[0]
    #dss['sHeight'] = dss.sHeight/np.sin(np.deg2rad(elv))[0]

#- now calculate the distance to the radar for each point. We will need that for the radar simulation.
radarPosZ = 0
sRange = np.sqrt((dss['x'] - radarPosX)**2 + (dss['y'] - radarPosY)**2 + (dss['sHeight'] - radarPosZ)**2)
# If sRange is a DataArray, extract .data for assignment
if hasattr(sRange, 'data'):
    dss['sRange'] = (('index',), sRange.data)
else:
    dss['sRange'] = (('index',), sRange)
print(maxRange,heightRes)
rangeVec = np.arange(0,maxRange,heightRes)
dssnew = xr.Dataset()
j = 0

#to get better Doppler Spectra we average over multiple beams next to each other
for i in range(number_of_beams):
    print(i)
    radarPosX = radarPosX + i*10
    beam_data = select_radar_beam_data(dss,radarPosX,radarPosY,0,270,elv,beamWidth,maxRange)
    if len(beam_data.sHeight) > 0:
        #if elv == 90:
        #    dss['vel'] = dss.vt+dss.w_vel # vel is combination of vertical wind and fall velocity
        #else:
        beam_data['vel'] = calculate_radial_velocity(beam_data, 270, elv)

        if i == 0 or j==0:
            beam_data = beam_data.assign_coords(index=beam_data.index)
        else:
            beam_data = beam_data.assign_coords(index=beam_data.index + dssnew.index.max().item()+1)
        dssnew = xr.merge([beam_data,dssnew])
        j += 1

# only if particles are found continue with the McRadar simulation, otherwise we can skip it and save some time.
if j > 0:
    dssnew = dssnew.where(~np.isnan(dssnew.sHeight),drop=True)
    dssnew = dssnew.where(np.isfinite(dssnew.mTot),drop=True)
    dssnew = dssnew.where(np.isfinite(dssnew.dia),drop=True)
    dssnew = dssnew.where(np.isfinite(dssnew.sNmono),drop=True)
    dssnew = dssnew.where(np.isfinite(dssnew.sPhi),drop=True)
    dssnew = dssnew.where(np.isfinite(dssnew.vel),drop=True)
    dssnew = calcRho(dssnew)
    dssnew['sRho_tot'] = dssnew.sRho_tot.where(dssnew.sRho_tot<918,918)
    #- calculate the volume in the radar beams
    gridVolume = calc_beam_volume(rangeVec, heightRes, beamWidth, elv)*number_of_beams
    #- now define all settings of the simulation
    settings_obj = RadarSettings(dataPath='', velVec=velVec, elv=elv, freq=freq, gridVolume=gridVolume, maxHeight=maxRange, minHeight=0,
                                 heightRes=heightRes, convolute=convolute, attenuation=attenuation, beta=beta, beta_std=beta_std, onlyIce=False,
                                 scatSet={'mode':scatMode, 'selmode':selMode, 'n_neighbors':n_neighbors, 'K2':0.93, 'lutPath':lutPath, 'orientational_avg':ori_avg, 'ice_core':ice_core})
    #print(settings_obj)
    dicSettings = settings_obj.settings
    radar_sim = RadarSimulation(dicSettings)

    #- now run the simulation:
    print("now running McRadar simulation")
    radar_sim.initialize(dssnew)
    radar_sim.run()
    print(radar_sim.results)
    output = radar_sim.results['spectra']
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
