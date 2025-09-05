import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
def grid_info(gridfile='Torus_Triangles_1024x4_150m.nc'):
    print(f"Trying to load grid info from {gridfile=}")
    with xr.open_dataset(gridfile) as grid:
       dx = dy = dz = float(grid.edge_length.isel(edge=0))
       vol = dx * dz * grid.domain_height
       domain_length = grid.domain_length
       domain_length_y = (grid.cartesian_y_vertices.max().item() - grid.cartesian_y_vertices.min().item())/np.sin(np.deg2rad(60))
       print(f"{dx=} {dz=} {domain_length=} {vol=}")
    return dict(dx=dx, dz=dz, vol=vol, domain_length=domain_length, domain_length_y=domain_length_y)


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

filePath = '/scratch/f/Fabian.Jakub/icon_mcsnow_build/experiments/mcsnow2d.nosani.sphabit3.rt3/'
ginfo = grid_info(filePath+'Torus_Triangles_1024x4_150m.nc') #gridfile

pfile=filePath+'particles00001800.000.nc'
ds = xr.open_dataset(pfile)

ds['x'] = ds.longitude / (2*np.pi) * ginfo['domain_length']
ds['y'] = ds.latitude              * ginfo['domain_length_y']


# ds['m_tot'] = ds_get_var(ds, 'm_tot')
# ds['m_tot'] = ds.m_tot / ginfo['vol']  
# ds['xi'] = ds_get_var(ds, 'xi')
# ds['xi'] = ds.xi / ginfo['vol']

ds = ds.rename({'addVar0001':'m_w','addVar0002':'m_i','addVar0003':'m_r',
                'addVar0004':'v_r','addVar0005':'d','addVar0006':'A',
                'addVar0007':'xi','addVar0008':'mm','addVar0009':'statusb',
                'addVar0010':'vt','addVar0011':'gblCellId','addVar0012':'jk',
                'addVar0013':'m_f','addVar0014':'T','addVar0015':'atmoT',
                'addVar0016':'dQdt','addVar0017':'V_i','addVar0018':'phi',
                'addVar0019':'pp','addVar0020':'dd'})

idx = np.arange(ds.noParts.size); np.random.shuffle(idx); dss=ds.isel(noParts=idx[:int(1e6)])
# print(dss)
# for var in dss.data_vars:
#     print(var)

dss['m_tot'] = dss.m_f+ dss.m_w+ dss.m_i+ dss.m_r # funktioniert nicht weil m_f,.. nicht defined
dss['prolate_ratio'] = dss.pp / dss.mm # funktioniert nicht weil pp,... nicht defined

fig,ax = plt.subplots(figsize=(15,6))
dss.plot.scatter(ax=ax,x='y', y='altitude', hue='m_tot', s=10, cmap='Spectral_r',norm=matplotlib.colors.LogNorm(1e-12,1e-4))
plt.tight_layout()
plt.savefig('m_tot_y.png')
plt.close()
fig,ax = plt.subplots(figsize=(15,6))
dss.plot.scatter(ax=ax,x='y', y='altitude', hue='xi', s=10, cmap='Spectral_r', norm=matplotlib.colors.LogNorm(1e6,1e10))
plt.tight_layout()
plt.savefig('xi_y.png')
plt.close()
dss.plot.scatter(x='y', y='altitude', hue='w_vel', s=10, cmap='Spectral_r')
plt.tight_layout()
plt.savefig('w_vel_y.png')
plt.close()
dss.plot.scatter(x='y', y='altitude', hue='prolate_ratio', s=10, cmap='viridis')
plt.tight_layout()
plt.savefig('prolate_ratio_y.png')
plt.close()


# try to get plot of particles within one beam of radar. Lets assume radar is at x=-40000

radarPosX = -40000; radarPosY = 100
beamWidth = 0.6 # in degree
beamWidthRad = np.deg2rad(beamWidth)
maxRange = 11000 # in m

# zenith view, cut beam out of data
maxWidth = maxRange/np.tan(beamWidthRad/2)