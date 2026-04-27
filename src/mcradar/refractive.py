from os import path
import numpy as np
from scipy import interpolate

ice_density = 0.9167
def ice_refractive(file):
    """
    Interpolator for the refractive indices of ice.

    Inputs:
       File to read the refractive index lookup table from.
       This is supplied as "ice_refr.dat", retrieved from
       http://www.atmos.washington.edu/ice_optical_constants/

    Returns:
       A callable object that takes as parameters the wavelength [mm]
       and the snow density [g/cm^3].
    """
    D = np.loadtxt(file)

    log_wl = np.log10(D[:,0]/1000)
    re = D[:,1]
    log_im = np.log10(D[:,2])

    iobj_re = interpolate.interp1d(log_wl, re)
    iobj_log_im = interpolate.interp1d(log_wl, log_im)

    def ref(wl, snow_density):
        lwl = np.log10(wl)
        try:
            len(lwl)
        except TypeError:
            mi_sqr = complex(iobj_re(lwl), 10**iobj_log_im(lwl))**2
        else:
            mi_sqr = np.array([complex(a,b) for (a,b) in zip(iobj_re(lwl), 
                10**iobj_log_im(lwl))])**2

        c = (mi_sqr-1)/(mi_sqr+2) * snow_density/ice_density
        return np.sqrt( (1+2*c) / (1-c) )

    return ref

module_path = path.split(path.abspath(__file__))[0]

mi = ice_refractive(module_path + '/ice_refr.dat')