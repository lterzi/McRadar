import numpy as np

def S2Z(S):
    """ Transformation of the amplitude matrix S into the scattering matrix Z
    as done within the notorious Mishchenko T-Matrix code
    https://www.giss.nasa.gov/staff/mmishchenko/t_matrix.html

    Parameters
    ----------
    S22, S11, S21, S12 : complex
        The 4 complex values of the amplitude matrix as defined in Mishchenko
        books and codes

    Returns
    -------
    Z : numpy 4x4 float
        The 4x4 real valued scattering matrix as defined in Mishchenko
        books and codes
    """
    S22, S11, S21, S12 = S

    S11c = S11.conjugate()
    S12c = S12.conjugate()
    S21c = S21.conjugate()
    S22c = S22.conjugate()

    Z11 = 0.5*( S11*S11c + S12*S12c + S21*S21c + S22*S22c).real
    Z12 = 0.5*( S11*S11c - S12*S12c + S21*S21c - S22*S22c).real
    Z13 = -(S11*S12c + S22*S21c).real
    Z14 = -(S11*S12c - S22*S21c).imag

    Z21 = 0.5*( S11*S11c + S12*S12c - S21*S21c - S22*S22c).real
    Z22 = 0.5*( S11*S11c - S12*S12c - S21*S21c + S22*S22c).real
    Z23 = -(S11*S12c - S22*S21c).real
    Z24 = -(S11*S12c + S22*S21c).imag

    Z31 = -(S11*S21c + S22*S12c).real
    Z32 = -(S11*S21c - S22*S12c).real
    Z33 = (S11*S22c + S12*S21c).real
    Z34 = (S11*S22c + S21*S12c).imag

    Z41 = -(S21*S11c + S22*S12c).imag
    Z42 = -(S21*S11c - S22*S12c).imag
    Z43 = (S22*S11c - S12*S21c).imag
    Z44 = (S22*S11c - S12*S21c).real

    Z = np.array([[Z11, Z12, Z13, Z14],
                  [Z21, Z22, Z23, Z24],
                  [Z31, Z32, Z33, Z34],
                  [Z41, Z42, Z43, Z44]])
    return Z

def S2Z_ds(ds):
    '''
    ds: xarray dataset with the amplitude matrix S
    '''
    S = (ds.adda_Sr + 1j*ds.adda_Si).transpose(..., 'amplitudemat')
    M = S.expand_dims(dict(muellermat_j=4)).rename(dict(amplitudemat='muellermat_i')).transpose(..., 'muellermat_i', 'muellermat_j').copy()
    M[:] = np.apply_along_axis(S2Z, -1, S.data)
    return M
