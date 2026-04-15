#! /usr/bin/env python3

import numpy as np
from pytmatrix import tmatrix, radar, orientation
from sys import argv
import time

def main():
    script, radius, wavelength, m, axis_ratio, canting, cantingStd, meanAngle, ndgs, theta0, phi0 = argv

    radius = float(radius)
    wavelength = float(wavelength)
    m = complex(m)
    axis_ratio = float(axis_ratio)
    canting = int(canting)
    cantingStd = float(cantingStd)
    meanAngle = float(meanAngle)
    ndgs = int(ndgs)
    theta0 = float(theta0)
    phi0 = float(phi0)

    def find_backward(th0, ph0):
        th = 180.0 - th0
        ph = (ph0 + 180.0) % 360.0
        return th, ph

    thet_back, phi_back = find_backward(theta0, phi0)

    scatterer = tmatrix.Scatterer(radius=radius,
                            wavelength=wavelength,
                            radius_type=tmatrix.Scatterer.RADIUS_MAXIMUM,
                            m=m,
                            axis_ratio=axis_ratio,
                            thet0=theta0,
                            thet=thet_back,
                            phi0=phi0,
                            phi=phi_back,
                            ndgs=ndgs
                            )
    if canting==True: 
            scatterer.or_pdf = orientation.gaussian_pdf(std=cantingStd,
                                                            mean=meanAngle)  
    #         scatterer.orient = orientation.orient_averaged_adaptive
            scatterer.orient = orientation.orient_averaged_fixed
    # start = time.time()
    # back_hh = radar.radar_xsect(scatterer, True)
    # back_vv = radar.radar_xsect(scatterer, False)
    # Z = scatterer.get_Z()
    # #back_hh = Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1]
    # #back_vv = Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1]
    # scatterer.thet = scatterer.thet0
    # scatterer.phi = (scatterer.phi0) % 360. #KDP geometry
    # S = scatterer.get_S()
    # sMat = (S[1,1]-S[0,0]).real
    # #Z = scatterer.get_Z()
    # Z11 = Z[0,0]; Z12 = Z[0,1]; Z21 = Z[1,0]; Z22 = Z[1,1]; Z33 = Z[2,2]; Z44 = Z[3,3]
    # S11i = S[0,0].imag
    # S22i = S[1,1].imag
    # end = time.time()
    # print('Time with calculating bck ', end-start, 's')

    #start = time.time()
    Z = scatterer.get_Z()
    back_hh = Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1]
    back_vv = Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1]
    scatterer.thet = scatterer.thet0
    scatterer.phi = (scatterer.phi0) % 360. #KDP geometry
    S = scatterer.get_S()
    sMat = (S[1,1]-S[0,0]).real
    #Z = scatterer.get_Z()
    #Z11 = Z[0,0]; Z12 = Z[0,1]; Z21 = Z[1,0]; Z22 = Z[1,1]; Z33 = Z[2,2]; Z44 = Z[3,3]
    S11i = S[0,0].imag
    S22i = S[1,1].imag
    #end = time.time()
    #print('Time without calculating bck ', end-start, 's')


    #print('Results ', back_hh, back_vv, sMat, Z11, Z12, Z21, Z22, Z33, Z44, S11i, S22i, ' DONE')
    print('Results ', back_hh, back_vv, sMat, S11i, S22i, ' DONE')



if __name__ == "__main__":
    main()