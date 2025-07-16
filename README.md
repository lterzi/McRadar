# McRadar
## project description:
McRadar is a tool used for forward simulating Particle microphysical properties into radar space. It was mainly developed for forward simulating McSnow output (hence the Mc in McRadar). McRadar uses look-up tables of scattering properties calculated with the discrete dipole approximation. These LUTs are stored in the subfolder LUT. It will calculate the Doppler spectra at horizontal and vertical polarization (Ze_hh, Ze_vv and Ze_hv) as well as KDP based on the given particle microphysical properties (for ice crystals Dmax, mass, aspect ratio and fall velocity of the particles are required, for aggregates mass, Dmax and fall velocity). In a next step, the Doppler spectrum can be convoluted with noise and turbulence, and attenuation can be added. 

## Installation: 
First clone the project from github: 

```
{
git clone https://github.com/lterzi/McRadar.git
}
```
Then navigate into the McRadar folder and install with 

```
{
pip install .
}
```


## McRadar modes
McRadar has two modes: forward simulating McSnow simulations and forward simulating a given particle size distribution. These two modes are shown in examples/forward_simulate_McSnow_casestudy.py and examples/forward_simulate_PSD.py

## selection of scattering properties:

To find the scattering properties of a particle which best fits with its microphysical properties (i.e. mass, size and aspect ratio) to the by the model predicted particle, McRadar performs a nearest neighbour selection or regression on the Level 2 LUTs, depending on user preference. This is implemented using the scikit-learn Python library, which supports efficient neighbour search algorithms. By default, McRadar uses neighbours-based regression, where the output is a weighted average of the n nearest neighbours based on  inverse distance to the query210 point. This supervised learning approach follows the methodology described by Roweis et al. (2004), and more details can be found in the official documentation: https://scikit-learn.org/1.5/modules/neighbors.html.

## description of which variables can be changed: 
- mode: if PSD==False, then McSnow simulation is expected, if PSD===True any PSD is expected (see examples)
- elevation and frequencies you want to have calculated: elevation can be varied between 0° (horizontal) and 90° (zenith), frequency between 5.6GHz, 9.6GHz, 35.6GHz, and 94GHz. 
- radar specific variables: 
    - nfft: number of ffts performed to obtain Doppler spectrum
    - nave: number of averages performed on Doppler spectrum
    - noise_pow: radar noise power
    - theta: beam width of radar
    - time_int: integration time of radar
    - maxVel, minVel: Nyquvist range
    - tau: pulse width
    - heightRes: range resolution of radar
- Atmospheric parameters: 
    - uwind: x component of wind velocity
    - eps_diss: eddy dissipation rate
    - k_theta, k_phi, k_r: wind shear in different directions
    - shear_height0, shear_height1: height of wind shear zone
- Simulation details:
    - gridBaseArea: area of grid base of your simulation
    - maxHeight, minHeight: maximum/minimum height of your simulation
- scatSet: 
    - mode: define the kind of DDA LUT you want to use:
        - azimuthal_random_orientation: this assumes that the particles orientation is random in azimuth. With this mode you can give a beta angel and beta standard deviation angle, so a wobbling of the particle around the angle beta is assumed.
        - fixed_orientation: this assumes that the particles orientation is fixed in space, no beta angle can be assumed. The particle properties where azimuthally averaged at a fixed orientation.
    - selmode: defines the selection mode from the LUTs:
        - KNeighborsRegressor: this ode uses the KNeighborsRegressor from sklearn. The n_neighbors closest neighbors in Dmax, aspect ratio and mass are selected and the corresponding scattering properties are averaged based on the inverse distance of the neighbors.
        - NearestNeighbors: the n_neighbors closest neighbors in Dmax, aspect ratio and mass are selected and the scattering properties of these points are averaged.
        - radius: this mode uses the radius_neighbors from sklearn. All neighbors (Dmax, aspect ratio, mass) within the predefined radius are selected and the scattering properties are averaged.
    - n_neighbors: number of neighbours to use for selmode 'KNeighborsRegressor' and 'NearestNeighbors'.
    - radius: radius in which the nearest neighbours are selected when scatSet['selmode'] is set to radius.
    
- define if you want to calculate attenuation
- define if you want noise, turbulence convolution

## More information: 
Please have a look at the examples provided in the examples folder. Further, this project will be published, and more information can be found in that publication. 