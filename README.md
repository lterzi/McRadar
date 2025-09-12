# McRadar
## project description:
McRadar is a tool used for forward simulating Particle microphysical properties into radar space. It was mainly developed for forward simulating McSnow output (hence the Mc in McRadar). McRadar uses look-up tables of scattering properties calculated with the discrete dipole approximation. These LUTs are stored in the subfolder LUT. It will calculate the Doppler spectra at horizontal and vertical polarization (Ze_hh, Ze_vv and Ze_hv) as well as KDP based on the given particle microphysical properties (for ice crystals Dmax, mass, aspect ratio and fall velocity of the particles are required, for aggregates mass, Dmax and fall velocity). In a next step, the Doppler spectrum can be convoluted with noise and turbulence, and attenuation can be added. 

## Installation: 
First clone the project from github: 

```
git clone https://github.com/lterzi/McRadar.git
```
Then navigate into the McRadar folder and install with 

```
pip install .
```

## the aggregate_habit_prediction branch: 
Ice crystals get treated the same way as in the master branch, however, here we are using the scattering properties of roughly 7 000 000 aggregate particles. For each of the aggregates the scattering properties were only calculated for one radar orientation (at 1 elevation and 1 azimuth angle) to enable a large variability of particle properties. This is especially suitable for models without fixed mass size relationships of particles. The scattering properties get selected from the database with a KDTree, were mass, Dmax, habit and elevation need to be provided. Habit refers to the monomer type the aggregate consists of, valid are needles, plates, dendrites, and mixtures of needles and plates, and needles and dendrites. The habit codes used can be found in mcradar/src/fullRadarOperator.py. 

## Publications related to McRadar:
- DDA look-up tables can be found here:  https://doi.org/10.5281/zenodo.17091066
- This setup was developed for the publication "On the geometry of aggregate snowflakes" by A. Seifert, F. Jakub, C. Siewert, L. von Terzi and S. Kneifel 
  

## More information: 
Please have a look at the examples provided in the examples folder. 