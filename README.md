# McRadar

McRadar is a flexible and modular radar forward operator designed for the simulation of radar observables from microphysical model output, particularly from McSnow. It leverages advanced scattering models and lookup tables (LUTs) to compute radar reflectivity (Ze), specific differential phase (Kdp), ZDR, Doppler spectra and other radar variables for a wide range of hydrometeor types and particle properties.

## Key Features
- **Support for Multiple Hydrometeor Types:** Handles ice, snow, graupel, hail, rain, and mixed-phase particles, including melted and partially melted particles.
- **Advanced Scattering Models:** Integrates T-matrix, DDA, and Mie scattering calculations, with support for external libraries such as pytmatrix and scattnlay.
- **Lookup Table (LUT) Integration:** Efficiently interpolates precomputed scattering properties for rapid simulation.
- **Flexible Input:** Reads microphysical tables from McSnow and other models in NetCDF/xarray format.
- **Customizable Settings:** User-configurable radar parameters (frequency, wavelength, elevation, polarization, etc.) and microphysical assumptions.
- **Batch Processing:** Supports simulation over multiple heights, time steps, and ensemble members.
- **Extensible Design:** Modular codebase allows easy extension for new hydrometeor types, scattering models, or radar variables.

## Typical Workflow
1. **Prepare Microphysical Input:** Generate or obtain microphysical output (e.g., from McSnow) in NetCDF/xarray format.
2. **Configure Radar Simulation:** Set radar and microphysical parameters in a settings file or via Python API.
3. **Run McRadar:** Use the provided scripts or Python interface to compute radar observables.
4. **Analyze Output:** Output is provided as xarray Datasets, ready for further analysis or visualization.

## Installation
Install McRadar and its dependencies using pip (after cloning the repository):

```bash
pip install .
```

## Dependencies
- numpy
- scipy
- xarray
- pandas
- netCDF4
- scikit-learn
- pytmatrix
- scattnlay

## Example Usage
```python
from mcradar import RadarSimulation, RadarSettings
settings = RadarSettings(frequency=35.5, wl=8.5, elv=90)
sim = RadarSimulation(settings)
sim.run(input_file="mcTable.nc", output_file="radar_output.nc")
```

## Citation
If you use McRadar in your research, please cite:
- [Your publication or DOI here]

## License
McRadar is licensed under the MIT License.

## Contact
For questions, bug reports, or contributions, contact Leonie Terzi (l.terzi@lmu.de).
