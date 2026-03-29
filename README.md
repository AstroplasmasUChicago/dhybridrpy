# dhybridrpy

![PyPI version](https://img.shields.io/pypi/v/dhybridrpy?label=PyPI&color=blue) [![Documentation Status](https://readthedocs.org/projects/dhybridrpy/badge/?version=latest)](https://dhybridrpy.readthedocs.io/en/latest/?badge=latest)

`dhybridrpy` allows you to easily load and plot data from `dHybridR` simulations. It provides programmatic access to simulation input and output data and the ability to quickly visualize that data.

## Features

- Efficiently access simulation input data and output data like timesteps, fields (e.g., magnetic field), phases (e.g., distribution functions), and particle tracks.
- Quickly plot 1D, 2D, and 3D output data.
- Compute FFT power spectra and 1D spatial averages.
- Follow individual particle trajectories across a simulation.
- Lazily load large datasets using `dask`.
- Perform arithmetic operations on data objects directly.

## Installation

The latest package version can be installed via pip:

```bash
pip install dhybridrpy
```

## Usage

Basic usage of the package:

```python
from dhybridrpy import DHybridrpy

# Enter your input file and output folder paths here
input_file = "examples/data/inputs/input"
output_folder = "examples/data/Output"

dpy = DHybridrpy(input_file=input_file, output_folder=output_folder)

# Print simulation timesteps
print(dpy.timesteps())

# Access an input variable
print(f"Timestep = {dpy.inputs['time']['dt']}")

# Access data at a specific timestep
ts = 1
Bx = dpy.timestep(ts).fields.Bx()
print(Bx.data)

# Plot data
import matplotlib.pyplot as plt
Bx.plot()
plt.show()
```

Further examples can be found in the `examples` folder and in the [online documentation](https://dhybridrpy.readthedocs.io/en/latest/examples/).

## Documentation

Full documentation is available at [dhybridrpy.readthedocs.io](https://dhybridrpy.readthedocs.io/en/latest/).

## License

Project licensed under the GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Authors

- Bricker Ostler
- Miha Cernetic

## License

Project licensed under the GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Authors

- Bricker Ostler
- Miha Cernetic