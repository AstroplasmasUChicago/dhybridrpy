# dhybridrpy

![PyPI version](https://img.shields.io/pypi/v/dhybridrpy?label=PyPI&color=blue) [![Documentation Status](https://readthedocs.org/projects/dhybridrpy/badge/?version=latest)](https://dhybridrpy.readthedocs.io/en/latest/?badge=latest)

`dhybridrpy` is an efficient Python package that allows you to easily load and plot data from `dHybridR` simulations. It provides programmatic access to simulation input and output data and the ability to quickly visualize that data.

## Features

- Efficiently access simulation input data and output data like timesteps, fields (e.g., magnetic field), phases (e.g., distribution functions), raw particle data, and particle tracks.
- Quickly plot 1D, 2D, and 3D output data, including an interactive slice viewer for 3D volumes.
- Read a quantity across many timesteps in one call, with files read by parallel worker processes.
- Compute FFT power spectra and 1D spatial averages.
- Follow individual particle trajectories across a simulation.
- Lazily load large datasets using `dask`.
- Perform arithmetic operations on data objects directly.
- Batch-render fields and phases from the command line with the `dplot` tool.

## Installation

The latest package version can be installed via pip:

```bash
pip install dhybridrpy
```

### Dependencies

The following packages are installed automatically with dhybridrpy:

| Package | Purpose |
|---------|--------|
| `h5py` | Reading HDF5 simulation output files |
| `numpy` | Numerical array operations |
| `matplotlib` | Plotting and visualization |
| `dask` | Lazy loading for large datasets |
| `f90nml` | Parsing Fortran namelist input files |
| `typer` | Command-line interface for the `dplot` tool |
| `joblib` | Parallel rendering in the `dplot` tool |
| `ruff` | Code linting (development tool) |
| `pre-commit` | Git hook management (development tool) |

Optional: `scipy`, when installed, is used to run FFTs across multiple threads. `ffmpeg` is needed on your system for `dplot` video output.

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

## CLI Tool: dplot

`dplot` is a command-line tool for visualizing dHybridR simulation fields and phases across all timesteps, saving PNG images and optionally creating MP4 videos. It is included when you install dhybridrpy.

### Discover available data

```bash
dplot -i path/to/input
```

### Plot specific fields

```bash
dplot -i path/to/input --fields Bx --fields By
```

### Plot phase-space distributions

```bash
dplot -i path/to/input --phases p1x1 --species 3
```

### Plot all fields with video output

```bash
dplot -i path/to/input --all-fields --video
```

Key options: `--video`, `--fps`, `--colormap`, `--dpi`, `--vmin`/`--vmax`, `--type`, `--plots-dir`, `-j` (parallel processes), `-v` (verbose). Large datasets are automatically downsampled to 1080p. See the [dplot documentation](https://dhybridrpy.readthedocs.io/en/latest/user-guide/dplot/) for full details.

## Documentation

Full documentation is available at [dhybridrpy.readthedocs.io](https://dhybridrpy.readthedocs.io/en/latest/).

## License

Project licensed under the GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Authors

- Bricker Ostler
- Miha Cernetic
