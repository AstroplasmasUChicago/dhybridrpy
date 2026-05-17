# Installation

## Requirements

- Python 3.9 or higher
- pip package manager

## Install from PyPI

The recommended way to install dhybridrpy is via pip:

```bash
pip install dhybridrpy
```

## Install from Source

To install the latest development version from GitHub:

```bash
git clone https://github.com/AstroplasmasUChicago/dhybridrpy.git
cd dhybridrpy
pip install -e .
```

## Dependencies

dhybridrpy depends on the following packages, which are automatically installed:

| Package | Purpose |
|---------|--------|
| `h5py` | Reading HDF5 simulation output files |
| `numpy` | Numerical array operations |
| `matplotlib` | Plotting and visualization |
| `dask` | Lazy loading for large datasets |
| `f90nml` | Parsing Fortran namelist input files |
| `typer` | Command-line interface for the `dplot` tool |
| `joblib` | Parallel rendering in the `dplot` tool |

