# Installation

## Requirements

- Python 3.8 or higher
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

## Verifying Installation

To verify that dhybridrpy is installed correctly:

```python
import dhybridrpy
print(dhybridrpy.__all__)
```

This should output:

```
['DHybridrpy', 'Timestep', 'Field', 'Phase', 'Raw']
```

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade dhybridrpy
```
