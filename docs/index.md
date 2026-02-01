# dhybridrpy

![PyPI version](https://img.shields.io/pypi/v/dhybridrpy?label=PyPI&color=blue)

**dhybridrpy** is a Python package that allows you to easily load and plot data from dHybridR simulations. It provides programmatic access to simulation input and output data with the ability to quickly visualize that data.

## Features

- **Easy Data Access**: Efficiently access simulation input data and output data like timesteps, fields (e.g., magnetic field), and phases (e.g., distribution functions).
- **Quick Visualization**: Plot 1D, 2D, and 3D output data with a single method call.
- **Lazy Loading**: Handle large datasets efficiently using [Dask](https://dask.org/) for lazy loading.
- **Flexible API**: Perform arithmetic operations on data objects directly.

## Quick Example

```python
from dhybridrpy import DHybridrpy

# Initialize with your simulation paths
dpy = DHybridrpy(
    input_file="path/to/input",
    output_folder="path/to/Output"
)

# Get available timesteps
print(dpy.timesteps())

# Access magnetic field data at timestep 1
Bx = dpy.timestep(1).fields.Bx()
print(Bx.data)

# Plot the data
import matplotlib.pyplot as plt
Bx.plot()
plt.show()
```

## Installation

Install via pip:

```bash
pip install dhybridrpy
```

See the [Installation Guide](installation.md) for more details.

## Getting Help

- Browse the [Quick Start Guide](quickstart.md) for an introduction
- Check the [API Reference](api/dhybridrpy.md) for detailed documentation
- View [Examples](examples.md) for common use cases
- Report issues on [GitHub](https://github.com/AstroplasmasUChicago/dhybridrpy/issues)

## License

This project is licensed under the GNU Affero General Public License v3.0. See the [LICENSE](https://github.com/AstroplasmasUChicago/dhybridrpy/blob/main/LICENSE) file for details.

## Authors

- Bricker Ostler
- Miha Cernetic
