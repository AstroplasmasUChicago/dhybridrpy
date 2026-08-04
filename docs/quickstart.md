# Quick Start

This guide will help you get started with dhybridrpy in just a few minutes.

## Basic Setup

First, import the main class and initialize it with your simulation paths:

```python
from dhybridrpy import DHybridrpy

# Point to your dHybridR simulation files
dpy = DHybridrpy(
    input_file="path/to/input",
    output_folder="path/to/Output"
)
```

## Exploring Your Data

### View Available Timesteps

```python
# Get an array of all available timesteps
timesteps = dpy.timesteps()
print(f"Available timesteps: {timesteps}")

# Get the simulation time of each timestep
times = dpy.times()
```

Raw particle data may be dumped at different intervals than fields and
phases. Use `dpy.raw_timesteps()` and `dpy.raw_times()` for those.

### Access Input Parameters

The simulation input parameters are parsed and available as a dictionary.
The sections and keys match the input file itself, for example `node_conf`,
`time`, and `grid_space`:

```python
# View all input sections
print(list(dpy.inputs.keys()))
# ['node_conf', 'time', 'grid_space', 'global_output', ...]

# Access the time step size
dt = dpy.inputs['time']['dt']
print(f"Time step: {dt}")

# Access the grid: cells per dimension and box size
ncells = dpy.inputs['grid_space']['ncells']
boxsize = dpy.inputs['grid_space']['boxsize']
print(f"Grid cells: {ncells}")   # e.g., [128, 64]
print(f"Box size: {boxsize}")    # e.g., [64.0, 32.0]
```

## Working with Timestep Data

### Access a Specific Timestep

```python
# Get data at timestep 1
ts = dpy.timestep(1)

# Or access by index (e.g., last timestep)
ts_last = dpy.timestep_index(-1)
```

### Explore Available Data

```python
# See what's available at this timestep
print(ts.fields)   # Available field data
print(ts.phases)   # Available phase data
print(ts.raw_files)  # Available raw files
```

## Accessing Fields

Fields include magnetic field (B), electric field (E), current density (J), and, when external forcing is enabled in the simulation, the external acceleration field (Accel):

```python
# Get the x-component of the magnetic field
Bx = dpy.timestep(1).fields.Bx()

# Access the actual data array
print(Bx.data)
print(Bx.data.shape)

# Get coordinate data
print(Bx.xdata)  # x coordinates
print(Bx.ydata)  # y coordinates (for 2D/3D)
```

### Field Types

Fields can be accessed by type: `Total` (default), `External`, or `Self`:

```python
# Get external magnetic field
Bx_ext = dpy.timestep(1).fields.Bx(type="External")

# Get self-consistent magnetic field
Bx_self = dpy.timestep(1).fields.Bx(type="Self")

# The external acceleration field is only written as an External field,
# so it must be requested with type="External"
Accelx = dpy.timestep(1).fields.Accelx(type="External")
```

## Accessing Phases

Phase data includes distribution functions and fluid quantities:

```python
# Get phase data for species 1 (default)
phase = dpy.timestep(1).phases.x2x1()

# Get phase data for a specific species
phase_s2 = dpy.timestep(1).phases.x2x1(species=2)
```

## Plotting Data

All data objects have a built-in `plot()` method:

```python
import matplotlib.pyplot as plt

# Simple plot
Bx = dpy.timestep(1).fields.Bx()
Bx.plot()
plt.show()
```

### Customizing Plots

```python
# Customize the plot
Bx.plot(
    colormap="plasma",
    title="Custom Title",
    xlabel="X Position",
    ylabel="Y Position",
    show_colorbar=True,
    colorbar_label="Bx"
)
plt.show()
```

## Data Operations

You can perform arithmetic operations directly on data objects:

```python
# Get field components
Bx = dpy.timestep(1).fields.Bx()
By = dpy.timestep(1).fields.By()
Bz = dpy.timestep(1).fields.Bz()

# Calculate magnetic field magnitude
import numpy as np
B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

# Plot the result
B_mag.plot(title="Magnetic Field Magnitude")
plt.show()
```

## Reading Across Timesteps

To study time evolution, avoid looping over timesteps one at a time.
Use `field_timeseries` (or `phase_timeseries`), which reads the files
in parallel worker processes:

```python
# All timesteps of Bx as one array of shape (num_timesteps, *grid)
Bx_all = dpy.field_timeseries("Bx")

# Reduce each timestep inside the workers instead of loading everything
import numpy as np
Bx_means = dpy.field_timeseries("Bx", apply=np.mean)
```

See [Working with Data](user-guide/working-with-data.md) for details,
including the guard needed when calling these from a script.

## Next Steps

- Learn more about [Working with Data](user-guide/working-with-data.md)
- Explore [Plotting Options](user-guide/plotting.md)
- Enable [Lazy Loading](user-guide/lazy-loading.md) for large datasets
- Check the full [API Reference](api/dhybridrpy.md)
