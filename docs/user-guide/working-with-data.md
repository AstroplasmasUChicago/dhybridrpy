# Working with Data

This guide covers the data access patterns in dhybridrpy.

## Data Hierarchy

The data in dhybridrpy is organized hierarchically:

```
DHybridrpy
├── inputs (simulation parameters)
└── timesteps
    ├── fields (B, E, J components)
    ├── phases (distribution functions, fluid quantities)
    └── raw_files (raw particle data)
```

## Accessing Timesteps

### By Timestep Number

```python
# Access a specific timestep
ts = dpy.timestep(1)
ts = dpy.timestep(100)
```

### By Index

```python
# Access by index (supports negative indexing)
ts_first = dpy.timestep_index(0)   # First timestep
ts_last = dpy.timestep_index(-1)   # Last timestep
```

### Get All Timesteps

```python
# Returns a numpy array of available timesteps
all_ts = dpy.timesteps()
print(all_ts)  # e.g., array([1, 2, 3, 4, 5])
```

## Field Data

Fields represent electromagnetic quantities on the simulation grid.

### Available Fields

| Field | Components | Description |
|-------|------------|-------------|
| Magnetic (B) | `Bx`, `By`, `Bz`, `Bmagnitude` | Magnetic field |
| Electric (E) | `Ex`, `Ey`, `Ez`, `Emagnitude` | Electric field |
| Current (J) | `Jx`, `Jy`, `Jz`, `Jmagnitude` | Current density |

### Accessing Fields

```python
# Get field components
Bx = dpy.timestep(1).fields.Bx()
By = dpy.timestep(1).fields.By()
Bz = dpy.timestep(1).fields.Bz()

# Get field magnitude
B_mag = dpy.timestep(1).fields.Bmagnitude()
```

### Field Types

Fields can be decomposed into:

- **Total** (default): Complete field
- **External**: Externally applied field
- **Self**: Self-consistent field from particle motion

```python
# Access different field types
Bx_total = dpy.timestep(1).fields.Bx()  # Default is Total
Bx_total = dpy.timestep(1).fields.Bx(type="Total")
Bx_ext = dpy.timestep(1).fields.Bx(type="External")
Bx_self = dpy.timestep(1).fields.Bx(type="Self")
```

## Phase Data

Phase data includes distribution functions and fluid quantities.

### Accessing Phase Data

```python
# Distribution function in x-y space
f_xy = dpy.timestep(1).phases.x2x1()

# Velocity distribution
f_vxvy = dpy.timestep(1).phases.p2p1()

# Phase space
f_xvx = dpy.timestep(1).phases.p1x1()
```

### Species Selection

Phase data is species-specific:

```python
# Default: species 1
phase_s1 = dpy.timestep(1).phases.x2x1()
phase_s1 = dpy.timestep(1).phases.x2x1(species=1)

# Species 2
phase_s2 = dpy.timestep(1).phases.x2x1(species=2)

# Total (all species)
phase_total = dpy.timestep(1).phases.x2x1(species="Total")
```

### Fluid Quantities

```python
# Bulk velocity components
Vx = dpy.timestep(1).phases.Vx(species=1)
Vy = dpy.timestep(1).phases.Vy(species=1)
Vz = dpy.timestep(1).phases.Vz(species=1)

# Pressure tensor components
Pxx = dpy.timestep(1).phases.Pxx(species=1)
Pxy = dpy.timestep(1).phases.Pxy(species=1)

# Scalar pressure
P = dpy.timestep(1).phases.P(species=1)
```

## Raw Particle Data

Raw files contain particle-level data:

```python
# Access raw data for species 1
raw = dpy.timestep(1).raw_files.raw(species=1)

# Get the data dictionary
data_dict = raw.dict
print(data_dict.keys())  # Available quantities
```

## Data Properties

All data objects (Field, Phase) share common properties:

```python
Bx = dpy.timestep(1).fields.Bx()

# Core data array
Bx.data        # The actual numpy/dask array
Bx.data.shape  # Array dimensions

# Coordinate arrays
Bx.xdata       # X coordinates
Bx.ydata       # Y coordinates (2D/3D)
Bx.zdata       # Z coordinates (3D)

# Coordinate limits
Bx.xlimdata    # [xmin, xmax]
Bx.ylimdata    # [ymin, ymax]
Bx.zlimdata    # [zmin, zmax]

# Metadata
Bx.name        # e.g., "Bx"
Bx.timestep    # Timestep number
Bx.time        # Simulation time
Bx.file_path   # Path to HDF5 file
```

## Arithmetic Operations

Data objects support arithmetic operations:

```python
import numpy as np

Bx = dpy.timestep(1).fields.Bx()
By = dpy.timestep(1).fields.By()

# Basic operations
B_sum = Bx + By
B_diff = Bx - By
B_scaled = Bx * 2.0
B_ratio = Bx / By
B_squared = Bx ** 2

# NumPy ufuncs work directly
B_abs = np.abs(Bx)
B_sin = np.sin(Bx)
B_mag = np.sqrt(Bx**2 + By**2)
```

!!! note
    Arithmetic operations require compatible data: same shape, same timestep, and (for Fields) same type.

## Inspecting Available Data

```python
ts = dpy.timestep(1)

# Print available fields
print(ts.fields)
# Output:
# Fields at timestep 1:
#   type = Total: Bx, By, Bz, Ex, Ey, Ez, Jx, Jy, Jz
#   type = External: Bx, By, Bz
#   type = Self: Bx, By, Bz

# Print available phases
print(ts.phases)
# Output:
# Phases at timestep 1:
#   species = 1: x2x1, p1x1, Vx, Vy, Vz, ...
#   species = 2: x2x1, p1x1, Vx, Vy, Vz, ...

# Print available raw files
print(ts.raw_files)
```
