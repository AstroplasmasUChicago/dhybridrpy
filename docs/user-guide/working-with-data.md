# Working with Data

This guide covers the data access patterns in dhybridrpy.

## Data Hierarchy

The data in dhybridrpy is organized hierarchically:

```
DHybridrpy
├── inputs (simulation parameters)
└── timesteps
    ├── fields (B, E, J components)
    ├── phases (phase diagrams, fluid quantities)
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

### By Closest Match

```python
# Access the closest available timestep to a target
ts = dpy.timestep_closest(100, verbose=True)
# INFO: Requested timestep: 100. Closest available timestep: 96.
```

### Get All Timesteps

```python
# Returns a numpy array of available timesteps
all_ts = dpy.timesteps()
print(all_ts)  # e.g., array([1, 2, 3, 4, 5])
```

### Get Simulation Times

`times()` returns the simulation time of each field/phase timestep, in
the same order as `timesteps()`:

```python
all_times = dpy.times()
```

### Raw Data Timesteps

Raw particle data may be dumped at different intervals than fields and
phases, so it has its own timestep and time arrays:

```python
raw_ts = dpy.raw_timesteps()
raw_times = dpy.raw_times()
```

## Field Data

Fields represent electromagnetic quantities on the simulation grid.

### Available Fields

| Field | Components | Description |
|-------|------------|-------------|
| Magnetic (B) | `Bx`, `By`, `Bz`, `Bmagnitude` | Magnetic field |
| Electric (E) | `Ex`, `Ey`, `Ez`, `Emagnitude` | Electric field |
| Current (J) | `Jx`, `Jy`, `Jz`, `Jmagnitude` | Current density |
| External acceleration (Accel) | `Accelx`, `Accely`, `Accelz`, `Accelmagnitude` | External force/driving acceleration field (only available with `type="External"`) |

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

!!! note
    The external acceleration field (`Accel`) is only ever written as an
    **External** field, so it must be requested with `type="External"`:

    ```python
    Accelx = dpy.timestep(1).fields.Accelx(type="External")
    ```

    Requesting it without `type="External"` (the default is `Total`) raises an
    error, since no `Total` or `Self` acceleration field exists.

## Phase Data

Phase data includes phase diagrams and fluid quantities. Phase diagrams
are charge density deposited on a 2D grid: each particle contributes its
charge, so purely spatial diagrams such as `x2x1` and `x3x2x1` are the
charge density of the species. The axis codes in a diagram name are:

| Code | Axis |
|------|------|
| `x1`, `x2`, `x3` | Position (x, y, z) |
| `p1`, `p2`, `p3` | Proper velocity component (gamma times velocity), unlike the `p` datasets in raw files |
| `pt` | Magnitude of the proper velocity |
| `et` | Kinetic energy on a natural log axis, so `etx1` is charge per log energy along x |

### Accessing Phase Data

```python
# Charge density in x-y space
f_xy = dpy.timestep(1).phases.x2x1()

# Charge density in proper velocity space
f_p2p1 = dpy.timestep(1).phases.p2p1()

# Charge per log energy along x
f_etx = dpy.timestep(1).phases.etx1()
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

## Reading Across Timesteps

A common task is following one quantity through the whole run. Looping
over timesteps works, but it reads the files one at a time:

```python
# Works, but slow: one file at a time
max_Bx = []
for ts in dpy.timesteps():
    max_Bx.append(dpy.timestep(ts).fields.Bx().data.max())
```

Prefer `field_timeseries` and `phase_timeseries`, which read the files
in parallel worker processes:

```python
import numpy as np

# Reduce each timestep inside the workers; only the results come back
max_Bx = dpy.field_timeseries("Bx", apply=np.max)

# Phase counterpart, per species
mean_Vx = dpy.phase_timeseries("Vx", species=1, apply=np.mean)
```

Without `apply`, the full selection is returned as one array of shape
`(num_timesteps, *grid)` and held in memory, so pass a `timesteps`
subset for very large runs:

```python
Bx_recent = dpy.field_timeseries("Bx", timesteps=dpy.timesteps()[-20:])
```

The `apply` function runs inside the workers, so only its results
travel back. This scales to runs whose full data would not fit in
memory. It must be a module-level function (such as `np.mean`), not a
lambda. `name` may also be a list of names, in which case `apply`
receives one array per name:

```python
def mean_bperp(bx, by):
    return np.sqrt(bx**2 + by**2).mean()

bperp = dpy.field_timeseries(["Bx", "By"], apply=mean_bperp)
```

Other arguments: `type` selects the field type (`field_timeseries`),
`species` selects the species (`phase_timeseries`), and `workers` sets
the pool size.

!!! note
    The workers are spawned processes. In a script, calls to these
    methods must run under an `if __name__ == "__main__":` guard.
    Notebooks and interactive sessions need no guard.

## Raw Particle Data

Raw files contain particle-level data:

```python
# Access raw data for species 1
raw = dpy.timestep(1).raw_files.raw(species=1)

# List available datasets without reading any data
print(raw.keys())  # e.g., ['ene', 'p1', 'p2', 'p3', 'x1', 'x2']

# Check for a dataset
if 'ene' in raw:
    print("energies available")
```

### Reading Selected Datasets

Index the raw object to read a single dataset, or use `load()` to read
several at once with the worker pool:

```python
# Read one dataset
x = raw['x1']

# Read several datasets in parallel worker processes
data = raw.load(['x1', 'x2', 'ene'])
print(data['ene'])

# Read everything in parallel
data = raw.load()
```

### Reading Everything

`raw.dict` reads the whole file into a dictionary in one pass:

```python
data_dict = raw.dict
print(data_dict.keys())  # Available quantities
```

Prefer `keys()`, indexing, or `load()` when you only need some of the
datasets, since `dict` reads them all.

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
