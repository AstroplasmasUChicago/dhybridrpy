# Working with data

Fields and species diagnostics provide arrays with coordinates for plotting
and analysis. For individual particles, use [raw snapshots](#read-particle-snapshots)
or [particle tracks](particle-tracks.md).

## Select an output

Open a run and inspect the quantities saved at its last field or phase timestep:

```python
from dhybridrpy import DHybridrpy

dpy = DHybridrpy("path/to/input", "path/to/Output")
ts = dpy.timestep_index(-1)
print(ts)
```

The listing shows names, field types, and species. Available quantities can
differ between timesteps; choose a name listed for the output you select.

| Call | Selects |
| --- | --- |
| `dpy.timestep(32)` | Iteration 32 |
| `dpy.timestep_index(0)` | First saved field or phase timestep |
| `dpy.timestep_index(-1)` | Last saved field or phase timestep |
| `dpy.timestep_closest(100)` | Saved field or phase timestep nearest iteration 100 |

`dpy.timesteps()` lists field and phase iterations; `dpy.times()` gives their
simulation times in the same order. These lists omit iteration zero by default.
Include it with `DHybridrpy(..., exclude_timestep_zero=False)`.

## Read a physical quantity

`ts.fields` holds electromagnetic fields and electron-fluid quantities.
`ts.phases` holds species charge density, bulk velocity, pressure, and
phase-space distributions. Species methods default to `species=1`;
`species="Total"` reads a saved total diagnostic rather than summing species.

### Fields

```python
Bx = ts.fields.Bx()
Bx_external = ts.fields.Bx(type="External")
```

The default field type is `"Total"`. For magnetic and electric fields,
`"External"` selects the imposed field and `"Self"` selects total minus external.

| Quantity | Names |
| --- | --- |
| Magnetic field | `Bx`, `By`, `Bz`, `Bmagnitude` |
| Electric field | `Ex`, `Ey`, `Ez`, `Emagnitude` |
| Total electric current density | `Jx`, `Jy`, `Jz`, `Jmagnitude` |
| Electron pressure and temperature | `Pe`, `Te` |
| Turbulence-driving acceleration | `Accelx`, `Accely`, `Accelz`, `Accelmagnitude` |

Acceleration requires `type="External"`. Magnitude methods read saved files;
calculate an unsaved magnitude with [arithmetic](#calculate-derived-quantities).

### Charge density

```python
density = ts.phases.x3x2x1(species=1)
```

`x3x2x1` is the species' signed charge density on the simulation grid.
The name is the same in 1D, 2D, and 3D runs.

`x2x1` gives an xy charge-density map, averaged over z in a 3D run.
It uses absolute charge weights. `x3x1` and `x3x2` give the xz and yz maps.

### Bulk velocity and pressure

```python
velocity = ts.phases.Vx(species=1)
pressure = ts.phases.Pmagnitude(species=1)
```

| Quantity | Names |
| --- | --- |
| Bulk velocity components and magnitude | `Vx`, `Vy`, `Vz`, `Vmagnitude` |
| Pressure tensor components | `Pxx`, `Pyy`, `Pzz`, `Pxy`, `Pyz`, `Pzx` |
| Scalar pressure, `(Pxx + Pyy + Pzz) / 3` | `Pmagnitude` |

### Velocity and energy distributions

These diagnostics show how particles are distributed in position and proper
velocity (`u = γv`) or kinetic energy:

| Name | Horizontal axis | Vertical axis |
| --- | --- | --- |
| `u1x1` | x position | x proper velocity |
| `u2u1` | x proper velocity | y proper velocity |
| `utx1` | x position | Proper speed |
| `etx1` | x position | Natural logarithm of kinetic energy |

```python
f = ts.phases.u1x1(species=1)
```

The vertical axis comes first in the name. These distributions use absolute
particle charge as the weight and average over spatial coordinates not shown.

Older output, including the sample data, uses names such as `p1x1` and `p2p1`.
Use the names printed by `ts.phases`; dhybridrpy preserves the saved names.

## Work with arrays

The fields and species diagnostics above share the same array, coordinate,
cropping, and arithmetic interface.

### Read values and coordinates

```python
values = Bx.data
x = Bx.xdata
```

By default, `.data` reads a NumPy array. Save it in a variable to avoid reading
the file again. With `lazy=True`, it returns a [Dask array](lazy-loading.md).

Spatial arrays have shape `(nx, ny)` in 2D and `(nx, ny, nz)` in 3D.
`xdata`, `ydata`, and `zdata` contain sample centers; `xlimdata`, `ylimdata`,
and `zlimdata` give axis edges. For distributions, coordinates follow the
displayed axes: `f.ydata` above contains proper-velocity bin centers.

### Crop a region

Use array indexing for individual values or slices. Use `crop()` to keep the
coordinates and plotting methods while selecting fractions of each axis:

```python
center = Bx.crop(x_range=(0.25, 0.75))  # Middle half of the x extent
```

### Calculate derived quantities

Arithmetic also preserves coordinates and plotting methods:

```python
import numpy as np

By = ts.fields.By()
B_xy = np.sqrt(Bx**2 + By**2)
```

Scalar arithmetic (`2 * Bx`) and comparisons (`Bx > 0`) work too. Combine
objects from the same grid, timestep, and field type or species. See the
[arithmetic reference](../api/data.md#arithmetic-and-comparisons) for
compatibility checks and boolean results.

## Read across timesteps

Use `field_timeseries()` or `phase_timeseries()` to read a quantity across
saved iterations. Pass `apply=np.mean` to return one mean per timestep:

```python
import numpy as np
from dhybridrpy import DHybridrpy

if __name__ == "__main__":
    dpy = DHybridrpy("path/to/input", "path/to/Output")
    mean_Bx = dpy.field_timeseries("Bx", apply=np.mean)
```

The result is a list in timestep order. Pass `timesteps=[32, 64]` to select
iterations, each of which must contain the quantity. Without `apply`, the
method returns full arrays stacked along a new time axis. Pass a list of
quantity names with `apply` to combine several arrays at each timestep.

These reads use worker processes, so scripts need the main guard shown
above. `apply` must be importable by those workers. See the
[time-series reference](../api/dhybridrpy.md#read-across-timesteps) for custom
functions and memory requirements, or [plot and export a field history](../examples/time-history.md).

## Read particle snapshots

Raw output stores individual particle data at a saved time. Its output interval
can differ from fields and phases, so select an iteration with `raw_timesteps()`:

```python
raw = dpy.timestep(dpy.raw_timesteps()[0]).raw_files.raw(species=1)
print(raw.keys())  # Dataset names; no array data is read
x = raw["x1"]
vx = raw["v1"]
```

`x1`, `x2`, and `x3` are positions; `v1`, `v2`, and `v3` are ordinary
velocity components. `q` is the signed macroparticle charge weight.
Use `"tag" in raw` to check whether a dataset exists.

For bulk reads, `raw.dict` reads all datasets and `raw.load()` reads them
in worker processes. See the [Raw reference](../api/data.md#raw) for selecting
datasets and using workers, or [lazy loading](lazy-loading.md#raw-particle-data)
to defer reads. For trajectories across time, use [particle tracks](particle-tracks.md).
