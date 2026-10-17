# Lazy Loading

Use `lazy=True` to work with Dask arrays. You can reduce a large field or
select a region before loading the result into memory:

```python
from dhybridrpy import DHybridrpy

dpy = DHybridrpy("path/to/input", "path/to/Output", lazy=True)
Bx = dpy.timestep_index(0).fields.Bx()
values = Bx.data  # Dask array; field values are not read yet
maximum = values.max().compute()  # Read the field and return one number
```

Metadata is read as needed. Array values are read when you call `.compute()`
or use a method that needs them. Calling `values.compute()` loads the full
NumPy array, which must fit in memory.

## Combine Calculations

Compute related reductions together so they can share file reads:

```python
import dask
import numpy as np

ts = dpy.timestep_index(0)
bx = ts.fields.Bx().data
by = ts.fields.By().data
B_xy = np.sqrt(bx**2 + by**2)
mean, maximum = dask.compute(B_xy.mean(), B_xy.max())
```

Arithmetic stays deferred until computation. Reuse the same Dask array
in related calculations; separate `.compute()` calls can reread the files.

## Read a Region

Slice before computing to read only the chunks that intersect the selection:

```python
print(values.chunks)
subset = values[..., :8].compute()  # First eight bins of the last axis
```

Large field and phase arrays are chunked along the last array axis. A slice
along another axis may touch every chunk; small arrays use one chunk, so
any slice reads the whole array.

Use `crop()` when you also need the selected region's coordinates or plotting
methods. It follows the same chunk limits:

```python
center = Bx.crop(x_range=(0.25, 0.75))
center_max = center.data.max().compute()
```

## Methods That Read Immediately

| Method | What is read |
|--------|--------------|
| `plot()` | The displayed data; 3D plots read the slice and, when shown, the cube's outer faces |
| `avg_1d()`, `fft_power()`, `fft_power_1d()` | The full input array |
| `field_timeseries()`, `phase_timeseries()` | Each selected array, even with `lazy=True` |

For time series, use `apply` to return a small result from each worker
([example](working-with-data.md#read-across-timesteps)).

## Raw Particle Data

Raw datasets also support deferred reductions:

```python
raw = dpy.timestep(dpy.raw_timesteps()[0]).raw_files.raw(species=1)
mean_vx = raw["v1"].mean().compute()
```

Each raw dataset is one chunk, so computing even a slice reads the entire
dataset. `raw.dict` returns lazy arrays for all datasets. `raw.load()`
requires `lazy=False`.
