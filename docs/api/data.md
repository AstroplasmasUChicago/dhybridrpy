# Data classes

`Field` and `Phase` share the array, analysis, and plotting API below.
Examples use a [Timestep](timestep.md) named `ts` and a `Field` named `obj`:

```python
import numpy as np

obj = ts.fields.Bx()
```

## Field and Phase

| Attribute or property | Value |
|---|---|
| `file_path`, `name` | Source HDF5 path and quantity name |
| `timestep`, `time` | Iteration number and simulation time |
| `lazy` | Whether array reads use Dask |
| `type` | `Field` only: `"Total"`, `"External"`, or `"Self"` |
| `species` | `Phase` only: species number or `"Total"` |
| `data` | Array in `(nx, ny, nz)` order; unused dimensions omitted |
| `xdata`, `ydata`, `zdata` | Axis cell centers |
| `xlimdata`, `ylimdata`, `zlimdata` | Axis boundaries `[low, high]` |

| Behavior | Details |
|---|---|
| Array type | NumPy by default; Dask for data and coordinates when `lazy=True`. Limits are always NumPy arrays |
| Coordinates | Follow the diagnostic axes; `ts.phases.u1x1().ydata` contains proper-velocity bin centers |
| Repeated reads | File-backed `.data` is not cached; arithmetic and crops retain their result array or Dask graph |

### Arithmetic and comparisons

| Operation | Example |
|---|---|
| Arithmetic | `obj + 1`, `obj - 1`, `2 * obj`, `obj / 2`, `obj ** 2`, `-obj` |
| Two data objects | `obj + ts.fields.By()` |
| NumPy elementwise functions | `np.sqrt(obj ** 2)` |
| Comparisons | `obj > 0`, `obj == ts.fields.By()` |
| Boolean reduction | `(obj > 0).data.any()` |

**Returns:** the same data class with preserved metadata; comparisons contain booleans.

**Requirements:** two objects must match in class, shape, iteration, and field type or species. Coordinates are not checked.

Use `.data` for multi-output NumPy functions. Data objects cannot be used as `if` conditions.

### crop

```text
crop(x_range=None, y_range=None, z_range=None) -> Field | Phase
```

| Item | Meaning |
|---|---|
| Each range | Fractional bounds `(low, high)` with `0 <= low < high <= 1`; `None` keeps the whole axis |
| Cell selection | Bounds rounded to indices; at least one cell retained |
| Returns | Same data class, with updated coordinates and limits |
| Reads | Eager: full source array. Lazy: intersecting chunks when computed |

```python
center = obj.crop(x_range=(0.2, 0.8))  # For 10 x cells, keep indices 2–7
```

## Analysis

All methods accept 1D, 2D, or 3D data, load the full selection, and return
NumPy arrays. `direction` selects an existing axis: `"x"`, `"y"`, or `"z"`.

### avg_1d

```python
coords, mean, lower, upper = obj.avg_1d(direction="x")
```

| Return | Meaning |
|---|---|
| `coords` | Cell centers along `direction` |
| `mean` | Arithmetic mean over all other axes |
| `lower`, `upper` | `mean - std`, `mean + std`, using population standard deviation |

For shape `(4, 3)`, `direction="x"` returns four means over three cells each.
For 1D data, `mean`, `lower`, and `upper` equal the original data.

### Spectra

Both methods return angular wavenumbers in inverse coordinate units and float64 power. Neither subtracts
the mean or applies a window. Axis lengths come from the object's boundary limits.

#### fft_power

```python
k, power = obj.fft_power()
```

| Return | Meaning |
|---|---|
| `k` | Radial bin centers, spaced by `2*pi / max(axis_lengths)` |
| `power` | FFT power summed within each radial bin |

For per-mode averages, use [`fft_power_iso(normalize=True)`](#fft_power_iso).

#### fft_power_1d

```python
k, mean, lower, upper = obj.fft_power_1d(direction="x")
```

One FFT along `direction` at each perpendicular position; statistics across those spectra:

| Return | Meaning |
|---|---|
| `k` | Wavenumbers starting at zero, spaced by `2*pi / axis_length` |
| `mean` | Geometric mean power: `exp(mean(log(power)))` |
| `lower`, `upper` | `exp(mean(log(power)) ± std(log(power)))` |

Zero-power entries are excluded from the statistics; bins with no positive power return zeros.

## Plotting

### plot

```text
plot(*, ax=None, dpi=100, title=None,
     xlabel=None, ylabel=None, zlabel=None,
     xlim=None, ylim=None, zlim=None,
     colormap="viridis", show_colorbar=True, colorbar_label=None,
     slice_axis="x", context_3d=True, **kwargs)
```

| Data | Plot | Returns |
|---|---|---|
| 1D | Line | `(Axes, Line2D)` |
| 2D | Image | `(Axes, AxesImage)` |
| 3D | Interactive slice | `(Axes, AxesImage)` for the slice panel |

| Option | Meaning |
|---|---|
| `ax`, `dpi` | Existing axes, or resolution of a new figure |
| `title` | Override the quantity/time title |
| `xlabel`, `ylabel`, `zlabel` | Labels for the corresponding coordinates |
| `xlim`, `ylim`, `zlim` | Limits for the corresponding coordinates |
| `colormap` | Image colormap |
| `show_colorbar`, `colorbar_label` | Show and label the image colorbar |
| `slice_axis` | `"x"`: yz plane; `"y"`: xz plane; `"z"`: xy plane |
| `context_3d` | Rotatable volume view; ignored when `ax` is supplied |
| `**kwargs` | Matplotlib `plot` (1D) or `imshow` (2D/3D) options |

- **1D labels and limits:** set the y-axis with `ax.set_ylabel(...)` and `ax.set_ylim(...)`.
- **3D controls:** slider and left/right arrows; requires an [interactive backend](../user-guide/plotting.md).
- **3D color limits:** rescale per slice unless fixed with `vmin`, `vmax`, or `norm`.

```python
ax, artist = obj.plot(colormap="RdBu_r", vmin=-1, vmax=1)  # 2D or 3D
```

### Profile and spectrum plots

**Returns:** `(Axes, Line2D)` for each method. All options except `direction` are keyword-only.

| Method | Plots | Options specific to this method |
|---|---|---|
| `plot_1d_avg(direction="x", ...)` | [`avg_1d`](#avg_1d) mean and standard-deviation band | Band options below |
| `plot_fft_power(...)` | [`fft_power`](#fft_power) radial sums | `loglog=True` |
| `plot_fft_power_1d(direction="x", ...)` | [`fft_power_1d`](#fft_power_1d) geometric mean and band | `loglog=True`; band options below |

| Keyword option | Default | Meaning |
|---|---|---|
| `ax`, `dpi` | `None`, `100` | Existing axes, or resolution of a new figure |
| `title`, `xlabel`, `ylabel` | `None` | Override automatic text |
| `xlim`, `ylim` | `None` | Axis limits |
| `loglog` | `True` | Spectrum methods only; `False` selects linear axes |
| `**kwargs` | — | Matplotlib line options, such as `label` or `linestyle` |

**Band options** (`plot_1d_avg` and `plot_fft_power_1d` only):

| Keyword option | Default | Meaning |
|---|---|---|
| `show_std`, `fill_alpha` | `True`, `0.3` | Show the band; set its opacity |
| `fill_color`, `line_color` | `None` | Band and mean-line colors; band defaults to line color |

```python
ax, line = obj.plot_1d_avg(direction="x", show_std=False)
ax, line = obj.plot_fft_power(loglog=True)
```

For velocity or energy profiles, override the default spatial axis label with `xlabel`.

## Raw

```python
raw = ts.raw_files.raw(species=1)
velocity = raw["v1"]
```

**Attributes:** `file_path`, `name` (`"raw"`), `timestep`, `time`, `lazy`, `species`.

| Access | Returns |
|---|---|
| `raw.keys()` | Dataset names; no array reads |
| `key in raw` | Whether a dataset exists |
| `raw[key]` | One NumPy array, or Dask array when lazy |
| `raw.dict` | Dictionary of all datasets; NumPy or Dask arrays |
| `raw.load(keys=None, workers=None)` | Dictionary of NumPy arrays read by worker processes |

| Detail | Value |
|---|---|
| Dataset names | Match the file; `x1`: position, `v1`: ordinary velocity in `v_A`, `q`: signed macroparticle charge weight |
| Reads | Not cached; missing keys raise `KeyError`; each lazy dataset is one chunk |
| `load` defaults | All datasets; `min(8, CPU count)` workers |
| `load` restrictions | `lazy=False`; scripts need the [time-series main guard](dhybridrpy.md#read-across-timesteps) |

## Standalone functions

Import from `dhybridrpy`. Spectrum functions take NumPy arrays.

### fft_power_iso

```text
fft_power_iso(data, Lx, Ly=None, Lz=None, normalize=False) -> (k, power)
```

| Parameter | Meaning |
|---|---|
| `Lx`, `Ly`, `Lz` | Axis lengths; `Ly` required for 2D/3D, `Lz` for 3D |
| `normalize` | `False`: radial sums; `True`: divide each bin by its mode count |

Returns match [`fft_power`](#fft_power).

### fft_power_1d_slices

```text
fft_power_1d_slices(data, L, direction="x") -> (k, mean, lower, upper)
```

**`L`:** length of the chosen axis. Returns match [`fft_power_1d`](#fft_power_1d).

### close_pooled_handles

```text
close_pooled_handles() -> None
```

Close HDF5 handles used for interactive 3D slicing before recreating those files
in the same process. Subsequent slices reopen handles.
