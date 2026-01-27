# Data Classes

This page documents the data classes: `Field`, `Phase`, and `Raw`.

## Field

Represents electromagnetic field data (B, E, J) on the simulation grid.

### Class Definition

```python
class Field(
    file_path: str,
    name: str,
    timestep: int,
    time: float,
    time_ndecimals: int,
    lazy: bool,
    field_type: str
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to the HDF5 file |
| `name` | `str` | Field name (e.g., "Bx", "Ey") |
| `timestep` | `int` | Timestep number |
| `time` | `float` | Simulation time |
| `lazy` | `bool` | Whether lazy loading is enabled |
| `type` | `str` | Field type: "Total", "External", or "Self" |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `data` | `ndarray` or `dask.Array` | The field data array |
| `xdata` | `ndarray` or `dask.Array` | X-coordinate values |
| `ydata` | `ndarray` or `dask.Array` | Y-coordinate values (2D/3D) |
| `zdata` | `ndarray` or `dask.Array` | Z-coordinate values (3D) |
| `xlimdata` | `ndarray` | X-axis limits [xmin, xmax] |
| `ylimdata` | `ndarray` | Y-axis limits [ymin, ymax] |
| `zlimdata` | `ndarray` | Z-axis limits [zmin, zmax] |

### Example

```python
Bx = dpy.timestep(1).fields.Bx()

print(Bx.name)       # "Bx"
print(Bx.type)       # "Total"
print(Bx.timestep)   # 1
print(Bx.time)       # e.g., 10.0
print(Bx.data.shape) # e.g., (256, 128)
```

---

## Phase

Represents phase space data (distribution functions, fluid quantities).

### Class Definition

```python
class Phase(
    file_path: str,
    name: str,
    timestep: int,
    time: float,
    time_ndecimals: int,
    lazy: bool,
    species: Union[int, str]
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to the HDF5 file |
| `name` | `str` | Phase name (e.g., "x2x1", "Vx") |
| `timestep` | `int` | Timestep number |
| `time` | `float` | Simulation time |
| `lazy` | `bool` | Whether lazy loading is enabled |
| `species` | `int` or `str` | Species identifier (1, 2, ... or "Total") |

### Properties

Same as Field: `data`, `xdata`, `ydata`, `zdata`, `xlimdata`, `ylimdata`, `zlimdata`

### Example

```python
phase = dpy.timestep(1).phases.x2x1(species=1)

print(phase.name)      # "x2x1"
print(phase.species)   # 1
print(phase.data.shape) # e.g., (128, 128)
```

---

## Raw

Represents raw particle data.

### Class Definition

```python
class Raw(
    file_path: str,
    name: str,
    timestep: int,
    time: float,
    lazy: bool,
    species: int
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to the HDF5 file |
| `name` | `str` | Always "raw" |
| `timestep` | `int` | Timestep number |
| `time` | `float` | Simulation time |
| `lazy` | `bool` | Whether lazy loading is enabled |
| `species` | `int` | Species number |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `dict` | `dict` | Dictionary of all raw data arrays |

### Example

```python
raw = dpy.timestep(1).raw_files.raw(species=1)

data_dict = raw.dict
print(data_dict.keys())  # e.g., ['x1', 'x2', 'x3', 'p1', 'p2', 'p3']

# Access specific quantities
x_positions = data_dict['x1']
momenta = data_dict['p1']
```

---

## Arithmetic Operations

Both `Field` and `Phase` support arithmetic operations:

### Binary Operations

```python
Bx = dpy.timestep(1).fields.Bx()
By = dpy.timestep(1).fields.By()

# Addition
B_sum = Bx + By

# Subtraction
B_diff = Bx - By

# Multiplication
B_scaled = Bx * 2.0
B_product = Bx * By

# Division
B_ratio = Bx / By

# Power
Bx_squared = Bx ** 2
```

### NumPy Ufuncs

```python
import numpy as np

Bx = dpy.timestep(1).fields.Bx()

# NumPy functions work directly
B_abs = np.abs(Bx)
B_sin = np.sin(Bx)
B_exp = np.exp(Bx)
B_log = np.log(np.abs(Bx) + 1e-10)

# Complex calculations
Bx = dpy.timestep(1).fields.Bx()
By = dpy.timestep(1).fields.By()
Bz = dpy.timestep(1).fields.Bz()

B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)
```

### Compatibility Requirements

Operations between data objects require:

1. **Same shape**: Grid dimensions must match
2. **Same timestep**: Data must be from the same timestep
3. **Same type** (for Fields): Field types must match
4. **Same species** (for Phases): Species must match

```python
# This works
Bx = dpy.timestep(1).fields.Bx(type="Total")
By = dpy.timestep(1).fields.By(type="Total")
result = Bx + By

# This raises ValueError (different types)
Bx_total = dpy.timestep(1).fields.Bx(type="Total")
Bx_ext = dpy.timestep(1).fields.Bx(type="External")
# result = Bx_total + Bx_ext  # Error!
```

---

## Plotting

Both `Field` and `Phase` have a `plot()` method:

### Method Signature

```python
def plot(
    self,
    *,
    ax: Optional[Axes] = None,
    dpi: int = 100,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    zlim: Optional[tuple] = None,
    colormap: str = "viridis",
    show_colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    slice_axis: Literal["x", "y", "z"] = "x",
    **kwargs
) -> Tuple[Axes, Union[Line2D, QuadMesh]]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ax` | `Axes` | `None` | Matplotlib Axes (creates new if None) |
| `dpi` | `int` | `100` | Figure resolution |
| `title` | `str` | `None` | Plot title (auto-generated if None) |
| `xlabel` | `str` | `None` | X-axis label |
| `ylabel` | `str` | `None` | Y-axis label |
| `zlabel` | `str` | `None` | Z-axis label (3D only) |
| `xlim` | `tuple` | `None` | X-axis limits |
| `ylim` | `tuple` | `None` | Y-axis limits |
| `zlim` | `tuple` | `None` | Z-axis limits (3D only) |
| `colormap` | `str` | `"viridis"` | Colormap name |
| `show_colorbar` | `bool` | `True` | Show colorbar |
| `colorbar_label` | `str` | `None` | Colorbar label |
| `slice_axis` | `str` | `"x"` | Slice axis for 3D data |
| `**kwargs` | | | Additional matplotlib kwargs |

### Returns

- `Axes`: The matplotlib Axes object
- `Line2D` or `QuadMesh`: The plot object

### Example

```python
import matplotlib.pyplot as plt

Bx = dpy.timestep(1).fields.Bx()

ax, mesh = Bx.plot(
    colormap="RdBu",
    title="Magnetic Field Bx",
    show_colorbar=True
)

plt.savefig("Bx.png")
plt.show()
```

---

## 1D Averaging

Both `Field` and `Phase` provide methods for computing and plotting 1D averages along a chosen direction.

### avg_1d Method

Computes the mean and standard deviation along a specified direction, averaging over all perpendicular axes.

```python
def avg_1d(
    self,
    direction: Literal["x", "y", "z"] = "x",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `direction` | `str` | `"x"` | Direction along which to compute the average ("x", "y", or "z") |

#### Returns

A tuple of four numpy arrays:

| Return Value | Description |
|--------------|-------------|
| `coords` | 1D array of coordinates along the specified direction |
| `mean` | 1D array of mean values |
| `std_lower` | 1D array of (mean - standard deviation) |
| `std_upper` | 1D array of (mean + standard deviation) |

#### Example

```python
Bx = dpy.timestep(1).fields.Bx()

# Get averaged data for custom analysis
coords, mean, std_lower, std_upper = Bx.avg_1d("x")

# Use in your own plotting or analysis
import matplotlib.pyplot as plt
plt.plot(coords, mean)
plt.fill_between(coords, std_lower, std_upper, alpha=0.3)
plt.show()
```

### plot_1d_avg Method

Plots the 1D average with a shaded region showing ±1 standard deviation.

```python
def plot_1d_avg(
    self,
    direction: Literal["x", "y", "z"] = "x",
    *,
    ax: Optional[Axes] = None,
    dpi: int = 100,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    fill_alpha: float = 0.3,
    fill_color: Optional[str] = None,
    line_color: Optional[str] = None,
    show_std: bool = True,
    **kwargs
) -> Tuple[Axes, Line2D]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `direction` | `str` | `"x"` | Direction along which to plot ("x", "y", or "z") |
| `ax` | `Axes` | `None` | Matplotlib Axes (creates new if None) |
| `dpi` | `int` | `100` | Figure resolution |
| `title` | `str` | `None` | Plot title (auto-generated if None) |
| `xlabel` | `str` | `None` | X-axis label |
| `ylabel` | `str` | `None` | Y-axis label |
| `xlim` | `tuple` | `None` | X-axis limits |
| `ylim` | `tuple` | `None` | Y-axis limits |
| `fill_alpha` | `float` | `0.3` | Transparency of the std deviation fill region |
| `fill_color` | `str` | `None` | Color for the fill region (defaults to line color) |
| `line_color` | `str` | `None` | Color for the mean line |
| `show_std` | `bool` | `True` | Whether to show the standard deviation fill |
| `**kwargs` | | | Additional matplotlib kwargs for the line plot |

#### Returns

- `Axes`: The matplotlib Axes object
- `Line2D`: The line plot object

#### Example

```python
import matplotlib.pyplot as plt

Bx = dpy.timestep(1).fields.Bx()

# Simple usage
ax, line = Bx.plot_1d_avg("x")
plt.show()

# Customized plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
Bx.plot_1d_avg("x", ax=axes[0], fill_alpha=0.2, line_color="blue")
Bx.plot_1d_avg("y", ax=axes[1], fill_alpha=0.2, line_color="red")
plt.tight_layout()
plt.savefig("Bx_averages.png")
```

#### Behavior by Dimension

| Data Dimension | Direction | Averaging Axes |
|----------------|-----------|----------------|
| 1D | x | None (data returned as-is) |
| 2D | x | y |
| 2D | y | x |
| 3D | x | y, z |
| 3D | y | x, z |
| 3D | z | x, y |

---

## See Also

- [Plotting Guide](../user-guide/plotting.md)
- [Lazy Loading Guide](../user-guide/lazy-loading.md)
- [Working with Data](../user-guide/working-with-data.md)
