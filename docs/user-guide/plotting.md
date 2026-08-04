# Plotting

dhybridrpy provides built-in plotting capabilities for quick visualization of simulation data.

## Basic Plotting

All data objects (Field, Phase) have a `plot()` method:

```python
import matplotlib.pyplot as plt

Bx = dpy.timestep(1).fields.Bx()
Bx.plot()
plt.show()
```

The plot method automatically detects the dimensionality of your data and creates the appropriate plot type.

## 1D Data

For 1D data, a line plot is created:

```python
# 1D field data
field_1d = dpy.timestep(1).fields.Bx()
ax, line = field_1d.plot()
plt.show()
```

## 2D Data

For 2D data, an image plot is created with `imshow`. The returned plot
object is a matplotlib `AxesImage`:

```python
# 2D field data
Bx = dpy.timestep(1).fields.Bx()
ax, image = Bx.plot(
    colormap="viridis",
    show_colorbar=True
)
plt.show()
```

The image is drawn with `origin="lower"`, `interpolation="nearest"`, and
`aspect="auto"` by default. Pass any of these as keyword arguments to
override them.

## 3D Data

For 3D data, an interactive slice viewer is created. The figure has two
panels:

- **Left**: the current 2D slice, drawn with `imshow`.
- **Right**: a rotatable context cube showing the outer faces of the
  volume, with a deep-red frame marking the position of the current
  slice.

```python
# 3D field data - interactive slice viewer
Bx = dpy.timestep(1).fields.Bx()
ax, image = Bx.plot(
    slice_axis="x"  # Slice along x-axis
)
plt.show()
```

Use the slider to navigate through slices along the chosen axis, or
press the left and right arrow keys to step one slice at a time. Each
slice is read from the file on demand, so the full 3D volume is never
held in memory.

The color limits rescale to each slice's data range as you move through
the volume. Pass `vmin`/`vmax` (or a `norm`) to fix them instead.

### Slice Axis Options

- `slice_axis="x"`: View y-z planes at different x positions
- `slice_axis="y"`: View x-z planes at different y positions  
- `slice_axis="z"`: View x-y planes at different z positions

### Disabling the Context Cube

Pass `context_3d=False` to show only the slice panel:

```python
ax, image = Bx.plot(slice_axis="z", context_3d=False)
```

The context cube is also skipped when you pass your own `ax`.

## Plot Customization

### All Plot Options

```python
Bx.plot(
    # Matplotlib Axes to use (creates new figure if None)
    ax=None,
    
    # Figure resolution
    dpi=100,
    
    # Labels and title
    title="Custom Title",
    xlabel="X Label",
    ylabel="Y Label",
    zlabel="Z Label",  # For 3D data
    
    # Axis limits
    xlim=(0, 10),
    ylim=(0, 5),
    zlim=(0, 5),  # For 3D data
    
    # Colormap (2D/3D only)
    colormap="viridis",
    
    # Colorbar options (2D/3D only)
    show_colorbar=True,
    colorbar_label="Field Value",
    
    # 3D slice options
    slice_axis="x",  # "x", "y", or "z"
    context_3d=True,  # Show the context cube next to the slice
)
```

The `plot()` method returns `(ax, artist)`. For 1D data the artist is a
`Line2D`; for 2D data and 3D slices it is an `AxesImage`.

## Subplots

Combine multiple plots in a figure:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

Bx = dpy.timestep(1).fields.Bx()
By = dpy.timestep(1).fields.By()
Bz = dpy.timestep(1).fields.Bz()

Bx.plot(ax=axes[0], title="Bx")
By.plot(ax=axes[1], title="By")
Bz.plot(ax=axes[2], title="Bz")

plt.tight_layout()
plt.show()
```

## Plotting Derived Quantities

Plot results of arithmetic operations:

```python
import numpy as np

Bx = dpy.timestep(1).fields.Bx()
By = dpy.timestep(1).fields.By()
Bz = dpy.timestep(1).fields.Bz()

# Calculate and plot magnetic field magnitude
B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
B_mag.plot(title="|B|")
plt.show()
```

## Time Series Animation

Create animations across timesteps:

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

def update(ts):
    ax.clear()
    Bx = dpy.timestep(ts).fields.Bx()
    Bx.plot(ax=ax)
    return ax,

ani = FuncAnimation(
    fig, update, 
    frames=dpy.timesteps(), 
    interval=200
)
plt.show()
```

A loop like this is fine for rendering frames. To compute numbers
across timesteps (means, maxima, profiles), use `field_timeseries`
instead of a loop; see
[Working with Data](working-with-data.md#reading-across-timesteps).

## Saving Plots

```python
Bx = dpy.timestep(1).fields.Bx()
Bx.plot()
plt.savefig("Bx_plot.png", dpi=300, bbox_inches="tight")
```

