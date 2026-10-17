# Plotting

Call `.plot()` on any field or phase. It chooses a line, image, or interactive
slice from the data's dimensions:

```python
import matplotlib.pyplot as plt
from dhybridrpy import DHybridrpy

dpy = DHybridrpy(input_file="run/input", output_folder="run/Output")
ts = dpy.timestep_index(-1)
Bx = ts.fields.Bx()
ax, artist = Bx.plot()
plt.show()
```

The returned `ax` and `artist` are Matplotlib objects. Use them to adjust
labels, add annotations, or save the figure.

## Set image colors

For a 2D field or phase, choose a colormap and fixed color limits:

```python
ax, image = Bx.plot(
    colormap="RdBu_r",
    vmin=-1,
    vmax=1,
    title="Bx",
    colorbar_label="Bx",
)
plt.show()
```

Use the same `vmin` and `vmax` to compare images on one scale.
`show_colorbar=False` hides the colorbar; `norm=` accepts a Matplotlib
normalization object, such as `LogNorm` for positive data.

## Arrange and save plots

Pass `ax=` to draw in an existing subplot:

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
Bx.plot(ax=axes[0], title="Bx")
ts.fields.By().plot(ax=axes[1], title="By")
fig.savefig("magnetic-components.png", dpi=150, bbox_inches="tight")
plt.show()
```

Use the returned axes for further changes, such as `ax.set_xlabel("x")`.

## Browse a 3D field

For a 3D field, choose the axis along which the slice moves:

```python
ax, image = Bx.plot(slice_axis="z")
plt.show()
```

| `slice_axis` | Displayed plane |
| --- | --- |
| `"x"` | yz plane |
| `"y"` | xz plane |
| `"z"` | xy plane |

Move the slider or press the left or right arrow key to change slices.
The controls need an interactive Matplotlib backend; a static notebook
image cannot respond to them. The adjacent cube shows the slice position;
hide it with `context_3d=False`. Supplying `ax=` also omits the cube.

For fields stored in HDF5, each slice is read as needed. The color scale
adjusts to each slice unless you pass `vmin`, `vmax`, or `norm`.

## Profiles and spectra

- [Magnetic magnitude and profile](../examples/magnetic-field.md):
  plot a derived field and its average over the transverse direction.
- [Power spectra](../examples/spectra.md):
  compare magnetic field components or spectra along different axes.
- [Plot options reference](../api/data.md): all options and return values.
