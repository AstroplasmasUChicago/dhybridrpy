# Quick start

Open a run, choose a saved output, and plot its magnetic field.
Install first with `pip install dhybridrpy`.

## Open a run

Replace the paths below with your simulation input file and `Output` directory:

```python
from dhybridrpy import DHybridrpy

dpy = DHybridrpy(input_file="run/input", output_folder="run/Output")
```

## Choose an output

```python
print(dpy.timesteps())
ts = dpy.timestep_index(-1)
print(ts.fields)
```

`timestep_index(-1)` selects the last timestep with field or phase output.
Use `dpy.timestep(32)` to select iteration 32 instead. `ts.fields` lists the
fields saved at that timestep.

## Plot a field

If `Bx` is listed under `Total`:

```python
import matplotlib.pyplot as plt

Bx = ts.fields.Bx()
Bx.plot()
plt.show()
```

`plot()` draws a line for 1D data, an image for 2D data, or a slice viewer for
3D data. For calculations, `Bx.data` returns the field as a NumPy array.

Continue with [data selection and calculations](user-guide/working-with-data.md),
[plot controls](user-guide/plotting.md), or a [focused example](examples.md).
