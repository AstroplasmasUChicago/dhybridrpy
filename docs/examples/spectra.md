# Power spectra

Compare magnetic power by field component or spatial direction. These
examples use total magnetic fields at the last saved timestep of a 2D run.
FFT methods load the selected field into memory, including with `lazy=True`;
they apply no window or mean subtraction.

## Compare field components

```python
import matplotlib.pyplot as plt
from dhybridrpy import DHybridrpy

dpy = DHybridrpy(input_file="input", output_folder="Output")
ts = dpy.timestep_index(-1)

fig, ax = plt.subplots()
for name in ("Bx", "By", "Bz"):
    field = getattr(ts.fields, name)()
    field.plot_fft_power(ax=ax, label=name)
ax.set_title("Magnetic component spectra")
ax.legend()
plt.show()
```

Each curve sums power in radial wavenumber bins without averaging over
modes. The plot uses logarithmic axes, omitting the first bin (which
contains the zero mode) and any nonpositive power.
Use `field.fft_power()` to obtain the `(k, power)` arrays.

## Compare directions

Using the same timestep, compare variations along x and y:

```python
Bx = ts.fields.Bx()
fig, ax = plt.subplots()
Bx.plot_fft_power_1d("x", ax=ax, label="Along x")
Bx.plot_fft_power_1d("y", ax=ax, label="Along y")
ax.set_title("Directional Bx spectra")
ax.legend()
plt.show()
```

For x, a separate FFT is computed at each y. The line is the geometric mean
of positive powers; the band shows one standard deviation in log power.
These directional averages have a different normalization from the radial
sums above; their amplitudes are not directly comparable.

Use `Bx.fft_power_1d("x")` to obtain `(k, mean, lower, upper)`. Each direction
has its own `k` array, measured in angular wavenumber (radians per unit length).
See the [spectrum reference](../api/data.md) for binning and normalization.
