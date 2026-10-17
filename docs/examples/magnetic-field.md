# Magnetic magnitude and profile

Compute `|B|`, plot its 2D map, and average over y to get a profile along x.
The example requires `Bx`, `By`, and `Bz` at the last saved timestep of a
2D run and saves `magnetic-field.png`.

```python
import matplotlib.pyplot as plt
import numpy as np
from dhybridrpy import DHybridrpy

dpy = DHybridrpy(input_file="input", output_folder="Output")
ts = dpy.timestep_index(-1)
Bx = ts.fields.Bx()
By = ts.fields.By()
Bz = ts.fields.Bz()
Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
Bmag.plot(ax=axes[0], title="Magnetic magnitude", colorbar_label="|B|")
Bmag.plot_1d_avg("x", ax=axes[1], title="Mean over y", ylabel="|B|")
fig.savefig("magnetic-field.png", dpi=150, bbox_inches="tight")
plt.show()
```

`plot_1d_avg("x")` keeps x and averages over the other axes. The shaded band
shows the spread across y: mean ± one standard deviation. To obtain those
arrays without plotting:

```python
x, mean, lower, upper = Bmag.avg_1d("x")
```
