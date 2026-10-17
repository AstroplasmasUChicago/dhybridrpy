# Field history

Reduce each saved `Bx` field to its maximum, plot the result against time,
and save `Bx-max.csv`. This requires total `Bx` output at every selected
timestep.

Save this as `field_history.py` beside your run's `input` and `Output`,
then run `python field_history.py`:

```python
import matplotlib.pyplot as plt
import numpy as np
from dhybridrpy import DHybridrpy


if __name__ == "__main__":
    dpy = DHybridrpy(input_file="input", output_folder="Output")
    steps = dpy.timesteps()
    times = dpy.times()
    maxima = dpy.field_timeseries("Bx", timesteps=steps, apply=np.max)

    np.savetxt(
        "Bx-max.csv",
        np.column_stack((times, maxima)),
        delimiter=",",
        header="time,max_Bx",
        comments="",
    )
    fig, ax = plt.subplots()
    ax.plot(times, maxima, "o-")
    ax.set(xlabel="Simulation time", ylabel="Maximum Bx")
    plt.show()
```

The `apply=np.max` argument returns one number per timestep instead of
stacking the full fields in memory. Reads run in worker processes, so
scripts need the `__main__` guard shown above.

If diagnostics have different output intervals, select only steps that
contain `Bx` and the matching entries from `dpy.times()`.
