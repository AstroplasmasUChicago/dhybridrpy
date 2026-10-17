# dhybridrpy

![PyPI version](https://img.shields.io/pypi/v/dhybridrpy?label=PyPI&color=blue)
[![Documentation Status](https://readthedocs.org/projects/dhybridrpy/badge/?version=latest)](https://dhybridrpy.readthedocs.io/en/latest/)

Read and plot dHybridR simulation output in Python: fields, phase diagrams,
fluid moments, raw particles, and particle tracks.

Browse 3D slices, calculate spatial profiles and spectra, or process a run
across timesteps. Dask support lets you defer large array reads.

```bash
pip install dhybridrpy
```

Requires Python 3.9 or later.

## Load and plot

```python
import matplotlib.pyplot as plt
from dhybridrpy import DHybridrpy

dpy = DHybridrpy(input_file="run/input", output_folder="run/Output")
ts = dpy.timestep_index(-1)
print(ts.fields)
ts.fields.Bx().plot()
plt.show()
```

Use paths to your run; this example requires a total `Bx` output.
The [quick start](https://dhybridrpy.readthedocs.io/en/latest/quickstart/)
explains how to select an output and plot a field.

## Plot from the terminal

```bash
# List available fields and phases.
dplot -i run/input -o run/Output

# Save Bx images for all available outputs.
dplot -i run/input -o run/Output --fields Bx
```

Add `--video` to make an MP4 (requires FFmpeg). See the
[dplot guide](https://dhybridrpy.readthedocs.io/en/latest/user-guide/dplot/)
for phase plots, species selection, and rendering options.

## Documentation

- [Data access and selection](https://dhybridrpy.readthedocs.io/en/latest/user-guide/working-with-data/)
- [Focused examples](https://dhybridrpy.readthedocs.io/en/latest/examples/)
- [API reference](https://dhybridrpy.readthedocs.io/en/latest/api/dhybridrpy/)

By Bricker Ostler and Miha Cernetic. Licensed under [AGPL-3.0](LICENSE).
