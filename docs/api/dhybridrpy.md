# DHybridrpy

Access a run's outputs, input parameters, and particle tracks.

## Constructor

```python
DHybridrpy(input_file, output_folder, lazy=False, exclude_timestep_zero=True)
```

| Parameter | Meaning |
| --- | --- |
| `input_file` | Input-file path; requires `time.dt` |
| `output_folder` | Output-directory path |
| `lazy` | Return Dask arrays instead of NumPy arrays; default `False` |
| `exclude_timestep_zero` | Omit iteration 0 from timestep/time lists; default `True` |

Reads file metadata; array values are loaded on access.

**Raises:** `FileNotFoundError` for a missing input, `NotADirectoryError` for
an invalid output directory, `KeyError` for missing `time`/`dt`, or
`ValueError` for inconsistent fixed timestep sizes in output files.

```python
from dhybridrpy import DHybridrpy

dpy = DHybridrpy("run/input", "run/Output")
```

## Attributes

| Attribute | Value |
| --- | --- |
| `input_file`, `output_folder`, `lazy`, `exclude_timestep_zero` | Constructor settings |
| `inputs` | Parsed `f90nml.Namelist`; e.g. `dpy.inputs["time"]["dt"]` |
| `dt` | Input timestep size as a float |
| `adaptive_dt` | Adaptive-timestep setting; default `False` |
| `start_time` | Legacy `t0`, or `0.0`; ignored when determining output times |

## Timestep selection { #select-a-timestep }

All methods return a [Timestep](timestep.md). `ts` is an iteration number.

| Method | Selection | Raises |
| --- | --- | --- |
| `timestep(ts)` | Exact iteration; includes raw-only output and iteration 0 | `ValueError` if absent |
| `timestep_index(index)` | Index in `timesteps()`; `0` first, `-1` last | `IndexError` if out of range |
| `timestep_closest(ts, verbose=False)` | Nearest iteration in `timesteps()`; ties choose the earlier | `ValueError` if no field/phase output |

`verbose=True` logs the requested and selected iterations.

## Iterations and times

Each method returns a 1D NumPy array.

| Method | Values |
| --- | --- |
| `timesteps()` | Sorted iterations with field or phase output |
| `times()` | Simulation times matching `timesteps()` |
| `raw_timesteps()` | Sorted iterations with raw particle output |
| `raw_times()` | Simulation times matching `raw_timesteps()` |

A quantity may be absent at some listed iterations. Times use HDF5 `TIME`,
or `iteration * dt` after a fixed-step consistency check.

## Time series { #read-across-timesteps }

```python
dpy.field_timeseries(name, type="Total", timesteps=None, apply=None, workers=None)
dpy.phase_timeseries(name, species=1, timesteps=None, apply=None, workers=None)
```

| Parameter | Meaning |
| --- | --- |
| `name` | Quantity name; a list of names requires `apply` |
| `type` / `species` | Field type / species; defaults shown above |
| `timesteps` | Iterations in result order; `None` uses `dpy.timesteps()` |
| `apply` | Function receiving one NumPy array per name, in name order |
| `workers` | Process count; `None` uses `min(8, CPU count)` |

| Mode | Returns | Requirements |
| --- | --- | --- |
| `apply=None` | NumPy array of shape `(number_of_timesteps, *grid_shape)` | One quantity; equal shapes and dtypes; memory for the stack and a copy |
| `apply=function` | List of results in timestep order | Importable function, such as `np.mean` or a module-level function |

- Reads eagerly, including with `lazy=True`.
- Every quantity must exist at every selected iteration. Empty selections raise `ValueError`.
- Scripts require a file and a main guard. In notebooks, put custom functions in an importable `.py` module.

**Example:** reduce one field or combine two fields; requires saved `Bx` and `By`.

```python
import numpy as np
from dhybridrpy import DHybridrpy

def mean_Bxy(bx, by):
    return np.sqrt(bx**2 + by**2).mean()

if __name__ == "__main__":
    dpy = DHybridrpy("run/input", "run/Output")
    means = dpy.field_timeseries("Bx", apply=np.mean)
    combined = dpy.field_timeseries(["Bx", "By"], apply=mean_Bxy)
```

## Tracks

| Method | Returns | Raises |
| --- | --- | --- |
| `tracks(species=1)` | NumPy array of `"rank-tag"` IDs, sorted numerically by rank and tag | `ValueError` if no track file for the species |
| `track(track_id, species=1)` | [Track](tracks.md#track) using `dpy.lazy` | `ValueError` for absent species; `KeyError` for absent ID |

## close_worker_pool { #release-worker-processes }

```python
from dhybridrpy import close_worker_pool

close_worker_pool()
```

**Returns:** `None`. Closes the worker pool; the next parallel read reopens it.
Cleanup also runs automatically at process exit.
