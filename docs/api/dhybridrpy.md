# DHybridrpy

The main class for loading and accessing dHybridR simulation data.

## Class Definition

```python
class DHybridrpy(
    input_file: str,
    output_folder: str,
    lazy: bool = False,
    exclude_timestep_zero: bool = True
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_file` | `str` | required | Path to the dHybridR input file |
| `output_folder` | `str` | required | Path to the dHybridR output folder |
| `lazy` | `bool` | `False` | Enable lazy loading via Dask |
| `exclude_timestep_zero` | `bool` | `True` | Exclude timestep 0 from the timesteps list |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `input_file` | `str` | Path to the input file |
| `output_folder` | `str` | Path to the output folder |
| `lazy` | `bool` | Whether lazy loading is enabled |
| `inputs` | `Namelist` | Parsed input file as a dictionary-like object |
| `dt` | `float` | Simulation timestep size (from input file) |
| `start_time` | `float` | Deprecated `t0` from the input file; 0.0 when absent (dHybridR never applied it) |
| `adaptive_dt` | `bool` | Whether the run used adaptive time stepping (from input file) |

## Methods

### `field_timeseries(name, type="Total", timesteps=None, apply=None, workers=None)`

One field across many timesteps. Prefer this over looping. Code like

```python
for ts in dpy.timesteps():
    data = dpy.timestep(ts).fields.Bx().data   # slow: one file at a time
```

reads files one at a time, while `field_timeseries` reads them with a
pool of parallel worker processes.

Without `apply`, returns an array of shape `(num_timesteps, *grid)`;
memory is the full selection, so pass a `timesteps` subset for very
large runs:

```python
B_x = dpy.field_timeseries("Bx", timesteps=dpy.timesteps()[-20:])
```

With `apply`, the function runs on each timestep's field(s) inside the
workers and only its results travel back. This scales to runs whose
full data would not fit in memory. `apply` must be importable (a
module-level function such as `np.mean`, not a lambda). `name` may be a
list, in which case `apply` receives one array per name:

```python
mean_Bx = dpy.field_timeseries("Bx", apply=np.mean)

def mean_bperp(bx, by):
    return np.sqrt(bx**2 + by**2).mean()

bperp = dpy.field_timeseries(["Bx", "By"], apply=mean_bperp)
```

`phase_timeseries(name, species=1, ...)` is the phase-quantity
counterpart.

Notes: the worker pool starts on first use and persists. Workers are
spawned processes, so in a script the call must run under an
`if __name__ == "__main__":` guard (notebooks and interactive sessions
need no guard; scripts piped through stdin cannot spawn workers at all).

---

### `timestep(ts: int) -> Timestep`

Access field, phase, and raw file information at a given timestep.

**Parameters:**

- `ts` (`int`): The timestep number to access

**Returns:** `Timestep` object

**Raises:** `ValueError` if the timestep is not found

**Example:**

```python
ts = dpy.timestep(1)
print(ts.fields)
print(ts.phases)
```

---

### `timestep_closest(ts: int, verbose: bool = False) -> Timestep`

Access field, phase, and raw file information at the closest available timestep.

**Parameters:**

- `ts` (`int`): The target timestep number to find the closest match for
- `verbose` (`bool`): If `True`, logs information about the requested and closest available timesteps

**Returns:** `Timestep` object

**Raises:** `ValueError` if there are no available timesteps

**Example:**

```python
ts_first = dpy.timestep_closest(100, verbose=True)
INFO:dhybridrpy.dhybridrpy:Requested timestep: 100. Closest available timestep: 96.
```

---

### `timestep_index(index: int) -> Timestep`

Access field, phase, and raw file information at a given timestep index.

**Parameters:**

- `index` (`int`): The index into the sorted timesteps array (supports negative indexing)

**Returns:** `Timestep` object

**Raises:** `IndexError` if the index is out of range

**Example:**

```python
# First timestep
ts_first = dpy.timestep_index(0)

# Last timestep
ts_last = dpy.timestep_index(-1)
```

---

### `timesteps() -> np.ndarray`

Retrieve an array of available timesteps for fields and phases. Raw particle data may be dumped at different intervals; use [`raw_timesteps()`](#raw_timesteps---npndarray) to get those.

**Returns:** NumPy array of field/phase timestep numbers (sorted)

**Example:**

```python
all_timesteps = dpy.timesteps()
print(f"Available: {all_timesteps}")
print(f"First: {all_timesteps[0]}, Last: {all_timesteps[-1]}")
```

---

### `times() -> np.ndarray`

Retrieve an array of simulation times corresponding to each field/phase timestep. Times are read from the HDF5 file `TIME` attribute.

**Returns:** NumPy array of simulation times (sorted by timestep)

**Example:**

```python
times = dpy.times()
print(f"Simulation times: {times}")
print(f"Start: {times[0]}, End: {times[-1]}")
```

---

### `raw_timesteps() -> np.ndarray`

Retrieve an array of available timesteps for raw particle data. Raw files may be dumped at different intervals than fields and phases.

**Returns:** NumPy array of raw timestep numbers (sorted)

**Example:**

```python
raw_ts = dpy.raw_timesteps()
print(f"Raw timesteps: {raw_ts}")
print(f"First: {raw_ts[0]}, Last: {raw_ts[-1]}")
```

---

### `raw_times() -> np.ndarray`

Retrieve an array of simulation times corresponding to each raw particle timestep. Times are read from the HDF5 file `TIME` attribute.

**Returns:** NumPy array of simulation times (sorted by raw timestep)

**Example:**

```python
raw_times = dpy.raw_times()
print(f"Raw simulation times: {raw_times}")
print(f"Start: {raw_times[0]}, End: {raw_times[-1]}")
```

## Usage Examples

### Basic Initialization

```python
from dhybridrpy import DHybridrpy

dpy = DHybridrpy(
    input_file="examples/data/inputs/input",
    output_folder="examples/data/Output"
)
```

### Accessing Input Parameters

```python
# View all input sections
print(dpy.inputs.keys())

# Access specific parameters
dt = dpy.inputs['time']['dt']

# Access pre-extracted time parameters
print(f"dt = {dpy.dt}")
print(f"t0 = {dpy.start_time}")
```

### Iterating Over Timesteps

```python
for ts_num in dpy.timesteps():
    ts = dpy.timestep(ts_num)
    Bx = ts.fields.Bx()
    print(f"Timestep {ts_num}: Bx max = {Bx.data.max()}")
```

### With Lazy Loading

```python
dpy = DHybridrpy(
    input_file="path/to/input",
    output_folder="path/to/Output",
    lazy=True
)

# Data is not loaded until needed
Bx = dpy.timestep(1).fields.Bx()
print(Bx.data)  # Dask array (not computed)
print(Bx.data.compute())  # NumPy array (computed)
```
