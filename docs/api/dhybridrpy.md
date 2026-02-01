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
| `dt` | `float` | Simulation timestep size |
| `start_time` | `float` | Simulation start time |

## Methods

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

Retrieve an array of all available timesteps.

**Returns:** NumPy array of timestep numbers (sorted)

**Example:**

```python
all_timesteps = dpy.timesteps()
print(f"Available: {all_timesteps}")
print(f"First: {all_timesteps[0]}, Last: {all_timesteps[-1]}")
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
for ts in dpy.timesteps():
    ts = dpy.timestep(ts)
    Bx = ts.fields.Bx()
    print(f"Timestep {ts}: Bx max = {Bx.data.max()}")
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