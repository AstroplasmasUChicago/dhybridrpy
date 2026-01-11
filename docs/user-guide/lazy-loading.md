# Lazy Loading

For large simulations with many timesteps or high-resolution data, dhybridrpy supports lazy loading through [Dask](https://dask.org/). This allows you to work with datasets that don't fit in memory.

## Enabling Lazy Loading

Enable lazy loading when initializing DHybridrpy:

```python
from dhybridrpy import DHybridrpy

dpy = DHybridrpy(
    input_file="path/to/input",
    output_folder="path/to/Output",
    lazy=True  # Enable lazy loading
)
```

## How It Works

With lazy loading enabled:

1. **Metadata is loaded immediately** - File paths, shapes, and dtypes are read
2. **Data is loaded on-demand** - Actual array data is only loaded when needed
3. **Operations are deferred** - Computations are built up as a task graph
4. **Execution happens on `.compute()`** - The actual computation runs when you request results

## Working with Lazy Arrays

### Checking Array Type

```python
Bx = dpy.timestep(1).fields.Bx()

# Check if it's a Dask array
import dask.array as da
print(isinstance(Bx.data, da.Array))  # True if lazy=True
```

### Explicit Computation

```python
# Get the lazy array
Bx = dpy.timestep(1).fields.Bx()
print(type(Bx.data))  # dask.array.Array

# Compute to get a NumPy array
data_numpy = Bx.data.compute()
print(type(data_numpy))  # numpy.ndarray
```

### Deferred Operations

```python
import numpy as np

# Build up a computation (no data loaded yet)
Bx = dpy.timestep(1).fields.Bx()
By = dpy.timestep(1).fields.By()
Bz = dpy.timestep(1).fields.Bz()

# This creates a task graph, not actual data
B_mag = np.sqrt(Bx.data**2 + By.data**2 + Bz.data**2)

# Now compute the result
result = B_mag.compute()
```

## Plotting with Lazy Data

The `plot()` method automatically handles lazy arrays:

```python
# This works seamlessly - data is computed when needed
Bx = dpy.timestep(1).fields.Bx()
Bx.plot()  # Automatically calls .compute() internally
plt.show()
```

## Memory Efficiency

### Processing Many Timesteps

```python
import numpy as np

# With lazy loading, this doesn't load all data into memory
max_values = []
for ts in dpy.timesteps():
    Bx = dpy.timestep(ts).fields.Bx()
    # Only loads data when .compute() is called
    max_val = Bx.data.max().compute()
    max_values.append(max_val)
```

### Chunked Processing

Dask automatically chunks large arrays:

```python
Bx = dpy.timestep(1).fields.Bx()

# View chunk structure
print(Bx.data.chunks)  # e.g., ((256, 256), (128, 128))
```

## When to Use Lazy Loading

| Scenario | Recommendation |
|----------|---------------|
| Small datasets (< 1 GB) | `lazy=False` (default) |
| Large datasets (> 1 GB) | `lazy=True` |
| Many timesteps | `lazy=True` |
| Quick exploration | `lazy=False` |
| Batch processing | `lazy=True` |
| Interactive plotting | Either (handled automatically) |

## Performance Tips

### 1. Minimize Compute Calls

```python
# Less efficient: multiple compute calls
mean_val = Bx.data.mean().compute()
std_val = Bx.data.std().compute()

# More efficient: single compute call
import dask
mean_val, std_val = dask.compute(Bx.data.mean(), Bx.data.std())
```

### 2. Use Appropriate Chunk Sizes

Dask chooses reasonable defaults, but you can optimize for your use case.

### 3. Consider Memory vs. Speed Trade-offs

Lazy loading reduces memory usage but may be slower for small datasets due to the overhead of building task graphs.

## Raw Data with Lazy Loading

Raw particle data also supports lazy loading:

```python
raw = dpy.timestep(1).raw_files.raw(species=1)

# Dictionary of lazy arrays
data_dict = raw.dict

# Compute specific quantities
positions_x = data_dict['x1'].compute()
```
