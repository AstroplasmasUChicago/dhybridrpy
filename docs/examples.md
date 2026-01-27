# Examples

This page provides practical examples for common use cases with dhybridrpy.

## Basic Usage

### Loading Simulation Data

```python
from dhybridrpy import DHybridrpy

# Initialize with paths to your simulation
dpy = DHybridrpy(
    input_file="examples/data/inputs/input",
    output_folder="examples/data/Output"
)

# Check available timesteps
print(f"Available timesteps: {dpy.timesteps()}")
```

### Accessing Input Parameters

```python
# The input file is parsed into a dictionary
print("Input sections:", list(dpy.inputs.keys()))

# Access specific parameters
dt = dpy.inputs['time']['dt']
print(f"Time step: {dt}")

# Access grid parameters
nx = dpy.inputs['grid']['nx']
print(f"Grid points in x: {nx}")
```

## Working with Fields

### Accessing Field Components

```python
# Get a timestep
ts = dpy.timestep(1)

# Access magnetic field components
Bx = ts.fields.Bx()
By = ts.fields.By()
Bz = ts.fields.Bz()

# Print data shape and range
print(f"Bx shape: {Bx.data.shape}")
print(f"Bx range: [{Bx.data.min():.3f}, {Bx.data.max():.3f}]")
```

### Comparing Field Types

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Compare Total, External, and Self fields
Bx_total = dpy.timestep(1).fields.Bx(type="Total")
Bx_ext = dpy.timestep(1).fields.Bx(type="External")
Bx_self = dpy.timestep(1).fields.Bx(type="Self")

Bx_total.plot(ax=axes[0], title="Bx (Total)")
Bx_ext.plot(ax=axes[1], title="Bx (External)")
Bx_self.plot(ax=axes[2], title="Bx (Self)")

plt.tight_layout()
plt.savefig("field_comparison.png", dpi=150)
plt.show()
```

## Calculating Derived Quantities

### Magnetic Field Magnitude

```python
import numpy as np
import matplotlib.pyplot as plt

ts = dpy.timestep(1)

Bx = ts.fields.Bx()
By = ts.fields.By()
Bz = ts.fields.Bz()

# Calculate magnitude using NumPy
B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

# Plot the result
B_mag.plot(title="|B|", colormap="plasma")
plt.savefig("B_magnitude.png", dpi=150)
plt.show()
```

### Energy Density

```python
import numpy as np

# Magnetic energy density: B²/2
Bx = dpy.timestep(1).fields.Bx()
By = dpy.timestep(1).fields.By()
Bz = dpy.timestep(1).fields.Bz()

B_squared = Bx**2 + By**2 + Bz**2
magnetic_energy = B_squared / 2

magnetic_energy.plot(title="Magnetic Energy Density")
plt.show()
```

## Time Evolution Analysis

### Track Maximum Field Over Time

```python
import matplotlib.pyplot as plt
import numpy as np

timesteps = dpy.timesteps()
max_Bx = []
times = []

for ts_num in timesteps:
    Bx = dpy.timestep(ts_num).fields.Bx()
    max_Bx.append(Bx.data.max())
    times.append(Bx.time)

plt.figure(figsize=(10, 6))
plt.plot(times, max_Bx, 'b-o')
plt.xlabel('Time')
plt.ylabel('Max Bx')
plt.title('Maximum Bx vs Time')
plt.grid(True)
plt.savefig("Bx_evolution.png", dpi=150)
plt.show()
```

### Create Time Animation

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 6))

def init():
    ax.clear()
    return []

def update(ts_num):
    ax.clear()
    Bx = dpy.timestep(ts_num).fields.Bx()
    Bx.plot(ax=ax)
    return []

ani = FuncAnimation(
    fig, update,
    frames=dpy.timesteps(),
    init_func=init,
    interval=200,
    blit=False
)

# Save as GIF
ani.save('Bx_animation.gif', writer='pillow', fps=5)
plt.show()
```

## Working with Phase Data

### Distribution Functions

```python
import matplotlib.pyplot as plt

# Get phase space distribution
f_xy = dpy.timestep(1).phases.x2x1(species=1)

# Plot distribution
f_xy.plot(
    colormap="hot",
    title="Particle Distribution (Species 1)"
)
plt.show()
```

### Comparing Species

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Compare species 1 and 2
f_s1 = dpy.timestep(1).phases.x2x1(species=1)
f_s2 = dpy.timestep(1).phases.x2x1(species=2)

f_s1.plot(ax=axes[0], title="Species 1")
f_s2.plot(ax=axes[1], title="Species 2")

plt.tight_layout()
plt.show()
```

### Velocity Space Analysis

```python
# Get velocity distribution
f_vxvy = dpy.timestep(1).phases.p2p1(species=1)

f_vxvy.plot(
    colormap="inferno",
    title="Velocity Distribution (vx vs vy)"
)
plt.show()
```

## Working with Raw Particle Data

### Accessing Particle Properties

```python
# Get raw particle data
raw = dpy.timestep(1).raw_files.raw(species=1)

# Access the data dictionary
data = raw.dict
print("Available quantities:", list(data.keys()))

# Get positions and momenta
x = data['x1']
p = data['p1']

print(f"Number of particles: {len(x)}")
```

### Particle Scatter Plot

```python
import matplotlib.pyplot as plt

raw = dpy.timestep(1).raw_files.raw(species=1)
data = raw.dict

x = data['x1']
y = data['x2']

plt.figure(figsize=(10, 8))
plt.scatter(x, y, s=1, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Particle Positions')
plt.savefig("particles.png", dpi=150)
plt.show()
```

## Lazy Loading for Large Datasets

### Memory-Efficient Processing

```python
# Enable lazy loading
dpy = DHybridrpy(
    input_file="path/to/input",
    output_folder="path/to/Output",
    lazy=True
)

# Process without loading all data into memory
for ts_num in dpy.timesteps():
    Bx = dpy.timestep(ts_num).fields.Bx()
    
    # Compute statistics without loading full array
    mean_val = Bx.data.mean().compute()
    max_val = Bx.data.max().compute()
    
    print(f"Timestep {ts_num}: mean={mean_val:.4f}, max={max_val:.4f}")
```

### Batch Statistics

```python
import dask

# Build up computations
means = []
maxes = []

for ts_num in dpy.timesteps():
    Bx = dpy.timestep(ts_num).fields.Bx()
    means.append(Bx.data.mean())
    maxes.append(Bx.data.max())

# Compute all at once (more efficient)
all_means, all_maxes = dask.compute(means, maxes)

print("Means:", all_means)
print("Maxes:", all_maxes)
```

## Multi-Panel Figures

### Field Overview

```python
import matplotlib.pyplot as plt

ts = dpy.timestep(1)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Magnetic field row
ts.fields.Bx().plot(ax=axes[0, 0], title="Bx")
ts.fields.By().plot(ax=axes[0, 1], title="By")
ts.fields.Bz().plot(ax=axes[0, 2], title="Bz")

# Electric field row
ts.fields.Ex().plot(ax=axes[1, 0], title="Ex")
ts.fields.Ey().plot(ax=axes[1, 1], title="Ey")
ts.fields.Ez().plot(ax=axes[1, 2], title="Ez")

# Current density row
ts.fields.Jx().plot(ax=axes[2, 0], title="Jx")
ts.fields.Jy().plot(ax=axes[2, 1], title="Jy")
ts.fields.Jz().plot(ax=axes[2, 2], title="Jz")

plt.tight_layout()
plt.savefig("field_overview.png", dpi=150)
plt.show()
```

## 1D Averaging Analysis

### Plot 1D Averages with Standard Deviation

```python
import matplotlib.pyplot as plt

# Get field data
Bx = dpy.timestep(1).fields.Bx()

# Plot 1D average along x with std deviation shading
ax, line = Bx.plot_1d_avg("x")
plt.savefig("Bx_avg_x.png", dpi=150)
plt.show()
```

### Compare Averages Along Different Directions

```python
import matplotlib.pyplot as plt

Bx = dpy.timestep(1).fields.Bx()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Average along x (shows variation in x, averaged over y)
Bx.plot_1d_avg("x", ax=axes[0], fill_alpha=0.3, line_color="blue")

# Average along y (shows variation in y, averaged over x)
Bx.plot_1d_avg("y", ax=axes[1], fill_alpha=0.3, line_color="red")

plt.tight_layout()
plt.savefig("Bx_directional_averages.png", dpi=150)
plt.show()
```

### Extract Averaged Data for Custom Analysis

```python
import matplotlib.pyplot as plt
import numpy as np

Bx = dpy.timestep(1).fields.Bx()

# Get the averaged data directly
coords, mean, std_lower, std_upper = Bx.avg_1d("x")

# Custom analysis: find where field exceeds threshold
threshold = 0.5 * mean.max()
exceeds = mean > threshold
print(f"Field exceeds threshold at x = {coords[exceeds]}")

# Custom plotting with more control
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(coords, mean, 'b-', linewidth=2, label='Mean')
ax.fill_between(coords, std_lower, std_upper, alpha=0.2, color='blue', label='±1σ')
ax.axhline(threshold, color='red', linestyle='--', label='Threshold')
ax.legend()
ax.set_xlabel('x / $d_i$')
ax.set_ylabel('Bx')
plt.savefig("Bx_custom_analysis.png", dpi=150)
plt.show()
```

### Time Evolution of 1D Profiles

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

for ts_num in dpy.timesteps()[::2]:  # Every other timestep
    Bx = dpy.timestep(ts_num).fields.Bx()
    coords, mean, _, _ = Bx.avg_1d("x")
    ax.plot(coords, mean, label=f't = {Bx.time:.2f}')

ax.set_xlabel('x / $d_i$')
ax.set_ylabel('Bx (averaged over y)')
ax.legend()
plt.savefig("Bx_profile_evolution.png", dpi=150)
plt.show()
```

### Phase Space Averaging

```python
import matplotlib.pyplot as plt

# Get density profile averaged along x
phase = dpy.timestep(1).phases.x3x2x1(species=1)

# Plot averaged density profile
ax, line = phase.plot_1d_avg("x", title="Density Profile (species 1)")
plt.savefig("density_profile.png", dpi=150)
plt.show()
```

## Saving Results

### Export to NumPy

```python
import numpy as np

Bx = dpy.timestep(1).fields.Bx()

# Save data array
np.save("Bx_data.npy", Bx.data)

# Save coordinates
np.savez(
    "Bx_full.npz",
    data=Bx.data,
    x=Bx.xdata,
    y=Bx.ydata
)
```

### Export to CSV

```python
import pandas as pd
import numpy as np

# For 1D data
Bx = dpy.timestep(1).fields.Bx()
df = pd.DataFrame({
    'x': Bx.xdata,
    'Bx': Bx.data
})
df.to_csv("Bx_1d.csv", index=False)
```

## See Also

- [Quick Start Guide](quickstart.md)
- [Working with Data](user-guide/working-with-data.md)
- [Plotting Guide](user-guide/plotting.md)
- [Lazy Loading Guide](user-guide/lazy-loading.md)
