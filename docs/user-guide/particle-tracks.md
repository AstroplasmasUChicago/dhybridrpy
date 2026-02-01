# Particle Tracks

This guide covers how to work with particle track data in dhybridrpy.

## Overview

Particle tracks record the trajectory of individual particles throughout the simulation. Each track contains position and momentum data at every tracked timestep, allowing you to follow a particle's motion through the simulation domain.

## Getting Started

### List Available Track IDs

```python
from dhybridrpy import DHybridrpy

dpy = DHybridrpy(
    input_file="path/to/input",
    output_folder="path/to/Output"
)

# Get array of all track IDs for species 1 (default)
track_ids = dpy.tracks()
print(f"Found {len(track_ids)} tracks")
print(track_ids[:5])  # e.g., ['0-1', '0-2', '0-3', '0-4', '0-5']
```

Track IDs are in the format `'rank-tag'` where `rank` is the MPI rank and `tag` is the particle's unique identifier.

### Access a Single Track

```python
# Get the first track
track_id = dpy.tracks()[0]
track = dpy.track(track_id)

# Or directly by ID
track = dpy.track('0-1465')
```

### Track Properties

Each track provides position and momentum data over time:

```python
track = dpy.track(track_id)

# Position components
x = track.x1  # x coordinate
y = track.x2  # y coordinate  
z = track.x3  # z coordinate

# Momentum components
px = track.p1  # x momentum
py = track.p2  # y momentum
pz = track.p3  # z momentum

# Time array
t = track.t
```

All properties return NumPy arrays with one value per tracked timestep.

## Iterating Over Tracks

### Using Track IDs

For bulk operations, use `tracks()` to get an array of track IDs and iterate:

```python
# Iterate over all tracks
for track_id in dpy.tracks():
    track = dpy.track(track_id)
    print(f"Track {track.track_id}: final x = {track.x1[-1]:.2f}")

# Get total number of tracks
n_tracks = len(dpy.tracks())
print(f"Total: {n_tracks} tracks")
```

### Filtering Tracks

Use list comprehensions to filter tracks by criteria:

```python
# Find tracks that reached x > 100
far_tracks = [dpy.track(tid) for tid in dpy.tracks() if dpy.track(tid).x1[-1] > 100]

# Find most energetic particle at end of simulation
import numpy as np

def final_energy(track):
    return np.sqrt(track.p1[-1]**2 + track.p2[-1]**2 + track.p3[-1]**2)

tracks = [dpy.track(tid) for tid in dpy.tracks()]
energetic_track = max(tracks, key=final_energy)
print(f"Most energetic: {energetic_track.track_id}")
```

## Working with Multiple Species

If your simulation tracks multiple species, specify the species number:

```python
# Get tracks for species 2
track_ids_sp2 = dpy.tracks(species=2)
track = dpy.track('0-100', species=2)

# Iterate over species 2 tracks
for track_id in dpy.tracks(species=2):
    track = dpy.track(track_id, species=2)
    print(track.track_id)
```

## Plotting Trajectories

### Basic Trajectory Plot

```python
import matplotlib.pyplot as plt

track = dpy.track(dpy.tracks()[0])

plt.figure(figsize=(10, 6))
plt.plot(track.x1, track.x2)
plt.xlabel('$x/d_i$')
plt.ylabel('$y/d_i$')
plt.title(f'Particle Trajectory: {track.track_id}')
plt.show()
```

### Trajectory with Time Coloring

```python
import matplotlib.pyplot as plt
import numpy as np

track = dpy.track(dpy.tracks()[0])

plt.figure(figsize=(10, 6))
scatter = plt.scatter(track.x1, track.x2, c=track.t, cmap='viridis', s=1)
plt.colorbar(scatter, label='$t \\omega_{ci}$')
plt.xlabel('$x/d_i$')
plt.ylabel('$y/d_i$')
plt.title('Particle Trajectory Colored by Time')
plt.show()
```

### Use Lazy Loading for Large Datasets

Enable lazy loading to defer data loading until needed:

```python
dpy = DHybridrpy(
    input_file="path/to/input",
    output_folder="path/to/Output",
    lazy=True  # Use Dask for lazy loading
)

track = dpy.track('0-1')
# Data isn't loaded until you access it
x = track.x1.compute()  # Now it loads
```