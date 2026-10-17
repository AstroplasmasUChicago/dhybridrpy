# Particle tracks

Use tracks to follow a particle's position, velocity, or energy over time.
Your run must have saved track output under `Output/Tracks`.

## Open a track

```python
from dhybridrpy import DHybridrpy

dpy = DHybridrpy(input_file="run/input", output_folder="run/Output")
track_ids = dpy.tracks(species=1)
print(track_ids)

track = dpy.track(track_ids[0], species=1)
print(track)
```

Choose an ID from `dpy.tracks()`; `print(track)` lists its available datasets.
IDs have the form `"rank-tag"`.

## Plot a trajectory

For a run with at least two spatial dimensions:

```python
import matplotlib.pyplot as plt

plt.plot(track.x1, track.x2)
plt.xlabel(r"$x/d_i$")
plt.ylabel(r"$y/d_i$")
plt.show()
```

Position datasets `x1`, `x2`, and `x3` correspond to x, y, and z.
Only the simulated dimensions are stored.

## Plot energy over time

```python
plt.plot(track.t, track.ene)
plt.xlabel(r"$t\omega_{ci}$")
plt.ylabel(r"$E_{\mathrm{kin}}/(m_p v_A^2)$")
plt.show()
```

Use each track's stored `t` array: tracks can start and end at different
times. The [dataset reference](../api/tracks.md#datasets) lists velocities,
iteration numbers, and optional fields.

## Compare many tracks

Read the same dataset from several tracks with `TrackCollection`. For
example, find each track's peak kinetic energy:

```python
from dhybridrpy import TrackCollection

with TrackCollection("run/Output/Tracks/Sp01/track_Sp01.h5", species=1) as collection:
    energies = collection.load_dataset("ene")
    peak_energy = {track_id: values.max() for track_id, values in energies.items()}
```

`load_dataset` returns a dictionary of arrays keyed by track ID; the arrays
may have different lengths. To select tracks, pass
`track_ids=collection.track_ids[:3]`. The `with` block closes the file when
finished.

## Load tracks lazily

```python
dpy = DHybridrpy("run/input", "run/Output", lazy=True)
track = dpy.track(dpy.tracks(species=1)[0], species=1)
x = track.x1.compute()
```

With `lazy=True`, datasets are Dask arrays until `.compute()` loads them.
`TrackCollection` accepts the same option. See [lazy loading](lazy-loading.md)
for calculations that defer reading data.
