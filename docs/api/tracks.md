# Tracks

`Track`: one particle's recorded samples. `TrackCollection`: tracks in one
species file. [Plotting examples](../user-guide/particle-tracks.md).

## Track

Returned by [`dpy.track(track_id, species=1)`](dhybridrpy.md#tracks) or
`collection[track_id]`.

| Access | Result |
|---|---|
| `track.x1`, `track.t`, etc. | NumPy arrays, or Dask arrays when `lazy=True`; `AttributeError` if absent |
| `print(track)` | Track ID, species, and available dataset names |

**Attributes:** `file_path`, `track_id`, `species`, `lazy`.
**Caching:** array reads are not cached; assign arrays you reuse to variables.

With an initialized [`DHybridrpy`](dhybridrpy.md) instance named `dpy`:

```python
track_id = dpy.tracks(species=1)[0]
track = dpy.track(track_id, species=1)
t, x = track.t, track.x1
```

### Datasets

One value per recorded sample. Suffixes 1, 2, and 3 mean x, y, and z.

| Dataset | Meaning | Units |
|---|---|---|
| `x1`, `x2`, `x3` | Position; only simulated dimensions are stored | `d_i` |
| `v1`, `v2`, `v3` | Ordinary velocity | `v_A` |
| `B1`, `B2`, `B3` | Magnetic field sampled at the particle, when enabled | `B_0` |
| `E1`, `E2`, `E3` | Electric field sampled at the particle, when enabled | `B_0 * v_A` |
| `t` | Simulation time | `1/omega_ci` |
| `n` | Simulation iteration number | — |
| `ene` | Kinetic energy per physical particle | `m_p * v_A**2` |
| `q` | Signed macroparticle charge weight | `e` |

Sample times (`t`), iterations (`n`), and array lengths can differ by track.

## TrackCollection

`TrackCollection(file_path, species, lazy=False)`

| Argument | Meaning |
|---|---|
| `file_path` | Track file, e.g. `Output/Tracks/Sp01/track_Sp01.h5` |
| `species` | Species number |
| `lazy` | Return Dask arrays when `True`; default `False` |

**Attributes:** `file_path`, `species`, `lazy`.

| Access | Result |
|---|---|
| `collection.track_ids` | NumPy array of `"rank-tag"` IDs sorted numerically by rank and tag |
| `len(collection)` | Number of tracks |
| `for track in collection` | Iterate over `Track` objects in ID order |
| `collection[track_id]` | `Track`; `KeyError` if absent |
| `collection.load_dataset(key, track_ids=None)` | `{track_id: array}` for dataset `key`; all tracks when IDs are omitted |
| `collection.close()` | Release the shared HDF5 handle; later reads reopen it |
| `with TrackCollection(...) as collection:` | Close the shared file handle on exit |

**Bulk reads:** arrays may differ in length; NumPy by default, Dask when
`lazy=True`. A missing track or dataset raises `KeyError`.

**Lazy reads:** arrays remain usable after closing the collection.

```python
from dhybridrpy import TrackCollection

with TrackCollection("Output/Tracks/Sp01/track_Sp01.h5", species=1) as collection:
    ids = collection.track_ids[:3]
    energies = collection.load_dataset("ene", track_ids=ids)
    times = collection.load_dataset("t", track_ids=ids)
```
