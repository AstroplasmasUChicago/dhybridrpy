# Tracks API

Classes for accessing particle track data from dHybridR simulations.

## Track

Represents a single particle track across all timesteps. Obtained via
[`dpy.track(track_id, species=...)`](#tracktrack_id-str-species-int--1---track).

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to the track HDF5 file |
| `track_id` | `str` | Particle identifier in format `'rank-tag'` |
| `species` | `int` | Species number |
| `lazy` | `bool` | Whether lazy loading is enabled |

### Dataset Access

Every dataset in the track file is available as an attribute of the same
name. Reads return `np.ndarray` (or `dask.array.Array` if lazy loading
is enabled). Accessing a name that is not a dataset in the file raises
`AttributeError`. Exactly which datasets exist depends on the run; print
a track to list them:

```python
track = dpy.track('0-1465')
print(track)
# Track (track_id=0-1465, species=1):
#   B1, B2, B3, E1, E2, E3, ene, n, q, t, v1, v2, v3, x1, x2
```

The datasets below are the ones dHybridR typically writes. The suffixes
1, 2, 3 correspond to the x, y, z directions.

#### Position

| Dataset | Description |
|---------|-------------|
| `x1` | X coordinate over time |
| `x2` | Y coordinate over time |
| `x3` | Z coordinate over time |

#### Velocity

Velocities are stored in units of the Alfven speed. These are regular
velocities, not momenta or proper velocities.

| Dataset | Description |
|---------|-------------|
| `v1` | X velocity over time |
| `v2` | Y velocity over time |
| `v3` | Z velocity over time |

#### Electromagnetic Fields

| Dataset | Description |
|---------|-------------|
| `B1` | X magnetic field at particle position over time |
| `B2` | Y magnetic field at particle position over time |
| `B3` | Z magnetic field at particle position over time |
| `E1` | X electric field at particle position over time |
| `E2` | Y electric field at particle position over time |
| `E3` | Z electric field at particle position over time |

#### Other

| Dataset | Description |
|---------|-------------|
| `t` | Simulation time |
| `n` | Iteration number at which each value was stored (e.g., `[10, 20, 30, ...]`) |
| `ene` | Particle energy over time |
| `q` | Particle charge |

!!! note
    The `n` array reflects the simulation iteration numbers at which data was recorded, controlled by the `track_nstore` parameter in the input file. For example, `track_nstore=10` means values are stored every 10 iterations.

### Example

```python
track = dpy.track('0-1465')

# Get trajectory
x, y = track.x1, track.x2

# Get velocity components
vx, vy, vz = track.v1, track.v2, track.v3

# Get fields at particle position over time
Bx, By, Bz = track.B1, track.B2, track.B3

# Get time and energy
t = track.t
ene = track.ene

# Get iteration numbers
n = track.n  # e.g., [10, 20, 30, ...]
```

---

## TrackCollection

Collection of all tracks for a given species. Discovered and constructed
automatically by `DHybridrpy` when track files are present in the output folder.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to the track HDF5 file |
| `species` | `int` | Species number |
| `lazy` | `bool` | Whether lazy loading is enabled |
| `track_ids` | `np.ndarray` | Array of all track IDs |

### Methods

#### `__getitem__(track_id: str) -> Track`

Get a track by its ID.

**Parameters:**

- `track_id` (`str`): Track ID in format `'rank-tag'`

**Returns:** `Track` object

**Raises:** `KeyError` if track ID not found

**Example:**

```python
track = dpy.track('0-1465')
```

#### `__len__()` / `__iter__()`

The number of tracks, and iteration over all `Track` objects in
rank-tag order:

```python
for track in collection:
    print(track.track_id, track.ene.max())
```

#### `load_dataset(key: str, track_ids=None) -> dict`

Load one dataset for many tracks in a single pass over the file. Much
faster than reading track by track.

**Parameters:**

- `key` (`str`): Dataset name, e.g. `'x1'` or `'ene'`
- `track_ids`: Iterable of track IDs; all tracks when `None`

**Returns:** `{track_id: array}`. Tracks may have different lengths, so
values are per track rather than stacked.

**Raises:** `KeyError` if a track ID or the dataset is not found

```python
energies = collection.load_dataset('ene')
```

#### `close()`

Reads share one open file handle (opened on first use, reopened if the
file is replaced). `close()` releases it, which is needed before
deleting or re-creating the file; reads reopen it automatically. `TrackCollection`
also works as a context manager.

## DHybridrpy Track Methods

Methods on the main `DHybridrpy` class for accessing tracks.

### `track(track_id: str, species: int = 1) -> Track`

Access a particle track by its ID.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `track_id` | `str` | required | Track ID in format `'rank-tag'` |
| `species` | `int` | `1` | Species number |

**Returns:** `Track` object

**Raises:** `ValueError` if species not found or no track data exists

**Example:**

```python
track = dpy.track('0-1465')
track_sp2 = dpy.track('0-100', species=2)
```

---

### `tracks(species: int = 1) -> np.ndarray`

Retrieve an array of track IDs for a given species.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `species` | `int` | `1` | Species number |

**Returns:** `np.ndarray` of track ID strings

**Raises:** `ValueError` if species not found or no track data exists

**Example:**

```python
# Get all track IDs
track_ids = dpy.tracks()
print(track_ids)  # ['0-1', '0-2', '0-3', ...]

# Get first and last track
first_id = dpy.tracks()[0]
last_id = dpy.tracks()[-1]
```