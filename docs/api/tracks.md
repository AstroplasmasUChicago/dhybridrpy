# Tracks API

Classes for accessing particle track data from dHybridR simulations.

## Track

Represents a single particle track across all timesteps.

### Class Definition

```python
class Track(
    file_path: str,
    group_name: str,
    track_id: str,
    species: int,
    lazy: bool = False
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `track_id` | `str` | Particle identifier in format `'rank-tag'` |
| `species` | `int` | Species number |
| `lazy` | `bool` | Whether lazy loading is enabled |

### Properties

All properties return `np.ndarray` (or `dask.array.Array` if lazy loading is enabled).

#### Position

| Property | Type | Description |
|----------|------|-------------|
| `x1` | `np.ndarray` | X coordinate over time |
| `x2` | `np.ndarray` | Y coordinate over time |
| `x3` | `np.ndarray` | Z coordinate over time |

#### Momentum

| Property | Type | Description |
|----------|------|-------------|
| `p1` | `np.ndarray` | X momentum over time |
| `p2` | `np.ndarray` | Y momentum over time |
| `p3` | `np.ndarray` | Z momentum over time |

#### Time

| Property | Type | Description |
|----------|------|-------------|
| `t` | `np.ndarray` | Time array |

### Example

```python
track = dpy.track('0-1465')

# Get trajectory
x, y, z = track.x1, track.x2, track.x3
```

---

## TrackCollection

Collection of all tracks for a given species. Supports iteration and indexing.

### Class Definition

```python
class TrackCollection(
    file_path: str,
    species: int,
    lazy: bool = False
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
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