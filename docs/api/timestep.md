# Timestep

Diagnostics at one iteration, returned by the
[DHybridrpy timestep methods](dhybridrpy.md#select-a-timestep).
Examples use an initialized `DHybridrpy` instance named `dpy`.

```python
ts = dpy.timestep_index(-1)
print(ts)  # Available names, field types, and species
```

## Attributes and access

| Access | Returns | Selection |
| --- | --- | --- |
| `ts.timestep` | Iteration number | — |
| `ts.fields.Bx(type="Total")` | [Field](data.md#field-and-phase) | `"Total"` (default), `"External"`, or `"Self"` |
| `ts.phases.x3x2x1(species=1)` | [Phase](data.md#field-and-phase) | Species number (default `1`) or `"Total"` |
| `ts.raw_files.raw(species=1)` | [Raw](data.md#raw) | Species number; default `1` |

Names are case-sensitive and match the saved output. Print `ts.fields`,
`ts.phases`, or `ts.raw_files` to list one container. See
[quantity names and meanings](../user-guide/working-with-data.md#read-a-physical-quantity).

## Selection rules

- Field type, species, and quantity must exist at the selected iteration.
- `species="Total"` reads a saved total; it does not sum species.
- Magnitude accessors read saved files; they do not calculate magnitudes.
- Legacy names such as `p1x1` remain unchanged.

**Examples:** use names and selections present in the listing.

```python
bx = ts.fields.Bx()
acceleration = ts.fields.Accelx(type="External")
pressure = ts.phases.Pmagnitude(species=1)
name = "u1x1"
phase = getattr(ts.phases, name)(species=1)
```

## Raw iterations

Use `raw_timesteps()` when raw output has a separate output interval:

```python
raw_ts = dpy.timestep(dpy.raw_timesteps()[-1])
raw = raw_ts.raw_files.raw(species=1)
```

## Lookup errors

| Exception | Cause |
| --- | --- |
| `AttributeError` | Missing name, field type, or species |
| `TypeError` | Unsupported keyword or more than one selection argument |
