# Timestep

Container class for accessing field, phase, and raw data at a specific timestep.

## Class Definition

```python
class Timestep(timestep: int)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `timestep` | `int` | The timestep number |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `timestep` | `int` | The timestep number |
| `fields` | `Container` | Access to field data |
| `phases` | `Container` | Access to phase data |
| `raw_files` | `Container` | Access to raw particle data |

## Accessing Fields

The `fields` attribute provides dynamic access to field data:

```python
ts = dpy.timestep(1)

# Access fields by name
Bx = ts.fields.Bx()
By = ts.fields.By()
Bz = ts.fields.Bz()
Ex = ts.fields.Ex()
Jx = ts.fields.Jx()

# Access magnitude
Bmagnitude = ts.fields.Bmagnitude()
```

### Field Types

Fields can be accessed by type using the `type` keyword argument:

```python
# Total field (default)
Bx_total = ts.fields.Bx()
Bx_total = ts.fields.Bx(type="Total")

# External field
Bx_ext = ts.fields.Bx(type="External")

# Self-consistent field
Bx_self = ts.fields.Bx(type="Self")
```

### Available Field Types

| Type | Description |
|------|-------------|
| `"Total"` | Complete field (default) |
| `"External"` | Externally applied field |
| `"Self"` | Self-consistent field from particles |

## Accessing Phases

The `phases` attribute provides dynamic access to phase space data:

```python
ts = dpy.timestep(1)

# Distribution functions
x2x1 = ts.phases.x2x1()  # x-y distribution
p1x1 = ts.phases.p1x1()  # x-px phase space
p2p1 = ts.phases.p2p1()  # px-py velocity space

# Fluid quantities
Vx = ts.phases.Vx()  # Bulk velocity x
Vy = ts.phases.Vy()  # Bulk velocity y
Vz = ts.phases.Vz()  # Bulk velocity z

# Pressure
P = ts.phases.P()     # Scalar pressure
Pxx = ts.phases.Pxx() # Pressure tensor component
```

### Species Selection

Phase data can be accessed by species using the `species` keyword argument:

```python
# Species 1 (default)
phase_s1 = ts.phases.x2x1()
phase_s1 = ts.phases.x2x1(species=1)

# Species 2
phase_s2 = ts.phases.x2x1(species=2)

# Total (all species)
phase_total = ts.phases.x2x1(species="Total")
```

## Accessing Raw Files

The `raw_files` attribute provides access to raw particle data:

```python
ts = dpy.timestep(1)

# Access raw data for species 1 (default)
raw = ts.raw_files.raw()
raw = ts.raw_files.raw(species=1)

# Access raw data for species 2
raw_s2 = ts.raw_files.raw(species=2)

# Get the data dictionary
data_dict = raw.dict
print(data_dict.keys())
```

## Inspecting Available Data

Print the containers to see what data is available:

```python
ts = dpy.timestep(1)

# View all available fields
print(ts.fields)
# Output:
# Fields at timestep 1:
#   type = Total: Bx, By, Bz, Bmagnitude, Ex, Ey, Ez, ...
#   type = External: Bx, By, Bz, ...
#   type = Self: Bx, By, Bz, ...

# View all available phases
print(ts.phases)
# Output:
# Phases at timestep 1:
#   species = 1: x2x1, p1x1, Vx, Vy, Vz, P, ...
#   species = 2: x2x1, p1x1, Vx, Vy, Vz, P, ...

# View all available raw files
print(ts.raw_files)
# Output:
# Raw files at timestep 1:
#   species = 1: raw
#   species = 2: raw

# View everything at once
print(ts)
```

## Error Handling

### Missing Data

```python
try:
    # Requesting non-existent field
    Bw = ts.fields.Bw()
except AttributeError as e:
    print(f"Field not found: {e}")

try:
    # Requesting non-existent species
    phase = ts.phases.x2x1(species=99)
except AttributeError as e:
    print(f"Species not found: {e}")
```

### Invalid Arguments

```python
try:
    # Too many arguments
    Bx = ts.fields.Bx("Total", "Extra")
except TypeError as e:
    print(f"Error: {e}")

try:
    # Wrong keyword argument
    Bx = ts.fields.Bx(species=1)  # Should be 'type'
except TypeError as e:
    print(f"Error: {e}")
```