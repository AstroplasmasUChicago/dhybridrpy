# dplot

Use `dplot` to save a PNG for each timestep of a 1D or 2D field or phase,
and optionally combine the frames into an MP4. It is installed with
dhybridrpy. For 3D data, use [interactive slices in Python](plotting.md).

## Find available data

From the simulation directory:

```bash
dplot -i input/input -o Output
```

This lists fields, phases, and the timestep range. Available names come
from the first saved field or phase timestep after timestep 0.

All commands below use the default paths `input/input` and `Output`. Add
`-i path/to/input -o path/to/Output` for a different run.

## Plot fields

Repeat `--fields` to select multiple quantities:

```bash
dplot --fields Bx --fields By --type Total
```

Use `--all-fields` for every field of the selected type. Field types are
`Total` (default), `External`, and `Self`.

For components relative to the z axis:

```bash
dplot --fields Bparallel --fields Bperp --parallel-axis z
```

Here `Bparallel = Bz` and `Bperp = sqrt(Bx**2 + By**2)`. The default parallel
axis is x. These are derived from the selected field type.

## Plot phases

```bash
dplot --phases u1x1 --species 1
```

`u1x1` has x position on the horizontal axis and x proper velocity on the
vertical axis. Choose names from the discovery output, or use
`--all-phases`. Repeat `--phases` or `--species` for multiple selections;
omitting `--species` includes all numbered species.

## Crop the data

```bash
dplot --fields Bx --x-range 0.1 0.9 --y-range 0.1 0.9
```

This keeps the middle 80% of each axis, excluding the edges from the plots
and automatic color limits. Each pair must satisfy `0 <= low < high <= 1`.
For `u1x1`, `--y-range` crops the proper-velocity axis.

## Set a shared color scale

Each field or phase/species selection uses one color scale across time.
By default, its limits are the 2nd and 98th percentiles from up to 10
sampled timesteps. Adjust them with `--pmin` and `--pmax`, or set explicit limits:

```bash
dplot --fields Bx --vmin -1 --vmax 1 --colormap RdBu_r
```

Setting one limit leaves the other automatic; setting both skips the scan.

## Choose output files

PNGs are saved under `plots/Fields/Bx/` or, for example,
`plots/Phases/u1x1_Sp01/`. Use `--plots-dir` to change the destination.

Add `--video` to create an MP4 from the frames. This requires `ffmpeg`
on your `PATH`:

```bash
dplot --fields Bx --video --fps 15
```

The movie is saved as `plots/Fields/Bx/Bx.mp4`; PNGs are kept.
Choose separate destinations when comparing field types or parallel axes,
since those settings are not included in filenames.

## Adjust rendering

| Option | Use |
| --- | --- |
| `-j 4` | Render with four processes; `-j -1` uses all CPU cores |
| `--dpi 300` | Increase PNG resolution from the default 150 DPI |
| `-v` | Show progress details |

Large 2D arrays are downsampled for rendering; color limits use the data
before downsampling. Run `dplot --help` for all options and defaults.
