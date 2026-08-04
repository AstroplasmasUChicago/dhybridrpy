# CLI Tool: dplot

`dplot` is a command-line tool that ships with dhybridrpy for visualizing dHybridR simulation fields and phases across all timesteps. It saves PNG images for every timestep and can optionally combine them into MP4 videos.

## Installation

`dplot` is included automatically when you install dhybridrpy:

```bash
pip install dhybridrpy
```

No additional setup is needed. For video output, you also need `ffmpeg` installed on your system (see [Requirements](#requirements)).

## Discovery Mode

Run `dplot` without specifying any fields or phases to see what data is available in your simulation:

```bash
dplot -i path/to/input
```

This prints all available fields (grouped by type), phases (grouped by species), and the number of timesteps:

```
Fields:
  type=Total: Bx, By, Bz, Ex, Ey, Ez, Jx, Jy, Jz
  type=External: Bx, By, Bz
  type=Self: Bx, By, Bz

Phases:
  species=1: p1x1, p2x1, x2x1
  species=2: p1x1, p2x1, x2x1

Timesteps: 50 (1 to 50)
```

## Plotting Fields

Use `--fields` to plot specific field components. Repeat the flag for multiple fields:

```bash
# Plot a single field
dplot -i path/to/input --fields Bx

# Plot multiple fields
dplot -i path/to/input --fields Bx --fields By --fields Bz
```

By default, `Total` fields are plotted. Use `--type` to select a different field type:

```bash
dplot -i path/to/input --fields Bx --type External
dplot -i path/to/input --fields Bx --type Self
```

### Plot All Fields

Use `--all-fields` to plot every field component available for the selected type:

```bash
dplot -i path/to/input --all-fields
dplot -i path/to/input --all-fields --type Self
```

## Bparallel and Bperp

In addition to the raw components (`Bx`, `By`, `Bz`), `--fields` accepts two derived magnetic-field quantities: `Bparallel` (the component along a chosen axis) and `Bperp` (the quadrature sum of the other two components). Use `--parallel-axis` to choose which axis is "parallel" (default: `x`); `Bperp` is derived automatically from whichever two axes remain:

```bash
# Parallel/perp relative to a guide field along x (the default)
dplot -i path/to/input --fields Bparallel --fields Bperp

# Guide field along z instead
dplot -i path/to/input --fields Bparallel --fields Bperp --parallel-axis z
```

`--type` still applies, so `--fields Bparallel --type Self` decomposes the self-generated field instead of `Total`.

## Plotting Phases

Use `--phases` to plot phase-space distributions. Use `--species` to select which species to include:

```bash
# Plot a phase for all available species
dplot -i path/to/input --phases p1x1

# Plot a phase for specific species
dplot -i path/to/input --phases p1x1 --species 1 --species 3

# Plot multiple phases
dplot -i path/to/input --phases p1x1 --phases x2x1 --species 2
```

If `--species` is not specified, all available species are plotted.

### Plot All Phases

Use `--all-phases` to plot every available phase-space distribution:

```bash
dplot -i path/to/input --all-phases
dplot -i path/to/input --all-phases --species 1
```

## Cropping the Box

Use `--x-range` and `--y-range` to keep only a fractional sub-region of the simulation box (each takes a `low high` pair between 0 and 1). This is useful for excluding domain-edge artifacts, such as absorbing boundaries, that would otherwise skew the color scale or dominate the plot:

```bash
# Drop the outer 10% on each side of both axes
dplot -i path/to/input --fields Bx --x-range 0.1 0.9 --y-range 0.1 0.9
```

The crop is applied before the color-scale percentile scan and before rendering, so excluded cells never influence `vmin`/`vmax` either. It works for both `--fields` and `--phases` (including `Bparallel`/`Bperp`), though for phase-space plots the second axis is often momentum rather than physical `y`, so check the axis labels in the output before relying on `--y-range` there.

## Creating Videos

Add `--video` to combine the saved PNGs into an MP4 video using ffmpeg:

```bash
dplot -i path/to/input --fields Bx --video
dplot -i path/to/input --all-fields --video --fps 15
```

The video is saved alongside the PNGs in each field/phase subdirectory.

## Output Directory Structure

By default, plots are saved under a `plots/` directory. The structure mirrors the data hierarchy:

```
plots/
├── Fields/
│   ├── Bx/
│   │   ├── Bx_00000001.png
│   │   ├── Bx_00000002.png
│   │   ├── ...
│   │   └── Bx.mp4          # if --video is used
│   └── By/
│       ├── By_00000001.png
│       └── ...
└── Phases/
    ├── p1x1_Sp01/
    │   ├── p1x1_Sp01_00000001.png
    │   └── ...
    └── p1x1_Sp03/
        ├── p1x1_Sp03_00000001.png
        └── ...
```

Use `--plots-dir` to change the base output directory:

```bash
dplot -i path/to/input --fields Bx --plots-dir my_plots
```

## Color Scale

`dplot` automatically determines a consistent color range across all timesteps by sampling ~10 equally spaced frames and taking the 2nd and 98th percentiles of the sampled values. This clips extreme outliers so features stay visible. You can override the percentiles with `--pmin` / `--pmax`, or set explicit limits with `--vmin` / `--vmax`:

```bash
# Default behavior: vmin = 2nd percentile, vmax = 98th percentile
dplot -i path/to/input --fields Bx

# Use tighter percentiles
dplot -i path/to/input --fields Bx --pmin 5 --pmax 95

# Set both limits explicitly (skips the percentile scan)
dplot -i path/to/input --fields Bx --vmin -1.0 --vmax 1.0

# Set only one limit (the other is auto-computed from the percentile)
dplot -i path/to/input --fields Bx --vmin 0
```

## Parallel Rendering

Use `-j` to render frames in parallel across multiple processes:

```bash
# Use 4 processes
dplot -i path/to/input --all-fields -j 4

# Use all available CPU cores
dplot -i path/to/input --all-fields -j -1
```

By default, dplot runs sequentially (`-j 1`). In parallel mode, data is loaded in the main process and rendering is distributed across worker processes.

## Large Datasets

For high-resolution simulations (e.g., 15000x3000 grids), dplot automatically downsamples data to fit within 1920x1080 pixels before rendering. This preserves the aspect ratio and keeps plotting fast. Downsampling is applied both in sequential and parallel modes.

Use `--verbose` to see downsampling details:

```bash
dplot -i path/to/input --fields Bx --verbose
#     Data (15000, 3000) → (2143, 429) (downsampled)
```

## Verbose Mode

Use `--verbose` / `-v` to print detailed information about what dplot is doing, including downsampling and parallel worker activity:

```bash
dplot -i path/to/input --fields Bx -v
dplot -i path/to/input --all-fields -j 4 -v
```

## CLI Options Reference

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-i` | `input/input` | Path to the dHybridR input file. |
| `--output` | `-o` | `Output` | Path to the simulation output folder. |
| `--fields` | | | Field names to plot. Repeat for multiple (e.g. `--fields Bx --fields By`). |
| `--phases` | | | Phase names to plot. Repeat for multiple (e.g. `--phases p1x1 --phases x2x1`). |
| `--all-fields` | | `False` | Plot all available fields for the selected type. |
| `--all-phases` | | `False` | Plot all available phase-space distributions. |
| `--type` | | `Total` | Field type: `Total`, `External`, or `Self`. |
| `--species` | | all | Species numbers for phases. Repeat for multiple (e.g. `--species 1 --species 3`). |
| `--jobs` | `-j` | `1` | Number of parallel rendering processes. Use `-1` for all cores. |
| `--verbose` | `-v` | `False` | Print verbose output (downsampling info, parallel progress). |
| `--video` | | `False` | Create an MP4 video from the saved PNGs using ffmpeg. |
| `--fps` | | `10` | Video framerate (frames per second). |
| `--colormap` | `-c` | `viridis` | Matplotlib colormap name. |
| `--dpi` | | `150` | Plot resolution in dots per inch. |
| `--vmin` | | auto | Fixed color scale minimum (overrides `--pmin`). |
| `--vmax` | | auto | Fixed color scale maximum (overrides `--pmax`). |
| `--pmin` | | `2.0` | Lower percentile used to estimate `vmin` during the initial scan. |
| `--pmax` | | `98.0` | Upper percentile used to estimate `vmax` during the initial scan. |
| `--plots-dir` | | `plots` | Base output directory for saved plots. |
| `--x-range` | | full box | Keep only this `low high` fraction (0-1) of the box along x. |
| `--y-range` | | full box | Keep only this `low high` fraction (0-1) of the box along y. |
| `--parallel-axis` | | `x` | Axis (`x`, `y`, or `z`) treated as parallel for `--fields Bparallel`/`Bperp`. |

## Examples

### Full workflow: discover, plot, and create video

```bash
# 1. See what's available
dplot -i sim/input

# 2. Plot magnetic field components with a diverging colormap
dplot -i sim/input --fields Bx --fields By --fields Bz \
    --colormap RdBu --dpi 200

# 3. Plot all phases for species 1 with video
dplot -i sim/input --all-phases --species 1 --video --fps 15

# 4. Plot everything with videos
dplot -i sim/input --all-fields --all-phases --video
```

### Custom output and input paths

```bash
dplot -i experiments/run42/input -o experiments/run42/Output \
    --fields Bx --plots-dir experiments/run42/figures
```

## Requirements

- **Python packages**: All dependencies are installed with dhybridrpy (matplotlib, numpy, typer).
- **ffmpeg**: Required only for `--video`. Install via your system package manager:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Conda
conda install ffmpeg
```
