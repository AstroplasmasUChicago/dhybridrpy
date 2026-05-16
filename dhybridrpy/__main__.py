import os
import re
import sys
import subprocess
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer
from joblib import Parallel, delayed

from .dhybridrpy import DHybridrpy
from .data import Data

MAX_W = 1920
MAX_H = 1080

app = typer.Typer(add_completion=False)


def print_progress(label: str, current: int, total: int) -> None:
    pct = current / total * 100
    sys.stderr.write(f"\r{label}: {current}/{total} ({pct:.0f}%)")
    sys.stderr.flush()
    if current == total:
        sys.stderr.write("\n")


_TITLE_TIME_RE = re.compile(r"(at time )([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def shorten_title_time(title: str, ndecimals: int = 1) -> str:
    """Round the 'at time <N>' value inside a plot title to `ndecimals` decimals."""
    return _TITLE_TIME_RE.sub(
        lambda m: f"{m.group(1)}{float(m.group(2)):.{ndecimals}f}",
        title,
        count=1,
    )


def compute_vlim(dpy, get_data, timesteps, pmin=2.0, pmax=98.0):
    """Sample ~10 equally spaced timesteps to estimate global vmin/vmax via percentiles."""
    indices = np.unique(
        np.linspace(0, len(timesteps) - 1, min(10, len(timesteps)), dtype=int)
    )
    sampled = timesteps[indices]
    chunks = []
    for i, ts_num in enumerate(sampled):
        try:
            data = get_data(dpy.timestep(ts_num)).data
        except (AttributeError, ValueError, OSError):
            continue
        try:
            if hasattr(data, "compute"):
                data = data.compute()
            chunks.append(np.asarray(data).ravel())
        except OSError:
            continue
        print_progress("Scanning color range", i + 1, len(sampled))
    if not chunks:
        typer.echo(
            "  Warning: could not sample any timesteps for color range; "
            "falling back to per-frame autoscale.",
            err=True,
        )
        return None, None
    combined = np.concatenate(chunks)
    lo, hi = np.percentile(combined, [pmin, pmax])
    return float(lo), float(hi)


def downsample(data, xdata, ydata):
    """Stride-sample 2D data to fit within MAX_W x MAX_H pixels, preserving aspect ratio."""
    step = max(1, max(data.shape[0] // MAX_W, data.shape[1] // MAX_H))
    if step > 1:
        return data[::step, ::step], xdata[::step], ydata[::step]
    return data, xdata, ydata


def vecho(msg: str) -> None:
    """Print only when verbose mode is active."""
    if getattr(app, "state", {}).get("verbose", False):
        typer.echo(msg)


def plot_frame(ax, data_obj, colormap, vmin, vmax):
    """Plot a single frame, downsampling large 2D data."""
    data = data_obj.data
    if hasattr(data, "compute"):
        data = data.compute()

    ndim = data.ndim
    orig_shape = data.shape
    if ndim == 1 or (ndim == 2 and max(data.shape) <= MAX_W):
        vecho(f"    Data {orig_shape} — no downsampling needed")
        data_obj.plot(ax=ax, colormap=colormap, vmin=vmin, vmax=vmax)
        return

    # Large 2D data — downsample and render manually
    xdata = data_obj.xdata
    ydata = data_obj.ydata
    if hasattr(xdata, "compute"):
        xdata = xdata.compute()
    if hasattr(ydata, "compute"):
        ydata = ydata.compute()

    data, xdata, ydata = downsample(data, xdata, ydata)
    vecho(f"    Data {orig_shape} → {data.shape} (downsampled)")

    X, Y = np.meshgrid(xdata, ydata, indexing="ij")
    mesh = ax.pcolormesh(
        X, Y, data, cmap=colormap, shading="auto", vmin=vmin, vmax=vmax
    )
    ax.set_title(data_obj._plot_title)
    xlabel, ylabel = Data._LABEL_MAPPINGS[data_obj.name]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(data_obj.xlimdata)
    ax.set_ylim(data_obj.ylimdata)
    plt.colorbar(mesh, ax=ax, label=data_obj.name)


def make_video(plot_dir: str, name: str, fps: int) -> None:
    """Combine PNGs in a directory into an MP4 using ffmpeg."""
    video_path = os.path.join(plot_dir, f"{name}.mp4")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            os.path.join(plot_dir, "*.png"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        typer.echo(f"ffmpeg error: {result.stderr}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Video saved: {video_path}")


def print_available(dpy: DHybridrpy) -> None:
    """Print available fields, phases, and timestep info."""
    first_ts = dpy.timestep(dpy.timesteps()[0])

    # Fields grouped by available types
    typer.echo("Fields:")
    types_available = [
        t for t in ["Total", "External", "Self"] if first_ts._fields_dict.get(t)
    ]
    for field_type in types_available:
        names = sorted(first_ts._fields_dict[field_type].keys())
        if names:
            typer.echo(f"  type={field_type}: {', '.join(names)}")

    # Phases grouped by species
    typer.echo("\nPhases:")
    for sp in sorted(k for k in first_ts._phases_dict.keys() if isinstance(k, int)):
        names = sorted(first_ts._phases_dict[sp].keys())
        if names:
            typer.echo(f"  species={sp}: {', '.join(names)}")
    if "Total" in first_ts._phases_dict:
        names = sorted(first_ts._phases_dict["Total"].keys())
        if names:
            typer.echo(f"  species=Total: {', '.join(names)}")

    # Timestep info
    timesteps = dpy.timesteps()
    typer.echo(f"\nTimesteps: {len(timesteps)} ({timesteps[0]} to {timesteps[-1]})")


def _render_one_frame(frame_data, label, plot_dir, colormap, dpi, vmin, vmax):
    """Render and save a single frame from pre-extracted data. Called by joblib."""
    data, xdata, ydata, xlim, ylim, name, title, ts_num = frame_data
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

        if data.ndim == 1:
            ax.plot(xdata, data)
            ax.set_xlabel("$x$")
            ax.set_ylabel(name)
            ax.set_xlim(xlim)
        else:
            X, Y = np.meshgrid(xdata, ydata, indexing="ij")
            mesh = ax.pcolormesh(
                X, Y, data, cmap=colormap, shading="auto", vmin=vmin, vmax=vmax
            )
            xlabel, ylabel = Data._LABEL_MAPPINGS[name]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.colorbar(mesh, ax=ax, label=name)

        ax.set_title(title)
        filename = f"{label}_{ts_num:08d}.png"
        fig.savefig(os.path.join(plot_dir, filename), bbox_inches="tight")
    except OSError:
        pass
    finally:
        if fig is not None:
            plt.close(fig)


def _extract_frame_data(dpy, get_data, ts_num):
    """Extract picklable numpy arrays from a data object. Runs in main thread."""
    try:
        data_obj = get_data(dpy.timestep(ts_num))
    except (AttributeError, ValueError, OSError):
        return None

    data_obj._plot_title = shorten_title_time(data_obj._plot_title)
    data = data_obj.data
    if hasattr(data, "compute"):
        data = data.compute()

    xdata = data_obj.xdata
    if hasattr(xdata, "compute"):
        xdata = xdata.compute()

    ydata = None
    ylim = None
    if data.ndim >= 2:
        ydata = data_obj.ydata
        if hasattr(ydata, "compute"):
            ydata = ydata.compute()
        ylim = data_obj.ylimdata
        # Downsample immediately to reduce memory footprint
        data, xdata, ydata = downsample(data, xdata, ydata)

    result = (
        data,
        xdata,
        ydata,
        data_obj.xlimdata,
        ylim,
        data_obj.name,
        data_obj._plot_title,
        ts_num,
    )

    # Clear cached HDF5 data to prevent memory leak in the main process
    data_obj._data_dict.clear()

    return result


def plot_data_series(
    dpy: DHybridrpy,
    get_data,
    label: str,
    plot_dir: str,
    colormap: str,
    dpi: int,
    vmin: Optional[float],
    vmax: Optional[float],
    video: bool,
    fps: int,
    jobs: int = 1,
    pmin: float = 2.0,
    pmax: float = 98.0,
) -> None:
    """Plot a data series across all timesteps and optionally create video."""
    os.makedirs(plot_dir, exist_ok=True)
    timesteps = dpy.timesteps()

    # Compute global color limits if not provided
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = compute_vlim(dpy, get_data, timesteps, pmin, pmax)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax
        if vmin is not None and vmax is not None:
            typer.echo(
                f"  Color range: [{vmin:.4g}, {vmax:.4g}] "
                f"(percentiles {pmin:g}/{pmax:g})"
            )

    if jobs == 1:
        # Sequential mode — no overhead from data extraction or process spawning
        for i, ts_num in enumerate(timesteps):
            try:
                data_obj = get_data(dpy.timestep(ts_num))
            except (AttributeError, ValueError, OSError):
                print_progress(f"  Plotting {label}", i + 1, len(timesteps))
                continue
            data_obj._plot_title = shorten_title_time(data_obj._plot_title)
            fig = None
            try:
                fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
                plot_frame(ax, data_obj, colormap, vmin, vmax)
                filename = f"{label}_{ts_num:08d}.png"
                fig.savefig(os.path.join(plot_dir, filename), bbox_inches="tight")
            except OSError:
                pass
            finally:
                if fig is not None:
                    plt.close(fig)
                data_obj._data_dict.clear()
            print_progress(f"  Plotting {label}", i + 1, len(timesteps))
    else:
        # Parallel mode — generator feeds frames on demand, joblib manages backpressure
        verbose_level = 10 if getattr(app, "state", {}).get("verbose", False) else 0

        def _load_frames():
            for ts_num in timesteps:
                frame = _extract_frame_data(dpy, get_data, ts_num)
                if frame is not None:
                    yield frame

        Parallel(n_jobs=jobs, verbose=verbose_level)(
            delayed(_render_one_frame)(
                frame, label, plot_dir, colormap, dpi, vmin, vmax
            )
            for frame in _load_frames()
        )

    typer.echo(f"  Saved {len(timesteps)} frames to {plot_dir}")

    # Create video if requested
    if video:
        make_video(plot_dir, label, fps)


@app.command()
def run(
    input: str = typer.Option(
        "input/input", "-i", "--input", help="Path to input file."
    ),
    output: str = typer.Option(
        "Output", "-o", "--output", help="Path to output folder."
    ),
    all_fields: bool = typer.Option(
        False, "--all-fields", help="Plot all available fields."
    ),
    all_phases: bool = typer.Option(
        False, "--all-phases", help="Plot all available phases."
    ),
    fields: Optional[List[str]] = typer.Option(
        None, "--fields", help="Field names to plot (e.g. --fields Bx --fields By)."
    ),
    phases: Optional[List[str]] = typer.Option(
        None, "--phases", help="Phase names to plot (e.g. --phases p1x1 --phases x2x1)."
    ),
    field_type: str = typer.Option(
        "Total", "--type", help="Field type: Total, External, or Self."
    ),
    species: Optional[List[int]] = typer.Option(
        None, "--species", help="Species numbers for phases (default: all available)."
    ),
    jobs: int = typer.Option(
        1, "-j", "--jobs", help="Number of parallel processes (-1 = all cores)."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print verbose output."
    ),
    video: bool = typer.Option(
        False, "--video", help="Create MP4 video from saved PNGs using ffmpeg."
    ),
    fps: int = typer.Option(10, "--fps", help="Video framerate."),
    colormap: str = typer.Option(
        "viridis", "-c", "--colormap", help="Matplotlib colormap."
    ),
    dpi: int = typer.Option(150, "--dpi", help="Plot resolution."),
    vmin: Optional[float] = typer.Option(
        None, "--vmin", help="Fixed color scale minimum."
    ),
    vmax: Optional[float] = typer.Option(
        None, "--vmax", help="Fixed color scale maximum."
    ),
    pmin: float = typer.Option(
        2.0, "--pmin", help="Lower percentile for auto vmin during scan."
    ),
    pmax: float = typer.Option(
        98.0, "--pmax", help="Upper percentile for auto vmax during scan."
    ),
    plots_dir: str = typer.Option(
        "plots", "--plots-dir", help="Base output directory for plots."
    ),
) -> None:
    """
    dplot: Visualize dHybridR simulation fields and phases.

    Run without --fields or --phases to see available data.
    """
    if not os.path.isfile(input):
        typer.echo(f"Error: '{input}' is not a valid file.", err=True)
        typer.echo("Use -i/--input to specify the path to the input file.", err=True)
        raise typer.Exit(1)

    if not os.path.isdir(output):
        typer.echo(f"Error: '{output}' is not a valid directory.", err=True)
        typer.echo(
            "Use -o/--output to specify the path to the output folder.", err=True
        )
        raise typer.Exit(1)

    state = {"verbose": verbose}
    app.state = state

    try:
        dpy = DHybridrpy(input, output, exclude_timestep_zero=True)
    except (FileNotFoundError, NotADirectoryError) as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo(
            "Use -i and -o to specify the input file and output folder.", err=True
        )
        raise typer.Exit(1)

    first_ts = dpy.timestep(dpy.timesteps()[0])

    # Expand --all-fields / --all-phases
    if all_fields:
        fields = sorted(first_ts._fields_dict.get(field_type, {}).keys())
    if all_phases:
        phases = sorted(
            {
                name
                for sp, names in first_ts._phases_dict.items()
                if isinstance(sp, int)
                for name in names
            }
        )

    # Discovery mode: print available data and exit
    if not fields and not phases:
        print_available(dpy)
        raise typer.Exit()

    # Plot fields
    if fields:
        for field_name in fields:
            typer.echo(f"Field: {field_name} (type={field_type})")
            plot_dir = os.path.join(plots_dir, "Fields", field_name)

            def get_field(ts, _name=field_name, _type=field_type):
                return getattr(ts.fields, _name)(type=_type)

            plot_data_series(
                dpy,
                get_field,
                field_name,
                plot_dir,
                colormap,
                dpi,
                vmin,
                vmax,
                video,
                fps,
                jobs,
                pmin,
                pmax,
            )

    # Plot phases
    if phases:
        available_species = sorted(
            k for k in first_ts._phases_dict.keys() if isinstance(k, int)
        )
        target_species = species if species else available_species

        for phase_name in phases:
            for sp in target_species:
                label = f"{phase_name}_Sp{sp:02d}"
                typer.echo(f"Phase: {phase_name} (species={sp})")
                plot_dir = os.path.join(plots_dir, "Phases", label)

                def get_phase(ts, _name=phase_name, _sp=sp):
                    return getattr(ts.phases, _name)(species=_sp)

                plot_data_series(
                    dpy,
                    get_phase,
                    label,
                    plot_dir,
                    colormap,
                    dpi,
                    vmin,
                    vmax,
                    video,
                    fps,
                    jobs,
                    pmin,
                    pmax,
                )


def main():
    app()


if __name__ == "__main__":
    main()
