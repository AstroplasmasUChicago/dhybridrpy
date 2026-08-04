"""Interactive slice viewer for a 3D field with a 3D context cube.

Left panel: one 2D slice of the field, stepped with the slider or the
left/right arrow keys. Right panel: the field's volume drawn as its six
outer faces (rotatable), with a deep-red frame marking the current slice
position.

Usage:
    python slice_context_3d.py INPUT_FILE OUTPUT_FOLDER [--field Bx]
        [--type Total] [--timestep-index -1] [--slice-axis x]
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.widgets import Slider

from dhybridrpy import DHybridrpy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file")
    parser.add_argument("output_folder")
    parser.add_argument("--field", default="Bx")
    parser.add_argument("--type", dest="field_type", default="Total")
    parser.add_argument("--timestep-index", type=int, default=-1)
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--downsample", type=int, default=16,
                        help="cube face downsampling stride")
    return parser.parse_args()


def main():
    args = parse_args()
    dp = DHybridrpy(args.input_file, args.output_folder)
    timestep = dp.timestep_index(args.timestep_index)
    field = getattr(timestep.fields, args.field)(args.field_type)
    shape = field._get_data_shape()
    if len(shape) != 3:
        raise SystemExit(f"{args.field} is {len(shape)}D; need a 3D field.")

    axis_index = {"x": 0, "y": 1, "z": 2}[args.slice_axis]
    n_along = shape[axis_index]
    coords = [np.asarray(c) for c in (field.xdata, field.ydata, field.zdata)]
    along = coords[axis_index]
    # the two axes spanned by a slice plane, in plot order
    plane_axes = [i for i in range(3) if i != axis_index]
    u, v = coords[plane_axes[0]], coords[plane_axes[1]]

    fig = plt.figure(figsize=(12, 5.5))
    fig.canvas.manager.set_window_title(f"{args.field}: slice + 3D context")
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    plt.subplots_adjust(bottom=0.18, wspace=0.05)

    axis_labels = ("$x$", "$y$", "$z$")
    slice0 = field._read_2d_slice(args.slice_axis, 0)
    img = ax2d.imshow(slice0.T, origin="lower",
                      extent=(u[0], u[-1], v[0], v[-1]),
                      cmap="viridis", interpolation="nearest", aspect="auto")
    ax2d.set_xlabel(axis_labels[plane_axes[0]])
    ax2d.set_ylabel(axis_labels[plane_axes[1]])
    fig.colorbar(img, ax=ax2d, label=args.field)

    # cube context: all six outer faces, downsampled with the endpoints
    # included so adjacent faces meet without gaps
    def ds_indices(n):
        idx = np.arange(0, n, args.downsample)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        return idx

    sampled = [ds_indices(n) for n in shape]
    sampled_coords = [c[i] for c, i in zip(coords, sampled)]
    face_specs = [(axis, end) for axis in "xyz" for end in (0, -1)]

    # The cube is convex: draw only faces whose outward normal points at
    # the camera, so no depth sorting is needed and the slice frame can sit
    # on a fixed high zorder, always fully visible.
    ax3d.computed_zorder = False
    all_faces = {}
    for axis, end in face_specs:
        i_axis = {"x": 0, "y": 1, "z": 2}[axis]
        others = [i for i in range(3) if i != i_axis]
        data = field._read_2d_slice(axis, shape[i_axis] - 1 if end else 0)
        all_faces[(axis, end)] = data[np.ix_(sampled[others[0]],
                                             sampled[others[1]])]
    vmin = min(a.min() for a in all_faces.values())
    vmax = max(a.max() for a in all_faces.values())
    face_rgba = colormaps["viridis"]
    norm = mcolors.Normalize(vmin, vmax)

    face_artists = {}
    face_normals = {}
    for (axis, end), face in all_faces.items():
        i_axis = {"x": 0, "y": 1, "z": 2}[axis]
        others = [i for i in range(3) if i != i_axis]
        A, B = np.meshgrid(sampled_coords[others[0]],
                           sampled_coords[others[1]], indexing="ij")
        const = np.full_like(A, coords[i_axis][end])
        xyz = [None, None, None]
        xyz[i_axis], xyz[others[0]], xyz[others[1]] = const, A, B
        normal = np.zeros(3)
        normal[i_axis] = 1.0 if end else -1.0
        face_artists[(axis, end)] = ax3d.plot_surface(
            *xyz, facecolors=face_rgba(norm(face)), shade=False,
            rstride=1, cstride=1, antialiased=False, linewidth=0, zorder=1,
        )
        face_normals[(axis, end)] = normal
    ax3d.set_xlabel("$x$")
    ax3d.set_ylabel("$y$")
    ax3d.set_zlabel("$z$")
    ax3d.set_box_aspect((1, 1, 1))

    def cull_back_faces(_event=None):
        azim, elev = np.deg2rad(ax3d.azim), np.deg2rad(ax3d.elev)
        toward_camera = np.array([
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ])
        changed = False
        for key, artist in face_artists.items():
            visible = float(face_normals[key] @ toward_camera) > 0
            if artist.get_visible() != visible:
                artist.set_visible(visible)
                changed = True
        return changed

    # slice marker: four deep-red flange strips outside the cube + outline
    pad = 0.10 * (u[-1] - u[0])
    deep_red = (0.55, 0.0, 0.0, 0.9)
    marker_artists = []

    def draw_marker(i):
        for artist in marker_artists:
            artist.remove()
        marker_artists.clear()
        u_lo, u_hi = u[0] - pad, u[-1] + pad
        v_lo, v_hi = v[0] - pad, v[-1] + pad
        strips = (
            ((u_lo, u[0]), (v_lo, v_hi)),
            ((u[-1], u_hi), (v_lo, v_hi)),
            ((u[0], u[-1]), (v_lo, v[0])),
            ((u[0], u[-1]), (v[-1], v_hi)),
        )
        for (ua, ub), (va, vb) in strips:
            U2, V2 = np.meshgrid([ua, ub], [va, vb], indexing="ij")
            const = np.full_like(U2, along[i])
            xyz = [None, None, None]
            xyz[axis_index] = const
            xyz[plane_axes[0]], xyz[plane_axes[1]] = U2, V2
            marker_artists.append(
                ax3d.plot_surface(*xyz, color=deep_red, shade=False,
                                  antialiased=False, linewidth=0, zorder=10)
            )
        ring_u = [u_lo, u_hi, u_hi, u_lo, u_lo]
        ring_v = [v_lo, v_lo, v_hi, v_hi, v_lo]
        line_xyz = [None, None, None]
        line_xyz[axis_index] = [along[i]] * 5
        line_xyz[plane_axes[0]], line_xyz[plane_axes[1]] = ring_u, ring_v
        marker_artists.append(
            ax3d.plot(*line_xyz, color="darkred", lw=2.5, zorder=11)[0]
        )

    ax_slider = fig.add_axes([0.12, 0.05, 0.45, 0.03])
    slider = Slider(ax_slider, f"{args.slice_axis.capitalize()} slice",
                    0, n_along - 1, valinit=0, valstep=1)

    def update(_val):
        i = int(slider.val)
        t0 = time.perf_counter()
        sl = field._read_2d_slice(args.slice_axis, i)
        read_ms = (time.perf_counter() - t0) * 1000
        img.set_data(sl.T)
        img.set_clim(float(sl.min()), float(sl.max()))
        ax2d.set_title(f"{args.slice_axis} = {along[i]:.2f}   "
                       f"(read {read_ms:.1f} ms)")
        draw_marker(i)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        if event.key == "right":
            slider.set_val(min(int(slider.val) + 1, n_along - 1))
        elif event.key == "left":
            slider.set_val(max(int(slider.val) - 1, 0))

    def on_motion(event):
        if event.inaxes is ax3d and cull_back_faces():
            fig.canvas.draw_idle()

    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    cull_back_faces()
    draw_marker(0)
    update(0)
    plt.show()


if __name__ == "__main__":
    main()
