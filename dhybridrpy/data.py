import math
import threading
import os

import h5py
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import operator
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.widgets import Slider

from matplotlib.axes import Axes
from matplotlib.backend_bases import key_press_handler
from matplotlib.collections import QuadMesh
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from typing import Tuple, Union, Optional, Literal
from dask.delayed import delayed


# Standalone functions for numpy arrays


def open_h5(file_path: str, **kwargs) -> h5py.File:
    """Open an HDF5 file read-only without file locking, which reading does
    not need and which costs extra round trips on network filesystems.

    Two consequences: a file being written by a running simulation opens
    without error (possibly reading a partial dump), and holding a separate
    default-locking handle to the same file in the same process fails with
    "file locking flag values don't match".
    """
    return h5py.File(file_path, "r", locking=False, **kwargs)


# Datasets kept open for repeated 2D slicing (the interactive 3D slider).
# HDF5's chunk cache lives on the OPEN DATASET and is freed when it closes,
# so reopening per slice re-inflates every touched chunk; a kept dataset
# with a cache sized to hold one slice's chunks makes neighboring slices
# nearly free.
_slice_handles = {}  # {(path, mtime_ns, size): (h5py.File, h5py.Dataset)}
_slice_handles_lock = threading.Lock()
_slice_handles_pid = os.getpid()
_SLICE_HANDLES_MAX = 8
_SLICE_CACHE_CAP = 512 * 1024**2  # per-file chunk cache ceiling
_SLOT_PRIMES = (1009, 10007, 100003, 1000003)


def close_pooled_handles() -> None:
    """Close all pooled slicing handles.

    Needed before deleting or re-creating a sliced file in this process:
    a pooled handle keeps the file open, so h5py.File(path, "w") on it
    fails until the handle is released.
    """
    with _slice_handles_lock:
        for handle, _ in _slice_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        _slice_handles.clear()


def _pooled_dataset(file_path: str) -> h5py.Dataset:
    global _slice_handles_pid
    with _slice_handles_lock:
        if os.getpid() != _slice_handles_pid:
            # forked child: parent handles are unusable; start fresh
            _slice_handles.clear()
            _slice_handles_pid = os.getpid()

        stat = os.stat(file_path)
        key = (file_path, stat.st_mtime_ns, stat.st_size)
        entry = _slice_handles.get(key)
        if entry is not None:
            _slice_handles[key] = _slice_handles.pop(key)  # refresh recency
            return entry[1]

        # A replaced file gets a fresh handle. Evicted and stale entries are
        # dropped WITHOUT close(): a caller may still hold their Dataset, and
        # the read-only file closes on garbage collection once released.
        for old_key in [k for k in _slice_handles if k[0] == file_path]:
            _slice_handles.pop(old_key, None)

        with open_h5(file_path) as probe:
            dataset = probe["DATA"]
            chunks, shape, itemsize = (
                dataset.chunks, dataset.shape, dataset.dtype.itemsize,
            )

        kwargs = {}
        if chunks is not None:
            chunk_bytes = math.prod(chunks) * itemsize
            chunk_counts = [-(-s // c) for s, c in zip(shape, chunks)]
            # most chunks any single 2D slice can touch
            slice_chunks = max(
                math.prod(chunk_counts) // n for n in chunk_counts
            )
            kwargs["rdcc_nbytes"] = min(
                2 * slice_chunks * chunk_bytes, _SLICE_CACHE_CAP
            )
            cached = kwargs["rdcc_nbytes"] // max(chunk_bytes, 1)
            kwargs["rdcc_nslots"] = next(
                (p for p in _SLOT_PRIMES if p >= 20 * cached), _SLOT_PRIMES[-1]
            )

        handle = open_h5(file_path, **kwargs)
        _slice_handles[key] = (handle, handle["DATA"])
        while len(_slice_handles) > _SLICE_HANDLES_MAX:
            _slice_handles.pop(next(iter(_slice_handles)), None)
        return _slice_handles[key][1]


def _add_slice_context(ax3d, source, slice_axis: str, colormap: str):
    """Draw `source`'s volume on `ax3d` as its six outer faces and return
    (draw_marker, cull_back_faces) closures for the moving slice frame.

    The cube is convex, so only the faces whose outward normal points at
    the camera are shown; with no overlapping geometry, matplotlib's
    per-artist depth sort is unnecessary and the slice frame can sit on a
    fixed high zorder, visible from every angle.
    """
    shape = source._get_data_shape()
    coords = [
        np.asarray(source._materialize(c))
        for c in (source.xdata, source.ydata, source.zdata)
    ]
    axis_index = {"x": 0, "y": 1, "z": 2}[slice_axis]
    plane_axes = [i for i in range(3) if i != axis_index]
    along = coords[axis_index]
    u, v = coords[plane_axes[0]], coords[plane_axes[1]]

    def sampled_indices(n):
        idx = np.arange(0, n, max(1, n // 24))
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)  # include the far edge: faces must meet
        return idx

    sampled = [sampled_indices(n) for n in shape]
    faces = {}
    for face_axis in "xyz":
        i_axis = {"x": 0, "y": 1, "z": 2}[face_axis]
        others = [i for i in range(3) if i != i_axis]
        picker = np.ix_(sampled[others[0]], sampled[others[1]])
        faces[(face_axis, 0)] = source._read_2d_slice(face_axis, 0)[picker]
        faces[(face_axis, -1)] = source._read_2d_slice(
            face_axis, shape[i_axis] - 1
        )[picker]

    norm = mcolors.Normalize(
        min(a.min() for a in faces.values()),
        max(a.max() for a in faces.values()),
    )
    face_rgba = colormaps[colormap] if isinstance(colormap, str) else colormap
    ax3d.computed_zorder = False
    face_artists = {}
    face_normals = {}
    for (face_axis, end), face in faces.items():
        i_axis = {"x": 0, "y": 1, "z": 2}[face_axis]
        others = [i for i in range(3) if i != i_axis]
        A, B = np.meshgrid(
            coords[others[0]][sampled[others[0]]],
            coords[others[1]][sampled[others[1]]],
            indexing="ij",
        )
        xyz = [None, None, None]
        xyz[i_axis] = np.full_like(A, coords[i_axis][end])
        xyz[others[0]], xyz[others[1]] = A, B
        face_artists[(face_axis, end)] = ax3d.plot_surface(
            *xyz, facecolors=face_rgba(norm(face)), shade=False,
            rstride=1, cstride=1, antialiased=False, linewidth=0, zorder=1,
        )
        normal = np.zeros(3)
        normal[i_axis] = -1.0 if end == 0 else 1.0
        face_normals[(face_axis, end)] = normal
    ax3d.set_xlabel("$x$")
    ax3d.set_ylabel("$y$")
    ax3d.set_zlabel("$z$")
    ax3d.set_box_aspect((1, 1, 1))

    def cull_back_faces() -> bool:
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

    # slice frame: four deep-red strips outside the cube plus an outline
    pad = 0.10 * (u[-1] - u[0])
    deep_red = (0.55, 0.0, 0.0, 0.9)
    marker_artists = []

    def draw_marker(slice_index: int) -> None:
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
            xyz = [None, None, None]
            xyz[axis_index] = np.full_like(U2, along[slice_index])
            xyz[plane_axes[0]], xyz[plane_axes[1]] = U2, V2
            marker_artists.append(
                ax3d.plot_surface(*xyz, color=deep_red, shade=False,
                                  antialiased=False, linewidth=0, zorder=10)
            )
        ring_u = [u_lo, u_hi, u_hi, u_lo, u_lo]
        ring_v = [v_lo, v_lo, v_hi, v_hi, v_lo]
        xyz = [None, None, None]
        xyz[axis_index] = [along[slice_index]] * 5
        xyz[plane_axes[0]], xyz[plane_axes[1]] = ring_u, ring_v
        marker_artists.append(
            ax3d.plot(*xyz, color="darkred", lw=2.5, zorder=11)[0]
        )

    cull_back_faces()
    return draw_marker, cull_back_faces


def _rfftn(data: np.ndarray) -> np.ndarray:
    """Real-input FFT, threaded when scipy is available."""
    try:
        from scipy import fft as scipy_fft
    except ImportError:
        return np.fft.rfftn(data)
    return scipy_fft.rfftn(data, workers=-1)


def _radial_power_spectrum(
    data: np.ndarray, box_lengths, normalize: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic power spectrum shared by the 1D, 2D, and 3D branches.

    Uses a real-input FFT over half the spectrum with conjugate-pair
    doubling, and sums shells in float64.
    """
    shape = data.shape
    ndim = data.ndim
    deltas = [length / n for length, n in zip(box_lengths, shape)]

    fft_data = _rfftn(data)
    power = fft_data.real.astype(np.float64) ** 2
    power += fft_data.imag.astype(np.float64) ** 2
    power *= float(np.prod(deltas)) / int(np.prod(shape))

    # Each spectral value on the halved last axis stands for a conjugate
    # pair, except the unpaired first plane and, for even sizes, the last.
    n_last = shape[-1]
    doubled = slice(1, -1) if n_last % 2 == 0 else slice(1, None)
    power[..., doubled] *= 2.0

    freqs = [
        np.fft.fftfreq(n, d=d) * 2 * np.pi
        for n, d in zip(shape[:-1], deltas[:-1])
    ]
    freqs.append(np.fft.rfftfreq(n_last, d=deltas[-1]) * 2 * np.pi)

    k_squared = np.zeros(power.shape)
    for axis, freq in enumerate(freqs):
        broadcast = [1] * ndim
        broadcast[axis] = len(freq)
        k_squared += (freq**2).reshape(broadcast)
    k_magnitude = np.sqrt(k_squared, out=k_squared).ravel()

    k_max = min(np.abs(freq).max() for freq in freqs)
    dk = 2 * np.pi / max(box_lengths)
    k_bins = np.arange(0, k_max + dk, dk)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    num_bins = len(k_bins) - 1
    if num_bins < 1:  # a size-1 axis pins k_max to zero
        return k_centers, np.zeros(0)

    # Same shell assignment as np.histogram: half-open bins, a closed last
    # edge, and modes beyond the last edge dropped.
    indices = np.searchsorted(k_bins, k_magnitude, side="right") - 1
    indices[k_magnitude == k_bins[-1]] = num_bins - 1
    valid = indices < num_bins
    indices = indices[valid]

    binned = np.bincount(
        indices, weights=power.ravel()[valid], minlength=num_bins
    )
    if normalize:
        pair_weights = np.ones(shape[:-1] + (power.shape[-1],))
        pair_weights[..., doubled] = 2.0
        counts = np.bincount(
            indices, weights=pair_weights.ravel()[valid], minlength=num_bins
        )
        binned = np.divide(
            binned, counts, out=np.zeros_like(binned), where=counts > 0
        )
    return k_centers, binned


def fft_power_iso(
    data: np.ndarray,
    Lx: float,
    Ly: Optional[float] = None,
    Lz: Optional[float] = None,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute isotropic FFT power spectrum of a numpy array.

    Args:
        data: 1D, 2D, or 3D numpy array
        Lx: Box size in x direction
        Ly: Box size in y direction (required for 2D/3D)
        Lz: Box size in z direction (required for 3D)
        normalize: if True, divide each radial |k| shell by its mode count, giving
            the per-mode (azimuthally averaged) power instead of the summed shell
            energy (default False).

    Returns:
        Tuple of (k, power) where:
            - k: 1D array of wavenumber values (in units of 2π/L)
            - power: 1D array of power spectral density at each k (float64
              regardless of input dtype)
    """
    num_dimensions = data.ndim
    if num_dimensions < 1 or num_dimensions > 3:
        raise NotImplementedError("fft_power_iso only supports 1D, 2D, or 3D data.")
    if num_dimensions >= 2 and Ly is None:
        raise ValueError("Ly required for 2D data")
    if num_dimensions == 3 and Lz is None:
        raise ValueError("Lz required for 3D data")

    box_lengths = [Lx, Ly, Lz][:num_dimensions]
    return _radial_power_spectrum(data, box_lengths, normalize)


def fft_power_1d_slices(
    data: np.ndarray,
    L: float,
    direction: Literal["x", "y", "z"] = "x",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 1D FFT power spectra along a chosen direction with statistics.

    Extracts 1D slices along the specified direction, computes FFT power
    spectrum for each slice, then returns the geometric mean and
    multiplicative standard deviation across all slices.

    Args:
        data: 1D, 2D, or 3D numpy array
        L: Box size along the chosen direction
        direction: Direction along which to compute 1D FFTs ("x", "y", or "z")

    Returns:
        Tuple of (k, power_mean, power_std_lower, power_std_upper) where:
            - k: 1D array of wavenumber values (in units of 2π/L)
            - power_mean: 1D array of geometric mean power at each k
            - power_std_lower: 1D array of geometric mean / multiplicative std
            - power_std_upper: 1D array of geometric mean × multiplicative std
        All power arrays are float64 regardless of input dtype.
    """
    if direction not in ["x", "y", "z"]:
        raise ValueError("Direction must be 'x', 'y', or 'z'.")

    num_dimensions = data.ndim
    if num_dimensions < 1 or num_dimensions > 3:
        raise NotImplementedError(
            "fft_power_1d_slices only supports 1D, 2D, or 3D data."
        )

    # Get grid points along the chosen direction
    if direction == "x":
        n = data.shape[0]
        axis = 0
    elif direction == "y":
        if num_dimensions < 2:
            raise ValueError("Cannot compute FFT along 'y' for 1D data.")
        n = data.shape[1]
        axis = 1
    else:  # z
        if num_dimensions < 3:
            raise ValueError("Cannot compute FFT along 'z' for 1D or 2D data.")
        n = data.shape[2]
        axis = 2

    # Use rfft for real input: returns n//2 + 1 bins including DC and (for even n)
    # the Nyquist bin, which np.fft.fftfreq's positive half misses.
    k = np.fft.rfftfreq(n, d=L / n) * 2 * np.pi

    fft_data = np.fft.rfft(data, axis=axis)
    power = np.abs(fft_data) ** 2 / n

    # Double interior bins to account for negative-frequency conjugate pairs.
    # DC (index 0) is unique. For even n the Nyquist bin (last index) is also
    # unique and must NOT be doubled; for odd n every non-DC bin has a pair.
    doubler = [slice(None)] * power.ndim
    doubler[axis] = slice(1, -1 if n % 2 == 0 else None)
    power[tuple(doubler)] *= 2

    # Move FFT axis to the end and collapse the remaining dims into "slices";
    # float64 so the naive variance below doesn't lose precision.
    power = np.moveaxis(power, axis, -1)
    power_spectra = power.reshape(-1, power.shape[-1]).astype(
        np.float64, copy=False
    )

    # Geometric statistics in log space, computed only over positive entries
    # per bin. Bins where every slice has zero power return 0 for mean/lo/hi
    # rather than the previous hardcoded 1e-50 floor.
    positive = power_spectra > 0
    with np.errstate(divide="ignore"):
        log_power = np.where(positive, np.log(power_spectra), 0.0)

    count = positive.sum(axis=0)
    sum_log = log_power.sum(axis=0)
    sum_log_sq = (log_power**2).sum(axis=0)

    safe_count = np.where(count > 0, count, 1)
    log_mean = sum_log / safe_count
    log_var = sum_log_sq / safe_count - log_mean**2
    log_std = np.sqrt(np.maximum(log_var, 0.0))

    power_mean = np.where(count > 0, np.exp(log_mean), 0.0)
    power_std_lower = np.where(count > 0, np.exp(log_mean - log_std), 0.0)
    power_std_upper = np.where(count > 0, np.exp(log_mean + log_std), 0.0)

    return k, power_mean, power_std_lower, power_std_upper


class BaseProperties:
    def __init__(
        self, file_path: str, name: str, timestep: int, time: float, lazy: bool
    ):
        self.file_path = file_path
        self.name = name
        self.timestep = timestep
        self.time = time
        self.lazy = lazy
        self._data_dict = {}

    def __repr__(self) -> str:
        attrs = ", ".join(
            f"{attr}={value}"
            for attr, value in self.__dict__.items()
            if not attr.startswith("_")
        )
        return f"{self.__class__.__name__}({attrs})"


class Data(BaseProperties):
    _X = "$x / d_i$"
    _Y = "$y / d_i$"
    _Z = "$z / d_i$"
    _PX = "$p_x / (m_i v_A)$"
    _PY = "$p_y / (m_i v_A)$"
    _PZ = "$p_z / (m_i v_A)$"
    _PTOT = "$p_{tot} / (m_i v_A)$"
    _ETOT = r"$\ln\left(\frac{e_{tot}}{m_i v_A^2}\right)$"

    _LABEL_MAPPINGS = {
        "p1x1": (_X, _PX),
        "p1x2": (_Y, _PX),
        "p1x3": (_Z, _PX),
        "p2x1": (_X, _PY),
        "p2x2": (_Y, _PY),
        "p2x3": (_Z, _PY),
        "p3x1": (_X, _PZ),
        "p3x2": (_Y, _PZ),
        "p3x3": (_Z, _PZ),
        "x2x1": (_X, _Y),
        "x3x1": (_X, _Z),
        "x3x2": (_Y, _Z),
        "p2p1": (_PX, _PY),
        "p3p1": (_PX, _PZ),
        "p3p2": (_PY, _PZ),
        "ptx1": (_X, _PTOT),
        "ptx2": (_Y, _PTOT),
        "ptx3": (_Z, _PTOT),
        "etx1": (_X, _ETOT),
        "etx2": (_Y, _ETOT),
        "etx3": (_Z, _ETOT),
    }

    @classmethod
    def _axis_labels(cls, name: str) -> tuple:
        """Look up plot axis labels for a Data name, falling back to (x, y)."""
        return cls._LABEL_MAPPINGS.get(name, (cls._X, cls._Y))

    # For derived object plot titles
    _BINOP_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "truediv": "/", "pow": "^"}

    def __init__(
        self,
        file_path: str,
        name: str,
        timestep: int,
        time: float,
        time_ndecimals: int,
        lazy: bool,
    ):
        super().__init__(file_path, name, timestep, time, lazy)
        self._time_ndecimals = time_ndecimals
        self._plot_title = rf"{name} at time {round(time, self._time_ndecimals)} $\omega_{{ci}}^{{-1}}$"
        self._data_shape = None
        self._data_dtype = None

    def _load_metadata(self) -> None:
        """Read shape, dtype, and all axis limits in a single file open."""
        with open_h5(self.file_path) as file:
            dataset = file["DATA"]
            if self._data_shape is None:
                # Reverse the data shape to be consistent with transpose in data @property
                self._data_shape = dataset.shape[::-1]
            if self._data_dtype is None:
                self._data_dtype = dataset.dtype
            axis_group = file.get("AXIS")
            if axis_group is not None:
                for axis_name in axis_group:
                    key = f"{axis_name} lims"
                    if key not in self._data_dict:
                        self._data_dict[key] = axis_group[axis_name][:]

    def _get_coordinate_limits(self, axis_name: str) -> np.ndarray:
        key = f"{axis_name} lims"
        if key not in self._data_dict:
            self._load_metadata()
        return self._data_dict[key]

    def _compute_coordinates(
        self, axis_name: str, size: int
    ) -> Union[np.ndarray, da.Array]:
        key = f"{axis_name} coords"
        if key not in self._data_dict:
            axis_limits = self._get_coordinate_limits(axis_name)
            delta = (axis_limits[1] - axis_limits[0]) / size
            grid = da.arange(size, chunks="auto") if self.lazy else np.arange(size)
            self._data_dict[key] = delta * grid + (delta / 2) + axis_limits[0]
        return self._data_dict[key]

    def _read_2d_slice(
        self, slice_axis: Literal["x", "y", "z"], idx: int
    ) -> np.ndarray:
        """Return a single 2D slice of 3D data without materializing the full cube.

        For derived Data (where the result is already in memory), this slices
        the cached array. For HDF5-backed Data, this issues a partial read so
        the slider doesn't have to hold the entire 3D volume.
        """
        if self.name in self._data_dict:
            arr = self._data_dict[self.name]
            if slice_axis == "x":
                arr = arr[idx, :, :]
            elif slice_axis == "y":
                arr = arr[:, idx, :]
            else:
                arr = arr[:, :, idx]
            # slice before materializing so a lazy array computes one plane,
            # not the whole cube
            return self._materialize(arr)

        # HDF5 stores (nz, ny, nx); after the .T in `data`, the numpy convention
        # is (nx, ny, nz). Mirror that here while reading only what's needed.
        ds = _pooled_dataset(self.file_path)
        if slice_axis == "x":
            return np.asarray(ds[:, :, idx]).T  # -> (ny, nz)
        if slice_axis == "y":
            return np.asarray(ds[:, idx, :]).T  # -> (nx, nz)
        return np.asarray(ds[idx, :, :]).T  # -> (nx, ny)

    def _materialize(self, arr: Union[np.ndarray, da.Array]) -> np.ndarray:
        """Return a numpy array for `arr`, calling .compute() if it's a dask array.

        Callers should bind `self.data` (or another property) once, then pass it
        in. Accessing `self.data` is a real HDF5 read in non-lazy mode.
        """
        if self.lazy and isinstance(arr, da.Array):
            return arr.compute()
        return arr

    def _parallel_read_worthwhile(self) -> bool:
        """Whether to read this dataset with worker processes: it must be
        large enough to benefit and spawning workers must be safe here."""
        from . import _parallel

        nbytes = (
            int(np.prod(self._get_data_shape()))
            * np.dtype(self._get_data_dtype()).itemsize
        )
        return (
            nbytes >= _parallel.SINGLE_FILE_MIN_BYTES
            and _parallel.spawn_is_safe()
        )

    def _get_data_shape(self) -> Tuple[int, ...]:
        """Retrieve the shape of the data without loading it."""
        if self._data_shape is None:
            self._load_metadata()
        return self._data_shape

    def _get_data_dtype(self) -> np.dtype:
        """Retrieve the type of the data without loading it."""
        if self._data_dtype is None:
            self._load_metadata()
        return self._data_dtype

    @property
    def data(self) -> Union[np.ndarray, da.Array]:
        """Retrieve the data at each grid point."""
        # If data was set programmatically (e.g. by arithmetic operators),
        # return the cached result since there is no HDF5 file to re-read.
        if self.name in self._data_dict:
            return self._data_dict[self.name]

        # Otherwise, always re-read from HDF5 without caching to avoid OOM
        # when iterating over many timesteps.
        def loader():
            with open_h5(self.file_path) as f:
                return f["DATA"][:].T

        if self.lazy:
            delayed_obj = delayed(loader)()
            return da.from_delayed(
                delayed_obj,
                shape=self._get_data_shape(),
                dtype=self._get_data_dtype(),
            )
        elif self._parallel_read_worthwhile():
            from . import _parallel

            try:
                arr = _parallel.parallel_read_data(self.file_path)
            except Exception:
                arr = None
            if arr is not None:
                return arr.T
            return loader()
        else:
            return loader()

    @property
    def xdata(self) -> Union[np.ndarray, da.Array]:
        """Retrieve the x (i.e. X1) grid coordinates."""
        return self._compute_coordinates("X1 AXIS", self._get_data_shape()[0])

    @property
    def ydata(self) -> Union[np.ndarray, da.Array]:
        """Retrieve the y (i.e. X2) grid coordinates."""
        return self._compute_coordinates("X2 AXIS", self._get_data_shape()[1])

    @property
    def zdata(self) -> Union[np.ndarray, da.Array]:
        """Retrieve the z (i.e. X3) grid coordinates."""
        return self._compute_coordinates("X3 AXIS", self._get_data_shape()[2])

    @property
    def xlimdata(self) -> Union[np.ndarray, da.Array]:
        """Retrieve the x (i.e. X1) grid axis limits."""
        return self._get_coordinate_limits("X1 AXIS")

    @property
    def ylimdata(self) -> Union[np.ndarray, da.Array]:
        """Retrieve the y (i.e. X2) grid axis limits."""
        return self._get_coordinate_limits("X2 AXIS")

    @property
    def zlimdata(self) -> Union[np.ndarray, da.Array]:
        """Retrieve the z (i.e. X3) grid axis limits."""
        return self._get_coordinate_limits("X3 AXIS")

    def _check_compatibility(self, other) -> None:
        """Raise if 'self' and 'other' cannot be operated on together."""

        if type(self) is not type(other):
            raise TypeError(
                f"Cannot combine {type(self).__name__} with "
                f"{type(other).__name__}: operands must be the same subclass."
            )
        if self._get_data_shape() != other._get_data_shape():
            raise ValueError(
                f"Incompatible grid shapes: {self._get_data_shape()} vs "
                f"{other._get_data_shape()}"
            )
        if self.timestep != other.timestep:
            raise ValueError(f"Timesteps differ: {self.timestep} vs {other.timestep}")

    @staticmethod
    def _short_operand_name(value) -> str:
        """Concise label for the right-hand operand in a derived Data name."""
        if isinstance(value, np.ndarray):
            return f"ndarray<{','.join(str(d) for d in value.shape)}>"
        if isinstance(value, da.Array):
            return f"darray<{','.join(str(d) for d in value.shape)}>"
        return str(value)

    def _apply_operation(self, other, op):
        """Apply a binary operation to self and another Data object or scalar."""

        if isinstance(other, Data):
            self._check_compatibility(other)
            left = self.data
            # expressions like bx * bx should read the file once, not twice
            right = left if other is self else other.data
            result = op(left, right)
            other_name = other.name
        else:
            result = op(self.data, other)
            other_name = self._short_operand_name(other)

        symbol = self._BINOP_SYMBOL.get(op.__name__, op.__name__)
        return self._create_new_instance(result, symbol, other_name, other)

    def _extra_init_args(self) -> tuple:
        """Positional args that the subclass's __init__ expects"""
        return ()

    @staticmethod
    def _trim_subtype(title: str) -> str:
        """Remove the trailing ' (type = ...)' or ' (species = ...)' for derived objects."""
        for token in (" (type =", " (species ="):
            index = title.find(token)
            if index != -1:
                return title[:index].rstrip()
        return title

    def _create_new_instance(
        self,
        result_array,
        op_symbol: str,
        other_name: str,
        other_obj=None,
    ):
        """Create a new Data instance with the result of the operation."""

        result_shape = getattr(result_array, "shape", ())
        if result_shape == ():
            raise ValueError(
                f"Operation '{op_symbol or other_name}' produced a 0-d result; "
                f"Data objects must have at least 1 dimension."
            )

        file_path = (
            other_obj.file_path if isinstance(other_obj, Data) else self.file_path
        )

        if op_symbol:
            if op_symbol == "^":
                new_name = f"({self.name}){op_symbol}{other_name}"
            else:
                new_name = f"{self.name}{op_symbol}{other_name}"
        else:  # name is supplied by ufunc wrapper
            new_name = other_name

        inst = self.__class__(
            file_path,
            new_name,
            self.timestep,
            self.time,
            self._time_ndecimals,
            self.lazy,
            *self._extra_init_args(),
        )
        # Only carry over cached AXIS coords/lims if shape didn't change;
        # otherwise the cached arrays would be the wrong length.
        if tuple(result_array.shape) == self._get_data_shape():
            inst._data_dict = {k: v for k, v in self._data_dict.items() if "AXIS" in k}
        else:
            inst._data_dict = {}
        inst._data_dict[new_name] = result_array
        inst._data_shape = tuple(result_array.shape)
        inst._data_dtype = getattr(result_array, "dtype", None)

        inst._plot_title = self._plot_title.replace(self.name, new_name)
        inst._plot_title = Data._trim_subtype(inst._plot_title)

        return inst

    def __add__(self, other):
        return self._apply_operation(other, operator.add)

    def __sub__(self, other):
        return self._apply_operation(other, operator.sub)

    def __mul__(self, other):
        return self._apply_operation(other, operator.mul)

    def __truediv__(self, other):
        return self._apply_operation(other, operator.truediv)

    def __pow__(self, other):
        return self._apply_operation(other, operator.pow)

    def __neg__(self):
        return self._create_new_instance(-self.data, "", f"-{self.name}", self)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return self._create_new_instance(
            other - self.data,
            "",
            f"{self._short_operand_name(other)}-{self.name}",
            self,
        )

    def __rtruediv__(self, other):
        return self._create_new_instance(
            other / self.data,
            "",
            f"{self._short_operand_name(other)}/{self.name}",
            self,
        )

    # Ensure that mixed Data and NumPy operations produce a Data object
    __array_priority__ = 20

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Allow NumPy ufuncs to be applied to Data objects"""

        if method != "__call__":
            return NotImplemented  # only allow element‑wise operations

        # Extract raw arrays and gather Data operands for naming / compat
        # checks. A Data object appearing more than once, as in
        # np.arctan2(by, by), is read only once.
        raw_inputs, data_operands = [], []
        materialized = {}
        for input in inputs:
            if isinstance(input, Data):
                data_operands.append(input)
                if id(input) not in materialized:
                    materialized[id(input)] = input.data
                raw_inputs.append(materialized[id(input)])
            else:
                raw_inputs.append(input)

        if data_operands:
            ref = data_operands[0]
            for other in data_operands[1:]:
                ref._check_compatibility(other)

        # Execute the ufunc on the underlying arrays
        result_array = ufunc(*raw_inputs, **kwargs)

        # No Data operands, so return the non-Data result
        if not data_operands:
            return result_array

        # Multi-output ufuncs (np.modf, np.divmod, np.frexp, ...) return a
        # tuple. Wrapping each output in its own Data needs a unique name per
        # output and isn't currently implemented; reject explicitly so the user
        # gets a clear message instead of a misleading "0-d result" further down.
        if isinstance(result_array, tuple):
            raise NotImplementedError(
                f"Multi-output ufunc {ufunc.__name__!r} is not supported on "
                f"Data objects. Compute on .data directly."
            )

        # Build a descriptive name: e.g. "sin(By)"
        names = ",".join(
            obj.name if isinstance(obj, Data) else self._short_operand_name(obj)
            for obj in inputs
        )
        new_name = f"{ufunc.__name__}({names})"

        # Choose a parent to copy metadata from (take self if it's a Data object)
        parent = next((obj for obj in data_operands if isinstance(obj, Data)), None)

        return parent._create_new_instance(result_array, "", new_name, parent)

    def avg_1d(
        self,
        direction: Literal["x", "y", "z"] = "x",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 1D average and standard deviation along a chosen direction.

        Averages the data along all axes perpendicular to the specified direction.

        Args:
            direction: The direction along which to compute the average ("x", "y", or "z").
                       The data is averaged over all other dimensions.

        Returns:
            Tuple of (coords, mean, std_lower, std_upper) where:
                - coords: 1D array of coordinates along the specified direction
                - mean: 1D array of mean values
                - std_lower: 1D array of mean - standard deviation
                - std_upper: 1D array of mean + standard deviation
        """
        if direction not in ["x", "y", "z"]:
            raise ValueError("Direction must be 'x', 'y', or 'z'.")

        num_dimensions = len(self._get_data_shape())
        if num_dimensions < 1:
            raise ValueError("Data must have at least 1 dimension.")

        data = self._materialize(self.data)

        # Determine which axis corresponds to the direction and compute mean/std
        if num_dimensions == 1:
            # For 1D data, just return as-is (no averaging needed)
            coord_data = self._materialize(self.xdata)
            mean_data = data
            std_data = np.zeros_like(data)
        elif num_dimensions == 2:
            # For 2D data: shape is (nx, ny); x -> axis 0, y -> axis 1
            if direction == "x":
                mean_data = np.mean(data, axis=1)
                std_data = np.std(data, axis=1)
                coord_data = self._materialize(self.xdata)
            elif direction == "y":
                mean_data = np.mean(data, axis=0)
                std_data = np.std(data, axis=0)
                coord_data = self._materialize(self.ydata)
            else:  # z
                raise ValueError(
                    "Cannot average along 'z' for 2D data. Use 'x' or 'y'."
                )
        elif num_dimensions == 3:
            # For 3D data: shape is (nx, ny, nz); x->0, y->1, z->2
            if direction == "x":
                mean_data = np.mean(data, axis=(1, 2))
                std_data = np.std(data, axis=(1, 2))
                coord_data = self._materialize(self.xdata)
            elif direction == "y":
                mean_data = np.mean(data, axis=(0, 2))
                std_data = np.std(data, axis=(0, 2))
                coord_data = self._materialize(self.ydata)
            else:  # z
                mean_data = np.mean(data, axis=(0, 1))
                std_data = np.std(data, axis=(0, 1))
                coord_data = self._materialize(self.zdata)
        else:
            raise NotImplementedError("avg_1d only supports 1D, 2D, or 3D data.")

        return coord_data, mean_data, mean_data - std_data, mean_data + std_data

    def crop(
        self,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
    ) -> "Data":
        """
        Return a new instance restricted to a fractional sub-region of the box.

        Each `*_range` is a (low, high) pair of fractions in [0, 1] along that
        axis (0 = start of box, 1 = end of box); axes left as None are kept
        whole. Useful for excluding domain-edge artifacts (e.g. absorbing
        boundaries) from color scales and other statistics computed downstream.

        Args:
            x_range: (low, high) fraction of the box to keep along x.
            y_range: (low, high) fraction of the box to keep along y.
            z_range: (low, high) fraction of the box to keep along z.

        Returns:
            A new instance of the same class holding only the cropped region,
            with axis coordinates and limits adjusted to match.
        """
        shape = self._get_data_shape()
        axis_names = ("X1 AXIS", "X2 AXIS", "X3 AXIS")[: len(shape)]
        fractions = (x_range, y_range, z_range)[: len(shape)]

        index_slices = []
        # Seed with the parent's axis entries so chaining crops keeps the
        # lims/coords of axes cropped in an earlier call.
        axis_overrides = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in self._data_dict.items()
            if "AXIS" in k
        }
        for axis_name, size, frac_range in zip(axis_names, shape, fractions):
            if frac_range is None:
                index_slices.append(slice(None))
                continue

            lo_frac, hi_frac = frac_range
            if not (0.0 <= lo_frac < hi_frac <= 1.0):
                raise ValueError(
                    f"Invalid crop range {frac_range} for {axis_name}; "
                    "expected 0 <= low < high <= 1."
                )
            i0 = int(round(lo_frac * size))
            i1 = max(int(round(hi_frac * size)), i0 + 1)
            index_slices.append(slice(i0, i1))

            limits = self._materialize(self._get_coordinate_limits(axis_name))
            delta = (limits[1] - limits[0]) / size
            axis_overrides[f"{axis_name} lims"] = np.array(
                [limits[0] + delta * i0, limits[0] + delta * i1]
            )
            coords = self._compute_coordinates(axis_name, size)
            sliced_coords = coords[i0:i1]
            if isinstance(sliced_coords, np.ndarray):
                sliced_coords = sliced_coords.copy()
            axis_overrides[f"{axis_name} coords"] = sliced_coords

        inst = self.__class__(
            self.file_path,
            self.name,
            self.timestep,
            self.time,
            self._time_ndecimals,
            self.lazy,
            *self._extra_init_args(),
        )
        inst._data_dict = axis_overrides
        cropped = self.data[tuple(index_slices)]
        # Copy: a numpy view here would keep the entire parent array alive.
        if isinstance(cropped, np.ndarray):
            cropped = cropped.copy()
        inst._data_dict[self.name] = cropped
        inst._data_shape = tuple(cropped.shape)
        inst._data_dtype = getattr(cropped, "dtype", None)
        inst._plot_title = self._plot_title
        return inst

    def fft_power(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the FFT power spectrum of the data.

        Computes the power spectral density as a function of wavenumber k,
        using the box size as the reference for wavenumber units.
        For multi-dimensional data, returns the radially-averaged (isotropic)
        power spectrum.

        Returns:
            Tuple of (k, power) where:
                - k: 1D array of wavenumber values (in units of 2π/L where L is box size)
                - power: 1D array of power spectral density at each k
        """

        data = self._materialize(self.data)
        num_dimensions = data.ndim

        xlim = self._materialize(self.xlimdata)
        Lx = xlim[1] - xlim[0]

        Ly = None
        Lz = None
        if num_dimensions >= 2:
            ylim = self._materialize(self.ylimdata)
            Ly = ylim[1] - ylim[0]
        if num_dimensions >= 3:
            zlim = self._materialize(self.zlimdata)
            Lz = zlim[1] - zlim[0]

        return fft_power_iso(data, Lx, Ly, Lz)

    def plot_fft_power(
        self,
        *,
        ax: Optional[Axes] = None,
        dpi: int = 100,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        loglog: bool = True,
        **kwargs,
    ) -> Tuple[Axes, Line2D]:
        """
        Plot the FFT power spectrum.

        Args:
            ax: Matplotlib Axes instance.
            dpi: Resolution of the plot.
            title: Plot title.
            xlabel, ylabel: Axis labels.
            xlim, ylim: Axis limits.
            loglog: Whether to use log-log scale (default True).
            **kwargs: Additional keyword arguments for the plot function.

        Returns:
            Tuple of (Axes, Line2D) for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

        k, power = self.fft_power()

        # Filter out zero/negative values for log plot
        if loglog:
            valid = (k > 0) & (power > 0)
            k_plot = k[valid]
            power_plot = power[valid]
            line = ax.loglog(k_plot, power_plot, **kwargs)[0]
        else:
            line = ax.plot(k, power, **kwargs)[0]

        default_title = rf"{self.name} power spectrum at time {round(self.time, self._time_ndecimals)} $\omega_{{ci}}^{{-1}}$"
        ax.set_title(title if title else default_title)
        ax.set_xlabel(xlabel if xlabel else r"$k \cdot d_i$")
        ax.set_ylabel(ylabel if ylabel else r"Power")

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.grid(True, alpha=0.3)

        return ax, line

    def fft_power_1d(
        self,
        direction: Literal["x", "y", "z"] = "x",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 1D FFT power spectra along a chosen direction with statistics.

        Extracts 1D slices along the specified direction, computes FFT power
        spectrum for each slice, then returns the geometric mean and
        multiplicative standard deviation across all slices. Statistics are
        computed in log space for proper representation on log-log plots.

        Args:
            direction: The direction along which to compute 1D FFTs ("x", "y", or "z").

        Returns:
            Tuple of (k, power_mean, power_std_lower, power_std_upper) where:
                - k: 1D array of wavenumber values (in units of 2π/L)
                - power_mean: 1D array of geometric mean power at each k
                - power_std_lower: 1D array of geometric mean / multiplicative std
                - power_std_upper: 1D array of geometric mean * multiplicative std
        """

        data = self._materialize(self.data)

        if direction == "x":
            lim = self._materialize(self.xlimdata)
        elif direction == "y":
            lim = self._materialize(self.ylimdata)
        else:  # z
            lim = self._materialize(self.zlimdata)
        L = lim[1] - lim[0]

        return fft_power_1d_slices(data, L, direction)

    def plot_fft_power_1d(
        self,
        direction: Literal["x", "y", "z"] = "x",
        *,
        ax: Optional[Axes] = None,
        dpi: int = 100,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        loglog: bool = True,
        fill_alpha: float = 0.3,
        fill_color: Optional[str] = None,
        line_color: Optional[str] = None,
        show_std: bool = True,
        **kwargs,
    ) -> Tuple[Axes, Line2D]:
        """
        Plot 1D FFT power spectrum along a chosen direction with std deviation band.

        Args:
            direction: The direction along which to compute 1D FFTs ("x", "y", or "z").
            ax: Matplotlib Axes instance.
            dpi: Resolution of the plot.
            title: Plot title.
            xlabel, ylabel: Axis labels.
            xlim, ylim: Axis limits.
            loglog: Whether to use log-log scale (default True).
            fill_alpha: Alpha (transparency) for the std deviation fill region.
            fill_color: Color for the fill region. Defaults to match line color.
            line_color: Color for the mean line.
            show_std: Whether to show the standard deviation fill region.
            **kwargs: Additional keyword arguments for the plot function.

        Returns:
            Tuple of (Axes, Line2D) for the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

        k, power_mean, power_std_lower, power_std_upper = self.fft_power_1d(direction)

        # Filter out zero/negative values for log plot
        if loglog:
            valid = (k > 0) & (power_mean > 0)
            k_plot = k[valid]
            power_plot = power_mean[valid]
            std_lower_plot = np.maximum(power_std_lower[valid], 1e-50)  # Avoid log(0)
            std_upper_plot = power_std_upper[valid]

            line = ax.loglog(k_plot, power_plot, color=line_color, **kwargs)[0]

            if show_std:
                fc = fill_color if fill_color else line.get_color()
                ax.fill_between(
                    k_plot, std_lower_plot, std_upper_plot, alpha=fill_alpha, color=fc
                )
        else:
            line = ax.plot(k, power_mean, color=line_color, **kwargs)[0]

            if show_std:
                fc = fill_color if fill_color else line.get_color()
                ax.fill_between(
                    k, power_std_lower, power_std_upper, alpha=fill_alpha, color=fc
                )

        default_title = rf"{self.name} 1D power spectrum (along {direction}) at time {round(self.time, self._time_ndecimals)} $\omega_{{ci}}^{{-1}}$"
        ax.set_title(title if title else default_title)
        ax.set_xlabel(xlabel if xlabel else r"$k \cdot d_i$")
        ax.set_ylabel(ylabel if ylabel else r"Power")

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.grid(True, alpha=0.3)

        return ax, line

    def plot_1d_avg(
        self,
        direction: Literal["x", "y", "z"] = "x",
        *,
        ax: Optional[Axes] = None,
        dpi: int = 100,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        fill_alpha: float = 0.3,
        fill_color: Optional[str] = None,
        line_color: Optional[str] = None,
        show_std: bool = True,
        **kwargs,
    ) -> Tuple[Axes, Line2D]:
        """
        Plot 1D average along a chosen direction with fill_between showing standard deviation.

        Averages the data along all axes perpendicular to the specified direction,
        then plots the mean with a shaded region representing ± one standard deviation.

        Args:
            direction: The direction along which to plot ("x", "y", or "z").
                       The data is averaged over all other dimensions.
            ax: Matplotlib Axes instance.
            dpi: Resolution of the plot.
            title: Plot title.
            xlabel, ylabel: Axis labels.
            xlim, ylim: Axis limits.
            fill_alpha: Alpha (transparency) for the standard deviation fill region.
            fill_color: Color for the fill region. Defaults to match line color.
            line_color: Color for the mean line.
            show_std: Whether to show the standard deviation fill region.
            **kwargs: Additional keyword arguments for the plot function.

        Returns:
            Tuple of (Axes, Line2D) for the plot.
        """
        num_dimensions = len(self._get_data_shape())

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

        # Get processed data
        coord_data, mean_data, std_lower, std_upper = self.avg_1d(direction)

        # Determine default xlabel based on direction
        default_xlabel = {"x": self._X, "y": self._Y, "z": self._Z}[direction]

        # Get coordinate limits
        if direction == "x":
            coord_lim = self.xlimdata
        elif direction == "y":
            coord_lim = self.ylimdata
        else:  # z
            coord_lim = self.zlimdata
        coord_lim = self._materialize(coord_lim)

        # Plot the mean line
        line = ax.plot(coord_data, mean_data, color=line_color, **kwargs)[0]

        # Add fill_between for standard deviation
        if show_std and num_dimensions > 1:
            fc = fill_color if fill_color else line.get_color()
            ax.fill_between(
                coord_data, std_lower, std_upper, alpha=fill_alpha, color=fc
            )

        # Set labels and title
        default_title = rf"{self.name} (avg along {direction}) at time {round(self.time, self._time_ndecimals)} $\omega_{{ci}}^{{-1}}$"
        ax.set_title(title if title else default_title)
        ax.set_xlabel(xlabel if xlabel else default_xlabel)
        ax.set_ylabel(ylabel if ylabel else f"{self.name}")
        ax.set_xlim(xlim if xlim else coord_lim)
        if ylim:
            ax.set_ylim(ylim)

        return ax, line

    def plot(
        self,
        *,
        ax: Optional[Axes] = None,
        dpi: int = 100,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        zlim: Optional[tuple] = None,
        colormap: str = "viridis",
        show_colorbar: bool = True,
        colorbar_label: Optional[str] = None,
        slice_axis: Literal["x", "y", "z"] = "x",
        context_3d: bool = True,
        **kwargs,
    ) -> Tuple[Axes, Union[Line2D, QuadMesh, AxesImage]]:
        """
        Plot 1D, 2D, or 3D data.

        Args:
            ax: Matplotlib Axes instance.
            dpi: Resolution of the plot.
            title: Plot title.
            xlabel, ylabel, zlabel: Axis labels.
            xlim, ylim, zlim: Axis limits.
            colormap: Colormap name for 2D/3D data.
            show_colorbar: Whether to display the colorbar.
            colorbar_label: Label for the colorbar.
            slice_axis: Slice axis for 3D data. Must be "x", "y", or "z".
                The left/right arrow keys step the slice by one.
            context_3d: For 3D data, also show a rotatable cube of the
                volume's outer faces with a frame marking the current slice.
                Ignored when `ax` is given.
            **kwargs: Additional keyword arguments for the plotting functions.

        Returns:
            Matplotlib Axes and plot object. For 3D data the slice is drawn
            with imshow, so the plot object is an AxesImage.
        """

        num_dimensions = len(self._get_data_shape())
        if not 1 <= num_dimensions <= 3:
            raise NotImplementedError("Plotting is restricted to 1D, 2D, or 3D data.")

        ax3d = None
        if ax is None:
            if num_dimensions == 3 and context_3d:
                fig = plt.figure(figsize=(12, 5.5), dpi=dpi)
                ax = fig.add_subplot(1, 2, 1)
                ax3d = fig.add_subplot(1, 2, 2, projection="3d")
                plt.subplots_adjust(bottom=0.2, wspace=0.05)
            else:
                fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
                if num_dimensions == 3:
                    plt.subplots_adjust(bottom=0.2)
        else:
            fig = ax.figure

        # For 3D we only need partial slices, so defer materializing `data` to
        # the slicing helper. For 1D/2D the full array is the plot.
        if num_dimensions < 3:
            data = self._materialize(self.data)
        xdata = self._materialize(self.xdata)
        xlimdata = self._materialize(self.xlimdata)

        if num_dimensions == 1:
            line = ax.plot(xdata, data, **kwargs)[0]
            ax.set_title(title if title else self._plot_title)
            ax.set_xlabel(xlabel if xlabel else "$x$")
            ax.set_ylabel(f"{self.name}")
            ax.set_xlim(xlim if xlim else xlimdata)

            return ax, line
        elif num_dimensions == 2:
            ydata = self._materialize(self.ydata)
            ylimdata = self._materialize(self.ylimdata)
            X, Y = np.meshgrid(xdata, ydata, indexing="ij")
            mesh = ax.pcolormesh(X, Y, data, cmap=colormap, shading="auto", **kwargs)
            ax.set_title(title if title else self._plot_title)
            xlabel_default, ylabel_default = self._axis_labels(self.name)
            ax.set_xlabel(xlabel if xlabel else xlabel_default)
            ax.set_ylabel(ylabel if ylabel else ylabel_default)
            ax.set_xlim(xlim if xlim else xlimdata)
            ax.set_ylim(ylim if ylim else ylimdata)
            if show_colorbar:
                cbar = plt.colorbar(mesh, ax=ax)
                cbar.set_label(colorbar_label if colorbar_label else f"{self.name}")

            return ax, mesh
        else:
            if slice_axis not in ["x", "y", "z"]:
                raise ValueError("Slice axis must be 'x', 'y', or 'z'.")

            ydata = self._materialize(self.ydata)
            ylimdata = self._materialize(self.ylimdata)
            zdata = self._materialize(self.zdata)
            zlimdata = self._materialize(self.zlimdata)

            axis_coords = {"x": xdata, "y": ydata, "z": zdata}[slice_axis]
            if slice_axis == "x":
                extent = (*ylimdata, *zlimdata)
                ax.set_xlabel(ylabel if ylabel else "$y$")
                ax.set_ylabel(zlabel if zlabel else "$z$")
                ax.set_xlim(ylim if ylim else ylimdata)
                ax.set_ylim(zlim if zlim else zlimdata)
            elif slice_axis == "y":
                extent = (*xlimdata, *zlimdata)
                ax.set_xlabel(xlabel if xlabel else "$x$")
                ax.set_ylabel(zlabel if zlabel else "$z$")
                ax.set_xlim(xlim if xlim else xlimdata)
                ax.set_ylim(zlim if zlim else zlimdata)
            else:
                extent = (*xlimdata, *ylimdata)
                ax.set_xlabel(xlabel if xlabel else "$x$")
                ax.set_ylabel(ylabel if ylabel else "$y$")
                ax.set_xlim(xlim if xlim else xlimdata)
                ax.set_ylim(ylim if ylim else ylimdata)

            initial_slice = 0
            initial_data_slice = self._read_2d_slice(slice_axis, initial_slice)
            # imshow instead of pcolormesh: the grid is uniform, and swapping
            # the image data per slider tick is far cheaper than re-rendering
            # a quad mesh
            kwargs.setdefault("origin", "lower")
            kwargs.setdefault("interpolation", "nearest")
            kwargs.setdefault("aspect", "auto")
            mesh = ax.imshow(
                initial_data_slice.T,
                extent=extent,
                cmap=colormap,
                **kwargs,
            )

            initial_position_str = (
                f"\n{slice_axis} = {axis_coords[initial_slice]:.2f}"
            )
            ax.set_title(
                title if title else f"{self._plot_title}{initial_position_str}"
            )
            if show_colorbar:
                cbar = plt.colorbar(mesh, ax=ax)
                cbar.set_label(colorbar_label if colorbar_label else f"{self.name}")

            if ax3d is not None:
                draw_marker, cull_back_faces = _add_slice_context(
                    ax3d, self, slice_axis, colormap
                )
                draw_marker(initial_slice)

                def on_motion(event) -> None:
                    if event.inaxes is ax3d and cull_back_faces():
                        fig.canvas.draw_idle()

                fig.canvas.mpl_connect("motion_notify_event", on_motion)
            else:
                draw_marker = None

            ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
            full_shape = self._get_data_shape()
            n_along = full_shape[{"x": 0, "y": 1, "z": 2}[slice_axis]]
            slider = Slider(
                ax_slider,
                f"{slice_axis.capitalize()} axis slice",
                0,
                n_along - 1,
                valinit=initial_slice,
                valstep=1,
            )

            def update(val: float) -> None:
                slice_index = int(slider.val)
                data_slice = self._read_2d_slice(slice_axis, slice_index)
                position_str = f"\n{slice_axis} = {axis_coords[slice_index]:.2f}"
                ax.set_title(title if title else f"{self._plot_title}{position_str}")
                mesh.set_data(data_slice.T)
                # Rescale color limits to the new slice's data range so the
                # colorbar reflects what is being shown.
                mesh.set_clim(float(data_slice.min()), float(data_slice.max()))
                if draw_marker is not None:
                    draw_marker(slice_index)
                fig.canvas.draw_idle()

            slider.on_changed(update)

            def step_slice(event) -> None:
                if event.key == "right":
                    slider.set_val(min(int(slider.val) + 1, n_along - 1))
                elif event.key == "left":
                    slider.set_val(max(int(slider.val) - 1, 0))
                else:
                    # keep matplotlib's other default shortcuts working
                    key_press_handler(
                        event, fig.canvas, getattr(fig.canvas, "toolbar", None)
                    )

            # matplotlib's default handler binds left/right to view history;
            # replace it so the arrow keys step through slices instead
            manager = fig.canvas.manager
            if manager is not None:
                fig.canvas.mpl_disconnect(manager.key_press_handler_id)
            fig.canvas.mpl_connect("key_press_event", step_slice)

            # Keep a strong reference to the slider on the Figure so it
            # isn't garbage collected after this function returns (which
            # would silently break the widget on some matplotlib backends).
            if not hasattr(fig, "_dhybridrpy_widgets"):
                fig._dhybridrpy_widgets = []
            fig._dhybridrpy_widgets.append(slider)
            return ax, mesh


class Field(Data):
    def __init__(
        self,
        file_path: str,
        name: str,
        timestep: int,
        time: float,
        time_ndecimals: int,
        lazy: bool,
        field_type: str,
    ):
        super().__init__(file_path, name, timestep, time, time_ndecimals, lazy)
        self.type = field_type  # The type of field, e.g., "External"
        self._plot_title += f" (type = {self.type})"

    def _check_compatibility(self, other):
        super()._check_compatibility(other)

        if self.type != other.type:
            raise ValueError("Field types do not match.")

    def _extra_init_args(self):
        return (self.type,)


class Phase(Data):
    def __init__(
        self,
        file_path: str,
        name: str,
        timestep: int,
        time: float,
        time_ndecimals: int,
        lazy: bool,
        species: Union[int, str],
    ):
        super().__init__(file_path, name, timestep, time, time_ndecimals, lazy)
        self.species = species
        self._plot_title += f" (species = {self.species})"

    def _check_compatibility(self, other):
        super()._check_compatibility(other)

        if self.species != other.species:
            raise ValueError("Phase species do not match.")

    def _extra_init_args(self):
        return (self.species,)


class Raw(BaseProperties):
    def __init__(
        self,
        file_path: str,
        name: str,
        timestep: int,
        time: float,
        lazy: bool,
        species: int,
    ):
        super().__init__(file_path, name, timestep, time, lazy)
        self.species = species

    def keys(self) -> list:
        """Dataset names in the raw file, without reading any data."""
        with open_h5(self.file_path) as file:
            return list(file.keys())

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def __getitem__(self, key: str) -> Union[np.ndarray, da.Array]:
        """Read a single dataset, e.g. raw["ene"]."""
        with open_h5(self.file_path) as file:
            if key not in file:
                raise KeyError(
                    f"Dataset '{key}' not found in {self.file_path}; "
                    f"available: {sorted(file.keys())}"
                )
            if not self.lazy:
                return file[key][:]
            shape = file[key].shape
            dtype = file[key].dtype

        def loader(k=key):
            with open_h5(self.file_path) as f:
                return f[k][:]

        return da.from_delayed(delayed(loader)(), shape=shape, dtype=dtype)

    def load(self, keys=None, workers: int = None) -> dict:
        """Read several datasets at once using the process worker pool.

        h5py cannot overlap reads across threads, so the datasets are read
        by separate processes. Args: keys (all datasets when None) and
        workers (pool size). Eager only; for lazy access use `dict` or
        indexing.
        """
        from . import _parallel

        if self.lazy:
            raise ValueError("load() reads eagerly; use dict or [] when lazy.")
        available = self.keys()
        if keys is None:
            keys = available
        else:
            missing = [key for key in keys if key not in available]
            if missing:
                raise KeyError(
                    f"Dataset(s) {missing} not found in {self.file_path}; "
                    f"available: {sorted(available)}"
                )
        pool = _parallel.get_pool(workers)
        futures = {
            key: pool.submit(_parallel.read_dataset, self.file_path, key)
            for key in keys
        }
        return {key: future.result() for key, future in futures.items()}

    @property
    def dict(self) -> dict:
        """Retrieve a dictionary of the raw file's keys and values.

        Always re-reads from HDF5 (or returns fresh dask-delayed views) to
        avoid OOM when iterating over many timesteps. Matches Data.data.
        """
        result = {}
        with open_h5(self.file_path) as file:
            for key in file.keys():
                if self.lazy:
                    shape = file[key].shape
                    dtype = file[key].dtype

                    def dict_helper(key=key):
                        with open_h5(self.file_path) as f:
                            return f[key][:]

                    delayed_helper = delayed(dict_helper)()
                    result[key] = da.from_delayed(
                        delayed_helper, shape=shape, dtype=dtype
                    )
                else:
                    result[key] = file[key][:]
        return result
