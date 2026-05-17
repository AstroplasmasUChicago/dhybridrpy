import h5py
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import operator
from matplotlib.widgets import Slider

from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D
from typing import Tuple, Union, Optional, Literal
from dask.delayed import delayed


# Standalone functions for numpy arrays


def fft_power_iso(
    data: np.ndarray,
    Lx: float,
    Ly: Optional[float] = None,
    Lz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute isotropic FFT power spectrum of a numpy array.

    Args:
        data: 1D, 2D, or 3D numpy array
        Lx: Box size in x direction
        Ly: Box size in y direction (required for 2D/3D)
        Lz: Box size in z direction (required for 3D)

    Returns:
        Tuple of (k, power) where:
            - k: 1D array of wavenumber values (in units of 2π/L)
            - power: 1D array of power spectral density at each k
    """
    num_dimensions = data.ndim
    if num_dimensions < 1 or num_dimensions > 3:
        raise NotImplementedError("fft_power_iso only supports 1D, 2D, or 3D data.")

    if num_dimensions == 1:
        nx = data.shape[0]
        dx = Lx / nx

        fft_data = np.fft.fft(data)
        power = np.abs(fft_data) ** 2 / nx * dx

        k = np.fft.fftfreq(nx, d=dx) * 2 * np.pi

        # Define bins (matching 2D approach)
        k_max = np.abs(k).max()
        dk = 2 * np.pi / Lx
        k_bins = np.arange(0, k_max + dk, dk)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        # Bin the spectrum
        binned_power, _ = np.histogram(np.abs(k), bins=k_bins, weights=power)

        return k_centers, binned_power

    elif num_dimensions == 2:
        if Ly is None:
            raise ValueError("Ly required for 2D data")
        nx, ny = data.shape

        dx = Lx / nx
        dy = Ly / ny

        # Compute FFT with proper normalization
        # Parseval normalization + physical units
        fft_data = np.fft.fft2(data)
        power_2d = np.abs(fft_data) ** 2 / (nx * ny) * dx * dy

        # Grid
        kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(KX**2 + KY**2)

        # Define Bins
        k_max = min(np.abs(kx).max(), np.abs(ky).max())
        dk = 2 * np.pi / max(Lx, Ly)
        k_bins = np.arange(0, k_max + dk, dk)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        # Binning
        K_flat = K.ravel()
        P_flat = power_2d.ravel()

        # Sum the power in each radial bin (energy spectrum)
        radial_sum, _ = np.histogram(K_flat, bins=k_bins, weights=P_flat)

        power_radial = radial_sum

        return k_centers, power_radial

    elif num_dimensions == 3:
        if Ly is None:
            raise ValueError("Ly required for 3D data")
        if Lz is None:
            raise ValueError("Lz required for 3D data")
        nx, ny, nz = data.shape

        dx = Lx / nx
        dy = Ly / ny
        dz = Lz / nz

        # Compute FFT with proper normalization
        # Parseval normalization + physical units
        fft_data = np.fft.fftn(data)
        power_3d = np.abs(fft_data) ** 2 / (nx * ny * nz) * dx * dy * dz

        # Grid
        kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
        kz = np.fft.fftfreq(nz, d=dz) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        K = np.sqrt(KX**2 + KY**2 + KZ**2)

        # Define Bins
        k_max = min(np.abs(kx).max(), np.abs(ky).max(), np.abs(kz).max())
        dk = 2 * np.pi / max(Lx, Ly, Lz)
        k_bins = np.arange(0, k_max + dk, dk)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        # Binning
        K_flat = K.ravel()
        P_flat = power_3d.ravel()

        # Sum the power in each radial bin (energy spectrum)
        radial_sum, _ = np.histogram(K_flat, bins=k_bins, weights=P_flat)

        power_radial = radial_sum

        return k_centers, power_radial

    raise RuntimeError("Unexpected number of dimensions in fft_power_iso.")


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

    # Move FFT axis to the end and collapse the remaining dims into "slices"
    power = np.moveaxis(power, axis, -1)
    power_spectra = power.reshape(-1, power.shape[-1])

    # Geometric statistics in log space, computed only over positive entries
    # per bin. Bins where every slice has zero power return 0 for mean/lo/hi
    # rather than the previous hardcoded 1e-50 floor.
    positive = power_spectra > 0
    with np.errstate(divide="ignore"):
        log_power = np.where(positive, np.log(power_spectra), 0.0)

    count = positive.sum(axis=0)
    sum_log = log_power.sum(axis=0)
    sum_log_sq = (log_power ** 2).sum(axis=0)

    safe_count = np.where(count > 0, count, 1)
    log_mean = sum_log / safe_count
    log_var = sum_log_sq / safe_count - log_mean ** 2
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

    def _get_coordinate_limits(self, axis_name: str) -> np.ndarray:
        key = f"{axis_name} lims"
        if key not in self._data_dict:
            with h5py.File(self.file_path, "r") as file:
                self._data_dict[key] = file["AXIS"][axis_name][:]
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
            arr = self._materialize(self._data_dict[self.name])
            if slice_axis == "x":
                return arr[idx, :, :]
            if slice_axis == "y":
                return arr[:, idx, :]
            return arr[:, :, idx]

        # HDF5 stores (nz, ny, nx); after the .T in `data`, the numpy convention
        # is (nx, ny, nz). Mirror that here while reading only what's needed.
        with h5py.File(self.file_path, "r") as f:
            ds = f["DATA"]
            if slice_axis == "x":
                return np.asarray(ds[:, :, idx]).T  # -> (ny, nz)
            if slice_axis == "y":
                return np.asarray(ds[:, idx, :]).T  # -> (nx, nz)
            return np.asarray(ds[idx, :, :]).T  # -> (nx, ny)

    def _materialize(
        self, arr: Union[np.ndarray, da.Array]
    ) -> np.ndarray:
        """Return a numpy array for `arr`, calling .compute() if it's a dask array.

        Callers should bind `self.data` (or another property) once, then pass it
        in. Accessing `self.data` is a real HDF5 read in non-lazy mode.
        """
        if self.lazy and isinstance(arr, da.Array):
            return arr.compute()
        return arr

    def _get_data_shape(self) -> Tuple[int, ...]:
        """Retrieve the shape of the data without loading it."""
        if self._data_shape is None:
            with h5py.File(self.file_path, "r") as file:
                # Reverse the data shape to be consistent with transpose in data @property
                self._data_shape = file["DATA"].shape[::-1]
        return self._data_shape

    def _get_data_dtype(self) -> np.dtype:
        """Retrieve the type of the data without loading it."""
        if self._data_dtype is None:
            with h5py.File(self.file_path, "r") as file:
                self._data_dtype = file["DATA"].dtype
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
            with h5py.File(self.file_path, "r") as f:
                return f["DATA"][:].T

        if self.lazy:
            delayed_obj = delayed(loader)()
            return da.from_delayed(
                delayed_obj,
                shape=self._get_data_shape(),
                dtype=self._get_data_dtype(),
            )
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
            result = op(self.data, other.data)
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
        inst._data_dict = {k: v for k, v in self._data_dict.items() if "AXIS" in k}
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
            other - self.data, "",
            f"{self._short_operand_name(other)}-{self.name}",
            self,
        )

    def __rtruediv__(self, other):
        return self._create_new_instance(
            other / self.data, "",
            f"{self._short_operand_name(other)}/{self.name}",
            self,
        )

    # Ensure that mixed Data and NumPy operations produce a Data object
    __array_priority__ = 20

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Allow NumPy ufuncs to be applied to Data objects"""

        if method != "__call__":
            return NotImplemented  # only allow element‑wise operations

        # Extract raw arrays and gather Data operands for naming / compat checks
        raw_inputs, data_operands = [], []
        for input in inputs:
            if isinstance(input, Data):
                data_operands.append(input)
                raw_inputs.append(input.data)
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
        **kwargs,
    ) -> Tuple[Axes, Union[Line2D, QuadMesh]]:
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
            **kwargs: Additional keyword arguments for the plotting functions.

        Returns:
            Matplotlib Axes and plot object.
        """

        num_dimensions = len(self._get_data_shape())
        if not 1 <= num_dimensions <= 3:
            raise NotImplementedError("Plotting is restricted to 1D, 2D, or 3D data.")

        if ax is None:
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

            initial_slice = 0
            initial_data_slice = self._read_2d_slice(slice_axis, initial_slice)
            if slice_axis == "x":
                A, B = np.meshgrid(ydata, zdata, indexing="ij")
                initial_position_str = f"\nx = {xdata[initial_slice]:.2f}"
                ax.set_xlabel(ylabel if ylabel else "$y$")
                ax.set_ylabel(zlabel if zlabel else "$z$")
                ax.set_xlim(ylim if ylim else ylimdata)
                ax.set_ylim(zlim if zlim else zlimdata)
            elif slice_axis == "y":
                A, B = np.meshgrid(xdata, zdata, indexing="ij")
                initial_position_str = f"\ny = {ydata[initial_slice]:.2f}"
                ax.set_xlabel(xlabel if xlabel else "$x$")
                ax.set_ylabel(zlabel if zlabel else "$z$")
                ax.set_xlim(xlim if xlim else xlimdata)
                ax.set_ylim(zlim if zlim else zlimdata)
            else:
                A, B = np.meshgrid(xdata, ydata, indexing="ij")
                initial_position_str = f"\nz = {zdata[initial_slice]:.2f}"
                ax.set_xlabel(xlabel if xlabel else "$x$")
                ax.set_ylabel(ylabel if ylabel else "$y$")
                ax.set_xlim(xlim if xlim else xlimdata)
                ax.set_ylim(ylim if ylim else ylimdata)

            mesh = ax.pcolormesh(
                A, B, initial_data_slice,
                cmap=colormap, shading="auto", **kwargs,
            )

            ax.set_title(
                title if title else f"{self._plot_title}{initial_position_str}"
            )
            if show_colorbar:
                cbar = plt.colorbar(mesh, ax=ax)
                cbar.set_label(colorbar_label if colorbar_label else f"{self.name}")

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
                if slice_axis == "x":
                    position_str = f"\nx = {xdata[slice_index]:.2f}"
                elif slice_axis == "y":
                    position_str = f"\ny = {ydata[slice_index]:.2f}"
                else:
                    position_str = f"\nz = {zdata[slice_index]:.2f}"

                ax.set_title(title if title else f"{self._plot_title}{position_str}")
                mesh.set_array(data_slice.ravel())
                # Rescale color limits to the new slice's data range so the
                # colorbar reflects what is being shown.
                mesh.set_clim(float(data_slice.min()), float(data_slice.max()))
                fig.canvas.draw_idle()

            slider.on_changed(update)
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

    @property
    def dict(self) -> dict:
        """Retrieve a dictionary of the raw file's keys and values.

        Always re-reads from HDF5 (or returns fresh dask-delayed views) to
        avoid OOM when iterating over many timesteps. Matches Data.data.
        """
        result = {}
        with h5py.File(self.file_path, "r") as file:
            for key in file.keys():
                if self.lazy:
                    shape = file[key].shape
                    dtype = file[key].dtype

                    def dict_helper(key=key):
                        with h5py.File(self.file_path, "r") as f:
                            return f[key][:]

                    delayed_helper = delayed(dict_helper)()
                    result[key] = da.from_delayed(
                        delayed_helper, shape=shape, dtype=dtype
                    )
                else:
                    result[key] = file[key][:]
        return result
