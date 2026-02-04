import h5py
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import operator
from matplotlib.widgets import Slider
from collections import defaultdict

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

    # Wavenumber array (positive frequencies only)
    k_full = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
    pos_mask = k_full >= 0
    k = k_full[pos_mask]

    # Extract slices
    if num_dimensions == 1:
        slices = [data]
    elif num_dimensions == 2:
        if axis == 0:
            slices = [data[:, j] for j in range(data.shape[1])]
        else:
            slices = [data[i, :] for i in range(data.shape[0])]
    else:  # 3D
        if axis == 0:
            slices = [
                data[:, j, kk]
                for j in range(data.shape[1])
                for kk in range(data.shape[2])
            ]
        elif axis == 1:
            slices = [
                data[i, :, kk]
                for i in range(data.shape[0])
                for kk in range(data.shape[2])
            ]
        else:
            slices = [
                data[i, j, :]
                for i in range(data.shape[0])
                for j in range(data.shape[1])
            ]

    # Compute power spectrum for each slice
    power_spectra = []
    for slice_data in slices:
        fft_data = np.fft.fft(slice_data)
        power = np.abs(fft_data) ** 2 / n
        power = power[pos_mask]
        power[1:] *= 2
        power_spectra.append(power)

    power_spectra = np.array(power_spectra)

    # Compute statistics in log space
    power_spectra_safe = np.maximum(power_spectra, 1e-50)
    log_power = np.log(power_spectra_safe)

    log_mean = np.mean(log_power, axis=0)
    log_std = np.std(log_power, axis=0)

    power_mean = np.exp(log_mean)
    power_std_lower = np.exp(log_mean - log_std)
    power_std_upper = np.exp(log_mean + log_std)

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

    _LABEL_MAPPINGS = defaultdict(
        lambda: (Data._X, Data._Y),
        {
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
        },
    )

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
        if self.name not in self._data_dict:

            def loader():
                with h5py.File(self.file_path, "r") as f:
                    return f["DATA"][:].T

            if self.lazy:
                delayed_obj = delayed(loader)()
                self._data_dict[self.name] = da.from_delayed(
                    delayed_obj,
                    shape=self._get_data_shape(),
                    dtype=self._get_data_dtype(),
                )
            else:
                self._data_dict[self.name] = loader()
        return self._data_dict[self.name]

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

    def _check_compatability(self, other) -> None:
        """Raise if 'self' and 'other' cannot be operated on together."""

        if self._get_data_shape() != other._get_data_shape():
            raise ValueError(
                f"Incompatible grid shapes: {self._get_data_shape()} vs "
                f"{other._get_data_shape()}"
            )
        if self.timestep != other.timestep:
            raise ValueError(f"Timesteps differ: {self.timestep} vs {other.timestep}")

    def _apply_operation(self, other, op):
        """Apply a binary operation to self and another Data object or scalar."""

        if isinstance(other, Data):
            self._check_compatability(other)
            result = op(self.data, other.data)
            other_name = other.name
        else:
            result = op(self.data, other)
            other_name = str(other)

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
        return self * (-1)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __rtruediv__(self, other):
        return self.__pow__(-1) * other

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
                ref._check_compatability(other)

        # Execute the ufunc on the underlying arrays
        result_array = ufunc(*raw_inputs, **kwargs)

        # No Data operands, so return the non-Data result
        if not data_operands:
            return result_array

        # Build a descriptive name: e.g. "sin(By)"
        names = ",".join(
            obj.name if isinstance(obj, Data) else str(obj) for obj in inputs
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

        def is_computable(arr: Union[np.ndarray, da.Array]) -> bool:
            return self.lazy and isinstance(arr, da.Array)

        data = self.data.compute() if is_computable(self.data) else self.data

        # Determine which axis corresponds to the direction and compute mean/std
        if num_dimensions == 1:
            # For 1D data, just return as-is (no averaging needed)
            coord_data = (
                self.xdata.compute() if is_computable(self.xdata) else self.xdata
            )
            mean_data = data
            std_data = np.zeros_like(data)
        elif num_dimensions == 2:
            # For 2D data: shape is (nx, ny)
            # x -> axis 0, y -> axis 1
            if direction == "x":
                # Average over y (axis 1)
                mean_data = np.mean(data, axis=1)
                std_data = np.std(data, axis=1)
                coord_data = (
                    self.xdata.compute() if is_computable(self.xdata) else self.xdata
                )
            elif direction == "y":
                # Average over x (axis 0)
                mean_data = np.mean(data, axis=0)
                std_data = np.std(data, axis=0)
                coord_data = (
                    self.ydata.compute() if is_computable(self.ydata) else self.ydata
                )
            else:  # z
                raise ValueError(
                    "Cannot average along 'z' for 2D data. Use 'x' or 'y'."
                )
        elif num_dimensions == 3:
            # For 3D data: shape is (nx, ny, nz)
            # x -> axis 0, y -> axis 1, z -> axis 2
            if direction == "x":
                # Average over y and z (axes 1 and 2)
                mean_data = np.mean(data, axis=(1, 2))
                std_data = np.std(data, axis=(1, 2))
                coord_data = (
                    self.xdata.compute() if is_computable(self.xdata) else self.xdata
                )
            elif direction == "y":
                # Average over x and z (axes 0 and 2)
                mean_data = np.mean(data, axis=(0, 2))
                std_data = np.std(data, axis=(0, 2))
                coord_data = (
                    self.ydata.compute() if is_computable(self.ydata) else self.ydata
                )
            else:  # z
                # Average over x and y (axes 0 and 1)
                mean_data = np.mean(data, axis=(0, 1))
                std_data = np.std(data, axis=(0, 1))
                coord_data = (
                    self.zdata.compute() if is_computable(self.zdata) else self.zdata
                )
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

        def is_computable(arr):
            return self.lazy and isinstance(arr, da.Array)

        data = self.data.compute() if is_computable(self.data) else self.data
        num_dimensions = data.ndim

        # Get box sizes from coordinate limits
        xlim = (
            self.xlimdata.compute() if is_computable(self.xlimdata) else self.xlimdata
        )
        Lx = xlim[1] - xlim[0]

        Ly = None
        Lz = None
        if num_dimensions >= 2:
            ylim = (
                self.ylimdata.compute()
                if is_computable(self.ylimdata)
                else self.ylimdata
            )
            Ly = ylim[1] - ylim[0]
        if num_dimensions >= 3:
            zlim = (
                self.zlimdata.compute()
                if is_computable(self.zlimdata)
                else self.zlimdata
            )
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

        def is_computable(arr):
            return self.lazy and isinstance(arr, da.Array)

        data = self.data.compute() if is_computable(self.data) else self.data

        # Get box size along the chosen direction
        if direction == "x":
            lim = (
                self.xlimdata.compute()
                if is_computable(self.xlimdata)
                else self.xlimdata
            )
        elif direction == "y":
            lim = (
                self.ylimdata.compute()
                if is_computable(self.ylimdata)
                else self.ylimdata
            )
        else:  # z
            lim = (
                self.zlimdata.compute()
                if is_computable(self.zlimdata)
                else self.zlimdata
            )
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
        def is_computable(arr: Union[np.ndarray, da.Array]) -> bool:
            return self.lazy and isinstance(arr, da.Array)

        if direction == "x":
            coord_lim = self.xlimdata
        elif direction == "y":
            coord_lim = self.ylimdata
        else:  # z
            coord_lim = self.zlimdata
        coord_lim = coord_lim.compute() if is_computable(coord_lim) else coord_lim

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

        def is_computable(arr: Union[np.ndarray, da.Array]) -> bool:
            return self.lazy and isinstance(arr, da.Array)

        data = self.data.compute() if is_computable(self.data) else self.data
        xdata = self.xdata.compute() if is_computable(self.xdata) else self.xdata
        xlimdata = (
            self.xlimdata.compute() if is_computable(self.xlimdata) else self.xlimdata
        )

        if num_dimensions == 1:
            line = ax.plot(xdata, data, **kwargs)[0]
            ax.set_title(title if title else self._plot_title)
            ax.set_xlabel(xlabel if xlabel else "$x$")
            ax.set_ylabel(f"{self.name}")
            ax.set_xlim(xlim if xlim else xlimdata)

            return ax, line
        elif num_dimensions == 2:
            ydata = self.ydata.compute() if is_computable(self.ydata) else self.ydata
            ylimdata = (
                self.ylimdata.compute()
                if is_computable(self.ylimdata)
                else self.ylimdata
            )
            X, Y = np.meshgrid(xdata, ydata, indexing="ij")
            mesh = ax.pcolormesh(X, Y, data, cmap=colormap, shading="auto", **kwargs)
            ax.set_title(title if title else self._plot_title)
            xlabel_default, ylabel_default = self._LABEL_MAPPINGS[self.name]
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

            ydata = self.ydata.compute() if is_computable(self.ydata) else self.ydata
            ylimdata = (
                self.ylimdata.compute()
                if is_computable(self.ylimdata)
                else self.ylimdata
            )
            zdata = self.zdata.compute() if is_computable(self.zdata) else self.zdata
            zlimdata = (
                self.zlimdata.compute()
                if is_computable(self.zlimdata)
                else self.zlimdata
            )

            initial_slice = 0
            if slice_axis == "x":
                Y, Z = np.meshgrid(ydata, zdata, indexing="ij")
                mesh = ax.pcolormesh(
                    Y,
                    Z,
                    data[initial_slice, :, :],
                    cmap=colormap,
                    shading="auto",
                    **kwargs,
                )
                initial_position_str = f"\nx = {xdata[initial_slice]:.2f}"
                ax.set_xlabel(ylabel if ylabel else "$y$")
                ax.set_ylabel(zlabel if zlabel else "$z$")
                ax.set_xlim(ylim if ylim else ylimdata)
                ax.set_ylim(zlim if zlim else zlimdata)
            elif slice_axis == "y":
                X, Z = np.meshgrid(xdata, zdata, indexing="ij")
                mesh = ax.pcolormesh(
                    X,
                    Z,
                    data[:, initial_slice, :],
                    cmap=colormap,
                    shading="auto",
                    **kwargs,
                )
                initial_position_str = f"\ny = {ydata[initial_slice]:.2f}"
                ax.set_xlabel(xlabel if xlabel else "$x$")
                ax.set_ylabel(zlabel if zlabel else "$z$")
                ax.set_xlim(xlim if xlim else xlimdata)
                ax.set_ylim(zlim if zlim else zlimdata)
            else:
                X, Y = np.meshgrid(xdata, ydata, indexing="ij")
                mesh = ax.pcolormesh(
                    X,
                    Y,
                    data[:, :, initial_slice],
                    cmap=colormap,
                    shading="auto",
                    **kwargs,
                )
                initial_position_str = f"\nz = {zdata[initial_slice]:.2f}"
                ax.set_xlabel(xlabel if xlabel else "$x$")
                ax.set_ylabel(ylabel if ylabel else "$y$")
                ax.set_xlim(xlim if xlim else xlimdata)
                ax.set_ylim(ylim if ylim else ylimdata)

            ax.set_title(
                title if title else f"{self._plot_title}{initial_position_str}"
            )
            if show_colorbar:
                cbar = plt.colorbar(mesh, ax=ax)
                cbar.set_label(colorbar_label if colorbar_label else f"{self.name}")

            ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
            data_shape = data.shape[{"x": 0, "y": 1, "z": 2}[slice_axis]]
            slider = Slider(
                ax_slider,
                f"{slice_axis.capitalize()} axis slice",
                0,
                data_shape - 1,
                valinit=initial_slice,
                valstep=1,
            )

            def update(val: float) -> None:
                slice_index = int(slider.val)
                if slice_axis == "x":
                    data_slice = data[slice_index, :, :]
                    position_str = f"\nx = {xdata[slice_index]:.2f}"
                elif slice_axis == "y":
                    data_slice = data[:, slice_index, :]
                    position_str = f"\ny = {ydata[slice_index]:.2f}"
                else:
                    data_slice = data[:, :, slice_index]
                    position_str = f"\nz = {zdata[slice_index]:.2f}"

                ax.set_title(title if title else f"{self._plot_title}{position_str}")
                mesh.set_array(data_slice.ravel())
                fig.canvas.draw_idle()

            slider.on_changed(update)
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

    def _check_compatability(self, other):
        super()._check_compatability(other)
        if isinstance(other, Field) and self.type != other.type:
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

    def _check_compatability(self, other):
        super()._check_compatability(other)
        if isinstance(other, Phase) and self.species != other.species:
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
        """Retrieve a dictionary of the raw file's keys and values."""
        if not self._data_dict:
            with h5py.File(self.file_path, "r") as file:

                def dict_helper():
                    with h5py.File(self.file_path, "r") as f:
                        return f[key][:]

                for key in file.keys():
                    if self.lazy:
                        shape = file[key].shape
                        dtype = file[key].dtype
                        delayed_helper = delayed(dict_helper)()
                        self._data_dict[key] = da.from_delayed(
                            delayed_helper, shape=shape, dtype=dtype
                        )
                    else:
                        self._data_dict[key] = file[key][:]
        return self._data_dict
