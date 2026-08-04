"""Regression tests for float32 spectral accumulation and crop() memory retention."""
import h5py
import numpy as np
import pytest

from dhybridrpy.data import Field, fft_power_iso, fft_power_1d_slices


@pytest.fixture
def field_f32():
    """Mean~1 float32 field: worst case for float32 shell accumulation."""
    rng = np.random.default_rng(7)
    return (1.0 + 0.1 * rng.standard_normal((64, 64, 64))).astype(np.float32)


def test_fft_power_iso_3d_float32_matches_float64(field_f32):
    k32, p32 = fft_power_iso(field_f32, 10.0, 10.0, 10.0)
    k64, p64 = fft_power_iso(field_f32.astype(np.float64), 10.0, 10.0, 10.0)
    np.testing.assert_allclose(k32, k64)
    # Pre-fix this failed badly: float32 histogram accumulation quantized bins
    # to ulp(total) and undercounted shells by tens of percent.
    np.testing.assert_allclose(p32, p64, rtol=1e-4)
    assert p32.dtype == np.float64


def test_fft_power_iso_2d_and_1d_float32(field_f32):
    plane = field_f32[:, :, 0]
    _, p32 = fft_power_iso(plane, 10.0, 10.0)
    _, p64 = fft_power_iso(plane.astype(np.float64), 10.0, 10.0)
    np.testing.assert_allclose(p32, p64, rtol=1e-4)

    line = field_f32[:, 0, 0]
    _, q32 = fft_power_iso(line, 10.0)
    _, q64 = fft_power_iso(line.astype(np.float64), 10.0)
    np.testing.assert_allclose(q32, q64, rtol=1e-4)


def test_fft_power_iso_normalize_float32(field_f32):
    _, p32 = fft_power_iso(field_f32, 10.0, 10.0, 10.0, normalize=True)
    _, p64 = fft_power_iso(
        field_f32.astype(np.float64), 10.0, 10.0, 10.0, normalize=True
    )
    np.testing.assert_allclose(p32, p64, rtol=1e-4)


def test_fft_power_1d_slices_float32(field_f32):
    r32 = fft_power_1d_slices(field_f32, 10.0, "x")
    r64 = fft_power_1d_slices(field_f32.astype(np.float64), 10.0, "x")
    for a32, a64 in zip(r32, r64):
        np.testing.assert_allclose(a32, a64, rtol=1e-4)


@pytest.fixture
def hdf5_field(tmp_path):
    """Minimal dHybridR-style grid file backing a Field object."""
    fp = tmp_path / "Bfld_00000100.h5"
    rng = np.random.default_rng(3)
    data = rng.standard_normal((20, 30, 40)).astype(np.float32)  # (nz, ny, nx)
    with h5py.File(fp, "w") as f:
        f.attrs.create("TIME", np.array([1.0], dtype=np.float32))
        f.attrs.create("ITER", np.array([100], dtype=np.int32))
        ax = f.create_group("AXIS")
        for i, ln in enumerate((40.0, 30.0, 20.0)):
            ax.create_dataset(
                f"X{i+1} AXIS", data=np.array([0.0, ln], dtype=np.float32)
            )
        f.create_dataset("DATA", data=data)
    return str(fp), data


def test_crop_does_not_retain_parent_array(hdf5_field):
    fp, raw = hdf5_field
    field = Field(fp, "Bx", 100, 1.0, 6, lazy=False, field_type="Total")
    cropped = field.crop(x_range=(0.25, 0.5), y_range=(0.0, 0.5))

    arr = cropped._data_dict["Bx"]
    assert isinstance(arr, np.ndarray)
    assert arr.base is None  # owns its memory; no view of the full array
    np.testing.assert_array_equal(arr, raw.T[10:20, 0:15, :])

    coords = cropped._data_dict["X1 AXIS coords"]
    assert coords.base is None


def test_crop_values_and_axes(hdf5_field):
    fp, _ = hdf5_field
    field = Field(fp, "Bx", 100, 1.0, 6, lazy=False, field_type="Total")
    cropped = field.crop(x_range=(0.25, 0.5))
    assert cropped._data_shape == (10, 30, 20)
    lims = cropped._data_dict["X1 AXIS lims"]
    np.testing.assert_allclose(lims, [10.0, 20.0])


def test_chained_crop_keeps_earlier_axis_metadata(hdf5_field):
    fp, _ = hdf5_field
    field = Field(fp, "Bx", 100, 1.0, 6, lazy=False, field_type="Total")
    twice = field.crop(x_range=(0.0, 0.5), y_range=(0.0, 0.5)).crop(
        x_range=(0.0, 0.5)
    )
    assert twice._data_shape == (10, 15, 20)
    np.testing.assert_allclose(twice._data_dict["X1 AXIS lims"], [0.0, 10.0])
    # Pre-fix, the second crop dropped the first crop's y metadata and fell
    # back to the full-box limits from the file.
    np.testing.assert_allclose(twice._data_dict["X2 AXIS lims"], [0.0, 15.0])
    assert len(twice._data_dict["X2 AXIS coords"]) == 15
