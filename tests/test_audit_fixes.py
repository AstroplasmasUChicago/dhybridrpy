"""Regression tests for the fixes from the bug audit."""
import h5py
import numpy as np
import pytest

from dhybridrpy import DHybridrpy
from dhybridrpy.data import Data, Field, Phase, fft_power_iso, fft_power_1d_slices
from dhybridrpy.tracks import TrackCollection


def write_field(dirpath, prefix, timestep, values, time):
    dirpath.mkdir(parents=True, exist_ok=True)
    fp = dirpath / f"{prefix}_{timestep:08d}.h5"
    with h5py.File(fp, "w") as f:
        f.attrs.create("TIME", np.array([time], dtype=np.float32))
        ax = f.create_group("AXIS")
        ax.create_dataset("X1 AXIS", data=np.array([0.0, 8.0], dtype=np.float32))
        ax.create_dataset("X2 AXIS", data=np.array([0.0, 4.0], dtype=np.float32))
        f.create_dataset("DATA", data=values)
    return fp


@pytest.fixture
def small_field(tmp_path):
    rng = np.random.default_rng(8)
    values = rng.standard_normal((4, 8)).astype(np.float32)
    fp = write_field(tmp_path, "Bfld", 10, values, 5.0)
    return Field(str(fp), "Bx", 10, 5.0, 6, lazy=False, field_type="Total"), values


def test_ufunc_out_plain_array(small_field):
    field, values = small_field
    target = np.zeros((8, 4), dtype=np.float32)
    result = np.add(field, field, out=target)
    assert result is target  # numpy's contract: the out array comes back
    np.testing.assert_array_equal(target, values.T * 2)


def test_ufunc_out_data_target_rejected(small_field):
    field, _ = small_field
    other = field * 1.0
    with pytest.raises(TypeError, match="plain arrays"):
        np.add(field, field, out=other)


def test_augmented_assignment_keeps_ndarray(small_field):
    field, values = small_field
    arr = np.ones((8, 4), dtype=np.float32)
    arr += field
    assert type(arr) is np.ndarray
    np.testing.assert_array_equal(arr, 1.0 + values.T)


def test_crop_near_edge_never_empty(small_field):
    field, _ = small_field
    cropped = field.crop(x_range=(0.99, 1.0))
    assert cropped._get_data_shape()[0] == 1  # the last cell, not nothing
    lims = cropped._data_dict["X1 AXIS lims"]
    assert lims[1] <= 8.0  # limits stay inside the box


def test_derived_phase_keeps_axis_labels(tmp_path):
    rng = np.random.default_rng(9)
    values = rng.standard_normal((4, 8)).astype(np.float32)
    fp = write_field(tmp_path, "p1x1_sp01", 10, values, 5.0)
    phase = Phase(str(fp), "p1x1", 10, 5.0, 6, lazy=False, species=1)
    doubled = phase * 2.0
    assert doubled._label_name == "p1x1"
    assert Data._axis_labels(doubled._label_name)[1] == Data._PX


def test_avg_1d_direction_guard_and_bool(small_field, tmp_path):
    field, values = small_field
    fp = tmp_path / "line_00000010.h5"
    with h5py.File(fp, "w") as f:
        f.attrs.create("TIME", np.array([5.0], dtype=np.float32))
        ax = f.create_group("AXIS")
        ax.create_dataset("X1 AXIS", data=np.array([0.0, 8.0], dtype=np.float32))
        f.create_dataset("DATA", data=np.arange(8, dtype=np.float32))
    line = Field(str(fp), "Bx", 10, 5.0, 6, lazy=False, field_type="Total")
    with pytest.raises(ValueError, match="1D data"):
        line.avg_1d("y")
    # a comparison result averages as floats instead of crashing
    mask = field > 0.0
    coords, mean, lo, hi = mask.avg_1d("x")
    np.testing.assert_allclose(mean, (values.T > 0).mean(axis=1))


def test_old_pickle_without_new_attributes(small_field):
    field, values = small_field
    old = Field.__new__(Field)
    state = dict(field.__dict__)
    state.pop("_data_chunks", None)
    state.pop("_label_name", None)
    old.__dict__.update(state)  # mimics a pickle from an older version
    np.testing.assert_array_equal(old.data, values.T)


def test_fft_power_units_agree():
    n, length = 128, 10.0
    x = np.linspace(0, length, n, endpoint=False)
    signal = np.sin(2 * np.pi * 5 * x / length)
    _, iso = fft_power_iso(signal, length)
    _, mean_1d, _, _ = fft_power_1d_slices(signal, length)
    np.testing.assert_allclose(iso.max(), mean_1d.max(), rtol=1e-9)


def test_corrupt_file_skipped_in_fallback_mode(tmp_path, caplog):
    (tmp_path / "input").write_text(
        "time\n{\n\tdt=0.5,\n\tniter=10,\n\tc=100.,\n}\n"
    )
    bx = tmp_path / "Output" / "Fields" / "Magnetic" / "Total" / "x"
    rng = np.random.default_rng(1)
    for ts in (10, 20):
        # times twice the deck dt: files agree with each other, so init
        # falls back to reading times from every file
        write_field(bx, "Bfld", ts, rng.standard_normal((4, 8)), ts * 1.0)
    (bx / "Bfld_00000030.h5").write_bytes(b"not an hdf5 file")
    dp = DHybridrpy(str(tmp_path / "input"), str(tmp_path / "Output"))
    assert list(dp.timesteps()) == [10, 20]  # corrupt file skipped, not fatal
    assert any("skipped" in r.message for r in caplog.records)


def test_stray_layouts_warn_and_skip(tmp_path, caplog):
    (tmp_path / "input").write_text(
        "time\n{\n\tdt=0.5,\n\tniter=10,\n\tc=100.,\n}\n"
    )
    bx = tmp_path / "Output" / "Fields" / "Magnetic" / "Total" / "x"
    rng = np.random.default_rng(2)
    write_field(bx, "Bfld", 10, rng.standard_normal((4, 8)), 5.0)
    write_field(tmp_path / "Output" / "Fields", "stray", 20,
                rng.standard_normal((4, 8)), 10.0)
    write_field(tmp_path / "Output" / "Fields" / "Magnetic" / "Odd" / "x",
                "Bfld", 30, rng.standard_normal((4, 8)), 15.0)
    dp = DHybridrpy(str(tmp_path / "input"), str(tmp_path / "Output"))
    assert list(dp.timesteps()) == [10]
    messages = " ".join(r.message for r in caplog.records)
    assert "Unrecognized folder layout" in messages
    assert "Unknown field type folder 'Odd'" in messages


def test_track_file_with_foreign_group(tmp_path, caplog):
    fp = tmp_path / "track_Sp01.h5"
    with h5py.File(fp, "w") as f:
        good = f.create_group("0-5")
        good.create_dataset("x1", data=np.arange(4, dtype=np.float32))
        f.create_group("metadata")  # not a rank-tag track
    coll = TrackCollection(str(fp), species=1)
    assert list(coll.track_ids) == ["0-5"]
    assert any("metadata" in r.message for r in caplog.records)


def test_timestep_closest_huge_argument(tmp_path):
    (tmp_path / "input").write_text(
        "time\n{\n\tdt=0.5,\n\tniter=10,\n\tc=100.,\n}\n"
    )
    bx = tmp_path / "Output" / "Fields" / "Magnetic" / "Total" / "x"
    write_field(bx, "Bfld", 10, np.zeros((4, 8)), 5.0)
    dp = DHybridrpy(str(tmp_path / "input"), str(tmp_path / "Output"))
    assert dp.timestep_closest(10**25).timestep == 10


def test_comparisons_are_elementwise(small_field):
    field, values = small_field
    same = field * 1.0
    equal = field == same
    assert equal.data.dtype == np.bool_
    assert equal.data.all()

    threshold = field > 0.0
    np.testing.assert_array_equal(threshold.data, values.T > 0)

    unequal = field != same
    assert not unequal.data.any()

    # ndarray on either side gives the same elementwise result
    left = field == values.T
    right = values.T == field
    np.testing.assert_array_equal(left.data, right.data)


def test_comparison_ambiguity_and_identity(small_field):
    field, _ = small_field
    assert field is field
    assert (field == None) is False  # noqa: E711  falls back to identity
    with pytest.raises(ValueError, match="ambiguous"):
        if field == field * 1.0:
            pass
    with pytest.raises(TypeError):
        {field}  # unhashable, like numpy arrays


def test_comparison_incompatible_shapes_raise(small_field, tmp_path):
    field, _ = small_field
    other_fp = write_field(tmp_path / "other", "Bfld", 10,
                           np.zeros((3, 5), dtype=np.float32), 5.0)
    other = Field(str(other_fp), "Bx", 10, 5.0, 6, lazy=False,
                  field_type="Total")
    with pytest.raises(ValueError, match="Incompatible grid shapes"):
        field == other
