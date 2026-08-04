"""Tests for pooled slicing handles and lazy slice materialization."""
import h5py
import numpy as np
import pytest

from dhybridrpy import data as data_mod
from dhybridrpy.data import Field


@pytest.fixture(autouse=True)
def fresh_pool():
    data_mod.close_pooled_handles()
    yield
    data_mod.close_pooled_handles()


def write_cube(fp, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    cube = rng.standard_normal((16, 24, 32)).astype(np.float32)  # (nz, ny, nx)
    with h5py.File(fp, "w") as f:
        f.attrs.create("TIME", np.array([1.0], dtype=np.float32))
        ax = f.create_group("AXIS")
        for i, ln in enumerate((32.0, 24.0, 16.0)):
            ax.create_dataset(
                f"X{i+1} AXIS", data=np.array([0.0, ln], dtype=np.float32)
            )
        f.create_dataset(
            "DATA", data=cube, chunks=(4, 4, 8), compression="gzip"
        )
    return cube


def make_field(fp):
    return Field(str(fp), "Bx", 100, 1.0, 6, lazy=False, field_type="Total")


def test_slices_match_direct_reads(tmp_path):
    fp = tmp_path / "Bfld_00000100.h5"
    cube = write_cube(fp)
    field = make_field(fp)
    np.testing.assert_array_equal(field._read_2d_slice("x", 5), cube[:, :, 5].T)
    np.testing.assert_array_equal(field._read_2d_slice("y", 7), cube[:, 7, :].T)
    np.testing.assert_array_equal(field._read_2d_slice("z", 3), cube[3, :, :].T)


def test_handle_reused_across_slices(tmp_path, monkeypatch):
    fp = tmp_path / "Bfld_00000100.h5"
    write_cube(fp)
    field = make_field(fp)
    field._read_2d_slice("x", 0)

    counts = {"n": 0}
    real_file = h5py.File

    class CountingFile(real_file):
        def __init__(self, *args, **kwargs):
            counts["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", CountingFile)
    for i in range(5):
        field._read_2d_slice("x", i)
    assert counts["n"] == 0  # all slices served by the pooled handle


def test_replaced_file_gets_fresh_handle(tmp_path):
    fp = tmp_path / "Bfld_00000100.h5"
    write_cube(fp, rng_seed=0)
    field = make_field(fp)
    before = field._read_2d_slice("z", 0).copy()

    import os

    # replace the file as an external writer would (write + rename)
    tmp = tmp_path / "replacement.h5"
    cube2 = write_cube(tmp, rng_seed=1)
    os.replace(tmp, fp)
    os.utime(fp)  # ensure a new mtime even on coarse filesystem clocks
    after = field._read_2d_slice("z", 0)
    assert not np.array_equal(before, after)
    np.testing.assert_array_equal(after, cube2[0, :, :].T)


def test_pool_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(data_mod, "_SLICE_HANDLES_MAX", 2)
    for i in range(4):
        fp = tmp_path / f"Bfld_{i:08d}.h5"
        write_cube(fp, rng_seed=i)
        make_field(fp)._read_2d_slice("x", 0)
    assert len(data_mod._slice_handles) <= 2


def test_contiguous_dataset_slices(tmp_path):
    fp = tmp_path / "Bfld_00000100.h5"
    rng = np.random.default_rng(2)
    cube = rng.standard_normal((8, 10, 12)).astype(np.float32)
    with h5py.File(fp, "w") as f:
        ax = f.create_group("AXIS")
        for i, ln in enumerate((12.0, 10.0, 8.0)):
            ax.create_dataset(
                f"X{i+1} AXIS", data=np.array([0.0, ln], dtype=np.float32)
            )
        f.create_dataset("DATA", data=cube)  # contiguous, no chunks
    field = make_field(fp)
    np.testing.assert_array_equal(field._read_2d_slice("y", 4), cube[:, 4, :].T)


def test_evicted_dataset_remains_usable(tmp_path, monkeypatch):
    monkeypatch.setattr(data_mod, "_SLICE_HANDLES_MAX", 1)
    fp1 = tmp_path / "Bfld_00000001.h5"
    fp2 = tmp_path / "Bfld_00000002.h5"
    cube1 = write_cube(fp1, rng_seed=1)
    write_cube(fp2, rng_seed=2)
    ds1 = data_mod._pooled_dataset(str(fp1))
    data_mod._pooled_dataset(str(fp2))  # evicts fp1's pool entry
    # the held dataset must still read (eviction drops the reference
    # without closing the file)
    np.testing.assert_array_equal(ds1[0, 0, :], cube1[0, 0, :])


def test_arrow_keys_step_slider(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backend_bases import KeyEvent

    fp = tmp_path / "Bfld_00000100.h5"
    write_cube(fp)
    field = make_field(fp)
    ax, mesh = field.plot(slice_axis="x")
    fig = ax.figure
    slider = fig._dhybridrpy_widgets[-1]

    def press(key):
        event = KeyEvent(name="key_press_event", canvas=fig.canvas, key=key)
        fig.canvas.callbacks.process("key_press_event", event)

    assert slider.val == 0
    press("left")
    assert slider.val == 0  # clamped at the low end
    press("right")
    press("right")
    assert slider.val == 2
    for _ in range(50):
        press("right")
    assert slider.val == 31  # clamped at n-1 for the 32-wide x axis

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_lazy_derived_slice_computes_one_plane(tmp_path):
    fp = tmp_path / "Bfld_00000100.h5"
    cube = write_cube(fp)
    lazy_field = Field(str(fp), "Bx", 100, 1.0, 6, lazy=True, field_type="Total")
    derived = lazy_field * 2.0
    plane = derived._read_2d_slice("z", 3)
    assert isinstance(plane, np.ndarray)
    np.testing.assert_allclose(plane, 2.0 * cube[3, :, :].T, rtol=1e-6)
