"""Tests for 2D imshow plots, shared coordinates, and chunked lazy arrays."""
import h5py
import numpy as np

from dhybridrpy import DHybridrpy
from dhybridrpy import data as data_mod
from dhybridrpy.data import Field


def write_plane(fp, shape=(6, 20), time=1.0):
    rng = np.random.default_rng(3)
    plane = rng.standard_normal(shape).astype(np.float32)  # (ny, nx)
    with h5py.File(fp, "w") as f:
        f.attrs.create("TIME", np.array([time], dtype=np.float32))
        ax = f.create_group("AXIS")
        ax.create_dataset("X1 AXIS", data=np.array([0.0, 40.0], dtype=np.float32))
        ax.create_dataset("X2 AXIS", data=np.array([0.0, 12.0], dtype=np.float32))
        f.create_dataset("DATA", data=plane, chunks=(2, 5),
                         compression="gzip")
    return plane


def test_plot2d_uses_imshow(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.image import AxesImage

    fp = tmp_path / "Bfld_00000100.h5"
    plane = write_plane(fp)
    field = Field(str(fp), "Bx", 100, 1.0, 6, lazy=False, field_type="Total")
    ax, img = field.plot()
    assert isinstance(img, AxesImage)
    np.testing.assert_array_equal(np.asarray(img.get_array()), plane)
    np.testing.assert_allclose(img.get_extent(), (0.0, 40.0, 0.0, 12.0))

    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def test_coordinates_shared_between_objects(tmp_path):
    fp = tmp_path / "Bfld_00000100.h5"
    write_plane(fp)
    field_a = Field(str(fp), "Bx", 100, 1.0, 6, lazy=False, field_type="Total")
    field_b = Field(str(fp), "By", 100, 1.0, 6, lazy=False, field_type="Total")
    assert field_a.xdata is field_b.xdata  # one shared array, not copies
    assert not field_a.xdata.flags.writeable

    cropped = field_a.crop(x_range=(0.0, 0.5))
    assert len(cropped.xdata) == 10
    assert len(field_a.xdata) == 20  # the original is untouched


def test_lazy_arrays_are_chunked(tmp_path, monkeypatch):
    import dask.array as da

    monkeypatch.setattr(data_mod, "_LAZY_SLAB_BYTES", 128)
    fp = tmp_path / "Bfld_00000100.h5"
    plane = write_plane(fp)
    lazy = Field(str(fp), "Bx", 100, 1.0, 6, lazy=True, field_type="Total")
    arr = lazy.data
    assert isinstance(arr, da.Array)
    assert len(arr.chunks[-1]) > 1  # several slabs along the file's first axis
    np.testing.assert_array_equal(arr.compute(), plane.T)

    # slicing one slab reads only that slab's file region
    counts = {"n": 0}
    real_file = h5py.File

    class CountingFile(real_file):
        def __init__(self, *args, **kwargs):
            counts["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", CountingFile)
    first_width = arr.chunks[-1][0]
    np.testing.assert_array_equal(
        arr[..., :first_width].compute(), plane.T[..., :first_width]
    )
    assert counts["n"] == 1


def test_lazy_small_files_stay_single_chunk(tmp_path):
    fp = tmp_path / "Bfld_00000100.h5"
    write_plane(fp)
    lazy = Field(str(fp), "Bx", 100, 1.0, 6, lazy=True, field_type="Total")
    assert len(lazy.data.chunks[-1]) == 1


def test_timestep_closest(tmp_path):
    (tmp_path / "input").write_text(
        "time\n{\n\tdt=0.5,\n\tniter=10,\n\tc=100.,\n}\n"
    )
    bx = tmp_path / "Output" / "Fields" / "Magnetic" / "Total" / "x"
    bx.mkdir(parents=True)
    for ts in (10, 20, 30):
        write_plane(bx / f"Bfld_{ts:08d}.h5", time=ts * 0.5)
    dp = DHybridrpy(str(tmp_path / "input"), str(tmp_path / "Output"))
    assert dp.timestep_closest(11).timestep == 10
    assert dp.timestep_closest(29).timestep == 30
    assert dp.timestep_closest(15).timestep == 10  # tie keeps the earlier one
