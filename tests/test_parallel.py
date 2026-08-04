"""Tests for the process worker pool, bulk field loading, and Raw access."""
import h5py
import numpy as np
import pytest

from dhybridrpy import DHybridrpy
from dhybridrpy import _parallel
from dhybridrpy.data import Raw

DT = 0.5


def mean_bperp(bx, by):
    """Module-level so spawned workers can import it."""
    return np.sqrt(bx**2 + by**2).mean()


@pytest.fixture(scope="module", autouse=True)
def shutdown_pool():
    yield
    _parallel.close_worker_pool()


def write_field(dirpath, prefix, timestep, values):
    dirpath.mkdir(parents=True, exist_ok=True)
    fp = dirpath / f"{prefix}_{timestep:08d}.h5"
    with h5py.File(fp, "w") as f:
        f.attrs.create("TIME", np.array([timestep * DT], dtype=np.float32))
        f.attrs.create("ITER", np.array([timestep], dtype=np.int32))
        ax = f.create_group("AXIS")
        ax.create_dataset("X1 AXIS", data=np.array([0.0, 8.0], dtype=np.float32))
        ax.create_dataset("X2 AXIS", data=np.array([0.0, 4.0], dtype=np.float32))
        f.create_dataset("DATA", data=values)


@pytest.fixture
def field_tree(tmp_path):
    (tmp_path / "input").write_text(
        "time\n{\n\tdt=0.5,\n\tniter=10,\n\tc=100.,\n}\n"
    )
    rng = np.random.default_rng(4)
    expected = {}
    bx = tmp_path / "Output" / "Fields" / "Magnetic" / "Total" / "x"
    by = tmp_path / "Output" / "Fields" / "Magnetic" / "Total" / "y"
    dens = tmp_path / "Output" / "Phase" / "x2x1" / "Sp01"
    for ts in (10, 20, 30):
        values = rng.standard_normal((4, 8)).astype(np.float32)  # (ny, nx)
        expected[ts] = values
        write_field(bx, "Bfld", ts, values)
        write_field(by, "Bfld", ts, values + 1.0)
        write_field(dens, "x2x1_sp01", ts, values * 2.0)
    return str(tmp_path / "input"), str(tmp_path / "Output"), expected


def test_field_timeseries_matches_serial(field_tree):
    inp, out, expected = field_tree
    dp = DHybridrpy(inp, out)
    stacked = dp.field_timeseries("Bx", workers=2)
    assert stacked.shape == (3, 8, 4)
    for i, ts in enumerate((10, 20, 30)):
        np.testing.assert_array_equal(stacked[i], expected[ts].T)
        np.testing.assert_array_equal(
            stacked[i], dp.timestep(ts).fields.Bx("Total").data
        )


def test_field_timeseries_timestep_subset(field_tree):
    inp, out, expected = field_tree
    dp = DHybridrpy(inp, out)
    stacked = dp.field_timeseries("Bx", timesteps=[30, 10], workers=2)
    np.testing.assert_array_equal(stacked[0], expected[30].T)
    np.testing.assert_array_equal(stacked[1], expected[10].T)


def test_field_timeseries_apply(field_tree):
    inp, out, expected = field_tree
    dp = DHybridrpy(inp, out)
    means = dp.field_timeseries("Bx", apply=np.mean, workers=2)
    for value, ts in zip(means, (10, 20, 30)):
        np.testing.assert_allclose(value, expected[ts].mean(), rtol=1e-6)


def test_field_timeseries_multi_field_apply(field_tree):
    inp, out, expected = field_tree
    dp = DHybridrpy(inp, out)
    results = dp.field_timeseries(["Bx", "By"], apply=mean_bperp, workers=2)
    for value, ts in zip(results, (10, 20, 30)):
        bx = expected[ts].T
        reference = np.sqrt(bx**2 + (bx + 1.0) ** 2).mean()
        np.testing.assert_allclose(value, reference, rtol=1e-6)


def test_field_timeseries_multi_field_needs_apply(field_tree):
    inp, out, _ = field_tree
    dp = DHybridrpy(inp, out)
    with pytest.raises(ValueError, match="needs `apply`"):
        dp.field_timeseries(["Bx", "By"], workers=2)


def test_phase_timeseries(field_tree):
    inp, out, expected = field_tree
    dp = DHybridrpy(inp, out)
    stacked = dp.phase_timeseries("x2x1", species=1, workers=2)
    np.testing.assert_array_equal(stacked[0], expected[10].T * 2.0)


def test_field_timeseries_uses_shared_memory_path(field_tree, monkeypatch):
    inp, out, _ = field_tree
    dp = DHybridrpy(inp, out)
    calls = {"n": 0}
    real_gather = _parallel.gather_data

    def counting_gather(paths, workers=None):
        calls["n"] += 1
        return real_gather(paths, workers)

    monkeypatch.setattr(_parallel, "gather_data", counting_gather)
    dp.field_timeseries("Bx", workers=2)
    assert calls["n"] == 1  # the shared-memory path, not pickled returns


def test_field_timeseries_empty_selection(field_tree):
    inp, out, _ = field_tree
    dp = DHybridrpy(inp, out)
    with pytest.raises(ValueError, match="No timesteps selected"):
        dp.field_timeseries("Bx", timesteps=[])


def test_raw_load_missing_key_names_it(raw_file):
    fp, _ = raw_file
    raw = Raw(fp, "raw", 10, 5.0, lazy=False, species=1)
    with pytest.raises(KeyError, match=r"\['nope'\] not found"):
        raw.load(keys=["x1", "nope"])


def test_pool_is_persistent_and_closable(field_tree):
    inp, out, _ = field_tree
    dp = DHybridrpy(inp, out)
    dp.field_timeseries("Bx", workers=2)
    pool_a = _parallel._pool
    dp.field_timeseries("Bx", workers=2)
    assert _parallel._pool is pool_a  # reused, not recreated
    _parallel.close_worker_pool()
    assert _parallel._pool is None


@pytest.fixture
def raw_file(tmp_path):
    fp = tmp_path / "raw_sp01_00000010.h5"
    rng = np.random.default_rng(5)
    data = {
        "x1": rng.uniform(0, 8, 100).astype(np.float32),
        "ene": rng.exponential(1.0, 100).astype(np.float32),
        "tag": rng.integers(0, 1000, 100, dtype=np.int64),
    }
    with h5py.File(fp, "w") as f:
        f.attrs.create("TIME", np.array([5.0], dtype=np.float32))
        for key, values in data.items():
            f.create_dataset(key, data=values)
    return str(fp), data


def test_raw_keys_and_getitem(raw_file):
    fp, data = raw_file
    raw = Raw(fp, "raw", 10, 5.0, lazy=False, species=1)
    assert sorted(raw.keys()) == ["ene", "tag", "x1"]
    assert "ene" in raw
    np.testing.assert_array_equal(raw["ene"], data["ene"])
    with pytest.raises(KeyError, match="'nope' not found"):
        raw["nope"]


def test_raw_getitem_lazy(raw_file):
    import dask.array as da

    fp, data = raw_file
    raw = Raw(fp, "raw", 10, 5.0, lazy=True, species=1)
    arr = raw["x1"]
    assert isinstance(arr, da.Array)
    np.testing.assert_array_equal(arr.compute(), data["x1"])


def test_raw_parallel_load(raw_file):
    fp, data = raw_file
    raw = Raw(fp, "raw", 10, 5.0, lazy=False, species=1)
    subset = raw.load(keys=["x1", "tag"], workers=2)
    assert set(subset) == {"x1", "tag"}
    np.testing.assert_array_equal(subset["x1"], data["x1"])
    np.testing.assert_array_equal(subset["tag"], data["tag"])

    everything = raw.load(workers=2)
    assert set(everything) == set(data)

    lazy_raw = Raw(fp, "raw", 10, 5.0, lazy=True, species=1)
    with pytest.raises(ValueError, match="reads eagerly"):
        lazy_raw.load()
