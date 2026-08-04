"""Tests for derived init times (timestep*dt) and the per-file fallback."""
import h5py
import numpy as np
import pytest

import dhybridrpy.dhybridrpy as dhy_mod
from dhybridrpy import DHybridrpy

DT = 0.5


def write_input(path, extra_time_lines=""):
    path.write_text(
        "time\n{\n"
        f"\tdt={DT},\n"
        "\tniter=10,\n"
        f"{extra_time_lines}"
        "\tc=100.,\n"
        "}\n"
    )


def write_field(dirpath, prefix, timestep, time_value):
    dirpath.mkdir(parents=True, exist_ok=True)
    fp = dirpath / f"{prefix}_{timestep:08d}.h5"
    with h5py.File(fp, "w") as f:
        f.attrs.create("TIME", np.array([time_value], dtype=np.float32))
        f.attrs.create("ITER", np.array([timestep], dtype=np.int32))
        ax = f.create_group("AXIS")
        ax.create_dataset("X1 AXIS", data=np.array([0.0, 8.0], dtype=np.float32))
        ax.create_dataset("X2 AXIS", data=np.array([0.0, 4.0], dtype=np.float32))
        f.create_dataset("DATA", data=np.zeros((4, 8), dtype=np.float32))


def make_tree(tmp_path, time_scale=1.0, extra_time_lines=""):
    inp = tmp_path / "input"
    write_input(inp, extra_time_lines)
    bx = tmp_path / "Output" / "Fields" / "Magnetic" / "Total" / "x"
    phase = tmp_path / "Output" / "Phase" / "x2x1" / "Sp01"
    for ts in (0, 10, 20):
        write_field(bx, "Bfld", ts, ts * DT * time_scale)
        write_field(phase, "x2x1_sp01", ts, ts * DT * time_scale)
    return str(inp), str(tmp_path / "Output")


def install_open_counter(monkeypatch):
    """Count h5py.File opens; install only AFTER the test tree is written."""
    counts = {"n": 0}
    real_file = h5py.File

    class CountingFile(real_file):
        def __init__(self, *args, **kwargs):
            counts["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(dhy_mod.h5py, "File", CountingFile)
    return counts


def test_derived_times_spot_check_opens_only(tmp_path, monkeypatch):
    inp, out = make_tree(tmp_path)
    counts = install_open_counter(monkeypatch)
    dp = DHybridrpy(inp, out)
    assert dp._derive_times
    assert counts["n"] == 2  # min+max spot checks, regardless of timestep count
    np.testing.assert_allclose(dp.times(), [5.0, 10.0])
    assert dp.timestep(10).phases.x2x1(species=1).time == 5.0


def test_fallback_when_deck_dt_mismatches(tmp_path, monkeypatch, caplog):
    # files agree with each other on dt, but not with the deck (stale input file)
    inp, out = make_tree(tmp_path, time_scale=2.0)
    counts = install_open_counter(monkeypatch)
    dp = DHybridrpy(inp, out)
    assert not dp._derive_times
    assert counts["n"] == 2 + 3  # both spot checks + one read per timestep
    np.testing.assert_allclose(dp.times(), [10.0, 20.0])
    assert any("does not match" in r.message for r in caplog.records)


def test_error_when_files_imply_different_dt(tmp_path):
    """A fixed-dt run cannot write files whose TIMEs imply two dt values."""
    inp = tmp_path / "input"
    write_input(inp)
    bx = tmp_path / "Output" / "Fields" / "Magnetic" / "Total" / "x"
    write_field(bx, "Bfld", 10, 10 * DT)
    write_field(bx, "Bfld", 20, 20 * DT + 3.0)
    with pytest.raises(ValueError, match="different time steps"):
        DHybridrpy(str(inp), str(tmp_path / "Output"))


def test_error_when_inconsistent_dt_spans_directories(tmp_path):
    """The consistency check must span the whole tree, not just the first
    folder walked."""
    inp = tmp_path / "input"
    write_input(inp)
    bx = tmp_path / "Output" / "Fields" / "Magnetic" / "Total" / "x"
    phase = tmp_path / "Output" / "Phase" / "x2x1" / "Sp01"
    write_field(bx, "Bfld", 10, 10 * DT)
    write_field(phase, "x2x1_sp01", 50, 50 * DT + 3.0)
    with pytest.raises(ValueError, match="different time steps"):
        DHybridrpy(str(inp), str(tmp_path / "Output"))


def test_adaptive_dt_reads_files(tmp_path, monkeypatch):
    inp, out = make_tree(tmp_path, extra_time_lines="\tadaptive_dt=.true.,\n")
    counts = install_open_counter(monkeypatch)
    dp = DHybridrpy(inp, out)
    assert not dp._derive_times
    assert counts["n"] == 3  # no spot check; one read per timestep


def test_t0_in_deck_still_parses(tmp_path):
    inp, out = make_tree(tmp_path, extra_time_lines="\tt0=0.,\n")
    dp = DHybridrpy(inp, out)
    assert dp.start_time == 0.0
    assert dp._derive_times


def test_currentdens_naming_with_shared_components(tmp_path):
    inp = tmp_path / "input"
    write_input(inp)
    jx = tmp_path / "Output" / "Fields" / "CurrentDens" / "x"
    # two files in one directory reuse that directory's components list
    for ts in (10, 20):
        write_field(jx, "Jfld", ts, ts * DT)
    dp = DHybridrpy(str(inp), str(tmp_path / "Output"))
    for ts in (10, 20):
        field = dp.timestep(ts).fields.Jx("Total")
        assert field.name == "Jx"
        assert field.type == "Total"
