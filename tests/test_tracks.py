"""Tests for the shared track-file handle, iteration, and bulk loading."""
import os

import h5py
import numpy as np
import pytest

from dhybridrpy.tracks import Track, TrackCollection

KEYS = ("x1", "ene", "B1")


def write_track_file(fp, lengths, seed=0):
    """Track file with ragged group lengths: {'0-5': 8, '1-2': 12, ...}."""
    rng = np.random.default_rng(seed)
    data = {}
    with h5py.File(fp, "w") as f:
        for track_id, length in lengths.items():
            group = f.create_group(track_id)
            data[track_id] = {}
            for key in KEYS:
                values = rng.standard_normal(length).astype(np.float32)
                group.create_dataset(key, data=values)
                data[track_id][key] = values
    return data


@pytest.fixture
def track_file(tmp_path):
    fp = tmp_path / "track_Sp01.h5"
    lengths = {"0-5": 8, "0-30": 12, "1-2": 12, "2-7": 5}
    return str(fp), write_track_file(fp, lengths)


def test_values_and_sorted_ids(track_file):
    fp, data = track_file
    coll = TrackCollection(fp, species=1)
    assert list(coll.track_ids) == ["0-5", "0-30", "1-2", "2-7"]
    for tid in coll.track_ids:
        np.testing.assert_array_equal(coll[tid].x1, data[tid]["x1"])


def test_single_open_for_many_reads(track_file, monkeypatch):
    fp, _ = track_file
    coll = TrackCollection(fp, species=1)
    coll[coll.track_ids[0]].x1  # warm: opens the shared handle

    counts = {"n": 0}
    real_file = h5py.File

    class CountingFile(real_file):
        def __init__(self, *args, **kwargs):
            counts["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", CountingFile)
    for tid in coll.track_ids:
        for key in KEYS:
            coll[tid]._load_dataset(key)
    assert counts["n"] == 0  # every read went through the shared handle


def test_len_and_iter(track_file):
    fp, data = track_file
    coll = TrackCollection(fp, species=1)
    assert len(coll) == 4
    seen = [track.track_id for track in coll]
    assert seen == ["0-5", "0-30", "1-2", "2-7"]
    for track in coll:
        assert isinstance(track, Track)


def test_bulk_load_dataset_ragged(track_file):
    fp, data = track_file
    coll = TrackCollection(fp, species=1)
    ene = coll.load_dataset("ene")
    assert set(ene) == set(data)
    for tid, values in ene.items():
        np.testing.assert_array_equal(values, data[tid]["ene"])
        assert len(values) == len(data[tid]["ene"])  # ragged lengths kept

    subset = coll.load_dataset("x1", track_ids=["1-2", "2-7"])
    assert set(subset) == {"1-2", "2-7"}


def test_bulk_load_dataset_lazy(track_file):
    fp, data = track_file
    coll = TrackCollection(fp, species=1, lazy=True)
    ene = coll.load_dataset("ene", track_ids=["0-5"])
    import dask.array as da

    assert isinstance(ene["0-5"], da.Array)
    np.testing.assert_array_equal(ene["0-5"].compute(), data["0-5"]["ene"])


def test_standalone_track_without_collection(track_file):
    fp, data = track_file
    track = Track(fp, "1-2", "1-2", species=1)
    np.testing.assert_array_equal(track.B1, data["1-2"]["B1"])


def test_replaced_file_reopens(track_file, tmp_path):
    fp, _ = track_file
    coll = TrackCollection(fp, species=1)
    before = coll["0-5"].x1.copy()

    tmp = tmp_path / "replacement.h5"
    new_data = write_track_file(tmp, {"0-5": 8, "0-30": 12, "1-2": 12, "2-7": 5},
                                seed=9)
    os.replace(tmp, fp)
    stat = os.stat(fp)
    os.utime(fp, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    after = coll["0-5"].x1
    assert not np.array_equal(before, after)
    np.testing.assert_array_equal(after, new_data["0-5"]["x1"])


def test_close_and_context_manager(track_file):
    fp, data = track_file
    with TrackCollection(fp, species=1) as coll:
        coll["0-5"].x1
        assert coll._file is not None
    assert coll._file is None
    # reads reopen after close
    np.testing.assert_array_equal(coll["0-5"].x1, data["0-5"]["x1"])


def test_pickle_after_use(track_file):
    import pickle

    fp, data = track_file
    coll = TrackCollection(fp, species=1)
    coll["0-5"].x1  # handle open
    clone = pickle.loads(pickle.dumps(coll))
    np.testing.assert_array_equal(clone["0-30"].x1, data["0-30"]["x1"])


def test_replaced_file_refreshes_track_ids(track_file, tmp_path):
    fp, _ = track_file
    coll = TrackCollection(fp, species=1)
    assert "9-9" not in coll.track_ids

    tmp = tmp_path / "replacement.h5"
    new_data = write_track_file(tmp, {"0-5": 8, "9-9": 6}, seed=3)
    os.replace(tmp, fp)
    stat = os.stat(fp)
    os.utime(fp, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    coll.handle()  # reopen invalidates the cached ids
    assert list(coll.track_ids) == ["0-5", "9-9"]
    np.testing.assert_array_equal(coll["9-9"].x1, new_data["9-9"]["x1"])


def test_bulk_load_errors_name_the_problem(track_file):
    fp, _ = track_file
    coll = TrackCollection(fp, species=1)
    with pytest.raises(KeyError, match="Track ID 'no-such'"):
        coll.load_dataset("x1", track_ids=["no-such"])
    with pytest.raises(KeyError, match="Dataset 'nope'"):
        coll.load_dataset("nope", track_ids=["0-5"])


def test_track_from_old_pickle_without_collection(track_file):
    fp, data = track_file
    old = Track.__new__(Track)
    old.__dict__.update(
        file_path=fp, track_id="1-2", species=1, lazy=False,
        _group_name="1-2", _available_keys=None,
    )  # no _collection: mimics a pickle from before the attribute existed
    np.testing.assert_array_equal(old.x1, data["1-2"]["x1"])


def test_threaded_reads_with_close_churn(track_file):
    import threading

    fp, data = track_file
    coll = TrackCollection(fp, species=1)
    errors = []

    def reader():
        try:
            for _ in range(50):
                np.testing.assert_array_equal(
                    coll["0-30"].x1, data["0-30"]["x1"]
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def closer():
        for _ in range(100):
            coll.close()

    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads.append(threading.Thread(target=closer))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
