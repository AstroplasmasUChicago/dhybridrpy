"""Process pool for parallel HDF5 reads.

h5py serializes every HDF5 call (including decompression) under one global
lock, so threads cannot overlap reads; separate processes can. Workers run
the functions below, which import only h5py/numpy, keeping matplotlib and
the rest of the package out of worker startup.
"""
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import h5py
import numpy as np


def _open(path):
    # read-only, no file locking; mirrors dhybridrpy.data.open_h5 without
    # importing that module (it pulls in matplotlib)
    return h5py.File(path, "r", locking=False)


def read_data(path: str) -> np.ndarray:
    """The DATA dataset in the package's (nx, ny[, nz]) orientation."""
    with _open(path) as f:
        return f["DATA"][:].T


def read_dataset(path: str, key: str) -> np.ndarray:
    with _open(path) as f:
        return f[key][:]


def map_data(paths, fn) -> object:
    """Apply `fn` to one timestep's field(s) worker-side; only the result
    returns. `paths` holds one file per field, passed to fn in order."""
    return fn(*(read_data(path) for path in paths))


def read_data_into(path: str, shm_name: str, index: int, shape, dtype_str):
    """Read DATA into slot `index` of a shared-memory stack."""
    from multiprocessing import shared_memory

    dtype = np.dtype(dtype_str)
    slot = int(np.prod(shape)) * dtype.itemsize
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        dest = np.ndarray(shape, dtype=dtype, buffer=shm.buf,
                          offset=index * slot)
        with _open(path) as f:
            data = f["DATA"][:].T
        if data.shape != tuple(shape):
            raise ValueError(
                f"{path} has shape {data.shape}, expected {tuple(shape)}"
            )
        if data.dtype != dtype:
            raise ValueError(
                f"{path} has dtype {data.dtype}, expected {dtype}"
            )
        dest[...] = data
    finally:
        shm.close()


def gather_data(paths, workers: int = None) -> np.ndarray:
    """Read many files' DATA into one stacked array.

    Workers write into shared memory instead of pickling arrays back, which
    would serialize in the parent and dominate the wall time for large
    fields. Needs the full selection's size in /dev/shm plus one copy.
    """
    if not paths:
        raise ValueError("No files to read.")
    with _open(paths[0]) as f:
        shape = f["DATA"].shape[::-1]
        dtype = f["DATA"].dtype
    from multiprocessing import shared_memory

    slot = int(np.prod(shape)) * dtype.itemsize
    shm = shared_memory.SharedMemory(create=True, size=slot * len(paths))
    try:
        pool = get_pool(workers)
        futures = [
            pool.submit(read_data_into, path, shm.name, i,
                        tuple(int(s) for s in shape), dtype.str)
            for i, path in enumerate(paths)
        ]
        for future in futures:
            future.result()
        result = np.ndarray(
            (len(paths), *shape), dtype=dtype, buffer=shm.buf
        ).copy()
    finally:
        shm.close()
        shm.unlink()
    return result


_pool = None
_pool_workers = None
_pool_pid = None
_pool_lock = threading.Lock()


def default_workers() -> int:
    return min(8, os.cpu_count() or 1)


def get_pool(workers: int = None) -> ProcessPoolExecutor:
    """A persistent spawn-based pool, recreated if the worker count grows."""
    global _pool, _pool_workers, _pool_pid
    workers = workers or default_workers()
    with _pool_lock:
        if (
            _pool is None
            or _pool_pid != os.getpid()
            or _pool_workers < workers
            or getattr(_pool, "_broken", False)  # self-heal after worker death
        ):
            if _pool is not None and _pool_pid == os.getpid():
                _pool.shutdown(wait=False)
            # spawn: workers never inherit open HDF5 handles or locks
            _pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn")
            )
            _pool_workers = workers
            _pool_pid = os.getpid()
        return _pool


def close_worker_pool() -> None:
    """Shut down the worker pool; it restarts on the next parallel call."""
    global _pool, _pool_workers, _pool_pid
    with _pool_lock:
        if _pool is not None and _pool_pid == os.getpid():
            _pool.shutdown(wait=True)
        _pool = None
        _pool_workers = None
        _pool_pid = None
