"""Process pool for parallel HDF5 reads.

h5py serializes every HDF5 call (including decompression) under one global
lock, so threads cannot overlap reads; separate processes can. Workers run
the functions below, which import only h5py/numpy, keeping matplotlib and
the rest of the package out of worker startup.
"""
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import h5py
import numpy as np

# Below this size, pool and copy overheads outweigh the parallel win for
# the shuffle+deflate files dHybridR writes.
SINGLE_FILE_MIN_BYTES = 512 * 1024**2


_in_worker = False


def _mark_worker():
    global _in_worker
    _in_worker = True


def spawn_is_safe() -> bool:
    """Whether worker processes can start without re-running user code.

    Spawned workers re-import __main__, which is only safe when __main__
    has no file to re-run (interactive sessions and notebooks). Scripts
    opt in by calling the parallel methods, which document the required
    __main__ guard. Worker processes themselves must never start nested
    pools.
    """
    if _in_worker:
        return False
    main = sys.modules.get("__main__")
    if main is None:
        return True
    # mirror multiprocessing's spawn preparation: it re-runs __main__ from
    # either its module spec or its file path
    spec_name = getattr(getattr(main, "__spec__", None), "name", None)
    return spec_name is None and not hasattr(main, "__file__")


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


def read_slab_into(path, shm_name, axis, start, stop, shape, dtype_str):
    """Read rows [start, stop) along `axis` of DATA into shared memory."""
    from multiprocessing import shared_memory

    dtype = np.dtype(dtype_str)
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        dest = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        selection = [slice(None)] * len(shape)
        selection[axis] = slice(start, stop)
        with _open(path) as f:
            dest[tuple(selection)] = f["DATA"][tuple(selection)]
    finally:
        shm.close()


def parallel_read_data(path: str, workers: int = None):
    """DATA read as chunk-aligned slabs by worker processes.

    Splits along the axis with the most chunks so that no chunk is
    decompressed twice. Returns the array in file order, or None when the
    dataset is too small to benefit or has nothing to split. Peak memory
    is about twice the dataset, and the shared block needs that much room
    in /dev/shm; allocation failure falls back to the serial read.
    """
    workers = workers or default_workers()
    with _open(path) as f:
        dataset = f["DATA"]
        shape, dtype, chunks = dataset.shape, dataset.dtype, dataset.chunks
    nbytes = int(np.prod(shape)) * dtype.itemsize
    if nbytes < SINGLE_FILE_MIN_BYTES:
        return None
    if chunks is None:
        axis, step = 0, 1
    else:
        counts = [-(-s // c) for s, c in zip(shape, chunks)]
        axis = int(np.argmax(counts))
        step = chunks[axis]
    blocks = -(-shape[axis] // step)
    if blocks < 2:
        return None

    n_slabs = min(workers, blocks)
    edges = [
        min(round(i * blocks / n_slabs) * step, shape[axis])
        for i in range(n_slabs + 1)
    ]
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    try:
        pool = get_pool(workers)
        futures = [
            pool.submit(read_slab_into, path, shm.name, axis, a, b,
                        tuple(int(s) for s in shape), dtype.str)
            for a, b in zip(edges, edges[1:]) if a < b
        ]
        for future in futures:
            future.result()
        result = np.ndarray(tuple(shape), dtype=dtype, buffer=shm.buf).copy()
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
    if _in_worker:
        raise RuntimeError(
            "Parallel reads cannot start inside a worker process."
        )
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
                max_workers=workers, mp_context=get_context("spawn"),
                initializer=_mark_worker,
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
