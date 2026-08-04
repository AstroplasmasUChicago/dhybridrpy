import logging
import os
import threading
import numpy as np
import dask.array as da
from dask.delayed import delayed
from typing import Union, List, Optional, Dict, Iterator

from .data import open_h5

logger = logging.getLogger(__name__)


class Track:
    """
    Represents a single particle track across all timesteps.

    Args:
        file_path: Path to the track HDF5 file.
        group_name: Name of the HDF5 group for this particle.
        track_id: The particle tag/ID.
        species: The species number.
        lazy: Whether to use lazy loading via dask.
        collection: The owning TrackCollection, whose shared file handle is
            used for reads. Standalone Tracks open the file per access.
    """

    # default for instances restored from pickles that predate _collection
    _collection = None

    def __init__(
        self,
        file_path: str,
        group_name: str,
        track_id: str,
        species: int,
        lazy: bool = False,
        collection: Optional["TrackCollection"] = None,
    ):
        self.file_path = file_path
        self.track_id = track_id  # Format: "rank-tag", e.g., "0-1465"
        self.species = species
        self.lazy = lazy
        self._group_name = group_name
        self._collection = collection
        self._available_keys: Optional[List[str]] = None

    def _get_available_keys(self) -> List[str]:
        """Get list of available datasets for this track."""
        if self._available_keys is None:
            if self._collection is not None:
                self._available_keys = self._collection._list_keys(
                    self._group_name
                )
            else:
                with open_h5(self.file_path) as f:
                    self._available_keys = list(f[self._group_name].keys())
        return self._available_keys

    def _load_dataset(self, key: str) -> Union[np.ndarray, da.Array]:
        """Load a dataset from the track file.

        Always re-reads from HDF5 (or returns a fresh dask-delayed view) so
        that iterating over many Tracks does not accumulate every dataset in
        memory. Mirrors Data.data and Raw.dict.
        """
        if key not in self._get_available_keys():
            raise AttributeError(
                f"Dataset '{key}' not available for track {self.track_id}."
            )

        if self.lazy:
            if self._collection is not None:
                shape, dtype = self._collection._dataset_meta(
                    self._group_name, key
                )
            else:
                with open_h5(self.file_path) as f:
                    shape = f[self._group_name][key].shape
                    dtype = f[self._group_name][key].dtype

            # the loader captures the path, not a handle, so computing the
            # array works after the collection's handle is closed
            def loader(k=key):
                with open_h5(self.file_path) as f:
                    return f[self._group_name][k][:]

            return da.from_delayed(delayed(loader)(), shape=shape, dtype=dtype)

        if self._collection is not None:
            return self._collection._read(self._group_name, key)
        with open_h5(self.file_path) as f:
            return f[self._group_name][key][:]

    @property
    def x1(self) -> Union[np.ndarray, da.Array]:
        """X coordinate over time."""
        return self._load_dataset("x1")

    @property
    def x2(self) -> Union[np.ndarray, da.Array]:
        """Y coordinate over time."""
        return self._load_dataset("x2")

    @property
    def x3(self) -> Union[np.ndarray, da.Array]:
        """Z coordinate over time."""
        return self._load_dataset("x3")

    @property
    def v1(self) -> Union[np.ndarray, da.Array]:
        """X component of velocity over time."""
        return self._load_dataset("v1")

    @property
    def v2(self) -> Union[np.ndarray, da.Array]:
        """Y component of velocity over time."""
        return self._load_dataset("v2")

    @property
    def v3(self) -> Union[np.ndarray, da.Array]:
        """Z component of velocity over time."""
        return self._load_dataset("v3")

    @property
    def B1(self) -> Union[np.ndarray, da.Array]:
        """X magnetic field at particle position over time."""
        return self._load_dataset("B1")

    @property
    def B2(self) -> Union[np.ndarray, da.Array]:
        """Y magnetic field at particle position over time."""
        return self._load_dataset("B2")

    @property
    def B3(self) -> Union[np.ndarray, da.Array]:
        """Z magnetic field at particle position over time."""
        return self._load_dataset("B3")

    @property
    def E1(self) -> Union[np.ndarray, da.Array]:
        """X electric field at particle position over time."""
        return self._load_dataset("E1")

    @property
    def E2(self) -> Union[np.ndarray, da.Array]:
        """Y electric field at particle position over time."""
        return self._load_dataset("E2")

    @property
    def E3(self) -> Union[np.ndarray, da.Array]:
        """Z electric field at particle position over time."""
        return self._load_dataset("E3")

    @property
    def t(self) -> Union[np.ndarray, da.Array]:
        """Simulation time."""
        return self._load_dataset("t")

    @property
    def n(self) -> Union[np.ndarray, da.Array]:
        """Iteration number at which each value was stored."""
        return self._load_dataset("n")

    @property
    def ene(self) -> Union[np.ndarray, da.Array]:
        """Particle energy over time."""
        return self._load_dataset("ene")

    @property
    def q(self) -> Union[np.ndarray, da.Array]:
        """Particle charge."""
        return self._load_dataset("q")

    def __getattr__(self, name: str) -> Union[np.ndarray, da.Array]:
        """Allow access to any dataset in the track file."""
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        try:
            return self._load_dataset(name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __repr__(self) -> str:
        keys = self._get_available_keys()
        lines = (
            f"Track (track_id={self.track_id}, species={self.species}):\n"
            f"  {', '.join(sorted(keys))}"
        )

        return lines


class TrackCollection:
    """
    Collection of all tracks for a given species.

    Reads share one open file handle: track files hold thousands of groups
    in a single file, and opening it per dataset re-parses the group
    metadata every time. The handle opens on first use, reopens if the file
    is replaced, and can be released with close() or a `with` block.

    Args:
        file_path: Path to the track HDF5 file.
        species: The species number.
        lazy: Whether to use lazy loading via dask.
    """

    def __init__(self, file_path: str, species: int, lazy: bool = False):
        self.file_path = file_path
        self.species = species
        self.lazy = lazy
        self._tracks: Dict[str, Track] = {}
        self._track_ids: Optional[np.ndarray] = None
        self._track_ids_set: Optional[set] = None
        self._file = None
        self._file_stat = None
        self._file_pid = None
        self._file_lock = threading.RLock()

    def handle(self):
        """The shared read-only file handle, (re)opened as needed."""
        with self._file_lock:
            stat = os.stat(self.file_path)
            stat_key = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            if (
                self._file is None
                or not self._file
                or self._file_pid != os.getpid()
                or self._file_stat != stat_key
            ):
                if self._file_stat is not None and self._file_stat != stat_key:
                    # the file changed: cached ids and keys may be stale
                    self._track_ids = None
                    self._track_ids_set = None
                    for track in self._tracks.values():
                        track._available_keys = None
                self.close()
                self._file = open_h5(self.file_path)
                self._file_stat = stat_key
                self._file_pid = os.getpid()
            return self._file

    def close(self) -> None:
        """Close the shared file handle; it reopens on the next read."""
        with self._file_lock:
            if self._file is not None and self._file_pid == os.getpid():
                try:
                    self._file.close()
                except Exception:
                    pass
            self._file = None
            self._file_stat = None
            self._file_pid = None

    def __enter__(self) -> "TrackCollection":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_file_stat"] = None
        state["_file_pid"] = None
        state["_file_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file_lock = threading.RLock()

    # reads hold the lock for their whole file access so close() in another
    # thread cannot pull the handle out from under them
    def _read(self, group_name: str, key: str) -> np.ndarray:
        with self._file_lock:
            return self.handle()[group_name][key][:]

    def _list_keys(self, group_name: str) -> List[str]:
        with self._file_lock:
            return list(self.handle()[group_name].keys())

    def _dataset_meta(self, group_name: str, key: str):
        with self._file_lock:
            dataset = self.handle()[group_name][key]
            return dataset.shape, dataset.dtype

    @property
    def track_ids(self) -> np.ndarray:
        """Array of all track IDs in this collection (format: 'rank-tag')."""
        if self._track_ids is None:
            with self._file_lock:
                names = list(self.handle().keys())
            ids = []
            for name in names:
                rank, _, tag = name.partition("-")
                if rank.isdigit() and tag.isdigit():
                    ids.append(name)
                else:
                    logger.warning(
                        f"Ignoring group '{name}' in {self.file_path}: "
                        f"not a rank-tag track."
                    )
            # Sort by (MPI rank, tag) numerically
            ids.sort(key=lambda x: tuple(map(int, x.split("-"))))
            self._track_ids = np.array(ids)
            self._track_ids_set = set(ids)
        return self._track_ids

    def __len__(self) -> int:
        return len(self.track_ids)

    def __iter__(self) -> Iterator[Track]:
        for track_id in self.track_ids:
            yield self[track_id]

    def __getitem__(self, track_id: str) -> Track:
        """Get a track by its ID (format: 'rank-tag')."""
        if track_id not in self._tracks:
            # Trigger track_ids cache population (also populates _track_ids_set)
            _ = self.track_ids
            if track_id not in self._track_ids_set:
                raise KeyError(
                    f"Track ID '{track_id}' not found for species {self.species}."
                )
            self._tracks[track_id] = Track(
                file_path=self.file_path,
                group_name=track_id,
                track_id=track_id,
                species=self.species,
                lazy=self.lazy,
                collection=self,
            )
        return self._tracks[track_id]

    def load_dataset(
        self, key: str, track_ids=None
    ) -> Dict[str, Union[np.ndarray, da.Array]]:
        """Load one dataset for many tracks in a single pass over the file.

        Args:
            key: Dataset name, e.g. "x1" or "ene".
            track_ids: Iterable of track IDs; all tracks when None.

        Returns:
            {track_id: array}. Tracks may have different lengths, so values
            are returned per track rather than stacked.
        """
        ids = self.track_ids if track_ids is None else list(track_ids)
        result = {}
        with self._file_lock:
            handle = self.handle()
            for track_id in ids:
                try:
                    dataset = handle[track_id][key]
                except KeyError:
                    if track_id not in handle:
                        raise KeyError(
                            f"Track ID '{track_id}' not found for species "
                            f"{self.species}."
                        ) from None
                    raise KeyError(
                        f"Dataset '{key}' not available for track {track_id}."
                    ) from None
                if self.lazy:
                    shape, dtype = dataset.shape, dataset.dtype

                    def loader(tid=track_id, k=key):
                        with open_h5(self.file_path) as f:
                            return f[tid][k][:]

                    result[track_id] = da.from_delayed(
                        delayed(loader)(), shape=shape, dtype=dtype
                    )
                else:
                    result[track_id] = dataset[:]
        return result
