import os
import re
import logging
import h5py
import numpy as np
import dask.array as da
from dask.delayed import delayed
from typing import Union, List, Optional, Dict

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
    """

    def __init__(self, file_path: str, group_name: str, track_id: str, 
                 species: int, lazy: bool = False):
        self.file_path = file_path
        self.track_id = track_id  # Format: "rank-tag", e.g., "0-1465"
        self.species = species
        self.lazy = lazy
        self._group_name = group_name
        self._data_cache: Dict[str, np.ndarray] = {}
        self._available_keys: Optional[List[str]] = None

    def _get_available_keys(self) -> List[str]:
        """Get list of available datasets for this track."""
        if self._available_keys is None:
            with h5py.File(self.file_path, 'r') as f:
                self._available_keys = list(f[self._group_name].keys())
        return self._available_keys

    def _load_dataset(self, key: str) -> Union[np.ndarray, da.Array]:
        """Load a dataset from the track file."""
        if key not in self._data_cache:
            if key not in self._get_available_keys():
                raise AttributeError(
                    f"Dataset '{key}' not available for track {self.track_id}. "
                )
            
            if self.lazy:
                with h5py.File(self.file_path, 'r') as f:
                    shape = f[self._group_name][key].shape
                    dtype = f[self._group_name][key].dtype
                
                def loader(k=key):
                    with h5py.File(self.file_path, 'r') as f:
                        return f[self._group_name][k][:]
                
                delayed_obj = delayed(loader)()
                self._data_cache[key] = da.from_delayed(delayed_obj, shape=shape, dtype=dtype)
            else:
                with h5py.File(self.file_path, 'r') as f:
                    self._data_cache[key] = f[self._group_name][key][:]
        
        return self._data_cache[key]

    @property
    def x1(self) -> Union[np.ndarray, da.Array]:
        """X coordinate over time."""
        return self._load_dataset('x1')

    @property
    def x2(self) -> Union[np.ndarray, da.Array]:
        """Y coordinate over time."""
        return self._load_dataset('x2')

    @property
    def x3(self) -> Union[np.ndarray, da.Array]:
        """Z coordinate over time."""
        return self._load_dataset('x3')

    @property
    def p1(self) -> Union[np.ndarray, da.Array]:
        """X component of momentum over time."""
        return self._load_dataset('p1')

    @property
    def p2(self) -> Union[np.ndarray, da.Array]:
        """Y component of momentum over time."""
        return self._load_dataset('p2')

    @property
    def p3(self) -> Union[np.ndarray, da.Array]:
        """Z component of momentum over time."""
        return self._load_dataset('p3')

    def __getattr__(self, name: str) -> Union[np.ndarray, da.Array]:
        """Allow access to any dataset in the track file."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
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
    Collection of all tracks for a given species. Provides iteration and bulk access to tracks.

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

    @property
    def track_ids(self) -> np.ndarray:
        """Array of all track IDs in this collection (format: 'rank-tag')."""
        if self._track_ids is None:
            with h5py.File(self.file_path, 'r') as f:
                ids = list(f.keys())
            # Sort by (MPI rank, tag) numerically
            ids.sort(key=lambda x: tuple(map(int, x.split('-'))))
            self._track_ids = np.array(ids)
        return self._track_ids

    def __getitem__(self, track_id: str) -> Track:
        """Get a track by its ID (format: 'rank-tag')."""
        if track_id not in self._tracks:
            if track_id not in self.track_ids:
                raise KeyError(
                    f"Track ID '{track_id}' not found for species {self.species}."
                )
            self._tracks[track_id] = Track(
                file_path=self.file_path,
                group_name=track_id,
                track_id=track_id,
                species=self.species,
                lazy=self.lazy
            )
        return self._tracks[track_id]