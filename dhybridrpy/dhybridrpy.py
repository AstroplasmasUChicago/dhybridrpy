import io
import logging
import math
import os
import re

import f90nml
import numpy as np
from f90nml import Namelist

from .containers import Timestep
from .data import Field, Phase, Raw, open_h5
from .tracks import Track, TrackCollection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFileParser:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.input_dict = self._parse_input_file()

    def _parse_input_file(self) -> Namelist:
        """
        Parses the input file and returns its content as a subclass of dictionary.
        """
        try:
            nml_content = self._create_nml_input_str()
            return f90nml.read(io.StringIO(nml_content))
        except Exception as e:
            logger.error(f"Failed to parse input file: {e}")
            raise

    def _create_nml_input_str(self) -> str:
        """
        Converts the input file content to a Fortran namelist format.
        """
        with open(self.input_file, "r") as infile:
            content = infile.read()

        # Regular expression to match sections with curly braces
        SECTION_PATTERN = re.compile(r"(\w+)\s*\{([^}]*)\}", re.DOTALL)

        # Generate namelist content
        namelist_content = []
        for match in SECTION_PATTERN.finditer(content):
            section_name, parameters = match.group(1), match.group(2).strip()

            # Begin the namelist section
            namelist_content.append(f"&{section_name}")
            namelist_content.extend(self._process_parameters(parameters))
            namelist_content.append("/")  # End the namelist section

        return "\n".join(namelist_content)

    def _process_parameters(self, parameters: str) -> list:
        """
        Processes the parameters within a section, retaining comments
        and ensuring valid namelist syntax.
        """
        processed_lines = []
        for line in parameters.splitlines():
            line = line.strip()
            if line.startswith("!") or not line:
                # Retain comments or skip empty lines
                processed_lines.append(line)
            else:
                # Ensure proper formatting by removing trailing commas
                processed_lines.append(line.rstrip(","))
        return processed_lines


class DHybridrpy:
    """
    Class for processing dHybridR input and output files.

    Args:
        input_file: Path to the dHybridR input file.
        output_folder: Path to the dHybridR output folder.
        lazy: Enables lazy loading of data via the dask library.
        exclude_timestep_zero: Excludes the zeroth timestep, if present, from the list of timesteps.
    """

    _FIELD_MAPPING = {
        "Magnetic": "B",
        "Electric": "E",
        "CurrentDens": "J",
        "ExtAccel": "Accel",
    }
    _PHASE_MAPPING = {"FluidVel": "V", "PressureTen": "P"}
    _COMPONENT_MAPPING = {"Intensity": "magnitude"}
    _SPECIES_PATTERN = re.compile(r"\d+")
    _TIMESTEP_PATTERN = re.compile(r"_(\d+)\.h5$")
    _TIME_NDECIMALS = 6

    def __init__(
        self,
        input_file: str,
        output_folder: str,
        lazy: bool = False,
        exclude_timestep_zero: bool = True,
    ):
        self.input_file = input_file
        self.output_folder = output_folder
        self.lazy = lazy
        self.exclude_timestep_zero = exclude_timestep_zero
        self._timesteps_dict = {}
        self._sorted_timesteps_cache = {}
        self._field_phase_timesteps = set()
        self._raw_timesteps = set()
        self._track_collections = {}  # {species: TrackCollection}
        self._timestep_times = {}  # {timestep: float} cache of TIME from HDF5
        self._validate_paths()
        self.inputs = InputFileParser(input_file).input_dict
        self._get_time_inputs()
        self._decide_time_mode()
        self._traverse_directory()
        self._discover_tracks()

    def _get_time_inputs(self) -> None:
        try:
            time_dict = self.inputs["time"]
            self.dt = time_dict["dt"]
        except KeyError as e:
            missing_key = e.args[0]
            raise KeyError(
                f"Required key '{missing_key}' not found in input file's "
                f"'time' section."
            ) from e
        # float(): f90nml parses literals like dt=1 as int
        self.dt = float(self.dt)
        # t0 is deprecated in dHybridR (never applied); old decks may still set it
        self.start_time = float(time_dict.get("t0", 0.0))
        self.adaptive_dt = time_dict.get("adaptive_dt", False)

    def _validate_paths(self) -> None:
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file {self.input_file} does not exist.")
        if not os.path.isdir(self.output_folder):
            raise NotADirectoryError(
                f"Output folder {self.output_folder} is not a directory."
            )

    def _decide_time_mode(self) -> None:
        """Decide whether times can be computed as timestep*dt instead of read
        from every file, which costs one file open per timestep. The earliest
        and latest output files are checked first; if their TIME attributes
        disagree with timestep*dt, times are read from files as before. Files
        that disagree with each other about dt raise an error, since a run
        with fixed dt cannot produce them.
        """
        self._derive_times = False
        if self.adaptive_dt:
            return

        candidates = []
        for dirpath, _, filenames in os.walk(self.output_folder):
            top = os.path.relpath(dirpath, self.output_folder).split(os.sep)[0]
            if top not in ("Fields", "Phase", "Raw"):
                continue
            for filename in filenames:
                match = self._TIMESTEP_PATTERN.search(filename)
                if match:
                    candidates.append(
                        (int(match.group(1)), os.path.join(dirpath, filename))
                    )
        if not candidates:
            return

        nonzero = [c for c in candidates if c[0] != 0] or candidates
        samples = []
        for timestep, filepath in sorted({min(nonzero), max(nonzero)}):
            try:
                with open_h5(filepath) as f:
                    file_time = float(f.attrs["TIME"][0])
            except (OSError, KeyError):
                logger.warning(
                    f"Could not read TIME from {filepath}; "
                    f"reading times from files instead."
                )
                return
            samples.append((timestep, filepath, file_time))

        # dt each file claims it was written with (TIME / iteration);
        # iteration-0 files say nothing about dt
        implied_dt = [
            (time / iteration, path) for iteration, path, time in samples
            if iteration != 0
        ]
        if len(implied_dt) > 1:
            (dt_lo, file_lo), (dt_hi, file_hi) = min(implied_dt), max(implied_dt)
            if not math.isclose(dt_lo, dt_hi, rel_tol=1e-5):
                raise ValueError(
                    f"Output folder contains files written with different time "
                    f"steps: {file_lo} implies dt={dt_lo:.6g} but {file_hi} "
                    f"implies dt={dt_hi:.6g}. A run with fixed dt cannot "
                    f"produce this; check that the output folder holds a "
                    f"single run's output."
                )

        for timestep, filepath, file_time in samples:
            derived = timestep * self.dt
            if not math.isclose(derived, file_time, rel_tol=1e-5, abs_tol=1e-9):
                logger.warning(
                    f"TIME attribute in {filepath} ({file_time}) does not match "
                    f"iteration*dt ({derived}); reading times from files instead."
                )
                return
        self._derive_times = True

    def _time_for_timestep(self, filepath: str, timestep: int) -> float:
        """Time for a timestep: derived as timestep*dt, or read from HDF5."""
        if timestep not in self._timestep_times:
            if self._derive_times:
                self._timestep_times[timestep] = timestep * self.dt
            else:
                with open_h5(filepath) as f:
                    self._timestep_times[timestep] = float(f.attrs["TIME"][0])
        return self._timestep_times[timestep]

    def _process_file(
        self, dirpath: str, filename: str, timestep: int, folder_components: list
    ) -> None:
        output_type = folder_components[0]

        if output_type == "Fields":
            self._process_field(dirpath, filename, timestep, folder_components)
        elif output_type == "Phase":
            self._process_phase(dirpath, filename, timestep, folder_components)
        elif output_type == "Raw":
            self._process_raw(dirpath, filename, timestep, folder_components)
        else:
            logger.warning(
                f"Unknown output type '{output_type}' for {filename}. File not processed."
            )

    def _process_field(
        self, dirpath: str, filename: str, timestep: int, folder_components: list
    ) -> None:
        category = folder_components[1]
        if category == "CurrentDens":
            folder_components.insert(2, "Total")
        field_type = folder_components[-2]
        component = folder_components[-1]
        if component in self._COMPONENT_MAPPING:
            component = self._COMPONENT_MAPPING[component]

        prefix = self._FIELD_MAPPING.get(category)
        if not prefix:
            logger.warning(f"Unknown category '{category}'. Skipping {filename}")
            return

        name = f"{prefix}{component}"
        if timestep not in self._timesteps_dict:
            self._timesteps_dict[timestep] = Timestep(timestep)
        self._field_phase_timesteps.add(timestep)
        filepath = os.path.join(dirpath, filename)
        time = self._time_for_timestep(filepath, timestep)
        field = Field(
            filepath,
            name,
            timestep,
            time,
            self._TIME_NDECIMALS,
            self.lazy,
            field_type,
        )
        self._timesteps_dict[timestep].add_field(field)

    def _process_phase(
        self, dirpath: str, filename: str, timestep: int, folder_components: list
    ) -> None:
        name = folder_components[1]

        # Manage bulk velocity, pressure tensor, and scalar pressure special cases
        if name == "FluidVel" or name == "PressureTen":
            species_str = folder_components[-2]
            component = folder_components[-1]
            if component in self._COMPONENT_MAPPING:
                component = self._COMPONENT_MAPPING[component]
            prefix = self._PHASE_MAPPING.get(name)
            if not prefix:
                logger.warning(f"Unknown name '{name}'. Skipping {filename}")
                return
            name = f"{prefix}{component}"
        else:
            species_str = folder_components[-1]

        if name == "x3x2x1" and "pres" in filename:
            name = "P"

        if species_str == "Total":
            species = species_str
        else:
            match = self._SPECIES_PATTERN.search(species_str)
            if match is None:
                logger.warning(
                    f"Could not parse species from folder '{species_str}'; "
                    f"skipping {filename}"
                )
                return
            species = int(match.group())
        if timestep not in self._timesteps_dict:
            self._timesteps_dict[timestep] = Timestep(timestep)
        self._field_phase_timesteps.add(timestep)
        filepath = os.path.join(dirpath, filename)
        time = self._time_for_timestep(filepath, timestep)
        phase = Phase(
            filepath,
            name,
            timestep,
            time,
            self._TIME_NDECIMALS,
            self.lazy,
            species,
        )
        self._timesteps_dict[timestep].add_phase(phase)

    def _process_raw(
        self, dirpath: str, filename: str, timestep: int, folder_components: list
    ) -> None:
        name = "raw"
        species_str = folder_components[-1]
        match = self._SPECIES_PATTERN.search(species_str)
        if match is None:
            logger.warning(
                f"Could not parse species from folder '{species_str}'; "
                f"skipping {filename}"
            )
            return
        species = int(match.group())
        if timestep not in self._timesteps_dict:
            self._timesteps_dict[timestep] = Timestep(timestep)
        self._raw_timesteps.add(timestep)
        filepath = os.path.join(dirpath, filename)
        time = self._time_for_timestep(filepath, timestep)
        raw = Raw(filepath, name, timestep, time, self.lazy, species)
        self._timesteps_dict[timestep].add_raw(raw)

    def _traverse_directory(self) -> None:
        for dirpath, _, filenames in os.walk(self.output_folder):
            components = os.path.relpath(dirpath, self.output_folder).split(os.sep)
            for filename in filenames:
                match = self._TIMESTEP_PATTERN.search(filename)
                if match:
                    timestep = int(match.group(1))
                    # copy: _process_field mutates the list for CurrentDens
                    self._process_file(dirpath, filename, timestep, list(components))

    def timestep(self, ts: int) -> Timestep:
        """Access field, phase, and raw file information at a given timestep."""

        if ts in self._timesteps_dict:
            return self._timesteps_dict[ts]
        raise ValueError(f"Timestep {ts} not found.")

    def timestep_closest(self, ts: int, verbose: bool = False) -> Timestep:
        """Access field, phase, and raw file information at the closest available timestep."""
        timesteps = self.timesteps()
        if len(timesteps) == 0:
            raise ValueError("No timesteps available.")
        closest_ts = min(timesteps, key=lambda x: abs(x - ts))
        if verbose:
            logger.info(
                f"Requested timestep: {ts}. Closest available timestep: {closest_ts}."
            )
        return self.timestep(closest_ts)

    def timestep_index(self, index: int) -> Timestep:
        """Access field, phase, and raw file information at a given timestep index."""

        timesteps = self.timesteps()
        num_timesteps = len(timesteps)
        if -num_timesteps <= index < num_timesteps:
            return self.timestep(timesteps[index])
        raise IndexError(
            f"Index {index} is out of range. Valid range: {-num_timesteps} to {num_timesteps - 1}."
        )

    def timesteps(self) -> np.ndarray:
        """Retrieve an array of the timesteps for fields and phases."""
        if "field_phase" not in self._sorted_timesteps_cache:
            self._sorted_timesteps_cache["field_phase"] = np.sort(
                list(self._field_phase_timesteps)
            )
        sorted_ts = self._sorted_timesteps_cache["field_phase"]
        if self.exclude_timestep_zero and len(sorted_ts) > 0 and sorted_ts[0] == 0:
            return sorted_ts[1:]
        return sorted_ts

    def times(self) -> np.ndarray:
        """Retrieve an array of simulation times corresponding to each field/phase timestep."""
        return np.array([self._timestep_times[ts] for ts in self.timesteps()])

    def raw_timesteps(self) -> np.ndarray:
        """Retrieve an array of the timesteps for raw particle data."""
        if "raw" not in self._sorted_timesteps_cache:
            self._sorted_timesteps_cache["raw"] = np.sort(list(self._raw_timesteps))
        sorted_ts = self._sorted_timesteps_cache["raw"]
        if self.exclude_timestep_zero and len(sorted_ts) > 0 and sorted_ts[0] == 0:
            return sorted_ts[1:]
        return sorted_ts

    def raw_times(self) -> np.ndarray:
        """Retrieve an array of simulation times corresponding to each raw timestep."""
        return np.array([self._timestep_times[ts] for ts in self.raw_timesteps()])

    def _timeseries_paths(self, kind: str, names: list, key, timesteps) -> list:
        """File paths per timestep, one inner list per requested name."""
        if timesteps is None:
            timesteps = self.timesteps()
        grouped = []
        for ts in timesteps:
            container = getattr(self.timestep(int(ts)), kind)
            grouped.append(
                [getattr(container, name)(key).file_path for name in names]
            )
        return grouped

    def _timeseries(self, kind: str, name, key, timesteps, apply, workers):
        from . import _parallel

        names = [name] if isinstance(name, str) else list(name)
        grouped = self._timeseries_paths(kind, names, key, timesteps)
        if not grouped:
            raise ValueError("No timesteps selected.")
        if apply is None:
            if len(names) != 1:
                raise ValueError(
                    "Reading several quantities at once needs `apply`; "
                    "call once per name to load them whole."
                )
            return _parallel.gather_data([g[0] for g in grouped], workers)
        pool = _parallel.get_pool(workers)
        return list(
            pool.map(_parallel.map_data, grouped, [apply] * len(grouped))
        )

    def field_timeseries(
        self,
        name,
        type: str = "Total",
        timesteps=None,
        apply=None,
        workers: int = None,
    ):
        """One field across many timesteps, read by parallel worker processes.

        h5py cannot overlap reads across threads, so files are read by a
        pool of worker processes instead.

        Args:
            name: Field name, e.g. "Bx", or a list of names when `apply`
                combines several fields.
            type: Field type ("Total", "External", "Self").
            timesteps: Iterable of timesteps; all field/phase timesteps
                when None.
            apply: Function applied to each timestep's field(s) inside the
                workers; only its results travel back, so this scales to
                runs whose full data would not fit in memory. Must be
                importable (a module-level function such as np.mean, not a
                lambda). Receives one array per name, in order.
            workers: Worker process count (default min(8, cpu count)).

        Returns:
            Without `apply`: array of shape (num_timesteps, *grid). The
            full selection is held in memory, so pass a timesteps subset
            for very large runs. With `apply`: a list of its results in
            timestep order.
        """
        return self._timeseries("fields", name, type, timesteps, apply, workers)

    def phase_timeseries(
        self,
        name,
        species=1,
        timesteps=None,
        apply=None,
        workers: int = None,
    ):
        """One phase quantity across many timesteps; see field_timeseries."""
        return self._timeseries(
            "phases", name, species, timesteps, apply, workers
        )

    def _discover_tracks(self) -> None:
        """Discover track files in the output folder."""

        tracks_folder = os.path.join(self.output_folder, "Tracks")
        if not os.path.isdir(tracks_folder):
            return

        species_pattern = re.compile(r"Sp(\d+)")
        track_file_pattern = re.compile(r"track_Sp(\d+)\.h5$")

        for species_folder in os.listdir(tracks_folder):
            species_match = species_pattern.match(species_folder)
            if not species_match:
                continue

            species = int(species_match.group(1))
            species_path = os.path.join(tracks_folder, species_folder)

            if not os.path.isdir(species_path):
                continue

            for filename in os.listdir(species_path):
                file_match = track_file_pattern.match(filename)
                if file_match:
                    file_path = os.path.join(species_path, filename)
                    if species in self._track_collections:
                        logger.warning(
                            f"Duplicate track file for species {species}: "
                            f"'{self._track_collections[species].file_path}' "
                            f"will be replaced by '{file_path}'."
                        )
                    self._track_collections[species] = TrackCollection(
                        file_path=file_path, species=species, lazy=self.lazy
                    )

    def track(self, track_id: str, species: int = 1) -> Track:
        """
        Access a particle track by its ID.

        Args:
            track_id: The track ID in format 'rank-tag' (e.g., '0-1465').
            species: The species number (default: 1).

        Returns:
            The corresponding Track object.
        """

        if species not in self._track_collections:
            tracks_exist = list(self._track_collections.keys())
            if not tracks_exist:
                raise ValueError("No track data found in output folder.")
            raise ValueError(
                f"No tracks found for species {species}. Tracks exist for species: {tracks_exist}"
            )
        return self._track_collections[species][track_id]

    def tracks(self, species: int = 1) -> np.ndarray:
        """
        Retrieve an array of track IDs for a given species.

        Args:
            species: The species number (default: 1).

        Returns:
            Array of track IDs.
        """

        if species not in self._track_collections:
            tracks_exist = list(self._track_collections.keys())
            if not tracks_exist:
                raise ValueError("No track data found in output folder.")
            raise ValueError(
                f"No tracks found for species {species}. Tracks exist for species: {tracks_exist}"
            )
        return self._track_collections[species].track_ids
