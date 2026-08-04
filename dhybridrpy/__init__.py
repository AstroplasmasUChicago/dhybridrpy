import importlib
from typing import TYPE_CHECKING

# Submodules import lazily so that lightweight consumers -- notably the
# spawned worker processes in _parallel, which need none of the plotting
# stack -- don't pay for matplotlib and dask at import time.
_EXPORTS = {
    "DHybridrpy": ".dhybridrpy",
    "Timestep": ".containers",
    "Field": ".data",
    "Phase": ".data",
    "Raw": ".data",
    "close_pooled_handles": ".data",
    "fft_power_iso": ".data",
    "fft_power_1d_slices": ".data",
    "Track": ".tracks",
    "TrackCollection": ".tracks",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:
    from .containers import Timestep as Timestep
    from .data import (
        Field as Field,
        Phase as Phase,
        Raw as Raw,
        close_pooled_handles as close_pooled_handles,
        fft_power_iso as fft_power_iso,
        fft_power_1d_slices as fft_power_1d_slices,
    )
    from .dhybridrpy import DHybridrpy as DHybridrpy
    from .tracks import Track as Track, TrackCollection as TrackCollection


def __getattr__(name):
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
