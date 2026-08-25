"""Find spikes, stuck sensors and baseline shifts in time-series readings."""

from .detectors import Finding, flatlines, level_shifts, spikes
from .engine import analyse

__all__ = ["Finding", "analyse", "flatlines", "level_shifts", "spikes"]
__version__ = "0.1.0"
