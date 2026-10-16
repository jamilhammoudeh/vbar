"""Identifiers, names, units and versions of the metrics (METRICS.md)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricInfo:
    """A metric's catalogue entry; `version` changes whenever its definition does."""

    id: str
    name: str
    units: str
    version: int = 1


CATALOGUE = (
    MetricInfo("M1", "success", "bool"),
    MetricInfo("M2", "terminal_pos_error", "m"),
    MetricInfo("M3", "terminal_vel_error", "m/s"),
    MetricInfo("M4", "time_to_complete", "s"),
    MetricInfo("M5", "progress", "fraction"),
)

METRICS = {info.name: info for info in CATALOGUE}
