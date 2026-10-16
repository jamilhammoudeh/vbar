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
    MetricInfo("E1", "delta_v", "m/s"),
    MetricInfo("E1b", "delta_v_axes", "m/s"),
    MetricInfo("E2", "propellant_mass", "kg"),
    MetricInfo("E4", "burn_count", "count"),
    MetricInfo("E5", "duty_cycle", "fraction"),
    MetricInfo("E6", "chatter", "1/s"),
)

METRICS = {info.name: info for info in CATALOGUE}
