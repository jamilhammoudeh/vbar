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
    MetricInfo("E3", "optimality_gap", "ratio"),
    MetricInfo("E3u", "optimality_gap_upper", "ratio"),
    MetricInfo("E3f", "below_lower_bound", "bool"),
    MetricInfo("E4", "burn_count", "count"),
    MetricInfo("E5", "duty_cycle", "fraction"),
    MetricInfo("E6", "chatter", "1/s"),
    MetricInfo("S1", "min_margin", "per constraint"),
    MetricInfo("S1b", "min_margin_bound", "per constraint"),
    MetricInfo("S2", "violation_depth", "per constraint"),
    MetricInfo("S3", "violation_duration", "s"),
    MetricInfo("S4", "passive_safety_margin", "m"),
    MetricInfo("S4b", "passive_safe_fraction", "fraction"),
    MetricInfo("S4i", "passive_forever_fraction", "fraction"),
    MetricInfo("S4t", "passive_threat_time", "s"),
    MetricInfo("S4o", "stuck_thruster_margin", "m"),
    MetricInfo("S5", "closing_speed_ratio", "ratio"),
    MetricInfo("S6", "min_time_to_keep_out", "s"),
    MetricInfo("S8", "cam_margin", "m"),
    MetricInfo("S9", "unprotected_time", "s"),
)

METRICS = {info.name: info for info in CATALOGUE}
