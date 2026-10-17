"""Identifiers, names, units and versions of the metrics (docs/metrics.md)."""

from __future__ import annotations

from dataclasses import dataclass

# How each metric is summarised across episodes: performance by its interquartile mean; safety by
# its unsafe tail, since the interquartile mean drops the quarter of runs where near misses are
# (Kochenderfer et al., Algorithms for Decision Making, 14.1); booleans as rates.
SUMMARIES = ("iqm", "lower_tail", "upper_tail", "rate", "none")


@dataclass(frozen=True)
class MetricInfo:
    """A metric's catalogue entry; `version` changes whenever its definition does.

    `summary` is how it ranks across episodes: "iqm", "lower_tail" (lower is less safe),
    "upper_tail" (higher is less safe), "rate", or "none" for labels, dictionaries and metrics
    that are already aggregates.
    """

    id: str
    name: str
    units: str
    version: int = 1
    summary: str = "iqm"

    def __post_init__(self):
        if self.summary not in SUMMARIES:
            raise ValueError(f"unknown summary {self.summary!r}, expected one of {SUMMARIES}")


CATALOGUE = (
    MetricInfo("M1", "success", "bool", summary="rate"),
    MetricInfo("M2", "terminal_pos_error", "m"),
    MetricInfo("M3", "terminal_vel_error", "m/s"),
    MetricInfo("M4", "time_to_complete", "s"),
    MetricInfo("M5", "progress", "fraction"),
    MetricInfo("M6", "contact_lateral_error", "m", summary="upper_tail"),
    MetricInfo("M7", "contact_axial_speed", "m/s"),
    MetricInfo("M8", "contact_lateral_speed", "m/s", summary="upper_tail"),
    MetricInfo("M9", "hold_dwell", "s"),
    MetricInfo("M10", "capture_margin", "ratio", summary="lower_tail"),
    MetricInfo("E1", "delta_v", "m/s"),
    MetricInfo("E1b", "delta_v_axes", "m/s"),
    MetricInfo("E2", "propellant_mass", "kg"),
    MetricInfo("E3", "optimality_gap", "ratio"),
    MetricInfo("E3m", "optimality_gap_model", "ratio"),
    MetricInfo("E3u", "optimality_gap_upper", "ratio"),
    MetricInfo("E3f", "below_lower_bound", "bool", summary="rate"),
    MetricInfo("E3c", "bound_certified", "bool", summary="rate"),
    MetricInfo("E3r", "bound_rung", "label", summary="none"),
    MetricInfo("E4", "burn_count", "count"),
    MetricInfo("E5", "duty_cycle", "fraction"),
    MetricInfo("E6", "chatter", "1/s"),
    MetricInfo("S1", "min_margin", "per constraint", summary="lower_tail"),
    MetricInfo("S1b", "min_margin_bound", "per constraint", summary="lower_tail"),
    MetricInfo("S2", "violation_depth", "per constraint", summary="upper_tail"),
    MetricInfo("S3", "violation_duration", "s", summary="upper_tail"),
    MetricInfo("S4", "passive_safety_margin", "m", summary="lower_tail"),
    MetricInfo("S4b", "passive_safe_fraction", "fraction", summary="lower_tail"),
    MetricInfo("S4i", "passive_forever_fraction", "fraction", summary="lower_tail"),
    MetricInfo("S4t", "passive_threat_time", "s", summary="lower_tail"),
    MetricInfo("S4o", "stuck_thruster_margin", "m", summary="lower_tail"),
    MetricInfo("S5", "closing_speed_ratio", "ratio", summary="upper_tail"),
    MetricInfo("S6", "min_time_to_keep_out", "s", summary="lower_tail"),
    MetricInfo("S8", "cam_margin", "m", summary="lower_tail"),
    MetricInfo("S9", "unprotected_time", "s", summary="upper_tail"),
    MetricInfo("S10", "plume_impulse", "N s", summary="upper_tail"),
    MetricInfo("S10t", "plume_time", "s", summary="upper_tail"),
    MetricInfo("S10m", "plume_margin", "rad", summary="lower_tail"),
    MetricInfo("F1", "intervention_rate", "fraction", version=2),
    MetricInfo("F1c", "clip_rate", "fraction"),
    MetricInfo("F2", "intervention_size", "m/s^2", version=2),
    MetricInfo("F2n", "intervention_size_normalized", "fraction of the widest change"),
    MetricInfo("F3", "safety_dv_cost", "m/s"),
    MetricInfo("F4", "filter_dependence", "difference of rates", summary="none"),
    MetricInfo("F5", "infeasible_steps", "count", summary="upper_tail"),
    MetricInfo("F5b", "solver_failures", "count", summary="upper_tail"),
    MetricInfo("F6", "filter_compute_median", "s"),
    MetricInfo("F6p", "filter_compute_p95", "s"),
    MetricInfo("A1", "latency_median", "s"),
    MetricInfo("A1p", "latency_p95", "s"),
    MetricInfo("A2", "compute", "s"),
    MetricInfo("A3i", "input_tokens", "tokens"),
    MetricInfo("A3o", "output_tokens", "tokens"),
    MetricInfo("A3", "token_cost", "USD"),
    MetricInfo("A4", "parse_failure_rate", "fraction"),
    MetricInfo("A5", "reasoning_faithfulness", "fraction"),
    MetricInfo("A6", "sign_check_pass", "bool", summary="rate"),
    MetricInfo("A7", "training_cost", "dict", summary="none"),
    MetricInfo("A8", "cost_of_learning", "dict", summary="none"),
    MetricInfo("C1", "composite_score", "score"),
    MetricInfo("C1c", "completion", "fraction"),
    MetricInfo("C1n", "infractions", "count per type", summary="none"),
    MetricInfo("T1", "failure_label", "label", summary="none"),
)

METRICS = {info.name: info for info in CATALOGUE}
