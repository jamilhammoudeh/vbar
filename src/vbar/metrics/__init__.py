"""Benchmark metrics: functions of a recorded episode (see docs/metrics.md)."""

from .efficiency import FuelBounds, Oracle, efficiency_metrics, optimality_gap
from .episode import Episode, record_episode
from .evaluate import evaluate
from .mission import captured, mission_metrics, success
from .passive import (
    TravellingEllipse,
    cam_impulses,
    escape_arc,
    passive_forever,
    passive_metrics,
)
from .plume import exhausts, plume_metrics, plume_normalization
from .registry import METRICS, MetricInfo
from .safety import safety_metrics
from .settings import MetricSettings
from .stl import Signal, Trace, complement, merge, minimum, negative_intervals, total

__all__ = [
    "METRICS",
    "Episode",
    "FuelBounds",
    "MetricInfo",
    "MetricSettings",
    "Oracle",
    "Signal",
    "Trace",
    "TravellingEllipse",
    "cam_impulses",
    "captured",
    "complement",
    "efficiency_metrics",
    "escape_arc",
    "evaluate",
    "exhausts",
    "merge",
    "minimum",
    "mission_metrics",
    "negative_intervals",
    "optimality_gap",
    "passive_forever",
    "passive_metrics",
    "plume_metrics",
    "plume_normalization",
    "record_episode",
    "safety_metrics",
    "success",
    "total",
]
