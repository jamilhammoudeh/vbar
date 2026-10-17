"""Benchmark metrics: functions of a recorded episode (see docs/metrics.md)."""

from .agent import AgentReport, agent_metrics, faithfulness, parse_belief
from .efficiency import FuelBounds, Oracle, efficiency_metrics, optimality_gap
from .episode import Episode, record_episode
from .evaluate import evaluate
from .filter import filter_metrics, iqm, paired_filter_metrics
from .mission import Contact, captured, completion_time, contact, mission_metrics, success
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
    "AgentReport",
    "Contact",
    "Episode",
    "FuelBounds",
    "MetricInfo",
    "MetricSettings",
    "Oracle",
    "Signal",
    "Trace",
    "TravellingEllipse",
    "cam_impulses",
    "agent_metrics",
    "captured",
    "complement",
    "completion_time",
    "contact",
    "efficiency_metrics",
    "escape_arc",
    "evaluate",
    "exhausts",
    "faithfulness",
    "filter_metrics",
    "iqm",
    "merge",
    "minimum",
    "mission_metrics",
    "negative_intervals",
    "optimality_gap",
    "paired_filter_metrics",
    "parse_belief",
    "passive_forever",
    "passive_metrics",
    "plume_metrics",
    "plume_normalization",
    "record_episode",
    "safety_metrics",
    "success",
    "total",
]
