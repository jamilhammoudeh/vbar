"""Benchmark metrics: functions of a recorded episode (see docs/metrics.md)."""

from .agent import AgentReport, agent_metrics, faithfulness, parse_belief
from .composite import WEIGHTS, composite_metrics, composite_score, infractions, weight_sensitivity
from .efficiency import FuelBounds, Oracle, efficiency_metrics, optimality_gap
from .episode import Episode, record_episode
from .evaluate import evaluate
from .filter import (
    filter_behaviour,
    filter_conservatism,
    filter_metrics,
    iqm,
    paired_filter_metrics,
)
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
from .summary import clopper_pearson, summarize, unsafe_tail
from .taxonomy import LABELS, failure_label

__all__ = [
    "METRICS",
    "AgentReport",
    "LABELS",
    "Contact",
    "Episode",
    "FuelBounds",
    "MetricInfo",
    "MetricSettings",
    "Oracle",
    "Signal",
    "Trace",
    "TravellingEllipse",
    "WEIGHTS",
    "cam_impulses",
    "agent_metrics",
    "captured",
    "clopper_pearson",
    "complement",
    "completion_time",
    "composite_metrics",
    "composite_score",
    "contact",
    "efficiency_metrics",
    "escape_arc",
    "evaluate",
    "exhausts",
    "failure_label",
    "faithfulness",
    "filter_behaviour",
    "filter_conservatism",
    "filter_metrics",
    "infractions",
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
    "summarize",
    "total",
    "unsafe_tail",
    "weight_sensitivity",
]
