"""Benchmark metrics: functions of a recorded episode (see docs/metrics.md)."""

from .agent import (
    AgentReport,
    Posterior,
    agent_metrics,
    belief_consistency,
    faithfulness,
    parse_belief,
    solver_metrics,
)
from .composite import WEIGHTS, composite_metrics, composite_score, infractions, weight_sensitivity
from .efficiency import FuelBounds, Oracle, efficiency_metrics, optimality_gap
from .episode import Episode, FilterPair, record_episode, record_pair
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
    "FilterPair",
    "FuelBounds",
    "MetricInfo",
    "MetricSettings",
    "Oracle",
    "Posterior",
    "Signal",
    "Trace",
    "TravellingEllipse",
    "WEIGHTS",
    "cam_impulses",
    "agent_metrics",
    "belief_consistency",
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
    "record_pair",
    "safety_metrics",
    "solver_metrics",
    "success",
    "summarize",
    "total",
    "unsafe_tail",
    "weight_sensitivity",
]
