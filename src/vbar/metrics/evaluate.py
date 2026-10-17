"""All metrics of one episode."""

from __future__ import annotations

from typing import Any

from .efficiency import FuelBounds, efficiency_metrics, optimality_gap
from .episode import Episode
from .mission import mission_metrics
from .passive import passive_metrics
from .plume import plume_metrics
from .safety import safety_metrics
from .settings import MetricSettings


def evaluate(
    ep: Episode, settings: MetricSettings | None = None, bounds: FuelBounds | None = None
) -> dict[str, Any]:
    """Every metric of `ep`, keyed by the names in `vbar.metrics.METRICS`.

    `settings` are the scenario's declared metric parameters and `bounds` the oracle's fuel
    bounds for the episode's start state (E3).
    """
    return {
        **mission_metrics(ep),
        **efficiency_metrics(ep),
        **optimality_gap(ep, bounds),
        **safety_metrics(ep),
        **passive_metrics(ep, settings),
        **plume_metrics(ep, settings),
    }
