"""All metrics of one episode."""

from __future__ import annotations

from typing import Any

from .agent import AgentReport, agent_metrics
from .efficiency import FuelBounds, efficiency_metrics, optimality_gap
from .episode import Episode
from .filter import filter_metrics
from .mission import mission_metrics
from .passive import passive_metrics
from .plume import plume_metrics
from .safety import safety_metrics
from .settings import MetricSettings


def evaluate(
    ep: Episode,
    settings: MetricSettings | None = None,
    bounds: FuelBounds | None = None,
    report: AgentReport | None = None,
) -> dict[str, Any]:
    """Every metric of `ep`, keyed by the names in `vbar.metrics.METRICS`.

    `settings` are the scenario's declared metric parameters, `bounds` the oracle's fuel bounds
    for the episode's start state (E3) and `report` the facts about the agent from outside the
    episode (A6 to A8, token prices).
    """
    return {
        **mission_metrics(ep),
        **efficiency_metrics(ep),
        **optimality_gap(ep, bounds),
        **safety_metrics(ep),
        **passive_metrics(ep, settings),
        **plume_metrics(ep, settings),
        **filter_metrics(ep, settings),
        **agent_metrics(ep, report),
    }
