"""All metrics of one episode."""

from __future__ import annotations

from typing import Any

from .efficiency import FuelBounds, efficiency_metrics, optimality_gap
from .episode import Episode
from .mission import mission_metrics


def evaluate(ep: Episode, bounds: FuelBounds | None = None) -> dict[str, Any]:
    """Every metric of `ep`, keyed by the names in `vbar.metrics.METRICS`.

    `bounds` are the oracle's fuel bounds for the episode's start state (E3).
    """
    return {**mission_metrics(ep), **efficiency_metrics(ep), **optimality_gap(ep, bounds)}
