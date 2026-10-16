"""All metrics of one episode."""

from __future__ import annotations

from typing import Any

from .efficiency import efficiency_metrics
from .episode import Episode
from .mission import mission_metrics


def evaluate(ep: Episode) -> dict[str, Any]:
    """Every metric of `ep`, keyed by the names in `vbar.metrics.METRICS`."""
    return {**mission_metrics(ep), **efficiency_metrics(ep)}
