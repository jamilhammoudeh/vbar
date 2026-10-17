"""Composite score C1 = completion x 1 / (1 + sum_j w_j n_j), with its breakdown.

Completion is 1 on success and the progress M5 otherwise. n_j counts infractions of each type:
one per contiguous interval in continuous time with a constraint margin below zero (keep-out
entry, approach-cone violation, speed or closing-speed breach), and one for a timeout without
success. The weights are a convention, not a measurement, so `weight_sensitivity` checks how
rankings move when each weight changes by +/- 50 %.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .episode import Episode
from .mission import mission_metrics
from .stl import Trace, negative_intervals

# Draft weights; the speed weight applies to both speed constraints.
WEIGHTS = {
    "keep_out": 0.7,
    "approach_cone": 0.5,
    "speed_limit": 0.3,
    "closing_speed_limit": 0.3,
    "timeout": 0.4,
}


def _completion_and_counts(
    ep: Episode, mission: Mapping[str, Any] | None = None
) -> tuple[float, dict[str, int]]:
    mission = mission or mission_metrics(ep)
    trace = Trace(ep)
    counts = {c.name: len(negative_intervals(c, trace)) for c in ep.constraints if ep.records}
    counts["timeout"] = int(ep.truncated and not mission["success"])
    done = 1.0 if mission["success"] else float(mission["progress"])
    return done, counts


def infractions(ep: Episode) -> dict[str, int]:
    """Infraction counts for every constraint of the episode, and timeouts.

    Constraints without a weight count but do not lower the score.
    """
    return _completion_and_counts(ep)[1]


def composite_score(
    done: float, counts: Mapping[str, int], weights: Mapping[str, float] = WEIGHTS
) -> float:
    """C1 from the completion and the infraction counts."""
    return done / (1.0 + sum(weights.get(name, 0.0) * n for name, n in counts.items()))


def composite_metrics(
    ep: Episode,
    weights: Mapping[str, float] = WEIGHTS,
    mission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """C1 and the breakdown it must be reported with; `mission` reuses computed M1 to M5."""
    done, counts = _completion_and_counts(ep, mission)
    return {
        "composite_score": composite_score(done, counts, weights),
        "completion": done,
        "infractions": counts,
    }


def weight_sensitivity(
    runs: Mapping[str, Sequence[Episode]],
    weights: Mapping[str, float] = WEIGHTS,
    factors: Sequence[float] = (0.5, 1.5),
) -> dict[str, Any]:
    """Rankings of agents by mean C1 at the given weights and with each weight scaled.

    Returns the base ranking, the ranking for every (weight, factor) and whether all of them
    equal the base ranking.
    """
    scored = {agent: [_completion_and_counts(ep) for ep in eps] for agent, eps in runs.items()}

    def ranking(w: Mapping[str, float]) -> tuple[str, ...]:
        means = {
            agent: float(np.mean([composite_score(d, c, w) for d, c in rows]))
            for agent, rows in scored.items()
        }
        return tuple(sorted(means, key=lambda agent: (-means[agent], agent)))

    base = ranking(weights)
    varied = {
        (name, factor): ranking({**weights, name: weights[name] * factor})
        for name in weights
        for factor in factors
    }
    return {
        "ranking": base,
        "varied": varied,
        "stable": all(order == base for order in varied.values()),
    }
