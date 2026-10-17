"""Safety metrics S1 to S3, S5 and S6 from constraint margins in continuous time.

S1 is each constraint's lowest margin over the episode: the environment's in-step minima (from
the dynamics model itself) for its own constraints, or the same search on the rebuilt trajectory
for any other constraint passed in. S3 is the time each margin spent below zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ..constraints import ClosingSpeedLimit, Constraint, KeepOut, closing_speed
from .episode import Episode
from .stl import Trace, minimum, negative_intervals, peak, total

# Closing speeds below this (m/s) count as not closing in S6.
_CLOSING_FLOOR = 1e-9


def lowest_margin(ep: Episode, constraint: Constraint, trace: Trace | None = None):
    """Lowest margin and its certified lower bound over the episode, start state included.

    The environment's own minima are used only for the very constraint it searched; a different
    one, even with the same name, is searched on the rebuilt trajectory.
    """
    at_start = float(constraint.margin(ep.start, 0.0))
    own = ep.constraint(constraint.name) == constraint
    found = [rec.segment.minima.get(constraint.name) if own else None for rec in ep.records]
    if not ep.records or any(f is None for f in found):
        whole = minimum(constraint, trace or Trace(ep)) if ep.records else None
        low = whole.margin if whole else math.inf
        bound = whole.lower_bound if whole else math.inf
    else:
        low = min(f.margin for f in found if f is not None)
        bound = min(f.lower_bound for f in found if f is not None)
    return min(low, at_start), min(bound, at_start)


def closing_speed_ratio(ep: Episode, limit: ClosingSpeedLimit, trace: Trace) -> float:
    """S5: highest closing speed over the allowed v0 + v1 rho.

    Found at the samples and refined on the two sample intervals around the highest one, without
    a certificate; the S1 margin of the closing-speed constraint carries the certified bound.
    """
    if not limit.v0 > 0.0:
        raise ValueError("the closing-speed ratio needs v0 > 0")

    def ratio(state, t=0.0):
        rho = np.linalg.norm(np.asarray(state)[..., :3], axis=-1)
        return closing_speed(state) / (limit.v0 + limit.v1 * rho)

    # The first segment's first sample is the start state, so the peak covers it.
    return peak(ratio, trace)


def time_to_keep_out(ep: Episode, keep_out: KeepOut) -> float:
    """S6: least time to reach the keep-out sphere over the step boundaries (s).

    (rho - R) / closing speed while closing, 0 at any boundary inside the sphere, and infinite
    if the chaser never closes outside it.
    """
    states = ep.states
    rho = np.linalg.norm(states[:, :3], axis=1)
    closing = closing_speed(states)
    times = np.where(
        closing > _CLOSING_FLOOR,
        (rho - keep_out.radius) / np.maximum(closing, _CLOSING_FLOOR),
        math.inf,
    )
    # Already inside, there is no time left whichever way the chaser moves.
    times = np.where(rho < keep_out.radius, 0.0, times)
    return float(times.min())


def safety_metrics(ep: Episode, constraints: Iterable[Constraint] | None = None) -> dict:
    """S1, S1 bound, S2, S3 per constraint, S5 and S6 for one episode."""
    trace = Trace(ep)
    chosen = tuple(ep.constraints if constraints is None else constraints)
    low, bound, depth, duration = {}, {}, {}, {}
    for c in chosen:
        low[c.name], bound[c.name] = lowest_margin(ep, c, trace)
        depth[c.name] = max(0.0, -low[c.name])
        duration[c.name] = total(negative_intervals(c, trace)) if ep.records else 0.0
    closing = next((c for c in chosen if isinstance(c, ClosingSpeedLimit)), None)
    keep_out = next((c for c in chosen if isinstance(c, KeepOut)), None)
    return {
        "min_margin": low,
        "min_margin_bound": bound,
        "violation_depth": depth,
        "violation_duration": duration,
        "closing_speed_ratio": closing_speed_ratio(ep, closing, trace)
        if closing is not None and ep.records
        else math.nan,
        "min_time_to_keep_out": time_to_keep_out(ep, keep_out) if keep_out else math.nan,
    }
