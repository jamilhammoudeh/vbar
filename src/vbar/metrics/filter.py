"""Safety-filter metrics F1 to F10.

Per episode they read each step's `FilterResult`; an episode without a filter gives NaN. The
paired metrics F3 and F4 compare the same seeds flown with the filter on and off.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import trim_mean

from ..agents.base import action_scale
from .efficiency import delta_v
from .episode import Episode
from .mission import success
from .settings import MetricSettings


def _decisions(ep: Episode, settings: MetricSettings):
    """Records with a filter result, and whether each decision was made outside the commit zone."""
    states = ep.states[:-1]
    outside = np.linalg.norm(states[:, :3], axis=1) > settings.commit_radius
    return [
        (rec, bool(out))
        for rec, out in zip(ep.records, outside, strict=True)
        if rec.filter_result is not None
    ]


def _active(result) -> bool:
    """Whether a safety row acted: as reported, or a change that was not only clipping."""
    if result.active is not None:
        return bool(result.active)
    return bool(result.intervened and not result.clipped)


def filter_metrics(ep: Episode, settings: MetricSettings | None = None) -> dict[str, float]:
    """F1, F1c, F2, F2n, F5, F5b and F6 for one episode.

    F1 counts decisions where a safety row was active, apart from changes that were only actuator
    clipping (F1c). F2 is the mean |filtered - command|_2 over those decisions in m/s^2 at the
    initial mass, and F2n the same over the largest command change the actuator allows, the
    distance from a full command to its opposite (2 sqrt(3) a_max for box limits, 2 a_max for a
    ball), so it lies in [0, 1]. F5 and F5b count decisions outside `commit_radius`, where the
    filter is expected to act.
    """
    settings = settings or MetricSettings()
    decisions = _decisions(ep, settings)
    out = {
        "intervention_rate": math.nan,
        "clip_rate": math.nan,
        "intervention_size": math.nan,
        "intervention_size_normalized": math.nan,
        "infeasible_steps": math.nan,
        "solver_failures": math.nan,
        "filter_compute_median": math.nan,
        "filter_compute_p95": math.nan,
    }
    if not decisions:
        return out
    # Action units to m/s^2: a full command asks for max_accel at the initial mass.
    to_accel = ep.view.max_accel / action_scale(ep.view)
    widest = 2.0 * ep.view.max_accel * (math.sqrt(3.0) if ep.view.thrust_limit == "box" else 1.0)
    acting = [rec for rec, _ in decisions if _active(rec.filter_result)]
    sizes = [np.linalg.norm(rec.filtered - rec.command) * to_accel for rec in acting]
    clipped = sum(bool(rec.filter_result.clipped) for rec, _ in decisions)
    compute = [rec.filter_result.compute_s for rec, _ in decisions]
    statuses = [rec.filter_result.status for rec, outside in decisions if outside]
    size = float(np.mean(sizes)) if sizes else math.nan
    out.update(
        intervention_rate=len(acting) / len(decisions),
        clip_rate=clipped / len(decisions),
        intervention_size=size,
        intervention_size_normalized=size / widest,
        infeasible_steps=float(statuses.count("infeasible_certified")),
        solver_failures=float(statuses.count("solver_failed")),
        filter_compute_median=float(np.median(compute)),
        filter_compute_p95=float(np.percentile(compute, 95)),
    )
    return out


def iqm(values: ArrayLike) -> float:
    """Interquartile mean: the mean of the middle half (Agarwal et al., 2021)."""
    return float(trim_mean(np.asarray(values, dtype=float), 0.25))


def paired_filter_metrics(on: Sequence[Episode], off: Sequence[Episode]) -> dict:
    """F3 and F4 from the same seeds flown with the filter on and off.

    F3 is the per-seed E1 difference (filter on minus off) and its interquartile mean; F4 the
    success rate with the filter minus without.
    """
    by_seed = {ep.seed: ep for ep in off}
    if len(by_seed) != len(off) or {ep.seed for ep in on} != set(by_seed) or len(on) != len(off):
        raise ValueError("filter on and off runs must cover the same distinct seeds")
    pairs = [(ep, by_seed[ep.seed]) for ep in on]
    costs = {a.seed: delta_v(a) - delta_v(b) for a, b in pairs}
    return {
        "safety_dv_cost": iqm(list(costs.values())),
        "safety_dv_cost_by_seed": costs,
        "filter_dependence": float(np.mean([success(a) for a, _ in pairs]))
        - float(np.mean([success(b) for _, b in pairs])),
    }
