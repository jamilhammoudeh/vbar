"""Safety-filter metrics F1 to F10 and the fallback and certificate columns.

Per episode they read each step's `FilterResult`; an episode without a filter gives NaN. The
paired metrics F3 and F4 compare the same seeds flown with the filter on and off.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import trim_mean

from ..agents.base import action_scale
from .efficiency import delta_v
from .episode import Episode
from .mission import success
from .passive import _arcs, _collision_radius, _drift_horizon, _settle
from .settings import MetricSettings

# Changes below this fraction of a full command are rounding, not intervention.
_MOVED = 1e-6


def _decisions(ep: Episode, settings: MetricSettings):
    """Records with a filter result, and whether each decision was made outside the commit zone."""
    states = ep.states[:-1]
    outside = np.linalg.norm(states[:, :3], axis=1) > settings.commit_radius
    return [
        (rec, bool(out))
        for rec, out in zip(ep.records, outside, strict=True)
        if rec.filter_result is not None
    ]


def _flown_change(ep: Episode, rec) -> float:
    """|flown(filtered) - flown(command)|_2 (m/s^2): the change the actuator would carry out."""
    return float(
        np.linalg.norm(_commanded_accel(ep, rec.filtered) - _commanded_accel(ep, rec.command))
    )


def _active(ep: Episode, rec) -> bool:
    """Whether a safety row acted: the filter says so, or the flown acceleration changed.

    Judged from the record, so a filter cannot hide a change by calling it clipping; a reported
    row only adds a decision whose action did not move.
    """
    return bool(rec.filter_result.active) or _flown_change(ep, rec) > _MOVED * ep.view.max_accel


def _clip_only(ep: Episode, rec) -> bool:
    """The action changed but the actuator flies the same acceleration: clipping only."""
    moved = float(np.max(np.abs(rec.filtered - rec.command))) > _MOVED * action_scale(ep.view)
    return moved and _flown_change(ep, rec) <= _MOVED * ep.view.max_accel


def filter_metrics(ep: Episode, settings: MetricSettings | None = None) -> dict[str, float]:
    """F1, F1c, F2, F2n, F5, F5b and F6 for one episode.

    F1 counts decisions where a safety row was active: the filter reports one, or the acceleration
    the actuator flies changed. Changes the actuator would make anyway, clipping to the thrust
    limit, are F1c. F2 is the mean change in flown acceleration over the active decisions, in
    m/s^2 at the initial mass, and F2n the same over the largest change the actuator allows, the
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
    widest = 2.0 * ep.view.max_accel * (math.sqrt(3.0) if ep.view.thrust_limit == "box" else 1.0)
    acting = [rec for rec, _ in decisions if _active(ep, rec)]
    sizes = [_flown_change(ep, rec) for rec in acting]
    clipped = sum(_clip_only(ep, rec) and not rec.filter_result.active for rec, _ in decisions)
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


def _commanded_accel(ep: Episode, command: np.ndarray) -> np.ndarray:
    """Hill acceleration (m/s^2) the spacecraft would fly for a command, at the initial mass.

    The thrust limit is taken at the initial mass; with constant thrust it grows in m/s^2 as
    propellant burns, so a change just above the initial limit late in a run counts as clipping.
    """
    u = np.asarray(command, dtype=float) / action_scale(ep.view)
    if ep.view.thrust_limit == "box":
        u = np.clip(u, -1.0, 1.0)
    else:
        u = u / max(1.0, float(np.linalg.norm(u)))
    return u * ep.view.max_accel


def _run_lengths(flags: list[bool], durations: list[float]) -> list[float]:
    """Durations (s) of the runs of consecutive True flags, each flag one decision of its duration.

    A run still open at the last flag never ended, so its duration is infinite.
    """
    runs, length, running = [], 0.0, False
    for flag, duration in zip(flags, durations, strict=True):
        if flag:
            length, running = length + duration, True
        elif running:
            runs.append(length)
            length, running = 0.0, False
    if running:
        runs.append(math.inf)
    return runs


def filter_behaviour(ep: Episode, settings: MetricSettings | None = None) -> dict[str, Any]:
    """F8, F9, F10, the fallback count and the certificate columns for one episode.

    F8 is the share of decisions with an active safety row whose own command, held for the step
    and then left to free drift over the drift horizon, would have stayed clear of the collision
    radius: interventions the filter did not need (the counterfactual of a filter-off run, from
    the state each decision was made in). Only the collision sphere is judged, so an intervention
    for another constraint counts as unnecessary. F9 is the mean time to get back inside the
    filter's safe set over the stretches where the filter reported `recovering`, infinite if the
    episode ends inside one; F10 the time with an active row
    while the chaser moved slower than `stall_speed`, a barrier holding it in place. The fallback
    count is decisions answered by a fallback (`fallback_level` > 0). NaN where the filter did not
    report the field.
    """
    settings = settings or MetricSettings()
    decisions = [(k, rec) for k, rec in enumerate(ep.records) if rec.filter_result is not None]
    out: dict[str, Any] = {
        "unnecessary_intervention_rate": math.nan,
        "reentry_time": math.nan,
        "stall_time": math.nan,
        "fallback_count": math.nan,
        "fallback_level_max": math.nan,
        "filter_certificates": (),
        "certificate_horizon_min": math.nan,
    }
    if not decisions:
        return out
    dt = ep.view.dt
    # A step cut short by a contact is shorter than dt, so the times come from the record.
    durations = [float(ep.times[k + 1] - ep.times[k]) for k, _ in decisions]
    states = ep.states
    acting = [(k, rec) for k, rec in decisions if _active(ep, rec)]
    if acting:
        starts = np.array([states[k] for k, _ in acting])
        accels = np.array([_commanded_accel(ep, rec.command) for _, rec in acting])
        radius, horizon = _collision_radius(ep, settings), _drift_horizon(ep, settings)
        arcs = _arcs(ep, settings, radius, starts, accels, dt, np.zeros(3), horizon)
        clear, _ = _settle(arcs, len(acting), 1)
        out["unnecessary_intervention_rate"] = float(np.mean(clear))
    results = [rec.filter_result for _, rec in decisions]
    if any(r.recovering is not None for r in results):
        runs = _run_lengths([bool(r.recovering) for r in results], durations)
        out["reentry_time"] = float(np.mean(runs)) if runs else 0.0
    speeds = np.linalg.norm(states[[k for k, _ in decisions], 3:], axis=1)
    slow = [
        _active(ep, rec) and s < settings.stall_speed
        for (_, rec), s in zip(decisions, speeds, strict=True)
    ]
    out["stall_time"] = float(sum(d for d, stalled in zip(durations, slow, strict=True) if stalled))
    levels = [r.fallback_level for r in results if r.fallback_level is not None]
    if levels:
        out["fallback_count"] = float(sum(level > 0 for level in levels))
        out["fallback_level_max"] = float(max(levels))
    kinds = sorted({r.certificate for r in results if r.certificate is not None})
    out["filter_certificates"] = tuple(kinds)
    horizons = [r.certificate_horizon for r in results if r.certificate_horizon is not None]
    if horizons:
        out["certificate_horizon_min"] = float(min(horizons))
    return out


def filter_conservatism(
    in_filter_set: Callable[[np.ndarray], bool],
    in_viable_set: Callable[[np.ndarray], bool],
    states: ArrayLike,
) -> float:
    """F7: share of the reference viable states that the filter's safe set leaves out.

    `states` sample a declared region (Hill, m and m/s); `in_viable_set` is a reference such as the
    exact braking set, and `in_filter_set` the set the filter keeps the chaser in. 0 means the
    filter keeps every viable state available; NaN if no sample is viable.
    """
    viable = [x for x in np.asarray(states, dtype=float) if in_viable_set(x)]
    if not viable:
        return math.nan
    return 1.0 - float(np.mean([bool(in_filter_set(x)) for x in viable]))


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
