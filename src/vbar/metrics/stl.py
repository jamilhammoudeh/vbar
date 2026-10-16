"""Signals over an episode in continuous time: minima and the intervals where they are negative.

A signal is a margin-like function of the state and time with a bound on its rate, like a
constraint: >= 0 where its predicate holds. The true trajectory between the samples of each
constant-acceleration segment is rebuilt from the step records alone: the state at time t is the
sample at or before t, propagated with the exact Clohessy-Wiltshire solution under the segment's
acceleration. That is exact on the CW model. On the two-body models it drops the difference
between their natural acceleration and CW's over at most one sample spacing (0.5 s at the default
20 samples per 10 s step): measured below 2e-7 m at 2 km without J2, and about 1e-6 m with J2,
whose Hill frame turns faster than n by about 1.5 J2 (R/a)^2 n; far below the 1 mm resolution of
the range constraints.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ..constraints import (
    Constraint,
    SegmentMinimum,
    interval_bounds,
    range_lower_bounds,
    segment_minimum,
)
from ..dynamics import cw_discrete
from .episode import Episode

# Bracket tolerance (s) for the time of a minimum or a sign change.
_TIME_TOL = 1e-9
# Intervals closer than this (s) are one interval.
_JOIN_TOL = 1e-9


@dataclass(frozen=True)
class Signal:
    """A margin-like function of the state with a rate bound, usable wherever a constraint is."""

    name: str
    margin_fn: Callable[[np.ndarray, np.ndarray | float], np.ndarray]
    rate_fn: Callable[[float, float, float], float]
    resolution: float

    def margin(self, state: np.ndarray, t: np.ndarray | float = 0.0) -> np.ndarray:
        return self.margin_fn(state, t)

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        return self.rate_fn(speed, accel, rho_min)


@dataclass(frozen=True)
class Segment:
    """One constant-acceleration piece of the trajectory and its evenly spaced samples."""

    t0: float
    duration: float
    accel: np.ndarray
    times: np.ndarray
    states: np.ndarray


class Trace:
    """The true trajectory of an episode in continuous time."""

    def __init__(self, episode: Episode):
        self.n = episode.view.n
        self.segments: list[Segment] = []
        for rec in episode.records:
            times, states = rec.segment.times, rec.segment.states
            # Each segment's samples run from its start to its end; the next segment repeats
            # the shared boundary time.
            breaks = [0, *(np.flatnonzero(np.diff(times) <= 0.0) + 1), len(times)]
            if len(breaks) - 1 != len(rec.segment.segments):
                raise ValueError("segment samples do not match the segments")
            for (t0, duration, accel), i, j in zip(
                rec.segment.segments, breaks[:-1], breaks[1:], strict=True
            ):
                self.segments.append(
                    Segment(t0, duration, np.asarray(accel, float), times[i:j], states[i:j])
                )
        self._starts = [s.t0 for s in self.segments]

    @property
    def end(self) -> float:
        last = self.segments[-1] if self.segments else None
        return last.t0 + last.duration if last else 0.0

    def segment_at(self, t: float) -> Segment:
        k = max(bisect_right(self._starts, t) - 1, 0)
        return self.segments[k]

    def state_in(self, segment: Segment, t: float) -> np.ndarray:
        """State at time `t` (s from the episode start) inside `segment`."""
        k = int(np.clip(np.searchsorted(segment.times, t, side="right") - 1, 0, None))
        tau = t - segment.times[k]
        if tau == 0.0:
            return segment.states[k].copy()
        phi, gamma = cw_discrete(self.n, tau)
        return phi @ segment.states[k] + gamma @ segment.accel

    def state_at(self, t: float) -> np.ndarray:
        return self.state_in(self.segment_at(t), t)

    def samples(self) -> tuple[np.ndarray, np.ndarray]:
        """All sample times (s) and states, segment by segment."""
        if not self.segments:
            return np.zeros(0), np.zeros((0, 6))
        return (
            np.concatenate([s.times for s in self.segments]),
            np.concatenate([s.states for s in self.segments]),
        )


def _rates(signal: Constraint, segment: Segment, n: float) -> tuple[float, np.ndarray]:
    """Sample spacing and the rate bound on each sample interval."""
    h = float(segment.times[1] - segment.times[0])
    thrust = float(np.linalg.norm(segment.accel))
    speed, accel = interval_bounds(segment.states, h, thrust, n)
    rho_min = range_lower_bounds(segment.states, h, speed)
    rates = [
        signal.rate_bound(v, a, rho_min=r) for v, a, r in zip(speed, accel, rho_min, strict=True)
    ]
    return h, np.array(rates)


def minimum(signal: Constraint, trace: Trace) -> SegmentMinimum:
    """Lowest value of `signal` over the whole trace, refined and certified per segment."""
    best: SegmentMinimum | None = None
    for seg in trace.segments:

        def state_at(tau: float, seg: Segment = seg) -> np.ndarray:
            return trace.state_in(seg, seg.t0 + tau)

        found = segment_minimum(
            signal,
            state_at,
            seg.times - seg.t0,
            seg.states,
            float(np.linalg.norm(seg.accel)),
            trace.n,
            seg.t0,
        )
        if best is None:
            best = found
        else:
            lowest = found if found.margin < best.margin else best
            best = SegmentMinimum(lowest.t, lowest.margin, min(best.lower_bound, found.lower_bound))
    if best is None:
        raise ValueError("the trace is empty")
    return best


def _root(f: Callable[[float], float], a: float, b: float) -> float:
    return float(brentq(f, a, b, xtol=_TIME_TOL))


def _extremum(f: Callable[[float], float], a: float, b: float) -> tuple[float, float]:
    found = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": _TIME_TOL})
    return float(found.x), float(found.fun)


def negative_intervals(signal: Constraint, trace: Trace) -> list[tuple[float, float]]:
    """Time intervals (s) where `signal` < 0, merged across samples, segments and steps.

    Sign changes between samples are found by root bracketing. An interval whose end samples
    share a sign is searched for a dip below zero (or a rise above it) only when the Lipschitz
    bound allows one; like the in-step minimum, this assumes at most one dip per sample interval.
    """
    found: list[tuple[float, float]] = []
    for seg in trace.segments:
        if len(seg.times) < 2:
            continue

        def f(t: float, seg: Segment = seg) -> float:
            return float(signal.margin(trace.state_in(seg, t), t))

        t = seg.times
        m = np.asarray(signal.margin(seg.states, t), dtype=float)
        h, rates = _rates(signal, seg, trace.n)
        for i in range(len(t) - 1):
            a, b = m[i], m[i + 1]
            if a < 0.0 and b < 0.0:
                if 0.5 * (a + b + rates[i] * h) > 0.0:
                    t_peak, minus_peak = _extremum(lambda s: -f(s), t[i], t[i + 1])
                    if minus_peak <= 0.0:
                        first = (t[i], _root(f, t[i], t_peak))
                        found += [first, (_root(f, t_peak, t[i + 1]), t[i + 1])]
                        continue
                found.append((t[i], t[i + 1]))
            elif a < 0.0:
                found.append((t[i], _root(f, t[i], t[i + 1])))
            elif b < 0.0:
                found.append((_root(f, t[i], t[i + 1]), t[i + 1]))
            elif 0.5 * (a + b - rates[i] * h) < 0.0:
                t_min, low = _extremum(f, t[i], t[i + 1])
                if low < 0.0:
                    found.append((_root(f, t[i], t_min), _root(f, t_min, t[i + 1])))
    return merge(found)


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of intervals, joining those that touch."""
    merged: list[tuple[float, float]] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + _JOIN_TOL:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def complement(intervals: list[tuple[float, float]], start: float, end: float):
    """Intervals of [start, end] not covered by `intervals` (which must be merged)."""
    gaps, t = [], start
    for lo, hi in intervals:
        if lo > t + _JOIN_TOL:
            gaps.append((t, min(lo, end)))
        t = max(t, hi)
    if end > t + _JOIN_TOL:
        gaps.append((t, end))
    return gaps


def total(intervals: list[tuple[float, float]]) -> float:
    return float(sum(hi - lo for lo, hi in intervals))
