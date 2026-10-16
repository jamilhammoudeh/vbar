"""Training rewards that agree with the benchmark's metrics.

A reward is a training signal, never the score, and scored runs never pass it to an agent. Each
term is a rate or an event in SI units, integrated exactly over the step's constant-acceleration
segments, and the return is discounted per second: a step of duration dt is discounted by
gamma = exp(-dt / tau), or not at all when no time constant is given. The same physical
trajectory then earns the same return at any decision interval, the semi-Markov return of
Sutton and Barto, Reinforcement Learning (2nd ed. draft, 2015), sec. 14.4.

Progress shaping is potential-based: gamma Phi(s') - Phi(s), with Phi = 0 at a true terminal, so
it shifts every policy's return by -Phi(s0) and leaves the optimum unchanged (Ng, Harada and
Russell, 1999). The success bonus needs the success predicate with no hard constraint violated
(metric M1), and constraint costs come from the segment trace's violation intervals, the
quantities behind S3 and C1. All fields may carry leading batch axes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import expm

from .dynamics import cw_system


@dataclass(frozen=True)
class Transition:
    """One decision step as rewards see it, from the truth (SI units, Hill frame).

    `segments` rows are (start offset s from the step start, duration s, ax, ay, az m/s^2) and
    `starts` the true state at each segment's start; `fuel_rates` is each segment's delta-v rate
    (m/s^2) in the actuator's norm (1-norm for box thrusters, 2-norm for ball). `violations` maps
    each constraint to its intervals of negative margin as (start, end) offsets (s), padded with
    empty intervals, and `entries` to the start offsets of the intervals that began in the step,
    padded with inf. `goal_distance` and `next_goal_distance` (m) are taken at the step's start
    and end. `success` is the success predicate at the end with no hard constraint violated in
    the episode; `terminal` is whether the episode ended there for a reason other than the time
    limit.
    """

    duration: Any
    segments: np.ndarray
    starts: np.ndarray
    fuel_rates: np.ndarray
    violations: Mapping[str, np.ndarray] = field(default_factory=dict)
    entries: Mapping[str, np.ndarray] = field(default_factory=dict)
    goal_distance: Any = 0.0
    next_goal_distance: Any = 0.0
    success: Any = False
    terminal: Any = False


class QuadraticCost:
    """Cost rate (x - x_ref)^T Q (x - x_ref) + a^T R a, integrated exactly under CW motion.

    `n` is the mean motion (rad/s), `q` (6x6) and `r` (3x3) the weights and `reference` the state
    deviations are measured from. Over a segment held at acceleration a the discounted integral is
    z^T W(d) z with z = (x_start, a, 1), W from Van Loan's matrix exponential (Van Loan, 1978);
    it is exact for the Clohessy-Wiltshire model and an approximation for any other.
    """

    def __init__(
        self,
        n: float,
        q: np.ndarray,
        r: np.ndarray,
        reference: np.ndarray | None = None,
    ):
        self.n = n
        self.q = np.array(q, dtype=float)
        self.r = np.array(r, dtype=float)
        self.reference = np.zeros(6) if reference is None else np.array(reference, dtype=float)
        if self.q.shape != (6, 6) or self.r.shape != (3, 3) or self.reference.shape != (6,):
            raise ValueError("q must be 6x6, r 3x3 and reference length 6")
        self._matrices: dict[tuple[float, float | None], np.ndarray] = {}

    def matrix(self, duration: float, tau: float | None) -> np.ndarray:
        """W (10x10) with int_0^d exp(-s / tau) cost(s) ds = z^T W z, z = (x_start, a, 1)."""
        key = (round(float(duration), 9), tau)
        if key not in self._matrices:
            a, b = cw_system(self.n)
            f = np.zeros((10, 10))
            f[:6, :6], f[:6, 6:9] = a, b
            if tau is not None:
                # exp(-s / tau) z^T M z = (exp(-s / 2 tau) z)^T M (exp(-s / 2 tau) z).
                f -= np.eye(10) / (2.0 * tau)
            qx = self.q @ self.reference
            w = np.zeros((10, 10))
            w[:6, :6], w[6:9, 6:9] = self.q, self.r
            w[:6, 9] = w[9, :6] = -qx
            w[9, 9] = self.reference @ qx
            block = np.zeros((20, 20))
            block[:10, :10], block[:10, 10:], block[10:, 10:] = -f.T, w, f
            g = expm(block * key[0])
            integral = g[10:, 10:].T @ g[:10, 10:]
            self._matrices[key] = 0.5 * (integral + integral.T)
        return self._matrices[key]

    def __call__(
        self, starts: np.ndarray, accels: np.ndarray, durations: np.ndarray, tau: float | None
    ) -> np.ndarray:
        """Integrated cost of each segment, without the discount of its start offset."""
        z = np.concatenate([starts, accels, np.ones((*starts.shape[:-1], 1))], axis=-1)
        cost = np.zeros(starts.shape[:-1])
        for d in np.unique(durations):
            sel = durations == d
            if d > 0.0:
                w = self.matrix(float(d), tau)
                cost[sel] = np.einsum("...i,ij,...j->...", z[sel], w, z[sel])
        return cost


@dataclass(frozen=True, kw_only=True)
class Reward:
    """A training reward made of metric-aligned terms; every weight defaults to off.

    `tau` (s) is the discount time constant, or None for no discount. `success_bonus` is paid at
    a terminal step whose success holds (M1). `fuel_weight` is per m/s of delta-v in the
    actuator's norm (E1 or E1b), `time_weight` per second (M4), `violation_weights` per second
    with each named constraint's margin negative (S3) and `entry_weights` per violation interval
    begun (C1). `progress_weight` (per m) sets the potential Phi = -progress_weight * distance to
    the goal. `quadratic` adds an integrated quadratic cost.
    """

    tau: float | None = None
    success_bonus: float = 0.0
    fuel_weight: float = 0.0
    time_weight: float = 0.0
    violation_weights: Mapping[str, float] = field(default_factory=dict)
    entry_weights: Mapping[str, float] = field(default_factory=dict)
    progress_weight: float = 0.0
    quadratic: QuadraticCost | None = None

    def __post_init__(self):
        if self.tau is not None and not self.tau > 0.0:
            raise ValueError("tau must be positive, or None for no discount")
        weights = [self.success_bonus, self.fuel_weight, self.time_weight, self.progress_weight]
        weights += [*self.violation_weights.values(), *self.entry_weights.values()]
        if not all(np.isfinite(w) and w >= 0.0 for w in weights):
            raise ValueError("reward weights must be finite and not negative")

    def discount(self, duration: Any) -> np.ndarray:
        """Discount of a step of `duration` s: exp(-duration / tau), or 1."""
        d = np.asarray(duration, dtype=float)
        return np.ones_like(d) if self.tau is None else np.exp(-d / self.tau)

    def _integral(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """int_start^end exp(-s / tau) ds, or end - start without discounting."""
        if self.tau is None:
            return end - start
        # expm1 keeps short intervals exact; the plain difference of exponentials cancels.
        return -self.tau * np.exp(-start / self.tau) * np.expm1(-(end - start) / self.tau)

    def __call__(self, tr: Transition) -> np.ndarray:
        gamma = self.discount(tr.duration)
        duration = np.asarray(tr.duration, dtype=float)
        offsets, durations = tr.segments[..., 0], tr.segments[..., 1]
        reward = -self.time_weight * self._integral(np.zeros_like(duration), duration)
        if self.fuel_weight:
            burn = self._integral(offsets, offsets + durations)
            reward = reward - self.fuel_weight * np.sum(tr.fuel_rates * burn, axis=-1)
        for name, weight in self.violation_weights.items():
            spans = tr.violations[name]
            inside = self._integral(spans[..., 0], spans[..., 1])
            reward = reward - weight * np.sum(inside, axis=-1)
        for name, weight in self.entry_weights.items():
            times = tr.entries[name]
            real = np.isfinite(times)
            begun = np.where(real, self.discount(np.where(real, times, 0.0)), 0.0)
            reward = reward - weight * np.sum(begun, axis=-1)
        if self.quadratic is not None:
            segment_cost = self.quadratic(tr.starts, tr.segments[..., 2:5], durations, self.tau)
            reward = reward - np.sum(self.discount(offsets) * segment_cost, axis=-1)
        terminal = np.asarray(tr.terminal, dtype=bool)
        if self.success_bonus:
            paid = terminal & np.asarray(tr.success, dtype=bool)
            reward = reward + self.success_bonus * gamma * paid
        if self.progress_weight:
            next_distance = np.where(terminal, 0.0, tr.next_goal_distance)
            shaping = np.asarray(tr.goal_distance) - gamma * next_distance
            reward = reward + self.progress_weight * shaping
        return reward
