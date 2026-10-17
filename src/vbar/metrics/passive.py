"""Passive safety and abort availability: S4, S4b, S4i, S4t, S4o, S8 and S9.

Fehse (Automated Rendezvous and Docking of Spacecraft, 2003, sections 4.4 and 4.5) asks of every
point of an approach that a loss of thrust leaves a collision-free drift, or else that a
collision avoidance manoeuvre (CAM) is available. Each check here is an escape arc: from a state,
hold an acceleration for a time (a stuck thruster, or none), add an impulse (the CAM, or none),
then drift; its margin is the lowest range minus the collision radius over the whole arc.

On the Clohessy-Wiltshire model all arcs of a check are sampled at once with the closed-form
transition matrices, bounded on every sample interval with the Lipschitz certificate of
`vbar.constraints`, and refined only where the answer depends on it: arcs whose sign is not
settled by their samples and bound, and arcs whose bound lies below the lowest value found. On
other models each arc is integrated by a fresh copy of the episode's dynamics model, whose target
starts from its initial orbit position.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.optimize import minimize_scalar

from ..constraints import KeepOut, interval_bounds, segment_minimum
from ..dynamics import cw_discrete, make_dynamics
from .episode import Episode
from .settings import MetricSettings
from .stl import Trace, cw_step

# Range resolution (m) of the arc minima, as for the keep-out constraint.
_RESOLUTION = 1e-3
# Arcs per batch on the closed-form path, to bound memory.
_BATCH = 256


def _grid(duration: float, spacing: float) -> np.ndarray:
    """Evenly spaced times over [0, duration], no further apart than `spacing`."""
    if duration <= 0.0:
        return np.zeros(1)
    return np.linspace(0.0, duration, max(2, math.ceil(duration / spacing) + 1))


def escape_arc(
    n: float,
    start: np.ndarray,
    accel: np.ndarray,
    burn_time: float,
    impulse: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """States of one escape arc at `times` (s) on Clohessy-Wiltshire, shape (len(times), 6).

    Holds `accel` (m/s^2, Hill) for `burn_time` s from `start`, adds the Hill `impulse` (m/s),
    then drifts.
    """
    start, accel = np.asarray(start, float), np.asarray(accel, float)
    times = np.asarray(times, dtype=float)
    phi, gamma = cw_discrete(n, np.minimum(times, burn_time))
    burning = phi @ start + gamma @ accel
    phi_b, gamma_b = cw_discrete(n, burn_time)
    after = phi_b @ start + gamma_b @ accel + np.r_[0.0, 0.0, 0.0, impulse]
    drift, _ = cw_discrete(n, np.maximum(times - burn_time, 0.0))
    return np.where((times <= burn_time)[:, None], burning, drift @ after)


@dataclass
class _Arcs:
    """Sampled minima, certified lower bounds and on-demand refinement for a set of arcs."""

    sampled: np.ndarray
    lower: np.ndarray
    refined: np.ndarray
    pieces: list

    def refine(self, k: int) -> float:
        if np.isnan(self.refined[k]):
            self.refined[k] = min(piece(k) for piece in self.pieces)
        return float(self.refined[k])


def _cw_arcs(n, radius, starts, accels, burn_time, impulses, horizon, spacing) -> _Arcs:
    """Arcs on Clohessy-Wiltshire, sampled in batches with the closed form."""
    keep_out = KeepOut(radius)
    t1, t2 = _grid(burn_time, spacing), _grid(horizon, spacing)
    phi1, gamma1 = cw_discrete(n, t1)
    phi2, _ = cw_discrete(n, t2)
    count = len(starts)
    sampled, lower = np.empty(count), np.empty(count)
    middles = np.empty((count, 6))
    for lo in range(0, count, _BATCH):
        x0, a = starts[lo : lo + _BATCH], accels[lo : lo + _BATCH]
        p1 = np.einsum("mij,kj->kmi", phi1, x0) + np.einsum("mij,kj->kmi", gamma1, a)
        x1 = p1[:, -1].copy()
        x1[:, 3:] += impulses[lo : lo + _BATCH]
        p2 = np.einsum("mij,kj->kmi", phi2, x1)
        middles[lo : lo + _BATCH] = x1
        low_s, low_b = np.full(len(x0), np.inf), np.full(len(x0), np.inf)
        for times, samples, thrust in ((t1, p1, np.linalg.norm(a, axis=1)), (t2, p2, 0.0)):
            if len(times) < 2:
                continue
            h = times[1] - times[0]
            margin = keep_out.margin(samples)
            speed, _ = interval_bounds(samples, h, np.reshape(thrust, (-1, 1)), n)
            bound = 0.5 * (margin[:, :-1] + margin[:, 1:] - speed * h)
            low_s = np.minimum(low_s, margin.min(axis=1))
            low_b = np.minimum(low_b, np.minimum(bound.min(axis=1), margin.min(axis=1)))
        sampled[lo : lo + _BATCH], lower[lo : lo + _BATCH] = low_s, low_b

    def piece(times, origin, accel_of):
        def minimum_of(k: int) -> float:
            if len(times) < 2:
                return math.inf
            x, a = origin(k), accel_of(k)
            states = escape_arc(n, x, a, times[-1], np.zeros(3), times)
            found = segment_minimum(
                keep_out,
                lambda tau: cw_step(n, tau, x, a),
                times,
                states,
                float(np.linalg.norm(a)),
                n,
            )
            return found.margin

        return minimum_of

    pieces = [
        piece(t1, lambda k: starts[k], lambda k: accels[k]),
        piece(t2, lambda k: middles[k], lambda k: np.zeros(3)),
    ]
    return _Arcs(sampled, lower, np.full(count, np.nan), pieces)


def _model_arcs(ep, radius, starts, accels, burn_time, impulses, horizon, spacing) -> _Arcs:
    """Arcs on the episode's own dynamics model, each integrated and refined in full."""
    keep_out = KeepOut(radius)
    model = make_dynamics(ep.dynamics, ep.body, ep.altitude_m, ep.inclination)
    t1, t2 = _grid(burn_time, spacing), _grid(horizon, spacing)
    count = len(starts)
    sampled, lower, refined = np.empty(count), np.empty(count), np.empty(count)
    for k in range(count):
        x, low_s, low_b, low = starts[k], math.inf, math.inf, math.inf
        for times, accel, kick in ((t1, accels[k], impulses[k]), (t2, np.zeros(3), None)):
            model.reset(x)
            if len(times) < 2:
                end = x
            else:
                end, sampled_states = model.advance(accel, times[-1], n_samples=len(times))
                samples = cast(np.ndarray, sampled_states)
                found = segment_minimum(
                    keep_out,
                    model.state_in_last_segment,
                    times,
                    samples,
                    float(np.linalg.norm(accel)),
                    model.n,
                )
                low_s = min(low_s, float(keep_out.margin(samples).min()))
                low_b, low = min(low_b, found.lower_bound), min(low, found.margin)
            x = end + (np.r_[0.0, 0.0, 0.0, kick] if kick is not None else 0.0)
        sampled[k], lower[k], refined[k] = low_s, low_b, low
    return _Arcs(sampled, lower, refined, [])


def _arcs(ep, settings, radius, starts, accels, burn_time, impulses, horizon) -> _Arcs:
    starts = np.asarray(starts, float).reshape(-1, 6)
    count = len(starts)
    accels = np.broadcast_to(np.asarray(accels, float), (count, 3))
    impulses = np.broadcast_to(np.asarray(impulses, float), (count, 3))
    spacing = 2.0 * np.pi / ep.view.n / settings.drift_samples_per_orbit
    if ep.dynamics == "cw":
        return _cw_arcs(ep.view.n, radius, starts, accels, burn_time, impulses, horizon, spacing)
    return _model_arcs(ep, radius, starts, accels, burn_time, impulses, horizon, spacing)


def _settle(arcs: _Arcs, rows: int, cols: int) -> tuple[np.ndarray, float]:
    """Per row, whether its best arc stays clear; and the least over rows of the best arc.

    Rows are checks (a state) and columns alternatives (CAM directions); a row is protected when
    any of its arcs keeps a non-negative margin. Arcs are refined only where samples and bounds
    leave the answer open.
    """
    sampled = arcs.sampled.reshape(rows, cols)
    lower = arcs.lower.reshape(rows, cols)
    value = sampled.copy()

    def refine(k: int, j: int) -> None:
        value[k, j] = arcs.refine(k * cols + j)

    safe = np.zeros(rows, dtype=bool)
    for k in range(rows):
        if lower[k].max() >= 0.0:
            safe[k] = True
            continue
        for j in np.argsort(-sampled[k]):
            if sampled[k, j] < 0.0:
                break
            refine(k, j)
            if value[k, j] >= 0.0:
                safe[k] = True
                break
    best = float(value.max(axis=1).min()) if rows else math.inf
    row_lower = lower.max(axis=1)
    for k in np.argsort(row_lower):
        if row_lower[k] >= best - _RESOLUTION:
            break
        for j in range(cols):
            refine(k, j)
        best = min(best, float(value[k].max()))
    return safe, best


@dataclass(frozen=True)
class TravellingEllipse:
    """Free-drift Clohessy-Wiltshire motion as a drifting ellipse (Hill axes, m, m/s).

    x(t) = x_c + (vx0 / n) sin nt - (3 x0 + 2 vy0 / n) cos nt, an oscillation of amplitude b
    about x_c = 4 x0 + 2 vy0 / n; y(t) = y_c(t) + 2 (vx0 / n) cos nt + 2 (3 x0 + 2 vy0 / n) sin nt
    about a centre y_c(t) = y0 - 2 vx0 / n - 1.5 n x_c t, so the along-track semi-axis is 2 b; the
    orbit-normal motion has amplitude c at the same frequency. This is Fehse's travelling ellipse
    (appendix A.1.5) in Hill axes, with his eq. A.69 corrected to x_F = x_c - 2 b cos(nt + phi).
    """

    n: float
    x0: float
    vx0: float
    radial_offset: float
    center_x: float
    center_y: float
    z0: float
    vz0: float

    @classmethod
    def from_state(cls, state: Sequence[float], n: float) -> TravellingEllipse:
        x, y, z, vx, vy, vz = (float(v) for v in state)
        return cls(
            n, x, vx, 3.0 * x + 2.0 * vy / n, 4.0 * x + 2.0 * vy / n, y - 2.0 * vx / n, z, vz
        )

    @property
    def semi_axis(self) -> float:
        """Radial semi-axis b (m); the along-track one is 2 b."""
        return math.hypot(self.vx0 / self.n, self.radial_offset)

    @property
    def drift_rate(self) -> float:
        """Along-track speed of the centre, -1.5 n x_c (m/s)."""
        return -1.5 * self.n * self.center_x

    def position(self, t: np.ndarray | float) -> np.ndarray:
        """Hill position (m) at `t` s, shape (..., 3)."""
        nt = self.n * np.asarray(t, dtype=float)
        s, c = np.sin(nt), np.cos(nt)
        a, b = self.vx0 / self.n, self.radial_offset
        x = self.center_x + a * s - b * c
        y = self.center_y + self.drift_rate * np.asarray(t, float) + 2.0 * (a * c + b * s)
        z = self.z0 * c + self.vz0 / self.n * s
        return np.stack([x, y, z], axis=-1)

    def cross_section_clearance(self) -> float:
        """Least sqrt(x^2 + z^2) over one cycle (m): the distance from the target to the tube."""
        a, b = self.vx0 / self.n, self.radial_offset

        def distance(theta: float) -> float:
            x = self.center_x + a * math.sin(theta) - b * math.cos(theta)
            z = self.z0 * math.cos(theta) + self.vz0 / self.n * math.sin(theta)
            return math.hypot(x, z)

        grid = np.linspace(0.0, 2.0 * np.pi, 721)
        k = int(np.argmin([distance(t) for t in grid]))
        found = minimize_scalar(
            distance, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, 720)]), method="bounded"
        )
        return float(min(found.fun, distance(grid[k])))


@dataclass(frozen=True)
class ForeverVerdict:
    """Whether free drift is certified never to come within the radius, and when it might."""

    safe: bool
    time_to_threat: float


def passive_forever(
    state: Sequence[float], n: float, radius: float, window_orbits: float = 10.0
) -> ForeverVerdict:
    """Certify that Clohessy-Wiltshire free drift from `state` never comes within `radius` (m).

    Safe if the drifting tube clears the target in the radial and orbit-normal plane, or if the
    tube's centre is moving away and already farther along-track than 2 b + radius. Otherwise the
    passage of the centre past the target is sampled when it lasts at most `window_orbits`
    orbits; a longer passage is not certified, with `time_to_threat` the time it begins.
    """
    ellipse = TravellingEllipse.from_state(state, n)
    if ellipse.cross_section_clearance() >= radius:
        return ForeverVerdict(True, math.inf)
    reach = 2.0 * ellipse.semi_axis + radius
    rate = ellipse.drift_rate
    period = 2.0 * np.pi / n
    if rate == 0.0:
        if abs(ellipse.center_y) > reach:
            return ForeverVerdict(True, math.inf)
        start, end = 0.0, period
    else:
        start, end = sorted(((-reach - ellipse.center_y) / rate, (reach - ellipse.center_y) / rate))
        if end < 0.0:
            return ForeverVerdict(True, math.inf)
        start = max(start, 0.0)
        if end - start > window_orbits * period:
            return ForeverVerdict(False, start)
    times = np.linspace(start, end, max(3, math.ceil((end - start) / period * 400) + 1))
    ranges = np.linalg.norm(ellipse.position(times), axis=1)
    if ranges.min() < radius:
        return ForeverVerdict(False, float(times[int(np.argmax(ranges < radius))]))
    k = int(np.argmin(ranges))
    found = minimize_scalar(
        lambda t: float(np.linalg.norm(ellipse.position(t))),
        bounds=(times[max(k - 1, 0)], times[min(k + 1, len(times) - 1)]),
        method="bounded",
    )
    if found.fun < radius:
        return ForeverVerdict(False, float(found.x))
    return ForeverVerdict(True, math.inf)


def _collision_radius(ep: Episode, settings: MetricSettings) -> float:
    if settings.collision_radius is not None:
        return settings.collision_radius
    keep_out = ep.constraint("keep_out")
    return float(getattr(keep_out, "radius", 0.0))


def _drift_horizon(ep: Episode, settings: MetricSettings) -> float:
    return settings.drift_horizon or 2.0 * (2.0 * np.pi / ep.view.n)


def cam_impulses(ep: Episode, settings: MetricSettings) -> np.ndarray:
    """The abort set (Hill delta-v, m/s), declared or opposite the approach.

    On a V-bar approach the CAM is along-track away from the target; on an R-bar approach it is
    radially away with a retrograde part of one fifth (Fehse sections 4.5 and 5.7). The approach
    side is the docking axis if the goal has one, else the hold point's direction.
    """
    if settings.cam_impulses is not None:
        return np.asarray(settings.cam_impulses, dtype=float).reshape(-1, 3)
    goal = ep.view.goal
    side = np.asarray(goal.get("port_axis", goal.get("hold_point", (0.0, 0.0, 0.0))), float)
    if not np.any(side[:2]):
        raise ValueError("no approach direction: declare cam_impulses")
    if abs(side[1]) >= abs(side[0]):
        return np.array([[0.0, settings.cam_dv * np.sign(side[1]), 0.0]])
    away = np.array([5.0 * np.sign(side[0]), -1.0, 0.0]) / math.sqrt(26.0)
    return settings.cam_dv * away[None, :]


def _stuck_accels(ep: Episode) -> np.ndarray:
    """Acceleration in effect at each step boundary: the last segment of the step before."""
    last = [np.asarray(rec.actuation.segments[-1][1], float) for rec in ep.records]
    return np.array([np.zeros(3), *last]).reshape(-1, 3)


def _partial_burn_points(ep: Episode, count: int) -> tuple[np.ndarray, np.ndarray]:
    """States inside thrusting segments, `count` per segment, and the acceleration there."""
    states, accels = [], []
    for seg in Trace(ep).segments:
        if not np.any(seg.accel) or len(seg.times) < 3 or count <= 0:
            continue
        picks = np.unique(np.linspace(0, len(seg.times) - 1, count + 2).round().astype(int)[1:-1])
        states.extend(seg.states[picks])
        accels.extend([seg.accel] * len(picks))
    return np.array(states).reshape(-1, 6), np.array(accels).reshape(-1, 3)


def passive_metrics(ep: Episode, settings: MetricSettings | None = None) -> dict[str, float]:
    """S4, S4b, S4i, S4t, S4o, S8 and S9 for one episode.

    The passive-safety checks start from every step boundary beyond `passive_exclusion`, the
    stuck-thruster checks also from points inside thrusting segments (the thrust continues for
    `stuck_time`, or stops there for a burn cut short), and the abort checks from every step
    boundary beyond `commit_radius`. Unprotected time counts each step whose starting state is
    beyond `commit_radius` with neither a safe drift nor a safe abort.
    """
    settings = settings or MetricSettings()
    radius = _collision_radius(ep, settings)
    horizon = _drift_horizon(ep, settings)
    states, times = ep.states, ep.times
    rho = np.linalg.norm(states[:, :3], axis=1)
    passive_zone = rho > settings.passive_exclusion
    abort_zone = rho > settings.commit_radius
    zero = np.zeros(3)
    out = {
        "passive_safety_margin": math.nan,
        "passive_safe_fraction": math.nan,
        "passive_forever_fraction": math.nan,
        "passive_threat_time": math.nan,
        "stuck_thruster_margin": math.nan,
        "cam_margin": math.nan,
        "unprotected_time": 0.0,
    }

    passive_ok = np.zeros(len(states), dtype=bool)
    if passive_zone.any():
        drift = _arcs(ep, settings, radius, states[passive_zone], zero, 0.0, zero, horizon)
        safe, least = _settle(drift, int(passive_zone.sum()), 1)
        passive_ok[passive_zone] = safe
        out["passive_safety_margin"] = least
        out["passive_safe_fraction"] = float(safe.mean())
        if ep.dynamics == "cw":
            verdicts = [passive_forever(x, ep.view.n, radius) for x in states[passive_zone]]
            out["passive_forever_fraction"] = float(np.mean([v.safe for v in verdicts]))
            out["passive_threat_time"] = min(v.time_to_threat for v in verdicts)

        burn_states, burn_accels = _partial_burn_points(ep, settings.partial_burn_samples)
        outside = np.linalg.norm(burn_states[:, :3], axis=1) > settings.passive_exclusion
        starts = np.vstack([states[passive_zone], burn_states[outside]])
        accels = np.vstack([_stuck_accels(ep)[passive_zone], burn_accels[outside]])
        stuck = _arcs(ep, settings, radius, starts, accels, settings.stuck_time, zero, horizon)
        _, least = _settle(stuck, len(starts), 1)
        if outside.any():
            cut = _arcs(ep, settings, radius, burn_states[outside], zero, 0.0, zero, horizon)
            least = min(least, _settle(cut, int(outside.sum()), 1)[1])
        out["stuck_thruster_margin"] = least

    cam_ok = np.zeros(len(states), dtype=bool)
    if abort_zone.any():
        cams = cam_impulses(ep, settings)
        count, cols = int(abort_zone.sum()), len(cams)
        starts = np.repeat(states[abort_zone], cols, axis=0)
        kicks = np.tile(cams, (count, 1))
        cam_horizon = settings.cam_horizon or horizon
        arcs = _arcs(ep, settings, radius, starts, zero, settings.cam_delay, kicks, cam_horizon)
        safe, least = _settle(arcs, count, cols)
        cam_ok[abort_zone] = safe
        out["cam_margin"] = least

    exposed = abort_zone[:-1] & ~(passive_ok[:-1] | cam_ok[:-1])
    out["unprotected_time"] = float(np.sum(np.diff(times)[exposed]))
    return out
