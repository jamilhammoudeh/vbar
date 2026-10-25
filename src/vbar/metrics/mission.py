"""Mission metrics M1 to M10: success, terminal errors, time, progress, contact and hold.

The goal comes from the scenario's public goal. A hold goal has `hold_point` (m),
`capture_radius` (m), `capture_speed` (m/s), optionally `goal_velocity` (m/s, default zero) and
a dwell `hold_time` (s); its capture predicate is |r - r_goal| < capture_radius and
|v - v_goal| < capture_speed. A docking goal has `port` (m), `port_axis` (unit vector from the port
toward the approach side), `lateral_limit` (m), `lateral_speed_limit` (m/s) and
`axial_speed_window` (m/s), and is judged at the contact event, the first crossing of the port
plane from the approach side within `contact_range` (m) of the port (Fehse, Automated Rendezvous
and Docking of Spacecraft, 2003, sections 7.1.1 and 8.3.5). The goal holds the port as briefed at
t = 0; on a recorded episode the port and its axis then turn with the target's true attitude mode,
and the speeds at contact are relative to the port's own velocity, as the environment judges them.
A capture goal has `capture_point` (m, body axes), `capture_radius`, `capture_speed` and
`hold_time`: a hold goal whose point is fixed to the target and moves with its true attitude mode,
the capture predicate then measuring the speed relative to the point (Fehse, section 8.3.6).
An inspection goal (`task` "inspect") is met when the coverage of the inspectable points reaches
`coverage`; that needs the lighting and the target's true attitude, so it is read from the
environment's verdict recorded in `Episode.outcome`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq

from ..scenarios.targets import body_point, body_point_velocity, briefed_mode
from .episode import Episode
from .stl import Signal, Trace, complement, negative_intervals


def goal_state(
    goal: Mapping[str, Any], t: Any = 0.0, target: Any = None, n: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Goal position (m) and velocity (m/s) in Hill axes at `t`.

    The hold point or the port, at rest; or a capture point fixed to the target, carried by its
    attitude mode `target` at mean motion `n`, with shape (*t.shape, 3).
    """
    if "capture_point" in goal:
        if target is None:
            raise ValueError("a capture goal moves with the target: record the target mode")
        point = goal["capture_point"]
        return body_point(target, point, t, n), body_point_velocity(target, point, t, n)
    r = np.asarray(goal["hold_point"] if "hold_point" in goal else goal["port"], dtype=float)
    v = np.asarray(goal.get("goal_velocity", (0.0, 0.0, 0.0)), dtype=float)
    return r, v


def goal_bounds(goal: Mapping[str, Any], target: Any = None, n: float = 0.0) -> tuple[float, float]:
    """Bounds on the goal point's speed (m/s) and acceleration (m/s^2) in the Hill frame.

    A point p fixed to a turning target moves at w x p and accelerates at
    dw/dt x p + w x (w x p), so the bounds are w_max |p| and (alpha_max + w_max^2) |p|; zero for
    a goal at rest.
    """
    if "capture_point" not in goal:
        return 0.0, 0.0
    reach = float(np.linalg.norm(goal["capture_point"]))
    rate = target.rate_bound(n)
    return rate * reach, (target.angular_acceleration_bound(n) + rate**2) * reach


def goal_errors(
    states: np.ndarray,
    goal: Mapping[str, Any],
    t: Any = 0.0,
    target: Any = None,
    n: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Position (m) and velocity (m/s) errors from the goal at `t`, for stacked states."""
    states = np.asarray(states)
    times = np.broadcast_to(np.asarray(t, dtype=float), states.shape[:-1])
    r, v = goal_state(goal, times, target, n)
    return (
        np.linalg.norm(states[..., :3] - r, axis=-1),
        np.linalg.norm(states[..., 3:6] - v, axis=-1),
    )


def captured(
    states: np.ndarray,
    goal: Mapping[str, Any],
    t: Any = 0.0,
    target: Any = None,
    n: float = 0.0,
) -> np.ndarray:
    """Whether each state satisfies the capture predicate at `t`."""
    position, velocity = goal_errors(states, goal, t, target, n)
    return (position < goal["capture_radius"]) & (velocity < goal["capture_speed"])


def docking(goal: Mapping[str, Any]) -> bool:
    """Whether the goal is judged at a contact event, which a contact window marks.

    A departure goal names the port it leaves from, so the port alone does not make a docking.
    """
    return "axial_speed_window" in goal


def hard_violated(ep: Episode) -> bool:
    """Whether any hard constraint's margin went below zero, at the start or within a step."""
    for c in ep.constraints:
        if c.name in ep.hard and float(c.margin(ep.start, 0.0)) < 0.0:
            return True
    return any(
        rec.segment.minima[name].margin < 0.0
        for rec in ep.records
        for name in ep.hard
        if name in rec.segment.minima
    )


def capture_time(ep: Episode) -> float | None:
    """First step boundary (s) at which the capture predicate holds, or None."""
    hits = np.flatnonzero(captured(ep.states[1:], ep.view.goal, ep.times[1:], ep.target, ep.view.n))
    return float(ep.times[1 + hits[0]]) if len(hits) else None


def hold_signal(goal: Mapping[str, Any], target: Any = None, n: float = 0.0) -> Signal:
    """The capture predicate as a margin: min((R - |r - r_g|) / R, (v_c - |v - v_g|) / v_c).

    Its rate is bounded by (V + V_g) / R + (A + A_g) / v_c, V and A the chaser's bounds and V_g
    and A_g the goal point's (`goal_bounds`).
    """
    radius, speed = goal["capture_radius"], goal["capture_speed"]
    point_speed, point_accel = goal_bounds(goal, target, n)

    def margin(state, t=0.0):
        position, velocity = goal_errors(state, goal, t, target, n)
        return np.minimum((radius - position) / radius, (speed - velocity) / speed)

    def rate(v, a, rho_min):
        return (v + point_speed) / radius + (a + point_accel) / speed

    return Signal("hold", margin, rate, 1e-9)


def hold_intervals(ep: Episode, trace: Trace | None = None) -> list[tuple[float, float]]:
    """Time intervals (s) during which the capture predicate holds, in continuous time."""
    if not ep.records:
        return []
    signal = hold_signal(ep.view.goal, ep.target, ep.view.n)
    outside = negative_intervals(signal, trace or Trace(ep))
    return complement(outside, 0.0, ep.duration)


@dataclass(frozen=True)
class Contact:
    """State at the first port-plane crossing: time (s), lateral offset (m), speeds (m/s)."""

    t: float
    lateral_error: float
    axial_speed: float
    lateral_speed: float
    margin: float


def port_motion(ep: Episode) -> Callable[[Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The port's position (m), unit axis and velocity (m/s) in Hill axes as functions of time.

    With the episode's true target mode recorded, the briefed port and axis at t = 0 are taken
    to body axes and carried by the true mode, the geometry the environment flies; without one
    (a hand-built episode) the goal's port is fixed.
    """
    goal = ep.view.goal
    port = np.asarray(goal["port"], dtype=float)
    axis = np.asarray(goal["port_axis"], dtype=float)
    axis = axis / np.linalg.norm(axis)
    mode, n = ep.target, ep.view.n
    if mode is None:

        def fixed(t):
            shape = (*np.shape(t), 3)
            return np.broadcast_to(port, shape), np.broadcast_to(axis, shape), np.zeros(shape)

        return fixed
    turn = briefed_mode(mode).matrix(0.0, n)
    port_body, axis_body = turn.T @ port, turn.T @ axis

    def moving(t):
        return (
            body_point(mode, port_body, t, n),
            body_point(mode, axis_body, t, n),
            body_point_velocity(mode, port_body, t, n),
        )

    return moving


def contact(ep: Episode, trace: Trace | None = None) -> Contact | None:
    """The contact event and its capture margin, or None if the port plane was never crossed.

    Contact is the first crossing between recorded samples at which the depth (r - p) . a_hat
    turns from positive to not positive, refined by Brent's method on the rebuilt trajectory, at a
    point within the goal's `contact_range` of the port (no limit when the goal has none): the
    rule the environment applies to the same samples. The margin is the least over the criteria
    of (limit - value) / limit, with the lower end of the axial window taken as
    (value - limit) / limit, or left out when it is zero; negative means out of tolerance.
    """
    goal = ep.view.goal
    trace = trace or Trace(ep)
    motion = port_motion(ep)
    reach = goal.get("contact_range", math.inf)

    def depth(t):
        port, axis, _ = motion(t)
        return float((trace.state_at(t)[:3] - port) @ axis)

    for rec in ep.records:
        times, states = rec.segment.times, rec.segment.states
        port, axis, _ = motion(times)
        depths = np.sum((states[:, :3] - port) * axis, axis=-1)
        for k in np.flatnonzero((depths[:-1] > 0.0) & (depths[1:] <= 0.0)):
            if depths[k + 1] == 0.0:
                t = float(times[k + 1])
            else:
                t = brentq(depth, times[k], times[k + 1], xtol=1e-12, rtol=4 * np.finfo(float).eps)
            state = trace.state_at(t)
            at = motion(t)
            if float(np.linalg.norm(state[:3] - at[0])) > reach:
                continue
            return _judge(goal, state, t, *at)
    return None


def _judge(goal, state, t, port, axis, port_velocity) -> Contact:
    """The contact at `t`: offsets from the port and speeds relative to its velocity."""
    offset = state[:3] - port
    velocity = state[3:6] - port_velocity
    axial = -float(velocity @ axis)
    lateral = float(np.linalg.norm(offset - (offset @ axis) * axis))
    sideways = float(np.linalg.norm(velocity + axial * axis))
    low, high = goal["axial_speed_window"]
    criteria = [
        (goal["lateral_limit"] - lateral) / goal["lateral_limit"],
        (goal["lateral_speed_limit"] - sideways) / goal["lateral_speed_limit"],
        (high - axial) / high,
    ]
    # A window open at zero (soft docking) has no lower criterion to normalise.
    if low > 0.0:
        criteria.append((axial - low) / low)
    return Contact(t, lateral, axial, sideways, min(criteria))


def inspection_outcome(ep: Episode) -> Mapping[str, Any] | None:
    """The recorded coverage verdict of an inspection episode, or None for other goals.

    Raises
    ------
    ValueError
        If the goal is an inspection and the episode carries no outcome.
    """
    if ep.view.goal.get("task") != "inspect":
        return None
    if ep.outcome is None:
        raise ValueError("an inspection episode needs its outcome; record it with record_episode")
    return ep.outcome


def completion_time(ep: Episode, trace: Trace | None = None) -> float | None:
    """When the goal is met (s), hard constraints aside, or None.

    For a docking goal, the contact event within tolerance; for a hold goal with a `hold_time`,
    the end of the first dwell that long; otherwise the first step boundary inside the capture
    predicate.
    """
    goal = ep.view.goal
    outcome = inspection_outcome(ep)
    if outcome is not None:
        return float(outcome["time_s"]) if outcome["met"] else None
    if docking(goal):
        event = contact(ep, trace)
        return event.t if event is not None and event.margin >= 0.0 else None
    hold_time = goal.get("hold_time", 0.0)
    if hold_time > 0.0:
        for lo, hi in hold_intervals(ep, trace):
            if hi - lo >= hold_time:
                return lo + hold_time
        return None
    return capture_time(ep)


def success(ep: Episode) -> bool:
    """M1: the goal was met and no hard constraint was violated."""
    return completion_time(ep) is not None and not hard_violated(ep)


def mission_metrics(ep: Episode) -> dict[str, float | bool]:
    """M1 to M10 for one episode.

    Contact metrics are NaN without a docking goal or a contact; the dwell is NaN without a capture
    predicate. An inspection has no goal position, so its terminal errors are NaN and its
    progress is the inspectable coverage.
    """
    goal = ep.view.goal
    trace = Trace(ep)
    outcome = inspection_outcome(ep)
    if outcome is not None:
        done = completion_time(ep, trace)
        return {
            "success": done is not None and not hard_violated(ep),
            "terminal_pos_error": math.nan,
            "terminal_vel_error": math.nan,
            "time_to_complete": ep.duration if done is None else done,
            "progress": float(outcome["progress"]),
            "contact_lateral_error": math.nan,
            "contact_axial_speed": math.nan,
            "contact_lateral_speed": math.nan,
            "hold_dwell": math.nan,
            "capture_margin": math.nan,
        }
    if docking(goal):
        # The port moves with the target, so the terminal errors and the progress are taken
        # from it where it stands.
        port, _, port_velocity = port_motion(ep)(ep.times)
        position = np.linalg.norm(ep.states[:, :3] - port, axis=-1)
        velocity = np.linalg.norm(ep.states[:, 3:6] - port_velocity, axis=-1)
    else:
        position, velocity = goal_errors(ep.states, goal, ep.times, ep.target, ep.view.n)
    done = completion_time(ep, trace)
    out: dict[str, float | bool] = {
        "success": done is not None and not hard_violated(ep),
        "terminal_pos_error": float(position[-1]),
        "terminal_vel_error": float(velocity[-1]),
        "time_to_complete": ep.duration if done is None else done,
        "progress": float(np.clip(1.0 - position[-1] / position[0], 0.0, 1.0))
        if position[0] > 0.0
        else 1.0,
        "contact_lateral_error": math.nan,
        "contact_axial_speed": math.nan,
        "contact_lateral_speed": math.nan,
        "hold_dwell": math.nan,
        "capture_margin": math.nan,
    }
    if docking(goal):
        event = contact(ep, trace)
        if event is not None:
            out.update(
                contact_lateral_error=event.lateral_error,
                contact_axial_speed=event.axial_speed,
                contact_lateral_speed=event.lateral_speed,
                capture_margin=event.margin,
            )
    if "capture_radius" in goal:
        out["hold_dwell"] = max((hi - lo for lo, hi in hold_intervals(ep, trace)), default=0.0)
    return out
