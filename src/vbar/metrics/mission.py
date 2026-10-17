"""Mission metrics M1 to M10: success, terminal errors, time, progress, contact and hold.

The goal comes from the scenario's public goal. A hold goal has `hold_point` (m),
`capture_radius` (m), `capture_speed` (m/s), optionally `goal_velocity` (m/s, default zero) and
a dwell `hold_time` (s); its capture predicate is |r - r_goal| < capture_radius and
|v - v_goal| < capture_speed. A docking goal has `port` (m), `port_axis` (unit vector from the port
toward the approach side), `lateral_limit` (m), `lateral_speed_limit` (m/s) and
`axial_speed_window` (m/s), and is judged at the contact event, the first crossing of the port
plane from the approach side (Fehse, Automated Rendezvous and Docking of Spacecraft, 2003,
sections 7.1.1 and 8.3.5).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .episode import Episode
from .stl import Signal, Trace, complement, negative_intervals


def goal_state(goal: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Goal position (m, the hold point or the port) and velocity (m/s) in Hill axes."""
    r = np.asarray(goal["hold_point"] if "hold_point" in goal else goal["port"], dtype=float)
    v = np.asarray(goal.get("goal_velocity", (0.0, 0.0, 0.0)), dtype=float)
    return r, v


def goal_errors(states: np.ndarray, goal: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Position (m) and velocity (m/s) errors from the goal, for stacked states."""
    r, v = goal_state(goal)
    states = np.asarray(states)
    return (
        np.linalg.norm(states[..., :3] - r, axis=-1),
        np.linalg.norm(states[..., 3:6] - v, axis=-1),
    )


def captured(states: np.ndarray, goal: Mapping[str, Any]) -> np.ndarray:
    """Whether each state satisfies the capture predicate."""
    position, velocity = goal_errors(states, goal)
    return (position < goal["capture_radius"]) & (velocity < goal["capture_speed"])


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
    hits = np.flatnonzero(captured(ep.states[1:], ep.view.goal))
    return float(ep.times[1 + hits[0]]) if len(hits) else None


def hold_signal(goal: Mapping[str, Any]) -> Signal:
    """The capture predicate as a margin: min((R - |r - r_g|) / R, (v_c - |v - v_g|) / v_c)."""
    radius, speed = goal["capture_radius"], goal["capture_speed"]

    def margin(state, t=0.0):
        position, velocity = goal_errors(state, goal)
        return np.minimum((radius - position) / radius, (speed - velocity) / speed)

    def rate(v, a, rho_min):
        return v / radius + a / speed

    return Signal("hold", margin, rate, 1e-9)


def hold_intervals(ep: Episode, trace: Trace | None = None) -> list[tuple[float, float]]:
    """Time intervals (s) during which the capture predicate holds, in continuous time."""
    if not ep.records:
        return []
    outside = negative_intervals(hold_signal(ep.view.goal), trace or Trace(ep))
    return complement(outside, 0.0, ep.duration)


@dataclass(frozen=True)
class Contact:
    """State at the first port-plane crossing: time (s), lateral offset (m), speeds (m/s)."""

    t: float
    lateral_error: float
    axial_speed: float
    lateral_speed: float
    margin: float


def contact(ep: Episode, trace: Trace | None = None) -> Contact | None:
    """The contact event and its capture margin, or None if the port plane was never crossed.

    The margin is the least over the criteria of (limit - value) / limit, with the lower end of
    the axial window taken as (value - limit) / limit, or left out when it is zero; negative
    means out of tolerance.
    """
    goal = ep.view.goal
    trace = trace or Trace(ep)
    port = np.asarray(goal["port"], dtype=float)
    axis = np.asarray(goal["port_axis"], dtype=float)
    axis = axis / np.linalg.norm(axis)

    def depth(state, t=0.0):
        return (np.asarray(state)[..., :3] - port) @ axis

    crossings = negative_intervals(Signal("port_plane", depth, lambda v, a, r: v, 1e-6), trace)
    starts = [lo for lo, _ in crossings if lo > 0.0]
    if not starts:
        return None
    t = starts[0]
    state = trace.state_at(t)
    offset = state[:3] - port
    velocity = state[3:6]
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
    margin = min(criteria)
    return Contact(t, lateral, axial, sideways, margin)


def completion_time(ep: Episode, trace: Trace | None = None) -> float | None:
    """When the goal is met (s), hard constraints aside, or None.

    For a docking goal, the contact event within tolerance; for a hold goal with a `hold_time`,
    the end of the first dwell that long; otherwise the first step boundary inside the capture
    predicate.
    """
    goal = ep.view.goal
    if "port" in goal:
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

    Contact metrics are NaN without a port or a contact; the dwell is NaN without a capture
    predicate.
    """
    goal = ep.view.goal
    trace = Trace(ep)
    position, velocity = goal_errors(ep.states, goal)
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
    if "port" in goal:
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
