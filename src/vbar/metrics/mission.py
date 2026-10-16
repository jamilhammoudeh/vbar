"""Mission metrics: success, terminal errors, time to complete and progress (METRICS M1 to M5).

The goal comes from the scenario's public goal: `hold_point` (m), `capture_radius` (m),
`capture_speed` (m/s) and optionally `goal_velocity` (m/s, default zero). The capture predicate
is |r - r_goal| < capture_radius and |v - v_goal| < capture_speed, checked at step boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .episode import Episode


def goal_state(goal: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Goal position (m) and velocity (m/s) in Hill axes."""
    r = np.asarray(goal["hold_point"], dtype=float)
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


def success(ep: Episode) -> bool:
    """M1: the capture predicate held at a step boundary and no hard constraint was violated."""
    return capture_time(ep) is not None and not hard_violated(ep)


def mission_metrics(ep: Episode) -> dict[str, float | bool]:
    """M1 to M5 for one episode."""
    goal = ep.view.goal
    position, velocity = goal_errors(ep.states, goal)
    reached = capture_time(ep)
    return {
        "success": success(ep),
        "terminal_pos_error": float(position[-1]),
        "terminal_vel_error": float(velocity[-1]),
        "time_to_complete": ep.duration if reached is None else reached,
        "progress": float(np.clip(1.0 - position[-1] / position[0], 0.0, 1.0))
        if position[0] > 0.0
        else 1.0,
    }
