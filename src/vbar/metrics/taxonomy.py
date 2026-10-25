"""Failure labels for failed runs, first match wins."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .episode import Episode
from .mission import (
    completion_time,
    docking,
    goal_bounds,
    goal_errors,
    hard_violated,
    hold_intervals,
)
from .settings import MetricSettings
from .stl import Signal, Trace, minimum

LABELS = (
    "filter_defect",
    "filter_infeasible",
    "keep_out_entry",
    "speed_limit_breach",
    "cone_violation",
    "propellant_exhaustion",
    "parse_failure",
    "sign_error",
    "oscillation",
    "contact_out_of_tolerance",
    "hold_not_maintained",
    "overshoot",
    "timeout_near",
    "timeout_far",
    "other",
)


def _below(minima: Mapping[str, float], *names: str) -> bool:
    return any(minima.get(name, math.inf) < 0.0 for name in names)


def _closest_to_goal(ep: Episode, trace: Trace) -> float:
    """Least distance (m) to the goal position over the episode, in continuous time."""
    goal = ep.view.goal

    def distance(state, t=0.0):
        return goal_errors(state, goal, t, ep.target, ep.view.n)[0]

    point_speed = goal_bounds(goal, ep.target, ep.view.n)[0]
    reach = Signal("goal_distance", distance, lambda v, a, r: v + point_speed, 1e-6)
    return min(float(distance(ep.start)), minimum(reach, trace).margin)


def failure_label(
    ep: Episode, metrics: Mapping[str, Any], settings: MetricSettings | None = None
) -> str | None:
    """The first matching label in `LABELS` for a failed run; None for a success.

    `metrics` are the episode's other metrics (success, minimum margins, filter, parse, chatter,
    contact and terminal figures), as `evaluate` gives them. Overshoot means the goal position
    came within the capture radius but the capture predicate never held, in continuous time;
    a timeout ending within twice the capture radius is near. An inspection timeout is near when
    the inspectable coverage misses the goal by at most twice the shortfall the goal allows.
    """
    if metrics["success"]:
        return None
    settings = settings or MetricSettings()
    trace = Trace(ep)
    minima = metrics["min_margin"]
    goal = ep.view.goal
    violated = hard_violated(ep)
    if violated and metrics["solver_failures"] > 0:
        return "filter_defect"
    if violated and metrics["infeasible_steps"] > 0:
        return "filter_infeasible"
    if _below(minima, "keep_out"):
        return "keep_out_entry"
    if _below(minima, "speed_limit", "closing_speed_limit"):
        return "speed_limit_breach"
    if _below(minima, "approach_cone"):
        return "cone_violation"
    if any(rec.actuation.empty for rec in ep.records):
        return "propellant_exhaustion"
    if metrics["parse_failure_rate"] > 0.5:
        return "parse_failure"
    if metrics["sign_check_pass"] is False:
        return "sign_error"
    if metrics["chatter"] > settings.oscillation_threshold:
        return "oscillation"
    if docking(goal) and metrics["capture_margin"] < 0.0:
        return "contact_out_of_tolerance"
    if goal.get("task") == "inspect":
        # Near means short of the goal by at most twice the shortfall the goal allows.
        near = metrics["progress"] >= 1.0 - 2.0 * (1.0 - goal["coverage"])
        if ep.truncated:
            return "timeout_near" if near else "timeout_far"
        return "other"
    held = goal.get("hold_time", 0.0) > 0.0 and hold_intervals(ep, trace)
    if held and completion_time(ep, trace) is None:
        return "hold_not_maintained"
    radius = goal.get("capture_radius", goal.get("lateral_limit"))
    if "capture_radius" in goal and _closest_to_goal(ep, trace) < radius:
        if not hold_intervals(ep, trace):
            return "overshoot"
    if ep.truncated:
        near = radius is not None and metrics["terminal_pos_error"] <= 2.0 * radius
        return "timeout_near" if near else "timeout_far"
    return "other"
