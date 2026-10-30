"""The task in words: a scenario's public goal and constraints, and the state measured against them.

`briefing(view)` describes the goal of every task kind (hold, berth, dock, inspect, depart) and
every public constraint in plain sentences; `situation(view, obs, info)` gives the quantities a
crew display shows for the current observation (distance to the hold point, offsets from a
moving port's axis, coverage). Both read only the `AgentView`, the observation and the agent's
`info`, so a prompt built from them carries nothing an agent may not know. Body-fixed geometry
(ports, corridors, the body keep-out) follows the briefed target mode in the goal's "target"
entry, rebuilt by `target_mode`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..constraints import _VBAR_COAST
from ..scenarios.targets import (
    FixedOffset,
    InertialPointing,
    Maneuvering,
    Passive,
    TargetMode,
    Tumbling,
    sun_at,
)
from .base import AgentView

# Goal keys each task needs before it can be described; any other task is described from its
# keys as they are.
TASK_KEYS: dict[str, tuple[str, ...]] = {
    "hold": ("hold_point", "capture_radius", "capture_speed"),
    "berth": ("hold_point", "capture_radius", "capture_speed", "capture_time"),
    "dock": ("port", "port_axis", "lateral_limit", "lateral_speed_limit", "axial_speed_window"),
    "inspect": ("points", "target_radius", "max_range", "sun_exclusion", "coverage"),
    "depart": ("port", "port_axis", "clear_range", "drift_horizon"),
}

# Goal keys the task sentences use, besides the task's own keys and "task" and "target"; the rest
# are described one by one. The inspection and departure goals carry a hold point and capture ball
# for the mission metrics; the task sentences state the real goal instead, and the departure's
# names the clear point.
_SCORED_BALL = ("hold_point", "capture_radius", "capture_speed")
_DESCRIBED = {
    "hold": ("hold_time",),
    "berth": ("hold_time",),
    "dock": ("contact_range",),
    "inspect": (*_SCORED_BALL, "shadow"),
    "depart": _SCORED_BALL,
}


def _num(value: float) -> str:
    return f"{float(value):.4g}"


def _vec(values: ArrayLike, digits: int = 2) -> str:
    return "[" + ", ".join(f"{float(c):.{digits}f}" for c in np.ravel(values)) + "]"


def _sig(values: ArrayLike) -> str:
    """A vector to four significant figures, so accelerations of any size keep their digits."""
    return "[" + ", ".join(f"{float(c):.4g}" for c in np.ravel(values)) + "]"


def _deg(angle: float) -> str:
    return f"{math.degrees(float(angle)):.4g} deg"


def _finite(where: str, key: str, value: Any) -> np.ndarray:
    """`value` as finite floats, refusing booleans, text and non-finite numbers."""
    array = np.asarray(value) if not isinstance(value, Mapping) else np.asarray(None)
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"the {where} {key} must be numeric and finite, got {value!r}")
    return array.astype(float)


def _mode_numbers(mode: str, description: Mapping[str, Any]) -> None:
    """Refuse a mode parameter that is not a finite number, or a zero spin axis."""
    for key in ("pitch", "limit_cycle_amplitude", "limit_cycle_period", "phase", "rate"):
        if key in description and _finite(f"{mode} target mode's", key, description[key]).shape:
            raise ValueError(f"the {mode} target mode's {key} must be a number")
    if "axis" in description:
        axis = _finite(f"{mode} target mode's", "axis", description["axis"])
        if axis.shape != (3,) or not np.linalg.norm(axis) > 0.0:
            raise ValueError(f"the {mode} target mode's axis must be a nonzero 3-vector")
    for burn in description.get("burns", ()):
        try:
            start, duration, accel = burn
        except (TypeError, ValueError):
            raise ValueError(f"a burn is (start, duration, acceleration), got {burn!r}") from None
        for key, value in (("start", start), ("duration", duration), ("acceleration", accel)):
            _finite("burn's", key, value)


def target_mode(description: Mapping[str, Any]) -> TargetMode:
    """The target attitude mode a goal's "target" entry describes.

    Parameters
    ----------
    description : mapping
        The entry: "mode" and the mode's parameters (rad, rad/s, s, m/s^2), as
        `TargetMode.public` gives them.

    Returns
    -------
    TargetMode
        The mode, which agents may propagate as the briefing tells it.

    Raises
    ------
    ValueError
        If the mode is not one of passive, fixed_offset, inertial, tumbling or maneuvering, a
        parameter it needs is missing, or a parameter is not a finite number.
    """
    mode = description.get("mode", "passive")
    _mode_numbers(mode, description)
    try:
        if mode == "maneuvering":
            attitude = target_mode(_attitude_of(description))
            return Maneuvering(tuple(description["burns"]), attitude)
        if mode == "passive":
            return Passive()
        if mode == "fixed_offset":
            return FixedOffset(
                description["pitch"],
                description.get("limit_cycle_amplitude", 0.0),
                description.get("limit_cycle_period", 0.0),
            )
        if mode == "inertial":
            return InertialPointing(description["phase"])
        if mode == "tumbling":
            return Tumbling(tuple(description["axis"]), description["rate"], description["phase"])
    except KeyError as error:
        raise ValueError(f"the {mode} target mode needs {error.args[0]!r}") from None
    raise ValueError(f"unknown target mode {mode!r}")


def _attitude_of(description: Mapping[str, Any]) -> dict[str, Any]:
    """The attitude a manoeuvring target keeps, from the entries its description merges in."""
    attitude = {k: v for k, v in description.items() if k not in ("mode", "burns")}
    if "axis" in attitude:
        return {**attitude, "mode": "tumbling"}
    if "pitch" in attitude:
        return {**attitude, "mode": "fixed_offset"}
    if "phase" in attitude:
        return {**attitude, "mode": "inertial"}
    return {"mode": "passive"}


def _attitude_text(description: Mapping[str, Any]) -> str:
    mode = description.get("mode", "passive")
    if mode == "maneuvering":
        burns = "; ".join(
            f"from t = {_num(s)} s for {_num(d)} s at {_sig(a)} m/s^2"
            for s, d, a in description["burns"]
        )
        return (
            f"{_attitude_text(_attitude_of(description))} The target fires its engines on a "
            f"schedule ({burns}, Hill axes); your relative motion feels the opposite acceleration."
        )
    if mode == "passive":
        return "The target keeps its axes aligned with this frame."
    if mode == "fixed_offset":
        text = f"The target is pitched by {_deg(description['pitch'])} about the z axis"
        amplitude = description.get("limit_cycle_amplitude", 0.0)
        if amplitude:
            text += (
                f", oscillating by {_deg(amplitude)} with a period of "
                f"{_num(description['limit_cycle_period'])} s"
            )
        return text + "; points on its body turn with it."
    if mode == "inertial":
        return (
            "The target points in a fixed inertial direction, so its body turns about the z axis "
            f"at {_num(description['rate'])} rad/s in this frame, from an angle of "
            f"{_deg(description['phase'])} at t = 0; points on its body turn with it."
        )
    if mode == "tumbling":
        return (
            f"The target tumbles at {_num(description['rate'])} rad/s about its body axis "
            f"{_vec(description['axis'], 3)}, from an angle of {_deg(description['phase'])} at "
            "t = 0; points on its body turn with it."
        )
    return f"Target: {dict(description)}."


def _task_text(view: AgentView) -> list[str]:
    goal = view.goal
    task = goal.get("task", "hold")
    if task == "hold":
        stay = goal.get("hold_time", 0.0)
        line = (
            f"Task: reach the hold point {_vec(goal['hold_point'])} m and stop there: within "
            f"{_num(goal['capture_radius'])} m of it at a speed below "
            f"{_num(goal['capture_speed'])} m/s"
        )
        if stay:
            line += f", without a break for {_num(stay)} s"
        return [line + "."]
    if task == "berth":
        return [
            f"Task: berthing. Bring the chaser to rest in the capture box: within "
            f"{_num(goal['capture_radius'])} m of {_vec(goal['hold_point'])} m at a speed below "
            f"{_num(goal['capture_speed'])} m/s. Once it is there at the end of a step the "
            f"thrusters are inhibited for {_num(goal['capture_time'])} s while the arm grapples, "
            "and it must stay in the box, drifting, until then."
        ]
    if task == "dock":
        low, high = goal["axial_speed_window"]
        contact = goal.get("contact_range")
        where = "" if contact is None else f" within {_num(contact)} m of the port"
        return [
            f"Task: dock at the target's port, at {_vec(goal['port'])} m at t = 0, with its axis "
            f"(from the port toward the approach side) {_vec(goal['port_axis'], 3)}.",
            f"Contact is the first crossing of the port plane{where}. At contact the lateral "
            f"offset from the axis must be at most {_num(goal['lateral_limit'])} m, the lateral "
            f"speed at most {_num(goal['lateral_speed_limit'])} m/s, and the closing speed along "
            f"the axis between {_num(low)} and {_num(high)} m/s.",
        ]
    if task == "inspect":
        lit = (
            "the Sun lights the point with the target outside the planet's shadow"
            if goal.get("shadow", True)
            else "the Sun lights the point (the planet's shadow is not modelled)"
        )
        return [
            f"Task: inspect the target, a sphere of radius {_num(goal['target_radius'])} m "
            f"carrying {goal['points']} surface points. A point is imaged when you are within "
            f"{_num(goal['max_range'])} m of the target on the side the point faces, {lit}, "
            "and the Sun is more "
            f"than {_deg(goal['sun_exclusion'])} from your camera's line of sight to the target. "
            f"Success is imaging {_num(100.0 * goal['coverage'])}% of the points that are lit at "
            "some time in the episode."
        ]
    if task == "depart":
        lines = [
            f"Task: depart from the target's port, at {_vec(goal['port'])} m at t = 0 with its "
            f"axis (from the port outward) {_vec(goal['port_axis'], 3)}. Reach a range of at "
            f"least {_num(goal['clear_range'])} m from the target on a path whose free drift (no "
            f"thrust) does not come back into the keep-out zone for {_num(goal['drift_horizon'])} "
            "s."
        ]
        if "hold_point" in goal:
            lines.append(
                f"Arrival is scored at the clear point {_vec(goal['hold_point'])} m, within "
                f"{_num(goal['capture_radius'])} m of it."
            )
        return lines
    return [f"Task: {task}."]


def _sun_text(sun0: ArrayLike, n: float) -> str:
    return (
        f"The Sun direction at t = 0 is {_vec(sun0, 3)}; the Sun is fixed in inertial space, so in "
        f"this frame it turns about the z axis at {_num(-n)} rad/s."
    )


def _goal_extras(view: AgentView) -> list[str]:
    goal = view.goal
    task = goal.get("task", "hold")
    described = {"task", "target", *TASK_KEYS.get(task, ()), *_DESCRIBED.get(task, ())}
    lines = []
    for key, value in goal.items():
        if key in described:
            continue
        if key == "waypoints":
            points = ", ".join(_vec(p) for p in value)
            lines.append(f"Suggested waypoints (not scored): {points} m.")
        elif key == "hold" and task == "dock":
            lines.append(
                f"A station-keeping point {_num(value[0])} m out on the axis, for a dwell of "
                f"{_num(value[1])} s, is available before the final approach (not scored)."
            )
        elif key == "thrust_inhibit":
            windows = ", ".join(f"{_num(a)} s to {_num(b)} s" for a, b in value)
            lines.append(f"The thrusters are inhibited (commands are ignored) from {windows}.")
        elif key == "passive_exclusion":
            where = "everywhere" if not value else f"farther than {_num(value)} m from the target"
            lines.append(
                f"Passive safety is scored {where}: if thrust stopped, the free drift should not "
                "collide with the target."
            )
        elif key == "commit_radius":
            lines.append(
                f"The ability to abort is scored except within {_num(value)} m of the target."
            )
        elif key == "plume_range":
            lines.append(
                f"Within {_num(value)} m of the target, thrust whose exhaust points at the target "
                "is penalised (plume impingement)."
            )
        elif key == "cam_impulses":
            if len(value):
                burns = ", ".join(_vec(dv, 3) for dv in value)
                lines.append(
                    f"The ability to abort is scored with the collision avoidance burns {burns} "
                    "m/s."
                )
        elif key == "out_of_bounds":
            lines.append(
                f"The episode ends as a failure if you go farther than {_num(value)} m from the "
                "target."
            )
        elif key == "end_on_arrival" and value:
            lines.append(
                f"The episode ends when you come within {_num(goal['capture_radius'])} m of the "
                f"hold point, a success only at a speed below {_num(goal['capture_speed'])} m/s."
            )
        elif key == "sun_direction":
            lines.append(_sun_text(value, view.n))
        elif key == "lighting_window" and any(
            c["type"] == "lighting_window" for c in view.constraints
        ):
            continue
        else:
            lines.append(f"Goal entry {key}: {value}.")
    return lines


def _body_fixed(spec: Mapping[str, Any]) -> bool:
    """Whether a constraint's geometry is in body axes and turns with the target."""
    kind = spec["type"]
    return (kind.startswith("approach_cone") and "n" in spec) or (
        kind.startswith("keep_out") and "port" in spec
    )


def _constraint_text(spec: Mapping[str, Any], mode: TargetMode, n: float) -> str:
    kind = spec["type"]
    rotate = mode.matrix(0.0, n) if _body_fixed(spec) else np.eye(3)
    turns = " (body-fixed: it turns with the target)" if _body_fixed(spec) else ""
    if kind.startswith("keep_out"):
        if "port" in spec:
            return (
                f"Never enter the sphere of radius {_num(spec['radius'])} m around the target's "
                "centre, except through the port, where the docking mechanism's capture volume "
                f"reaches {_num(spec.get('reception', 0.0))} m behind the port plane."
            )
        return f"Never come within {_num(spec['radius'])} m of the target's centre (the origin)."
    if kind.startswith("approach_cone"):
        reach = spec.get("active_range", math.inf)
        apex, axis = rotate @ np.asarray(spec["apex"]), rotate @ np.asarray(spec["axis"])
        lead = f"From the point {_vec(apex)} m"
        if math.isfinite(reach):
            lead = f"Within {_num(reach)} m of the point {_vec(apex)} m"
        return (
            f"Corridor{turns}: {lead}, stay inside the cone of half-angle "
            f"{_deg(spec['half_angle'])} around the axis {_vec(axis, 3)} (at t = 0) from that "
            "point."
        )
    if kind.startswith("speed_limit"):
        return (
            f"Your speed must stay below {_num(spec['v0'])} m/s plus {_num(spec['v1'])} m/s per "
            "metre of distance to the target."
        )
    if kind.startswith("closing_speed_limit"):
        return (
            f"Your closing speed (the rate at which your distance to the target falls) must stay "
            f"below {_num(spec['v0'])} m/s plus {_num(spec['v1'])} m/s per metre of distance."
        )
    if kind.startswith("gate_table"):
        gates = ", ".join(f"{_num(r)} m: {_num(v)} m/s" for r, v in sorted(spec["gates"]))
        return (
            "Your closing speed must stay below these limits by distance to the target, "
            f"interpolated between them and held beyond the table: {gates}."
        )
    if kind.startswith("passive_speed"):
        extent = _num(spec["extent"])
        if spec["axis"] == "vbar":
            law = f"n (range - {extent}) / {_VBAR_COAST:.5f} m/s"
        else:
            law = f"n sqrt((4 range - {extent})^2 - 9 range^2) m/s, and zero within {extent} m"
        return (
            "Your closing speed must stay low enough that a thrust failure still misses "
            f"structure reaching {extent} m from the target: at most {law}, with "
            f"n = {spec['n']:.6f} rad/s and the range in m."
        )
    if kind.startswith("lighting_window"):
        low, high = spec["window"]
        return (
            f"Within {_num(spec['active_range'])} m of the port the Sun must light the target and "
            f"its direction in the orbit plane must be between {_deg(low)} and {_deg(high)} from "
            "the approach axis."
        )
    if kind.startswith("zone_entry"):
        return (
            f"Do not enter the ellipsoid with semi-axes {_vec(spec['semi_axes'], 1)} m (Hill axes) "
            f"around {_vec(spec['center'], 1)} m before t = {_num(spec['go_time'])} s."
        )
    params = {k: v for k, v in spec.items() if k != "type"}
    return f"Constraint {kind}: {params}."


# The shape and sign of every numeric goal entry the briefing prints: "vector" a 3-vector,
# "window" (low, high) with 0 <= low < high, "standoff" (distance > 0, dwell >= 0), "vectors"
# and "spans" lists of 3-vectors and of (start, end) pairs with 0 <= start <= end, "positive"
# > 0, "count" a positive integer, "nonnegative" >= 0, "fraction" in (0, 1], "angle" in [0, pi]
# rad. A Sun direction must also be nonzero.
_GOAL_FORMS = {
    "hold_point": "vector",
    "port": "vector",
    "port_axis": "vector",
    "sun_direction": "vector",
    "axial_speed_window": "window",
    "cam_impulses": "vectors",
    "waypoints": "vectors",
    "hold": "standoff",
    "thrust_inhibit": "spans",
    "capture_radius": "positive",
    "capture_speed": "positive",
    "capture_time": "positive",
    "lateral_limit": "positive",
    "lateral_speed_limit": "positive",
    "target_radius": "positive",
    "max_range": "positive",
    "clear_range": "positive",
    "drift_horizon": "positive",
    "points": "count",
    "contact_range": "positive",
    "out_of_bounds": "positive",
    "hold_time": "nonnegative",
    "commit_radius": "nonnegative",
    "plume_range": "nonnegative",
    "passive_exclusion": "nonnegative",
    "coverage": "fraction",
    "sun_exclusion": "angle",
}


def _goal_entry(task: str, key: str, value: Any) -> None:
    """Refuse a goal entry whose numbers, shape or sign the briefing cannot describe."""
    where = f"{task} goal's"
    form = _GOAL_FORMS[key]
    if form in ("vectors", "spans") and np.size(np.asarray(value, dtype=object)) == 0:
        return
    x = _finite(where, key, value)
    shapes = {"vector": (3,), "window": (2,), "standoff": (2,)}
    if form in shapes and x.shape != shapes[form]:
        raise ValueError(f"the {where} {key} must have shape {shapes[form]}, got {value!r}")
    if form in ("vectors", "spans") and (
        x.ndim != 2 or x.shape[1] != (3 if form == "vectors" else 2)
    ):
        raise ValueError(f"the {where} {key} has the wrong shape, got {value!r}")
    if form not in ("vector", "window", "standoff", "vectors", "spans") and x.shape != ():
        raise ValueError(f"the {where} {key} must be a number, got {value!r}")
    if form == "window":
        bad = not 0.0 <= x[0] < x[1]
    elif form == "standoff":
        bad = not (x[0] > 0.0 and x[1] >= 0.0)
    elif form == "spans":
        bad = bool(np.any(x[:, 0] < 0.0) or np.any(x[:, 0] > x[:, 1]))
    elif form == "count":
        bad = not (x > 0.0 and x == np.floor(x))
    elif form == "positive":
        bad = not x > 0.0
    elif form == "nonnegative":
        bad = not x >= 0.0
    elif form == "fraction":
        bad = not 0.0 < x <= 1.0
    elif form == "angle":
        bad = not 0.0 <= x <= math.pi
    else:
        bad = False
    if key == "sun_direction" and not np.linalg.norm(x) > 0.0:
        bad = True
    if bad:
        raise ValueError(f"the {where} {key} is out of range, got {value!r}")


def check_goal(view: AgentView) -> None:
    """Refuse a goal the briefing cannot describe.

    Parameters
    ----------
    view : AgentView
        The scenario's public description.

    Raises
    ------
    ValueError
        If the goal of a known task lacks a key the task needs, a numeric entry the briefing
        prints is not finite numbers of the right shape and sign, its port axis is zero, or its
        target mode is unknown or has a parameter that is not a finite number.
    """
    task = view.goal.get("task", "hold")
    missing = [k for k in TASK_KEYS.get(task, ()) if k not in view.goal]
    if missing:
        raise ValueError(f"the {task} goal has no {', '.join(missing)}")
    for key, value in view.goal.items():
        if key in _GOAL_FORMS:
            _goal_entry(task, key, value)
    if "port_axis" in TASK_KEYS.get(task, ()):
        axis = np.asarray(view.goal["port_axis"], dtype=float)
        if not np.linalg.norm(axis) > 0.0:
            raise ValueError(f"the {task} goal's port_axis must be a nonzero 3-vector")
    target_mode(view.goal.get("target", {"mode": "passive"}))


def briefing(view: AgentView) -> str:
    """The goal, the target's motion and every public constraint, in sentences.

    Parameters
    ----------
    view : AgentView
        The scenario's public description; positions in m in Hill axes at t = 0.

    Returns
    -------
    str
        One line per sentence.

    Raises
    ------
    ValueError
        As `check_goal`.
    """
    check_goal(view)
    description = view.goal.get("target", {"mode": "passive"})
    mode = target_mode(description)
    lines = [*_task_text(view), _attitude_text(description), *_goal_extras(view)]
    if view.constraints:
        lines.append("Constraints:")
        lines.extend(f"- {_constraint_text(c, mode, view.n)}" for c in view.constraints)
    return "\n".join(lines)


def situation(view: AgentView, obs: np.ndarray, info: Mapping[str, Any]) -> str:
    """The observed state measured against the goal, one line per quantity.

    Distances in m, speeds in m/s, from the observation `obs` (Hill) and the agent's `info`.

    Parameters
    ----------
    view : AgentView
        The scenario's public description.
    obs : numpy.ndarray
        The observation; its first six entries are the observed relative state.
    info : mapping
        The agent's info: "time_s" and, for inspection, "coverage".

    Returns
    -------
    str
        The lines, indented for a prompt.
    """
    goal = view.goal
    task = goal.get("task", "hold")
    t = float(info["time_s"])
    r, v = np.asarray(obs[:3], dtype=float), np.asarray(obs[3:6], dtype=float)
    rng = float(np.linalg.norm(r))
    closing = -float(r @ v) / rng if rng > 0.0 else float(np.linalg.norm(v))
    lines = [
        f"distance to the target = {rng:.2f} m, closing speed = {closing:.4f} m/s, "
        f"speed = {float(np.linalg.norm(v)):.4f} m/s"
    ]
    mode = target_mode(goal.get("target", {"mode": "passive"}))
    turn = mode.matrix(t, view.n) @ mode.matrix(0.0, view.n).T
    turning = mode.rate_bound(view.n) > 0.0
    if task in ("hold", "berth") and "hold_point" in goal:
        hold = np.asarray(goal["hold_point"], dtype=float)
        lines.append(f"distance to the hold point = {float(np.linalg.norm(r - hold)):.2f} m")
    if task in ("dock", "depart") and "port" in goal:
        port = turn @ np.asarray(goal["port"], dtype=float)
        axis = turn @ np.asarray(goal["port_axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        offset = r - port
        depth = float(offset @ axis)
        lateral = float(np.linalg.norm(offset - depth * axis))
        axial = float(v @ axis)
        sideways = float(np.linalg.norm(v - axial * axis))
        if turning:
            lines.append(f"port now at {_vec(port)} m, axis now {_vec(axis, 4)}")
        lines.append(
            f"distance along the port axis = {depth:.3f} m, lateral offset from the axis = "
            f"{lateral:.3f} m"
        )
        toward = "toward the port" if task == "dock" else "outward"
        speed = -axial if task == "dock" else axial
        lines.append(
            f"speed along the axis {toward} = {speed:.4f} m/s, lateral speed = {sideways:.4f} m/s"
        )
    if task == "depart" and "hold_point" in goal:
        clear = np.asarray(goal["hold_point"], dtype=float)
        lines.append(f"distance to the clear point = {float(np.linalg.norm(r - clear)):.2f} m")
    if task == "inspect":
        if "coverage" in info:
            lines.append(f"points imaged so far = {100.0 * float(info['coverage']):.1f}% of all")
        if "sun_direction" in goal:
            lines.append(f"Sun direction now = {_vec(sun_at(goal['sun_direction'], view.n, t), 3)}")
    if turning:
        for spec in view.constraints:
            if spec["type"].startswith("approach_cone") and _body_fixed(spec):
                apex = mode.matrix(t, view.n) @ np.asarray(spec["apex"], dtype=float)
                axis = mode.matrix(t, view.n) @ np.asarray(spec["axis"], dtype=float)
                lines.append(f"corridor now: apex {_vec(apex)} m, axis {_vec(axis, 4)}")
    for a, b in goal.get("thrust_inhibit", ()):
        if a <= t < b:
            lines.append(f"thrusters inhibited until t = {_num(b)} s")
    return "\n".join(f"  {line}" for line in lines)


__all__ = ["TASK_KEYS", "briefing", "check_goal", "situation", "target_mode"]
