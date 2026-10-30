"""Replay of a recorded episode in Rerun (needs pip install "vbar[replay]").

`export_rrd` writes an `Episode` from `vbar.metrics.record_episode` to a `.rrd` file that the Rerun
viewer opens (`rerun episode.rrd`). Everything in the file is read from the step records and the
scenario's public description; nothing is simulated again. Positions and times are in the target's
Hill frame, in SI units (m, m/s, m/s^2, s), with x radial, y along-track and z orbit normal.

The viewer draws positions as 32-bit floats. The same values are also stored as 64-bit
components on the same entities (`position_m`, `velocity_mps`, `accel_mps2`), so a file reads
back exactly what the record holds.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .agents.base import command_fraction
from .constraints import ApproachCone, KeepOut
from .conventions import G0
from .metrics.episode import Episode
from .metrics.evaluate import evaluate
from .metrics.filter import _active, _flown_change
from .metrics.registry import METRICS
from .metrics.settings import MetricSettings
from .viz import INK_SECONDARY, KEEP_OUT_FILL, SERIES_COLORS

__all__ = ["APPLICATION_ID", "export_rrd"]

logger = logging.getLogger(__name__)

APPLICATION_ID = "vbar"
_INSTALL = 'pip install "vbar[replay]"'
# Violation intervals closer than this (s) are one interval, as in the metrics.
_JOIN_TOL = 1e-9
# Fraction of the episode's greatest range that a full command is drawn as.
_ARROW_FRACTION = 0.1
_RING_POINTS = 64
_RINGS = (1 / 3, 2 / 3, 1.0)
_RULINGS = 8


def _rerun():
    try:
        import rerun
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(f"replay export needs {err.name}: {_INSTALL}") from err
    return rerun


def _rgba(color: str) -> tuple[int, int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16), 255


def _joined_violations(ep: Episode) -> dict[str, list[tuple[float, float]]]:
    """Each constraint's intervals (s) of negative margin, joined where steps touch."""
    out: dict[str, list[tuple[float, float]]] = {c.name: [] for c in ep.constraints}
    for rec in ep.records:
        for name, spans in rec.segment.violations.items():
            joined = out.setdefault(name, [])
            for lo, hi in spans:
                if joined and lo <= joined[-1][1] + _JOIN_TOL:
                    joined[-1] = (joined[-1][0], max(joined[-1][1], hi))
                else:
                    joined.append((lo, hi))
    return out


def _cone_strips(cone: ApproachCone, reach: float) -> list[np.ndarray]:
    """Rings and rulings that outline a corridor out to slant length `reach` (m) from its apex."""
    axis = np.asarray(cone.axis, dtype=float)
    axis /= np.linalg.norm(axis)
    apex = np.asarray(cone.apex, dtype=float)
    across = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, across)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    slant = cone.active_range if math.isfinite(cone.active_range) else reach
    theta = np.linspace(0.0, 2.0 * np.pi, _RING_POINTS + 1)
    circle = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)

    def ring(fraction: float) -> np.ndarray:
        s = fraction * slant
        return apex + s * math.cos(cone.half_angle) * axis + s * math.sin(cone.half_angle) * circle

    rim = ring(1.0)
    rulings = [np.array([apex, rim[i]]) for i in range(0, _RING_POINTS, _RING_POINTS // _RULINGS)]
    return [ring(f) for f in _RINGS] + rulings


def _numeric_leaves(card: Mapping[str, Any]) -> Iterator[tuple[tuple[str, ...], float]]:
    """Finite numbers in a metric card as (key path, value); dictionaries nest, labels drop out."""
    for key, value in card.items():
        if isinstance(value, Mapping):
            for sub, number in _numeric_leaves(value):
                yield (str(key), *sub), number
        elif isinstance(value, bool | int | float | np.number) and np.isfinite(value):
            yield (str(key),), float(value)


def _series_label(parts: tuple[str, ...]) -> str:
    """The legend entry of a metric series: identifier, name, constraint and unit."""
    info = METRICS.get(parts[0])
    if info is None:
        return "/".join(parts)
    label = f"{info.id} {'/'.join(parts)}"
    # A margin has the unit of its constraint, which the catalogue cannot name, and a dictionary
    # metric's entries have their own.
    return label if info.units in ("per constraint", "dict") else f"{label} ({info.units})"


def _entity(rr, *parts: str) -> str:
    """An entity path from parts, each escaped so a constraint name cannot add or change levels."""
    return "/".join(rr.escape_entity_path_part(part) for part in parts)


def _running_card(ep: Episode) -> dict[tuple[str, ...], np.ndarray]:
    """Metric values over the first k steps for k = 1 .. steps, by the card's own names.

    Each series follows the definition of the metric in `vbar.metrics`, accumulated step by step,
    so its last value equals the full card's and each earlier value the metric's on that prefix
    of the record. Evaluating the metric on every prefix of the record gives the same numbers
    and takes time quadratic in the steps.
    """
    series: dict[tuple[str, ...], list[float]] = {}

    def add(name: str, value: float) -> None:
        series.setdefault((name,), []).append(float(value))

    view = ep.view
    box = view.thrust_limit == "box"
    # Python's sum compensates float rounding, so summing the lists each step gives the
    # metric's own value; a running total would differ in the last digit.
    dv_l2: list[float] = []
    dv_l1: list[float] = []
    on_time: list[float] = []
    bursts = changes = 0
    was_thrusting = False
    last_sign: list[float | None] = [None, None, None]
    decisions = acting = 0
    sizes: list[float] = []
    for rec in ep.records:
        thrusting = False
        for duration, accel in rec.actuation.segments:
            a = np.asarray(accel, dtype=float)
            if np.any(a):
                thrusting = True
                on_time.append(float(duration))
            for axis in range(3):
                if a[axis] != 0.0:
                    sign = float(np.sign(a[axis]))
                    changes += last_sign[axis] is not None and sign != last_sign[axis]
                    last_sign[axis] = sign
        dv_l2.append(rec.actuation.dv_l2)
        dv_l1.append(rec.actuation.dv_l1)
        bursts += thrusting and not was_thrusting
        was_thrusting = thrusting
        total_l2, total_l1 = sum(dv_l2), sum(dv_l1)
        add("delta_v", total_l2)
        add("delta_v_axes", total_l1)
        if view.mass_kg is not None:
            spent = total_l1 if box else total_l2
            add("propellant_mass", -view.mass_kg * math.expm1(-spent / (ep.isp * G0)))
        add("burn_count", bursts)
        add("duty_cycle", sum(on_time) / rec.t if rec.t > 0.0 else 0.0)
        add("chatter", changes / rec.t if rec.t > 0.0 else 0.0)
        if rec.filter_result is not None:
            decisions += 1
            if _active(ep, rec):
                acting += 1
                sizes.append(_flown_change(ep, rec))
            add("intervention_rate", acting / decisions)
            add("intervention_size", float(np.mean(sizes)) if sizes else math.nan)

    spans = _joined_violations(ep)
    ends = np.array([rec.t for rec in ep.records])
    for c in ep.constraints:
        minima = [rec.segment.minima.get(c.name) for rec in ep.records]
        if any(m is None for m in minima):
            continue
        margins = [float(c.margin(ep.start, 0.0)), *(m.margin for m in minima if m is not None)]
        lowest = np.minimum.accumulate(margins)[1:]
        series["min_margin", c.name] = list(lowest)
        series["violation_depth", c.name] = list(np.maximum(0.0, -lowest))
        series["violation_duration", c.name] = [
            sum(max(0.0, min(hi, t) - lo) for lo, hi in spans[c.name]) for t in ends
        ]
    return {key: np.array(values) for key, values in series.items() if np.isfinite(values).any()}


def export_rrd(
    ep: Episode,
    path: str | PathLike[str],
    *,
    card: Mapping[str, Any] | None = None,
    settings: MetricSettings | None = None,
    arrow_m: float | None = None,
) -> Path:
    """Write a recorded episode as a Rerun recording.

    The file holds, on the timeline `sim_time` (episode seconds, with `step` the decision index):
    the chaser's position at every decision (`hill/chaser`) with a trail through its recorded
    states, the target, the keep-out spheres and approach corridors of the scenario, the hold
    point or port of the goal, the agent's command, the filtered command and the flown
    acceleration as arrows at the decision that issued them, keep-out and other constraint
    violations and safety-filter interventions as events, and the metric card as scalars.

    Parameters
    ----------
    ep : Episode
        A recorded episode (`vbar.metrics.record_episode`). It is only read.
    path : str or path-like
        The `.rrd` file to write; an existing file is replaced.
    card : mapping, optional
        The episode's metrics as `vbar.metrics.evaluate` returns them. Computed with `settings`
        when not given.
    settings : MetricSettings, optional
        The scenario's declared metric settings, used when the card is computed. Default settings
        when not given.
    arrow_m : float, optional
        Length (m) that a full command is drawn as. Default 0.1 times the greatest range of the
        chaser from the target over the episode. Only the drawing depends on it; the stored
        `accel_mps2` values are in m/s^2.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    ModuleNotFoundError
        If the `replay` extra is not installed; the message gives the install command.
    ValueError
        If the episode has no steps or `arrow_m` is not a positive finite number.

    Notes
    -----
    Constraints that move with a turning target are not drawn, and the other constraint types
    appear only through their violation events and margins.
    """
    rr = _rerun()
    if not ep.records:
        raise ValueError("the episode has no steps")
    if arrow_m is not None and not (math.isfinite(arrow_m) and arrow_m > 0.0):
        raise ValueError("arrow_m must be a positive finite number")
    settings = settings or MetricSettings()
    card = evaluate(ep, settings) if card is None else card

    states, times = ep.states, ep.times
    view = ep.view
    ranges = np.linalg.norm(states[:, :3], axis=1)
    reach = max(float(ranges.max()), 1.0)
    arrow_m = _ARROW_FRACTION * reach if arrow_m is None else float(arrow_m)

    target = Path(path)
    stream = rr.RecordingStream(APPLICATION_ID)
    stream.save(target)
    stream.set_log_time_enabled(False)
    try:
        _log_scene(stream, rr, ep, states, reach)
        _log_chaser(stream, rr, ep, states, times, arrow_m)
        _log_events(stream, rr, ep, times)
        _log_card(stream, rr, ep, times, card)
    finally:
        stream.disconnect()
    logger.info("wrote %d steps of %s to %s", len(ep.records), view.scenario_id, target)
    return target


def _log_scene(stream, rr, ep: Episode, states: np.ndarray, reach: float) -> None:
    """Static entities: frame, target, constraints, goal and a description of the episode."""
    view = ep.view
    stream.log("hill", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    stream.log(
        "hill/target",
        rr.Points3D([[0.0, 0.0, 0.0]], labels=["target"], radii=rr.Radius.ui_points(7.0)),
        static=True,
    )
    for c in ep.constraints:
        if isinstance(c, KeepOut) and c.radius > 0.0:
            stream.log(
                _entity(rr, "hill", "constraints", c.name),
                rr.Ellipsoids3D(
                    centers=[[0.0, 0.0, 0.0]],
                    half_sizes=[[c.radius] * 3],
                    colors=[_rgba(KEEP_OUT_FILL)],
                    line_radii=rr.Radius.ui_points(1.0),
                    fill_mode=rr.components.FillMode.DenseWireframe,
                    labels=[f"{c.name} {c.radius:g} m"],
                ),
                rr.AnyValues(radius_m=[c.radius]),
                static=True,
            )
        elif isinstance(c, ApproachCone):
            apex_reach = float(np.linalg.norm(states[:, :3] - np.asarray(c.apex), axis=1).max())
            stream.log(
                _entity(rr, "hill", "constraints", c.name),
                rr.LineStrips3D(
                    _cone_strips(c, max(apex_reach, reach)),
                    colors=[_rgba(INK_SECONDARY)],
                    radii=rr.Radius.ui_points(0.8),
                ),
                rr.AnyValues(
                    half_angle_rad=[c.half_angle],
                    apex_m=[np.asarray(c.apex, dtype=float)],
                    axis=[np.asarray(c.axis, dtype=float)],
                    active_range_m=[c.active_range],
                ),
                static=True,
            )
        else:
            logger.debug("constraint %s is not drawn", c.name)
    for key, label in (("hold_point", "hold point"), ("port", "port")):
        point = view.goal.get(key)
        if point is not None:
            stream.log(
                f"hill/goal/{key}",
                rr.Points3D([point], labels=[label], radii=rr.Radius.ui_points(5.0)),
                rr.AnyValues(position_m=[np.asarray(point, dtype=float)]),
                static=True,
            )
    description = "\n".join(
        [
            f"scenario: {view.scenario_id}",
            f"dynamics: {ep.dynamics}, body: {ep.body}, altitude: {ep.altitude_m:g} m",
            f"step: {view.dt:g} s, action mode: {view.action_mode}",
            "frame: Hill, x radial, y along-track, z orbit normal; SI units",
            f"vbar {__version__}",
        ]
    )
    stream.log("meta/episode", rr.TextDocument(description), static=True)


def _log_chaser(stream, rr, ep: Episode, states: np.ndarray, times: np.ndarray, arrow_m) -> None:
    """The chaser's states, trail and per-decision arrows, one row per decision."""
    view = ep.view
    has_filter = any(rec.filter_result is not None for rec in ep.records)
    blue = _rgba(SERIES_COLORS[0])
    steps = len(ep.records)
    for k in range(steps + 1):
        stream.set_time("sim_time", duration=float(times[k]))
        stream.set_time("step", sequence=k)
        r, v = states[k, :3], states[k, 3:]
        stream.log(
            "hill/chaser",
            rr.Points3D([r], labels=["chaser"], colors=[blue], radii=rr.Radius.ui_points(5.0)),
            rr.AnyValues(position_m=[r], velocity_mps=[v]),
        )
        if k > 0:
            stream.log(
                "hill/chaser/trail",
                rr.LineStrips3D(
                    [states[: k + 1, :3]], colors=[blue], radii=rr.Radius.ui_points(1.5)
                ),
            )
        if k == steps:
            continue
        rec = ep.records[k]
        commanded = command_fraction(view, rec.command)
        stream.log(
            "hill/chaser/command",
            rr.Arrows3D(
                origins=[r],
                vectors=[commanded * arrow_m],
                colors=[_rgba(SERIES_COLORS[1])],
                radii=rr.Radius.ui_points(1.5),
            ),
            rr.AnyValues(accel_mps2=[commanded * view.max_accel]),
        )
        if has_filter:
            filtered = command_fraction(view, rec.filtered)
            stream.log(
                "hill/chaser/filtered",
                rr.Arrows3D(
                    origins=[r],
                    vectors=[filtered * arrow_m],
                    colors=[_rgba(SERIES_COLORS[2])],
                    radii=rr.Radius.ui_points(1.5),
                ),
                rr.AnyValues(accel_mps2=[filtered * view.max_accel]),
            )
        dv = sum((d * np.asarray(a, dtype=float) for d, a in rec.actuation.segments), np.zeros(3))
        span = sum(d for d, _ in rec.actuation.segments)
        flown = dv / span if span > 0.0 else np.zeros(3)
        stream.log(
            "hill/chaser/flown",
            rr.Arrows3D(
                origins=[r],
                vectors=[flown / view.max_accel * arrow_m],
                colors=[_rgba(INK_SECONDARY)],
                radii=rr.Radius.ui_points(1.0),
            ),
            rr.AnyValues(accel_mps2=[flown], dv_mps=[dv]),
        )
    stream.reset_time()
    columns = rr.TimeColumn("sim_time", duration=times)
    stream.send_columns(
        "state/range_m",
        [columns],
        rr.Scalars.columns(scalars=np.linalg.norm(states[:, :3], axis=1)),
    )
    stream.send_columns(
        "state/speed_mps",
        [columns],
        rr.Scalars.columns(scalars=np.linalg.norm(states[:, 3:], axis=1)),
    )
    for name, label in (("state/range_m", "range (m)"), ("state/speed_mps", "speed (m/s)")):
        stream.log(name, rr.SeriesLines(names=[label]), static=True)


def _log_events(stream, rr, ep: Episode, times: np.ndarray) -> None:
    """Violations and filter decisions as timed events."""
    for name, spans in _joined_violations(ep).items():
        level = "ERROR" if name in ep.hard else "WARN"
        for lo, hi in spans:
            stream.set_time("sim_time", duration=lo)
            stream.log(
                _entity(rr, "events", "violation", name),
                rr.TextLog(f"{name} violated for {hi - lo:.3g} s", level=level),
                rr.AnyValues(start_s=[lo], end_s=[hi]),
            )
    for k, rec in enumerate(ep.records):
        result = rec.filter_result
        if result is None:
            continue
        stream.set_time("sim_time", duration=float(times[k]))
        stream.set_time("step", sequence=k)
        if _active(ep, rec):
            change = _flown_change(ep, rec)
            stream.log(
                "events/filter/intervention",
                rr.TextLog(f"filter changed the flown acceleration by {change:.3g} m/s^2"),
                rr.AnyValues(change_mps2=[change], status=[result.status]),
            )
        if result.status != "feasible":
            stream.log(
                "events/filter/status",
                rr.TextLog(f"filter status {result.status}", level="ERROR"),
                rr.AnyValues(status=[result.status], conflicting=[list(result.conflicting)]),
            )


def _log_card(stream, rr, ep: Episode, times: np.ndarray, card) -> None:
    """The metric card: running values at every step end, then the final values at the last."""
    for parts, values in _running_card(ep).items():
        path = _entity(rr, "card", *parts)
        stream.send_columns(
            path,
            [rr.TimeColumn("sim_time", duration=times[1:])],
            rr.Scalars.columns(scalars=values),
        )
        stream.log(path, rr.SeriesLines(names=[_series_label(parts)]), static=True)
    end = float(times[-1])
    stream.set_time("sim_time", duration=end)
    stream.set_time("step", sequence=len(ep.records))
    for parts, value in _numeric_leaves(card):
        path = _entity(rr, "card", "final", *parts)
        stream.log(path, rr.Scalars(value))
        stream.log(path, rr.SeriesLines(names=[_series_label(parts)]), static=True)
    # A card without a verdict (the caller passed {}) gets no outcome event.
    if "success" in card:
        outcome = "success" if card["success"] else f"not successful: {card.get('failure_label')}"
        stream.log("events/outcome", rr.TextLog(outcome, level="INFO"))
