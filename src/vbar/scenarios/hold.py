"""Hold-point scenarios: reach a point near the target and hold it for a dwell.

The goal is a hold point with a capture ball and a dwell: success when the capture predicate,
|r - r_hold| < capture_radius and |v| < capture_speed, holds without a break for `hold_time`
(Fehse, Automated Rendezvous and Docking of Spacecraft, 2003, section 3.3.3: a state inside
such a box leaves it within a fraction of an orbit unless it is controlled, so only a dwell is
a hold). Constraints come from the parameters: a keep-out sphere, a closing-speed law (the
linear form or a table of range-rate gates), the passive-speed envelope, an approach
ellipsoid that may be entered only after a GO time, and an approach cone fixed to the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..constraints import (
    ClosingSpeedLimit,
    Constraint,
    GateTable,
    KeepOut,
    PassiveSpeed,
    SpeedLimit,
    ZoneEntry,
)
from ..metrics.settings import MetricSettings
from .base import StepRecord
from .scenario import ScenarioEnv, ScenarioParams
from .targets import RotatingCone


@dataclass(frozen=True, kw_only=True)
class HoldParams(ScenarioParams):
    """Parameters of a hold-point scenario (m, m/s, s, rad).

    `keep_out` is the keep-out sphere's radius (0 for none). `closing_limit` is (v0, v1) of the
    linear closing-speed law, `gates` a table of (range, closing speed) pairs, `passive_axis`
    "vbar" or "rbar" for the passive-speed envelope with the keep-out radius as the structure
    extent, and `full_speed_limit` (v0, v1) a limit on the whole relative speed. `zone` gives
    the semi-axes of an approach ellipsoid closed until the episode's drawn `go_time`.
    `cone` is (half-angle, active range) of a corridor whose apex is the target's centre and
    whose axis points from the centre toward the hold point, turning with the target.
    `waypoints` are informational. `plume_range`, `passive_exclusion` and `cam_impulses` (the
    abort set, needed where the hold point is off the orbit plane) are declared for the metrics.

    Parameters
    ----------
    hold_point : tuple of float
        r_hold, the hold point (m), Hill axes.
    capture_radius : float
        Radius of the capture ball around the hold point (m), positive.
    capture_speed : float
        Speed below which the chaser counts as held (m/s), positive.
    hold_time : float
        Dwell the capture predicate must hold without a break (s), not negative.
    keep_out : float
        Keep-out sphere radius (m); 0 for none.
    closing_limit : tuple of float, optional
        (v0 m/s, v1 1/s) of the closing-speed law v0 + v1 rho.
    gates : tuple of tuple of float, optional
        (range m, closing speed m/s) gates.
    passive_axis : {"vbar", "rbar"}, optional
        Axis of the passive-speed envelope.
    full_speed_limit : tuple of float, optional
        (v0 m/s, v1 1/s) of a limit v0 + v1 rho on the whole relative speed.
    zone : tuple of float, optional
        Semi-axes of the approach ellipsoid (m), Hill axes.
    cone : tuple of float, optional
        (half-angle rad, active range m) of the corridor toward the hold point.
    waypoints : tuple of tuple of float
        Informational waypoints (m), Hill axes.
    plume_range : float, optional
        Range within which thrust is scored for plume impingement (m).
    passive_exclusion : float
        Range from the target inside which passive safety is not evaluated (m).
    cam_impulses : tuple of tuple of float, optional
        The abort set, delta-v vectors (m/s), Hill axes.
    terminate_on_violation : bool
        Whether a hard-constraint violation ends the episode.
    thrust_inhibit, soft
        As in `ScenarioParams`.

    Raises
    ------
    ValueError
        If the capture limits are not positive, `hold_time` or `keep_out` is negative, or
        `passive_axis` is not None, "vbar" or "rbar".
    """

    hold_point: tuple[float, float, float]
    capture_radius: float = 5.0
    capture_speed: float = 0.05
    hold_time: float = 600.0
    keep_out: float = 50.0
    closing_limit: tuple[float, float] | None = None
    gates: tuple[tuple[float, float], ...] | None = None
    passive_axis: str | None = None
    full_speed_limit: tuple[float, float] | None = None
    zone: tuple[float, float, float] | None = None
    cone: tuple[float, float] | None = None
    waypoints: tuple[tuple[float, float, float], ...] = ()
    plume_range: float | None = None
    passive_exclusion: float = 0.0
    cam_impulses: tuple[tuple[float, float, float], ...] | None = None
    terminate_on_violation: bool = False

    def __post_init__(self):
        super().__post_init__()
        if min(self.capture_radius, self.capture_speed) <= 0.0 or self.hold_time < 0.0:
            raise ValueError("capture_radius and capture_speed are positive, hold_time >= 0")
        if self.keep_out < 0.0:
            raise ValueError("keep_out must not be negative")
        if self.passive_axis not in (None, "vbar", "rbar"):
            raise ValueError("passive_axis is None, 'vbar' or 'rbar'")
        object.__setattr__(self, "hold_point", tuple(float(c) for c in self.hold_point))


class HoldPointEnv(ScenarioEnv):
    """Reach a hold point near the target and hold it for a dwell.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec; its parameters are `HoldParams`.
    config : mapping, optional
        `EnvConfig` field overrides (SI units).

    Raises
    ------
    ValueError
        If the parameters are invalid or a `thrust_inhibit` edge is not a whole step.
    """

    params_type: ClassVar[type] = HoldParams
    params: HoldParams

    def __init__(self, spec, config=None):
        super().__init__(spec, config)
        self.hold_point = np.array(self.params.hold_point)

    def _constraints(self) -> tuple[Constraint, ...]:
        p = self.params
        out: list[Constraint] = []
        if p.keep_out > 0.0:
            out.append(KeepOut(p.keep_out))
        if p.closing_limit is not None:
            out.append(ClosingSpeedLimit(*p.closing_limit))
        if p.gates is not None:
            out.append(GateTable(p.gates))
        if p.passive_axis is not None:
            out.append(PassiveSpeed(p.passive_axis, p.keep_out, self.n))
        if p.full_speed_limit is not None:
            out.append(SpeedLimit(*p.full_speed_limit))
        if p.zone is not None:
            out.append(ZoneEntry(p.zone, go_time=float(self.draw.get("go_time", math.inf))))
        if p.cone is not None:
            axis = np.array(p.hold_point) / np.linalg.norm(p.hold_point)
            out.append(
                RotatingCone(
                    p.cone[0], tuple(axis), (0.0, 0.0, 0.0), p.cone[1], self.target, self.n
                )
            )
        return tuple(out)

    def _goal(self) -> dict[str, Any]:
        p = self.params
        goal: dict[str, Any] = {
            **super()._goal(),
            "task": "hold",
            "hold_point": p.hold_point,
            "capture_radius": p.capture_radius,
            "capture_speed": p.capture_speed,
            "hold_time": p.hold_time,
            "passive_exclusion": p.passive_exclusion,
        }
        if p.plume_range is not None:
            goal["plume_range"] = p.plume_range
        if p.waypoints:
            goal["waypoints"] = p.waypoints
        return goal

    def metric_settings(self) -> MetricSettings:
        """The metric parameters the scenario declares.

        Returns
        -------
        MetricSettings
            Passive exclusion (m), plume range (m) and abort set (m/s, Hill axes).
        """
        return MetricSettings(
            passive_exclusion=self.params.passive_exclusion,
            plume_range=self.params.plume_range,
            cam_impulses=self.params.cam_impulses,
        )

    def reset(self, *, seed=None, options=None):
        """Start an episode with the dwell and violation record cleared (`BaseEnv.reset`).

        Parameters
        ----------
        seed : int, optional
            Episode seed.
        options : dict, optional
            Training-only reset options.

        Returns
        -------
        tuple
            The first observation and the info dict.

        Raises
        ------
        ValueError
            If options are given outside training or are unknown.
        """
        self.violated = False
        self.violation_steps = 0
        self.held_since: float | None = None
        self.dwell = 0.0
        self.success = False
        obs, info = super().reset(seed=seed, options=options)
        self.violated = any(self.min_margins[name] < 0.0 for name in self._hard())
        self.violation_steps = int(self.violated)
        return obs, self._info()

    def _hard(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.constraints if c.name not in self.soft_constraints)

    def _inside(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        position = np.linalg.norm(states[..., :3] - self.hold_point, axis=-1)
        speed = np.linalg.norm(states[..., 3:6], axis=-1)
        return (position < self.params.capture_radius) & (speed < self.params.capture_speed)

    def _goal_distance(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(state)[:3] - self.hold_point))

    def _success(self) -> bool:
        return self.success

    def _evaluate(self, record: StepRecord) -> tuple[float, bool, dict[str, Any]]:
        trace = record.segment
        step_violation = any(trace.minima[name].margin < 0.0 for name in self._hard())
        if step_violation:
            self.violated = True
            self.violation_steps += 1
        # The predicate is checked on every sample of the step, so a dwell is continuous to the
        # sampling resolution; the metrics settle it in continuous time.
        if bool(np.all(self._inside(trace.states))) and bool(self._inside(self.state)):
            if self.held_since is None:
                self.held_since = float(trace.times[0])
            self.dwell = self.t - self.held_since
        elif bool(self._inside(self.state)):
            self.held_since = self.t
            self.dwell = 0.0
        else:
            self.held_since = None
            self.dwell = 0.0
        held = self.held_since is not None and self.dwell >= self.params.hold_time - 1e-9
        self.success = held and not self.violated
        terminated = held or (step_violation and self.params.terminate_on_violation)
        return 0.0, terminated, {}

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        rho = float(np.linalg.norm(self.state[:3]))
        keep_out = self.min_margins.get("keep_out")
        info.update(
            {
                "dist_to_hold_m": self._goal_distance(self.state),
                "range_to_target_m": rho,
                "speed_mps": float(np.linalg.norm(self.state[3:])),
                "min_range_to_target_m": rho
                if keep_out is None
                else keep_out + self.params.keep_out,
                "keep_out_violated": self.violated,
                "violation_steps": self.violation_steps,
                "captured": bool(self._inside(self.state)),
                "dwell_s": self.dwell,
                "success": self.success,
                "inhibited": self.inhibited(),
            }
        )
        return info


__all__ = ["HoldParams", "HoldPointEnv"]
