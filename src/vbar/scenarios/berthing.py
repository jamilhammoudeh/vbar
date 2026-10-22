"""Berthing: bring the chaser to rest inside a capture box and let the arm take it.

Success when the capture predicate, |r - r_box| < capture_radius and |v| < capture_speed,
holds from a step boundary through `capture_time` of free drift: once it holds, the
environment inhibits the thrusters for the capture window (Fehse, Automated Rendezvous and
Docking of Spacecraft, 2003, section 8.1.2: thrust is inhibited at capture and the arm grapples
within one to two minutes), and the predicate must still hold on every sample until the window
ends. Released at rest 15 m below the V-bar the chaser drifts 0.41 m in 120 s (Fehse 5.3.1), so
a 0.5 m ball with a 60 s window is reachable from rest and lost with a residual speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..constraints import ApproachCone, Constraint, KeepOut
from ..metrics.settings import MetricSettings
from .base import StepRecord
from .scenario import ScenarioEnv, ScenarioParams


@dataclass(frozen=True, kw_only=True)
class BerthingParams(ScenarioParams):
    """Parameters of a berthing scenario (m, m/s, s, rad).

    Parameters
    ----------
    box : tuple of float
        r_box, the capture box centre (m), Hill axes.
    capture_radius : float
        Radius of the capture ball around the box centre (m), positive.
    capture_speed : float
        Speed below which the chaser can be captured (m/s), positive.
    capture_time : float
        Free-drift capture window (s), positive and a whole number of steps.
    keep_out : float
        Keep-out sphere radius (m), positive.
    cone : tuple of float, optional
        (half-angle rad, active range m) of the corridor toward the box.
    terminate_on_violation : bool
        Whether a hard-constraint violation ends the episode.
    thrust_inhibit, soft
        As in `ScenarioParams`.

    Raises
    ------
    ValueError
        If `capture_radius`, `capture_speed`, `capture_time` or `keep_out` is not positive.
    """

    box: tuple[float, float, float] = (-15.0, 0.0, 0.0)
    capture_radius: float = 0.5
    capture_speed: float = 0.01
    capture_time: float = 60.0
    keep_out: float = 10.0
    cone: tuple[float, float] | None = (math.radians(10.0), 50.0)
    terminate_on_violation: bool = False

    def __post_init__(self):
        super().__post_init__()
        if min(self.capture_radius, self.capture_speed, self.capture_time, self.keep_out) <= 0.0:
            raise ValueError("capture_radius, capture_speed, capture_time, keep_out are positive")
        object.__setattr__(self, "box", tuple(float(c) for c in self.box))


class BerthingEnv(ScenarioEnv):
    """Bring the chaser to rest inside a capture box and drift there through the capture window.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec; its parameters are `BerthingParams`.
    config : mapping, optional
        `EnvConfig` field overrides (SI units).

    Raises
    ------
    ValueError
        If the parameters are invalid, or `capture_time` or a `thrust_inhibit` edge is not a
        whole number of steps.
    """

    params_type: ClassVar[type] = BerthingParams
    params: BerthingParams

    def __init__(self, spec, config=None):
        super().__init__(spec, config)
        if not self._on_grid(self.params.capture_time):
            raise ValueError("capture_time must be a whole number of steps")

    @property
    def box(self) -> np.ndarray:
        """Capture box centre (m), Hill axes, shape (3,)."""
        return np.array(self.params.box)

    def _constraints(self) -> tuple[Constraint, ...]:
        p = self.params
        out: list[Constraint] = [KeepOut(p.keep_out)]
        if p.cone is not None:
            # The apex sits behind the box so the corridor is as wide as the capture ball there.
            axis = self.box / np.linalg.norm(self.box)
            setback = p.capture_radius / math.tan(p.cone[0])
            apex = tuple(self.box - setback * axis)
            out.append(ApproachCone(p.cone[0], tuple(axis), apex, p.cone[1] + setback))
        return tuple(out)

    def _goal(self) -> dict[str, Any]:
        p = self.params
        return {
            **super()._goal(),
            "task": "berth",
            "hold_point": p.box,
            "capture_radius": p.capture_radius,
            "capture_speed": p.capture_speed,
            "hold_time": p.capture_time,
            "capture_time": p.capture_time,
        }

    def metric_settings(self) -> MetricSettings:
        """The metric parameters the scenario declares.

        Returns
        -------
        MetricSettings
            The collision radius, the keep-out radius (m).
        """
        return MetricSettings(collision_radius=self.params.keep_out)

    def reset(self, *, seed=None, options=None):
        """Start an episode with the capture window and violation record cleared.

        See `BaseEnv.reset`; a start inside the capture predicate arms the window at t = 0.

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
        self.inhibit_until = -math.inf
        self.armed_at: float | None = None
        self.success = False
        obs, info = super().reset(seed=seed, options=options)
        self.violated = any(self.min_margins[name] < 0.0 for name in self._hard())
        self.violation_steps = int(self.violated)
        if bool(self._inside(self.state)):
            self.armed_at = 0.0
            self.inhibit_until = self.params.capture_time
        return obs, self._info()

    def _hard(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.constraints if c.name not in self.soft_constraints)

    def _inside(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        position = np.linalg.norm(states[..., :3] - self.box, axis=-1)
        speed = np.linalg.norm(states[..., 3:6], axis=-1)
        return (position < self.params.capture_radius) & (speed < self.params.capture_speed)

    def _limit_command(self, fraction: np.ndarray) -> np.ndarray:
        fraction = super()._limit_command(fraction)
        if self.t < self.inhibit_until - 1e-9:
            return np.zeros_like(fraction)
        return fraction

    def inhibited(self) -> bool:
        """Whether commands are being zeroed now.

        Returns
        -------
        bool
            True inside a `thrust_inhibit` window or the capture window.
        """
        return super().inhibited() or self.t < self.inhibit_until - 1e-9

    def _goal_distance(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(state)[:3] - self.box))

    def _success(self) -> bool:
        return self.success

    def _evaluate(self, record: StepRecord) -> tuple[float, bool, dict[str, Any]]:
        trace = record.segment
        step_violation = any(trace.minima[name].margin < 0.0 for name in self._hard())
        if step_violation:
            self.violated = True
            self.violation_steps += 1
        terminated = step_violation and self.params.terminate_on_violation
        if self.armed_at is not None:
            if not bool(np.all(self._inside(trace.states))):
                self.armed_at = None
                self.inhibit_until = -math.inf
            elif self.t >= self.armed_at + self.params.capture_time - 1e-9:
                self.success = not self.violated
                terminated = True
        if self.armed_at is None and bool(self._inside(self.state)):
            self.armed_at = self.t
            self.inhibit_until = self.t + self.params.capture_time
        return 0.0, terminated, {}

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        rho = float(np.linalg.norm(self.state[:3]))
        info.update(
            {
                "dist_to_hold_m": self._goal_distance(self.state),
                "range_to_target_m": rho,
                "speed_mps": float(np.linalg.norm(self.state[3:])),
                "min_range_to_target_m": self.min_margins["keep_out"] + self.params.keep_out
                if "keep_out" in self.min_margins
                else rho,
                "keep_out_violated": self.violated,
                "violation_steps": self.violation_steps,
                "captured": bool(self._inside(self.state)),
                "armed": self.armed_at is not None,
                "inhibited": self.inhibited(),
                "success": self.success,
            }
        )
        return info


__all__ = ["BerthingEnv", "BerthingParams"]
