"""Departure: leave the port through a departure corridor onto a non-returning path.

Success at a step boundary when the range exceeds `clear_range` and the free drift from that
state, over `drift_horizon`, never comes within `collision_radius` of the target, checked with
the Clohessy-Wiltshire transition matrix. The departure corridor is a cone on the port axis,
active near the port. The public goal names the clear point on the axis as a hold point with a
wide capture ball, so the mission metrics score the arrival in the clear zone; the drift check
is the scenario's own and is reported in `info`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import brentq

from ..constraints import ApproachCone, Constraint, KeepOut
from ..dynamics import cw_discrete, cw_stm
from ..metrics.settings import MetricSettings
from .base import StepRecord
from .scenario import ScenarioEnv, ScenarioParams
from .targets import body_point, briefed_mode


@dataclass(frozen=True, kw_only=True)
class DepartureParams(ScenarioParams):
    """Parameters of a departure scenario (m, m/s, s, rad); body axes at zero attitude.

    Parameters
    ----------
    port : tuple of float
        Port position (m), body axes.
    port_axis : tuple of float
        Departure axis from the port outward, body axes.
    body_radius : float
        Radius of the target's keep-out sphere and of the drift check (m), positive.
    cone : tuple of float, optional
        (half-angle rad, active range m) of the departure corridor with its apex at the port.
    clear_range : float
        Range from the target the chaser must reach (m), positive.
    clear_radius : float
        Radius of the capture ball around the clear point in the public goal (m), positive.
    drift_horizon : float, optional
        Horizon of the free-drift check (s); None for two orbits.
    plume_range : float, optional
        Range within which thrust is scored for plume impingement (m).
    terminate_on_violation : bool
        Whether a hard-constraint violation ends the episode.
    thrust_inhibit, soft
        As in `ScenarioParams`.

    Raises
    ------
    ValueError
        If `body_radius`, `clear_range` or `clear_radius` is not positive.
    """

    port: tuple[float, float, float] = (0.0, 30.0, 0.0)
    port_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    body_radius: float = 30.0
    cone: tuple[float, float] | None = (math.radians(10.0), 122.0)
    clear_range: float = 610.0
    clear_radius: float = 200.0
    drift_horizon: float | None = None
    plume_range: float | None = None
    terminate_on_violation: bool = False

    def __post_init__(self):
        super().__post_init__()
        if min(self.body_radius, self.clear_range, self.clear_radius) <= 0.0:
            raise ValueError("body_radius, clear_range and clear_radius must be positive")
        object.__setattr__(self, "port", tuple(float(c) for c in self.port))
        object.__setattr__(self, "port_axis", tuple(float(c) for c in self.port_axis))


def drift_minimum_range(state: np.ndarray, n: float, horizon: float, samples: int = 4000) -> float:
    """Least range (m) of the free Clohessy-Wiltshire drift from `state` over `horizon` s.

    The drift is sampled on a uniform grid and an interior minimum is refined by root-finding
    on the radial rate r . v.

    Parameters
    ----------
    state : numpy.ndarray
        Relative state (m, m/s), Hill axes, shape (6,).
    n : float
        Mean motion (rad/s).
    horizon : float
        Drift duration (s).
    samples : int
        Number of grid points, endpoints included.

    Returns
    -------
    float
        Least range from the target (m).
    """
    state = np.asarray(state, dtype=float)
    grid = np.linspace(0.0, horizon, samples)
    path = cw_discrete(n, grid)[0] @ state
    ranges = np.linalg.norm(path[:, :3], axis=1)
    k = int(np.argmin(ranges))
    if k in (0, samples - 1):
        return float(ranges[k])

    def radial_rate(t):
        s = cw_stm(n, t) @ state
        return float(s[:3] @ s[3:])

    lo, hi = grid[k - 1], grid[k + 1]
    if radial_rate(lo) < 0.0 < radial_rate(hi):
        t = brentq(radial_rate, lo, hi, xtol=1e-9)
        return float(np.linalg.norm((cw_stm(n, t) @ state)[:3]))
    return float(ranges[k])


class DepartureEnv(ScenarioEnv):
    """Leave the port through a departure corridor onto a non-returning path.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec; its parameters are `DepartureParams`.
    config : mapping, optional
        `EnvConfig` field overrides (SI units).

    Raises
    ------
    ValueError
        If the parameters are invalid or a `thrust_inhibit` edge is not a whole step.
    """

    params_type: ClassVar[type] = DepartureParams
    params: DepartureParams

    def __init__(self, spec, config=None):
        super().__init__(spec, config)
        self.axis_body = np.array(self.params.port_axis) / np.linalg.norm(self.params.port_axis)
        self.clear_point = np.array(self.params.port) + self.params.clear_range * self.axis_body

    def _constraints(self) -> tuple[Constraint, ...]:
        p = self.params
        out: list[Constraint] = [KeepOut(p.body_radius)]
        if p.cone is not None:
            out.append(ApproachCone(p.cone[0], p.port_axis, p.port, p.cone[1]))
        return tuple(out)

    def _goal(self) -> dict[str, Any]:
        p = self.params
        told = briefed_mode(self.target)
        goal: dict[str, Any] = {
            **super()._goal(),
            "port": tuple(float(c) for c in body_point(told, p.port, 0.0, self.n)),
            "port_axis": tuple(float(c) for c in body_point(told, self.axis_body, 0.0, self.n)),
            "hold_point": tuple(float(c) for c in self.clear_point),
            "capture_radius": p.clear_radius,
            "capture_speed": 10.0,
            "clear_range": p.clear_range,
            "drift_horizon": self.horizon,
        }
        if p.plume_range is not None:
            goal["plume_range"] = p.plume_range
        return goal

    @property
    def horizon(self) -> float:
        """Free-drift check horizon (s): `drift_horizon`, else two orbits."""
        return self.params.drift_horizon or 2.0 * (2.0 * math.pi / self.n)

    def metric_settings(self) -> MetricSettings:
        """The metric parameters the scenario declares.

        Returns
        -------
        MetricSettings
            Collision radius and plume range (m).
        """
        return MetricSettings(
            collision_radius=self.params.body_radius, plume_range=self.params.plume_range
        )

    def _sample_start(self, rng: np.random.Generator) -> np.ndarray:
        # The start is at the port, displaced along the axis by the drawn separation and moving
        # out at the drawn speed; the port follows the target's attitude at t = 0.
        port = body_point(self.target, self.params.port, 0.0, self.n)
        axis = body_point(self.target, self.axis_body, 0.0, self.n)
        gap = float(self.draw.get("separation", 0.0))
        speed = float(self.draw.get("separation_speed", 0.0))
        return np.concatenate([port + gap * axis, speed * axis])

    def reset(self, *, seed=None, options=None):
        """Start an episode with the drift margin and violation record cleared.

        See `BaseEnv.reset`.

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
        self.success = False
        self.drift_margin: float | None = None
        obs, info = super().reset(seed=seed, options=options)
        self.violated = any(self.min_margins[name] < 0.0 for name in self._hard())
        self.violation_steps = int(self.violated)
        return obs, self._info()

    def _hard(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.constraints if c.name not in self.soft_constraints)

    def _goal_distance(self, state: np.ndarray) -> float:
        return max(self.params.clear_range - float(np.linalg.norm(np.asarray(state)[:3])), 0.0)

    def _success(self) -> bool:
        return self.success

    def _evaluate(self, record: StepRecord) -> tuple[float, bool, dict[str, Any]]:
        step_violation = any(record.segment.minima[name].margin < 0.0 for name in self._hard())
        if step_violation:
            self.violated = True
            self.violation_steps += 1
        terminated = step_violation and self.params.terminate_on_violation
        if np.linalg.norm(self.state[:3]) >= self.params.clear_range:
            least = drift_minimum_range(self.state, self.n, self.horizon)
            self.drift_margin = least - self.params.body_radius
            if self.drift_margin >= 0.0:
                self.success = not self.violated
                terminated = True
        return 0.0, terminated, {}

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        rho = float(np.linalg.norm(self.state[:3]))
        info.update(
            {
                "dist_to_hold_m": float(np.linalg.norm(self.state[:3] - self.clear_point)),
                "range_to_target_m": rho,
                "speed_mps": float(np.linalg.norm(self.state[3:])),
                "min_range_to_target_m": self.min_margins["keep_out"] + self.params.body_radius
                if "keep_out" in self.min_margins
                else rho,
                "keep_out_violated": self.violated,
                "violation_steps": self.violation_steps,
                "captured": rho >= self.params.clear_range,
                "drift_margin_m": self.drift_margin,
                "success": self.success,
            }
        )
        return info


__all__ = ["DepartureEnv", "DepartureParams", "drift_minimum_range"]
