"""Inspection: image the target's surface points under lighting constraints.

The target carries `points` surface points on a sphere (a Fibonacci lattice, equal weights).
A point with outward normal n_i at p_i is inspected at time t when the chaser faces it,
n_i . (r - p_i) > 0, the Sun lights it, n_i . s(t) > 0 with the target outside the Earth's
shadow, the range is within `max_range`, and the Sun is farther than `sun_exclusion` from the
camera's line of sight. The Sun is fixed in inertial space, so it turns at -n about z in the
Hill frame; at beta = 0 and 500 km the target is in eclipse 38% of each orbit (Montenbruck and
Gill, Satellite Orbits, section 3.4). Points whose normal never faces the Sun during the
episode cannot be inspected by anyone, so success is a coverage of `coverage` of the points
lit at some time in the episode; both fractions are reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..constraints import Constraint, KeepOut
from ..dynamics import get_body
from ..metrics.settings import MetricSettings
from .base import StepRecord
from .scenario import ScenarioEnv, ScenarioParams
from .targets import sun_at, sun_direction, sunlit


def fibonacci_sphere(count: int) -> np.ndarray:
    """`count` unit vectors spread evenly over the sphere (the Fibonacci lattice).

    Parameters
    ----------
    count : int
        Number of points.

    Returns
    -------
    numpy.ndarray
        Unit vectors, shape (count, 3).
    """
    k = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (k + 0.5) / count
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    angle = math.pi * (3.0 - math.sqrt(5.0)) * k
    return np.stack([radius * np.cos(angle), radius * np.sin(angle), z], axis=1)


@dataclass(frozen=True, kw_only=True)
class InspectionParams(ScenarioParams):
    """Parameters of an inspection scenario (m, rad).

    Parameters
    ----------
    points : int
        Number of surface points, at least 1.
    target_radius : float
        Radius of the sphere carrying the points (m), positive.
    max_range : float
        Greatest range at which a point is inspected (m), positive.
    sun_exclusion : float
        Least angle between the Sun and the camera's line of sight (rad).
    keep_out : float
        Keep-out sphere radius (m), positive.
    coverage : float
        Fraction of the inspectable points needed for success, in (0, 1].
    lit_resolution : float
        Sampling interval of the lighting check that finds the inspectable points (s).
    thrust_inhibit, soft
        As in `ScenarioParams`.

    Raises
    ------
    ValueError
        If `points`, `target_radius`, `max_range` or `keep_out` is not positive, or
        `coverage` is outside (0, 1].
    """

    points: int = 100
    target_radius: float = 5.0
    max_range: float = 200.0
    sun_exclusion: float = math.radians(30.0)
    keep_out: float = 10.0
    coverage: float = 0.95
    lit_resolution: float = 60.0

    def __post_init__(self):
        super().__post_init__()
        if self.points < 1 or min(self.target_radius, self.max_range, self.keep_out) <= 0.0:
            raise ValueError("points, target_radius, max_range and keep_out must be positive")
        if not 0.0 < self.coverage <= 1.0:
            raise ValueError("coverage is a fraction in (0, 1]")


class InspectionEnv(ScenarioEnv):
    """Image the target's surface points under lighting constraints.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec; its parameters are `InspectionParams`.
    config : mapping, optional
        `EnvConfig` field overrides (SI units).

    Raises
    ------
    ValueError
        If the parameters are invalid or a `thrust_inhibit` edge is not a whole step.
    """

    params_type: ClassVar[type] = InspectionParams
    params: InspectionParams

    def __init__(self, spec, config=None):
        super().__init__(spec, config)
        self.normals_body = fibonacci_sphere(self.params.points)
        self.sun0 = sun_direction(0.0, 0.0)
        self.inspectable = np.ones(self.params.points, dtype=bool)

    def _constraints(self) -> tuple[Constraint, ...]:
        return (KeepOut(self.params.keep_out),)

    def _goal(self) -> dict[str, Any]:
        p = self.params
        return {
            **super()._goal(),
            "task": "inspect",
            "points": p.points,
            "target_radius": p.target_radius,
            "max_range": p.max_range,
            "sun_exclusion": p.sun_exclusion,
            "coverage": p.coverage,
            "sun_direction": tuple(float(c) for c in self.sun0),
        }

    def metric_settings(self) -> MetricSettings:
        """The metric parameters the scenario declares.

        Returns
        -------
        MetricSettings
            The abort set, one 0.3 m/s impulse along +R-bar.
        """
        # The goal has no approach direction, so the abort set is declared: radially away.
        return MetricSettings(cam_impulses=((0.3, 0.0, 0.0),))

    def _draw_episode(self, rng: np.random.Generator) -> None:
        super()._draw_episode(rng)
        self.sun0 = sun_direction(float(self.draw.get("beta", 0.0)), float(self.draw["sun_phase"]))
        self.inspectable = self._inspectable()

    def normals(self, t) -> np.ndarray:
        """Outward normals in Hill axes at `t`, shape (..., points, 3).

        Parameters
        ----------
        t : array_like
            Time (s), any shape.

        Returns
        -------
        numpy.ndarray
            Unit normals, shape (*t.shape, points, 3).
        """
        return np.einsum("...ij,pj->...pi", self.target.matrix(t, self.n), self.normals_body)

    def inspected(self, states: np.ndarray, t) -> np.ndarray:
        """Which points each state inspects at the matching time, shape (..., points).

        Parameters
        ----------
        states : numpy.ndarray
            Relative states (m, m/s), Hill axes, shape (..., 6).
        t : array_like
            Times (s), shape matching the states' leading shape.

        Returns
        -------
        numpy.ndarray
            Boolean, shape (..., points).
        """
        p = self.params
        states = np.asarray(states, dtype=float)
        times = np.asarray(t, dtype=float)
        r = states[..., :3]
        rho = np.linalg.norm(r, axis=-1)
        normals = self.normals(times)
        sun = sun_at(self.sun0, self.n, times)
        offsets = r[..., None, :] - p.target_radius * normals
        facing = np.sum(normals * offsets, axis=-1) > 0.0
        lit = (np.sum(normals * sun[..., None, :], axis=-1) > 0.0) & (
            self.target_lit(times)[..., None] > 0.0
        )
        look = -r
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(rho > 0.0, np.sum(look * sun, axis=-1) / rho, 1.0)
        glare = np.arccos(np.clip(cos, -1.0, 1.0)) >= p.sun_exclusion
        close = rho <= p.max_range
        return facing & lit & (glare & close)[..., None]

    @property
    def orbit_radius(self) -> float:
        """Target's distance from the configured central body's centre (m)."""
        return get_body(self.cfg.body).radius + self.cfg.altitude_m

    def target_lit(self, t) -> np.ndarray:
        """1 where the target is outside the central body's shadow cylinder at `t` (s), else 0.

        Parameters
        ----------
        t : float or array_like
            Times since the episode start (s).

        Returns
        -------
        numpy.ndarray
            Shape of `t`.
        """
        return sunlit(self.sun0, self.n, t, self.orbit_radius, get_body(self.cfg.body))

    def reset(self, *, seed=None, options=None):
        """Start an episode with the inspected points and violation record cleared.

        See `BaseEnv.reset`; the points the start state inspects count as seen.

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
        self.seen = np.zeros(self.params.points, dtype=bool)
        self.violated = False
        self.violation_steps = 0
        self.success = False
        self.time_to_coverage: float | None = None
        obs, info = super().reset(seed=seed, options=options)
        self.violated = self.min_margins.get("keep_out", 0.0) < 0.0
        self.violation_steps = int(self.violated)
        self.seen |= self.inspected(self.state, 0.0)
        return obs, self._info()

    def _inspectable(self) -> np.ndarray:
        """Points lit at some time in the episode, sampled every `lit_resolution` s."""
        horizon = self.cfg.max_steps * self.cfg.dt
        times = np.arange(
            0.0, horizon + 0.5 * self.params.lit_resolution, self.params.lit_resolution
        )
        normals = self.normals(times)
        sun = sun_at(self.sun0, self.n, times)
        lit = (np.sum(normals * sun[:, None, :], axis=-1) > 0.0) & (
            self.target_lit(times)[:, None] > 0.0
        )
        return np.any(lit, axis=0)

    def coverage(self) -> float:
        """Fraction of all points inspected so far.

        Returns
        -------
        float
            Coverage in [0, 1].
        """
        return float(np.mean(self.seen))

    def inspectable_coverage(self) -> float:
        """Fraction of the inspectable points inspected so far, 1 when none is inspectable.

        Returns
        -------
        float
            Coverage in [0, 1].
        """
        count = int(np.sum(self.inspectable))
        return 1.0 if count == 0 else float(np.sum(self.seen & self.inspectable) / count)

    def _goal_distance(self, state: np.ndarray) -> float:
        return 1.0 - self.inspectable_coverage()

    def _success(self) -> bool:
        return self.success

    def task_outcome(self) -> dict[str, Any]:
        """The coverage verdict, which needs the lighting and the target's true attitude.

        Returns
        -------
        dict
            "met" once the inspectable coverage reached `coverage`, "time_s" (s) when it did,
            and "progress", the inspectable coverage.
        """
        return {
            "met": self.time_to_coverage is not None,
            "time_s": self.time_to_coverage,
            "progress": self.inspectable_coverage(),
        }

    def _evaluate(self, record: StepRecord) -> tuple[float, bool, dict[str, Any]]:
        trace = record.segment
        step_violation = "keep_out" in trace.minima and trace.minima["keep_out"].margin < 0.0
        if step_violation:
            self.violated = True
            self.violation_steps += 1
        self.seen |= np.any(self.inspected(trace.states, trace.times), axis=0)
        done = self.inspectable_coverage() >= self.params.coverage
        if done and self.time_to_coverage is None:
            self.time_to_coverage = self.t
        self.success = done and not self.violated
        return 0.0, done, {}

    def agent_info(self) -> dict[str, Any]:
        """The part of `info` agents may see, with the coverage.

        Returns
        -------
        dict
            "time_s" (s) and "coverage", the fraction of all points inspected.
        """
        # The camera knows which points it has imaged, so the count is the agent's to see.
        return {**super().agent_info(), "coverage": self.coverage()}

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        rho = float(np.linalg.norm(self.state[:3]))
        coverage = self.coverage()
        dv = max(self.dv_l2, 1e-12)
        info.update(
            {
                "coverage": coverage,
                "inspectable_coverage": self.inspectable_coverage(),
                "inspectable_fraction": float(np.mean(self.inspectable)),
                "coverage_per_mps": coverage / dv,
                "time_to_coverage_s": self.time_to_coverage,
                "dist_to_hold_m": rho,
                "range_to_target_m": rho,
                "speed_mps": float(np.linalg.norm(self.state[3:])),
                "min_range_to_target_m": self.min_margins["keep_out"] + self.params.keep_out
                if "keep_out" in self.min_margins
                else rho,
                "keep_out_violated": self.violated,
                "violation_steps": self.violation_steps,
                "captured": self.success,
                "success": self.success,
            }
        )
        return info


__all__ = ["InspectionEnv", "InspectionParams", "fibonacci_sphere"]
