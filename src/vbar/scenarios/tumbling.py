"""Capture of a tumbling target: reach a capture-ready state at an interface on the turning body.

The capture-ready state is defined at the capture interface, not at the centres of mass (Fehse,
Automated Rendezvous and Docking of Spacecraft, 2003, sections 7.1.1, 8.1.2 to 8.1.3 and 8.3.6):
the chaser's capture point holds within `capture_radius` of the target's capture point c(t),
a point fixed to the body at arm's reach beyond the grapple interface, with a velocity relative
to it below `capture_speed`, without a break for `dwell` s. The target point moves at
c_dot = w x c, so a chaser that holds still in the Hill frame is not capture-ready on a turning
target; it must follow the point. The target's body is a capsule (a segment swept by a sphere)
that turns with it and is the hard keep-out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..constraints import Constraint
from ..metrics.settings import MetricSettings
from .base import StepRecord
from .scenario import ScenarioEnv, ScenarioParams
from .targets import Passive, TargetMode, body_point


@dataclass(frozen=True)
class RotatingCapsule:
    """A capsule fixed to a turning target; margin dist(r, segment) - radius, m.

    The segment's ends `ends` are body-fixed and follow `target`. The distance to a moving set
    changes at most at the chaser's speed plus the fastest speed of the set's points, and the
    segment's points move at most at the attitude rate times their distance from the target's
    centre, the larger of the ends' distances; so the rate bound is
    speed + rate max(|a|, |b|). The margin is 1-Lipschitz in position.

    Parameters
    ----------
    ends : tuple of tuple of float
        The segment's ends (a, b) in body axes (m).
    radius : float
        The capsule's radius (m), positive.
    target : TargetMode
        The target's attitude mode; kept out of the public spec.
    n : float
        Mean motion (rad/s).
    name : str
        Constraint name.

    Raises
    ------
    ValueError
        If `radius` is not positive or an end is not a 3-vector.
    """

    ends: tuple[tuple[float, float, float], tuple[float, float, float]]
    radius: float
    target: TargetMode
    n: float
    name: str = "keep_out"
    resolution: ClassVar[float] = 1e-3
    private_fields: ClassVar[tuple[str, ...]] = ("target",)

    def __post_init__(self):
        if not self.radius > 0.0:
            raise ValueError("radius must be positive")
        a, b = (tuple(float(c) for c in np.asarray(end).reshape(3)) for end in self.ends)
        object.__setattr__(self, "ends", (a, b))

    def margin(self, state: np.ndarray, t: float | np.ndarray = 0.0) -> np.ndarray:
        """Distance from the capsule's surface where it stands at `t`.

        Parameters
        ----------
        state : numpy.ndarray
            Relative state (m, m/s), Hill axes, shape (..., 6).
        t : float or numpy.ndarray
            Time (s), broadcast against the state's leading shape.

        Returns
        -------
        numpy.ndarray
            Margin (m), positive outside, shape (...,).
        """
        state = np.asarray(state, dtype=float)
        times = np.broadcast_to(np.asarray(t, dtype=float), state.shape[:-1])
        a = body_point(self.target, self.ends[0], times, self.n)
        b = body_point(self.target, self.ends[1], times, self.n)
        d = b - a
        length = np.sum(d * d, axis=-1)
        r = state[..., :3]
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(length > 0.0, np.sum((r - a) * d, axis=-1) / length, 0.0)
        s = np.clip(s, 0.0, 1.0)
        closest = a + s[..., None] * d
        return np.linalg.norm(r - closest, axis=-1) - self.radius

    def rate_bound(self, speed: float, accel: float, rho_min: float = 0.0) -> float:
        """Bound on |d margin / dt|: speed + rate max(|a|, |b|).

        Parameters
        ----------
        speed : float
            Bound on the relative speed (m/s).
        accel : float
            Bound on the relative acceleration (m/s^2), unused.
        rho_min : float
            Lower bound on the range (m), unused.

        Returns
        -------
        float
            Bound on the margin's rate (m/s).
        """
        reach = max(float(np.linalg.norm(end)) for end in self.ends)
        return speed + self.target.rate_bound(self.n) * reach


@dataclass(frozen=True, kw_only=True)
class CaptureParams(ScenarioParams):
    """Parameters of a capture scenario (m, m/s, s); body axes for the body geometry.

    Parameters
    ----------
    capture_point : tuple of float
        c, the target's capture point (m), body axes: where the chaser's capture point must be,
        at arm's reach beyond the grapple interface.
    interface : tuple of float
        The grapple interface (m), body axes; informational.
    capture_radius : float
        Greatest distance from the capture point (m), positive.
    capture_speed : float
        Greatest speed relative to the moving capture point (m/s), positive.
    dwell : float
        Time the capture-ready state must hold without a break (s), not negative.
    body_ends : tuple of tuple of float
        Ends of the body's capsule segment (m), body axes.
    body_radius : float
        Radius of the body's capsule (m), positive; the hard keep-out.
    commit_radius : float
        Range inside which abort availability is not evaluated (m).
    passive_exclusion : float
        Range inside which passive safety is not evaluated (m).
    plume_range : float, optional
        Range within which thrust is scored for plume impingement (m).
    cam_impulses : tuple of tuple of float, optional
        The abort set, delta-v vectors (m/s), Hill axes.
    terminate_on_violation : bool
        Whether a hard-constraint violation ends the episode.
    thrust_inhibit, soft
        As in `ScenarioParams`.

    Raises
    ------
    ValueError
        If the capture limits or the body radius are not positive, `dwell` is negative, or the
        capture ball reaches into the body.
    """

    capture_point: tuple[float, float, float]
    interface: tuple[float, float, float] = (0.0, 0.0, 0.0)
    capture_radius: float = 0.2
    capture_speed: float = 0.02
    dwell: float = 30.0
    body_ends: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-3.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    )
    body_radius: float = 1.5
    commit_radius: float = 0.0
    passive_exclusion: float = 0.0
    plume_range: float | None = None
    cam_impulses: tuple[tuple[float, float, float], ...] | None = None
    terminate_on_violation: bool = False

    def __post_init__(self):
        super().__post_init__()
        if min(self.capture_radius, self.capture_speed, self.body_radius) <= 0.0:
            raise ValueError("capture_radius, capture_speed and body_radius must be positive")
        if self.dwell < 0.0:
            raise ValueError("dwell must not be negative")
        for name in ("capture_point", "interface"):
            object.__setattr__(self, name, tuple(float(c) for c in getattr(self, name)))
        # The whole capture ball lies outside the body, so a capture-ready chaser is clear of it.
        clearance = RotatingCapsule(self.body_ends, self.body_radius, Passive(), 0.0)
        if float(clearance.margin(np.r_[self.capture_point, 0.0, 0.0, 0.0])) <= self.capture_radius:
            raise ValueError("the capture ball reaches into the body")


class CaptureEnv(ScenarioEnv):
    """Reach a capture-ready state at the capture point of a turning target and hold it.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec; its parameters are `CaptureParams`.
    config : mapping, optional
        `EnvConfig` field overrides (SI units).

    Raises
    ------
    ValueError
        If the parameters are invalid or a `thrust_inhibit` edge is not a whole step.
    """

    params_type: ClassVar[type] = CaptureParams
    params: CaptureParams

    def capture_point_at(self, t) -> tuple[np.ndarray, np.ndarray]:
        """The target's capture point and its velocity in Hill axes at `t`.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.

        Returns
        -------
        tuple of numpy.ndarray
            Position (m) and velocity (m/s), each shape (*t.shape, 3).
        """
        # The predicate, the info and the goal distance all ask for the point at the step's end;
        # the attitude is computed once for it.
        cached = getattr(self, "_capture_cache", None)
        if np.ndim(t) == 0 and cached is not None and cached[0] == float(t):
            return cached[1], cached[2]
        point = body_point(self.target, self.params.capture_point, t, self.n)
        velocity = np.cross(self.target.angular_velocity(t, self.n), point)
        if np.ndim(t) == 0:
            # Handed out on every later call for this time, so not to be changed in place.
            point.setflags(write=False)
            velocity.setflags(write=False)
            self._capture_cache = (float(t), point, velocity)
        return point, velocity

    def _constraints(self) -> tuple[Constraint, ...]:
        p = self.params
        return (RotatingCapsule(p.body_ends, p.body_radius, self.target, self.n),)

    def _goal(self) -> dict[str, Any]:
        p = self.params
        goal: dict[str, Any] = {
            **super()._goal(),
            "task": "capture",
            "capture_point": p.capture_point,
            "interface": p.interface,
            "capture_radius": p.capture_radius,
            "capture_speed": p.capture_speed,
            "hold_time": p.dwell,
            "body_ends": p.body_ends,
            "body_radius": p.body_radius,
            "commit_radius": p.commit_radius,
            "passive_exclusion": p.passive_exclusion,
        }
        if p.plume_range is not None:
            goal["plume_range"] = p.plume_range
        return goal

    def metric_settings(self) -> MetricSettings:
        """The metric parameters the scenario declares.

        Returns
        -------
        MetricSettings
            The body's swept radius as the collision radius, the passive exclusion, commit
            radius and plume range (m), and the abort set (m/s, Hill axes).
        """
        p = self.params
        swept = max(float(np.linalg.norm(end)) for end in p.body_ends) + p.body_radius
        return MetricSettings(
            collision_radius=swept,
            passive_exclusion=p.passive_exclusion,
            commit_radius=p.commit_radius,
            plume_range=p.plume_range,
            cam_impulses=p.cam_impulses,
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
        # A new episode may draw a new tumble.
        self._capture_cache: tuple[float, np.ndarray, np.ndarray] | None = None
        obs, info = super().reset(seed=seed, options=options)
        self.violated = any(self.min_margins[name] < 0.0 for name in self._hard())
        self.violation_steps = int(self.violated)
        return obs, self._info()

    def _hard(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.constraints if c.name not in self.soft_constraints)

    def _errors(self, states: np.ndarray, t) -> tuple[np.ndarray, np.ndarray]:
        """Distance (m) from the capture point and speed (m/s) relative to it."""
        states = np.asarray(states)
        point, velocity = self.capture_point_at(t)
        return (
            np.linalg.norm(states[..., :3] - point, axis=-1),
            np.linalg.norm(states[..., 3:6] - velocity, axis=-1),
        )

    def _inside(self, states: np.ndarray, t) -> np.ndarray:
        distance, speed = self._errors(states, t)
        return (distance < self.params.capture_radius) & (speed < self.params.capture_speed)

    def _goal_distance(self, state: np.ndarray) -> float:
        return float(self._errors(state, self.t)[0])

    def _success(self) -> bool:
        return self.success

    def _evaluate(self, record: StepRecord) -> tuple[float, bool, dict[str, Any]]:
        trace = record.segment
        step_violation = any(trace.minima[name].margin < 0.0 for name in self._hard())
        if step_violation:
            self.violated = True
            self.violation_steps += 1
        # The predicate is checked on every sample of the step, so the dwell is continuous to
        # the sampling resolution; the metrics settle it in continuous time.
        now = bool(self._inside(self.state, self.t))
        if bool(np.all(self._inside(trace.states, trace.times))) and now:
            if self.held_since is None:
                self.held_since = float(trace.times[0])
            self.dwell = self.t - self.held_since
        elif now:
            self.held_since = self.t
            self.dwell = 0.0
        else:
            self.held_since = None
            self.dwell = 0.0
        held = self.held_since is not None and self.dwell >= self.params.dwell - 1e-9
        self.success = held and not self.violated
        terminated = held or (step_violation and self.params.terminate_on_violation)
        return 0.0, terminated, {}

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        distance, speed = self._errors(self.state, self.t)
        rho = float(np.linalg.norm(self.state[:3]))
        clearance = self.min_margins.get("keep_out")
        p = self.params
        info.update(
            {
                "dist_to_hold_m": float(distance),
                "speed_to_capture_mps": float(speed),
                "range_to_target_m": rho,
                "speed_mps": float(np.linalg.norm(self.state[3:])),
                "min_clearance_m": clearance,
                # For the capsule, the least distance to the body's axis segment.
                "min_range_to_target_m": rho if clearance is None else clearance + p.body_radius,
                "keep_out_violated": self.violated,
                "violation_steps": self.violation_steps,
                "captured": bool(self._inside(self.state, self.t)),
                "dwell_s": self.dwell,
                "success": self.success,
            }
        )
        return info


__all__ = ["CaptureEnv", "CaptureParams", "RotatingCapsule"]
