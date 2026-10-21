"""Docking: make contact at the port inside the mechanism's reception envelope and a corridor.

Contact is an event, the first crossing of the port plane (r - r_port) . a_hat = 0 from the
approach side within `contact_range` of the port, found in the exact segment trace; the step
ends there. Success at contact (Fehse, Automated Rendezvous and Docking of Spacecraft, 2003,
sections 7.1.1 and 8.3.5, appendix C): lateral offset within `lateral_limit`, lateral speed
within `lateral_speed_limit`, axial closing speed inside the mechanism's window, and no hard
constraint violated on the way. The port and its axis are fixed to the target, so they move
with the target's attitude mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import brentq

from ..constraints import ApproachCone, ClosingSpeedLimit, Constraint, GateTable, SpeedLimit
from ..metrics.settings import MetricSettings
from .base import StepRecord
from .scenario import ScenarioEnv, ScenarioParams
from .targets import (
    BodyKeepOut,
    LightingWindow,
    Passive,
    RotatingCone,
    body_point,
    briefed_mode,
    sun_direction,
)

# Axial contact-speed windows (m/s) by mechanism: Fehse appendix C and section 8.3 (ATV
# probe-drogue nominal 0.05 to 0.10; Soyuz and Progress 0.1 to 0.35; Hermes-Columbus 0.01 to
# 0.03) and the Shuttle APDS, below 0.05 m/s with a lower end of 0.025 m/s (ours, just below the
# planned 0.030 m/s of STS-130; Goodman, History of Space Shuttle Rendezvous, 2011, p. 242).
PRESETS: dict[str, tuple[float, float]] = {
    "probe-drogue-atv": (0.05, 0.10),
    "impact": (0.10, 0.35),
    "low-impact": (0.01, 0.03),
    "apds": (0.025, 0.05),
}


# The record ends this far (m) past the port plane, in steps of this many seconds.
_PAST_PLANE = 1e-9
_PLANE_STEP = 1e-8


@dataclass(frozen=True, kw_only=True)
class DockingParams(ScenarioParams):
    """Parameters of a docking scenario (m, m/s, s, rad); body axes at zero attitude.

    `port` and `port_axis` (from the port toward the approach side) are body-fixed.
    `body_radius` is the target's keep-out sphere. `preset` names the axial contact window, or
    `axial_window` gives it. `cones` lists (half-angle, active range) corridors with their apex
    at the port, in order from the outer one. `speed_limit` is (v0, v1) of the hard limit on
    the whole relative speed, v0 + v1 rho; `None` for v1 means 4 n and `None` for v0 the
    window's upper end. `gates` and `closing_limit` are soft closing-speed laws. `lighting` is
    (low, high, active range) of a Sun-angle window near the port. `hold` is an optional
    station-keeping point (range from the port, dwell) agents may use, informational.
    `reception_depth` is how far the mechanism's capture volume reaches behind the port plane;
    the body keep-out is open that deep at the port.

    Parameters
    ----------
    port : tuple of float
        r_port, the port position (m), body axes.
    port_axis : tuple of float
        a_hat, from the port toward the approach side, nonzero, body axes.
    body_radius : float
        Radius of the target's keep-out sphere (m), positive.
    lateral_limit : float
        Greatest lateral offset from the port axis at contact (m), positive.
    lateral_speed_limit : float
        Greatest lateral speed at contact (m/s), positive.
    preset : str
        Key of `PRESETS` naming the axial contact window, used when `axial_window` is None.
    axial_window : tuple of float, optional
        (low, high) axial closing speed at contact (m/s).
    cones : tuple of tuple of float
        (half-angle rad, active range m) corridors with their apex at the port, outer first.
    speed_limit : tuple, optional
        (v0 m/s, v1 1/s) of the hard limit v0 + v1 rho; None for no limit.
    gates : tuple of tuple of float, optional
        Soft (range m, closing speed m/s) gates.
    closing_limit : tuple of float, optional
        Soft (v0 m/s, v1 1/s) closing-speed law.
    commit_radius : float
        Range from the target inside which abort availability is not evaluated (m).
    passive_exclusion : float
        Range from the target inside which passive safety is not evaluated (m).
    plume_range : float, optional
        Range within which thrust is scored for plume impingement (m).
    lighting : tuple of float, optional
        (low rad, high rad, active range m) of the Sun-angle window.
    contact_range : float
        Distance from the port within which a crossing of the port plane is contact (m).
    hold : tuple of float, optional
        (range from the port m, dwell s) of an informational station-keeping point.
    reception_depth : float
        Depth of the capture volume behind the port plane (m).
    terminate_on_violation : bool
        Whether a hard-constraint violation ends the episode.
    thrust_inhibit, soft
        As in `ScenarioParams`.

    Raises
    ------
    ValueError
        If `preset` is unknown with no `axial_window`, a lateral limit or `body_radius` is not
        positive, or `port_axis` is zero.
    """

    port: tuple[float, float, float] = (0.0, -2.0, 0.0)
    port_axis: tuple[float, float, float] = (0.0, -1.0, 0.0)
    body_radius: float = 2.0
    lateral_limit: float = 0.1
    lateral_speed_limit: float = 0.03
    preset: str = "probe-drogue-atv"
    axial_window: tuple[float, float] | None = None
    cones: tuple[tuple[float, float], ...] = ((math.radians(10.0), 50.0),)
    speed_limit: tuple[float | None, float | None] | None = (None, None)
    gates: tuple[tuple[float, float], ...] | None = None
    closing_limit: tuple[float, float] | None = None
    commit_radius: float = 1.0
    passive_exclusion: float = 50.0
    plume_range: float | None = None
    lighting: tuple[float, float, float] | None = None
    contact_range: float = 50.0
    hold: tuple[float, float] | None = None
    reception_depth: float = 0.1
    terminate_on_violation: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.axial_window is None and self.preset not in PRESETS:
            raise ValueError(f"unknown preset {self.preset!r}, expected one of {sorted(PRESETS)}")
        if min(self.lateral_limit, self.lateral_speed_limit, self.body_radius) <= 0.0:
            raise ValueError("lateral limits and body_radius must be positive")
        object.__setattr__(self, "port", tuple(float(c) for c in self.port))
        object.__setattr__(self, "port_axis", tuple(float(c) for c in self.port_axis))
        if not np.linalg.norm(self.port_axis) > 0.0:
            raise ValueError("port_axis must be nonzero")

    @property
    def window(self) -> tuple[float, float]:
        """(low, high) axial closing speed at contact (m/s), checked 0 <= low < high."""
        low, high = self.axial_window if self.axial_window is not None else PRESETS[self.preset]
        if not 0.0 <= low < high:
            raise ValueError("the axial window needs 0 <= low < high")
        return float(low), float(high)


@dataclass(frozen=True)
class Contact:
    """The contact event: time (s), lateral offset (m), speeds (m/s) and the capture margin.

    Parameters
    ----------
    t : float
        Contact time (s).
    lateral_error : float
        Offset from the port axis (m).
    axial_speed : float
        Closing speed along the port axis (m/s).
    lateral_speed : float
        Speed across the port axis (m/s).
    margin : float
        Least normalised margin to the reception limits, dimensionless; not negative inside.
    """

    t: float
    lateral_error: float
    axial_speed: float
    lateral_speed: float
    margin: float


class DockingEnv(ScenarioEnv):
    """Make contact at the port inside the mechanism's reception envelope and a corridor.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec; its parameters are `DockingParams`.
    config : mapping, optional
        `EnvConfig` field overrides (SI units).

    Raises
    ------
    ValueError
        If the parameters are invalid or a `thrust_inhibit` edge is not a whole step.
    """

    params_type: ClassVar[type] = DockingParams
    params: DockingParams

    @property
    def axis_body(self) -> np.ndarray:
        """Unit port axis, body axes, shape (3,)."""
        axis = np.array(self.params.port_axis)
        return axis / np.linalg.norm(axis)

    @property
    def moving(self) -> bool:
        """Whether the port turns in the Hill frame (any target mode but `Passive`)."""
        return not isinstance(self.target, Passive)

    def port_at(self, t) -> np.ndarray:
        """Port position in Hill axes at `t`.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.

        Returns
        -------
        numpy.ndarray
            Position (m), shape (*t.shape, 3).
        """
        return body_point(self.target, self.params.port, t, self.n)

    def axis_at(self, t) -> np.ndarray:
        """Unit port axis in Hill axes at `t`.

        Parameters
        ----------
        t : array_like
            Time (s), any shape.

        Returns
        -------
        numpy.ndarray
            Unit vector, shape (*t.shape, 3).
        """
        return body_point(self.target, self.axis_body, t, self.n)

    def _sun0(self) -> tuple[float, float, float]:
        s = sun_direction(float(self.draw.get("beta", 0.0)), float(self.draw.get("sun_phase", 0.0)))
        return float(s[0]), float(s[1]), float(s[2])

    def _constraints(self) -> tuple[Constraint, ...]:
        p = self.params
        out: list[Constraint] = []
        for k, (half_angle, active) in enumerate(p.cones):
            name = "approach_cone" if k == 0 else f"approach_cone_{k + 1}"
            # The apex sits behind the port so the corridor is as wide as the lateral
            # reception limit at the port plane; a contact inside the envelope is inside it.
            setback = p.lateral_limit / math.tan(half_angle)
            apex = tuple(np.array(p.port) - setback * self.axis_body)
            if self.moving:
                cone: Constraint = RotatingCone(
                    half_angle, p.port_axis, apex, active + setback, self.target, self.n, name=name
                )
            else:
                cone = ApproachCone(half_angle, p.port_axis, apex, active + setback, name=name)
            out.append(cone)
        out.append(
            BodyKeepOut(p.body_radius, p.port, p.port_axis, self.target, self.n, p.reception_depth)
        )
        if p.speed_limit is not None:
            v0, v1 = p.speed_limit
            out.append(
                SpeedLimit(p.window[1] if v0 is None else v0, 4.0 * self.n if v1 is None else v1)
            )
        if p.gates is not None:
            out.append(GateTable(p.gates))
        if p.closing_limit is not None:
            out.append(ClosingSpeedLimit(*p.closing_limit))
        if p.lighting is not None:
            low, high, active = p.lighting
            radius = self.dynamics.body.radius if hasattr(self.dynamics, "body") else 6_378_137.0
            out.append(
                LightingWindow(
                    (low, high),
                    p.port_axis,
                    p.port,
                    active,
                    self._sun0(),
                    self.n,
                    radius + self.cfg.altitude_m,
                )
            )
        return tuple(out)

    def _goal(self) -> dict[str, Any]:
        p = self.params
        # The port and axis as the briefing puts them in the Hill frame at t = 0; a turning
        # target carries them on from there (the goal's target entry says how).
        told = briefed_mode(self.target)
        goal: dict[str, Any] = {
            **super()._goal(),
            "port": tuple(float(c) for c in body_point(told, p.port, 0.0, self.n)),
            "port_axis": tuple(float(c) for c in body_point(told, self.axis_body, 0.0, self.n)),
            "lateral_limit": p.lateral_limit,
            "lateral_speed_limit": p.lateral_speed_limit,
            "axial_speed_window": p.window,
            "contact_range": p.contact_range,
            "commit_radius": p.commit_radius,
            "passive_exclusion": p.passive_exclusion,
        }
        if p.plume_range is not None:
            goal["plume_range"] = p.plume_range
        if p.hold is not None:
            goal["hold"] = p.hold
        if p.lighting is not None:
            goal["sun_direction"] = self._sun0()
            goal["lighting_window"] = p.lighting
        return goal

    def metric_settings(self) -> MetricSettings:
        """The metric parameters the scenario declares.

        Returns
        -------
        MetricSettings
            Collision radius, passive exclusion, commit radius and plume range (m).
        """
        p = self.params
        return MetricSettings(
            collision_radius=p.body_radius,
            passive_exclusion=p.passive_exclusion,
            commit_radius=p.commit_radius,
            plume_range=p.plume_range,
        )

    def reset(self, *, seed=None, options=None):
        """Start an episode with the contact and violation record cleared (`BaseEnv.reset`).

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
        self.contact: Contact | None = None
        self.pending_contact: float | None = None
        obs, info = super().reset(seed=seed, options=options)
        self.violated = any(self.min_margins[name] < 0.0 for name in self._hard())
        self.violation_steps = int(self.violated)
        return obs, self._info()

    def _hard(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.constraints if c.name not in self.soft_constraints)

    def _depth(self, state: np.ndarray, t) -> np.ndarray:
        """Signed distance in front of the port plane: positive on the approach side."""
        offset = np.asarray(state)[..., :3] - self.port_at(t)
        return np.sum(offset * self.axis_at(t), axis=-1)

    def _event_time(self, tau: np.ndarray, samples: np.ndarray, t0: float) -> float | None:
        depth = self._depth(samples, t0 + tau)
        near = np.linalg.norm(samples[:, :3] - self.port_at(t0 + tau), axis=-1)
        crossing = np.flatnonzero((depth[:-1] > 0.0) & (depth[1:] <= 0.0))
        for k in crossing:
            if min(near[k], near[k + 1]) > self.params.contact_range:
                continue

            def f(s):
                return float(self._depth(self.dynamics.state_in_last_segment(s), t0 + s))

            if depth[k + 1] == 0.0:
                root = float(tau[k + 1])
            else:
                root = brentq(f, tau[k], tau[k + 1], xtol=1e-12, rtol=4 * np.finfo(float).eps)
            # Propagation between samples is exact only to rounding (the step matrices are
            # cached on a rounded time), so the step ends a nanometre past the plane, where the
            # crossing shows in the record whatever the rounding; the body keep-out is open
            # far deeper than that at the port.
            for _ in range(2000):
                if f(root) < -_PAST_PLANE or root >= tau[-1]:
                    break
                root += _PLANE_STEP
            root = min(root, float(tau[-1]))
            self.pending_contact = t0 + root
            return root
        return None

    def _contact(self, t: float) -> Contact:
        p = self.params
        axis = self.axis_at(t)
        offset = self.state[:3] - self.port_at(t)
        velocity = self.state[3:6]
        axial = -float(velocity @ axis)
        lateral = float(np.linalg.norm(offset - (offset @ axis) * axis))
        sideways = float(np.linalg.norm(velocity + axial * axis))
        low, high = p.window
        criteria = [
            (p.lateral_limit - lateral) / p.lateral_limit,
            (p.lateral_speed_limit - sideways) / p.lateral_speed_limit,
            (high - axial) / high,
        ]
        if low > 0.0:
            criteria.append((axial - low) / low)
        return Contact(t, lateral, axial, sideways, min(criteria))

    def _goal_distance(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(state)[:3] - self.port_at(self.t)))

    def _success(self) -> bool:
        return self.contact is not None and self.contact.margin >= 0.0 and not self.violated

    def _evaluate(self, record: StepRecord) -> tuple[float, bool, dict[str, Any]]:
        step_violation = any(record.segment.minima[name].margin < 0.0 for name in self._hard())
        if step_violation:
            self.violated = True
            self.violation_steps += 1
        terminated = step_violation and self.params.terminate_on_violation
        if self.pending_contact is not None:
            self.contact = self._contact(self.pending_contact)
            self.pending_contact = None
            terminated = True
        return 0.0, terminated, {}

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        c = self.contact
        rho = float(np.linalg.norm(self.state[:3]))
        info.update(
            {
                "dist_to_hold_m": self._goal_distance(self.state),
                "range_to_target_m": rho,
                "speed_mps": float(np.linalg.norm(self.state[3:])),
                "min_range_to_target_m": self.min_margins["keep_out"] + self.params.body_radius
                if "keep_out" in self.min_margins
                else rho,
                "keep_out_violated": self.violated,
                "violation_steps": self.violation_steps,
                "captured": c is not None and c.margin >= 0.0,
                "contact": c is not None,
                # None rather than NaN before contact: Gymnasium's checker compares info dicts
                # across resets, and NaN never equals itself.
                "contact_time_s": None if c is None else c.t,
                "contact_lateral_error_m": None if c is None else c.lateral_error,
                "contact_axial_speed_mps": None if c is None else c.axial_speed,
                "contact_lateral_speed_mps": None if c is None else c.lateral_speed,
                "capture_margin": None if c is None else c.margin,
                "success": self._success(),
            }
        )
        return info


__all__ = ["PRESETS", "Contact", "DockingEnv", "DockingParams"]
