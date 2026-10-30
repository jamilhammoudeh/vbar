"""Set-point tracker: flies a reference given as a point, a moving point or a velocity.

The tracker turns a destination into a continuous command in the view's action mode. A leg to a
point is a velocity law on the Clohessy-Wiltshire model with the natural acceleration cancelled
(Fehse 6.5, forced motion): the demanded speed along the line to the point is the least of the
vehicle speed, a gain times the distance and the public speed laws, and a PD hold takes over
within `capture_range` of the last point. Legs that would pass within `clearance` keep-out radii
of the target go round it on a sphere. A docking leg stands off outside every approach cone, then
descends the port axis at a speed in the mechanism's window; an orbit leg joins a circle about
the target that turns with the Sun (Fehse 5.7.3 inspection). It reads only the `AgentView`, the
observation and the time.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from ..constraints import _VBAR_COAST
from ..scenarios.targets import rotation_z
from .base import AgentView, action_scale
from .briefing import target_mode

# A rung is reached within this distance (m) and below this speed (m/s) of its reference.
ARRIVAL_RADIUS = 5.0
ARRIVAL_SPEED = 0.05

_ARC_STEP = math.radians(45.0)
# A detour goes round the target at this many keep-out radii and at least 1.3 times the
# clearance. The passive-speed law falls to zero at the keep-out radius, so a detour nearer the
# target would crawl; farther out, the arc is long.
_DETOUR_RADII = 3.0
# Passing a detour point within this distance (m) is close enough to turn toward the next one.
_MIN_TURN = 3.0
# Seconds over which an orbit leg turns from the start direction to the circle's.
_JOIN_S = 900.0
# Final docking leg: closing speed rises 0.003 m/s per metre of depth above the window's middle
# (inside the hard speed limit, whose slope is about 0.0044 1/s); lateral velocity -0.03 per metre
# of offset; the velocity loop stiffens to 0.2 1/s inside 2 m so the speed settles in the window.
_DEPTH_SLOPE = 0.003
_LATERAL_GAIN = 0.03
_STIFF_DEPTH = 2.0
_STIFF_GAIN = 0.2
# Beyond this angle (rad) from the port axis the standoff is approached by walking round the
# sphere about the port in steps of at most _WALK_STEP.
_WALK_ANGLE = 0.15
_WALK_STEP = 0.35
_DOCK_MARGIN = 10.0


def _laws(view: AgentView, rho: float) -> tuple[float, float]:
    """The lowest closing-speed law and the lowest full-speed law at range rho (m/s)."""
    closing = full = math.inf
    for spec in view.constraints:
        kind = spec["type"]
        if kind.startswith("closing_speed_limit"):
            closing = min(closing, spec["v0"] + spec["v1"] * rho)
        elif kind.startswith("speed_limit"):
            full = min(full, spec["v0"] + spec["v1"] * rho)
        elif kind.startswith("gate_table"):
            ranges, speeds = zip(*sorted(spec["gates"]), strict=True)
            closing = min(closing, float(np.interp(rho, ranges, speeds)))
        elif kind.startswith("passive_speed"):
            extent, n = spec["extent"], spec["n"]
            if rho <= extent:
                law = 0.0
            elif spec["axis"] == "vbar":
                law = n * (rho - extent) / _VBAR_COAST
            else:
                law = n * math.sqrt(max((4.0 * rho - extent) ** 2 - 9.0 * rho**2, 0.0))
            closing = min(closing, law)
    return closing, full


def speed_cap(view: AgentView, rho: float) -> float:
    """The lowest public speed law at range `rho` from the target.

    Parameters
    ----------
    view : AgentView
        The scenario's public description; the closing-speed, speed, gate-table and
        passive-speed constraints are read.
    rho : float
        Range from the target (m).

    Returns
    -------
    float
        The highest speed (m/s) every law allows at `rho`; `math.inf` if the view has none.
    """
    return min(_laws(view, rho))


def _real(name: str, value, low: float, *, closed: bool = False, high: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    number = float(value)
    if not (number >= low if closed else number > low) or number > high:
        raise ValueError(f"{name} out of range, got {value!r}")
    return number


def _cw_terms(state: np.ndarray, n: float) -> np.ndarray:
    """Natural Clohessy-Wiltshire acceleration of a relative state (m/s^2, Hill)."""
    x, _, z, vx, vy, _ = state[:6]
    return np.array([3 * n**2 * x + 2 * n * vy, -2 * n * vx, -(n**2) * z])


def _segment_clearance(a: np.ndarray, b: np.ndarray) -> float:
    """Least distance from the origin to the segment from a to b (m)."""
    d = b - a
    length = float(d @ d)
    s = 0.0 if length == 0.0 else float(np.clip(-(a @ d) / length, 0.0, 1.0))
    return float(np.linalg.norm(a + s * d))


def _around(a: np.ndarray, b: np.ndarray, radius: float) -> list[np.ndarray]:
    """Points on the sphere of `radius` from the direction of a to that of b, 45 deg apart."""
    ua, ub = a / np.linalg.norm(a), b / np.linalg.norm(b)
    angle = math.acos(float(np.clip(ua @ ub, -1.0, 1.0)))
    if angle < 1e-9:
        return [radius * ua]
    if math.pi - angle < 1e-6:
        # Opposite directions: the great circle is undefined, so go through a perpendicular one.
        side = np.array([0.0, 0.0, 1.0]) if abs(ua[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        mid = np.cross(ua, side)
        mid /= np.linalg.norm(mid)
        return [radius * ua, *_around(radius * mid, b, radius)]
    count = max(1, math.ceil(angle / _ARC_STEP))
    points = []
    for k in range(count + 1):
        f = k / count
        w = (math.sin((1 - f) * angle) * ua + math.sin(f * angle) * ub) / math.sin(angle)
        points.append(radius * w)
    return points


def _toward(u: np.ndarray, target: np.ndarray, angle: float) -> np.ndarray:
    """Unit vector `u` turned toward the unit vector `target` by `angle` (rad)."""
    k = np.cross(u, target)
    if np.linalg.norm(k) < 1e-9:
        # Equal or opposite directions: turn about any axis perpendicular to `u`.
        k = np.cross(u, [0.0, 0.0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0.0, 0.0])
    k = k / np.linalg.norm(k)
    return u * math.cos(angle) + np.cross(k, u) * math.sin(angle)


class SetpointTracker:
    """Flies the reference a supervised agent gives it, in the view's action mode.

    Parameters
    ----------
    v_max : float
        Highest speed along a leg (m/s).
    gain : float
        Speed per metre of distance to the last point of a leg (1/s); the leg slows to a stop.
    k_v : float
        Gain of the velocity loop (1/s).
    wn : float
        Natural frequency (rad/s) of the PD hold within `capture_range` of the last point.
    zeta : float
        Damping ratio of the PD hold.
    capture_range : float
        Distance (m) to the last point of a leg inside which the PD hold acts.
    speed_margin : float
        Fraction, at most 1, of the public speed laws the demanded speed may use.
    clearance : float
        Keep-out radii, at least 1, a leg must pass the target at; a leg that comes nearer goes
        round the target on a sphere.
    hold_accel : float
        Fraction, at most 1, of the thruster acceleration that holding a place may cost against
        the tidal acceleration there; a supervised agent refuses a HOLD that costs more.

    Raises
    ------
    ValueError
        If a parameter is not a finite number in its range.
    """

    def __init__(
        self,
        *,
        v_max: float = 0.5,
        gain: float = 0.01,
        k_v: float = 0.05,
        wn: float = 0.02,
        zeta: float = 1.0,
        capture_range: float = 20.0,
        speed_margin: float = 0.8,
        clearance: float = 1.2,
        hold_accel: float = 0.5,
    ):
        self.v_max = _real("v_max", v_max, 0.0)
        self.gain = _real("gain", gain, 0.0)
        self.k_v = _real("k_v", k_v, 0.0)
        self.wn = _real("wn", wn, 0.0)
        self.zeta = _real("zeta", zeta, 0.0)
        self.capture_range = _real("capture_range", capture_range, 0.0)
        self.speed_margin = _real("speed_margin", speed_margin, 0.0, high=1.0)
        self.clearance = _real("clearance", clearance, 1.0, closed=True)
        self.hold_accel = _real("hold_accel", hold_accel, 0.0, high=1.0)
        if not math.isfinite(self.wn * self.wn) or not math.isfinite(2.0 * self.zeta * self.wn):
            raise ValueError("wn and zeta give a hold gain that is not finite")

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode: read the public view and stop commanding.

        Parameters
        ----------
        view : AgentView
            The scenario's public description.
        rng : numpy.random.Generator, optional
            Unused; the tracker is deterministic.

        Raises
        ------
        ValueError
            If the view's action mode is not continuous.
        """
        scale = action_scale(view)
        goal = view.goal
        self._view = view
        self._scale = scale
        self._n = view.n
        self._dt = view.dt
        self._a_max = view.max_accel
        self._ball = view.thrust_limit == "ball"
        self._keep_out = max(
            (float(c["radius"]) for c in view.constraints if c["type"].startswith("keep_out")),
            default=0.0,
        )
        self._dock = goal.get("task") == "dock"
        if self._dock:
            self._port0 = np.array(goal["port"], dtype=float)
            self._axis0 = np.array(goal["port_axis"], dtype=float)
            low, high = goal["axial_speed_window"]
            self._v_contact = 0.5 * (low + high)
            reach = max(
                (
                    c["active_range"]
                    for c in view.constraints
                    if c["type"].startswith("approach_cone")
                ),
                default=0.0,
            )
            self._standoff = reach + _DOCK_MARGIN
            self._mode_of_target = target_mode(goal.get("target", {"mode": "passive"}))
        self._mode = "coast"
        self._dest = np.zeros(3)
        self._speed = self.v_max
        self._route: list[np.ndarray] | None = None
        self._k = 0
        self._stage = "standoff"
        self._start: np.ndarray | None = None
        self._t0 = 0.0
        self._v_cmd = np.zeros(3)
        self._remaining = np.zeros(3)
        self._r_ref = np.zeros(3)
        self._v_ref = np.zeros(3)

    def fly_to(self, point, kind: str = "hold", *, speed: float | None = None) -> None:
        """Fly to a point, going round the target if the straight leg passes too near it.

        Parameters
        ----------
        point : array_like
            Destination (m, Hill), shape (3,); for "contact" and "orbit" legs the rung position
            at t = 0.
        kind : str
            "hold", "goal" or "waypoint" for a point at rest; "contact" for the docking port,
            reached by a standoff and a descent along the port axis; "orbit" for the circle about
            the target that keeps its angle to the Sun.
        speed : float, optional
            Speed along the leg (m/s), at most `v_max`; default `v_max`.
        """
        self._dest = np.array(point, dtype=float)
        self._speed = self.v_max if speed is None else min(float(speed), self.v_max)
        self._route = None
        self._k = 0
        self._stage = "standoff"
        self._start = None
        self._mode = {"contact": "dock", "orbit": "orbit"}.get(kind, "leg")
        self._v_ref = np.zeros(3)
        self._r_ref = self._dest

    def hold(self, r) -> None:
        """Stop at the place `r` (m, Hill) and hold it."""
        self._dest = np.array(r, dtype=float)
        self._speed = self.v_max
        self._route = [self._dest]
        self._k = 0
        self._mode = "leg"
        self._r_ref, self._v_ref = self._dest, np.zeros(3)

    def track_velocity(self, r, v) -> None:
        """Fly at the velocity `v` (m/s, Hill) from the place `r` (m, kept as the reference)."""
        self._v_cmd = np.array(v, dtype=float)
        self._mode = "velocity"
        self._r_ref, self._v_ref = np.array(r, dtype=float), self._v_cmd

    def coast(self) -> None:
        """Command no thrust."""
        self._mode = "coast"

    def fire(self, dv) -> None:
        """Burn the velocity change `dv` (m/s, Hill) over as many steps as the thrust needs."""
        self._remaining = np.array(dv, dtype=float)
        self._mode = "burn"

    def reference(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Position (m) and velocity (m/s, Hill) the tracker is flying to at time `t` (s)."""
        if self._mode == "orbit":
            r, v, _ = self._circle(t)
            return r, v
        if self._mode == "dock":
            port, _, spin = self._port(t)
            return port, np.cross(spin, port)
        return self._r_ref.copy(), self._v_ref.copy()

    def act(self, obs, t: float) -> np.ndarray:
        """The command, in the view's action mode, for observation `obs` at time `t` (s)."""
        obs = np.asarray(obs, dtype=float)
        v = obs[3:6]
        if self._mode == "coast":
            return np.zeros(3)
        if self._mode == "burn":
            return self._burn()
        if self._mode == "velocity":
            a = -_cw_terms(obs, self._n) + self.k_v * (self._v_cmd - v)
        elif self._mode == "orbit":
            a = self._orbit(obs, t)
        elif self._mode == "dock":
            a = self._descent(obs, t)
        else:
            a = self._follow(obs)
        return self._fraction(a) * self._scale

    def _fraction(self, a: np.ndarray) -> np.ndarray:
        """Acceleration as a fraction of a full command, inside the thrust limit."""
        f = a / self._a_max
        if self._ball:
            size = float(np.linalg.norm(f))
            return f / size if size > 1.0 else f
        return np.clip(f, -1.0, 1.0)

    def _burn(self) -> np.ndarray:
        f = self._fraction(self._remaining / self._dt)
        self._remaining = self._remaining - f * self._a_max * self._dt
        if float(np.linalg.norm(self._remaining)) < 1e-9:
            self._mode = "coast"
        return f * self._scale

    def _limit(self, r: np.ndarray, u: np.ndarray) -> float:
        """Speed (m/s) along unit direction `u` from `r` that the public laws allow."""
        rho = float(np.linalg.norm(r))
        closing, full = _laws(self._view, rho)
        limit = min(self._speed, self.speed_margin * max(full, 0.0))
        toward = -float(r @ u) / rho if rho > 0.0 else 0.0
        if toward > 1e-9:
            limit = min(limit, self.speed_margin * max(closing, 0.0) / toward)
        return limit

    def _law(
        self,
        obs: np.ndarray,
        point: np.ndarray,
        v_point: np.ndarray,
        *,
        last: bool,
    ) -> np.ndarray:
        """Acceleration toward `point`; the last point of a leg is approached slowing to a stop."""
        r, v = obs[:3], obs[3:6]
        d = point - r
        dist = float(np.linalg.norm(d))
        u = d / dist if dist > 1e-9 else np.zeros(3)
        speed = self._limit(r, u)
        k = self.k_v
        if last:
            gain = self.gain
            if dist < self.capture_range:
                # The PD law -wn^2 (r - p) - 2 zeta wn (v - v_p) as a velocity loop, so the speed
                # laws still bound the speed it demands.
                k, gain = 2.0 * self.zeta * self.wn, self.wn / (2.0 * self.zeta)
            speed = min(speed, gain * dist)
        return -_cw_terms(obs, self._n) + k * (v_point + speed * u - v)

    def _plan(self, start: np.ndarray, dest: np.ndarray) -> list[np.ndarray]:
        """Points to fly through: detours on a sphere about the target, then `dest`."""
        clear = self.clearance * self._keep_out
        if clear <= 0.0 or _segment_clearance(start, dest) >= clear:
            return [dest]
        radius = max(_DETOUR_RADII, 1.3 * self.clearance) * self._keep_out
        arc = _around(start, dest, radius)
        if float(np.linalg.norm(dest)) > radius:
            arc = arc[:-1]
        if float(np.linalg.norm(start)) > radius:
            arc = arc[1:]
        return [*arc, dest]

    def _follow(self, obs: np.ndarray) -> np.ndarray:
        r = obs[:3]
        if self._route is None:
            self._route = self._plan(r, self._dest)
            self._k = 0
        last = len(self._route) - 1
        turn = max(_MIN_TURN, self._speed / self.k_v)
        while self._k < last and float(np.linalg.norm(self._route[self._k] - r)) < turn:
            self._k += 1
        return self._law(obs, self._route[self._k], np.zeros(3), last=self._k == last)

    def _port(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Port position, outward axis and angular velocity of the target at time t (Hill)."""
        mode, n = self._mode_of_target, self._n
        turn = mode.matrix(t, n) @ mode.matrix(0.0, n).T
        return turn @ self._port0, turn @ self._axis0, np.asarray(mode.angular_velocity(t, n))

    def _descent(self, obs: np.ndarray, t: float) -> np.ndarray:
        r, v = obs[:3], obs[3:6]
        port, axis, spin = self._port(t)
        standoff = port + self._standoff * axis
        if self._stage == "standoff":
            if float(np.linalg.norm(standoff - r)) < ARRIVAL_RADIUS:
                self._stage = "final"
            else:
                return self._standoff_leg(obs, port, axis, standoff, spin)
        offset = r - port
        depth = float(offset @ axis)
        lateral = offset - depth * axis
        speed = min(self.v_max, self._v_contact + _DEPTH_SLOPE * max(depth, 0.0))
        # A point at this depth on a turning axis moves with the axis (Fehse 5.3.1).
        v_des = (
            np.cross(spin, port + max(depth, 0.0) * axis) - speed * axis - _LATERAL_GAIN * lateral
        )
        k = self.k_v if depth > _STIFF_DEPTH else _STIFF_GAIN
        return -_cw_terms(obs, self._n) + k * (v_des - v)

    def _standoff_leg(self, obs, port, axis, standoff, spin) -> np.ndarray:
        """Fly to the standoff, walking round the sphere about the port if far off the axis."""
        r = obs[:3]
        rel = r - port
        point = standoff
        if float(np.linalg.norm(standoff - r)) >= self.capture_range:
            angle = math.acos(float(np.clip(rel @ axis / np.linalg.norm(rel), -1.0, 1.0)))
            if angle > _WALK_ANGLE:
                u = _toward(rel / np.linalg.norm(rel), axis, min(angle, _WALK_STEP))
                point = port + self._standoff * u
        return self._law(obs, point, np.cross(spin, point), last=True)

    def _circle(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The circle point, its velocity and acceleration (Hill) at `t`.

        The point turns at -n about the orbit normal, so it keeps its angle to the Sun, which is
        fixed in inertial space.
        """
        spin = np.array([0.0, 0.0, -self._n])
        r = rotation_z(-self._n * t) @ self._dest
        v = np.cross(spin, r)
        return r, v, np.cross(spin, v)

    def _join(self, t: float, start: np.ndarray) -> np.ndarray:
        """Point of the path from the start to the circle, blended over _JOIN_S seconds."""
        circle = self._circle(t)[0]
        tau = t - self._t0
        if tau >= _JOIN_S:
            return circle
        s = 0.5 - 0.5 * math.cos(math.pi * tau / _JOIN_S)
        u0, uc = start / np.linalg.norm(start), circle / np.linalg.norm(circle)
        angle = math.acos(float(np.clip(u0 @ uc, -1.0, 1.0)))
        radius = (1.0 - s) * np.linalg.norm(start) + s * np.linalg.norm(self._dest)
        return radius * _toward(u0, uc, s * angle)

    def _orbit(self, obs: np.ndarray, t: float) -> np.ndarray:
        if self._start is None:
            self._start, self._t0 = obs[:3].copy(), t
        start = self._start
        if t - self._t0 >= _JOIN_S:
            r_ref, v_ref, a_ref = self._circle(t)
        else:
            h = 1.0
            before = max(t - h, self._t0)
            r_ref = self._join(t, start)
            v_ref = (self._join(t + h, start) - self._join(before, start)) / (t + h - before)
            a_ref = (self._join(t + h, start) - 2.0 * r_ref + self._join(before, start)) / (h * h)
        a = a_ref - _cw_terms(np.concatenate([r_ref, v_ref]), self._n)
        return a - self.wn**2 * (obs[:3] - r_ref) - 2.0 * self.zeta * self.wn * (obs[3:6] - v_ref)
