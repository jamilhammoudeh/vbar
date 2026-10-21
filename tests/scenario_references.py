"""Reference controllers for the scenario audit: each scenario must be solvable by one of them.

They are hand-written forced-motion and two-impulse controllers with Clohessy-Wiltshire
feedforward, built on the observation and the public view alone, so a run through them is a
run any agent could make. They are not baselines and make no claim to efficiency.
"""

from __future__ import annotations

import math

import numpy as np

from vbar.agents import action_scale
from vbar.dynamics import cw_stm
from vbar.scenarios.targets import rotation_z, sun_at


def cw_terms(state, n):
    """Natural Clohessy-Wiltshire acceleration of a relative state (Hill, m/s^2)."""
    x, y, z, vx, vy, vz = state[:6]
    return np.array([3 * n**2 * x + 2 * n * vy, -2 * n * vx, -(n**2) * z])


def cw_lambert(r0, rf, t, n):
    """Velocity that carries r0 to rf in t s under free Clohessy-Wiltshire motion."""
    phi = cw_stm(n, t)
    return np.linalg.solve(phi[:3, 3:], np.asarray(rf) - phi[:3, :3] @ np.asarray(r0))


def segment_clearance(a, b):
    """Least distance from the origin to the segment from a to b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    length = float(d @ d)
    s = 0.0 if length == 0.0 else float(np.clip(-(a @ d) / length, 0.0, 1.0))
    return float(np.linalg.norm(a + s * d))


ARC_STEP = math.radians(45.0)
GRAPPLE_LEAD = math.radians(46.0)


def around(a, b, radius, step=ARC_STEP):
    """Points on the sphere of `radius` between the directions of a and b, 45 deg apart."""
    ua, ub = np.asarray(a) / np.linalg.norm(a), np.asarray(b) / np.linalg.norm(b)
    angle = math.acos(float(np.clip(ua @ ub, -1.0, 1.0)))
    if angle < 1e-9:
        return [radius * ua]
    if math.pi - angle < 1e-6:
        side = np.array([0.0, 0.0, 1.0]) if abs(ua[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        mid = np.cross(ua, side)
        mid /= np.linalg.norm(mid)
        return [radius * ua, *around(radius * mid, radius * ub, radius, step)[0:]]
    count = max(1, math.ceil(angle / step))
    points = []
    for k in range(count + 1):
        f = k / count
        w = (math.sin((1 - f) * angle) * ua + math.sin(f * angle) * ub) / math.sin(angle)
        points.append(radius * w)
    return points


def plan(start, waypoints, clearance):
    """Waypoints with detours on a sphere of 1.3 `clearance` wherever a leg comes closer."""
    out = []
    here = np.asarray(start[:3], dtype=float)
    for point in waypoints:
        point = np.asarray(point, dtype=float)
        if clearance > 0.0 and segment_clearance(here, point) < clearance:
            radius = 1.3 * clearance
            out += around(here, point, radius)[1:-1] if np.linalg.norm(here) > radius else []
            if np.linalg.norm(here) <= radius:
                out += around(here, point, radius)
        out.append(point)
        here = point
    return out


class Glide:
    """Fly straight legs between waypoints with a speed profile, then hold the last one.

    `waypoints` may be points or callables of time giving (point, velocity); the speed along a
    leg is min(v_max, gain d), and the last waypoint is held with a PD law. `clearance` adds
    detours around the target.
    """

    name = "glide"

    def __init__(
        self,
        waypoints,
        v_max=0.5,
        gain=0.01,
        k_v=0.05,
        wn=0.02,
        zeta=1.0,
        tol=3.0,
        clearance=0.0,
        name="glide",
    ):
        self.waypoints = list(waypoints)
        self.v_max, self.gain, self.k_v, self.wn, self.zeta = v_max, gain, k_v, wn, zeta
        self.tol, self.clearance, self.name = tol, clearance, name

    def reset(self, view, rng=None):
        self.n = view.n
        self.max_accel = view.max_accel
        self.scale = action_scale(view)
        self.route = None
        self.k = 0

    def target(self, k, t):
        point = self.route[k]
        if callable(point):
            return point(t)
        return np.asarray(point, dtype=float), np.zeros(3)

    def act(self, obs, info):
        t = info["time_s"]
        if self.route is None:
            fixed = [p for p in self.waypoints if not callable(p)]
            moving = [p for p in self.waypoints if callable(p)]
            self.route = plan(obs, fixed, self.clearance) + moving if fixed else moving
        point, v_point = self.target(self.k, t)
        d = point - obs[:3]
        dist = float(np.linalg.norm(d))
        if dist < self.tol and self.k < len(self.route) - 1:
            self.k += 1
            point, v_point = self.target(self.k, t)
            d = point - obs[:3]
            dist = float(np.linalg.norm(d))
        a = -cw_terms(obs, self.n)
        if self.k == len(self.route) - 1 and dist < 20.0:
            a += -(self.wn**2) * (obs[:3] - point) - 2 * self.zeta * self.wn * (obs[3:6] - v_point)
        else:
            v_des = v_point + min(self.v_max, self.gain * dist) * d / max(dist, 1e-9)
            a += self.k_v * (v_des - obs[3:6])
        return np.clip(a / self.max_accel, -1.0, 1.0) * self.scale


class Lambert:
    """Two-impulse transfers re-solved every step, then a PD hold at the last point.

    `legs` are (point, depart s, arrive s); before a leg's departure the chaser holds where it
    is. Burns are spread over as many steps as the thrust limit needs.
    """

    name = "lambert"

    def __init__(self, legs, wn=0.02, zeta=1.0, name="lambert"):
        self.legs, self.wn, self.zeta, self.name = list(legs), wn, zeta, name

    def reset(self, view, rng=None):
        self.n, self.dt = view.n, view.dt
        self.max_accel = view.max_accel
        self.scale = action_scale(view)
        self.k = 0
        self.anchor = None

    def act(self, obs, info):
        t = info["time_s"]
        while self.k < len(self.legs) - 1 and t >= self.legs[self.k][2]:
            self.k += 1
            self.anchor = None
        point, depart, arrive = self.legs[self.k]
        point = np.asarray(point, dtype=float)
        a = -cw_terms(obs, self.n)
        remaining = arrive - t
        if t < depart or remaining < 1.5 * self.dt:
            hold = point if remaining < 1.5 * self.dt else self._anchor(obs)
            a += -(self.wn**2) * (obs[:3] - hold) - 2 * self.zeta * self.wn * obs[3:6]
        else:
            v_req = cw_lambert(obs[:3], point, remaining, self.n)
            a += (v_req - obs[3:6]) / self.dt
        return np.clip(a / self.max_accel, -1.0, 1.0) * self.scale

    def _anchor(self, obs):
        if self.anchor is None:
            self.anchor = np.array(obs[:3])
        return self.anchor


class Track:
    """Track a reference trajectory given as t -> (r, v, acceleration) with CW feedforward."""

    name = "track"

    def __init__(self, reference, wn=0.02, zeta=1.0, name="track"):
        self.reference, self.wn, self.zeta, self.name = reference, wn, zeta, name

    def reset(self, view, rng=None):
        self.n = view.n
        self.max_accel = view.max_accel
        self.scale = action_scale(view)

    def act(self, obs, info):
        r, v, acc = self.reference(info["time_s"])
        a = acc - cw_terms(np.concatenate([r, v]), self.n)
        a += -(self.wn**2) * (obs[:3] - r) - 2 * self.zeta * self.wn * (obs[3:6] - v)
        return np.clip(a / self.max_accel, -1.0, 1.0) * self.scale


def circle(radius, rate, theta0, t_end, n, then=None):
    """Reference on a circle in the orbit plane at `rate` (rad/s) from angle theta0 (from +x).

    After `t_end` the reference stays at the end point (or `then` takes over).
    """

    def ref(t):
        tau = min(t, t_end)
        theta = theta0 + rate * tau
        c, s = math.cos(theta), math.sin(theta)
        r = radius * np.array([c, s, 0.0])
        if t >= t_end:
            return (r, np.zeros(3), np.zeros(3)) if then is None else then(t)
        v = radius * rate * np.array([-s, c, 0.0])
        acc = -radius * rate**2 * np.array([c, s, 0.0])
        return r, v, acc

    return ref


class Dock:
    """Approach the port along its axis at a speed inside the mechanism's window.

    The route goes to a point `standoff` m out on the axis, outside the corridor's activation,
    then down the axis with speed min(v_max, gain depth), never below the window's middle. The
    port follows the target mode in the goal. With a lighting window in the goal the final leg
    waits at the standoff until the window is open for the leg's duration.
    """

    name = "dock"

    def __init__(self, standoff=60.0, v_max=0.3, gain=0.01, k_v=0.05, clearance=0.0, name="dock"):
        self.standoff, self.v_max, self.gain, self.k_v = standoff, v_max, gain, k_v
        self.clearance, self.name = clearance, name

    def reset(self, view, rng=None):
        self.n = view.n
        self.max_accel = view.max_accel
        self.scale = action_scale(view)
        goal = view.goal
        self.port0 = np.array(goal["port"], dtype=float)
        self.axis0 = np.array(goal["port_axis"], dtype=float)
        low, high = goal["axial_speed_window"]
        self.v_contact = 0.5 * (low + high)
        mode = goal.get("target", {})
        self.phase = float(mode.get("phase", 0.0)) if mode.get("mode") == "inertial" else None
        self.pitch = float(mode.get("pitch", 0.0)) if mode.get("mode") == "fixed_offset" else None
        self.rate = -self.n if self.phase is not None else 0.0
        self.window = goal.get("lighting_window")
        # Stand off beyond every corridor's activation range, so the final leg enters the
        # outer corridor from its mouth.
        reach = max(c["active_range"] for c in view.constraints if c["type"] == "approach_cone")
        self.standoff = max(self.standoff, reach + 10.0)
        self.sun0 = goal.get("sun_direction")
        self.orbit_radius = (3.986004418e14 / self.n**2) ** (1.0 / 3.0)
        self.route = None
        self.k = 0
        self.leg = False

    def geometry(self, t):
        """Port, axis and the rotation rate vector at `t`; the goal holds the t = 0 values."""
        angle = self.rate * t
        spin = np.array([0.0, 0.0, self.rate])
        if angle == 0.0:
            return self.port0, self.axis0, spin
        m = rotation_z(angle)
        return m @ self.port0, m @ self.axis0, spin

    def window_open(self, t, duration):
        if self.window is None:
            return True
        low, high, _ = self.window
        for tau in np.linspace(t, t + duration, 9):
            sun = sun_at(self.sun0, self.n, tau)
            planar = np.linalg.norm(sun[:2])
            axis = self.axis0[:2] / np.linalg.norm(self.axis0[:2])
            angle = math.acos(float(np.clip(sun[:2] @ axis / planar, -1.0, 1.0)))
            lit = sun[0] >= 0.0 or self.orbit_radius * math.sqrt(1 - sun[0] ** 2) > 6_378_137.0
            if not (low < angle < high and lit):
                return False
        return True

    def act(self, obs, info):
        t = info["time_s"]
        port, axis, spin = self.geometry(t)
        standoff = port + self.standoff * axis
        v_standoff = np.cross(spin, standoff)
        a = -cw_terms(obs, self.n)
        offset = obs[:3] - port
        depth = float(offset @ axis)
        lateral = offset - depth * axis
        if not self.leg:
            if self.route is None:
                fixed = self.phase is None
                self.route = plan(obs, [standoff], self.clearance) if fixed else [standoff]
            last = self.k == len(self.route) - 1
            point = standoff if last else self.route[self.k]
            d = point - obs[:3]
            dist = float(np.linalg.norm(d))
            if dist < 3.0 and not last:
                self.k += 1
                last = self.k == len(self.route) - 1
                point = standoff if last else self.route[self.k]
                d = point - obs[:3]
                dist = float(np.linalg.norm(d))
            if (
                last
                and dist < 5.0
                and self.window_open(t, self.standoff / self.v_contact * 0.5 + 100.0)
            ):
                self.leg = True
            elif last and dist < 20.0:
                a += -(0.02**2) * (obs[:3] - point) - 2 * 0.02 * (obs[3:6] - v_standoff)
                return np.clip(a / self.max_accel, -1.0, 1.0) * self.scale
            else:
                v_des = (v_standoff if last else 0.0) + min(self.v_max, self.gain * dist) * d / max(
                    dist, 1e-9
                )
                a += self.k_v * (v_des - obs[3:6])
                return np.clip(a / self.max_accel, -1.0, 1.0) * self.scale
        # Inside the hard speed limit v0 + 4 n rho, whose slope is about 0.0044 1/s.
        speed = min(self.v_max, self.v_contact + 0.003 * max(depth, 0.0))
        # A point at this depth on a turning axis moves with the axis (Fehse 5.3.1).
        v_des = np.cross(spin, port + max(depth, 0.0) * axis) - speed * axis - 0.03 * lateral
        a += self.k_v * (v_des - obs[3:6]) if depth > 2.0 else 0.2 * (v_des - obs[3:6])
        return np.clip(a / self.max_accel, -1.0, 1.0) * self.scale


class Grapple:
    """Hold below an inertially pointing target and descend onto its axis when it comes round."""

    name = "grapple"

    def __init__(self, descend_speed=0.05, lead=GRAPPLE_LEAD, name="grapple"):
        self.descend_speed, self.lead, self.name = descend_speed, lead, name

    def reset(self, view, rng=None):
        self.n = view.n
        self.max_accel = view.max_accel
        self.scale = action_scale(view)
        goal = view.goal
        self.hold = np.array(goal["hold_point"], dtype=float)
        self.wait = np.array(goal["waypoints"][0], dtype=float)
        self.phase = float(goal["target"]["phase"])
        self.inner = Glide([self.wait], v_max=0.5, tol=2.0)
        self.inner.reset(view, rng)
        self.descending = False

    def axis_angle(self, t):
        """Angle (rad) from the axis to the hold direction, decreasing as the target turns."""
        angle = (self.phase - self.n * t) % (2 * math.pi)
        return angle if angle <= math.pi else angle - 2 * math.pi

    def act(self, obs, info):
        t = info["time_s"]
        a = -cw_terms(obs, self.n)
        waiting = float(np.linalg.norm(obs[:3] - self.wait)) < 3.0
        if not self.descending and waiting and 0.0 < self.axis_angle(t) < self.lead:
            self.descending = True
        if not self.descending:
            return self.inner.act(obs, info)
        d = self.hold - obs[:3]
        dist = float(np.linalg.norm(d))
        if dist < 1.5:
            a += -(0.05**2) * (obs[:3] - self.hold) - 2 * 0.05 * obs[3:6]
        else:
            v_des = min(self.descend_speed, 0.02 * dist) * d / dist
            a += 0.1 * (v_des - obs[3:6])
        return np.clip(a / self.max_accel, -1.0, 1.0) * self.scale


class Inspect:
    """Circle the target at `radius` while keeping the Sun's in-plane direction behind."""

    name = "inspect"

    def __init__(self, radius=60.0, name="inspect"):
        self.radius, self.name = radius, name

    def reset(self, view, rng=None):
        self.n = view.n
        sun = np.array(view.goal["sun_direction"], dtype=float)
        planar = sun[:2] / np.linalg.norm(sun[:2])
        theta0 = math.atan2(planar[1], planar[0])
        self.inner = Track(circle(self.radius, -self.n, theta0, math.inf, self.n))
        self.inner.reset(view, rng)

    def act(self, obs, info):
        return self.inner.act(obs, info)


class Closing:
    """Hold at S2 until the GO, then a half-orbit two-impulse transfer to S3."""

    name = "closing"

    def reset(self, view, rng=None):
        go = next(c["go_time"] for c in view.constraints if c["type"] == "zone_entry")
        period = 2 * math.pi / view.n
        hold = tuple(view.goal["hold_point"])
        self.inner = Lambert([(hold, go, go + 0.5 * period)])
        self.inner.reset(view, rng)

    def act(self, obs, info):
        return self.inner.act(obs, info)


class Torva:
    """The twice-orbital-rate quarter lap from +R-bar to +V-bar, then a glide to the hold."""

    name = "torva"

    def reset(self, view, rng=None):
        n = view.n
        radius = 575.0 * 0.3048
        t_end = math.pi / 2 / (2 * n)
        self.inner = Track(circle(radius, -2 * n, math.pi, t_end, n))
        self.t_end = t_end
        self.glide = Glide([tuple(view.goal["hold_point"])], v_max=0.2)
        self.glide.reset(view, rng)
        self.inner.reset(view, rng)

    def act(self, obs, info):
        if info["time_s"] < self.t_end:
            return self.inner.act(obs, info)
        return self.glide.act(obs, info)


class Approach:
    """Glide to the hold point around the keep-out sphere, reading both from the view."""

    name = "approach"

    def __init__(self, axis):
        self.axis = axis

    def reset(self, view, rng=None):
        hold = np.array(view.goal["hold_point"], dtype=float)
        keep_out = max(
            (c["radius"] for c in view.constraints if c["type"] == "keep_out"), default=0.0
        )
        side = 3.0 * keep_out
        if self.axis == "vbar":
            points = [(side, 2.0 * keep_out, 0.0), (side, hold[1], 0.0), tuple(hold)]
        elif self.axis == "hbar":
            points = [(0.0, side, 0.0), (0.0, side, hold[2]), tuple(hold)]
        else:
            points = [tuple(hold)]
        self.inner = Glide(points, clearance=1.2 * keep_out)
        self.inner.reset(view, rng)

    def act(self, obs, info):
        return self.inner.act(obs, info)


def reference_for(name: str, n: float = 0.0011458864877737333):
    """The reference controller of a built-in scenario, by its base name."""
    ft = 0.3048
    period = 2 * math.pi / n
    if name in ("approach-vbar", "approach-rbar", "approach-hbar"):
        return Approach(name.split("-")[1])
    if name in ("docking", "docking-offset", "inertial-target", "vbar-final-shuttle"):
        return Dock(clearance=60.0)
    if name == "docking-afrl":
        return Glide([(0.0, 0.0, 0.0)], v_max=0.15)
    if name == "inspection":
        return Inspect()
    if name == "closing":
        return Closing()
    if name == "rbar-acquisition":
        return Lambert([((-500.0, 0.0, 0.0), 0.0, 0.5 * period)])
    if name == "berthing-box":
        return Glide([(-30.0, 0.0, 0.0), (-15.0, 0.0, 0.0)], v_max=0.1, wn=0.03)
    if name == "orbt-rbar":
        return Lambert(
            [
                ((-1823.0 * ft, -885.0 * ft, 0.0), 0.0, 76.9 * 60.0),
                ((-600.0 * ft, 0.0, 0.0), 76.9 * 60.0, 89.9 * 60.0),
            ]
        )
    if name == "rpm-hold":
        return Glide([(-600.0 * ft, 0.0, 0.0)])
    if name == "torva":
        return Torva()
    if name == "hst-rbar-grapple":
        return Grapple()
    if name == "separation-flyaround":
        return Glide([(0.0, 30.0 + 130.0, 0.0), (0.0, 700.0, 0.0)], v_max=0.5)
    raise KeyError(name)
