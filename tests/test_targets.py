import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from vbar.constraints import ApproachCone, KeepOut, segment_minimum
from vbar.dynamics import EARTH, CWDynamics, cw_discrete, cw_stm, mean_motion
from vbar.scenarios.targets import (
    BodyKeepOut,
    FixedOffset,
    GravityGradientTumble,
    InertialPointing,
    LightingWindow,
    Maneuvering,
    ManeuveringDynamics,
    Misbriefed,
    Passive,
    RotatingCone,
    TorqueFree,
    Tumbling,
    body_point,
    body_point_velocity,
    burn_boundaries,
    rotation_z,
    split_at,
    sun_at,
    sun_direction,
    sunlit,
)

N = mean_motion(EARTH.radius + 500e3)
T = 2.0 * math.pi / N


def test_inertially_pointing_port_moves_at_the_lateral_speed_fehse_quotes():
    # Fehse 5.3.1: on the docking axis of an inertially pointing target the chaser needs a
    # lateral speed r n, 0.113 m/s at 100 m (0.1131 m/s with n at 400 km).
    mode = InertialPointing()
    r = 100.0
    axis = np.array([0.0, -1.0, 0.0])
    h = 1e-3
    p0 = r * body_point(mode, axis, 0.0, N)
    p1 = r * body_point(mode, axis, h, N)
    assert np.linalg.norm(p1 - p0) / h == pytest.approx(r * N, rel=1e-6)
    assert r * mean_motion(EARTH.radius + 400e3) == pytest.approx(0.1131, abs=5e-4)


def test_fixed_offset_puts_the_port_above_the_vbar_by_the_tangent_rule():
    # Fehse 5.7.3, p. 168: a port whose along-track distance is 20 m, on a target pitched
    # 30 deg, sits 20 tan(30 deg) = 11.547 m off the V-bar; here the port is 20 m from the
    # centre along the body axis, so a pitch of 30 deg puts it 20 sin(30 deg) = 10 m up.
    port = body_point(FixedOffset(math.radians(30.0)), (0.0, -20.0, 0.0), 0.0, N)
    assert port == pytest.approx([10.0, -20.0 * math.cos(math.radians(30.0)), 0.0])
    assert 20.0 * math.tan(math.radians(30.0)) == pytest.approx(11.547, abs=5e-4)
    cycle = FixedOffset(0.3, amplitude=math.radians(1.0), period=600.0)
    assert cycle.angle(150.0) == pytest.approx(0.3 + math.radians(1.0))
    assert cycle.rate_bound(N) == pytest.approx(2 * math.pi * math.radians(1.0) / 600.0)
    with pytest.raises(ValueError):
        FixedOffset(0.1, amplitude=0.1)


def test_tumbling_about_the_normal_returns_after_one_period_and_misbriefing_hides_it():
    mode = Tumbling((0.0, 0.0, 1.0), rate=0.01)
    point = (2.0, 0.0, 0.0)
    assert body_point(mode, point, 2 * math.pi / 0.01, N) == pytest.approx(point)
    assert body_point(mode, point, 0.25 * 2 * math.pi / 0.01, N) == pytest.approx([0.0, 2.0, 0.0])
    lied = Misbriefed(actual=mode, briefed=Passive())
    assert lied.public(N) == {"mode": "passive"}
    assert np.allclose(lied.matrix(10.0, N), mode.matrix(10.0, N))
    assert Passive().matrix([0.0, 1.0], N).shape == (2, 3, 3)


def test_rotating_cone_equals_the_fixed_cone_turned_by_the_attitude():
    half, axis, apex, active = math.radians(10.0), (0.0, -1.0, 0.0), (0.0, -2.0, 0.0), 50.0
    target = InertialPointing(phase=0.4)
    rotating = RotatingCone(half, axis, apex, active, target, N)
    t = 700.0
    m = target.matrix(t, N)
    fixed = ApproachCone(half, tuple(m @ axis), tuple(m @ apex), active)
    rng = np.random.default_rng(1)
    states = np.concatenate([rng.uniform(-60, 60, (200, 3)), rng.uniform(-0.1, 0.1, (200, 3))], 1)
    assert np.allclose(rotating.margin(states, t), fixed.margin(states, t), atol=1e-9)
    # The rate bound covers the cone's own motion: a chaser at rest inside the active range sees
    # the margin change only because the cone turns, at up to the attitude rate times its reach.
    h = 1e-3
    near = []
    while len(near) < 400:
        d = rng.normal(size=3)
        d *= rng.uniform(1.0, active) / np.linalg.norm(d)
        near.append(np.concatenate([np.asarray(fixed.apex) + d, np.zeros(3)]))
    near = np.array(near)
    rates = np.abs(rotating.margin(near, t + h) - rotating.margin(near, t)) / h
    assert np.max(rates) <= rotating.rate_bound(0.0, 0.0)
    # The bound is not loose by more than the reach's factor: points far out turn fastest.
    assert np.max(rates) > 0.5 * target.rate_bound(N) * active
    with pytest.raises(ValueError):
        RotatingCone(half, axis, apex, math.inf, target, N)


def test_rotating_cone_margin_is_certified_through_a_segment():
    # A chaser crossing the turning corridor: the certified minimum must not exceed the true
    # minimum found on a fine grid.
    target = InertialPointing()
    cone = RotatingCone(math.radians(10.0), (0.0, -1.0, 0.0), (0.0, 0.0, 0.0), 40.0, target, N)
    model = CWDynamics(N)
    start = np.array([-5.0, -30.0, 0.0, 0.02, 0.0, 0.0])
    model.reset(start)
    accel = np.zeros(3)
    end, samples = model.advance(accel, 600.0, n_samples=21)
    tau = np.linspace(0.0, 600.0, 21)
    found = segment_minimum(cone, model.state_in_last_segment, tau, samples, 0.0, N, 1000.0)
    fine = np.linspace(0.0, 600.0, 20001)
    states = np.einsum("tij,j->ti", cw_discrete(N, fine)[0], start)
    truth = float(np.min(cone.margin(states, 1000.0 + fine)))
    assert found.lower_bound <= truth + 1e-9
    assert found.margin == pytest.approx(truth, abs=1e-3)


class _WithoutRotation:
    """A turning constraint whose rate bound leaves out the target's rotation."""

    def __init__(self, constraint):
        self.constraint = constraint
        self.name = constraint.name
        self.resolution = constraint.resolution

    def margin(self, state, t=0.0):
        return self.constraint.margin(state, t)

    def rate_bound(self, speed, accel, rho_min=0.0):
        return speed


def _certified_minimum(constraint, start, duration, samples, t0):
    """The certified minimum of a free-drift segment from `samples` samples."""
    model = CWDynamics(N)
    model.reset(start)
    _, states = model.advance(np.zeros(3), duration, n_samples=samples)
    tau = np.linspace(0.0, duration, samples)
    return segment_minimum(constraint, model.state_in_last_segment, tau, states, 0.0, N, t0)


def _true_minimum(constraint, start, duration, t0):
    """The minimum over a free-drift segment: a fine grid, then each of its lowest dips refined."""
    fine = np.linspace(0.0, duration, 20001)
    path = np.einsum("tij,j->ti", cw_discrete(N, fine)[0], start)
    values = constraint.margin(path, t0 + fine)
    dips = [k for k in range(1, len(fine) - 1) if values[k] <= min(values[k - 1], values[k + 1])]
    best = float(values.min())
    for k in sorted(dips, key=lambda k: values[k])[:5]:
        found = minimize_scalar(
            lambda tau: float(constraint.margin(cw_stm(N, tau) @ start, t0 + tau)),
            bounds=(fine[k - 1], fine[k + 1]),
            method="bounded",
            options={"xatol": 1e-9},
        )
        best = min(best, float(found.fun))
    return best


@pytest.mark.parametrize(
    "target", [InertialPointing(0.3), Tumbling((0.0, 0.0, 1.0), 0.05)], ids=["inertial", "tumble"]
)
def test_turning_cone_certificate_holds_for_a_chaser_crossing_it_at_speed(target):
    # The docking corridor (apex set back 0.567 m, 50 m active) crossed at 5 cm/s while the
    # target turns at n or at 0.05 rad/s; 30 s between samples.
    cone = RotatingCone(math.radians(10.0), (0.0, -1.0, 0.0), (0.0, -2.567, 0.0), 50.567, target, N)
    start = np.array([-15.0, -30.0, 0.0, 0.05, 0.0, 0.0])
    truth = _true_minimum(cone, start, 600.0, 1000.0)
    found = _certified_minimum(cone, start, 600.0, 21, 1000.0)
    assert found.lower_bound <= truth
    assert found.margin == pytest.approx(truth, abs=1e-6)
    if isinstance(target, Tumbling):
        # Without the rotation term the bound sits above the true minimum: the term is needed.
        loose = _certified_minimum(_WithoutRotation(cone), start, 600.0, 21, 1000.0)
        assert loose.lower_bound > truth + 0.5


def test_turning_port_keep_out_certificate_needs_the_rotation_term():
    # A chaser drifting through the reception volume at the port while the body spins at
    # 0.16 rad/s about a skew axis; the open port turns away and back between the 15 s samples.
    target = Tumbling((0.07, 0.1, -0.39), 0.16)
    body = BodyKeepOut(2.0, (0.0, -2.0, 0.0), (0.0, -1.0, 0.0), target, N, 0.1)
    start = np.array([-0.027, -1.957, -0.146, 0.0011, -0.0025, 0.001])
    truth = _true_minimum(body, start, 60.0, 0.0)
    found = _certified_minimum(body, start, 60.0, 5, 0.0)
    assert truth < 0.0
    assert found.lower_bound <= truth
    assert found.margin == pytest.approx(truth, abs=1e-6)
    loose = _certified_minimum(_WithoutRotation(body), start, 60.0, 5, 0.0)
    assert loose.lower_bound > truth + 0.02
    # The same port on an inertially pointing target, crossed at 2 mm/s inside the depth.
    slow = BodyKeepOut(2.0, (0.0, -2.0, 0.0), (0.0, -1.0, 0.0), InertialPointing(), N, 0.1)
    crossing = np.array([-0.5, -1.97, 0.0, 0.002, 0.0, 0.0])
    truth = _true_minimum(slow, crossing, 600.0, 0.0)
    found = _certified_minimum(slow, crossing, 600.0, 21, 0.0)
    assert found.lower_bound <= truth
    assert found.margin == pytest.approx(truth, abs=1e-6)


def test_body_keep_out_is_open_at_the_port_and_closed_elsewhere():
    body = BodyKeepOut(2.0, (0.0, -2.0, 0.0), (0.0, -1.0, 0.0), Passive(), N)
    at_port = np.array([0.0, -2.0, 0.0, 0.0, 0.1, 0.0])
    assert body.margin(at_port) == pytest.approx(0.1, abs=1e-12)
    just_past = np.array([0.0, -1.95, 0.0, 0.0, 0.1, 0.0])
    assert body.margin(just_past) == pytest.approx(0.05, abs=1e-12)
    deep = np.array([0.0, -1.5, 0.0, 0.0, 0.0, 0.0])
    assert body.margin(deep) < 0.0
    in_front = np.array([0.05, -2.5, 0.0, 0.0, 0.0, 0.0])
    assert body.margin(in_front) > 0.0
    inside = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    assert body.margin(inside) < 0.0
    behind_plane_outside = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    assert body.margin(behind_plane_outside) == pytest.approx(
        KeepOut(2.0).margin(behind_plane_outside)
    )
    assert body.rate_bound(0.3, 0.02) == pytest.approx(0.3)
    # Far in front of the port the clipped depth term is a constant, so a turning port plane
    # cannot move the margin faster than the bound: a chaser at rest 4 m off the axis 100 m out.
    turning = BodyKeepOut(1.0, (0.0, -1.0, 0.0), (0.0, -1.0, 0.0), InertialPointing(), N)
    far = np.array([4.0, -101.0, 0.0, 0.0, 0.0, 0.0])
    h = 1e-3
    rate = (turning.margin(far, h) - turning.margin(far, 0.0)) / h
    assert abs(rate) <= turning.rate_bound(0.0, 0.0)
    near = np.array([0.3, -1.05, 0.0, 0.0, 0.0, 0.0])
    rate = (turning.margin(near, h) - turning.margin(near, 0.0)) / h
    assert abs(rate) <= turning.rate_bound(0.0, 0.0)


def test_a_turning_port_plane_moves_the_keep_out_margin_at_w_dot_a_cross_r():
    # At rest, d/dt [(r - R p) . R a] = -(w x R p) . R a + (r - R p) . (w x R a) = w . (R a x r):
    # the port's own offset cancels, so the depth term turns at most at |w| rho, below
    # |w| (R + d_r) where it is the larger term, and the bound's |port| is only margin. The port
    # sits on the sphere off the body axes, its axis not along it, on a fast tumble.
    target = TorqueFree((1000.0, 4200.0, 4700.0), 4650.0, math.radians(2.0), 0.3, (0.2, -0.4, 0.9))
    radius, reception, port, axis = 1.5, 0.1, (-1.4, 0.5, 0.2), (-1.0, 0.0, 0.0)
    body = BodyKeepOut(radius, port, axis, target, N, reception)
    rng = np.random.default_rng(3)
    r = np.array(port) + rng.normal(scale=0.3, size=(4000, 3))
    r = r[np.linalg.norm(r, axis=1) <= radius + reception]
    t = rng.uniform(0.0, 600.0, len(r))
    states = np.concatenate([r, np.zeros_like(r)], axis=1)
    h = 1e-4
    rate = (body.margin(states, t + h) - body.margin(states, t - h)) / (2.0 * h)
    depth = np.sum((r - body_point(target, port, t, N)) * body_point(target, axis, t, N), axis=1)
    rho = np.linalg.norm(r, axis=1)
    # Behind the port plane, where the depth term is the larger one by a step's worth.
    active = (depth < -1e-3) & (depth + reception > rho - radius + 1e-3)
    assert 100 < np.sum(active) < len(r)
    w = target.angular_velocity(t, N)
    a = body_point(target, axis, t, N)
    predicted = np.sum(w * np.cross(a, r), axis=1)
    # Central differences at h = 1e-4 s leave an error of order |w|^3 rho h^2.
    assert rate[active] == pytest.approx(predicted[active], abs=1e-9)
    assert np.max(np.abs(rate)) <= target.rate_bound(N) * (radius + reception)
    assert np.max(np.abs(rate)) <= body.rate_bound(0.0, 0.0)


def test_sun_direction_and_eclipse_fraction_match_montenbruck():
    # Montenbruck and Gill, sec. 3.4: with the Sun in the orbit plane a 500 km target is in
    # shadow 37.8% of each orbit, 23.1% at beta = 60 deg.
    for beta, fraction in ((0.0, 0.378), (math.radians(60.0), 0.231)):
        sun0 = sun_direction(beta, 0.3)
        times = np.linspace(0.0, T, 20001)[:-1]
        lit = sunlit(sun0, N, times, EARTH.radius + 500e3)
        assert 1.0 - np.mean(lit) == pytest.approx(fraction, abs=2e-3)
    s0 = sun_direction(0.2, 0.5)
    assert np.linalg.norm(s0) == pytest.approx(1.0)
    assert math.asin(s0[2]) == pytest.approx(0.2)
    # The Hill frame turns at n, so the Sun comes back after one orbit.
    assert np.allclose(sun_at(s0, N, T), s0)


def test_lighting_window_opens_only_near_the_port_inside_the_window_on_the_day_side():
    sun0 = tuple(sun_direction(0.0, math.radians(90.0 + 180.0)))  # the Sun along -y at t = 0
    window = LightingWindow(
        (math.radians(15.0), math.radians(75.0)),
        (0.0, -1.0, 0.0),
        (0.0, -2.0, 0.0),
        10.0,
        sun0,
        N,
        EARTH.radius + 500e3,
    )
    near = np.array([0.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    far = np.array([0.0, -60.0, 0.0, 0.0, 0.0, 0.0])
    assert window.sun_angle(0.0) == pytest.approx(0.0)
    assert window.margin(near, 0.0) < 0.0 < window.margin(far, 0.0)
    # With the Sun 45 deg off the axis on the day side (x > 0) the window is open near the port.
    t45 = math.radians(315.0) / N
    assert window.sun_angle(t45) == pytest.approx(math.radians(45.0))
    assert window.margin(near, t45) > 0.0
    # The margin never changes faster than the rate bound, on this axis and on an R-bar axis
    # whose window is open while the Sun crosses the terminator.
    times = np.linspace(0.0, T, 4001)
    values = window.margin(np.tile(near, (len(times), 1)), times)
    assert np.max(np.abs(np.diff(values))) / (times[1] - times[0]) <= window.rate_bound(0.0, 0.0)
    rbar = LightingWindow(
        (math.radians(30.0), math.radians(150.0)),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        10.0,
        tuple(sun_direction(0.0, 0.0)),
        N,
        EARTH.radius + 500e3,
    )
    close = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    values = rbar.margin(np.tile(close, (len(times), 1)), times)
    assert np.max(np.abs(np.diff(values))) / (times[1] - times[0]) <= rbar.rate_bound(0.0, 0.0)
    assert np.any(values > 0.0) and np.any(values < 0.0)


def test_a_lighting_window_needs_an_axis_with_an_in_plane_part():
    # An axis along the orbit normal has no in-plane angle to the Sun: refused, not NaN.
    with pytest.raises(ValueError):
        LightingWindow(
            (0.2, 1.2),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 2.0),
            10.0,
            (1.0, 0.0, 0.0),
            N,
            EARTH.radius + 500e3,
        )


def test_a_target_burn_is_the_opposite_acceleration_on_the_relative_state():
    burn = (100.0, 50.0, (0.0, 4e-4, 0.0))
    mode = Maneuvering(schedule=(burn,))
    moving = ManeuveringDynamics(CWDynamics(N), mode)
    plain = CWDynamics(N)
    start = np.array([10.0, -300.0, 0.0, 0.0, 0.01, 0.0])
    moving.reset(start)
    plain.reset(start)
    accel = np.array([1e-4, 0.0, 0.0])
    for duration, extra in ((100.0, np.zeros(3)), (50.0, -np.array(burn[2])), (100.0, np.zeros(3))):
        a, _ = moving.advance(accel, duration)
        b, _ = plain.advance(accel + extra, duration)
        assert np.allclose(a, b, atol=1e-12)
    with pytest.raises(ValueError):
        ManeuveringDynamics(CWDynamics(N), mode).advance(accel, 120.0)
    assert burn_boundaries(mode) == (100.0, 150.0)
    pieces = split_at([(10.0, np.zeros(3)), (200.0, np.ones(3))], 0.0, (100.0, 150.0))
    assert [d for d, _ in pieces] == pytest.approx([10.0, 90.0, 50.0, 60.0])
    with pytest.raises(ValueError):
        Maneuvering(schedule=((0.0, 10.0, (0, 0, 0)), (5.0, 10.0, (0, 0, 0))))


def test_free_drift_identity_holds_for_the_wrapped_model():
    moving = ManeuveringDynamics(
        CWDynamics(N), Maneuvering(schedule=((900.0, 10.0, (1e-4, 0, 0)),))
    )
    start = np.array([5.0, -100.0, 3.0, 0.001, 0.0, -0.002])
    moving.reset(start)
    end, _ = moving.advance(np.zeros(3), 600.0)
    assert np.allclose(end, cw_stm(N, 600.0) @ start, atol=1e-9)


@pytest.mark.parametrize(
    "mode",
    [
        Passive(),
        FixedOffset(0.3, amplitude=math.radians(2.0), period=500.0),
        InertialPointing(0.4),
        Tumbling((0.3, -1.0, 0.5), 0.07, phase=0.2),
        Maneuvering(schedule=((10.0, 5.0, (0.0, 1e-4, 0.0)),), attitude=InertialPointing(1.0)),
        Misbriefed(actual=Tumbling((0.0, 1.0, 1.0), -0.02), briefed=Passive()),
        TorqueFree((1000.0, 4200.0, 4700.0), 4300.0, 0.03, 0.2, (0.3, -0.5, 0.8), 1.0),
        TorqueFree((1000.0, 4200.0, 4700.0), 1300.0, 0.05, 0.7, (0.0, 0.0, -1.0), 2.0),
        # A pure spin about the major axis with H off z: only the Hill frame's turn moves its rate.
        TorqueFree((1000.0, 4200.0, 4700.0), 4700.0, 0.03, 0.0, (1.0, 0.0, 0.0)),
        GravityGradientTumble((1000.0, 4200.0, 4700.0), 4650.0, 0.004, 0.3, (0.2, -0.4, 0.9), 0.7),
    ],
    ids=lambda mode: mode.name,
)
def test_angular_velocity_is_the_rate_of_the_attitude_matrix(mode):
    # dR/dt R^T is the skew matrix of the body rate in Hill axes; the central difference at 1e-3 s
    # is off by about w (w h)^2 / 6, about 6e-11 rad/s at the fastest rate here (0.07 rad/s),
    # inside the 1e-9 rad/s tolerance. The times stay after t = 0, before which an integrated
    # tumble is not defined.
    times = np.array([1.0, 137.0, 911.0])
    h = 1e-3
    skew = (mode.matrix(times + h, N) - mode.matrix(times - h, N)) / (2.0 * h)
    skew = skew @ np.swapaxes(mode.matrix(times, N), -1, -2)
    w = np.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], axis=-1)
    assert mode.angular_velocity(times, N) == pytest.approx(w, abs=1e-9)
    assert np.all(np.linalg.norm(w, axis=-1) <= mode.rate_bound(N) * (1.0 + 1e-6))
    point = (0.5, -2.0, 1.0)
    moved = (body_point(mode, point, times + h, N) - body_point(mode, point, times - h, N)) / (
        2 * h
    )
    assert body_point_velocity(mode, point, times, N) == pytest.approx(moved, abs=1e-9)
    # The rate's derivative stays within its bound over a dense set of times.
    dense = np.linspace(0.0, 2000.0, 4001)
    spin_up = np.diff(mode.angular_velocity(dense, N), axis=0) / np.diff(dense)[:, None]
    assert np.max(np.linalg.norm(spin_up, axis=-1)) <= mode.angular_acceleration_bound(N) + 1e-12


def test_a_torque_free_tumble_keeps_its_momentum_fixed_inertially():
    # Schaub and Junkins, section 4.2.1: H is constant in inertial space only, so in the Hill
    # frame, which turns at n, it turns at -n about z.
    mode = TorqueFree((1000.0, 4200.0, 4700.0), 4300.0, 0.03, 0.2, (0.3, -0.5, 0.8), 1.0)
    times = np.linspace(0.0, 3000.0, 301)
    moments = np.array(mode.moments)
    h_hill = np.einsum("tij,tj->ti", mode.matrix(times, N), moments * mode.motion.omega(times))
    expected = np.einsum("tij,j->ti", rotation_z(-N * times), mode.motion.momentum)
    assert h_hill == pytest.approx(expected, abs=1e-10 * mode.motion.h)
    assert mode.motion.h == pytest.approx(4300.0 * 0.03)
    direction = np.array([0.3, -0.5, 0.8]) / math.sqrt(0.98)
    assert mode.motion.momentum / mode.motion.h == pytest.approx(direction)
    # The public description rebuilds the same tumble, so an agent can predict it.
    public = mode.public(N)
    keys = ("moments", "effective_inertia", "rate", "phase", "momentum_direction", "spin_angle")
    rebuilt = TorqueFree(*(public[key] for key in keys))
    assert rebuilt.matrix(times, N) == pytest.approx(mode.matrix(times, N), abs=1e-12)
    assert public["omega"] == pytest.approx(mode.motion.omega(0.0))
    with pytest.raises(ValueError):
        TorqueFree((1000.0, 4200.0, 4700.0), 4200.0, 0.03)
    with pytest.raises(ValueError):
        TorqueFree((1000.0, 4200.0, 4700.0), 4300.0, 0.03, momentum_direction=(0.0, 0.0, 0.0))


def test_the_gravity_gradient_tumble_starts_as_briefed_and_departs_from_the_torque_free_one():
    # Both modes start from the same briefed state; the torque then turns a slow tumble away from
    # the torque-free motion within an orbit, while a body with no orbit (n = 0) follows it.
    keys = dict(moments=(1000.0, 4200.0, 4700.0), effective_inertia=4650.0, rate=0.004, phase=0.3)
    torqued = GravityGradientTumble(**keys, momentum_direction=(0.2, -0.4, 0.9), spin_angle=0.7)
    free = TorqueFree(**keys, momentum_direction=(0.2, -0.4, 0.9), spin_angle=0.7)
    assert torqued.matrix(0.0, N) == pytest.approx(free.matrix(0.0, N), abs=1e-15)
    assert torqued.public(N) == {**free.public(N), "mode": "gravity_gradient"}
    later = np.array([3000.0])
    apart = np.linalg.norm(torqued.matrix(later, N) - free.matrix(later, N))
    assert apart > 1e-3
    assert torqued.matrix(later, 0.0) == pytest.approx(free.matrix(later, 0.0), abs=1e-8)
