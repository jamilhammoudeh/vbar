import math

import numpy as np
import pytest

from vbar.constraints import ApproachCone, KeepOut, segment_minimum
from vbar.dynamics import EARTH, CWDynamics, cw_discrete, cw_stm, mean_motion
from vbar.scenarios.targets import (
    BodyKeepOut,
    FixedOffset,
    InertialPointing,
    LightingWindow,
    Maneuvering,
    ManeuveringDynamics,
    Misbriefed,
    Passive,
    RotatingCone,
    Tumbling,
    body_point,
    burn_boundaries,
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
