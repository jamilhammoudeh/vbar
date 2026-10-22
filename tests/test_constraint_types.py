import numpy as np
import pytest
from scipy.optimize import minimize

from vbar import constraints as constraints_module
from vbar.agents import PDAgent
from vbar.constraints import (
    ApproachCone,
    ClosingSpeedLimit,
    GateTable,
    PassiveSpeed,
    SpeedLimit,
    SunExclusion,
    ZoneEntry,
    closing_speed,
    interval_bounds,
    segment_minima,
    segment_minimum,
)
from vbar.dynamics import EARTH, CWDynamics, cw_discrete, mean_motion
from vbar.scenarios.approach import ApproachConfig, ApproachEnv

N = mean_motion(EARTH.radius + 500e3)
# Dunlap et al. 2021, eq. 14: v0 = 0.2 m/s, v1 = 4 n.
SPEED = SpeedLimit(0.2, 4 * N)
CLOSING = ClosingSpeedLimit(0.2, 4 * N)
# Docking corridor: 10 deg from a port 2 m behind the target, active within 50 m (Fehse 5.6).
CONE = ApproachCone(
    np.radians(10.0), axis=(0.0, -1.0, 0.0), apex=(0.0, -2.0, 0.0), active_range=50.0
)
FT = 0.3048
# STS-130 +V-bar final approach gates, range ft and closing rate ft/s (Goodman, fig. 25.22).
STS130_VBAR = ((320.0, 0.20), (110.0, 0.15), (75.0, 0.10), (30.0, 0.07))
GATES = GateTable(tuple((r * FT, v * FT) for r, v in STS130_VBAR))


def flyby(seed):
    rng = np.random.default_rng(seed)
    position = rng.normal(size=3)
    position *= rng.uniform(20.0, 200.0) / np.linalg.norm(position)
    direction = -position / np.linalg.norm(position) + rng.normal(scale=0.5, size=3)
    state = np.r_[position, rng.uniform(0.1, 3.0) * direction / np.linalg.norm(direction)]
    accel = rng.uniform(-0.02, 0.02, 3) if seed % 2 else np.zeros(3)
    return state, accel, rng.uniform(5.0, 30.0)


def found_and_dense(constraint, seed, samples=20):
    state, accel, duration = flyby(seed)
    m = CWDynamics(N)
    m.reset(state)
    _, sampled = m.advance(accel, duration, n_samples=samples)
    times = np.linspace(0.0, duration, samples)
    thrust = float(np.linalg.norm(accel))
    found = segment_minimum(constraint, m.state_in_last_segment, times, sampled, thrust, N)
    grid = np.linspace(0.0, duration, 4001)
    phi, gamma = cw_discrete(N, grid)
    dense = phi @ state + gamma @ accel
    return found, constraint.margin(dense, grid)


def test_speed_limit_margin_is_the_allowed_minus_the_actual_speed():
    state = np.array([0.0, -100.0, 0.0, 0.3, 0.4, 0.0])
    assert SPEED.margin(state) == pytest.approx(0.2 + 400 * N - 0.5)


def test_closing_speed_counts_only_motion_toward_the_target():
    approaching = np.array([0.0, -100.0, 0.0, 0.1, 0.3, 0.0])
    receding = np.array([0.0, -100.0, 0.0, 0.1, -0.3, 0.0])
    crossing = np.array([0.0, -100.0, 0.0, 0.5, 0.0, 0.0])
    assert closing_speed(approaching) == pytest.approx(0.3)
    assert closing_speed(receding) == 0.0
    assert closing_speed(crossing) == 0.0
    assert CLOSING.margin(approaching) == pytest.approx(0.2 + 400 * N - 0.3)


def test_closing_speed_rate_bound_needs_a_range_bound():
    assert CLOSING.rate_bound(1.0, 0.0) == np.inf
    assert CLOSING.rate_bound(1.0, 0.01, rho_min=10.0) == pytest.approx(4 * N + 0.1 + 0.01)


def cone_distance_by_search(point, cone):
    """Signed distance to the cone surface by a direct search over its generators."""
    axis = np.asarray(cone.axis) / np.linalg.norm(cone.axis)
    e1 = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    d = np.asarray(point) - np.asarray(cone.apex)
    theta = cone.half_angle

    def gap(p):
        s, psi = max(p[0], 0.0), p[1]
        g = np.cos(theta) * axis + np.sin(theta) * (np.cos(psi) * e1 + np.sin(psi) * e2)
        return np.linalg.norm(d - s * g)

    starts = [
        (s, psi)
        for s in np.linspace(0.0, 2.0 * np.linalg.norm(d), 9)
        for psi in np.linspace(0, 6, 7)
    ]
    best = min(
        (
            minimize(gap, x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
            for x0 in starts
        ),
        key=lambda r: r.fun,
    )
    rho = np.linalg.norm(d)
    inside = rho == 0.0 or d @ axis / rho >= np.cos(theta)
    return best.fun if inside else -best.fun


def test_gate_table_interpolates_between_gates_and_holds_beyond_them():
    gates = GateTable(((100.0, 0.3), (10.0, 0.0), (300.0, 0.5)))
    assert gates.allowed_speed([5.0, 10.0, 55.0, 200.0, 400.0]) == pytest.approx(
        [0.0, 0.0, 0.15, 0.4, 0.5]
    )
    # Closing at 0.1 m/s from 55 m: allowed 0.15 there.
    assert gates.margin(np.array([0.0, 55.0, 0.0, 0.0, -0.1, 0.0])) == pytest.approx(0.05)


def test_gate_table_bound_holds_across_a_steep_gate():
    # 0.45 m/s falls to 0.05 m/s over 10 m; closing at 0.3 m/s from 205 m crosses it in 60 s
    # with four samples, so the bound must carry the 0.045 1/s slope as well as the speed change.
    gates = GateTable(((200.0, 0.5), (190.0, 0.05)))
    state = np.array([0.0, 205.0, 0.0, 0.0, -0.3, 0.0])
    m = CWDynamics(N)
    m.reset(state)
    _, sampled = m.advance(np.zeros(3), 60.0, n_samples=4)
    found = segment_minimum(gates, m.state_in_last_segment, np.linspace(0, 60, 4), sampled, 0.0, N)
    grid = np.linspace(0.0, 60.0, 20001)
    dense = gates.margin(cw_discrete(N, grid)[0] @ state, grid).min()
    assert found.lower_bound <= dense
    assert found.margin == pytest.approx(dense, abs=1e-5)


def test_shuttle_rbar_gates_sit_three_to_eleven_times_below_the_afrl_law():
    # Goodman, table 24.3, +R-bar from ORBT (ft, ft/s), against 0.2 + 4 n rho at 500 km
    # (Dunlap et al. 2021, eq. 14).
    orbt = ((2000, 3.0), (1000, 1.5), (600, 0.8), (500, 0.5), (200, 0.2), (100, 0.1))
    rho = np.array([r for r, _ in orbt]) * FT
    gates = GateTable(tuple((r * FT, v * FT) for r, v in orbt))
    ratio = CLOSING.allowed_speed(rho) / gates.allowed_speed(rho)
    assert ratio == pytest.approx([3.2, 3.4, 4.1, 5.7, 7.7, 11.0], abs=0.05)


def test_gate_table_rejects_a_single_gate_repeated_ranges_and_negative_speeds():
    for gates in (
        ((10.0, 0.1),),
        ((10.0, 0.1), (10.0, 0.2)),
        ((10.0, -0.1), (20.0, 0.1)),
        ((-10.0, 0.1), (20.0, 0.1)),
        ((float("nan"), 0.1), (20.0, 0.1)),
    ):
        with pytest.raises(ValueError):
            GateTable(gates)
    listed = GateTable([[10.0, 0.1], [20.0, 0.2]])
    assert hash(listed) == hash(GateTable(((10.0, 0.1), (20.0, 0.2))))


def test_shuttle_vbar_gates_sit_inside_the_vbar_passive_speed_envelope():
    # Goodman, fig. 25.22: envelope n rho / 0.47755 = 2.094 n rho at 185 n.mi. over the flown gate.
    n = mean_motion(EARTH.radius + 185.0 * 1852.0)
    rho = np.array([r for r, _ in STS130_VBAR]) * FT
    ratio = PassiveSpeed("vbar", 0.0, n).speed_envelope(rho) / GATES.allowed_speed(rho)
    assert ratio == pytest.approx([3.84, 1.76, 1.80, 1.03], abs=0.01)


@pytest.mark.parametrize("seed", range(12))
def test_cone_margin_is_the_signed_distance_to_the_cone(seed):
    rng = np.random.default_rng(seed)
    cone = ApproachCone(
        rng.uniform(0.1, 1.4), axis=tuple(rng.normal(size=3)), apex=tuple(rng.normal(size=3))
    )
    point = np.asarray(cone.apex) + rng.normal(size=3) * rng.uniform(1.0, 30.0)
    expected = cone_distance_by_search(point, cone)
    assert cone.margin(np.r_[point, 0.0, 0.0, 0.0]) == pytest.approx(expected, abs=1e-6)


def test_cone_margin_behind_the_apex_is_minus_the_distance_to_it():
    # Boyd and Vandenberghe, exercise 8.3: past phi = theta + pi/2 the apex is the nearest point.
    behind = np.array([0.0, 8.0, 3.0, 0.0, 0.0, 0.0])
    assert CONE.margin(behind) == pytest.approx(-np.hypot(10.0, 3.0))


def test_cone_margin_on_the_axis_and_on_the_surface():
    on_axis = np.array([0.0, -32.0, 0.0, 0.0, 0.0, 0.0])
    assert CONE.margin(on_axis) == pytest.approx(30.0 * np.sin(np.radians(10.0)))
    on_surface = np.array([30.0 * np.tan(np.radians(10.0)), -32.0, 0.0, 0.0, 0.0, 0.0])
    assert CONE.margin(on_surface) == pytest.approx(0.0, abs=1e-12)


def test_cone_turns_on_without_a_jump_at_its_active_range():
    # Far off the axis, the margin is the distance still to go to the active range.
    lateral = 40.0
    along = np.sqrt(50.0**2 - lateral**2)
    edge = np.array([lateral, -2.0 - along, 0.0, 0.0, 0.0, 0.0])
    step = np.array([0.0, -1e-6, 0.0, 0.0, 0.0, 0.0])
    assert CONE.margin(edge) == pytest.approx(0.0, abs=1e-9)
    assert abs(CONE.margin(edge + step) - CONE.margin(edge - step)) < 1e-5
    assert CONE.margin(np.array([0.0, -200.0, 100.0, 0.0, 0.0, 0.0])) > 0.0


def test_cone_rejects_a_half_angle_outside_its_range():
    with pytest.raises(ValueError, match="half_angle"):
        ApproachCone(np.pi / 2)


SUN = SunExclusion(np.radians(20.0), sun_direction=(0.3, 1.0, 0.2), rate=N)
PASSIVE_VBAR = PassiveSpeed("vbar", 15.0, N)
PASSIVE_RBAR = PassiveSpeed("rbar", 15.0, N)
ZONE = ZoneEntry((30.0, 60.0, 30.0))


def test_sun_exclusion_margin_is_the_angle_between_line_of_sight_and_sun_minus_the_half_angle():
    sun = SunExclusion(np.radians(20.0), sun_direction=(0.0, 1.0, 0.0), rate=N)
    behind = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    # Looking along +y at the target, straight into the Sun.
    assert sun.margin(behind, 0.0) == pytest.approx(-np.radians(20.0))
    above = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert sun.margin(above, 0.0) == pytest.approx(np.radians(70.0))


def test_sun_turns_backward_in_the_hill_frame_by_a_quarter_turn_per_quarter_orbit():
    sun = SunExclusion(np.radians(20.0), sun_direction=(0.0, 1.0, 0.0), rate=N)
    assert np.allclose(sun.sun_at(0.5 * np.pi / N), [1.0, 0.0, 0.0], atol=1e-12)
    below = np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert sun.margin(below, 0.5 * np.pi / N) == pytest.approx(-np.radians(20.0))


N400 = mean_motion(EARTH.radius + 400e3)


@pytest.mark.parametrize(("height", "envelope"), [(50.0, 0.1225), (30.0, 0.0612), (20.0, 0.0283)])
def test_rbar_passive_speed_envelope_matches_fehse_derived_values(height, envelope):
    # 15 m of structure at 400 km; Fehse 5.7.2 flies 0.1, 0.05, 0.02 m/s inside this envelope.
    rbar = PassiveSpeed("rbar", 15.0, N400)
    assert rbar.speed_envelope(height) == pytest.approx(envelope, abs=5e-5)


@pytest.mark.parametrize("height", [20.0, 50.0, 300.0])
def test_free_drift_from_the_rbar_envelope_just_touches_the_structure(height):
    rbar = PassiveSpeed("rbar", 15.0, N400)
    v = float(rbar.speed_envelope(height))
    start = np.array([-height, 0.0, 0.0, v, 0.0, 0.0])
    phi, _ = cw_discrete(N400, np.linspace(0.0, 2 * np.pi / N400, 200001))
    assert np.abs(phi @ start)[:, 0].min() == pytest.approx(15.0, rel=1e-8)
    assert rbar.margin(start) == pytest.approx(0.0, abs=1e-15)


def test_vbar_coast_after_thrust_stops_is_0_47755_v_over_n():
    # Fehse eq. 3.28, fig. 3.13: 0.1 m/s at 400 km coasts forward 42.21 m.
    phi, _ = cw_discrete(N400, np.linspace(0.0, 2 * np.pi / N400, 200001))
    coast = (phi @ np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0]))[:, 1].max()
    assert coast == pytest.approx(42.21, abs=5e-3)
    assert coast == pytest.approx(0.47755 * 0.1 / N400, rel=1e-5)


@pytest.mark.parametrize("gap", [5.0, 50.0, 500.0])
def test_free_drift_from_the_vbar_envelope_just_reaches_the_structure(gap):
    vbar = PassiveSpeed("vbar", 15.0, N400)
    v = float(vbar.speed_envelope(15.0 + gap))
    start = np.array([0.0, -15.0 - gap, 0.0, 0.0, v, 0.0])
    phi, _ = cw_discrete(N400, np.linspace(0.0, 2 * np.pi / N400, 200001))
    assert (phi @ start)[:, 1].max() == pytest.approx(-15.0, abs=1e-6 * gap)
    assert vbar.margin(start) == pytest.approx(0.0, abs=1e-15)


def test_passive_speed_rejects_an_unknown_axis():
    with pytest.raises(ValueError, match="axis"):
        PassiveSpeed("hbar", 15.0, N)


def test_zone_margin_is_zero_on_the_ellipsoid_and_infinite_after_go():
    iss = ZoneEntry((1000.0, 2000.0, 1000.0), go_time=600.0)
    surface = np.array([0.0, -2000.0, 0.0, 0.0, 0.0, 0.0])
    assert iss.margin(surface, 0.0) == pytest.approx(0.0)
    assert iss.margin(np.array([0.0, -3000.0, 0.0, 0, 0, 0]), 0.0) == pytest.approx(500.0)
    assert iss.margin(surface * 0.5, 600.0) < 0.0
    assert iss.margin(surface * 0.5, 600.001) == np.inf


@pytest.mark.parametrize("seed", range(8))
def test_zone_margin_lies_below_the_distance_to_the_ellipsoid(seed):
    rng = np.random.default_rng(seed)
    axes = rng.uniform(10.0, 100.0, 3)
    zone = ZoneEntry(tuple(axes))
    point = rng.normal(size=3) * 150.0
    u = rng.normal(size=(200000, 3))
    surface = axes * u / np.linalg.norm(u, axis=1, keepdims=True)
    distance = np.linalg.norm(surface - point, axis=1).min()
    margin = float(zone.margin(np.r_[point, 0.0, 0.0, 0.0]))
    if margin > 0.0:
        assert margin <= distance
    sphere = ZoneEntry((40.0, 40.0, 40.0))
    assert sphere.margin(np.r_[point, 0, 0, 0]) == pytest.approx(np.linalg.norm(point) - 40.0)


@pytest.mark.parametrize(
    "constraint",
    [SPEED, CLOSING, GATES, CONE, SUN, PASSIVE_VBAR, PASSIVE_RBAR, ZONE],
    ids=lambda c: f"{c.name}_{getattr(c, 'axis', '')}",
)
@pytest.mark.parametrize("seed", range(12))
def test_certificate_and_refined_minimum_bracket_the_dense_minimum(constraint, seed):
    found, dense = found_and_dense(constraint, seed)
    assert np.isfinite(found.lower_bound)
    # The bound can equal the last sample's margin, while the dense reference computes that state
    # through the closed-form matrices instead of the propagator; the two agree to about 1e-9 m
    # and 1e-12 m/s, so the margins agree to well under 1e-10 of their size.
    assert found.lower_bound <= dense.min() + 1e-10 * max(1.0, float(np.abs(dense).max()))
    # 4001 samples over at most 30 s miss a smooth minimum by far less than 1e-5 m/s.
    assert found.margin <= dense.min() + 1e-6
    assert found.margin >= found.lower_bound


def test_vector_parameters_given_as_lists_or_arrays_are_held_as_tuples():
    axis = [0.0, -1.0, 0.0]
    cone = ApproachCone(0.2, axis=axis, apex=np.array([0.0, -2.0, 0.0]))
    zone = ZoneEntry([1000.0, 2000.0, 1000.0], center=np.zeros(3))
    sun = SunExclusion(0.1, [1.0, 0.0, 0.0], N)
    axis[1] = 1.0
    assert cone.axis == (0.0, -1.0, 0.0) and cone.apex == (0.0, -2.0, 0.0)
    assert zone.semi_axes == (1000.0, 2000.0, 1000.0) and zone.center == (0.0, 0.0, 0.0)
    assert sun.sun_direction == (1.0, 0.0, 0.0)
    assert len({cone, zone, sun}) == 3
    with pytest.raises(ValueError, match="three components"):
        ZoneEntry([1000.0, 2000.0])


@pytest.mark.parametrize("axis", ["vbar", "rbar"])
@pytest.mark.parametrize("rho_min", [16.0, 20.0, 50.0, 300.0])
def test_passive_speed_rate_bound_uses_the_envelope_slope_at_the_range_bound(axis, rho_min):
    # Beyond the closing-speed term the bound is the envelope's slope at rho_min times the speed;
    # the R-bar slope falls with range, so rho_min gives the largest one over the interval.
    passive = PassiveSpeed(axis, 15.0, N400)
    speed, accel = 0.3, 1e-3
    envelope_term = passive.rate_bound(speed, accel, rho_min) - ClosingSpeedLimit(
        0.0, 0.0
    ).rate_bound(speed, accel, rho_min)
    h = 1e-4
    slope = (passive.speed_envelope(rho_min + h) - passive.speed_envelope(rho_min - h)) / (2 * h)
    assert envelope_term == pytest.approx(slope * speed, rel=1e-6)


SPEED_TYPES = [SPEED, CLOSING, GATES]


def speed_type_segments(count=180, seed=5):
    """CW segments that stress the speed bounds: flybys coasting and thrusting, velocity
    reversals under thrust, near-zero speeds, straight radial runs and passes within half a
    metre of the target."""
    rng = np.random.default_rng(seed)
    segments = []
    for k in range(count):
        duration = rng.uniform(5.0, 30.0)
        samples = int(rng.integers(2, 21))
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        kind = k % 6
        if kind in (0, 1):
            r = rng.uniform(20.0, 200.0) * rng.normal(size=3) / np.sqrt(3.0)
            v = rng.uniform(0.1, 3.0) * (-r / np.linalg.norm(r) + rng.normal(scale=0.5, size=3))
            a = rng.uniform(-0.02, 0.02, 3) if kind else np.zeros(3)
        elif kind == 2:
            r = rng.uniform(10.0, 100.0) * rng.normal(size=3)
            v = rng.uniform(0.0, 0.3) * d
            a = -0.02 * d
        elif kind == 3:
            r = rng.uniform(10.0, 100.0) * rng.normal(size=3)
            v = rng.uniform(0.0, 1e-4) * d
            a = rng.uniform(-1e-3, 1e-3, 3)
        elif kind == 5:
            # Straight radial motion under thrust along the line: |v| is linear but for the
            # CW terms, so only the jerk allowance separates the margin from its chord.
            r = rng.uniform(20.0, 300.0) * d
            v = rng.uniform(0.05, 3.0) * d * rng.choice([-1.0, 1.0])
            a = rng.uniform(0.0, 0.02) * np.sign(v @ d) * d
        else:
            speed = rng.uniform(0.5, 2.0)
            miss = np.cross(d, rng.normal(size=3))
            miss *= rng.uniform(0.0, 0.5) / max(np.linalg.norm(miss), 1e-12)
            r = miss - d * speed * rng.uniform(0.2, 0.8) * duration
            v = speed * d
            a = rng.uniform(-0.02, 0.02, 3) * (k % 2)
        segments.append((np.r_[r, v], a, duration, samples))
    # A slow thrusting segment 240 m out whose closing-speed margin dips 6.7e-6 m/s below the
    # gate table's bound once the jerk allowance is dropped; found by search, one in 12000.
    pinned = np.array(
        [-217.16515462097917, 89.7300631871679, -63.52001646538965]
        + [0.05014025551335433, -0.02115754973021164, -0.013336738862452307]
    )
    thrust = np.array([-0.006451479902743831, -0.01914988595717226, 3.579511850209868e-05])
    segments.append((pinned, thrust, 13.062257520505783, 4))
    return segments


def dense_and_bounds(constraint, state, accel, duration, samples, per_interval=100):
    """Each interval's second-order bound and the least margin of the dense states inside it,
    the samples taken from the same closed-form states as the dense grid."""
    grid = np.linspace(0.0, duration, (samples - 1) * per_interval + 1)
    phi, gamma = cw_discrete(N, grid)
    dense = phi @ state + gamma @ accel
    sampled = dense[::per_interval]
    h = duration / (samples - 1)
    speed, accel_bound = interval_bounds(sampled, h, float(np.linalg.norm(accel)), N)
    jerk = constraints_module._jerk_bounds(speed, accel_bound, N)
    bounds = constraint.second_order_bounds(sampled, h, speed, accel_bound, jerk)
    margins = constraint.margin(dense)
    lowest = np.array(
        [margins[i * per_interval : (i + 1) * per_interval + 1].min() for i in range(samples - 1)]
    )
    return bounds, lowest


@pytest.mark.parametrize("constraint", SPEED_TYPES, ids=lambda c: c.name)
def test_speed_second_order_bounds_lie_below_the_dense_minimum_of_every_interval(constraint):
    for state, accel, duration, samples in speed_type_segments():
        bounds, lowest = dense_and_bounds(constraint, state, accel, duration, samples)
        # Samples and dense states come from one computation, so only the bound's own
        # rounding separates them.
        assert np.all(bounds <= lowest + 1e-12)


def test_speed_bound_on_a_steep_fall_is_the_last_samples_margin():
    # Thrust along the motion raises |v| at 0.02 m/s^2, far faster than the curvature allowance
    # can bend the chord, so the lowest point of each interval's quadratic is its end.
    state = np.array([0.0, 100.0, 0.0, 0.0, 0.1, 0.0])
    accel = np.array([0.0, 0.02, 0.0])
    grid = np.linspace(0.0, 10.0, 20)
    phi, gamma = cw_discrete(N, grid)
    sampled = phi @ state + gamma @ accel
    h = grid[1]
    speed, accel_bound = interval_bounds(sampled, h, 0.02, N)
    jerk = constraints_module._jerk_bounds(speed, accel_bound, N)
    margins = SPEED.margin(sampled)
    bounds = SPEED.second_order_bounds(sampled, h, speed, accel_bound, jerk)
    np.testing.assert_allclose(bounds, margins[1:], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("constraint", SPEED_TYPES, ids=lambda c: c.name)
def test_speed_second_order_bounds_without_a_jerk_bound_are_minus_infinity(constraint):
    grid = np.linspace(0.0, 10.0, 20)
    phi, _ = cw_discrete(N, grid)
    sampled = phi @ np.array([0.0, 100.0, 0.0, 0.0, -0.1, 0.0])
    speed, accel = interval_bounds(sampled, grid[1], 0.0, N)
    assert np.all(constraint.second_order_bounds(sampled, grid[1], speed, accel) == -np.inf)


@pytest.mark.parametrize("constraint", [SPEED, CLOSING], ids=lambda c: c.name)
def test_speed_second_order_bound_is_minus_infinity_where_the_range_can_reach_zero(constraint):
    # A pass straight through the target: the interval holding it has a zero range bound.
    grid = np.linspace(0.0, 4.0, 5)
    sampled = np.array([[-1.5 + t, 0.0, 0.0, 1.0, 0.0, 0.0] for t in grid])
    speed, accel = interval_bounds(sampled, 1.0, 0.0, N)
    jerk = constraints_module._jerk_bounds(speed, accel, N)
    bounds = constraint.second_order_bounds(sampled, 1.0, speed, accel, jerk)
    rho_min = constraints_module.range_lower_bounds(sampled, 1.0, speed)
    assert rho_min[1] == 0.0 and bounds[1] == -np.inf


def test_gate_bound_is_minus_infinity_on_an_interval_that_can_reach_a_gate():
    gate = STS130_VBAR[1][0] * FT
    # Samples on the V-bar, closing at 0.1 m/s, with only the middle interval spanning the
    # 110 ft gate; the rule reads only the samples and the bounds, so the jump across the gate
    # need not match the speed, and it keeps the outer intervals clear of the gate.
    offsets = (0.6, 0.5, -0.5, -0.6)
    sampled = np.array([[0.0, gate + d, 0.0, 0.0, -0.1, 0.0] for d in offsets])
    speed, accel = interval_bounds(sampled, 1.0, 0.0, N)
    jerk = constraints_module._jerk_bounds(speed, accel, N)
    bounds = GATES.second_order_bounds(sampled, 1.0, speed, accel, jerk)
    assert bounds[1] == -np.inf
    assert np.isfinite(bounds[0]) and np.isfinite(bounds[2])


@pytest.mark.parametrize("constraint", SPEED_TYPES, ids=lambda c: c.name)
@pytest.mark.parametrize("samples", [2, 20])
def test_batched_speed_minima_equal_the_scalar_ones(constraint, samples):
    segments = speed_type_segments(count=30, seed=9)
    states = np.array([s[0] for s in segments])
    accels = np.array([s[1] for s in segments])
    times = np.linspace(0.0, 10.0, samples)
    phis, gammas = cw_discrete(N, times)
    sampled = np.einsum("mij,nj->nmi", phis, states) + np.einsum("mij,nj->nmi", gammas, accels)

    def state_at(rows, tau):
        phi, gamma = cw_discrete(N, tau)
        return np.einsum("aij,aj->ai", phi, states[rows]) + np.einsum(
            "aij,aj->ai", gamma, accels[rows]
        )

    thrust = np.linalg.norm(accels, axis=1)
    _, margins, bounds = segment_minima(
        constraint, state_at, times, sampled, thrust, N, np.zeros(len(states))
    )
    for k in range(len(states)):

        def one(tau, k=k):
            phi, gamma = cw_discrete(N, np.array([tau]))
            return (phi @ states[k] + gamma @ accels[k])[0]

        found = segment_minimum(constraint, one, times, sampled[k], float(thrust[k]), N)
        # Same samples and formulas: the bounds agree to rounding.
        assert bounds[k] == pytest.approx(found.lower_bound, rel=1e-12, abs=1e-15)
        # Golden section to 1e-6 s and Brent land within 1e-7 m/s of the same dip.
        assert margins[k] == pytest.approx(found.margin, abs=1e-7)


@pytest.mark.parametrize("constraint", SPEED_TYPES, ids=lambda c: c.name)
def test_speed_bounds_spare_the_refinement_on_a_pd_approach(monkeypatch, constraint):
    # A 30-step PD approach from 400 m: with first-order bounds alone the minimum is refined
    # 166, 190 and 74 times for the three types (measured); the second-order bound leaves none.
    calls = []
    real = constraints_module.minimize_scalar

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(constraints_module, "minimize_scalar", counting)
    start = (0.0, -400.0, 30.0, 0.0, 0.0, 0.0)
    pd = PDAgent()
    pd.reset(ApproachEnv(ApproachConfig(start_state=start)).agent_view())
    dyn = CWDynamics(N)
    x = np.array(start)
    dyn.reset(x)
    times = np.linspace(0.0, 10.0, 20)
    for k in range(30):
        u = np.clip(pd.act(x, {}), -1.0, 1.0) * 0.02
        x, sampled = dyn.advance(u, 10.0, 20)
        thrust = float(np.linalg.norm(u))
        segment_minimum(constraint, dyn.state_in_last_segment, times, sampled, thrust, N, 10.0 * k)
    assert len(calls) <= 10


# A flat speed cap has no range term to absorb the CW jerk, so only the jerk allowance keeps its
# bound sound where |v| bends between samples.
FLAT = SpeedLimit(0.05, 0.0)
# Coasting CW segments near the target where the bound without the jerk allowance lies above
# the exact minimum by 3.7e-10 to 3.6e-6 m/s (found by search over random passes).
FLAT_PINNED = [
    (
        [-8.512268687337283, -4.5816356503564055, 19.086841347259163]
        + [0.09924779892108913, -0.13070478816855716, -2.7588801485899066],
        16.174723678345668,
        5,
    ),
    (
        [-0.4601501976730352, -0.9589593079593223, 2.482664485586285]
        + [0.008865641363431438, 0.18535200869152368, -0.12043260996055892],
        26.15290790724532,
        5,
    ),
    (
        [0.5381253161172106, -0.002255556919430512, -0.5843674673423601]
        + [-0.003172288740382911, 0.021028930011550387, 0.07938371056212991],
        20.244984733344392,
        10,
    ),
    (
        [0.04829468883316452, -0.6408693338400149, 2.1716962803919375]
        + [-0.16220440273697237, 0.9678578387734551, -0.75779064388136],
        17.128064244967,
        10,
    ),
    (
        [-0.292589060436217, 0.38230360518405193, 0.22361748470537274]
        + [0.008616647762206598, -0.06630629716763056, -0.03781049927790027],
        47.737681475323896,
        10,
    ),
    (
        [5.443612240050586, 4.40407325880459, -8.213820696827401]
        + [-0.255369413471183, 0.0408834216497586, 0.5853790007332254],
        8.923616295404688,
        3,
    ),
]


@pytest.mark.parametrize("case", range(len(FLAT_PINNED)))
def test_flat_speed_cap_bound_lies_below_the_dense_minimum_on_pinned_passes(case):
    state, duration, samples = FLAT_PINNED[case]
    bounds, lowest = dense_and_bounds(
        FLAT, np.array(state), np.zeros(3), duration, samples, per_interval=400
    )
    assert np.all(bounds <= lowest + 1e-12)


def flat_cap_passes(count=200, seed=21):
    """Coasting passes within 25 m of the target at up to 3 m/s, 2 to 20 samples."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        r = rng.normal(size=3) * rng.uniform(0.5, 25.0) / np.sqrt(3.0)
        v = rng.normal(size=3) * rng.uniform(0.0, 3.0) / np.sqrt(3.0)
        yield np.r_[r, v], rng.uniform(5.0, 50.0), int(rng.integers(2, 21))


def test_flat_speed_cap_bound_lies_below_the_dense_minimum_on_coasting_passes():
    for state, duration, samples in flat_cap_passes():
        bounds, lowest = dense_and_bounds(FLAT, state, np.zeros(3), duration, samples)
        assert np.all(bounds <= lowest + 1e-12)
