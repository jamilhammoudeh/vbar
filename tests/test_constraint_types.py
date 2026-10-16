import numpy as np
import pytest
from scipy.optimize import minimize

from vbar.constraints import (
    ApproachCone,
    ClosingSpeedLimit,
    SpeedLimit,
    SunExclusion,
    closing_speed,
    segment_minimum,
)
from vbar.dynamics import EARTH, CWDynamics, cw_discrete, mean_motion

N = mean_motion(EARTH.radius + 500e3)
# Dunlap et al. 2021, eq. 14: v0 = 0.2 m/s, v1 = 4 n.
SPEED = SpeedLimit(0.2, 4 * N)
CLOSING = ClosingSpeedLimit(0.2, 4 * N)
# Docking corridor: 10 deg from a port 2 m behind the target, active within 50 m (Fehse 5.6).
CONE = ApproachCone(
    np.radians(10.0), axis=(0.0, -1.0, 0.0), apex=(0.0, -2.0, 0.0), active_range=50.0
)


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


@pytest.mark.parametrize(
    "constraint",
    [SPEED, CLOSING, CONE, SUN],
    ids=lambda c: f"{c.name}_{getattr(c, 'axis', '')}",
)
@pytest.mark.parametrize("seed", range(12))
def test_certificate_and_refined_minimum_bracket_the_dense_minimum(constraint, seed):
    found, dense = found_and_dense(constraint, seed)
    assert np.isfinite(found.lower_bound)
    assert found.lower_bound <= dense.min() + 1e-12
    # 4001 samples over at most 30 s miss a smooth minimum by far less than 1e-5 m/s.
    assert found.margin <= dense.min() + 1e-6
    assert found.margin >= found.lower_bound
