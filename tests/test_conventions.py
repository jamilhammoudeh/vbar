import numpy as np
import pytest

from vbar.conventions import (
    HILL_TO_ISS,
    hill_frame,
    hill_to_inertial,
    hill_to_iss,
    inertial_to_hill,
    iss_to_hill,
)
from vbar.dynamics.bodies import EARTH

A_LEO = EARTH.radius + 500e3
V_LEO = np.sqrt(EARTH.mu / A_LEO)
N_LEO = V_LEO / A_LEO


def random_orbits(rng, m):
    """Inertial (r, v) pairs with random orientation, radius and flight-path angle."""
    r_hat = rng.normal(size=(m, 3))
    r_hat /= np.linalg.norm(r_hat, axis=1, keepdims=True)
    radius = rng.uniform(6.6e6, 4.2e7, (m, 1))
    side = np.cross(r_hat, rng.normal(size=(m, 3)))
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    gamma = rng.uniform(-0.3, 0.3, (m, 1))
    speed = np.sqrt(EARTH.mu / radius) * rng.uniform(0.8, 1.3, (m, 1))
    v = speed * (np.cos(gamma) * side + np.sin(gamma) * r_hat)
    return radius * r_hat, v


def circular_state(u, i=0.0, radius=A_LEO):
    """Position and velocity at argument of latitude `u` on a circular orbit, node on +X."""
    speed = np.sqrt(EARTH.mu / radius)
    p = np.array([np.cos(u), np.cos(i) * np.sin(u), np.sin(i) * np.sin(u)])
    q = np.array([-np.sin(u), np.cos(i) * np.cos(u), np.sin(i) * np.cos(u)])
    return radius * p, speed * q


def test_hill_axes_are_orthonormal_and_right_handed():
    r, v = random_orbits(np.random.default_rng(0), 200)
    C, _ = hill_frame(r, v)
    assert np.allclose(np.swapaxes(C, -1, -2) @ C, np.eye(3), atol=1e-15)
    assert np.allclose(np.linalg.det(C), 1.0, atol=1e-15)
    assert np.allclose(np.cross(C[..., 0], C[..., 1]), C[..., 2], atol=1e-15)


def test_hill_frame_of_prograde_equatorial_orbit_is_inertial_at_node():
    C, omega = hill_frame([A_LEO, 0.0, 0.0], [0.0, V_LEO, 0.0])
    assert np.array_equal(C, np.eye(3))
    assert omega == pytest.approx([0.0, 0.0, N_LEO], rel=1e-15)


def test_hill_x_is_radial_and_y_is_velocity_on_circular_orbit():
    r, v = circular_state(u=1.1, i=0.9)
    C, _ = hill_frame(r, v)
    assert np.allclose(C[:, 0], r / np.linalg.norm(r), atol=1e-15)
    assert np.allclose(C[:, 1], v / np.linalg.norm(v), atol=1e-15)


def test_chaser_on_same_circular_orbit_is_at_rest_in_hill_frame():
    dphi = 1e-4  # ahead of the target by 690 m of arc
    r_t, v_t = circular_state(u=0.4, i=0.9)
    r_c, v_c = circular_state(u=0.4 + dphi, i=0.9)
    rel = inertial_to_hill(r_t, v_t, r_c - r_t, v_c - v_t)
    expected = [A_LEO * (np.cos(dphi) - 1.0), A_LEO * np.sin(dphi), 0.0]
    # Differences of 7e6 m vectors carry about 1e-9 m and 1e-12 m/s of rounding.
    assert np.allclose(rel[:3], expected, rtol=0.0, atol=1e-8)
    assert np.allclose(rel[3:], 0.0, atol=1e-11)


def test_radial_offset_on_circular_orbit_drifts_along_track_at_minus_1p5_n_dx():
    dx = 100.0
    r_t, v_t = circular_state(u=0.0)
    r_c, v_c = circular_state(u=0.0, radius=A_LEO + dx)
    rel = inertial_to_hill(r_t, v_t, r_c - r_t, v_c - v_t)
    # Linear CW result; the nonlinear correction is of order dx / a = 1.5e-5.
    assert rel[4] == pytest.approx(-1.5 * N_LEO * dx, rel=1e-4)
    assert rel[0] == pytest.approx(dx, rel=1e-9)


def test_orbit_normal_acceleration_turns_frame_about_radial_axis():
    a_z = 1e-2
    _, omega = hill_frame([A_LEO, 0.0, 0.0], [0.0, V_LEO, 0.0], accel=[3.0, -2.0, a_z])
    assert omega == pytest.approx([a_z / V_LEO, 0.0, N_LEO], rel=1e-15)


def test_inertial_hill_round_trip_is_machine_precision():
    rng = np.random.default_rng(1)
    m = 200
    r, v = random_orbits(rng, m)
    accel = rng.normal(scale=1e-2, size=(m, 3))
    rel = np.c_[rng.uniform(-5e3, 5e3, (m, 3)), rng.uniform(-5.0, 5.0, (m, 3))]
    dr, dv = hill_to_inertial(r, v, rel, accel)
    back = inertial_to_hill(r, v, dr, dv, accel)
    ulps = 16 * np.finfo(float).eps
    assert np.allclose(back[:, :3], rel[:, :3], rtol=0.0, atol=ulps * 5e3)
    assert np.allclose(back[:, 3:], rel[:, 3:], rtol=0.0, atol=ulps * 5.0)


def test_conversions_broadcast_over_stacked_samples():
    rng = np.random.default_rng(2)
    r, v = random_orbits(rng, 5)
    rel = rng.normal(size=(5, 6))
    stacked = inertial_to_hill(r, v, rel[:, :3], rel[:, 3:])
    single = [inertial_to_hill(r[k], v[k], rel[k, :3], rel[k, 3:]) for k in range(5)]
    assert np.array_equal(stacked, np.array(single))


@pytest.mark.parametrize(
    "r_km, v_km_s, rho_km, rho_dot_km_s",
    [
        # Alfriend et al., Spacecraft Formation Flying, example 7.3: deputy on a 500 m projected
        # circular orbit at phase 0.
        (
            [5023.437579954, 5023.679067423, 0.469973680],
            [-1.810792589537, 1.810419297938, 7.041300610075],
            [-0.000288947081, 0.500033326318, 0.000175666681],
            [0.000263388377, 0.000000272412, 0.000527371445],
        ),
        # Example 7.4: the same chief and orbit at phase 90 deg.
        (
            [5024.067715322, 5023.402914470, 0.171195964],
            [-1.810892863426, 1.810892391776, 7.040872374521],
            [0.250014418391, 0.000198338483, 0.500288022195],
            [-0.000000124335, -0.000527557529, -0.000000019840],
        ),
    ],
)
def test_inertial_to_hill_reproduces_published_examples(r_km, v_km_s, rho_km, rho_dot_km_s):
    r_chief = np.array([5023.558528005, 5023.558528005, 0.0]) * 1e3
    v_chief = np.array([-1.810956397226, 1.810956397226, 7.041120373157]) * 1e3
    dr = np.array(r_km) * 1e3 - r_chief
    dv = np.array(v_km_s) * 1e3 - v_chief
    rel = inertial_to_hill(r_chief, v_chief, dr, dv)
    # The inputs are printed to 1e-6 m and 1e-9 m/s, so their differences carry that rounding.
    assert np.max(np.abs(rel[:3] - np.array(rho_km) * 1e3)) < 1e-6
    assert np.max(np.abs(rel[3:] - np.array(rho_dot_km_s) * 1e3)) < 3e-9


def test_iss_lvlh_axes_map_as_specified():
    x_hill, y_hill, z_hill = np.eye(3)
    assert np.array_equal(hill_to_iss(y_hill), [1.0, 0.0, 0.0])
    assert np.array_equal(hill_to_iss(z_hill), [0.0, -1.0, 0.0])
    assert np.array_equal(hill_to_iss(x_hill), [0.0, 0.0, -1.0])
    assert np.linalg.det(HILL_TO_ISS) == pytest.approx(1.0, abs=1e-15)


def test_iss_lvlh_round_trip_on_states_is_exact():
    states = np.random.default_rng(3).normal(size=(7, 6))
    assert np.array_equal(iss_to_hill(hill_to_iss(states)), states)
    assert np.array_equal(hill_to_iss(states)[:, 3:], hill_to_iss(states[:, 3:]))
