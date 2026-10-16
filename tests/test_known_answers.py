"""Published worked examples reproduced through the package, to the precision they are printed.

Sources: Fehse, Automated Rendezvous and Docking of Spacecraft (2003), whose local orbital frame
is the ISS-style LVLH frame; Alfriend et al., Spacecraft Formation Flying (2010). Fehse's
examples are recomputed for a 400 km orbit with this package's Earth constants; his printed
values use r = 6766 km and agree with these to within 0.6%.
"""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from vbar.conventions import hill_to_iss, iss_to_hill
from vbar.dynamics import EARTH, cw_discrete, cw_stm, mean_motion
from vbar.dynamics.models import CWDynamics, TwoBodyDynamics

N_400 = mean_motion(EARTH.radius + 400e3)
PERIOD_400 = 2 * np.pi / N_400


def fly(state, t, accel=(0.0, 0.0, 0.0)):
    phi, gamma = cw_discrete(N_400, t)
    return phi @ np.asarray(state, dtype=float) + gamma @ np.asarray(accel, dtype=float)


def fehse_constant_force_solution(w, t, s0, force):
    """Fehse eq 3.22: CW motion under constant specific force, in his (V-bar, H-bar, R-bar) axes."""
    x0, y0, z0, vx0, vy0, vz0 = s0
    fx, fy, fz = force
    s, c = math.sin(w * t), math.cos(w * t)
    x = (
        (4 * vx0 / w - 6 * z0) * s
        - 2 * vz0 / w * c
        + (6 * w * z0 - 3 * vx0) * t
        + (x0 + 2 * vz0 / w)
        + 2 / w**2 * fz * (w * t - s)
        + fx * (4 / w**2 * (1 - c) - 1.5 * t * t)
    )
    y = y0 * c + vy0 / w * s + fy / w**2 * (1 - c)
    z = (
        (2 * vx0 / w - 3 * z0) * c
        + vz0 / w * s
        + (4 * z0 - 2 * vx0 / w)
        + 2 / w**2 * fx * (s - w * t)
        + fz / w**2 * (1 - c)
    )
    return np.array([x, y, z])


def test_fehse_constant_force_solution_matches_cw_through_iss_axes():
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = np.r_[rng.uniform(-500.0, 500.0, 3), rng.uniform(-0.5, 0.5, 3)]
        force = rng.uniform(-1e-3, 1e-3, 3)
        t = rng.uniform(1.0, 12000.0)
        hill = fly(iss_to_hill(state), t, iss_to_hill(force))
        expected = fehse_constant_force_solution(N_400, t, state, force)
        # Fehse's secular terms reach 1e5 m at 12000 s; their rounding sets the floor.
        assert np.max(np.abs(hill_to_iss(hill[:3]) - expected)) < 1e-9


def test_release_from_rest_below_target_drifts_ahead_376_99_m_per_orbit():
    # Fehse eq 3.26: released 10 m below the target (+R-bar side, Hill x = -10 m).
    start = [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert fly(start, PERIOD_400 / 2)[0] == pytest.approx(-70.0, abs=1e-9)
    end = fly(start, PERIOD_400)
    assert end[0] == pytest.approx(-10.0, abs=1e-9)
    assert end[1] == pytest.approx(376.99, abs=5e-3)


def test_lower_coorbiting_start_drifts_ahead_94_25_m_per_orbit():
    # Fehse eqs 3.24 and 3.25: 10 m below on a circular orbit, vy = -1.5 n x.
    end = fly([-10.0, 0.0, 0.0, 0.0, 15.0 * N_400, 0.0], PERIOD_400)
    assert end[0] == pytest.approx(-10.0, abs=1e-9)
    assert end[1] == pytest.approx(94.25, abs=5e-3)


def test_out_of_plane_release_crosses_the_orbit_plane_at_0_01131_m_per_s():
    # Fehse eq 3.27; his printed 0.0138 m/s is a misprint.
    quarter = fly([0.0, 0.0, 10.0, 0.0, 0.0, 0.0], PERIOD_400 / 4)
    assert quarter[2] == pytest.approx(0.0, abs=1e-12)
    assert abs(quarter[5]) == pytest.approx(0.01131, abs=5e-6)


def test_forward_push_at_target_comes_back_1666_1_m_behind():
    # Fehse eq 3.28: 0.1 m/s along +V-bar, then free drift.
    v = 0.1
    assert fly([0.0, 0.0, 0.0, 0.0, v, 0.0], PERIOD_400)[1] == pytest.approx(-1666.1, abs=0.05)
    turn = fly([0.0, 0.0, 0.0, 0.0, v, 0.0], math.acos(0.75) / N_400)
    assert turn[4] == pytest.approx(0.0, abs=1e-15)
    assert turn[1] == pytest.approx(42.21, abs=5e-3)


@pytest.mark.parametrize(
    "impulse, t_fraction, axis, expected",
    [
        # Fehse eq 3.28 and fig 3.15: tangential impulse.
        ((0.0, 0.01, 0.0), 0.5, 0, 35.36),
        ((0.0, 0.01, 0.0), 1.0, 1, -166.61),
        # Fehse eq 3.34 and fig 3.20: radial impulse toward the Earth.
        ((-0.01, 0.0, 0.0), 0.5, 1, 35.36),
        ((-0.01, 0.0, 0.0), 0.25, 0, -8.839),
        ((-0.01, 0.0, 0.0), 1.0, 1, 0.0),
    ],
)
def test_impulse_responses_match_published_values(impulse, t_fraction, axis, expected):
    end = fly([0.0, 0.0, 0.0, *impulse], t_fraction * PERIOD_400)
    assert end[axis] == pytest.approx(expected, abs=5e-3)


def test_radial_hold_needs_3_n_squared_x_and_stays_put():
    # Fehse eq 3.55: 100 m below the target at 400 km.
    accel = 3 * N_400**2 * 100.0
    assert accel == pytest.approx(3.840e-4, abs=5e-8)
    end = fly([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0], PERIOD_400, (accel, 0.0, 0.0))
    assert np.allclose(end, [-100.0, 0.0, 0.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-9)


def test_one_orbit_of_constant_thrust_matches_published_displacements():
    # Fehse eqs 3.59 to 3.74 with a = 1e-5 m/s^2 from rest.
    along = fly([0.0] * 6, PERIOD_400, (0.0, 1e-5, 0.0))
    assert along[:2] == pytest.approx([98.1755, -462.641], abs=5e-4)
    assert along[3] == pytest.approx(0.0, abs=1e-15)
    radial = fly([0.0] * 6, PERIOD_400, (1e-5, 0.0, 0.0))
    assert radial[0] == pytest.approx(0.0, abs=1e-9)
    assert fly([0.0] * 6, PERIOD_400 / 2, (1e-5, 0.0, 0.0))[0] == pytest.approx(15.6251, abs=5e-5)


@pytest.mark.parametrize("d, expected", [(20.0, 753.98), (50.0, 1884.96), (100.0, 3769.91)])
def test_radial_escape_impulse_drifts_12_pi_d_whatever_its_size(d, expected):
    # Fehse figs 4.17 and 4.18: collision avoidance from d below the target, 0.3 m/s away.
    assert fly([-d, 0.0, 0.0, -0.3, 0.0, 0.0], PERIOD_400)[1] == pytest.approx(expected, abs=5e-3)


def test_two_impulse_rendezvous_matches_corrected_published_example():
    # Alfriend et al. example 5.2 in units of a and n. The printed numbers belong to n tf = 2
    # rad, and the second impulse is -v(tf).
    phi = cw_stm(1.0, 2.0)
    r0 = np.array([0.01, 0.02, 0.015])
    v0 = np.array([0.001, 0.001, 0.001])
    v_transfer = np.linalg.solve(phi[:3, 3:], -phi[:3, :3] @ r0)
    dv1 = v_transfer - v0
    dv2 = -(phi[3:, :3] @ r0 + phi[3:, 3:] @ v_transfer)
    assert np.all(np.abs(dv1 - [-0.00180, -0.01927, 0.005865]) <= [5e-6, 5e-6, 5e-7])
    assert np.all(np.abs(dv2 - [0.00562, -0.00173, 0.0165]) <= [5e-6, 5e-6, 5e-5])
    assert np.linalg.norm(dv1) + np.linalg.norm(dv2) == pytest.approx(0.03774, abs=5e-6)


def test_in_plane_targeting_is_singular_at_published_transfer_times():
    # Alfriend et al. p. 92: 8 cos(n t) + 3 n t sin(n t) = 8 besides n t = 2 k pi.
    def det(nt):
        return np.linalg.det(cw_stm(1.0, nt)[:2, 3:5])

    assert brentq(det, 2.7 * np.pi, 2.9 * np.pi) / np.pi == pytest.approx(2.81346, abs=5e-6)
    assert brentq(det, 4.8 * np.pi, 4.95 * np.pi) / np.pi == pytest.approx(4.89060, abs=5e-6)


def test_short_cw_targeting_matches_published_velocities():
    # Alfriend et al. example 9.2: 7170 km circular orbit, null the position in 0.01 T.
    n = mean_motion(7170e3)
    phi = cw_stm(n, 0.01 * 2 * np.pi / n)
    v0 = np.linalg.solve(phi[:3, 3:], -phi[:3, :3] @ np.array([25.0, 25.0, 50.0]))
    assert v0 == pytest.approx([-0.3889, -0.4392, -0.8264], abs=5e-5)


def test_chaser_ahead_on_the_target_circle_is_a_two_body_equilibrium():
    # Alfriend et al. eqs 4.77, 5.70 and 5.71: exact co-orbital equilibrium at angle s, which
    # CW mistakes for a semi-major axis offset and drifts.
    twobody = TwoBodyDynamics()
    a = EARTH.radius + twobody.altitude
    s = 1e-3
    start = [a * (math.cos(s) - 1.0), a * math.sin(s), 0.0, 0.0, 0.0, 0.0]
    twobody.reset(start)
    end, _ = twobody.advance(np.zeros(3), 2 * np.pi / twobody.n)
    assert np.allclose(end, start, rtol=0.0, atol=1e-8)
    cw = CWDynamics(twobody.n)
    cw.reset(start)
    drifted, _ = cw.advance(np.zeros(3), 2 * np.pi / twobody.n)
    assert abs(drifted[1] - start[1]) > 100.0
