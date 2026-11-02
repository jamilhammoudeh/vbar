"""Sternberg's matched-acceleration equation is not a usable approach reference."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

OMEGA_Z = np.deg2rad(5.0)  # flat spin of 5 deg/s about z, as in Hettrick's listing 3.1


def sternberg_rhs(t, state, omega, omega_dot):
    """State derivative of eq. 3.6: I r'' - 2 [w]x r' + C r = 0 with C as printed."""
    w1, w2, w3 = omega
    d1, d2, d3 = omega_dot
    skew = np.array([[0.0, -w3, w2], [w3, 0.0, -w1], [-w2, w1, 0.0]])
    c = np.array(
        [
            [-(w3**2) - w2**2, d3 + w1 * w2, -d2 + w1 * w3],
            [-d3 + w1 * w2, -(w1**2) - w3**2, d1 + w2 * w3],
            [d2 + w1 * w3, -d1 + w2 * w3, -(w1**2) - w2**2],
        ]
    )
    r, v = state[:3], state[3:]
    return np.concatenate([v, 2.0 * skew @ v - c @ r])


def test_sternberg_equation_spirals_away_from_the_target_under_flat_spin():
    # Hettrick, "Autonomous Rendezvous and Docking with Tumbling, Uncooperative, and Fragile
    # Targets" (MIT S.M. thesis, 2019), eq. 3.6 and listing 3.1, pp. 54 to 56: from [10, 0, 0] m
    # at rest for 100 s the chaser spirals away (fig. 3-7, last point about 88 m out).
    omega = np.array([0.0, 0.0, OMEGA_Z])
    start = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    times = np.linspace(0.0, 100.0, 101)
    sol = solve_ivp(
        sternberg_rhs, (0.0, 100.0), start, t_eval=times, args=(omega, np.zeros(3)),
        rtol=1e-10, atol=1e-10,
    )  # fmt: skip
    ranges = np.linalg.norm(sol.y[:3], axis=0)
    # A synchronous radial approach would close the range; this one never gets below the start
    # and ends at least five times farther out.
    assert np.all(ranges >= ranges[0] - 1e-9)
    assert ranges[-1] > 5.0 * ranges[0]
    # In the frame spinning with the target the motion is a straight line at the start speed
    # omega * 10 m, so the range is 10 sqrt(1 + (omega t)^2).
    assert ranges[-1] == pytest.approx(10.0 * np.hypot(1.0, OMEGA_Z * 100.0), rel=1e-8)
