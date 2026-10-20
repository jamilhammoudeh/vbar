import numpy as np
import pytest

from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.dynamics.cw import cw_rhs
from vbar.solvers import HillScaling

N = mean_motion(EARTH.radius + 400e3)
SCALING = HillScaling(length=250.0, n=N)


def random_states(rng, count):
    r = rng.uniform(-2000.0, 2000.0, (count, 3))
    v = rng.uniform(-2.0, 2.0, (count, 3))
    return np.hstack([r, v])


def test_units_are_length_speed_n_l_and_acceleration_n2_l():
    x = SCALING.state([250.0, 0.0, 0.0, 0.0, N * 250.0, 0.0])
    np.testing.assert_allclose(x, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], rtol=1e-15)
    assert SCALING.accel(N * N * 250.0) == pytest.approx(1.0, rel=1e-15)
    assert SCALING.time(1.0 / N) == pytest.approx(1.0, rel=1e-15)


def test_round_trips_return_the_si_values():
    rng = np.random.default_rng(11)
    x = random_states(rng, 100)
    a = rng.uniform(-0.02, 0.02, (100, 3))
    np.testing.assert_allclose(SCALING.state_si(SCALING.state(x)), x, rtol=1e-15)
    np.testing.assert_allclose(SCALING.accel_si(SCALING.accel(a)), a, rtol=1e-15)
    assert SCALING.time_si(SCALING.time(5400.0)) == pytest.approx(5400.0, rel=1e-15)


def test_scaled_cw_equations_have_unit_mean_motion():
    rng = np.random.default_rng(12)
    for x, a in zip(random_states(rng, 50), rng.uniform(-0.02, 0.02, (50, 3)), strict=True):
        rate_si = cw_rhs(0.0, x, N, a)
        rate = cw_rhs(0.0, SCALING.state(x), 1.0, SCALING.accel(a))
        # d(r/L)/d(nt) = v/(nL) and d(v/(nL))/d(nt) = (dv/dt)/(n^2 L).
        expected = np.concatenate([SCALING.state(x)[3:], SCALING.accel(rate_si[3:])])
        np.testing.assert_allclose(rate, expected, rtol=1e-13, atol=1e-13 * np.abs(expected).max())


def test_scaled_transition_matches_si_propagation():
    rng = np.random.default_rng(13)
    for dt in (1e-3, 10.0, 600.0, 5400.0):
        phi, gamma = cw_discrete(N, dt)
        phi_1, gamma_1 = cw_discrete(1.0, SCALING.time(dt))
        for x, a in zip(random_states(rng, 20), rng.uniform(-0.02, 0.02, (20, 3)), strict=True):
            si = phi @ x + gamma @ a
            scaled = SCALING.state_si(phi_1 @ SCALING.state(x) + gamma_1 @ SCALING.accel(a))
            # Different rounding paths through the trigonometric terms; relative to |x|.
            np.testing.assert_allclose(scaled, si, rtol=0, atol=1e-12 * np.abs(si).max())


@pytest.mark.parametrize(
    "length, n", [(0.0, N), (-1.0, N), (100.0, 0.0), (np.inf, N), (100.0, np.nan)]
)
def test_scale_must_be_finite_and_positive(length, n):
    with pytest.raises(ValueError, match="finite and positive"):
        HillScaling(length=length, n=n)


def test_states_need_six_components():
    with pytest.raises(ValueError, match="six components"):
        SCALING.state(np.zeros(3))
    with pytest.raises(ValueError, match="six components"):
        SCALING.state_si(np.zeros((2, 5)))
