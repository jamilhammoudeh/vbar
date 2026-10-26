import dataclasses
import functools
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from test_constraints import decimal_sin_cos

from vbar.dynamics import EARTH, cw_discrete, cw_states, mean_motion
from vbar.safety.barriers import (
    BrakingBarrier,
    ClosingSpeedCbf,
    ConeEcbf,
    KeepOutEcbf,
    PassiveBarrier,
    SpeedCbf,
    braking_deceleration,
)
from vbar.solvers import ActuatorSet, Rows, project

N = mean_motion(EARTH.radius + 500e3)
R = 50.0
T_CAP = 10 * 2 * np.pi / N
FINITE = {"allow_nan": False, "allow_infinity": False, "allow_subnormal": False}


def a_cw(x, n=N):
    return np.array([3 * n * n * x[0] + 2 * n * x[4], -2 * n * x[3], -n * n * x[2]])


def held(x, u, tau):
    """States tau s along the held segment from x, exact CW."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    return cw_states(N, tau, np.tile(x, (len(tau), 1)), np.tile(u, (len(tau), 1)))


def rates(f, x, u, tau=0.05):
    """f and its first two time derivatives at x under the held command u.

    Fourth-order central differences: with a 0.05 s step the truncation error is near
    tau^4 |f^(6)| / 90 and the rounding near 1e-16 |f| / tau^2, far below the 1e-6 relative
    tolerance used here even where |v| is a few cm/s and its higher derivatives are large.
    """
    m2, m1, mid, p1, p2 = (f(s) for s in held(x, u, [-2 * tau, -tau, 0.0, tau, 2 * tau]))
    first = (m2 - 8 * m1 + 8 * p1 - p2) / (12 * tau)
    second = (-m2 + 16 * m1 - 30 * mid + 16 * p1 - p2) / (12 * tau**2)
    return mid, first, second


def rho_of(s):
    return float(np.linalg.norm(s[:3]))


def random_states(rng, count, low=80.0, high=400.0, speed=0.3):
    d = rng.standard_normal((count, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = d * rng.uniform(low, high, (count, 1))
    return np.hstack([r, rng.uniform(-speed, speed, (count, 3))])


# Exponential CBF for the keep-out sphere: Ames et al., Control Barrier Functions: Theory and
# Applications, ECC 2019, sec. IV (Definition 7, Theorem 8), with real poles p1, p2.


@pytest.mark.parametrize("poles", [(0.05, 0.05), (0.02, 0.1), (1.0, 3.0)])
def test_ecbf_closed_loop_eigenvalues_are_minus_the_poles(poles):
    # Ames et al. sec. IV.B prints the roots as p_i; the closed loop F - G K has roots -p_i.
    k0, k1 = KeepOutEcbf(radius=R, poles=poles).gains
    F = np.array([[0.0, 1.0], [0.0, 0.0]])
    G = np.array([[0.0], [1.0]])
    K = np.array([[k0, k1]])
    eigenvalues = np.sort(np.linalg.eigvals(F - G @ K).real)
    assert eigenvalues == pytest.approx(np.sort([-poles[0], -poles[1]]), abs=1e-9)


def test_ecbf_gains_are_the_pole_product_and_sum():
    k0, k1 = KeepOutEcbf(radius=R, poles=(0.02, 0.1)).gains
    assert k0 == pytest.approx(0.002, rel=1e-15)
    assert k1 == pytest.approx(0.12, rel=1e-15)


def closing_start(h0=10.0, hd0=-0.5):
    return np.array([0.0, R + h0, 0.0, 0.0, hd0, 0.0])


@pytest.mark.parametrize("poles", [(0.06, 0.06), (0.05, 0.08), (0.04, 0.5), (0.5, 0.04)])
def test_corrected_theorem_8_start_condition_keeps_the_barrier_nonnegative(poles):
    # Theorem 8 with the sign corrected: max(p1, p2) >= -hdot0 / h0 = 0.05 1/s.
    barrier = KeepOutEcbf(radius=R, poles=poles)
    assert barrier.contains(closing_start())
    k0, k1 = barrier.gains
    t = np.arange(0.0, 600.5, 0.5)
    flows = expm(np.array([[0.0, 1.0], [-k0, -k1]]) * t[:, None, None])
    h = flows[:, 0, :] @ np.array([10.0, -0.5])
    assert h.min() >= -1e-9


def test_double_pole_below_the_start_condition_is_rejected_and_crosses_zero_at_100_s():
    barrier = KeepOutEcbf(radius=R, poles=(0.04, 0.04))
    assert not barrier.contains(closing_start())
    k0, k1 = barrier.gains
    flow = np.array([[0.0, 1.0], [-k0, -k1]])

    def h(t):
        return float((expm(flow * t) @ np.array([10.0, -0.5]))[0])

    # Closed form for the double pole: h = (h0 + (hdot0 + p h0) t) exp(-p t), zero where
    # 10 - 0.1 t = 0 and lowest where its derivative (-0.5 + 0.004 t) exp(-p t) vanishes.
    assert abs(h(100.0)) <= 1e-6
    assert h(125.0) == pytest.approx(-2.5 * math.exp(-5.0), abs=1e-6)
    assert round(h(125.0), 4) == -0.0168
    assert h(125.0) < min(h(124.0), h(126.0))


@pytest.mark.parametrize(
    ("state", "inside"),
    [
        (closing_start(10.0, -0.5), True),
        (closing_start(10.0, -0.6000001), False),
        (closing_start(0.0, 0.0), True),
        (closing_start(-1e-6, 0.1), False),
        (closing_start(20.0, 0.3), True),
    ],
)
def test_ecbf_safe_set_is_h_nonnegative_and_hdot_plus_fast_pole_h_nonnegative(state, inside):
    assert KeepOutEcbf(radius=R, poles=(0.02, 0.06)).contains(state) == inside


@settings(max_examples=25, deadline=None)
@given(
    h0=st.floats(0.01, 300.0, **FINITE),
    hd0=st.floats(-1.0, 1.0, **FINITE),
    p1=st.floats(0.001, 1.0, **FINITE),
    p2=st.floats(0.001, 1.0, **FINITE),
)
def test_ecbf_contains_matches_the_corrected_start_condition(h0, hd0, p1, p2):
    slack = hd0 + max(p1, p2) * h0
    # Away from the boundary, where rounding of rho could flip the answer.
    if abs(slack) < 1e-9:
        return
    assert KeepOutEcbf(radius=R, poles=(p1, p2)).contains(closing_start(h0, hd0)) == (slack > 0)


@pytest.mark.parametrize(
    "poles",
    [(0.05 + 0.02j, 0.05 - 0.02j), (0.0, 0.05), (-0.05, 0.05), (0.05, -1.0), (math.nan, 0.05)],
)
def test_ecbf_refuses_complex_or_nonpositive_poles(poles):
    # The comparison bound after Ames et al. eq. 18 holds only for real closed-loop poles.
    with pytest.raises(ValueError):
        KeepOutEcbf(radius=R, poles=poles)


def test_ecbf_row_at_a_hand_state():
    x = np.array([30.0, -40.0, 0.0, -0.1, 0.05, 0.02])
    a, b = KeepOutEcbf(radius=20.0, poles=(0.02, 0.1)).row(x, a_cw(x))
    rhat = np.array([0.6, -0.8, 0.0])
    rhodot = -0.1
    centripetal = (0.0129 - 0.01) / 50.0
    expected = -centripetal - rhat @ a_cw(x) - 0.12 * rhodot - 0.002 * 30.0
    assert np.asarray(a) == pytest.approx(rhat, rel=1e-14)
    assert b == pytest.approx(expected, rel=1e-12)


def test_ecbf_row_is_hddot_plus_gains_times_h_and_hdot_along_the_held_segment():
    barrier = KeepOutEcbf(radius=R, poles=(0.03, 0.07))
    k0, k1 = barrier.gains
    rng = np.random.default_rng(11)
    for x in random_states(rng, 20):
        u = rng.uniform(-0.02, 0.02, 3)
        a, b = barrier.row(x, a_cw(x))
        rho, rhodot, rhoddot = rates(rho_of, x, u)
        lhs = rhoddot + k1 * rhodot + k0 * (rho - R)
        assert a @ u - b == pytest.approx(lhs, rel=1e-6, abs=1e-10)


def test_speed_row_at_a_hand_state():
    x = np.array([30.0, -40.0, 0.0, -0.1, 0.05, 0.02])
    a, b = SpeedCbf(v0=0.2, v1=0.004, gamma=0.05).row(x, a_cw(x))
    speed = math.sqrt(0.0129)
    vhat = x[3:] / speed
    s = 0.2 + 0.004 * 50.0 - speed
    expected = vhat @ a_cw(x) - 0.004 * (-0.1) - 0.05 * s
    assert np.asarray(a) == pytest.approx(-vhat, rel=1e-14)
    assert b == pytest.approx(expected, rel=1e-12)


def test_closing_speed_row_at_a_hand_state():
    x = np.array([30.0, -40.0, 0.0, -0.1, 0.05, 0.02])
    a, b = ClosingSpeedCbf(v0=0.05, v1=0.002, gamma=0.1).row(x, a_cw(x))
    rhat = np.array([0.6, -0.8, 0.0])
    centripetal = (0.0129 - 0.01) / 50.0
    s = 0.05 + 0.002 * 50.0 - 0.1
    expected = -centripetal - rhat @ a_cw(x) - 0.002 * (-0.1) - 0.1 * s
    assert np.asarray(a) == pytest.approx(rhat, rel=1e-14)
    assert b == pytest.approx(expected, rel=1e-12)


def test_speed_rows_are_sdot_plus_gamma_s_along_the_held_segment():
    speed_cbf = SpeedCbf(v0=0.2, v1=0.004, gamma=0.05)
    closing_cbf = ClosingSpeedCbf(v0=0.05, v1=0.002, gamma=0.1)

    def speed_margin(s):
        return 0.2 + 0.004 * rho_of(s) - float(np.linalg.norm(s[3:]))

    def closing_margin(s):
        return 0.05 + 0.002 * rho_of(s) + float(s[:3] @ s[3:]) / rho_of(s)

    rng = np.random.default_rng(13)
    closing = 0
    for x in random_states(rng, 30):
        u = rng.uniform(-0.02, 0.02, 3)
        a, b = speed_cbf.row(x, a_cw(x))
        s, sdot, _ = rates(speed_margin, x, u)
        assert a @ u - b == pytest.approx(sdot + 0.05 * s, rel=1e-6, abs=1e-10)
        if x[:3] @ x[3:] < 0.0:
            closing += 1
            a, b = closing_cbf.row(x, a_cw(x))
            s, sdot, _ = rates(closing_margin, x, u)
            assert a @ u - b == pytest.approx(sdot + 0.1 * s, rel=1e-6, abs=1e-10)
    assert closing >= 5


@pytest.mark.parametrize(("speed", "kept"), [(0.0, False), (5e-10, False), (2e-9, True)])
def test_speed_row_is_dropped_below_1e_9_m_per_s(speed, kept):
    x = np.array([30.0, -40.0, 0.0, 0.0, 0.0, speed])
    assert (SpeedCbf(v0=0.2, v1=0.004, gamma=0.05).row(x, a_cw(x)) is not None) is kept


@pytest.mark.parametrize(("rhodot", "kept"), [(0.1, False), (0.0, False), (-1e-6, True)])
def test_closing_speed_row_is_dropped_unless_closing(rhodot, kept):
    x = np.array([0.0, 50.0, 0.0, 0.05, rhodot, 0.0])
    row = ClosingSpeedCbf(v0=0.05, v1=0.002, gamma=0.1).row(x, a_cw(x))
    assert (row is not None) is kept


# Approach-cone barrier h_c = a . (r - p) - cos(theta) |r - p|, relative degree 2, with the
# curvature term of |r - p|; known answers at theta = 10 deg, rho = 5 m.
THETA = math.radians(10.0)
AXIS = np.array([0.0, -1.0, 0.0])
APEX = np.array([0.0, -2.0, 0.0])


def cone(poles=(0.05, 0.05)):
    return ConeEcbf(half_angle=THETA, axis=AXIS, apex=APEX, active_range=200.0, poles=poles)


def test_cone_row_on_the_surface_has_gradient_sin_theta_and_curvature_term():
    e = math.cos(THETA) * AXIS + math.sin(THETA) * np.array([1.0, 0.0, 0.0])
    x = np.concatenate([APEX + 5.0 * e, [0.0, 0.0, 0.03]])
    barrier = cone()
    assert abs(barrier.value(x)) <= 1e-12
    a, b = barrier.row(x, a_cw(x))
    grad = AXIS - math.cos(THETA) * e
    assert np.linalg.norm(a) == pytest.approx(0.173648, rel=1e-5)
    assert np.linalg.norm(a) == pytest.approx(math.sin(THETA), rel=1e-12)
    assert np.asarray(a) == pytest.approx(grad, rel=1e-12, abs=1e-15)
    # On the surface with hdot = 0 the row is -(hddot at zero thrust); the CW part is grad . a_cw.
    hddot_free = -b - grad @ a_cw(x)
    assert hddot_free == pytest.approx(-1.77265e-4, rel=1e-5)
    assert hddot_free == pytest.approx(-math.cos(THETA) * 0.03**2 / 5.0, rel=1e-12)


def test_cone_row_is_hddot_plus_gains_inside_the_cone():
    barrier = cone(poles=(0.04, 0.09))
    k0, k1 = barrier.gains
    rng = np.random.default_rng(17)
    for _ in range(20):
        off = rng.uniform(0.0, 0.7 * THETA)
        phase = rng.uniform(0.0, 2 * np.pi)
        lateral = math.cos(phase) * np.array([1.0, 0, 0]) + math.sin(phase) * np.array([0, 0, 1.0])
        r = APEX + rng.uniform(5.0, 150.0) * (math.cos(off) * AXIS + math.sin(off) * lateral)
        x = np.concatenate([r, rng.uniform(-0.1, 0.1, 3)])
        u = rng.uniform(-0.02, 0.02, 3)
        a, b = barrier.row(x, a_cw(x))
        h, hdot, hddot = rates(barrier.value, x, u)
        assert a @ u - b == pytest.approx(hddot + k1 * hdot + k0 * h, rel=1e-6, abs=1e-10)


@pytest.mark.parametrize(("v_lat", "radius"), [(0.01, 0.028), (0.03, 0.255), (0.05, 0.709)])
def test_cone_is_infeasible_inside_cot_theta_v_squared_over_a_max(v_lat, radius):
    rho = cone().infeasible_radius(v_lat, 0.02)
    assert rho == pytest.approx(v_lat**2 / (math.tan(THETA) * 0.02), rel=1e-12)
    assert round(rho, 3) == radius


# Free-drift barrier: Ames et al. sec. III, eq. 12 and Theorem 4 with the backup beta = 0.
PASSIVE = PassiveBarrier(radius=R, n=N)
TAU = np.arange(0.0, T_CAP + 5.0, 5.0)
PHI_R = cw_discrete(N, TAU)[0][:, :3, :]


@functools.cache
def capped_barrier():
    """The drift barrier certified up to t_cap only, which the filter keeps under error sets."""
    return PassiveBarrier(radius=R, n=N, terminal=False)


@functools.cache
def terminal_barrier():
    """The drift barrier whose members drift clear of the keep-out for all time."""
    return PassiveBarrier(radius=R, n=N, terminal=True)


def drift_horizon(x):
    drift = abs(6 * N * x[0] + 3 * x[4])
    bound = 6 * abs(x[0]) + abs(x[1]) + 4 * abs(x[3]) / N + 4 * abs(x[4]) / N
    return T_CAP if drift == 0.0 else min(T_CAP, (R + bound) / drift + 5.0)


def exact_drift_minimum(x):
    """Lowest free-drift range minus R over the horizon and its time, refined between samples."""
    m = int(np.searchsorted(TAU, drift_horizon(x))) + 1
    rho = np.linalg.norm(PHI_R[:m] @ x, axis=1)
    k = int(np.argmin(rho))
    found = minimize_scalar(
        lambda t: np.linalg.norm(cw_discrete(N, t)[0][:3] @ x),
        bounds=(TAU[max(k - 1, 0)], TAU[min(k + 1, m - 1)]),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return found.fun - R, found.x


def approach_starts(rng, count):
    return np.column_stack(
        [
            rng.uniform(-200, 200, count),
            rng.uniform(200, 600, count),
            rng.uniform(-50, 50, count),
            rng.uniform(-0.1, 0.1, (count, 3)),
        ]
    )


def test_capped_passive_value_is_a_certified_and_tight_lower_bound_on_the_drift_margin():
    barrier = capped_barrier()
    rng = np.random.default_rng(1903)
    tau = np.arange(0.0, T_CAP, 2.0)
    phi = cw_discrete(N, tau)[0]
    for x in approach_starts(rng, 50):
        drift = phi @ x
        margin = np.linalg.norm(drift[:, :3], axis=1) - R
        within = tau <= drift_horizon(x)
        speed = np.linalg.norm(drift[within, 3:], axis=1).max()
        value = barrier.value(x)
        assert value <= margin[within].min() + 1e-6
        # Past the horizon the secular drift keeps the range above R, so no violation is missed.
        assert min(value, 0.0) <= min(margin.min(), 0.0) + 1e-6
        # The sampled bound loses at most about |v| times the 5 s sample spacing.
        assert value >= margin[within].min() - 5.0 * (speed + 0.05)
        assert barrier.contains(x) == (value >= 0.0)


def test_capped_passive_horizon_ends_where_the_secular_drift_clears_the_keep_out():
    fast = np.array([100.0, 400.0, 0.0, 0.0, 0.0, 0.0])
    horizon, proven = capped_barrier().horizon(fast)
    assert proven
    assert horizon == pytest.approx((R + 600.0 + 400.0) / (600.0 * N) + 5.0, rel=1e-12)
    slow = np.array([1.0, 400.0, 0.0, 0.0, 0.0, 0.0])
    horizon, proven = capped_barrier().horizon(slow)
    assert not proven
    assert horizon == pytest.approx(T_CAP, rel=1e-12)


def test_passive_gradient_matches_central_differences():
    # The envelope theorem gives grad h_p = Phi_r(tau*)^T rhat(tau*) at a unique interior
    # minimiser tau*. States drawn as the reference computation draws them.
    rng = np.random.default_rng(1903)
    for _ in range(5):
        while True:
            x = np.concatenate([rng.uniform(-300, 300, 3), rng.uniform(-0.2, 0.2, 3)])
            _, tau = exact_drift_minimum(x)
            if 50.0 < tau < 0.9 * T_CAP:
                break
        fd = np.zeros(6)
        for i in range(6):
            e = np.zeros(6)
            e[i] = 1e-4 if i < 3 else 1e-7
            fd[i] = (exact_drift_minimum(x + e)[0] - exact_drift_minimum(x - e)[0]) / (2 * e[i])
        grad = np.asarray(PASSIVE.gradient(x))
        assert grad.shape == (6,)
        assert np.abs(grad - fd).max() <= 1e-6 * np.abs(fd).max()


@pytest.mark.parametrize(
    "x",
    [
        # Closed relative orbit: 6 n x + 3 vy = 0, the minimum repeats every orbit.
        np.array([100.0, 300.0, 0.0, 0.0, -2.0 * N * 100.0, 0.0]),
        # Below the target and drifting ahead from rest: the range only grows, tau* = 0.
        np.array([-50.0, 300.0, 0.0, 0.0, 0.0, 0.0]),
        # Far ahead and closing slowly: still approaching when the horizon ends.
        np.array([10.0, 5000.0, 0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_passive_gradient_refuses_degenerate_minimisers(x):
    with pytest.raises(ValueError):
        PASSIVE.gradient(x)


def periodic_along_track(x, tau):
    """y(t) + (6 n x0 + 3 vy0) t over one orbit, from the exact CW solution, (len(tau), k)."""
    y = cw_discrete(N, tau)[0][:, 1, :] @ x.T
    return y + np.outer(tau, 6 * N * x[:, 0] + 3 * x[:, 4])


def drift_bound(x):
    return (
        6 * np.abs(x[..., 0])
        + np.abs(x[..., 1])
        + 4 * np.abs(x[..., 3]) / N
        + 4 * np.abs(x[..., 4]) / N
    )


def test_drift_bound_holds_on_1000_random_states():
    rng = np.random.default_rng(1903)
    x = np.hstack([rng.uniform(-2000, 2000, (1000, 3)), rng.uniform(-1.0, 1.0, (1000, 3))])
    tau = np.linspace(0.0, 2 * np.pi / N, 400)
    periodic = np.abs(periodic_along_track(x, tau)).max(axis=0)
    assert np.all(periodic <= drift_bound(x) * (1 + 1e-12))


@settings(max_examples=25, deadline=None)
@given(
    st.lists(st.floats(-5000.0, 5000.0, **FINITE), min_size=3, max_size=3),
    st.lists(st.floats(-2.0, 2.0, **FINITE), min_size=3, max_size=3),
)
def test_drift_bound_holds_on_drawn_states(r, v):
    x = np.array([r + v])
    tau = np.linspace(0.0, 2 * np.pi / N, 400)
    periodic = np.abs(periodic_along_track(x, tau)).max()
    assert periodic <= drift_bound(x[0]) * (1 + 1e-12) + 1e-9


def test_passive_backup_is_free_drift():
    x = np.array([30.0, 400.0, 0.0, 0.0, -0.1, 0.0])
    for layout in ("box", "ball"):
        assert np.array_equal(PASSIVE.backup(x, ActuatorSet(0.02, layout)), np.zeros(3))


# Braking barrier h_b = rho - R - max(0, -rhodot)^2 / (2 a_b), the keep-out analogue of the
# lane-keeping barrier of Ames et al. sec. V.B, with the outer barrier
# h_o = rho_cap - rho - max(0, rhodot)^2 / (2 a_b) and the speed cap |v| <= v_cap. The approach
# box: 0.02 m/s^2, dt 10 s, R 50 m, rho_cap 2000 m, v_cap 0.3 m/s.
A_MAX = 0.02
RHO_CAP = 2000.0


@dataclasses.dataclass(frozen=True)
class Braking:
    """The constants of the braking backup, from its definition in docs/safety-filters.md."""

    a_r: float
    rho_mid: float
    eps: float
    v_hi: float
    a_b: float
    delta: float
    dt: float
    v_cap: float
    rho_cap: float


def braking_constants(a_max=A_MAX, v_cap=0.3, dt=10.0, radius=R, rho_cap=RHO_CAP, n=N):
    # What is left of a_max after cancelling the CW acceleration, split equally between the
    # radial and the tangential parts.
    budget = a_max - (3 * n * n * rho_cap + 2 * n * v_cap)
    a_r = budget / math.sqrt(2.0)
    eps = (3 * n * n * v_cap + 2 * n * budget) * dt / (1 - 2 * n * dt - 1.5 * n * n * dt * dt)
    v_hi = v_cap + eps * dt / 2
    theta = v_hi * dt / radius
    rho_mid = (radius + rho_cap) / 2
    sigma = a_r * theta**2 / 2 + a_r * theta + eps + v_hi**2 / (rho_mid - v_hi * dt)
    return Braking(a_r, rho_mid, eps, v_hi, a_r - sigma, sigma * dt, dt, v_cap, rho_cap)


APPROACH = braking_constants()


def braking_terms(x):
    """rho, rhat, rho_dot, the velocity across the line of sight and its size."""
    rho = float(np.linalg.norm(x[:3]))
    rhat = x[:3] / rho
    p = float(rhat @ x[3:])
    v_t = x[3:] - p * rhat
    return rho, rhat, p, v_t, float(np.linalg.norm(v_t))


def braking_law(x, c=APPROACH):
    """beta' = a_r rhat - b that, the backup's acceleration once the CW part is cancelled."""
    rho, rhat, p, v_t, w = braking_terms(x)
    inner = rho < c.rho_mid
    if inner and p < 0.0:
        a = c.a_r
    elif not inner and p > 0.0:
        a = -c.a_r
    else:
        target = c.delta if inner else -c.delta
        a = float(np.clip((target - p) / c.dt, -c.a_r, c.a_r))
    tangential = np.zeros(3) if w == 0.0 else -min(c.a_r, w / c.dt) * v_t / w
    return a * rhat + tangential


def braking_barrier(a_b=None, v_cap=0.3, dt=10.0, radius=R, rho_cap=RHO_CAP):
    if a_b is None:
        a_b = braking_deceleration(
            a_max=A_MAX, v_cap=v_cap, dt=dt, radius=radius, rho_cap=rho_cap, n=N
        )
    return BrakingBarrier(
        radius=radius, a_b=a_b, rho_cap=rho_cap, v_cap=v_cap, n=N, dt=dt, a_max=A_MAX
    )


def unit(rng):
    d = rng.standard_normal(3)
    return d / np.linalg.norm(d)


def braking_draws(c, rng, count):
    """States of the set: uniform, and on its inner, outer and middle boundaries."""
    a_b, v_cap, rho_cap, s = c.a_b, c.v_cap, c.rho_cap, 1 - 1e-6
    states = []
    for _ in range(count):
        rho = R * (rho_cap / R) ** rng.uniform()
        v = unit(rng) * v_cap * rng.uniform() ** (1 / 3)
        states.append(np.concatenate([rho * unit(rng), v]))
    for kind in ("inner", "outer", "middle"):
        for _ in range(count // 2):
            d = unit(rng)
            if kind == "inner":
                rho = R + 10 ** rng.uniform(-6, 1.5)
                radial = -min(v_cap, math.sqrt(2 * a_b * (rho - R))) * s
            elif kind == "outer":
                rho = rho_cap - 10 ** rng.uniform(-6, 1.5)
                radial = min(v_cap, math.sqrt(2 * a_b * (rho_cap - rho))) * s
            else:
                rho = c.rho_mid + rng.uniform(-5.0, 5.0)
                radial = rng.uniform(-v_cap, v_cap) * s
            t = unit(rng)
            t -= (t @ d) * d
            t /= np.linalg.norm(t)
            w = math.sqrt(max(0.0, (v_cap * s) ** 2 - radial**2)) * rng.choice([1.0, rng.uniform()])
            states.append(np.concatenate([rho * d, radial * d + w * t]))
    return np.array(states)


def braking_values(states, a_b, rho_cap=RHO_CAP):
    """h_b and h_o of an array of states, m."""
    rho = np.linalg.norm(states[..., :3], axis=-1)
    p = np.einsum("...i,...i", states[..., :3], states[..., 3:]) / rho
    h_b = rho - R - np.maximum(0.0, -p) ** 2 / (2 * a_b)
    h_o = rho_cap - rho - np.maximum(0.0, p) ** 2 / (2 * a_b)
    return h_b, h_o


@pytest.mark.parametrize(
    ("v", "value"),
    [
        ((-0.03, -0.04, 0.1), 30.0 - 0.05**2 / (2 * 0.005)),
        ((0.03, 0.04, 0.1), 30.0),
        ((-0.08, 0.06, 0.0), 30.0),
    ],
)
def test_braking_value_at_hand_states(v, value):
    barrier = braking_barrier(a_b=0.005, radius=20.0)
    x = np.concatenate([[30.0, 40.0, 0.0], v])
    assert barrier.value(x) == pytest.approx(value, rel=1e-14)


@pytest.mark.parametrize(
    ("v", "value"),
    [
        ((0.03, 0.04, 0.1), 1950.0 - 0.05**2 / (2 * 0.005)),
        ((-0.03, -0.04, 0.1), 1950.0),
        ((-0.08, 0.06, 0.0), 1950.0),
    ],
)
def test_braking_outer_value_at_hand_states(v, value):
    barrier = braking_barrier(a_b=0.005, radius=20.0)
    x = np.concatenate([[30.0, 40.0, 0.0], v])
    assert barrier.outer_value(x) == pytest.approx(value, rel=1e-14)


def test_braking_contains_states_with_room_to_stop():
    barrier = braking_barrier(a_b=0.005, radius=20.0)
    assert barrier.contains(np.array([30.0, 40.0, 0.0, -0.03, -0.04, 0.1]))
    # 1 m of room, closing at 0.2 m/s: stopping takes 4 m.
    assert not barrier.contains(np.array([0.0, 21.0, 0.0, 0.0, -0.2, 0.0]))


def test_braking_contains_needs_room_to_stop_before_rho_cap():
    barrier = braking_barrier()
    # 1 m inside rho_cap, opening at 0.2 m/s: stopping takes 2.6 m.
    assert not barrier.contains(np.array([0.0, RHO_CAP - 1.0, 0.0, 0.0, 0.2, 0.0]))
    assert barrier.contains(np.array([0.0, RHO_CAP - 1.0, 0.0, 0.0, 0.0, 0.0]))
    assert not barrier.contains(np.array([0.0, RHO_CAP + 1e-6, 0.0, 0.0, 0.0, 0.0]))


def test_braking_contains_needs_the_speed_cap():
    barrier = braking_barrier()
    assert barrier.contains(np.array([0.0, 1000.0, 0.0, 0.3, 0.0, 0.0]))
    assert not barrier.contains(np.array([0.0, 1000.0, 0.0, 0.3 + 1e-9, 0.0, 0.0]))


def default_deceleration(**kw):
    args = {"a_max": A_MAX, "v_cap": 0.3, "dt": 10.0, "radius": R, "rho_cap": RHO_CAP, "n": N}
    return braking_deceleration(**(args | kw))


@pytest.mark.parametrize(
    ("kw", "rounded"),
    [({}, 0.0075776), ({"v_cap": 1.0}, 0.004492), ({"v_cap": 1.0, "dt": 20.0}, 0.00225)],
    ids=["approach box", "v_cap 1", "v_cap 1 dt 20"],
)
def test_default_braking_deceleration_is_the_radial_share_less_the_losses_of_a_step(kw, rounded):
    c = braking_constants(**kw)
    assert default_deceleration(**kw) == pytest.approx(c.a_b, rel=1e-12)
    assert default_deceleration(**kw) == pytest.approx(rounded, abs=5e-6)


def test_default_braking_deceleration_for_the_approach_box_is_0_00758():
    assert round(default_deceleration(), 5) == 0.00758


def test_braking_constants_of_the_approach_box():
    # B = 0.02 - (3 n^2 2000 + 2 n 0.3) = 0.011986 m/s^2; E = eps dt / 2 is the most the speed
    # rises above v_cap within a step and theta the turn of rhat in a step at 50 m.
    c = APPROACH
    assert c.a_r * math.sqrt(2.0) == pytest.approx(0.011986, abs=5e-7)
    assert c.a_r == pytest.approx(0.008475, abs=5e-7)
    assert c.eps == pytest.approx(2.83e-4, abs=5e-7)
    assert c.v_hi - 0.3 == pytest.approx(1.413e-3, abs=5e-7)
    assert c.v_hi * 10.0 / R == pytest.approx(0.0603, abs=5e-5)
    assert c.delta == pytest.approx(0.00898, abs=5e-6)


@pytest.mark.parametrize(
    "kw",
    [
        {"rho_cap": 20000.0},
        {"dt": 460.0},
        {"v_cap": 1.0, "dt": 30.0},
        {"dt": 45.0},
        {"dt": 60.0},
        {"rho_cap": 60.0},
    ],
    ids=[
        "CW terms above a_max",
        "2 n dt above 1",
        "losses above the radial share",
        "speed cap lost at dt 45",
        "speed cap lost at dt 60",
        "no room to stop between R and rho_cap",
    ],
)
def test_braking_deceleration_is_zero_when_the_backup_cannot_keep_its_set(kw):
    # Each breaks the reset condition its id names. The conditions delta <= a_R dt and
    # eps <= a_R follow from a_b > 0, so they cannot fail alone.
    assert default_deceleration(**kw) == 0.0


def test_braking_deceleration_admits_the_approach_box_up_to_dt_33_37_s():
    expected = braking_constants(dt=33.37).a_b
    assert default_deceleration(dt=33.37) == pytest.approx(expected, rel=1e-12)
    assert default_deceleration(dt=33.4) == 0.0


def test_braking_deceleration_takes_no_layout():
    with pytest.raises(TypeError):
        braking_deceleration(
            a_max=A_MAX, layout="box", v_cap=0.3, dt=10.0, radius=R, rho_cap=RHO_CAP, n=N
        )


@pytest.mark.parametrize("a_b", [0.0, -1e-3, math.nan])
def test_braking_barrier_refuses_a_nonpositive_deceleration(a_b):
    with pytest.raises(ValueError):
        braking_barrier(a_b=a_b)


def test_braking_barrier_refuses_a_region_the_thrust_cannot_hold():
    # At 20 km the CW terms alone exceed 0.02 m/s^2.
    assert default_deceleration(rho_cap=20000.0) <= 0.0
    with pytest.raises(ValueError):
        braking_barrier(a_b=1e-3, rho_cap=20000.0)


def test_braking_barrier_refuses_an_a_b_its_backup_cannot_deliver():
    derived = default_deceleration()
    braking_barrier(a_b=derived)
    braking_barrier(a_b=0.5 * derived)
    with pytest.raises(ValueError):
        braking_barrier(a_b=derived * (1 + 1e-9))
    with pytest.raises(ValueError):
        braking_barrier(a_b=0.00991)


def test_braking_backup_refuses_an_actuator_weaker_than_its_own():
    x = np.array([30.0, -40.0, 10.0, -0.1, 0.05, 0.0])
    with pytest.raises(ValueError):
        braking_barrier().backup(x, ActuatorSet(0.019, "box"))


@functools.cache
def in_set_draws():
    barrier = braking_barrier()
    states = braking_draws(APPROACH, np.random.default_rng(1903), 200)
    return tuple(x for x in states if barrier.contains(x))


def test_braking_backup_cancels_the_cw_acceleration():
    barrier = braking_barrier()
    draws = in_set_draws()
    assert len(draws) >= 400
    for x in draws:
        u = np.asarray(barrier.backup(x, ActuatorSet(A_MAX, "box")))
        np.testing.assert_allclose(u + a_cw(x), braking_law(x), rtol=0.0, atol=1e-15)


def test_braking_backup_is_the_same_command_in_both_layouts_and_inside_the_ball():
    barrier = braking_barrier()
    for x in in_set_draws():
        box = np.asarray(barrier.backup(x, ActuatorSet(A_MAX, "box")))
        ball = np.asarray(barrier.backup(x, ActuatorSet(A_MAX, "ball")))
        np.testing.assert_array_equal(box, ball)
        assert np.linalg.norm(box) <= A_MAX


def held_change(barrier, x, dt=10.0):
    """v0 + beta' dt along rhat0, along that0 and across both, m/s."""
    u = np.asarray(barrier.backup(x, ActuatorSet(A_MAX, "box")))
    _, rhat, _, v_t, w = braking_terms(x)
    that = v_t / w
    end = x[3:] + (u + a_cw(x)) * dt
    return end @ rhat, end @ that, abs(end @ np.cross(rhat, that))


@pytest.mark.parametrize("w", [0.02, 0.0847, 0.2, 0.29])
def test_braking_backup_brakes_tangential_speed_without_reversing_it(w):
    # a_T dt = 0.08475 m/s: below it the tangential speed is zeroed in one step, above it cut by
    # that much.
    barrier = braking_barrier()
    for rho, p in ((300.0, -0.01), (300.0, 0.01), (1500.0, 0.01), (1500.0, -0.01)):
        x = np.array([0.0, rho, 0.0, w, p, 0.0])
        _, along, across = held_change(barrier, x)
        assert along == pytest.approx(max(0.0, w - APPROACH.a_r * 10.0), abs=1e-15)
        assert across <= 1e-15


@pytest.mark.parametrize("p", [0.0, 0.02, 0.05, 0.09])
def test_braking_backup_targets_a_slow_opening_inside_and_a_slow_closing_outside(p):
    # Unclipped while |target - rho_dot| <= a_R dt = 0.08475 m/s; rho_mid is 1025 m.
    barrier = braking_barrier()
    inside = np.array([0.0, 300.0, 0.0, 0.1, p, 0.0])
    outside = np.array([0.0, 1500.0, 0.0, 0.1, -p, 0.0])
    assert held_change(barrier, inside)[0] == pytest.approx(APPROACH.delta, rel=1e-12)
    assert held_change(barrier, outside)[0] == pytest.approx(-APPROACH.delta, rel=1e-12)


@pytest.mark.parametrize(
    ("rho", "p", "a"),
    [
        (300.0, -0.2, APPROACH.a_r),
        (300.0, -1e-6, APPROACH.a_r),
        (300.0, 0.25, -APPROACH.a_r),
        (1500.0, 0.2, -APPROACH.a_r),
        (1500.0, 1e-6, -APPROACH.a_r),
        (1500.0, -0.25, APPROACH.a_r),
    ],
)
def test_braking_backup_brakes_at_the_radial_share_when_far_from_its_target(rho, p, a):
    barrier = braking_barrier()
    x = np.array([0.0, rho, 0.0, 0.1, p, 0.0])
    assert held_change(barrier, x)[0] == pytest.approx(p + a * 10.0, rel=1e-12)


@functools.cache
def held_steps(dt):
    """The set's draws at dt, their backup commands and their states on 401 step samples."""
    c = braking_constants(dt=dt)
    barrier = braking_barrier(dt=dt)
    states = braking_draws(c, np.random.default_rng(20261025), 400)
    states = np.array([x for x in states if barrier.contains(x)])
    actuator = ActuatorSet(A_MAX, "box")
    u = np.array([barrier.backup(x, actuator) for x in states])
    phi, gamma = cw_discrete(N, np.linspace(0.0, dt, 401))
    path = np.einsum("tij,mj->tmi", phi, states) + np.einsum("tij,mj->tmi", gamma, u)
    return barrier, c, states, u, path


@pytest.mark.parametrize("layout", ["box", "ball"])
@pytest.mark.parametrize("dt", [1.0, 10.0, 33.37])
def test_braking_backup_keeps_the_set_over_a_held_step(dt, layout):
    barrier, c, states, u, path = held_steps(dt)
    assert len(states) >= 800
    size = np.max(np.abs(u), axis=1) if layout == "box" else np.linalg.norm(u, axis=1)
    assert np.all(size <= A_MAX)
    h_b, _ = braking_values(path, barrier.a_b)
    assert h_b.min() >= -1e-9
    # Between samples the speed may exceed v_cap by E = eps dt / 2, 1.4 mm/s at dt 10 s.
    assert np.linalg.norm(path[..., 3:], axis=-1).max() <= c.v_hi
    for x1 in path[-1]:
        assert barrier.contains(x1)
        assert np.linalg.norm(x1[3:]) <= 0.3


# 100 starts take about 4 s of Python calls; the default run flies 20.
FLOWN = (pytest.param(100, marks=pytest.mark.slow), 20)


@pytest.mark.parametrize("count", FLOWN)
def test_braking_backup_flown_alone_never_leaves_its_set(count):
    barrier = braking_barrier()
    actuator = ActuatorSet(A_MAX, "box")
    phi, gamma = cw_discrete(N, 10.0)
    draws = in_set_draws()
    starts = draws[:: len(draws) // count][:count]
    assert len(starts) == count
    for x in starts:
        path = []
        for _ in range(360):
            x = phi @ x + gamma @ np.asarray(barrier.backup(x, actuator))
            path.append(x)
        path = np.array(path)
        h_b, h_o = braking_values(path, barrier.a_b)
        assert h_b.min() >= 0.0 and h_o.min() >= 0.0
        assert np.linalg.norm(path[:, 3:], axis=1).max() <= 0.3


# Fast states outside the set, beyond rho_cap and opening at 0.87 m/s: cancelling the CW
# acceleration and braking at the full shares would take 0.0202 to 0.0206 m/s^2 on the largest
# axis and 0.0205 to 0.0208 m/s^2 in size.
FAST_OUTSIDE = (
    np.array([2100.0, -950.0, 0.0, 0.45, 0.74, 0.0]),
    np.array([2000.0, -900.0, 0.0, 0.45, 0.74, 0.0]),
)


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_braking_backup_from_a_fast_state_outside_its_set_stays_within_the_thrust_limit(layout):
    barrier = braking_barrier()
    order = np.inf if layout == "box" else 2
    for x in FAST_OUTSIDE:
        assert not barrier.contains(x)
        assert np.linalg.norm(braking_law(x) - a_cw(x), order) > A_MAX
        u = np.asarray(barrier.backup(x, ActuatorSet(A_MAX, layout)))
        assert np.linalg.norm(u, order) <= A_MAX


# Sample-and-hold of the exponential CBF on a double integrator, h'' = u with
# u_k = -k0 h_k - k1 hdot_k held for dt; in units of 1/p only s = p dt matters.


def hold_worst(s, rays, steps):
    """Lowest h (inter-sample dips included) and lowest hdot + p h from unit starts on C0 n C1."""
    k0, k1 = KeepOutEcbf(radius=R, poles=(1.0, 1.0)).gains
    th = np.linspace(0.0, 1.0, rays)
    # Rays from the C1 edge (1, -1) to the C0 edge (0, 1) of the cone, normalised.
    h, hd = 1.0 - th, -(1.0 - th) + th
    norm = np.hypot(h, hd)
    h, hd = h / norm, hd / norm
    low_h, low_nu = np.inf, np.inf
    for _ in range(steps):
        u = -k0 * h - k1 * hd
        h1 = h + hd * s + 0.5 * u * s * s
        hd1 = hd + u * s
        pushing = u > 0
        tau = np.where(pushing, -hd / np.where(pushing, u, 1.0), 0.0)
        inside = pushing & (hd < 0) & (tau < s)
        dip = np.where(inside, h + hd * tau + 0.5 * u * tau * tau, np.inf)
        low_h = min(low_h, float(np.min(np.minimum(np.minimum(h, h1), dip))))
        h, hd = h1, hd1
        low_nu = min(low_nu, float(np.min(hd + h)))
    return low_h, low_nu


def first_failure(fails, lo=0.5, hi=0.9):
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if fails(mid) else (mid, hi)
    return 0.5 * (lo + hi)


def test_double_integrator_hold_thresholds_lie_above_the_pole_rule():
    s_set = first_failure(lambda s: hold_worst(s, 181, 60)[1] < -1e-12)
    s_h = first_failure(lambda s: hold_worst(s, 181, 60)[0] < -1e-12)
    # From the start (h, hdot) = (0, 1) the second step's dip touches zero at the root of
    # 2 s^4 + 4 s^3 - 6 s^2 + 1 in (0.6, 1); C1 is lost first on the same ray at (sqrt 5 - 1) / 2.
    roots = np.roots([2.0, 4.0, -6.0, 0.0, 1.0])
    (root,) = roots[(abs(roots.imag) < 1e-12) & (roots.real > 0.6) & (roots.real < 1.0)].real
    assert s_set == pytest.approx((math.sqrt(5.0) - 1.0) / 2.0, abs=1e-4)
    assert s_set == pytest.approx(0.618034, abs=1e-4)
    assert s_h == pytest.approx(root, abs=1e-4)
    assert s_h == pytest.approx(0.786578, abs=1e-4)
    assert 0.6 < s_set < s_h


def head_on_fails(p, dt, t_end=1500.0):
    """Whether the ECBF without a thrust limit lets a reckless inward push break the keep-out."""
    barrier = KeepOutEcbf(radius=R, poles=(p, p))
    phi, gamma = cw_discrete(N, np.linspace(0.0, dt, 201))
    x = np.array([0.0, R, 0.0, 0.0, 0.1, 0.0])
    for _ in range(int(np.ceil(t_end / dt))):
        a, b = barrier.row(x, a_cw(x))
        u_nom = -x[:3] / np.linalg.norm(x[:3])
        u = project(u_nom, Rows(np.array([a]), np.array([b]), ("keep_out",))).u
        segment = phi @ x + gamma @ u
        if np.linalg.norm(segment[:, :3], axis=1).min() - R < -1e-3:
            return True
        x = segment[-1]
    return False


@pytest.mark.parametrize("p", [0.05, 0.1])
def test_cw_head_on_first_fails_at_the_double_integrator_threshold_over_p(p):
    # The double-integrator threshold 0.78658 / p carries over to CW, where n << p.
    predicted = 0.78658 / p
    assert not head_on_fails(p, 0.5 * predicted)
    assert not head_on_fails(p, 0.997 * predicted)
    assert head_on_fails(p, 1.003 * predicted)


def drift_positions(x, t):
    """Free-drift positions (k, m, 3) of states x (k, 6) at times t (m,), closed-form CW."""
    n = N
    c, s = np.cos(n * t), np.sin(n * t)
    x0, y0, z0, vx, vy, vz = (x[:, i : i + 1] for i in range(6))
    rx = 4 * x0 - 3 * x0 * c + vx / n * s + 2 * vy / n * (1 - c)
    ry = y0 + 6 * x0 * (s - n * t) - 2 * vx / n * (1 - c) + vy / n * (4 * s - 3 * n * t)
    rz = z0 * c + vz / n * s
    return np.stack([rx, ry, rz], axis=-1)


def exact_drift_minima(x, t_cap, after=None):
    """Lowest |r_drift| - R over [after_i, t_cap] (after_i = 0 by default), refined to 1e-9 s."""
    t = np.arange(0.0, t_cap + 2.0, 2.0)
    starts = np.zeros(len(x)) if after is None else np.asarray(after, dtype=float)
    rho = np.linalg.norm(drift_positions(x, t), axis=-1)
    rho = np.where((t[None, :] >= starts[:, None]) & (t[None, :] <= t_cap), rho, np.inf)
    minima = []
    for state, row, start in zip(x, rho, starts, strict=True):
        k = int(np.argmin(row))
        found = minimize_scalar(
            lambda tau, state=state: np.linalg.norm(drift_positions(state[None], np.array([tau]))),
            bounds=(max(t[k] - 2.0, start), min(t[k] + 2.0, t_cap)),
            method="bounded",
            options={"xatol": 1e-9},
        )
        minima.append(min(float(found.fun), float(row[k])) - R)
    return np.array(minima)


def test_capped_passive_value_never_exceeds_the_exact_drift_minimum_for_slow_drift_and_large_vx():
    # Slow secular drift |6 n x + 3 vy| with up to 0.5 m/s radial speed: the periodic along-track
    # swing 4 |vx| / n of the CW solution must enter the drift bound B, or the horizon ends before
    # the drift reaches its lowest range.
    rng = np.random.default_rng(11)
    k = 12
    x0 = rng.uniform(-50.0, 50.0, k)
    states = np.column_stack(
        [
            x0,
            rng.uniform(-300.0, 300.0, k),
            rng.uniform(-50.0, 50.0, k),
            rng.uniform(-0.5, 0.5, k),
            -2.0 * N * x0 + rng.uniform(-1e-3, 1e-3, k),
            rng.uniform(-0.05, 0.05, k),
        ]
    )
    states = np.vstack([states, [2.2966, -194.6066, 3.6318, -0.4973, 0.0093, 0.0]])
    states = states[np.linalg.norm(states[:, :3], axis=1) > R + 1.0]
    barrier = capped_barrier()
    assert np.all(np.abs(6 * N * states[:, 0] + 3 * states[:, 4]) > 1e-9)
    values = barrier.value(states)
    exact = exact_drift_minima(states, barrier.t_cap)
    assert np.all(values <= exact + 1e-6)
    horizons = [barrier.horizon(x) for x in states]
    proven = np.array([p for _, p in horizons])
    later = exact_drift_minima(states[proven], barrier.t_cap, [t for t, p in horizons if p])
    assert np.all(later >= values[proven] - 1e-6)


# Terminal free-drift set. In the CW solution r = (x_c + b cos psi, y_c - 2 b sin psi,
# a_z cos(psi + delta)), psi = n t + alpha, with x_c = 4 x + 2 vy / n, y_c = y - 2 vx / n,
# b = |(3 x + 2 vy / n, vx / n)|, a_z = |(z, vz / n)| and dy_c/dt = -1.5 n x_c. Only y_c moves
# under free drift, so a set judged on the other invariants, or on a y_c that only recedes, is
# kept by every drift step.
START_BOX = ((-200.0, 200.0, -50.0, -0.1, -0.1, -0.1), (200.0, 600.0, 50.0, 0.1, 0.1, 0.1))
NEAR_BOX = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))


def test_hold_at_rest_on_the_v_bar_has_its_static_far_margin_exactly():
    x = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    assert terminal_barrier().value(x) == pytest.approx(50.0, abs=1e-12)


def test_hold_receding_from_the_target_is_in_with_its_far_margin():
    # vy = 1e-4 m/s at y = -100 m: x_c = 2 vy / n = 0.18 m, so y_c moves away at 1.5 n x_c.
    vy = 1e-4
    x = np.array([0.0, -100.0, 0.0, 0.0, vy, 0.0])
    b = 2.0 * vy / N
    assert terminal_barrier().contains(x)
    assert terminal_barrier().value(x) == pytest.approx(100.0 - 2.0 * b - R, abs=1e-9)


def test_hold_drifting_slowly_toward_the_target_is_out():
    # vy = -1e-6 m/s at y = -100 m: the centre closes at 3e-6 m/s and slides through the target
    # after about 5e7 s, far past t_cap, so no finite check proves this drift clear.
    x = np.array([0.0, -100.0, 0.0, 0.0, -1e-6, 0.0])
    assert terminal_barrier().value(x) < 0.0
    assert not terminal_barrier().contains(x)


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_closed_orbit_clear_of_the_keep_out_is_in_the_terminal_set(side):
    # x = vy = 0 makes x_c exactly zero: a 2:1 ellipse about the target, 60 m radial and 120 m
    # along-track, crossing the V-bar at y = -120 and y = 120 m and nearest the target at 60 m.
    x = np.array([0.0, side * 120.0, 0.0, side * 60.0 * N, 0.0, 0.0])
    period = np.linspace(0.0, 2.0 * np.pi / N, 4001)
    rho = np.linalg.norm(drift_positions(x[None], period)[0], axis=1)
    assert rho.min() == pytest.approx(60.0, abs=1e-3)
    assert 9.99 <= terminal_barrier().value(x) <= 10.0


def test_closed_orbit_through_the_keep_out_is_out():
    x = np.array([0.0, -40.0, 0.0, -20.0 * N, 0.0, 0.0])
    assert not terminal_barrier().contains(x)


@pytest.mark.parametrize(
    ("x", "low", "high"),
    [
        # Helix with b = a_z = 60 m, delta = pi / 2, x_c = 0: |(x, z)| is 60 m at every phase.
        (np.array([0.0, -120.0, -60.0, -60.0 * N, 0.0, 0.0]), 9.99, 10.0),
        # x_c = 100 m, b = 30 m, a_z = 0: the centre slides through the target along-track
        # while |x| >= 70 m.
        (np.array([130.0, 300.0, 0.0, 0.0, -210.0 * N, 0.0]), 19.99, 20.0),
    ],
)
def test_state_whose_radial_and_cross_track_tube_clears_the_keep_out_is_in(x, low, high):
    assert low <= terminal_barrier().value(x) <= high
    t = np.arange(0.0, 2.0 * T_CAP, 10.0)
    r = drift_positions(x[None], t)[0]
    assert np.linalg.norm(r[:, [0, 2]], axis=1).min() >= R + low
    assert r[:, 1].min() < 0.0 < r[:, 1].max()


def test_tube_value_stays_below_the_exact_tube_minimum_between_phase_samples():
    # x_c = 100 m, b = 30 m, a_z = 40 m and delta = 0.3 rad, taken at psi = 0: the cross-track
    # term moves the lowest tube radius off pi, between the phases the bound samples.
    x_c, b, a_z, delta = 100.0, 30.0, 40.0, 0.3
    x = np.array(
        [
            x_c + b,
            300.0,
            a_z * math.cos(delta),
            0.0,
            -N * (1.5 * x_c + 2.0 * b),
            -a_z * N * math.sin(delta),
        ]
    )

    def tube(psi):
        return math.hypot(x_c + b * math.cos(psi), a_z * math.cos(psi + delta))

    psi = np.linspace(0.0, 2.0 * np.pi, 100001)
    k = int(np.argmin(np.hypot(x_c + b * np.cos(psi), a_z * np.cos(psi + delta))))
    lowest = minimize_scalar(
        tube, bounds=(psi[k - 1], psi[k + 1]), method="bounded", options={"xatol": 1e-12}
    ).fun
    value = terminal_barrier().value(x)
    assert value <= lowest - R
    assert value >= lowest - R - 0.01


def test_terminal_flag_must_be_a_bool():
    with pytest.raises(TypeError):
        PassiveBarrier(radius=R, n=N, terminal=1)


@pytest.mark.parametrize("box", [START_BOX, NEAR_BOX], ids=["start", "near"])
def test_terminal_set_lies_inside_the_capped_set(box):
    xs = np.random.default_rng(1135).uniform(box[0], box[1], (100, 6))
    inside = terminal_barrier().value(xs) >= 0.0
    assert inside.any()
    assert np.all(capped_barrier().value(xs[inside]) >= 0.0)


def test_terminal_set_states_drift_clear_of_the_keep_out_for_three_cap_horizons():
    xs = np.random.default_rng(1136).uniform(NEAR_BOX[0], NEAR_BOX[1], (24, 6))
    values = terminal_barrier().value(xs)
    xs, values = xs[values >= 0.0], values[values >= 0.0]
    assert len(xs) >= 12
    # Sampled minima lie above the true ones, so coarse samples can only miss a violation.
    t = np.arange(0.0, 3.0 * T_CAP, 4.0)
    margin = np.linalg.norm(drift_positions(xs, t), axis=-1).min(axis=1) - R
    assert np.all(values <= margin + 1e-6)


def exact_drift_margin(x, t):
    """Range minus R (m) `t` s along the free drift from `x`, to 50 digits.

    The closed-form CW solution, taking the floats in `x` and N as exact. The series for sin n t
    has terms up to 6e22 at n t = 55 rad, so the sums carry 80 digits.
    """
    with localcontext() as ctx:
        ctx.prec = 80
        rx0, ry0, rz0, vx, vy, vz = (Decimal(float(c)) for c in x)
        n, t = Decimal(float(N)), Decimal(float(t))
        s, c = decimal_sin_cos(n * t)
        rx = (4 - 3 * c) * rx0 + s / n * vx + 2 * (1 - c) / n * vy
        ry = 6 * (s - n * t) * rx0 + ry0 - 2 * (1 - c) / n * vx + (4 * s - 3 * n * t) / n * vy
        rz = c * rz0 + s / n * vz
        return (rx * rx + ry * ry + rz * rz).sqrt() - Decimal(R)


def test_drift_margin_at_a_sample_minimum_never_exceeds_the_exact_minimum():
    # A horizon that is a whole number of 5 s samples puts the last sample at the horizon.
    capped = PassiveBarrier(radius=R, n=N, t_cap=50000.0)
    cases = []
    # Secular drifts from 2.5 km along-track that reach the sphere at the horizon, closing at
    # 1.5 n d, so their lowest range is at the last sample.
    for d in np.linspace(20.0, 45.0, 6):
        for k in range(5):
            y_end = math.sqrt(R * R - d * d) + 1e-9 * k
            x = np.array([d, y_end + 1.5 * N * d * capped.t_cap, 0.0, 0.0, -1.5 * N * d, 0.0])
            cases.append((capped, x, capped.t_cap))
    # Starts on the sphere moving radially out, lowest at the first sample; off the axes so the
    # norm rounds.
    rng = np.random.default_rng(1137)
    for _ in range(20):
        u = np.array([1.0, 0.05 * rng.standard_normal(), 0.05 * rng.standard_normal()])
        u /= np.linalg.norm(u)
        x = np.concatenate([u * (R + 1e-12 * rng.integers(0, 5)), 0.1 * u])
        cases += [(capped_barrier(), x, 0.0), (terminal_barrier(), x, 0.0)]
    excess = np.array(
        [float(Decimal(float(b.value(x))) - exact_drift_margin(x, t)) for b, x, t in cases]
    )
    assert excess.max() <= 0.0
    # Rounding over a 50000 s drift is near 1e-8 m; 1e-6 m catches an allowance far larger.
    assert excess.min() >= -1e-6
