"""Exact constrained-LQ oracle on the zero-order-hold CW model.

With the Riccati matrix P as terminal weight, the box-constrained finite-horizon optimum V_N
equals the constrained infinite-horizon value once its predicted terminal state lies in the
maximal admissible set of the LQR loop, and z0' P z0 bounds the cost of any controller from
below (Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed., chapter 2, section 2.5.4
and the chapter notes; the steady-state target of chapter 1, section 1.5.1). The admissible set
is the Gilbert and Tan (1991) set of the LQR loop with the thrust box. Every case is for a 500 km
circular orbit of the WGS 84 Earth, a 10 s step and a 0.02 m/s^2 box per axis, with the weights
Q = diag(1/10^2 x3, 1/0.01^2 x3) / 2 and R = I / 0.02^2 / 2: the book's stage cost carries a
factor 1/2 and the oracle's does not, so the 1/2 is folded into the weights.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import lsq_linear

from vbar.agents.lqr import dlqr, steady_state_input
from vbar.agents.mpc import admissible_set
from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.oracles.lq import LQBound, LQOracle
from vbar.rewards import QuadraticCost

N = mean_motion(EARTH.radius + 500e3)
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
PHI, GAMMA = cw_discrete(N, DT)
Q_HALF = np.diag([1 / 10.0**2] * 3 + [1 / 0.01**2] * 3) / 2
R_HALF = np.eye(3) / A_MAX**2 / 2
BOX = np.full(3, A_MAX)
ALONG_TRACK = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
RADIAL = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Three times the along-track offset at rest at which the LQR first saturates.
OFFSET_VALUE = 1.855711116e5
OFFSET_LOWER = 1.845593979e5
# Half a unit in the last printed digit of the two values above.
PRINTED = 5e-4
# The finite-horizon value and the bound come from different solves (a QP and a Riccati
# equation); their agreement and that of the oracle with the reference QP sit near 1e-13 here.
REL = 1e-9


def oracle(**kwargs):
    return LQOracle(n=N, dt=DT, q=Q_HALF, r=R_HALF, a_max=A_MAX, **kwargs)


@pytest.fixture(scope="module")
def design():
    k, p = dlqr(PHI, GAMMA, Q_HALF, R_HALF)
    rows, _ = admissible_set(PHI - GAMMA @ k, k, BOX)
    s_max = A_MAX / np.max(np.abs(rows @ ALONG_TRACK))
    # 201.5 m from the Gilbert-Tan rows of this loop (t* = 1).
    assert abs(s_max - 201.5) < 0.05
    return SimpleNamespace(k=k, p=p, rows=rows, s_max=s_max)


def set_ratio(rows, z):
    """Largest |row . z| over the box limit; at most 1 inside the admissible set."""
    return float(np.max(np.abs(rows @ z)) / A_MAX)


def reference_qp(z0, steps, p_f, *, lo=-BOX, hi=BOX, q=Q_HALF, r=R_HALF, s=None):
    """Box-constrained optimum of the stage sum plus z_N' p_f z_N, and its terminal state.

    Condensed over the inputs and solved by bounded-variable least squares on a Cholesky factor
    of the Hessian, independent of the oracle's own solve.
    """
    s = np.zeros((6, 3)) if s is None else s
    powers = [np.eye(6)]
    for _ in range(steps):
        powers.append(PHI @ powers[-1])
    pushed = [p @ GAMMA for p in powers]
    su = np.zeros((6 * (steps + 1), 3 * steps))
    for i in range(1, steps + 1):
        for j in range(i):
            su[6 * i : 6 * i + 6, 3 * j : 3 * j + 3] = pushed[i - 1 - j]
    free = np.concatenate([p @ z0 for p in powers])
    qb = np.zeros((6 * (steps + 1), 6 * (steps + 1)))
    sb = np.zeros((6 * (steps + 1), 3 * steps))
    for i in range(steps):
        qb[6 * i : 6 * i + 6, 6 * i : 6 * i + 6] = q
        sb[6 * i : 6 * i + 6, 3 * i : 3 * i + 3] = s
    qb[-6:, -6:] = p_f
    h = su.T @ qb @ su + np.kron(np.eye(steps), r) + su.T @ sb + sb.T @ su
    f = (su.T @ qb + sb.T) @ free
    chol = np.linalg.cholesky(0.5 * (h + h.T))
    sol = lsq_linear(
        chol.T,
        -np.linalg.solve(chol, f),
        bounds=(np.tile(lo, steps), np.tile(hi, steps)),
        method="bvls",
        tol=1e-15,
    )
    u = sol.x.reshape(steps, 3)
    z, value = np.array(z0, dtype=float), 0.0
    for uk in u:
        value += z @ q @ z + uk @ r @ uk + 2.0 * z @ s @ uk
        z = PHI @ z + GAMMA @ uk
    return value + z @ p_f @ z, z


def clipped_lqr_cost(z0, k, p, rows, *, q=Q_HALF, r=R_HALF, s=None, max_steps=500):
    """Infinite-horizon cost of the LQR clipped to the box.

    Inside the admissible set the clipped law is the LQR, whose remaining cost is z' P z.
    """
    s = np.zeros((6, 3)) if s is None else s
    z, total = np.array(z0, dtype=float), 0.0
    for _ in range(max_steps):
        if set_ratio(rows, z) <= 1.0:
            return total + z @ p @ z
        u = np.clip(-k @ z, -A_MAX, A_MAX)
        total += z @ q @ z + u @ r @ u + 2.0 * z @ s @ u
        z = PHI @ z + GAMMA @ u
    raise AssertionError("the clipped LQR did not reach the admissible set")


@pytest.mark.parametrize("steps", [5, 10, 20, 40, 80, 120])
def test_constrained_value_from_three_saturation_offsets_is_the_same_at_every_horizon(
    design, steps
):
    bound = oracle(horizons=(steps,))(3.0 * design.s_max * ALONG_TRACK)
    assert isinstance(bound, LQBound)
    assert bound.valid
    assert bound.horizon == steps
    assert abs(bound.value - OFFSET_VALUE) <= PRINTED
    assert abs(bound.lower - OFFSET_LOWER) <= PRINTED


def test_horizons_are_tried_in_order_and_the_first_valid_one_is_reported(design):
    x0 = 3.0 * design.s_max * ALONG_TRACK
    assert oracle()(x0).horizon == 5
    # After one step the predicted state is still twice outside the set; after two it is inside.
    bound = oracle(horizons=(1, 2, 3))(x0)
    assert bound.valid
    assert bound.horizon == 2
    assert abs(bound.value - OFFSET_VALUE) <= PRINTED


def test_one_step_horizon_ends_outside_the_set_and_reports_only_the_bound(design):
    x0 = 3.0 * design.s_max * ALONG_TRACK
    _, z1 = reference_qp(x0, 1, design.p)
    assert set_ratio(design.rows, z1) > 1.9
    bound = oracle(horizons=(1,))(x0)
    assert bound.valid is False
    assert np.isnan(bound.value)
    assert bound.horizon is None
    assert abs(bound.lower - OFFSET_LOWER) <= PRINTED


@pytest.mark.parametrize(
    ("steps", "offsets", "inside"),
    [
        (1, 1.5, True),
        (1, 3.0, False),
        (2, 2.0, True),
        (2, 5.0, False),
        (3, 3.0, True),
        (3, 5.0, False),
        (5, 5.0, True),
        (5, 8.0, False),
    ],
)
def test_validity_flag_is_false_exactly_when_the_terminal_state_leaves_the_set(
    design, steps, offsets, inside
):
    x0 = offsets * design.s_max * ALONG_TRACK
    value, z_n = reference_qp(x0, steps, design.p)
    ratio = set_ratio(design.rows, z_n)
    assert (ratio <= 1.0) == inside
    # Every case is at least 2% from the boundary, far above the solvers' rounding.
    assert abs(ratio - 1.0) > 0.02
    bound = oracle(horizons=(steps,))(x0)
    assert bound.valid is inside
    assert bound.lower == pytest.approx(x0 @ design.p @ x0, rel=REL)
    if inside:
        assert bound.horizon == steps
        assert bound.value == pytest.approx(value, rel=REL)
    else:
        assert bound.horizon is None
        assert np.isnan(bound.value)


def test_bound_and_value_are_sound_against_the_clipped_lqr_from_random_starts(design):
    rng = np.random.default_rng(20261022)
    starts = rng.uniform(-1.0, 1.0, (10, 6)) * np.array([600.0, 600.0, 200.0, 0.3, 0.3, 0.3])
    strict = 0
    lqr = oracle()
    for x0 in starts:
        # Every start saturates the first LQR command, so the constraints bind.
        assert np.max(np.abs(design.k @ x0)) > A_MAX
        clipped = clipped_lqr_cost(x0, design.k, design.p, design.rows)
        # Once the terminal state is inside the set, V_N no longer changes with N.
        long_value, z_n = reference_qp(x0, 10, design.p)
        assert set_ratio(design.rows, z_n) < 1.0
        bound = lqr(x0)
        assert bound.valid
        assert bound.lower <= bound.value
        assert bound.lower == pytest.approx(x0 @ design.p @ x0, rel=REL)
        assert bound.value == pytest.approx(long_value, rel=REL)
        assert clipped >= bound.value * (1.0 - REL)
        strict += clipped > bound.value * (1.0 + 1e-7)
    # The clipped LQR is optimal from some of these starts and strictly dearer from others.
    assert strict >= 2


@pytest.mark.parametrize(
    "x0",
    [
        np.array([1500.0, -1500.0, 500.0, 1.0, -1.0, 0.5]),
        np.array([-800.0, 1200.0, -300.0, -0.5, 1.5, 1.0]),
    ],
)
def test_value_skips_horizons_that_end_outside_the_set(design, x0):
    # The first LQR command asks for 13 times the thrust limit.
    _, z_10 = reference_qp(x0, 10, design.p)
    value_20, z_20 = reference_qp(x0, 20, design.p)
    assert set_ratio(design.rows, z_10) > 1.0
    assert set_ratio(design.rows, z_20) < 1.0
    bound = oracle()(x0)
    assert bound.valid
    assert bound.horizon == 20
    assert bound.value == pytest.approx(value_20, rel=REL)
    assert bound.lower <= bound.value


def test_start_too_far_for_every_horizon_keeps_a_sound_bound(design):
    x0 = 100e3 * ALONG_TRACK
    long_value, z_n = reference_qp(x0, 20, design.p)
    assert set_ratio(design.rows, z_n) > 1.0
    bound = oracle(horizons=(5, 10, 20))(x0)
    assert bound.valid is False
    assert np.isnan(bound.value)
    assert bound.horizon is None
    assert bound.lower == pytest.approx(x0 @ design.p @ x0, rel=REL)
    assert bound.lower <= long_value


@pytest.mark.parametrize(
    "x0",
    [
        0.5 * 201.5 * ALONG_TRACK,
        0.99 * 201.5 * ALONG_TRACK,
        np.array([20.0, -40.0, 15.0, 0.01, -0.005, 0.002]),
    ],
)
def test_value_equals_the_bound_when_no_constraint_binds(design, x0):
    assert set_ratio(design.rows, x0) < 1.0
    bound = oracle()(x0)
    assert bound.valid
    assert bound.lower == pytest.approx(x0 @ design.p @ x0, rel=REL)
    assert bound.value == pytest.approx(bound.lower, rel=REL)


def test_start_at_the_hold_point_costs_nothing():
    bound = oracle(hold=(0.0, -100.0, 0.0))(np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0]))
    assert bound.valid
    assert bound.lower == 0.0
    assert bound.value == pytest.approx(0.0, abs=1e-12)


def test_vbar_hold_point_only_shifts_the_start(design):
    hold = np.array([0.0, -100.0, 0.0])
    x0 = np.concatenate([hold, np.zeros(3)]) + 3.0 * design.s_max * ALONG_TRACK
    bound = oracle(hold=tuple(hold))(x0)
    assert bound.valid
    assert abs(bound.value - OFFSET_VALUE) <= PRINTED
    assert abs(bound.lower - OFFSET_LOWER) <= PRINTED


def test_radial_hold_shifts_the_thrust_box_by_the_holding_input():
    # Hold 100 m below the target: the holding input is 3 n^2 100 = 3.6749e-4 m/s^2 radial, so
    # the deviation input may span [-0.02 - us, 0.02 - us]. Values from the box-constrained QP
    # over 40 steps with that shifted box; its terminal state lies inside the LQR set. With an
    # unshifted box both starts would cost 45512.67924967781.
    hold = np.array([-100.0, 0.0, 0.0])
    xs = np.concatenate([hold, np.zeros(3)])
    lqr = oracle(hold=tuple(hold))
    above = lqr(xs + 300.0 * RADIAL)
    below = lqr(xs - 300.0 * RADIAL)
    assert above.valid and below.valid
    assert above.value == pytest.approx(45508.69042384245, rel=REL)
    assert below.value == pytest.approx(45516.80517143117, rel=REL)
    assert above.lower == pytest.approx(below.lower, rel=REL)
    assert above.lower <= above.value


def test_terminal_state_that_later_overdrives_a_thrusting_hold_is_not_valid():
    # Holding 5000 m below the target takes 92% of the radial limit. With these weights the
    # admissible set of the remaining radial headroom needs the rows h F^t up to t = 4, one more
    # than the set of the full box. The start was built so that the one-step optimum ends where
    # the LQR keeps within the box for four steps and then asks for 0.9% too much radial thrust.
    q = np.diag([1e-2] * 3 + [1.0] * 3)
    r = np.eye(3) / A_MAX**2
    hold = np.array([-5000.0, 0.0, 0.0])
    xs = np.concatenate([hold, np.zeros(3)])
    us = steady_state_input(PHI, GAMMA, xs)
    k, p = dlqr(PHI, GAMMA, q, r)
    z0 = np.array([1.39017932, 52.05816075, -47.96364239, 0.07347916, -0.6080067, 0.80137682])
    _, z = reference_qp(z0, 1, p, lo=-A_MAX - us, hi=A_MAX - us, q=q, r=r)
    full_rows, _ = admissible_set(PHI - GAMMA @ k, k, BOX)
    headroom = np.tile(A_MAX - np.abs(us), len(full_rows) // 3)
    assert np.max(np.abs(full_rows @ z) / headroom) < 0.995
    accels = []
    for _ in range(6):
        accels.append(us - k @ z)
        z = (PHI - GAMMA @ k) @ z
    accels = np.array(accels)
    assert np.max(np.abs(accels[:4])) <= A_MAX
    assert accels[4, 0] > 1.005 * A_MAX
    bound = LQOracle(n=N, dt=DT, q=q, r=r, a_max=A_MAX, hold=tuple(hold), horizons=(1,))(xs + z0)
    assert bound.valid is False
    assert np.isnan(bound.value)
    assert bound.lower == pytest.approx(z0 @ p @ z0, rel=REL)


def test_hold_point_beyond_the_thrust_limit_is_refused():
    # Holding 1000 km below the target needs 3 n^2 1e6 = 3.7 m/s^2.
    with pytest.raises(ValueError):
        oracle(hold=(-1e6, 0.0, 0.0))


def sampled_cost(reference):
    qc = np.diag([1 / 10.0**2] * 3 + [1 / 0.01**2] * 3) / 2
    rc = np.eye(3) / A_MAX**2 / 2
    return QuadraticCost(N, qc, rc, reference=reference)


def test_from_cost_value_is_the_quadratic_cost_of_the_lqr_it_implies():
    reference = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    cost = sampled_cost(reference)
    w = cost.matrix(DT, None)
    q, r, s = w[:6, :6], w[6:9, 6:9], w[:6, 6:9]
    k, p = dlqr(PHI, GAMMA, q, r, s)
    _, p_without_cross = dlqr(PHI, GAMMA, q, r)
    z0 = np.array([30.0, 79.7, -20.0, 0.01, -0.02, 0.005])
    x = reference + z0
    starts, accels = [], []
    # The loop's spectral radius is 0.99005, so after 2000 steps the remaining cost is 1e-17
    # of the total.
    for _ in range(2000):
        a = -k @ (x - reference)
        starts.append(x)
        accels.append(a)
        x = PHI @ x + GAMMA @ a
    accels = np.array(accels)
    assert np.max(np.abs(accels)) < A_MAX
    total = cost(np.array(starts), accels, np.full(len(accels), DT), None).sum()
    bound = LQOracle.from_cost(cost, dt=DT, a_max=A_MAX)(reference + z0)
    assert bound.valid
    assert bound.value == pytest.approx(total, rel=REL)
    assert bound.lower == pytest.approx(total, rel=REL)
    assert bound.lower == pytest.approx(z0 @ p @ z0, rel=REL)
    # The cross weight of the sampled cost moves the bound by 1.8e-3, so a Riccati solution
    # without it fails the line above.
    assert abs(z0 @ p_without_cross @ z0 / (z0 @ p @ z0) - 1.0) > 1e-3


def test_from_cost_value_includes_the_cross_weight_when_the_box_binds():
    # QP over 40 steps with the sampled weights including the cross term; the clipped LQR from
    # this start (three times its along-track saturation offset, 159.39 m) costs 1152388.0035.
    reference = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    x0 = reference + 478.1784912834536 * ALONG_TRACK
    bound = LQOracle.from_cost(sampled_cost(reference), dt=DT, a_max=A_MAX)(x0)
    assert bound.valid
    assert bound.value == pytest.approx(1152387.583084355, rel=REL)
    assert bound.lower == pytest.approx(1146632.9439076555, rel=REL)
    assert bound.value <= 1152388.003498732


def test_cost_on_the_acceleration_toward_a_thrusting_hold_is_refused():
    # Holding off the V-bar needs a steady input, and a cost on a rather than a - us then
    # grows without limit: the oracle raises or reports no value.
    reference = np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x0 = reference + 50.0 * ALONG_TRACK
    try:
        bound = LQOracle.from_cost(sampled_cost(reference), dt=DT, a_max=A_MAX)(x0)
    except ValueError:
        return
    assert bound.valid is False
    assert np.isnan(bound.value)


def test_moving_reference_is_refused():
    with pytest.raises(ValueError):
        LQOracle.from_cost(
            sampled_cost(np.array([0.0, -100.0, 0.0, 0.1, 0.0, 0.0])), dt=DT, a_max=A_MAX
        )


def test_ball_layout_is_refused():
    with pytest.raises(ValueError):
        oracle(layout="ball")
    with pytest.raises(ValueError):
        LQOracle.from_cost(sampled_cost(np.zeros(6)), dt=DT, a_max=A_MAX, layout="ball")


@pytest.mark.parametrize("delay", [0.5 * DT, 2.5, 1.5 * DT])
def test_input_delay_off_the_step_grid_is_refused(delay):
    # The value is exact for inputs held from one step start to the next; a fractional delay
    # splits each held input across two steps.
    with pytest.raises(ValueError):
        oracle(delay=delay)


def test_zero_delay_is_the_default(design):
    x0 = 3.0 * design.s_max * ALONG_TRACK
    assert oracle(delay=0.0)(x0) == oracle()(x0)


def test_bound_carries_the_cw_rung(design):
    # The value is exact for CW motion only; on another dynamics rung it is a model value.
    assert oracle()(3.0 * design.s_max * ALONG_TRACK).rung == "cw"
    assert oracle(horizons=(1,))(3.0 * design.s_max * ALONG_TRACK).rung == "cw"


@pytest.mark.parametrize(
    "x0",
    [np.zeros(3), np.zeros(7), np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])],
)
def test_malformed_start_is_refused(x0):
    with pytest.raises(ValueError):
        oracle()(x0)


@pytest.mark.parametrize("a_max", [0.0, -0.02, np.inf])
def test_thrust_limit_must_be_positive_and_finite(a_max):
    with pytest.raises(ValueError):
        LQOracle(n=N, dt=DT, q=Q_HALF, r=R_HALF, a_max=a_max)
