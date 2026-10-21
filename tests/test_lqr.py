"""Discrete LQR design on the exact zero-order-hold CW model.

Known answers follow Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed.: the hold-point
target and the one-step actuation delay of chapter 1 (sections 1.2.5 and 1.5.1, exercise 1.61),
the sampled-data stage cost of chapter 8 (section 8.5.2, Van Loan 1978) and the steady offset
under a constant disturbance of chapter 5 (section 5.5). The gain scaling with mean motion follows
Alfriend et al., Spacecraft Formation Flying, section 10.1.2. Every number is for a 500 km circular
orbit of the WGS 84 Earth and a 10 s step unless stated.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vbar.agents import lqr
from vbar.dynamics import EARTH, KERBIN, cw_discrete, cw_system, mean_motion

N = mean_motion(EARTH.radius + 500e3)
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
# Default scales: 10 m and 1 cm/s of error and full thrust each cost one unit per step.
Q_DEFAULT = np.diag([1 / 10.0**2] * 3 + [1 / 0.01**2] * 3)
R_DEFAULT = np.eye(3) / A_MAX**2
EPS = np.finfo(float).eps
R_BAR_HOLD = np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0])

coordinates = st.floats(-2000.0, 2000.0, allow_subnormal=False)


def spectral_radius(m):
    return np.max(np.abs(np.linalg.eigvals(m)))


def block_diag(*blocks):
    size = sum(b.shape[0] for b in blocks)
    out = np.zeros((size, size))
    at = 0
    for b in blocks:
        out[at : at + b.shape[0], at : at + b.shape[0]] = b
        at += b.shape[0]
    return out


def quadrature_cost(qc, rc, dt, panels=40):
    """Integrated ZOH stage cost by composite 10-node Gauss-Legendre quadrature.

    Integrates phi(s)' Qc phi(s), phi(s)' Qc gamma(s) and gamma(s)' Qc gamma(s) + Rc over
    [0, dt] with the closed-form matrices, an independent path to Van Loan's exponential.
    """
    nodes, weights = np.polynomial.legendre.leggauss(10)
    edges = np.linspace(0.0, dt, panels + 1)
    qd, rd, sd = np.zeros((6, 6)), np.zeros((3, 3)), np.zeros((6, 3))
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        s = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * weights
        phi, gamma = cw_discrete(N, s)
        qd += np.einsum("m,mji,jk,mkl->il", w, phi, qc, phi)
        sd += np.einsum("m,mji,jk,mkl->il", w, phi, qc, gamma)
        rd += np.einsum("m,mji,jk,mkl->il", w, gamma, qc, gamma) + np.sum(w) * rc
    return qd, rd, sd


def test_gain_closes_the_loop_with_the_negative_feedback_convention():
    phi, gamma = cw_discrete(N, DT)
    k, _ = lqr.dlqr(phi, gamma, Q_DEFAULT, R_DEFAULT)
    assert k.shape == (3, 6)
    assert spectral_radius(phi - gamma @ k) < 1.0
    # The opposite sign is far from stable (radius about 2), so the convention is unambiguous.
    assert spectral_radius(phi + gamma @ k) > 1.5


def test_riccati_solution_is_symmetric_and_satisfies_the_dare():
    phi, gamma = cw_discrete(N, DT)
    k, p = lqr.dlqr(phi, gamma, Q_DEFAULT, R_DEFAULT)
    assert np.array_equal(p, p.T)
    assert np.all(np.linalg.eigvalsh(p) > 0.0)
    gain = np.linalg.solve(R_DEFAULT + gamma.T @ p @ gamma, gamma.T @ p @ phi)
    assert np.allclose(k, gain, rtol=1e-12, atol=0.0)
    residual = Q_DEFAULT + phi.T @ p @ phi - (phi.T @ p @ gamma) @ gain - p
    # The DARE residual of a converged solution is at rounding level (observed 1e-14).
    assert np.max(np.abs(residual)) <= 1e-10 * np.max(np.abs(p))


def test_dlqr_refuses_an_unstabilizable_pair():
    phi, _ = cw_discrete(N, DT)
    with pytest.raises(ValueError, match="riccati solution did not converge|closed loop is not"):
        lqr.dlqr(phi, np.zeros((6, 3)), Q_DEFAULT, R_DEFAULT)


@pytest.mark.parametrize("dt", [1.0, 10.0, 60.0])
def test_r_bar_hold_needs_three_n_squared_times_the_offset(dt):
    phi, gamma = cw_discrete(N, dt)
    us = lqr.steady_state_input(phi, gamma, R_BAR_HOLD)
    assert us.shape == (3,)
    # 3 n^2 100 m = 3.6749087912e-4 m/s^2 at this n; the discrete target input equals the
    # continuous one because a constant input at an equilibrium is held exactly by the ZOH map.
    assert us[0] == pytest.approx(3.6749087912e-4, rel=1e-10)
    assert us[0] == pytest.approx(3.0 * N**2 * 100.0, rel=1e-10)
    assert np.max(np.abs(us[1:])) < 1e-15


@pytest.mark.parametrize("dt", [1.0, 10.0, 60.0])
def test_v_bar_hold_needs_no_input(dt):
    phi, gamma = cw_discrete(N, dt)
    us = lqr.steady_state_input(phi, gamma, np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0]))
    assert np.max(np.abs(us)) < 1e-15


@pytest.mark.parametrize("dt", [1.0, 10.0, 60.0])
@settings(max_examples=60, deadline=None)
@given(coordinates, coordinates, coordinates)
def test_hold_input_cancels_the_tidal_and_normal_terms(dt, x, y, z):
    phi, gamma = cw_discrete(N, dt)
    us = lqr.steady_state_input(phi, gamma, np.array([x, y, z, 0.0, 0.0, 0.0]))
    # At rest, x'' = 3 n^2 x + a_x and z'' = -n^2 z + a_z vanish for a_x = -3 n^2 x and
    # a_z = n^2 z; y costs nothing. Rounding is measured from a 1 m offset so zeros count too.
    expected = np.array([-3.0 * N**2 * x, 0.0, N**2 * z])
    floor = 3.0 * N**2 * max(abs(x), abs(y), abs(z), 1.0)
    assert np.max(np.abs(us - expected)) <= 1e-10 * floor


def test_hold_point_that_is_not_a_steady_state_is_refused():
    phi, gamma = cw_discrete(N, DT)
    # A hold point moving along-track cannot be held by any constant acceleration.
    with pytest.raises(ValueError, match="not a steady state"):
        lqr.steady_state_input(phi, gamma, np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0]))


def test_regulating_to_the_hold_point_without_feedforward_leaves_an_offset():
    phi, gamma = cw_discrete(N, DT)
    k, _ = lqr.dlqr(phi, gamma, Q_DEFAULT, R_DEFAULT)
    # Fixed point of x+ = phi x - gamma k (x - xs) at these weights only; the offset changes
    # with the weights.
    fixed = np.linalg.solve(np.eye(6) - phi + gamma @ k, gamma @ k @ R_BAR_HOLD)
    assert (fixed - R_BAR_HOLD)[:3] == pytest.approx([-3.702, 0.041, 0.0], abs=1e-3)


def test_feedforward_removes_the_hold_point_offset():
    phi, gamma = cw_discrete(N, DT)
    k, _ = lqr.dlqr(phi, gamma, Q_DEFAULT, R_DEFAULT)
    us = lqr.steady_state_input(phi, gamma, R_BAR_HOLD)
    fixed = np.linalg.solve(np.eye(6) - phi + gamma @ k, gamma @ (us + k @ R_BAR_HOLD))
    # Observed 5e-13 m; the bound leaves three decades above it.
    assert np.max(np.abs(fixed - R_BAR_HOLD)) < 1e-9


@pytest.mark.parametrize(
    "pos_scale, vel_scale, radius_ignored",
    [(1.0, 1.0, 1.3921), (100.0, 0.01, 1.0057), (100.0, 1.0, 0.9012)],
)
def test_one_step_delay_ignored_by_the_design_changes_the_closed_loop_radius(
    pos_scale, vel_scale, radius_ignored
):
    phi, gamma = cw_discrete(N, DT)
    q = np.diag([1 / pos_scale**2] * 3 + [1 / vel_scale**2] * 3)
    k, _ = lqr.dlqr(phi, gamma, q, R_DEFAULT)
    # The command computed from x_k reaches the plant at k + 1.
    loop = np.block([[phi, gamma], [-k, np.zeros((3, 3))]])
    assert spectral_radius(loop) == pytest.approx(radius_ignored, abs=1e-4)


@pytest.mark.parametrize("pos_scale, vel_scale", [(1.0, 1.0), (100.0, 0.01), (100.0, 1.0)])
def test_delay_augmented_design_recovers_the_undelayed_radius(pos_scale, vel_scale):
    phi, gamma = cw_discrete(N, DT)
    q = np.diag([1 / pos_scale**2] * 3 + [1 / vel_scale**2] * 3)
    k, _ = lqr.dlqr(phi, gamma, q, R_DEFAULT)
    phi_a, gamma_a = lqr.delay_augmented(phi, gamma, 1)
    assert phi_a.shape == (9, 9)
    assert gamma_a.shape == (9, 3)
    k_a, _ = lqr.dlqr(phi_a, gamma_a, block_diag(q, np.zeros((3, 3))), R_DEFAULT)
    undelayed = spectral_radius(phi - gamma @ k)
    assert spectral_radius(phi_a - gamma_a @ k_a) == pytest.approx(undelayed, abs=1e-4)


def test_delay_augmented_model_delays_the_input_by_d_steps():
    phi, gamma = cw_discrete(N, DT)
    d = 2
    phi_a, gamma_a = lqr.delay_augmented(phi, gamma, d)
    assert phi_a.shape == (6 + 3 * d, 6 + 3 * d)
    assert gamma_a.shape == (6 + 3 * d, 3)
    rng = np.random.default_rng(1)
    x = rng.normal(size=6) * 100.0
    u = rng.normal(size=(12, 3)) * 1e-3
    z = np.concatenate([x, u[0], u[1]])
    for k in range(2, 12):
        z = phi_a @ z + gamma_a @ u[k]
        x = phi @ x + gamma @ u[k - d]
    # Both paths multiply the same matrices; the augmented one adds zero blocks.
    assert np.allclose(z[:6], x, rtol=1e-12, atol=0.0)
    assert np.array_equal(z[6:9], u[10])
    assert np.array_equal(z[9:], u[11])


def test_delay_augmented_with_no_delay_is_the_plain_model():
    phi, gamma = cw_discrete(N, DT)
    phi_a, gamma_a = lqr.delay_augmented(phi, gamma, 0)
    assert np.array_equal(phi_a, phi)
    assert np.array_equal(gamma_a, gamma)


# Sampled-data cost: Qc = diag(1e-4 x3, 1 x3), Rc = 1e3 I, x0 = (50, -80, 10 m, 0.02, -0.01,
# 0.005 m/s). The optimal cost values come from an independent computation of the DARE cost at
# these constants; the residual of that computation was 3e-14 and it matched the integrated
# continuous cost of the closed loop.
QC_SAMPLED = np.diag([1e-4] * 3 + [1.0] * 3)
RC_SAMPLED = np.eye(3) * 1e3
X0_SAMPLED = np.array([50.0, -80.0, 10.0, 0.02, -0.01, 0.005])


@pytest.mark.parametrize(
    "dt, cost, gain_gap", [(10.0, 116.65796914, 0.0055), (60.0, 122.99249126, 0.066)]
)
def test_sampled_data_cost_gives_the_exact_zoh_optimum(dt, cost, gain_gap):
    a, b = cw_system(N)
    phi, gamma = cw_discrete(N, dt)
    qd, rd, sd = lqr.sampled_data_cost(a, b, QC_SAMPLED, RC_SAMPLED, dt)
    assert qd.shape == (6, 6)
    assert rd.shape == (3, 3)
    assert sd.shape == (6, 3)
    k, p = lqr.dlqr(phi, gamma, qd, rd, sd)
    assert X0_SAMPLED @ p @ X0_SAMPLED == pytest.approx(cost, rel=1e-10)
    # Weighting the sampled state by Q dt and R dt is a different design, by 0.55% of the
    # gain at 10 s and 6.6% at 60 s.
    k_endpoint, _ = lqr.dlqr(phi, gamma, QC_SAMPLED * dt, RC_SAMPLED * dt)
    gap = np.max(np.abs(k - k_endpoint)) / np.max(np.abs(k))
    assert gap == pytest.approx(gain_gap, rel=0.1)


@pytest.mark.parametrize("dt", [1e-6, 1e-3, 10.0, 1e3, 1e4])
def test_van_loan_cost_matches_gauss_legendre_quadrature(dt):
    a, b = cw_system(N)
    qd, rd, sd = lqr.sampled_data_cost(a, b, Q_DEFAULT, R_DEFAULT, dt)
    qq, rq, sq = quadrature_cost(Q_DEFAULT, R_DEFAULT, dt)
    # Two paths to the same integral: the exponential and the quadrature of closed forms.
    # Observed agreement is 2e-12 at the longest step and rounding below it.
    assert np.max(np.abs(qd - qq)) <= 1e-9 * np.max(np.abs(qq))
    assert np.max(np.abs(rd - rq)) <= 1e-9 * np.max(np.abs(rq))
    assert np.max(np.abs(sd - sq)) <= 1e-9 * np.max(np.abs(sq))


@pytest.mark.parametrize("dt", [1e-6, 1e-3, 10.0, 1e3, 1e4])
def test_sampled_cost_block_is_symmetric_and_positive_semidefinite(dt):
    a, b = cw_system(N)
    qd, rd, sd = lqr.sampled_data_cost(a, b, Q_DEFAULT, R_DEFAULT, dt)
    full = np.block([[qd, sd], [sd.T, rd]])
    scale = np.max(np.abs(full))
    assert np.max(np.abs(qd - qd.T)) <= 4 * EPS * scale
    assert np.max(np.abs(rd - rd.T)) <= 4 * EPS * scale
    eigenvalues = np.linalg.eigvalsh(full)
    assert eigenvalues[0] >= -1e-12 * eigenvalues[-1]


@pytest.mark.parametrize("dt", [1e-6, 1e-3, 10.0, 1e3, 1e4])
def test_sampled_data_design_is_stable_at_every_step_length(dt):
    a, b = cw_system(N)
    phi, gamma = cw_discrete(N, dt)
    qd, rd, sd = lqr.sampled_data_cost(a, b, Q_DEFAULT, R_DEFAULT, dt)
    k, _ = lqr.dlqr(phi, gamma, qd, rd, sd)
    assert spectral_radius(phi - gamma @ k) < 1.0


def test_sampled_data_cost_is_the_same_integral_at_every_step_length():
    # The stage cost of two half steps composes into the full step: the integral over
    # [0, 2h] equals the integral over [0, h] plus the one over [h, 2h] mapped through phi(h).
    a, b = cw_system(N)
    h = 30.0
    qd1, rd1, sd1 = lqr.sampled_data_cost(a, b, Q_DEFAULT, R_DEFAULT, h)
    qd2, rd2, sd2 = lqr.sampled_data_cost(a, b, Q_DEFAULT, R_DEFAULT, 2 * h)
    phi, gamma = cw_discrete(N, h)
    # Over [h, 2h] the state is phi x + gamma u, so the second half contributes
    # (phi x + gamma u)' Qd1 (phi x + gamma u) + 2 (phi x + gamma u)' Sd1 u + u' Rd1 u.
    qd_sum = qd1 + phi.T @ qd1 @ phi
    sd_sum = sd1 + phi.T @ qd1 @ gamma + phi.T @ sd1
    rd_sum = rd1 + gamma.T @ qd1 @ gamma + 2 * gamma.T @ sd1 + rd1
    rd_sum = 0.5 * (rd_sum + rd_sum.T)
    assert np.allclose(qd2, qd_sum, rtol=1e-12, atol=1e-12 * np.max(np.abs(qd2)))
    assert np.allclose(sd2, sd_sum, rtol=1e-12, atol=1e-12 * np.max(np.abs(sd2)))
    assert np.allclose(rd2, rd_sum, rtol=1e-12, atol=1e-12 * np.max(np.abs(rd2)))


def test_constant_disturbance_leaves_the_known_steady_offset():
    phi, gamma = cw_discrete(N, DT)
    q = np.diag([1e-2] * 3 + [10.0] * 3)
    r = np.eye(3) * 1e3
    k, _ = lqr.dlqr(phi, gamma, q, r)
    d = np.array([2e-7, 1e-6, -3e-7])  # unmodelled acceleration, m/s^2
    offset = np.linalg.solve(np.eye(6) - phi + gamma @ k, gamma @ d)
    assert offset[:3] == pytest.approx([1.288762e-4, 5.842058e-4, -1.759851e-4], rel=1e-6)


def alfriend_weights(n):
    """Position 1 m, velocity n m/s and acceleration n^2 m/s^2 each cost one unit."""
    return np.diag([1.0] * 3 + [n**-2] * 3), np.eye(3) / n**4


def test_gains_scale_with_mean_motion_when_the_step_scales_with_the_period():
    n_earth = N
    n_kerbin = mean_motion(KERBIN.radius + 150e3, KERBIN.mu)
    dt_earth = DT
    dt_kerbin = dt_earth * n_earth / n_kerbin
    gains = []
    for n, dt in ((n_earth, dt_earth), (n_kerbin, dt_kerbin)):
        phi, gamma = cw_discrete(n, dt)
        q, r = alfriend_weights(n)
        k, _ = lqr.dlqr(phi, gamma, q, r)
        gains.append(k)
    k_earth, k_kerbin = gains
    ratio = n_kerbin / n_earth
    predicted = np.hstack([ratio**2 * k_earth[:, :3], ratio * k_earth[:, 3:]])
    assert np.max(np.abs(k_kerbin - predicted)) <= 1e-9 * np.max(np.abs(k_kerbin))


def test_rate_augmented_model_carries_the_previous_command():
    phi, gamma = cw_discrete(N, DT)
    s = np.zeros((6, 3))
    rd = np.eye(3) * 0.1 / A_MAX**2
    phi_r, gamma_r, q_r, r_r, s_r = lqr.rate_augmented(phi, gamma, Q_DEFAULT, R_DEFAULT, s, rd)
    assert phi_r.shape == (9, 9)
    assert gamma_r.shape == (9, 3)
    assert q_r.shape == (9, 9)
    assert r_r.shape == (3, 3)
    assert s_r.shape == (9, 3)
    rng = np.random.default_rng(2)
    x = rng.normal(size=6) * 100.0
    u_prev, u = rng.normal(size=(2, 3)) * 1e-3
    z = np.concatenate([x, u_prev])
    z_next = phi_r @ z + gamma_r @ u
    assert np.allclose(z_next[:6], phi @ x + gamma @ u, rtol=1e-12, atol=0.0)
    assert np.array_equal(z_next[6:], u)


def test_rate_augmented_cost_adds_the_command_change_penalty():
    phi, gamma = cw_discrete(N, DT)
    rng = np.random.default_rng(4)
    s = rng.normal(size=(6, 3)) * 1e-3
    rd = np.eye(3) * 0.1 / A_MAX**2
    _, _, q_r, r_r, s_r = lqr.rate_augmented(phi, gamma, Q_DEFAULT, R_DEFAULT, s, rd)
    for _ in range(10):
        x = rng.normal(size=6) * 100.0
        u_prev, u = rng.normal(size=(2, 3)) * 1e-3
        z = np.concatenate([x, u_prev])
        augmented = z @ q_r @ z + u @ r_r @ u + 2 * z @ s_r @ u
        du = u - u_prev
        plain = x @ Q_DEFAULT @ x + u @ R_DEFAULT @ u + 2 * x @ s @ u + du @ rd @ du
        # The two quadratic forms sum the same terms in different orders.
        assert augmented == pytest.approx(plain, rel=1e-12)


def test_zero_rate_weight_recovers_the_plain_gain():
    phi, gamma = cw_discrete(N, DT)
    k, _ = lqr.dlqr(phi, gamma, Q_DEFAULT, R_DEFAULT)
    phi_r, gamma_r, q_r, r_r, s_r = lqr.rate_augmented(
        phi, gamma, Q_DEFAULT, R_DEFAULT, np.zeros((6, 3)), np.zeros((3, 3))
    )
    k_r, _ = lqr.dlqr(phi_r, gamma_r, q_r, r_r, s_r)
    # The previous command is then uncoupled from the cost, so its gain is zero and the state
    # gain is the plain one; both DARE solves are at rounding level of the same problem.
    assert np.allclose(k_r[:, :6], k, rtol=1e-9, atol=0.0)
    assert np.max(np.abs(k_r[:, 6:])) <= 1e-9 * np.max(np.abs(k))


def test_negative_delay_is_refused():
    phi, gamma = cw_discrete(N, 10.0)
    with pytest.raises(ValueError, match="delay must not be negative"):
        lqr.delay_augmented(phi, gamma, -1)
