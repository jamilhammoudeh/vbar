import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import solve_ivp
from scipy.linalg import solve_discrete_are
from scipy.stats import norm

from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.reach import LinearLoop, Zonotope, cw_loop, support
from vbar.safety.tightening import (
    BallError,
    BoxError,
    EllipsoidError,
    box_hull,
    chance_margin,
    delay_margin,
    ecbf_bias,
    first_order_bias,
    halfspace_margin,
    keep_out_margin,
    per_draw_sigma,
    scenario_margin,
    scenario_sample_count,
    tube_margin,
)
from vbar.solvers import ActuatorSet

N_500 = mean_motion(EARTH.radius + 500e3)
# Execution error at full thrust a_max = 0.02 m/s^2: 2% scale, 0.5% noise and 1 deg
# misalignment, per axis a_max (0.025 + sqrt(3) sin 1 deg) = 1.1046e-3 m/s^2 (5.5% of a_max).
DA_EXE = 0.02 * (0.025 + math.sqrt(3.0) * math.sin(math.radians(1.0)))


def unit_rows(rng, count, dim):
    d = rng.standard_normal((count, dim))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def random_covariance(rng):
    root = rng.uniform(-1.0, 1.0, (6, 6)) * np.array([10.0] * 3 + [0.01] * 3)[:, None]
    return root @ root.T


def test_box_support_equals_the_largest_vertex_value():
    rng = np.random.default_rng(41)
    for _ in range(100):
        error = BoxError(r_pos=rng.uniform(0.1, 50.0), r_vel=rng.uniform(1e-4, 0.1))
        a = rng.uniform(-2.0, 2.0, 6)
        radii = np.array([error.r_pos] * 3 + [error.r_vel] * 3)
        vertices = np.array(list(itertools.product((-1.0, 1.0), repeat=6))) * radii
        assert halfspace_margin(a, error) == pytest.approx(np.max(vertices @ a), rel=1e-12)


def test_ball_support_is_attained_and_never_exceeded_by_random_directions():
    rng = np.random.default_rng(43)
    for _ in range(100):
        error = BallError(r_pos=rng.uniform(0.1, 50.0), r_vel=rng.uniform(1e-4, 0.1))
        a = rng.uniform(-2.0, 2.0, 6)
        margin = halfspace_margin(a, error)
        best = np.concatenate(
            [
                error.r_pos * a[:3] / np.linalg.norm(a[:3]),
                error.r_vel * a[3:] / np.linalg.norm(a[3:]),
            ]
        )
        assert best @ a == pytest.approx(margin, rel=1e-12)
        samples = np.hstack(
            [error.r_pos * unit_rows(rng, 2000, 3), error.r_vel * unit_rows(rng, 2000, 3)]
        )
        assert np.max(samples @ a) <= margin * (1.0 + 1e-12)
        # 2000 directions per ball come within a few percent of the maximum.
        assert np.max(samples @ a) >= 0.9 * margin


def test_ellipsoid_support_is_attained_and_never_exceeded_on_the_boundary():
    rng = np.random.default_rng(47)
    for _ in range(100):
        error = EllipsoidError(P=rng.uniform(-1.0, 1.0, (6, 6)))
        a = rng.uniform(-2.0, 2.0, 6)
        margin = halfspace_margin(a, error)
        z = error.P.T @ a
        assert (error.P @ (z / np.linalg.norm(z))) @ a == pytest.approx(margin, rel=1e-12)
        boundary = unit_rows(rng, 5000, 6) @ error.P.T
        assert np.max(boundary @ a) <= margin * (1.0 + 1e-12)


def test_gaussian_ellipsoid_has_shape_k_squared_covariance():
    rng = np.random.default_rng(53)
    cov = random_covariance(rng)
    error = EllipsoidError.from_gaussian(cov, 3.0)
    np.testing.assert_allclose(
        error.P @ error.P.T, 9.0 * cov, rtol=0.0, atol=1e-12 * 9.0 * np.max(cov)
    )


@pytest.mark.parametrize("k", [1.0, 2.0, 3.0, 4.0])
def test_gaussian_ellipsoid_exceedance_is_the_six_dof_chi_square_tail(k):
    # The chi-square survival function with 6 degrees of freedom is exp(-x/2)(1 + x/2 + x^2/8).
    x = k * k
    tail = math.exp(-x / 2.0) * (1.0 + x / 2.0 + x * x / 8.0)
    assert EllipsoidError.from_gaussian(np.eye(6), k).exceedance == pytest.approx(tail, rel=1e-12)


def test_gaussian_ellipsoid_exceedance_matches_sampled_draws():
    rng = np.random.default_rng(59)
    cov = random_covariance(rng)
    error = EllipsoidError.from_gaussian(cov, 3.0)
    draws = rng.multivariate_normal(np.zeros(6), cov, size=200_000)
    outside = np.mean(np.sum(np.linalg.solve(error.P, draws.T) ** 2, axis=0) > 1.0)
    # Binomial standard deviation at p = 0.0611 over 2e5 draws is 5.4e-4; allow 5 of them.
    assert outside == pytest.approx(error.exceedance, abs=5 * 5.4e-4)


def test_ellipsoid_support_of_a_gaussian_equals_the_chance_margin_at_phi_k():
    rng = np.random.default_rng(61)
    cov = random_covariance(rng)
    a = rng.uniform(-1.0, 1.0, 6)
    k = 2.5
    expected = chance_margin(a, cov, norm.cdf(k))
    margin = halfspace_margin(a, EllipsoidError.from_gaussian(cov, k))
    assert margin == pytest.approx(expected, rel=1e-10)


def test_chance_margin_at_one_in_a_million_is_4_753_sigma():
    # Boyd and Vandenberghe, sec. 7.4.2: for a standard Gaussian P(a . x >= 4.753) = 1e-6.
    a = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 4.0])
    assert chance_margin(a, np.eye(6), 1.0 - 1e-6) / 5.0 == pytest.approx(4.753, abs=1e-3)


def test_chance_margin_is_exceeded_with_probability_one_minus_eta():
    rng = np.random.default_rng(67)
    cov = random_covariance(rng)
    a = rng.uniform(-1.0, 1.0, 6)
    margin = chance_margin(a, cov, 0.9)
    draws = rng.multivariate_normal(np.zeros(6), cov, size=200_000)
    # Binomial standard deviation at p = 0.1 over 2e5 draws is 6.7e-4; allow 5 of them.
    assert np.mean(draws @ a > margin) == pytest.approx(0.1, abs=5 * 6.7e-4)


def test_chance_margin_refuses_eta_below_one_half_and_at_one():
    a = np.ones(6)
    assert chance_margin(a, np.eye(6), 0.5) == 0.0
    with pytest.raises(ValueError, match="eta"):
        chance_margin(a, np.eye(6), 0.4)
    with pytest.raises(ValueError, match="eta"):
        chance_margin(a, np.eye(6), 0.5 - 1e-12)
    with pytest.raises(ValueError, match="eta"):
        chance_margin(a, np.eye(6), 1.0)


def test_keep_out_margin_of_an_ellipsoid_is_attained_by_the_top_singular_vector():
    rng = np.random.default_rng(71)
    for _ in range(50):
        phi_r = rng.uniform(-1.0, 1.0, (3, 6)) * np.array([1.0] * 3 + [100.0] * 3)
        error = EllipsoidError(P=rng.uniform(-1.0, 1.0, (6, 6)))
        margin = keep_out_margin(phi_r, error)
        top = np.linalg.svd(phi_r @ error.P)[2][0]
        assert np.linalg.norm(phi_r @ error.P @ top) == pytest.approx(margin, rel=1e-12)
        boundary = unit_rows(rng, 5000, 6) @ error.P.T
        assert np.max(np.linalg.norm(boundary @ phi_r.T, axis=1)) <= margin * (1.0 + 1e-12)


def test_keep_out_margin_of_a_box_is_its_largest_vertex_and_bounds_interior_points():
    rng = np.random.default_rng(73)
    phi_r = rng.uniform(-1.0, 1.0, (3, 6)) * np.array([1.0] * 3 + [100.0] * 3)
    error = BoxError(r_pos=2.0, r_vel=0.01)
    margin = keep_out_margin(phi_r, error)
    radii = np.array([2.0] * 3 + [0.01] * 3)
    vertex_ranges = [
        np.linalg.norm(phi_r @ (np.array(signs) * radii))
        for signs in itertools.product((-1.0, 1.0), repeat=6)
    ]
    assert margin == pytest.approx(max(vertex_ranges), rel=1e-12)
    inside = rng.uniform(-1.0, 1.0, (5000, 6)) * radii
    assert np.max(np.linalg.norm(inside @ phi_r.T, axis=1)) <= margin * (1.0 + 1e-12)


def test_keep_out_margin_of_two_balls_bounds_samples_and_is_exact_for_aligned_blocks():
    rng = np.random.default_rng(79)
    error = BallError(r_pos=2.0, r_vel=0.01)
    phi_r = rng.uniform(-1.0, 1.0, (3, 6)) * np.array([1.0] * 3 + [100.0] * 3)
    margin = keep_out_margin(phi_r, error)
    samples = np.hstack([2.0 * unit_rows(rng, 5000, 3), 0.01 * unit_rows(rng, 5000, 3)])
    assert np.max(np.linalg.norm(samples @ phi_r.T, axis=1)) <= margin * (1.0 + 1e-12)
    # With Phi_r = [A, c A] both blocks peak along the same input direction v1 of A.
    A = rng.uniform(-1.0, 1.0, (3, 3))
    aligned = np.hstack([A, 50.0 * A])
    v1 = np.linalg.svd(A)[2][0]
    attained = np.linalg.norm(aligned @ np.concatenate([2.0 * v1, 0.01 * v1]))
    assert keep_out_margin(aligned, error) == pytest.approx(attained, rel=1e-12)


@pytest.mark.parametrize(
    ("p", "da", "depth"),
    [(0.05, 2e-4, 0.08), (0.1, 2e-4, 0.02), (0.05, 2e-5, 0.008)],
)
def test_ecbf_bias_of_a_double_pole_is_da_over_p_squared(p, da, depth):
    assert ecbf_bias(da, (p, p)) == pytest.approx(depth, abs=1e-5)


@pytest.mark.parametrize("poles", [(0.05, 0.05), (0.03, 0.08), (0.02, 0.5)])
def test_ecbf_bias_is_the_steady_state_of_the_barrier_dynamics(poles):
    # h'' + (p1 + p2) h' + p1 p2 h = -da from rest at h = 0; 60 slowest time constants.
    p1, p2 = poles
    da = 2e-4

    def rhs(t, y):
        return [y[1], -(p1 + p2) * y[1] - p1 * p2 * y[0] - da]

    end = solve_ivp(rhs, (0.0, 60.0 / min(poles)), [0.0, 0.0], rtol=1e-11, atol=1e-14).y[0, -1]
    assert end == pytest.approx(-ecbf_bias(da, poles), abs=1e-5, rel=1e-6)


def test_first_order_bias_is_the_steady_state_of_the_barrier_dynamics():
    gamma, da = 0.04, 2e-4
    end = solve_ivp(
        lambda t, y: [-gamma * y[0] - da], (0.0, 60.0 / gamma), [0.0], rtol=1e-11, atol=1e-14
    ).y[0, -1]
    assert end == pytest.approx(-first_order_bias(da, gamma), rel=1e-6)


def test_error_sets_and_biases_refuse_invalid_parameters():
    with pytest.raises(ValueError, match="r_pos"):
        BoxError(r_pos=-1.0, r_vel=0.0)
    with pytest.raises(ValueError, match="r_vel"):
        BallError(r_pos=1.0, r_vel=math.inf)
    with pytest.raises(ValueError, match="6x6"):
        EllipsoidError(P=np.eye(3))
    with pytest.raises(ValueError, match="exceedance"):
        EllipsoidError(P=np.eye(6), exceedance=1.5)
    with pytest.raises(ValueError, match="symmetric"):
        EllipsoidError.from_gaussian(np.triu(np.ones((6, 6))), 3.0)
    with pytest.raises(ValueError, match="semidefinite"):
        EllipsoidError.from_gaussian(-np.eye(6), 3.0)
    with pytest.raises(ValueError, match="k must"):
        EllipsoidError.from_gaussian(np.eye(6), 0.0)
    with pytest.raises(ValueError, match="poles"):
        ecbf_bias(1e-4, (0.05, 0.0))
    with pytest.raises(ValueError, match="da_max"):
        ecbf_bias(-1e-4, (0.05, 0.05))
    with pytest.raises(ValueError, match="gamma"):
        first_order_bias(1e-4, 0.0)
    with pytest.raises(TypeError, match="unknown"):
        halfspace_margin(np.ones(6), "box")


def lqr_gain(phi, gamma, q, r):
    """The discrete LQR gain K of u = K x."""
    p = solve_discrete_are(phi, gamma, q, r)
    return -np.linalg.solve(r + gamma.T @ p @ gamma, gamma.T @ p @ phi)


def support_sum(a_k, w_bar, a, terms):
    """sum_{j < terms} |a . A_K^j| w_bar, the support of the box tube along a."""
    rows, lam = np.empty((terms, len(w_bar))), np.asarray(a, dtype=float)
    for j in range(terms):
        rows[j] = lam
        lam = lam @ a_k
    return float(np.sum(np.abs(rows) @ w_bar))


def k22_loop():
    """The CW LQR loop with Q = diag(1e-2, 10), R = 1e3 I at dt = 10 s, 500 km."""
    phi, gamma = cw_discrete(N_500, 10.0)
    k = lqr_gain(phi, gamma, np.diag([1e-2] * 3 + [10.0] * 3), 1e3 * np.eye(3))
    # The disturbance bound is |Gamma 1| x 2e-4, the image of one corner of the 2e-4 m/s^2 box,
    # not its box hull |Gamma| 1 x 2e-4; the known answers below are computed with this w_bar.
    w_bar = np.abs(gamma @ np.full(3, 2e-4))
    return phi + gamma @ k, w_bar


def test_box_hull_of_a_box_is_the_largest_vertex_image_per_axis():
    _, gamma = cw_discrete(N_500, 10.0)
    w = Zonotope.box(np.zeros(3), np.full(3, DA_EXE))
    vertices = np.array(list(itertools.product((-1.0, 1.0), repeat=3))) * DA_EXE
    h = box_hull(gamma, w)
    assert h.shape == (6,)
    np.testing.assert_allclose(h, np.max(vertices @ gamma.T, axis=0), rtol=1e-12)
    np.testing.assert_allclose(h, np.abs(gamma) @ np.full(3, DA_EXE), rtol=1e-12)


def test_box_hull_of_a_thrust_ball_is_a_max_times_the_row_norms():
    _, gamma = cw_discrete(N_500, 10.0)
    h = box_hull(gamma, ActuatorSet(0.02, "ball"))
    np.testing.assert_allclose(h, 0.02 * np.linalg.norm(gamma, axis=1), rtol=1e-12)


def test_box_hull_along_track_exceeds_the_image_of_one_corner():
    _, gamma = cw_discrete(N_500, 10.0)
    h = box_hull(gamma, Zonotope.box(np.zeros(3), np.full(3, 2e-4)))
    corner = np.abs(gamma @ np.full(3, 2e-4))
    # The Coriolis entries of Gamma add along y and v_y in the hull and cancel at the corner:
    # 1.00734e-2 m against 9.92581e-3 m.
    assert h[1] > corner[1] * 1.01
    assert np.all(h >= corner * (1.0 - 1e-12))


def test_tube_margin_of_the_scalar_loop_is_two():
    # Rawlings, Mayne and Diehl (2nd ed.), sec. 5.3.5, p. 347: A_K = 0.5, W = [-1, 1] gives
    # phi_inf = 1 / (1 - 0.5) = 2, which (1 - alpha)^-1 phi_N reproduces at every N. The printed
    # factor (1 - alpha^-1) gives a negative margin, and phi_N alone misses the tail.
    assert tube_margin(np.array([[0.5]]), np.array([1.0]), np.array([1.0])) == pytest.approx(
        2.0, rel=1e-12
    )


def test_tube_margin_of_the_cw_lqr_loop_bounds_the_infinite_sum():
    a_k, w_bar = k22_loop()
    c = np.eye(6)[0]
    bound = tube_margin(a_k, w_bar, c)
    # First N with max_i (|A_K^N| w_bar)_i / w_bar_i < 0.5 is N = 8, alpha = 0.3752,
    # phi_8 = 0.10660 m, so phi_8 / (1 - alpha) = 0.17061 m; the factor is (1 - alpha)^-1 per
    # the corrected sec. 5.3.5 of Rawlings, Mayne and Diehl.
    assert bound == pytest.approx(0.17061, rel=1e-4)
    phi_8 = support_sum(a_k, w_bar, c, 8)
    assert phi_8 == pytest.approx(0.10660, rel=1e-4)
    # 20000 terms of a loop with spectral radius 0.712 leave no tail.
    phi_inf = support_sum(a_k, w_bar, c, 20000)
    assert phi_inf == pytest.approx(0.11978, rel=1e-4)
    assert phi_8 < phi_inf <= bound


def test_tube_margin_uses_the_first_n_below_alpha_and_its_own_ratio():
    a_k, w_bar = k22_loop()
    c = np.eye(6)[0]
    m8 = np.linalg.matrix_power(a_k, 8)
    alpha_8 = float(np.max((np.abs(m8) @ w_bar) / w_bar))
    assert alpha_8 == pytest.approx(0.3752, abs=1e-4)
    expected = support_sum(a_k, w_bar, c, 8) / (1.0 - alpha_8)
    assert tube_margin(a_k, w_bar, c) == pytest.approx(expected, rel=1e-12)
    assert tube_margin(a_k, w_bar, c, alpha=0.3753) == pytest.approx(expected, rel=1e-12)
    assert tube_margin(a_k, w_bar, c, alpha=0.3751) != pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "alpha, big_n, alpha_n",
    # Rawlings, Mayne and Diehl, Example 3.13, pp. 231 to 233: alpha(N) = 2, 1.4, 0.72, 0.296,
    # 0.0928 for N = 1 to 5, so alpha < 1 first at N = 3.
    [(1.0, 3, 0.72), (0.3, 4, 0.296), (0.1, 5, 0.0928)],
)
def test_tube_margin_reproduces_the_contraction_steps_of_example_3_13(alpha, big_n, alpha_n):
    a_k = np.array([[1.0, 1.0], [0.0, 1.0]]) + np.array([[0.0], [1.0]]) @ np.array([[-0.4, -1.2]])
    w_bar = np.array([0.1, 0.1])
    c = np.array([1.0, 0.0])
    expected = support_sum(a_k, w_bar, c, big_n) / (1.0 - alpha_n)
    assert tube_margin(a_k, w_bar, c, alpha=alpha) == pytest.approx(expected, rel=1e-9)


def test_example_3_25_tube_half_width_is_one_minus_two_to_the_minus_i():
    # Rawlings, Mayne and Diehl, Example 3.25, p. 254: K = -1/2 (printed 1/2), A_K = 1/2,
    # w uniform on [-1/2, 1/2], so S_K(i) has half-width 1 - 2^-i and S_K(inf) half-width 1.
    loop = LinearLoop(np.array([[0.5]]), np.array([[1.0]]), np.zeros(1))
    w = Zonotope.box(np.zeros(1), np.array([0.5]))
    for i in range(6):
        assert support(loop, np.zeros(1), w, np.ones(1), i) == pytest.approx(1 - 2.0**-i, abs=1e-12)
    assert tube_margin(np.array([[0.5]]), np.array([0.5]), np.ones(1)) == pytest.approx(
        1.0, rel=1e-12
    )


def test_tube_margin_bounds_the_lqr_tubes_of_execution_and_navigation_error():
    a_max = 0.02
    phi, gamma = cw_discrete(N_500, 10.0)
    k = lqr_gain(phi, gamma, np.diag([1 / 10.0**2] * 3 + [1 / 0.1**2] * 3), np.eye(3) / a_max**2)
    a_k = phi + gamma @ k
    exe_loop = LinearLoop(a_k, gamma, np.zeros(6))
    exe_box = Zonotope.box(np.zeros(3), np.full(3, DA_EXE))
    nav_loop = cw_loop(N_500, 10.0, gain=-k)
    # Navigation error treated as a 3-sigma box of 3 m and 3 mm/s.
    nav_box = Zonotope.box(np.zeros(6), np.array([3.0] * 3 + [3e-3] * 3))
    exe_hull = box_hull(gamma, exe_box)
    nav_hull = box_hull(nav_loop.B, nav_box)
    exe_tube, nav_tube, exe_bound, nav_bound = [], [], [], []
    for e in np.eye(6):
        # Spectral radius 0.905: 400 terms leave a tail below 1e-16 of the sum.
        exe_tube.append(support(exe_loop, np.zeros(6), exe_box, e, 400))
        nav_tube.append(support(nav_loop, np.zeros(6), nav_box, e, 400))
        exe_bound.append(tube_margin(a_k, exe_hull, e))
        nav_bound.append(tube_margin(a_k, nav_hull, e))
    # Largest per-axis tubes of this loop: 1.42 m and 0.024 m/s under execution error,
    # 3.35 m and 0.057 m/s under the navigation box.
    assert max(exe_tube[:3]) == pytest.approx(1.42, abs=0.005)
    assert max(exe_tube[3:]) == pytest.approx(0.024, abs=0.0005)
    assert max(nav_tube[:3]) == pytest.approx(3.35, abs=0.015)
    assert max(nav_tube[3:]) == pytest.approx(0.057, abs=0.0005)
    assert np.all(np.array(exe_bound) >= np.array(exe_tube))
    assert np.all(np.array(nav_bound) >= np.array(nav_tube))


@settings(max_examples=25, deadline=None)
@given(
    dim=st.integers(2, 4),
    entries=st.lists(st.floats(-1.0, 1.0), min_size=16, max_size=16),
    radius=st.floats(0.1, 0.9),
    widths=st.lists(st.floats(1e-3, 1.0), min_size=4, max_size=4),
    direction=st.lists(st.floats(-1.0, 1.0), min_size=4, max_size=4),
)
def test_tube_margin_bounds_every_finite_reach_support(dim, entries, radius, widths, direction):
    a_k = np.array(entries[: dim * dim]).reshape(dim, dim)
    rho = float(np.max(np.abs(np.linalg.eigvals(a_k))))
    if rho > 1e-6:
        a_k *= radius / rho
    w_bar = np.array(widths[:dim])
    a = np.array(direction[:dim])
    loop = LinearLoop(a_k, np.eye(dim), np.zeros(dim))
    finite = support(loop, np.zeros(dim), Zonotope.box(np.zeros(dim), w_bar), a, 150)
    assert finite <= tube_margin(a_k, w_bar, a) * (1.0 + 1e-9) + 1e-12


def test_tube_margin_refuses_unstable_loops_zero_widths_and_bad_alpha():
    with pytest.raises(ValueError):
        tube_margin(np.diag([1.0, 0.5]), np.array([0.1, 0.1]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        tube_margin(np.diag([1.2, 0.3]), np.array([0.1, 0.1]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="w_bar"):
        tube_margin(np.diag([0.5, 0.5]), np.array([0.1, 0.0]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        tube_margin(np.diag([0.5, 0.5]), np.array([0.1, 0.1]), np.array([1.0, 0.0]), alpha=0.0)
    with pytest.raises(ValueError):
        tube_margin(np.diag([0.5, 0.5]), np.array([0.1, 0.1]), np.array([1.0, 0.0]), alpha=1.5)


def test_tube_margin_at_alpha_one_waits_for_a_strict_contraction():
    # A_K = [[0, 1], [0, 0]] maps the unit box onto the segment |x_1| <= 1, x_2 = 0, so
    # alpha_1 = 1 is not below 1; A_K^2 = 0 gives alpha_2 = 0 and phi_2 = 1 + 1 = 2.
    a_k = np.array([[0.0, 1.0], [0.0, 0.0]])
    margin = tube_margin(a_k, np.array([1.0, 1.0]), np.array([1.0, 0.0]), alpha=1.0)
    assert margin == pytest.approx(2.0, rel=1e-12)


def test_delay_margin_of_one_step_is_the_execution_error_through_gamma():
    _, gamma = cw_discrete(N_500, 10.0)
    w = Zonotope.box(np.zeros(3), np.full(3, DA_EXE))
    # Cross-track the step is uncoupled: dt^2 / 2 x 1.1046e-3 = 0.0552 m and
    # dt x 1.1046e-3 = 0.01105 m/s.
    assert delay_margin(N_500, 10.0, 10.0, w, np.eye(6)[2]) == pytest.approx(0.0552, rel=0.01)
    assert delay_margin(N_500, 10.0, 10.0, w, np.eye(6)[5]) == pytest.approx(0.01105, rel=0.01)
    hull = box_hull(gamma, w)
    for i, e in enumerate(np.eye(6)):
        assert delay_margin(N_500, 10.0, 10.0, w, e) == pytest.approx(hull[i], rel=1e-12)


def test_delay_margin_is_the_open_loop_reach_support():
    rng = np.random.default_rng(47)
    w = Zonotope.box(np.zeros(3), np.full(3, DA_EXE))
    loop = cw_loop(N_500, 10.0)
    for _ in range(20):
        a = rng.uniform(-1.0, 1.0, 6)
        expected = support(loop, np.zeros(6), w, a, 3)
        assert delay_margin(N_500, 10.0, 30.0, w, a) == pytest.approx(expected, rel=1e-12)
    assert delay_margin(N_500, 10.0, 0.0, w, np.ones(6)) == 0.0


def test_delay_margin_grows_without_bound_along_track_with_no_feedback():
    w = Zonotope.box(np.zeros(3), np.full(3, DA_EXE))
    e_y = np.eye(6)[1]
    # Open-loop along-track half-widths of the 5.5% execution error: 719 m after 100 steps and
    # 2.47e4 m after 360 steps (one hour at dt = 10 s); a feedback gain would keep them near 1.4 m.
    assert delay_margin(N_500, 10.0, 1000.0, w, e_y) == pytest.approx(719.0, abs=0.5)
    assert delay_margin(N_500, 10.0, 3600.0, w, e_y) == pytest.approx(2.47e4, abs=50.0)


def test_delay_margin_refuses_a_negative_delay():
    w = Zonotope.box(np.zeros(3), np.full(3, DA_EXE))
    with pytest.raises(ValueError):
        delay_margin(N_500, 10.0, -10.0, w, np.ones(6))


def test_delay_margin_takes_only_whole_steps_rounded_to_the_nearest():
    w = Zonotope.box(np.zeros(3), np.full(3, DA_EXE))
    with pytest.raises(ValueError, match="whole number"):
        delay_margin(N_500, 10.0, 15.0, w, np.ones(6))
    # 0.3 / 0.1 is 2.9999999999999996 in floating point: three steps, not two.
    expected = support(cw_loop(N_500, 0.1), np.zeros(6), w, np.ones(6), 3)
    assert delay_margin(N_500, 0.1, 0.3, w, np.ones(6)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("draws, sigma", [(1080, 4.91), (2160, 5.04)])
def test_per_draw_sigma_splits_eps_over_both_tails_of_every_draw(draws, sigma):
    # Phi^-1(1 - 1e-3 / (2 draws)) = 4.9068 at 1080 draws and 5.0411 at 2160; without the union
    # over draws it would be Phi^-1(1 - 5e-4) = 3.29.
    assert per_draw_sigma(1e-3, draws) == pytest.approx(sigma, abs=0.005)
    assert per_draw_sigma(1e-3, draws) == pytest.approx(norm.isf(1e-3 / (2 * draws)), rel=1e-12)


def test_per_draw_sigma_refuses_invalid_eps_and_draws():
    with pytest.raises(ValueError):
        per_draw_sigma(0.0, 1080)
    with pytest.raises(ValueError):
        per_draw_sigma(1.5, 1080)
    with pytest.raises(ValueError):
        per_draw_sigma(1e-3, 0)
    with pytest.raises(TypeError):
        per_draw_sigma(1e-3, True)


@pytest.mark.parametrize("eps, count", [(0.1, 113), (0.01, 1122), (0.001, 11211)])
def test_scenario_sample_count_is_the_ceiling_of_eq_3_25(eps, count):
    # Rawlings, Mayne and Diehl, eq. (3.25): M* = (2 / eps)(ln(1 / beta) + n_theta) = 112.10,
    # 1121.03 and 11210.3 at beta = 0.01 and n_theta = 1.
    m = scenario_sample_count(eps=eps, beta=0.01, n_theta=1)
    assert m == count
    assert isinstance(m, int)


def test_scenario_sample_count_refuses_invalid_eps_and_beta():
    with pytest.raises(ValueError):
        scenario_sample_count(eps=0.0, beta=0.01, n_theta=1)
    with pytest.raises(ValueError):
        scenario_sample_count(eps=0.1, beta=0.0, n_theta=1)
    with pytest.raises(TypeError):
        scenario_sample_count(eps=0.1, beta=0.01, n_theta=True)


@pytest.mark.parametrize("eps, count", [(0.1, 133), (0.01, 1322)])
def test_scenario_sample_count_grows_with_the_decision_variables(eps, count):
    # Eq. (3.25) with n_theta = 2 and beta = 0.01: (2 / eps)(ln 100 + 2) = 132.10 and 1321.03.
    assert scenario_sample_count(eps=eps, beta=0.01, n_theta=2) == count


def test_scenario_margin_has_one_decision_variable():
    # The margin program min theta s.t. a . e(i) <= theta has n_theta = 1, so 113 draws suffice
    # at eps = 0.1, beta = 0.01, fewer than the 133 that n_theta = 2 would need.
    rng = np.random.default_rng(61)
    samples = rng.standard_normal((113, 6))
    a = np.ones(6)
    assert scenario_margin(a, samples, eps=0.1, beta=0.01) == pytest.approx(
        float(np.max(samples @ a)), rel=1e-12
    )


def test_scenario_functions_take_eps_and_beta_by_keyword_only():
    with pytest.raises(TypeError):
        scenario_sample_count(0.1, 0.01)
    with pytest.raises(TypeError):
        scenario_margin(np.ones(6), np.zeros((113, 6)), 0.1, 0.01)


def test_scenario_margin_is_the_largest_draw_and_needs_m_star_draws():
    rng = np.random.default_rng(53)
    a = rng.uniform(-1.0, 1.0, 6)
    samples = rng.standard_normal((113, 6))
    margin = scenario_margin(a, samples, eps=0.1, beta=0.01)
    assert margin == pytest.approx(float(np.max(samples @ a)), rel=1e-12)
    with pytest.raises(ValueError):
        scenario_margin(a, samples[:112], eps=0.1, beta=0.01)


def test_scenario_margin_is_exceeded_less_often_than_eps():
    rng = np.random.default_rng(59)
    root = rng.uniform(-1.0, 1.0, (6, 6))
    a = rng.uniform(-1.0, 1.0, 6)
    margin = scenario_margin(a, rng.standard_normal((1122, 6)) @ root.T, eps=0.01, beta=0.01)
    fresh = rng.standard_normal((200000, 6)) @ root.T
    # The maximum of 1122 draws is exceeded with probability 1 / 1123 on average.
    assert np.mean(fresh @ a > margin) < 0.01
