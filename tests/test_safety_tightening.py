import itertools
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.stats import norm

from vbar.safety.tightening import (
    BallError,
    BoxError,
    EllipsoidError,
    chance_margin,
    ecbf_bias,
    first_order_bias,
    halfspace_margin,
    keep_out_margin,
)


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
