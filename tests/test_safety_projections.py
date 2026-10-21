import numpy as np
import pytest
from scipy.optimize import minimize

from vbar.safety.projections import (
    ball,
    box,
    halfspace,
    keep_out_projection,
    second_order_cone,
)


def test_halfspace_projection_moves_along_the_normal_onto_the_plane():
    # Boyd and Vandenberghe, Convex Optimization, sec. 8.1.1: u0 + (b - a . u0) a / |a|^2.
    np.testing.assert_array_equal(halfspace([0.0, 0.0], [1.0, 1.0], 2.0), [1.0, 1.0])
    np.testing.assert_array_equal(
        halfspace([1.0, -2.0, 0.0], [0.0, 2.0, 0.0], 2.0), [1.0, 1.0, 0.0]
    )


def test_halfspace_projection_leaves_points_inside_unchanged():
    np.testing.assert_array_equal(halfspace([3.0, 0.0], [1.0, 1.0], 2.0), [3.0, 0.0])
    np.testing.assert_array_equal(halfspace([1.0, 1.0], [1.0, 1.0], 2.0), [1.0, 1.0])


def test_halfspace_refuses_a_zero_normal():
    with pytest.raises(ValueError, match="nonzero"):
        halfspace([1.0, 0.0], [0.0, 0.0], 1.0)


def test_box_clips_each_axis_and_ball_scales_radially():
    np.testing.assert_array_equal(box([3.0, -0.5, -4.0], 1.0), [1.0, -0.5, -1.0])
    np.testing.assert_allclose(ball([3.0, 4.0, 0.0], 1.0), [0.6, 0.8, 0.0], rtol=1e-15)
    np.testing.assert_array_equal(ball([0.3, 0.4, 0.0], 1.0), [0.3, 0.4, 0.0])
    with pytest.raises(ValueError, match="a_max"):
        box([1.0], 0.0)
    with pytest.raises(ValueError, match="a_max"):
        ball([1.0], -1.0)


def test_cone_projection_maps_the_polar_region_to_the_apex():
    # Boyd and Vandenberghe, exercise 8.3(c), first regime: |x| <= -t.
    x, t = second_order_cone([3.0, 4.0], -6.0)
    np.testing.assert_array_equal(x, [0.0, 0.0])
    assert t == 0.0
    x, t = second_order_cone([3.0, 4.0], -5.0)
    np.testing.assert_array_equal(x, [0.0, 0.0])
    assert t == 0.0


def test_cone_projection_keeps_points_of_the_cone():
    x, t = second_order_cone([3.0, 4.0], 6.0)
    np.testing.assert_array_equal(x, [3.0, 4.0])
    assert t == 6.0


def test_cone_projection_third_regime_is_on_the_boundary_with_orthogonal_residual():
    # (1/2)(1 + t/|x|)(x, |x|): t = 0 gives (1.5, 2, 2.5), t = -1 gives (1.2, 1.6, 2).
    x, t = second_order_cone([3.0, 4.0], 0.0)
    np.testing.assert_allclose(x, [1.5, 2.0], rtol=1e-15)
    assert t == pytest.approx(2.5, rel=1e-15)
    x, t = second_order_cone([3.0, 4.0], -1.0)
    np.testing.assert_allclose(x, [1.2, 1.6], rtol=1e-15)
    assert t == pytest.approx(2.0, rel=1e-15)
    rng = np.random.default_rng(83)
    for _ in range(200):
        x0 = rng.uniform(-5.0, 5.0, 3)
        t0 = rng.uniform(-0.9, 0.9) * np.linalg.norm(x0)
        x, t = second_order_cone(x0, t0)
        assert t == pytest.approx(np.linalg.norm(x), rel=1e-14)
        residual = np.append(x0 - x, t0 - t)
        assert abs(residual @ np.append(x, t)) <= 1e-13 * np.linalg.norm(x0) ** 2


def test_negative_cone_is_not_mistaken_for_the_cone():
    # t^2 >= |x|^2 also holds on the negative cone; only t >= 0 separates them.
    x, t = second_order_cone([0.3, 0.4], -1.0)
    np.testing.assert_array_equal(x, [0.0, 0.0])
    assert t == 0.0


def sphere_search(G, h, u0, radius):
    """Smallest |u - u0|^2 with |G u + h| >= R, over the sphere of step directions from u0.

    Along a unit direction d the first boundary crossing solves |w0 + t G d| = R, a quadratic
    in t with one positive root since u0 is inside. A Fibonacci grid of 4000 directions picks
    two starts, each refined by the simplex method.
    """
    w0 = G @ u0 + h

    def cost(d):
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        g = d @ G.T
        a = np.sum(g * g, axis=-1)
        b = g @ w0
        c = w0 @ w0 - radius**2
        return ((-b + np.sqrt(b * b - a * c)) / a) ** 2

    i = np.arange(4000) + 0.5
    z = 1.0 - 2.0 * i / 4000
    phi = np.pi * (1.0 + 5.0**0.5) * i
    grid = np.stack([np.sqrt(1.0 - z * z) * np.cos(phi), np.sqrt(1.0 - z * z) * np.sin(phi), z], -1)
    best = np.inf
    for k in np.argsort(cost(grid))[:2]:
        refined = minimize(
            lambda d: float(cost(d)),
            grid[k],
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-14},
        )
        best = min(best, refined.fun)
    return best


def random_keep_out_cases(count):
    # Draw order of the reference check: G, h, R, then u0 deep inside the keep-out.
    rng = np.random.default_rng(3)
    for _ in range(count):
        G = rng.standard_normal((3, 3))
        h = rng.standard_normal(3)
        radius = 2.0 + rng.random()
        u0 = np.linalg.solve(G, 0.3 * rng.standard_normal(3) - h)
        yield G, h, radius, u0


def test_keep_out_projection_matches_a_global_search_over_directions():
    worst = 0.0
    for G, h, radius, u0 in random_keep_out_cases(200):
        projection = keep_out_projection(u0, G, h, radius)
        assert not projection.hard_case
        cost = float(np.sum((projection.u - u0) ** 2))
        brute = sphere_search(G, h, u0, radius)
        worst = max(worst, abs(cost - brute) / brute)
        assert np.linalg.norm(G @ projection.u + h) == pytest.approx(radius, rel=1e-12)
    # Measured agreement is 5e-14; 1e-9 leaves room for the simplex stopping tolerance.
    assert worst <= 1e-9


def test_keep_out_multiplier_satisfies_the_stationarity_condition():
    for G, h, radius, u0 in random_keep_out_cases(200):
        projection = keep_out_projection(u0, G, h, radius)
        lam = projection.lam
        smax = np.linalg.svd(G, compute_uv=False)[0]
        assert 0.0 < lam < 1.0 / smax**2
        matrix = np.eye(3) - lam * G.T @ G
        u = np.linalg.solve(matrix, u0 + lam * G.T @ h)
        # The solve loses accuracy in proportion to the condition number of I - lam G^T G.
        tol = 1e-12 * np.linalg.cond(matrix) * np.linalg.norm(u)
        assert np.linalg.norm(u - projection.u) <= tol


def test_keep_out_projection_is_never_worse_than_the_tangent_plane_projection():
    # |G u + h| is convex, so its tangent halfspace at u0 lies inside the feasible region.
    for G, h, radius, u0 in random_keep_out_cases(200):
        w0 = G @ u0 + h
        normal = G.T @ w0 / np.linalg.norm(w0)
        tangent = halfspace(u0, normal, radius - np.linalg.norm(w0) + normal @ u0)
        exact = keep_out_projection(u0, G, h, radius).u
        assert np.sum((exact - u0) ** 2) <= np.sum((tangent - u0) ** 2) * (1.0 + 1e-12)


def test_keep_out_projection_returns_an_outside_command_unchanged():
    G = np.diag([2.0, 1.0, 0.5])
    u0 = np.array([3.0, 0.0, 0.0])
    projection = keep_out_projection(u0, G, np.zeros(3), 2.0)
    np.testing.assert_array_equal(projection.u, u0)
    assert projection.lam == 0.0
    assert not projection.hard_case


def hard_case_problem(rotate):
    G = np.diag([2.0, 1.0, 0.5])
    u0 = np.array([0.0, 0.3, 0.2])
    if rotate:
        rng = np.random.default_rng(11)
        q1, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        q2, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        G = q1 @ G @ q2.T
        u0 = q2 @ u0
    return G, np.zeros(3), 2.0, u0


@pytest.mark.parametrize("rotate", [False, True])
def test_hard_case_is_closed_with_a_top_singular_component(rotate):
    # G u0 + h = (0, 0.3, 0.1) has no component on the top singular direction, and at
    # lam = 1/4 the other two reach only |(0.4, 0.1067)| = 0.414 m < R.
    G, h, radius, u0 = hard_case_problem(rotate)
    projection = keep_out_projection(u0, G, h, radius)
    assert projection.hard_case
    assert projection.lam == pytest.approx(0.25, rel=1e-15)
    assert np.linalg.norm(G @ projection.u + h) >= radius * (1.0 - 1e-12)
    cost = float(np.sum((projection.u - u0) ** 2))
    assert cost == pytest.approx(sphere_search(G, h, u0, radius), rel=1e-9)


def test_near_hard_case_is_continuous_with_the_hard_case():
    G, h, radius, u0 = hard_case_problem(False)
    hard = keep_out_projection(u0, G, h, radius)
    near = keep_out_projection(u0 + np.array([1e-8, 0.0, 0.0]), G, h, radius)
    assert not near.hard_case
    # The hard case is optimal with either sign of the top component; compare magnitudes.
    np.testing.assert_allclose(np.abs(near.u), np.abs(hard.u), rtol=0.0, atol=1e-7)
    assert near.lam == pytest.approx(hard.lam, rel=1e-7)


def test_keep_out_projection_with_more_commands_than_axes_moves_only_in_the_row_space():
    rng = np.random.default_rng(29)
    G = rng.standard_normal((2, 4))
    h = rng.standard_normal(2)
    u0 = np.linalg.lstsq(G, 0.2 * rng.standard_normal(2) - h, rcond=None)[0]
    projection = keep_out_projection(u0, G, h, 1.5)
    assert np.linalg.norm(G @ projection.u + h) == pytest.approx(1.5, rel=1e-12)
    null = np.linalg.svd(G)[2][2:]
    np.testing.assert_allclose(null @ (projection.u - u0), 0.0, atol=1e-14)


def test_keep_out_projection_refuses_a_zero_map_and_bad_inputs():
    with pytest.raises(ValueError, match="cannot be left"):
        keep_out_projection(np.zeros(3), np.zeros((3, 3)), np.zeros(3), 1.0)
    with pytest.raises(ValueError, match="radius"):
        keep_out_projection(np.zeros(3), np.eye(3), np.zeros(3), 0.0)
    with pytest.raises(ValueError, match="expected"):
        keep_out_projection(np.zeros(2), np.eye(3), np.zeros(3), 1.0)
    with pytest.raises(ValueError, match="finite"):
        keep_out_projection(np.zeros(3), np.eye(3), np.array([np.nan, 0.0, 0.0]), 1.0)
