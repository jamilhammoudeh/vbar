import itertools
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.linalg import solve_discrete_are
from scipy.optimize import linprog, lsq_linear

from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.reach import (
    LinearLoop,
    Polytope,
    Zonotope,
    _certificate,
    ball_distance,
    contains,
    cw_loop,
    drift_clearance,
    gravity_remainder_bound,
    invariant_step,
    loop_clearance,
    min_time_steps,
    propagate,
    support,
)
from vbar.solvers import ActuatorSet

N = mean_motion(EARTH.radius + 500e3)
STEP = 60.0
PHI, GAMMA = cw_discrete(N, STEP)
HOLD = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
FAR = np.array([0.0, -1000.0, 0.0, 0.0, 0.0, 0.0])
TWO_ORBITS = 4.0 * math.pi / N

finite = {"allow_nan": False, "allow_infinity": False, "allow_subnormal": False}


def lqr_gain():
    q = np.diag([1.0, 1.0, 1.0, 1e4, 1e4, 1e4])
    r = 1e8 * np.eye(3)
    p = solve_discrete_are(PHI, GAMMA, q, r)
    return np.linalg.solve(r + GAMMA.T @ p @ GAMMA, GAMMA.T @ p @ PHI)


K = lqr_gain()


def extent(z, d):
    """Scale of the terms of a support value, for relative comparisons near zero."""
    return abs(float(d @ z.center)) + float(np.sum(np.abs(z.generators.T @ d)))


def box_polytope(center, half_widths):
    dim = len(center)
    h = np.vstack([np.eye(dim), -np.eye(dim)])
    return Polytope(h, np.r_[center + half_widths, half_widths - center])


def triangle(vertices):
    v = np.asarray(vertices, dtype=float)
    rows, rhs = [], []
    for i in range(3):
        a, b, other = v[i], v[(i + 1) % 3], v[(i + 2) % 3]
        normal = np.array([b[1] - a[1], a[0] - b[0]])
        if normal @ (other - a) > 0.0:
            normal = -normal
        rows.append(normal)
        rhs.append(normal @ a)
    return Polytope(np.array(rows), np.array(rhs))


def unit_directions(dim, count, seed):
    d = np.random.default_rng(seed).normal(size=(count, dim))
    return np.vstack([np.eye(dim), -np.eye(dim), d / np.linalg.norm(d, axis=1)[:, None]])


def corners(z):
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=z.generators.shape[1])))
    return z.center + signs @ z.generators.T


def in_zonotope(z, x):
    m = z.generators.shape[1]
    lp = linprog(
        np.zeros(m), A_eq=z.generators, b_eq=x - z.center, bounds=[(-1.0, 1.0)] * m, method="highs"
    )
    return lp.status == 0


def test_zonotope_support_matches_its_four_vertices():
    # Kochenderfer, Katz, Corso and Moss, Algorithms for Validation, exercise 8.6: center (1, 2),
    # generators (0, 1) and (1, 1) have vertices (0, 0), (0, 2), (2, 2), (2, 4).
    z = Zonotope([1.0, 2.0], [[0.0, 1.0], [1.0, 1.0]])
    vertices = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 4.0]])
    for d in itertools.product(range(-3, 4), repeat=2):
        assert z.support(np.array(d, dtype=float)) == np.max(vertices @ d)


def test_linear_image_of_a_triangle_is_the_hull_of_its_vertex_images():
    # Algorithms for Validation, exercise 8.3: A = [[1, 1], [-1, -1]] maps (-1, -1), (2, 1),
    # (3, 5) to (-2, 2), (3, -3), (8, -8).
    loop = LinearLoop(np.array([[1.0, 1.0], [-1.0, -1.0]]), np.zeros((2, 1)), np.zeros(2))
    start = triangle([[-1.0, -1.0], [2.0, 1.0], [3.0, 5.0]])
    images = np.array([[-2.0, 2.0], [3.0, -3.0], [8.0, -8.0]])
    for d in unit_directions(2, 12, seed=1):
        # A linear program's optimum, accurate to its feasibility tolerance on unit-sized data.
        assert support(loop, start, None, d, 1) == pytest.approx(np.max(images @ d), abs=1e-9)


def test_polytope_support_of_triangle_along_book_direction():
    # Algorithms for Validation, exercise 8.8: the support vector of conv{(1, 1), (2, 3), (3, 2)}
    # along (-4, 1) is (1, 1), so the support value is -3.
    p = triangle([[1.0, 1.0], [2.0, 3.0], [3.0, 2.0]])
    assert p.support([-4.0, 1.0]) == pytest.approx(-3.0, abs=1e-9)


@pytest.mark.parametrize(
    ("noise", "horizon", "expected"),
    [(0.1, 20, 0.2), (1.0, 20, 0.2), (2.5, 20, 0.3504), (2.5, 40, 0.3947)],
)
def test_mass_spring_damper_reaches_avoid_set_only_at_largest_noise(noise, horizon, expected):
    # Algorithms for Validation, examples 8.3 to 8.5 and appendix A.4: m 1, k 10, c 2, dt 0.05,
    # u = -v observed, p0 in [-0.2, 0.2], v0 = 0, observation noise box +-noise on both axes.
    # R_1 is the start set, so R_1:h is steps 0 to h - 1; the values are recomputed, not printed
    # in the book (figure 8.4 shows them qualitatively).
    dt = 0.05
    ts = np.array([[1.0, dt], [-10.0 * dt, 1.0 - 2.0 * dt]])
    ta_alpha = np.array([[0.0, 0.0], [0.0, -dt]])
    loop = LinearLoop(ts + ta_alpha, ta_alpha, np.zeros(2))
    start = Zonotope.box([0.0, 0.0], [0.2, 0.0])
    w = Zonotope.box([0.0, 0.0], [noise, noise])
    reach = max(
        support(loop, start, w, sign * np.array([1.0, 0.0]), k)
        for k in range(horizon)
        for sign in (1.0, -1.0)
    )
    assert reach == pytest.approx(expected, abs=1e-4)
    assert (reach >= 0.3) == (noise == 2.5)


def lp_support_reference(d, center, half, a_max, steps):
    phi_n = np.linalg.matrix_power(PHI, steps)
    blocks = [np.linalg.matrix_power(PHI, steps - 1 - k) @ GAMMA for k in range(steps)]
    cost = -np.r_[d @ phi_n, np.concatenate([d @ b for b in blocks])]
    bounds = [(c - h, c + h) for c, h in zip(center, half, strict=True)]
    lp = linprog(cost, bounds=bounds + [(-a_max, a_max)] * (3 * steps), method="highs")
    return -lp.fun


def test_cw_support_closed_form_equals_the_linear_program():
    # Algorithms for Validation, eq. 8.20 to 8.27, on CW at 500 km with 60 s steps, 10 steps,
    # a 1e-3 m/s^2 per-axis box and a start box of 10 m and 0.01 m/s around (0, -1000, 0) m.
    loop = cw_loop(N, STEP)
    half = np.array([10.0, 10.0, 10.0, 0.01, 0.01, 0.01])
    a_max = 1e-3
    start, start_h = Zonotope.box(FAR, half), box_polytope(FAR, half)
    thrust_h = box_polytope(np.zeros(3), np.full(3, a_max))
    directions = np.vstack([np.eye(6), unit_directions(6, 3, seed=3)[12:]])
    values = []
    for d in directions:
        reference = lp_support_reference(d, FAR, half, a_max, 10)
        closed = support(loop, start, ActuatorSet(a_max, "box"), d, 10)
        closed_zonotope = support(loop, start, Zonotope.box(np.zeros(3), np.full(3, a_max)), d, 10)
        lp_path = support(loop, start_h, thrust_h, d, 10)
        assert closed == pytest.approx(reference, abs=1e-6)
        assert closed_zonotope == pytest.approx(closed, rel=1e-12, abs=1e-12)
        assert lp_path == pytest.approx(reference, abs=1e-6)
        values.append(closed)
    assert values[:3] == pytest.approx([277.215353723, -747.144546626, 186.925173596], abs=1e-6)


def test_ball_thrust_support_is_a_max_times_the_euclidean_norm():
    loop = cw_loop(N, STEP)
    d = unit_directions(6, 1, seed=5)[-1]
    lam = d.copy()
    expected = 0.0
    for _ in range(4):
        expected += 1e-3 * np.linalg.norm(GAMMA.T @ lam)
        lam = PHI.T @ lam
    expected += lam @ HOLD
    assert support(loop, HOLD, ActuatorSet(1e-3, "ball"), d, 4) == pytest.approx(
        expected, rel=1e-12
    )


def closed_loop_trajectory(x0, errors):
    """Hold-point LQR flown step by step on the exact CW step, navigation error on each estimate."""
    states = [x0]
    x = x0
    for e in errors:
        u = -K @ (x + e - HOLD)
        x = PHI @ x + GAMMA @ u
        states.append(x)
    return np.array(states)


START_HALF = np.array([5.0, 5.0, 5.0, 5e-3, 5e-3, 5e-3])
NAV_HALF = np.array([0.5, 0.5, 0.5, 5e-4, 5e-4, 5e-4])


def test_propagated_sets_contain_every_sampled_closed_loop_trajectory():
    steps = 12
    sets = propagate(
        cw_loop(N, STEP, gain=K, hold=HOLD),
        Zonotope.box(HOLD, START_HALF),
        Zonotope.box(np.zeros(6), NAV_HALF),
        steps,
    )
    assert len(sets) == steps + 1
    rng = np.random.default_rng(11)
    directions = unit_directions(6, 30, seed=12)
    # Rounding of the two computation paths, far below the metre-scale sets.
    limits = np.array([[z.support(d) + 1e-12 * extent(z, d) for d in directions] for z in sets])
    for _ in range(200):
        x0 = HOLD + START_HALF * rng.choice([-1.0, 1.0], size=6)
        errors = NAV_HALF * rng.uniform(-1.0, 1.0, size=(steps, 6))
        errors[rng.random(steps) < 0.5] = NAV_HALF * rng.choice([-1.0, 1.0], size=6)
        assert np.all(closed_loop_trajectory(x0, errors) @ directions.T <= limits)


def test_propagated_support_is_attained_by_the_extreme_trajectory():
    # The support of R_N along d is d . x_N for the start vertex and the error vertices that
    # the backward direction picks; the trajectory is flown independently of the set code.
    steps = 12
    loop = cw_loop(N, STEP, gain=K, hold=HOLD)
    start, nav = Zonotope.box(HOLD, START_HALF), Zonotope.box(np.zeros(6), NAV_HALF)
    sets = propagate(loop, start, nav, steps)
    a_cl = PHI - GAMMA @ K
    b_cl = -GAMMA @ K
    for d in unit_directions(6, 10, seed=13):
        lam = [d]
        for _ in range(steps):
            lam.append(a_cl.T @ lam[-1])
        x0 = HOLD + START_HALF * np.sign(lam[steps])
        errors = np.array([NAV_HALF * np.sign(b_cl.T @ lam[steps - 1 - k]) for k in range(steps)])
        attained = d @ closed_loop_trajectory(x0, errors)[-1]
        scale = extent(sets[steps], d)
        assert sets[steps].support(d) == pytest.approx(attained, abs=1e-12 * scale)
        assert support(loop, start, nav, d, steps) == pytest.approx(attained, abs=1e-12 * scale)


def test_closed_loop_matrix_is_formed_before_the_set_operation():
    loop = cw_loop(N, STEP, gain=K, hold=HOLD)
    np.testing.assert_allclose(loop.A, PHI - GAMMA @ K, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(loop.B, -GAMMA @ K, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(loop.c, GAMMA @ K @ HOLD, rtol=1e-12, atol=1e-15)
    # Algorithms for Validation, example 8.2: composing the open-loop sets treats the state in
    # phi x and in -gamma K x as independent, which inflates the set (the dependency effect).
    start = Zonotope.box(HOLD, START_HALF)
    exact = propagate(loop, start, None, 1)[1]
    composed = start.map(PHI) + start.map(-GAMMA @ K) + GAMMA @ K @ HOLD
    for d in unit_directions(6, 20, seed=14):
        assert exact.support(d) <= composed.support(d) + 1e-12 * extent(composed, d)
    widths = [
        (exact.support(e) + exact.support(-e), composed.support(e) + composed.support(-e))
        for e in np.eye(6)
    ]
    assert any(tight < 0.9 * loose for tight, loose in widths)


def test_reduced_zonotope_keeps_the_generators_furthest_from_the_axes():
    # Girard, HSCC 2005: rank by |g|_1 - |g|_inf, box the lowest. Order 1.5 in 2-D keeps one
    # generator, (2, -1.5) with score 1.5, and boxes (1, 1), (3, 0.1), (0.2, 2).
    g = np.array([[1.0, 3.0, 0.2, 2.0], [1.0, 0.1, 2.0, -1.5]])
    z = Zonotope([1.0, -2.0], g)
    reduced = z.reduce(1.5)
    expected = Zonotope([1.0, -2.0], [[2.0, 4.2, 0.0], [-1.5, 0.0, 3.1]])
    assert reduced.generators.shape[1] == 3
    for d in unit_directions(2, 40, seed=15):
        assert reduced.support(d) == pytest.approx(expected.support(d), rel=1e-12)
    assert contains(reduced, z)


@settings(max_examples=60, deadline=None)
@given(
    dim=st.integers(2, 4),
    count=st.integers(1, 12),
    order=st.sampled_from([1.0, 1.5, 2.0, 3.0]),
    seed=st.integers(0, 2**32 - 1),
    scale=st.floats(1e-3, 1e3, **finite),
)
def test_order_reduction_is_an_outer_set_of_the_stated_order(dim, count, order, seed, scale):
    rng = np.random.default_rng(seed)
    z = Zonotope(scale * rng.normal(size=dim), scale * rng.normal(size=(dim, count)))
    reduced = z.reduce(order)
    limit = math.floor(order * dim)
    assert reduced.generators.shape[1] <= limit
    np.testing.assert_array_equal(reduced.center, z.center)
    if count <= limit:
        np.testing.assert_array_equal(reduced.generators, z.generators)
    for d in unit_directions(dim, 30, seed=seed):
        assert reduced.support(d) >= z.support(d) - 1e-12 * extent(z, d)
    # Boxing part of the generators keeps the interval hull, so the axis supports are equal.
    for e in np.vstack([np.eye(dim), -np.eye(dim)]):
        assert reduced.support(e) == pytest.approx(z.support(e), rel=1e-12, abs=1e-12 * scale)


def test_ball_distance_is_exact_when_the_closest_point_is_a_vertex():
    z = Zonotope.box([10.0, 10.0, 10.0], [1.0, 1.0, 1.0])
    true = 9.0 * math.sqrt(3.0) - 5.0
    lower, upper = ball_distance(z, 5.0)
    assert lower <= true <= upper + 1e-12
    assert lower == pytest.approx(true, abs=1e-6)
    assert upper == pytest.approx(true, abs=1e-6)


def test_ball_distance_is_minus_radius_when_the_set_reaches_the_origin():
    z = Zonotope([1.0, -2.0, 0.5], [[3.0, 0.0, 1.0], [1.0, 4.0, 0.0], [0.0, 1.0, 2.0]])
    lower, upper = ball_distance(z, 50.0)
    assert lower == pytest.approx(-50.0, abs=1e-9)
    assert upper == pytest.approx(-50.0, abs=1e-9)


def test_ball_distance_lower_bound_is_below_dense_face_samples():
    rng = np.random.default_rng(16)
    z = Zonotope([40.0, -25.0, 30.0], rng.normal(scale=8.0, size=(3, 4)))
    grid = np.linspace(-1.0, 1.0, 9)
    points = [corners(z)]
    for j in range(4):
        for side in (-1.0, 1.0):
            beta = np.array(list(itertools.product(grid, repeat=3)))
            beta = np.insert(beta, j, side, axis=1)
            points.append(z.center + beta @ z.generators.T)
    sampled = float(np.min(np.linalg.norm(np.vstack(points), axis=1))) - 50.0
    lower, upper = ball_distance(z, 50.0)
    assert lower <= sampled
    assert lower <= upper
    # The closest point is the exact minimum over the set; the grid can only sit above it.
    assert upper <= sampled + 1e-9


@settings(max_examples=100, deadline=None)
@given(
    center=st.lists(st.floats(-100.0, 100.0, **finite), min_size=3, max_size=3),
    count=st.integers(1, 6),
    seed=st.integers(0, 2**32 - 1),
    spread=st.floats(0.0, 50.0, **finite),
    radius=st.floats(0.0, 100.0, **finite),
)
def test_ball_distance_brackets_the_true_margin(center, count, seed, spread, radius):
    c = np.array(center)
    g = spread * np.random.default_rng(seed).normal(size=(3, count))
    z = Zonotope(c, g)
    lower, upper = ball_distance(z, radius)
    nearest = lsq_linear(g, -c, bounds=(-1.0, 1.0), method="bvls", tol=1e-14).x
    p = c + g @ nearest
    true = min(np.linalg.norm(p), np.min(np.linalg.norm(corners(z), axis=1))) - radius
    # A separating hyperplane through the reference point: a certified bound of our own.
    norm = np.linalg.norm(p)
    lam = p / norm if norm > 0.0 else np.zeros(3)
    certified = lam @ c - np.sum(np.abs(lam @ g)) - radius
    tol = 1e-9 * (np.linalg.norm(c) + np.sum(np.abs(g)) + radius + 1.0)
    assert lower <= true + tol
    assert upper >= max(certified, -radius) - tol
    assert lower <= upper + tol
    assert lower >= -radius - tol


def vertex_drift_minimum(x0, samples):
    ts = np.linspace(0.0, TWO_ORBITS, samples)
    phis, _ = cw_discrete(N, ts)
    states = np.einsum("tij,vj->tvi", phis, corners(x0))
    return float(np.min(np.linalg.norm(states[..., :3], axis=-1)))


def test_point_at_vbar_hold_drifts_clear_by_fifty_metres():
    clearance = drift_clearance(N, Zonotope.point(HOLD), 50.0, TWO_ORBITS, 801)
    assert clearance.found == pytest.approx(50.0, abs=1e-3)
    assert clearance.certified <= clearance.found
    assert clearance.clear


def test_three_sigma_box_of_three_metres_reaches_the_target():
    # A 3 m, 3 mm/s box around the hold reaches the origin, so the found margin is -radius; the
    # certified bound loses at most v h / 2 between 2001 samples over two orbits.
    x0 = Zonotope.box(HOLD, [3.0, 3.0, 3.0, 3e-3, 3e-3, 3e-3])
    clearance = drift_clearance(N, x0, 50.0, TWO_ORBITS, 2001)
    assert clearance.found == pytest.approx(-50.0, abs=1e-3)
    assert -50.35 <= clearance.certified <= clearance.found
    assert not clearance.clear


def test_box_of_sixty_centimetres_still_enters_the_keep_out():
    x0 = Zonotope.box(HOLD, [0.6, 0.6, 0.6, 3e-4, 3e-4, 3e-4])
    clearance = drift_clearance(N, x0, 50.0, TWO_ORBITS, 1001)
    assert clearance.found == pytest.approx(-6.157, abs=1e-3)
    assert clearance.certified <= clearance.found
    assert clearance.certified <= vertex_drift_minimum(x0, 8001) - 50.0


def exact_margin(r, radius):
    """|r| - radius of float inputs in 60-digit decimal arithmetic (m)."""
    with localcontext() as context:
        context.prec = 60
        return sum(Decimal(float(c)) ** 2 for c in r).sqrt() - Decimal(float(radius))


def test_certified_clearance_of_a_receding_point_is_not_above_its_exact_margin():
    # The point is a trajectory of the set, so its margin at t = 0 bounds every certified value
    # from above. Rounding of the norm put the certified one 5e-14 m above it, 1.5e-16 of its range.
    x0 = [123.77143839022689, 118.30198966826268, 281.01877035297036]
    x0 += [0.011008728372664167, 0.010522253656756901, 0.024994937044217295]
    radius = 48.30784222506006
    clearance = drift_clearance(0.0011, x0, radius, 1e-6, 2)
    assert Decimal(clearance.certified) <= exact_margin(x0[:3], radius)


@pytest.mark.parametrize(("duration", "samples"), [(1e-6, 2), (1e-6, 3)])
def test_certified_clearance_is_never_above_the_exact_margin_of_a_receding_start(duration, samples):
    # Starts from 1 cm to 100 km out, receding at up to 1% of their range per second, so the
    # margin at t = 0 is the minimum and the exact value is |r| - R. The certified margin may fall
    # below it by the rounding allowance and the speed bound's slack over a 1 us interval, which
    # stay under 1e-12 of the range.
    rng = np.random.default_rng(5)
    above, below = [], []
    for _ in range(100):
        scale = 10.0 ** rng.uniform(-2.0, 5.0)
        r = rng.normal(size=3)
        r *= scale / np.linalg.norm(r)
        v = r * rng.uniform(1e-6, 1e-2)
        radius = scale * rng.uniform(0.1, 0.99)
        clearance = drift_clearance(0.0011, np.concatenate([r, v]), radius, duration, samples)
        excess = float(Decimal(clearance.certified) - exact_margin(r, radius)) / scale
        (above if excess > 0.0 else below).append(excess)
    assert not above, f"{len(above)} of 100 certified margins above the exact one, by {max(above)}"
    assert min(below) >= -1e-12


def segment_with_vertex_nearest(rng):
    """Centre c, generator g and range (m) of a segment nearest the origin at one of its ends."""
    scale = 10.0 ** rng.uniform(-2.0, 5.0)
    while True:
        c = rng.normal(size=3)
        c *= scale / np.linalg.norm(c)
        g = rng.normal(size=3) * scale * rng.uniform(0.02, 0.6)
        # The line through the segment is nearest the origin at beta = -c.g / g.g; a factor of
        # 1.5 puts that point beyond an end, clear of any rounding in this test.
        if abs(c @ g) >= 1.5 * (g @ g):
            return c, g, scale


def exact_end_margin(c, g, radius):
    """The smaller of |c + g| and |c - g|, less the radius, in 60-digit decimal arithmetic (m)."""
    with localcontext() as context:
        context.prec = 60
        c, g = [Decimal(float(x)) for x in c], [Decimal(float(x)) for x in g]
        ends = [sum((a + s * b) ** 2 for a, b in zip(c, g, strict=True)).sqrt() for s in (1, -1)]
        return min(ends) - Decimal(float(radius))


def segment_lower_bound(c, g, radius):
    return ball_distance(Zonotope(c, g[:, None]), radius)[0]


def segment_certified_clearance(c, g, radius):
    # At rest, so the speed bound's slack over 1 us is about 1e-18 of the range, below rounding,
    # and the margin at t = 0 bounds every certified value from above.
    x0 = Zonotope(np.concatenate([c, np.zeros(3)]), np.concatenate([g, np.zeros(3)])[:, None])
    return drift_clearance(0.0011, x0, radius, 1e-6, 2).certified


@pytest.mark.parametrize(
    "certified", [segment_lower_bound, segment_certified_clearance], ids=["ball", "drift"]
)
def test_certified_bound_of_a_segment_is_not_above_the_exact_margin_of_its_nearest_end(certified):
    # With the nearest point an end, the separating hyperplane through it is exact in exact
    # arithmetic, so the bound lands within rounding of the margin and sits above it in about
    # half of the cases unless the generator term carries its own rounding allowance.
    rng = np.random.default_rng(11)
    above = []
    for _ in range(100):
        c, g, scale = segment_with_vertex_nearest(rng)
        radius = scale * rng.uniform(0.0, 0.9)
        excess = float(Decimal(certified(c, g, radius)) - exact_end_margin(c, g, radius)) / scale
        if excess > 0.0:
            above.append(excess)
    assert not above, f"{len(above)} of 100 bounds above the exact margin, by {max(above)} of range"


def point_lower_bound(r, v, radius):
    return ball_distance(Zonotope.point(r), radius)[0]


def receding_certified_clearance(r, v, radius):
    return drift_clearance(0.0011, np.concatenate([r, v]), radius, 1e-6, 2).certified


@pytest.mark.parametrize(
    "certified", [point_lower_bound, receding_certified_clearance], ids=["ball", "drift"]
)
def test_certified_bound_is_not_above_the_exact_margin_of_a_distant_keep_out(certified):
    # Keep-out radii from 2 to 1000 times the range: the margin |r| - R is then rounded at the
    # size of R, which exceeds the allowance for the norm alone. The velocity is along r, so the
    # speed bound is tight and the exact margin at t = 0 is the one to stay below.
    rng = np.random.default_rng(7)
    above, loose = [], []
    for _ in range(300):
        scale = 10.0 ** rng.uniform(-2.0, 5.0)
        r = rng.normal(size=3)
        r *= scale / np.linalg.norm(r)
        v = r * rng.uniform(1e-6, 1e-2)
        radius = scale * 10.0 ** rng.uniform(math.log10(2.0), 3.0)
        excess = float(Decimal(certified(r, v, radius)) - exact_margin(r, radius))
        if excess > 0.0:
            above.append(excess)
        # The norm's allowance, 32 eps |r| for a point, is at most 16 eps R here, and the radius
        # term adds at most 4 eps R; 15.8 eps R was the loosest seen, so 24 eps R catches a radius
        # term even ten times too large.
        if excess < -24.0 * np.finfo(float).eps * radius:
            loose.append(excess / radius)
    assert not above, f"{len(above)} of 300 bounds above the exact margin, by {max(above)} m"
    assert not loose, f"{len(loose)} of 300 bounds looser than 24 eps R, by {min(loose)} of R"


def test_largest_passively_safe_velocity_box_lies_between_1_4140_and_1_4160_mm_per_s():
    # With exact positions at the hold, the along-track closing over two orbits is
    # v / n (2 (1 - cos t) + 3 t - 4 sin t) at worst, which reaches 50 m at v = 1.4147 mm/s.
    tau = np.linspace(0.0, N * TWO_ORBITS, 200001)
    closed_form = 50.0 * N / np.max(2.0 * (1.0 - np.cos(tau)) + 3.0 * tau - 4.0 * np.sin(tau))
    assert 1.4140e-3 < closed_form < 1.4160e-3
    safe = drift_clearance(
        N, Zonotope.box(HOLD, [0, 0, 0, *[1.4140e-3] * 3]), 50.0, TWO_ORBITS, 4001
    )
    unsafe = drift_clearance(
        N, Zonotope.box(HOLD, [0, 0, 0, *[1.4160e-3] * 3]), 50.0, TWO_ORBITS, 801
    )
    assert safe.clear
    assert safe.certified > 0.0
    assert unsafe.found < 0.0
    assert not unsafe.clear


def test_hold_point_lqr_loop_reaches_an_invariant_set():
    loop = cw_loop(N, STEP, gain=K, hold=HOLD)
    start = Zonotope.box(HOLD, [10.0, 10.0, 10.0, 1e-2, 1e-2, 1e-2])
    nav = Zonotope.box(np.zeros(6), [0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4])
    step = invariant_step(loop, start, nav, 50)
    assert step is not None
    assert 1 <= step <= 50


def test_free_drift_sets_are_never_invariant():
    start = Zonotope.box(HOLD, [10.0, 10.0, 10.0, 1e-2, 1e-2, 1e-2])
    assert invariant_step(cw_loop(N, STEP), start, None, 50) is None


def test_containment_is_exact_at_the_boundary():
    z = Zonotope(
        [1.0, 2.0, -1.0], [[2.0, 0.5, 0.0, 1.0], [0.0, 1.0, 1.5, -1.0], [1.0, 0.0, 2.0, 0.5]]
    )
    shrunk = Zonotope(z.center, (1.0 - 1e-6) * z.generators)
    assert contains(z, z)
    assert contains(z, shrunk)
    assert not contains(shrunk, z)
    assert contains(z, z.support_point([1.0, 1.0, 1.0]))
    assert not contains(shrunk, z.support_point([1.0, 1.0, 1.0]))


def test_containment_agrees_with_vertex_membership():
    rng = np.random.default_rng(17)
    for _ in range(100):
        outer = Zonotope(rng.normal(size=2), rng.normal(size=(2, int(rng.integers(2, 5)))))
        if np.linalg.matrix_rank(outer.generators) < 2:
            continue
        s = rng.choice([rng.uniform(0.3, 0.9), rng.uniform(1.1, 1.6)])
        inner = Zonotope(
            outer.center + 0.1 * rng.normal(size=2),
            s * outer.generators[:, : int(rng.integers(1, outer.generators.shape[1] + 1))],
        )
        expected = all(in_zonotope(outer, v) for v in corners(inner))
        assert contains(outer, inner) == expected


def test_containment_refuses_flat_or_oversized_outer_sets():
    flat = Zonotope([0.0, 0.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        contains(flat, Zonotope.point([0.0, 0.0, 0.0]))
    # 2 C(20, 5) = 31008 facets in 6-D.
    big = Zonotope(np.zeros(6), np.random.default_rng(18).normal(size=(6, 20)))
    with pytest.raises(ValueError):
        contains(big, Zonotope.point(np.zeros(6)))


def test_minimum_time_to_hold_is_28_steps_with_a_certificate_at_27():
    a_max = 1e-3
    result = min_time_steps(PHI, GAMMA, FAR, HOLD, a_max, 40)
    assert result.steps == 28
    plan = np.asarray(result.plan)
    assert plan.shape == (28, 3)
    assert np.max(np.abs(plan)) <= a_max * (1.0 + 1e-12)
    x = FAR.copy()
    for u in plan:
        x = PHI @ x + GAMMA @ u
    assert np.max(np.abs(x - HOLD)) <= 1e-6
    assert result.residual <= 1e-6
    # The separating direction for 27 steps, checked in closed form from nu alone.
    assert result.nu is not None
    nu = np.asarray(result.nu)
    blocks = np.hstack([np.linalg.matrix_power(PHI, 26 - k) @ GAMMA for k in range(27)])
    miss = HOLD - np.linalg.matrix_power(PHI, 27) @ FAR
    rho = a_max * np.sum(np.abs(blocks.T @ nu))
    gap = nu @ miss - rho
    assert gap > 1e-9 * (abs(nu @ miss) + rho)
    assert result.gap == pytest.approx(gap, rel=1e-9)


def horizon_matrices(steps):
    blocks = [np.linalg.matrix_power(PHI, steps - 1 - k) @ GAMMA for k in range(steps)]
    return np.hstack(blocks), HOLD - np.linalg.matrix_power(PHI, steps) @ FAR


def test_separating_direction_is_withheld_where_the_target_is_reachable():
    # The minimum-time search reaches this check only at horizons its linear program already
    # found infeasible, so the reachable case is exercised on the helper directly. Every nu has
    # nu . b <= rho(nu) when b is reachable, so no direction may be returned as a proof.
    a_max = 1e-3
    m, b = horizon_matrices(28)
    nu, gap = _certificate(m, b, a_max)
    assert nu is None
    assert gap <= 1e-9 * np.max(np.abs(b))
    u = 0.5 * a_max * np.random.default_rng(19).uniform(-1.0, 1.0, size=m.shape[1])
    nu, gap = _certificate(m, m @ u, a_max)
    assert nu is None
    m, b = horizon_matrices(27)
    nu, gap = _certificate(m, b, a_max)
    assert nu is not None
    assert gap > 0.0


def test_minimum_time_reports_no_horizon_below_the_first_feasible_one():
    result = min_time_steps(PHI, GAMMA, FAR, HOLD, 1e-3, 27)
    assert result.steps is None
    assert result.plan is None


def test_minimum_time_refuses_a_ball_limit():
    with pytest.raises(ValueError):
        min_time_steps(PHI, GAMMA, FAR, HOLD, 1e-3, 40, layout="ball")


R_500 = EARTH.radius + 500e3


def gravity_remainders(rho):
    """|g(r + rho u) - g(r) - G rho u| over a 181 x 121 grid of directions u, m/s^2."""
    mu = EARTH.mu
    th, ph = np.meshgrid(np.linspace(0.0, math.pi, 181), np.linspace(0.0, 2.0 * math.pi, 121))
    u = np.stack([np.cos(th), np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)], axis=-1)
    s = np.array([R_500, 0.0, 0.0]) + rho * u
    g = -mu * s / np.linalg.norm(s, axis=-1, keepdims=True) ** 3
    g0 = np.array([-mu / R_500**2, 0.0, 0.0])
    grad = mu / R_500**3 * np.diag([2.0, -1.0, -1.0])
    return np.linalg.norm(g - g0 - rho * u @ grad.T, axis=-1)


def test_gravity_remainder_leading_coefficient_is_three():
    # Algorithms for Validation, sec. 9.3 (conservative linearization), at r = 6878.137 km.
    rho = 50.0
    unit = EARTH.mu * rho**2 / R_500**4
    assert np.max(gravity_remainders(rho)) / unit == pytest.approx(3.0, abs=1e-4)
    assert gravity_remainder_bound(EARTH.mu, R_500, rho) / unit == pytest.approx(3.0, abs=1e-4)


@pytest.mark.parametrize("rho", [100.0, 1e3, 1e4])
def test_gravity_remainder_bound_holds_in_every_direction(rho):
    bound = gravity_remainder_bound(EARTH.mu, R_500, rho)
    worst = np.max(gravity_remainders(rho))
    assert worst <= bound
    assert worst >= 0.99 * bound


def test_gravity_remainder_needs_the_inner_radius():
    # Radially inward at 10 km the remainder exceeds 3 mu rho^2 / r^4 by about 0.19 %.
    rho = 1e4
    assert np.max(gravity_remainders(rho)) > 3.0 * EARTH.mu * rho**2 / R_500**4
    with pytest.raises(ValueError):
        gravity_remainder_bound(EARTH.mu, R_500, R_500)


def test_loop_matrices_must_be_square_and_finite():
    with pytest.raises(ValueError):
        LinearLoop(np.ones((2, 3)), np.ones((2, 1)), np.zeros(2))
    with pytest.raises(ValueError):
        LinearLoop(np.eye(2), np.ones((3, 1)), np.zeros(2))
    with pytest.raises(ValueError):
        Zonotope([0.0, math.nan], np.eye(2))
    with pytest.raises(ValueError):
        cw_loop(N, STEP, gain=np.ones((6, 3)))


def test_loop_clearance_refuses_a_loop_that_saturates_on_the_set():
    loop = cw_loop(N, STEP, gain=K, hold=HOLD)
    start = Zonotope.box(HOLD, [1e3, 1e3, 1e3, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="saturates"):
        loop_clearance(loop, start, None, 50.0, 5, 10, actuator=ActuatorSet(1e-3, "box"))
