import numpy as np
import pytest
from scipy.optimize import minimize

from vbar.dynamics import EARTH, mean_motion
from vbar.solvers import ActuatorSet, HillScaling, Rows, project, qp
from vbar.solvers.qp import BOX_FACES

# Face rows in the documented order: "box+x" is -u_x >= -a_max, "box-x" is u_x >= -a_max.
FACE_NORMALS = np.array([s * e for e in np.eye(3) for s in (-1.0, 1.0)])


def named_rows(a, b):
    return Rows(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        tuple(f"r{i}" for i in range(len(b))),
    )


def on_actuator(u, actuator):
    if actuator.layout == "box":
        return np.clip(u, -actuator.a_max, actuator.a_max)
    radius = np.linalg.norm(u)
    return u * (actuator.a_max / radius) if radius > actuator.a_max else u


def assert_certificate_proves_infeasibility(projection, rows, actuator):
    lam = projection.certificate
    assert lam is not None and np.all(lam >= 0.0)
    k = len(rows.b)
    if actuator is not None and actuator.layout == "box":
        a = np.vstack([rows.a, FACE_NORMALS])
        b = np.append(rows.b, np.full(6, -actuator.a_max))
        assert lam.shape == (k + 6,)
    else:
        a, b = rows.a, rows.b
        assert lam.shape == (k,)
    if actuator is not None and actuator.layout == "ball":
        # For every |u| <= a_max: lam^T A u <= a_max |A^T lam| < b^T lam, so some row fails.
        assert b @ lam > actuator.a_max * np.linalg.norm(a.T @ lam)
    else:
        scale = np.sum(lam * np.linalg.norm(a, axis=1))
        assert np.max(np.abs(a.T @ lam)) <= 1e-9 * scale
        assert b @ lam > 1e-9 * scale * np.max(np.abs(b))


def test_boyd_exercise_4_3_box_qp_optimum_and_multipliers():
    # Boyd and Vandenberghe, Convex Optimization, exercise 4.3: min 1/2 x^T P x + q^T x + 1 on
    # -1 <= x_i <= 1 has x* = (1, 0.5, -1), p* = -21.625 and bound multipliers 1 (x1 <= 1) and
    # 2 (x3 >= -1). With P = L L^T and y = L^T x it is a Euclidean projection of y0 = L^T x0,
    # x0 = -P^-1 q, onto six rows; multipliers do not change under the change of variables.
    p = np.array([[13.0, 12.0, -2.0], [12.0, 17.0, 6.0], [-2.0, 6.0, 12.0]])
    q = np.array([-22.0, -14.5, 13.0])
    chol = np.linalg.cholesky(p)
    inv = np.linalg.inv(chol)
    x0 = -np.linalg.solve(p, q)
    a, b, names = [], [], []
    for i in range(3):
        a += [-inv[:, i], inv[:, i]]
        b += [-1.0, -1.0]
        names += [f"x{i + 1}<=1", f"x{i + 1}>=-1"]
    rows = Rows(np.array(a), np.array(b), tuple(names))
    result = project(chol.T @ x0, rows)
    x = np.linalg.solve(chol.T, result.u)
    assert result.status == "feasible"
    np.testing.assert_allclose(x, [1.0, 0.5, -1.0], rtol=0, atol=1e-12)
    p_star = result.objective + 1.0 - 0.5 * x0 @ p @ x0
    assert p_star == pytest.approx(-21.625, rel=1e-12)
    np.testing.assert_allclose(result.multipliers, [1.0, 0, 0, 0, 0, 2.0], rtol=0, atol=1e-12)
    assert result.active == ("x1<=1", "x3>=-1")


def test_halfspace_projection_matches_closed_form():
    # Boyd and Vandenberghe, sec. 8.1.1: u = u0 + max(0, b - a . u0) a / |a|^2.
    rng = np.random.default_rng(1)
    for _ in range(200):
        u0 = rng.uniform(-1.0, 1.0, 3)
        a = rng.uniform(-1.0, 1.0, 3)
        b = rng.uniform(-1.0, 1.0)
        step = max(0.0, b - a @ u0) / (a @ a)
        result = project(u0, named_rows([a], [b]))
        np.testing.assert_allclose(result.u, u0 + step * a, rtol=0, atol=1e-12 * max(1.0, step))
        assert result.multipliers[0] == pytest.approx(step, rel=1e-12, abs=1e-14)
        assert result.active == (("r0",) if step > 0 else ())


def test_box_projection_without_rows_is_clipping_with_the_faces_named():
    rng = np.random.default_rng(2)
    box = ActuatorSet(0.5, "box")
    for _ in range(100):
        u0 = rng.uniform(-1.0, 1.0, 3)
        weight = rng.uniform(0.1, 10.0, 3)
        result = project(u0, named_rows(np.zeros((0, 3)), []), box, weight)
        np.testing.assert_allclose(result.u, np.clip(u0, -0.5, 0.5), rtol=0, atol=1e-15)
        beyond = np.stack([u0 > 0.5, u0 < -0.5], axis=1).ravel()
        assert result.active == tuple(np.array(BOX_FACES)[beyond])
        assert result.phase1_margin == -np.inf


def test_ball_projection_without_rows_scales_radially():
    rng = np.random.default_rng(3)
    ball = ActuatorSet(0.5, "ball")
    for _ in range(100):
        u0 = rng.uniform(-1.0, 1.0, 3)
        result = project(u0, named_rows(np.zeros((0, 3)), []), ball)
        np.testing.assert_allclose(result.u, on_actuator(u0, ball), rtol=0, atol=1e-12 * 0.5)
        assert result.active == (("ball",) if np.linalg.norm(u0) > 0.5 else ())


def dual_search(u_nom, a, b, actuator):
    # u(lam) = P_U(u_nom + lam a) for H = I, and a . u(lam) is nondecreasing in lam.
    def u_of(lam):
        return on_actuator(u_nom + lam * a, actuator)

    if a @ u_of(0.0) >= b:
        return u_of(0.0)
    lo, hi = 0.0, 1.0
    while a @ u_of(hi) < b:
        lo, hi = hi, 2.0 * hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if a @ u_of(mid) < b else (lo, mid)
    return u_of(hi)


@pytest.mark.parametrize("layout, norm_ord", [("box", 1), ("ball", 2)])
def test_one_row_feasibility_rule_and_solution_match_a_dual_search(layout, norm_ord):
    # One row under a box or ball is feasible iff a_max |a|_1 >= b (box) or a_max |a|_2 >= b
    # (ball), the support function of the actuator set.
    rng = np.random.default_rng(4)
    a_max = 0.02
    actuator = ActuatorSet(a_max, layout)
    feasible_count = 0
    for _ in range(500):
        u_nom = rng.uniform(-2 * a_max, 2 * a_max, 3)
        a = rng.uniform(-1.0, 1.0, 3)
        while np.linalg.norm(a) < 0.1:
            a = rng.uniform(-1.0, 1.0, 3)
        support = a_max * np.linalg.norm(a, norm_ord)
        b = rng.uniform(-1.2, 1.2) * support
        # Instances on the boundary sit at the rounding floor of both methods.
        while abs(b - support) < 1e-6 * support:
            b = rng.uniform(-1.2, 1.2) * support
        result = project(u_nom, named_rows([a], [b]), actuator)
        if b <= support:
            feasible_count += 1
            assert result.status == "feasible"
            expected = dual_search(u_nom, a, b, actuator)
            np.testing.assert_allclose(result.u, expected, rtol=0, atol=1e-9 * a_max)
        else:
            assert result.status == "infeasible_certified"
            assert result.conflicting[0] == "r0"
            assert_certificate_proves_infeasibility(result, named_rows([a], [b]), actuator)
    assert 300 < feasible_count < 500


def slsqp(u_nom, rows, actuator, weight, start):
    constraints = [{"type": "ineq", "fun": lambda u: rows.a @ u - rows.b, "jac": lambda u: rows.a}]
    bounds = None
    if actuator.layout == "box":
        bounds = [(-actuator.a_max, actuator.a_max)] * 3
    else:
        constraints.append(
            {"type": "ineq", "fun": lambda u: actuator.a_max**2 - u @ u, "jac": lambda u: -2 * u}
        )
    res = minimize(
        lambda u: 0.5 * np.sum(weight * (u - u_nom) ** 2),
        start,
        jac=lambda u: weight * (u - u_nom),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    # Status 8 is a line search that cannot improve on its point; the comparison still checks it.
    assert res.status in (0, 8), res.message
    return res.x, res.fun


def random_feasible_instance(rng, layout, k):
    a_max = 1.0
    inside = rng.uniform(-1.0, 1.0, 3)
    inside *= (
        rng.uniform(0.0, 0.8)
        * a_max
        / np.max(np.abs(inside) if layout == "box" else np.linalg.norm(inside))
    )
    a = rng.normal(0.0, 1.0, (k, 3))
    b = a @ inside - rng.uniform(0.0, 0.3, k) * a_max * np.linalg.norm(a, axis=1)
    u_nom = rng.uniform(-3.0, 3.0, 3) * a_max
    weight = rng.uniform(0.2, 5.0, 3)
    return named_rows(a, b), ActuatorSet(a_max, layout), u_nom, weight, inside


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_several_rows_match_slsqp_with_tight_active_rows(layout):
    rng = np.random.default_rng(5 if layout == "box" else 6)
    for _ in range(200):
        rows, actuator, u_nom, weight, inside = random_feasible_instance(
            rng, layout, rng.integers(2, 7)
        )
        result = project(u_nom, rows, actuator, weight)
        assert result.status == "feasible"
        u_ref, f_ref = slsqp(u_nom, rows, actuator, weight, inside)
        assert result.objective == pytest.approx(f_ref, rel=1e-6, abs=1e-12)
        np.testing.assert_allclose(result.u, u_ref, rtol=0, atol=1e-6)
        assert np.min(rows.a @ result.u - rows.b) >= -1e-12
        tight = rows.a @ result.u - rows.b
        for name, gap in zip(rows.names, tight, strict=True):
            if name in result.active:
                assert abs(gap) <= 1e-12


def test_32_rows_under_a_box_match_slsqp():
    rng = np.random.default_rng(7)
    rows, actuator, u_nom, weight, inside = random_feasible_instance(rng, "box", 32)
    result = project(u_nom, rows, actuator, weight)
    u_ref, f_ref = slsqp(u_nom, rows, actuator, weight, inside)
    assert result.status == "feasible"
    assert result.objective == pytest.approx(f_ref, rel=1e-6, abs=1e-12)
    np.testing.assert_allclose(result.u, u_ref, rtol=0, atol=1e-6)


def test_more_than_32_rows_are_refused():
    with pytest.raises(ValueError, match="at most 32 rows"):
        named_rows(np.ones((33, 3)), np.zeros(33))


def test_multipliers_satisfy_the_kkt_conditions_without_an_actuator():
    rng = np.random.default_rng(8)
    for _ in range(200):
        k = rng.integers(1, 9)
        inside = rng.uniform(-1.0, 1.0, 3)
        a = rng.normal(0.0, 1.0, (k, 3))
        rows = named_rows(a, a @ inside - rng.uniform(0.0, 0.5, k))
        u_nom = rng.uniform(-3.0, 3.0, 3)
        weight = rng.uniform(0.2, 5.0, 3)
        result = project(u_nom, rows, weight=weight)
        lam = result.multipliers
        gap = rows.a @ result.u - rows.b
        scale = max(1.0, np.max(np.abs(result.u)))
        assert result.status == "feasible"
        assert np.all(lam >= 0.0) and np.min(gap) >= -1e-12 * scale
        np.testing.assert_allclose(
            weight * (result.u - u_nom), rows.a.T @ lam, rtol=0, atol=1e-11 * scale
        )
        assert np.max(np.abs(lam * gap)) <= 1e-11 * scale


def test_weighted_halfspace_projection_matches_closed_form():
    # min 1/2 (u - u0)^T W (u - u0) on a . u >= b:
    # u = u0 + max(0, b - a . u0) W^-1 a / (a^T W^-1 a).
    rng = np.random.default_rng(9)
    for _ in range(200):
        u0 = rng.uniform(-1.0, 1.0, 3)
        a = rng.uniform(-1.0, 1.0, 3)
        b = rng.uniform(-1.0, 1.0)
        weight = rng.uniform(0.1, 10.0, 3)
        lam = max(0.0, b - a @ u0) / (a @ (a / weight))
        result = project(u0, named_rows([a], [b]), weight=weight)
        np.testing.assert_allclose(
            result.u, u0 + lam * a / weight, rtol=0, atol=1e-12 * max(1.0, lam)
        )
        assert result.multipliers[0] == pytest.approx(lam, rel=1e-12, abs=1e-14)


def test_uniform_weight_scales_multipliers_and_objective_only():
    rng = np.random.default_rng(10)
    rows, actuator, u_nom, weight, _ = random_feasible_instance(rng, "box", 5)
    one = project(u_nom, rows, actuator, weight)
    seven = project(u_nom, rows, actuator, 7.0 * weight)
    np.testing.assert_allclose(seven.u, one.u, rtol=0, atol=1e-14)
    np.testing.assert_allclose(seven.multipliers, 7.0 * one.multipliers, rtol=1e-12, atol=1e-14)
    assert seven.objective == pytest.approx(7.0 * one.objective, rel=1e-12)
    assert seven.active == one.active


@pytest.mark.parametrize("weight", [[1.0, 0.0, 1.0], [1.0, 1.0, np.inf], [1.0, 1.0]])
def test_weight_must_be_three_positive_finite_numbers(weight):
    with pytest.raises(ValueError, match="weight"):
        project(np.zeros(3), named_rows([[1.0, 0.0, 0.0]], [0.0]), weight=np.array(weight))


def test_opposite_rows_are_certified_and_both_named():
    rows = Rows(
        np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), np.array([1.0, 0.0]), ("ahead", "behind")
    )
    for actuator in (None, ActuatorSet(5.0, "box"), ActuatorSet(5.0, "ball")):
        result = project(np.array([0.3, 0.2, 0.1]), rows, actuator)
        assert result.status == "infeasible_certified"
        assert result.conflicting == ("ahead", "behind")
        assert_certificate_proves_infeasibility(result, rows, actuator)
        assert result.phase1_margin == pytest.approx(1.0, rel=1e-12)
        np.testing.assert_allclose(result.multipliers, 0.0)


def test_row_beyond_the_box_is_certified_with_the_faces_it_needs():
    # u_x + u_y >= 3 under |u_i| <= 1: the box reaches at most 2 along (1, 1, 0).
    rows = Rows(
        np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), np.array([3.0, -0.9]), ("reach", "floor")
    )
    box = ActuatorSet(1.0, "box")
    result = project(np.array([0.0, 0.0, 0.4]), rows, box)
    assert result.status == "infeasible_certified"
    assert result.conflicting == ("reach", "box+x", "box+y")
    assert_certificate_proves_infeasibility(result, rows, box)
    # The least-violation command does the best the box allows and leaves z alone.
    np.testing.assert_allclose(result.u, [1.0, 1.0, 0.4], rtol=0, atol=1e-12)
    assert result.phase1_margin == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-12)


def test_ball_too_small_for_a_row_is_certified():
    rows = Rows(np.array([[1.0, 1.0, 0.0]]), np.array([3.0]), ("reach",))
    ball = ActuatorSet(1.0, "ball")
    result = project(np.array([0.0, -1.0, 0.0]), rows, ball)
    assert result.status == "infeasible_certified"
    assert result.conflicting == ("reach", "ball")
    assert_certificate_proves_infeasibility(result, rows, ball)
    np.testing.assert_allclose(result.u, [np.sqrt(0.5), np.sqrt(0.5), 0.0], rtol=0, atol=1e-12)


def test_zero_row_with_positive_bound_conflicts_on_its_own():
    rows = Rows(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.array([0.1, 0.0]), ("degree", "x"))
    result = project(np.zeros(3), rows, ActuatorSet(1.0, "box"))
    assert result.status == "infeasible_certified"
    assert result.conflicting == ("degree",)
    holds = project(
        np.zeros(3), Rows(rows.a, np.array([-0.1, 0.0]), rows.names), ActuatorSet(1.0, "box")
    )
    assert holds.status == "feasible"


@pytest.mark.parametrize("layout", [None, "box", "ball"])
def test_random_conflicts_are_certified_with_valid_certificates(layout):
    rng = np.random.default_rng({None: 14, "box": 15, "ball": 16}[layout])
    actuator = None if layout is None else ActuatorSet(1.0, layout)
    for _ in range(100):
        d = rng.normal(0.0, 1.0, 3)
        t = rng.uniform(-0.25, 0.25) * np.linalg.norm(d)
        gap = rng.uniform(0.01, 0.25) * np.linalg.norm(d)
        # d . u >= t and d . u <= t - gap cannot both hold, but each holds somewhere in the
        # actuator set; the extra rows hold with slack on all of it, so a certificate needs both.
        extra = rng.normal(0.0, 1.0, (0 if actuator is None else rng.integers(0, 4), 3))
        a = np.vstack([d, -d, extra])
        b = np.concatenate(
            [[t, gap - t], -rng.uniform(2.0, 3.0, len(extra)) * np.linalg.norm(extra, 1, axis=1)]
        )
        rows = named_rows(a, b)
        u_nom = rng.uniform(-2.0, 2.0, 3)
        result = project(u_nom, rows, actuator)
        assert result.status == "infeasible_certified"
        assert_certificate_proves_infeasibility(result, rows, actuator)
        assert {"r0", "r1"} <= set(result.conflicting)
        if actuator is not None:
            np.testing.assert_allclose(
                on_actuator(result.u, actuator), result.u, rtol=0, atol=1e-15
            )


def test_single_vertex_feasible_set_is_solved():
    # u_x + u_y >= 2 and u_y + u_z >= 2 meet the unit box only at (1, 1, 1), where five
    # constraints are active and only three are independent.
    rows = Rows(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]), np.array([2.0, 2.0]), ("xy", "yz"))
    rng = np.random.default_rng(17)
    for u_nom in rng.uniform(-3.0, 3.0, (50, 3)):
        result = project(u_nom, rows, ActuatorSet(1.0, "box"))
        assert result.status == "feasible"
        np.testing.assert_allclose(result.u, [1.0, 1.0, 1.0], rtol=0, atol=1e-14)


def test_multipliers_at_a_degenerate_vertex_are_nonnegative_and_stationary():
    # Four rows meet at the origin; several active sets give the same point, and only some have
    # nonnegative multipliers. The first subset tried, {sum, x, y}, has a negative one.
    rows = Rows(np.vstack([np.ones(3), np.eye(3)]), np.zeros(4), ("sum", "x", "y", "z"))
    rng = np.random.default_rng(21)
    for _ in range(50):
        u_nom = -rng.uniform(0.1, 2.0, 3)
        weight = rng.uniform(0.2, 5.0, 3)
        result = project(u_nom, rows, weight=weight)
        np.testing.assert_allclose(result.u, 0.0, rtol=0, atol=1e-15)
        assert np.all(result.multipliers >= 0.0)
        np.testing.assert_allclose(
            weight * (result.u - u_nom), rows.a.T @ result.multipliers, rtol=1e-12, atol=1e-14
        )


def test_row_tangent_to_the_ball_gives_its_single_point_without_finite_multipliers():
    # The analogue of Boyd and Vandenberghe, exercise 5.26: the feasible set is one point,
    # Slater's condition fails and no finite multipliers exist for this u_nom.
    rows = Rows(np.array([[2.0, 0.0, 0.0]]), np.array([2.0]), ("x",))
    result = project(np.array([0.0, 2.0, 0.0]), rows, ActuatorSet(1.0, "ball"))
    assert result.status == "feasible"
    np.testing.assert_allclose(result.u, [1.0, 0.0, 0.0], rtol=0, atol=1e-15)
    assert result.active == ("x", "ball")
    assert result.multipliers[0] == np.inf


def test_same_input_gives_bitwise_identical_output():
    rng = np.random.default_rng(18)
    cases = []
    for layout in ("box", "ball"):
        rows, actuator, u_nom, weight, _ = random_feasible_instance(rng, layout, 6)
        cases.append((u_nom, rows, actuator, weight))
    reach = Rows(np.array([[1.0, 1.0, 0.0]]), np.array([3.0]), ("reach",))
    cases += [(np.zeros(3), reach, ActuatorSet(1.0, "box"), None)]
    cases += [(np.zeros(3), reach, ActuatorSet(1.0, "ball"), None)]
    for u_nom, rows, actuator, weight in cases:
        first = project(u_nom, rows, actuator, weight)
        second = project(u_nom, rows, actuator, weight)
        assert first.u.tobytes() == second.u.tobytes()
        assert first.multipliers.tobytes() == second.multipliers.tobytes()
        assert (first.status, first.active, first.conflicting) == (
            second.status,
            second.active,
            second.conflicting,
        )
        assert first.objective == second.objective and first.phase1_margin == second.phase1_margin
        if first.certificate is not None:
            assert first.certificate.tobytes() == second.certificate.tobytes()


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_projection_in_si_and_in_hill_units_agree(layout):
    scaling = HillScaling(length=100.0, n=mean_motion(EARTH.radius + 400e3))
    rng = np.random.default_rng(19 if layout == "box" else 20)
    a_max = 2e-3
    statuses = set()
    for _ in range(100):
        k = rng.integers(1, 6)
        a = rng.normal(0.0, 1.0, (k, 3))
        b = rng.uniform(-1.5, 1.0, k) * a_max * np.linalg.norm(a, axis=1)
        u_nom = rng.uniform(-3.0, 3.0, 3) * a_max
        weight = rng.uniform(0.2, 5.0, 3)
        names = tuple(f"r{i}" for i in range(k))
        si = project(u_nom, Rows(a, b, names), ActuatorSet(a_max, layout), weight)
        scaled = project(
            scaling.accel(u_nom),
            Rows(a, scaling.accel(b), names),
            ActuatorSet(float(scaling.accel(a_max)), layout),
            weight,
        )
        statuses.add(si.status)
        assert scaled.status == si.status
        assert (scaled.active, scaled.conflicting) == (si.active, si.conflicting)
        np.testing.assert_allclose(scaling.accel_si(scaled.u), si.u, rtol=0, atol=1e-12 * a_max)
        # The objective scales as u^2 and each row as u, so multipliers scale as u.
        np.testing.assert_allclose(
            scaling.accel_si(scaled.multipliers),
            si.multipliers,
            rtol=1e-12,
            atol=1e-12 * np.max(si.multipliers, initial=0.0),
        )
    assert statuses == {"feasible", "infeasible_certified"}


def test_invalid_rows_and_actuators_are_refused():
    with pytest.raises(ValueError, match="shape"):
        Rows(np.zeros((2, 2)), np.zeros(2), ("a", "b"))
    with pytest.raises(ValueError, match="unique"):
        Rows(np.zeros((2, 3)), np.zeros(2), ("a", "a"))
    with pytest.raises(ValueError, match="unique"):
        Rows(np.zeros((1, 3)), np.zeros(1), ("ball",))
    with pytest.raises(ValueError, match="finite"):
        Rows(np.full((1, 3), np.nan), np.zeros(1), ("a",))
    with pytest.raises(ValueError, match="layout"):
        ActuatorSet(1.0, "cone")
    with pytest.raises(ValueError, match="positive"):
        ActuatorSet(0.0, "box")
    with pytest.raises(ValueError, match="u_nom"):
        project(np.zeros(2), Rows(np.zeros((0, 3)), np.zeros(0), ()))


@pytest.mark.parametrize("layout", ["box", "ball"])
@pytest.mark.parametrize("weighted", [False, True])
def test_huge_nominal_command_does_not_loosen_the_rows(layout, weighted):
    a_max = 2e-3
    actuator = ActuatorSet(a_max, layout)
    weight = np.array([1e-3, 1.0, 1e3]) if weighted else None
    rows = Rows(np.array([[0.0, 1.0, 0.0]]), np.array([1e-3]), ("y",))
    result = project(np.array([1e8, 0.0, 0.0]), rows, actuator, weight)
    assert result.status == "feasible"
    assert result.u[1] >= 1e-3 - 1e-10 * a_max
    rng = np.random.default_rng(22 + 2 * weighted + (layout == "ball"))
    for _ in range(100):
        k = rng.integers(1, 6)
        inside = rng.uniform(-0.5, 0.5, 3) * a_max
        a = rng.normal(0.0, 1.0, (k, 3))
        b = a @ inside - rng.uniform(0.0, 0.3, k) * a_max * np.linalg.norm(a, axis=1)
        u_nom = rng.normal(0.0, 1.0, 3) * 10.0 ** rng.uniform(6.0, 12.0)
        result = project(u_nom, named_rows(a, b), actuator, weight)
        assert result.status == "feasible"
        gap = (a @ result.u - b) / np.linalg.norm(a, axis=1)
        assert np.min(gap) >= -1e-10 * a_max
        np.testing.assert_allclose(
            on_actuator(result.u, actuator), result.u, rtol=0, atol=1e-12 * a_max
        )


def corrupt_dual(monkeypatch, marginals_of):
    real = qp.linprog

    def wrong_dual(*args, **kwargs):
        res = real(*args, **kwargs)
        res.ineqlin.marginals = marginals_of(np.asarray(res.ineqlin.marginals))
        return res

    monkeypatch.setattr(qp, "linprog", wrong_dual)


OPPOSITE = Rows(
    np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), np.array([1.0, 0.0]), ("ahead", "behind")
)


def test_dual_with_the_wrong_sign_is_not_a_certificate(monkeypatch):
    corrupt_dual(monkeypatch, lambda m: -m)
    result = project(np.zeros(3), OPPOSITE)
    assert result.status == "solver_failed"
    assert result.certificate is None and result.conflicting == ()


def test_dual_with_nonzero_a_transpose_lam_is_not_a_certificate(monkeypatch):
    # Weight on "ahead" only: b^T lam = 1 > 0 but A^T lam = (1, 0, 0).
    corrupt_dual(monkeypatch, lambda m: np.array([-1.0, 0.0]))
    result = project(np.zeros(3), OPPOSITE)
    assert result.status == "solver_failed"
    assert result.certificate is None and result.conflicting == ()


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_dual_that_the_actuator_set_can_meet_is_not_a_certificate(monkeypatch, layout):
    # Weight on "ahead" only: u_x >= 1 holds at u_x = a_max = 5, so b^T lam = 1 is below the
    # actuator set's support a_max |A^T lam| = 5.
    corrupt_dual(monkeypatch, lambda m: np.array([-1.0, 0.0]))
    result = project(np.zeros(3), OPPOSITE, ActuatorSet(5.0, layout))
    assert result.status == "solver_failed"
    assert result.certificate is None and result.conflicting == ()


def test_row_a_rounding_error_off_a_box_face_is_solved_not_singular():
    # A speed row -vhat near rest with a 2e-16 m/s rounding residue across the motion: its
    # normal sits 2.2e-12 off the -y face, so the face and the row are numerically one row.
    rows = Rows(np.array([[0.0, -1.0, 2.220446049250313e-12]]), np.array([0.01]), ("speed",))
    cases = [([0.0, 0.0, 0.0], [0.0, -0.01, 0.0]), ([0.0, -0.0625, -0.0625], [0.0, -0.02, -0.02])]
    for u_nom, u_star in cases:
        found = project(np.array(u_nom), rows, ActuatorSet(0.02, "box"))
        assert found.status == "feasible"
        assert rows.a[0] @ found.u >= 0.01 - 1e-12
        np.testing.assert_allclose(found.u, u_star, atol=1e-12)
