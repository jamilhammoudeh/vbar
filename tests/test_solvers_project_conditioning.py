import itertools
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import linprog

from vbar.solvers import ActuatorSet, Rows, project

# Face rows in the documented order: "box+x" is -u_x >= -a_max, "box-x" is u_x >= -a_max.
FACE_NORMALS = np.array([s * e for e in np.eye(3) for s in (-1.0, 1.0)])


def named_rows(a, b):
    return Rows(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        tuple(f"r{i}" for i in range(len(b))),
    )


def actuator_of(layout, a_max):
    return None if layout == "none" else ActuatorSet(a_max, layout)


def assert_certificate_proves_infeasibility(projection, rows, actuator):
    lam = projection.certificate
    assert lam is not None and np.all(lam >= 0.0)
    if actuator is not None and actuator.layout == "box":
        a = np.vstack([rows.a, FACE_NORMALS])
        b = np.append(rows.b, np.full(6, -actuator.a_max))
    else:
        a, b = rows.a, rows.b
    assert lam.shape == (len(b),)
    if actuator is not None and actuator.layout == "ball":
        assert b @ lam > actuator.a_max * np.linalg.norm(a.T @ lam)
    else:
        weight = np.sum(lam * np.linalg.norm(a, axis=1))
        assert np.max(np.abs(a.T @ lam)) <= 1e-9 * weight
        assert b @ lam > 1e-9 * weight * np.max(np.abs(b))


def holds_exactly(a, b, u):
    """Whether a_i . u >= b_i for every row, in exact rational arithmetic."""
    return all(
        sum(Fraction(float(a_ij)) * Fraction(float(u_j)) for a_ij, u_j in zip(row, u, strict=True))
        >= Fraction(float(b_i))
        for row, b_i in zip(a, b, strict=True)
    )


def _solve_exact(m, rhs):
    n = len(m)
    m = [row[:] + [rhs[i]] for i, row in enumerate(m)]
    for c in range(n):
        p = next((r for r in range(c, n) if m[r][c] != 0), None)
        if p is None:
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c], strict=True)]
    return [m[i][n] / m[i][i] for i in range(n)]


def exact_kkt_optimum(a, b, u_nom, w):
    """The QP optimum from the KKT conditions checked in exact rational arithmetic.

    The QP is strictly convex, so the one active set whose KKT point satisfies every row with
    nonnegative multipliers gives the global optimum (Boyd and Vandenberghe, sec. 5.5.3).
    """
    a = [[Fraction(float(x)) for x in row] for row in a]
    b = [Fraction(float(x)) for x in b]
    u_nom = [Fraction(float(x)) for x in u_nom]
    w = [Fraction(float(x)) for x in w]
    for size in range(4):
        for subset in itertools.combinations(range(len(b)), size):
            n = 3 + size
            m = [[Fraction(0)] * n for _ in range(n)]
            rhs = [Fraction(0)] * n
            for i in range(3):
                m[i][i] = w[i]
                rhs[i] = w[i] * u_nom[i]
                for j, row in enumerate(subset):
                    m[i][3 + j] = -a[row][i]
                    m[3 + j][i] = a[row][i]
            for j, row in enumerate(subset):
                rhs[3 + j] = b[row]
            x = _solve_exact(m, rhs)
            if x is None:
                continue
            u, lam = x[:3], x[3:]
            primal = all(
                sum(r[i] * u[i] for i in range(3)) >= b_i for r, b_i in zip(a, b, strict=True)
            )
            if primal and all(lam_j >= 0 for lam_j in lam):
                objective = sum(w[i] * (u[i] - u_nom[i]) ** 2 for i in range(3)) / 2
                return [float(x) for x in u], float(objective)
    raise AssertionError("no active set satisfies the KKT conditions")


@pytest.mark.parametrize("y_nom", [1e6, 1e9, 1e10, 3e10, 1e11, 1e12])
def test_huge_nominal_command_without_a_limit_does_not_loosen_a_row_across_it(y_nom):
    # x >= 1 with u_nom = (0, Y, 0): the optimum is (1, Y, 0) for every Y, and a_x . u has no
    # rounding at all, so nothing but the tolerance can let u_x below 1.
    rows = Rows(np.array([[1.0, 0.0, 0.0]]), np.array([1.0]), ("x_min",))
    result = project(np.array([0.0, y_nom, 0.0]), rows)
    assert result.status == "feasible"
    assert result.u[0] >= 1.0 - 1e-10
    np.testing.assert_allclose(result.u, [1.0, y_nom, 0.0], rtol=1e-15, atol=1e-10)


@pytest.mark.parametrize("layout", ["none", "box", "ball"])
@pytest.mark.parametrize("y_nom", [1e6, 1e10, 1e11, 1e12])
def test_conflicting_rows_stay_certified_under_a_huge_nominal_command(layout, y_nom):
    # x >= 1 and x <= 0.99 conflict for every command; lam = (1, 1) proves it exactly.
    rows = Rows(np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), np.array([1.0, -0.99]), ("lo", "hi"))
    actuator = actuator_of(layout, 5.0)
    result = project(np.array([0.0, y_nom, 0.0]), rows, actuator)
    assert result.status == "infeasible_certified"
    assert {"lo", "hi"} <= set(result.conflicting)
    assert_certificate_proves_infeasibility(result, rows, actuator)


NEARLY_PARALLEL_FEASIBLE = [
    # x >= 1 and -x + eps y >= -0.5, met at x = 1, y = 0.5 / eps.
    *[([[1.0, 0.0, 0.0], [-1.0, eps, 0.0]], [1.0, -0.5]) for eps in (1e-9, 1e-11, 1e-13, 1e-15)],
    # The wedge 1 <= x <= 1 + d (y - Y0), met for y >= Y0 = 1000.
    ([[1.0, 0.0, 0.0], [-1.0, 1e-9, 0.0]], [1.0, -1.0 + 1e-9 * 1e3]),
]


@pytest.mark.parametrize("a, b", NEARLY_PARALLEL_FEASIBLE)
def test_rows_that_can_be_met_are_never_certified_infeasible(a, b):
    a, b = np.array(a), np.array(b)
    # The LP on the raw rows is refused by HiGHS's absolute tolerance; with t = a_2y y as the
    # second variable it is well scaled, and the point it finds is checked exactly.
    a_2y = a[1, 1]
    lp = linprog(
        np.zeros(3),
        A_ub=-(a / np.array([1.0, a_2y, 1.0])),
        b_ub=-(b + 1e-6),
        bounds=[(None, None)] * 3,
        method="highs",
    )
    assert lp.status == 0
    witness = lp.x / np.array([1.0, a_2y, 1.0])
    assert holds_exactly(a, b, witness)
    result = project(np.zeros(3), named_rows(a, b))
    assert result.status != "infeasible_certified"
    assert result.certificate is None


# A sweep problem whose active rows r2 and r6 are 1.4e-7 off parallel (smallest singular value
# of their unit normals) and whose inactive rows r3 and r4 are 5.8e-12 off parallel.
PARALLEL_RECORD = dict(
    a=[
        [88916.51605398221, -44404.86807819112, 79317.3556458124],
        [5356228.589042437, 1373911.8249632288, -10184917.042241931],
        [-349628.1233956301, -18649.47214698447, -818225.7587552713],
        [1.9714330206728368e-07, 7.618421696779627e-08, 1.2932262003985228e-08],
        [0.015098954850032549, 0.005834852314105602, 0.000990465504459906],
        [9186.3118706671, 3549.96577078127, 602.6063918725831],
        [14200.311761714047, 757.450813005909, 33232.61832050159],
    ],
    b=[
        10.534972681197646,
        1253.1230962114867,
        -36.578465319668275,
        3.253243039253542e-11,
        2.645900729169667e-06,
        1.6097848813966473,
        1.8892126839052696,
    ],
    u_nom=[7.180638696914962e-07, 2.5423000009407806e-07, -7.612948083495233e-07],
    w=[57.45583999198223, 3648.1159134553554, 1.020564773289755],
)


def assert_rows_hold_to_the_documented_tolerance(a, b, u, scale):
    gap = (a @ u - b) / np.linalg.norm(a, axis=1)
    assert np.min(gap) >= -1e-10 * scale


def test_nearly_parallel_active_rows_give_the_exact_kkt_optimum():
    a, b = np.array(PARALLEL_RECORD["a"]), np.array(PARALLEL_RECORD["b"])
    u_nom, w = np.array(PARALLEL_RECORD["u_nom"]), np.array(PARALLEL_RECORD["w"])
    _, f_star = exact_kkt_optimum(a, b, u_nom, w)
    result = project(u_nom, named_rows(a, b), weight=w)
    assert result.status == "feasible"
    scale = np.max(np.abs(b / np.linalg.norm(a, axis=1)))
    assert_rows_hold_to_the_documented_tolerance(a, b, result.u, scale)
    # The multipliers (6.4e10 on each unit row) times the row tolerance are worth 2e-3, 6e-9 of
    # f*, so 1e-5 leaves room for rounding only.
    assert result.objective <= f_star * (1.0 + 1e-5)


@pytest.mark.parametrize(
    "y0, d", [(10.0, 1e-9), (10.0, 1e-10), (1e3, 1e-9), (1e3, 1e-10), (1e3, 1e-11)]
)
def test_narrow_wedge_projects_to_its_apex(y0, d):
    # 1 <= x <= 1 + d (y - Y0) with u_nom = (1, 0, 0): every feasible point has y >= Y0, so
    # the optimum is the apex (1, y_apex, 0) with y_apex = (b_2 + 1) / d of the stored rows.
    # The unit normals' smallest singular value is about d / 2, above the 1e-12 rank threshold.
    a = np.array([[1.0, 0.0, 0.0], [-1.0, d, 0.0]])
    b = np.array([1.0, -1.0 + d * y0])
    y_apex = (Fraction(float(b[1])) + 1) / Fraction(d)
    f_star = float(y_apex**2 / 2)
    result = project(np.array([1.0, 0.0, 0.0]), named_rows(a, b))
    assert result.status == "feasible"
    assert_rows_hold_to_the_documented_tolerance(a, b, result.u, 1.0)
    assert result.objective <= f_star * (1.0 + 1e-6)


def unit_problems(s):
    """Rows, nominal command and limit in command units of size s; the conflict is 1% of s."""
    conflict = Rows(
        np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), np.array([0.6 * s, -0.59 * s]), ("lo", "hi")
    )
    reach = Rows(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([1.01 * s, -s]), ("reach", "free")
    )
    return [
        (conflict, np.array([0.1, 0.2, -0.3]) * s, "none"),
        (conflict, np.array([0.1, 0.2, -0.3]) * s, "box"),
        (conflict, np.array([0.1, 0.2, -0.3]) * s, "ball"),
        (reach, np.zeros(3), "box"),
        (reach, np.zeros(3), "ball"),
    ]


@pytest.mark.parametrize("s", [1e3, 1e-3, 1e-5, 1e-6, 1e-8, 1e-12])
@pytest.mark.parametrize("case", range(5))
def test_infeasibility_certificate_survives_a_change_of_command_units(s, case):
    rows_1, u_nom_1, layout = unit_problems(1.0)[case]
    rows_s, u_nom_s, _ = unit_problems(s)[case]
    actuator_1, actuator_s = actuator_of(layout, 1.0), actuator_of(layout, s)
    reference = project(u_nom_1, rows_1, actuator_1)
    assert reference.status == "infeasible_certified"
    result = project(u_nom_s, rows_s, actuator_s)
    assert result.status == reference.status
    assert result.conflicting == reference.conflicting
    assert_certificate_proves_infeasibility(result, rows_s, actuator_s)
    # The rows' normals do not change with s, so the certificate's direction does not either.
    np.testing.assert_allclose(
        result.certificate / np.sum(result.certificate),
        reference.certificate / np.sum(reference.certificate),
        rtol=0,
        atol=1e-9,
    )
