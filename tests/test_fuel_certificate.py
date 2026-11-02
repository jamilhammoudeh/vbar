"""The fuel oracle accepts an LP solve only when its own residuals prove it optimal: primal
feasibility, dual feasibility with the multipliers' signs, and a zero duality gap (Boyd and
Vandenberghe, Convex Optimization, 2004, section 5.5.3). The status HiGHS reports is not enough.
"""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, linprog

from vbar.oracles import FuelOracle, fuel
from vbar.scenarios import make
from vbar.scenarios.seeds import SeedSplit

VBAR_BOX = "vbar:approach-vbar:cw:box:dv:v1"

# Boyd and Vandenberghe, Exercise 5.28 (p282): x* = (1, 1, 1, 1), p* = 64, multipliers
# (3, 2, 2, 7, 0); scipy's inequality marginals are their negatives.
BOYD_528 = {
    "c": np.array([47.0, 93.0, 17.0, -93.0]),
    "A_ub": np.array(
        [
            [-1.0, -6.0, 1.0, 3.0],
            [-1.0, -2.0, 7.0, 1.0],
            [0.0, 3.0, -10.0, -1.0],
            [-6.0, -11.0, -2.0, 12.0],
            [1.0, 6.0, -1.0, -3.0],
        ]
    ),
    "b_ub": np.array([-3.0, 5.0, -8.0, -7.0, 4.0]),
    "bounds": (None, None),
}

# min x1 + x2 with x1 + x2 = 1 and x >= 0: p* = 1, y = 1, bound marginals 0 at x = (1, 0).
SPLIT = {
    "c": np.array([1.0, 1.0]),
    "A_eq": np.array([[1.0, 1.0]]),
    "b_eq": np.array([1.0]),
    "bounds": (0.0, None),
}


def claim(x, y, lower, upper=(0.0, 0.0)):
    """A solve of `SPLIT` that reports `x`, equality marginal `y` and bound marginals."""
    return OptimizeResult(
        x=np.asarray(x, dtype=float),
        eqlin=OptimizeResult(marginals=np.atleast_1d(np.asarray(y, dtype=float))),
        lower=OptimizeResult(marginals=np.asarray(lower, dtype=float)),
        upper=OptimizeResult(marginals=np.asarray(upper, dtype=float)),
    )


def free_claim(x, z):
    """A solve of an LP with rows A_ub x <= b_ub and x >= 0 that reports `x` and marginals `z`."""
    zero = OptimizeResult(marginals=np.zeros(len(x)))
    return OptimizeResult(
        x=np.asarray(x, dtype=float),
        ineqlin=OptimizeResult(marginals=np.asarray(z, dtype=float)),
        lower=zero,
        upper=zero,
    )


def test_boyd_exercise_5_28_optimum_and_multipliers_are_certified():
    res = linprog(**BOYD_528, method="highs")
    assert res.fun == pytest.approx(64.0, rel=1e-9)
    assert res.x == pytest.approx(np.ones(4), abs=1e-9)
    assert -res.ineqlin.marginals == pytest.approx([3.0, 2.0, 2.0, 7.0, 0.0], abs=1e-9)
    assert fuel._certified(BOYD_528, res)


def test_perturbed_multipliers_do_not_fit_the_costs():
    res = linprog(**BOYD_528, method="highs")
    tampered = OptimizeResult(
        {**res, "ineqlin": OptimizeResult(marginals=res.ineqlin.marginals * 1.001)}
    )
    assert not fuel._certified(BOYD_528, tampered)


def test_the_exact_optimum_of_the_split_lp_is_certified():
    assert fuel._certified(SPLIT, claim([1.0, 0.0], 1.0, [0.0, 1e-12]))


def test_a_primal_value_below_the_dual_bound_is_rejected():
    # Cost 0.8 under the dual bound 1, which weak duality forbids for a feasible point: the
    # point misses x1 + x2 = 1.
    assert not fuel._certified(SPLIT, claim([0.4, 0.4], 1.0, [0.0, 0.0]))


def test_a_point_outside_its_bounds_is_rejected_with_no_gap():
    # x1 + x2 = 1 and c^T x = 1 = the dual objective, but x2 = -0.5 breaks x >= 0.
    assert not fuel._certified(SPLIT, claim([1.5, -0.5], 1.0, [0.0, 0.0]))


def test_multipliers_off_stationarity_are_rejected_with_no_gap():
    # The dual objective is 1 = c^T x, but c - y - lower = (0, -0.5).
    assert not fuel._certified(SPLIT, claim([1.0, 0.0], 1.0, [0.0, 0.5]))


def test_a_bound_multiplier_of_the_wrong_sign_is_dual_infeasible():
    # c - y - lower = 0 holds exactly, and the dual objective 2 would exceed p* = 1, but a
    # multiplier of x >= 0 must not be negative.
    assert not fuel._certified(SPLIT, claim([1.0, 0.0], 2.0, [-1.0, -1.0]))


def test_wrong_sign_multipliers_are_rejected_with_no_gap_and_no_residual():
    # Each claim is stationary with a dual objective equal to c^T x; only the multiplier signs
    # are wrong: a negative lower marginal, a positive upper marginal at x <= 1, and a positive
    # marginal of x <= 1.
    lower = {**SPLIT, "c": np.array([1.0, 0.0])}
    assert not fuel._certified(lower, claim([1.0, 0.0], 1.0, [0.0, -1.0]))
    upper = {"c": np.array([1.0]), "bounds": (0.0, 1.0)}
    assert not fuel._certified(
        upper,
        OptimizeResult(
            x=np.array([1.0]),
            lower=OptimizeResult(marginals=np.array([0.0])),
            upper=OptimizeResult(marginals=np.array([1.0])),
        ),
    )
    rows = {"c": np.array([1.0]), "A_ub": np.array([[1.0], [-1.0]]), "b_ub": np.array([1.0, 0.0])}
    assert not fuel._certified(rows, free_claim([1.0], [1.0, 0.0]))


def test_points_off_their_rows_are_rejected_with_no_gap():
    # Stationary multipliers and a zero gap, but x = 0 misses x1 + x2 = 1 and breaks -x <= -1.
    assert not fuel._certified(SPLIT, claim([0.0, 0.0], 0.0, [1.0, 1.0]))
    rows = {"c": np.array([0.0]), "A_ub": np.array([[-1.0]]), "b_ub": np.array([-1.0])}
    assert not fuel._certified(rows, free_claim([0.0], [0.0]))


def test_dual_feasible_multipliers_with_a_gap_are_rejected():
    # Dual feasible with objective 0.5 against the primal 1: not complementary.
    assert not fuel._certified(SPLIT, claim([1.0, 0.0], 0.5, [0.5, 0.5]))


def test_a_solve_without_a_point_is_not_certified():
    assert not fuel._certified(SPLIT, OptimizeResult(x=None))


def test_an_optimal_status_that_fails_the_check_is_retried(monkeypatch):
    solve = fuel.linprog
    tried = []

    def linprog_tampered_first(*args, method, **options):
        res = solve(*args, method=method, **options)
        tried.append(method)
        if len(tried) == 1:
            return OptimizeResult({**res, "x": res.x + 0.1})
        return res

    monkeypatch.setattr(fuel, "linprog", linprog_tampered_first)
    res = fuel._highs(True, **SPLIT)
    assert tried == ["highs", "highs-ipm"]
    assert res.status == 0 and fuel._certified(SPLIT, res)


def test_an_optimal_status_no_attempt_certifies_is_undecided(monkeypatch):
    solve = fuel.linprog

    def linprog_tampered(*args, method, **options):
        res = solve(*args, method=method, **options)
        return OptimizeResult({**res, "x": res.x + 0.1})

    monkeypatch.setattr(fuel, "linprog", linprog_tampered)
    res = fuel._highs(True, **SPLIT)
    assert res.status == fuel._NO_VERDICT
    with pytest.raises(fuel._Undecided, match="certificate check"):
        fuel._checked(res.status, res.message, (2,))


def test_approach_vbar_validation_start_19_window_301_optimum_is_certified(monkeypatch):
    # Dual simplex ends this LP at model status 15 with primal and dual objectives 0.544044 and
    # 0.543151 in the LP's units; the interior point method's optimum is 0.5431506395, which is
    # 0.3493889553 m/s at this start's scale. The certificate's gap tolerance is 1e-7, so the
    # value is held to that, 1e4 times finer than the disagreement of the undecided end.
    env = make(VBAR_BOX)
    start = env.reset(seed=SeedSplit(VBAR_BOX, "validation", 100).seed(19))[1]["true_state"]
    checked = []
    certify = fuel._certified

    def recorded(problem, res):
        checked.append(certify(problem, res))
        return checked[-1]

    outer = fuel._outer
    outer_checks = []

    def recorded_outer(*args, **kwargs):
        lp = outer(*args, **kwargs)
        outer_checks.append(len(checked))
        return lp

    monkeypatch.setattr(fuel, "_certified", recorded)
    monkeypatch.setattr(fuel, "_outer", recorded_outer)
    sol = FuelOracle().solve(env.agent_view(), start, arrival_step=301, hard=frozenset())
    # Older HiGHS releases report an optimum here that fails the check; the retry behind
    # finite_optimum must pass it.
    n = outer_checks[0]
    assert n > 0 and checked[n - 1]
    assert sol.finite_optimum == pytest.approx(0.3493889553, rel=1e-7)
    assert 0.0 < sol.impulsive_lower <= sol.finite_lower
