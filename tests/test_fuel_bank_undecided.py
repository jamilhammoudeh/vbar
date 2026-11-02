"""A window LP that HiGHS ends without a verdict is retried, and a start still undecided after the
retries leaves the bank uncertified while the other starts keep their bounds.

HiGHS reports model status 15 (unknown, primal feasible) on some long windows, which scipy's
`linprog` returns as status 4. A bound from any multipliers is valid (Boyd and Vandenberghe, Convex
Optimization, 2004, section 5.1.3), so a retried solve gives a valid bound like the first one.
"""

import math
import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from vbar.agents.base import AgentView
from vbar.dynamics import mean_motion
from vbar.dynamics.cw import R_EARTH
from vbar.oracles import DualBank, FuelOracle, OracleConfig, fuel
from vbar.scenarios import make
from vbar.scenarios.seeds import SeedSplit

VBAR_BOX = "vbar:approach-vbar:cw:box:dv:v1"
# A stride past the 301 windows of approach-vbar solves only windows 1 and 301, the last.
ENDS_ONLY = OracleConfig(bank_stride=1000)


def small_view():
    """Box thrusters, 30 steps of 20 s at 500 km, a 1 m and 0.01 m/s ball at the origin."""
    return AgentView(
        scenario_id="oracle-test",
        dt=20.0,
        max_steps=30,
        n=mean_motion(R_EARTH + 500e3),
        action_mode="accel",
        max_accel=0.02,
        thrust_limit="box",
        goal={"hold_point": (0.0, 0.0, 0.0), "capture_radius": 1.0, "capture_speed": 0.01},
    )


def starts(count):
    rng = np.random.default_rng(7)
    return np.c_[rng.uniform(-200.0, 200.0, (count, 3)), rng.uniform(-0.2, 0.2, (count, 3))]


def test_approach_vbar_validation_start_19_gets_a_bound_from_the_retry():
    # With the HiGHS that scipy 1.18.1 bundles, dual simplex ends window 301 of this start at
    # model status 15 with and without presolve, and the interior point method solves it. On a
    # platform where the first solve succeeds this is a plain regression check.
    env = make(VBAR_BOX)
    start = env.reset(seed=SeedSplit(VBAR_BOX, "validation", 100).seed(19))[1]["true_state"]
    bank = DualBank.build(env.agent_view(), [start], config=ENDS_ONLY)
    assert bank.uncertified == ()
    lows = bank.window_lowers(start)
    assert math.isfinite(lows[300]) and lows[300] > 0.0
    assert bank.bounds(start).lower > 0.0


def undecided_linprog(monkeypatch, decided_by=(), after=0):
    """Makes every HiGHS solve for the start set as `poisoned["row"]`, after its first `after`,
    end at status 4 unless its method is in `decided_by`; returns `poisoned` and the methods
    tried for that start."""
    solve = fuel.linprog
    poisoned = {"on": False, "row": None}
    tried = []
    check = fuel._start

    def start(row):
        poisoned["on"] = poisoned["row"] is not None and bool(np.array_equal(row, poisoned["row"]))
        return check(row)

    def linprog(*args, method, **options):
        res = solve(*args, method=method, **options)
        if not poisoned["on"]:
            return res
        tried.append(method)
        if len(tried) <= after or method in decided_by:
            return res
        return OptimizeResult({**res, "status": 4, "message": "HiGHS Status 15: Unknown"})

    monkeypatch.setattr(fuel, "_start", start)
    monkeypatch.setattr(fuel, "linprog", linprog)
    return poisoned, tried


def test_a_start_left_undecided_is_uncertified_and_the_other_starts_keep_their_bounds(
    monkeypatch,
):
    view = small_view()
    points = starts(3)
    reference = DualBank.build(view, points[[0, 2]])
    # Window 1 of the second start is infeasible with and without the limit (two LPs), window
    # 11 adds multipliers and window 21 is left undecided.
    poisoned, tried = undecided_linprog(monkeypatch, after=3)
    poisoned["row"] = points[1]
    bank = DualBank.build(view, points)
    assert bank.uncertified == (1,)
    assert tried == ["highs"] * 4 + ["highs-ipm", "highs-ds"]
    # The undecided start kept none of its multipliers, so the bank is the bank of the other two.
    for point in points:
        assert np.array_equal(bank.window_lowers(point), reference.window_lowers(point))
        assert bank.bounds(point) == reference.bounds(point)
    assert bank.bounds(points[0]).lower > 0.0


def test_a_status_the_retry_decides_leaves_every_start_certified(monkeypatch):
    view = small_view()
    points = starts(2)
    poisoned, tried = undecided_linprog(monkeypatch, decided_by=("highs-ipm",))
    poisoned["row"] = points[0]
    bank = DualBank.build(view, points)
    assert bank.uncertified == ()
    assert "highs-ds" not in tried and tried.count("highs-ipm") == tried.count("highs") > 0
    assert bank.bounds(points[0]).lower > 0.0


def test_a_fixed_window_left_undecided_raises(monkeypatch):
    view = small_view()
    point = starts(1)[0]
    poisoned, _ = undecided_linprog(monkeypatch)
    poisoned["row"] = point
    with pytest.raises(RuntimeError, match="HiGHS failed"):
        FuelOracle().solve(view, point, arrival_step=20, hard=frozenset())


def no_window_feasible(monkeypatch, row):
    """Makes every window LP of the start `row` infeasible, with and without the limit."""
    real_window, real_outer = fuel._window, fuel._outer
    from_row = {}

    def window(view, x0, *args):
        win = real_window(view, x0, *args)
        from_row[id(win)] = np.array_equal(x0, row)
        return win

    def outer(win, norm, config):
        lp, optimum, cuts, verdict = real_outer(win, norm, config)
        if from_row[id(win)]:
            return None, math.inf, cuts, fuel.INFEASIBLE_WITHOUT_LIMIT
        return lp, optimum, cuts, verdict

    monkeypatch.setattr(fuel, "_window", window)
    monkeypatch.setattr(fuel, "_outer", outer)


def test_a_start_with_no_feasible_window_is_listed_as_infeasible(monkeypatch):
    view = small_view()
    points = starts(3)
    reference = DualBank.build(view, points[[0, 2]])
    no_window_feasible(monkeypatch, points[1])
    bank = DualBank.build(view, points)
    assert bank.infeasible == {1: "no_feasible_window"}
    assert bank.uncertified == ()
    # The other starts' multipliers are those of a bank without it.
    assert np.array_equal(bank.window_lowers(points[0]), reference.window_lowers(points[0]))


def test_a_free_final_time_with_no_feasible_window_reports_it_on_the_bounds(monkeypatch):
    view = small_view()
    point = starts(1)[0]
    no_window_feasible(monkeypatch, point)
    bounds = FuelOracle().bounds(view, point)
    assert bounds.infeasible == "no_feasible_window" and bounds.lower == 0.0


def test_a_bank_start_with_no_feasible_window_scores_uncertified_through_its_bounds(monkeypatch):
    # The pattern of benchmarks/lower_bound_check.py: build the bank from the true starts, then
    # score each run with bank.bounds(ep.start).
    from vbar.agents import DriftAgent
    from vbar.metrics import optimality_gap, record_episode
    from vbar.metrics.mission import success

    env = make(VBAR_BOX)
    ep = record_episode(env, DriftAgent(), seed=3, agent_seed=4)
    other = starts(1)[0]
    no_window_feasible(monkeypatch, np.asarray(ep.start, dtype=float))
    bank = DualBank.build(ep.view, np.array([ep.start, other]), config=ENDS_ONLY)
    assert bank.infeasible == {0: "no_feasible_window"}
    assert bank.bounds(other).infeasible == ""
    bounds = bank.bounds(ep.start)
    assert bounds.infeasible == "no_feasible_window"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gap = optimality_gap(ep, bounds)
    assert not gap["bound_certified"] and gap["bound_reason"] == "no_feasible_window"
    assert not any(isinstance(v, float) and math.isinf(v) for v in gap.values())
    assert gap["bound_solver_error"] == bool(success(ep))


def test_a_start_equal_but_for_the_sign_of_a_zero_matches_its_bank_row(monkeypatch):
    view = small_view()
    row = starts(1)[0]
    row[2] = 0.0
    no_window_feasible(monkeypatch, row)
    bank = DualBank.build(view, row[None, :])
    assert bank.infeasible == {0: "no_feasible_window"}
    signed = row.copy()
    signed[2] = -0.0
    assert bank.bounds(row).infeasible == bank.bounds(signed).infeasible == "no_feasible_window"
