"""Finite-horizon and soft modes of the predictive safety filter.

The interface these tests assume, beyond that of `tests/test_safety_predictive.py`:

- `PredictiveFilter(*, mode="terminal", drift_horizon=None, alpha=None, beta=None, ...)` keeps
  `mode` as an attribute; `mode` is "terminal", "finite-horizon" or "soft", and any other value
  raises ValueError. The names are "predictive-n{N}", "predictive-fh-n{N}" and
  "predictive-soft-n{N}". Both new modes declare `counts_toward_certification` False, so
  `audit_filter` refuses them; the terminal mode still counts.
- Finite-horizon mode declares one barrier,
  `PassiveBarrier(radius=R, n=n, grid=dt / ceil(dt / 5), t_cap=drift_horizon, terminal=False)`,
  and `contains(x)` is that barrier's value >= 0. A certified decision reports certificate
  "finite-horizon" with `certificate_horizon` equal to drift_horizon. Its terminal rows `C_N` are
  at most the one gradient row; the far-part rows of the terminal mode are left out.
- Finite-horizon mode reports `recovering` True for every decision at a state outside its
  declared set (`contains(x)` False), a certified plan included, since free drift does not keep
  that set. The other modes report it when nothing is certified, which happens only outside
  the set.
- Soft mode passes `solve_horizon` a `StageQP` whose inputs are the Hill-frame acceleration
  (m/s^2) followed by at least one nonnegative slack (m) that no dynamics column reads; the
  linear slack weights alpha and beta sit in the stage gradients `q[:, nx + 3:]`. A certified
  decision is "nominal" with horizon inf. When nothing is certified, braking comes first in its
  region; otherwise the soft solution is applied as "feasible" with its predicted shortfall (m)
  as `slack` and no certificate. Soft mode never reports
  "infeasible_certified" or "solver_failed".

Known answers, recomputed here at 500 km, R 50 m, dt 10 s, drift grid 5 s and drift_horizon
20 pi / n: at the G-1135 state x = (5, 521, 0, 0, -7.5 n, 0) the capped free-drift margin is
+0.0099 m (G-1135 states +0.0147 m) and one drift step later it is -0.0727 m (G-1135 states a
0.068 m shortfall); the terminal margin at x is -1 mm, the value for a drift that reaches no
terminal state.
"""

import dataclasses
import math

import numpy as np
import pytest

from vbar.agents.base import action_scale
from vbar.constraints import KeepOut
from vbar.dynamics import cw_discrete
from vbar.safety import predictive as predictive_module
from vbar.safety.audit import audit_filter
from vbar.safety.barriers import PassiveBarrier
from vbar.safety.cbf import BackupFilter
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.solvers.ipm import HorizonSolution

PredictiveFilter = predictive_module.PredictiveFilter

R = 50.0
A_MAX = 0.02
DT = 10.0
N = ApproachEnv().n
PHI, GAMMA = cw_discrete(N, DT)
DEPTH = KeepOut.resolution
TEN_ORBITS = 20.0 * math.pi / N
TWO_ORBITS = 4.0 * math.pi / N
FAR = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])
V_BAR = np.array([0.0, 150.0, 0.0, 0.0, 0.0, 0.0])
# At rest 70 m out on V-bar. The reckless command, full thrust toward the target, clears the
# keep-out by 19.0 m over its held step, but the drift after it enters by 25.6 m, so the
# program runs; zero thrust is certified with +20.0 m, and the zero-thrust reference ends in the
# far part of the terminal set.
REST = np.array([0.0, 70.0, 0.0, 0.0, 0.0, 0.0])
# test_safety_predictive.py: one step before the flyby X1, which passes the target at 1.5 m/s.
X1 = np.array([60.0, 78.935, 0.0, 0.0, -1.5, 0.0])
X0 = np.linalg.solve(PHI, X1 - GAMMA @ np.array([0.002, 0.0, 0.001]))
NEAR_BOX = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))
# G-1135: x_c = 4 x + 2 vy / n = 5 m and no ellipse, so the drift runs down V-bar at 7.5 n m/s
# and reaches the keep-out just after ten orbits.
G1135 = np.array([5.0, 521.0, 0.0, 0.0, -7.5 * N, 0.0])
# 5 m outside the keep-out closing at 1 m/s: full braking needs 25 m, so every plan enters.
UNAVOIDABLE = np.array([0.0, -55.0, 0.0, 0.0, 1.0, 0.0])
ZERO = np.zeros(3)


def approach(dt=DT, steps=1, start=FAR, **cfg):
    start = tuple(float(s) for s in start)
    return ApproachEnv(ApproachConfig(dt=dt, max_steps=steps, start_state=start, **cfg))


def ready(filt, dt=DT, **cfg):
    filt.reset(approach(dt, **cfg).agent_view())
    return filt


def finite_horizon(drift_horizon=TEN_ORBITS):
    return ready(PredictiveFilter(mode="finite-horizon", drift_horizon=drift_horizon))


def toward_target(x):
    """The reckless command: 1 m/s^2 toward the target in normalized units, far past the box."""
    view = approach().agent_view()
    return -x[:3] / np.linalg.norm(x[:3]) * action_scale(view) / view.max_accel


def in_set_states(count):
    backup = ready(BackupFilter(barrier="passive"))
    xs = np.random.default_rng(1104).uniform(*NEAR_BOX, (2 * count, 6))
    inside = [x for x in xs if backup.contains(x)]
    assert len(inside) >= count
    return inside[:count]


def spy(seen):
    """The real solver, recording each program it is given."""
    solve = predictive_module.solve_horizon

    def record(qp, *args, **kwargs):
        seen.append(qp)
        return solve(qp, *args, **kwargs)

    return record


def stopped(status):
    """A solver that stops with `status` and no plan."""

    def solve(qp, *args, **kwargs):
        rows = (0 if qp.C is None else qp.C.shape[0] * qp.C.shape[1]) + (
            0 if qp.C_N is None else qp.C_N.shape[0]
        )
        if status != "infeasible_certified":
            return HorizonSolution(None, None, status, 7, math.nan, 1.0, 1.0, 1.0, None)
        return HorizonSolution(
            None,
            None,
            status,
            7,
            math.nan,
            1.0,
            1.0,
            1.0,
            None,
            conflicting=((0, 0),),
            certificate=np.ones(rows),
        )

    return solve


# Construction and declaration.


@pytest.mark.parametrize(
    ("kw", "name"),
    [
        ({}, "predictive-n10"),
        ({"mode": "terminal"}, "predictive-n10"),
        ({"mode": "finite-horizon", "drift_horizon": TEN_ORBITS}, "predictive-fh-n10"),
        ({"mode": "soft"}, "predictive-soft-n10"),
        ({"mode": "soft", "horizon": 4}, "predictive-soft-n4"),
    ],
)
def test_the_name_states_the_mode_and_the_horizon(kw, name):
    assert PredictiveFilter(**kw).name == name


def test_the_terminal_mode_is_the_default():
    assert PredictiveFilter().mode == "terminal"


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        PredictiveFilter(mode="explicit")


@pytest.mark.parametrize(
    "kw",
    [{"mode": "finite-horizon", "drift_horizon": TEN_ORBITS}, {"mode": "soft"}],
    ids=["finite-horizon", "soft"],
)
def test_both_new_modes_declare_they_do_not_count_toward_certification(kw):
    assert ready(PredictiveFilter()).declaration.counts_toward_certification is True
    assert ready(PredictiveFilter(**kw)).declaration.counts_toward_certification is False


@pytest.mark.parametrize(
    "kw",
    [{"mode": "finite-horizon", "drift_horizon": TEN_ORBITS}, {"mode": "soft"}],
    ids=["finite-horizon", "soft"],
)
def test_the_audit_refuses_both_new_modes(kw):
    with pytest.raises(ValueError, match="does not count toward certification"):
        audit_filter(PredictiveFilter(**kw), ApproachEnv(), starts=1, max_steps=1)


# Finite-horizon mode.


@pytest.mark.parametrize("drift_horizon", [TEN_ORBITS, TWO_ORBITS], ids=["ten", "two"])
def test_finite_horizon_set_is_the_free_drift_set_capped_at_the_drift_horizon(drift_horizon):
    filt = finite_horizon(drift_horizon)
    expected = PassiveBarrier(radius=R, n=N, grid=5.0, t_cap=drift_horizon, terminal=False)
    assert filt.declaration.barriers == (expected,)
    result = filt.filter(FAR.copy(), ZERO.copy())
    assert result.status == "feasible" and result.recovering is False
    assert result.certificate == "finite-horizon"
    assert result.certificate_horizon == pytest.approx(drift_horizon, rel=1e-12)


def test_g1135_state_is_certified_for_the_drift_horizon_and_one_drift_step_later_is_not():
    filt = finite_horizon()
    (barrier,) = filt.declaration.barriers
    assert barrier.value(G1135) == pytest.approx(0.0099, abs=1e-4)
    first = filt.filter(G1135.copy(), ZERO.copy())
    assert first.status == "feasible" and first.fallback_level == 0
    assert first.recovering is False
    assert first.certificate == "finite-horizon"
    assert first.certificate_horizon == pytest.approx(TEN_ORBITS, rel=1e-12)
    np.testing.assert_array_equal(first.action, ZERO)
    # The set is not invariant under drift: the capped window slides onto the later entry.
    x1 = PHI @ G1135
    assert barrier.value(x1) == pytest.approx(-0.0727, abs=1e-4)
    assert filt.contains(x1) is False
    assert filt.filter(x1, ZERO.copy()).recovering is True


def test_terminal_mode_at_the_g1135_state_is_recovering():
    filt = ready(PredictiveFilter())
    assert filt.declaration.barriers[0].value(G1135) == pytest.approx(-DEPTH, abs=1e-12)
    assert filt.contains(G1135) is False
    assert filt.filter(G1135.copy(), ZERO.copy()).recovering is True


def test_finite_horizon_terminal_rows_are_the_gradient_row_only(monkeypatch):
    programs = {}
    for mode, filt in (("terminal", ready(PredictiveFilter())), ("fh", finite_horizon())):
        seen = []
        monkeypatch.setattr(predictive_module, "solve_horizon", spy(seen))
        filt.filter(REST.copy(), toward_target(REST))
        assert seen
        programs[mode] = seen[0]
    assert programs["terminal"].C_N.shape[0] == 5
    c_n = programs["fh"].C_N
    assert c_n is None or c_n.shape[0] <= 1


# Soft mode.


def soft_and_terminal_first_inputs(states, soft_solver=None, monkeypatch=None):
    """First inputs (normalized) of the terminal and soft modes, each fresh, on `states`."""
    terminal_u, soft_u, intervened, soft_results = [], [], 0, []
    for x in states:
        action = toward_target(x)
        hard = ready(PredictiveFilter()).filter(x.copy(), action.copy())
        if soft_solver is not None:
            monkeypatch.setattr(predictive_module, "solve_horizon", soft_solver)
        soft = ready(PredictiveFilter(mode="soft")).filter(x.copy(), action.copy())
        if soft_solver is not None:
            monkeypatch.undo()
        terminal_u.append(np.asarray(hard.action, dtype=float))
        soft_u.append(np.asarray(soft.action, dtype=float))
        intervened += bool(hard.intervened)
        soft_results.append(soft)
    return np.array(terminal_u), np.array(soft_u), intervened, soft_results


EXACT_STATES = 20


def test_soft_first_input_equals_the_terminal_first_input_on_in_set_states():
    terminal_u, soft_u, intervened, results = soft_and_terminal_first_inputs(
        in_set_states(EXACT_STATES)
    )
    # The rows bind on these states, so a penalty that is not exact would show.
    assert intervened >= 5
    # Normalized units: 1e-6 of a_max.
    assert np.max(np.abs(soft_u - terminal_u)) <= 1e-6
    assert all(r.status == "feasible" and r.recovering is False for r in results)
    assert all(r.certificate == "nominal" for r in results)
    assert all(r.certificate_horizon == math.inf for r in results)


def quadratic_penalty(solutions):
    """The real solver on the soft program with each penalty alpha s turned into alpha s^2."""
    solve = predictive_module.solve_horizon

    def mutant(qp, *args, **kwargs):
        nx, nu = qp.B.shape[1], qp.B.shape[2]
        assert nu > 3
        slack = np.arange(nx + 3, nx + nu)
        Q, q = np.array(qp.Q), np.array(qp.q)
        Q[:, slack, slack] += 2.0 * np.abs(q[:, slack])
        q[:, slack] = 0.0
        solution = solve(dataclasses.replace(qp, Q=Q, q=q), *args, **kwargs)
        solutions.append(solution)
        return solution

    return mutant


def hard_solver(multipliers):
    """The real solver on the terminal program, recording the stage-0 keep-out multipliers."""
    solve = predictive_module.solve_horizon

    def record(qp, *args, **kwargs):
        solution = solve(qp, *args, **kwargs)
        stage = qp.C.shape[0] * qp.C.shape[1]
        lam = np.asarray(solution.multipliers)[:stage].reshape(qp.C.shape[:2])
        # Each stage's first six rows are the input box; the rest are its keep-out rows, which
        # one slack relaxes together.
        multipliers.append(float(lam[0, 6:].sum()))
        return solution

    return record


# The reckless flyby from X0: its sixth decision binds a stage-0 keep-out row with a hard
# multiplier of 0.172, so the quadratic penalty alpha s^2 with alpha = 1 leaves a slack of up to
# lambda / (2 alpha) = 0.086 m there.
FLYBY_DECISIONS = 6


def test_exact_penalty_keeps_the_hard_input_where_a_quadratic_penalty_moves_it(monkeypatch):
    terminal = ready(PredictiveFilter())
    soft = ready(PredictiveFilter(mode="soft", alpha=1.0, beta=1.0))
    mutant = ready(PredictiveFilter(mode="soft", alpha=1.0, beta=1.0))
    x, multipliers, solutions = X0.copy(), [], []
    for _ in range(FLYBY_DECISIONS):
        action = toward_target(x)
        multipliers.clear()
        monkeypatch.setattr(predictive_module, "solve_horizon", hard_solver(multipliers))
        hard = terminal.filter(x.copy(), action.copy())
        monkeypatch.undo()
        exact = soft.filter(x.copy(), action.copy())
        solutions.clear()
        monkeypatch.setattr(predictive_module, "solve_horizon", quadratic_penalty(solutions))
        moved = mutant.filter(x.copy(), action.copy())
        monkeypatch.undo()
        np.testing.assert_allclose(exact.action, hard.action, rtol=0.0, atol=1e-6)
        x = PHI @ x + GAMMA @ (np.asarray(hard.action, dtype=float) * A_MAX)
    lam = multipliers[0]
    assert hard.fallback_level == 0 and hard.iterations is not None
    assert lam > 0.1
    # The row's multiplier falls as its slack grows, so 2 alpha s = lambda(s) settles below
    # lambda / (2 alpha); measured 0.018 m.
    s_0 = float(solutions[0].u[0, 3])
    assert 0.01 < s_0 <= lam / 2.0
    assert np.max(np.abs(np.asarray(moved.action) - np.asarray(hard.action))) > 1e-6


def test_soft_program_has_slack_inputs_no_dynamics_column_reads(monkeypatch):
    seen = []
    monkeypatch.setattr(predictive_module, "solve_horizon", spy(seen))
    ready(PredictiveFilter(mode="soft")).filter(REST.copy(), toward_target(REST))
    assert seen
    qp = seen[0]
    assert qp.B.shape[2] > 3
    np.testing.assert_array_equal(qp.B[:, :, 3:], 0.0)
    # Positive weights on the slacks: the program pays for every metre of enlargement.
    assert np.all(qp.q[:, qp.A.shape[1] + 3 :] >= 0.0)
    assert np.any(qp.q[:, qp.A.shape[1] + 3 :] > 0.0)


def test_soft_mode_applies_an_unavoidable_entry_as_feasible_with_its_shortfall():
    filt = ready(PredictiveFilter(mode="soft"))
    assert filt.contains(UNAVOIDABLE) is False
    result = filt.filter(UNAVOIDABLE.copy(), ZERO.copy())
    assert result.status == "feasible" and result.proof is None
    assert result.recovering is True and result.slack > 0.0
    assert result.certificate is None and result.certificate_horizon is None


@pytest.mark.parametrize("status", ["iteration_cap", "solver_failed", "infeasible_certified"])
@pytest.mark.parametrize("start", [UNAVOIDABLE, V_BAR], ids=["unavoidable", "v-bar"])
def test_soft_mode_never_reports_a_solver_status(monkeypatch, status, start):
    filt = ready(PredictiveFilter(mode="soft"))
    monkeypatch.setattr(predictive_module, "solve_horizon", stopped(status))
    result = filt.filter(start.copy(), toward_target(start))
    assert result.status == "feasible" and result.proof is None
    assert result.recovering is (not filt.contains(start))
