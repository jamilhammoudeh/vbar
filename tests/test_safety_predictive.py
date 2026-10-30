"""Predictive safety filter: certified plans, fallbacks, the terminal set and delay prediction.

The interface these tests assume:

- `PredictiveFilter(*, horizon=10, row_samples=5, eps=1e-3, settings=None, delay_s=0.0, ...)`
  takes keyword arguments only and keeps each as an attribute of the same name; `horizon` is the
  number of steps N, `row_samples` the keep-out row samples per step, and `settings` the
  `IPMSettings` given to the solver.
- `reset(view)` checks the view as `BackupFilter` does, except that the view must carry exactly
  ceil(delay_s / dt) pending command slots. `declaration.barriers` holds one barrier, the
  terminal free-drift barrier that `BackupFilter(barrier="passive")` declares.
- `filter(obs, action)` returns a `FilterResult`. A certified agent command inside the thrust
  limit comes back as the agent's own array, bit for bit, at level 0; a converged and certified
  solution is level 0, the stored plan shifted by one step and re-certified level 1, zero thrust
  level 2. The stored plan is the one applied: the solution, the shifted plan, or the agent's
  command followed by zeros. `iterations` is the solver's count when it ran.
- A decision that solves calls `vbar.safety.predictive.solve_horizon(qp, ...)` with a `StageQP`
  of `horizon` stages whose state is the Hill state (m, m/s) and whose input is the Hill-frame
  acceleration (m/s^2).
- `predict(obs)` returns the Hill state (m, m/s) delay_s after the observation, propagated
  through the commands the observation echoes.
"""

import math

import numpy as np
import pytest

from vbar.agents.base import action_scale
from vbar.constraints import KeepOut
from vbar.dynamics import cw_discrete
from vbar.safety import FilteredAgent
from vbar.safety import predictive as predictive_module
from vbar.safety.barriers import PassiveBarrier
from vbar.safety.cbf import BackupFilter
from vbar.safety.sampled import held_segment_bound
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.solvers.ipm import HorizonSolution, IPMSettings
from vbar.spacecraft import PerfectNav

PredictiveFilter = predictive_module.PredictiveFilter

R = 50.0
A_MAX = 0.02
DT = 10.0
N = ApproachEnv().n
PHI, GAMMA = cw_discrete(N, DT)
# A violation is a certified in-step bound deeper than the keep-out's resolution.
DEPTH = KeepOut.resolution
FAR = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])
V_BAR = np.array([0.0, 150.0, 0.0, 0.0, 0.0, 0.0])
START_BOX = ((-200.0, 200.0, -50.0, -0.1, -0.1, -0.1), (200.0, 600.0, 50.0, 0.1, 0.1, 0.1))
NEAR_BOX = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))
# X1 passes the target at 1.5 m/s, 55.2 m from its centre about 58 s on; X0 is one step before
# it under PLAN[0] and X2 one step after under PLAN[1]. PUSH, full thrust toward the y axis, takes
# each of the three into the keep-out.
X1 = np.array([60.0, 78.935, 0.0, 0.0, -1.5, 0.0])
PLAN = np.array([[0.002, 0.0, 0.001], [0.001, 0.0, -0.001], [0.0005, 0.0, 0.0005]])
X0 = np.linalg.solve(PHI, X1 - GAMMA @ PLAN[0])
X2 = PHI @ X1 + GAMMA @ PLAN[1]
PUSH = np.array([-1.0, 0.0, 0.0])


def approach(dt=DT, steps=1, start=FAR, **cfg):
    start = tuple(float(s) for s in start)
    return ApproachEnv(ApproachConfig(dt=dt, max_steps=steps, start_state=start, **cfg))


def ready(filt, dt=DT, **cfg):
    filt.reset(approach(dt, **cfg).agent_view())
    return filt


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def assert_in_actuator_set(action, layout):
    action = np.asarray(action, dtype=float)
    if layout == "box":
        assert np.all(np.abs(action) <= 1.0)
    else:
        assert np.linalg.norm(action) <= 1.0


class Reckless:
    """Pushes at 1 m/s^2 toward the target, far beyond any thrust limit."""

    name = "reckless"

    def reset(self, view, rng=None):
        self.scale = action_scale(view) / view.max_accel

    def act(self, obs, info):
        return -obs[:3] / np.linalg.norm(obs[:3]) * self.scale


def fly(filt, agent, start, steps, **cfg):
    env = approach(DT, steps, start, sensor=PerfectNav(), **cfg)
    pilot = FilteredAgent(agent, filt)
    obs, _ = env.reset(seed=0)
    pilot.reset(env.agent_view(), np.random.default_rng(0))
    worst, results = math.inf, []
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(pilot.act(obs, env.agent_info()))
        worst = min(worst, env.record.segment.minima["keep_out"].lower_bound)
        results.append(pilot.last_result)
        assert_in_actuator_set(pilot.last_result.action, cfg.get("thrust_limit", "box"))
        if terminated or truncated:
            break
    return worst, results


# Construction and declaration.


def test_defaults_are_ten_steps_five_row_samples_and_eps_1e_3():
    filt = PredictiveFilter()
    assert filt.horizon == 10 and filt.row_samples == 5 and filt.eps == 1e-3
    assert filt.delay_s == 0.0


@pytest.mark.parametrize("delay_s", [-1.0, math.nan, math.inf])
def test_construction_refuses_a_delay_that_is_negative_or_not_finite(delay_s):
    with pytest.raises(ValueError):
        PredictiveFilter(delay_s=delay_s)


def test_declared_set_is_the_terminal_free_drift_set_of_the_backup_filter():
    declared = ready(PredictiveFilter()).declaration
    assert declared.barriers == ready(BackupFilter(barrier="passive")).declaration.barriers
    assert declared.certificate == "nominal"
    assert declared.backup == "zero thrust, else the braking backup outside the set"


@pytest.mark.parametrize("dt", [1.0, 7.0, 10.0, 20.0])
def test_drift_grid_divides_the_step(dt):
    (barrier,) = ready(PredictiveFilter(), dt).declaration.barriers
    assert isinstance(barrier, PassiveBarrier) and barrier.terminal
    k = dt / barrier.grid
    assert abs(k - round(k)) <= 1e-12 * k


def test_free_drift_step_never_lowers_h_p():
    barrier = ready(PredictiveFilter()).declaration.barriers[0]
    rng = np.random.default_rng(1103)
    xs = np.vstack([rng.uniform(*box, (100, 6)) for box in (START_BOX, NEAR_BOX)])
    before = barrier.value(xs)
    assert np.any(before >= 0.0) and np.any(before < 0.0)
    # 1e-9 m: the two sample sets agree to about 1e-12 m of rounding.
    assert np.all(barrier.value(xs @ PHI.T) >= before - 1e-9)


# The agent's command.


@pytest.mark.parametrize(
    ("mode", "action"),
    [
        ("normalized", [0.1, -0.3, 0.05]),
        ("accel", [0.002, -0.006, 0.001]),
        ("thrust", [1.0, -3.0, 0.5]),
    ],
)
def test_certified_agent_command_passes_through_bit_for_bit(mode, action):
    action = np.array(action)
    backup = ready(BackupFilter(barrier="passive"), action_mode=mode)
    assert not backup.filter(FAR.copy(), action.copy()).active
    result = ready(PredictiveFilter(), action_mode=mode).filter(FAR.copy(), action.copy())
    assert same_bits(result.action, action)
    assert result.intervened is False and result.fallback_level == 0
    assert result.status == "feasible" and result.certificate == "nominal"
    assert result.certificate_horizon == math.inf


def in_set_states(count):
    backup = ready(BackupFilter(barrier="passive"))
    xs = np.random.default_rng(1104).uniform(*NEAR_BOX, (2 * count, 6))
    inside = [x for x in xs if backup.contains(x)]
    assert len(inside) >= count
    return inside[:count]


def check_one_step_filter_accepts_what_the_backup_filter_accepts(count):
    backup = ready(BackupFilter(barrier="passive"))
    predictive = ready(PredictiveFilter(horizon=1))
    accepted = []
    for x in in_set_states(count):
        # Toward the target at full thrust on its largest axis, inside the box.
        action = -x[:3] / np.max(np.abs(x[:3]))
        by_backup = not backup.filter(x.copy(), action.copy()).active
        result = predictive.filter(x.copy(), action.copy())
        assert same_bits(result.action, action) == by_backup
        accepted.append(by_backup)
    assert any(accepted) and not all(accepted)


# Each rejected command costs a solve, so the test runs the first 40 of the 200 seeded states;
# a benchmark runs all of them.
ACCEPT_PREFIX = 40


def test_one_step_filter_accepts_exactly_the_commands_the_backup_filter_accepts():
    check_one_step_filter_accepts_what_the_backup_filter_accepts(ACCEPT_PREFIX)


# Flights.


@pytest.mark.parametrize("start", [V_BAR, X0], ids=["v-bar", "flyby"])
def test_reckless_agent_is_held_out_with_a_nominal_certificate_on_every_step(start):
    worst, results = fly(PredictiveFilter(), Reckless(), start, 20)
    assert len(results) == 20
    assert all(r.status == "feasible" and r.recovering is False for r in results)
    assert all(r.certificate == "nominal" for r in results)
    assert all(r.certificate_horizon == math.inf for r in results)
    assert all(r.fallback_level in (0, 1) for r in results)
    assert any(r.intervened for r in results)
    assert worst >= -DEPTH


# Fallbacks, with the solver replaced. DIP and LATE are the plans the solver returns from X1.
# 3.077 mm/s^2 toward the y axis for seven steps takes the pass 15 mm inside the keep-out 1.25 s
# into the seventh step, midway between two row samples, which stay 16 mm outside; the drift
# after it recedes.
DIP = np.tile([-0.003077, 0.0, 0.0], (7, 1))
# One step at 15 mm/s^2 toward the y axis, clear of the keep-out, sets up a drift that enters it.
LATE = np.array([[-0.015, 0.0, 0.0]])


def row_count(qp):
    stages = 0 if qp.C is None else qp.C.shape[0] * qp.C.shape[1]
    return stages + (0 if qp.C_N is None else qp.C_N.shape[0])


def converged(inputs):
    """A solver that returns `inputs`, then zeros, as a converged plan rolled out exactly."""

    def solve(qp, *args, **kwargs):
        steps, _, nu = qp.B.shape
        u = np.zeros((steps, nu))
        u[: len(inputs)] = inputs
        x = [np.array(qp.x0, dtype=float)]
        for k in range(steps):
            x.append(qp.A[k] @ x[-1] + qp.B[k] @ u[k] + qp.c[k])
        return HorizonSolution(
            np.array(x), u, "converged", 7, 0.0, 0.0, 0.0, 0.0, np.zeros(row_count(qp))
        )

    return solve


def stopped(status):
    """A solver that stops with `status` and no plan."""

    def solve(qp, *args, **kwargs):
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
            certificate=np.ones(row_count(qp)),
        )

    return solve


def planned(monkeypatch):
    """The filter after a first decision at X0 that applied the converged PLAN."""
    filt = ready(PredictiveFilter())
    monkeypatch.setattr(predictive_module, "solve_horizon", converged(PLAN))
    first = filt.filter(X0.copy(), PUSH.copy())
    assert first.status == "feasible" and first.fallback_level == 0
    np.testing.assert_allclose(first.action, PLAN[0] / A_MAX, rtol=0.0, atol=1e-12)
    return filt


def assert_shifted(result, u):
    """Level 1: the stored plan's next input, certified for all time."""
    assert result.status == "feasible" and result.fallback_level == 1
    assert result.certificate == "nominal" and result.certificate_horizon == math.inf
    np.testing.assert_allclose(result.action, u / A_MAX, rtol=0.0, atol=1e-12)


def test_a_plan_dipping_into_the_keep_out_between_row_samples_gives_the_stored_plan(monkeypatch):
    phis, gammas = cw_discrete(N, np.linspace(0.0, DT, 5))
    x, lowest = X1, math.inf
    for u in DIP[:-1]:
        lowest = min(lowest, np.linalg.norm((phis @ x + gammas @ u)[:, :3], axis=1).min())
        x = PHI @ x + GAMMA @ u
    rows = (phis @ x + gammas @ DIP[-1])[:, :3]
    assert min(lowest, np.linalg.norm(rows, axis=1).min()) > R + 0.01
    phis, gammas = cw_discrete(N, np.linspace(0.0, DT, 4001))
    assert np.linalg.norm((phis @ x + gammas @ DIP[-1])[:, :3], axis=1).min() < R - 0.01
    filt = planned(monkeypatch)
    assert filt.declaration.barriers[0].value(PHI @ x + GAMMA @ DIP[-1]) >= 0.0
    monkeypatch.setattr(predictive_module, "solve_horizon", converged(DIP))
    assert_shifted(filt.filter(X1.copy(), PUSH.copy()), PLAN[1])


def test_a_plan_whose_drift_after_the_last_thrust_enters_the_keep_out_gives_the_stored_plan(
    monkeypatch,
):
    assert held_segment_bound(X1, LATE[0], DT, 20, R, N) > 0.0
    filt = planned(monkeypatch)
    assert filt.declaration.barriers[0].value(PHI @ X1 + GAMMA @ LATE[0]) < 0.0
    monkeypatch.setattr(predictive_module, "solve_horizon", converged(LATE))
    assert_shifted(filt.filter(X1.copy(), PUSH.copy()), PLAN[1])


@pytest.mark.parametrize("status", ["iteration_cap", "solver_failed"])
def test_a_solve_without_a_plan_applies_the_stored_plan_step_by_step(monkeypatch, status):
    filt = planned(monkeypatch)
    monkeypatch.setattr(predictive_module, "solve_horizon", stopped(status))
    assert_shifted(filt.filter(X1.copy(), PUSH.copy()), PLAN[1])
    assert_shifted(filt.filter(X2.copy(), PUSH.copy()), PLAN[2])


def test_infeasible_certified_from_the_solver_is_never_reported(monkeypatch):
    monkeypatch.setattr(predictive_module, "solve_horizon", stopped("infeasible_certified"))
    first = ready(PredictiveFilter()).filter(X0.copy(), PUSH.copy())
    assert first.status == "feasible" and first.proof is None
    assert first.fallback_level == 2 and first.certificate == "nominal"
    np.testing.assert_array_equal(first.action, np.zeros(3))
    filt = planned(monkeypatch)
    monkeypatch.setattr(predictive_module, "solve_horizon", stopped("infeasible_certified"))
    later = filt.filter(X1.copy(), PUSH.copy())
    assert later.proof is None
    assert_shifted(later, PLAN[1])


# Outside the set: XO passes 6 m closer than X1 and its drift enters the keep-out; XO0 is one
# step before it under W0. OUT, 0.408 mm/s^2 outward for seven steps, clears the pass but for a
# 15 mm dip in its sixth step, and its drift after the last thrust recedes.
XO = np.array([54.0, 78.935, 0.0, 0.0, -1.5, 0.0])
W0 = np.array([0.001, 0.0, 0.001])
XO0 = np.linalg.solve(PHI, XO - GAMMA @ W0)
OUT = np.tile([0.000408, 0.0, 0.0], (7, 1))
# One small outward step that leaves the drift entering the keep-out.
SHORT = np.array([[0.0005, 0.0, 0.0005]])


def recovered(monkeypatch, stored):
    """The filter after a recovering decision at XO0 that applied W0 and then `stored`."""
    filt = ready(PredictiveFilter())
    assert filt.declaration.barriers[0].value(XO0) < 0.0
    monkeypatch.setattr(predictive_module, "solve_horizon", converged(np.vstack([W0, stored])))
    first = filt.filter(XO0.copy(), PUSH.copy())
    assert first.recovering is True and first.certificate is None
    np.testing.assert_allclose(first.action, W0 / A_MAX, rtol=0.0, atol=1e-12)
    monkeypatch.setattr(predictive_module, "solve_horizon", stopped("iteration_cap"))
    return filt


def assert_uncertified_stored_plan(result, u):
    assert result.status == "feasible" and result.fallback_level == 1
    assert result.recovering is True and result.slack > 0.0
    assert result.certificate is None and result.certificate_horizon is None
    np.testing.assert_allclose(result.action, u / A_MAX, rtol=0.0, atol=1e-12)


def test_a_stored_plan_whose_drift_enters_the_keep_out_gets_no_certificate(monkeypatch):
    filt = recovered(monkeypatch, SHORT)
    assert held_segment_bound(XO, SHORT[0], DT, 20, R, N) > 0.0
    assert filt.declaration.barriers[0].value(PHI @ XO + GAMMA @ SHORT[0]) < 0.0
    assert_uncertified_stored_plan(filt.filter(XO.copy(), PUSH.copy()), SHORT[0])


def test_a_stored_plan_short_by_more_than_the_keep_out_resolution_gets_no_certificate(
    monkeypatch,
):
    filt = recovered(monkeypatch, OUT)
    x, held = XO, []
    for u in OUT:
        held.append(held_segment_bound(x, u, DT, 20, R, N))
        x = PHI @ x + GAMMA @ u
    # Short by 15 mm on one step: beyond the 1 mm allowed for rounding, far within 1 m.
    assert -0.1 < min(held) < -10.0 * DEPTH
    assert filt.declaration.barriers[0].value(x) >= 0.0
    assert_uncertified_stored_plan(filt.filter(XO.copy(), PUSH.copy()), OUT[0])


def test_one_newton_step_gives_the_shifted_plan():
    filt = ready(PredictiveFilter(settings=IPMSettings(max_iter=1)))
    first = filt.filter(X0.copy(), PLAN[0] / A_MAX)
    assert first.fallback_level == 0 and first.intervened is False
    result = filt.filter(X1.copy(), PUSH.copy())
    assert result.iterations == 1
    assert result.status == "feasible" and result.fallback_level == 1
    assert result.certificate == "nominal" and result.certificate_horizon == math.inf
    np.testing.assert_array_equal(result.action, np.zeros(3))


# Delay.


def fly_delayed(delay_s, layout, steps):
    """Predicted and true states delay_s after each observation, and the deepest bound."""
    env = approach(
        DT,
        steps,
        V_BAR,
        delay_s=delay_s,
        observation_channels=("pending_commands",),
        thrust_limit=layout,
    )
    filt = PredictiveFilter(delay_s=delay_s)
    pilot = FilteredAgent(Reckless(), filt)
    obs, _ = env.reset(seed=0)
    pilot.reset(env.agent_view(), np.random.default_rng(0))
    predicted, truth, worst = [], {}, math.inf
    for _ in range(steps):
        predicted.append((round(env.t + delay_s, 6), filt.predict(obs)))
        obs, _, _, _, _ = env.step(pilot.act(obs, env.agent_info()))
        segment = env.record.segment
        # A command taking effect within a step starts a segment there, so its start is truth.
        for (t0, _, _), x in zip(segment.segments, segment.starts, strict=True):
            truth[round(t0, 6)] = x
        truth[round(env.t, 6)] = env.state.copy()
        worst = min(worst, segment.minima["keep_out"].lower_bound)
        assert_in_actuator_set(pilot.last_result.action, layout)
    pairs = [(x, truth[t]) for t, x in predicted if t in truth]
    return pairs, worst


@pytest.mark.parametrize("layout", ["box", "ball"])
@pytest.mark.parametrize("delay_s", [10.0, 15.0])
def test_predicted_state_through_pending_commands_is_the_true_state(delay_s, layout):
    steps = 8
    pairs, worst = fly_delayed(delay_s, layout, steps)
    assert len(pairs) >= steps - 2
    for x, true in pairs:
        assert np.linalg.norm(x - true) <= 1e-9 * np.linalg.norm(true)
    assert worst >= -DEPTH


@pytest.mark.parametrize(
    ("declared", "delay_s", "channels"),
    [
        (10.0, 10.0, ()),
        (10.0, 0.0, ("pending_commands",)),
        (0.0, 10.0, ("pending_commands",)),
        (25.0, 10.0, ("pending_commands",)),
        (10.0, 25.0, ("pending_commands",)),
    ],
    ids=["no-slots", "no-delay", "undeclared", "too-many", "too-few"],
)
def test_reset_refuses_a_delay_that_disagrees_with_the_pending_slots(declared, delay_s, channels):
    with pytest.raises(ValueError):
        ready(PredictiveFilter(delay_s=declared), delay_s=delay_s, observation_channels=channels)


@pytest.mark.parametrize("name", ["horizon", "row_samples"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_construction_refuses_a_count_that_is_not_finite(name, value):
    with pytest.raises(ValueError):
        PredictiveFilter(**{name: value})
