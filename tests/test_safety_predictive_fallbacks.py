"""Braking recovery, the declared CAM backup and the delay-value check of the predictive filter.

The interface these tests assume, beyond that of `tests/test_safety_predictive.py`:

- `PredictiveFilter(*, backup="passive", brake_v_cap=1.0, rho_cap=2000.0, ...)` keeps each as
  an attribute of the same name; `backup` is "passive" or "cam", and any other value raises
  ValueError.
- At reset, a_b = braking_deceleration(a_max, layout, v_cap=brake_v_cap, dt, radius, rho_cap, n).
  When a_b > 0 there is a braking level: a decision with nothing certified, at a state x where
  `BrakingBarrier(radius=R, a_b=a_b, rho_cap=rho_cap, v_cap=brake_v_cap).contains(x)`, applies
  that barrier's `backup(x, actuator)` at `fallback_level` 2 as "feasible" with `recovering`
  True and no certificate. When a_b <= 0 reset still succeeds and there is no braking level.
  `declaration.backup` is "zero thrust, else full outward braking outside the set" when the
  braking level exists and does not name braking otherwise. Braking comes before an
  uncertified soft solution.
- With backup="cam" the declared set is the CAM set: `declaration.barriers` is the one
  `CamBarrier` that `BackupFilter(barrier="cam")` declares for the same view, and `contains` is
  its membership. Level 2 is `CamBarrier.backup(x)` wherever its value is >= 0, with
  certificate "nominal" and horizon inf, and the stored plan is the remaining CAM steps, so the
  next decision with no solution replays them at level 1. Its name is "predictive-cam-n{N}",
  and it is refused together with mode="finite-horizon".
- A decision with nothing certified, which happens only outside the declared set, reports
  `recovering` True.
- With delay_s > dt, the second and every later decision raises ValueError unless the latest
  pending slot's wait equals delay_s - dt within 1e-6 s. With delay_s <= dt every wait is 0 and
  the value is not checked.

BRAKE, 60 m out on -V-bar closing at 0.26 m/s, stands in for a faster closing start: from
that start, (0, 80, 0, 0, -0.5, 0), the current filter certifies every decision and clears the
keep-out by 22 mm, and it lies outside the braking set at the defaults (h_b = -5.63 m with
a_b = 3.508e-3 m/s^2). BRAKE is inside it (h_b = +0.37 m) and the filter without braking drifts
about 35 m into the keep-out from it.
"""

import math

import numpy as np
import pytest

from vbar.constraints import KeepOut
from vbar.dynamics import cw_discrete
from vbar.safety import FilteredAgent
from vbar.safety import predictive as predictive_module
from vbar.safety.barriers import BrakingBarrier, CamBarrier, braking_deceleration
from vbar.safety.cbf import BackupFilter
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.solvers.ipm import HorizonSolution
from vbar.solvers.qp import ActuatorSet
from vbar.spacecraft import PerfectNav

PredictiveFilter = predictive_module.PredictiveFilter

R = 50.0
A_MAX = 0.02
DT = 10.0
N = ApproachEnv().n
PHI, GAMMA = cw_discrete(N, DT)
BOX = ActuatorSet(a_max=A_MAX, layout="box")
DEPTH = KeepOut.resolution
FAR = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])
NEAR_BOX = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))
BRAKE = np.array([0.0, -60.0, 0.0, 0.0, 0.26, 0.0])
# test_safety_cam.py: free drift enters the keep-out, the two-step CAM along -y clears it.
CLOSING = np.array([0.0, -60.0, 0.0, 0.0, 0.3, 0.0])
CAM = np.array([0.0, -A_MAX, 0.0])
ZERO = np.zeros(3)


def approach(dt=DT, steps=1, start=FAR, **cfg):
    start = tuple(float(s) for s in start)
    return ApproachEnv(ApproachConfig(dt=dt, max_steps=steps, start_state=start, **cfg))


def ready(filt, dt=DT, **cfg):
    filt.reset(approach(dt, **cfg).agent_view())
    return filt


def braking_barrier(dt=DT, v_cap=1.0, rho_cap=2000.0):
    a_b = braking_deceleration(
        a_max=A_MAX, layout="box", v_cap=v_cap, dt=dt, radius=R, rho_cap=rho_cap, n=N
    )
    return BrakingBarrier(radius=R, a_b=a_b, rho_cap=rho_cap, v_cap=v_cap)


def toward_target(x):
    return -x[:3] / np.linalg.norm(x[:3])


class Zero:
    name = "zero"

    def reset(self, view, rng=None):
        pass

    def act(self, obs, info):
        return np.zeros(3)


def fly(filt, start, steps):
    """Fly the zero agent through `filt`; the deepest certified bound and every result."""
    env = approach(DT, steps, start, sensor=PerfectNav())
    pilot = FilteredAgent(Zero(), filt)
    obs, _ = env.reset(seed=0)
    pilot.reset(env.agent_view(), np.random.default_rng(0))
    worst, results = math.inf, []
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(pilot.act(obs, env.agent_info()))
        worst = min(worst, env.record.segment.minima["keep_out"].lower_bound)
        results.append(pilot.last_result)
        if terminated or truncated:
            break
    return worst, results


def stopped(qp, *args, **kwargs):
    """A solver that stops at its iteration cap with no plan."""
    return HorizonSolution(None, None, "iteration_cap", 7, math.nan, 1.0, 1.0, 1.0, None)


# Construction.


def test_defaults_are_the_passive_backup_and_braking_up_to_one_metre_per_second():
    filt = PredictiveFilter()
    assert filt.backup == "passive"
    assert filt.brake_v_cap == 1.0 and filt.rho_cap == 2000.0


def test_an_unknown_backup_is_refused():
    with pytest.raises(ValueError):
        PredictiveFilter(backup="braking")


# Braking recovery.


def test_braking_is_applied_when_nothing_is_certified_inside_the_braking_set(monkeypatch):
    filt = ready(PredictiveFilter())
    assert filt.contains(BRAKE) is False
    assert braking_barrier().value(BRAKE) == pytest.approx(0.366, abs=1e-3)
    assert braking_barrier().contains(BRAKE)
    monkeypatch.setattr(predictive_module, "solve_horizon", stopped)
    result = filt.filter(BRAKE.copy(), ZERO.copy())
    assert result.status == "feasible" and result.fallback_level == 2
    assert result.recovering is True
    assert result.certificate is None and result.certificate_horizon is None
    expected = braking_barrier().backup(BRAKE, BOX) / A_MAX
    np.testing.assert_allclose(result.action, expected, rtol=0.0, atol=1e-12)


# 2000 s is asked for; the free drift from BRAKE meets the keep-out within 40 s, and braking
# hands over to a certified decision within a few steps, so 12 decisions show the claim.
BRAKE_STEPS = 12


def test_braking_keeps_a_fast_close_approach_out_until_a_decision_is_certified():
    worst, results = fly(PredictiveFilter(), BRAKE, BRAKE_STEPS)
    assert len(results) == BRAKE_STEPS
    assert worst >= -DEPTH
    braked = [
        r.fallback_level == 2 and r.recovering and np.any(np.asarray(r.action) != 0.0)
        for r in results
    ]
    assert braked[0]
    certified = [r.certificate == "nominal" for r in results]
    assert any(certified[: BRAKE_STEPS - 1])
    assert all(r.status == "feasible" for r in results)


# The entry passes 30 m deep about 170 s in and bottoms out near 35 m about 200 s in.
MUTANT_STEPS = 18


def test_a_no_braking_mutant_enters_the_keep_out_by_about_35_m(monkeypatch):
    monkeypatch.setattr(BrakingBarrier, "contains", lambda self, x: False)
    worst, _ = fly(PredictiveFilter(), BRAKE, MUTANT_STEPS)
    assert -40.0 < worst < -30.0


def test_braking_is_never_applied_inside_the_declared_set(monkeypatch):
    filt = ready(PredictiveFilter())
    # The first 100 of the 200 seeded in-set states, for the 2 s budget; each costs a reset and
    # a decision.
    xs = np.random.default_rng(1104).uniform(*NEAR_BOX, (200, 6))
    inside = [x for x in xs if filt.contains(x)][:100]
    assert len(inside) == 100
    brakeable = braking_barrier()
    # Most of them are in the braking set too, so a braking level tried first would show.
    assert sum(brakeable.contains(x) for x in inside) >= 50
    monkeypatch.setattr(predictive_module, "solve_horizon", stopped)
    for x in inside:
        filt.reset(approach().agent_view())
        result = filt.filter(x.copy(), toward_target(x))
        assert result.recovering is False and result.certificate == "nominal"
        if result.fallback_level == 2:
            np.testing.assert_array_equal(result.action, ZERO)


def test_without_a_positive_braking_deceleration_there_is_no_braking_level(monkeypatch):
    a_b = braking_deceleration(
        a_max=A_MAX, layout="box", v_cap=1.0, dt=30.0, radius=R, rho_cap=2000.0, n=N
    )
    assert a_b <= 0.0
    with_braking = ready(PredictiveFilter()).declaration.backup
    without = ready(PredictiveFilter(), dt=30.0)
    assert with_braking == "zero thrust, else full outward braking outside the set"
    assert "braking" not in without.declaration.backup
    monkeypatch.setattr(predictive_module, "solve_horizon", stopped)
    result = without.filter(BRAKE.copy(), ZERO.copy())
    assert result.fallback_level == 2 and result.recovering is True
    np.testing.assert_array_equal(result.action, ZERO)


# The declared CAM backup.


def test_the_cam_backup_is_named_for_it():
    assert PredictiveFilter(backup="cam").name == "predictive-cam-n10"


def test_the_soft_mode_with_the_cam_backup_is_named_for_both():
    assert PredictiveFilter(mode="soft", backup="cam").name == "predictive-soft-cam-n10"


@pytest.mark.parametrize(
    ("kw", "backup"),
    [
        ({"backup": "cam"}, "zero thrust, else the CAM, else full outward braking outside the set"),
        ({"mode": "finite-horizon"}, "zero thrust, else full outward braking outside the set"),
    ],
    ids=["cam", "finite-horizon"],
)
def test_the_declared_backup_names_each_fallback_in_order(kw, backup):
    assert ready(PredictiveFilter(**kw)).declaration.backup == backup


def test_the_cam_backup_is_refused_in_the_finite_horizon_mode():
    # The CAM set holds for all time; a finite-horizon declaration cannot carry it.
    with pytest.raises(ValueError):
        PredictiveFilter(backup="cam", mode="finite-horizon")


def test_cam_backup_declares_the_cam_set_of_the_backup_filter():
    filt = ready(PredictiveFilter(backup="cam"))
    (barrier,) = filt.declaration.barriers
    assert isinstance(barrier, CamBarrier)
    assert filt.declaration.barriers == ready(BackupFilter(barrier="cam")).declaration.barriers
    assert filt.contains(CLOSING) is True
    assert ready(PredictiveFilter()).contains(CLOSING) is False


def test_cam_level_2_is_the_cam_and_the_stored_plan_replays_its_remaining_step(monkeypatch):
    filt = ready(PredictiveFilter(backup="cam"))
    monkeypatch.setattr(predictive_module, "solve_horizon", stopped)
    first = filt.filter(CLOSING.copy(), ZERO.copy())
    assert first.status == "feasible" and first.fallback_level == 2
    assert first.recovering is False
    assert first.certificate == "nominal" and first.certificate_horizon == math.inf
    np.testing.assert_allclose(first.action, CAM / A_MAX, rtol=0.0, atol=1e-12)
    second = filt.filter(PHI @ CLOSING + GAMMA @ CAM, ZERO.copy())
    assert second.fallback_level == 1 and second.recovering is False
    assert second.certificate == "nominal"
    np.testing.assert_allclose(second.action, CAM / A_MAX, rtol=0.0, atol=1e-12)


# 2000 s is asked for; the CAM ends after two steps and the drift after it recedes, so 15
# decisions show the claim.
CAM_STEPS = 15


def test_cam_backup_keeps_the_closing_start_out_with_nominal_decisions_only():
    worst, results = fly(PredictiveFilter(backup="cam"), CLOSING, CAM_STEPS)
    assert len(results) == CAM_STEPS
    assert worst >= -DEPTH
    assert all(r.status == "feasible" and r.recovering is False for r in results)
    assert all(r.certificate == "nominal" for r in results)


def test_passive_backup_is_recovering_at_the_closing_start():
    result = ready(PredictiveFilter(backup="passive")).filter(CLOSING.copy(), ZERO.copy())
    assert result.recovering is True and result.certificate is None


# The delay value. A command issued at t takes effect at t + delay_s, so with delay_s > dt the
# latest pending slot waits delay_s - dt at every decision after the first.


def delayed_decisions(declared, delay_s, decisions):
    """Run the zero agent's decisions through the filter, the true delay `delay_s`."""
    env = approach(
        DT,
        decisions,
        FAR,
        delay_s=delay_s,
        observation_channels=("pending_commands",),
        sensor=PerfectNav(),
    )
    filt = PredictiveFilter(delay_s=declared)
    obs, _ = env.reset(seed=0)
    filt.reset(env.agent_view())
    done = 0
    for _ in range(decisions):
        result = filt.filter(obs, ZERO.copy())
        done += 1
        obs, _, _, _, _ = env.step(np.asarray(result.action, dtype=float))
    return done


@pytest.mark.parametrize("declared", [15.0, 11.0])
def test_a_delay_with_the_right_slot_count_and_the_wrong_value_raises_at_the_second_decision(
    declared,
):
    assert delayed_decisions(declared, 12.0, 1) == 1
    with pytest.raises(ValueError):
        delayed_decisions(declared, 12.0, 2)


def test_a_short_observation_on_the_second_decision_is_refused_as_malformed():
    env = approach(
        DT, 2, FAR, delay_s=12.0, observation_channels=("pending_commands",), sensor=PerfectNav()
    )
    filt = PredictiveFilter(delay_s=12.0)
    obs, _ = env.reset(seed=0)
    filt.reset(env.agent_view())
    obs, _, _, _, _ = env.step(np.asarray(filt.filter(obs, ZERO.copy()).action, dtype=float))
    with pytest.raises(ValueError):
        filt.filter(obs[:3], ZERO.copy())


def test_the_declared_delay_runs_twenty_decisions():
    assert delayed_decisions(12.0, 12.0, 20) == 20


def test_a_delay_within_one_step_cannot_be_checked():
    # 5 s and 8 s both fill one slot whose wait is always 0, so the mismatch goes unseen.
    assert delayed_decisions(5.0, 8.0, 20) == 20


def test_the_slot_wait_is_the_delay_less_one_step():
    env = approach(
        DT, 3, FAR, delay_s=12.0, observation_channels=("pending_commands",), sensor=PerfectNav()
    )
    obs, _ = env.reset(seed=0)
    obs, _, _, _, _ = env.step(np.array([0.1, 0.0, 0.0]))
    names = env.agent_view().observation
    assert obs[names.index("pending_1_t")] == pytest.approx(2.0, abs=1e-6)
