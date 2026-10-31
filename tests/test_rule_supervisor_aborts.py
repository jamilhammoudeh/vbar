"""The abort rules of the rule supervisor, on constructed situations.

Rule A aborts on a predicted collision: the free drift of the observed state, propagated with
the exact transition matrix and sampled every 5 s up to `warning_s`, comes within the largest
keep-out radius, and stopping before that sphere needs more than `brake_margin` times the
view's max_accel (closing^2 / (2 (range - radius)), or the chaser is inside the sphere and
closing). Rule B aborts on a braking deficit: the chaser was told to stop (phase
"stopped" or "holding") and still closes faster than `overspeed` speed caps without slowing, on
`strikes` consecutive consults. The first consult after a reset counts as not having slowed.

The view has a 50 m keep-out sphere and a constant closing-speed law of 0.2 m/s, so the
overspeed threshold of the defaults is 0.24 m/s.
"""

import math

import numpy as np
import pytest

from vbar import run_episode
from vbar.agents.base import AgentView
from vbar.agents.rules import RuleSupervisor
from vbar.agents.supervision import Command, Rung, Situation, SupervisedAgent
from vbar.agents.tracker import SetpointTracker
from vbar.dynamics import cw_stm
from vbar.scenarios import make, seeds
from vbar.scenarios.suites import SUITES

CONTINUE, GO, HOLD, RETREAT, CAM, INHIBIT = (
    Command.CONTINUE,
    Command.GO,
    Command.HOLD,
    Command.RETREAT,
    Command.CAM,
    Command.THRUST_INHIBIT,
)

N = 0.0011
KEEP_OUT = 50.0
CAP = 0.2
SAMPLE_S = 5.0
MAX_ACCEL = 0.01
# Neither the overspeed hold nor the braking rule acts in this phase
PHASE = "retreating"

LIMIT = ({"type": "closing_speed_limit", "v0": CAP, "v1": 0.0},)
SPHERE = ({"type": "keep_out", "radius": KEEP_OUT},) + LIMIT

LADDER = (
    Rung((0.0, -200.0, 0.0), "hold", 5.0, 0.05),
    Rung((0.0, -100.0, 0.0), "goal", 5.0, 0.05),
)
DOCK_LADDER = (
    Rung((0.0, -60.0, 0.0), "hold", 5.0, 0.05),
    Rung((0.0, -2.0, 0.0), "contact", 0.1, 0.1),
)


def view_of(task, constraints):
    return AgentView(
        scenario_id="rules-test",
        dt=10.0,
        max_steps=100,
        n=N,
        action_mode="continuous",
        max_accel=MAX_ACCEL,
        thrust_limit="box",
        goal={"task": task},
        constraints=constraints,
    )


def supervisor(task="hold", constraints=SPHERE, **parameters):
    rules = RuleSupervisor(**parameters)
    rules.reset(view_of(task, constraints), LADDER)
    return rules


def situation(phase, obs, *, kind="goal", t=0.0, refuse=()):
    mask = [True] * len(Command)
    mask[Command.RECONFIGURE] = False
    for command in refuse:
        mask[command] = False
    return Situation(
        t=t,
        obs=np.asarray(obs, dtype=float),
        phase=phase,
        rung=1 if kind in ("goal", "contact") else 0,
        ladder=DOCK_LADDER if kind == "contact" else LADDER,
        mask=tuple(mask),
        reasons={},
        refused=None,
        info={},
    )


def sampled_range(obs, warning_s=600.0):
    """Range (m) from the target of the free drift at 5 s, 10 s, ... up to `warning_s`."""
    times = np.arange(SAMPLE_S, warning_s + 0.5 * SAMPLE_S, SAMPLE_S)
    return np.array([np.linalg.norm((cw_stm(N, t) @ obs)[:3]) for t in times])


def first_sample_inside(obs, warning_s=600.0):
    """Time (s) of the first sample within the keep-out radius."""
    inside = np.nonzero(sampled_range(obs, warning_s) < KEEP_OUT)[0]
    assert len(inside) > 0
    return SAMPLE_S * (inside[0] + 1)


def drift_from(offset, speed):
    """100 m out on the V-bar, closing at `speed`, `offset` m off the V-bar in x."""
    return np.array([offset, -100.0, 0.0, 0.0, speed, 0.0])


def offset_with_closest_approach(clearance, speed):
    """The x offset whose sampled drift comes `clearance` m from the target, by bisection."""
    low, high = -10.0, 100.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if sampled_range(drift_from(mid, speed)).min() < clearance:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def braking_need(obs):
    """Deceleration (m/s^2) that stops the chaser at the keep-out sphere."""
    rho = np.linalg.norm(obs[:3])
    closing = -(obs[:3] @ obs[3:6]) / rho
    return closing**2 / (2.0 * (rho - KEEP_OUT))


# 1 m either side of the 50 m sphere, closing at 1 m/s: stopping needs 0.01 m/s^2, twice the
# default brake_margin times max_accel
CLEARS = drift_from(offset_with_closest_approach(KEEP_OUT + 1.0, 1.0), 1.0)
ENTERS = drift_from(offset_with_closest_approach(KEEP_OUT - 1.0, 1.0), 1.0)
# The same entry at the speed cap: stopping needs a twentieth of the acceleration
EASY = drift_from(offset_with_closest_approach(KEEP_OUT - 1.0, CAP), CAP)

# Closing at 0.3 m/s, above 1.2 speed caps, on a view with no sphere so that only rule B acts
FAST = np.array([0.0, -100.0, 0.0, 0.0, 0.3, 0.0])


def at_speed(closing):
    return np.array([0.0, -100.0, 0.0, 0.0, closing, 0.0])


def braking(**parameters):
    return supervisor(constraints=LIMIT, **parameters)


def test_the_fixtures_clear_and_enter_the_sphere_by_1_m():
    assert sampled_range(CLEARS).min() == pytest.approx(KEEP_OUT + 1.0, abs=1e-6)
    assert sampled_range(ENTERS).min() == pytest.approx(KEEP_OUT - 1.0, abs=1e-6)
    assert sampled_range(EASY).min() == pytest.approx(KEEP_OUT - 1.0, abs=1e-6)
    for obs in (CLEARS, ENTERS, EASY):
        assert np.linalg.norm(obs[:3]) > 1.1 * KEEP_OUT
    for obs in (CLEARS, ENTERS):
        assert braking_need(obs) > 0.5 * MAX_ACCEL
    assert braking_need(EASY) < 0.5 * MAX_ACCEL


def test_the_defaults_are_a_warning_of_600_s_a_brake_margin_of_0_5_and_2_strikes():
    rules = RuleSupervisor()
    assert rules.brake_margin == 0.5
    assert rules.warning_s == 600.0
    assert rules.strikes == 2


@pytest.mark.parametrize(
    ("obs", "expected"), [(CLEARS, CONTINUE), (ENTERS, CAM)], ids=["clears", "enters"]
)
def test_a_drift_entering_the_keep_out_sphere_gets_a_cam_and_one_clearing_it_does_not(
    obs, expected
):
    assert supervisor().decide(situation(PHASE, obs)) == expected


@pytest.mark.parametrize("phase", ["holding", "stopped", "retreating", "inhibited"])
def test_a_predicted_collision_gets_a_cam_in_any_phase_outside_the_commit(phase):
    assert supervisor().decide(situation(phase, ENTERS)) == CAM


def test_the_collision_radius_is_the_largest_keep_out_radius():
    constraints = (
        {"type": "keep_out", "radius": 20.0},
        {"type": "keep_out", "radius": KEEP_OUT},
    ) + LIMIT
    assert supervisor(constraints=constraints).decide(situation(PHASE, ENTERS)) == CAM
    assert supervisor(constraints=constraints).decide(situation(PHASE, CLEARS)) == CONTINUE


def test_a_view_without_a_keep_out_sphere_never_predicts_a_collision():
    assert supervisor(constraints=LIMIT).decide(situation(PHASE, ENTERS)) == CONTINUE


def test_an_entry_at_the_last_sample_in_the_warning_fires_and_one_sample_later_does_not():
    t_in = first_sample_inside(ENTERS)
    assert t_in > 2 * SAMPLE_S
    assert supervisor(warning_s=t_in).decide(situation(PHASE, ENTERS)) == CAM
    assert supervisor(warning_s=t_in - SAMPLE_S).decide(situation(PHASE, ENTERS)) == CONTINUE


def test_a_sphere_entered_just_after_the_warning_does_not_fire():
    t_in = first_sample_inside(ENTERS)
    assert supervisor(warning_s=t_in - 1.0).decide(situation(PHASE, ENTERS)) == CONTINUE


def test_a_longer_warning_sees_a_collision_the_default_does_not():
    # the drift enters the sphere only after the default warning
    obs = np.array([-100.0, -200.0, 0.0, 0.0, CAP, 0.0])
    t_in = first_sample_inside(obs, 3000.0)
    assert t_in > 600.0
    rules = supervisor(brake_margin=1e-6)
    assert braking_need(obs) > 1e-6 * MAX_ACCEL
    assert rules.decide(situation(PHASE, obs)) == CONTINUE
    assert supervisor(warning_s=t_in, brake_margin=1e-6).decide(situation(PHASE, obs)) == CAM


def test_a_drift_entering_the_sphere_is_not_aborted_when_stopping_is_easy():
    assert supervisor().decide(situation(PHASE, EASY)) == CONTINUE


def test_a_chaser_closing_hard_whose_drift_clears_the_sphere_is_not_aborted():
    assert braking_need(CLEARS) > 0.5 * MAX_ACCEL
    assert supervisor().decide(situation(PHASE, CLEARS)) == CONTINUE


def test_brake_margin_sets_how_much_of_max_accel_stopping_may_need():
    need = braking_need(EASY) / MAX_ACCEL
    assert supervisor(brake_margin=0.9 * need).decide(situation(PHASE, EASY)) == CAM
    assert supervisor(brake_margin=1.1 * need).decide(situation(PHASE, EASY)) == CONTINUE


def test_a_chaser_inside_the_sphere_and_closing_is_aborted_whatever_the_braking():
    obs = np.array([0.0, -40.0, 0.0, 0.0, 0.01, 0.0])
    assert supervisor().decide(situation(PHASE, obs, refuse=(CAM,))) == RETREAT


def test_a_chaser_inside_the_sphere_and_opening_is_not_aborted_by_the_collision_rule():
    obs = np.array([0.0, -40.0, 0.0, 0.0, -0.05, 0.0])
    assert supervisor().decide(situation(PHASE, obs, refuse=(CAM,))) == CONTINUE


def test_a_retreat_is_ordered_for_a_predicted_collision_when_the_mask_refuses_the_cam():
    assert supervisor().decide(situation(PHASE, ENTERS, refuse=(CAM,))) == RETREAT


def test_a_cam_is_preferred_to_a_retreat_for_a_predicted_collision():
    assert supervisor().decide(situation(PHASE, ENTERS, refuse=(RETREAT,))) == CAM
    assert supervisor().decide(situation(PHASE, ENTERS)) == CAM


def test_nothing_is_ordered_for_a_predicted_collision_when_the_mask_refuses_both():
    sit = situation(PHASE, ENTERS, refuse=(CAM, RETREAT))
    assert supervisor().decide(sit) == CONTINUE


def test_the_contact_rung_is_exempt_from_the_predicted_collision_rule():
    sit = situation(PHASE, ENTERS, kind="contact")
    assert supervisor("dock").decide(sit) == CONTINUE


def test_the_commit_phase_is_exempt_from_the_predicted_collision_rule():
    assert supervisor("hold").decide(situation("commit", ENTERS)) == CONTINUE


def test_the_commit_inhibit_beats_the_predicted_collision():
    assert supervisor("dock").decide(situation("commit", ENTERS)) == INHIBIT


def test_a_docking_commit_that_cannot_inhibit_is_not_aborted_for_a_predicted_collision():
    sit = situation("commit", ENTERS, refuse=(INHIBIT,))
    assert supervisor("dock").decide(sit) == CONTINUE


def test_the_predicted_collision_rule_beats_the_overspeed_hold():
    # closing at 1 m/s is above 1.2 speed caps
    assert supervisor(constraints=LIMIT).decide(situation("transfer", ENTERS)) == HOLD
    assert supervisor().decide(situation("transfer", ENTERS)) == CAM


@pytest.mark.parametrize("phase", ["stopped", "holding"])
def test_a_chaser_told_to_stop_that_keeps_closing_fast_is_retreated_on_the_second_consult(phase):
    rules = braking()
    assert rules.decide(situation(phase, FAST, t=0.0)) == CONTINUE
    assert rules.decide(situation(phase, FAST, t=10.0)) == RETREAT


def test_a_stopped_chaser_closing_just_under_1_2_speed_caps_is_not_retreated():
    rules = braking()
    assert rules.decide(situation("stopped", at_speed(0.23), t=0.0)) == GO
    assert rules.decide(situation("stopped", at_speed(0.23), t=10.0)) == GO


def test_a_stopped_chaser_closing_just_over_1_2_speed_caps_is_retreated():
    rules = braking()
    rules.decide(situation("stopped", at_speed(0.25), t=0.0))
    assert rules.decide(situation("stopped", at_speed(0.25), t=10.0)) == RETREAT


def test_overspeed_scales_the_threshold_of_the_braking_rule():
    rules = braking(overspeed=2.0)
    rules.decide(situation("stopped", at_speed(0.30), t=0.0))
    assert rules.decide(situation("stopped", at_speed(0.30), t=10.0)) != RETREAT
    rules.decide(situation("stopped", at_speed(0.45), t=20.0))
    assert rules.decide(situation("stopped", at_speed(0.45), t=30.0)) == RETREAT


def test_a_chaser_that_is_slowing_has_no_braking_deficit():
    rules = braking()
    for k, closing in enumerate((0.40, 0.35, 0.30, 0.27)):
        assert rules.decide(situation("stopped", at_speed(closing), t=10.0 * k)) == CONTINUE


def test_a_chaser_whose_closing_speed_rises_is_counted():
    rules = braking()
    rules.decide(situation("stopped", at_speed(0.30), t=0.0))
    assert rules.decide(situation("stopped", at_speed(0.31), t=10.0)) == RETREAT


def test_an_opening_chaser_has_no_braking_deficit():
    rules = braking()
    for k in range(4):
        assert rules.decide(situation("holding", at_speed(-0.5), t=10.0 * k)) != RETREAT


def test_a_retreat_for_a_braking_deficit_falls_back_to_a_cam_when_the_mask_refuses_it():
    rules = braking()
    rules.decide(situation("stopped", FAST, refuse=(RETREAT,)))
    assert rules.decide(situation("stopped", FAST, t=10.0, refuse=(RETREAT,))) == CAM


def test_a_retreat_is_preferred_to_a_cam_for_a_braking_deficit():
    rules = braking()
    rules.decide(situation("stopped", FAST))
    assert rules.decide(situation("stopped", FAST, t=10.0, refuse=(CAM,))) == RETREAT
    rules = braking()
    rules.decide(situation("stopped", FAST))
    assert rules.decide(situation("stopped", FAST, t=10.0)) == RETREAT


def test_nothing_is_ordered_for_a_braking_deficit_when_the_mask_refuses_both():
    rules = braking()
    both = (RETREAT, CAM)
    rules.decide(situation("stopped", FAST, refuse=both))
    assert rules.decide(situation("stopped", FAST, t=10.0, refuse=both)) == CONTINUE


def test_strikes_sets_how_many_consecutive_consults_a_braking_deficit_needs():
    rules = braking(strikes=3)
    assert rules.decide(situation("stopped", FAST, t=0.0)) == CONTINUE
    assert rules.decide(situation("stopped", FAST, t=10.0)) == CONTINUE
    assert rules.decide(situation("stopped", FAST, t=20.0)) == RETREAT


def test_one_strike_orders_the_retreat_at_the_first_consult():
    assert braking(strikes=1).decide(situation("holding", FAST)) == RETREAT


def test_a_consult_where_the_closing_speed_fell_resets_the_strikes():
    rules = braking(strikes=3)
    rules.decide(situation("stopped", at_speed(0.30), t=0.0))
    rules.decide(situation("stopped", at_speed(0.31), t=10.0))
    assert rules.decide(situation("stopped", at_speed(0.29), t=20.0)) == CONTINUE
    assert rules.decide(situation("stopped", at_speed(0.29), t=30.0)) == CONTINUE
    assert rules.decide(situation("stopped", at_speed(0.29), t=40.0)) == RETREAT


def test_dropping_below_the_threshold_resets_the_strikes():
    rules = braking()
    rules.decide(situation("stopped", at_speed(0.30), t=0.0))
    rules.decide(situation("stopped", at_speed(0.10), t=10.0))
    assert rules.decide(situation("stopped", at_speed(0.30), t=20.0)) == CONTINUE
    assert rules.decide(situation("stopped", at_speed(0.30), t=30.0)) == RETREAT


def test_leaving_the_stopped_and_holding_phases_resets_the_strikes():
    rules = braking()
    rules.decide(situation("stopped", FAST, t=0.0))
    rules.decide(situation("transfer", FAST, t=10.0))
    assert rules.decide(situation("stopped", FAST, t=20.0)) == CONTINUE


def test_a_new_episode_resets_the_strikes():
    rules = braking()
    rules.decide(situation("stopped", FAST, t=0.0))
    rules.reset(view_of("hold", LIMIT), LADDER)
    assert rules.decide(situation("stopped", FAST, t=10.0)) == CONTINUE
    assert rules.decide(situation("stopped", FAST, t=20.0)) == RETREAT


def test_the_braking_rule_does_not_apply_in_the_transfer_phase():
    rules = braking()
    assert rules.decide(situation("transfer", FAST, t=0.0)) == HOLD
    assert rules.decide(situation("transfer", FAST, t=10.0)) == HOLD


@pytest.mark.parametrize("phase", ["retreating", "inhibited", "aborted"])
def test_the_braking_rule_does_not_apply_outside_stopped_and_holding(phase):
    rules = braking()
    rules.decide(situation(phase, FAST, t=0.0))
    assert rules.decide(situation(phase, FAST, t=10.0)) == CONTINUE


def test_the_contact_rung_is_exempt_from_the_braking_rule():
    rules = supervisor("dock", LIMIT)
    rules.decide(situation("stopped", FAST, kind="contact", t=0.0))
    assert rules.decide(situation("stopped", FAST, kind="contact", t=10.0)) != RETREAT


def test_the_commit_phase_is_exempt_from_the_braking_rule():
    rules = braking()
    rules.decide(situation("commit", FAST, t=0.0))
    assert rules.decide(situation("commit", FAST, t=10.0)) == CONTINUE


def test_the_braking_rule_beats_the_go_ahead_of_a_settled_holding_chaser():
    rules = braking(settle_s=10.0, strikes=3)
    rules.decide(situation("holding", FAST, t=0.0))
    assert rules.decide(situation("holding", FAST, t=10.0)) == GO
    assert rules.decide(situation("holding", FAST, t=20.0)) == RETREAT


def test_the_predicted_collision_rule_beats_the_braking_rule():
    assert supervisor().decide(situation("stopped", ENTERS, t=0.0)) == CAM


def test_the_commit_inhibit_beats_the_braking_rule():
    rules = supervisor("dock", LIMIT)
    rules.decide(situation("commit", FAST, t=0.0))
    assert rules.decide(situation("commit", FAST, t=10.0)) == INHIBIT


@pytest.mark.parametrize("name", ["warning_s", "strikes", "brake_margin"])
@pytest.mark.parametrize(
    "value", [0, 0.0, -1, -1.0, math.nan, math.inf, -math.inf, True, "5", None]
)
def test_a_parameter_that_is_not_a_positive_finite_number_is_refused(name, value):
    with pytest.raises(ValueError):
        RuleSupervisor(**{name: value})


@pytest.mark.parametrize("value", [1.5, 0.5])
def test_strikes_must_be_a_whole_number(value):
    with pytest.raises(ValueError):
        RuleSupervisor(strikes=value)


def test_a_positive_integer_strikes_and_a_positive_warning_are_accepted():
    rules = RuleSupervisor(warning_s=30, strikes=4, brake_margin=2)
    assert rules.warning_s == 30.0
    assert rules.strikes == 4
    assert rules.brake_margin == 2.0


CORE = tuple(
    entry.scenario for entry in SUITES["core-v1"].entries if not entry.scenario.endswith(":nav1")
)


@pytest.mark.slow
@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("name", CORE)
def test_the_abort_rules_leave_a_nominal_approach_to_succeed_without_a_keep_out_violation(
    name, index
):
    env = make(name)
    seed, agent_seed = seeds(name, "validation", 3).pairs()[index]
    agent = SupervisedAgent(RuleSupervisor(), tracker=SetpointTracker())
    result = run_episode(env, agent, seed=seed, agent_seed=agent_seed)
    info = env._info()
    assert info["success"], info
    assert not result["keep_out_violated"]
