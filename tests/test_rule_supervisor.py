"""The rule supervisor, one rule at a time on constructed situations.

Each case builds the `Situation` a rule reads (phase, rung kind, observation, mask, time) and
checks the command it gives; no flight is run. The view has a 50 m keep-out sphere and a constant
closing-speed law of 0.2 m/s, so the speed cap is 0.2 m/s at every range and the overspeed
threshold of the defaults is 0.24 m/s. The chaser sits on the V-bar below the target, so its
closing speed is the y velocity.
"""

import numpy as np
import pytest

from vbar.agents.base import AgentView
from vbar.agents.rules import RuleSupervisor
from vbar.agents.supervision import Command, Rung, Situation

CONTINUE, GO, HOLD, CAM, INHIBIT = (
    Command.CONTINUE,
    Command.GO,
    Command.HOLD,
    Command.CAM,
    Command.THRUST_INHIBIT,
)

KEEP_OUT = 50.0
CAP = 0.2

LADDER = (
    Rung((0.0, -200.0, 0.0), "hold", 5.0, 0.05),
    Rung((0.0, -100.0, 0.0), "goal", 5.0, 0.05),
)
DOCK_LADDER = (
    Rung((0.0, -60.0, 0.0), "hold", 5.0, 0.05),
    Rung((0.0, -2.0, 0.0), "contact", 0.1, 0.1),
)


def view_of(task, constraints=None):
    if constraints is None:
        constraints = (
            {"type": "keep_out", "radius": KEEP_OUT},
            {"type": "closing_speed_limit", "v0": CAP, "v1": 0.0},
        )
    return AgentView(
        scenario_id="rules-test",
        dt=10.0,
        max_steps=100,
        n=0.0011,
        action_mode="continuous",
        max_accel=0.01,
        thrust_limit="box",
        goal={"task": task},
        constraints=constraints,
    )


def supervisor(task="hold", constraints=None, **parameters):
    rules = RuleSupervisor(**parameters)
    rules.reset(view_of(task, constraints), LADDER)
    return rules


def situation(phase, *, rho=100.0, closing=0.0, kind="goal", t=0.0, refuse=()):
    """A situation at range `rho` (m) on the V-bar, closing at `closing` (m/s)."""
    mask = [True] * len(Command)
    mask[Command.RECONFIGURE] = False
    for command in refuse:
        mask[command] = False
    rungs = DOCK_LADDER if kind == "contact" else LADDER
    rung = 1 if kind in ("goal", "contact") else 0
    return Situation(
        t=t,
        obs=np.array([0.0, -rho, 0.0, 0.0, closing, 0.0]),
        phase=phase,
        rung=rung,
        ladder=rungs,
        mask=tuple(mask),
        reasons={},
        refused=None,
        info={},
    )


def test_the_defaults_are_a_settle_time_of_60_s_and_an_overspeed_factor_of_1_2():
    rules = RuleSupervisor()
    assert rules.settle_s == 60.0
    assert rules.overspeed == 1.2


def test_a_chaser_inside_the_commit_radius_of_a_docking_is_told_to_inhibit_thrust():
    assert supervisor("dock").decide(situation("commit")) == INHIBIT


def test_the_commit_phase_of_a_task_other_than_a_docking_does_not_inhibit_thrust():
    assert supervisor("hold").decide(situation("commit")) == CONTINUE


def test_a_docking_outside_the_commit_phase_does_not_inhibit_thrust():
    assert supervisor("dock").decide(situation("transfer")) == CONTINUE


def test_thrust_is_not_inhibited_when_the_mask_refuses_it():
    assert supervisor("dock").decide(situation("commit", refuse=(INHIBIT,))) == CONTINUE


@pytest.mark.parametrize(("rho", "expected"), [(54.0, CAM), (56.0, CONTINUE)])
def test_a_chaser_closing_inside_1_1_keep_out_radii_is_told_to_fire_a_cam_and_not_outside(
    rho, expected
):
    # 1.1 x 50 m = 55 m
    assert supervisor().decide(situation("transfer", rho=rho, closing=0.05)) == expected


def test_a_chaser_that_is_opening_inside_the_cam_range_is_not_told_to_fire_a_cam():
    assert supervisor().decide(situation("transfer", rho=54.0, closing=-0.05)) == CONTINUE


def test_a_chaser_at_rest_inside_the_cam_range_is_not_told_to_fire_a_cam():
    assert supervisor().decide(situation("transfer", rho=54.0, closing=0.0)) == CONTINUE


def test_the_contact_rung_is_exempt_from_the_cam_rule():
    rules = supervisor("dock")
    assert rules.decide(situation("transfer", rho=54.0, closing=0.05, kind="contact")) == CONTINUE


def test_a_cam_is_not_ordered_when_the_mask_refuses_it():
    sit = situation("transfer", rho=54.0, closing=0.05, refuse=(CAM,))
    assert supervisor().decide(sit) == CONTINUE


def test_a_view_without_a_keep_out_sphere_never_gets_a_cam():
    limit = ({"type": "closing_speed_limit", "v0": CAP, "v1": 0.0},)
    rules = supervisor(constraints=limit)
    assert rules.decide(situation("transfer", rho=5.0, closing=0.05)) == CONTINUE


@pytest.mark.parametrize(
    ("closing", "expected"),
    [(0.25, HOLD), (0.23, CONTINUE), (-0.5, CONTINUE)],
    ids=["above-1.2-caps", "between-1-and-1.2-caps", "opening"],
)
def test_a_transferring_chaser_closing_faster_than_1_2_speed_caps_is_told_to_hold(
    closing, expected
):
    assert supervisor().decide(situation("transfer", closing=closing)) == expected


def test_overspeed_scales_the_threshold_on_the_speed_cap():
    assert supervisor(overspeed=2.0).decide(situation("transfer", closing=0.25)) == CONTINUE
    assert supervisor(overspeed=2.0).decide(situation("transfer", closing=0.41)) == HOLD


def test_an_overspeeding_chaser_is_not_told_to_hold_when_the_mask_refuses_it():
    sit = situation("transfer", closing=0.25, refuse=(HOLD,))
    assert supervisor().decide(sit) == CONTINUE


def test_overspeed_outside_the_transfer_phase_is_not_answered_with_a_hold():
    assert supervisor().decide(situation("retreating", closing=0.25)) == CONTINUE


def test_a_stopped_chaser_below_the_overspeed_threshold_is_told_to_go():
    assert supervisor().decide(situation("stopped", closing=0.23)) == GO


def test_a_stopped_chaser_still_closing_above_the_threshold_is_not_told_to_go():
    assert supervisor().decide(situation("stopped", closing=0.25)) == CONTINUE


def test_a_stopped_chaser_is_not_told_to_go_when_the_mask_refuses_it():
    assert supervisor().decide(situation("stopped", refuse=(GO,))) == CONTINUE


def test_a_holding_chaser_is_told_to_go_once_the_arrival_has_been_held_for_60_s():
    rules = supervisor()
    assert rules.decide(situation("holding", t=100.0)) == CONTINUE
    assert rules.decide(situation("holding", t=159.9)) == CONTINUE
    assert rules.decide(situation("holding", t=160.0)) == GO


def test_settle_s_sets_how_long_the_arrival_is_held_before_the_go_ahead():
    rules = supervisor(settle_s=10.0)
    assert rules.decide(situation("holding", t=0.0)) == CONTINUE
    assert rules.decide(situation("holding", t=9.9)) == CONTINUE
    assert rules.decide(situation("holding", t=10.0)) == GO


def test_losing_the_go_ahead_restarts_the_settle_time():
    rules = supervisor()
    assert rules.decide(situation("holding", t=0.0)) == CONTINUE
    assert rules.decide(situation("holding", t=30.0, refuse=(GO,))) == CONTINUE
    assert rules.decide(situation("holding", t=60.0)) == CONTINUE
    assert rules.decide(situation("holding", t=119.9)) == CONTINUE
    assert rules.decide(situation("holding", t=120.0)) == GO


def test_leaving_the_holding_phase_restarts_the_settle_time():
    rules = supervisor()
    assert rules.decide(situation("holding", t=0.0)) == CONTINUE
    assert rules.decide(situation("transfer", t=30.0)) == CONTINUE
    assert rules.decide(situation("holding", t=60.0)) == CONTINUE
    assert rules.decide(situation("holding", t=119.9)) == CONTINUE
    assert rules.decide(situation("holding", t=120.0)) == GO


def test_a_new_episode_starts_the_settle_time_afresh():
    rules = supervisor()
    assert rules.decide(situation("holding", t=0.0)) == CONTINUE
    rules.reset(view_of("hold"), LADDER)
    assert rules.decide(situation("holding", t=60.0)) == CONTINUE


@pytest.mark.parametrize("phase", ["transfer", "inhibited", "aborted"])
def test_a_situation_no_rule_covers_gets_continue(phase):
    assert supervisor().decide(situation(phase)) == CONTINUE


def test_the_commit_rule_beats_the_cam_rule():
    sit = situation("commit", rho=54.0, closing=0.05)
    assert supervisor("dock").decide(sit) == INHIBIT


def test_the_cam_rule_beats_the_overspeed_rule():
    sit = situation("transfer", rho=54.0, closing=0.25)
    assert supervisor().decide(sit) == CAM


def test_the_cam_rule_beats_the_go_ahead_of_a_stopped_chaser():
    sit = situation("stopped", rho=54.0, closing=0.05)
    assert supervisor().decide(sit) == CAM


def test_the_cam_rule_beats_the_go_ahead_of_a_settled_holding_chaser():
    rules = supervisor()
    assert rules.decide(situation("holding", t=0.0)) == CONTINUE
    assert rules.decide(situation("holding", rho=54.0, closing=0.05, t=60.0)) == CAM
