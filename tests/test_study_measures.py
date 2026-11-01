"""The supervisor study's episode measures: outcome, decision points, disagreement, chattering.

Flights use approach-vbar cut to 12 steps (120 s) with a consult at every step, so a run is a
second or two; the measures are counts over consults and do not need a whole approach.
"""

import math

import numpy as np
import pytest
from supervision_support import Always, Cycle

from vbar.agents import (
    BudgetedModel,
    LLMSupervisor,
    Price,
    Reply,
    RuleSupervisor,
    SupervisedAgent,
)
from vbar.agents.supervision import Command, Situation
from vbar.eval.study import (
    Consult,
    StudyAgent,
    StudyTask,
    WallLimitReached,
    chattering,
    decision_point,
    disagreement,
    outcome,
    study_metrics,
)
from vbar.metrics import record_episode
from vbar.safety import FilteredAgent, FilterResult, PassThroughFilter
from vbar.scenarios import make, seeds

SEED, AGENT_SEED = seeds("approach-vbar", "validation", 1).pairs()[0]


def short_env():
    return make("approach-vbar", max_steps=12)


def fly(agent, safety_filter=None):
    env, task = short_env(), StudyTask()
    pilot = agent if safety_filter is None else FilteredAgent(agent, safety_filter)
    recorded = task.record(env, pilot, SEED, AGENT_SEED)
    return recorded, task.score(env, recorded)


class Halving:
    """Halves every command and reports the change."""

    name = "halving"

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(0.5 * np.asarray(action, dtype=float), True)


@pytest.mark.parametrize(
    ("success", "entered", "aborted", "expected"),
    [
        (True, False, False, "success"),
        (True, False, True, "success"),
        (False, False, True, "safe_abort"),
        (False, False, False, "timeout"),
        (True, True, False, "unsafe"),
        (False, True, True, "unsafe"),
        (False, True, False, "unsafe"),
        (True, True, True, "unsafe"),
    ],
)
def test_a_keep_out_entry_is_unsafe_whatever_else_happened(success, entered, aborted, expected):
    assert outcome(success, entered, aborted) == expected


def situation(phase, mask_on, r, v):
    mask = [False] * len(Command)
    mask[Command.CONTINUE] = True
    for command in mask_on:
        mask[command] = True
    return Situation(
        t=0.0,
        obs=np.array([*r, *v], dtype=float),
        phase=phase,
        rung=0,
        ladder=(),
        mask=tuple(mask),
        reasons={},
        refused=None,
        info={},
    )


CLOSING_NEAR = ((0.0, 140.0, 0.0), (0.0, -0.1, 0.0))


@pytest.mark.parametrize(
    ("phase", "mask_on", "state", "radius", "expected"),
    [
        ("holding", [Command.GO], ((0.0, 400.0, 0.0), (0.0, 0.0, 0.0)), 50.0, True),
        ("stopped", [Command.GO], ((0.0, 400.0, 0.0), (0.0, 0.0, 0.0)), 50.0, True),
        ("transfer", [Command.GO], ((0.0, 400.0, 0.0), (0.0, 0.0, 0.0)), 50.0, False),
        ("transfer", [Command.CAM], CLOSING_NEAR, 50.0, True),
        ("transfer", [Command.RETREAT], CLOSING_NEAR, 50.0, True),
        ("transfer", [Command.HOLD], CLOSING_NEAR, 50.0, False),
        # 150 m is three radii: the range is not inside it.
        ("transfer", [Command.CAM], ((0.0, 150.0, 0.0), (0.0, -0.1, 0.0)), 50.0, False),
        ("transfer", [Command.CAM], ((0.0, 140.0, 0.0), (0.0, 0.1, 0.0)), 50.0, False),
        ("transfer", [Command.CAM], CLOSING_NEAR, 0.0, False),
        # Moving across the line of sight neither closes nor opens the range.
        ("transfer", [Command.CAM], ((0.0, 140.0, 0.0), (0.1, 0.0, 0.0)), 50.0, False),
    ],
)
def test_a_decision_point_is_a_go_at_a_rung_or_an_abort_while_closing_near(
    phase, mask_on, state, radius, expected
):
    assert decision_point(situation(phase, mask_on, *state), radius) is expected


def test_disagreement_counts_consults_and_decision_points_that_differ():
    consults = [
        Consult(0.0, Command.CONTINUE, Command.CONTINUE, False),
        Consult(60.0, Command.HOLD, Command.CONTINUE, False),
        Consult(120.0, Command.GO, Command.GO, True),
        Consult(180.0, Command.HOLD, Command.GO, True),
        Consult(240.0, None, Command.CONTINUE, False),
    ]
    found = disagreement(consults)
    assert found["consults"] == 5 and found["consults_differing"] == 3
    assert found["decision_points"] == 2 and found["decision_points_differing"] == 1
    assert found["disagreement"] == pytest.approx(3 / 5)
    assert found["decision_point_share"] == pytest.approx(2 / 5)
    assert found["decision_point_disagreement"] == pytest.approx(1 / 2)
    assert found["first_difference_s"] == 60.0


def test_disagreement_without_consults_or_differences_is_nan_and_none():
    empty = disagreement([])
    assert math.isnan(empty["disagreement"]) and math.isnan(empty["decision_point_share"])
    assert math.isnan(empty["decision_point_disagreement"]) and empty["first_difference_s"] is None
    same = disagreement([Consult(0.0, Command.GO, Command.GO, False)])
    assert same["disagreement"] == 0.0 and same["first_difference_s"] is None
    assert math.isnan(same["decision_point_disagreement"])


def test_disagreement_needs_a_shadow_at_every_consult():
    with pytest.raises(ValueError, match="shadow"):
        disagreement([Consult(0.0, Command.GO, None, False)])


def test_a_flip_is_a_command_differing_from_the_last_command_that_was_not_continue():
    commands = [
        Command.GO,
        Command.CONTINUE,
        Command.GO,
        Command.HOLD,
        None,
        Command.CONTINUE,
        Command.HOLD,
        Command.GO,
    ]
    consults = [Consult(60.0 * i, c, None, False) for i, c in enumerate(commands)]
    # GO -> HOLD and HOLD -> GO; the repeated GO and HOLD, CONTINUE and the non-command are not.
    assert chattering(consults, 400) == (2, 0.5)
    assert chattering([], 1) == (0, 0.0)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True, "12"])
def test_chattering_refuses_a_step_count_that_is_not_a_positive_integer(steps):
    with pytest.raises(ValueError, match="steps"):
        chattering([], steps)


@pytest.mark.parametrize(
    "fields",
    [
        {"shadow": 1},
        {"shadow": "yes"},
        {"wall_limit_s": 0.0},
        {"wall_limit_s": -1.0},
        {"wall_limit_s": math.nan},
        {"wall_limit_s": math.inf},
        {"wall_limit_s": True},
        {"wall_limit_s": "60"},
        {"period_s": math.nan},
        {"mode": "thrust"},
    ],
)
def test_a_study_agent_refuses_bad_settings(fields):
    with pytest.raises(ValueError):
        StudyAgent(RuleSupervisor(), **fields)


class Accepting:
    """A plain SupervisedAgent that keeps whether it accepted each consult's order."""

    def __init__(self, agent):
        self.agent, self.name, self.accepted = agent, agent.name, []

    def reset(self, view, rng=None):
        self.agent.reset(view, rng)

    def act(self, obs, info):
        action = self.agent.act(obs, info)
        if self.agent.last_step.consulted:
            self.accepted.append(self.agent.last_step.accepted)
        return action


def test_the_study_agent_keeps_each_consult_with_its_acceptance():
    (episode, consults, *_), _ = fly(StudyAgent(Cycle(), period_s=10.0))
    assert len(consults) == len(episode.records) == 12
    assert [c.command for c in consults] == [Cycle.ORDERS[i % 7] for i in range(12)]
    plain = Accepting(SupervisedAgent(Cycle(), period_s=10.0))
    record_episode(short_env(), plain, SEED, AGENT_SEED)
    assert [c.accepted for c in consults] == plain.accepted
    assert False in plain.accepted and True in plain.accepted


def test_the_shadow_is_never_flown():
    with_shadow = fly(StudyAgent(Always(Command.HOLD), shadow=True, period_s=10.0))[0][0]
    without = fly(StudyAgent(Always(Command.HOLD), shadow=False, period_s=10.0))[0][0]
    np.testing.assert_array_equal(with_shadow.states, without.states)
    assert [r.command.tolist() for r in with_shadow.records] == [
        r.command.tolist() for r in without.records
    ]


def test_the_shadow_gives_the_rule_supervisors_command_for_the_same_situation():
    recorded, metrics = fly(StudyAgent(RuleSupervisor(), shadow=True, period_s=10.0))
    consults = recorded[1]
    assert all(c.command == c.shadow for c in consults)
    assert metrics["disagreement"] == 0.0 and metrics["first_difference_s"] is None
    _, held = fly(StudyAgent(Always(Command.HOLD), shadow=True, period_s=10.0))
    assert held["consults_differing"] > 0 and held["first_difference_s"] == 0.0


def test_a_consult_past_the_wall_limit_raises():
    agent = StudyAgent(RuleSupervisor(), wall_limit_s=1e-9, period_s=10.0)
    with pytest.raises(WallLimitReached, match="limit"):
        fly(agent)


def test_the_task_flies_only_a_study_agent():
    env, task = short_env(), StudyTask()

    class Sub(StudyAgent):
        pass

    for agent in (SupervisedAgent(RuleSupervisor()), Sub(RuleSupervisor())):
        with pytest.raises(ValueError, match="StudyAgent"):
            task.record(env, agent, SEED, AGENT_SEED)


def test_the_filter_count_is_the_number_of_decisions_it_changed():
    _, halved = fly(StudyAgent(Always(Command.GO), period_s=10.0), Halving())
    assert halved["intervention_count"] == 12
    _, passed = fly(StudyAgent(Always(Command.GO), period_s=10.0), PassThroughFilter())
    assert passed["intervention_count"] == 0
    _, off = fly(StudyAgent(Always(Command.GO), period_s=10.0))
    assert math.isnan(off["intervention_count"])


def metrics_of(success=False, keep_out=10.0):
    return {"success": success, "min_margin": {"keep_out": keep_out}}


def test_only_an_accepted_abort_counts_as_an_abort():
    refused = [Consult(0.0, Command.CAM, None, False, accepted=False)]
    assert study_metrics(metrics_of(), refused, 10)["outcome"] == "timeout"
    for command in (Command.CAM, Command.RETREAT):
        accepted = [Consult(0.0, command, None, False, accepted=True)]
        found = study_metrics(metrics_of(), accepted, 10)
        assert found["aborted"] and found["outcome"] == "safe_abort"
    held = [Consult(0.0, Command.HOLD, None, False, accepted=True)]
    assert study_metrics(metrics_of(), held, 10)["outcome"] == "timeout"


def test_a_negative_keep_out_margin_is_an_entry_and_no_keep_out_is_none():
    assert study_metrics(metrics_of(True, -1e-9), [], 10)["outcome"] == "unsafe"
    no_sphere = {"success": True, "min_margin": {"speed_limit": -1.0}}
    assert study_metrics(no_sphere, [], 10)["outcome"] == "success"


def test_without_a_shadow_the_disagreement_fields_are_none():
    found = study_metrics(metrics_of(), [Consult(0.0, Command.GO, None, True)], 10)
    assert found["consults"] == 1 and found["decision_points"] == 1
    assert found["disagreement"] is None and found["first_difference_s"] is None


class Replying:
    """Replies with the given texts in turn, reporting 1000 input and 20 output tokens."""

    def __init__(self, *texts):
        self.texts, self.k = texts, 0

    def __call__(self, messages, tools):
        text = self.texts[self.k % len(self.texts)]
        self.k += 1
        return Reply(text, 1000, 20)


def test_model_usage_comes_from_the_budget_ledger_and_parse_failures_set_the_rate():
    price = Price(input_per_mtok=1.0, output_per_mtok=10.0)
    budget = BudgetedModel(
        Replying('{"command": "CONTINUE"}', "no json here"),
        price=price,
        max_usd=1.0,
        max_output_tokens=100,
    )
    supervisor = LLMSupervisor(budget, price=price)
    (_, consults, flown, _), metrics = fly(StudyAgent(supervisor, shadow=True, period_s=10.0))
    assert flown is supervisor and len(consults) == 12
    assert metrics["model_calls"] == 12 and metrics["input_tokens"] == 12_000
    assert metrics["cost_usd"] == pytest.approx(budget.spent_usd)
    assert metrics["cost_usd"] == pytest.approx(12 * price.cost(1000, 20))
    assert metrics["parse_failures"] == 6
    assert metrics["invalid_action_rate"] == pytest.approx(6 / 12)


def test_a_rule_supervisor_costs_nothing():
    _, metrics = fly(StudyAgent(RuleSupervisor(), period_s=10.0))
    assert metrics["model_calls"] == 0 and metrics["cost_usd"] == 0.0
    assert metrics["invalid_action_rate"] == 0.0


def test_each_episode_on_a_shared_budget_is_charged_its_own_calls():
    budget_price = Price(input_per_mtok=1.0, output_per_mtok=10.0)
    budget = BudgetedModel(
        Replying('{"command": "CONTINUE"}'),
        price=budget_price,
        max_usd=1.0,
        max_output_tokens=100,
    )
    # The supervisor declares another price; the ledger, not the supervisor, is the cost.
    supervisor = LLMSupervisor(budget, price=Price(input_per_mtok=5.0, output_per_mtok=50.0))
    agent = StudyAgent(supervisor, period_s=10.0)
    first = fly(agent)[1]["cost_usd"]
    second = fly(agent)[1]["cost_usd"]
    assert first == pytest.approx(12 * budget_price.cost(1000, 20))
    assert second == pytest.approx(first)
    assert budget.spent_usd == pytest.approx(2 * first)


def test_an_order_for_a_refused_command_is_invalid_but_not_a_parse_failure():
    # RECONFIGURE is never valid, so every reply is an invalid action that parsed.
    supervisor = LLMSupervisor(Replying('{"command": "RECONFIGURE"}'))
    _, metrics = fly(StudyAgent(supervisor, period_s=10.0))
    assert metrics["invalid_actions"] == 12 and metrics["parse_failures"] == 0
    assert metrics["invalid_action_rate"] == 0.0


class Mutating:
    """Orders HOLD after rewriting the situation it was shown: no valid command, another phase,
    and the chaser moved far away."""

    name = "mutating"

    def reset(self, view, ladder, rng=None):
        pass

    def decide(self, situation):
        object.__setattr__(situation, "mask", (True,) + (False,) * (len(Command) - 1))
        object.__setattr__(situation, "phase", "aborted")
        situation.obs.setflags(write=True)
        situation.obs[:] = 1e6
        return Command.HOLD


def test_a_supervisor_that_rewrites_its_situation_changes_neither_shadow_nor_decision_points():
    honest = fly(StudyAgent(Always(Command.HOLD), shadow=True, period_s=10.0))[0][1]
    cheat = fly(StudyAgent(Mutating(), shadow=True, period_s=10.0))[0][1]
    assert [c.shadow for c in cheat] == [c.shadow for c in honest]
    assert [c.decision_point for c in cheat] == [c.decision_point for c in honest]
    assert any(c.decision_point for c in honest) and any(c.shadow != Command.HOLD for c in honest)
