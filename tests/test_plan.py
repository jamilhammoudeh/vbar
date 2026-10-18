import dataclasses
import json

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent, Reply
from vbar.agents.plan import LLMPlanner, PlanAgent, PlanVerifier, parse_plan
from vbar.dynamics import CWDynamics
from vbar.spacecraft import GaussianNav

CLEAR = (-150.0, -400.0, 0.0, 0.0, 0.0, 0.0)
THROUGH = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)
AROUND = [(120.0, 150.0, 0.0), (120.0, -100.0, 0.0)]


def rollout(points=()):
    """A classical planner: the PD agent flown on the CW model through waypoints to the goal."""

    def planner(obs, view, verdict, steps_left):
        pd = PDAgent()
        pd.reset(view)
        route = [np.array(p, dtype=float) for p in points] + [pd.hold.copy()]
        model = CWDynamics(view.n)
        model.reset(obs[:6])
        state, plan, k = np.asarray(obs[:6], dtype=float), [], 0
        for _ in range(steps_left):
            if k < len(route) - 1 and np.linalg.norm(state[:3] - route[k]) < 10.0:
                k += 1
            pd.hold = route[k]
            plan.append(pd.act(state, {}))
            state, _ = model.advance(plan[-1] * view.max_accel, view.dt)
            done = np.linalg.norm(state[:3] - route[-1]) < view.goal["capture_radius"]
            if done and np.linalg.norm(state[3:]) < view.goal["capture_speed"]:
                break
        return np.array(plan)

    return planner


class Replay:
    """Flies fixed rows of fractions open loop."""

    name = "replay"

    def __init__(self, rows):
        self.rows = rows

    def reset(self, view, rng=None):
        self.k = 0

    def act(self, obs, info):
        self.k += 1
        return self.rows[self.k - 1] if self.k <= len(self.rows) else np.zeros(3)


def first_verdict(start, planner, sensor=None):
    cfg = (
        ApproachConfig(start_state=start)
        if sensor is None
        else ApproachConfig(start_state=start, sensor=sensor)
    )
    env = ApproachEnv(cfg)
    obs, _ = env.reset(seed=0)
    view = env.agent_view()
    plan = planner(obs, view, None, view.max_steps)
    return PlanVerifier(view).verify(obs, plan, view.max_steps), plan, cfg


def test_verifier_predicts_what_the_environment_flies():
    verdict, plan, cfg = first_verdict(CLEAR, rollout())
    assert verdict.ok and verdict.capture_step == len(plan)
    flown = run_episode(ApproachEnv(cfg), Replay(plan))
    assert flown["captured"] and len(flown["actions"]) == verdict.capture_step
    # Same exact CW solution from the same state: agreement to rounding.
    assert flown["min_range_to_target_m"] - 50.0 == pytest.approx(
        verdict.min_margins["keep_out"], rel=1e-12
    )
    assert flown["final_dist_to_hold_m"] == pytest.approx(verdict.final_distance_m, abs=1e-9)
    assert flown["dv_used_mps"] == pytest.approx(verdict.delta_v_mps, rel=1e-12)


def test_verifier_rejects_a_plan_through_the_keep_out_zone_and_says_why():
    verdict, _, _ = first_verdict(THROUGH, rollout())
    assert not verdict.ok and verdict.min_margins["keep_out"] < -40.0
    assert any(p.startswith("keep_out violated from step") for p in verdict.problems)
    assert verdict.summary().startswith("Plan rejected:")
    verdict, _, _ = first_verdict(THROUGH, rollout(AROUND))
    assert verdict.ok and verdict.min_margins["keep_out"] > 0.0


def test_verifier_predicts_from_the_observation_it_is_given():
    noisy = GaussianNav(20.0, 0.02)
    env = ApproachEnv(ApproachConfig(start_state=CLEAR, sensor=noisy))
    obs, info = env.reset(seed=0)
    view = env.agent_view()
    plan = rollout()(obs, view, None, 50)
    from_obs = PlanVerifier(view).verify(obs, plan)
    from_truth = PlanVerifier(view).verify(np.r_[info["true_state"]], plan)
    assert abs(from_obs.final_distance_m - from_truth.final_distance_m) > 1.0


def test_plan_agent_revises_until_the_verifier_accepts():
    attempts = []

    def planner(obs, view, verdict, steps_left):
        attempts.append(verdict)
        points = () if verdict is None else AROUND
        return rollout(points)(obs, view, verdict, steps_left)

    agent = PlanAgent(planner)
    result = run_episode(ApproachEnv(ApproachConfig(start_state=THROUGH)), agent, seed=0)
    assert result["captured"] and not result["keep_out_violated"]
    assert agent.log[0]["verifier_calls"] == 2 and agent.log[0]["verdict"].ok
    assert attempts[0] is None and not attempts[1].ok
    assert len(agent.log) == 1


def test_a_plan_that_never_passes_is_not_flown():
    agent = PlanAgent(lambda obs, view, verdict, steps_left: np.zeros((0, 3)), max_revisions=2)
    result = run_episode(ApproachEnv(ApproachConfig(start_state=THROUGH, max_steps=4)), agent)
    assert result["dv_used_mps"] == 0.0
    assert [e["verifier_calls"] for e in agent.log] == [3, 3, 3, 3]
    assert "the plan is empty" in agent.log[0]["verdict"].problems


def test_llm_planner_sends_the_verdict_back_and_reads_segments():
    replies = [
        Reply(json.dumps([{"ax": 0.0, "ay": -1.0, "az": 0.0, "steps": 3}]), 400, 30),
        Reply("I cannot plan this."),
    ]
    prompts = []

    def complete(messages, tools):
        prompts.append(messages[0]["content"])
        return replies[min(len(prompts), 2) - 1]

    planner = LLMPlanner(complete)
    agent = PlanAgent(planner, max_revisions=1)
    run_episode(ApproachEnv(ApproachConfig(start_state=THROUGH, max_steps=1)), agent)
    assert "Your previous plan" not in prompts[0]
    assert "Plan rejected:" in prompts[1] and "At most 1 steps remain" in prompts[1]
    assert planner.log[0]["input_tokens"] == 400
    assert parse_plan('[{"ax": 0.5, "ay": 0, "az": 0, "steps": 2}]').tolist() == [[0.5, 0, 0]] * 2
    assert parse_plan("no plan").shape == (0, 3)


def test_plan_agent_rejects_the_discrete_mode():
    view = ApproachEnv(ApproachConfig(action_mode="disc27")).agent_view()
    with pytest.raises(ValueError, match="disc27"):
        PlanAgent(rollout()).reset(view)
    with pytest.raises(ValueError, match="cannot check"):
        PlanVerifier(dataclasses.replace(view, constraints=({"type": "cone"},)))


@pytest.mark.parametrize(
    "reply",
    [
        '[{"ax": 0, "ay": 0, "az": 0, "steps": 1e999}]',
        '[{"ax": ' + "9" * 400 + ', "ay": 0, "az": 0, "steps": 1}]',
        "[" * 3000 + "]" * 3000,
        '[{"ax": 0, "ay": 0, "az": 0, "steps": ' + "9" * 400 + "}]",
        '[{"ax": 0, "ay": 0, "az": 0, "steps": -3}]',
    ],
    ids=["infinite steps", "huge fraction", "deep nesting", "huge steps", "negative steps"],
)
def test_unusable_plan_replies_are_empty_plans(reply):
    assert parse_plan(reply).shape == (0, 3)


def test_plans_longer_than_the_time_left_are_refused_without_flying_them():
    assert parse_plan('[{"ax": 0, "ay": 0, "az": 0, "steps": 400}]', max_rows=360).shape == (0, 3)
    view = ApproachEnv(ApproachConfig(start_state=CLEAR)).agent_view()
    verdict = PlanVerifier(view).verify(np.array(CLEAR), np.zeros((100_000, 3)), steps_left=5)
    assert "only 5 remain" in " ".join(verdict.problems) and not verdict.ok
