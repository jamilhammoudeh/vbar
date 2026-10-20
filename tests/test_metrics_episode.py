import numpy as np
import pytest
from metric_cases import episode

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import LLMAgent, PDAgent
from vbar.agents.llm import text_model
from vbar.agents.plan import PlanAgent
from vbar.agents.tools import transfer_burn
from vbar.metrics import (
    evaluate,
    record_episode,
)
from vbar.safety import FilteredAgent


class HalvingFilter:
    name = "halve"

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return 0.5 * action


def test_recorded_states_match_run_episode_for_the_same_seeds():
    config = ApproachConfig(max_steps=30)
    recorded = record_episode(ApproachEnv(config), PDAgent(), seed=4, agent_seed=9)
    ran = run_episode(ApproachEnv(config), PDAgent(), seed=4, agent_seed=9)
    assert np.array_equal(recorded.states, ran["states"])
    assert recorded.hard == frozenset({"keep_out"})
    assert recorded.agent_seed == 9 and recorded.truncated and not recorded.terminated


def test_records_keep_the_raw_command_and_the_filter_result():
    env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0), max_steps=4))
    ep = record_episode(env, FilteredAgent(PDAgent(), HalvingFilter()), seed=0, agent_seed=1)
    for rec in ep.records:
        assert np.allclose(rec.filtered, 0.5 * rec.command)
        assert rec.filter_result.intervened
        assert rec.agent_latency_s >= 0.0


def test_an_llm_agent_log_is_kept():
    reply = '{"ax": 0.1, "ay": 0.0, "az": 0.0}'
    ep = episode(LLMAgent(text_model(lambda prompt: reply)), max_steps=3)
    assert len(ep.agent_log) == 3 and not ep.agent_log[0]["parse_failed"]


class SparseLogger(PDAgent):
    """Logs only its first decision, as a planner logs its planning points."""

    def reset(self, view, rng=None):
        super().reset(view, rng)
        self.log = []

    def act(self, obs, info):
        if not self.log:
            self.log.append({"time_s": info["time_s"]})
        return super().act(obs, info)


def test_a_log_that_is_not_per_decision_is_left_out_with_a_warning():
    with pytest.warns(UserWarning, match="1 entries for 3 decisions"):
        ep = episode(SparseLogger(), max_steps=3)
    assert ep.agent_log == ()
    assert np.isnan(evaluate(ep)["parse_failure_rate"])


def test_a_plan_agent_episode_is_scored():
    # A two-burn transfer to the hold point is accepted at the first decision and flown open loop.
    def planner(obs, view, verdict, steps_left):
        seconds = 600.0
        burns = transfer_burn(obs, view, {"x": 0, "y": -100, "z": 0, "seconds": seconds})
        k = int(seconds / view.dt)
        plan = np.zeros((k + 1, 3))
        plan[0] = np.array(burns["first_burn_mps"]) / view.dt / view.max_accel
        plan[k] = np.array(burns["second_burn_mps"]) / view.dt / view.max_accel
        return plan

    agent = PlanAgent(planner, max_revisions=0)
    env = ApproachEnv(ApproachConfig(start_state=(0, -200, 0, 0, 0, 0), max_steps=100))
    ep = record_episode(env, agent, seed=0, agent_seed=1)
    assert agent.planning_log[0]["verdict"].ok and len(ep.records) > 1
    assert ep.agent_log == ()
    assert evaluate(ep)["success"]
