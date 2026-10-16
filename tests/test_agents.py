import re

import numpy as np

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import LLMAgent, PDAgent
from vbar.agents.base import AgentView
from vbar.spacecraft import GaussianNav


class Spy:
    """Records what the runner hands an agent."""

    name = "spy"

    def reset(self, view, rng=None):
        self.reset_args = (view, rng)
        self.info_keys = set()

    def act(self, obs, info):
        self.info_keys |= set(info)
        return np.zeros(3)


def test_agents_receive_only_the_view_a_random_stream_and_the_time():
    spy = Spy()
    run_episode(ApproachEnv(ApproachConfig(max_steps=5)), spy, seed=0)
    view, rng = spy.reset_args
    assert isinstance(view, AgentView) and isinstance(rng, np.random.Generator)
    assert view.goal["hold_point"] == (0.0, -100.0, 0.0)
    assert view.constraints == ({"type": "keep_out", "radius": 50.0},)
    assert spy.info_keys == {"time_s"}


def test_llm_prompt_is_built_from_the_observation_not_the_truth():
    cfg = ApproachConfig(start_state=(0, 400, 0, 0, 0, 0), max_steps=3, sensor=GaussianNav(5.0))
    env = ApproachEnv(cfg)
    agent = LLMAgent(lambda prompt: '{"ax": 0, "ay": 0, "az": 0}')
    obs, _ = env.reset(seed=1)
    agent.reset(env.agent_view())
    agent.act(obs, env.agent_info())
    prompt = agent.log[0]["prompt"]
    reported = float(re.search(r"distance to target = ([\d.]+) m", prompt).group(1))
    assert reported == round(float(np.linalg.norm(obs[:3])), 1)
    assert abs(reported - 400.0) > 0.05


def test_pd_agent_flies_the_same_in_accel_action_mode():
    start = (-150.0, -400.0, 0.0, 0.0, 0.0, 0.0)
    normalized = run_episode(ApproachEnv(ApproachConfig(start_state=start)), PDAgent())
    accel = run_episode(
        ApproachEnv(ApproachConfig(start_state=start, action_mode="accel")), PDAgent()
    )
    assert np.allclose(normalized["states"], accel["states"], rtol=0.0, atol=1e-9)
    assert np.allclose(0.02 * normalized["actions"], accel["actions"], rtol=1e-14, atol=0.0)
