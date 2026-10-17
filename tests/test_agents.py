import dataclasses
import re

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import LLMAgent, PDAgent, action_scale
from vbar.agents.base import AgentView
from vbar.agents.llm import text_model
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
    result = run_episode(ApproachEnv(ApproachConfig(max_steps=5)), spy, seed=0)
    view, rng = spy.reset_args
    assert isinstance(view, AgentView) and isinstance(rng, np.random.Generator)
    assert view.goal["hold_point"] == (0.0, -100.0, 0.0)
    assert view.constraints == ({"type": "keep_out", "radius": 50.0},)
    assert spy.info_keys == {"time_s"}
    # The stream is a root seeded by the agent seed alone, not a child of the episode seed.
    assert rng.bit_generator.seed_seq.entropy == result["agent_seed"]
    assert rng.bit_generator.seed_seq.spawn_key == ()


NOISE = dict(sigma_pos=10.0, sigma_vel=0.01, bias_pos=5.0, bias_vel=0.005)


class SiblingCheat:
    """Rebuilds the episode's streams from the seed sequence behind its own generator."""

    name = "sibling-cheat"

    def reset(self, view, rng=None):
        seq = rng.bit_generator.seed_seq
        sibling = np.random.SeedSequence(
            seq.entropy, spawn_key=(*seq.spawn_key[:-1], 0), pool_size=seq.pool_size
        )
        self.replica = GaussianNav(**NOISE)
        self.replica.reset(np.random.default_rng(sibling))
        master = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seq.entropy)))
        self.start_guess = ApproachEnv()._sample_start(master)

    def denoise(self, obs, info):
        return obs - self.replica.observe(np.zeros(6), info["time_s"])


def cheat_errors(rng, seed):
    """Truth recovered by the cheat minus the truth, over 20 steps, and its start-state error."""
    env = ApproachEnv(ApproachConfig(sensor=GaussianNav(**NOISE)))
    obs, info = env.reset(seed=seed)
    cheat = SiblingCheat()
    cheat.reset(env.agent_view(), rng)
    start_error = np.linalg.norm(cheat.start_guess - info["true_state"])
    errors = []
    for _ in range(20):
        errors.append(cheat.denoise(obs, env.agent_info()) - info["true_state"])
        obs, _, _, _, info = env.step(np.zeros(3))
    return np.array(errors), start_error


def test_agent_stream_reveals_neither_the_sensor_noise_nor_the_start_state():
    # A child of the episode seed, as an agent stream spawned by the environment would be,
    # gives the cheat the truth to rounding, so the attack itself works.
    child = np.random.default_rng(np.random.SeedSequence(7, spawn_key=(3,)))
    errors, start_error = cheat_errors(child, 7)
    assert np.max(np.abs(errors)) < 1e-9 and start_error < 1e-9
    # The runner's stream leaves the cheat with two independent noise draws, sqrt(2) x 10 m.
    spy = Spy()
    run_episode(ApproachEnv(ApproachConfig(max_steps=1)), spy, seed=7)
    errors, start_error = cheat_errors(spy.reset_args[1], 7)
    assert np.median(np.abs(errors[:, :3])) > 1.0 and start_error > 1.0


class RandomAgent:
    name = "random"

    def reset(self, view, rng=None):
        self.rng = rng

    def act(self, obs, info):
        return self.rng.uniform(-1.0, 1.0, 3)


def test_agent_seed_reproduces_the_run_and_is_recorded():
    env = ApproachEnv(ApproachConfig(max_steps=20))
    first = run_episode(env, RandomAgent(), seed=3, agent_seed=11)
    again = run_episode(env, RandomAgent(), seed=3, agent_seed=11)
    other = run_episode(env, RandomAgent(), seed=3, agent_seed=12)
    assert first["agent_seed"] == 11
    assert np.array_equal(first["actions"], again["actions"])
    assert np.array_equal(first["states"], again["states"])
    assert not np.array_equal(first["actions"], other["actions"])
    drawn = run_episode(env, RandomAgent(), seed=3)
    replayed = run_episode(env, RandomAgent(), seed=3, agent_seed=drawn["agent_seed"])
    assert np.array_equal(drawn["actions"], replayed["actions"])


def test_llm_prompt_is_built_from_the_observation_not_the_truth():
    cfg = ApproachConfig(start_state=(0, 400, 0, 0, 0, 0), max_steps=3, sensor=GaussianNav(5.0))
    env = ApproachEnv(cfg)
    agent = LLMAgent(text_model(lambda prompt: '{"ax": 0, "ay": 0, "az": 0}'))
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


def test_pd_agent_commands_newtons_in_thrust_action_mode():
    # Under constant thrust a command in newtons and the same fraction of a full command give the
    # same thrust, so the two runs must fly the same trajectory.
    start = (-150.0, -400.0, 0.0, 0.0, 0.0, 0.0)
    base = dict(start_state=start, drive="thrust", propellant_kg=100.0)
    normalized = run_episode(ApproachEnv(ApproachConfig(**base)), PDAgent())
    thrust = run_episode(ApproachEnv(ApproachConfig(action_mode="thrust", **base)), PDAgent())
    assert thrust["captured"]
    assert np.allclose(normalized["states"], thrust["states"], rtol=0.0, atol=1e-9)
    assert np.allclose(10.0 * normalized["actions"], thrust["actions"], rtol=1e-14, atol=0.0)


def test_agents_read_only_the_navigated_state_when_channels_are_on():
    channels = ("t_remaining", "propellant_fraction", "pending_commands")
    start = (-150.0, -400.0, 0.0, 0.0, 0.0, 0.0)
    plain = run_episode(ApproachEnv(ApproachConfig(start_state=start, delay_s=5.0)), PDAgent())
    cfg = ApproachConfig(start_state=start, delay_s=5.0, observation_channels=channels)
    extended = run_episode(ApproachEnv(cfg), PDAgent())
    assert np.array_equal(plain["states"], extended["states"])
    llm = LLMAgent(text_model(lambda prompt: '{"ax": 0, "ay": 0, "az": 0}'))
    assert run_episode(ApproachEnv(cfg), llm)["dv_used_mps"] == 0.0


@pytest.mark.parametrize(
    ("mode", "scale"), [("normalized", 1.0), ("accel", 0.02), ("thrust", 10.0)]
)
def test_llm_fractions_are_converted_to_the_action_mode(mode, scale):
    agent = LLMAgent(text_model(lambda prompt: '{"ax": 0.5, "ay": 0, "az": 0}'))
    cfg = ApproachConfig(start_state=(0, 400, 0, 0, 0, 0), max_steps=3, action_mode=mode)
    result = run_episode(ApproachEnv(cfg), agent)
    assert np.array_equal(result["actions"][0], [0.5 * scale, 0.0, 0.0])
    # Half thrust for 30 s; in thrust mode the mass falls by 1.4e-4 of itself meanwhile.
    assert result["dv_used_mps"] == pytest.approx(0.3, rel=2e-4)


def test_unsupported_action_modes_are_rejected():
    view = ApproachEnv(ApproachConfig(action_mode="thrust")).agent_view()
    with pytest.raises(ValueError, match="mass_kg"):
        PDAgent().reset(dataclasses.replace(view, mass_kg=None))
    with pytest.raises(ValueError, match="action mode"):
        action_scale(dataclasses.replace(view, action_mode="impulse"))
