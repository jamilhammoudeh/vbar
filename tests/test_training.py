import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import DISC27, DriftAgent, LLMAgent, PDAgent, command_fraction, zero_action

START = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)


def fly(env, actions, seed=0):
    env.reset(seed=seed)
    states = [env.state.copy()]
    for a in actions:
        env.step(a)
        states.append(env.state.copy())
    return np.array(states)


def test_disc27_indices_fly_their_bang_off_bang_commands():
    indices = np.random.default_rng(0).integers(0, 27, 30)
    discrete = fly(ApproachEnv(ApproachConfig(start_state=START, action_mode="disc27")), indices)
    continuous = fly(ApproachEnv(ApproachConfig(start_state=START)), DISC27[indices])
    assert np.array_equal(discrete, continuous)
    assert np.array_equal(DISC27[13], np.zeros(3))
    assert len({tuple(c) for c in DISC27}) == 27


def test_disc27_is_its_own_scenario_variant_and_follows_the_gymnasium_api():
    env = ApproachEnv(ApproachConfig(action_mode="disc27"))
    assert env.scenario_id == "vbar:approach-vbar:cw:box:dv:v0:disc27"
    assert env.action_space.n == 27
    check_env(env, skip_render_check=True)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="0 to 26"):
        env.step(27)


def test_continuous_agents_reject_the_discrete_mode_and_drift_coasts():
    cfg = ApproachConfig(start_state=START, action_mode="disc27", max_steps=5)
    for agent in (PDAgent(), LLMAgent(lambda prompt: "{}")):
        with pytest.raises(ValueError, match="disc27"):
            run_episode(ApproachEnv(cfg), agent)
    drift = run_episode(ApproachEnv(cfg), DriftAgent())
    assert drift["dv_used_mps"] == 0.0 and np.all(drift["actions"] == 13)


@pytest.mark.parametrize(
    ("mode", "action", "fraction"),
    [
        ("normalized", [0.5, -1.0, 0.0], [0.5, -1.0, 0.0]),
        ("accel", [0.01, -0.02, 0.0], [0.5, -1.0, 0.0]),
        ("thrust", [5.0, -10.0, 0.0], [0.5, -1.0, 0.0]),
        ("disc27", 5, [-1.0, 0.0, 1.0]),
    ],
)
def test_command_fraction_reads_every_action_mode(mode, action, fraction):
    view = ApproachEnv(ApproachConfig(action_mode=mode)).agent_view()
    assert np.allclose(command_fraction(view, action), fraction, rtol=1e-15, atol=0.0)
    assert not np.any(command_fraction(view, zero_action(view)))
