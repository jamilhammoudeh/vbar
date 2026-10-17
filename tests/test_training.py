import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import DISC27, DriftAgent, LLMAgent, PDAgent, command_fraction, zero_action
from vbar.rewards import Reward

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


def stepwise_and_coasted(tau, seed=0):
    """The same burns and coasts flown one step per decision and as impulse_coast decisions."""
    rng = np.random.default_rng(seed)
    plan = [(rng.uniform(-0.3, 0.3, 3), int(rng.integers(0, 6))) for _ in range(8)]
    reward = Reward(
        tau=tau,
        fuel_weight=2.0,
        time_weight=1e-3,
        violation_weights={"keep_out": 0.1},
        entry_weights={"keep_out": 3.0},
    )
    base = dict(start_state=(3.0, 150.0, -2.0, 0.0, -0.6, 0.0), capture_radius=0.0, reward=reward)
    stepwise = ApproachEnv(ApproachConfig(**base))
    coasted = ApproachEnv(ApproachConfig(action_mode="impulse_coast", **base))
    stepwise.reset(seed=0)
    coasted.reset(seed=0)
    one, many = [], []
    for burn, coast in plan:
        total, weight = 0.0, 1.0
        for command in [burn] + [np.zeros(3)] * coast:
            _, r, _, _, info = stepwise.step(command)
            total += weight * r
            weight *= info["discount"]
        one.append((stepwise.state.copy(), total, weight, info["min_margins"]["keep_out"]))
        _, r, _, _, info = coasted.step(np.r_[burn, coast * 10.0])
        many.append((coasted.state.copy(), r, info["discount"], info["min_margins"]["keep_out"]))
    assert coasted.steps == stepwise.steps == sum(1 + coast for _, coast in plan)
    return one, many


@pytest.mark.parametrize("tau", [None, 900.0])
def test_impulse_coast_decision_equals_the_steps_it_replaces(tau):
    # Sutton and Barto 14.4: one semi-Markov decision earns the discounted sum of the rewards of
    # the steps it spans and is discounted by exp(-duration / tau).
    one, many = stepwise_and_coasted(tau)
    assert min(m[3] for m in many) < 0.0
    for (s1, r1, g1, m1), (s2, r2, g2, m2) in zip(one, many, strict=True):
        # One long coast against several short ones: same closed form, different rounding.
        assert np.allclose(s2, s1, rtol=0.0, atol=1e-11 * np.max(np.abs(s1)))
        assert r2 == pytest.approx(r1, rel=1e-9, abs=1e-12)
        assert g2 == pytest.approx(g1, rel=1e-14)
        assert m2 == pytest.approx(m1, abs=1e-6)


def test_coast_is_cut_at_the_time_limit():
    env = ApproachEnv(ApproachConfig(action_mode="impulse_coast", max_steps=20))
    env.reset(seed=0)
    _, _, _, truncated, info = env.step([0.0, 0.0, 0.0, 120.0])
    assert env.steps == 13 and info["time_s"] == 130.0 and not truncated
    _, _, _, truncated, info = env.step([0.0, 0.0, 0.0, 1e6])
    assert env.steps == 20 and info["time_s"] == 200.0 and truncated
    with pytest.raises(ValueError, match="four finite"):
        env.step([0.0, 0.0, 0.0, np.nan])


# The coast entry is in seconds, which the checker flags as an unnormalized box.
@pytest.mark.filterwarnings("ignore:.*symmetric and normalized")
def test_impulse_coast_follows_the_gymnasium_api_and_continuous_agents_reject_it():
    cfg = ApproachConfig(action_mode="impulse_coast", max_steps=50)
    check_env(ApproachEnv(cfg), skip_render_check=True)
    with pytest.raises(ValueError, match="impulse_coast"):
        run_episode(ApproachEnv(cfg), PDAgent())
    drift = run_episode(ApproachEnv(cfg), DriftAgent())
    assert drift["dv_used_mps"] == 0.0 and drift["actions"].shape == (50, 4)
