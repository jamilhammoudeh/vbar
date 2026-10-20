import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.policy import MLPPolicy, PolicyAgent

# Observation scale of the approach scenario: hundreds of metres, tenths of a metre per second.
MEAN = np.array([0.0, 300.0, 0.0, 0.0, 0.0, 0.0])
STD = np.array([150.0, 200.0, 50.0, 0.1, 0.1, 0.1])


def policy(**options):
    return MLPPolicy.random(MEAN, STD, hidden=(64, 64), rng=np.random.default_rng(0), **options)


def observations(count=500, seed=1):
    rng = np.random.default_rng(seed)
    return MEAN + STD * rng.normal(size=(count, 6))


def test_policy_normalizes_then_squashes_into_fractions():
    p = policy()
    out = p(observations())
    assert out.shape == (500, 3) and np.all(np.abs(out) < 1.0)
    # Normalization is part of the policy: shifting the mean shifts the input it sees.
    moved = MLPPolicy(p.weights, p.biases, MEAN + STD, STD)
    assert np.allclose(moved(MEAN + STD), p(MEAN), rtol=0.0, atol=1e-15)
    clipped = policy(squash="clip", activation="relu")
    assert np.all(np.abs(clipped(10.0 * observations())) <= 1.0)


@pytest.mark.parametrize(
    ("weights", "message"),
    [((np.zeros((5, 3)),), "chain"), ((np.zeros((6, 4)),), "3 outputs")],
)
def test_layer_shapes_are_checked(weights, message):
    with pytest.raises(ValueError, match=message):
        MLPPolicy(weights, tuple(np.zeros(w.shape[1]) for w in weights), MEAN, STD)


def test_policy_agent_flies_in_every_continuous_mode():
    p = policy()
    start = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)
    normalized = run_episode(
        ApproachEnv(ApproachConfig(start_state=start, max_steps=20)), PolicyAgent(p)
    )
    accel = run_episode(
        ApproachEnv(ApproachConfig(start_state=start, max_steps=20, action_mode="accel")),
        PolicyAgent(p),
    )
    assert np.allclose(normalized["states"], accel["states"], rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_policy_agent_refuses_modes_without_a_continuous_command(mode):
    env = ApproachEnv(ApproachConfig(action_mode=mode, max_steps=2))
    with pytest.raises(ValueError, match=f"'{mode}' takes no continuous command"):
        run_episode(env, PolicyAgent(policy()))
