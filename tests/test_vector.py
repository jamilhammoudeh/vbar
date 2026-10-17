import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv
from vbar.agents import PDAgent
from vbar.rewards import QuadraticCost, Reward
from vbar.scenarios import ApproachVectorEnv
from vbar.spacecraft import ExecutionError, GaussianNav, GaussMarkovNav

HOLD = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
REWARD = Reward(
    tau=1800.0,
    success_bonus=10.0,
    fuel_weight=2.0,
    time_weight=1e-3,
    violation_weights={"keep_out": 0.1},
    entry_weights={"keep_out": 5.0},
    progress_weight=0.01,
    quadratic=QuadraticCost(ApproachEnv().n, np.diag([1e-6] * 3 + [1e-2] * 3), np.eye(3), HOLD),
)


class Mixed:
    """PD on even copies, a seeded random walk of commands on odd ones, read from the obs."""

    name = "mixed"

    def __init__(self, k, scale=1.0, discrete=False):
        self.k, self.scale, self.discrete = k, scale, discrete

    def reset(self, view, rng=None):
        self.pd = PDAgent()
        self.pd.reset(view if view.action_mode != "disc27" else _as_normalized(view))
        self.rng = np.random.default_rng(self.k)

    def act(self, obs, info):
        if self.discrete:
            return int(self.rng.integers(0, 27))
        if self.k % 2 == 0:
            return self.pd.act(obs, info)
        return self.scale * self.rng.uniform(-1.0, 1.0, 3)


def _as_normalized(view):
    import dataclasses

    return dataclasses.replace(view, action_mode="normalized")


def compare(cfg, num_envs=6, steps=80, seed=10, scale=1.0, discrete=False):
    """Fly the vector env and `num_envs` single envs with the same seeds and actions."""
    vec = ApproachVectorEnv(cfg, num_envs)
    singles = [ApproachEnv(cfg) for _ in range(num_envs)]
    agents = [Mixed(k, scale, discrete) for k in range(num_envs)]
    obs_v, info_v = vec.reset(seed=seed)
    obs_s = []
    for k, env in enumerate(singles):
        obs_s.append(env.reset(seed=seed + k)[0])
        agents[k].reset(env.agent_view())
    assert np.allclose(obs_v, np.array(obs_s), rtol=0.0, atol=0.0)
    endings = {"terminated": 0, "truncated": 0}
    for _ in range(steps):
        actions = [
            a.act(o, {"time_s": e.t}) for a, o, e in zip(agents, obs_s, singles, strict=True)
        ]
        obs_v, rew_v, term_v, trunc_v, info_v = vec.step(np.array(actions))
        for k, env in enumerate(singles):
            obs, rew, term, trunc, info = env.step(actions[k])
            assert (term_v[k], trunc_v[k]) == (term, trunc)
            assert rew_v[k] == pytest.approx(rew, rel=1e-9, abs=1e-9)
            row = info_v["final_info"] if (term or trunc) else info_v
            assert np.allclose(row["true_state"][k], info["true_state"], rtol=1e-9, atol=1e-9)
            for key in ("time_s", "dv_used_mps", "propellant_used_kg", "mass_kg"):
                assert row[key][k] == pytest.approx(info[key], rel=1e-12, abs=1e-12)
            for key in ("keep_out_violated", "violation_steps", "captured"):
                assert row[key][k] == info[key]
            margin = row["min_margins"]["keep_out"][k]
            assert margin == pytest.approx(info["min_margins"]["keep_out"], abs=1e-6)
            if "discount" in info:
                assert info_v["discount"][k] == pytest.approx(info["discount"], rel=1e-14)
            if term or trunc:
                endings["terminated" if term else "truncated"] += 1
                # The true final observation survives the same-step reset.
                assert info_v["_final_obs"][k]
                assert np.allclose(info_v["final_obs"][k], obs, rtol=1e-9, atol=1e-9)
                obs = env.reset()[0]
                agents[k].reset(env.agent_view())
            else:
                assert not info_v.get("_final_obs", np.zeros(num_envs, bool))[k]
            assert np.allclose(obs_v[k], obs, rtol=1e-9, atol=1e-9)
            obs_s[k] = obs
    return endings


def test_vector_env_matches_single_envs_through_captures_and_time_limits():
    # Starts near the hold point: PD copies capture (terminated), random ones run out the clock.
    cfg = ApproachConfig(
        start_pos_range=((-15.0, 15.0), (-115.0, -85.0), (-15.0, 15.0)),
        start_vel_range=((-0.02, 0.02),) * 3,
        max_steps=25,
    )
    endings = compare(cfg, steps=90, scale=0.2)
    assert endings["terminated"] >= 3 and endings["truncated"] >= 3


def test_vector_env_matches_single_envs_with_violations_rewards_and_run_out():
    # Starts closing on the target at about 1 m/s; a 0.05 kg tank runs dry within a few burns,
    # mostly inside a step, and entering the keep-out zone ends an episode.
    cfg = ApproachConfig(
        start_pos_range=((-60.0, 60.0), (200.0, 300.0), (-30.0, 30.0)),
        start_vel_range=((-0.05, 0.05), (-1.2, -0.8), (-0.05, 0.05)),
        max_steps=40,
        propellant_kg=0.05,
        terminate_on_violation=True,
        reward=REWARD,
        observation_channels=("t_remaining", "propellant_fraction"),
    )
    endings = compare(cfg, num_envs=6, steps=60)
    assert endings["terminated"] >= 1 and endings["truncated"] >= 1


@pytest.mark.parametrize(
    "fields",
    [
        dict(action_mode="accel", thrust_limit="ball", min_impulse=0.5, sensor=GaussianNav()),
        dict(action_mode="thrust", sensor=GaussMarkovNav(), reward=REWARD),
    ],
)
def test_vector_env_matches_single_envs_in_other_modes_and_sensors(fields):
    cfg = ApproachConfig(max_steps=40, **fields)
    compare(cfg, num_envs=4, steps=45, scale=0.3 if fields["action_mode"] == "accel" else 1.0)


def test_vector_env_matches_single_envs_with_discrete_actions():
    cfg = ApproachConfig(max_steps=30, action_mode="disc27", reward=REWARD)
    compare(cfg, num_envs=4, steps=35, discrete=True)


def test_seeds_are_per_copy_and_a_list_gives_each_its_own():
    vec = ApproachVectorEnv(ApproachConfig(), 3)
    obs, _ = vec.reset(seed=5)
    assert np.array_equal(obs, [ApproachEnv().reset(seed=5 + k)[0] for k in range(3)])
    obs, _ = vec.reset(seed=[9, 2, 7])
    assert np.array_equal(obs, [ApproachEnv().reset(seed=s)[0] for s in (9, 2, 7)])
    with pytest.raises(ValueError, match="expected 3 seeds"):
        vec.reset(seed=[1, 2])


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (dict(dynamics="twobody"), "dynamics"),
        (dict(delay_s=2.0), "delay"),
        (dict(drive="thrust", propellant_kg=100.0), "constant-thrust"),
        (dict(execution=ExecutionError(scale_sigma=0.01)), "execution"),
        (dict(action_mode="impulse_coast"), "impulse_coast"),
    ],
)
def test_unsupported_options_are_refused(fields, message):
    with pytest.raises(ValueError, match=message):
        ApproachVectorEnv(ApproachConfig(**fields), 2)
