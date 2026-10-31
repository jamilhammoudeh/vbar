import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from test_pursuit_evasion import FLOW, NaivePursuitAgent

import vbar  # noqa: F401  (registers the environments)
from vbar.eval import Job, run_jobs
from vbar.games import (
    GYMNASIUM_IDS,
    AgentEnvRunner,
    NormalEscapeEvader,
    PassiveEvader,
    PE1_E1_I3_Env,
    PE1_E3_I3_Env,
    PursuitEvasionGymEnv,
    PursuitTask,
)


def test_the_scenarios_fly_the_i3_start_against_their_bots():
    assert isinstance(PE1_E1_I3_Env().evader, PassiveEvader)
    assert isinstance(PE1_E3_I3_Env().evader, NormalEscapeEvader)
    assert PE1_E3_I3_Env().reset()[0] == PE1_E1_I3_Env().reset()[0]


def test_kspdg_example_runner_call_works_as_written():
    runner = AgentEnvRunner(
        agent=NaivePursuitAgent(),
        env_cls=PE1_E1_I3_Env,
        env_kwargs=None,
        runner_timeout=100,
        debug=True,
        enable_telemetry=False,
    )
    results = runner.run()["agent_env_results"]
    assert results["is_episode_done"] and results["steps"] == 240
    assert results["pursuer_fuel_usage"] == pytest.approx(240 * FLOW, rel=1e-12)


PE1_IDS = sorted(key for key in gym.registry if key.startswith("vbar/PursuitEvasion"))


def test_every_scenario_is_registered_under_its_gymnasium_id():
    assert PE1_IDS == sorted(GYMNASIUM_IDS.values())
    for key, env_id in GYMNASIUM_IDS.items():
        assert gym.spec(env_id).kwargs == {"scenario": key}


@pytest.mark.parametrize("env_id", PE1_IDS)
def test_the_scenarios_are_registered_and_pass_the_gymnasium_checker(env_id):
    env = gym.make(env_id, episode_timeout=30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env.unwrapped, skip_render_check=True)
    assert isinstance(env.unwrapped, PursuitEvasionGymEnv)
    observation, _ = env.reset(seed=4)
    assert observation.shape == (15,) and observation.dtype == np.float64


def test_gymnasium_terminates_at_capture_and_truncates_at_the_timeout():
    env = PursuitEvasionGymEnv("pe1-e1-i3", capture_dist=2000.0)
    env.reset()
    action = {"burn_vec": np.array([1.0, 0.0, 0.0, 10.0]), "ref_frame": 0, "vec_type": 0}
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(action)
    assert terminated and not truncated and reward < 0.0
    env = PursuitEvasionGymEnv("pe1-e1-i3", episode_timeout=10.0)
    env.reset()
    _, _, terminated, truncated, _ = env.step(
        {"burn_vec": np.zeros(4) + [0, 0, 0, 10.0], "ref_frame": 0, "vec_type": 0}
    )
    assert truncated and not terminated
    with pytest.raises(ValueError):
        PursuitEvasionGymEnv("pe1-e9-i3")


class Coast:
    def get_action(self, observation):
        return [0.0, 0.0, 0.0, 10.0]


def test_the_seed_changes_nothing():
    env = PursuitEvasionGymEnv("pe1-e3-i3")
    first = env.reset(seed=1)[0]
    assert np.array_equal(env.reset(seed=99)[0], first)
    task = PursuitTask()
    runs = []
    for seed, agent_seed in ((10**30, 10**31), (5, 6)):
        env = task.make("pe1-e1-i3")
        runs.append(task.score(env, task.record(env, Coast(), seed, agent_seed)))
    assert runs[0] == runs[1]


def test_a_run_without_capture_is_not_a_success():
    task = PursuitTask()
    env = task.make("pe1-e1-i3")
    metrics = task.score(env, task.record(env, Coast(), 10**30, 10**31))
    assert metrics["success"] is False
    assert metrics["steps"] == 24 and metrics["closest_approach"] > 2600.0


def test_episodes_run_in_parallel_give_the_same_results():
    jobs = [Job(name, "validation", 0, 10**30, 10**31) for name in ("pe1-e1-i3", "pe1-e3-i3")]
    agent = "vbar.games.pursuers:VelocityMatchingPursuer"
    serial = run_jobs(jobs, agent, task=PursuitTask(), workers=1)
    parallel = run_jobs(jobs, agent, task=PursuitTask(), workers=2)
    for one, other in zip(serial, parallel, strict=True):
        assert one["error"] is None and other["error"] is None
        assert one["metrics"] == other["metrics"]
        assert one["provenance"] == other["provenance"]
    assert (
        serial[0]["metrics"]["success"] and serial[0]["scenario_id"] == "vbar/PursuitEvasionE1I3-v1"
    )
