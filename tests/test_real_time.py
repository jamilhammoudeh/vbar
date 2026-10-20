import time

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, RealTime, run_episode
from vbar.agents import PDAgent
from vbar.spacecraft import Spacecraft

START = (-150.0, -400.0, 0.0, 0.0, 0.0, 0.0)


class Slow:
    """A PD agent that takes `seconds` of wall time per decision."""

    name = "slow"

    def __init__(self, seconds):
        self.seconds = seconds

    def reset(self, view, rng=None):
        self.pd = PDAgent()
        self.pd.reset(view)

    def act(self, obs, info):
        time.sleep(self.seconds)
        return self.pd.act(obs, info)


def test_recorded_latencies_act_like_an_actuation_delay():
    # A command answering the observation at t takes effect at t + 3.7 s either way.
    steps = 40
    delayed = run_episode(
        ApproachEnv(ApproachConfig(start_state=START, delay_s=3.7, max_steps=steps)), PDAgent()
    )
    timed = run_episode(
        ApproachEnv(ApproachConfig(start_state=START, max_steps=steps)),
        PDAgent(),
        real_time=RealTime(latencies=(3.7,) * steps),
    )
    assert np.array_equal(delayed["states"], timed["states"])
    assert np.array_equal(timed["latencies_s"], [3.7] * steps)


def test_wall_time_scaled_into_simulated_latency():
    # A 120 s step keeps the scaled latency inside the step even when a slow runner sleeps
    # 0.01 s for 0.5 s; macOS CI runners overshot a 10 s step.
    env = ApproachEnv(ApproachConfig(start_state=START, max_steps=2, dt=120.0))
    result = run_episode(env, Slow(0.01), real_time=RealTime(time_scale=200.0))
    latency = 200.0 * result["latencies_s"][-1]
    assert latency >= 2.0
    assert env.record.agent_latency_s == latency
    # The previous command stays on until the new one arrives.
    first, second = env.record.segment.segments
    assert first[1] == pytest.approx(latency, abs=1e-9)
    assert np.array_equal(first[2], 0.02 * result["actions"][0])
    assert np.array_equal(second[2], 0.02 * result["actions"][1])


def test_a_late_command_never_takes_effect_after_a_newer_one():
    craft = Spacecraft()
    craft.apply([1.0, 0.0, 0.0], 10.0, latency=15.0)
    actuation = craft.apply([0.0, 1.0, 0.0], 10.0, latency=0.0)
    # Both arrive at 15 s; the newer command wins and the first never acts.
    assert [round(d, 12) for d, _ in actuation.segments] == [5.0, 5.0]
    assert np.array_equal(actuation.segments[0][1], np.zeros(3))
    assert np.array_equal(actuation.segments[1][1], [0.0, 0.02, 0.0])
    with pytest.raises(ValueError, match="latency"):
        craft.apply(np.zeros(3), 10.0, latency=-1.0)


def test_every_run_records_decision_times_and_short_recordings_are_refused():
    env = ApproachEnv(ApproachConfig(start_state=START, max_steps=5))
    result = run_episode(env, PDAgent())
    assert len(result["latencies_s"]) == 5 and np.all(result["latencies_s"] >= 0.0)
    with pytest.raises(ValueError, match="ran out at decision 3"):
        run_episode(env, PDAgent(), real_time=RealTime(latencies=(0.1, 0.2, 0.3)))
    with pytest.raises(ValueError, match="time_scale"):
        RealTime(time_scale=-1.0)
