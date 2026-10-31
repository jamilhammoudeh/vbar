"""A supervised LLM flight in real time: the model's latency delays the orders, and a run repeats
exactly from its recorded replies and latencies."""

import time

import numpy as np
import pytest

from vbar import RealTime, run_episode
from vbar.agents import (
    LLMSupervisor,
    RecordedModel,
    ReplayModel,
    Reply,
    SetpointTracker,
    SupervisedAgent,
)
from vbar.scenarios import make, seeds

NAME = "approach-vbar"


class Slow:
    """Gives GO when the prompt allows it, else CONTINUE, after a fixed wall-clock wait."""

    def __init__(self, wait_s):
        self.wait_s = wait_s

    def __call__(self, messages, tools):
        time.sleep(self.wait_s)
        valid = messages[0]["content"].split("Commands you may give now:", 1)[-1].split("\n", 1)[0]
        command = "GO" if "GO" in [c.strip() for c in valid.split(",")] else "CONTINUE"
        return Reply(f'{{"command": "{command}"}}', 900, 20)


def fly(model, real_time=None, steps=120):
    env = make(NAME, max_steps=steps)
    seed, agent_seed = seeds(NAME, "validation", 1).pairs()[0]
    agent = SupervisedAgent(LLMSupervisor(model), tracker=SetpointTracker())
    return run_episode(env, agent, seed=seed, agent_seed=agent_seed, real_time=real_time)


@pytest.mark.slow
def test_the_models_latency_delays_the_supervised_flight():
    paused = fly(Slow(0.02))
    # 0.02 s of wall time at a scale of 1000 is 20 s of flight per consult.
    late = fly(Slow(0.02), real_time=RealTime(time_scale=1000.0))
    assert not np.array_equal(paused["states"], late["states"])
    assert late["latencies_s"].max() >= 0.02


@pytest.mark.slow
def test_a_real_time_supervised_run_replays_with_its_recorded_replies_and_latencies():
    recorder = RecordedModel(Slow(0.01))
    recorded = fly(recorder, real_time=RealTime(time_scale=1000.0))
    repeat = RealTime(time_scale=1000.0, latencies=tuple(recorded["latencies_s"]))
    replayed = fly(ReplayModel(recorder.entries), real_time=repeat)
    assert np.array_equal(recorded["states"], replayed["states"])
