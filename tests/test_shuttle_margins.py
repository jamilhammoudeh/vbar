"""Keep-out margins predicted over the coasts the elevation-timed agent commits.

The agent coasts from its first decision to the scheduled burn, and from the burn to the
arrival burn of its transfer. Each commitment is logged with the predicted minimum keep-out
margin and a certified lower bound on it, and the agent flies on whatever they say.
"""

import math

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv
from vbar.agents.base import agent_generator, command_fraction
from vbar.agents.shuttle import ElevationTimedAgent
from vbar.metrics import record_episode
from vbar.spacecraft import PerfectNav

FT = 0.3048
TRIGGER = math.radians(42.0)


def flown(agent, seed=0, **config):
    config.setdefault("sensor", PerfectNav())
    env = ApproachEnv(ApproachConfig(**config))
    return record_episode(env, agent, seed=seed, agent_seed=seed + 1000)


def coelliptic(n, x0, y0, t):
    """State on the coelliptic path at height x0 (Hill, m) through (x0, y0) at t = 0."""
    return np.array([x0, y0 - 1.5 * n * x0 * t, 0.0, 0.0, -1.5 * n * x0, 0.0])


def late_start(n):
    """4000 ft below the target, reaching the trigger elevation 303 s after the start."""
    x0 = -4000.0 * FT
    return coelliptic(n, x0, x0 / math.tan(TRIGGER) + 1.5 * n * x0 * 303.0, 0.0)


def committed(log):
    return [entry for entry in log if "coast_s" in entry]


@pytest.mark.parametrize("mode", ["normalized", "impulse_coast"])
def test_the_wait_and_the_transfer_each_log_their_predicted_margin(mode):
    n = ApproachEnv(ApproachConfig()).agent_view().n
    ep = flown(
        ElevationTimedAgent(trigger=TRIGGER),
        start_state=tuple(late_start(n)),
        action_mode=mode,
        max_steps=60,
    )
    log = ep.agent_log
    # The burn is scheduled at the 300 s decision from the first one, and fires there.
    assert log[0]["phase"] == "wait"
    assert log[0]["coast_s"] == 300.0
    assert log[30]["phase"] == "transfer"
    assert np.any(command_fraction(ep.view, ep.records[30].command) != 0.0)
    # The burn of about 1 m/s takes several saturated steps at 0.02 m/s^2; the first decision
    # whose impulse fits in one step commits the coast to the arrival, 903.5 s (n tf = 1 rad)
    # after the burn, at step 120, cut at the end of the 60-step episode.
    k = next(j for j in range(30, len(log)) if not log[j]["saturated"])
    assert k > 30
    assert all("coast_s" not in entry for entry in log[30:k])
    arrival = round((300.0 + 1.0 / n) / 10.0)
    assert log[k]["coast_s"] == min(arrival - k - 1, 60 - k - 1) * 10.0
    assert [entry["t"] for entry in committed(log)] == [0.0, k * 10.0]
    for entry in committed(log):
        assert entry["predicted_margin_bound"] <= entry["predicted_margin"]
        assert entry["predicted_margin"] > 0.0
    if min(entry["predicted_margin_bound"] for entry in committed(log)) >= 0.0:
        assert min(r.segment.minima["keep_out"].margin for r in ep.records) >= 0.0


def test_a_moved_burn_logs_the_new_wait():
    # Every observation after the first places the chaser 40 s further back on its path, so
    # the second decision moves the burn from 300 s to 340 s and commits a new wait.
    view = ApproachEnv(ApproachConfig()).agent_view()
    start = late_start(view.n)
    agent = ElevationTimedAgent(trigger=TRIGGER)
    agent.reset(view, agent_generator(1))
    for k in range(3):
        t = k * view.dt
        state = coelliptic(view.n, start[0], start[1], t - (40.0 if k else 0.0))
        agent.act(state, {"time_s": t})
    assert [entry.get("coast_s") for entry in agent.log] == [300.0, 330.0, None]


def test_a_coast_predicted_clear_of_the_keep_out_zone_stays_clear():
    # Random starts on the default ranges, every coast checked against the continuous-time
    # minimum the environment records. Over seeds 0 to 7 in the normalized and impulse_coast
    # modes, 6 of 8 per mode were predicted clear and none of them violated. Seeds 0 and 1 are
    # predicted clear; 2 predicts a violation in the transfer and 3 in the wait before it. The four
    # impulse_coast episodes take about 3.2 s on a loaded machine; four seeds are the fewest that
    # include both kinds of predicted violation next to clear ones.
    clear = predicted_violation = 0
    for seed in range(4):
        ep = flown(
            ElevationTimedAgent(trigger=TRIGGER),
            seed,
            action_mode="impulse_coast",
            max_steps=120,
        )
        entries = committed(ep.agent_log)
        assert entries
        for entry in entries:
            assert entry["predicted_margin_bound"] <= entry["predicted_margin"]
        bound = min(entry["predicted_margin_bound"] for entry in entries)
        truth = min(r.segment.minima["keep_out"].margin for r in ep.records)
        if bound >= 0.0:
            clear += 1
            assert truth >= 0.0
        else:
            predicted_violation += 1
    assert clear == 2 and predicted_violation == 2
