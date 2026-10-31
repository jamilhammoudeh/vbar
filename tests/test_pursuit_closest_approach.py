import logging

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vbar.dynamics import KERBIN
from vbar.dynamics.kepler import propagate_kepler
from vbar.games import AgentEnvRunner, PE1_E1_I3_Env
from vbar.games.pursuers import ClosestApproachAgent, closest_approach
from vbar.games.pursuit import INITIAL_CONDITIONS


def start(initial):
    pursuer, evader = INITIAL_CONDITIONS[initial]
    return (*pursuer.state(KERBIN.mu), *evader.state(KERBIN.mu))


def integrated_ranges(state, max_time):
    """Ranges at whole seconds by DOP853, independent of the Kepler propagation under test."""
    mu = KERBIN.mu

    def rhs(t, y):
        out = []
        for k in (0, 6):
            r = y[k : k + 3]
            out += [*y[k + 3 : k + 6], *(-mu * r / np.linalg.norm(r) ** 3)]
        return out

    y0 = np.concatenate(state)
    times = np.arange(max_time + 1.0)
    solution = solve_ivp(rhs, (0.0, max_time), y0, t_eval=times, rtol=1e-12, atol=1e-6)
    return np.linalg.norm(solution.y[6:9] - solution.y[0:3], axis=0), solution.y


@pytest.mark.parametrize("initial", ["i1", "i4"])
def test_the_predicted_approach_is_the_closest_whole_second_of_an_integrated_coast(initial):
    state = start(initial)
    (r1, v1, r2, v2), time = closest_approach(state, 300)
    ranges, y = integrated_ranges(state, 300)
    assert time == int(np.argmin(ranges))
    assert np.linalg.norm(r2 - r1) == pytest.approx(ranges[time], abs=1e-4)
    assert np.allclose(np.concatenate([r1, v1, r2, v2]), y[:, time], rtol=1e-9)


def test_a_vessel_already_at_its_closest_approach_gives_time_zero():
    r_p, v_p, r_e, v_e = start("i2")  # the I2 vessels separate from the start
    assert closest_approach((r_p, v_p, r_e, v_e), 300)[1] == 0
    assert closest_approach((r_p, v_p, r_e, v_e), 0)[1] == 0


@pytest.mark.parametrize("max_time", [-1, 1.5, True, "300"])
def test_a_bad_horizon_is_refused(max_time):
    with pytest.raises(ValueError):
        closest_approach(start("i3"), max_time)


R, V = [7.5e5, 0.0, 0.0], [0.0, 2170.0, 0.0]


@pytest.mark.parametrize(
    ("state", "mu", "message"),
    [
        ((R, V, R), KERBIN.mu, "four 3-vectors"),
        (5, KERBIN.mu, "four 3-vectors"),
        ((R, V, "far", V), KERBIN.mu, "four 3-vectors"),
        ((R[:2], V, R, V), KERBIN.mu, "four finite 3-vectors"),
        (([1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]), KERBIN.mu, "four finite 3-vectors"),
        ((R, V, [np.nan, 0.0, 0.0], V), KERBIN.mu, "four finite 3-vectors"),
        ((R, [np.inf, 0.0, 0.0], R, V), KERBIN.mu, "four finite 3-vectors"),
        ((R, V, R, V), -1.0, "mu must be positive"),
        ((R, V, R, V), float("nan"), "mu must be positive"),
        ((R, V, R, V), True, "mu must be a number"),
    ],
)
def test_a_bad_state_or_mu_is_refused(state, mu, message):
    with pytest.raises(ValueError, match=message):
        closest_approach(state, 10, mu)


class Branches(ClosestApproachAgent):
    """Records the predicted approach time of each call."""

    def __init__(self):
        super().__init__()
        self.times = []

    def get_action(self, observation):
        obs = np.asarray(observation)
        self.times.append(closest_approach((obs[3:6], obs[6:9], obs[9:12], obs[12:15]), 300)[1])
        return super().get_action(observation)


def test_each_branch_fires_the_burn_of_the_original_agent():
    agent = ClosestApproachAgent()
    r_p, v_p, r_e, v_e = start("i3")
    (r1, v1, r2, v2), time = closest_approach((r_p, v_p, r_e, v_e), 300)
    action = agent.get_action([0.0, 6685.0, 1200.0, *r_p, *v_p, *r_e, *v_e])
    assert action["ref_frame"] == 1 and action["burn_vec"][3] == 5.0
    # I3 coasts to its closest approach at 121 s, beyond 80 s: 0.7 toward the evader.
    assert time == 121
    toward = (r_e - r_p) / np.linalg.norm(r_e - r_p)
    assert np.allclose(action["burn_vec"][:3], 0.7 * toward, rtol=1e-12)
    # Back up the coast so the approach is a set time ahead: each branch, both edges, and the
    # burn computed as the original computes it, so that it rounds alike.
    for seconds in (80, 79, 40, 25, 24, 10):
        p, w_p = propagate_kepler(r1, v1, -float(seconds), KERBIN.mu)
        e, w_e = propagate_kepler(r2, v2, -float(seconds), KERBIN.mu)
        (q1, u1, q2, u2), ahead = closest_approach((p, w_p, e, w_e), 300)
        assert ahead == seconds
        burn = np.array(agent.get_action([0.0, 6685.0, 1200.0, *p, *w_p, *e, *w_e])["burn_vec"])
        miss = (q2 - q1) / np.linalg.norm(q1 - q2)
        if seconds >= 80:
            expected = 0.7 * ((e - p) / np.linalg.norm(p - e))
        elif seconds >= 25:
            expected = 0.7 * miss
        else:
            expected = 0.4 * miss - 0.8 * ((u1 - u2) / np.linalg.norm(u1 - u2))
        assert np.array_equal(burn[:3], expected), seconds


def test_the_agent_logs_each_prediction_when_debugging(caplog):
    r_p, v_p, r_e, v_e = start("i3")
    observation = [0.0, 6685.0, 1200.0, *r_p, *v_p, *r_e, *v_e]
    with caplog.at_level(logging.DEBUG):
        ClosestApproachAgent(debug=True).get_action(observation)
    assert "closest approach in 121 s" in caplog.text


def test_the_agent_captures_the_passive_evader_through_all_three_branches():
    agent = Branches()
    info = AgentEnvRunner(agent, PE1_E1_I3_Env, None).run()["agent_env_results"]
    assert info["closest_approach"] == pytest.approx(5.0, abs=1e-6)
    assert info["closest_approach_time"] < 240.0
    branches = {0 if t >= 80 or t == 0 else 1 if t >= 25 else 2 for t in agent.times}
    assert branches == {0, 1, 2}
