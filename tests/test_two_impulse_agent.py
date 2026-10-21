"""The two-impulse CW agent and its fuel bound, flown in the approach scenario.

Plans follow Fehse, Automated Rendezvous and Docking of Spacecraft, section 3.3.2, eqs 3.41 to
3.43, and section 4.4.1 for tracking a nominal trajectory. Reference numbers quoted below come
from an independent computation with the closed-form CW transition matrix.
"""

import inspect
import warnings

import gymnasium as gym
import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.base import AgentView, agent_generator
from vbar.agents.cw_targeting import TwoImpulseAgent, TwoImpulseBound
from vbar.agents.sign_check import sign_check
from vbar.dynamics import CWDynamics, TwoBodyDynamics
from vbar.metrics import FuelBounds, evaluate, record_episode, success
from vbar.spacecraft import GaussianNav, PerfectNav, Spacecraft

REST = (0.0, -300.0, 0.0, 0.0, 0.0, 0.0)
HOLD = np.array([0.0, -100.0, 0.0])
DT = 10.0
N = ApproachEnv().agent_view().n
TF = 1.0 / N  # the default n tf = 1 rad, 903.5 s at 500 km


def flown(agent, seed=0, **config):
    config.setdefault("start_state", REST)
    env = ApproachEnv(ApproachConfig(**config))
    return record_episode(env, agent, seed=seed, agent_seed=seed + 1000)


def burns(ep):
    """Records of the steps that fired, in order."""
    return [r for r in ep.records if np.any(np.asarray(r.command)[:3] != 0.0)]


@pytest.fixture(scope="module")
def open_loop_run():
    agent = TwoImpulseAgent(mode="open_loop")
    return flown(agent, action_mode="impulse_coast", sensor=PerfectNav())


def test_open_loop_transfer_captures_at_the_planned_arrival(open_loop_run):
    ep = open_loop_run
    final = ep.records[-1]
    assert ep.terminated
    assert np.linalg.norm(final.true_state[:3] - HOLD) < 5.0
    assert np.linalg.norm(final.true_state[3:]) < 0.05
    assert abs(final.t - TF) <= DT
    # After the first burn step the coast of 89.35 steps is flown as 89, so the arrival burn
    # starts 3.5 s early and leaves 0.842 m to the hold point (an independent computation);
    # spreading each burn over one step adds nothing measurable on this rest-to-rest transfer.
    second = burns(ep)[1]
    assert np.linalg.norm(second.true_state[:3] - HOLD) < 2.0 * 0.842


# Seed 13 is among them: there the open-loop plan, made on one noisy observation, ends 5.1 m
# from the hold point and never captures (an independent computation; the only miss in 20 seeds).
NOISY_SEEDS = range(11, 16)


def test_retargeting_captures_through_navigation_noise_where_open_loop_misses():
    # About 5 s: each episode simulates about 90 steps with the keep-out zone refined in
    # continuous time, and five seeds are the fewest that keep one open-loop miss.
    retarget, open_loop = [], []
    for seed in NOISY_SEEDS:
        for mode, outcomes in (("retarget", retarget), ("open_loop", open_loop)):
            ep = flown(
                TwoImpulseAgent(mode=mode),
                seed,
                action_mode="impulse_coast",
                sensor=GaussianNav(1.0, 1e-3),
            )
            outcomes.append(success(ep))
    assert all(retarget)
    assert not all(open_loop)


def test_tracking_the_nominal_arrives_closer_than_flying_the_plan_open_loop():
    # The miss at the planned arrival, from noise in the one observation the plan is made on.
    # An independent computation over these five seeds gives medians of 0.82 m tracked and
    # 3.1 m open loop (0.92 m and 2.4 m over 20 seeds). About 5 s: ten episodes of 91 steps,
    # the fewest seeds whose medians are still far apart.
    deviations = {"track": [], "open_loop": []}
    for seed in range(10, 15):
        for mode, found in deviations.items():
            ep = flown(
                TwoImpulseAgent(mode=mode),
                seed,
                action_mode="accel",
                sensor=GaussianNav(1.0, 1e-3),
                max_steps=100,
            )
            arrival = next(r for r in ep.records if r.t >= TF)
            found.append(np.linalg.norm(arrival.true_state[:3] - HOLD))
    assert np.median(deviations["track"]) < np.median(deviations["open_loop"])


def test_first_command_points_at_the_goal_on_every_axis():
    assert sign_check(TwoImpulseAgent()).passed


def test_a_transfer_of_1_5_rad_first_burns_away_from_a_goal_above():
    # From rest the radial departure speed is n (4 sin nt - 3 nt) dx / D, D > 0, which turns
    # negative past nt = 1.2757 (the root of 4 sin x = 3 x): the planner correctly dips first.
    check = sign_check(TwoImpulseAgent(n_tf=1.5))
    assert check.failures == ("+x", "-x")
    assert check.commands["+x"][0] < 0.0


def unit_view(thrust_limit, max_steps=1000):
    """Alfriend's example 5.2 in normalized units: n = 1, the goal at the origin."""
    return AgentView(
        scenario_id="approach-vbar",
        dt=0.01,
        max_steps=max_steps,
        n=1.0,
        action_mode="normalized",
        max_accel=1.0,
        thrust_limit=thrust_limit,
        goal={"hold_point": (0.0, 0.0, 0.0), "capture_radius": 1e-4, "capture_speed": 1e-5},
    )


EX52_START = np.array([0.01, 0.02, 0.015, 0.001, 0.001, 0.001])


def test_bound_reaches_the_cheapest_two_impulse_transfer_of_alfriend_example_5_2():
    # Alfriend, Vadali, Gurfil, How and Breger, Spacecraft Formation Flying, example 5.2. The
    # printed minimum (at n tf = 2) is a misprint; recomputed at the stated inputs the least
    # sum of 2-norms is 0.0357201783 at n tf = 4.6404, and no grid point can go below it.
    found = TwoImpulseBound().bounds(unit_view("ball"), EX52_START)
    assert isinstance(found, FuelBounds)
    assert (found.lower, found.norm, found.rung) == (0.0, "l2", "cw")
    assert found.upper == pytest.approx(0.03572, abs=1e-3)
    assert found.upper >= 0.0357201783 * (1.0 - 1e-9)


def test_box_thrusters_get_a_1_norm_bound():
    # The least sum of 1-norms on a 400-point grid, from an independent computation.
    found = TwoImpulseBound().bounds(unit_view("box"), EX52_START)
    assert (found.lower, found.norm, found.rung) == (0.0, "l1", "cw")
    assert found.upper == pytest.approx(0.04573, abs=1e-3)


def test_bound_searches_only_transfers_that_fit_the_episode():
    # (max_steps - 1) dt = 2 cuts the search at n tf = 2, where the least cost is 0.037186,
    # above the 0.03572 found past it (an independent computation).
    found = TwoImpulseBound().bounds(unit_view("ball", max_steps=201), EX52_START)
    assert found.upper > 0.0365


def test_open_loop_run_flies_exactly_the_planned_two_impulse_cost(open_loop_run):
    # Bounded by the plan the agent flew (n tf = 1), the delivered 1-norm delta-v is the plan's
    # 0.676 m/s up to rounding: neither burn saturates the 0.2 m/s a step can deliver.
    ep = open_loop_run
    planned = TwoImpulseBound(n_tf_grid=(1.0,)).bounds(ep.view, ep.start)
    scored = evaluate(ep, bounds=planned)
    assert scored["success"]
    assert scored["optimality_gap_upper"] == pytest.approx(1.0, abs=1e-6)
    # Over the default grid a longer transfer costs 0.0526 m/s, so the flown plan is 12.8
    # times the bound (an independent computation).
    searched = evaluate(ep, bounds=TwoImpulseBound().bounds(ep.view, ep.start))
    assert searched["optimality_gap_upper"] > 1.0


def test_a_coast_predicted_clear_of_the_keep_out_zone_stays_clear():
    # Random starts on the default ranges, each coast checked against the continuous-time
    # minimum the environment records. An independent computation over 50 seeds found 37
    # predicted clear, none of them violating. Four seeds keep the test near 4 s (each episode
    # refines a 900 s coast) and still cover both outcomes: seeds 0 and 1 are predicted clear,
    # 2 and 3 predict a violation.
    clear = predicted_violation = 0
    for seed in range(4):
        ep = flown(
            TwoImpulseAgent(mode="open_loop"),
            seed,
            start_state=None,
            action_mode="impulse_coast",
            max_steps=100,
        )
        committed = [
            entry
            for entry, record in zip(ep.agent_log, ep.records, strict=True)
            if np.asarray(record.command)[3] > 0.0
        ]
        assert committed
        for entry in committed:
            assert entry["predicted_margin_bound"] <= entry["predicted_margin"]
        bound = min(entry["predicted_margin_bound"] for entry in committed)
        truth = min(r.segment.minima["keep_out"].margin for r in ep.records)
        if bound >= 0.0:
            clear += 1
            assert truth >= 0.0
        else:
            predicted_violation += 1
    assert clear >= 2 and predicted_violation >= 1


def test_agent_keeps_one_log_entry_per_decision():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ep = flown(TwoImpulseAgent(), max_steps=5)
    assert len(ep.agent_log) == len(ep.records) == 5


@pytest.mark.parametrize("mode", ["retarget", "open_loop", "track"])
def test_disc27_actions_are_refused_at_reset(mode):
    view = ApproachEnv(ApproachConfig(action_mode="disc27")).agent_view()
    with pytest.raises(ValueError):
        TwoImpulseAgent(mode=mode).reset(view, agent_generator(1))


def test_tracking_needs_a_continuous_action_mode():
    view = ApproachEnv(ApproachConfig(action_mode="impulse_coast")).agent_view()
    with pytest.raises(ValueError):
        TwoImpulseAgent(mode="track").reset(view, agent_generator(1))


def test_an_unknown_execution_mode_is_refused():
    view = ApproachEnv(ApproachConfig()).agent_view()
    with pytest.raises(ValueError):
        TwoImpulseAgent(mode="hover").reset(view, agent_generator(1))


def test_agent_takes_the_protocol_arguments_by_name():
    agent = TwoImpulseAgent(name="two-impulse-test")
    assert agent.name == "two-impulse-test"
    assert list(inspect.signature(agent.reset).parameters) == ["view", "rng"]
    assert list(inspect.signature(agent.act).parameters) == ["obs", "info"]
    env = ApproachEnv(ApproachConfig(start_state=REST))
    obs, _ = env.reset(seed=0)
    agent.reset(view=env.agent_view(), rng=agent_generator(1))
    assert np.asarray(agent.act(obs=obs, info=env.agent_info())).shape == (3,)


class Witness:
    """Passes everything to the agent and records the info keys it was handed."""

    def __init__(self, agent):
        self.agent, self.name, self.info_keys = agent, agent.name, set()

    def reset(self, view, rng=None):
        self.agent.reset(view, rng)

    def act(self, obs, info):
        self.info_keys |= set(info)
        return self.agent.act(obs, info)


def held_objects(agent):
    """The agent's attributes and the items of any container among them."""
    found = []
    for value in vars(agent).values():
        found.append(value)
        if isinstance(value, dict):
            found.extend(value.values())
        elif isinstance(value, list | tuple):
            found.extend(value)
    return found


@pytest.mark.parametrize("mode", ["retarget", "open_loop"])
def test_agent_sees_only_the_time_and_holds_no_simulator_object(mode):
    agent = TwoImpulseAgent(mode=mode)
    witness = Witness(agent)
    run_episode(
        ApproachEnv(ApproachConfig(start_state=REST, max_steps=5)), witness, seed=0, agent_seed=1
    )
    assert witness.info_keys == {"time_s"}
    forbidden = (gym.Env, CWDynamics, TwoBodyDynamics, Spacecraft, PerfectNav, GaussianNav)
    assert not any(isinstance(value, forbidden) for value in held_objects(agent))
