"""The Lambert targeting agent on CW and two-body truth.

The agent rebuilds a circular target orbit from the view's mean motion and its declared
gravitational parameter, a = (mu / n^2)^(1/3), and solves Lambert's problem (Curtis, Orbital
Mechanics for Engineering Students, algorithm 5.2) from the chaser's inertial state to the hold
point. Every number is for a 500 km circular orbit of the WGS 84 Earth; reference numbers come from
an independent computation with a separate universal-variable solver flown in the same environment.
"""

import functools
import inspect
import math

import gymnasium as gym
import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.base import agent_generator, command_fraction
from vbar.agents.cw_targeting import TwoImpulseAgent
from vbar.agents.lambert import LambertAgent
from vbar.agents.sign_check import sign_check
from vbar.dynamics import EARTH, CWDynamics, TwoBodyDynamics
from vbar.metrics import record_episode, success
from vbar.spacecraft import GaussianNav, PerfectNav, Spacecraft

N = ApproachEnv().agent_view().n
A = (EARTH.mu / N**2) ** (1.0 / 3.0)
HOLD = np.array([0.0, -100.0, 0.0])
INCLINATION = math.radians(51.6)


def rest_behind(rho):
    return (0.0, -rho, 0.0, 0.0, 0.0, 0.0)


def first_action(agent, start, **config):
    env = ApproachEnv(ApproachConfig(start_state=start, sensor=PerfectNav(), **config))
    obs, _ = env.reset(seed=0)
    view = env.agent_view()
    agent.reset(view, agent_generator(1))
    return command_fraction(view, agent.act(obs, env.agent_info()))


def flown(agent, start, seed=0, **config):
    config.setdefault("sensor", PerfectNav())
    env = ApproachEnv(ApproachConfig(start_state=start, **config))
    return record_episode(env, agent, seed=seed, agent_seed=seed + 1000)


@pytest.mark.parametrize("rho", [300.0, 3000.0])
def test_first_burn_agrees_with_cw_targeting_to_the_range_over_the_orbit_radius(rho):
    # The CW plan is Lambert's linearised about the target orbit, so the two first impulses
    # differ by a relative amount of order rho / a: 0.57 rho / a at 300 m and 0.32 rho / a at
    # 3000 m (an independent computation); the solver's rounding floor sits near 1e-11. An
    # agent built in the wrong frame or on the wrong orbit differs by order one.
    config = {"action_mode": "impulse_coast"}
    lambert = first_action(LambertAgent(), rest_behind(rho), **config)
    cw = first_action(TwoImpulseAgent(), rest_behind(rho), **config)
    gap = np.linalg.norm(lambert[:3] - cw[:3]) / np.linalg.norm(cw[:3])
    assert gap < rho / A


# Open-loop transfers of 900 s, a whole number of 10 s steps, with no burn saturating (2 m/s^2).
# Each one-step burn misses by itself (1.2 m at 300 m to 17 m at 3000 m), so the model error is
# the distance from the arrival of the CW plan flown on CW truth, where the model is exact and
# only the execution misses. Two keep-out samples per step: no claim here is about the zone.
TRANSFER = {
    "action_mode": "impulse_coast",
    "max_steps": 95,
    "max_accel": 2.0,
    "violation_samples": 2,
}
N_TF = 900.0 * N
RANGES = (300.0, 600.0, 1000.0, 2000.0, 3000.0)


PLANNERS = {
    "lambert": lambda: LambertAgent(n_tf=N_TF, mode="open_loop"),
    "lambert_8_mu": lambda: LambertAgent(n_tf=N_TF, mu=8.0 * EARTH.mu, mode="open_loop"),
    "cw": lambda: TwoImpulseAgent(n_tf=N_TF, mode="open_loop"),
}


@functools.cache
def arrival(planner, rho, dynamics):
    """Position at the end of the coast the departure burn commits (Hill, m).

    Cached so the tests below share their episodes: a two-body episode propagates the 900 s
    coast at the sample points and takes 1.0 s to 1.4 s on a loaded machine.
    """
    ep = flown(PLANNERS[planner](), rest_behind(rho), dynamics=dynamics, **TRANSFER)
    departure = next(r for r in ep.records if np.asarray(r.command)[3] > 0.0)
    assert departure.t == pytest.approx(900.0)
    return departure.true_state[:3].copy()


def model_error(planner, rho):
    """Distance between the planner's two-body arrival and the CW plan's arrival on CW truth (m)."""
    return float(np.linalg.norm(arrival(planner, rho, "twobody") - arrival("cw", rho, "cw")))


# Model errors of Lambert and of the CW plan from an independent computation: 0.000124 and
# 0.00718 m at 300 m, 0.000513 and 0.0257 at 600 m, 0.00144 and 0.0684 at 1000 m, 0.00580 and
# 0.265 at 2000 m, 0.0131 and 0.589 m at 3000 m; the CW error is 45 to 58 times Lambert's.
@pytest.mark.parametrize("rho", RANGES)
def test_on_two_body_truth_lambert_arrives_ten_times_closer_than_cw_targeting(rho):
    # Two two-body episodes and one CW episode per start, shared with the next test; the five
    # starts span the ranges the claim is made for.
    assert model_error("lambert", rho) < 0.1 * model_error("cw", rho)


def test_declared_gravitational_parameter_sets_the_model_orbit():
    # Declaring eight times Earth's mu doubles the model orbit radius at the same n and halves
    # the curvature the agent corrects for: from 3000 m the model error grows from 0.0131 m to
    # 0.301 m (an independent computation). An agent that ignores the declared mu errs the same
    # in both. One two-body episode beyond those shared with the test above.
    assert model_error("lambert_8_mu", 3000.0) > 10.0 * model_error("lambert", 3000.0)


def burn_commands(agent, dynamics="cw"):
    ep = flown(
        agent,
        rest_behind(300.0),
        action_mode="impulse_coast",
        dynamics=dynamics,
        max_steps=100,
        violation_samples=2,
    )
    return ep, [np.asarray(r.command, dtype=float)[:3] for r in ep.records if np.any(r.command[:3])]


def test_arrival_burn_agrees_with_the_cw_plan_in_the_hill_axes():
    # The planned arrival impulses from rest at 300 m differ by 0.28 rho / a; the arrival
    # velocity left in inertial axes, 1 rad of orbit after departure, differs by 0.96 of its
    # size (an independent computation).
    _, lambert = burn_commands(LambertAgent(mode="open_loop"))
    _, cw = burn_commands(TwoImpulseAgent(mode="open_loop"))
    assert len(lambert) >= 2 and len(cw) >= 2
    assert np.linalg.norm(lambert[1] - cw[1]) <= 300.0 / A * np.linalg.norm(cw[1])


def test_open_loop_transfer_on_two_body_truth_captures_at_rest():
    ep, _ = burn_commands(LambertAgent(mode="open_loop"), dynamics="twobody")
    assert ep.terminated and success(ep)
    assert np.linalg.norm(ep.records[-1].true_state[3:]) < 0.05


@pytest.mark.parametrize("mode", ["retarget", "open_loop"])
def test_commands_do_not_depend_on_the_orbit_inclination(mode):
    # Lambert's problem is invariant under rotation, and the agent sees only the Hill-frame
    # observation, so the commands agree up to the rounding of the inertial truth (about 1e-12
    # of a 300 m range). Three decisions per orbit: retarget re-solves at each, open loop
    # commits its coast at the first.
    commands = []
    for inclination in (0.0, INCLINATION):
        ep = flown(
            LambertAgent(mode=mode),
            rest_behind(300.0),
            action_mode="impulse_coast",
            dynamics="twobody",
            inclination=inclination,
            max_steps=3,
            violation_samples=2,
        )
        commands.append(np.array([r.command for r in ep.records]))
    assert commands[0].shape == commands[1].shape
    scale = np.max(np.abs(commands[0]))
    assert np.max(np.abs(commands[0] - commands[1])) <= 1e-9 * scale


def test_agent_runs_on_j2_truth_whose_perturbation_is_outside_its_model():
    # The model is an unperturbed circular orbit; J2 and eccentricity are model error here.
    # Three retargeting decisions, enough to re-solve on perturbed truth.
    ep = flown(
        LambertAgent(),
        rest_behind(300.0),
        action_mode="impulse_coast",
        dynamics="twobody_j2",
        max_steps=3,
        violation_samples=2,
    )
    assert len(ep.records) == 3
    assert all(np.all(np.isfinite(r.command)) for r in ep.records)


def test_first_command_points_at_the_goal_on_every_axis():
    assert sign_check(LambertAgent(n_tf=1.0)).passed


def test_committed_coasts_log_the_predicted_keep_out_margin():
    ep = flown(
        LambertAgent(mode="open_loop"),
        rest_behind(300.0),
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


@pytest.mark.parametrize("mode", ["retarget", "open_loop"])
def test_agent_keeps_one_log_entry_per_decision(mode):
    ep = flown(LambertAgent(mode=mode), rest_behind(300.0), max_steps=5)
    assert len(ep.agent_log) == len(ep.records) == 5


@pytest.mark.parametrize("mode", ["retarget", "open_loop"])
def test_disc27_actions_are_refused_at_reset(mode):
    view = ApproachEnv(ApproachConfig(action_mode="disc27")).agent_view()
    with pytest.raises(ValueError):
        LambertAgent(mode=mode).reset(view, agent_generator(1))


def test_an_unknown_execution_mode_is_refused():
    with pytest.raises(ValueError):
        agent = LambertAgent(mode="hover")
        agent.reset(ApproachEnv().agent_view(), agent_generator(1))


@pytest.mark.parametrize("mu", [0.0, -EARTH.mu, math.inf, math.nan])
def test_a_gravitational_parameter_that_is_not_positive_and_finite_is_refused(mu):
    with pytest.raises(ValueError):
        LambertAgent(mu=mu)


def test_declared_model_knowledge_is_keyword_only_with_earth_as_default():
    parameters = inspect.signature(LambertAgent).parameters
    assert parameters["mu"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["mu"].default == EARTH.mu
    with pytest.raises(TypeError):
        LambertAgent(EARTH.mu)


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


@pytest.mark.parametrize("dynamics", ["cw", "twobody"])
def test_agent_sees_only_the_time_and_holds_no_simulator_object(dynamics):
    agent = LambertAgent()
    witness = Witness(agent)
    env = ApproachEnv(
        ApproachConfig(start_state=rest_behind(300.0), dynamics=dynamics, max_steps=5)
    )
    run_episode(env, witness, seed=0, agent_seed=1)
    assert witness.info_keys == {"time_s"}
    forbidden = (gym.Env, CWDynamics, TwoBodyDynamics, Spacecraft, PerfectNav, GaussianNav)
    assert not any(isinstance(value, forbidden) for value in held_objects(agent))
