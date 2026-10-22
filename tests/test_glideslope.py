"""Glideslope guidance: the exponential range profile, its pulses and the agent that flies them.

The law is that of Hablani, Tapper and Dana-Bashian, Guidance and control of an autonomous
rendezvous spacecraft, JGCD 25(3), 2002, with the equations as reproduced in Yuan, Li and Wang,
Aerospace 9, 242, 2022, eqs 14 to 25: the range to go follows rho_dot = a rho + b with
a = (rho_dot0 - rho_dotT) / rho0 and b = rho_dotT, so
rho(t) = (rho0 + b / a) exp(a t) - b / a and the arrival time is T = ln(rho_dotT / rho_dot0) / a.
"""

import math

import gymnasium as gym
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.glideslope import GlideslopeAgent, glideslope_profile, glideslope_pulses
from vbar.agents.sign_check import sign_check
from vbar.dynamics import EARTH, CWDynamics, TwoBodyDynamics, cw_discrete, mean_motion
from vbar.scenarios.base import BaseEnv
from vbar.spacecraft import GaussianNav, PerfectNav, Spacecraft

N = mean_motion(EARTH.radius + 500e3)
REST = np.array([0.0, -300.0, 0.0, 0.0, 0.0, 0.0])
HOLD = np.array([0.0, -100.0, 0.0])
OFF_AXIS = np.array([30.0, -250.0, 20.0, 0.01, 0.0, -0.005])


def range_at(profile, rho0, t):
    return (rho0 + profile.b / profile.a) * math.exp(profile.a * t) - profile.b / profile.a


def default_profile(state, hold=HOLD, rate=1e-3, arrival_speed=0.02):
    """The profile the agent's defaults ask for: a = -rate, b = -arrival_speed."""
    rho0 = float(np.linalg.norm(state[:3] - hold))
    return glideslope_profile(rho0, -(rate * rho0 + arrival_speed), -arrival_speed)


def coast_through(n, state, impulses):
    """States just before each impulse and just after the last, flown impulsively on exact CW."""
    x = np.array(state, dtype=float)
    before = []
    for k, burn in enumerate(impulses):
        before.append(x.copy())
        x[3:] += burn.dv
        if k + 1 < len(impulses):
            x = cw_discrete(n, impulses[k + 1].t - burn.t)[0] @ x
    return np.array(before), x


def test_the_profile_closes_at_a_negative_rate_and_arrives_at_the_closed_form_time():
    profile = glideslope_profile(200.0, -0.22, -0.02)
    assert profile.a == pytest.approx(-1e-3, rel=1e-12)
    assert profile.b == pytest.approx(-0.02, rel=1e-12)
    assert profile.duration == pytest.approx(1000.0 * math.log(11.0), rel=1e-12)


def test_the_profile_meets_its_boundary_range_and_rates():
    rho0, rho_dot0, rho_dot_t = 420.0, -0.35, -0.015
    profile = glideslope_profile(rho0, rho_dot0, rho_dot_t)
    assert range_at(profile, rho0, 0.0) == pytest.approx(rho0, rel=1e-12)
    # rho(T) is a difference of two terms of size |b / a| = 7.8 m; 1e-12 of rho0 is the
    # rounding floor of that difference.
    assert abs(range_at(profile, rho0, profile.duration)) < 1e-12 * rho0
    assert profile.a * rho0 + profile.b == pytest.approx(rho_dot0, rel=1e-12)
    assert profile.b == pytest.approx(rho_dot_t, rel=1e-12)


def test_shuttle_exponential_braking_is_the_limit_of_a_vanishing_arrival_rate():
    # Fehse, Automated Rendezvous and Docking of Spacecraft, 6.2.3 p. 183 and App. B p. 443:
    # x_dot = -x / lambda with lambda = 1000 s inside 300 m gives 0.3, 0.1 and 0.01 m/s at 300,
    # 100 and 10 m, and x(t) = 300 exp(-t / lambda). An arrival rate of 1e-9 m/s moves each
    # speed by 1e-9 m/s.
    rho_dot_t = -1e-9
    profile = glideslope_profile(300.0, -0.3 + rho_dot_t, rho_dot_t)
    assert profile.a == pytest.approx(-1e-3, rel=1e-6)
    for rho, speed in ((300.0, 0.3), (100.0, 0.1), (10.0, 0.01)):
        assert -(profile.a * rho + profile.b) == pytest.approx(speed, abs=1e-6)
    for rho in (100.0, 10.0):
        t = 1000.0 * math.log(300.0 / rho)
        assert range_at(profile, 300.0, t) == pytest.approx(rho, rel=1e-6)


@pytest.mark.parametrize(
    ("rho0", "rho_dot0", "rho_dot_t"),
    [
        (300.0, -0.3, 0.0),  # no arrival rate: the range never closes
        (300.0, -0.01, -0.02),  # slower at the start than at arrival: a positive rate
        (300.0, 0.1, -0.02),  # opening at the start
        (300.0, -0.3, 0.02),  # opening at arrival
        (0.0, -0.3, -0.02),
        (-5.0, -0.3, -0.02),
    ],
)
def test_profiles_that_do_not_close_on_the_aim_point_are_refused(rho0, rho_dot0, rho_dot_t):
    with pytest.raises(ValueError):
        glideslope_profile(rho0, rho_dot0, rho_dot_t)


@pytest.mark.parametrize("state", [REST, OFF_AXIS])
def test_pulses_start_now_end_at_arrival_and_are_spaced_at_most_the_pulse_interval(state):
    profile = default_profile(state)
    impulses = glideslope_pulses(N, state, HOLD, profile, 60.0)
    times = np.array([burn.t for burn in impulses])
    assert len(impulses) == math.ceil(profile.duration / 60.0) + 1
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(profile.duration, rel=1e-12)
    assert np.all(np.diff(times) > 0.0)
    assert np.all(np.diff(times) <= 60.0 * (1.0 + 1e-12))


@pytest.mark.parametrize("state", [REST, OFF_AXIS])
def test_each_pulse_fires_on_the_line_of_sight_at_the_profile_range(state):
    profile = default_profile(state)
    impulses = glideslope_pulses(N, state, HOLD, profile, 60.0)
    before, _ = coast_through(N, state, impulses)
    rho0 = float(np.linalg.norm(state[:3] - HOLD))
    u = (state[:3] - HOLD) / rho0
    for burn, x in zip(impulses, before, strict=True):
        offset = x[:3] - HOLD
        # Forty CW solves and propagations round to about 1e-13 of the range; 1e-9 of it is far
        # above that and far below any waypoint placed off the line.
        assert np.linalg.norm(offset - (offset @ u) * u) < 1e-9 * rho0
        assert offset @ u == pytest.approx(range_at(profile, rho0, burn.t), abs=1e-9 * rho0)


@pytest.mark.parametrize("state", [REST, OFF_AXIS])
def test_the_last_pulse_leaves_the_chaser_at_rest_on_the_aim_point(state):
    profile = default_profile(state)
    impulses = glideslope_pulses(N, state, HOLD, profile, 60.0)
    before, after = coast_through(N, state, impulses)
    assert np.linalg.norm(before[-1][:3] - HOLD) < 1e-9 * np.linalg.norm(state[:3] - HOLD)
    # Just before the last pulse the chaser closes at the arrival rate, 0.02 m/s; the CW terms
    # turn the velocity by about n times the 60 s interval, 0.066 rad, between pulses.
    assert np.linalg.norm(before[-1][3:]) == pytest.approx(0.02, rel=0.1)
    assert np.linalg.norm(after[3:]) < 1e-12


@settings(max_examples=12, deadline=None)
@given(
    position=st.tuples(
        st.floats(-200.0, 200.0, allow_subnormal=False),
        st.floats(-600.0, -160.0, allow_subnormal=False),
        st.floats(-50.0, 50.0, allow_subnormal=False),
    ),
    velocity=st.tuples(*[st.floats(-0.05, 0.05, allow_subnormal=False)] * 3),
    rate=st.floats(5e-4, 5e-3, allow_subnormal=False),
    arrival_speed=st.floats(5e-3, 0.05, allow_subnormal=False),
)
def test_every_pulse_plan_rides_the_line_of_sight_and_stops_on_the_aim_point(
    position, velocity, rate, arrival_speed
):
    state = np.array([*position, *velocity])
    profile = default_profile(state, rate=rate, arrival_speed=arrival_speed)
    impulses = glideslope_pulses(N, state, HOLD, profile, 120.0)
    before, after = coast_through(N, state, impulses)
    rho0 = float(np.linalg.norm(state[:3] - HOLD))
    u = (state[:3] - HOLD) / rho0
    offsets = before[:, :3] - HOLD
    off_line = offsets - np.outer(offsets @ u, u)
    # Up to 71 segments over 1.5 orbits; the rounding of the chain stays near 1e-12 of the range.
    assert np.max(np.linalg.norm(off_line, axis=1)) < 1e-9 * rho0
    assert np.all(np.diff(offsets @ u) < 0.0)
    assert np.linalg.norm(after[3:]) < 1e-9 * (rate * rho0 + arrival_speed)


def test_the_pulse_cost_converges_to_first_order_in_the_pulse_interval():
    profile = default_profile(REST)
    totals = {
        dt: sum(np.linalg.norm(burn.dv) for burn in glideslope_pulses(N, REST, HOLD, profile, dt))
        for dt in (60.0, 30.0, 10.0)
    }
    assert totals[60.0] < totals[30.0] < totals[10.0]
    # The total falls by about 3.9e-4 m/s per second of pulse interval. Extrapolated to zero
    # interval from (60, 30) s and from (30, 10) s it is 0.7258 m/s; from an independent
    # computation at 5 and 1 s, 0.72579 m/s. Dropping the last pulse lowers it by 0.0206 m/s.
    coarse = totals[30.0] + (totals[30.0] - totals[60.0])
    fine = totals[10.0] + (totals[10.0] - totals[30.0]) / 2.0
    assert coarse == pytest.approx(0.72579, rel=1e-3)
    assert fine == pytest.approx(0.72579, rel=1e-3)


def burn_starts(actions, dt):
    """Decision indices and times of the decisions that start a burn in impulse_coast."""
    times = np.concatenate([[0.0], np.cumsum(dt * (1 + np.round(actions[:, 3] / dt)))])
    thrust = np.any(actions[:, :3] != 0.0, axis=1)
    starts = [
        k
        for k in range(len(actions))
        if thrust[k] and (k == 0 or not thrust[k - 1] or actions[k - 1, 3] > 0.0)
    ]
    return np.array(starts), times[starts]


def test_open_loop_pulses_fly_down_the_v_bar_and_stop_at_the_hold_point_on_time():
    # Capture speed below the 0.02 m/s arrival rate, so the run captures only after the last
    # pulse has stopped the chaser; at 0.05 m/s it would capture inside 5 m, before arrival.
    cfg = ApproachConfig(action_mode="impulse_coast", start_state=tuple(REST), capture_speed=0.01)
    env = ApproachEnv(cfg)
    agent = GlideslopeAgent(mode="open_loop")
    result = run_episode(env, agent, seed=1, agent_seed=2)
    assert result["captured"] and not result["keep_out_violated"]
    actions = result["actions"]
    starts, times = burn_starts(actions, cfg.dt)
    # The line of sight is the V-bar, so a pulse position's distance from it is |(x, z)|.
    positions = result["states"][starts, :3]
    assert np.max(np.hypot(positions[:, 0], positions[:, 2])) < 1e-6
    arrival = default_profile(REST).duration
    assert times[-1] == round(arrival / cfg.dt) * cfg.dt
    assert len(agent.log) == len(actions)
    assert not any(isinstance(value, BaseEnv) for value in vars(agent).values())


def test_committed_coasts_log_their_predicted_keep_out_margin_and_its_bound():
    cfg = ApproachConfig(action_mode="impulse_coast", start_state=tuple(REST), capture_speed=0.01)
    agent = GlideslopeAgent(mode="open_loop")
    result = run_episode(ApproachEnv(cfg), agent, seed=1, agent_seed=2)
    coasting = [entry for entry, a in zip(agent.log, result["actions"], strict=True) if a[3] > 0]
    assert coasting
    for entry in coasting:
        assert entry["predicted_margin_bound"] <= entry["predicted_margin"]
        # The glideslope stays on the V-bar behind the hold point, 50 m clear of the sphere.
        assert entry["predicted_margin_bound"] > 0.0


def test_the_first_command_points_toward_the_goal_on_every_axis():
    check = sign_check(GlideslopeAgent())
    assert check.passed, check.commands


def test_a_pulse_interval_off_the_step_grid_is_refused():
    view = ApproachEnv(ApproachConfig(action_mode="impulse_coast")).agent_view()
    GlideslopeAgent(pulse_interval=30.0).reset(view, np.random.default_rng(0))
    with pytest.raises(ValueError):
        GlideslopeAgent(pulse_interval=25.0).reset(view, np.random.default_rng(0))


def test_bang_off_bang_actions_are_refused_at_reset():
    view = ApproachEnv(ApproachConfig(action_mode="disc27")).agent_view()
    with pytest.raises(ValueError):
        GlideslopeAgent().reset(view, np.random.default_rng(0))


@pytest.mark.parametrize("action_mode", ["impulse_coast", "normalized"])
def test_a_glideslope_started_on_the_hold_point_commands_zero_and_captures(action_mode):
    cfg = ApproachConfig(action_mode=action_mode, start_state=(0.0, -100.0, 0.0, 0.0, 0.0, 0.0))
    agent = GlideslopeAgent()
    result = run_episode(ApproachEnv(cfg), agent, seed=1, agent_seed=2)
    assert result["captured"]
    assert np.all(result["actions"] == 0.0)
    assert len(agent.log) == len(result["actions"])


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
def test_the_agent_sees_only_the_time_and_holds_no_simulator_object(mode):
    agent = GlideslopeAgent(mode=mode)
    witness = Witness(agent)
    cfg = ApproachConfig(action_mode="impulse_coast", start_state=tuple(REST), max_steps=20)
    run_episode(ApproachEnv(cfg), witness, seed=0, agent_seed=1)
    assert witness.info_keys == {"time_s"}
    forbidden = (gym.Env, CWDynamics, TwoBodyDynamics, Spacecraft, PerfectNav, GaussianNav)
    assert not any(isinstance(value, forbidden) for value in held_objects(agent))


@pytest.mark.parametrize("mode", ["retarget", "open_loop"])
def test_reset_clears_the_previous_episode(mode):
    first = ApproachConfig(action_mode="impulse_coast", start_state=tuple(REST), max_steps=40)
    second = ApproachConfig(action_mode="impulse_coast", start_state=tuple(OFF_AXIS), max_steps=40)
    reused = GlideslopeAgent(mode=mode)
    run_episode(ApproachEnv(first), reused, seed=1, agent_seed=2)
    again = run_episode(ApproachEnv(second), reused, seed=3, agent_seed=4)
    fresh = run_episode(ApproachEnv(second), GlideslopeAgent(mode=mode), seed=3, agent_seed=4)
    # The same computation repeated on the same inputs, so the actions agree exactly.
    assert np.array_equal(again["actions"], fresh["actions"])
    assert len(reused.log) == len(again["actions"])
