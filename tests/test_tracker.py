"""The set-point tracker behind the supervisory commands: the speed law it flies under, the size
of its commands, the action modes it refuses, the keep-out sphere it flies round, and each base
core scenario flown with a GO at every consult. The flights are marked `slow`.
"""

import dataclasses
import math

import numpy as np
import pytest
from supervision_support import Always
from test_briefing import every_scenario

from vbar import run_episode
from vbar.agents.base import AgentView, action_scale
from vbar.agents.supervision import Command, SupervisedAgent
from vbar.agents.tracker import SetpointTracker, speed_cap
from vbar.dynamics import cw_discrete
from vbar.scenarios import make, seeds

BASE = ("approach-vbar", "approach-rbar", "docking:axis", "docking-afrl", "inspection")
N = 0.0011067834463349404
# The constant of the V-bar passive-safety speed law, n (range - extent) / VBAR_COAST, as the task
# briefing states it.
VBAR_COAST = math.sqrt(7.0) - 3.0 * math.acos(0.75)


def view_with(*constraints):
    goal = {"task": "hold"}
    return AgentView("test", 10.0, 100, N, "normalized", 0.02, "box", 500.0, goal, constraints)


# speed_cap: the lowest public speed law at a range


def test_a_closing_speed_limit_gives_its_linear_law_in_the_range():
    view = view_with({"type": "closing_speed_limit", "v0": 0.2, "v1": 0.004})
    assert math.isclose(speed_cap(view, 100.0), 0.6, rel_tol=1e-12)
    assert math.isclose(speed_cap(view, 0.0), 0.2, rel_tol=1e-12)


def test_a_full_speed_limit_gives_its_linear_law_in_the_range():
    view = view_with({"type": "speed_limit", "v0": 0.1, "v1": 0.004})
    assert math.isclose(speed_cap(view, 50.0), 0.3, rel_tol=1e-12)


GATES = ((100.0, 0.5), (10.0, 0.05), (50.0, 0.2))


@pytest.mark.parametrize(
    ("rho", "cap"),
    [(30.0, 0.125), (75.0, 0.35), (10.0, 0.05), (100.0, 0.5), (5.0, 0.05), (300.0, 0.5)],
)
def test_a_gate_table_is_interpolated_between_its_gates_and_held_beyond_them(rho, cap):
    # Listed far to near, as the catalogue lists its tables.
    view = view_with({"type": "gate_table", "gates": GATES})
    assert math.isclose(speed_cap(view, rho), cap, rel_tol=1e-12)


def test_a_v_bar_passive_speed_law_is_zero_at_the_extent_and_grows_with_the_range():
    spec = {"type": "passive_speed", "axis": "vbar", "extent": 50.0, "n": N}
    view = view_with(spec)
    assert math.isclose(speed_cap(view, 100.0), N * 50.0 / VBAR_COAST, rel_tol=1e-9)
    assert math.isclose(speed_cap(view, 50.0), 0.0, abs_tol=1e-12)


def test_an_r_bar_passive_speed_law_is_zero_within_the_extent_and_follows_the_briefed_formula():
    spec = {"type": "passive_speed", "axis": "rbar", "extent": 50.0, "n": N}
    view = view_with(spec)
    assert math.isclose(speed_cap(view, 100.0), N * math.sqrt(32500.0), rel_tol=1e-9)
    assert speed_cap(view, 40.0) == 0.0


def test_the_cap_is_the_lowest_of_the_laws_that_apply():
    closing = {"type": "closing_speed_limit", "v0": 0.2, "v1": 0.004}
    gates = {"type": "gate_table", "gates": GATES}
    view = view_with(closing, gates)
    # At 75 m the closing law gives 0.5 and the table 0.35; at 1 m 0.204 against 0.05.
    assert math.isclose(speed_cap(view, 75.0), 0.35, rel_tol=1e-12)
    assert math.isclose(speed_cap(view, 1.0), 0.05, rel_tol=1e-12)
    # Far out the table holds 0.5 and the closing law, 0.2 + 0.004 * 300 = 1.4, is higher.
    assert math.isclose(speed_cap(view, 300.0), 0.5, rel_tol=1e-12)
    slow_table = {"type": "gate_table", "gates": ((10.0, 2.0), (500.0, 3.0))}
    assert math.isclose(speed_cap(view_with(closing, slow_table), 100.0), 0.6, rel_tol=1e-12)


def expected_cap(view, rho):
    """The lowest speed law of a view's public constraints at range rho, from their specs."""
    caps = []
    for c in view.constraints:
        kind = c["type"]
        if kind in ("closing_speed_limit", "speed_limit"):
            caps.append(c["v0"] + c["v1"] * rho)
        elif kind == "gate_table":
            ranges, speeds = zip(*sorted(c["gates"]), strict=True)
            caps.append(float(np.interp(rho, ranges, speeds)))
        elif kind == "passive_speed":
            extent, n = c["extent"], c["n"]
            if rho <= extent:
                caps.append(0.0)
            elif c["axis"] == "vbar":
                caps.append(n * (rho - extent) / VBAR_COAST)
            else:
                caps.append(n * math.sqrt((4.0 * rho - extent) ** 2 - 9.0 * rho**2))
    return min(caps)


LAWS = ("closing_speed_limit", "speed_limit", "gate_table", "passive_speed")
WITH_A_LAW = [
    name
    for name in every_scenario()
    if any(c["type"] in LAWS for c in make(name).agent_view().constraints)
]


def test_the_catalogue_has_scenarios_for_every_kind_of_speed_law():
    kinds = {c["type"] for name in WITH_A_LAW for c in make(name).agent_view().constraints}
    assert set(LAWS) <= kinds


@pytest.mark.parametrize("name", WITH_A_LAW)
def test_the_cap_of_every_catalogue_scenario_is_the_lowest_of_its_public_laws(name):
    view = make(name).agent_view()
    for rho in (10.0, 45.0, 80.0, 130.0, 300.0, 900.0, 3000.0):
        assert math.isclose(
            speed_cap(view, rho), expected_cap(view, rho), rel_tol=1e-9, abs_tol=1e-12
        )


# Commands


def far_states(count, seed=4):
    rng = np.random.default_rng(seed)
    r = rng.uniform(-800.0, 800.0, (count, 3))
    v = rng.uniform(-1.0, 1.0, (count, 3))
    return np.hstack([r, v])


@pytest.mark.parametrize("action_mode", ["normalized", "accel", "thrust"])
@pytest.mark.parametrize("thrust_limit", ["box", "ball"])
def test_the_command_never_exceeds_a_full_command_per_axis_or_in_length(thrust_limit, action_mode):
    view = make("approach-vbar", thrust_limit=thrust_limit, action_mode=action_mode).agent_view()
    # A tenth of the authority, so the law asks for more than a full command most of the time.
    view = dataclasses.replace(view, max_accel=view.max_accel / 10.0)
    agent = SupervisedAgent(Always(Command.CONTINUE), tracker=SetpointTracker())
    agent.reset(view, None)
    scale = action_scale(view)
    fractions = []
    for k, state in enumerate(far_states(200)):
        fractions.append(np.asarray(agent.act(state, {"time_s": 10.0 * k}), dtype=float) / scale)
    fractions = np.array(fractions)
    size = np.linalg.norm(fractions, axis=1) if thrust_limit == "ball" else np.abs(fractions).max(1)
    assert size.max() <= 1.0 + 1e-9
    assert size.max() > 0.999


@pytest.mark.parametrize("action_mode", ["disc27", "impulse_coast"])
def test_the_tracker_refuses_an_action_mode_that_is_not_continuous_at_reset(action_mode):
    view = make("approach-vbar", action_mode=action_mode).agent_view()
    with pytest.raises(ValueError, match=action_mode):
        SetpointTracker().reset(view)


# Flight on the Clohessy-Wiltshire model


def fly(agent, view, state, steps):
    phi, gamma = cw_discrete(view.n, view.dt)
    states = [np.array(state, dtype=float)]
    for k in range(steps):
        command = np.asarray(agent.act(states[-1], {"time_s": k * view.dt}), dtype=float)
        states.append(phi @ states[-1] + gamma @ (command * view.max_accel))
    return np.array(states)


def test_the_tracker_flies_round_the_keep_out_sphere_when_the_leg_would_cross_it():
    view = make("approach-vbar").agent_view()
    keep_out = next(c["radius"] for c in view.constraints if c["type"] == "keep_out")
    start = (0.0, 150.0, 0.0, 0.0, 0.0, 0.0)
    agent = SupervisedAgent(Always(Command.CONTINUE), tracker=SetpointTracker())
    agent.reset(view, None)
    states = fly(agent, view, start, view.max_steps)
    ranges = np.linalg.norm(states[:, :3], axis=1)
    rung_0 = np.array([0.0, -200.0, 0.0])
    # The straight leg from the start to the first rung passes through the target.
    assert np.linalg.norm(np.array(start[:3]) - rung_0) > 300.0
    assert ranges.min() >= keep_out
    assert (
        np.linalg.norm(states[-1, :3] - rung_0)
        < np.linalg.norm(np.array(start[:3]) - rung_0) - 100.0
    )


# Flights


@pytest.mark.slow
@pytest.mark.parametrize("name", BASE)
def test_the_tracker_given_go_at_every_consult_flies_each_base_scenario(name):
    env = make(name)
    seed, agent_seed = seeds(name, "validation", 1).pairs()[0]
    agent = SupervisedAgent(Always(Command.GO), tracker=SetpointTracker())
    result = run_episode(env, agent, seed=seed, agent_seed=agent_seed)
    info = env._info()
    assert info["success"], info
    assert not result["keep_out_violated"]


def test_holding_at_rest_off_v_bar_cancels_the_cw_acceleration():
    # At rest at (x, y, z) the natural Clohessy-Wiltshire acceleration is (3 n^2 x, 0, -n^2 z), so
    # with no position or velocity error the hold command is its negative (Fehse section 3.3).
    view = make("approach-vbar").agent_view()
    tracker = SetpointTracker()
    tracker.reset(view)
    r = np.array([30.0, -200.0, 12.0])
    tracker.hold(r)
    command = tracker.act(np.concatenate([r, np.zeros(3)]), 0.0)
    n = view.n
    expected = -np.array([3 * n**2 * r[0], 0.0, -(n**2) * r[2]]) / view.max_accel
    assert np.all(np.abs(expected) < 1.0)
    assert command == pytest.approx(expected * action_scale(view), rel=1e-12)


def test_tracking_a_velocity_cancels_the_coriolis_terms():
    # Flying at exactly the commanded velocity, the command is -(3 n^2 x + 2 n vy, -2 n vx, -n^2 z).
    view = make("approach-vbar").agent_view()
    tracker = SetpointTracker()
    tracker.reset(view)
    r, v = np.array([0.0, -300.0, 0.0]), np.array([0.05, 0.1, 0.0])
    tracker.track_velocity(r, v)
    command = tracker.act(np.concatenate([r, v]), 0.0)
    n = view.n
    expected = -np.array([2 * n * v[1], -2 * n * v[0], 0.0]) / view.max_accel
    assert np.all(np.abs(expected) < 1.0)
    assert command == pytest.approx(expected * action_scale(view), rel=1e-12)
