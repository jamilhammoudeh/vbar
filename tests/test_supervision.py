"""The supervisory command channel: the ladder of rungs a goal gives, the validity mask of each
command, refusal of invalid orders, determinism, the consult cadence and the information a
supervisor is given.

A supervised agent turns a supervisor's commands (GO, HOLD, RETREAT, CAM, THRUST_INHIBIT) into
set points for a built-in tracker. Mask conditions are checked on constructed observations; the
free-drift states are verified here against the Clohessy-Wiltshire state transition matrix, so
the cases do not depend on the code under test. Whole flights are marked `slow`.
"""

import dataclasses
import math
import re

import numpy as np
import pytest
from supervision_support import (
    Always,
    Cycle,
    Flight,
    GoWhenPossible,
    Session,
    plain_situation,
)

from vbar import run_episode
from vbar.agents.briefing import TASK_KEYS
from vbar.agents.rules import RuleSupervisor
from vbar.agents.supervision import (
    PHASES,
    Command,
    Order,
    Rung,
    Situation,
    SupervisedAgent,
    ladder,
)
from vbar.agents.tracker import SetpointTracker
from vbar.dynamics import cw_stm
from vbar.scenarios import make, seeds
from vbar.scenarios.suites import SUITES

BASE = ("approach-vbar", "approach-rbar", "docking:axis", "docking-afrl", "inspection")

APPROACH = make("approach-vbar").agent_view()
DOCK = make("docking:axis").agent_view()
N = APPROACH.n


def with_keep_out(view, radius):
    constraints = tuple(
        {**c, "radius": radius} if c["type"] == "keep_out" else c for c in view.constraints
    )
    return dataclasses.replace(view, constraints=constraints)


# The same task with the keep-out sphere larger than the first rung's distance, so a chaser at rest
# on that rung is inside it.
BIG_KEEP_OUT = with_keep_out(APPROACH, 250.0)
COMMIT = dataclasses.replace(APPROACH, goal={**APPROACH.goal, "commit_radius": 30.0})

# Observations (Hill, m and m/s) on the approach-vbar ladder: rung 0 is (0, -200, 0), rung 1
# (0, -100, 0). All are at rest unless a velocity is listed.
FAR = (0.0, -300.0, 0.0, 0.0, 0.0, 0.0)
R0 = (0.0, -200.0, 0.0, 0.0, 0.0, 0.0)
MID = (0.0, -150.0, 0.0, 0.0, 0.0, 0.0)
R1 = (0.0, -100.0, 0.0, 0.0, 0.0, 0.0)
OFF_R0 = (0.0, -215.0, 0.0, 0.0, 0.0, 0.0)
# Closing on the target along V-bar at 0.1, 0.3 and 0.8 m/s from 100 m. The free drift of the
# first stays beyond 50 m, that of the other two enters it. A CAM impulse of (0, -0.3, 0) m/s
# clears 50 m from the second and not from the third (test below).
SLOW_CLOSING = (0.0, -100.0, 0.0, 0.0, 0.1, 0.0)
CLOSING = (0.0, -100.0, 0.0, 0.0, 0.3, 0.0)
FAST = (0.0, -100.0, 0.0, 0.0, 0.8, 0.0)
INSIDE = (0.0, -20.0, 0.0, 0.0, 0.0, 0.0)
FAR_DOCK = (0.0, -150.0, 0.0, 0.0, 0.0, 0.0)
IN_DOCK = (0.0, -0.5, 0.0, 0.0, 0.0, 0.0)

# Station keeping against the tidal acceleration (-3 n^2 x, 0, n^2 z) costs at most half the
# thruster authority (Fehse 3.3.3, 6.5).
HALF_AUTHORITY = 0.5 * APPROACH.max_accel
X_OK = (0.8 * HALF_AUTHORITY / (3 * N**2), -300.0, 0.0, 0.0, 0.0, 0.0)
X_BAD = (1.2 * HALF_AUTHORITY / (3 * N**2), -300.0, 0.0, 0.0, 0.0, 0.0)
Z_BAD = (0.0, -300.0, 1.2 * HALF_AUTHORITY / N**2, 0.0, 0.0, 0.0)

GO, HOLD, RETREAT, CAM, INHIBIT = (
    Command.GO,
    Command.HOLD,
    Command.RETREAT,
    Command.CAM,
    Command.THRUST_INHIBIT,
)


def min_drift_range(state, view, impulse=None, orbits=2.0, step=5.0):
    """Least range from the target of the free drift of `state` over `orbits` orbits (m)."""
    state = np.array(state, dtype=float)
    if impulse is not None:
        state[3:] += impulse
    times = np.arange(0.0, orbits * 2.0 * math.pi / view.n + step, step)
    return min(float(np.linalg.norm((cw_stm(view.n, t) @ state)[:3])) for t in times)


def test_the_commands_and_phases_are_the_declared_ones():
    assert [(c.name, int(c)) for c in Command] == [
        ("CONTINUE", 0),
        ("GO", 1),
        ("HOLD", 2),
        ("RETREAT", 3),
        ("CAM", 4),
        ("THRUST_INHIBIT", 5),
        ("RECONFIGURE", 6),
    ]
    assert PHASES == (
        "transfer",
        "holding",
        "stopped",
        "retreating",
        "inhibited",
        "commit",
        "aborted",
    )


def test_the_free_drift_states_of_the_mask_cases_have_the_drift_they_are_built_for():
    keep_out = 50.0
    assert min_drift_range(SLOW_CLOSING, APPROACH) > keep_out
    assert min_drift_range(CLOSING, APPROACH) < keep_out
    cam = (0.0, -0.3, 0.0)
    assert min_drift_range(CLOSING, APPROACH, cam) > keep_out
    assert min_drift_range(FAST, APPROACH, cam) < keep_out
    assert min_drift_range(FAR, APPROACH) > keep_out
    assert min_drift_range(R0, APPROACH) > keep_out
    assert min_drift_range(R0, BIG_KEEP_OUT) < 250.0


# Ladder


def test_the_base_core_scenarios_are_the_five_the_ladder_covers():
    names = {e.scenario for e in SUITES["core-v1"].entries if not e.scenario.endswith(":nav1")}
    assert names == set(BASE)


def first_observation(name, seed=1):
    return make(name).reset(seed=seed)[0]


@pytest.mark.parametrize(
    ("name", "hold", "goal"),
    [
        ("approach-vbar", (0.0, -200.0, 0.0), (0.0, -100.0, 0.0)),
        ("approach-rbar", (-200.0, 0.0, 0.0), (-100.0, 0.0, 0.0)),
    ],
)
def test_a_hold_goal_off_the_origin_gets_a_hold_rung_twice_as_far_and_the_goal(name, hold, goal):
    view = make(name).agent_view()
    rungs = ladder(view, first_observation(name))
    assert len(rungs) == 2 and all(isinstance(r, Rung) for r in rungs)
    assert [r.kind for r in rungs] == ["hold", "goal"]
    assert np.allclose(rungs[0].position, hold) and np.allclose(rungs[1].position, goal)
    assert rungs[1].radius == view.goal["capture_radius"]


def test_a_hold_goal_at_the_origin_gets_a_rung_50_m_along_the_line_of_sight():
    view = make("docking-afrl").agent_view()
    obs0 = np.array([300.0, 400.0, 0.0, 0.0, 0.0, 0.0])
    rungs = ladder(view, obs0)
    assert [r.kind for r in rungs] == ["hold", "goal"]
    assert np.allclose(rungs[0].position, [30.0, 40.0, 0.0])
    assert np.allclose(rungs[1].position, [0.0, 0.0, 0.0])
    assert rungs[1].radius == view.goal["capture_radius"]


def test_the_line_of_sight_rung_follows_the_first_observation_of_a_drawn_seed():
    name = "docking-afrl"
    view = make(name).agent_view()
    seed, _ = seeds(name, "validation", 1).pairs()[0]
    obs0 = first_observation(name, seed)
    rung = ladder(view, obs0)[0]
    assert np.allclose(rung.position, 50.0 * obs0[:3] / np.linalg.norm(obs0[:3]))


def test_a_docking_goal_gets_a_standoff_outside_every_cone_and_the_port_as_contact():
    view = DOCK
    rungs = ladder(view, first_observation("docking:axis"))
    port = np.array(view.goal["port"])
    axis = np.array(view.goal["port_axis"])
    reach = max(c["active_range"] for c in view.constraints if c["type"] == "approach_cone")
    assert [r.kind for r in rungs] == ["hold", "contact"]
    assert np.allclose(rungs[0].position, port + (reach + 10.0) * axis)
    assert np.allclose(rungs[1].position, port)


def test_a_docking_goal_with_a_station_keeping_point_gets_a_rung_there_before_the_port():
    goal = {**DOCK.goal, "hold": (20.0, 300.0)}
    view = dataclasses.replace(DOCK, goal=goal)
    rungs = ladder(view, first_observation("docking:axis"))
    port, axis = np.array(goal["port"]), np.array(goal["port_axis"])
    assert [r.kind for r in rungs] == ["hold", "hold", "contact"]
    assert np.allclose(rungs[1].position, port + 20.0 * axis)


def test_an_inspection_goal_gets_an_entry_point_at_60_m_and_the_sun_behind_orbit_point():
    view = make("inspection").agent_view()
    rungs = ladder(view, first_observation("inspection"))
    sun = np.array(view.goal["sun_direction"], dtype=float)
    planar = np.array([sun[0], sun[1], 0.0]) / np.linalg.norm(sun[:2])
    assert [r.kind for r in rungs] == ["hold", "orbit"]
    assert math.isclose(float(np.linalg.norm(rungs[0].position)), 60.0, rel_tol=1e-9)
    assert np.allclose(rungs[1].position, 60.0 * planar)


LADDER_TASKS = {"hold": "approach-vbar", "dock": "docking:axis", "inspect": "inspection"}


@pytest.mark.parametrize(("task", "key"), [(t, k) for t in LADDER_TASKS for k in TASK_KEYS[t]])
def test_a_goal_missing_a_key_its_task_needs_raises_value_error_naming_the_key(task, key):
    name = LADDER_TASKS[task]
    view = make(name).agent_view()
    goal = {k: v for k, v in view.goal.items() if k != key}
    with pytest.raises(ValueError, match=re.escape(key)):
        ladder(dataclasses.replace(view, goal=goal), first_observation(name))


def test_an_inspection_goal_without_a_sun_direction_raises_value_error_naming_it():
    view = make("inspection").agent_view()
    goal = {k: v for k, v in view.goal.items() if k != "sun_direction"}
    with pytest.raises(ValueError, match="sun_direction"):
        ladder(dataclasses.replace(view, goal=goal), first_observation("inspection"))


# Validity mask


def at_rung_0(s):
    s.settle(FAR)
    return s.settle(R0)


def to_transfer(s):
    return s.settle(FAR)


def to_holding(s):
    return at_rung_0(s)


def to_transfer_1(s):
    at_rung_0(s)
    s.step(R0, GO)
    return s.settle(MID)


def to_holding_1(s):
    to_transfer_1(s)
    return s.settle(R1)


def to_stopped(s):
    s.settle(FAR)
    s.step(FAR, HOLD)
    return s.settle(FAR)


def to_stopped_1(s):
    to_transfer_1(s)
    s.step(MID, HOLD)
    return s.settle(MID)


def to_retreating(s):
    to_transfer_1(s)
    s.step(MID, RETREAT)
    return s.settle(MID)


def to_inhibited(s):
    s.settle(FAR)
    s.step(FAR, INHIBIT)
    return s.settle(FAR)


def to_aborted(s):
    s.settle(FAR)
    s.step(FAR, CAM)
    return s.settle(FAR)


def to_commit(s):
    dock = s.view.goal["task"] == "dock"
    s.settle(FAR_DOCK if dock else FAR)
    return s.settle(IN_DOCK if dock else INSIDE)


PATHS = {
    "transfer": to_transfer,
    "holding": to_holding,
    "transfer1": to_transfer_1,
    "holding1": to_holding_1,
    "stopped": to_stopped,
    "stopped1": to_stopped_1,
    "retreating": to_retreating,
    "inhibited": to_inhibited,
    "aborted": to_aborted,
    "commit": to_commit,
}


def test_the_first_consult_is_at_t_0_in_transfer_to_rung_0_with_the_ladder_and_mask():
    s = Session(APPROACH)
    s.step(FAR)
    sit = s.sit
    assert isinstance(sit, Situation)
    assert sit.t == 0.0 and sit.phase == "transfer" and sit.rung == 0
    assert [r.kind for r in sit.ladder] == ["hold", "goal"]
    assert len(sit.mask) == len(Command) and sit.refused is None


@pytest.mark.parametrize(
    ("path", "phase", "rung"),
    [
        ("holding", "holding", 0),
        ("transfer1", "transfer", 1),
        ("holding1", "holding", 1),
        ("stopped", "stopped", 0),
        ("stopped1", "stopped", 1),
        ("retreating", "retreating", None),
        ("inhibited", "inhibited", 0),
        ("aborted", "aborted", 0),
        ("commit", "commit", None),
    ],
)
def test_the_orders_and_arrivals_that_reach_each_phase_put_the_agent_in_it(path, phase, rung):
    view = COMMIT if path == "commit" else APPROACH
    sit = PATHS[path](Session(view))
    assert sit.phase == phase
    if rung is not None:
        assert sit.rung == rung


CELLS = []


def cell(path, label, obs, command, valid, view=APPROACH):
    CELLS.append(pytest.param(path, obs, view, command, valid, id=f"{path}-{command.name}-{label}"))


# Transfer: GO is not allowed before the arrival, HOLD and CAM and INHIBIT depend on the state.
cell("transfer", "not-arrived", FAR, GO, False)
cell("transfer", "outside-commit-and-gentle-hold", FAR, HOLD, True)
cell("transfer", "station-keeping-within-half-authority", X_OK, HOLD, True)
cell("transfer", "radial-station-keeping-too-costly", X_BAD, HOLD, False)
cell("transfer", "normal-station-keeping-too-costly", Z_BAD, HOLD, False)
cell("transfer", "no-previous-rung", FAR, RETREAT, False)
cell("transfer1", "previous-rung-exists", MID, RETREAT, True)
cell("transfer1", "not-arrived", MID, GO, False)
cell("transfer", "cam-arc-clears-keep-out", CLOSING, CAM, True)
cell("transfer", "cam-arc-enters-keep-out", FAST, CAM, False)
cell("transfer", "drift-clears-keep-out", SLOW_CLOSING, INHIBIT, True)
cell("transfer", "drift-enters-keep-out", CLOSING, INHIBIT, False)
# A commit radius in the goal changes nothing outside it.
cell("transfer", "outside-commit-radius", FAR, HOLD, True, COMMIT)
cell("transfer1", "outside-commit-radius", MID, RETREAT, True, COMMIT)
cell("transfer", "outside-commit-radius", CLOSING, CAM, True, COMMIT)

# Retreating: GO resumes, RETREAT is not repeated.
cell("retreating", "resume", MID, GO, True)
cell("retreating", "gentle-hold", MID, HOLD, True)
cell("retreating", "radial-station-keeping-too-costly", X_BAD, HOLD, False)
cell("retreating", "already-retreating", MID, RETREAT, False)
cell("retreating", "cam-arc-clears-keep-out", CLOSING, CAM, True)
cell("retreating", "cam-arc-enters-keep-out", FAST, CAM, False)
cell("retreating", "drift-clears-keep-out", SLOW_CLOSING, INHIBIT, True)
cell("retreating", "drift-enters-keep-out", CLOSING, INHIBIT, False)

# Holding at rung 0 of 2 with the arrival held: GO is allowed; at the last rung it is not.
cell("holding", "arrival-held-and-not-last-rung", R0, GO, True)
cell("holding1", "last-rung", R1, GO, False)
cell("holding", "arrival-lost", OFF_R0, GO, False)
cell("holding", "already-holding", R0, HOLD, False)
cell("holding1", "already-holding", R1, HOLD, False)
cell("holding", "no-previous-rung", R0, RETREAT, False)
cell("holding1", "previous-rung-exists", R1, RETREAT, True)
cell("holding", "cam-arc-clears-keep-out", R0, CAM, True)
cell("holding", "cam-arc-enters-keep-out", R0, CAM, False, BIG_KEEP_OUT)
cell("holding", "drift-clears-keep-out", R0, INHIBIT, True)
cell("holding", "drift-enters-keep-out", R0, INHIBIT, False, BIG_KEEP_OUT)

# Stopped: GO restarts, HOLD is already in effect.
cell("stopped", "restart", FAR, GO, True)
cell("stopped", "already-stopped", FAR, HOLD, False)
cell("stopped", "no-previous-rung", FAR, RETREAT, False)
cell("stopped1", "previous-rung-exists", MID, RETREAT, True)
cell("stopped", "cam-arc-clears-keep-out", CLOSING, CAM, True)
cell("stopped", "cam-arc-enters-keep-out", FAST, CAM, False)
cell("stopped", "drift-clears-keep-out", SLOW_CLOSING, INHIBIT, True)
cell("stopped", "drift-enters-keep-out", CLOSING, INHIBIT, False)

# Inhibited: GO restarts, HOLD depends on the station-keeping cost, INHIBIT is in effect.
cell("inhibited", "restart", FAR, GO, True)
cell("inhibited", "gentle-hold", FAR, HOLD, True)
cell("inhibited", "radial-station-keeping-too-costly", X_BAD, HOLD, False)
cell("inhibited", "no-previous-rung", FAR, RETREAT, False)
cell("inhibited", "cam-arc-clears-keep-out", CLOSING, CAM, True)
cell("inhibited", "cam-arc-enters-keep-out", FAST, CAM, False)
cell("inhibited", "already-inhibited", FAR, INHIBIT, False)

# Inside the commit radius only THRUST_INHIBIT remains, and only on a docking task (Fehse 8.3.6).
for command in (GO, HOLD, RETREAT, CAM):
    cell("commit", "inside-commit-radius", INSIDE, command, False, COMMIT)
    cell("commit", "inside-commit-radius", IN_DOCK, command, False, DOCK)
cell("commit", "task-is-hold", INSIDE, INHIBIT, False, COMMIT)
cell("commit", "task-is-dock", IN_DOCK, INHIBIT, True, DOCK)

# Aborted: nothing is allowed.
for command in (GO, HOLD, RETREAT, CAM, INHIBIT):
    cell("aborted", "after-a-cam", FAR, command, False)


@pytest.mark.parametrize(("path", "obs", "view", "command", "valid"), CELLS)
def test_the_mask_allows_a_command_exactly_when_the_phase_and_state_conditions_hold(
    path, obs, view, command, valid
):
    s = Session(view)
    PATHS[path](s)
    sit = s.settle(obs)
    assert bool(sit.mask[command]) is valid
    if not valid:
        assert isinstance(sit.reasons[command], str) and sit.reasons[command]


@pytest.mark.parametrize("path", sorted(PATHS))
def test_continue_is_always_valid_and_reconfigure_is_refused_for_want_of_a_redundant_unit(path):
    view = COMMIT if path == "commit" else APPROACH
    s = Session(view)
    sit = PATHS[path](s)
    assert bool(sit.mask[Command.CONTINUE]) is True
    assert bool(sit.mask[Command.RECONFIGURE]) is False
    assert sit.reasons[Command.RECONFIGURE] == "no redundant units"


def test_an_order_the_mask_refuses_is_recorded_as_refused_and_shown_at_the_next_consult():
    s = Session(APPROACH)
    s.settle(FAR)
    step = s.step(FAR, GO)
    assert step.consulted and step.accepted is False and step.reason
    s.step(FAR)
    assert s.sit.refused is not None


def test_a_situation_is_frozen_and_holds_copies_of_the_observation():
    s = Session(APPROACH)
    obs = np.array(FAR)
    s.agent.act(obs, {"time_s": 0.0})
    sit = s.supervisor.seen[-1]
    with pytest.raises(AttributeError):
        sit.phase = "holding"
    assert not np.shares_memory(sit.obs, obs)
    assert np.array_equal(obs, FAR)


# Invalid orders


def refusals():
    """Orders a supervisor can give that the agent must refuse, by supervision mode."""
    nan = float("nan")
    return {
        "ladder": {
            "unknown-name": lambda: "LAUNCH",
            "unknown-number": lambda: 99,
            "point-in-ladder-mode": lambda: Order(GO, point=(0.0, -150.0, 0.0), speed=0.1),
            "velocity-in-ladder-mode": lambda: Order(GO, velocity=(0.0, 0.1, 0.0)),
        },
        "waypoint": {
            "velocity-in-waypoint-mode": lambda: Order(GO, velocity=(0.0, 0.1, 0.0)),
            "nan-point": lambda: Order(GO, point=(nan, -150.0, 0.0), speed=0.1),
            "nan-speed": lambda: Order(GO, point=(0.0, -150.0, 0.0), speed=nan),
            "point-inside-keep-out": lambda: Order(GO, point=(0.0, -10.0, 0.0), speed=0.1),
            "zero-speed": lambda: Order(GO, point=(0.0, -150.0, 0.0), speed=0.0),
            "negative-speed": lambda: Order(GO, point=(0.0, -150.0, 0.0), speed=-0.1),
            "speed-above-v-max": lambda: Order(GO, point=(0.0, -150.0, 0.0), speed=0.7),
        },
        "velocity": {
            "point-in-velocity-mode": lambda: Order(GO, point=(0.0, -150.0, 0.0), speed=0.1),
            "nan-velocity": lambda: Order(GO, velocity=(nan, 0.0, 0.0)),
            "velocity-above-v-max": lambda: Order(GO, velocity=(0.6, 0.0, 0.0)),
        },
    }


VALID = {
    "ladder": (lambda: Command.CONTINUE, lambda: Command.CONTINUE),
    "waypoint": (lambda: Order(GO, point=(0.0, -250.0, 0.0), speed=0.1),) * 2,
    "velocity": (lambda: Order(GO, velocity=(0.0, 0.1, 0.0)),) * 2,
}

REFUSED = [
    pytest.param(mode, label, make_order, id=f"{mode}-{label}")
    for mode, cases in refusals().items()
    for label, make_order in cases.items()
]


@pytest.mark.parametrize(("mode", "label", "make_order"), REFUSED)
def test_an_invalid_order_is_refused_and_recorded_without_moving_the_reference(
    mode, label, make_order
):
    tracker = SetpointTracker(v_max=0.5)
    s = Session(APPROACH, mode=mode, tracker=tracker)
    s.step(FAR, VALID[mode][0])
    before = s.agent.last_step
    kept = (np.array(before.r_ref), np.array(before.v_ref))
    step = s.step(FAR, make_order)
    assert step.consulted and step.accepted is False
    assert isinstance(step.reason, str) and step.reason
    assert np.array_equal(step.r_ref, kept[0]) and np.array_equal(step.v_ref, kept[1])


def test_a_valid_waypoint_order_moves_the_reference_to_the_point():
    s = Session(APPROACH, mode="waypoint", tracker=SetpointTracker(v_max=0.5))
    step = s.step(FAR, lambda: Order(GO, point=(0.0, -250.0, 0.0), speed=0.1))
    assert step.accepted is True
    assert np.allclose(step.r_ref, [0.0, -250.0, 0.0])


def test_a_valid_velocity_order_sets_the_reference_velocity():
    s = Session(APPROACH, mode="velocity", tracker=SetpointTracker(v_max=0.5))
    step = s.step(FAR, lambda: Order(GO, velocity=(0.0, 0.1, 0.0)))
    assert step.accepted is True
    assert np.allclose(step.v_ref, [0.0, 0.1, 0.0])


def test_a_valid_go_moves_the_ladder_reference_to_the_next_rung():
    s = Session(APPROACH)
    at_rung_0(s)
    assert np.allclose(s.agent.last_step.r_ref, [0.0, -200.0, 0.0])
    step = s.step(R0, GO)
    assert step.accepted is True
    s.step(R0)
    assert np.allclose(s.agent.last_step.r_ref, [0.0, -100.0, 0.0])


@pytest.mark.parametrize("action_mode", ["disc27", "impulse_coast"])
def test_a_supervised_agent_refuses_an_action_mode_that_is_not_continuous(action_mode):
    view = make("approach-vbar", action_mode=action_mode).agent_view()
    agent = SupervisedAgent(Always(Command.CONTINUE), tracker=SetpointTracker())
    with pytest.raises(ValueError, match=action_mode):
        agent.reset(view, None)


# Determinism


def cycle_flight(pair, tracker=None, steps=150):
    seed, agent_seed = pair
    agent = SupervisedAgent(Cycle(), tracker=tracker)
    return Flight("approach-vbar", agent, seed, agent_seed, max_steps=steps)


PAIRS = seeds("approach-vbar", "validation", 2).pairs()


def test_the_same_seeds_give_equal_supervision_records_and_commands():
    first = cycle_flight(PAIRS[0]).run(150)
    second = cycle_flight(PAIRS[0]).run(150)
    assert first.records == second.records
    assert np.array_equal(first.actions, second.actions)
    assert any(r[3] is False for r in first.records)
    assert any(r[3] is True and "CONTINUE" not in r[2] for r in first.records)


def test_two_agents_from_one_config_stepped_interleaved_equal_each_flying_alone():
    tracker = SetpointTracker()
    a = cycle_flight(PAIRS[0], tracker)
    b = cycle_flight(PAIRS[1], tracker)
    for _ in range(150):
        a.advance()
        b.advance()
    alone_a = cycle_flight(PAIRS[0]).run(150)
    alone_b = cycle_flight(PAIRS[1]).run(150)
    assert a.records == alone_a.records and np.array_equal(a.actions, alone_a.actions)
    assert b.records == alone_b.records and np.array_equal(b.actions, alone_b.actions)
    assert not np.array_equal(alone_a.actions, alone_b.actions)


# Cadence


@pytest.mark.parametrize("period", [10.0, 30.0, 60.0, 120.0])
def test_the_supervisor_is_consulted_at_t_0_and_every_period_while_nothing_changes(period):
    s = Session(APPROACH, period_s=period)
    consulted, recorded = [], []
    for _ in range(40):
        s.step(FAR)
        consulted.append(s.consulted)
        recorded.append(s.agent.last_step.consulted)
    expected = [(10.0 * k) % period == 0.0 for k in range(40)]
    assert consulted == expected and recorded == expected


def test_a_phase_change_is_consulted_between_periods_by_arrival_and_by_an_order():
    s = Session(APPROACH, period_s=60.0)
    flags = []
    plan = [(FAR, Command.CONTINUE)] * 2 + [(R0, GO), (MID, Command.CONTINUE)]
    plan += [(MID, Command.CONTINUE)] * 2
    for obs, order in plan:
        s.step(obs, order)
        flags.append(s.consulted)
        if len(flags) == 4:
            assert s.sit.phase == "transfer" and s.sit.rung == 1
    # t = 0 by the schedule, t = 20 by the arrival at rung 0, t = 30 by the accepted GO.
    assert flags == [True, False, True, True, False, False]


def test_the_supervisor_is_consulted_at_most_once_per_step_however_often_the_phase_changes():
    s = Session(APPROACH, period_s=60.0, supervisor=Cycle())
    places = [FAR, R0, MID, R1, FAR, R0, INSIDE, MID]
    for k in range(120):
        s.step(places[k % len(places)])
    assert 20 < len(s.supervisor.seen) <= 120


def test_pending_gives_the_situation_the_next_act_records_and_changes_no_state():
    s = Session(APPROACH, period_s=60.0, supervisor=Cycle())
    places = [FAR, R0, MID, R1, FAR, R0, INSIDE, MID]
    s.step(places[0])
    asked = 0
    for k in range(1, 120):
        obs = places[k % len(places)]
        info = {"time_s": s.t}
        tracker = s.agent._tracker
        before = (repr(s.agent.last_step), *tracker.reference(s.t))
        first = s.agent.pending(obs, info)
        second = s.agent.pending(obs, info)
        after = (repr(s.agent.last_step), *tracker.reference(s.t))
        assert before[0] == after[0]
        assert all(np.array_equal(b, a) for b, a in zip(before[1:], after[1:], strict=True))
        assert (first is None) == (second is None)
        s.step(obs)
        assert (first is not None) == s.consulted == s.agent.last_step.consulted
        if first is not None:
            asked += 1
            assert plain_situation(first) == plain_situation(second) == plain_situation(s.sit)
    assert asked > 20


@pytest.mark.parametrize("period", [5.0, 65.0, 0.0, -10.0])
def test_a_period_that_is_not_a_whole_number_of_steps_raises_value_error(period):
    with pytest.raises(ValueError):
        agent = SupervisedAgent(Always(Command.CONTINUE), period_s=period)
        agent.reset(APPROACH, None)


# Information the supervisor is given


def scripted_observation(t):
    """The same track whatever the true state: closing on rung 0, resting on it, then on rung 1."""
    if t < 200.0:
        return np.array([0.0, -300.0 + 0.5 * t, 0.0, 0.0, 0.5, 0.0])
    if t < 500.0:
        return np.array(R0)
    if t < 700.0:
        return np.array([0.0, -200.0 + 0.5 * (t - 500.0), 0.0, 0.0, 0.5, 0.0])
    return np.array(R1)


class Scripted:
    """A sensor that reports one observation sequence whatever the true state."""

    name = "scripted"

    def reset(self, rng):
        pass

    def observe(self, true_state, t):
        return scripted_observation(t)


FLOATS = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def scripted_flight(pair, steps=90):
    seed, agent_seed = pair
    supervisor = GoWhenPossible()
    agent = SupervisedAgent(supervisor)
    flight = Flight("approach-vbar", agent, seed, agent_seed, max_steps=steps, sensor=Scripted())
    flight.supervisor = supervisor
    return flight.run(steps)


def test_runs_from_true_starts_more_than_10_m_apart_give_identical_situations_and_commands():
    a = scripted_flight(PAIRS[0])
    b = scripted_flight(PAIRS[1])
    assert np.linalg.norm(a.truths[0][:3] - b.truths[0][:3]) > 10.0
    seen_a, seen_b = a.supervisor.seen, b.supervisor.seen
    assert len(seen_a) == len(seen_b) > 5
    assert [plain_situation(s) for s in seen_a] == [plain_situation(s) for s in seen_b]
    assert a.records == b.records
    assert np.array_equal(a.actions, b.actions)
    assert {s.phase for s in seen_a} >= {"transfer", "holding"}


def truth_printed(texts, truth):
    """The true values above 1 cm or 1 cm/s that some number in `texts` equals.

    numpy prints arrays to eight significant digits, so a relative 1e-8 still finds a leaked
    array; an honest run has no match up to 1e-7, while 1e-6 matched a step time of 810.0 s against
    a true y of 810.0005 m.
    """
    numbers = np.array([float(x) for text in texts for x in FLOATS.findall(text)])
    values = truth[np.abs(truth) > 1e-2]
    return [c for c in values if np.any(np.abs(numbers - c) <= 1e-8 * abs(c))]


def test_the_repr_of_a_situation_and_of_a_step_holds_no_number_of_the_true_state():
    flight = scripted_flight(PAIRS[0])
    texts = [repr(s) for s in flight.supervisor.seen] + flight.reprs
    assert FLOATS.findall(" ".join(texts))
    assert truth_printed(texts, np.concatenate(flight.truths)) == []


def test_the_repr_scan_finds_a_leaked_true_state():
    flight = scripted_flight(PAIRS[0])
    truth = np.concatenate(flight.truths)
    leaked = [repr(s) for s in flight.supervisor.seen] + [
        repr(np.asarray(t)) for t in flight.truths
    ]
    assert len(truth_printed(leaked, truth)) > 0.9 * np.count_nonzero(np.abs(truth) > 1e-2)


# Whole flights


@pytest.mark.slow
@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("name", BASE)
def test_the_rule_supervisor_over_the_tracker_succeeds_without_a_keep_out_violation(name, index):
    env = make(name)
    seed, agent_seed = seeds(name, "validation", 5).pairs()[index]
    agent = SupervisedAgent(RuleSupervisor(), tracker=SetpointTracker())
    result = run_episode(env, agent, seed=seed, agent_seed=agent_seed)
    info = env._info()
    assert info["success"], info
    assert not result["keep_out_violated"]


@pytest.mark.slow
@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("name", BASE)
def test_a_supervisor_that_only_continues_never_succeeds_and_violates_nothing(name, index):
    env = make(name)
    seed, agent_seed = seeds(name, "validation", 5).pairs()[index]
    agent = SupervisedAgent(Always(Command.CONTINUE), tracker=SetpointTracker())
    result = run_episode(env, agent, seed=seed, agent_seed=agent_seed)
    info = env._info()
    assert not info["success"], info
    assert not result["keep_out_violated"]
    if not env.soft_constraints:
        assert info["violation_steps"] == 0
