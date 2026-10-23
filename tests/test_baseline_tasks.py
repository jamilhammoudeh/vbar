"""The classical baselines fly to a hold point, so they refuse goals whose task is not a hold."""

import dataclasses
import functools
import re

import numpy as np
import pytest

from vbar.agents import (
    CWTrackingAgent,
    DriftAgent,
    Element,
    ElementAgent,
    ElevationTimedAgent,
    FehseSequenceAgent,
    GlideslopeAgent,
    Impulse,
    LambertAgent,
    LQGAgent,
    LQRAgent,
    LyapunovAgent,
    MissionManagerAgent,
    Mode,
    Mutant,
    NominalTrackingAgent,
    PDAgent,
    PilotAgent,
    TorvaAgent,
    TwoImpulseAgent,
    TwoImpulseBound,
)
from vbar.scenarios import make

START = np.array([0.0, -300.0, 0.0, 0.0, 0.0, 0.0])


class Bound:
    """Puts `TwoImpulseBound.bounds` behind a reset, the call that reads the view's goal."""

    def reset(self, view, rng=None):
        TwoImpulseBound(grid_points=20).bounds(view, START)


DISC27 = {"action_mode": "disc27"}
IMPULSE_COAST = {"action_mode": "impulse_coast"}

# Each agent, the name its refusal gives, and view changes it refuses on their own.
AGENTS = {
    "two-impulse": (lambda: TwoImpulseAgent(), "two-impulse agent", [DISC27]),
    # The bound reads no action mode; an unknown thrust layout is what it refuses.
    "two-impulse-bound": (Bound, "two-impulse bound", [{"thrust_limit": "ring"}]),
    # A 20 m final leg keeps the hop and the leg inside the hold scenario's 3600 s episode.
    "fehse-sequence": (
        lambda: FehseSequenceAgent(final_leg=20.0),
        "Fehse sequence agent",
        [DISC27, IMPULSE_COAST],
    ),
    "nominal-tracking": (
        lambda: NominalTrackingAgent(TwoImpulseAgent(mode="open_loop")),
        "nominal tracking agent",
        [DISC27, IMPULSE_COAST],
    ),
    "glideslope": (lambda: GlideslopeAgent(), "glideslope agent", [DISC27]),
    "lambert": (lambda: LambertAgent(), "Lambert agent", [DISC27]),
    "lqr": (lambda: LQRAgent(), "LQR agent", [DISC27, IMPULSE_COAST]),
    "lqg": (lambda: LQGAgent(), "LQG agent", [DISC27, IMPULSE_COAST]),
    "cw-tracking": (lambda: CWTrackingAgent(), "CW tracking agent", [DISC27, IMPULSE_COAST]),
    "lyapunov": (lambda: LyapunovAgent(), "Lyapunov agent", [DISC27, IMPULSE_COAST]),
    "elevation-timed": (
        lambda: ElevationTimedAgent(trigger=0.3),
        "elevation-timed agent",
        [DISC27],
    ),
    "pilot": (lambda: PilotAgent(), "pilot agent", [DISC27]),
    # The manager has no check of its own: it resets the first mode's agent at reset and each
    # later mode's agent on entry, so the refusal comes from the agents it runs.
    "mission": (
        lambda: MissionManagerAgent([Mode(name="hold", agent=LQRAgent())]),
        "LQR agent",
        [DISC27, IMPULSE_COAST],
    ),
    # The wrapper resets the inner agent with the same view.
    "mutant-pd": (lambda: Mutant(PDAgent(), "delay", 1.0), "PD agent", [DISC27, IMPULSE_COAST]),
}

REFUSED_MODES = [
    pytest.param(name, change, id=f"{name}-{'-'.join(map(str, change.values()))}")
    for name, (_, _, changes) in AGENTS.items()
    for change in changes
]

# These fly a given element or a fly-around and never read the goal, so any task suits them; a
# fly-around is an inspection manoeuvre.
GOAL_FREE = {
    "element": lambda: ElementAgent(
        lambda view, state: Element(
            impulses=(Impulse(0.0, np.array([0.0, 0.01, 0.0])),), duration=0.0
        )
    ),
    "torva": lambda: TorvaAgent(stop_angle=1.0),
}


@functools.cache
def approach_view():
    return make("approach-vbar").agent_view()


def hold_view(view_changes=None, **goal):
    """The approach-vbar view with its goal entries replaced; None removes an entry."""
    view = approach_view()
    merged = {**view.goal, **goal}
    goal = {k: v for k, v in merged.items() if v is not None}
    return dataclasses.replace(view, goal=goal, **(view_changes or {}))


def refusal(agent, task):
    return re.escape(f"the {agent} flies to a hold point and cannot fly the {task} task")


@pytest.mark.parametrize("task", ["dock", "inspect", "depart"])
@pytest.mark.parametrize("name", AGENTS)
def test_baselines_refuse_a_task_that_is_not_a_hold_at_reset(name, task):
    # The goal keeps its hold point: inspection and departure goals carry one for the metrics.
    make_agent, agent, _ = AGENTS[name]
    with pytest.raises(ValueError, match=refusal(agent, task)):
        make_agent().reset(hold_view(task=task), np.random.default_rng(0))


@pytest.mark.parametrize(("name", "change"), REFUSED_MODES)
def test_the_task_refusal_comes_before_the_action_mode_refusal(name, change):
    make_agent, agent, _ = AGENTS[name]
    with pytest.raises(ValueError, match=refusal(agent, "dock")):
        make_agent().reset(hold_view(change, task="dock"), np.random.default_rng(0))


@pytest.mark.parametrize("task", ["berth", None])
@pytest.mark.parametrize("name", [*AGENTS, *GOAL_FREE])
def test_baselines_accept_a_berthing_goal_and_a_goal_without_a_task(name, task):
    make_agent = AGENTS[name][0] if name in AGENTS else GOAL_FREE[name]
    make_agent().reset(hold_view(task=task), np.random.default_rng(0))


@pytest.mark.parametrize("name", AGENTS)
def test_baselines_refuse_a_hold_goal_without_a_hold_point(name):
    make_agent, agent, _ = AGENTS[name]
    with pytest.raises(
        ValueError, match=re.escape(f"the {agent} flies to a hold point and this goal has none")
    ):
        make_agent().reset(hold_view(hold_point=None), np.random.default_rng(0))


@pytest.mark.parametrize("name", GOAL_FREE)
def test_element_and_fly_around_agents_fly_an_inspection_goal_without_a_hold_point(name):
    agent = GOAL_FREE[name]()
    agent.reset(hold_view(task="inspect", hold_point=None), np.random.default_rng(0))
    action = agent.act(START, {"time_s": 0.0})
    assert action.shape == (3,) and np.all(np.isfinite(action))


def test_a_mission_refuses_when_it_enters_a_hold_mode_on_a_docking_goal():
    mission = MissionManagerAgent(
        [
            Mode(name="coast", agent=DriftAgent(), exit=lambda obs, t: True, next="hold"),
            Mode(name="hold", agent=LQRAgent()),
        ]
    )
    mission.reset(hold_view(task="dock"), np.random.default_rng(0))
    with pytest.raises(ValueError, match=refusal("LQR agent", "dock")):
        mission.act(START, {"time_s": 0.0})
