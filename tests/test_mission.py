"""Mission manager: a timeline of modes with next, contingency and abort exits.

Fehse, Automated Rendezvous and Docking of Spacecraft, section 6.3, pp. 203 to 207: each mode
runs until its exit gate holds (next), runs a contingency mode when its time runs out without the
gate, and leaves for an abort mode when safety demands it.
"""

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.base import AgentView, DriftAgent
from vbar.agents.mission import MissionManagerAgent, Mode
from vbar.scenarios.base import BaseEnv

DT = 10.0
# From rest on the V-bar with 0.1 m/s across the track, z(t) = (0.1 / n) sin(n t) is 1.0, 2.0,
# 3.0, 4.0, 5.0 m at the decisions 10 to 50 s to within 3 mm (an independent computation), so
# position gates at half-metre thresholds fire at known decisions.
CROSSING = (0.0, -300.0, 0.0, 0.0, 0.0, 0.1)


def never(obs, t_in_mode):
    return False


def mode(name, agent=None, **exits):
    fields = dict(exit=None, t_max=None, next=None, contingency=None, abort=None)
    fields.update(exits)
    return Mode(name=name, agent=agent or Counting(), **fields)


class Counting(DriftAgent):
    """Drifts, and counts its resets and its decisions since the last reset."""

    name = "counting"

    def __init__(self):
        self.resets = []
        self.acts = 0

    def reset(self, view, rng=None):
        super().reset(view, rng)
        self.resets.append((view, rng))
        self.acts = 0

    def act(self, obs, info):
        self.acts += 1
        return super().act(obs, info)


class DoneAfter(Counting):
    """Declares itself done on its `k`-th decision."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def reset(self, view, rng=None):
        super().reset(view, rng)
        self.done = False

    def act(self, obs, info):
        action = super().act(obs, info)
        self.done = self.acts >= self.k
        return action


class Spy(DriftAgent):
    """Records what the manager hands a mode agent."""

    name = "spy"

    def reset(self, view, rng=None):
        super().reset(view, rng)
        self.reset_args = (view, rng)
        self.info_keys = set()
        self.obs_shapes = set()

    def act(self, obs, info):
        self.info_keys |= set(info)
        self.obs_shapes.add(np.shape(obs))
        return super().act(obs, info)


class CoastThrough(DriftAgent):
    """Commits a 300 s coast at its first decision, then drifts step by step."""

    name = "coast-through"

    def reset(self, view, rng=None):
        super().reset(view, rng)
        self.first = True

    def act(self, obs, info):
        if self.first:
            self.first = False
            return np.array([0.0, 0.0, 0.0, 300.0])
        return np.zeros(4)


def fly(manager, steps, start=CROSSING, **config):
    cfg = ApproachConfig(start_state=start, max_steps=steps, **config)
    return run_episode(ApproachEnv(cfg), manager, seed=1, agent_seed=2)


def modes_flown(manager):
    return [entry["mode"] for entry in manager.log]


def transitions(manager):
    """(decision, transition) for each decision that logged one."""
    return [
        (k, str(entry["transition"])) for k, entry in enumerate(manager.log) if entry["transition"]
    ]


def timeline():
    return [
        mode("approach", exit=lambda obs, t: obs[2] > 4.5, next="hold"),
        mode("hold", exit=never, t_max=35.0, next="final", contingency="recover"),
        mode("recover", exit=lambda obs, t: t > 15.0, next="final"),
        mode("final"),
    ]


def test_a_timeline_moves_on_at_its_gates_and_runs_the_contingency_when_time_runs_out():
    manager = MissionManagerAgent(timeline())
    result = fly(manager, 20)
    # Gate z > 4.5 m holds first at 50 s; hold times out at 40 s in mode (above 35 s);
    # recover's gate t > 15 s holds at 20 s in mode.
    expected = ["approach"] * 6 + ["hold"] * 5 + ["recover"] * 3 + ["final"] * 6
    assert modes_flown(manager) == expected
    assert len(manager.log) == len(result["actions"]) == 20
    found = transitions(manager)
    assert [k for k, _ in found] == [5, 10, 13]
    assert "next" in found[0][1] and "contingency" in found[1][1] and "next" in found[2][1]
    t_in_mode = [entry["t_in_mode"] for entry in manager.log]
    assert t_in_mode == pytest.approx(
        [*np.arange(6) * DT, *np.arange(5) * DT, *np.arange(3) * DT, *np.arange(6) * DT],
        abs=1e-9,
    )


def test_a_second_episode_starts_again_from_the_first_mode():
    manager = MissionManagerAgent(timeline())
    fly(manager, 20)
    first = modes_flown(manager)
    fly(manager, 20)
    assert modes_flown(manager) == first


def test_the_contingency_fires_only_once_the_time_in_mode_is_above_its_limit():
    manager = MissionManagerAgent(
        [
            mode("hold", exit=never, t_max=30.0, contingency="recover", next="final"),
            mode("recover", exit=never, next="final"),
            mode("final"),
        ]
    )
    fly(manager, 8)
    # At 30 s in mode the limit is reached, not exceeded; the switch comes at 40 s.
    assert modes_flown(manager)[:6] == ["hold"] * 5 + ["recover"]
    assert transitions(manager)[0][0] == 4


def test_a_gate_that_holds_after_the_time_limit_moves_on_instead_of_running_the_contingency():
    manager = MissionManagerAgent(
        [
            mode(
                "hold",
                exit=lambda obs, t: t > 35.0,
                t_max=35.0,
                contingency="recover",
                next="final",
            ),
            mode("recover", exit=never, next="final"),
            mode("final"),
        ]
    )
    fly(manager, 8)
    assert modes_flown(manager) == ["hold"] * 5 + ["final"] * 3
    k, transition = transitions(manager)[0]
    assert k == 4 and "next" in transition and "contingency" not in transition


def test_an_agent_that_declares_itself_done_moves_the_timeline_on():
    manager = MissionManagerAgent([mode("first", DoneAfter(3), next="second"), mode("second")])
    fly(manager, 6)
    assert modes_flown(manager) == ["first"] * 3 + ["second"] * 3
    assert transitions(manager)[0][0] == 2


def test_the_abort_predicate_wins_over_the_gate_and_the_time_limit():
    # At 30 s, z = 3.0 m: the abort predicate, the gate and the time limit all hold, and the
    # abort is checked first. The predicate holds at no other decision (z = 2.0 and 4.0 m).
    manager = MissionManagerAgent(
        [
            mode(
                "approach",
                exit=lambda obs, t: obs[2] > 2.5,
                t_max=25.0,
                next="final",
                contingency="hold",
                abort="safe",
            ),
            mode("hold", exit=never, next="final"),
            mode("safe", exit=never, next="final"),
            mode("final"),
        ],
        abort_when=lambda obs, t: 2.5 < obs[2] < 3.5,
    )
    fly(manager, 8)
    assert modes_flown(manager) == ["approach"] * 4 + ["safe"] * 4
    k, transition = transitions(manager)[0]
    assert k == 3 and "abort" in transition


@pytest.mark.parametrize(
    "modes",
    [
        [mode("a", next="nowhere"), mode("b")],
        [mode("a", next="b", contingency="nowhere"), mode("b")],
        [mode("a", next="b", abort="nowhere"), mode("b")],
        [mode("a", next="b"), mode("b", next="a")],
    ],
    ids=["unknown next", "unknown contingency", "unknown abort", "last mode with a next"],
)
def test_timelines_that_name_missing_modes_or_never_end_are_refused(modes):
    with pytest.raises(ValueError):
        MissionManagerAgent(modes)


def test_a_mode_agent_is_reset_with_the_view_and_stream_each_time_its_mode_is_entered():
    looped, recovery, final = Counting(), Counting(), Counting()
    manager = MissionManagerAgent(
        [
            mode("loop", looped, exit=never, t_max=15.0, contingency="recover", next="final"),
            mode("recover", recovery, exit=lambda obs, t: t > 5.0, next="loop"),
            mode("final", final),
        ]
    )
    env = ApproachEnv(ApproachConfig(start_state=CROSSING, max_steps=7))
    obs, _ = env.reset(seed=1)
    view = env.agent_view()
    rng = np.random.default_rng(5)
    manager.reset(view, rng)
    for _ in range(7):
        obs = env.step(manager.act(obs, env.agent_info()))[0]
    # loop: 0 to 20 s, contingency; recover: 0 and 10 s, next; loop again from 50 s.
    assert modes_flown(manager) == ["loop"] * 3 + ["recover"] * 2 + ["loop"] * 2
    assert len(looped.resets) == 2 and len(recovery.resets) == 1 and not final.resets
    for reset_view, reset_rng in looped.resets + recovery.resets:
        assert reset_view is view and reset_rng is rng
    assert looped.acts == 2


def test_mode_agents_see_only_the_view_a_stream_the_observation_and_the_time():
    spies = [Spy(), Spy(), Spy()]
    manager = MissionManagerAgent(
        [
            mode("a", spies[0], exit=lambda obs, t: t > 15.0, next="b"),
            mode("b", spies[1], exit=lambda obs, t: t > 15.0, next="c"),
            mode("c", spies[2]),
        ]
    )
    result = fly(manager, 8)
    for spy in spies:
        view, rng = spy.reset_args
        assert isinstance(view, AgentView) and isinstance(rng, np.random.Generator)
        assert spy.info_keys == {"time_s"}
        assert spy.obs_shapes == {(6,)}
        assert not any(isinstance(value, BaseEnv) for value in vars(spy).values())
    assert not any(isinstance(value, BaseEnv) for value in vars(manager).values())
    assert len(manager.log) == len(result["actions"])


# From 100 m behind the target closing at 0.5 m/s, a free coast passes 21 m from it: over 300 s
# the predicted keep-out margin is -28.65 m (an independent computation), while the first
# 10 s alone stay 45 m clear.
CLOSING = (0.0, -100.0, 0.0, 0.0, 0.5, 0.0)


def test_a_coast_predicted_to_enter_the_keep_out_zone_switches_to_the_abort_mode():
    manager = MissionManagerAgent(
        [
            mode("close", CoastThrough(), exit=never, next="final", abort="safe"),
            mode("safe", exit=never, next="final"),
            mode("final"),
        ]
    )
    result = fly(manager, 40, start=CLOSING, action_mode="impulse_coast")
    assert "abort" in str(manager.log[0]["transition"])
    assert manager.log[1]["mode"] == "safe"
    # The committed coast is not flown: the next decision comes one step later.
    assert round(result["actions"][0][3] / DT) == 0


def test_without_an_abort_mode_the_manager_flies_the_coast_and_logs_its_negative_margin():
    manager = MissionManagerAgent(
        [mode("close", CoastThrough(), exit=never, next="final"), mode("final")]
    )
    result = fly(manager, 40, start=CLOSING, action_mode="impulse_coast")
    assert not manager.log[0]["transition"]
    assert manager.log[0]["mode"] == "close"
    assert result["actions"][0][3] == 300.0
    assert manager.log[0]["predicted_margin"] < -20.0
    assert manager.log[0]["predicted_margin_bound"] <= manager.log[0]["predicted_margin"]
    assert result["keep_out_violated"]
