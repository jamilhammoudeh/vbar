"""Agents that fly trajectory elements: element execution, the twice-rate fly-around, the Fehse
sequence and nominal-trajectory tracking.

Elements follow Fehse, Automated Rendezvous and Docking of Spacecraft, sections 3.3 (transfers and
forced motion), 4.4 (nominal trajectories and their tracking) and 5.7 (approach strategies); the
twice-orbital-rate fly-around follows Goodman, History of Space Shuttle Rendezvous (JSC-63400),
p. 137. Every number is for a 500 km circular orbit of the WGS 84 Earth on exact CW motion with a
10 s step unless stated; reference numbers come from an independent computation with scratch
agents flown in the same environment.
"""

import inspect
import math

import gymnasium as gym
import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.base import agent_generator
from vbar.agents.cw_targeting import TwoImpulseAgent
from vbar.agents.elements import (
    ElementAgent,
    FehseSequenceAgent,
    NominalTrackingAgent,
    TorvaAgent,
    radial_transfer,
    twice_rate_cost,
)
from vbar.agents.sign_check import sign_check
from vbar.dynamics import CWDynamics, TwoBodyDynamics, cw_discrete
from vbar.metrics import record_episode, success
from vbar.spacecraft import GaussianNav, PerfectNav, Spacecraft

N = ApproachEnv().agent_view().n
T = 2.0 * math.pi / N
DT = 10.0
HOLD = np.array([0.0, -100.0, 0.0])
HOP_START = (0.0, -600.0, 0.0, 0.0, 0.0, 0.0)
R_TORVA = 100.0
TORVA_START = (-R_TORVA, 0.0, 0.0, 0.0, 0.0, 0.0)


def flown(agent, seed=0, **config):
    config.setdefault("sensor", PerfectNav())
    env = ApproachEnv(ApproachConfig(**config))
    return record_episode(env, agent, seed=seed, agent_seed=seed + 1000)


def fired(ep, axis=None):
    """Indices of the records whose command thrusts (on `axis`, or on any axis)."""
    cmds = [np.asarray(r.command, dtype=float)[:3] for r in ep.records]
    if axis is None:
        return [i for i, c in enumerate(cmds) if np.any(c != 0.0)]
    return [i for i, c in enumerate(cmds) if c[axis] != 0.0]


def radial_hop(*_):
    # Fehse's radial transfer along the V-bar, 500 m from (0, -600, 0) to the hold point in T/2.
    return radial_transfer(N, 500.0)


def view_for(**config):
    return ApproachEnv(ApproachConfig(**config)).agent_view()


# Runs whose claims are not about the keep-out zone check it at two samples per step: the
# environment refines the zone's minimum in continuous time, which dominates their run time.
CHEAP = {"violation_samples": 2}


@pytest.fixture(scope="module")
def impulsive_hop():
    # One decision with a 2830 s coast, checked at the environment's default 20 samples per step
    # because the margin test compares the agent's prediction with that check; 2.7 s to 4.6 s
    # measured on a loaded machine, nearly all of it the refinement over the coast.
    return flown(ElementAgent(radial_hop), start_state=HOP_START, action_mode="impulse_coast")


@pytest.fixture(scope="module", params=["impulse_coast", "accel"])
def hop_run(request):
    if request.param == "impulse_coast":
        return request.getfixturevalue("impulsive_hop")
    # 285 steps of 10 s, the hop's half orbit, which the end point and the cost need whole; the
    # step sets the 0.209 m. 2.8 s alone and up to 6 s on a loaded machine, about 8 ms a step
    # in the environment's keep-out refinement even at two samples.
    return flown(ElementAgent(radial_hop), start_state=HOP_START, action_mode="accel", **CHEAP)


def test_radial_hop_ends_at_rest_at_the_hold_point_within_the_one_step_burns(hop_run):
    # Each impulse is delivered over one 10 s step and the arrival burn falls on the step after
    # T/2 = 2838.5 s; together they leave the chaser 0.209 m from the hold point in both modes
    # (an independent computation). A radial burn flown along Fehse's x axis (the V-bar) instead
    # of Hill x ends hundreds of metres away.
    ep = hop_run
    burns = fired(ep)
    assert len(burns) >= 2
    end = ep.records[burns[1]]
    assert np.linalg.norm(end.true_state[:3] - HOLD) < 2.0 * 0.209
    assert np.linalg.norm(end.true_state[3:]) < 0.05
    assert abs(end.t - T / 2.0) <= 2.0 * DT
    assert ep.terminated and success(ep)


def test_radial_hop_departs_toward_earth_to_move_forward_along_the_vbar(hop_run):
    # A radial impulse dvx moves the chaser by -4 dvx / n in half an orbit (Fehse eq 3.34), so a
    # forward hop of 500 m starts with dvx = -n 500 / 4 and no along-track part.
    first = np.asarray(hop_run.records[0].command, dtype=float)[:3]
    assert first[0] < 0.0
    assert first[1] == 0.0 and first[2] == 0.0


def test_radial_hop_costs_n_dy_over_2_in_either_mode(hop_run):
    # Fehse eq 3.37: two radial impulses of n dy / 4. The one-step burns deliver the planned
    # impulses; 1% leaves room for an arrival burn that cancels the measured residual velocity
    # (0.46 mm/s, an independent computation) instead of the planned one.
    dv = sum(r.actuation.dv_l1 for r in hop_run.records)
    assert dv == pytest.approx(N * 500.0 / 2.0, rel=1e-2)


def test_radial_hop_coast_is_predicted_clear_with_the_margin_the_environment_records(
    impulsive_hop,
):
    ep = impulsive_hop
    committed = [
        (entry, record)
        for entry, record in zip(ep.agent_log, ep.records, strict=True)
        if np.asarray(record.command)[3] > 0.0
    ]
    assert committed
    for entry, record in committed:
        truth = record.segment.minima["keep_out"].margin
        assert entry["predicted_margin_bound"] <= entry["predicted_margin"]
        assert entry["predicted_margin_bound"] <= truth
        # The same CW coast sampled and refined by the same rule from the exact state: the hop
        # passes 50.0043 m outside the keep-out sphere in both (an independent computation).
        assert entry["predicted_margin"] == pytest.approx(truth, rel=1e-6)


def torva_run(stop_angle, **config):
    config = {
        "start_state": TORVA_START,
        "action_mode": "accel",
        "max_accel": 0.05,
        "hold_point": (0.0, R_TORVA, 0.0),
        "max_steps": 80,
        **CHEAP,
        **config,
    }
    return flown(TorvaAgent(stop_angle=stop_angle), **config)


@pytest.fixture(scope="module")
def torva():
    # 72 steps. On a circle of constant range the environment's continuous-time refinement of
    # the keep-out minimum finds a flat minimum in every step and runs slowly (a known cost of
    # that refinement, not of the agent): 7 s to 23 s at the default 20 samples per step, so the
    # run checks two. The whole quarter turn is needed, since the claims are the arrival at T/8,
    # the range held over the arc and thrust losses up to 90 deg.
    return torva_run(math.pi / 2.0)


def stop_index(ep):
    """The record of the stop burn: the first along-track command after the start burn."""
    along = fired(ep, axis=1)
    assert along[0] == 0
    return along[1]


def test_torva_starts_along_plus_vbar_from_below_and_turns_clockwise(torva):
    # On the clockwise twice-rate circle v = (2 n y, -2 n x, 0), so from (-R, 0, 0) the start
    # impulse is 2 n R along +y and the chaser swings toward +V-bar (Goodman p. 137).
    first = np.asarray(torva.records[0].command, dtype=float)
    assert first[1] > 0.0 and first[2] == 0.0
    early = torva.records[5].true_state
    assert early[1] > 0.0
    assert early[0] * early[4] - early[1] * early[3] < 0.0


def test_torva_thrusts_only_the_rbar_hover_force_on_the_arc(torva):
    # Between the start and stop burns the command is -3 n^2 x at the observed x and nothing
    # else; a force with the wrong sign or an along-track term fails here. The observation is
    # exact, so the law and its recomputation differ by rounding only.
    stop = stop_index(torva)
    for prev, record in zip(torva.records[: stop - 1], torva.records[1:stop], strict=True):
        command = np.asarray(record.command, dtype=float)
        hover = -3.0 * N**2 * prev.true_state[0]
        assert command[1] == 0.0 and command[2] == 0.0
        assert command[0] == pytest.approx(hover, rel=1e-9, abs=1e-15)


def test_torva_reaches_plus_vbar_after_an_eighth_of_an_orbit(torva):
    # At rate 2 n a quarter circle takes T/8 = 709.6 s at 500 km (Goodman p. 137).
    stop = stop_index(torva)
    decided = torva.records[stop].t - DT
    assert abs(decided - T / 8.0) <= DT
    at = torva.records[stop].true_state
    assert np.linalg.norm(at[:3] - [0.0, R_TORVA, 0.0]) < 0.03 * R_TORVA
    assert np.linalg.norm(at[3:]) < 0.01


def test_torva_holds_its_radius_to_the_zero_order_hold_error(torva):
    # Holding the hover force over each 10 s step lets the range wander from R = 100 m by
    # 0.177 m when the start step carries the hover force and 1.497 m when it does not (an
    # independent computation, sampled within every step). With the force's sign flipped the
    # chaser falls tens of metres off the circle within the quarter turn.
    stop = stop_index(torva)
    ranges = np.concatenate(
        [np.linalg.norm(r.segment.states[:, :3], axis=1) for r in torva.records[:stop]]
    )
    assert np.max(np.abs(ranges - R_TORVA)) < 2.0 * 1.497


def test_torva_quarter_turn_costs_the_twice_rate_arc_from_rest_to_rest(torva):
    # 4 n R + (3 n R / 2) sin(90 deg) = 5.5 n R (Goodman p. 137); the one-step start and stop
    # burns add 1.0% to 1.5% (an independent computation).
    stop = stop_index(torva)
    dv = sum(r.actuation.dv_l1 for r in torva.records[: stop + 1])
    assert dv == pytest.approx(twice_rate_cost(N, R_TORVA, math.pi / 2.0), rel=3e-2)


# Angle flown from the +R-bar start, and Goodman's greatest free range after a thrust loss there
# in units of R (losses at 180, 150, 120 and 90 deg, the angle counted from +x toward +y): 2,
# 4.145751, 6.203627 and 7 R, recomputed at the stated constants.
THRUST_LOSS = [(0.0, 2.0), (30.0, 4.145751), (60.0, 6.203627), (90.0, 7.0)]


@pytest.mark.parametrize(("flown_deg", "greatest"), THRUST_LOSS)
def test_free_ellipse_after_torva_thrust_loss_never_comes_closer_than_the_range_at_loss(
    torva, flown_deg, greatest
):
    # vy + 2 n x = 0 on the arc, so the free motion after a thrust loss is a closed ellipse with
    # no drift whose least range is the range at the loss. The flown state sits within a step of
    # the stated angle (1.3 deg), which moves the greatest range by up to 0.065 R, and the hold
    # error lowers the least range by up to 1.8e-4 of it (an independent computation).
    stop = stop_index(torva)
    angles = [
        math.degrees(math.atan2(r.true_state[1], -r.true_state[0])) for r in torva.records[:stop]
    ]
    lost = torva.records[int(np.argmin(np.abs(np.array(angles) - flown_deg)))].true_state
    assert abs(lost[4] + 2.0 * N * lost[0]) <= 1e-9 * N * R_TORVA
    phi, _ = cw_discrete(N, 1.0)
    state, ranges = lost.copy(), []
    for _ in range(math.ceil(T)):
        state = phi @ state
        ranges.append(np.linalg.norm(state[:3]))
    at_loss = np.linalg.norm(lost[:3])
    assert min(ranges) >= at_loss * (1.0 - 1e-3)
    assert max(ranges) == pytest.approx(greatest * R_TORVA, abs=0.1 * R_TORVA)


def test_torva_fails_the_sign_check_by_construction():
    # The circle's start impulse is set by where the chaser is, not by where the goal is: from
    # the V-bar start of the check it points radially, so the along-track goals fail.
    check = sign_check(TorvaAgent(stop_angle=math.pi / 2.0), config=ApproachConfig(max_steps=360))
    assert not check.passed
    assert {"+y", "-y"} <= set(check.failures)


@pytest.fixture(scope="module")
def sequence_run():
    # 399 steps of 10 s: half an orbit of hop and 1250 s of final leg, all needed for the mode
    # sequence, the capture and the speed held to the end. Two samples per step of the keep-out
    # check; 4 s to 11 s measured on a loaded machine at the default 20.
    return flown(
        FehseSequenceAgent(), start_state=HOP_START, action_mode="accel", max_steps=900, **CHEAP
    )


def test_fehse_sequence_hops_then_flies_the_final_leg_and_captures(sequence_run):
    ep = sequence_run
    modes = [entry["mode"] for entry in ep.agent_log]
    distinct = [m for i, m in enumerate(modes) if i == 0 or m != modes[i - 1]]
    assert distinct == ["hop", "final"]
    assert ep.terminated and success(ep)
    # The hop ends one final leg (50 m) behind the hold point; the one-step burns leave it
    # 0.19 m off (an independent computation), well inside the 10 m waypoint gate.
    hop_end = ep.records[modes.index("final") - 1].true_state
    assert np.linalg.norm(hop_end[:3] - [0.0, -150.0, 0.0]) < 1.0


def test_fehse_sequence_hop_starts_with_a_burn_toward_earth(sequence_run):
    # A forward radial hop starts with dvx = -n dy / 4 (Fehse eq 3.34).
    first = np.asarray(sequence_run.records[0].command, dtype=float)
    assert first[0] < 0.0
    assert abs(first[1]) <= 1e-12 * abs(first[0]) and first[2] == 0.0


def test_fehse_sequence_final_leg_keeps_constant_speed_up_to_capture(sequence_run):
    # Fehse section 5.7: the final approach keeps its speed to contact rather than braking near
    # the target. The scratch run holds 0.04 m/s to 7e-7 m/s (an independent computation).
    ep = sequence_run
    modes = [entry["mode"] for entry in ep.agent_log]
    start = modes.index("final")
    speeds = [np.linalg.norm(r.true_state[3:]) for r in ep.records[start + 1 :]]
    assert speeds
    assert np.all(np.abs(np.array(speeds) - 0.04) <= 1e-3)


def test_fehse_sequence_longer_than_the_episode_is_refused_at_reset_with_its_duration():
    # Half an orbit of hop and 50 m at 0.04 m/s take 2838.5 + 1250 s, above the 3600 s episode.
    with pytest.raises(ValueError, match=r"4\d{3}"):
        FehseSequenceAgent().reset(view_for(action_mode="accel", max_steps=360), agent_generator(1))
    FehseSequenceAgent().reset(view_for(action_mode="accel", max_steps=900), agent_generator(1))


def first_command(agent, start, goal, **config):
    env = ApproachEnv(
        ApproachConfig(start_state=start, hold_point=goal, sensor=PerfectNav(), **config)
    )
    obs, _ = env.reset(seed=0)
    agent.reset(env.agent_view(), agent_generator(0))
    return np.asarray(agent.act(obs, env.agent_info()), dtype=float)


def test_fehse_sequence_first_command_is_radial_so_the_sign_check_fails_on_the_vbar():
    # The convention check judges the first command; the radial hop first burns -x for a goal
    # ahead (+y) and +x for a goal behind, so both along-track directions fail by design.
    ahead = first_command(
        FehseSequenceAgent(), HOP_START, (0.0, -400.0, 0.0), action_mode="accel", max_steps=900
    )
    behind = first_command(
        FehseSequenceAgent(), HOP_START, (0.0, -800.0, 0.0), action_mode="accel", max_steps=900
    )
    assert ahead[0] < 0.0 and ahead[1] == 0.0
    assert behind[0] > 0.0 and behind[1] == 0.0


# The arrival falls in step 91 (t = 910 s), the last one these runs need.
NOISY = {"action_mode": "accel", "sensor": GaussianNav(1.0, 1e-3), "max_steps": 92, **CHEAP}
TRANSFER_START = (0.0, -300.0, 0.0, 0.0, 0.0, 0.0)


def test_tracker_first_command_is_the_planned_burn():
    # The nominal starts at the first observation, so the tracking error is zero and the command
    # is the plan's feedforward alone; a tracker that drops the feedforward commands only the
    # CW cancellation terms here.
    env = ApproachEnv(ApproachConfig(start_state=TRANSFER_START, **NOISY))
    obs, _ = env.reset(seed=3)
    view = env.agent_view()
    planner = TwoImpulseAgent(mode="open_loop")
    planner.reset(view, agent_generator(1))
    planned = np.asarray(planner.act(obs, env.agent_info()), dtype=float)
    tracker = NominalTrackingAgent(TwoImpulseAgent(mode="open_loop"))
    tracker.reset(view, agent_generator(1))
    tracked = np.asarray(tracker.act(obs, env.agent_info()), dtype=float)
    assert np.linalg.norm(tracked - planned) <= 1e-9 * np.linalg.norm(planned)


def test_tracking_the_nominal_arrives_closer_than_the_open_loop_plan():
    # Fehse section 4.4.1: controlling the deviation from the nominal keeps the arrival where the
    # plan put it. Seeds 10 to 14 give medians of 2.11 m tracked and 3.14 m open loop (1.72 m
    # and 2.43 m over 20 seeds; an independent computation with the CW tracking law, k = 2).
    # Ten episodes of 92 steps; five seeds are the fewest whose medians stay apart: seeds 10 to
    # 12 alone give medians of 2.11 m and 2.42 m in the same computation, too close to rest a
    # claim on. 7.5 s to 13 s measured on a loaded machine at 20 keep-out samples per step; two
    # here.
    tf = 1.0 / N
    deviations = {"track": [], "open_loop": []}
    for seed in range(10, 15):
        for key, agent in (
            ("track", NominalTrackingAgent(TwoImpulseAgent(mode="open_loop"))),
            ("open_loop", TwoImpulseAgent(mode="open_loop")),
        ):
            ep = flown(agent, seed, start_state=TRANSFER_START, **NOISY)
            arrival = next(r for r in ep.records if r.t >= tf)
            deviations[key].append(np.linalg.norm(arrival.true_state[:3] - HOLD))
    assert np.median(deviations["track"]) < np.median(deviations["open_loop"])


AGENTS = {
    "element": lambda: ElementAgent(radial_hop),
    "torva": lambda: TorvaAgent(stop_angle=math.pi / 2.0),
    "sequence": lambda: FehseSequenceAgent(),
    "tracking": lambda: NominalTrackingAgent(TwoImpulseAgent(mode="open_loop")),
}


@pytest.mark.parametrize("make", AGENTS.values(), ids=AGENTS.keys())
def test_disc27_actions_are_refused_at_reset(make):
    with pytest.raises(ValueError):
        make().reset(view_for(action_mode="disc27", max_steps=900), agent_generator(1))


@pytest.mark.parametrize("key", ["torva", "sequence", "tracking"])
def test_continuous_thrust_agents_refuse_impulse_coast_at_reset(key):
    with pytest.raises(ValueError):
        AGENTS[key]().reset(
            view_for(action_mode="impulse_coast", max_steps=900), agent_generator(1)
        )


CASES = {
    "element": (HOP_START, {"action_mode": "impulse_coast"}),
    "torva": (TORVA_START, {"action_mode": "accel", "max_accel": 0.05}),
    "sequence": (HOP_START, {"action_mode": "accel", "max_steps": 900}),
    "tracking": (TRANSFER_START, {"action_mode": "accel"}),
}


def short(key, steps):
    """Start and config of a short run.

    The sequence refuses an episode its 4089 s plan does not fit in, so it gets 50 s steps (82
    decisions to capture) instead of 900 steps of 10 s; these claims do not depend on the step.
    """
    start, config = CASES[key]
    if key == "sequence":
        return start, {**config, **CHEAP, "dt": 50.0, "max_steps": 100}
    return start, {**config, **CHEAP, "max_steps": steps}


@pytest.mark.parametrize("key", CASES)
def test_agent_keeps_one_log_entry_per_decision(key):
    start, config = short(key, 8)
    ep = flown(AGENTS[key](), start_state=start, **config)
    assert len(ep.agent_log) == len(ep.records)


@pytest.mark.parametrize("key", CASES)
def test_agent_takes_the_protocol_arguments_by_name(key):
    agent = AGENTS[key]()
    assert list(inspect.signature(agent.reset).parameters) == ["view", "rng"]
    assert list(inspect.signature(agent.act).parameters) == ["obs", "info"]
    start, config = CASES[key]
    env = ApproachEnv(ApproachConfig(start_state=start, **config))
    obs, _ = env.reset(seed=0)
    agent.reset(view=env.agent_view(), rng=agent_generator(1))
    agent.act(obs=obs, info=env.agent_info())


class Witness:
    """Passes everything to the agent and records the info keys it was handed."""

    def __init__(self, agent):
        self.agent, self.name, self.info_keys = agent, agent.name, set()

    def reset(self, view, rng=None):
        self.agent.reset(view, rng)

    def act(self, obs, info):
        self.info_keys |= set(info)
        return self.agent.act(obs, info)


def held_objects(agent, depth=2):
    """The agent's attributes, the items of containers among them, and the same one level down."""
    found = []
    for value in vars(agent).values():
        found.append(value)
        if isinstance(value, dict):
            found.extend(value.values())
        elif isinstance(value, list | tuple):
            found.extend(value)
    if depth > 1:
        for value in list(found):
            if hasattr(value, "__dict__") and not isinstance(value, type):
                found.extend(held_objects(value, depth - 1))
    return found


@pytest.mark.parametrize("key", CASES)
def test_agent_sees_only_the_time_and_holds_no_simulator_object(key):
    agent = AGENTS[key]()
    witness = Witness(agent)
    start, config = short(key, 5)
    env = ApproachEnv(ApproachConfig(start_state=start, **config))
    run_episode(env, witness, seed=0, agent_seed=1)
    assert witness.info_keys == {"time_s"}
    forbidden = (gym.Env, CWDynamics, TwoBodyDynamics, Spacecraft, PerfectNav, GaussianNav)
    assert not any(isinstance(value, forbidden) for value in held_objects(agent))
