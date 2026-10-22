"""Shuttle rendezvous practice: CW aim points, elevation-timed burns and the pilot model.

Goodman, History of Space Shuttle Rendezvous, JSC-63400 Rev 3 (2011). Goodman's LVLH frame has +X
along the V-bar and +Z toward the Earth, so a point Z0 "above" the target in the book is -x in the
Hill frame: x_Hill = -Z, y_Hill = X (`vbar.conventions.iss_to_hill`). Book units are ft, ft/s and
n.mi. (1 ft = 0.3048 m, 1 n.mi. = 1852 m); orbit radii are the WGS 84 equatorial radius plus the
altitude. Reference numbers quoted below as "from an independent computation" come from a separate
implementation of the same rules flown in the same scenario.
"""

import inspect
import math

import gymnasium as gym
import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.base import agent_generator, command_fraction
from vbar.agents.cw_targeting import cw_lambert, two_impulse
from vbar.agents.planning import impulse_action
from vbar.agents.shuttle import (
    ElevationTimedAgent,
    PilotAgent,
    elevation,
    football_aim,
    nsr_aim,
)
from vbar.agents.sign_check import sign_check
from vbar.conventions import iss_to_hill
from vbar.dynamics import EARTH, CWDynamics, TwoBodyDynamics, cw_discrete, mean_motion
from vbar.metrics import record_episode, success
from vbar.spacecraft import GaussianNav, PerfectNav, Spacecraft

FT = 0.3048
NMI = 1852.0
N140 = mean_motion(EARTH.radius + 140.0 * NMI)
T140 = 2.0 * math.pi / N140
N190 = mean_motion(EARTH.radius + 190.0 * NMI)
TRIGGER = math.radians(42.0)
PULSE = 0.1 * FT
REST = (0.0, -300.0, 0.0, 0.0, 0.0, 0.0)


def lvlh_ft(X, Z):
    """Goodman's in-plane LVLH point (X along the V-bar, Z toward the Earth, ft) in Hill metres."""
    return iss_to_hill(np.array([X, 0.0, Z]) * FT)


def flown(agent, seed=0, **config):
    config.setdefault("start_state", REST)
    config.setdefault("sensor", PerfectNav())
    env = ApproachEnv(ApproachConfig(**config))
    return record_episode(env, agent, seed=seed, agent_seed=seed + 1000)


def view_of(**config):
    return ApproachEnv(ApproachConfig(**config)).agent_view()


def decision_obs(ep):
    """The observation each decision was made on (navigation is exact in these runs)."""
    return [np.asarray(ep.start, dtype=float)] + [r.obs[:6] for r in ep.records[:-1]]


def fired(ep):
    """Indices of the decisions that commanded thrust."""
    return [k for k, r in enumerate(ep.records) if np.any(np.asarray(r.command)[:3] != 0.0)]


def logged(log, label):
    """Whether any log entry carries `label`, as a key set to a true value or as a string value."""
    return any(
        entry.get(label) is True or any(isinstance(v, str) and v == label for v in entry.values())
        for entry in log
    )


def coelliptic(n, x0, y0, t):
    """State on the coelliptic path at height x0 (Hill, m) through (x0, y0) at t = 0."""
    return np.array([x0, y0 - 1.5 * n * x0 * t, 0.0, 0.0, -1.5 * n * x0, 0.0])


def start_before_crossing(n, x0, t_cross):
    """Coelliptic start below the target whose elevation reaches the trigger after t_cross."""
    y_cross = x0 / math.tan(TRIGGER)
    return coelliptic(n, x0, y_cross + 1.5 * n * x0 * t_cross, 0.0)


N500 = view_of().n
# 4000 ft below at 500 km, reaching the trigger 303 s after the start.
LATE = tuple(start_before_crossing(N500, -4000.0 * FT, 303.0))


# Aim-point rules, Goodman chapter 23, pp. 178 to 180 (STS-39), Tables 23.4 and 23.5.


def test_nsr_transfer_takes_4_over_3n_which_is_19_03_min_at_140_nmi():
    # The printed expression (4/3) w is wrong; recomputed, t = 4 / (3 n) = 19.033 min, which
    # matches the 19.0 min of Tables 23.4 and 23.5.
    _, tf = nsr_aim(lvlh_ft(-32808.0, 2000.0), N140)
    assert tf * N140 == pytest.approx(4.0 / 3.0, rel=1e-12)
    assert tf / 60.0 == pytest.approx(19.03, abs=0.005)


def test_football_transfer_takes_a_quarter_period_which_is_22_42_min_at_140_nmi():
    _, tf = football_aim(lvlh_ft(-32808.0, 2000.0), N140)
    assert tf == pytest.approx(T140 / 4.0, rel=1e-12)
    assert tf / 60.0 == pytest.approx(22.42, abs=0.005)
    assert 4.0 * tf / 60.0 == pytest.approx(89.69, abs=0.005)


def test_aim_points_are_two_heights_ahead_on_the_start_height_and_on_the_v_bar():
    # Goodman's (X0 + 2 Z0, Z0) and (X0 + 2 Z0, 0) in LVLH are (x0, y0 - 2 x0, 0) and
    # (0, y0 - 2 x0, 0) in Hill. A start out of the orbit plane is aimed back into it.
    r0 = np.array([-609.6, -10000.0, 25.0])
    nsr, _ = nsr_aim(r0, N140)
    football, _ = football_aim(r0, N140)
    np.testing.assert_allclose(nsr, [-609.6, -10000.0 + 1219.2, 0.0], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(football, [0.0, -10000.0 + 1219.2, 0.0], rtol=1e-12, atol=0.0)


# Starts of Tables 23.4 and 23.5 in LVLH ft: 10 km behind at 2000 ft below and 1200 ft above,
# the near-field point and the Ti point.
STARTS = [(-32808.0, 2000.0), (-32808.0, -1200.0), (-7382.0, 400.0), (-48600.0, -1200.0)]


@pytest.mark.parametrize(("X0", "Z0"), STARTS)
def test_nsr_aim_turns_cw_targeting_into_a_coelliptic_burn(X0, Z0):
    r0 = lvlh_ft(X0, Z0)
    rf, tf = nsr_aim(r0, N140)
    v0, vf = cw_lambert(N140, r0, rf, tf)
    coelliptic_v = np.array([0.0, -1.5 * N140 * r0[0], 0.0])
    # Rounding is near 1e-15 m/s and 1e-13 m (an independent computation), so 1e-12 m/s and
    # 1e-9 m sit far above it.
    np.testing.assert_allclose(v0, coelliptic_v, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(vf, v0, rtol=0.0, atol=1e-12)
    phi, _ = cw_discrete(N140, tf)
    assert np.linalg.norm((phi @ np.r_[r0, v0])[:3] - rf) < 1e-9
    # The coelliptic drift 1.5 n Z0 in ft/s.
    assert np.linalg.norm(v0) / FT == pytest.approx(1.5 * N140 * abs(Z0), rel=1e-9)


def test_nsr_burn_from_2000_ft_below_is_3_50261_ft_per_s():
    rf, tf = nsr_aim(lvlh_ft(-32808.0, 2000.0), N140)
    v0, _ = cw_lambert(N140, lvlh_ft(-32808.0, 2000.0), rf, tf)
    assert v0[1] > 0.0
    assert np.linalg.norm(v0) / FT == pytest.approx(3.50261, rel=1e-5)


@pytest.mark.parametrize(("X0", "Z0"), STARTS)
def test_football_aim_gives_a_no_drift_ellipse_that_returns_after_one_period(X0, Z0):
    r0 = lvlh_ft(X0, Z0)
    rf, tf = football_aim(r0, N140)
    v0, _ = cw_lambert(N140, r0, rf, tf)
    # Rounding is below 1e-15 m/s and the return miss near 1e-11 m (an independent computation).
    assert abs(v0[0]) < 1e-12
    assert abs(v0[1] + 2.0 * N140 * r0[0]) < 1e-12
    phi, _ = cw_discrete(N140, T140)
    assert np.linalg.norm((phi @ np.r_[r0, v0])[:3] - r0) < 1e-6


def test_elevation_is_positive_below_the_target_and_zero_on_the_v_bar():
    assert elevation(np.array([-10.0, 0.0, 0.0])) == pytest.approx(math.pi / 2, rel=1e-12)
    assert elevation(np.array([10.0, 0.0, 0.0])) == pytest.approx(-math.pi / 2, rel=1e-12)
    assert elevation(np.array([0.0, -300.0, 0.0])) == 0.0
    dH = 4000.0
    below_behind = lvlh_ft(-dH / math.tan(TRIGGER), dH)
    assert elevation(below_behind) == pytest.approx(TRIGGER, rel=1e-12)


# STORRM terminal-phase initiation, Goodman pp. 150 and 154, at 190 n.mi.: coelliptic 4000 to
# 5000 ft below, burn at a line-of-sight elevation of 42 deg, aim 330 ft behind and 1000 ft below.
# The burn is mostly radial up (+x in Hill), as flown on p. 154, not mostly posigrade.
TPI_AIM = lvlh_ft(-330.0, 1000.0)
TRANSFER_ANGLES = (math.pi / 2, 2 * math.pi / 3)


@pytest.mark.parametrize("dH", [4000.0, 5000.0])
@pytest.mark.parametrize("n_tf", TRANSFER_ANGLES)
def test_terminal_phase_burn_at_42_deg_elevation_is_mostly_radial_up(dH, n_tf):
    r0 = lvlh_ft(-dH / math.tan(TRIGGER), dH)
    assert elevation(r0) == pytest.approx(TRIGGER, rel=1e-12)
    v_req, _ = cw_lambert(N190, r0, TPI_AIM, n_tf / N190)
    dv = v_req - np.array([0.0, -1.5 * N190 * r0[0], 0.0])
    assert 2.35 <= np.linalg.norm(dv) / FT <= 3.61
    assert dv[0] / np.linalg.norm(dv) > 0.9


# The elevation-timed agent. Slip limits are the project's own; Goodman describes the practice.


def test_elevation_timed_burn_fires_at_the_predicted_crossing_step():
    # From 4000 ft below at 500 km the elevation reaches 42 deg after 303 s: 41.87 deg at the
    # 300 s decision and 42.30 deg at 310 s (an independent computation). The burn is scheduled
    # at the crossing rounded to a step, so it fires at 300 s, before the elevation gets there.
    ep = flown(ElevationTimedAgent(trigger=TRIGGER), start_state=LATE, max_steps=35)
    burns = fired(ep)
    assert burns and burns[0] == 30
    assert not logged(ep.agent_log, "trigger_not_reached")


@pytest.mark.parametrize("dH", [4000.0, 5000.0])
@pytest.mark.parametrize("n_tf", TRANSFER_ANGLES)
def test_elevation_timed_terminal_phase_burn_matches_cw_targeting_at_190_nmi(dH, n_tf):
    # The crossing is 103 s after the start, so the burn is at the 100 s decision, at 41.87 deg.
    # |dv| there is 2.9482 and 2.9515 ft/s from 4000 ft, 3.5895 and 3.5421 ft/s from 5000 ft,
    # for transfers of 90 and 120 deg (an independent computation). A transfer of 180 deg sits
    # on the cross-track singular set of CW targeting and is refused.
    start = start_before_crossing(N190, -dH * FT, 103.0)
    ep = flown(
        ElevationTimedAgent(trigger=TRIGGER, n_tf=n_tf),
        start_state=tuple(start),
        altitude_m=190.0 * NMI,
        hold_point=tuple(TPI_AIM),
        max_accel=0.2,
        max_steps=12,
    )
    burns = fired(ep)
    assert burns and burns[0] == 10
    view = ep.view
    dv = command_fraction(view, ep.records[10].command) * view.max_accel * view.dt
    state = coelliptic(view.n, start[0], start[1], 100.0)
    v_req, _ = cw_lambert(view.n, state[:3], TPI_AIM, n_tf / view.n)
    # Exact CW stepping and the closed-form coelliptic state differ at the rounding level.
    np.testing.assert_allclose(dv, v_req - state[3:], rtol=1e-9, atol=1e-12)
    assert 2.35 <= np.linalg.norm(dv) / FT <= 3.61
    assert dv[0] / np.linalg.norm(dv) > 0.9


@pytest.mark.parametrize(("start", "max_steps"), [(REST, 40), (LATE, 20)])
def test_a_trigger_never_reached_fires_two_impulse_targeting_at_once(start, max_steps):
    # From rest on the V-bar the elevation stays 0; from 4000 ft below the crossing at 303 s
    # comes after a 20-step episode ends.
    agent = ElevationTimedAgent(trigger=TRIGGER, n_tf=1.0)
    ep = flown(agent, start_state=start, max_steps=max_steps)
    assert fired(ep)[0] == 0
    assert logged(ep.agent_log, "trigger_not_reached")
    view = ep.view
    dv1, _ = two_impulse(view.n, np.asarray(start), view.goal["hold_point"], 1.0 / view.n)
    expected, _ = impulse_action(view, dv1, 0.0)
    np.testing.assert_allclose(ep.records[0].command, expected, rtol=1e-9, atol=1e-15)


def slipped_run(delay_s, slip_s=60.0):
    """Decision steps at which the agent fires when every observation after the first places
    the chaser `delay_s` further back along the same coelliptic path."""
    view = view_of()
    x0 = -4000.0 * FT
    start = start_before_crossing(view.n, x0, 303.0)
    agent = ElevationTimedAgent(trigger=TRIGGER, slip_s=slip_s)
    agent.reset(view, agent_generator(1))
    steps = []
    for k in range(40):
        t = k * view.dt
        state = coelliptic(view.n, x0, start[1], t - (delay_s if k else 0.0))
        if np.any(command_fraction(view, agent.act(state, {"time_s": t})) != 0.0):
            steps.append(k)
    return steps, agent.log


def test_a_re_predicted_crossing_within_the_slip_limit_moves_the_burn():
    # 303 s scheduled at 300 s; 40 s later is 343 s, inside the 60 s limit, so the burn moves
    # to 340 s.
    steps, log = slipped_run(40.0)
    assert steps[0] == 34
    assert not logged(log, "slip_exceeded")


def test_a_re_predicted_crossing_beyond_the_slip_limit_is_ignored_and_logged():
    steps, log = slipped_run(200.0)
    assert steps[0] == 30
    assert logged(log, "slip_exceeded")


@pytest.mark.parametrize("n_tf", [2 * math.pi, 2 * math.pi + 1e-3, 4 * math.pi])
def test_transfer_angles_near_a_whole_revolution_are_refused_at_reset(n_tf):
    with pytest.raises(ValueError):
        ElevationTimedAgent(trigger=TRIGGER, n_tf=n_tf).reset(view_of(), agent_generator(1))


def test_elevation_timed_agent_first_command_points_at_the_goal_on_every_axis():
    # From rest on the V-bar the trigger is never reached, so the agent fires the n tf = 1 rad
    # two-impulse transfer at once.
    assert sign_check(ElevationTimedAgent(trigger=TRIGGER, n_tf=1.0)).passed


# The pilot model: 0.1 ft/s pulses (Goodman p. 143, STS-92) that null line-of-sight drift and keep
# the closing rate on a 1000 s time constant gate (pp. 57 and 74) down to the 0.1 ft/s final rate
# (Table 25.8). The band, the read time and the 1e-4 rad/s line-of-sight limit are the project's
# own choices.


def pilot_command(obs, t=0.0, agent=None, **config):
    """Fraction of a full command the pilot asks for on one observation, from a fresh reset."""
    view = view_of(**config)
    if agent is None:
        agent = PilotAgent()
        agent.reset(view, agent_generator(1))
    return command_fraction(view, agent.act(np.asarray(obs, dtype=float), {"time_s": t}))


FRACTION = PULSE / (0.02 * 10.0)  # 0.1524 of a full step at the default 0.02 m/s^2 and 10 s


def test_pilot_pulses_toward_the_hold_point_when_closing_too_slowly():
    np.testing.assert_allclose(pilot_command(REST), [0.0, FRACTION, 0.0], rtol=1e-12)


def test_pilot_pulse_is_0_1_ft_per_s_along_the_line_of_sight_in_three_dimensions():
    r = np.array([30.0, -300.0, 40.0])
    u = (np.array([0.0, -100.0, 0.0]) - r) / np.linalg.norm([0.0, -100.0, 0.0] - r)
    np.testing.assert_allclose(pilot_command(np.r_[r, 0.0, 0.0, 0.0]), FRACTION * u, rtol=1e-12)


def test_pilot_pulses_away_when_closing_too_fast():
    # 200 m out the gate is 0.2 m/s; 0.3 m/s is above the band.
    command = pilot_command((0.0, -300.0, 0.0, 0.0, 0.3, 0.0))
    np.testing.assert_allclose(command, [0.0, -FRACTION, 0.0], rtol=1e-12)


def test_pilot_coasts_while_the_closing_rate_is_on_the_gate():
    assert np.all(pilot_command((0.0, -300.0, 0.0, 0.0, 0.2, 0.0)) == 0.0)


def test_pilot_gate_never_drops_below_0_1_ft_per_s():
    # 20 m out the 1000 s gate alone would be 0.02 m/s, which a 0.02 m/s closing rate meets; the
    # floor of 0.03048 m/s puts it below the band.
    command = pilot_command((0.0, -120.0, 0.0, 0.0, 0.02, 0.0))
    np.testing.assert_allclose(command, [0.0, FRACTION, 0.0], rtol=1e-12)


def test_pilot_nulls_the_velocity_inside_the_capture_radius():
    command = pilot_command((0.0, -102.0, 0.0, 0.04, 0.0, 0.0))
    np.testing.assert_allclose(command, [-FRACTION, 0.0, 0.0], rtol=1e-12)
    # Below half a pulse a further pulse would overshoot.
    assert np.all(pilot_command((0.0, -102.0, 0.0, 0.01, 0.0, 0.0)) == 0.0)


def test_pilot_waits_the_read_time_after_every_pulse():
    # A pulse at 0 s takes the 10 s step; the 60 s read follows, so the next pulse may come at
    # 70 s even though the closing rate is still below the band.
    view = view_of(action_mode="accel")
    agent = PilotAgent()
    agent.reset(view, agent_generator(1))
    pulses = [
        t
        for t in np.arange(0.0, 150.0, view.dt)
        if np.any(pilot_command(REST, t, agent, action_mode="accel") != 0.0)
    ]
    assert pulses == [0.0, 70.0, 140.0]


def test_pilot_asks_for_the_read_time_as_the_coast_of_an_impulse():
    view = view_of(action_mode="impulse_coast")
    agent = PilotAgent()
    agent.reset(view, agent_generator(1))
    action = np.asarray(agent.act(np.asarray(REST), {"time_s": 0.0}))
    np.testing.assert_allclose(action, [0.0, FRACTION, 0.0, 60.0], rtol=1e-12)


@pytest.mark.parametrize("mode", ["impulse_coast", "accel"])
def test_each_pilot_pulse_delivers_0_1_ft_per_s(mode):
    ep = flown(PilotAgent(), action_mode=mode, max_steps=60)
    pulses = [ep.records[k] for k in fired(ep)]
    assert len(pulses) >= 3
    for record in pulses:
        assert record.actuation.dv_l2 == pytest.approx(PULSE, rel=1e-12)


def test_twelve_pulses_from_far_behind_add_up_to_1_2_ft_per_s():
    # STS-92, Goodman p. 143: 12 +X pulses gave 1.2 ft/s. From rest 1400 m behind the hold point
    # the closing rate stays below the band for over 30 pulses; CW coupling turns the line of
    # sight by at most 3.5 deg over the first 12 (an independent computation). Line-of-sight
    # pulses, which the coupling also calls for, are switched off to isolate the closing pulses.
    ep = flown(
        PilotAgent(los_rate_limit=None),
        action_mode="impulse_coast",
        start_state=(0, -1500, 0, 0, 0, 0),
        max_steps=90,
    )
    pulses = [ep.records[k] for k in fired(ep)][:12]
    assert len(pulses) == 12
    view = ep.view
    for record in pulses:
        dv = command_fraction(view, record.command) * view.max_accel * view.dt
        assert dv[1] / np.linalg.norm(dv) > math.cos(math.radians(10.0))
    assert sum(r.actuation.dv_l2 for r in pulses) / FT == pytest.approx(1.2, rel=1e-12)


@pytest.mark.parametrize("mode", ["impulse_coast", "accel"])
def test_every_pilot_decision_outside_the_read_time_follows_the_gate_rule(mode):
    # 60 steps (10 min) hold several pulses and read times; under 1 s per mode.
    ep = flown(PilotAgent(), action_mode=mode, max_steps=60)
    view = ep.view
    hold = np.array(view.goal["hold_point"])
    quiet_until = -1.0
    t = 0.0
    for obs, record in zip(decision_obs(ep), ep.records, strict=True):
        command = command_fraction(view, record.command)
        d = hold - obs[:3]
        rho = np.linalg.norm(d)
        u = d / rho
        closing = obs[3:6] @ u
        transverse = obs[3:6] - closing * u
        gate = max(rho / 1000.0, PULSE)
        if t < quiet_until - 1e-9:
            assert np.all(command == 0.0)
        elif np.linalg.norm(np.cross(d, obs[3:6])) / rho**2 > 1e-4:
            expected = -FRACTION * transverse / np.linalg.norm(transverse)
            np.testing.assert_allclose(command, expected, rtol=1e-9, atol=1e-15)
        elif closing < 0.75 * gate:
            np.testing.assert_allclose(command, FRACTION * u, rtol=1e-9, atol=1e-15)
        elif closing > 1.25 * gate:
            np.testing.assert_allclose(command, -FRACTION * u, rtol=1e-9, atol=1e-15)
        else:
            assert np.all(command == 0.0)
        if np.any(command != 0.0):
            quiet_until = t + view.dt + 60.0
        t = record.t
    assert len(fired(ep)) >= 3


@pytest.mark.parametrize("mode", ["impulse_coast", "accel"])
def test_pilot_captures_at_the_hold_point_within_the_hour(mode):
    # The claim is about the whole approach, so the episode cannot be shortened; it ends at the
    # capture, at 3180 s in impulse_coast and 3160 s in accel (an independent computation), up
    # to 316 decisions. The 1000 s gate ends at 30.48 m, where the 0.1 ft/s floor takes over.
    ep = flown(PilotAgent(), action_mode=mode)
    assert success(ep)
    assert ep.records[-1].t <= 3600.0


def test_pilot_without_line_of_sight_nulling_drifts_away_instead_of_capturing():
    # Range-rate pulses along the line of sight feed the CW coupling between radial and
    # along-track motion, which nothing nulls: the chaser ends the hour 3.1 km from the hold
    # point (an independent computation). The claim is about the full 360-step hour; impulse
    # coasts keep it to about 60 decisions.
    ep = flown(PilotAgent(los_rate_limit=None), action_mode="impulse_coast")
    assert not success(ep)
    hold = np.array(ep.view.goal["hold_point"])
    assert np.linalg.norm(ep.records[-1].true_state[:3] - hold) > 1000.0


def test_pilot_checks_line_of_sight_drift_before_the_closing_rate():
    # At rest in range the closing rate is below the band, but 0.05 m/s across the line of
    # sight 200 m out is 2.5e-4 rad/s, above the limit, and is nulled first.
    command = pilot_command((0.0, -300.0, 0.0, 0.0, 0.0, 0.05))
    np.testing.assert_allclose(command, [0.0, 0.0, -FRACTION], rtol=1e-12, atol=1e-15)


def test_pilot_nulls_line_of_sight_drift_above_the_limit():
    # 200 m from the hold point closing on the gate at 0.2 m/s with 0.05 m/s across the line of
    # sight: the line-of-sight rate is 2.5e-4 rad/s.
    start = (0.0, -300.0, 0.0, 0.0, 0.2, 0.05)
    limited = flown(PilotAgent(), action_mode="impulse_coast", start_state=start, max_steps=5)
    assert fired(limited)[0] == 0
    first = command_fraction(limited.view, limited.records[0].command)
    np.testing.assert_allclose(first, [0.0, 0.0, -FRACTION], rtol=1e-12, atol=1e-15)

    free = flown(
        PilotAgent(los_rate_limit=None),
        action_mode="impulse_coast",
        start_state=start,
        max_steps=40,
    )
    k = fired(free)[0]
    assert k > 0
    obs = decision_obs(free)[k]
    u = (np.array(free.view.goal["hold_point"]) - obs[:3]) / np.linalg.norm(
        np.array(free.view.goal["hold_point"]) - obs[:3]
    )
    command = command_fraction(free.view, free.records[k].command)
    assert abs(command @ u) == pytest.approx(FRACTION, rel=1e-9)


def test_pilot_first_pulse_points_at_the_goal_on_every_axis():
    assert sign_check(PilotAgent()).passed


def test_pilot_predicts_the_keep_out_margin_over_every_read_coast():
    ep = flown(PilotAgent(), action_mode="impulse_coast", max_steps=60)
    committed = [
        entry
        for entry, record in zip(ep.agent_log, ep.records, strict=True)
        if np.asarray(record.command)[3] > 0.0
    ]
    assert len(committed) >= 3
    for entry in committed:
        assert entry["coast_s"] == 60.0
        assert entry["predicted_margin_bound"] <= entry["predicted_margin"]
        assert entry["predicted_margin"] > 0.0


# Protocol, refusals and public information, for both agents.

AGENTS = {
    "elevation": lambda **kw: ElevationTimedAgent(trigger=TRIGGER, **kw),
    "pilot": lambda **kw: PilotAgent(**kw),
}


@pytest.mark.parametrize("kind", AGENTS)
def test_disc27_actions_are_refused_at_reset(kind):
    with pytest.raises(ValueError):
        AGENTS[kind]().reset(view_of(action_mode="disc27"), agent_generator(1))


@pytest.mark.parametrize("kind", AGENTS)
@pytest.mark.parametrize("mode", ["normalized", "impulse_coast"])
def test_agent_keeps_one_log_entry_per_decision(kind, mode):
    ep = flown(AGENTS[kind](), action_mode=mode, max_steps=20)
    assert len(ep.agent_log) == len(ep.records)


@pytest.mark.parametrize("kind", AGENTS)
def test_agent_takes_the_protocol_arguments_by_name(kind):
    agent = AGENTS[kind](name=f"{kind}-test")
    assert agent.name == f"{kind}-test"
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


@pytest.mark.parametrize("kind", AGENTS)
def test_agent_sees_only_the_time_and_holds_no_simulator_object(kind):
    agent = AGENTS[kind]()
    witness = Witness(agent)
    env = ApproachEnv(ApproachConfig(start_state=REST, max_steps=5, sensor=GaussianNav(1.0, 1e-3)))
    run_episode(env, witness, seed=0, agent_seed=1)
    assert witness.info_keys == {"time_s"}
    forbidden = (gym.Env, CWDynamics, TwoBodyDynamics, Spacecraft, PerfectNav, GaussianNav)
    assert not any(isinstance(value, forbidden) for value in held_objects(agent))
