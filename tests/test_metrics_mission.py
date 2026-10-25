from dataclasses import replace

import numpy as np
import pytest
from metric_cases import DOCKING_GOAL, episode, pd_episode
from scipy.optimize import brentq

from vbar.agents import DriftAgent, PDAgent
from vbar.dynamics import EARTH, cw_stm, mean_motion
from vbar.metrics import METRICS, contact, evaluate, mission_metrics, record_episode
from vbar.scenarios import SCENARIOS, Fixed, InertialPointing, Misbriefed, Passive, make
from vbar.scenarios.registry import entry_class
from vbar.scenarios.targets import body_point, body_point_velocity

N = mean_motion(EARTH.radius + 500e3)


def test_pd_from_behind_succeeds_and_stops_inside_the_capture_ball():
    ep = pd_episode()
    m = mission_metrics(ep)
    assert m["success"] and ep.terminated
    assert m["terminal_pos_error"] < 5.0 and m["terminal_vel_error"] < 0.05
    assert m["time_to_complete"] == ep.duration
    # Captured at the end means at least 1 - 5 / d0 of the way there.
    assert m["progress"] >= 1.0 - 5.0 / np.linalg.norm(ep.start[:3] - (0.0, -100.0, 0.0))


def test_resting_at_the_vbar_hold_point_succeeds_at_the_first_step():
    # The V-bar point is a Clohessy-Wiltshire equilibrium, so free drift stays on it exactly.
    ep = episode(start_state=(0.0, -100.0, 0.0, 0.0, 0.0, 0.0))
    m = mission_metrics(ep)
    assert m["success"]
    assert m["time_to_complete"] == 10.0
    assert m["terminal_pos_error"] == 0.0 and m["terminal_vel_error"] == 0.0
    assert m["progress"] == 1.0


def test_a_capture_after_entering_the_keep_out_zone_is_not_a_success():
    # The target lies between the start and the hold point; PD flies straight through it.
    ep = episode(PDAgent(), start_state=(0.0, 400.0, 0.0, 0.0, 0.0, 0.0), max_steps=300)
    m = mission_metrics(ep)
    assert ep.terminated and not m["success"]
    assert m["time_to_complete"] == ep.duration


def test_free_drift_through_the_target_fails_with_known_progress():
    ep = episode()
    m = mission_metrics(ep)
    d0 = 250.0  # from (0, 150, 0) to (0, -100, 0)
    d_end = np.linalg.norm(ep.states[-1, :3] - (0.0, -100.0, 0.0))
    assert not m["success"]
    assert m["time_to_complete"] == 200.0
    assert m["terminal_pos_error"] == pytest.approx(d_end, rel=1e-12)
    assert m["progress"] == pytest.approx(max(0.0, 1.0 - d_end / d0), abs=1e-12)


def test_moving_away_gives_zero_progress():
    ep = episode(start_state=(0.0, -200.0, 0.0, 0.0, -1.0, 0.0), max_steps=10)
    assert mission_metrics(ep)["progress"] == 0.0


def test_terminal_errors_do_not_depend_on_the_decision_step():
    reference = mission_metrics(episode())
    for dt in (5.0, 20.0):
        m = mission_metrics(episode(dt=dt, max_steps=int(200 / dt)))
        for name in ("terminal_pos_error", "terminal_vel_error", "progress"):
            assert m[name] == pytest.approx(reference[name], rel=1e-9)


def test_mission_metrics_are_unchanged_by_mirroring_the_orbit_normal():
    # Clohessy-Wiltshire motion is symmetric under z -> -z, and so is the goal on the V-bar.
    start = np.array([20.0, -400.0, 30.0, 0.01, 0.02, -0.03])
    mirrored = start * [1, 1, -1, 1, 1, -1]
    a = mission_metrics(pd_episode(start_state=tuple(start)))
    b = mission_metrics(pd_episode(start_state=tuple(mirrored)))
    for name in a:
        assert a[name] == pytest.approx(b[name], rel=1e-9, nan_ok=True)


def test_every_reported_metric_is_in_the_catalogue():
    assert set(evaluate(episode())) <= set(METRICS)
    assert [METRICS[name].id for name in ("success", "progress")] == ["M1", "M5"]


def drift_state(start, t):
    return cw_stm(N, t) @ np.asarray(start, dtype=float)


def test_contact_is_scored_at_the_exact_port_plane_crossing():
    # Coasting at 0.08 m/s toward the port 2 m behind the target, 5 cm off the axis. A longer
    # coast would drift off the axis: Clohessy-Wiltshire coupling moves it 0.9 m radially in 100 s.
    # The episode stops before the coast carries it inside the 1 m keep-out sphere.
    start = (0.05, -3.0, 0.0, 0.0, 0.08, 0.004)
    ep = episode(goal=DOCKING_GOAL, start_state=start, max_steps=2, keep_out_radius=1.0)
    t = brentq(lambda s: drift_state(start, s)[1] + 2.0, 0.0, 30.0, xtol=1e-12)
    x, y, z, vx, vy, vz = drift_state(start, t)
    event = contact(ep)
    assert event.t == pytest.approx(t, abs=1e-6)
    assert event.lateral_error == pytest.approx(np.hypot(x, z), abs=1e-9)
    assert event.axial_speed == pytest.approx(vy, abs=1e-9)
    assert event.lateral_speed == pytest.approx(np.hypot(vx, vz), abs=1e-9)
    expected = min(
        (0.1 - np.hypot(x, z)) / 0.1,
        (0.03 - np.hypot(vx, vz)) / 0.03,
        (0.1 - vy) / 0.1,
        (vy - 0.05) / 0.05,
    )
    m = mission_metrics(ep)
    assert m["capture_margin"] == pytest.approx(expected, abs=1e-8)
    assert m["success"] and m["time_to_complete"] == pytest.approx(t, abs=1e-6)


def test_a_contact_too_fast_for_the_mechanism_fails_with_a_negative_margin():
    start = (0.0, -3.0, 0.0, 0.0, 0.2, 0.0)
    m = mission_metrics(
        episode(goal=DOCKING_GOAL, start_state=start, max_steps=3, keep_out_radius=1.0)
    )
    assert not m["success"] and m["time_to_complete"] == 30.0
    # The 0.2 m/s impact is twice the 0.10 m/s top of the probe-drogue window.
    assert m["capture_margin"] == pytest.approx(-1.0, abs=1e-3)


def test_the_shuttle_iss_contact_speed_is_judged_by_the_scenarios_window():
    # STS-130 contact at 0.113 ft/s = 0.0344 m/s (Goodman p. 244): inside an APDS-like window of
    # 0.025 to 0.05 m/s (upper bound Fehse app. C, lower bound ours), below the probe-drogue window
    # of 0.05 to 0.10.
    start = (0.0, -2.5, 0.0, 0.0, 0.113 * 0.3048, 0.0)
    apds = {**DOCKING_GOAL, "axial_speed_window": (0.025, 0.05)}
    accepted = episode(goal=apds, start_state=start, max_steps=2, keep_out_radius=1.0)
    rejected = episode(goal=DOCKING_GOAL, start_state=start, max_steps=2, keep_out_radius=1.0)
    assert contact(accepted).axial_speed == pytest.approx(0.0344, abs=1e-4)
    assert mission_metrics(accepted)["success"]
    m = mission_metrics(rejected)
    assert not m["success"]
    assert m["capture_margin"] == pytest.approx((0.0344 - 0.05) / 0.05, abs=2e-3)


def test_a_contact_between_two_samples_ten_seconds_apart_is_found():
    start = (0.0, -2.5, 0.0, 0.0, 0.08, 0.0)
    ep = episode(
        goal=DOCKING_GOAL, start_state=start, max_steps=2, keep_out_radius=1.0, violation_samples=2
    )
    t = brentq(lambda s: drift_state(start, s)[1] + 2.0, 0.0, 20.0, xtol=1e-12)
    assert contact(ep).t == pytest.approx(t, abs=1e-6)


def test_no_crossing_means_no_contact_metrics():
    m = mission_metrics(episode(goal=DOCKING_GOAL, start_state=(0, -10.0, 0, 0, 0, 0), max_steps=3))
    assert not m["success"]
    assert np.isnan(m["contact_lateral_error"]) and np.isnan(m["capture_margin"])


def test_hold_dwell_is_the_time_until_drift_leaves_the_capture_ball():
    # From rest 2 m above the hold point, drift falls back along-track (Fehse eq 3.26).
    start = (2.0, -100.0, 0.0, 0.0, 0.0, 0.0)
    goal = {"hold_point": (0.0, -100.0, 0.0), "capture_radius": 10.0, "capture_speed": 0.05}

    def gap(t):
        x, y = drift_state(start, t)[:2]
        return np.hypot(x, y + 100.0) - 10.0

    leave = brentq(gap, 100.0, 3000.0, xtol=1e-12)
    for hold_time, succeeds in ((0.9 * leave, True), (1.1 * leave, False)):
        ep = episode(goal={**goal, "hold_time": hold_time}, start_state=start, max_steps=200)
        m = mission_metrics(ep)
        assert m["hold_dwell"] == pytest.approx(leave, abs=1e-6)
        assert m["success"] is succeeds
        if succeeds:
            assert m["time_to_complete"] == pytest.approx(hold_time, abs=1e-9)


def test_a_window_open_at_zero_speed_has_no_lower_criterion():
    goal = {**DOCKING_GOAL, "axial_speed_window": (0.0, 0.1)}
    start = (0.0, -3.0, 0.0, 0.0, 0.08, 0.0)
    ep = episode(goal=goal, start_state=start, max_steps=2, keep_out_radius=1.0)
    event = contact(ep)
    assert mission_metrics(ep)["capture_margin"] == pytest.approx(
        min(
            (0.1 - event.axial_speed) / 0.1,
            1.0 - event.lateral_error / 0.1,
            1.0 - event.lateral_speed / 0.03,
        )
    )


def test_a_start_behind_the_port_plane_is_not_a_contact():
    # Starting 0.5 m past the plane and backing away crosses it from the wrong side.
    start = (0.0, -1.5, 0.0, 0.0, -0.1, 0.0)
    ep = episode(goal=DOCKING_GOAL, start_state=start, max_steps=2, keep_out_radius=1.0)
    assert contact(ep) is None


def test_a_plane_crossing_far_from_the_port_is_not_a_contact():
    # Passing the port plane 60 m beside the target is no contact when contact counts only
    # within 50 m of the port; without a range the same crossing reads as a 60 m miss.
    start = (60.0, -6.0, 0.0, 0.0, 0.2, 0.0)
    far = episode(goal=DOCKING_GOAL, start_state=start, max_steps=8, keep_out_radius=1.0)
    event = contact(far)
    assert event is not None and event.lateral_error > 55.0
    ranged = {**DOCKING_GOAL, "contact_range": 50.0}
    near = episode(goal=ranged, start_state=start, max_steps=8, keep_out_radius=1.0)
    assert contact(near) is None
    assert np.isnan(mission_metrics(near)["capture_margin"])


@pytest.mark.parametrize("briefed", [None, Passive()], ids=["told", "misbriefed"])
def test_contact_at_a_turning_port_is_judged_as_the_environment_judges_it(briefed):
    # An inertially pointing target turns its port at -n; the chaser starts 3 m out on the axis
    # moving with the port and closing at 0.07 m/s. The metrics follow the true mode, even when
    # agents were told the target is LVLH-stabilised, and measure speeds against the port.
    truth = InertialPointing(0.7)
    spec = SCENARIOS["inertial-target"]
    n = make("inertial-target").n
    port = body_point(truth, (0.0, -2.0, 0.0), 0.0, n)
    axis = body_point(truth, (0.0, -1.0, 0.0), 0.0, n)
    v_port = body_point_velocity(truth, (0.0, -2.0, 0.0), 0.0, n)
    flown = replace(
        spec,
        target=truth if briefed is None else Misbriefed(actual=truth, briefed=briefed),
        dispersions={
            "start_pos": Fixed(tuple(port + 3.0 * axis)),
            # 4 mm/s across the axis offsets the Coriolis drift over the 43 s coast.
            "start_vel": Fixed(tuple(v_port - 0.07 * axis + 0.004 * np.cross(axis, (0, 0, 1)))),
        },
    )
    env = entry_class(flown)(flown)
    ep = record_episode(env, DriftAgent(), seed=1, agent_seed=2)
    info = env._info()
    event = contact(ep)
    assert event.t == pytest.approx(info["contact_time_s"], abs=1e-6)
    assert event.lateral_error == pytest.approx(info["contact_lateral_error_m"], abs=1e-9)
    assert event.axial_speed == pytest.approx(info["contact_axial_speed_mps"], abs=1e-9)
    assert event.lateral_speed == pytest.approx(info["contact_lateral_speed_mps"], abs=1e-9)
    assert event.margin == pytest.approx(info["capture_margin"], abs=1e-7)
    m = mission_metrics(ep)
    assert m["success"] is info["success"] is True
    # The terminal errors are taken from the port where it stands at the end, which has turned
    # about 0.1 m from where the goal briefed it at t = 0.
    end = ep.records[-1].true_state
    port_end = body_point(truth, (0.0, -2.0, 0.0), ep.duration, n)
    v_end = body_point_velocity(truth, (0.0, -2.0, 0.0), ep.duration, n)
    assert m["terminal_pos_error"] == pytest.approx(np.linalg.norm(end[:3] - port_end), abs=1e-12)
    assert m["terminal_vel_error"] == pytest.approx(np.linalg.norm(end[3:] - v_end), abs=1e-12)
    assert np.linalg.norm(port_end - port) > 0.05
    # Riding with the port, the chaser's speed across it stays near zero, while its Hill-frame
    # velocity carries the port's 2 n = 2.2 mm/s across the axis.
    v = ep.records[-1].true_state[3:6]
    a = body_point(truth, (0.0, -1.0, 0.0), event.t, n)
    hill_lateral = float(np.linalg.norm(v - (v @ a) * a))
    assert event.lateral_speed < 1e-3 < hill_lateral


@pytest.mark.parametrize("beside", [49.95, 50.06], ids=["inside", "outside"])
def test_contact_range_is_judged_at_the_crossing_in_both(beside):
    # A pass through the port plane about 50 m beside the port, sliding 0.38 m/s sideways: the
    # environment and the metrics judge the 50 m contact range at the crossing point, so a
    # crossing at the range's edge counts for both or for neither.
    spec = SCENARIOS["docking"]
    flown = replace(
        spec,
        config={**spec.config, "max_steps": 2},
        dispersions={
            "start_pos": Fixed((beside, -2.0026, 0.0)),
            "start_vel": Fixed((-0.38, 0.02, 0.0)),
        },
    )
    env = entry_class(flown)(flown)
    ep = record_episode(env, DriftAgent(), seed=1, agent_seed=2)
    info = env._info()
    event = contact(ep)
    assert (event is not None) is info["contact"] is (beside < 50.0)
    if event is not None:
        # The environment judges the state where the step ends, a nanometre past the plane:
        # 5e-8 s later at 0.02 m/s closing, 2e-8 m further sideways at 0.38 m/s.
        assert event.t == pytest.approx(info["contact_time_s"], abs=1e-6)
        assert event.lateral_error == pytest.approx(info["contact_lateral_error_m"], abs=1e-6)


def test_a_departure_goal_is_judged_at_its_clear_point_not_at_its_port():
    # A departure goal names its port and a clear point as a hold point; crossing the port plane
    # on the way is no contact, and the terminal error is the distance to the clear point.
    goal = {
        "port": (0.0, -2.0, 0.0),
        "port_axis": (0.0, -1.0, 0.0),
        "hold_point": (0.0, -100.0, 0.0),
        "capture_radius": 50.0,
        "capture_speed": 10.0,
    }
    start = (0.0, -3.0, 0.0, 0.0, 0.08, 0.0)
    ep = episode(goal=goal, start_state=start, max_steps=3, keep_out_radius=1.0)
    m = mission_metrics(ep)
    assert np.isnan(m["capture_margin"])
    end = ep.records[-1].true_state
    assert m["terminal_pos_error"] == pytest.approx(np.linalg.norm(end[:3] - (0, -100.0, 0)))
    assert not m["success"]
