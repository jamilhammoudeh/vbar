import numpy as np
import pytest
from metric_cases import episode, pd_episode

from vbar.agents import PDAgent
from vbar.metrics import METRICS, evaluate, mission_metrics


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
        assert a[name] == pytest.approx(b[name], rel=1e-9)


def test_every_reported_metric_is_in_the_catalogue():
    assert set(evaluate(episode())) <= set(METRICS)
    assert [METRICS[name].id for name in ("success", "progress")] == ["M1", "M5"]
