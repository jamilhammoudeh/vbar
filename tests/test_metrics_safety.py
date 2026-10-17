import numpy as np
import pytest
from metric_cases import FLYBY, drift_closest_approach, drift_range_crossings, episode, pd_episode

from vbar.agents import PDAgent
from vbar.constraints import ApproachCone, ClosingSpeedLimit, SpeedLimit, ZoneEntry, closing_speed
from vbar.dynamics import EARTH, cw_discrete, cw_stm, mean_motion
from vbar.metrics import safety_metrics

N = mean_motion(EARTH.radius + 500e3)
# Dunlap et al. 2021, eq. 14.
SPEED = SpeedLimit(0.2, 4 * N)
CLOSING = ClosingSpeedLimit(0.2, 4 * N)


def test_keep_out_margin_depth_and_duration_match_the_closed_form_flyby():
    ep = episode()
    m = safety_metrics(ep)
    closest = drift_closest_approach(FLYBY, ep.view.n, 200.0)
    entry, exit_ = drift_range_crossings(FLYBY, ep.view.n, 50.0, 200.0)
    assert m["min_margin"]["keep_out"] == pytest.approx(closest - 50.0, abs=1e-3)
    assert m["min_margin_bound"]["keep_out"] <= m["min_margin"]["keep_out"]
    assert m["violation_depth"]["keep_out"] == pytest.approx(50.0 - closest, abs=1e-3)
    assert m["violation_duration"]["keep_out"] == pytest.approx(exit_ - entry, abs=1e-6)


def test_a_clean_approach_has_a_positive_margin_and_no_violation_time():
    m = safety_metrics(pd_episode())
    assert m["min_margin"]["keep_out"] > 0.0
    assert m["violation_depth"]["keep_out"] == 0.0
    assert m["violation_duration"]["keep_out"] == 0.0


def test_margins_of_constraints_outside_the_environment_match_the_environment_search():
    # The environment searches its own constraints on the dynamics model; others are searched on
    # the trajectory rebuilt from the records. On Clohessy-Wiltshire both are exact.
    start = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)
    guarded = episode(PDAgent(), extra=(SPEED, CLOSING), start_state=start, max_steps=15)
    plain = episode(PDAgent(), start_state=start, max_steps=15)
    inside = safety_metrics(guarded)
    outside = safety_metrics(plain, (*plain.constraints, SPEED, CLOSING))
    for name in ("speed_limit", "closing_speed_limit"):
        assert outside["min_margin"][name] == pytest.approx(inside["min_margin"][name], abs=1e-9)
        assert outside["violation_duration"][name] == pytest.approx(
            inside["violation_duration"][name], abs=1e-6
        )


def test_closing_speed_ratio_matches_a_dense_closed_form_maximum():
    start = np.array([0.0, 300.0, 20.0, 0.05, -0.8, 0.0])
    ep = episode(extra=(CLOSING,), start_state=tuple(start), max_steps=12)
    phi, _ = cw_discrete(N, np.linspace(0.0, 120.0, 120001))
    states = phi @ start
    rho = np.linalg.norm(states[:, :3], axis=1)
    dense = np.max(closing_speed(states) / (0.2 + 4 * N * rho))
    assert safety_metrics(ep)["closing_speed_ratio"] == pytest.approx(dense, abs=1e-7)


def test_time_to_keep_out_at_the_start_of_a_straight_closing():
    # 250 m from the 50 m sphere at 0.5 m/s straight in: 500 s.
    start = np.array([0.0, 300.0, 0.0, 0.0, -0.5, 0.0])
    ep = episode(start_state=tuple(start), max_steps=1)
    later = cw_stm(N, 10.0) @ start
    rho = np.linalg.norm(later[:3])
    expected = min(500.0, (rho - 50.0) / (-(later[:3] @ later[3:]) / rho))
    assert safety_metrics(ep)["min_time_to_keep_out"] == pytest.approx(expected, rel=1e-12)


def test_time_to_keep_out_is_infinite_when_never_closing():
    ep = episode(start_state=(0.0, -100.0, 0.0, 0.0, 0.0, 0.0), max_steps=3)
    assert safety_metrics(ep)["min_time_to_keep_out"] == np.inf
    assert np.isnan(safety_metrics(ep)["closing_speed_ratio"])


def test_safety_metrics_do_not_depend_on_the_decision_step():
    cone = ApproachCone(np.radians(10.0), active_range=120.0)
    extra = (SPEED, CLOSING, cone)
    reference = safety_metrics(episode(extra=extra))
    for dt in (4.0, 25.0):
        m = safety_metrics(episode(extra=extra, dt=dt, max_steps=int(200 / dt)))
        for name in ("keep_out", "speed_limit", "closing_speed_limit", "approach_cone"):
            assert m["min_margin"][name] == pytest.approx(reference["min_margin"][name], abs=1e-6)
            assert m["violation_duration"][name] == pytest.approx(
                reference["violation_duration"][name], abs=1e-5
            )
        assert m["closing_speed_ratio"] == pytest.approx(reference["closing_speed_ratio"], abs=1e-7)


def test_safety_metrics_are_unchanged_by_mirroring_the_orbit_normal():
    start = np.array([20.0, 300.0, 30.0, 0.01, -0.6, -0.03])
    mirrored = start * [1, 1, -1, 1, 1, -1]
    a = safety_metrics(episode(extra=(SPEED, CLOSING), start_state=tuple(start)))
    b = safety_metrics(episode(extra=(SPEED, CLOSING), start_state=tuple(mirrored)))
    for key in ("min_margin", "violation_duration"):
        for name, value in a[key].items():
            assert b[key][name] == pytest.approx(value, abs=1e-9)


def test_entering_a_zone_in_the_last_sample_interval_before_go_counts():
    # The flyby enters a 100 m sphere at about 33.3 s; GO comes at the step boundary 33.6 s.
    zone = ZoneEntry((100.0, 100.0, 100.0), go_time=33.6)
    ep = episode(extra=(zone,), soft=("zone_entry",), dt=8.4, max_steps=6)
    entry = drift_range_crossings(FLYBY, N, 100.0, 40.0)[0]
    at_go = np.linalg.norm((cw_stm(N, 33.6) @ np.array(FLYBY))[:3])
    m = safety_metrics(ep)
    assert entry < 33.6
    assert m["min_margin"]["zone_entry"] == pytest.approx(at_go - 100.0, abs=1e-3)
    assert m["min_margin_bound"]["zone_entry"] <= m["min_margin"]["zone_entry"]
    assert m["violation_duration"]["zone_entry"] == pytest.approx(33.6 - entry, abs=1e-6)
