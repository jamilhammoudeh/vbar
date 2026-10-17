import numpy as np
import pytest
from metric_cases import episode, pd_episode

from vbar.agents import PDAgent
from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.metrics import WEIGHTS, composite_metrics, composite_score, weight_sensitivity
from vbar.metrics import composite as composite_module

N = mean_motion(EARTH.radius + 500e3)


def test_a_clean_success_scores_one():
    m = composite_metrics(pd_episode())
    assert m["composite_score"] == 1.0 and m["completion"] == 1.0
    assert m["infractions"] == {"keep_out": 0, "timeout": 0}


def test_a_flyby_through_the_target_is_penalized_for_the_entry_and_the_timeout():
    ep = episode()
    m = composite_metrics(ep)
    assert m["infractions"] == {"keep_out": 1, "timeout": 1}
    assert m["composite_score"] == pytest.approx(m["completion"] / (1.0 + 0.7 + 0.4))


def test_each_separate_entry_counts_once():
    # A relative orbit centred on the target with radial amplitude 40 m and along-track 80 m
    # enters the 50 m sphere each time it crosses the radial axis.
    start = np.array([40.0, 0.0, 0.0, 0.0, -80.0 * N, 0.0])
    ep = episode(start_state=tuple(start), max_steps=500)

    phi, _ = cw_discrete(N, np.linspace(0.0, 5000.0, 50001))
    values = np.linalg.norm((phi @ start)[:, :3], axis=1) - 50.0
    entries = np.sum((values[:-1] >= 0) & (values[1:] < 0)) + int(values[0] < 0)
    assert entries >= 2
    assert composite_metrics(ep)["infractions"]["keep_out"] == entries


def test_score_formula_with_custom_weights():
    assert composite_score(0.8, {"keep_out": 2, "timeout": 1}) == pytest.approx(0.8 / 2.8)
    assert composite_score(1.0, {"keep_out": 1}, {"keep_out": 0.35}) == pytest.approx(1 / 1.35)
    assert composite_score(1.0, {"passive_speed": 3}) == 1.0


def test_baseline_ranking_is_stable_under_half_and_one_and_a_half_weights():
    runs = {
        "pd": [pd_episode(max_steps=150)],
        "pd_through_target": [episode(PDAgent(), start_state=(0, 400, 0, 0, 0, 0), max_steps=300)],
        "drift": [episode()],
    }
    found = weight_sensitivity(runs)
    assert found["ranking"] == ("pd", "pd_through_target", "drift")
    assert found["stable"] and len(found["varied"]) == 2 * len(WEIGHTS)


def test_sensitivity_reports_a_ranking_that_depends_on_a_weight(monkeypatch):
    # Two speed breaches against one keep-out entry: 1 / 1.6 beats 1 / 1.7, but not 1 / 1.35.
    rows = {"a": (1.0, {"speed_limit": 2}), "b": (1.0, {"keep_out": 1})}
    monkeypatch.setattr(composite_module, "_completion_and_counts", lambda ep: rows[ep])
    found = weight_sensitivity({"a": ["a"], "b": ["b"]})
    assert found["ranking"] == ("a", "b") and not found["stable"]
    assert found["varied"][("keep_out", 0.5)] == ("b", "a")


def test_a_success_that_runs_to_the_step_limit_is_not_a_timeout():
    goal = {"hold_point": (0.0, -100.0, 0.0), "capture_radius": 5.0, "capture_speed": 0.05}
    ep = episode(
        goal={**goal, "hold_time": 30.0}, start_state=(0.0, -100.0, 0, 0, 0, 0), max_steps=6
    )
    m = composite_metrics(ep)
    assert ep.truncated and m["completion"] == 1.0
    assert m["infractions"]["timeout"] == 0 and m["composite_score"] == 1.0
