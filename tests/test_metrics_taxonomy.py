import numpy as np
import pytest
from metric_cases import DOCKING_GOAL, ConstantAgent, episode, pd_episode

from vbar.agents import LLMAgent, PDAgent
from vbar.agents.llm import text_model
from vbar.constraints import ApproachCone, SpeedLimit, SunExclusion, ZoneEntry
from vbar.dynamics import EARTH, cw_stm, mean_motion
from vbar.metrics import (
    LABELS,
    AgentReport,
    agent_metrics,
    efficiency_metrics,
    evaluate,
    failure_label,
    filter_metrics,
    mission_metrics,
    safety_metrics,
)
from vbar.safety import FilteredAgent, FilterResult

N = mean_motion(EARTH.radius + 500e3)


class StatusFilter:
    name = "status"

    def __init__(self, status):
        self.status = status

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(0.5 * action, True, self.status)


class Alternating:
    name = "alternating"

    def reset(self, view, rng=None):
        self.k = 0

    def act(self, obs, info):
        self.k += 1
        return np.array([0.2 if self.k % 2 else -0.2, 0.0, 0.0])


def label(ep, report=None):
    metrics = {
        **mission_metrics(ep),
        **efficiency_metrics(ep),
        **safety_metrics(ep),
        **filter_metrics(ep),
        **agent_metrics(ep, report),
    }
    return failure_label(ep, metrics)


AT_REST_FAR = (0.0, -300.0, 0.0, 0.0, 0.0, 0.0)
TARGET_IN_THE_WAY = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)


def test_a_success_has_no_label_and_evaluate_reports_one_for_a_failure():
    assert label(pd_episode()) is None
    assert evaluate(episode())["failure_label"] == "keep_out_entry"


@pytest.mark.parametrize(
    ("status", "expected"),
    [("solver_failed", "filter_defect"), ("infeasible_certified", "filter_infeasible")],
)
def test_a_filter_that_lets_a_hard_violation_through_is_blamed(status, expected):
    agent = FilteredAgent(PDAgent(), StatusFilter(status))
    assert label(episode(agent, start_state=TARGET_IN_THE_WAY, max_steps=300)) == expected


def test_constraint_breaches_in_label_order():
    # The flyby passes 16 m from the target at 1.5 m/s, from the +V-bar side.
    assert label(episode()) == "keep_out_entry"
    speed = SpeedLimit(0.2, 4 * N)
    assert label(episode(extra=(speed,), keep_out_radius=5.0)) == "speed_limit_breach"
    cone = ApproachCone(np.radians(10.0), active_range=120.0)
    assert label(episode(extra=(cone,), keep_out_radius=5.0)) == "cone_violation"


def test_running_out_of_propellant():
    agent = ConstantAgent([1.0, 0.0, 0.0])
    ep = episode(agent, start_state=AT_REST_FAR, max_steps=10, propellant_kg=0.2)
    assert any(rec.actuation.empty for rec in ep.records)
    assert label(ep) == "propellant_exhaustion"


def test_unreadable_replies_and_a_failed_sign_check():
    ep = episode(
        LLMAgent(text_model(lambda prompt: "thinking")), start_state=AT_REST_FAR, max_steps=5
    )
    assert label(ep) == "parse_failure"
    drift = episode(start_state=AT_REST_FAR, max_steps=5)
    assert label(drift, AgentReport(sign_check_pass=False)) == "sign_error"


def test_a_command_flipping_every_step_is_an_oscillation():
    assert label(episode(Alternating(), start_state=AT_REST_FAR, max_steps=20)) == "oscillation"


def test_a_contact_outside_the_mechanism_window():
    # 0.2 m/s at the port, then stopped before the coast reaches the 1 m keep-out sphere.
    start = (0.0, -2.5, 0.0, 0.0, 0.2, 0.0)
    ep = episode(goal=DOCKING_GOAL, start_state=start, max_steps=1, dt=5.0, keep_out_radius=1.0)
    assert label(ep) == "contact_out_of_tolerance"


def test_a_hold_that_did_not_last():
    goal = {"hold_point": (0.0, -100.0, 0.0), "capture_radius": 10.0, "capture_speed": 0.05}
    ep = episode(
        goal={**goal, "hold_time": 5000.0}, start_state=(2.0, -100.0, 0, 0, 0, 0), max_steps=200
    )
    assert label(ep) == "hold_not_maintained"


def test_a_departure_beyond_its_clear_range_on_a_drift_back_to_the_target():
    # 320 m behind, beyond the 300 m clear range, on the drift that reaches the target in 300 s.
    phi = cw_stm(N, 300.0)
    r0 = np.array([0.0, -320.0, 0.0])
    v0 = -np.linalg.solve(phi[:3, 3:], phi[:3, :3] @ r0)
    goal = {
        "task": "depart",
        "hold_point": (0.0, -330.0, 0.0),
        "capture_radius": 100.0,
        "capture_speed": 10.0,
        "clear_range": 300.0,
        "drift_horizon": 4.0 * np.pi / N,
        "drift_radius": 30.0,
    }
    assert label(episode(goal=goal, start_state=(*r0, *v0), max_steps=3)) == "returning_drift"
    # Short of the clear range the same drift is a timeout, near the clear point.
    short = {**goal, "clear_range": 400.0}
    assert label(episode(goal=short, start_state=(*r0, *v0), max_steps=3)) == "timeout_near"
    # At rest the drift stays clear, so a run failed by a forbidden zone is no returning drift.
    zone = ZoneEntry((400.0, 400.0, 400.0))
    cleared = episode(goal=goal, extra=(zone,), start_state=(*r0, 0, 0, 0), max_steps=3)
    assert mission_metrics(cleared)["time_to_complete"] == 10.0
    assert label(cleared) not in (None, "returning_drift")


def test_passing_through_the_capture_ball_too_fast_is_an_overshoot():
    ep = episode(start_state=(0.0, -60.0, 0.0, 0.0, -0.5, 0.0), keep_out_radius=10.0)
    assert label(ep) == "overshoot"


def test_timeouts_near_and_far():
    # 8 m from the hold point at rest on the V-bar, an equilibrium outside the 5 m ball.
    assert label(episode(start_state=(0.0, -108.0, 0, 0, 0, 0), max_steps=3)) == "timeout_near"
    assert label(episode(start_state=AT_REST_FAR, max_steps=3)) == "timeout_far"


def test_a_capture_inside_a_forbidden_zone_ends_as_other():
    zone = ZoneEntry((200.0, 200.0, 200.0))
    ep = episode(extra=(zone,), start_state=(0.0, -100.0, 0, 0, 0, 0))
    assert ep.terminated and label(ep) == "other"


def test_labels_are_listed_in_order():
    assert LABELS[0] == "filter_defect" and LABELS[-1] == "other" and len(LABELS) == 16


def test_a_hold_that_lasted_but_broke_another_hard_constraint_is_not_called_unmaintained():
    # At rest on the hold point for 120 s with a 60 s dwell, looking straight into the Sun; the
    # run fails on the Sun constraint and ends at its step limit.
    sun = SunExclusion(np.radians(20.0), sun_direction=(0.0, 1.0, 0.0), rate=N)
    goal = {
        "hold_point": (0.0, -100.0, 0.0),
        "capture_radius": 5.0,
        "capture_speed": 0.05,
        "hold_time": 60.0,
    }
    ep = episode(extra=(sun,), goal=goal, start_state=(0.0, -100.0, 0, 0, 0, 0), max_steps=12)
    assert label(ep) == "timeout_near"


def test_a_speed_breach_comes_before_a_cone_violation():
    both = (SpeedLimit(0.2, 4 * N), ApproachCone(np.radians(10.0), active_range=120.0))
    assert label(episode(extra=both, keep_out_radius=5.0)) == "speed_limit_breach"


def test_a_timeout_just_beyond_twice_the_capture_radius_is_far():
    # 12 m from the hold point at rest on the V-bar, against a 10 m limit for near.
    assert label(episode(start_state=(0.0, -112.0, 0, 0, 0, 0), max_steps=3)) == "timeout_far"
