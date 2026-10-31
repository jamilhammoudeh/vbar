"""The LQR baseline on the comparison tasks, as docs/scenarios.md quotes it.

The quoted figures are the table `python -m vbar.eval reproduce reproduce.toml` writes for 100
validation seeds per task; these tests fly the first seed of each task and check the outcome the
table reports. The two longer flights are marked `slow`.
"""

import pytest

from vbar.agents import LQRAgent
from vbar.eval import Job, SpaceTask, run_job
from vbar.scenarios import seeds


def fly(name):
    task = SpaceTask()
    env = task.make(name)
    seed, agent_seed = seeds(name, "validation", 1).pairs()[0]
    return task.score(env, task.record(env, LQRAgent(), seed, agent_seed))


def test_lqr_ends_the_spacegym_docking_task_at_the_step_limit_short_of_the_goal():
    metrics = fly("spacegym-docking")
    assert metrics["success"] is False and metrics["failure_label"] == "timeout_far"
    # The goal is within 1 m; the table's 100 runs end 11 to 23 m away.
    assert metrics["terminal_pos_error"] > 10.0


@pytest.mark.slow
def test_lqr_ends_the_spacegym_rendezvous_task_at_the_step_limit_short_of_the_goal():
    metrics = fly("spacegym-rendezvous")
    assert metrics["success"] is False and metrics["failure_label"] == "timeout_far"
    # The goal is within 10 m; the table's 100 runs end 44 to 74 m away.
    assert metrics["terminal_pos_error"] > 40.0


@pytest.mark.slow
def test_lqr_docks_in_the_afrl_docking_task():
    metrics = fly("afrl-docking")
    assert metrics["success"] is True and metrics["failure_label"] is None


def test_lqr_refuses_the_inspection_task_and_the_run_is_recorded_as_an_error():
    seed, agent_seed = seeds("afrl-inspection", "validation", 1).pairs()[0]
    result = run_job(
        Job("afrl-inspection", "validation", 0, seed, agent_seed, "comparison-v1"),
        SpaceTask(),
        LQRAgent,
    )
    assert result["error"].startswith("ValueError") and "inspect" in result["error"]
    assert result["metrics"]["success"] is False and result["failure_label"] == "error"
