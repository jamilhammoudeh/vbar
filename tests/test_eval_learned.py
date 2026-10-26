"""Learned-agent entries: training runs are the unit, and a failed run is never replaced."""

import math

import numpy as np
import pytest

from vbar.eval.learned import LearnedEntry, TrainingRun, by_tuning, normalized, run_scores


def runs(count=5, failed=()):
    return tuple(
        TrainingRun(
            seed=100 + k,
            status="diverged" if k in failed else "completed",
            env_steps=10_000,
            simulated_time_s=1e5,
            wall_clock_s=60.0,
            violations=3,
            propellant_kg=1.5,
        )
        for k in range(count)
    )


def entry(**changes):
    fields = {"name": "ppo", "algorithm": "PPO", "runs": runs(), "tuned_on": ("approach-vbar",)}
    return LearnedEntry(**{**fields, **changes})


def test_fewer_than_five_training_runs_are_refused():
    with pytest.raises(ValueError, match="at least 5"):
        LearnedEntry(name="ppo", algorithm="PPO", runs=runs(4), tuned_on=("approach-vbar",))
    assert len(entry().runs) == 5


def test_a_diverged_run_counts_as_a_failure_and_cannot_be_swapped_for_a_rerun():
    declared = LearnedEntry(
        name="ppo", algorithm="PPO", runs=runs(5, failed=(2,)), tuned_on=("approach-vbar",)
    )
    assert declared.failed_runs == 1
    scores = {100: 0.9, 101: 0.8, 103: 0.95, 104: 0.85}
    found = run_scores(declared, scores)
    assert np.array_equal(found, [0.9, 0.8, 0.0, 0.95, 0.85])
    # A rerun under a new seed is not one of the declared runs.
    with pytest.raises(ValueError, match="undeclared"):
        run_scores(declared, {**scores, 999: 0.97})
    with pytest.raises(ValueError, match="no score"):
        run_scores(declared, {100: 0.9})


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"action_mode": "sampled"}, "fixed action seeds"),
        ({"action_mode": "greedy"}, "action mode"),
        ({"tau_s": 0.0}, "tau_s"),
        ({"trace_lambda": 1.5}, "trace_lambda"),
        ({"tuning": "shared", "tuned_on": ()}, "tuned on"),
        ({"configurations_tried": 0}, "configurations_tried"),
        ({"offline": True}, "data source"),
    ],
)
def test_an_entry_must_declare_what_shapes_its_result(changes, message):
    with pytest.raises(ValueError, match=message):
        entry(**changes)


def test_runs_need_distinct_seeds_and_honest_costs():
    twins = (*runs(4), runs(1)[0])
    with pytest.raises(ValueError, match="distinct"):
        LearnedEntry(name="ppo", algorithm="PPO", runs=twins, tuned_on=("approach-vbar",))
    with pytest.raises(ValueError, match="status"):
        TrainingRun(
            seed=1,
            status="restarted",
            env_steps=1,
            simulated_time_s=1.0,
            wall_clock_s=1.0,
            violations=0,
            propellant_kg=0.0,
        )
    with pytest.raises(ValueError, match="violations"):
        TrainingRun(
            seed=1,
            status="completed",
            env_steps=1,
            simulated_time_s=1.0,
            wall_clock_s=1.0,
            violations=-1,
            propellant_kg=0.0,
        )


def test_the_discount_is_declared_as_a_time_constant():
    assert entry(tau_s=1000.0).gamma(10.0) == pytest.approx(math.exp(-0.01))
    assert entry().gamma(10.0) == 1.0


def test_scores_normalised_per_start_state_map_the_references_to_zero_and_one():
    low, high = np.array([0.2, 0.5]), np.array([1.2, 4.5])
    assert np.allclose(normalized(low, low, high), 0.0)
    assert np.allclose(normalized(high, low, high), 1.0)
    assert normalized([0.7], [0.2], [1.2])[0] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="distinct"):
        normalized([1.0], [1.0], [1.0])


def test_shared_and_per_scenario_tuning_are_aggregated_apart():
    shared = entry()
    tuned = entry(tuning="per_scenario", tuned_on=())
    groups = by_tuning([shared, tuned])
    assert groups["shared"] == [shared] and groups["per_scenario"] == [tuned]
