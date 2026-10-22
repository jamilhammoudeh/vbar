import dataclasses
import math
import time
from dataclasses import replace

import numpy as np
import pytest
from metric_cases import ConstantAgent, GuardedApproach, episode, pd_episode
from scipy.stats import trim_mean

from vbar import ApproachConfig
from vbar.agents import PDAgent
from vbar.constraints import SegmentMinimum, SpeedLimit
from vbar.metrics import (
    MetricSettings,
    escape_arc,
    filter_behaviour,
    filter_conservatism,
    filter_metrics,
    iqm,
    paired_filter_metrics,
    record_episode,
)
from vbar.metrics.efficiency import delta_v
from vbar.safety import FilteredAgent, FilterResult, PassThroughFilter, run_filter
from vbar.scenarios import SCENARIOS, Fixed
from vbar.scenarios.registry import entry_class


class ScriptedFilter:
    """Halves every command and reports the statuses and compute times it is given, in turn."""

    name = "scripted"

    def __init__(self, statuses=("feasible",), compute=(0.001,), **fields):
        self.statuses, self.compute, self.fields = statuses, compute, fields

    def reset(self, view):
        self.k = 0

    def filter(self, obs, action):
        status = self.statuses[self.k % len(self.statuses)]
        spent = self.compute[self.k % len(self.compute)]
        self.k += 1
        return FilterResult(0.5 * action, True, status, compute_s=spent, **self.fields)


def test_halving_every_command_gives_rate_one_and_the_mean_half_command_size():
    agent = FilteredAgent(PDAgent(), ScriptedFilter())
    ep = episode(agent, start_state=(0.0, -400.0, 30.0, 0.0, 0.0, 0.0), max_steps=30)
    m = filter_metrics(ep)
    # Normalized commands: a full command is 0.02 m/s^2; the widest box change is 2 sqrt(3) of it.
    sizes = [np.linalg.norm(0.5 * rec.command) * 0.02 for rec in ep.records]
    assert m["intervention_rate"] == 1.0 and m["clip_rate"] == 0.0
    assert m["intervention_size"] == pytest.approx(np.mean(sizes), rel=1e-12)
    assert m["intervention_size_normalized"] == pytest.approx(
        np.mean(sizes) / (2 * math.sqrt(3.0) * 0.02), rel=1e-12
    )


def test_pass_through_never_intervenes_and_no_filter_gives_nan():
    ep = episode(FilteredAgent(PDAgent(), PassThroughFilter()), start_state=(0, -400, 0, 0, 0, 0))
    m = filter_metrics(ep)
    assert m["intervention_rate"] == 0.0 and math.isnan(m["intervention_size"])
    assert m["infeasible_steps"] == 0.0 and m["solver_failures"] == 0.0
    assert all(math.isnan(v) for v in filter_metrics(pd_episode(max_steps=3)).values())


def test_certified_infeasibility_and_solver_failures_are_counted_apart():
    statuses = ("feasible", "infeasible_certified", "solver_failed", "infeasible_certified")
    agent = FilteredAgent(PDAgent(), ScriptedFilter(statuses))
    m = filter_metrics(episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=8))
    assert m["infeasible_steps"] == 4.0 and m["solver_failures"] == 2.0


def test_decisions_inside_the_commit_zone_are_not_counted():
    # Inside the commit zone, the last metre before contact, the filter only monitors (Fehse 4.4).
    agent = FilteredAgent(PDAgent(), ScriptedFilter(("solver_failed",)))
    ep = episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=4)
    assert filter_metrics(ep)["solver_failures"] == 4.0
    assert filter_metrics(ep, MetricSettings(commit_radius=500.0))["solver_failures"] == 0.0


class PatternSleepFilter:
    """Sleeps 0, 2 and 20 ms in turn and claims no compute time."""

    name = "pattern-sleep"
    sleeps = (0.0, 0.002, 0.020)

    def reset(self, view):
        self.k = 0

    def filter(self, obs, action):
        slept = self.sleeps[self.k % len(self.sleeps)]
        self.k += 1
        if slept:
            time.sleep(slept)
        return FilterResult(action, False, compute_s=0.0)


def test_compute_time_median_and_95th_percentile_are_of_the_measured_times():
    agent = FilteredAgent(PDAgent(), PatternSleepFilter())
    ep = episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=9)
    measured = [rec.filter_result.compute_s for rec in ep.records]
    m = filter_metrics(ep)
    assert m["filter_compute_median"] == pytest.approx(np.median(measured), rel=1e-12)
    assert m["filter_compute_p95"] == pytest.approx(np.percentile(measured, 95), rel=1e-12)
    # Three decisions of each sleep: the median is a 2 ms decision, the 95th percentile lies
    # between two 20 ms ones.
    for slept, spent in zip(PatternSleepFilter.sleeps * 3, measured, strict=True):
        assert spent >= slept
    assert m["filter_compute_median"] >= 0.002
    assert m["filter_compute_p95"] >= 0.020


def test_paired_costs_compare_the_same_seeds_with_and_without_the_filter():
    config = {"start_state": None, "max_steps": 40}
    on = [episode(FilteredAgent(PDAgent(), ScriptedFilter()), seed=s, **config) for s in range(5)]
    off = [episode(PDAgent(), seed=s, **config) for s in range(5)]
    m = paired_filter_metrics(on, off[::-1])
    costs = [delta_v(a) - delta_v(b) for a, b in zip(on, off, strict=True)]
    assert m["safety_dv_cost_by_seed"] == pytest.approx(dict(enumerate(costs)))
    assert m["safety_dv_cost"] == pytest.approx(trim_mean(costs, 0.25))
    assert -1.0 <= m["filter_dependence"] <= 1.0
    with pytest.raises(ValueError, match="same distinct seeds"):
        paired_filter_metrics(on, off[:4])


def test_interquartile_mean_drops_the_outer_quarters():
    assert iqm([1.0, 2.0, 3.0, 4.0, 100.0, -50.0, 5.0, 6.0]) == pytest.approx(3.5)


class Rewriting:
    """Returns `rewrite(action)`, reported as an intervention, with the given fields."""

    name = "rewriting"

    def __init__(self, rewrite, **fields):
        self.rewrite, self.fields = rewrite, fields

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(self.rewrite(action), True, **self.fields)


def test_clipping_to_the_thrust_limit_is_counted_apart_from_safety_activity():
    over = ConstantAgent([0.0, 3.0, 0.0])
    agent = FilteredAgent(over, Rewriting(lambda a: np.clip(a, -1.0, 1.0)))
    m = filter_metrics(episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=5))
    assert m["intervention_rate"] == 0.0 and m["clip_rate"] == 1.0
    assert np.isnan(m["intervention_size"])


def test_scaling_a_command_onto_the_ball_limit_is_clipping():
    over = ConstantAgent([2.0, 2.0, 0.0])
    agent = FilteredAgent(over, Rewriting(lambda a: a / np.linalg.norm(a)))
    ep = episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=3, thrust_limit="ball")
    m = filter_metrics(ep)
    assert m["intervention_rate"] == 0.0 and m["clip_rate"] == 1.0


def test_a_reported_row_on_a_clipped_command_is_activity_not_clipping():
    over = ConstantAgent([0.0, 3.0, 0.0])
    agent = FilteredAgent(over, Rewriting(lambda a: np.clip(a, -1.0, 1.0), active=True))
    m = filter_metrics(episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=3))
    assert m["intervention_rate"] == 1.0 and m["clip_rate"] == 0.0


def test_a_filter_cannot_hide_a_change_by_reporting_no_active_row():
    agent = FilteredAgent(PDAgent(), Rewriting(lambda a: 0.0 * a, active=False))
    ep = episode(agent, start_state=(0, -400, 30, 0, 0, 0), max_steps=5)
    m, behaviour = filter_metrics(ep), filter_behaviour(ep)
    assert m["intervention_rate"] == 1.0 and m["clip_rate"] == 0.0
    assert m["intervention_size"] > 0.0
    # Zeroing PD's commands from rest keeps the chaser below the 0.01 m/s stall speed.
    assert behaviour["stall_time"] == 50.0


def test_a_filter_that_zeroes_the_command_in_place_still_intervenes():
    def zero_in_place(action):
        action[:] = 0.0
        return action

    agent = FilteredAgent(PDAgent(), Rewriting(zero_in_place, active=False))
    ep = episode(agent, start_state=(0, -400, 30, 0, 0, 0), max_steps=5)
    assert filter_metrics(ep)["intervention_rate"] == 1.0
    assert np.linalg.norm(ep.records[0].command) > 0.0


def test_only_the_flown_change_counts_so_intervention_size_stays_within_the_actuator():
    # Reversing a command ten times the box limit on every axis flies the widest change, F2n = 1.
    agent = FilteredAgent(ConstantAgent([10.0, 10.0, 10.0]), Rewriting(lambda a: -a))
    m = filter_metrics(episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=3))
    assert m["intervention_size"] == pytest.approx(2 * math.sqrt(3) * 0.02, rel=1e-12)
    assert m["intervention_size_normalized"] == pytest.approx(1.0, rel=1e-12)
    # Halving 1.5 of the limit flies 0.75 instead of 1.
    agent = FilteredAgent(ConstantAgent([0.0, 1.5, 0.0]), Rewriting(lambda a: 0.5 * a))
    m = filter_metrics(episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=3))
    assert m["intervention_size"] == pytest.approx(0.25 * 0.02, rel=1e-12)


def test_a_reported_safety_row_counts_even_when_the_action_did_not_move():
    class Watching:
        name = "watching"

        def reset(self, view):
            pass

        def filter(self, obs, action):
            return FilterResult(action, False, active=True)

    m = filter_metrics(
        episode(
            FilteredAgent(PDAgent(), Watching()), start_state=(0, -400, 0, 0, 0, 0), max_steps=4
        )
    )
    assert m["intervention_rate"] == 1.0 and m["intervention_size"] == 0.0


@pytest.mark.parametrize(
    ("mode", "layout", "widest"),
    [("thrust", "box", 2 * math.sqrt(3) * 0.02), ("accel", "ball", 2 * 0.02)],
)
def test_intervention_size_is_in_metres_per_second_squared_in_every_action_mode(
    mode, layout, widest
):
    # The same command inside the thrust limit in each mode's units: N at 500 kg, or m/s^2.
    u = np.array([0.3, 0.4, 0.2])
    scale = {"thrust": 0.02 * 500.0, "accel": 0.02}[mode]
    agent = FilteredAgent(ConstantAgent(u * scale), ScriptedFilter())
    ep = episode(
        agent,
        start_state=(0, -400, 30, 0, 0, 0),
        max_steps=5,
        action_mode=mode,
        thrust_limit=layout,
    )
    m = filter_metrics(ep)
    size = 0.5 * np.linalg.norm(u) * 0.02
    assert m["intervention_size"] == pytest.approx(size, rel=1e-12)
    assert m["intervention_size_normalized"] == pytest.approx(size / widest, rel=1e-12)


class SequenceFilter:
    """Passes commands through (or zeroes them) and reports a given field value per decision."""

    name = "sequence"

    def __init__(self, zero=False, **sequences):
        self.zero, self.sequences = zero, sequences

    def reset(self, view):
        self.k = 0

    def filter(self, obs, action):
        fields = {key: values[self.k % len(values)] for key, values in self.sequences.items()}
        self.k += 1
        out = np.zeros(3) if self.zero else action
        return FilterResult(out, bool(self.zero), **fields)


def test_unnecessary_interventions_are_those_whose_own_command_would_have_stayed_clear():
    # Full thrust toward the target for one step from rest 100 m behind coasts to 45 m from it,
    # inside the 50 m sphere, so zeroing it was needed; a tenth of it from 400 m was not.
    ep = episode(
        FilteredAgent(ConstantAgent([0.0, 1.0, 0.0]), SequenceFilter(zero=True)),
        start_state=(0.0, -100.0, 0.0, 0.0, 0.0, 0.0),
        max_steps=3,
    )
    n = ep.view.n
    times = np.linspace(0.0, 10.0 + 4 * np.pi / n, 40001)
    clear = [
        np.linalg.norm(
            escape_arc(n, x, np.array([0.0, 0.02, 0.0]), 10.0, np.zeros(3), times)[:, :3], axis=1
        ).min()
        >= 50.0
        for x in ep.states[:-1]
    ]
    m = filter_behaviour(ep)
    assert m["unnecessary_intervention_rate"] == pytest.approx(np.mean(clear))
    far = episode(
        FilteredAgent(ConstantAgent([0.0, 0.1, 0.0]), SequenceFilter(zero=True)),
        start_state=(0.0, -400.0, 0.0, 0.0, 0.0, 0.0),
        max_steps=3,
    )
    assert filter_behaviour(far)["unnecessary_intervention_rate"] == 1.0
    assert m["unnecessary_intervention_rate"] == 0.0


def test_reentry_time_is_the_mean_length_of_the_recovering_stretches():
    flags = (False, True, True, True, False, False, True, False)
    agent = FilteredAgent(PDAgent(), SequenceFilter(recovering=flags))
    m = filter_behaviour(episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=8))
    # Stretches of 3 and 1 decisions, 10 s each.
    assert m["reentry_time"] == pytest.approx(20.0)
    # Recovering until the episode ends is never getting back.
    unfinished = FilteredAgent(PDAgent(), SequenceFilter(recovering=(False, True, True)))
    m = filter_behaviour(episode(unfinished, start_state=(0, -400, 0, 0, 0, 0), max_steps=3))
    assert m["reentry_time"] == np.inf
    assert np.isnan(
        filter_behaviour(episode(FilteredAgent(PDAgent(), SequenceFilter()), max_steps=3))[
            "reentry_time"
        ]
    )


def test_a_barrier_that_holds_the_chaser_still_counts_as_stall_time():
    # Every command zeroed with the safety row active, from rest: the chaser never moves.
    agent = FilteredAgent(PDAgent(), SequenceFilter(zero=True, active=(True,)))
    m = filter_behaviour(episode(agent, start_state=(0.0, -300.0, 0.0, 0, 0, 0), max_steps=6))
    assert m["stall_time"] == pytest.approx(60.0)
    moving = filter_behaviour(
        episode(FilteredAgent(PDAgent(), SequenceFilter(active=(True,))), max_steps=6)
    )
    assert moving["stall_time"] == 0.0


def test_stall_time_counts_only_decisions_slower_than_the_stall_speed():
    # 0.03 of a full command from rest gains 0.006 m/s per 10 s step, so the first two of six
    # decisions are slower than 0.01 m/s and all six are slower than 0.1 m/s.
    agent = FilteredAgent(ConstantAgent([0.0, 0.03, 0.0]), SequenceFilter(active=(True,)))
    ep = episode(agent, start_state=(0.0, -300.0, 0.0, 0, 0, 0), max_steps=6)
    assert filter_behaviour(ep)["stall_time"] == pytest.approx(20.0)
    assert filter_behaviour(ep, MetricSettings(stall_speed=0.1))["stall_time"] == pytest.approx(
        60.0
    )


def test_stall_time_ends_at_a_contact_that_cuts_the_last_step():
    # 5 cm out drifting in at 8 mm/s, inside a soft window: contact at 6.25 s cuts the second
    # 5 s step, so the stalled time is the contact time, not two whole steps.
    spec = SCENARIOS["docking"]
    spec = replace(
        spec,
        parameters={**spec.parameters, "axial_window": (0.0, 0.1)},
        dispersions={"start_pos": Fixed((0.0, -2.05, 0.0)), "start_vel": Fixed((0.0, 0.008, 0.0))},
    )
    env = entry_class(spec)(spec)
    agent = FilteredAgent(ConstantAgent([0.0, 0.0, 0.0]), SequenceFilter(zero=True, active=(True,)))
    ep = record_episode(env, agent, seed=1, agent_seed=2)
    assert env._info()["success"] and len(ep.records) == 2
    assert ep.duration == pytest.approx(0.05 / 0.008, rel=1e-3)
    assert filter_behaviour(ep)["stall_time"] == pytest.approx(ep.duration, rel=1e-12)


def test_a_planned_hold_without_an_active_row_is_not_stall_time():
    agent = FilteredAgent(ConstantAgent([0.0, 0.0, 0.0]), SequenceFilter())
    ep = episode(agent, start_state=(0.0, -300.0, 0.0, 0, 0, 0), max_steps=6)
    assert filter_behaviour(ep)["stall_time"] == 0.0


def test_unnecessary_interventions_do_not_depend_on_drift_sampling_or_mirroring():
    start = np.array([5.0, -140.0, 8.0, 0.0, 0.3, 0.01])
    mirrored = start * [1, 1, -1, 1, 1, -1]

    def rate(state, samples):
        agent = FilteredAgent(ConstantAgent([0.0, 1.0, 0.2]), SequenceFilter(zero=True))
        ep = episode(agent, start_state=tuple(state), max_steps=12)
        settings = MetricSettings(drift_samples_per_orbit=samples)
        return filter_behaviour(ep, settings)["unnecessary_intervention_rate"]

    # Coasting in at 0.3 m/s, the held full-thrust command clears from two of the twelve decisions.
    reference = rate(start, 200)
    assert reference == pytest.approx(2 / 12)
    assert rate(start, 50) == rate(start, 800) == reference
    agent = FilteredAgent(ConstantAgent([0.0, 1.0, -0.2]), SequenceFilter(zero=True))
    ep = episode(agent, start_state=tuple(mirrored), max_steps=12)
    assert filter_behaviour(ep)["unnecessary_intervention_rate"] == reference


def test_fallbacks_and_certificates_are_reported_as_given():
    agent = FilteredAgent(
        PDAgent(),
        SequenceFilter(
            fallback_level=(0, 1, 0, 2),
            certificate=("implicit",),
            certificate_horizon=(600.0, 450.0),
        ),
    )
    m = filter_behaviour(episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=4))
    assert m["fallback_count"] == 2.0 and m["fallback_level_max"] == 2.0
    assert m["filter_certificates"] == ("implicit",) and m["certificate_horizon_min"] == 450.0
    silent = filter_behaviour(episode(FilteredAgent(PDAgent(), SequenceFilter()), max_steps=3))
    assert np.isnan(silent["fallback_count"]) and np.isnan(silent["fallback_level_max"])


def test_conservatism_is_the_share_of_viable_states_the_filter_leaves_out():
    states = np.zeros((201, 6))
    states[:, 0] = np.linspace(0.0, 2.0, 201)
    found = filter_conservatism(lambda x: x[0] <= 0.6, lambda x: x[0] <= 1.0, states)
    # 101 viable samples from 0 to 1 m, of which 61 are inside the filter's set.
    assert found == pytest.approx(40 / 101)
    assert np.isnan(filter_conservatism(lambda x: True, lambda x: False, states))


class ClockedFilter:
    """Sleeps `sleep_s` per decision and reports `claimed_s` as its compute time."""

    name = "clocked"

    def __init__(self, sleep_s, claimed_s):
        self.sleep_s, self.claimed_s = sleep_s, claimed_s

    def reset(self, view):
        pass

    def filter(self, obs, action):
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return FilterResult(action, False, compute_s=self.claimed_s)


def test_run_filter_reports_the_measured_compute_time_not_the_filters_claim():
    obs, command = np.zeros(6), np.zeros(3)
    slow, _ = run_filter(ClockedFilter(0.01, 0.0), obs, command)
    assert slow.compute_s >= 0.01
    boastful, elapsed = run_filter(ClockedFilter(0.0, 100.0), obs, command)
    assert boastful.compute_s == pytest.approx(elapsed) and boastful.compute_s < 1.0


def test_compute_time_metric_reads_the_measured_time():
    slow = episode(FilteredAgent(PDAgent(), ClockedFilter(0.01, 0.0)), max_steps=5)
    assert filter_metrics(slow)["filter_compute_median"] >= 0.01
    boastful = episode(FilteredAgent(PDAgent(), ClockedFilter(0.0, 100.0)), max_steps=5)
    assert filter_metrics(boastful)["filter_compute_median"] < 1.0


def record_pair(*args, **kwargs):
    # Imported here so the rest of this file runs while pair recording is being built.
    from vbar.metrics.episode import record_pair as recorded

    return recorded(*args, **kwargs)


PAIR_SEED = 0x9E3779B97F4A7C15F39CC0605CEDC834
PAIR_AGENT_SEED = 0x6A09E667F3BCC908B2FB1366EA957D3E


class HalvingFilter:
    """Halves every command and reports each decision as an active intervention."""

    name = "halving"

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(0.5 * np.asarray(action, dtype=float), True, active=True)


def speeding_pair(soft, seed=PAIR_SEED, steps=4):
    """A coast at 0.2 m/s 300 m behind the target, over a 0.01 m/s speed limit, with a thrust
    the filter halves; the keep-out sphere stays far away."""
    env = GuardedApproach(
        ApproachConfig(start_state=(0.0, -300.0, 0.0, 0.0, 0.2, 0.0), max_steps=steps),
        extra=(SpeedLimit(0.01, 0.0),),
        soft=("speed_limit",) if soft else (),
    )
    return record_pair(
        env, lambda: ConstantAgent([0.0, 0.1, 0.0]), HalvingFilter(), seed=seed,
        agent_seed=seed + 1,
    )  # fmt: skip


def test_paired_unnecessary_interventions_ignore_a_soft_violation():
    # The unfiltered run breaks only the soft speed limit, so every intervention was unnecessary.
    pair = speeding_pair(soft=True)
    m = paired_filter_metrics([pair.on], [pair.off])
    assert m["unnecessary_intervention_rate_paired"] == 1.0


def test_paired_unnecessary_interventions_are_none_when_the_unfiltered_run_breaks_a_hard_one():
    pair = speeding_pair(soft=False)
    m = paired_filter_metrics([pair.on], [pair.off])
    assert m["unnecessary_intervention_rate_paired"] == 0.0


def test_paired_unnecessary_interventions_pool_the_active_decisions_of_every_pair():
    # 4 active decisions in a clean pair and 6 in a pair whose unfiltered run breaks a hard
    # constraint: a lower bound of 4 / 10, the second pair's decisions all counted as needed.
    clean = speeding_pair(soft=True, seed=PAIR_SEED, steps=4)
    broken = speeding_pair(soft=False, seed=PAIR_SEED + 2, steps=6)
    m = paired_filter_metrics([clean.on, broken.on], [clean.off, broken.off])
    assert m["unnecessary_intervention_rate_paired"] == pytest.approx(0.4)


def test_paired_metrics_refuse_a_pair_with_different_agent_seeds():
    first = speeding_pair(soft=True)
    env = GuardedApproach(
        ApproachConfig(start_state=(0.0, -300.0, 0.0, 0.0, 0.2, 0.0), max_steps=4),
        extra=(SpeedLimit(0.01, 0.0),),
        soft=("speed_limit",),
    )
    other = record_pair(
        env, lambda: ConstantAgent([0.0, 0.1, 0.0]), HalvingFilter(), seed=PAIR_SEED,
        agent_seed=PAIR_SEED + 7,
    )  # fmt: skip
    with pytest.raises(ValueError, match="agent"):
        paired_filter_metrics([first.on], [other.off])


def test_paired_metrics_refuse_a_pair_from_different_scenarios():
    first = speeding_pair(soft=True, steps=4)
    longer = speeding_pair(soft=True, steps=5)
    with pytest.raises(ValueError, match="scenario|view"):
        paired_filter_metrics([first.on], [longer.off])


class ReplacingFilter:
    """Replaces every command with `command` and reports an active intervention."""

    name = "replacing"

    def __init__(self, command):
        self.command = np.asarray(command, dtype=float)

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(self.command.copy(), True, active=True)


def limited_pair(safety_filter, steps):
    """A coast at 0.2 m/s 300 m behind the target under a hard 0.25 m/s speed limit, the agent
    adding 0.02 m/s per step along the motion; the keep-out sphere stays far away."""
    env = GuardedApproach(
        ApproachConfig(start_state=(0.0, -300.0, 0.0, 0.0, 0.2, 0.0), max_steps=steps),
        extra=(SpeedLimit(0.25, 0.0),),
    )
    return record_pair(
        env, lambda: ConstantAgent([0.0, 0.1, 0.0]), safety_filter, seed=PAIR_SEED,
        agent_seed=PAIR_SEED + 1,
    )  # fmt: skip


def test_paired_rate_is_zero_when_only_the_unfiltered_run_breaks_a_hard_constraint():
    # Unfiltered: 0.2 + 4 x 0.02 = 0.28 m/s, over the limit. Filtered at half the command:
    # 0.24 m/s, inside it. The interventions were needed.
    pair = limited_pair(HalvingFilter(), steps=4)
    assert min(r.segment.minima["speed_limit"].lower_bound for r in pair.on.records) > 0.0
    m = paired_filter_metrics([pair.on], [pair.off])
    assert m["unnecessary_intervention_rate_paired"] == 0.0


def test_paired_rate_is_one_when_only_the_filtered_run_breaks_a_hard_constraint():
    # Unfiltered: 0.24 m/s after 2 steps, inside the limit. The filter's full burn reaches
    # 0.6 m/s; its interventions were unnecessary, whatever they did to its own run.
    pair = limited_pair(ReplacingFilter([0.0, 1.0, 0.0]), steps=2)
    assert min(r.segment.minima["speed_limit"].margin for r in pair.on.records) < 0.0
    m = paired_filter_metrics([pair.on], [pair.off])
    assert m["unnecessary_intervention_rate_paired"] == 1.0


def with_lower_bound(ep, name, bound):
    """A copy of the episode whose every step reports `bound` as the certified bound of `name`."""
    records = []
    for rec in ep.records:
        minima = dict(rec.segment.minima)
        found = minima[name]
        minima[name] = SegmentMinimum(found.t, found.margin, bound)
        records.append(
            dataclasses.replace(rec, segment=dataclasses.replace(rec.segment, minima=minima))
        )
    return dataclasses.replace(ep, records=tuple(records))


def test_an_unfiltered_run_counts_as_clean_only_on_its_certified_bound():
    # The refined minima stay inside the limit, but a certified bound 1 m/s below it could hide
    # a dip, so the unfiltered run is not shown clean and the interventions are not counted
    # as unnecessary: the paired rate stays a lower bound.
    pair = limited_pair(ReplacingFilter([0.0, 1.0, 0.0]), steps=2)
    unproven = with_lower_bound(pair.off, "speed_limit", -1.0)
    assert min(r.segment.minima["speed_limit"].margin for r in unproven.records) > 0.0
    m = paired_filter_metrics([pair.on], [unproven])
    assert m["unnecessary_intervention_rate_paired"] == 0.0
    # Just at minus the resolution the run still counts as clean.
    edge = with_lower_bound(pair.off, "speed_limit", -SpeedLimit.resolution)
    m = paired_filter_metrics([pair.on], [edge])
    assert m["unnecessary_intervention_rate_paired"] == 1.0
