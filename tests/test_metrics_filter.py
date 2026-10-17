import math

import numpy as np
import pytest
from metric_cases import ConstantAgent, episode, pd_episode
from scipy.stats import trim_mean

from vbar.agents import PDAgent
from vbar.metrics import (
    MetricSettings,
    escape_arc,
    filter_behaviour,
    filter_conservatism,
    filter_metrics,
    iqm,
    paired_filter_metrics,
)
from vbar.metrics.efficiency import delta_v
from vbar.safety import FilteredAgent, FilterResult, PassThroughFilter


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


def test_compute_time_median_and_95th_percentile():
    compute = (0.001, 0.002, 0.003, 0.004, 0.100)
    agent = FilteredAgent(PDAgent(), ScriptedFilter(compute=compute))
    m = filter_metrics(episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=5))
    assert m["filter_compute_median"] == pytest.approx(0.003)
    assert m["filter_compute_p95"] == pytest.approx(np.percentile(compute, 95))


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


def test_clipping_is_counted_apart_from_safety_activity():
    agent = FilteredAgent(PDAgent(), ScriptedFilter(clipped=True))
    m = filter_metrics(episode(agent, start_state=(0, -400, 0, 0, 0, 0), max_steps=5))
    assert m["intervention_rate"] == 0.0 and m["clip_rate"] == 1.0
    assert np.isnan(m["intervention_size"])


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
    agent = FilteredAgent(PDAgent(), ScriptedFilter())
    ep = episode(
        agent,
        start_state=(0, -400, 30, 0, 0, 0),
        max_steps=5,
        action_mode=mode,
        thrust_limit=layout,
    )
    m = filter_metrics(ep)
    to_accel = 0.02 / {"thrust": 0.02 * 500.0, "accel": 0.02}[mode]
    sizes = [np.linalg.norm(0.5 * rec.command) * to_accel for rec in ep.records]
    assert m["intervention_size"] == pytest.approx(np.mean(sizes), rel=1e-12)
    assert m["intervention_size_normalized"] == pytest.approx(np.mean(sizes) / widest, rel=1e-12)


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


def test_conservatism_is_the_share_of_viable_states_the_filter_leaves_out():
    states = np.zeros((201, 6))
    states[:, 0] = np.linspace(0.0, 2.0, 201)
    found = filter_conservatism(lambda x: x[0] <= 0.6, lambda x: x[0] <= 1.0, states)
    # 101 viable samples from 0 to 1 m, of which 61 are inside the filter's set.
    assert found == pytest.approx(40 / 101)
    assert np.isnan(filter_conservatism(lambda x: True, lambda x: False, states))
