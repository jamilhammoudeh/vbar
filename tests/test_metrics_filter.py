import math

import numpy as np
import pytest
from metric_cases import episode, pd_episode
from scipy.stats import trim_mean

from vbar.agents import PDAgent
from vbar.metrics import MetricSettings, filter_metrics, iqm, paired_filter_metrics
from vbar.metrics.efficiency import delta_v
from vbar.safety import FilteredAgent, FilterResult, PassThroughFilter


class ScriptedFilter:
    """Halves every command and reports the statuses and compute times it is given, in turn."""

    name = "scripted"

    def __init__(self, statuses=("feasible",), compute=(0.001,)):
        self.statuses, self.compute = statuses, compute

    def reset(self, view):
        self.k = 0

    def filter(self, obs, action):
        status = self.statuses[self.k % len(self.statuses)]
        spent = self.compute[self.k % len(self.compute)]
        self.k += 1
        return FilterResult(0.5 * action, True, status, compute_s=spent)


def test_halving_every_command_gives_rate_one_and_the_mean_half_command_size():
    agent = FilteredAgent(PDAgent(), ScriptedFilter())
    ep = episode(agent, start_state=(0.0, -400.0, 30.0, 0.0, 0.0, 0.0), max_steps=30)
    m = filter_metrics(ep)
    sizes = [np.linalg.norm(0.5 * rec.command) / math.sqrt(3.0) for rec in ep.records]
    assert m["intervention_rate"] == 1.0
    assert m["intervention_size"] == pytest.approx(np.mean(sizes), rel=1e-12)


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
