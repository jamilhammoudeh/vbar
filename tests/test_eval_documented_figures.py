"""Figures quoted in docs/evaluation.md and docs/validation.md, asserted where the docs state them.

The coverage table comes from `benchmarks/coverage_table.py`; these tests hold the claims that
rest on it, with the table's seeds, and the safety-tail figures that rest on closed forms.
"""

import inspect
import re

import numpy as np
import pytest
from scipy.stats import binom

from vbar.eval.coverage import (
    iqm_coverage,
    mcnemar_false_positives,
    paired_coverage,
    rate_coverage,
)
from vbar.eval.stats import (
    paired_difference,
    probability_of_improvement,
    stratified_bootstrap,
    summary_interval,
)
from vbar.eval.tails import compare_rates, conformal_coverage_risk, tail_summary, tolerance_order
from vbar.metrics import clopper_pearson

RATES = (0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999)


def test_the_exact_rate_interval_covers_at_least_95_percent_at_every_count_from_1_to_200():
    # The coverage at a true rate p is the probability, under Binomial(n, p), of the counts whose
    # interval holds p: a sum, no simulation. Its lowest value on these rates is 0.9501 (n = 190,
    # p = 0.5), so the 1e-9 slack only absorbs rounding.
    for n in range(1, 201):
        counts = np.arange(n + 1)
        bounds = np.array([clopper_pearson(k, n) for k in range(n + 1)])
        for p in RATES:
            holds = (bounds[:, 0] <= p) & (p <= bounds[:, 1])
            coverage = binom.pmf(counts, n, p)[holds].sum()
            assert coverage >= 0.95 - 1e-9, (n, p, coverage)


@pytest.mark.parametrize("n", [5, 10, 20, 50, 100])
def test_mcnemar_calls_equal_controllers_different_at_most_5_percent_of_the_time(n):
    # The table's cells, seed 5: 0.000, 0.000, 0.000, 0.014 and 0.034.
    assert mcnemar_false_positives(0.8, n, 1000, rng=5).estimate <= 0.05


@pytest.mark.slow
def test_the_iqm_and_paired_intervals_cover_the_truth_at_20_runs():
    # The table's cells are 0.939 and 0.943 against the threshold of 0.929.
    assert iqm_coverage(20, 1000, 1000, rng=3).holds
    assert paired_coverage(20, 1000, 1000, rng=4).holds


@pytest.mark.slow
def test_the_iqm_and_paired_intervals_fall_short_at_5_runs():
    # The table's cells are 0.889 and 0.886.
    assert not iqm_coverage(5, 1000, 1000, rng=3).holds
    assert not paired_coverage(5, 1000, 1000, rng=4).holds


def test_the_documented_defaults_are_the_defaults():
    for function in (
        stratified_bootstrap,
        probability_of_improvement,
        paired_difference,
        summary_interval,
    ):
        parameters = inspect.signature(function).parameters
        assert parameters["reps"].default == 10_000, function.__name__
        assert parameters["confidence"].default == 0.95, function.__name__
    for function in (rate_coverage, iqm_coverage, paired_coverage, mcnemar_false_positives):
        assert inspect.signature(function).parameters["repeats"].default == 1000, function.__name__
    for function in (iqm_coverage, paired_coverage):
        assert inspect.signature(function).parameters["reps"].default == 1000, function.__name__
    for function in (rate_coverage, iqm_coverage, paired_coverage):
        assert inspect.signature(function).parameters["confidence"].default == 0.95
    assert inspect.signature(paired_coverage).parameters["shared"].default == 0.9
    parameters = inspect.signature(mcnemar_false_positives).parameters
    assert (parameters["alpha"].default, parameters["shared"].default) == (0.05, 0.5)
    parameters = inspect.signature(tail_summary).parameters
    assert (parameters["level"].default, parameters["confidence"].default) == (0.05, 0.95)


@pytest.mark.parametrize("n, rank, confidence", [(100, 2, 0.963), (1000, 39, 0.957)])
def test_the_tolerance_bound_of_100_and_1000_runs_has_the_quoted_confidence(n, rank, confidence):
    assert tolerance_order(n, 0.95, 0.95) == rank
    assert round(float(binom.sf(rank - 1, n, 0.05)), 3) == confidence
    # The next rank falls short of 95%, so this is the largest one that holds.
    assert binom.sf(rank, n, 0.05) < 0.95


@pytest.mark.parametrize("n, risk", [(100, 0.436), (1000, 0.480)])
def test_the_conformal_risk_of_100_and_1000_runs_rounds_to_the_quoted_figure(n, risk):
    assert round(conformal_coverage_risk(n, 0.95), 3) == risk


def test_twenty_runs_cannot_bound_the_5th_percentile_from_below():
    lower, upper = tail_summary(np.arange(20.0), "low", reps=100, rng=0)["quantile_interval"]
    assert lower == -np.inf and np.isfinite(upper)


def test_70_of_100_against_50_of_100_differ_by_0_20_inside_the_0_203_width_and_are_refused():
    found = compare_rates(70, 100, 50, 100)
    lower, upper = found["interval_b"]
    assert round(upper - lower, 4) == 0.2034
    assert found["difference"] == pytest.approx(0.20, abs=1e-12)
    assert found["resolved"] is False
    message = "differ by 0.2, smaller than the interval width 0.203"
    with pytest.raises(ValueError, match=re.escape(message)):
        compare_rates(70, 100, 50, 100, require_resolved=True)


def test_3_failures_in_10000_runs_give_the_quoted_normal_and_clopper_pearson_figures():
    runs = np.r_[np.ones(3), np.zeros(9997)]
    normal_lower = runs.mean() - 1.96 * runs.std(ddof=1) / np.sqrt(len(runs))
    assert f"{normal_lower:.1e}" == "-3.9e-05"
    lower, upper = clopper_pearson(3, 10_000)
    assert (f"{lower:.2e}", f"{upper:.2e}") == ("6.19e-05", "8.76e-04")
