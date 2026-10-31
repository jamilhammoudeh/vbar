"""Scores across scenarios: interquartile mean, stratified bootstrap, profiles and comparisons.

Methods from Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice,
NeurIPS 2021.
"""

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import mannwhitneyu

from vbar.eval.stats import (
    Interval,
    aggregate_iqm,
    iqm,
    paired_difference,
    performance_profile,
    probability_of_improvement,
    stratified_bootstrap,
    summary_interval,
)

# At 500 resamples a 97.5% end (the 488th smallest) or a 2.5% end (the 13th) misses an event that
# a resample holds with probability q only when 488 resamples lack it, P(Binomial(500, 1 - q) >=
# 488); every q below is at least 0.26, so that chance is below 1e-47 in each test.
REPS = 500

scores = st.lists(
    st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False, allow_subnormal=False),
    min_size=1,
    max_size=30,
)


def pooled_mean(samples):
    return float(np.mean(np.concatenate([samples[k] for k in sorted(samples)])))


def test_an_interval_is_frozen_and_reports_its_width():
    interval = Interval(estimate=1.0, lower=0.5, upper=2.0)
    assert interval.width == 1.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        interval.lower = 0.0


def test_the_iqm_is_the_mean_of_the_middle_half():
    # Eight values: the lowest two and the highest two are trimmed, leaving 2, 3, 4, 5.
    assert iqm(np.arange(8.0)) == 3.5
    assert iqm([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1e12]) == 3.5
    assert iqm([4.0]) == 4.0


def test_the_iqm_ignores_infinite_values_in_the_trimmed_quarters():
    assert iqm([-np.inf, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf]) == 2.5


def test_the_iqm_of_no_values_is_refused():
    with pytest.raises(ValueError):
        iqm([])


def test_the_aggregate_iqm_pools_runs_across_scenarios():
    samples = {"a": np.arange(8.0), "b": np.array([100.0, 101.0, 102.0, 103.0])}
    # Twelve pooled runs, three trimmed at each end: 3, 4, 5, 6, 7 and 100. The mean of the two
    # per-scenario IQMs would be (3.5 + 101.5) / 2 = 52.5.
    assert aggregate_iqm(samples) == pytest.approx(125.0 / 6.0, rel=1e-12)


def test_the_aggregate_iqm_refuses_no_scenarios():
    with pytest.raises(ValueError):
        aggregate_iqm({})


def test_the_bootstrap_estimate_is_the_statistic_on_the_data():
    samples = {"a": np.arange(8.0), "b": np.array([100.0, 101.0, 102.0, 103.0])}
    found = stratified_bootstrap(samples, aggregate_iqm, reps=100, rng=0)
    assert isinstance(found, Interval)
    assert found.estimate == aggregate_iqm(samples)
    assert found.lower <= found.estimate <= found.upper


def test_the_bootstrap_resamples_with_replacement_within_each_scenario():
    samples = {"a": np.arange(10.0), "b": 100.0 + np.arange(5.0)}
    seen = []

    def record(resampled):
        seen.append({k: np.array(v) for k, v in resampled.items()})
        return 0.0

    stratified_bootstrap(samples, record, reps=200, rng=1)
    draws = [s for s in seen if set(s) == {"a", "b"}]
    assert len(draws) >= 200
    for s in draws:
        assert len(s["a"]) == 10 and len(s["b"]) == 5
        assert set(s["a"]) <= set(samples["a"]) and set(s["b"]) <= set(samples["b"])
    assert any(len(set(s["a"])) < 10 for s in draws)


def test_the_bootstrap_resamples_scenarios_independently():
    x = np.random.default_rng(2).standard_normal(100)

    def gap(s):
        return float(np.mean(s["a"]) - np.mean(s["b"]))

    found = stratified_bootstrap({"a": x, "b": x.copy()}, gap, reps=2000, rng=3)
    # Independent resamples of the same 100 values: sd of the gap sqrt(2 var / 100).
    expected = 2 * 1.96 * np.sqrt(2 * np.var(x) / 100)
    assert found.estimate == 0.0
    assert found.width > 0.7 * expected


def test_the_stratified_interval_excludes_the_spread_between_scenarios():
    rng = np.random.default_rng(4)
    samples = {"a": rng.standard_normal(200), "b": 10.0 + rng.standard_normal(200)}
    found = stratified_bootstrap(samples, pooled_mean, reps=4000, rng=5)
    # Within scenarios: sd of the pooled mean sqrt(var_a / 200 + var_b / 200) / 2, about 0.05,
    # so the 95% width is about 0.196. Resampling the pooled runs would add the spread between
    # the two scenario means and give about 1.0; a 90% or 99% interval would be 16% narrower or
    # 31% wider.
    sd = np.sqrt(np.var(samples["a"]) / 200 + np.var(samples["b"]) / 200) / 2
    assert found.width == pytest.approx(2 * 1.96 * sd, rel=0.07)


def test_the_bootstrap_is_reproducible_from_a_seed_or_a_generator():
    samples = {"a": np.arange(20.0), "b": np.arange(7.0) ** 2}
    first = stratified_bootstrap(samples, aggregate_iqm, reps=100, rng=11)
    assert stratified_bootstrap(samples, aggregate_iqm, reps=100, rng=11) == first
    from_generator = stratified_bootstrap(
        samples, aggregate_iqm, reps=100, rng=np.random.default_rng(11)
    )
    again = stratified_bootstrap(samples, aggregate_iqm, reps=100, rng=np.random.default_rng(11))
    assert from_generator == again


def test_the_order_of_scenarios_does_not_change_the_interval():
    a, b = np.arange(20.0), np.arange(7.0) ** 2
    forward = stratified_bootstrap({"a": a, "b": b}, aggregate_iqm, reps=100, rng=12)
    backward = stratified_bootstrap({"b": b, "a": a}, aggregate_iqm, reps=100, rng=12)
    assert forward == backward


def test_doubling_every_score_doubles_the_iqm_and_its_interval():
    rng = np.random.default_rng(13)
    samples = {"a": rng.standard_normal(30), "b": rng.exponential(size=12)}
    doubled = {k: 2.0 * v for k, v in samples.items()}
    one = stratified_bootstrap(samples, aggregate_iqm, reps=200, rng=14)
    two = stratified_bootstrap(doubled, aggregate_iqm, reps=200, rng=14)
    assert two.estimate == pytest.approx(2.0 * one.estimate, rel=1e-12)
    assert two.lower == pytest.approx(2.0 * one.lower, rel=1e-12)
    assert two.upper == pytest.approx(2.0 * one.upper, rel=1e-12)


def test_equal_scores_and_single_runs_give_an_interval_of_zero_width():
    found = stratified_bootstrap({"a": [3.0, 3.0, 3.0], "b": [5.0]}, pooled_mean, reps=100, rng=0)
    assert (found.estimate, found.lower, found.upper) == (3.5, 3.5, 3.5)
    assert found.width == 0.0


@pytest.mark.parametrize(
    "samples, kwargs",
    [
        ({}, {}),
        ({"a": [1.0, 2.0], "b": []}, {}),
        ({"a": [1.0, 2.0]}, {"reps": 99}),
        ({"a": [1.0, 2.0]}, {"confidence": 0.0}),
        ({"a": [1.0, 2.0]}, {"confidence": 1.0}),
        ({"a": [1.0, 2.0]}, {"confidence": 1.5}),
    ],
)
def test_the_bootstrap_refuses_empty_scenarios_few_reps_and_bad_confidence(samples, kwargs):
    with pytest.raises(ValueError):
        stratified_bootstrap(samples, pooled_mean, rng=0, **kwargs)


def test_one_hundred_reps_are_enough_for_the_bootstrap():
    found = stratified_bootstrap({"a": [1.0, 2.0, 4.0]}, pooled_mean, reps=100, rng=0)
    assert 1.0 <= found.lower <= found.upper <= 4.0


def test_the_performance_profile_weights_each_scenario_equally():
    samples = {"a": np.array([1.0, 2.0, 3.0, 4.0]), "b": np.array([10.0, 20.0])}
    found = performance_profile(samples, [0.0, 2.0, 5.0, 15.0, 25.0])
    # Scenario a: 1, 1/2, 0, 0, 0 above each threshold; b: 1, 1, 1, 1/2, 0. A score equal to the
    # threshold is not above it; pooling the six runs would give 4/6 at 2.
    np.testing.assert_allclose(found, [1.0, 0.75, 0.5, 0.25, 0.0], rtol=0, atol=1e-15)
    assert isinstance(found, np.ndarray) and found.shape == (5,)


@settings(max_examples=40, deadline=None)
@given(
    a=scores,
    b=scores,
    taus=st.lists(st.floats(-150.0, 150.0, allow_subnormal=False), min_size=1, max_size=10),
)
def test_the_performance_profile_falls_from_one_to_zero(a, b, taus):
    samples = {"a": np.array(a), "b": np.array(b)}
    found = performance_profile(samples, np.sort(taus))
    assert np.all((found >= 0.0) & (found <= 1.0))
    assert np.all(np.diff(found) <= 0.0)
    np.testing.assert_array_equal(performance_profile(samples, [-200.0, 200.0]), [1.0, 0.0])


def test_the_probability_of_improvement_counts_ties_as_half():
    # Pairs of (1, 2, 3) against (2, 2): 0, 0, 1/2, 1/2, 1, 1, so 3/6. Second scenario: every
    # run of x beats every run of y. Scenarios weigh equally: (0.5 + 1) / 2.
    x = {"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0]}
    y = {"a": [2.0, 2.0], "b": [1.0, 2.0]}
    found = probability_of_improvement(x, y, reps=200, rng=0)
    assert isinstance(found, Interval)
    assert found.estimate == 0.75
    assert 0.0 <= found.lower <= found.estimate <= found.upper <= 1.0


def test_the_probability_of_improvement_weights_each_scenario_equally():
    x = {"a": np.full(100, 2.0), "b": [0.0, 0.0]}
    y = {"a": np.full(100, 1.0), "b": [1.0, 1.0]}
    assert probability_of_improvement(x, y, reps=200, rng=0).estimate == 0.5


@settings(max_examples=20, deadline=None)
@given(a=scores, b=scores)
def test_the_probability_of_improvement_is_the_mann_whitney_fraction(a, b):
    found = probability_of_improvement({"s": a}, {"s": b}, reps=100, rng=0).estimate
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    assert found == pytest.approx(u / (len(a) * len(b)), rel=1e-12, abs=1e-12)
    reverse = probability_of_improvement({"s": b}, {"s": a}, reps=100, rng=0).estimate
    assert found + reverse == pytest.approx(1.0, rel=1e-12)


def test_a_controller_that_always_wins_has_an_interval_at_one():
    x = {"a": [5.0, 6.0, 7.0], "b": [3.0, 9.0]}
    y = {"a": [1.0, 2.0], "b": [0.0, 1.0, 2.0]}
    found = probability_of_improvement(x, y, reps=200, rng=0)
    assert (found.estimate, found.lower, found.upper) == (1.0, 1.0, 1.0)


def test_the_probability_of_improvement_interval_is_reproducible():
    rng = np.random.default_rng(21)
    x = {"a": rng.standard_normal(15), "b": rng.standard_normal(9)}
    y = {"a": rng.standard_normal(15), "b": rng.standard_normal(11)}
    first = probability_of_improvement(x, y, reps=300, rng=22)
    assert probability_of_improvement(x, y, reps=300, rng=22) == first
    assert first.width > 0.0


def test_the_probability_of_improvement_needs_the_same_scenarios():
    with pytest.raises(ValueError):
        probability_of_improvement({"a": [1.0], "b": [2.0]}, {"a": [1.0]}, reps=100, rng=0)
    with pytest.raises(ValueError):
        probability_of_improvement({"a": [1.0]}, {"a": []}, reps=100, rng=0)


def test_the_paired_difference_is_the_iqm_of_per_seed_differences():
    # Differences 0, 1, 2, -7: the middle half is 0 and 1. The difference of the two IQMs would be
    # 1.5 - 0 = 1.5.
    x = {"a": [0.0, 1.0, 2.0, 3.0]}
    y = {"a": [0.0, 0.0, 0.0, 10.0]}
    found = paired_difference(x, y, reps=200, rng=0)
    assert isinstance(found, Interval)
    assert found.estimate == 0.5


def test_common_random_numbers_cancel_in_the_paired_difference():
    rng = np.random.default_rng(31)
    noise = {"a": 10.0 * rng.standard_normal(40), "b": 10.0 * rng.standard_normal(25)}
    x = {k: v + 0.1 for k, v in noise.items()}
    found = paired_difference(x, noise, reps=200, rng=32)
    # Each seed's noise is shared, so every difference is 0.1 and every resample agrees; resampling
    # x and y separately would give an interval about 5 wide.
    assert found.estimate == pytest.approx(0.1, rel=1e-9)
    assert found.width < 1e-9


def test_the_paired_difference_pools_pairs_across_scenarios():
    x = {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 10.0]}
    y = {"a": [0.0, 0.0, 0.0, 0.0], "b": [0.0, 0.0]}
    # Six differences 1, 2, 3, 4, 10, 10; one trimmed at each end leaves 2, 3, 4, 10.
    assert paired_difference(x, y, reps=200, rng=0).estimate == 4.75


def test_the_paired_difference_interval_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(33)
    x = {"a": rng.standard_normal(30), "b": rng.standard_normal(20)}
    y = {"a": rng.standard_normal(30), "b": rng.standard_normal(20)}
    first = paired_difference(x, y, reps=150, rng=34)
    assert paired_difference(x, y, reps=150, rng=34) == first
    assert first.lower < first.estimate < first.upper


@pytest.mark.parametrize(
    "x, y",
    [
        ({"a": [1.0, 2.0]}, {"a": [1.0]}),
        ({"a": [1.0], "b": [2.0]}, {"a": [1.0]}),
        ({"a": []}, {"a": []}),
    ],
)
def test_the_paired_difference_refuses_unpaired_runs(x, y):
    with pytest.raises(ValueError):
        paired_difference(x, y, reps=100, rng=0)


CLEAN = {"approach-vbar": [1.0, 2.0, 3.0, 4.0], "inspection": [2.0, 3.0, 5.0]}
NAN_RUN = {"approach-vbar": [1.0, 2.0, 3.0, 4.0], "inspection": [2.0, np.nan, 5.0]}
INF_RUN = {"approach-vbar": [1.0, 2.0, 3.0, 4.0], "inspection": [2.0, np.inf, 5.0]}
CHEAP = {"reps": 100, "rng": 0}
BOOTSTRAPS = [
    (stratified_bootstrap, {"samples": CLEAN, "statistic": aggregate_iqm}),
    (probability_of_improvement, {"x": CLEAN, "y": CLEAN}),
    (paired_difference, {"x": CLEAN, "y": CLEAN}),
    (summary_interval, {"values": [1.0, 2.0, 3.0]}),
]
BAD_SETTINGS = [
    ("reps-99", {"reps": 99}),
    ("reps-a-float", {"reps": 100.0}),
    ("reps-true", {"reps": True}),
    ("reps-nan", {"reps": np.nan}),
    ("confidence-0", {"confidence": 0.0}),
    ("confidence-1", {"confidence": 1.0}),
    ("confidence-nan", {"confidence": np.nan}),
    ("confidence-1.5", {"confidence": 1.5}),
]


def hostile(function, label, match=None, **arguments):
    return pytest.param(function, arguments, match, id=f"{function.__name__}-{label}")


def nan_when_a_run_repeats(resampled):
    # The data has no repeated run, so only resamples give NaN.
    runs = resampled["a"].tolist()
    return math.nan if len(set(runs)) < len(runs) else 0.0


@pytest.mark.parametrize(
    "function, arguments, match",
    [
        hostile(iqm, "nan-among-finite", values=[1.0, np.nan, 3.0]),
        hostile(iqm, "only-nan", values=[np.nan]),
        hostile(aggregate_iqm, "nan-score", "inspection", samples=NAN_RUN),
        hostile(
            stratified_bootstrap,
            "nan-score",
            "inspection",
            samples=NAN_RUN,
            statistic=aggregate_iqm,
            **CHEAP,
        ),
        hostile(
            stratified_bootstrap,
            "nan-score-with-another-statistic",
            "inspection",
            samples=NAN_RUN,
            statistic=pooled_mean,
            **CHEAP,
        ),
        hostile(
            stratified_bootstrap,
            "statistic-nan-on-a-resample",
            samples={"a": np.arange(8.0)},
            statistic=nan_when_a_run_repeats,
            **CHEAP,
        ),
        hostile(
            stratified_bootstrap,
            "statistic-nan-on-the-data",
            "on the data",
            samples={"a": np.arange(8.0)},
            statistic=lambda resampled: math.nan,
            **CHEAP,
        ),
        hostile(performance_profile, "nan-score", "inspection", samples=NAN_RUN, taus=[0.0, 2.0]),
        hostile(performance_profile, "nan-threshold", samples=CLEAN, taus=[0.0, np.nan]),
        hostile(probability_of_improvement, "nan-in-x", "inspection", x=NAN_RUN, y=CLEAN, **CHEAP),
        hostile(probability_of_improvement, "nan-in-y", "inspection", x=CLEAN, y=NAN_RUN, **CHEAP),
        hostile(paired_difference, "nan-in-x", "inspection", x=NAN_RUN, y=CLEAN, **CHEAP),
        hostile(paired_difference, "nan-in-y", "inspection", x=CLEAN, y=NAN_RUN, **CHEAP),
        hostile(paired_difference, "inf-minus-inf", "inspection", x=INF_RUN, y=INF_RUN, **CHEAP),
        hostile(summary_interval, "nan-score", values=[1.0, np.nan, 3.0], **CHEAP),
        *[
            hostile(function, label, **{**CHEAP, **base, **setting})
            for function, base in BOOTSTRAPS
            for label, setting in BAD_SETTINGS
        ],
    ],
)
def test_hostile_statistics_inputs_are_refused(function, arguments, match):
    with pytest.raises(ValueError, match=match):
        function(**arguments)


def test_infinite_scores_are_accepted_by_the_iqm_and_the_bootstrap():
    assert iqm([np.inf, np.inf, np.inf, 1.0]) == np.inf
    assert iqm([-np.inf, -np.inf, -np.inf, 1.0]) == -np.inf
    runs = {"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.inf]}
    found = stratified_bootstrap(runs, aggregate_iqm, **CHEAP)
    assert found.estimate == 3.5
    assert not math.isnan(found.lower) and not math.isnan(found.upper)
    largest = stratified_bootstrap({"a": [0.0, 1.0, np.inf]}, lambda s: np.max(s["a"]), **CHEAP)
    assert largest.estimate == np.inf and largest.upper == np.inf
    assert not math.isnan(largest.lower)


def test_infinite_thresholds_and_scores_are_accepted_by_the_profile():
    np.testing.assert_array_equal(performance_profile(CLEAN, [-np.inf, np.inf]), [1.0, 0.0])
    np.testing.assert_allclose(performance_profile(INF_RUN, [10.0]), [1.0 / 6.0], rtol=1e-12)


def test_the_bootstrap_can_draw_the_last_run_of_every_scenario():
    # A resample of five draws holds the last of five runs with probability 1 - 0.8**5 = 0.672, and
    # both scenarios' last runs together with 0.452, so the largest sum is reached in 45% of the
    # resamples and the 97.5% end equals it.
    samples = {"a": np.arange(5.0), "b": 10.0 + np.arange(5.0)}
    found = stratified_bootstrap(samples, lambda s: s["a"].max() + s["b"].max(), reps=REPS, rng=0)
    assert (found.estimate, found.upper) == (18.0, 18.0)


def test_the_bootstrap_can_draw_the_first_run_of_every_scenario():
    samples = {"a": np.arange(5.0), "b": 10.0 + np.arange(5.0)}
    found = stratified_bootstrap(samples, lambda s: s["a"].min() + s["b"].min(), reps=REPS, rng=0)
    assert (found.estimate, found.lower) == (10.0, 10.0)


def test_the_pooled_iqm_bootstrap_can_draw_the_last_run():
    # Of [0, 0, 0, 1] the IQM is the mean of the middle two, 0; a resample reaches 0.5 or 1 when the
    # last run is drawn twice or more, 26% of them, so the 97.5% end is above 0 and only a bootstrap
    # that can draw the last run gets there.
    runs = [0.0, 0.0, 0.0, 1.0]
    pooled = summary_interval(runs, reps=REPS, rng=0)
    generic = stratified_bootstrap({"runs": runs}, lambda s: aggregate_iqm(s), reps=REPS, rng=0)
    assert pooled.estimate == 0.0 and generic.estimate == 0.0
    assert pooled.upper > 0.0 and generic.upper > 0.0


def test_the_pooled_iqm_bootstrap_equals_the_statistic_applied_to_each_resample():
    # A lambda is not `aggregate_iqm`, so it takes the path that calls the statistic per resample.
    gen = np.random.default_rng(1)
    samples = {"a": gen.standard_normal(12), "b": gen.exponential(size=7), "c": gen.normal(size=5)}
    fast = stratified_bootstrap(samples, aggregate_iqm, reps=300, rng=2)
    slow = stratified_bootstrap(samples, lambda s: aggregate_iqm(s), reps=300, rng=2)
    assert fast.estimate == slow.estimate
    assert fast.lower == pytest.approx(slow.lower, rel=1e-12)
    assert fast.upper == pytest.approx(slow.upper, rel=1e-12)


def test_the_probability_of_improvement_bootstrap_can_draw_the_last_run_of_x():
    # Only the last run of x beats any run of y. A resample of x holds it with probability 0.672,
    # so the 97.5% end of P(X > Y) is above 0; with x never holding it, every resample gives 0.
    x = {"s": [0.0, 0.0, 0.0, 0.0, 1.0]}
    y = {"s": [0.5] * 5}
    found = probability_of_improvement(x, y, reps=REPS, rng=0)
    assert found.estimate == 0.2
    assert found.upper > 0.0


def test_the_probability_of_improvement_bootstrap_can_draw_the_last_run_of_y():
    # Only the last run of y beats any run of x, so P(X > Y) falls below 1 in a resample of y only
    # when it holds that run.
    x = {"s": [0.5] * 5}
    y = {"s": [0.0, 0.0, 0.0, 0.0, 1.0]}
    found = probability_of_improvement(x, y, reps=REPS, rng=0)
    assert found.estimate == 0.8
    assert found.lower < 1.0
