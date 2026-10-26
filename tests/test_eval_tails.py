"""Safety tails with intervals, tolerance bounds and success-rate comparisons.

Order-statistic intervals and tolerance bounds are distribution free: their confidence is a
binomial sum that depends only on n (Kochenderfer, Katz, Corso and Moss, Algorithms for
Validation, 2026, sections 7.1 and 12.2).
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import binom, norm

from vbar.eval.tails import compare_rates, conformal_coverage_risk, tail_summary, tolerance_order
from vbar.metrics import unsafe_tail


def bounds(interval):
    if hasattr(interval, "lower"):
        return interval.lower, interval.upper
    lower, upper = interval
    return lower, upper


def no_bound(value):
    return value is None or np.isnan(value)


def ranks(n, worse, **kwargs):
    """1-based ranks, from the low end, of the order statistics the quantile interval picks."""
    x = np.random.default_rng(n).permutation(np.arange(1.0, n + 1.0))
    lower, upper = bounds(tail_summary(x, worse, reps=100, rng=0, **kwargs)["quantile_interval"])
    assert lower == int(lower) and upper == int(upper)
    return int(lower), int(upper)


@pytest.mark.parametrize("worse", ["low", "high"])
def test_the_point_values_are_the_unsafe_tail_of_the_metrics(worse):
    x = np.concatenate([np.random.default_rng(1).standard_normal(120), [np.nan, np.nan]])
    found = tail_summary(x, worse, reps=200, rng=0)
    tail = unsafe_tail(x, worse)
    for key in ("worst", "quantile", "tail_mean", "n", "missing"):
        assert found[key] == tail[key]
    assert (found["n"], found["missing"]) == (120, 2)
    clean = tail_summary(x[:120], worse, reps=200, rng=0)
    for key in ("quantile_interval", "tail_mean_interval"):
        assert bounds(found[key]) == bounds(clean[key])
    assert found["worst_bound"] == clean["worst_bound"]


def test_the_order_statistic_interval_covers_the_true_quantile():
    n = 200
    lo, hi = ranks(n, "low")
    # Distribution free: the interval holds the 5th percentile when at least lo and fewer than hi
    # runs fall below it.
    coverage = binom.cdf(hi - 1, n, 0.05) - binom.cdf(lo - 1, n, 0.05)
    assert 0.95 <= coverage < 0.99
    x = np.sort(np.random.default_rng(2).standard_normal((2000, n)), axis=1)
    q = norm.ppf(0.05)
    covered = np.mean((x[:, lo - 1] <= q) & (q <= x[:, hi - 1]))
    # Equal tails give 0.967 here, the shortest pair 0.953; 0.94 is 2.8 standard errors below.
    assert covered >= 0.94


def test_the_upper_tail_interval_covers_the_true_95th_percentile():
    n = 200
    lo, hi = ranks(n, "high")
    coverage = binom.cdf(n - lo, n, 0.05) - binom.cdf(n - hi, n, 0.05)
    assert 0.95 <= coverage < 0.99
    x = np.sort(np.random.default_rng(3).standard_normal((2000, n)), axis=1)
    q = norm.ppf(0.95)
    assert np.mean((x[:, lo - 1] <= q) & (q <= x[:, hi - 1])) >= 0.94


def test_the_quantile_interval_holds_the_point_quantile_and_is_made_of_runs():
    x = np.random.default_rng(4).standard_normal(200)
    found = tail_summary(x, "low", reps=100, rng=0)
    lower, upper = bounds(found["quantile_interval"])
    assert lower <= found["quantile"] <= upper
    lo, hi = ranks(200, "low")
    assert (lower, upper) == (np.sort(x)[lo - 1], np.sort(x)[hi - 1])


def test_a_higher_confidence_widens_the_quantile_interval():
    lo95, hi95 = ranks(200, "low")
    lo99, hi99 = ranks(200, "low", confidence=0.99)
    assert lo99 <= lo95 and hi99 >= hi95 and (lo99, hi99) != (lo95, hi95)


def test_too_few_runs_give_no_finite_lower_end_for_the_5th_percentile():
    # With 20 runs, P(at least one run below the 5th percentile) = 1 - 0.95**20 = 0.64 < 0.95, so
    # no order statistic bounds it from below at 95%.
    x = np.arange(20.0)
    lower, upper = bounds(tail_summary(x, "low", reps=100, rng=0)["quantile_interval"])
    assert not lower >= x.min()


def test_mirrored_values_give_the_mirrored_upper_tail():
    x = np.random.default_rng(5).standard_normal(150)
    low = tail_summary(x, "low", reps=500, rng=6)
    high = tail_summary(-x, "high", reps=500, rng=6)
    for key in ("worst", "quantile", "tail_mean", "worst_bound"):
        assert high[key] == -low[key]
    lower, upper = bounds(low["quantile_interval"])
    assert bounds(high["quantile_interval"]) == (-upper, -lower)
    lower, upper = bounds(low["tail_mean_interval"])
    width = upper - lower
    high_lower, high_upper = bounds(high["tail_mean_interval"])
    assert high_lower == pytest.approx(-upper, abs=0.1 * width)
    assert high_upper == pytest.approx(-lower, abs=0.1 * width)


@pytest.mark.parametrize("n, rank", [(100, 2), (1000, 39), (59, 1)])
def test_the_worst_bound_is_the_tolerance_order_statistic(n, rank):
    # Algorithms for Validation, section 12.2: P(Binomial(n, 0.05) >= rank) is 0.963 at n = 100
    # and 0.957 at n = 1000; at n = 59, 1 - 0.95**59 = 0.952.
    x = np.random.default_rng(n).permutation(np.arange(1.0, n + 1.0))
    assert tail_summary(x, "low", reps=100, rng=0)["worst_bound"] == rank
    assert tail_summary(x, "high", reps=100, rng=0)["worst_bound"] == n + 1 - rank


def test_58_runs_are_too_few_for_a_95_95_bound():
    # 1 - 0.95**58 = 0.949 < 0.95.
    assert no_bound(tail_summary(np.arange(58.0), "low", reps=100, rng=0)["worst_bound"])
    assert no_bound(tail_summary([1.0], "high", reps=100, rng=0)["worst_bound"])


def test_the_worst_bound_lies_below_the_true_5th_percentile_with_95_percent_confidence():
    n = 100
    rank = tail_summary(np.arange(1.0, n + 1.0), "low", reps=100, rng=0)["worst_bound"]
    x = np.sort(np.random.default_rng(7).standard_normal((4000, n)), axis=1)
    # 0.963 expected; 0.95 is 4 standard errors below.
    assert np.mean(x[:, int(rank) - 1] <= norm.ppf(0.05)) >= 0.95


def test_the_tail_mean_interval_stays_between_the_worst_value_and_the_largest():
    x = np.random.default_rng(8).standard_normal(400)
    found = tail_summary(x, "low", reps=400, rng=9)
    lower, upper = bounds(found["tail_mean_interval"])
    assert found["worst"] <= lower < found["tail_mean"] < upper <= x.max()


def test_the_tail_summary_is_reproducible_from_a_seed_or_a_generator():
    x = np.random.default_rng(10).standard_normal(80)
    first = tail_summary(x, "high", reps=300, rng=11)
    again = tail_summary(x, "high", reps=300, rng=11)
    assert bounds(again["tail_mean_interval"]) == bounds(first["tail_mean_interval"])
    a = tail_summary(x, "high", reps=300, rng=np.random.default_rng(12))
    b = tail_summary(x, "high", reps=300, rng=np.random.default_rng(12))
    assert bounds(a["tail_mean_interval"]) == bounds(b["tail_mean_interval"])


def test_doubling_every_value_doubles_the_tail_and_its_intervals():
    x = np.random.default_rng(13).standard_normal(120)
    one = tail_summary(x, "low", reps=400, rng=14)
    two = tail_summary(2.0 * x, "low", reps=400, rng=14)
    for key in ("worst", "quantile", "tail_mean", "worst_bound"):
        assert two[key] == pytest.approx(2.0 * one[key], rel=1e-12)
    for key in ("quantile_interval", "tail_mean_interval"):
        np.testing.assert_allclose(bounds(two[key]), 2.0 * np.array(bounds(one[key])), rtol=1e-12)


def test_infinite_values_stay_episode_values_in_every_interval():
    # Ratios that are infinite by design (a hold gate, a target never closed on) in 3 of 100 runs.
    x = np.concatenate([np.linspace(0.1, 0.9, 97), [np.inf] * 3])
    found = tail_summary(x, "high", reps=500, rng=15)
    assert found["worst"] == np.inf and found["worst_bound"] == np.inf
    assert found["tail_mean"] == np.inf
    assert found["quantile"] == unsafe_tail(x, "high")["quantile"]
    for key in ("quantile_interval", "tail_mean_interval"):
        lower, upper = bounds(found[key])
        assert not np.isnan(lower) and not np.isnan(upper) and lower <= upper
    # Margins that are infinite when nothing was approached sit far from the lower tail.
    margins = tail_summary(np.concatenate([np.arange(97.0), [np.inf] * 3]), "low", reps=500, rng=16)
    assert np.isfinite(margins["tail_mean"]) and np.all(
        np.isfinite(bounds(margins["tail_mean_interval"]))
    )


def test_one_run_and_equal_runs_give_degenerate_tails():
    one = tail_summary([3.0], "low", reps=100, rng=0)
    assert (one["worst"], one["quantile"], one["tail_mean"], one["n"]) == (3.0, 3.0, 3.0, 1)
    equal = tail_summary(np.full(300, 2.5), "low", reps=200, rng=0)
    assert bounds(equal["quantile_interval"]) == (2.5, 2.5)
    assert bounds(equal["tail_mean_interval"]) == (2.5, 2.5)
    assert equal["worst_bound"] == 2.5


@pytest.mark.parametrize("values, worse", [([], "low"), ([1.0, 2.0], "worst"), ([1.0, 2.0], "")])
def test_the_tail_summary_refuses_no_values_and_unknown_sides(values, worse):
    with pytest.raises(ValueError):
        tail_summary(values, worse, reps=100, rng=0)


@pytest.mark.parametrize(
    "n, coverage, confidence, rank",
    [
        # Algorithms for Validation, section 12.2: 95/95 at n = 100 and 1000.
        (100, 0.95, 0.95, 2),
        (1000, 0.95, 0.95, 39),
        # Zero-failure counts, the smallest n with 1 - coverage**n >= confidence (section 7.1).
        (59, 0.95, 0.95, 1),
        (230, 0.99, 0.90, 1),
        (299, 0.99, 0.95, 1),
    ],
)
def test_the_tolerance_order_reproduces_the_known_order_statistics(n, coverage, confidence, rank):
    assert tolerance_order(n, coverage, confidence) == rank


@pytest.mark.parametrize(
    "n, coverage, confidence",
    [(58, 0.95, 0.95), (229, 0.99, 0.90), (298, 0.99, 0.95), (1, 0.95, 0.95)],
)
def test_one_run_fewer_than_the_zero_failure_count_gives_no_bound(n, coverage, confidence):
    assert tolerance_order(n, coverage, confidence) is None


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(1, 3000),
    coverage=st.floats(0.5, 0.999, allow_subnormal=False),
    confidence=st.floats(0.5, 0.999, allow_subnormal=False),
)
def test_the_tolerance_order_is_the_highest_rank_with_the_stated_confidence(
    n, coverage, confidence
):
    rank = tolerance_order(n, coverage, confidence)
    p = 1.0 - coverage
    if rank is None:
        assert binom.sf(0, n, p) < confidence
        return
    assert 1 <= rank <= n
    assert binom.sf(rank - 1, n, p) >= confidence
    assert rank == n or binom.sf(rank, n, p) < confidence


@pytest.mark.parametrize(
    "coverage, confidence", [(0.0, 0.95), (1.0, 0.95), (0.95, 0.0), (0.95, 1.0)]
)
def test_the_tolerance_order_refuses_probabilities_outside_zero_and_one(coverage, confidence):
    with pytest.raises(ValueError):
        tolerance_order(100, coverage, confidence)


@pytest.mark.parametrize("n, risk", [(100, 0.436), (1000, 0.480)])
def test_the_conformal_bound_falls_short_of_its_coverage_almost_half_the_time(n, risk):
    # Split conformal takes order statistic floor(0.05 (n + 1)): 5 of 100, 50 of 1000. Its
    # coverage given the calibration runs falls below 0.95 when fewer than that many runs lie
    # below the 5th percentile: P(Binomial(n, 0.05) <= rank - 1) (Algorithms for Validation,
    # section 12.2).
    assert conformal_coverage_risk(n, 0.95) == pytest.approx(risk, abs=1e-3)


def test_more_calibration_runs_do_not_remove_the_conformal_risk():
    # 0.494 at n = 10000: the risk tends to one half, unlike a tolerance bound's.
    assert conformal_coverage_risk(10_000, 0.95) > 0.45


def test_compare_rates_uses_clopper_pearson_intervals():
    found = compare_rates(90, 100, 3, 10_000)
    # Algorithms for Decision Making, section 14.1: 90 of 100 and 3 failures in 10000.
    np.testing.assert_allclose(found["interval_a"], (0.8238, 0.9510), atol=1e-4)
    np.testing.assert_allclose(found["interval_b"], (6.187e-5, 8.765e-4), rtol=1e-3)
    assert found["difference"] == pytest.approx(0.9 - 3e-4, rel=1e-12)
    assert found["resolved"] is True


def test_the_rare_event_interval_stays_positive_where_the_normal_one_does_not():
    # The normal interval for 3 in 10000 (sample sd 0.0173188, standard error 1.73188e-4) has
    # lower end 3e-4 - 1.96 * 1.73188e-4 = -3.945e-5.
    p = np.r_[np.ones(3), np.zeros(9997)]
    assert p.mean() - 1.96 * p.std(ddof=1) / np.sqrt(len(p)) == pytest.approx(-3.945e-5, rel=1e-3)
    assert compare_rates(3, 10_000, 3, 10_000)["interval_a"][0] > 0.0


def test_a_difference_inside_the_wider_interval_width_is_unresolved():
    # Clopper-Pearson width at 50 of 100 is 0.2034 and is the wider of the two: 70 of 100 differs
    # by 0.20, 71 of 100 by 0.21.
    assert compare_rates(70, 100, 50, 100)["resolved"] is False
    assert compare_rates(71, 100, 50, 100)["resolved"] is True
    # 1 of 100 against 3 of 10000 differ by 0.0097: wider than the narrow interval (8.1e-4), not
    # the wide one (0.054).
    assert compare_rates(1, 100, 3, 10_000)["resolved"] is False


def test_an_unresolved_comparison_is_refused_on_request():
    with pytest.raises(ValueError, match="smaller than the interval width"):
        compare_rates(45, 50, 42, 50, require_resolved=True)
    found = compare_rates(95, 100, 40, 100, require_resolved=True)
    assert found["resolved"] is True
    assert found["difference"] == pytest.approx(0.55, rel=1e-12)


def test_the_difference_is_rate_a_minus_rate_b():
    assert compare_rates(40, 100, 95, 100)["difference"] == pytest.approx(-0.55, rel=1e-12)


def test_a_higher_confidence_widens_both_rate_intervals():
    at95 = compare_rates(30, 60, 20, 60)
    at99 = compare_rates(30, 60, 20, 60, confidence=0.99)
    for key in ("interval_a", "interval_b"):
        assert at99[key][0] < at95[key][0] and at99[key][1] > at95[key][1]


@pytest.mark.parametrize("args", [(5, 4, 1, 10), (1, 10, -1, 10), (0, 0, 1, 10)])
def test_compare_rates_refuses_impossible_counts(args):
    with pytest.raises(ValueError):
        compare_rates(*args)


@settings(max_examples=60, deadline=None)
@given(
    values=st.lists(
        st.one_of(
            st.floats(-50.0, 50.0, allow_subnormal=False),
            st.sampled_from([0.0, 1.0, np.inf, -np.inf]),
        ),
        min_size=1,
        max_size=60,
    ),
    worse=st.sampled_from(["low", "high"]),
)
def test_the_bootstrap_tail_means_are_the_unsafe_tail_of_each_resample(values, worse):
    # The resampled tail means are computed for all resamples at once; each must equal the
    # metrics' own tail mean of that resample, ties and infinite values included.
    from vbar.eval.tails import _lower_tail_means

    x = np.array(values)
    sign = 1.0 if worse == "low" else -1.0
    rows = np.random.default_rng(0).choice(x, size=(5, len(x)))
    found = sign * _lower_tail_means(np.sort(sign * rows, axis=1), 0.05)
    expected = np.array([unsafe_tail(row, worse)["tail_mean"] for row in rows])
    # Two summation orders over the same values: equal to rounding, not bit for bit.
    np.testing.assert_allclose(found, expected, rtol=1e-12, atol=1e-12)
