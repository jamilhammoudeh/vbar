"""Multiple-comparison corrections and the paired sign-flip test.

Holm, S., A simple sequentially rejective multiple test procedure, Scandinavian Journal of
Statistics 6, 1979. Benjamini, Y. and Hochberg, Y., Controlling the false discovery rate, Journal
of the Royal Statistical Society B 57, 1995.
"""

import dataclasses
import itertools
import math

import numpy as np
import pytest

from vbar.eval.inference import (
    Correction,
    PairedTest,
    benjamini_hochberg,
    holm,
    sign_flip_test,
)

P = [0.01, 0.04, 0.03, 0.005]
PROCEDURES = [holm, benjamini_hochberg]


def hostile(function, label, match=None, **arguments):
    return pytest.param(function, arguments, match, id=f"{function.__name__}-{label}")


def test_a_correction_is_frozen():
    found = holm([0.01])
    assert isinstance(found, Correction)
    with pytest.raises(dataclasses.FrozenInstanceError):
        found.adjusted = np.zeros(1)


def test_holm_matches_the_hand_computed_adjustment():
    found = holm(P)
    np.testing.assert_allclose(found.adjusted, [0.03, 0.06, 0.06, 0.02], rtol=1e-12)
    np.testing.assert_array_equal(found.reject, [True, False, False, True])
    assert found.reject.dtype == bool


def test_benjamini_hochberg_matches_the_hand_computed_adjustment():
    found = benjamini_hochberg(P)
    np.testing.assert_allclose(found.adjusted, [0.02, 0.04, 0.04, 0.02], rtol=1e-12)
    np.testing.assert_array_equal(found.reject, [True, True, True, True])


def test_the_level_decides_the_rejections():
    np.testing.assert_array_equal(holm(P, alpha=0.1).reject, [True, True, True, True])
    np.testing.assert_array_equal(holm(P, alpha=0.025).reject, [False, False, False, True])
    np.testing.assert_array_equal(benjamini_hochberg(P, q=0.03).reject, [True, False, False, True])


@pytest.mark.parametrize("procedure", PROCEDURES)
def test_a_single_p_value_is_not_adjusted(procedure):
    found = procedure([0.2])
    np.testing.assert_allclose(found.adjusted, [0.2], rtol=1e-12)
    np.testing.assert_array_equal(found.reject, [False])
    assert procedure([0.03]).reject[0]


def test_tied_p_values_get_the_same_adjustment():
    np.testing.assert_allclose(holm([0.01, 0.01, 0.01]).adjusted, [0.03] * 3, rtol=1e-12)
    np.testing.assert_allclose(benjamini_hochberg([0.01, 0.01, 0.01]).adjusted, [0.01] * 3)
    # Sorted 0.01, 0.01, 0.2, 0.2 with m = 4: steps 4 * 0.01, 3 * 0.01, 2 * 0.2, 1 * 0.2 =
    # 0.04, 0.03, 0.4, 0.2; the running maximum gives 0.04, 0.04, 0.4, 0.4.
    np.testing.assert_allclose(holm([0.2, 0.01, 0.2, 0.01]).adjusted, [0.4, 0.04, 0.4, 0.04])


@pytest.mark.parametrize("procedure", PROCEDURES)
def test_no_p_values_give_empty_arrays(procedure):
    found = procedure([])
    assert found.adjusted.shape == (0,)
    assert found.reject.shape == (0,)
    assert found.reject.dtype == bool


@pytest.mark.parametrize("procedure", PROCEDURES)
def test_adjustment_does_not_depend_on_input_order(procedure):
    gen = np.random.default_rng(1)
    p = gen.random(12)
    order = gen.permutation(12)
    base, shuffled = procedure(p), procedure(p[order])
    np.testing.assert_allclose(shuffled.adjusted, base.adjusted[order], rtol=1e-12)
    np.testing.assert_array_equal(shuffled.reject, base.reject[order])


@pytest.mark.parametrize("procedure", PROCEDURES)
def test_adjusted_values_lie_between_the_raw_value_and_one(procedure):
    gen = np.random.default_rng(2)
    for _ in range(50):
        p = gen.random(gen.integers(1, 15)) ** 3
        adjusted = procedure(p).adjusted
        assert np.all(adjusted >= p - 1e-15)
        assert np.all(adjusted <= 1.0)
    # Large raw values hit the cap.
    assert procedure([0.9, 0.8]).adjusted.max() <= 1.0


@pytest.mark.parametrize("procedure", PROCEDURES)
def test_adjusted_values_never_decrease_with_the_raw_value(procedure):
    gen = np.random.default_rng(3)
    for _ in range(50):
        p = gen.random(gen.integers(2, 15)) ** 2
        adjusted = procedure(p).adjusted[np.argsort(p)]
        assert np.all(np.diff(adjusted) >= -1e-15)


def test_holm_rejects_at_least_what_bonferroni_does_and_at_most_what_bh_does():
    gen = np.random.default_rng(4)
    for _ in range(300):
        m = int(gen.integers(1, 20))
        p = gen.random(m) ** 4
        level = float(gen.choice([0.01, 0.05, 0.1]))
        bonferroni = m * p <= level
        reject_holm = holm(p, alpha=level).reject
        reject_bh = benjamini_hochberg(p, q=level).reject
        assert np.all(reject_holm[bonferroni])
        assert np.all(reject_bh[reject_holm])


def test_holm_keeps_the_family_wise_error_rate_under_the_global_null():
    gen = np.random.default_rng(5)
    families, size = 2000, 20
    p = gen.random((families, size))
    share = np.mean([holm(row).reject.any() for row in p])
    assert share <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / families)


def test_benjamini_hochberg_keeps_the_false_discovery_rate_under_the_global_null():
    # With every hypothesis null, a discovery is a false discovery, so the FDR equals the share
    # of families with any rejection.
    gen = np.random.default_rng(6)
    families, size = 2000, 20
    p = gen.random((families, size))
    share = np.mean([benjamini_hochberg(row).reject.any() for row in p])
    assert share <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / families)


def test_holm_rejects_a_clear_effect_among_nulls():
    p = [0.4, 1e-6, 0.7, 0.2, 0.9, 0.55, 0.3, 0.8, 0.65, 0.5]
    found = holm(p)
    assert found.reject[1]
    assert found.reject.sum() == 1


BAD_P = [
    ("nan", [0.1, np.nan]),
    ("negative", [0.1, -0.01]),
    ("above-one", [0.1, 1.01]),
    ("two-dimensional", [[0.1, 0.2], [0.3, 0.4]]),
    ("bool-items", [True, False]),
    ("string-items", ["a", "b"]),
    ("numeric-strings", ["0.1", "0.2"]),
]
BAD_LEVEL = [("zero", 0.0), ("one", 1.0), ("nan", math.nan), ("bool", True)]


@pytest.mark.parametrize(
    "function, arguments, match",
    [
        *[hostile(f, label, pvalues=v) for f in PROCEDURES for label, v in BAD_P],
        *[hostile(holm, f"alpha-{k}", pvalues=[0.1, 0.2], alpha=v) for k, v in BAD_LEVEL],
        *[hostile(benjamini_hochberg, f"q-{k}", pvalues=[0.1, 0.2], q=v) for k, v in BAD_LEVEL],
    ],
)
def test_hostile_correction_inputs_are_refused(function, arguments, match):
    with pytest.raises(ValueError, match=match):
        function(**arguments)


REPS = 200
DIFFERENCES = np.array([3.0, -1.0, 4.0, 1.0, 5.0, -9.0, 2.0, 6.0, -5.0, 3.0])


def exact_p(differences):
    flips = np.array(list(itertools.product([-1.0, 1.0], repeat=len(differences))))
    totals = np.abs(flips @ differences) / len(differences)
    observed = abs(differences.mean())
    return float(np.mean(totals >= observed * (1.0 - 1e-12)))


def test_a_paired_test_is_frozen():
    found = sign_flip_test({"a": [1.0, 2.0, 3.0]}, {"a": [0.0, 0.0, 0.0]}, reps=1000, rng=0)
    assert isinstance(found, PairedTest)
    with pytest.raises(dataclasses.FrozenInstanceError):
        found.p_value = 0.0


def test_the_statistic_is_the_mean_difference_and_pairs_are_counted():
    x = {"a": [4.0, 6.0, 8.0], "b": [1.0, 1.0]}
    y = {"a": [3.0, 3.0, 3.0], "b": [0.0, 3.0]}
    found = sign_flip_test(x, y, reps=1000, rng=0)
    assert found.pairs == 5
    # Differences 1, 3, 5, 1, -2 pooled over both scenarios.
    assert found.statistic == pytest.approx(8.0 / 5.0, rel=1e-12)


def test_the_small_case_p_value_is_near_the_exact_one():
    # Differences 1, 2, 3: only the observed and the all-flipped assignment reach |sum| = 6, so
    # the exact two-sided p is 2 / 8.
    reps = 20_000
    found = sign_flip_test({"a": [1.0, 2.0, 3.0]}, {"a": [0.0, 0.0, 0.0]}, reps=reps, rng=0)
    assert found.p_value == pytest.approx(0.25, abs=4 * math.sqrt(0.25 * 0.75 / reps))


def test_the_p_value_matches_full_enumeration_of_sign_assignments():
    exact = exact_p(DIFFERENCES)
    reps = 20_000
    y = np.arange(10.0)
    found = sign_flip_test({"a": y + DIFFERENCES}, {"a": y}, reps=reps, rng=7)
    assert found.p_value == pytest.approx(exact, abs=4 * math.sqrt(exact * (1 - exact) / reps))


def test_a_clear_effect_gets_the_smallest_possible_p_value():
    # With 30 positive differences only the observed pattern and its mirror reach the observed
    # |T|; the mirror has probability 2**-29 per draw, so no random draw does.
    reps = 1000
    found = sign_flip_test({"a": np.arange(1.0, 31.0)}, {"a": np.zeros(30)}, reps=reps, rng=0)
    assert found.p_value == 1.0 / (reps + 1)


def test_all_zero_differences_give_a_p_value_of_one():
    found = sign_flip_test({"a": [1.0, 2.0, 3.0]}, {"a": [1.0, 2.0, 3.0]}, reps=500, rng=0)
    assert found.p_value == 1.0
    assert found.statistic == 0.0


def test_swapping_the_agents_negates_the_statistic_and_keeps_the_p_value():
    gen = np.random.default_rng(8)
    x = {"a": gen.normal(1.0, 1.0, 12), "b": gen.normal(0.0, 1.0, 9)}
    y = {"a": gen.normal(0.0, 1.0, 12), "b": gen.normal(0.0, 1.0, 9)}
    forward = sign_flip_test(x, y, reps=2000, rng=3)
    backward = sign_flip_test(y, x, reps=2000, rng=3)
    assert backward.statistic == pytest.approx(-forward.statistic, rel=1e-12)
    assert backward.p_value == forward.p_value


def test_the_same_seed_gives_the_same_p_value():
    gen = np.random.default_rng(9)
    x, y = {"a": gen.normal(0.3, 1.0, 20)}, {"a": gen.normal(0.0, 1.0, 20)}
    first = sign_flip_test(x, y, reps=500, rng=11)
    again = sign_flip_test(x, y, reps=500, rng=11)
    assert first == again
    seen = {sign_flip_test(x, y, reps=500, rng=seed).p_value for seed in range(12)}
    assert len(seen) > 1


def test_a_generator_can_be_passed_as_the_seed():
    x, y = {"a": [1.0, 2.0, 0.5, 3.0]}, {"a": [0.0, 0.0, 0.0, 0.0]}
    first = sign_flip_test(x, y, reps=300, rng=np.random.default_rng(4))
    again = sign_flip_test(x, y, reps=300, rng=np.random.default_rng(4))
    assert first == again


def test_the_order_of_scenarios_in_the_mapping_does_not_matter():
    gen = np.random.default_rng(10)
    a, b = gen.normal(0.5, 1.0, 10), gen.normal(0.5, 1.0, 7)
    ya, yb = gen.normal(size=10), gen.normal(size=7)
    forward = sign_flip_test({"a": a, "b": b}, {"a": ya, "b": yb}, reps=1000, rng=5)
    backward = sign_flip_test({"b": b, "a": a}, {"b": yb, "a": ya}, reps=1000, rng=5)
    assert forward.statistic == pytest.approx(backward.statistic, rel=1e-12)
    assert forward.pairs == backward.pairs == 17
    assert forward.p_value == backward.p_value


def test_splitting_a_scenario_keeps_the_statistic_and_the_pair_count():
    gen = np.random.default_rng(12)
    x, y = gen.normal(1.0, 1.0, 12), gen.normal(size=12)
    whole = sign_flip_test({"a": x}, {"a": y}, reps=500, rng=0)
    split = sign_flip_test({"a": x[:5], "b": x[5:]}, {"a": y[:5], "b": y[5:]}, reps=500, rng=0)
    assert split.statistic == pytest.approx(whole.statistic, rel=1e-12)
    assert split.pairs == whole.pairs == 12


def test_runs_are_paired_by_index_within_a_scenario():
    found = sign_flip_test({"a": [5.0, 1.0, 3.0]}, {"a": [4.0, 0.0, 2.0]}, reps=500, rng=0)
    assert found.statistic == pytest.approx(1.0, rel=1e-12)


def test_the_test_is_calibrated_under_the_null():
    gen = np.random.default_rng(13)
    trials, n = 400, 30
    rejected = 0
    for seed in range(trials):
        d = gen.normal(0.0, 1.0, n)
        found = sign_flip_test({"a": d}, {"a": np.zeros(n)}, reps=REPS, rng=seed)
        rejected += found.p_value <= 0.05
    assert 0.02 <= rejected / trials <= 0.09


def test_the_test_detects_a_one_sigma_shift():
    gen = np.random.default_rng(14)
    trials, n = 100, 30
    rejected = 0
    for seed in range(trials):
        d = gen.normal(1.0, 1.0, n)
        found = sign_flip_test({"a": d}, {"a": np.zeros(n)}, reps=REPS, rng=seed)
        rejected += found.p_value <= 0.05
    assert rejected / trials > 0.8


def test_scenarios_on_very_different_scales_stay_calibrated():
    gen = np.random.default_rng(15)
    trials, n = 400, 15
    rejected = 0
    for seed in range(trials):
        x = {"small": gen.normal(0.0, 0.01, n), "large": gen.normal(0.0, 100.0, n)}
        y = {"small": np.zeros(n), "large": np.zeros(n)}
        rejected += sign_flip_test(x, y, reps=REPS, rng=seed).p_value <= 0.05
    assert 0.02 <= rejected / trials <= 0.09


GOOD = {"a": [1.0, 2.0, 3.0]}
BAD_REPS = [99, 0, -100, 100.5, math.nan, "1000", None]


@pytest.mark.parametrize(
    "function, arguments, match",
    [
        hostile(sign_flip_test, "different-scenarios", x={"a": [1.0]}, y={"b": [1.0]}),
        hostile(sign_flip_test, "extra-scenario", x={"a": [1.0], "b": [1.0]}, y={"a": [1.0]}),
        hostile(sign_flip_test, "unequal-counts", x={"a": [1.0, 2.0]}, y={"a": [1.0]}),
        hostile(
            sign_flip_test,
            "unequal-counts-in-one-of-two",
            x={"a": [1.0, 2.0], "b": [1.0, 2.0]},
            y={"a": [1.0, 2.0], "b": [1.0]},
        ),
        hostile(sign_flip_test, "nan-in-x", "NaN", x={"a": [1.0, np.nan]}, y={"a": [1.0, 2.0]}),
        hostile(sign_flip_test, "nan-in-y", "NaN", x={"a": [1.0, 2.0]}, y={"a": [1.0, np.nan]}),
        hostile(sign_flip_test, "infinite-score", x={"a": [1.0, np.inf]}, y={"a": [1.0, 2.0]}),
        hostile(sign_flip_test, "inf-minus-inf", x={"a": [np.inf, 1.0]}, y={"a": [np.inf, 2.0]}),
        hostile(sign_flip_test, "no-scenarios", x={}, y={}),
        hostile(sign_flip_test, "no-runs", x={"a": []}, y={"a": []}),
        *[hostile(sign_flip_test, f"reps-{r!r}", x=GOOD, y=GOOD, reps=r) for r in BAD_REPS],
    ],
)
def test_hostile_paired_test_inputs_are_refused(function, arguments, match):
    with pytest.raises(ValueError, match=match):
        function(**arguments)
