import numpy as np
import pytest

from vbar.stress.falsify import cross_entropy

# 128-bit root seeds; the search refuses anything below 2**64, as the audit does.
SEEDS = [(1 << 64) + 7919 * k + 0x5DEECE66D for k in range(20)]


def two_wells(x):
    """A shallow well at 0.15 and the global minimum, a narrow one, at 0.83."""
    x = float(x[0])
    return min((x - 0.15) ** 2 + 0.05, 20.0 * (x - 0.83) ** 2)


FAILURE = np.array([0.71, 0.23, 0.58, 0.36, 0.84, 0.12])


def bowl_with_a_failure_ball(x):
    """A smooth bowl whose bottom holds a ball of radius 0.03 where the score is negative."""
    return float(np.sum((np.asarray(x) - FAILURE) ** 2)) - 0.03**2


def everywhere(x):
    return True


@pytest.mark.parametrize("seed", SEEDS)
def test_the_search_finds_the_narrow_global_well_in_one_dimension(seed):
    found = cross_entropy(two_wells, [0.0], [1.0], accept=everywhere, budget=500, seed=seed)
    assert abs(float(found.x[0]) - 0.83) < 1e-3
    assert found.evaluations <= 500


@pytest.mark.parametrize("seed", SEEDS[:5])
def test_the_search_finds_a_small_failure_ball_in_six_dimensions(seed):
    found = cross_entropy(
        bowl_with_a_failure_ball, np.zeros(6), np.ones(6), accept=everywhere, budget=1000,
        seed=seed,
    )  # fmt: skip
    assert found.score < 0.0
    assert np.linalg.norm(np.asarray(found.x) - FAILURE) < 0.03


def test_uniform_draws_with_the_same_budget_miss_the_failure_ball():
    # A ball of radius 0.03 fills about 4e-6 of the unit 6-cube; 1000 uniform draws miss it.
    draws = np.random.default_rng(SEEDS[0]).uniform(size=(1000, 6))
    assert min(bowl_with_a_failure_ball(x) for x in draws) > 0.0


def test_the_same_seed_gives_the_same_history_with_one_or_two_workers():
    # A numpy score, so each worker process imports numpy alone and not this package; 40
    # evaluations of 10 per chain are two generations of each of the two chains.
    kwargs = {"budget": 40, "population": 10, "seed": SEEDS[1]}
    one = cross_entropy(np.linalg.norm, [-1.0, -1.0], [2.0, 2.0], workers=1, **kwargs)
    two = cross_entropy(np.linalg.norm, [-1.0, -1.0], [2.0, 2.0], workers=2, **kwargs)
    assert [g.chain for g in one.generations] == [0, 1, 0, 1]
    np.testing.assert_array_equal(one.points, two.points)
    np.testing.assert_array_equal(one.scores, two.scores)
    for a, b in zip(one.generations, two.generations, strict=True):
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.sigma, b.sigma)
    assert one.score == two.score


def test_the_spread_never_falls_below_its_floor():
    # Without the floor the cross-entropy spread collapses onto the first elites.
    found = cross_entropy(two_wells, [0.0], [1.0], accept=everywhere, budget=500, seed=SEEDS[2])
    assert len(found.generations) > 1
    assert all(np.all(np.asarray(g.sigma) >= 0.02) for g in found.generations)


def test_rejected_draws_are_counted_and_never_scored():
    def left_half(x):
        return float(x[0]) < 0.5

    seen = []

    def score(x):
        seen.append(np.array(x, dtype=float))
        return float(np.sum((np.asarray(x) - 0.9) ** 2))

    found = cross_entropy(score, [0.0, 0.0], [1.0, 1.0], accept=left_half, budget=200,
                          seed=SEEDS[3])  # fmt: skip
    assert found.rejected > 0
    assert len(seen) == found.evaluations == len(found.points)
    assert all(left_half(x) for x in seen)
    assert all(left_half(x) for x in found.points)
    # The best point the accepted half allows lies on its edge, toward (0.9, 0.9).
    assert float(found.x[0]) < 0.5


def test_proposals_that_leave_the_box_are_rejected_and_never_scored():
    lo, hi = np.array([-80.0, 52.0]), np.array([80.0, 150.0])
    # The first 100 draws are uniform; the 300 after them are normal proposals pushed against
    # the low edge of x, some of which fall outside the box.
    found = cross_entropy(lambda x: float(x[0]), lo, hi, accept=everywhere, budget=400,
                          seed=SEEDS[4])  # fmt: skip
    assert len(found.generations) == 8
    assert found.rejected > 0
    points = np.asarray(found.points)
    assert np.all(points >= lo) and np.all(points <= hi)
    assert float(found.x[0]) < lo[0] + 1.0


def test_the_same_seed_repeats_and_the_seed_is_recorded():
    a = cross_entropy(two_wells, [0.0], [1.0], accept=everywhere, budget=100, seed=SEEDS[5])
    b = cross_entropy(two_wells, [0.0], [1.0], accept=everywhere, budget=100, seed=SEEDS[5])
    np.testing.assert_array_equal(a.points, b.points)
    assert a.seed == SEEDS[5]


def test_a_drawn_seed_is_128_bit_and_recorded():
    found = cross_entropy(two_wells, [0.0], [1.0], accept=everywhere, budget=50, seed=None)
    assert (1 << 64) <= found.seed < (1 << 128)


@pytest.mark.parametrize("seed", [(1 << 64) - 1, 7, -1, 1 << 128, True])
def test_a_small_or_out_of_range_seed_is_refused(seed):
    with pytest.raises(ValueError, match="seed"):
        cross_entropy(two_wells, [0.0], [1.0], accept=everywhere, budget=50, seed=seed)


def test_the_smallest_allowed_seed_is_accepted():
    found = cross_entropy(two_wells, [0.0], [1.0], accept=everywhere, budget=50, seed=1 << 64)
    assert found.seed == 1 << 64
