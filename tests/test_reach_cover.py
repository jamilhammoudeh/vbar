import dataclasses
import math

import numpy as np
import pytest

from vbar.constraints import KeepOut, interval_bounds
from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.reach import Zonotope, _nearest, drift_clearance, drift_cover
from vbar.safety.barriers import PassiveBarrier
from vbar.safety.cbf import BackupFilter
from vbar.scenarios.approach import ApproachConfig, ApproachEnv

N = mean_motion(EARTH.radius + 500e3)
R = 50.0
HOLD = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
TWO_ORBITS = 4.0 * math.pi / N
# The audited run of the approach scenario: 360 steps of 10 s, plus the step after the last.
AUDIT = 3610.0
TINY = np.array([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6])
DEPTH = KeepOut.resolution
# The closed form of test_reach.py: in-plane closing at this speed from the hold reaches 50 m.
PASSIVE_LIMIT = 1.4147e-3


def declared_everywhere(x):
    return True


def passive_set(radius):
    """The declared set of the free-drift backup filter on the approach scenario."""
    filt = BackupFilter(barrier="passive")
    filt.reset(ApproachEnv(ApproachConfig(keep_out_radius=radius)).agent_view())
    return filt.contains


def closed_drift(delta, y0):
    """x0 = delta and vy0 = -1.5 n delta: x stays at delta and y falls at 1.5 n delta."""
    return np.array([delta, y0, 0.0, 0.0, -1.5 * N * delta, 0.0])


def dense_margins(states, duration, extra=()):
    """Lowest keep-out margin of each state's free drift at 1 s steps and the times `extra`, m."""
    ts = np.concatenate([np.arange(0.0, duration, 1.0), [duration], extra])
    phis, _ = cw_discrete(N, ts)
    ranges = np.linalg.norm(np.einsum("tij,sj->sti", phis[:, :3], states), axis=-1)
    return ranges.min(axis=1) - R


def per_sample_drift_clearance(x0, duration, samples):
    """(found, certified, t) of drift_clearance with every sample set built as its own zonotope."""
    times = np.linspace(0.0, duration, samples)
    h = float(times[1] - times[0])
    sets = [x0.map(phi) for phi in cw_discrete(N, times)[0]]
    positions = [z.map(np.eye(6)[:3]) for z in sets]
    velocities = [z.map(np.eye(6)[3:]) for z in sets]
    lower, upper = _nearest(
        np.array([p.center for p in positions]), np.array([p.generators for p in positions])
    )
    lower, upper = lower - R, upper - R
    largest = np.zeros((samples, 6))
    for i, (p, v) in enumerate(zip(positions, velocities, strict=True)):
        largest[i, 0] = np.linalg.norm(p.center) + np.sum(np.linalg.norm(p.generators, axis=0))
        largest[i, 3] = np.linalg.norm(v.center) + np.sum(np.linalg.norm(v.generators, axis=0))
    speed, _ = interval_bounds(largest, h, 0.0, N)
    k = int(np.argmin(upper))
    return float(upper[k]), float(np.min(0.5 * (lower[:-1] + lower[1:] - speed * h))), k * h


# The cases of the drift_clearance tests in test_reach.py. The 2001- and 4001-sample ones take
# several seconds and run in benchmarks/drift_cover_checks.py.
CLEARANCE_CASES = {
    "point": ([0.0] * 6, 801),
    "three_metres": ([3.0] * 3 + [3e-3] * 3, 2001),
    "sixty_centimetres": ([0.6] * 3 + [3e-4] * 3, 1001),
    "safe_velocities": ([0.0] * 3 + [1.4140e-3] * 3, 4001),
    "unsafe_velocities": ([0.0] * 3 + [1.4160e-3] * 3, 801),
}
QUICK_CLEARANCE_CASES = ("point", "sixty_centimetres", "unsafe_velocities")


def check_stacked_drift_clearance_equals_the_per_sample_sets(case):
    half_widths, samples = CLEARANCE_CASES[case]
    x0 = Zonotope.box(HOLD, half_widths)
    clearance = drift_clearance(N, x0, R, TWO_ORBITS, samples)
    found, certified, t = per_sample_drift_clearance(x0, TWO_ORBITS, samples)
    # The two paths form the same products in a different memory layout; 1e-12 of the 50 m
    # scale covers their last-bit differences.
    assert clearance.found == pytest.approx(found, rel=1e-12, abs=1e-12 * R)
    assert clearance.certified == pytest.approx(certified, rel=1e-12, abs=1e-12 * R)
    assert clearance.t == pytest.approx(t, rel=1e-12)


@pytest.mark.parametrize("case", QUICK_CLEARANCE_CASES)
def test_stacked_drift_clearance_equals_the_per_sample_sets(case):
    check_stacked_drift_clearance_equals_the_per_sample_sets(case)


def velocity_box(half_width):
    half = np.array([0.0, 0.0, 0.0, half_width, half_width, half_width])
    return HOLD - half, HOLD + half


def test_velocity_box_of_1_4140_mm_per_s_at_the_hold_is_certified_as_one_box():
    lo, hi = velocity_box(1.4140e-3)
    cover = drift_cover(N, lo, hi, R, TWO_ORBITS, 4001, declared_everywhere, 8, 4)
    assert cover.boxes == 1
    assert cover.certified == 1.0
    assert cover.excluded == 0.0
    assert cover.undecided == 0.0
    np.testing.assert_array_equal(cover.certified_boxes, [[lo, hi]])
    assert cover.counterexamples == ()


def test_velocity_box_of_1_4160_mm_per_s_at_the_hold_is_split():
    lo, hi = velocity_box(1.4160e-3)
    cover = drift_cover(N, lo, hi, R, TWO_ORBITS, 4001, declared_everywhere, 2, 4)
    assert cover.boxes == 2
    assert cover.certified < 1.0
    # Boxes left when the budget is spent are undecided, so the fractions cover the box.
    assert cover.certified + cover.excluded + cover.undecided == 1.0
    assert len(cover.undecided_boxes) >= 1


def test_drift_entering_after_the_one_hour_horizon_is_a_counterexample():
    # The capped barrier checks the drift up to t_cap = 3600 s only, and this drift enters at
    # 3602 s.
    x = closed_drift(5.0, math.sqrt(R * R - 25.0) + 1.5 * N * 5.0 * 3602.0)
    contains = PassiveBarrier(radius=R, n=N, t_cap=3600.0).contains
    assert contains(x)
    cover = drift_cover(N, x - TINY, x + TINY, R, AUDIT, 3611, contains, 16, 4)
    entering = [
        (margin, t)
        for state, margin, t in cover.counterexamples
        if contains(state) and margin < -DEPTH and 3602.0 < t <= AUDIT
    ]
    assert entering


def test_declared_radius_of_45_m_gives_a_counterexample_three_metres_deep():
    # The drift passes y = 0 at 1281 s with x = 47 m, so the 50 m keep-out is entered by 3 m.
    x = closed_drift(47.0, 100.0)
    contains = passive_set(45.0)
    assert contains(x)
    cover = drift_cover(N, x - TINY, x + TINY, R, AUDIT, 3611, contains, 16, 4)
    assert len(cover.counterexamples) >= 1
    for state, margin, _ in cover.counterexamples:
        assert contains(state)
        assert -3.0 <= margin <= -2.9


def test_declared_radius_of_50_m_gives_no_counterexample():
    x = closed_drift(47.0, 100.0)
    cover = drift_cover(N, x - TINY, x + TINY, R, AUDIT, 3611, passive_set(R), 16, 4)
    # Every drift of the box enters, and the declared set holds none of the centers checked.
    assert cover.excluded == 1.0
    assert cover.counterexamples == ()


def test_drift_entering_by_more_than_the_resolution_is_excluded():
    # y0 puts the drift at y = 0, x = R - 1.5 mm on the sample at 1000 s.
    delta = R - 1.5e-3
    x = closed_drift(delta, 1.5 * N * delta * 1000.0)
    cover = drift_cover(N, x, x, R, AUDIT, 3611, declared_everywhere, 4, 4)
    assert cover.excluded == 1.0
    ((state, margin, t),) = cover.counterexamples
    np.testing.assert_array_equal(state, x)
    assert margin == pytest.approx(-1.5e-3, abs=1e-9)
    assert t == 1000.0


def test_drift_entering_by_less_than_the_resolution_is_neither_excluded_nor_a_counterexample():
    delta = R - 0.5e-3
    x = closed_drift(delta, 1.5 * N * delta * 1000.0)
    cover = drift_cover(N, x, x, R, AUDIT, 3611, declared_everywhere, 4, 4)
    assert cover.excluded == 0.0
    assert cover.certified == 0.0
    assert cover.undecided == 1.0
    assert cover.counterexamples == ()


def test_box_ending_at_a_counterexample_counts_as_undecided():
    # Four metres along y keep every sample set wider than the 3 m incursion of the center, so
    # the box is neither excluded nor certified and stops at its center's counterexample.
    x = closed_drift(47.0, 100.0)
    half = np.array([0.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    cover = drift_cover(N, x - half, x + half, R, AUDIT, 3611, declared_everywhere, 16, 4)
    assert cover.boxes == 1
    assert cover.undecided == 1.0
    assert cover.excluded == 0.0
    np.testing.assert_array_equal(cover.undecided_boxes, [[x - half, x + half]])
    (state, margin, t), *_ = cover.counterexamples
    np.testing.assert_array_equal(state, x)
    assert margin < -DEPTH


def test_box_is_halved_along_the_axis_that_moves_its_closest_sample_most():
    # At the start only the 0.1 m radial width moves the position; over two orbits the radial
    # width drifts y by about 7.5 m and the 2 mm/s along-track width by about 68 m, and the box
    # comes closest late.
    half = np.array([0.1, 0.0, 0.0, 0.0, 2e-3, 0.0])
    lo, hi = HOLD - half, HOLD + half
    cover = drift_cover(N, lo, hi, R, TWO_ORBITS, 801, declared_everywhere, 1, 4)
    assert cover.boxes == 1
    assert cover.certified == 0.0
    assert cover.excluded == 0.0
    halved = 2.0 * half
    halved[4] = half[4]
    widths = cover.undecided_boxes[:, 1] - cover.undecided_boxes[:, 0]
    np.testing.assert_allclose(widths, [halved, halved], rtol=1e-12)


def test_box_at_the_depth_limit_is_undecided():
    lo, hi = velocity_box(1.4160e-3)
    cover = drift_cover(N, lo, hi, R, TWO_ORBITS, 801, declared_everywhere, 10_000, 0)
    assert cover.boxes == 1
    assert cover.undecided == 1.0


def check_certified_boxes_drift_clear_and_counterexamples_enter(boxes, states):
    """Fly `states` drawn states of each of `boxes` certified boxes, and every counterexample.

    The boxes come from covers of seeded boxes of in-plane closing velocities at the hold that
    straddle the passive limit, so the true margins of their states lie within about 0.5 m of
    zero; the flights are at 1 s steps over the two orbits covered.
    """
    rng = np.random.default_rng(23)
    flown = entered = 0
    for _ in range(2 * boxes):
        if flown == boxes:
            break
        speed = PASSIVE_LIMIT - rng.uniform(0.0, 8e-6)
        half = rng.uniform(8e-6, 2e-5)
        lo, hi = HOLD.copy(), HOLD.copy()
        lo[3:5], hi[3:5] = -speed - half, -speed + half
        cover = drift_cover(N, lo, hi, R, TWO_ORBITS, 801, declared_everywhere, 8, 8)
        for box_lo, box_hi in cover.certified_boxes[: boxes - flown]:
            drawn = rng.uniform(box_lo, box_hi, size=(states, 6))
            assert np.all(dense_margins(drawn, TWO_ORBITS) >= 0.0)
            flown += 1
        for state, margin, t in cover.counterexamples:
            assert np.all(lo <= state) and np.all(state <= hi)
            at_t = np.linalg.norm(cw_discrete(N, t)[0][:3] @ state) - R
            assert at_t == pytest.approx(margin, abs=1e-9)
            assert margin < -DEPTH
            assert dense_margins(state[None], TWO_ORBITS, [t])[0] < -DEPTH
            entered += 1
    assert flown == boxes
    assert entered >= 1


def test_certified_boxes_at_the_passive_limit_drift_clear_and_counterexamples_enter():
    # A certificate loosened by 0.5 m certifies the first box whole, and about 2 in 5 of its
    # states then enter.
    check_certified_boxes_drift_clear_and_counterexamples_enter(1, 16)


def test_cover_result_is_frozen_and_read_only():
    lo, hi = velocity_box(1.4140e-3)
    cover = drift_cover(N, lo, hi, R, TWO_ORBITS, 801, declared_everywhere, 1, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cover.boxes = 2  # type: ignore[misc]
    with pytest.raises(ValueError):
        cover.certified_boxes[0, 0, 0] = 1.0


@pytest.mark.parametrize(
    "change",
    [
        {"lo": HOLD + 1.0},
        {"lo": np.zeros(5)},
        {"hi": np.full(6, math.nan)},
        {"max_boxes": 0},
        {"depth": -1},
        {"samples": 1},
        {"radius": -1.0},
        {"duration": 0.0},
    ],
)
def test_cover_refuses_bad_inputs(change):
    lo, hi = velocity_box(1e-3)
    args = {
        "n": N,
        "lo": lo,
        "hi": hi,
        "radius": R,
        "duration": TWO_ORBITS,
        "samples": 801,
        "contains": declared_everywhere,
        "max_boxes": 4,
        "depth": 2,
    }
    with pytest.raises(ValueError):
        drift_cover(**(args | change))
