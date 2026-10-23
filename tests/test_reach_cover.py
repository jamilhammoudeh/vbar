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


BURN = 30.0
# One orbit after mid-burn, so the closest approach of the CAM arcs below is the middle sample.
CAM_DURATION = 2.0 * math.pi / N + BURN
T_CLOSEST = math.pi / N + 0.5 * BURN
YZ_BOX = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0])


def radial(a):
    return np.array([a, 0.0, 0.0])


def cam_closest_range(alpha):
    """Least range of the arc from the hold under an inward radial CAM of alpha m/s^2, m.

    For t >= t_b the arc is (Gamma(t) - Gamma(t - t_b)) a: with phi = n t - n t_b / 2 and
    s = sin(n t_b / 2), x = -b sin(phi) and y = -D - 2 b cos(phi), b = 2 alpha s / n^2 and
    D = 100 - 2 alpha t_b / n. Then rho^2 = b^2 + D^2 + 4 b D cos(phi) + 3 b^2 cos(phi)^2 is
    least at cos(phi) = -1 while D >= 1.5 b, at t = pi / n + t_b / 2, where rho = D - 2 b.
    """
    s = math.sin(0.5 * N * BURN)
    return 100.0 - 2.0 * alpha * BURN / N - 4.0 * alpha * s / N**2


def cam_arcs(states, accel, ts, burn_time=BURN):
    """States at times `ts` of the arcs from `states` under `accel` held for `burn_time`.

    The burn adds (Gamma(t) - Gamma(t - t_b)) a once it ends, Gamma(t) a before.
    """
    phis, gammas = cw_discrete(N, ts)
    _, after = cw_discrete(N, np.maximum(ts - burn_time, 0.0))
    offset = gammas @ accel - np.where((ts > burn_time)[:, None], after @ accel, 0.0)
    return np.einsum("tij,sj->sti", phis, states) + offset


def cam_cover(lo, hi, accel, contains=declared_everywhere, max_boxes=4, depth=4):
    """Cover of [lo, hi] under `accel` (Hill frame, m/s^2) held for BURN, over CAM_DURATION."""
    return drift_cover(
        N, lo, hi, R, CAM_DURATION, 801, contains, max_boxes, depth, accel=accel, burn_time=BURN
    )


def dense_cam_margins(states, accel, duration, extra=()):
    """Lowest keep-out margin of each state's CAM arc at 1 s steps and the times `extra`, m."""
    ts = np.concatenate([np.arange(0.0, duration, 1.0), [duration], extra])
    ranges = np.linalg.norm(cam_arcs(states, accel, ts)[..., :3], axis=-1)
    return ranges.min(axis=1) - R


def assert_same_cover(a, b):
    assert a.certified == b.certified
    assert a.excluded == b.excluded
    assert a.undecided == b.undecided
    assert a.boxes == b.boxes
    np.testing.assert_array_equal(a.certified_boxes, b.certified_boxes)
    np.testing.assert_array_equal(a.undecided_boxes, b.undecided_boxes)
    assert len(a.counterexamples) == len(b.counterexamples)
    for (state_a, margin_a, t_a), (state_b, margin_b, t_b) in zip(
        a.counterexamples, b.counterexamples, strict=True
    ):
        np.testing.assert_array_equal(state_a, state_b)
        assert margin_a == margin_b
        assert t_a == t_b


def split_velocity_box():
    lo, hi = velocity_box(1.4160e-3)
    return (N, lo, hi, R, TWO_ORBITS, 801, declared_everywhere, 4, 4)


def counterexample_45_m():
    x = closed_drift(47.0, 100.0)
    return (N, x - TINY, x + TINY, R, AUDIT, 3611, passive_set(45.0), 16, 4)


def halved_box():
    half = np.array([0.1, 0.0, 0.0, 0.0, 2e-3, 0.0])
    return (N, HOLD - half, HOLD + half, R, TWO_ORBITS, 801, declared_everywhere, 1, 4)


@pytest.mark.parametrize("case", [split_velocity_box, counterexample_45_m, halved_box])
@pytest.mark.parametrize(
    "burn",
    [
        {"accel": radial(1e-3), "burn_time": 0.0},
        # D2: a burn changes only the centres, by Gamma terms that vanish for a zero acceleration.
        {"accel": np.zeros(3), "burn_time": 600.0},
    ],
)
def test_zero_burn_gives_the_free_drift_cover_to_the_bit(case, burn):
    args = case()
    assert_same_cover(drift_cover(*args, **burn), drift_cover(*args))


def test_inward_cam_from_the_hold_reaches_the_closed_form_range_half_an_orbit_after_mid_burn():
    # 5e-4 m/s^2 for 30 s: D = 72.894 m, 1.5 b = 20.33 m, least range 45.790 m.
    cover = cam_cover(HOLD, HOLD, radial(-5e-4))
    assert cover.excluded == 1.0
    ((state, margin, t),) = cover.counterexamples
    np.testing.assert_array_equal(state, HOLD)
    assert margin == pytest.approx(cam_closest_range(5e-4) - R, abs=1e-6)
    assert t == pytest.approx(T_CLOSEST, rel=1e-12)


def test_box_at_the_hold_under_a_3e_4_inward_cam_is_certified_whole():
    # Least range of the center's arc 67.474 m; the 0.5 m widths in y and z move it by under 1 m.
    cover = cam_cover(HOLD - YZ_BOX, HOLD + YZ_BOX, radial(-3e-4))
    assert cover.boxes == 1
    assert cover.certified == 1.0
    assert cover.counterexamples == ()


def test_box_at_the_hold_under_a_5e_4_inward_cam_is_excluded_with_its_center_entering():
    cover = cam_cover(HOLD - YZ_BOX, HOLD + YZ_BOX, radial(-5e-4))
    assert cover.boxes == 1
    assert cover.excluded == 1.0
    ((state, margin, t),) = cover.counterexamples
    np.testing.assert_array_equal(state, HOLD)
    assert margin == pytest.approx(cam_closest_range(5e-4) - R, abs=1e-6)
    assert t == pytest.approx(T_CLOSEST, rel=1e-12)


def test_outward_cam_of_the_same_size_keeps_the_hold_box_clear():
    # Outward thrust drives the hold backwards along V-bar, so the range never drops below 100 m;
    # a cover that flips the burn's sign sees the 45.79 m approach instead.
    cover = cam_cover(HOLD - YZ_BOX, HOLD + YZ_BOX, radial(5e-4))
    assert cover.certified == 1.0
    assert cover.counterexamples == ()


def test_cam_into_the_keep_out_gives_a_counterexample_the_declared_set_holds():
    contains = passive_set(R)
    assert contains(HOLD)
    args = (N, HOLD - TINY, HOLD + TINY, R, CAM_DURATION, 801, contains, 16, 4)
    assert drift_cover(*args).counterexamples == ()
    cover = drift_cover(*args, accel=radial(-5e-4), burn_time=BURN)
    assert cover.counterexamples
    for state, margin, t in cover.counterexamples:
        assert contains(state)
        assert np.all(HOLD - TINY <= state) and np.all(state <= HOLD + TINY)
        assert margin == pytest.approx(cam_closest_range(5e-4) - R, abs=1e-2)
        assert t == pytest.approx(T_CLOSEST, abs=10.0)


def test_certified_boxes_under_a_cam_fly_clear_and_counterexamples_enter():
    # The center's arc clears by 0.3 m; 0.6 m along y and 2e-5 m/s along-track move the least
    # range by up to about 0.8 m, so the box straddles the keep-out.
    # The least range falls linearly with the CAM's size.
    accel = radial(-(100.0 - R - 0.3) / (100.0 - cam_closest_range(1.0)))
    half = np.array([0.0, 0.6, 0.0, 0.0, 2e-5, 0.0])
    lo, hi = HOLD - half, HOLD + half
    cover = cam_cover(lo, hi, accel, max_boxes=16, depth=8)
    assert len(cover.certified_boxes) >= 1
    rng = np.random.default_rng(29)
    for box_lo, box_hi in cover.certified_boxes[:4]:
        drawn = rng.uniform(box_lo, box_hi, size=(8, 6))
        assert np.all(dense_cam_margins(drawn, accel, CAM_DURATION) >= 0.0)
    assert len(cover.counterexamples) >= 1
    for state, margin, t in cover.counterexamples:
        assert np.all(lo <= state) and np.all(state <= hi)
        at_t = np.linalg.norm(cam_arcs(state[None], accel, np.array([t]))[0, 0, :3]) - R
        assert at_t == pytest.approx(margin, abs=1e-9)
        assert margin < -DEPTH


def test_pass_that_dips_in_between_samples_after_a_burn_inside_the_interval_is_not_certified():
    # 1.7 m/s^2 along V-bar for 2 s carries the state about 200 m in the 60 s between the two
    # samples; the mean of the end speeds, 0 and 3.4 m/s, allows only 102 m. The pass at
    # x = 46.7 m dips about 67 mm into the keep-out near 29.5 s, with both samples over 60 m clear.
    x = np.array([46.7, -100.0, 0.0, 0.0, 0.0, 0.0])
    accel = np.array([0.0, 1.7, 0.0])
    ts = np.linspace(0.0, 60.0, 6001)
    ranges = np.linalg.norm(cam_arcs(x[None], accel, ts, 2.0)[0, :, :3], axis=-1)
    assert ranges.min() - R < -0.06
    assert ranges[[0, -1]].min() - R > 60.0
    cover = drift_cover(N, x, x, R, 60.0, 2, declared_everywhere, 1, 0, accel=accel, burn_time=2.0)
    assert cover.certified == 0.0


@pytest.mark.parametrize(
    "burn",
    [
        {"accel": radial(-5e-4), "burn_time": -1.0},
        {"burn_time": BURN},
        {"accel": radial(-5e-4), "burn_time": math.nan},
        {"accel": radial(-5e-4), "burn_time": math.inf},
        {"accel": np.zeros(2), "burn_time": BURN},
        {"accel": np.array([math.nan, 0.0, 0.0]), "burn_time": BURN},
        {"accel": np.array([math.inf, 0.0, 0.0]), "burn_time": BURN},
    ],
)
def test_cover_refuses_bad_burns(burn):
    with pytest.raises(ValueError):
        drift_cover(N, HOLD, HOLD, R, CAM_DURATION, 801, declared_everywhere, 4, 4, **burn)
