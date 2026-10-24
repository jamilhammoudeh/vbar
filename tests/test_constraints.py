from decimal import Decimal, localcontext
from functools import cache

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from vbar.constraints import (
    ClosingSpeedLimit,
    KeepOut,
    interval_bounds,
    range_lower_bounds,
    segment_minima,
    segment_minimum,
    violation_intervals,
)
from vbar.dynamics import (
    EARTH,
    CWDynamics,
    TwoBodyDynamics,
    cw_discrete,
    cw_rhs,
    cw_stm,
    mean_motion,
)

N = mean_motion(EARTH.radius + 500e3)
KEEP_OUT = KeepOut(50.0)


def closest_approach(state, accel, duration):
    """Minimum range over a segment from an independent reference.

    The closed-form STM when coasting, DOP853 at rtol 1e-13 when thrusting; the minimum is where
    r . v changes sign.
    """
    if np.any(accel):
        sol = solve_ivp(
            cw_rhs,
            (0.0, duration),
            state,
            method="DOP853",
            args=(N, accel),
            rtol=1e-13,
            atol=1e-14,
            dense_output=True,
        )

    def at(t):
        return sol.sol(t) if np.any(accel) else cw_stm(N, t) @ state

    ts = np.linspace(0.0, duration, 2001)
    ranges = [np.linalg.norm(at(t)[:3]) for t in ts]
    k = int(np.argmin(ranges))
    if k in (0, len(ts) - 1):
        return ranges[k]
    t = brentq(lambda t: at(t)[:3] @ at(t)[3:], ts[k - 1], ts[k + 1], xtol=1e-12)
    return np.linalg.norm(at(t)[:3])


def detected_minimum(state, accel, duration, samples=20):
    m = CWDynamics(N)
    m.reset(state)
    _, sampled = m.advance(accel, duration, n_samples=samples)
    times = np.linspace(0.0, duration, samples)
    thrust = float(np.linalg.norm(accel))
    found = segment_minimum(KEEP_OUT, m.state_in_last_segment, times, sampled, thrust, N)
    return found, float(KEEP_OUT.margin(sampled).min())


def test_keep_out_margin_is_range_minus_radius():
    states = np.array([[30.0, 40.0, 0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]])
    assert np.allclose(KEEP_OUT.margin(states), [0.0, -40.0])


def test_fast_flyby_minimum_is_found_between_samples():
    state = np.array([0.0, 60.0, 5.0, 0.0, -12.0, 0.0])
    found, sampled = detected_minimum(state, np.zeros(3), 10.0)
    exact = closest_approach(state, np.zeros(3), 10.0) - 50.0
    assert found.margin == pytest.approx(exact, abs=1e-3)
    # The best of 20 samples misses the passage by most of a metre.
    assert sampled - exact > 0.1


@pytest.mark.parametrize("seed", range(20))
def test_minimum_range_matches_the_exact_cw_trajectory_to_a_millimetre(seed):
    rng = np.random.default_rng(seed)
    position = rng.normal(size=3)
    position *= rng.uniform(20.0, 200.0) / np.linalg.norm(position)
    speed = rng.uniform(0.5, 20.0)
    direction = -position / np.linalg.norm(position) + rng.normal(scale=0.3, size=3)
    state = np.r_[position, speed * direction / np.linalg.norm(direction)]
    accel = rng.uniform(-0.02, 0.02, 3) if seed % 2 else np.zeros(3)
    duration = rng.uniform(5.0, 30.0)
    found, _ = detected_minimum(state, accel, duration)
    assert found.margin == pytest.approx(closest_approach(state, accel, duration) - 50.0, abs=1e-3)


def test_minimum_at_a_segment_end_is_the_end_sample():
    state = np.array([0.0, 300.0, 0.0, 0.0, -1.0, 0.0])
    found, sampled = detected_minimum(state, np.zeros(3), 10.0)
    assert found.margin == sampled
    assert found.t == pytest.approx(10.0)


def random_flyby(seed):
    rng = np.random.default_rng(seed)
    position = rng.normal(size=3)
    position *= rng.uniform(20.0, 200.0) / np.linalg.norm(position)
    direction = -position / np.linalg.norm(position) + rng.normal(scale=0.3, size=3)
    state = np.r_[position, rng.uniform(0.5, 20.0) * direction / np.linalg.norm(direction)]
    accel = rng.uniform(-0.02, 0.02, 3) if seed % 2 else np.zeros(3)
    return state, accel, rng.uniform(5.0, 30.0)


@pytest.mark.parametrize("seed", range(20))
def test_certified_bound_lies_below_the_exact_minimum_within_l_h_over_2(seed):
    state, accel, duration = random_flyby(seed)
    found, _ = detected_minimum(state, accel, duration)
    exact = closest_approach(state, accel, duration) - 50.0
    # 1e-6 m below: the closed-form reference and the model's propagation differ by a few 1e-9 m,
    # and the second-order bound is tight enough to see that.
    assert found.lower_bound <= exact + 1e-6
    assert exact <= found.margin + 1e-9
    # Never looser than the plain form min_i m_i - L h / 2 with the largest interval L.
    m = CWDynamics(N)
    m.reset(state)
    _, sampled = m.advance(accel, duration, n_samples=20)
    h = duration / 19
    speed, _ = interval_bounds(sampled, h, float(np.linalg.norm(accel)), N)
    assert found.lower_bound >= KEEP_OUT.margin(sampled).min() - speed.max() * h / 2 - 1e-12


def test_bound_flags_a_dip_between_samples_and_refinement_finds_it():
    # A 12 m/s pass 5 m from the target, sampled only at the ends of a 10 s segment.
    state = np.array([0.0, 60.0, 5.0, 0.0, -12.0, 0.0])
    found, sampled = detected_minimum(state, np.zeros(3), 10.0, samples=2)
    assert sampled > 0.0 and found.lower_bound < 0.0
    assert found.margin == pytest.approx(
        closest_approach(state, np.zeros(3), 10.0) - 50.0, abs=1e-3
    )


def dense_states(m, state, accel, duration, samples):
    """`samples` evenly spaced states of the segment `m` flies from `state`."""
    # A fresh model flown alike repeats the integration; `advance` then evaluates every time in
    # one call, where `state_in_last_segment` takes one at a time.
    m.reset(state)
    return m.advance(accel, duration, n_samples=samples)[1]


@pytest.mark.parametrize("seed", range(6))
def test_speed_bounds_and_certificate_hold_on_two_body_motion(seed):
    rng = np.random.default_rng(seed)

    def model():
        return TwoBodyDynamics(inclination=0.9, j2=bool(seed % 2))

    m = model()
    position = rng.normal(size=3)
    position *= rng.uniform(100.0, 5000.0) / np.linalg.norm(position)
    state = np.r_[position, rng.uniform(-5.0, 5.0, 3)]
    accel = rng.uniform(-0.02, 0.02, 3)
    m.reset(state)
    duration = 60.0
    _, sampled = m.advance(accel, duration, n_samples=20)
    times = np.linspace(0.0, duration, 20)
    found = segment_minimum(
        KEEP_OUT, m.state_in_last_segment, times, sampled, 0.02 * np.sqrt(3), m.n
    )
    dense = dense_states(model(), state, accel, duration, 3801)
    assert found.lower_bound <= KEEP_OUT.margin(dense).min() <= found.margin + 1e-6
    speed, _ = interval_bounds(sampled, times[1], 0.02 * np.sqrt(3), m.n)
    dense_speed = np.linalg.norm(dense[:, 3:], axis=1)
    for i in range(19):
        assert dense_speed[200 * i : 200 * (i + 1) + 1].max() <= speed[i]


def test_coarse_sampling_gives_no_finite_bound_rather_than_a_wrong_one():
    # 1000 s between samples is past the 350 s at which the speed bound stops closing at 500 km;
    # the bound used to come out above the true minimum here (212.4 m against 208.3 m).
    start = [0.0, 400.0, 0.0, 0.3, 0.0, 0.0]
    m = CWDynamics(N)
    m.reset(start)
    _, sampled = m.advance(np.zeros(3), 2000.0, n_samples=3)
    times = np.linspace(0.0, 2000.0, 3)
    speed, accel = interval_bounds(sampled, times[1], 0.0, N)
    assert np.all(np.isinf(speed)) and np.all(np.isinf(accel))
    found = segment_minimum(KEEP_OUT, m.state_in_last_segment, times, sampled, 0.0, N)
    dense = dense_states(CWDynamics(N), start, np.zeros(3), 2000.0, 20001)
    exact = KEEP_OUT.margin(dense).min()
    assert found.lower_bound == -np.inf
    assert found.margin == pytest.approx(exact, abs=1e-3)


def test_flat_margin_is_not_refined_past_the_resolution():
    # On the V-bar the range is constant; every interval's bound is within 1 mm of the samples.
    found, sampled = detected_minimum(np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), 10.0)
    assert found.margin == sampled
    assert 0.0 < found.margin - found.lower_bound < KEEP_OUT.resolution


@pytest.mark.parametrize("seed", range(10))
def test_range_bound_lies_below_the_dense_range_on_every_interval(seed):
    state, accel, duration = random_flyby(seed)
    m = CWDynamics(N)
    m.reset(state)
    _, sampled = m.advance(accel, duration, n_samples=20)
    h = duration / 19
    speed, _ = interval_bounds(sampled, h, float(np.linalg.norm(accel)), N)
    lower = range_lower_bounds(sampled, h, speed)
    dense = np.array([m.state_in_last_segment(t) for t in np.linspace(0.0, duration, 1901)])
    dense_range = np.linalg.norm(dense[:, :3], axis=1)
    for i in range(19):
        assert dense_range[100 * i : 100 * (i + 1) + 1].min() >= lower[i]


def test_range_bound_is_zero_when_the_speed_bound_is_infinite():
    states = np.array([[0.0, 400.0, 0.0, 0.3, 0.0, 0.0], [0.0, 500.0, 0.0, 0.3, 0.0, 0.0]])
    assert range_lower_bounds(states, 1000.0, np.array([np.inf]))[0] == 0.0


def test_bounds_on_stacked_trajectories_equal_the_bounds_on_each():
    trajectories = []
    for seed in range(4):
        state, _, _ = random_flyby(seed)
        m = CWDynamics(N)
        m.reset(state)
        trajectories.append(m.advance(np.zeros(3), 10.0, n_samples=20)[1])
    stacked = np.array(trajectories)
    thrust = np.array([[0.0], [0.01], [0.02], [0.03]])
    speed, accel = interval_bounds(stacked, 10.0 / 19, thrust, N)
    lower = range_lower_bounds(stacked, 10.0 / 19, speed)
    for k, samples in enumerate(trajectories):
        one_speed, one_accel = interval_bounds(samples, 10.0 / 19, float(thrust[k, 0]), N)
        assert np.array_equal(speed[k], one_speed) and np.array_equal(accel[k], one_accel)
        assert np.array_equal(lower[k], range_lower_bounds(samples, 10.0 / 19, one_speed))


def exact_crossings(state, duration):
    """Times a coasting CW trajectory crosses the 50 m sphere, from the closed-form STM."""
    ts = np.linspace(0.0, duration, 4001)
    margins = [np.linalg.norm((cw_stm(N, t) @ state)[:3]) - 50.0 for t in ts]
    return [
        brentq(lambda t: np.linalg.norm((cw_stm(N, t) @ state)[:3]) - 50.0, ts[i], ts[i + 1])
        for i in range(len(ts) - 1)
        if (margins[i] < 0.0) != (margins[i + 1] < 0.0)
    ]


def detected_violations(state, duration, samples=20):
    m = CWDynamics(N)
    m.reset(state)
    _, sampled = m.advance(np.zeros(3), duration, n_samples=samples)
    times = np.linspace(0.0, duration, samples)
    found = segment_minimum(KEEP_OUT, m.state_in_last_segment, times, sampled, 0.0, N, 100.0)
    return violation_intervals(KEEP_OUT, m.state_in_last_segment, times, sampled, found, 100.0)


def test_violation_interval_is_bounded_by_the_exact_entry_and_exit():
    state = np.array([3.0, 100.0, -2.0, 0.0, -10.0, 0.0])
    ((start, end),) = detected_violations(state, 20.0)
    entry, exit_ = exact_crossings(state, 20.0)
    assert start - 100.0 == pytest.approx(entry, abs=1e-8)
    assert end - 100.0 == pytest.approx(exit_, abs=1e-8)


def test_violation_between_two_samples_is_found_from_the_refined_minimum():
    state = np.array([0.0, 60.0, 5.0, 0.0, -12.0, 0.0])
    ((start, end),) = detected_violations(state, 10.0, samples=2)
    entry, exit_ = exact_crossings(state, 10.0)
    assert (start - 100.0, end - 100.0) == pytest.approx((entry, exit_), abs=1e-8)


def test_violation_open_at_either_end_of_the_segment_runs_to_that_end():
    leaving = np.array([0.0, 10.0, 0.0, 0.0, 10.0, 0.0])
    ((start, end),) = detected_violations(leaving, 10.0)
    assert start == 100.0
    assert end - 100.0 == pytest.approx(exact_crossings(leaving, 10.0)[0], abs=1e-8)
    staying = np.array([0.0, 20.0, 0.0, 0.0, 0.1, 0.0])
    assert detected_violations(staying, 10.0) == ((100.0, 110.0),)
    clear = np.array([0.0, 300.0, 0.0, 0.0, -1.0, 0.0])
    assert detected_violations(clear, 10.0) == ()


def batched(states, accels, duration, samples, constraint=KEEP_OUT):
    """The batched minimum over CW segments that all start at 100 s."""
    phis, gammas = cw_discrete(N, np.linspace(0.0, duration, samples))
    sampled = np.einsum("mij,nj->nmi", phis, states) + np.einsum("mij,nj->nmi", gammas, accels)

    def state_at(rows, tau):
        phi, gamma = cw_discrete(N, tau)
        return np.einsum("aij,aj->ai", phi, states[rows]) + np.einsum(
            "aij,aj->ai", gamma, accels[rows]
        )

    times = np.linspace(0.0, duration, samples)
    thrust = np.linalg.norm(accels, axis=1)
    t0 = np.full(len(states), 100.0)
    return segment_minima(constraint, state_at, times, sampled, thrust, N, t0)


@pytest.mark.parametrize("samples", [2, 20])
def test_batched_minimum_agrees_with_the_scalar_one(samples):
    # The scalar minimum is checked against the exact trajectory above; this checks agreement.
    flybys = [random_flyby(seed) for seed in range(40)]
    states = np.array([f[0] for f in flybys])
    accels = np.array([f[1] for f in flybys])
    _, margin, bound = batched(states, accels, 10.0, samples)
    for k, (state, accel, _) in enumerate(flybys):
        found, _ = detected_minimum(state, accel, 10.0, samples)
        # Golden section to 1e-6 s and Brent to 1e-5 s land within 1e-7 m of the same dip.
        assert margin[k] == pytest.approx(found.margin, abs=1e-7)
        assert bound[k] == pytest.approx(found.lower_bound, rel=1e-12, abs=1e-12)


def test_batched_minimum_uses_the_range_bound_as_the_scalar_one_does():
    # A closing-speed limit's rate bound needs a positive lower bound on the range; without it
    # the bound is infinite and the batched certificate collapses to -inf.
    limit = ClosingSpeedLimit(0.2, 4.0 * N)
    flybys = [random_flyby(seed) for seed in range(20)]
    states = np.array([f[0] for f in flybys])
    accels = np.array([f[1] for f in flybys])
    _, _, bound = batched(states, accels, 10.0, 20, constraint=limit)
    for k, (state, accel, _) in enumerate(flybys):
        m = CWDynamics(N)
        m.reset(state)
        _, sampled = m.advance(accel, 10.0, n_samples=20)
        found = segment_minimum(
            limit,
            m.state_in_last_segment,
            np.linspace(0.0, 10.0, 20),
            sampled,
            float(np.linalg.norm(accel)),
            N,
            100.0,
        )
        assert np.isfinite(bound[k])
        assert bound[k] == pytest.approx(found.lower_bound, rel=1e-12, abs=1e-12)


def flyby(rng, coast, duration=None):
    """A pass aimed near the target from 20 to 200 m at 0.5 to 20 m/s, 5 to 30 s unless given."""
    position = rng.normal(size=3)
    position *= rng.uniform(20.0, 200.0) / np.linalg.norm(position)
    direction = -position / np.linalg.norm(position) + rng.normal(scale=0.3, size=3)
    state = np.r_[position, rng.uniform(0.5, 20.0) * direction / np.linalg.norm(direction)]
    accel = np.zeros(3) if coast else rng.uniform(-0.02, 0.02, 3)
    return state, accel, duration if duration is not None else rng.uniform(5.0, 30.0)


@cache
def flyby_tables():
    rng = np.random.default_rng(7)
    dense = [(*flyby(rng, True, 10.0), 20) for _ in range(200)]
    thrusting = [(*flyby(rng, k % 2 == 0), 20) for k in range(200)]
    coarse = [(*flyby(rng, True, 10.0), 5) for _ in range(200)]
    return {"dense": dense, "thrusting": thrusting, "coarse": coarse}


@cache
def random_segments():
    """CW segments up to 60 s at up to 20 m/s, half passing within 5 m of the target."""
    rng = np.random.default_rng(11)
    segments = []
    for k in range(320):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        w = rng.normal(size=3)
        w -= (w @ u) * u
        miss = rng.uniform(0.0, 5.0) if k % 2 else rng.uniform(5.0, 150.0)
        speed, duration = rng.uniform(0.5, 20.0), rng.uniform(1.0, 60.0)
        position = miss * w / np.linalg.norm(w) - u * speed * rng.uniform(-0.2, 1.2) * duration
        accel = rng.uniform(-0.02, 0.02, 3) if k % 3 == 0 else np.zeros(3)
        samples = int(rng.choice([2, 3, 4, 5, 6, 10, 20]))
        segments.append((np.r_[position, speed * u], accel, duration, samples))
    return segments


FLYBY_TABLES = ("dense", "thrusting", "coarse")
ALL_TABLES = (*FLYBY_TABLES, "random")


def cases(table):
    return random_segments() if table == "random" else flyby_tables()[table]


@cache
def exact_margins(table):
    """Exact minimum keep-out margin of each segment from the closed-form CW solution.

    A 2001-point grid, then four 201-point grids each spanning the two intervals around the lowest
    point; the last spacing is below 1e-11 s.
    """
    margins = []
    rows = cases(table)
    for start in range(0, len(rows), 25):
        chunk = rows[start : start + 25]
        states = np.array([c[0] for c in chunk])
        accels = np.array([c[1] for c in chunk])
        lo, hi = np.zeros(len(chunk)), np.array([c[2] for c in chunk])
        best = np.full(len(chunk), np.inf)
        index = np.arange(len(chunk))
        for points in (2001, 201, 201, 201, 201):
            t = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, points)
            phi, gamma = cw_discrete(N, t)
            r = np.einsum("cpij,cj->cpi", phi[..., :3, :], states)
            r += np.einsum("cpij,cj->cpi", gamma[..., :3, :], accels)
            rho = np.linalg.norm(r, axis=-1)
            k = np.argmin(rho, axis=1)
            best = np.minimum(best, rho[index, k])
            lo, hi = t[index, np.maximum(k - 1, 0)], t[index, np.minimum(k + 1, points - 1)]
        margins.append(best - KEEP_OUT.radius)
    return np.concatenate(margins)


class FirstOrderKeepOut:
    """The keep-out constraint without its second-order bound."""

    name = "keep_out"
    resolution = KeepOut.resolution

    def __init__(self, radius):
        self.keep_out = KeepOut(radius)

    def margin(self, state, t=0.0):
        return self.keep_out.margin(state, t)

    def rate_bound(self, speed, accel, rho_min=0.0):
        return self.keep_out.rate_bound(speed, accel, rho_min)


def certified(table, constraint=KEEP_OUT, count=None):
    """Lower bound from `segment_minimum` on each segment, or the first `count`, by the CW model."""
    found = []
    for state, accel, duration, samples in cases(table)[:count]:
        m = CWDynamics(N)
        m.reset(state)
        _, sampled = m.advance(accel, duration, n_samples=samples)
        times = np.linspace(0.0, duration, samples)
        thrust = float(np.linalg.norm(accel))
        found.append(
            segment_minimum(constraint, m.state_in_last_segment, times, sampled, thrust, N)
        )
    return np.array([f.lower_bound for f in found])


@cache
def lower_bounds(table):
    return certified(table)


# The first-order and loosened bounds refine many more intervals than the certified one, so
# their claims run on the first segments of each table: 100 hold 6 random segments where the
# first-order bound is the larger, and the dense segment where a doubled curvature bound leaves
# its 1.90 m gap. The full tables run in benchmarks/keep_out_bound_tables.py.
PREFIX = 100


@cache
def first_order_lower_bounds(table, count=PREFIX):
    return certified(table, FirstOrderKeepOut(KEEP_OUT.radius), count)


def check_first_order_bound_misses_the_tightness_table(count):
    gap = exact_margins("dense")[:count] - first_order_lower_bounds("dense", count)
    assert np.median(gap) > 0.1


def check_certified_bound_is_never_below_the_first_order_bound(table, count):
    first_order = first_order_lower_bounds(table, count)
    assert np.all(lower_bounds(table)[:count] >= first_order - 1e-12)


def check_loosened_bound_is_sound_and_outside_the_tightness_table(count):
    bounds = {t: certified(t, count=count) for t in FLYBY_TABLES}
    assert all(np.all(bounds[t] <= exact_margins(t)[:count] + 1e-6) for t in FLYBY_TABLES)
    assert (exact_margins("dense")[:count] - bounds["dense"]).max() > 0.03


def test_curvature_bound_is_twice_speed_squared_plus_extent_times_acceleration():
    assert KEEP_OUT.curvature_bound(3.0, 0.01, 100.0) == pytest.approx(20.0, rel=1e-15)
    curvature = KEEP_OUT.curvature_bound(
        np.array([0.0, 3.0]), np.array([0.02, 0.0]), np.array([50.0, 10.0])
    )
    assert np.allclose(curvature, [2.0, 18.0], rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("table", ALL_TABLES)
def test_certified_bound_lies_below_the_exact_minimum_on_every_segment(table):
    # 1e-6 m: the closed form and the model's propagation differ by about 1e-9 m.
    assert np.all(lower_bounds(table) <= exact_margins(table) + 1e-6)


# Ceilings about 1.5 times the gaps of the second-order bound alone on these flybys (median
# 4.9e-5 m and maximum 0.019 m dense, maximum 0.143 m thrusting, 1.24 m coarse); the first-order
# bound alone leaves 1.76 m median and 5.02 m maximum on the dense table.
@pytest.mark.parametrize(
    ("table", "median", "maximum"),
    [("dense", 1e-4, 0.03), ("thrusting", np.inf, 0.2), ("coarse", np.inf, 1.5)],
)
def test_certified_bound_is_within_the_tightness_table_of_the_exact_minimum(table, median, maximum):
    gap = exact_margins(table) - lower_bounds(table)
    assert np.median(gap) <= median
    assert gap.max() <= maximum


def test_first_order_bound_alone_misses_the_tightness_table():
    check_first_order_bound_misses_the_tightness_table(PREFIX)


@pytest.mark.parametrize("table", ALL_TABLES)
def test_certified_bound_is_never_below_the_first_order_bound(table):
    check_certified_bound_is_never_below_the_first_order_bound(table, PREFIX)


@pytest.mark.parametrize("samples", [2, 20])
def test_certified_bound_is_exact_on_a_straight_line_pass(samples):
    # With n = 0 and no thrust, f = |r|^2 is quadratic in time with f'' = 2 |v|^2, and the speed
    # bound is the constant speed exactly, so the curvature bound equals f'' and the chord minus
    # M s (h - s) / 2 is f itself; its minimum is the closest approach, 5 m at t = 5 s.
    r0, v = np.array([0.0, 60.0, 5.0]), np.array([0.0, -12.0, 0.0])

    def state_at(tau):
        return np.r_[r0 + v * tau, v]

    times = np.linspace(0.0, 10.0, samples)
    states = np.array([state_at(t) for t in times])
    found = segment_minimum(KEEP_OUT, state_at, times, states, 0.0, 0.0)
    assert found.lower_bound == pytest.approx(5.0 - KEEP_OUT.radius, abs=1e-9)


def test_second_order_bound_is_minus_infinity_on_coarse_samples():
    # 1000 s between samples puts the speed and acceleration bounds out of their regime.
    m = CWDynamics(N)
    m.reset([0.0, 400.0, 0.0, 0.3, 0.0, 0.0])
    _, sampled = m.advance(np.zeros(3), 2000.0, n_samples=3)
    times = np.linspace(0.0, 2000.0, 3)
    speed, accel = interval_bounds(sampled, times[1], 0.0, N)
    assert np.all(KEEP_OUT.second_order_bounds(sampled, times[1], speed, accel) == -np.inf)
    found = segment_minimum(KEEP_OUT, m.state_in_last_segment, times, sampled, 0.0, N)
    assert found.lower_bound == -np.inf


def test_second_order_bound_is_minus_infinity_on_each_interval_with_an_infinite_bound():
    states = np.array([[0.0, 400.0 + 10.0 * k, 0.0, 0.0, 10.0, 0.0] for k in range(4)])
    speed = np.array([np.inf, 10.0, 10.0])
    accel = np.array([1e-4, np.inf, 1e-4])
    bounds = KEEP_OUT.second_order_bounds(states, 1.0, speed, accel)
    assert bounds[0] == -np.inf and bounds[1] == -np.inf
    assert 349.0 < bounds[2] <= 420.0 - KEEP_OUT.radius


def test_second_order_bound_without_motion_is_the_margin_and_divides_by_nothing():
    states = np.array([[0.0, 400.0, 0.0, 0.0, 0.0, 0.0]] * 2)
    with np.errstate(all="raise"):
        bounds = KEEP_OUT.second_order_bounds(states, 1.0, np.zeros(1), np.zeros(1))
    assert np.array_equal(bounds, [350.0])


def test_second_order_bounds_on_stacked_trajectories_equal_the_bounds_on_each():
    trajectories = []
    for seed in range(4):
        state, accel, _ = random_flyby(seed)
        m = CWDynamics(N)
        m.reset(state)
        trajectories.append(m.advance(accel, 10.0, n_samples=20)[1])
    stacked = np.array(trajectories)
    thrust = np.array([[0.0], [0.01], [0.02], [0.03]])
    speed, accel = interval_bounds(stacked, 10.0 / 19, thrust, N)
    bounds = KEEP_OUT.second_order_bounds(stacked, 10.0 / 19, speed, accel)
    assert bounds.shape == (4, 19)
    for k, samples in enumerate(trajectories):
        one_speed, one_accel = interval_bounds(samples, 10.0 / 19, float(thrust[k, 0]), N)
        one = KEEP_OUT.second_order_bounds(samples, 10.0 / 19, one_speed, one_accel)
        assert np.array_equal(bounds[k], one)


def test_halved_curvature_bound_is_caught_by_the_soundness_check(monkeypatch):
    curvature_bound = KeepOut.curvature_bound
    monkeypatch.setattr(
        KeepOut, "curvature_bound", lambda self, *args: 0.5 * curvature_bound(self, *args)
    )
    assert any(np.any(certified(t) > exact_margins(t) + 1e-6) for t in FLYBY_TABLES)


def test_doubled_curvature_bound_stays_sound_and_is_caught_by_the_tightness_table(monkeypatch):
    curvature_bound = KeepOut.curvature_bound
    monkeypatch.setattr(
        KeepOut, "curvature_bound", lambda self, *args: 2.0 * curvature_bound(self, *args)
    )
    # The doubled bound leaves a largest dense gap of 1.90 m against the 0.03 m ceiling.
    check_loosened_bound_is_sound_and_outside_the_tightness_table(PREFIX)


def test_batched_minimum_reports_the_scalar_second_order_bound():
    flybys = cases("dense")
    states = np.array([f[0] for f in flybys])
    accels = np.array([f[1] for f in flybys])
    _, _, bound = batched(states, accels, 10.0, 20)
    assert np.allclose(bound, lower_bounds("dense"), rtol=0.0, atol=1e-9)


def decimal_sin_cos(x):
    """sin x and cos x from their Taylor series, in the current decimal context."""
    term, sin, cos, k = Decimal(1), Decimal(0), Decimal(1), 0
    while abs(term) > Decimal(10) ** -60:
        k += 1
        term = term * x / k
        if k % 4 == 1:
            sin += term
        elif k % 4 == 2:
            cos -= term
        elif k % 4 == 3:
            sin -= term
        else:
            cos += term
    return sin, cos


def exact_coasting_range(state, t):
    """Range (m) `t` s after `state` on the coasting CW trajectory, to 50 digits.

    The closed-form solution, taking the floats in `state` and N as exact.
    """
    with localcontext() as ctx:
        ctx.prec = 50
        x, y, z, vx, vy, vz = (Decimal(float(c)) for c in state)
        n, t = Decimal(float(N)), Decimal(t)
        s, c = decimal_sin_cos(n * t)
        rx = (4 - 3 * c) * x + s / n * vx + 2 * (1 - c) / n * vy
        ry = 6 * (s - n * t) * x + y - 2 * (1 - c) / n * vx + (4 * s - 3 * n * t) / n * vy
        rz = c * z + s / n * vz
        return (rx * rx + ry * ry + rz * rz).sqrt()


@cache
def radial_hop():
    """Start states of the 10 s steps of a radial hop and each step's exact minimum margin.

    The hop goes from 300 m to 60 m on V-bar, 4 vx / n, in pi / n s. The range falls through the
    whole hop, so each step's closest approach is at its last sample. The first step is left out:
    the range is stationary where the hop starts, and the curvature term there puts the bound
    2e-9 m below the end sample.
    """
    m = CWDynamics(N)
    state = np.array([0.0, 300.0, 0.0, 60.0 * N, 0.0, 0.0])
    starts = []
    for _ in range(int(np.pi / N / 10.0) - 1):
        m.reset(state)
        state, _ = m.advance(np.zeros(3), 10.0)
        starts.append(state)
    exact = [exact_coasting_range(s, 10.0) - Decimal(KEEP_OUT.radius) for s in starts]
    return np.array(starts), exact


def excess_over_exact(bounds, exact):
    return np.array([float(Decimal(float(b)) - e) for b, e in zip(bounds, exact, strict=True)])


def test_certified_bound_at_a_sample_minimum_never_exceeds_the_exact_minimum():
    # There the second-order bound is the sample's own margin, which rounding in the propagated
    # state put up to about 1e-13 m above the exact range on half of these steps.
    starts, exact = radial_hop()
    times = np.linspace(0.0, 10.0, 11)
    bounds = []
    for state in starts:
        m = CWDynamics(N)
        m.reset(state)
        _, sampled = m.advance(np.zeros(3), 10.0, n_samples=11)
        found = segment_minimum(KEEP_OUT, m.state_in_last_segment, times, sampled, 0.0, N)
        bounds.append(found.lower_bound)
    excess = excess_over_exact(bounds, exact)
    assert excess.max() <= 0.0
    # An allowance of a few ulp is enough; 1e-9 m catches one far larger than rounding needs.
    assert excess.min() >= -1e-9


def test_batched_bound_at_a_sample_minimum_never_exceeds_the_exact_minimum():
    starts, exact = radial_hop()
    _, _, bounds = batched(starts, np.zeros((len(starts), 3)), 10.0, 11)
    excess = excess_over_exact(bounds, exact)
    assert excess.max() <= 0.0
    assert excess.min() >= -1e-9
