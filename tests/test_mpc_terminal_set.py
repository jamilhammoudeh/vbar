"""Maximal constraint admissible sets of a stable closed loop, used as MPC terminal sets.

The set {z : |h_i F^t z| <= bound_i for all t >= 0} is computed by the algorithm of Gilbert and
Tan, "Linear systems with state and control constraints: the theory and application of maximal
output admissible sets", IEEE Transactions on Automatic Control 36(9), 1991, and checked against
Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed., Exercise 2.5. Rows are returned
stacked by time, h, h F, ..., h F^t*, each with the bound of its row of h. The CW cases are for a
500 km circular orbit of the WGS 84 Earth, a 10 s step and a 0.02 m/s^2 box limit per axis
unless stated.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

from vbar.agents import mpc
from vbar.agents.lqr import LQRConfig, dlqr, sampled_data_cost, steady_state_input
from vbar.agents.mpc import admissible_set
from vbar.constraints import interval_bounds
from vbar.dynamics import EARTH, cw_discrete, cw_system, mean_motion
from vbar.scenarios import make

N = mean_motion(EARTH.radius + 500e3)
A_MAX = 0.02  # m/s^2 per axis
EX25_F = np.array([[0.9, 1.0], [0.0, 0.09]])
EX25_H = np.array([[1.0, 1.0]])
ALONG_TRACK = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


def row_bounds(rows, bound):
    """Bound of each stacked row: the bounds of h repeated once per time."""
    bound = np.asarray(bound, dtype=float)
    return np.tile(bound, len(rows) // len(bound))


def discrete_weight_loop(dt):
    """Closed loop and gain of the LQR with Q = diag(1e-2 x3, 1e4 x3), R = 2500 I per step.

    These are 10 m, 0.01 m/s and 0.02 m/s^2 costing 1, summed once per step.
    """
    phi, gamma = cw_discrete(N, dt)
    k, _ = dlqr(phi, gamma, np.diag([1e-2] * 3 + [1e4] * 3), 2500.0 * np.eye(3))
    return phi - gamma @ k, k


def worst_image(rows, bound, f):
    """Largest |row_j F z| / bound_j over the set, one LP per row and sign."""
    b = row_bounds(rows, bound)
    a_ub = np.vstack([rows, -rows])
    b_ub = np.concatenate([b, b])
    worst = 0.0
    for row, bj in zip(rows @ f, b, strict=True):
        for sign in (1.0, -1.0):
            lp = linprog(-sign * row, A_ub=a_ub, b_ub=b_ub, bounds=[(None, None)] * f.shape[0])
            assert lp.status == 0, "the set is unbounded along an image row"
            worst = max(worst, -lp.fun / bj)
    return worst


def test_exercise_2_5_set_is_determined_after_two_steps():
    rows, t_star = admissible_set(EX25_F, EX25_H, np.array([1.0]))
    assert t_star == 2
    expected = np.array([[1.0, 1.0], [0.9, 1.09], [0.81, 0.9981]])
    assert rows.shape == expected.shape
    np.testing.assert_allclose(rows, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(("f22", "t_star"), [(0.5, 4), (0.9, 12), (0.99, 30)])
def test_determination_index_grows_as_the_slow_pole_nears_one(f22, t_star):
    # Indices from an independent computation of the same algorithm.
    f = np.array([[0.9, 1.0], [0.0, f22]])
    rows, found = admissible_set(f, EX25_H, np.array([1.0]))
    assert found == t_star
    expected = np.vstack([EX25_H @ np.linalg.matrix_power(f, t) for t in range(t_star + 1)])
    np.testing.assert_allclose(rows, expected, rtol=1e-12, atol=1e-15)


def test_set_is_positively_invariant_under_the_closed_loop():
    for f22 in (0.09, 0.5):
        f = np.array([[0.9, 1.0], [0.0, f22]])
        rows, _ = admissible_set(f, EX25_H, np.array([1.0]))
        assert worst_image(rows, [1.0], f) <= 1.0 + 1e-9


@pytest.mark.parametrize("a", [0.1, 0.19, 0.43])
def test_row_whose_maximum_equals_its_bound_is_redundant(a):
    # Over the box |z_i| <= 0.02 the row (a, 1 - a) of h F peaks at exactly 0.02, so h alone is
    # the set (t* = 0). At these a the LP returns 0.02 plus a rounding error, so a test without
    # a relative tolerance adds the row (found by an independent computation).
    f = np.array([[a, 1.0 - a], [0.0, 0.5]])
    rows, t_star = admissible_set(f, np.eye(2), np.array([0.02, 0.02]))
    assert t_star == 0
    np.testing.assert_array_equal(rows, np.eye(2))


@pytest.mark.parametrize(("a", "degrees"), [(0.1, 30.0), (0.1, 45.0), (0.19, 15.0)])
def test_row_tied_with_its_bound_only_up_to_rounding_is_redundant(a, degrees):
    # The tie above in coordinates turned by an orthogonal Q: the loop Q' F Q and the output Q.
    # The set is the same square turned, so t* = 0, but the rows no longer divide by their
    # bounds to exact entries and the LP maximum lands 1 + O(1e-16) above the bound at these
    # angles (found by an independent computation).
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    q = np.array([[c, -s], [s, c]])
    f = q.T @ np.array([[a, 1.0 - a], [0.0, 0.5]]) @ q
    rows, t_star = admissible_set(f, q, np.array([1.0, 1.0]))
    assert t_star == 0
    np.testing.assert_array_equal(rows, q)


def test_row_unbounded_over_the_rows_so_far_is_kept():
    # Over |z1| <= 1 alone the row h F = (0.5, 0.5) has no maximum; the set needs it and
    # h F^2 = (0.25, 0.5), which peaks at 1.25 over the first two rows.
    f = np.array([[0.5, 0.5], [0.0, 0.5]])
    rows, t_star = admissible_set(f, np.array([[1.0, 0.0]]), np.array([1.0]))
    assert t_star == 2
    np.testing.assert_array_equal(rows, [[1.0, 0.0], [0.5, 0.5], [0.25, 0.5]])


def test_unobservable_state_leaves_the_set_unbounded_along_it():
    # z2 never reaches the output, so the set is |z1| <= 1 with z2 free and h alone describes it.
    rows, t_star = admissible_set(np.diag([0.5, 0.5]), np.array([[1.0, 0.0]]), np.array([1.0]))
    assert t_star == 0
    np.testing.assert_array_equal(rows, [[1.0, 0.0]])


def test_unobservable_state_appended_to_exercise_2_5_keeps_its_rows():
    # The set is the exercise's set times a free third coordinate.
    f = np.zeros((3, 3))
    f[:2, :2] = EX25_F
    f[2, 2] = 0.5
    rows, t_star = admissible_set(f, np.array([[1.0, 1.0, 0.0]]), np.array([1.0]))
    assert t_star == 2
    expected = [[1.0, 1.0, 0.0], [0.9, 1.09, 0.0], [0.81, 0.9981, 0.0]]
    np.testing.assert_allclose(rows, expected, rtol=1e-12, atol=0.0)


def test_pair_unobservable_only_up_to_rounding_is_refused():
    # The unobservable pair diag(0.5, 0.4), h = (1, 0) in coordinates z = T w. Rounding in
    # T^-1 F T and h T leaves each row h F^t a tiny component along the direction the set leaves
    # unbounded, so the LP keeps finding the next row unbounded and no exact certificate of a
    # finite description exists; the search runs into the cap (20 here for time: the default
    # 200 takes about 6 s).
    t = np.array([[1.0, 0.3], [0.2, 1.0]])
    f = np.linalg.solve(t, np.diag([0.5, 0.4]) @ t)
    with pytest.raises(ValueError, match="within t_max"):
        admissible_set(f, np.array([[1.0, 0.0]]) @ t, np.array([1.0]), t_max=20)


def test_certificate_bounds_the_maximum_by_absolute_multipliers():
    # Over the box |z_i| <= 1, z1 - z2 reaches 2 although its multipliers (1, -1) sum to 0, so
    # only the sum of their absolute values bounds it; (0.5, 0.5) reaches exactly 1.
    box = np.eye(2)
    limits = np.ones(2)
    assert not mpc._certified(np.array([1.0, -1.0]), 1.0, box, limits, np.array([1.0, -1.0]))
    assert mpc._certified(np.array([0.5, 0.5]), 1.0, box, limits, np.array([1.0, 1.0]))


@pytest.mark.parametrize("status", [1, 2, 4])
def test_failed_redundancy_lp_raises(monkeypatch, status):
    # Iteration limit, infeasible and numerical trouble are not answers about redundancy.
    def failing(*args, **kwargs):
        return SimpleNamespace(status=status, fun=None, x=None, message="stub failure")

    monkeypatch.setattr(mpc, "linprog", failing)
    with pytest.raises(RuntimeError):
        admissible_set(EX25_F, EX25_H, np.array([1.0]))


@pytest.mark.parametrize("h", [[[0.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]])
def test_output_row_of_zeros_is_refused(h):
    with pytest.raises(ValueError):
        admissible_set(EX25_F, np.array(h), np.ones(len(h)))


@pytest.mark.parametrize("scale", [1e-12, 1e12])
def test_set_does_not_depend_on_the_units_of_the_bound(scale):
    # An absolute LP tolerance of 1e-9 stops at t* = 1 when the bound is 1e-12.
    rows, t_star = admissible_set(EX25_F, EX25_H, np.array([scale]))
    assert t_star == 2
    np.testing.assert_allclose(rows, [[1.0, 1.0], [0.9, 1.09], [0.81, 0.9981]], rtol=1e-12)


def test_rows_follow_a_change_of_state_coordinates():
    # In coordinates w = T^-1 z the loop is T^-1 F T and the output h T; the rows map by T.
    t = np.random.default_rng(3).normal(size=(2, 2)) + 2.0 * np.eye(2)
    rows, t_star = admissible_set(EX25_F, EX25_H, np.array([1.0]))
    rows_w, t_star_w = admissible_set(np.linalg.solve(t, EX25_F @ t), EX25_H @ t, np.array([1.0]))
    assert t_star_w == t_star
    np.testing.assert_allclose(rows_w, rows @ t, rtol=0.0, atol=1e-12 * np.max(np.abs(rows @ t)))


def test_each_output_row_keeps_its_own_bound():
    # |z1 + z2| <= 1 and |z1| <= 0.5: t* = 2 from an independent computation; the bound 0.5 on
    # both rows would give t* = 1.
    h = np.array([[1.0, 1.0], [1.0, 0.0]])
    rows, t_star = admissible_set(EX25_F, h, np.array([1.0, 0.5]))
    assert t_star == 2
    expected = [[1.0, 1.0], [1.0, 0.0], [0.9, 1.09], [0.9, 1.0], [0.81, 0.9981], [0.81, 0.99]]
    np.testing.assert_allclose(rows, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("f22", [1.0, 1.0 - 1e-13, 1.05])
def test_closed_loop_without_decay_is_refused(f22):
    # At radius 1 - 1e-13 the index runs far beyond any cap, so without the radius check the
    # cap raises instead, with its own message.
    f = np.array([[0.9, 1.0], [0.0, f22]])
    with pytest.raises(ValueError, match="stable closed loop"):
        admissible_set(f, EX25_H, np.array([1.0]))


def test_free_drift_cw_step_is_refused():
    # The CW step has eigenvalue 1 (along-track drift), so no finite set of rows describes it.
    phi, _ = cw_discrete(N, 10.0)
    with pytest.raises(ValueError, match="stable closed loop"):
        admissible_set(phi, np.eye(6)[:3], np.full(3, 100.0))


def test_determination_index_above_the_cap_is_refused():
    # t* = 4 at F22 = 0.5 and 30 at 0.99.
    f = np.array([[0.9, 1.0], [0.0, 0.5]])
    with pytest.raises(ValueError, match="within t_max"):
        admissible_set(f, EX25_H, np.array([1.0]), t_max=3)
    assert admissible_set(f, EX25_H, np.array([1.0]), t_max=4)[1] == 4
    with pytest.raises(ValueError, match="within t_max"):
        admissible_set(np.array([[0.9, 1.0], [0.0, 0.99]]), EX25_H, np.array([1.0]), t_max=20)


@pytest.mark.parametrize(
    ("dt", "t_star", "offset_m"),
    [(1.0, 5, 24.154232), (10.0, 1, 201.509569), (60.0, 1, 1237.851968)],
)
def test_lqr_thrust_box_set_and_its_along_track_reach(dt, t_star, offset_m):
    # t* and the largest along-track offset at rest inside the set, from an independent
    # computation; at 10 s that is 201.5 m with six rows.
    f, k = discrete_weight_loop(dt)
    rows, found = admissible_set(f, k, np.full(3, A_MAX))
    assert found == t_star
    expected = np.vstack([k @ np.linalg.matrix_power(f, t) for t in range(t_star + 1)])
    assert rows.shape == expected.shape
    np.testing.assert_allclose(rows, expected, rtol=0.0, atol=1e-12 * np.max(np.abs(expected)))
    along = np.abs(rows @ ALONG_TRACK)
    reach = np.min(row_bounds(rows, np.full(3, A_MAX))[along > 0.0] / along[along > 0.0])
    assert reach == pytest.approx(offset_m, rel=1e-6)
    assert worst_image(rows, np.full(3, A_MAX), f) <= 1.0 + 1e-9


def scaled_loop(pos_scale_m, vel_scale_mps, accel_scale_mps2, dt=10.0):
    """Closed loop and gain of the LQR whose scales cost 1, summed once per step."""
    phi, gamma = cw_discrete(N, dt)
    q = np.diag([pos_scale_m**-2] * 3 + [vel_scale_mps**-2] * 3)
    k, _ = dlqr(phi, gamma, q, np.eye(3) / accel_scale_mps2**2)
    return phi - gamma @ k, k


def test_badly_scaled_gain_rows_give_a_set():
    # Position gains sit about 1e-8 below velocity gains here; an LP with presolve on these
    # nearly parallel rows reported a feasible program as infeasible. t* = 2 from an independent
    # interior-point computation on column-scaled rows (the t = 3 rows peak 2e-6 below the bound).
    f, k = scaled_loop(1e4, 1e-4, 0.002)
    rows, t_star = admissible_set(f, k, np.full(3, A_MAX))
    assert t_star == 2
    expected = np.vstack([k @ np.linalg.matrix_power(f, t) for t in range(3)])
    np.testing.assert_allclose(rows, expected, rtol=0.0, atol=1e-12 * np.max(np.abs(expected)))


def test_set_that_cannot_be_certified_is_refused():
    # Without the certificate this returns t* = 2, yet an independent interior-point LP puts a
    # row of h F^3 at 1.0001 times its bound over that set: a wrong set, so it must raise.
    f, k = scaled_loop(10**2.5, 1e-4, 0.2)
    # Platform arithmetic decides whether the recursion stalls first or the certificate fails.
    with pytest.raises((RuntimeError, ValueError), match="could not be certified|not finitely"):
        admissible_set(f, k, np.full(3, A_MAX))


@pytest.mark.parametrize(
    "scales", [(10.0, 0.01, 0.02), (1.0, 1e-3, 0.02), (100.0, 0.1, 0.02), (10.0, 0.01, 0.002)]
)
def test_returned_set_is_invariant_for_ordinary_weights(scales):
    f, k = scaled_loop(*scales)
    rows, _ = admissible_set(f, k, np.full(3, A_MAX))
    assert worst_image(rows, np.full(3, A_MAX), f) <= 1.0 + 1e-9


def hold_loop():
    """The approach-vbar view's LQR loop with the default weights, and its rows and bounds.

    The hold row keeps n_h . r >= R + b_h, written about the hold point as |n_h . dr| <=
    |hold| - R - b_h. b_h = 2.25 m covers the chord-to-arc allowance A dt^2 / 8 = 1.4593 m between
    samples, A being the acceleration bound of `interval_bounds` at the set's largest range and
    speed with sqrt(3) max_accel of thrust; the range is |hold| plus the norm of the set's extent
    along each position axis, the speed the norm of its extents along the velocity axes (from an
    independent computation).
    """
    view = make("approach-vbar").agent_view()
    cfg = LQRConfig()
    phi, gamma = cw_discrete(view.n, view.dt)
    a, b = cw_system(view.n)
    qc = np.diag([cfg.pos_scale_m**-2] * 3 + [cfg.vel_scale_mps**-2] * 3)
    rc = np.eye(3) / view.max_accel**2
    k, _ = dlqr(phi, gamma, *sampled_data_cost(a, b, qc, rc, view.dt))
    hold = np.array(view.goal["hold_point"], dtype=float)
    xs = np.concatenate([hold, np.zeros(3)])
    us = steady_state_input(phi, gamma, xs)
    radius = next(c["radius"] for c in view.constraints if c["type"] == "keep_out")
    n_h = hold / np.linalg.norm(hold)
    h = np.vstack([k, np.concatenate([n_h, np.zeros(3)])])
    bound = np.concatenate([view.max_accel - np.abs(us), [np.linalg.norm(hold) - radius - 2.25]])
    return view, phi - gamma @ k, k, xs, us, radius, h, bound


def ray_states(rows, bound, count, seed, boundary):
    """Deviations on rays from the hold point, mostly toward the target and slow.

    Directions are drawn with scales (1, 50, 1) m and 0.05 m/s; each ray is cut where it leaves
    the set, then shortened by a random fraction or by 1e-9 (`boundary`).
    """
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(count, 6)) * np.array([1.0, 50.0, 1.0, 0.05, 0.05, 0.05])
    reach = np.min(row_bounds(rows, bound) / np.abs(d @ rows.T), axis=1)
    fraction = np.full(count, 1.0 - 1e-9) if boundary else rng.uniform(size=count) ** (1 / 6)
    return d * (reach * fraction)[:, None]


def test_hold_row_keeps_the_box_and_the_keep_out_over_three_orbits():
    # About 1.5 s: 200 states, 1704 steps of 10 s each.
    view, f, k, xs, us, radius, h, bound = hold_loop()
    rows, t_star = admissible_set(f, h, bound)
    assert t_star == 1  # independent computation
    b = row_bounds(rows, bound)
    on_axis = linprog(-ALONG_TRACK, A_ub=np.vstack([rows, -rows]), b_ub=np.concatenate([b, b]))
    assert -on_axis.fun <= (np.linalg.norm(xs[:3]) - radius - 2.25) * (1.0 + 1e-9)

    e = np.vstack(
        [ray_states(rows, bound, 100, 0, boundary=True)]
        + [ray_states(rows, bound, 100, 1, boundary=False)]
    )
    steps = int(np.ceil(3.0 * 2.0 * np.pi / (view.n * view.dt)))
    flown = np.empty((steps + 1, *e.shape))
    flown[0] = e
    for j in range(steps):
        flown[j + 1] = flown[j] @ f.T
    u = us - flown[:-1] @ k.T
    assert np.max(np.abs(u)) <= view.max_accel * (1.0 + 1e-9)

    # First-order certified range bound over each held step from its two ends (an interval of
    # dt), the bound `segment_minimum` uses with the keep-out rate |v|.
    x = flown + xs
    ends = np.stack([x[:-1], x[1:]], axis=-2)
    speed, _ = interval_bounds(ends, view.dt, np.linalg.norm(u, axis=-1)[..., None], view.n)
    margin = np.linalg.norm(ends[..., :3], axis=-1) - radius
    certified = 0.5 * (margin[..., 0] + margin[..., 1] - speed[..., 0] * view.dt)
    assert np.min(certified) >= 0.0


def test_box_rows_alone_admit_states_inside_the_keep_out_sphere():
    # Without the hold row the same draws start inside 50 m, so the rows above are what keeps
    # those flights out.
    _, f, k, xs, _, radius, h, bound = hold_loop()
    rows, _ = admissible_set(f, h[:3], bound[:3])
    e = ray_states(rows, bound[:3], 100, 1, boundary=False)
    assert np.min(np.linalg.norm((e + xs)[:, :3], axis=1)) < radius
