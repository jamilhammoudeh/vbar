"""Primer vector and the finite-thrust dual lower bound on CW minimum-fuel transfers.

For any multipliers nu_j at constraint times t_j, nu^T d minus the capture-ball support terms minus
a times the integral of max(0, |p(t)|_* - 1) is at most the delta-v of every input with
|a(t)| <= a on CW motion, where p(t) is Lawden's primer vector B^T Phi(t_j, t)^T nu_j summed over
t_j > t and |.|_* is the dual of the actuator norm: l_inf per axis for box thrusters (cost l1),
l2 for a ball (Boyd and Vandenberghe, Convex Optimization, 2004, section 5.1.3 and example 3.26;
D. F. Lawden, Optimal Trajectories for Space Navigation, 1963). With accel = inf the bound is the
impulsive one, D(nu) / sup |p|_*. Both must hold over the whole interval, not at evaluation
points only, so the tests include primers that peak between them.

The known answers are the half-orbit transfer of x0 = (-200, -1500, 50) m at rest to
(0, -100, 0) m at rest on a 6778 km circular orbit with |a_i| <= 2e-3 m/s^2, and an out-of-plane
transfer with a closed-form optimum. Values marked "independent computation" were computed with
`cw_discrete` and scipy's HiGHS and a separate implementation of the cell bound, not with the
module under test.
"""

import functools
import math

import numpy as np
import pytest
from scipy.optimize import linprog

from vbar.agents.base import AgentView
from vbar.dynamics import cw_discrete, cw_stm, cw_system, mean_motion
from vbar.dynamics.cw import R_EARTH
from vbar.oracles.primer import _lipschitz, dual_bound, primer

N_HALF = mean_motion(6778e3)
T_HALF = math.pi / N_HALF
X0_HALF = np.array([-200.0, -1500.0, 50.0, 0.0, 0.0, 0.0])
HOLD_HALF = (0.0, -100.0, 0.0)
A_HALF = 2e-3
# The 640-step ZOH optimum of the box problem, a feasible plan and so an upper bound on the
# continuous optimum (independent computation). The margin covers HiGHS's feasibility tolerance.
P640 = 0.8947292762
P640_MARGIN = 1e-6
# The 2-norm problem with |a|_2 <= 2e-3: equality multipliers of a 40-step LP over 400 outer
# polyhedral directions, and the exact 2-norm cost of a 40-step plan from an inner polyhedron,
# |u_k|_2 <= a checked on every step and the target reached to 3e-12 (independent computation).
NU_BALL = np.array(
    [
        0.0019899957123063593,
        0.0005028098178603568,
        0.0010684558619793439,
        -1.2080758403822702,
        0.28813086168409263,
        0.03691134065725177,
    ]
)
UB_BALL = 0.8324776


def make_view(n, accel, hold_point, layout, *, dt, steps=1, radius=0.0, speed=0.0, **extra):
    goal = {"hold_point": hold_point, "capture_radius": radius, "capture_speed": speed, **extra}
    return AgentView(
        scenario_id="primer-test",
        dt=dt,
        max_steps=steps,
        n=n,
        action_mode="accel",
        max_accel=accel,
        thrust_limit=layout,
        goal=goal,
    )


def half_view(layout="box", **kwargs):
    kwargs.setdefault("dt", T_HALF)
    return make_view(N_HALF, A_HALF, HOLD_HALF, layout, **kwargs)


def primer_matrix(n, tau):
    """(..., 3, 6) matrix C with p = C nu, from the cancellation-free ZOH closed form."""
    phi, _ = cw_discrete(n, np.asarray(tau, dtype=float))
    return np.swapaxes(phi[..., :, 3:6], -1, -2)


def displacement(n, start, goal_state, nus, times, radius=0.0, speed=0.0):
    """Sum over j of nu_j^T (x_g - Phi(t_j) x0) - radius |nu_j,r| - speed |nu_j,v|."""
    total = 0.0
    for nu, t in zip(nus, times, strict=True):
        phi, _ = cw_discrete(n, t)
        total += nu @ (goal_state - phi @ start)
        total -= radius * np.linalg.norm(nu[:3]) + speed * np.linalg.norm(nu[3:])
    return total


@functools.cache
def midpoint_matrices(n, times, points):
    """Per t_j, the count m of midpoints of [0, t_J] before t_j and C(t_j - t) on them, (3 m, 6)."""
    t = (np.arange(points) + 0.5) * times[-1] / points
    counts = [int(np.searchsorted(t, tj)) for tj in times]
    return [
        (m, primer_matrix(n, tj - t[:m]).reshape(-1, 6))
        for m, tj in zip(counts, times, strict=True)
    ]


def primer_samples(n, nus, times, points=200_000):
    """Primers (points, 3, K) at the midpoints of [0, t_J] for K sets of multipliers (K, J, 6)."""
    p = np.zeros((points, 3, len(nus)))
    for j, (m, c) in enumerate(midpoint_matrices(n, tuple(times), points)):
        p[:m] += (c @ nus[:, j].T).reshape(m, 3, -1)
    return p


def midpoint_excess(n, nus, times, norm, points=200_000):
    """Midpoint sums (K,) of max(0, |p|_* - 1) over [0, t_J] (s), in batches of ten."""
    sums = []
    for batch in np.array_split(nus, max(1, len(nus) // 10)):
        p = primer_samples(n, batch, times, points)
        size = np.abs(p) if norm == "l1" else np.linalg.norm(p, axis=1, keepdims=True)
        sums.append(np.maximum(0.0, size - 1.0).sum(axis=(0, 1)) * times[-1] / points)
    return np.concatenate(sums)


@functools.cache
def box_optimum(steps):
    """Optimum (m/s) and equality multipliers of the box problem on a ZOH grid of `steps` steps."""
    dt = T_HALF / steps
    phi, gamma = cw_discrete(N_HALF, dt)
    powers = [np.eye(6)]
    for _ in range(steps):
        powers.append(phi @ powers[-1])
    cols = [powers[steps - 1 - k] @ gamma for k in range(steps)]
    goal_state = np.r_[HOLD_HALF, 0.0, 0.0, 0.0]
    d = goal_state - powers[steps] @ X0_HALF
    a_eq = np.hstack([np.hstack([g, -g]) for g in cols])
    res = linprog(
        np.full(6 * steps, dt),
        A_eq=a_eq,
        b_eq=d,
        bounds=[(0.0, A_HALF)] * (6 * steps),
        method="highs",
    )
    assert res.status == 0, res.message
    return res.fun, res.eqlin.marginals


def half_bound(nu, *, norm="l1", cells=20_000, accel=None, view=None):
    if view is None:
        view = half_view("box" if norm == "l1" else "ball")
    return dual_bound(
        N_HALF, view, X0_HALF, np.atleast_2d(nu), [T_HALF], norm=norm, accel=accel, cells=cells
    )


def test_primer_matches_the_closed_form_transition_matrix():
    n = 1.1e-3
    rng = np.random.default_rng(1)
    nus = [*np.eye(6), *rng.standard_normal((3, 6))]
    for n_tau in (0.3, 4.0):
        tau = n_tau / n
        c_ref = (cw_stm(n, tau)[:, 3:6]).T
        for nu in nus:
            p = primer(n, nu, tau)
            assert p.shape == (3,)
            # Different paths to the same closed form: tolerance relative to the summed terms.
            scale = np.abs(c_ref) @ np.abs(nu)
            assert np.all(np.abs(p - c_ref @ nu) <= 1e-12 * scale)


def test_primer_keeps_full_precision_at_short_times():
    # At n tau = 1e-6 the 1 - cos(n tau) terms of the textbook matrix lose four digits; the
    # power series of expm(A tau) B converges to rounding within nine terms.
    n = 1.1e-3
    tau = 1e-6 / n
    a_mat, b_mat = cw_system(n)
    series = np.zeros((6, 3))
    term = b_mat.copy()
    for k in range(9):
        series += term
        term = a_mat @ term * tau / (k + 1)
    c_ref = series.T
    for nu in np.eye(6):
        p = primer(n, nu, tau)
        scale = np.abs(c_ref) @ np.abs(nu)
        assert np.all(np.abs(p - c_ref @ nu) <= 1e-12 * scale)
    # p_x from nu_y is -2 (1 - cos n tau) / n alone, the term that cancels.
    assert primer(n, np.eye(6)[1], tau)[0] == pytest.approx(c_ref[0, 1], rel=1e-12)


def test_primer_at_the_constraint_time_is_the_velocity_multiplier():
    n = 1.1e-3
    nu = np.array([0.3, -0.2, 0.1, 1.5, -0.7, 0.4])
    taus = np.array([0.0, 10.0, 1000.0])
    p = primer(n, nu, taus)
    assert p.shape == (3, 3)
    np.testing.assert_array_equal(p[0], nu[3:])
    np.testing.assert_allclose(p[1:], primer_matrix(n, taus[1:]) @ nu, rtol=1e-12)


def test_box_bound_from_a_coarse_dual_is_below_the_finer_optima():
    p40, nu40 = box_optimum(40)
    p160, _ = box_optimum(160)
    # Boyd and Vandenberghe, sections 4.3 and 5.1.3: the ZOH problem is an LP; values to their
    # printed digits, independent computation (0.8977056 and 0.8949323).
    assert p40 == pytest.approx(0.8977056, abs=1e-7)
    assert p160 == pytest.approx(0.8949323, abs=1e-7)
    lb = half_bound(nu40)
    # The continuous-time value of this dual is 0.8930605 (200000-point midpoint sum,
    # independent computation); the cell rule (larger endpoint + L h / 2 on 20000 cells) gives
    # 0.8921478. A bound above the first is unsound for some nu, below the second it is looser
    # than the rule, and a halved acceleration term would give 0.9139.
    assert 0.892140 <= lb <= 0.8930615
    assert lb < p160 < p40


def test_box_bound_from_a_finer_dual_is_tighter():
    _, nu40 = box_optimum(40)
    _, nu160 = box_optimum(160)
    lb = half_bound(nu160)
    # Midpoint value 0.8946294, cell rule 0.8937805 (independent computation).
    assert 0.893775 <= lb <= 0.8946304
    assert lb > half_bound(nu40)


def test_bound_never_exceeds_a_feasible_cost_for_perturbed_duals():
    _, nu40 = box_optimum(40)
    _, nu160 = box_optimum(160)
    rng = np.random.default_rng(2)
    far = [half_bound(nu40 * (1 + 0.3 * rng.standard_normal(6)), cells=2000) for _ in range(200)]
    near = [half_bound(nu160 * (1 + 1e-3 * rng.standard_normal(6))) for _ in range(10)]
    assert max(far) <= P640 + P640_MARGIN
    # Near the maximum the cell rule sits 8.8e-4 below P640 (independent computation).
    assert max(near) <= P640 + P640_MARGIN
    assert max(near) > 0.8935


def test_bound_is_concave_in_the_multipliers():
    _, nu40 = box_optimum(40)
    rng = np.random.default_rng(3)
    for _ in range(5):
        direction = 0.5 * nu40 * rng.standard_normal(6)
        s = np.linspace(-1.0, 1.0, 50)
        lb = np.array([half_bound(nu40 + si * direction, cells=2000) for si in s])
        # Equal spacing: each interior point is the midpoint of its neighbours.
        chord = 0.5 * (lb[:-2] + lb[2:])
        assert np.all(lb[1:-1] >= chord - 1e-12 * (1.0 + np.abs(chord)))


def test_cell_bound_covers_the_fine_midpoint_integral():
    _, nu40 = box_optimum(40)
    goal_state = np.r_[HOLD_HALF, 0.0, 0.0, 0.0]
    rng = np.random.default_rng(4)
    for times in ([T_HALF], [0.6 * T_HALF, T_HALF]):
        steps = 1 if len(times) == 1 else 5
        nus = nu40 * (1 + 0.2 * rng.standard_normal((25, len(times), 6)))
        nus[:, 1:] *= 0.05
        for norm, layout in (("l1", "box"), ("l2", "ball")):
            view = half_view(layout, dt=T_HALF / steps, steps=steps)
            excess = midpoint_excess(N_HALF, nus, times, norm, points=20_000)
            assert np.count_nonzero(excess) >= 20
            for k, nu in enumerate(nus):
                cells = (1, 3, 40, 2000)[k % 4]
                lb = dual_bound(N_HALF, view, X0_HALF, nu, times, norm=norm, cells=cells)
                d = displacement(N_HALF, X0_HALF, goal_state, nu, times)
                # In units of 1 m/s + a E: on these cases the 20000-point sum is within 1.5e-8
                # of the 200000-point one and the cell rule at 2000 cells lies at least 1.4e-3
                # below it (independent computation); the margin is 1e-6.
                assert lb <= d - A_HALF * excess[k] + 1e-6 * (1.0 + A_HALF * excess[k])


def out_of_plane(layout="box"):
    """z0 = 500 m at rest to the origin at rest in half an orbit at 500 km, a = 1e-3 m/s^2.

    One burn centred on the node with half-width delta, 2 a sin(delta) = n^2 z0, costs
    2 a delta / n; nu_z = n / cos(delta) gives the primer sin(n tau) / cos(delta), above 1 exactly
    on the burn, and its bound equals that cost, so both are optimal. The impulsive optimum is
    n z0, one impulse at the node, certified by nu_z = n (primer sin(n tau)).
    """
    n = mean_motion(R_EARTH + 500e3)
    z0, accel = 500.0, 1e-3
    delta = math.asin(n**2 * z0 / (2 * accel))
    return {
        "n": n,
        "tf": math.pi / n,
        "view": make_view(n, accel, (0.0, 0.0, 0.0), layout, dt=math.pi / n),
        "start": np.array([0.0, 0.0, z0, 0.0, 0.0, 0.0]),
        "nu": np.array([[0.0, 0.0, n / math.cos(delta), 0.0, 0.0, 0.0]]),
        "nu_impulsive": np.array([[0.0, 0.0, n, 0.0, 0.0, 0.0]]),
        "finite": 2 * accel * delta / n,
        "impulsive": n * z0,
    }


def oop_bound(case, nu, *, norm="l1", cells, accel=None):
    return dual_bound(
        case["n"],
        case["view"],
        case["start"],
        nu,
        [case["tf"]],
        norm=norm,
        accel=accel,
        cells=cells,
    )


@pytest.mark.parametrize("layout,norm", [("box", "l1"), ("ball", "l2")])
def test_bound_reaches_the_out_of_plane_finite_thrust_optimum(layout, norm):
    case = out_of_plane(layout)
    lb = oop_bound(case, case["nu"], norm=norm, cells=20_000)
    # The cell rule is 9.5e-5 below the optimum here, and a halved acceleration term would give
    # 1.017 times it (independent computation).
    assert case["finite"] * (1 - 2e-4) <= lb <= case["finite"] * (1 + 1e-12)


def test_primer_peak_between_cells_is_counted():
    case = out_of_plane()
    # Three cells: the primer is 0.910 at every cell edge and peaks at 1.050 mid-cell, so a
    # maximum taken at the edges sees no excess and returns nu^T d = 1.034 times the optimum.
    nu = case["nu"][0]
    edges = primer(case["n"], nu, np.linspace(0.0, case["tf"], 4))[:, 2]
    peak = primer(case["n"], nu, 0.5 * case["tf"])[2]
    assert np.all(np.abs(edges) < 1.0) and peak > 1.0
    assert oop_bound(case, case["nu"], cells=3) <= case["finite"] * (1 + 1e-12)


@pytest.mark.parametrize("layout,norm", [("box", "l1"), ("ball", "l2")])
def test_one_cell_bound_stays_below_the_out_of_plane_optima(layout, norm):
    case = out_of_plane(layout)
    # One cell over half an orbit: the primer is 0 at both ends and peaks mid-cell, so all the
    # margin is the slack L h / 2 = pi / 2 of the peak. With half of it the finite bound would
    # be nu^T d = 1.034 times the optimum and the impulsive one n z0 / (pi / 4) = 1.27 n z0.
    finite = oop_bound(case, case["nu"], norm=norm, cells=1)
    assert finite <= case["finite"] * (1 + 1e-12)
    impulsive = oop_bound(case, case["nu_impulsive"], norm=norm, cells=1, accel=np.inf)
    assert impulsive <= case["impulsive"] * (1 + 1e-12)


def test_impulsive_bound_reaches_the_out_of_plane_impulsive_optimum():
    case = out_of_plane()
    fine = oop_bound(case, case["nu_impulsive"], cells=20_000, accel=np.inf)
    # The cell rule is 7.9e-5 below n z0 here (independent computation).
    assert case["impulsive"] * (1 - 2e-4) <= fine <= case["impulsive"] * (1 + 1e-12)
    # Three cells: sup |p| = 1 at the node, mid-cell; the edges see sin(pi / 3), which would
    # give n z0 / sin(pi / 3) = 1.155 n z0.
    coarse = oop_bound(case, case["nu_impulsive"], cells=3, accel=np.inf)
    assert coarse <= case["impulsive"] * (1 + 1e-12)


def test_impulsive_bound_is_invariant_to_scaling_the_multipliers():
    _, nu40 = box_optimum(40)
    base = half_bound(nu40, accel=np.inf, cells=2000)
    assert base > 0.0
    for lam in (1e-3, 0.5, 7.0, 1e4):
        assert half_bound(lam * nu40, accel=np.inf, cells=2000) == pytest.approx(base, rel=1e-12)


@pytest.mark.parametrize("accel", [None, np.inf])
def test_bound_is_homogeneous_in_the_displacement(accel):
    # CW is linear: scaling the start, the goal, the capture ball and the acceleration limit by
    # lam scales every admissible cost, and the bound for the same nu, by lam.
    _, nu40 = box_optimum(40)

    def bound(lam):
        hold = tuple(lam * np.array(HOLD_HALF))
        view = make_view(
            N_HALF, lam * A_HALF, hold, "box", dt=T_HALF, radius=5 * lam, speed=0.05 * lam
        )
        x0 = lam * X0_HALF
        return dual_bound(
            N_HALF, view, x0, nu40[None], [T_HALF], norm="l1", accel=accel, cells=2000
        )

    base = bound(1.0)
    for lam in (0.01, 3.7):
        assert bound(lam) == pytest.approx(lam * base, rel=1e-12)


def test_capture_ball_lowers_the_bound_by_its_support_function():
    # min over |r - r_g| <= R, |v - v_g| <= V of nu^T x is nu^T x_g - R |nu_r|_2 - V |nu_v|_2.
    _, nu40 = box_optimum(40)
    point = half_bound(nu40)
    ball = half_bound(nu40, view=half_view(radius=5.0, speed=0.05))
    drop = 5.0 * np.linalg.norm(nu40[:3]) + 0.05 * np.linalg.norm(nu40[3:])
    assert point - ball == pytest.approx(drop, rel=1e-12)


def test_box_and_ball_bounds_use_their_own_dual_norm():
    _, nu40 = box_optimum(40)
    nu = nu40 * 0.98 / np.abs(primer_samples(N_HALF, nu40[None, None], [T_HALF])).max()
    p = primer_samples(N_HALF, nu[None, None], [T_HALF])[..., 0]
    # Every axis stays below 1 while the 2-norm reaches 1.31: no box excess, a large ball excess.
    assert np.abs(p).max() < 0.981 and np.linalg.norm(p, axis=1).max() > 1.3
    goal_state = np.r_[HOLD_HALF, 0.0, 0.0, 0.0]
    d = displacement(N_HALF, X0_HALF, goal_state, [nu], [T_HALF])
    box = half_bound(nu, norm="l1")
    assert box == pytest.approx(d, rel=1e-12)
    ball_excess = A_HALF * midpoint_excess(N_HALF, nu[None, None], [T_HALF], "l2")[0]
    assert ball_excess > 0.05
    ball = half_bound(nu, norm="l2")
    # The cell rule leaves 0.86% of the excess (independent computation).
    assert d - 1.02 * ball_excess <= ball <= d - ball_excess + 1e-8


def test_ball_bound_stays_below_a_verified_ball_plan():
    lb = half_bound(NU_BALL, norm="l2")
    # Midpoint value 0.8271793, cell rule 0.8267173; with the box dual norm this nu would give
    # 0.8333464, above the verified plan (independent computation).
    assert 0.826710 <= lb <= 0.8271803
    assert lb < UB_BALL
    rng = np.random.default_rng(5)
    perturbed = [
        half_bound(NU_BALL * (1 + 1e-2 * rng.standard_normal(6)), norm="l2", cells=2000)
        for _ in range(200)
    ]
    assert max(perturbed) < UB_BALL


def test_dwell_bound_stays_below_a_feasible_cost():
    # The hold point is on the V-bar, a CW equilibrium: the 640-step plan arrives at T and then
    # stays with no thrust, so P640 bounds the cost of being there at T and at 1.25 T.
    _, nu160 = box_optimum(160)
    view = half_view(dt=0.25 * T_HALF, steps=5)
    times = [T_HALF, 1.25 * T_HALF]
    rng = np.random.default_rng(6)
    for _ in range(100):
        nus = np.array([nu160 * (1 + 0.05 * rng.standard_normal(6)), 1e-6 * rng.standard_normal(6)])
        lb = dual_bound(N_HALF, view, X0_HALF, nus, times, norm="l1", cells=2000)
        assert lb <= P640 + P640_MARGIN


def test_invalid_inputs_raise():
    _, nu40 = box_optimum(40)
    with pytest.raises(ValueError):
        half_bound(nu40, norm="l3")
    no_radius = AgentView(
        scenario_id="primer-test",
        dt=T_HALF,
        max_steps=1,
        n=N_HALF,
        action_mode="accel",
        max_accel=A_HALF,
        thrust_limit="box",
        goal={"hold_point": HOLD_HALF},
    )
    with pytest.raises(ValueError):
        half_bound(nu40, view=no_radius)
    view = half_view(dt=0.25 * T_HALF, steps=5)
    with pytest.raises(ValueError):
        dual_bound(N_HALF, view, X0_HALF, np.array([nu40, nu40]), [T_HALF, 0.5 * T_HALF], norm="l1")
    with pytest.raises(ValueError):
        dual_bound(N_HALF, view, X0_HALF, np.array([nu40, nu40]), [T_HALF], norm="l1")


def test_mean_motion_must_match_the_view():
    # A bound computed for another orbit than the scenario's is not certified for it.
    _, nu40 = box_optimum(40)
    view = half_view()
    for n in (N_HALF * (1 + 1e-9), N_HALF * (1 - 1e-9), 2 * N_HALF):
        with pytest.raises(ValueError):
            dual_bound(n, view, X0_HALF, nu40[None], [T_HALF], norm="l1", cells=200)
    close = dual_bound(
        N_HALF * (1 + 1e-14), view, X0_HALF, nu40[None], [T_HALF], norm="l1", cells=200
    )
    assert close == pytest.approx(half_bound(nu40, cells=200), rel=1e-9)


def test_default_cells_are_eight_per_step():
    _, nu40 = box_optimum(40)
    view = half_view(dt=T_HALF / 40, steps=40)
    default = dual_bound(N_HALF, view, X0_HALF, nu40[None], [T_HALF], norm="l1")
    explicit = dual_bound(N_HALF, view, X0_HALF, nu40[None], [T_HALF], norm="l1", cells=8 * 40)
    assert default == explicit


@pytest.mark.parametrize("layout,norm", [("box", "l2"), ("ball", "l1")])
def test_norm_must_match_the_thruster_layout(layout, norm):
    _, nu40 = box_optimum(40)
    with pytest.raises(ValueError):
        half_bound(nu40, norm=norm, view=half_view(layout))


def test_thrust_inhibit_windows_leave_the_bound_unchanged():
    # Leaving the windows out lets the bound admit inputs the scenario forbids, which only
    # lowers it: the bound stays valid, and the windows must not change it.
    _, nu40 = box_optimum(40)
    view = half_view(dt=T_HALF / 40, steps=40)
    windows = ((0.25 * T_HALF, 0.5 * T_HALF), (0.75 * T_HALF, 0.8 * T_HALF))
    inhibited = half_view(dt=T_HALF / 40, steps=40, thrust_inhibit=windows)
    plain = dual_bound(N_HALF, view, X0_HALF, nu40[None], [T_HALF], norm="l1", cells=2000)
    held = dual_bound(N_HALF, inhibited, X0_HALF, nu40[None], [T_HALF], norm="l1", cells=2000)
    assert held == plain


def primer_rate(n, nu, tau):
    """d p / d tau (..., 3) = (A Phi(tau) B)^T nu; |dp/dt| is the same."""
    a_mat, _ = cw_system(n)
    phi, _ = cw_discrete(n, np.asarray(tau, dtype=float))
    return np.swapaxes((a_mat @ phi)[..., :, 3:6], -1, -2) @ nu


def test_primer_rate_bounds_are_sharp_for_unit_multipliers():
    # Over half an orbit every unit multiplier drives each primer component at its full rate
    # somewhere (n tau = 0, pi / 2 or pi, all on the grid), so the per-axis bounds on |dp_i/dt|
    # must hold and be reached.
    n = 1.1e-3
    tau = np.linspace(0.0, math.pi / n, 20_001)
    for nu in np.eye(6):
        rate = np.abs(primer_rate(n, nu, tau)).max(axis=0)
        bound = _lipschitz(n, nu)
        assert np.all(rate <= bound * (1 + 1e-12) + 1e-18)
        assert np.all(rate >= bound * (1 - 1e-8))


def test_ball_cell_slack_covers_the_primer_speed():
    # One cell over half an orbit: the impulsive bound is D / top, with top the larger end
    # value of |p|_2 plus a slope times h / 2. At n tau = pi this primer's y and z components
    # both change at their full rate 7e-3 / s, so |dp/dt|_2 reaches 7 sqrt(2) 1e-3 / s, above
    # every single-axis rate; the slope must cover it, and |L|_2 = 1.0206 times it does.
    nu = np.array([0.0, 1e-3, 7e-3, 0.0, 0.0, 0.0])
    goal_state = np.r_[HOLD_HALF, 0.0, 0.0, 0.0]
    d = displacement(N_HALF, X0_HALF, goal_state, [nu], [T_HALF])
    assert abs(d) > 0.1
    nu, d = np.sign(d) * nu, abs(d)
    top = d / half_bound(nu, norm="l2", accel=np.inf, cells=1)
    ends = np.linalg.norm(primer(N_HALF, nu, np.array([0.0, T_HALF])), axis=1).max()
    slope = 2 * (top - ends) / T_HALF
    tau = np.linspace(0.0, T_HALF, 20_001)
    speed = np.linalg.norm(primer_rate(N_HALF, nu, tau), axis=1).max()
    assert speed == pytest.approx(7e-3 * math.sqrt(2), rel=1e-8)
    assert speed * (1 - 1e-9) <= slope <= 1.03 * speed
