"""Piecewise-affine MPC: the constrained regulator solved offline as a multiparametric QP.

min 1/2 U'HU + (F x)'U subject to G U <= w + S x, for x in a box, has one affine law per active
set on a polyhedral critical region (Bemporad, Morari, Dua and Pistikopoulos, "The explicit
linear quadratic regulator for constrained systems", Automatica 38(1), 2002; Rawlings, Mayne and
Diehl, Model Predictive Control, 2nd ed., sections 7.2 to 7.4). References marked "independent
computation" come from a bounded-variable least-squares solve of the same QP or from a separate
enumeration of every active set, not from the module under test. CW cases are for a 500 km
circular orbit of the WGS 84 Earth and a 10 s step.
"""

import dataclasses
import functools
import itertools

import numpy as np
import pytest
from metric_cases import episode
from scipy.linalg import solve_discrete_are
from scipy.optimize import linprog, lsq_linear

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import mpc_pwa
from vbar.agents.base import AgentView, agent_generator, command_fraction
from vbar.agents.lqr import LQRAgent, LQRConfig, dlqr, sampled_data_cost, steady_state_input
from vbar.dynamics import EARTH, cw_discrete, cw_system, mean_motion
from vbar.scenarios import make

N = mean_motion(EARTH.radius + 500e3)
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
VBAR_HOLD = (0.0, -100.0, 0.0)
RBAR_HOLD = (-100.0, 0.0, 0.0)
# Half widths of the parameter box on the deviation from the hold point, Hill order (m, m/s).
BOX = (300.0, 600.0, 300.0, 1.0, 1.0, 1.0)
INFO = {"time_s": 0.0}

# Rawlings et al. Example 7.11: V = x^2/2 - u x + u^2 subject to u >= 1, u >= 2 - x/2 and
# u >= 2 - x; the constant x^2/2 does not move the minimizer.
EX711 = {
    "h": np.array([[2.0]]),
    "f": np.array([[-1.0]]),
    "g": np.array([[-1.0], [-1.0], [-1.0]]),
    "w": np.array([-1.0, -2.0, -2.0]),
    "s": np.array([[0.0], [0.5], [1.0]]),
    "bounds": [(-1.0, 4.0)],
}
# Rawlings et al. Example 7.12: H = [[3, 1], [1, 2]], linear term (2x, x), |u_i| <= 1; rows in
# the order u1 <= 1, u2 <= 1, -u1 <= 1, -u2 <= 1.
EX712 = {
    "h": np.array([[3.0, 1.0], [1.0, 2.0]]),
    "f": np.array([[2.0], [1.0]]),
    "g": np.vstack([np.eye(2), -np.eye(2)]),
    "w": np.ones(4),
    "s": np.zeros((4, 1)),
    "bounds": [(-4.0, 4.0)],
}

# CW subsystems with the constants of the region-count study: |a| <= 2e-3 m/s^2, stage weights
# below, R = 1e4 per input, DARE terminal cost; the parameter box bounds the state, the input
# limits are the only rows of G.
CW_A_MAX = 2e-3
SUBSYSTEMS = {
    "cross_track": ([2, 5], [2], np.diag([1.0, 1e3]), [(-300.0, 300.0), (-1.0, 1.0)]),
    "in_plane": (
        [0, 1, 3, 4],
        [0, 1],
        np.diag([1.0, 1.0, 1e3, 1e3]),
        [(-300.0, 300.0), (-600.0, 600.0), (-1.0, 1.0), (-1.0, 1.0)],
    ),
}


def box_qp(h, fx, lo, hi):
    """Minimizer of 1/2 U'HU + fx'U over lo <= U <= hi by bounded-variable least squares."""
    chol = np.linalg.cholesky(h)
    rhs = -np.linalg.solve(chol, fx)
    return lsq_linear(chol.T, rhs, bounds=(lo, hi), method="bvls", tol=1e-15).x


def law(region, x):
    return region.gain @ np.asarray(x, dtype=float) + region.offset


def interval(region):
    """The interval a x <= b of a region with a scalar parameter."""
    a = region.a[:, 0]
    b = region.b
    return np.max(b[a < 0] / a[a < 0]), np.min(b[a > 0] / a[a > 0])


def by_active(regions, active):
    (region,) = [r for r in regions if r.active == active]
    return region


def condensed(subsystem, horizon):
    """H and F of the condensed CW regulator on one subsystem, cost sum x'Qx + u'Ru + x_N'P x_N."""
    xi, ui, q, _ = SUBSYSTEMS[subsystem]
    phi, gamma = cw_discrete(N, DT)
    a = phi[np.ix_(xi, xi)]
    b = gamma[np.ix_(xi, ui)]
    nx, nu = b.shape
    r = 1e4 * np.eye(nu)
    p = solve_discrete_are(a, b, q, r)
    sx = np.zeros((nx * horizon, nx))
    su = np.zeros((nx * horizon, nu * horizon))
    for k in range(horizon):
        sx[nx * k : nx * (k + 1)] = np.linalg.matrix_power(a, k + 1)
        for j in range(k + 1):
            su[nx * k : nx * (k + 1), nu * j : nu * (j + 1)] = np.linalg.matrix_power(a, k - j) @ b
    qbar = np.kron(np.eye(horizon), q)
    qbar[-nx:, -nx:] = p
    return su.T @ qbar @ su + np.kron(np.eye(horizon), r), su.T @ qbar @ sx


@functools.cache
def cw_problem(subsystem, horizon):
    h, f = condensed(subsystem, horizon)
    nu = h.shape[0]
    g = np.vstack([np.eye(nu), -np.eye(nu)])
    w = np.full(2 * nu, CW_A_MAX)
    s = np.zeros((2 * nu, f.shape[1]))
    bounds = SUBSYSTEMS[subsystem][3]
    return h, f, g, w, s, bounds, mpc_pwa.mpqp_regions(h, f, g, w, s, bounds)


def box_points(bounds, count, seed):
    lo, hi = np.array(bounds).T
    return np.random.default_rng(seed).uniform(lo, hi, size=(count, len(lo)))


def test_example_7_11_has_one_region_per_active_constraint_and_drops_the_degenerate_sets():
    # Two of the three rows are active together only at x = 0 or x = 2, and rows 0 and 1 are
    # the same row in u, so no region of positive width has two active rows.
    regions = mpc_pwa.mpqp_regions(**EX711)
    assert sorted(r.active for r in regions) == [(), (1,), (2,)]


def test_example_7_11_second_constraint_is_active_on_0_to_2_with_u_two_minus_half_x():
    region = by_active(mpc_pwa.mpqp_regions(**EX711), (1,))
    lo, hi = interval(region)
    # Entries of order 1, so the rounding of the KKT solve is near 1e-16.
    assert abs(lo - 0.0) <= 1e-12 and abs(hi - 2.0) <= 1e-12
    assert abs(region.gain[0, 0] + 0.5) <= 1e-12
    assert abs(region.offset[0] - 2.0) <= 1e-12


def test_example_7_11_law_equals_the_qp_minimizer_on_a_1e_3_grid():
    regions = mpc_pwa.mpqp_regions(**EX711)
    second = [r.active for r in regions].index((1,))
    for x in np.linspace(-1.0, 4.0, 5001):
        index, _ = mpc_pwa.lookup(regions, np.array([x]))
        # The scalar QP's minimizer is the unconstrained x/2 raised to the largest lower bound.
        assert abs(law(regions[index], [x])[0] - max(x / 2, 1.0, 2 - x / 2, 2 - x)) <= 1e-12
        if 1e-9 < x < 2.0 - 1e-9:
            assert index == second


def test_example_7_12_only_u1_at_its_lower_bound_on_5_3_to_3_with_u2_half_minus_half_x():
    region = by_active(mpc_pwa.mpqp_regions(**EX712), (2,))
    lo, hi = interval(region)
    assert abs(lo - 5 / 3) <= 1e-12 and abs(hi - 3.0) <= 1e-12
    assert np.max(np.abs(region.gain[:, 0] - [0.0, -0.5])) <= 1e-12
    assert np.max(np.abs(region.offset - [-1.0, 0.5])) <= 1e-12


def test_example_7_12_has_five_regions_mirrored_about_zero():
    # Independent computation: unconstrained on |x| <= 5/3, one input at its bound to |x| = 3,
    # both beyond.
    regions = mpc_pwa.mpqp_regions(**EX712)
    assert sorted(r.active for r in regions) == [(), (0,), (0, 1), (2,), (2, 3)]


def box_qp_by_faces(h, fx, bound):
    """Minimizers of 1/2 U'HU + fx_i'U over |U| <= bound for each row fx_i, by trying every face.

    Each face's stationary point that lies in the box is a candidate; the minimizer of a strictly
    convex QP is the candidate of least cost.
    """
    nu = h.shape[0]
    best = np.full(len(fx), np.inf)
    u_best = np.zeros_like(fx)
    for pattern in itertools.product((-1.0, 0.0, 1.0), repeat=nu):
        fixed = np.array(pattern) != 0
        u = np.tile(np.array(pattern) * bound, (len(fx), 1))
        if not fixed.all():
            free = ~fixed
            rhs = fx[:, free] + u[:, fixed] @ h[np.ix_(free, fixed)].T
            u[:, free] = -np.linalg.solve(h[np.ix_(free, free)], rhs.T).T
        cost = 0.5 * np.einsum("ij,jk,ik->i", u, h, u) + np.einsum("ij,ij->i", fx, u)
        better = np.all(np.abs(u) <= bound * (1 + 1e-15), axis=1) & (cost < best)
        best[better] = cost[better]
        u_best[better] = u[better]
    return u_best


def test_example_7_12_law_equals_the_box_qp_on_a_1e_3_grid():
    regions = mpc_pwa.mpqp_regions(**EX712)
    grid = np.linspace(-4.0, 4.0, 8001)
    u_qp = box_qp_by_faces(EX712["h"], grid[:, None] @ EX712["f"].T, 1.0)
    for x, u in zip(grid, u_qp, strict=True):
        index, _ = mpc_pwa.lookup(regions, np.array([x]))
        # Independent computation; inputs are of order 1, so rounding is near 1e-16.
        assert np.max(np.abs(law(regions[index], [x]) - u)) <= 1e-12


@pytest.mark.parametrize("c", [0.5, 2.0, 3.0])
def test_saturated_scalar_minimum_is_a_minus_or_plus_b_plus_half_c(c):
    # min a + b u + c u^2 / 2 over |u| <= 1 with b the parameter (Rawlings et al. section 7.2).
    # The values on the saturated pieces are a - b + c/2 and a + b + c/2; the printed c^2/2 is a
    # misprint, and c != 1 here so a law reproducing it fails.
    regions = mpc_pwa.mpqp_regions(
        h=np.array([[c]]),
        f=np.array([[1.0]]),
        g=np.array([[1.0], [-1.0]]),
        w=np.ones(2),
        s=np.zeros((2, 1)),
        bounds=[(-3 * c, 3 * c)],
    )
    assert sorted(r.active for r in regions) == [(), (0,), (1,)]
    rng = np.random.default_rng(6)
    saturated = 0
    for _ in range(200):
        a, b = rng.normal(), rng.uniform(-3 * c, 3 * c)
        index, _ = mpc_pwa.lookup(regions, np.array([b]))
        u = law(regions[index], [b])[0]
        value = a + b * u + c * u * u / 2
        scale = abs(a) + abs(b) + c
        if b / c >= 1:
            assert abs(value - (a - b + c / 2)) <= 1e-14 * scale
            saturated += 1
        elif b / c <= -1:
            assert abs(value - (a + b + c / 2)) <= 1e-14 * scale
            saturated += 1
        else:
            assert abs(value - (a - b * b / (2 * c))) <= 1e-14 * scale
    assert saturated > 50


def test_regions_come_breadth_first_from_the_empty_active_set():
    regions = cw_problem("cross_track", 4)[-1]
    sizes = [len(r.active) for r in regions]
    assert regions[0].active == ()
    assert sizes == sorted(sizes)
    assert all(list(r.active) == sorted(r.active) for r in regions)


@pytest.mark.parametrize(
    "subsystem, horizon, count",
    # Independent computation by enumerating every active set; the breadth-first search finds the
    # same regions. Build times (reference enumeration, one machine): 0.07, 0.5, 0.08 and 0.7 s;
    # cross-track 73 and 129 at N = 6 and 8 and in-plane 417 at N = 3 take 2 to 6 s and stay out
    # of the suite.
    [("cross_track", 2, 9), ("cross_track", 4, 33), ("in_plane", 1, 9), ("in_plane", 2, 81)],
)
def test_cw_region_counts(subsystem, horizon, count):
    assert len(cw_problem(subsystem, horizon)[-1]) == count


@pytest.mark.parametrize("subsystem, horizon", [("cross_track", 4), ("in_plane", 2)])
def test_cw_law_equals_the_box_qp_at_200_states_in_the_box(subsystem, horizon):
    h, f, _, _, _, bounds, regions = cw_problem(subsystem, horizon)
    for x in box_points(bounds, 200, seed=7):
        index, _ = mpc_pwa.lookup(regions, x)
        u_qp = box_qp(h, f @ x, -CW_A_MAX, CW_A_MAX)
        # The least-squares solve meets a direct solve on its active set to 2e-14 a_max; the
        # KKT system of a region costs some digits more (7e-13 a_max in a bordered solve).
        assert np.max(np.abs(law(regions[index], x) - u_qp)) <= 1e-12 * CW_A_MAX


@pytest.mark.parametrize("subsystem, horizon", [("cross_track", 4), ("in_plane", 2)])
def test_every_state_in_the_box_lies_in_exactly_one_region(subsystem, horizon):
    *_, bounds, regions = cw_problem(subsystem, horizon)
    for x in box_points(bounds, 200, seed=8):
        assert sum(bool(np.all(r.a @ x <= r.b)) for r in regions) == 1


@pytest.mark.parametrize("subsystem, horizon", [("cross_track", 4), ("in_plane", 2)])
def test_lookup_returns_the_first_region_containing_the_state_and_counts_its_visits(
    subsystem, horizon
):
    *_, bounds, regions = cw_problem(subsystem, horizon)
    for x in box_points(bounds, 200, seed=9):
        index, visits = mpc_pwa.lookup(regions, x)
        assert np.all(regions[index].a @ x <= regions[index].b)
        assert not any(np.all(r.a @ x <= r.b) for r in regions[:index])
        assert visits == index + 1 <= len(regions)


def test_lookup_outside_the_box_finds_no_region_after_visiting_every_one():
    *_, bounds, regions = cw_problem("cross_track", 4)
    for x in ([301.0, 0.0], [0.0, -1.01], [-1e4, 5.0]):
        assert mpc_pwa.lookup(regions, np.array(x)) == (None, len(regions))


@pytest.mark.parametrize(
    "problem, x",
    [
        (EX711, 0.0),
        (EX711, 2.0),
        (EX711, -1.0),
        (EX711, 4.0),
        (EX712, -3.0),
        (EX712, -5 / 3),
        (EX712, 5 / 3),
        (EX712, 3.0),
        (EX712, 4.0),
    ],
)
def test_a_state_on_a_region_boundary_goes_to_the_first_region_it_touches(problem, x):
    regions = mpc_pwa.mpqp_regions(**problem)
    point = np.array([x])
    index, _ = mpc_pwa.lookup(regions, point)
    assert index is not None
    # Every region the sequential search passed over misses the point by more than rounding.
    for r in regions[:index]:
        assert np.max(r.a @ point - r.b) > 1e-9
    if problem is EX712:
        u_qp = box_qp(problem["h"], problem["f"] @ point, -1.0, 1.0)
    else:
        u_qp = np.array([max(x / 2, 1.0, 2 - x / 2, 2 - x)])
    # The law is continuous, so either region's law gives the minimizer on the boundary.
    assert np.max(np.abs(law(regions[index], point) - u_qp)) <= 1e-12


def facet_points(regions):
    """Each region's Chebyshev centre projected onto each of its rows, where it stays a face."""
    for r in regions:
        nx = r.a.shape[1]
        norms = np.linalg.norm(r.a, axis=1)
        res = linprog(
            np.r_[np.zeros(nx), -1.0],
            A_ub=np.column_stack([r.a, norms]),
            b_ub=r.b,
            bounds=[(None, None)] * nx + [(0.0, None)],
            method="highs",
        )
        centre = res.x[:nx]
        for a_i, b_i in zip(r.a, r.b, strict=True):
            # A row of zeros (an input bound that cannot bind in a saturated region) has no face.
            if not a_i.any():
                continue
            x = centre + (b_i - a_i @ centre) / (a_i @ a_i) * a_i
            if np.all(r.a @ x <= r.b + 1e-9 * (1.0 + np.abs(r.b))):
                yield x


@pytest.mark.parametrize("subsystem, horizon", [("cross_track", 4), ("in_plane", 1)])
def test_states_on_region_facets_are_found_and_take_the_qp_minimizer(subsystem, horizon):
    h, f, _, _, _, _, regions = cw_problem(subsystem, horizon)
    points = list(facet_points(regions))
    assert len(points) >= 2 * len(regions)
    for x in points:
        index, _ = mpc_pwa.lookup(regions, x)
        assert index is not None
        u_qp = box_qp(h, f @ x, -CW_A_MAX, CW_A_MAX)
        # The law is continuous across facets, so the region found gives the minimizer.
        assert np.max(np.abs(law(regions[index], x) - u_qp)) <= 1e-12 * CW_A_MAX


def hand_view(mode="accel", limit="box", hold=VBAR_HOLD, task=None):
    """An AgentView built without an environment."""
    goal = {"hold_point": hold, "capture_radius": 5.0, "capture_speed": 0.05}
    if task is not None:
        goal["task"] = task
    return AgentView(
        scenario_id="hand-made",
        dt=DT,
        max_steps=1000,
        n=N,
        action_mode=mode,
        max_accel=A_MAX,
        thrust_limit=limit,
        mass_kg=500.0,
        goal=goal,
        constraints=({"type": "keep_out", "radius": 50.0},),
    )


def config(horizon=1):
    return mpc_pwa.PiecewiseAffineConfig(horizon=horizon, box=BOX)


def command(agent, view, x):
    return command_fraction(view, agent.act(np.asarray(x, dtype=float), dict(INFO))) * A_MAX


def full_qp_first_move(view, x0, horizon):
    """First move of the 6-state condensed hold-point QP, by bounded-variable least squares.

    Stage cost x'Qx + u'Ru + 2x'Su on the deviation from the hold point (the zero-order-hold
    sampled cost of the default LQR weights), DARE terminal cost, |u_i| <= max_accel on the
    absolute input.
    """
    regulator = LQRConfig()
    phi, gamma = cw_discrete(view.n, view.dt)
    a, b = cw_system(view.n)
    qc = np.diag([regulator.pos_scale_m**-2] * 3 + [regulator.vel_scale_mps**-2] * 3)
    rc = np.eye(3) / view.max_accel**2
    q, r, s = sampled_data_cost(a, b, qc, rc, view.dt)
    _, p = dlqr(phi, gamma, q, r, s)
    xs = np.concatenate([view.goal["hold_point"], np.zeros(3)])
    us = steady_state_input(phi, gamma, xs)
    steps = horizon + 1
    sx = np.zeros((6 * steps, 6))
    su = np.zeros((6 * steps, 3 * horizon))
    qbar = np.zeros((6 * steps, 6 * steps))
    sbar = np.zeros((6 * steps, 3 * horizon))
    for k in range(steps):
        sx[6 * k : 6 * k + 6] = np.linalg.matrix_power(phi, k)
        for j in range(k):
            su[6 * k : 6 * k + 6, 3 * j : 3 * j + 3] = (
                np.linalg.matrix_power(phi, k - 1 - j) @ gamma
            )
        if k < horizon:
            qbar[6 * k : 6 * k + 6, 6 * k : 6 * k + 6] = q
            sbar[6 * k : 6 * k + 6, 3 * k : 3 * k + 3] = s
    qbar[-6:, -6:] = p
    h = 2 * (su.T @ qbar @ su + np.kron(np.eye(horizon), r) + su.T @ sbar + sbar.T @ su)
    f = 2 * (su.T @ qbar @ sx + sbar.T @ sx)
    lo = np.tile(-view.max_accel - us, horizon)
    hi = np.tile(view.max_accel - us, horizon)
    return us + box_qp(h, f @ (np.asarray(x0, dtype=float) - xs), lo, hi)[:3]


def test_config_is_frozen_and_keyword_only():
    with pytest.raises(TypeError):
        mpc_pwa.PiecewiseAffineConfig(LQRConfig())
    with pytest.raises(dataclasses.FrozenInstanceError):
        config().horizon = 3


@pytest.mark.parametrize(
    "fields", [{"horizon": 0}, {"box": (300.0, 600.0, 0.0, 1.0, 1.0, 1.0)}, {"box": (1.0,) * 5}]
)
def test_config_refuses_an_empty_horizon_and_a_degenerate_box(fields):
    with pytest.raises(ValueError):
        mpc_pwa.PiecewiseAffineConfig(**fields)


@pytest.mark.parametrize("hold", [VBAR_HOLD, RBAR_HOLD])
def test_command_at_the_hold_point_at_rest_is_the_steady_hold_input(hold):
    view = hand_view(hold=hold)
    agent = mpc_pwa.PiecewiseAffineMPCAgent(config())
    agent.reset(view, np.random.default_rng(0))
    phi, gamma = cw_discrete(N, DT)
    xs = np.concatenate([hold, np.zeros(3)])
    us = steady_state_input(phi, gamma, xs)
    # The R-bar hold needs 3.7e-4 m/s^2 radially; the V-bar hold needs none.
    assert np.max(np.abs(command(agent, view, xs) - us)) <= 1e-12 * A_MAX


@pytest.mark.parametrize("hold", [VBAR_HOLD, RBAR_HOLD])
def test_command_equals_the_lqr_agent_where_no_input_saturates(hold):
    view = hand_view(hold=hold)
    agent = mpc_pwa.PiecewiseAffineMPCAgent(config(horizon=2))
    agent.reset(view, np.random.default_rng(0))
    lqr = LQRAgent(LQRConfig())
    lqr.reset(view)
    xs = np.concatenate([hold, np.zeros(3)])
    # Offsets under a metre ask for corrections under 2e-4 m/s^2, far from every input limit.
    for dev in ([0.5, -0.5, 0.3, 0.0, 0.0, 0.0], [-0.2, 0.4, -0.1, 1e-4, -1e-4, 2e-4]):
        x = xs + np.array(dev)
        u = command(agent, view, x)
        assert np.max(np.abs(u - command(lqr, view, x))) <= 1e-12 * A_MAX
        assert agent.log[-1]["level"] == 0


@pytest.mark.parametrize("hold", [VBAR_HOLD, RBAR_HOLD])
def test_command_equals_the_first_move_of_the_full_qp_at_50_states_in_the_box(hold):
    view = hand_view(hold=hold)
    agent = mpc_pwa.PiecewiseAffineMPCAgent(config(horizon=2))
    agent.reset(view, np.random.default_rng(0))
    xs = np.concatenate([hold, np.zeros(3)])
    rng = np.random.default_rng(10)
    saturated = 0
    for _ in range(50):
        # Scaled draws: the whole box saturates some input at every state, a thousandth of it at
        # none, and the mix exercises both kinds of region.
        x = xs + rng.uniform(-1.0, 1.0, 6) * np.array(BOX) * 10 ** rng.uniform(-3.0, 0.0)
        u = command(agent, view, x)
        u_qp = full_qp_first_move(view, x, horizon=2)
        # Independent computation on the coupled 6-state problem: the decoupled laws agree to
        # 1.4e-15 a_max at worst over both holds.
        assert np.max(np.abs(u - u_qp)) <= 1e-12 * A_MAX
        saturated += np.any(np.abs(u_qp) >= A_MAX * (1 - 1e-12))
    assert 0 < saturated < 50


def test_outside_the_box_the_command_is_the_clipped_lqr_law_and_is_logged():
    view = hand_view()
    agent = mpc_pwa.PiecewiseAffineMPCAgent(config())
    agent.reset(view, np.random.default_rng(0))
    lqr = LQRAgent(LQRConfig())
    lqr.reset(view)
    # 700 m along track from the hold point, past the 600 m of the box; the LQR law asks for
    # 0.251 m/s^2 along track, so the clip is exercised.
    x = np.array([0.0, 600.0, 0.0, 0.0, 0.0, 0.0])
    u = command(agent, view, x)
    assert np.max(np.abs(u - command(lqr, view, x))) <= 1e-12 * A_MAX
    assert abs(u[1]) == pytest.approx(A_MAX, rel=1e-12)
    assert agent.log[-1]["level"] == 2
    assert agent.log[-1]["regions"] is None
    assert agent.fallbacks[2] == 1 and agent.fallbacks[0] == 0
    command(agent, view, np.array([0.0, -50.0, 0.0, 0.0, 0.0, 0.0]))
    assert agent.log[-1]["level"] == 0
    assert len(agent.log[-1]["regions"]) == 2
    assert agent.fallbacks[0] == 1 and len(agent.log) == 2


def test_cross_track_state_outside_the_box_alone_falls_back_to_the_clipped_lqr_law():
    view = hand_view()
    agent = mpc_pwa.PiecewiseAffineMPCAgent(config())
    agent.reset(view, np.random.default_rng(0))
    lqr = LQRAgent(LQRConfig())
    lqr.reset(view)
    # In plane half a metre from the hold point, inside the box; 400 m cross-track, past its
    # 300 m. The LQR law asks for -0.143 m/s^2 cross-track, so the command is -a_max there.
    x = np.array([0.5, -100.5, 400.0, 0.0, 0.0, 0.0])
    u = command(agent, view, x)
    assert np.max(np.abs(u - command(lqr, view, x))) <= 1e-12 * A_MAX
    assert u[2] == pytest.approx(-A_MAX, rel=1e-12)
    assert agent.log[-1]["level"] == 2
    assert agent.log[-1]["regions"] is None
    assert agent.fallbacks == [0, 0, 1]


def test_in_plane_state_outside_the_box_falls_back_while_cross_track_saturates_inside():
    view = hand_view()
    agent = mpc_pwa.PiecewiseAffineMPCAgent(config())
    agent.reset(view, np.random.default_rng(0))
    lqr = LQRAgent(LQRConfig())
    lqr.reset(view)
    # 650 m along track, past the 600 m of the box; 250 m cross-track, inside its 300 m, where
    # the LQR law asks for -0.089 m/s^2 and both axes clip to a_max.
    x = np.array([0.0, 550.0, 250.0, 0.0, 0.0, 0.0])
    u = command(agent, view, x)
    assert np.max(np.abs(u - command(lqr, view, x))) <= 1e-12 * A_MAX
    assert u[1] == pytest.approx(-A_MAX, rel=1e-12)
    assert u[2] == pytest.approx(-A_MAX, rel=1e-12)
    assert agent.log[-1]["level"] == 2
    assert agent.log[-1]["regions"] is None
    assert agent.fallbacks == [0, 0, 1]


def test_a_ball_thrust_layout_is_refused_at_reset():
    with pytest.raises(ValueError, match="ball"):
        mpc_pwa.PiecewiseAffineMPCAgent(config()).reset(hand_view(limit="ball"))


def test_an_observation_with_pending_commands_is_refused_at_reset():
    # The regions are built on the six-state model; a delay would need the pending commands as
    # parameters.
    pending = tuple(f"pending_0_{c}" for c in "txyz")
    view = dataclasses.replace(hand_view(), observation=hand_view().observation + pending)
    with pytest.raises(ValueError, match="pending"):
        mpc_pwa.PiecewiseAffineMPCAgent(config()).reset(view)


def test_a_rate_weight_is_refused_at_reset():
    regulator = LQRConfig(rate_weight=0.1)
    agent = mpc_pwa.PiecewiseAffineMPCAgent(
        mpc_pwa.PiecewiseAffineConfig(regulator=regulator, horizon=1, box=BOX)
    )
    with pytest.raises(ValueError, match="rate weight"):
        agent.reset(hand_view())


@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_discrete_and_impulsive_action_modes_are_refused_at_reset(mode):
    view = ApproachEnv(ApproachConfig(action_mode=mode)).agent_view()
    with pytest.raises(ValueError, match="takes no continuous command"):
        mpc_pwa.PiecewiseAffineMPCAgent(config()).reset(view, np.random.default_rng(0))


def hold_view(**goal):
    """The approach-vbar view with its goal entries replaced; None removes an entry."""
    view = make("approach-vbar").agent_view()
    merged = {**view.goal, **goal}
    return dataclasses.replace(view, goal={k: v for k, v in merged.items() if v is not None})


@pytest.mark.parametrize("task", ["dock", "inspect", "depart"])
def test_a_task_that_is_not_a_hold_is_refused_at_reset(task):
    with pytest.raises(ValueError, match=f"cannot fly the {task} task"):
        mpc_pwa.PiecewiseAffineMPCAgent(config()).reset(hold_view(task=task))


def test_a_hold_goal_without_a_hold_point_is_refused_at_reset():
    with pytest.raises(ValueError, match="this goal has none"):
        mpc_pwa.PiecewiseAffineMPCAgent(config()).reset(hold_view(hold_point=None))


def test_commands_are_equal_in_fractions_in_every_continuous_mode():
    views = {
        m: ApproachEnv(ApproachConfig(action_mode=m)).agent_view()
        for m in ("normalized", "accel", "thrust")
    }
    rng = np.random.default_rng(3)
    for _ in range(3):
        obs = np.array([*VBAR_HOLD, 0.0, 0.0, 0.0]) + rng.uniform(-1.0, 1.0, 6) * np.array(BOX)
        fractions = {}
        for m, view in views.items():
            agent = mpc_pwa.PiecewiseAffineMPCAgent(config())
            agent.reset(view, np.random.default_rng(0))
            fractions[m] = command_fraction(view, agent.act(obs.copy(), dict(INFO)))
        ref = fractions["normalized"]
        assert np.max(np.abs(ref)) > 0.0
        for m in ("accel", "thrust"):
            assert np.max(np.abs(fractions[m] - ref)) <= 1e-12 * np.max(np.abs(ref))


A_START = (60.0, -300.0, -20.0, 0.01, 0.0, 0.0)
B_START = (0.0, -450.0, 30.0, 0.0, 0.0, 0.0)
STEPS = 30


def actions(agent, start):
    env = ApproachEnv(ApproachConfig(start_state=start, max_steps=STEPS))
    return run_episode(env, agent, seed=11, agent_seed=12)["actions"]


def test_reset_wipes_the_previous_episode():
    agent = mpc_pwa.PiecewiseAffineMPCAgent(config())
    actions(agent, A_START)
    reused = actions(agent, B_START)
    fresh = actions(mpc_pwa.PiecewiseAffineMPCAgent(config()), B_START)
    assert np.array_equal(reused, fresh)
    assert len(agent.log) == STEPS
    assert sum(agent.fallbacks) == STEPS


def test_two_agents_from_one_config_on_interleaved_environments_match_separate_runs():
    starts = (A_START, B_START)
    envs = [ApproachEnv(ApproachConfig(start_state=s, max_steps=STEPS)) for s in starts]
    agents = [mpc_pwa.PiecewiseAffineMPCAgent(config()) for _ in starts]
    obs, logs = [], [[], []]
    for env, agent in zip(envs, agents, strict=True):
        o, _ = env.reset(seed=11)
        agent.reset(env.agent_view(), agent_generator(12))
        obs.append(o)
    for _ in range(STEPS):
        for i, (env, agent) in enumerate(zip(envs, agents, strict=True)):
            a = agent.act(obs[i], env.agent_info())
            logs[i].append(a)
            obs[i], *_ = env.step(a)
    for i, start in enumerate(starts):
        alone = actions(mpc_pwa.PiecewiseAffineMPCAgent(config()), start)
        assert np.array_equal(np.array(logs[i]), np.array(alone))


class ReadLog(dict):
    """An info mapping that records which keys are read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


def test_agent_runs_on_a_hand_made_view_and_reads_only_the_time():
    view = hand_view(mode="normalized")
    obs = np.array([5.0, -380.0, 3.0, 0.01, 0.0, -0.002])
    agent = mpc_pwa.PiecewiseAffineMPCAgent(config())
    agent.reset(view)
    plain = agent.act(obs.copy(), {"time_s": 100.0})
    agent.reset(view)
    poisoned = ReadLog(time_s=100.0, true_state=np.full(6, 1e6), dv_used_mps=1e6)
    action = agent.act(obs.copy(), poisoned)
    assert poisoned.read <= {"time_s"}
    assert np.array_equal(action, plain)
    assert action.shape == (3,) and np.all(np.isfinite(action))


def test_recorded_episode_keeps_one_log_entry_per_decision():
    start = (0.0, -400.0, 30.0, 0.0, 0.0, 0.0)
    ep = episode(mpc_pwa.PiecewiseAffineMPCAgent(config()), start_state=start, max_steps=15)
    assert len(ep.agent_log) == len(ep.records) == 15
    assert {e["level"] for e in ep.agent_log} <= {0, 2}
