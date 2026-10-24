import dataclasses
import math

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from scipy.optimize import OptimizeResult, minimize

import vbar.solvers.ipm
from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.solvers import (
    HillScaling,
    HorizonSolution,
    IPMSettings,
    StageQP,
    riccati_solve,
    solve_horizon,
)

N_500 = mean_motion(EARTH.radius + 500e3)
NX, NU = 6, 3
NZ = NX + NU
A_MAX = 0.02
STATUSES = {"converged", "iteration_cap", "infeasible_certified", "solver_failed"}


def box_rows(n, a_max=A_MAX):
    """|u_i| <= a_max on every stage as rows C_k z_k >= e_k with zero state columns."""
    c = np.zeros((n, 2 * NU, NZ))
    c[:, :NU, NX:] = np.eye(NU)
    c[:, NU:, NX:] = -np.eye(NU)
    return c, np.full((n, 2 * NU), -a_max)


def approach_qp(n, dt=20.0, start=-150.0, hold=-100.0, rows=True, a_max=A_MAX, terminal=None):
    """V-bar approach from rest at `start` to a hold point at `hold` (both m along y).

    Stage cost 1/2 (x - x_h)^T Q_x (x - x_h) + 1/2 u^T R u without the constant term, a stiffer
    terminal cost, and the thrust box as stage rows.
    """
    a, b = cw_discrete(N_500, np.full(n, dt))
    x_hold = np.array([0.0, hold, 0.0, 0.0, 0.0, 0.0])
    qx = np.diag([1e-2] * 3 + [1.0] * 3)
    p_n = np.diag([1.0] * 3 + [100.0] * 3)
    q = np.zeros((NZ, NZ))
    q[:NX, :NX] = qx
    q[NX:, NX:] = 1e2 * np.eye(NU)
    q_lin = np.zeros(NZ)
    q_lin[:NX] = -qx @ x_hold
    c, e = box_rows(n, a_max) if rows else (None, None)
    c_n, e_n = terminal if terminal is not None else (None, None)
    return StageQP(
        x0=np.array([0.0, start, 0.0, 0.0, 0.0, 0.0]),
        A=a,
        B=b,
        c=np.zeros((n, NX)),
        Q=np.repeat(q[None], n, axis=0),
        q=np.repeat(q_lin[None], n, axis=0),
        P=p_n,
        p=-p_n @ x_hold,
        C=c,
        e=e,
        C_N=c_n,
        e_N=e_n,
    )


def assemble(qp):
    """The horizon problem as one QP in w = (x_0, u_0, x_1, ..., x_N), built independently.

    min 1/2 w^T H w + g^T w subject to E w = f (x_0 = x0 and the dynamics) and G w >= h
    (stage rows in stage order, then terminal rows). Returns H, g, E, f, G, h.
    """
    n, nx, nu = qp.B.shape
    nz = nx + nu
    nw = n * nz + nx
    h_mat = np.zeros((nw, nw))
    g = np.zeros(nw)
    for k in range(n):
        h_mat[k * nz : (k + 1) * nz, k * nz : (k + 1) * nz] = qp.Q[k]
        g[k * nz : (k + 1) * nz] = qp.q[k]
    h_mat[n * nz :, n * nz :] = qp.P
    g[n * nz :] = qp.p
    e_mat = np.zeros(((n + 1) * nx, nw))
    f = np.zeros((n + 1) * nx)
    e_mat[:nx, :nx] = np.eye(nx)
    f[:nx] = qp.x0
    for k in range(n):
        rows = slice((k + 1) * nx, (k + 2) * nx)
        e_mat[rows, k * nz : k * nz + nx] = qp.A[k]
        e_mat[rows, k * nz + nx : (k + 1) * nz] = qp.B[k]
        e_mat[rows, (k + 1) * nz : (k + 1) * nz + nx] = -np.eye(nx)
        f[rows] = -qp.c[k]
    g_rows, h_rows = [], []
    if qp.C is not None:
        for k in range(n):
            block = np.zeros((qp.C.shape[1], nw))
            block[:, k * nz : (k + 1) * nz] = qp.C[k]
            g_rows.append(block)
            h_rows.append(qp.e[k])
    if qp.C_N is not None:
        block = np.zeros((qp.C_N.shape[0], nw))
        block[:, n * nz :] = qp.C_N
        g_rows.append(block)
        h_rows.append(qp.e_N)
    g_mat = np.vstack(g_rows) if g_rows else np.zeros((0, nw))
    h_vec = np.concatenate(h_rows) if h_rows else np.zeros(0)
    return h_mat, g, e_mat, f, g_mat, h_vec


def stack(qp, x, u):
    n, nx, nu = qp.B.shape
    w = np.zeros(n * (nx + nu) + nx)
    for k in range(n):
        w[k * (nx + nu) : k * (nx + nu) + nx] = x[k]
        w[k * (nx + nu) + nx : (k + 1) * (nx + nu)] = u[k]
    w[n * (nx + nu) :] = x[n]
    return w


def row_count(qp):
    stage = 0 if qp.C is None else qp.C.shape[0] * qp.C.shape[1]
    return stage + (0 if qp.C_N is None else qp.C_N.shape[0])


def row_multipliers(sol, qp):
    """Row duals flattened in stage order, then the terminal rows.

    Read as the leading entries of the flattened `multipliers`, which covers both an (N, m) array
    and a flat array of stage then terminal duals.
    """
    return np.asarray(sol.multipliers, dtype=float).ravel()[: row_count(qp)]


def kkt_residuals(qp, sol):
    """Relative stationarity, absolute primal infeasibility and complementarity of a solution.

    The dynamics multipliers are not reported, so stationarity is the least-squares residual of
    H w + g - G^T lam + E^T nu = 0 over nu, relative to the size of the gradient terms.
    """
    h_mat, g, e_mat, f, g_mat, h_vec = assemble(qp)
    w = stack(qp, sol.x, sol.u)
    lam = row_multipliers(sol, qp) if len(h_vec) else np.zeros(0)
    grad = h_mat @ w + g - g_mat.T @ lam
    nu, *_ = np.linalg.lstsq(e_mat.T, -grad, rcond=None)
    stationarity = np.linalg.norm(grad + e_mat.T @ nu) / max(
        1.0, np.linalg.norm(h_mat @ w + g), np.linalg.norm(g_mat.T @ lam)
    )
    slack = g_mat @ w - h_vec
    primal = max(
        float(np.max(np.abs(e_mat @ w - f))),
        float(np.max(-slack, initial=0.0)),
    )
    complementarity = float(np.max(np.abs(lam * slack), initial=0.0))
    return stationarity, primal, complementarity, lam


def relative_kkt_residuals(qp, sol):
    """Largest KKT residuals of a solution in the scale-free form the IPM stops on.

    Each stationarity and equality component is divided by the sum of the magnitudes of the terms
    it sums and a row's violation by sum_j |G_ij w_j| + |h_i|. A row is complementary to the
    smaller of that relative slack and its force,
    lam_i max_j G_ij^2 / (|G_ij| sigma_j + H_jj sum_l |G_il w_l|), with sigma_j the cost and
    dynamics terms of component j and x_0 excluded (docs/solvers.md). The fallbacks for unused
    axes and rounding-only rows are left out. The dynamics multipliers are the least-squares
    ones, as in `kkt_residuals`.
    """
    h_mat, g, e_mat, f, g_mat, h_vec = assemble(qp)
    nx = qp.A.shape[1]
    w = stack(qp, sol.x, sol.u)
    lam = row_multipliers(sol, qp)
    grad = h_mat @ w + g - g_mat.T @ lam
    nu, *_ = np.linalg.lstsq(e_mat.T, -grad, rcond=None)
    sigma = np.abs(h_mat) @ np.abs(w) + np.abs(g) + np.abs(e_mat.T) @ np.abs(nu)
    stationarity = np.max(np.abs(grad + e_mat.T @ nu) / (sigma + np.abs(g_mat.T) @ np.abs(lam)))
    equality = np.abs(e_mat @ w - f) / (np.abs(e_mat) @ np.abs(w) + np.abs(f))
    g_abs = np.abs(g_mat)
    terms = g_abs @ np.abs(w)
    slack = g_mat @ w - h_vec
    row_scale = terms + np.abs(h_vec)
    primal = max(np.max(equality), np.max(-slack / row_scale, initial=0.0))
    resisted = g_abs * sigma + np.diagonal(h_mat) * terms[:, None]
    force = np.divide(g_abs**2, resisted, out=np.zeros(g_abs.shape), where=g_abs > 0.0)
    force[:, :nx] = 0.0
    complementarity = np.max(
        np.minimum(np.abs(slack) / row_scale, lam * np.max(force, axis=1)), initial=0.0
    )
    return float(stationarity), float(primal), float(complementarity), lam


def objective(qp, x, u):
    h_mat, g, *_ = assemble(qp)
    w = stack(qp, x, u)
    return 0.5 * w @ h_mat @ w + g @ w


def relative(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def condensed_reference(qp):
    """Independent optimum by SLSQP over the inputs alone, the states eliminated by simulation."""
    n = qp.B.shape[0]

    def states(u_flat):
        u = u_flat.reshape(n, NU)
        x = [qp.x0]
        for k in range(n):
            x.append(qp.A[k] @ x[-1] + qp.B[k] @ u[k] + qp.c[k])
        return np.array(x), u

    w0 = stack(qp, *states(np.zeros(n * NU)))
    t = np.column_stack([stack(qp, *states(col)) - w0 for col in np.eye(n * NU)])
    h_mat, g, _, _, g_mat, h_vec = assemble(qp)
    # SLSQP stops on an absolute change in the objective, so the inputs are measured in units of
    # the thrust limit and the objective in units of its largest Hessian entry, without the
    # constant term.
    t = t * A_MAX
    hu = t.T @ h_mat @ t
    weight = np.abs(hu).max()
    hu = hu / weight
    gu = t.T @ (h_mat @ w0 + g) / weight
    const = 0.5 * w0 @ h_mat @ w0 + g @ w0
    gu_rows = g_mat @ t
    hu_rows = h_vec - g_mat @ w0
    found = minimize(
        lambda v: 0.5 * v @ hu @ v + gu @ v,
        np.zeros(n * NU),
        jac=lambda v: hu @ v + gu,
        constraints=[
            {"type": "ineq", "fun": lambda v: gu_rows @ v - hu_rows, "jac": lambda v: gu_rows}
        ],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    assert found.success, found.message
    return const + weight * found.fun, A_MAX * found.x.reshape(n, NU)


@pytest.fixture(scope="module")
def box_case():
    qp = approach_qp(10)
    return qp, solve_horizon(qp)


def test_unconstrained_horizon_equals_riccati_solve():
    qp = approach_qp(10, rows=False)
    sol = solve_horizon(qp)
    x, u, _ = riccati_solve(qp)
    assert sol.status == "converged"
    # One Newton step solves an equality-constrained QP exactly; 1e-9 allows the stopping test.
    assert relative(sol.u, u) < 1e-9
    assert relative(sol.x, x) < 1e-9


def test_inactive_rows_leave_the_riccati_solution_unchanged():
    # A 10 m/s^2 box is never touched by this approach (the unconstrained plan stays near
    # 0.11 m/s^2), so the rows only add a barrier term that vanishes at convergence.
    qp_free = approach_qp(10, rows=False)
    qp = approach_qp(10, a_max=10.0)
    sol = solve_horizon(qp)
    x, u, _ = riccati_solve(qp_free)
    assert sol.status == "converged"
    assert relative(sol.u, u) < 1e-9
    assert relative(sol.x, x) < 1e-9


def test_box_is_active_over_part_of_the_approach(box_case):
    qp, sol = box_case
    assert sol.status == "converged"
    saturated = np.isclose(np.abs(sol.u).max(axis=1), A_MAX, rtol=1e-6)
    assert saturated.any() and not saturated.all()


def test_box_constrained_solution_satisfies_kkt(box_case):
    qp, sol = box_case
    stationarity, primal, complementarity, lam = kkt_residuals(qp, sol)
    assert stationarity < 1e-7
    assert primal < 1e-7
    assert complementarity < 1e-7
    assert np.all(lam >= 0.0)
    assert np.all(np.abs(sol.u) <= A_MAX * (1 + 1e-7))


def test_box_constrained_objective_matches_an_independent_condensed_solve(box_case):
    qp, sol = box_case
    reference, _ = condensed_reference(qp)
    assert sol.objective == pytest.approx(objective(qp, sol.x, sol.u), rel=1e-9)
    assert sol.objective == pytest.approx(reference, rel=1e-8)


def test_box_constrained_solve_converges_within_forty_iterations(box_case):
    _, sol = box_case
    assert sol.iterations <= 40


def test_hundred_stage_approach_converges_within_forty_iterations():
    qp = approach_qp(100, start=-300.0)
    sol = solve_horizon(qp)
    assert sol.status == "converged"
    assert sol.iterations <= 40
    stationarity, primal, complementarity, _ = kkt_residuals(qp, sol)
    assert max(stationarity, primal, complementarity) < 1e-7


def unreachable_terminal_qp():
    # From rest at y = -150 m, 200 s under a 0.02 m/s^2 box moves the chaser at most about
    # 0.5 * 0.02 * 200^2 = 400 m along y, so y_N >= 5000 m is out of reach by a wide margin.
    c_n = np.zeros((1, NX))
    c_n[0, 1] = 1.0
    return approach_qp(10, terminal=(c_n, np.array([5000.0])))


def test_unreachable_terminal_row_is_certified_infeasible():
    qp = unreachable_terminal_qp()
    sol = solve_horizon(qp)
    assert sol.status == "infeasible_certified"
    assert sol.x is None and sol.u is None
    assert math.isnan(sol.objective)
    n = qp.B.shape[0]
    assert sol.conflicting
    for stage, row in sol.conflicting:
        assert 0 <= stage <= n
        assert 0 <= row < (qp.C_N.shape[0] if stage == n else qp.C.shape[1])


def test_infeasibility_certificate_verifies_by_farkas():
    qp = unreachable_terminal_qp()
    sol = solve_horizon(qp)
    assert sol.certificate is not None
    _, _, e_mat, f, g_mat, h_vec = assemble(qp)
    lam = np.asarray(sol.certificate, dtype=float).ravel()[: len(h_vec)]
    # Farkas: lam >= 0 and some nu with G^T lam + E^T nu = 0 and h^T lam + f^T nu > 0 rule out
    # every w with G w >= h and E w = f. nu is recomputed, so only the row multipliers count.
    assert np.all(lam >= 0.0) and lam.sum() > 0.0
    nu, *_ = np.linalg.lstsq(e_mat.T, -g_mat.T @ lam, rcond=None)
    scale = np.abs(lam).sum() * max(1.0, np.abs(g_mat).max())
    assert np.linalg.norm(g_mat.T @ lam + e_mat.T @ nu) <= 1e-7 * scale
    # The shortfall is thousands of metres, so the certificate's margin is far above rounding.
    assert h_vec @ lam + f @ nu > 1e-6 * np.abs(lam).sum() * np.abs(h_vec).max()
    # The box alone is feasible (zero thrust), so every valid certificate weights the terminal row.
    assert lam[-1] > 1e-6 * np.abs(lam).sum()


def corrupted_linprog(monkeypatch, corrupt):
    """Replace the LP solver seen by the interior-point module with one whose dual is altered."""
    exact = vbar.solvers.ipm.linprog

    def altered(*args, **kwargs):
        res = exact(*args, **kwargs)
        marginals = np.asarray(res.ineqlin.marginals, dtype=float).copy()
        res.ineqlin = OptimizeResult(res.ineqlin, marginals=corrupt(marginals))
        return res

    monkeypatch.setattr(vbar.solvers.ipm, "linprog", altered)


def wrong_sign(marginals):
    return -marginals


def inputs_not_cancelled(marginals):
    # The first stage row is the box face u_x >= -a_max, which has no state columns, so extra
    # weight on it leaves an input component of G^T lam + E^T nu that no nu can cancel.
    changed = marginals.copy()
    changed[0] -= 1.0 + np.abs(marginals).max()
    return changed


def margin_not_positive(marginals):
    # Equal weight on the two faces of one box pair cancels exactly in G^T lam with nu = 0, but
    # h^T lam = -2 a_max < 0: a combination of feasible rows, not a certificate.
    changed = np.zeros_like(marginals)
    changed[0] = changed[NU] = -1.0
    return changed


@pytest.mark.parametrize("corrupt", [wrong_sign, inputs_not_cancelled, margin_not_positive])
def test_unverified_phase_one_dual_is_not_reported_as_a_certificate(monkeypatch, corrupt):
    corrupted_linprog(monkeypatch, corrupt)
    sol = solve_horizon(unreachable_terminal_qp())
    assert sol.status == "solver_failed"
    assert sol.certificate is None
    assert sol.conflicting == ()
    assert sol.x is None and sol.u is None


def test_forty_stage_approach_at_ten_second_steps_converges():
    qp = approach_qp(40, dt=10.0, start=-300.0)
    sol = solve_horizon(qp)
    assert sol.status == "converged"
    # Measured at 14 iterations from the zero-thrust rollout: a change that quadruples the count
    # runs past 60.
    assert sol.iterations <= 60
    stationarity, primal, complementarity, lam = kkt_residuals(qp, sol)
    assert max(stationarity, primal, complementarity) < 1e-7
    assert np.all(lam >= 0.0)


def test_approach_from_ahead_through_the_target_converges():
    # Zero thrust meets every row, so the problem is feasible and a stalled primal residual is
    # no reason to stop: from 300 m ahead to the hold point 100 m behind in 400 s.
    qp = approach_qp(40, dt=10.0, start=300.0, hold=-100.0)
    sol = solve_horizon(qp)
    assert sol.status == "converged"
    stationarity, primal, complementarity, _ = kkt_residuals(qp, sol)
    assert max(stationarity, primal, complementarity) < 1e-7


def test_iteration_cap_returns_no_plan(box_case):
    qp, _ = box_case
    sol = solve_horizon(qp, IPMSettings(max_iter=3))
    assert sol.status == "iteration_cap"
    assert sol.iterations == 3
    assert sol.x is None and sol.u is None
    assert math.isnan(sol.objective)
    assert np.isfinite(sol.residual_primal) and np.isfinite(sol.residual_dual)


def test_solution_in_hill_units_equals_the_si_solution(box_case):
    qp, sol = box_case
    scaling = HillScaling(length=100.0, n=N_500)
    scaled = solve_horizon(scaling.stage(qp))
    assert scaled.status == "converged"
    assert relative(scaling.state_si(scaled.x), sol.x) < 1e-9
    assert relative(scaling.accel_si(scaled.u), sol.u) < 1e-9


def test_same_input_gives_bitwise_identical_output(box_case):
    qp, sol = box_case
    again = solve_horizon(qp)
    assert isinstance(again, HorizonSolution)
    assert again.status == sol.status and again.iterations == sol.iterations
    assert np.array_equal(again.x, sol.x) and np.array_equal(again.u, sol.u)
    assert np.array_equal(again.multipliers, sol.multipliers)
    assert again.objective == sol.objective
    assert (again.residual_primal, again.residual_dual, again.gap) == (
        sol.residual_primal,
        sol.residual_dual,
        sol.gap,
    )


def test_warm_start_that_breaks_the_dynamics_converges_to_the_cold_start_plan(box_case):
    qp, sol = box_case
    rng = np.random.default_rng(21)
    # States far from any rollout of u: the start leaves a dynamics residual of hundreds of
    # metres, so only the infeasible-start steps can remove it.
    x_bad = 50.0 * sol.x * (1.0 + rng.uniform(-0.1, 0.1, sol.x.shape))
    warm = solve_horizon(qp, warm=(x_bad, sol.u))
    assert warm.status == "converged"
    stationarity, primal, complementarity, lam = kkt_residuals(qp, warm)
    assert max(stationarity, primal, complementarity) < 1e-7
    assert np.all(lam >= 0.0)
    assert relative(warm.u, sol.u) < 1e-8
    assert relative(warm.x, sol.x) < 1e-8


def test_warm_start_from_the_solution_converges_to_it(box_case):
    qp, sol = box_case
    warm = solve_horizon(qp, warm=(sol.x, sol.u))
    assert warm.status == "converged"
    assert relative(warm.u, sol.u) < 1e-7


@pytest.mark.parametrize("slack", [0.0, 1e-9])
def test_rows_touching_the_optimum_with_zero_multipliers_converge_to_it(slack):
    qp_free = approach_qp(10, rows=False)
    x_free, u_free, _ = riccati_solve(qp_free)
    n = qp_free.B.shape[0]
    # Each input component is bounded above at its unconstrained optimum plus `slack`, so the
    # optimum sits on (or 1e-9 m/s^2 inside) every row at once with zero multipliers, the weakly
    # active rows a receding-horizon controller meets whenever a limit is just reached.
    c = np.zeros((n, NU, NZ))
    c[:, :, NX:] = -np.eye(NU)
    qp = dataclasses.replace(qp_free, C=c, e=-(u_free + slack))
    sol = solve_horizon(qp)
    assert sol.status == "converged"
    stationarity, primal, complementarity, _ = kkt_residuals(qp, sol)
    assert max(stationarity, primal, complementarity) < 1e-6
    assert relative(sol.u, u_free) < 1e-6
    assert relative(sol.x, x_free) < 1e-6


def test_a_weakly_active_row_on_the_box_limited_approach_leaves_its_plan(box_case):
    qp, sol = box_case
    assert sol.status == "converged"
    # A stage and axis where the box is slack; the new row caps that input at its optimal value,
    # so it is exactly tight at the plan with a zero multiplier.
    free = np.argwhere(np.abs(sol.u) < 0.5 * A_MAX)
    k, i = free[len(free) // 2]
    n = qp.B.shape[0]
    extra_c = np.zeros((n, 1, NZ))
    extra_e = np.full((n, 1), -1.0)
    extra_c[k, 0, NX + i] = -1.0
    extra_e[k, 0] = -sol.u[k, i]
    tight = dataclasses.replace(
        qp,
        C=np.concatenate([qp.C, extra_c], axis=1),
        e=np.concatenate([qp.e, extra_e], axis=1),
    )
    again = solve_horizon(tight)
    assert again.status == "converged"
    assert relative(again.u, sol.u) < 1e-6
    assert relative(again.x, sol.x) < 1e-6
    stationarity, primal, complementarity, _ = kkt_residuals(tight, again)
    assert max(stationarity, primal, complementarity) < 1e-6


def small_stage_problem(seed, n, nx, nu, m, margin):
    """Random stage problem whose rows hold with slack at the unconstrained optimum."""
    rng = np.random.default_rng(seed)
    nz = nx + nu
    q = np.zeros((n, nz, nz))
    for k in range(n):
        root = rng.normal(size=(nz, nz))
        q[k] = root @ root.T
        q[k, nx:, nx:] += 0.1 * np.eye(nu)
        q[k] = 0.5 * (q[k] + q[k].T)
    root = rng.normal(size=(nx, nx))
    fields = dict(
        x0=rng.normal(size=nx),
        A=rng.normal(size=(n, nx, nx)) * 0.5 + np.eye(nx),
        B=rng.normal(size=(n, nx, nu)),
        c=rng.normal(size=(n, nx)) * 0.1,
        Q=q,
        q=rng.normal(size=(n, nz)),
        P=0.5 * (root @ root.T + (root @ root.T).T),
        p=rng.normal(size=nx),
    )
    free = StageQP(**fields, C=None, e=None, C_N=None, e_N=None)
    x, u, _ = riccati_solve(free)
    z = np.concatenate([x[:-1], u], axis=1)
    c = rng.normal(size=(n, m, nz))
    e = np.einsum("kij,kj->ki", c, z) - margin * (1.0 + np.abs(np.einsum("kij,kj->ki", c, z)))
    return StageQP(**fields, C=c, e=e, C_N=None, e_N=None), u


@st.composite
def small_stage_problems(draw):
    return small_stage_problem(
        seed=draw(st.integers(0, 2**32 - 1)),
        n=draw(st.integers(2, 6)),
        nx=draw(st.integers(1, 4)),
        nu=draw(st.integers(1, 3)),
        m=draw(st.integers(1, 3)),
        margin=draw(st.floats(0.05, 1.0, allow_nan=False, allow_subnormal=False)),
    )


@settings(max_examples=25, deadline=None)
@given(small_stage_problems())
# Converged with absolute complementarity 1.1e-6: the multipliers of these inactive rows are
# judged per row against the row's force, not as products lam_i s_i.
@example(small_stage_problem(seed=286, n=4, nx=4, nu=2, m=1, margin=0.71875))
def test_random_small_problems_converge_to_kkt_points(case):
    qp, u_free = case
    sol = solve_horizon(qp)
    assert sol.status == "converged"
    stationarity, primal, complementarity, lam = relative_kkt_residuals(qp, sol)
    # The default tol_feas and tol_gap. Stationarity is measured with the least-squares dynamics
    # multipliers, not the solver's, which moves it up to 2.2 tol_feas; complementarity uses the
    # slack C z - e, not the solver's s. A solver stopping at 10 tol_gap exceeds 6 tol_gap here.
    tol = 1e-8
    assert max(stationarity, primal) <= 4 * tol
    assert complementarity <= 2 * tol
    assert np.all(lam >= 0.0)
    # The rows hold with slack at the unconstrained optimum, so it is also the constrained one.
    assert np.max(np.abs(sol.u - u_free)) < 1e-6 * max(1.0, np.max(np.abs(u_free)))


def test_shape_errors_raise_value_error():
    qp = approach_qp(4)
    with pytest.raises(ValueError):
        solve_horizon(qp, warm=(np.zeros((3, NX)), np.zeros((4, NU))))


def scaled_costs(qp, k):
    """The same problem with every cost term (Q, q, P, p) multiplied by k."""
    return dataclasses.replace(qp, Q=k * qp.Q, q=k * qp.q, P=k * qp.P, p=k * qp.p)


@pytest.fixture(scope="module")
def forty_stage():
    qp = approach_qp(40, dt=10.0, start=-300.0)
    return qp, solve_horizon(qp)


@pytest.mark.parametrize("k", [1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6])
def test_solution_does_not_depend_on_the_cost_scale(forty_stage, k):
    # Scaling the objective changes no minimiser, so the stopping rule must not see the scale.
    qp, reference = forty_stage
    assert reference.status == "converged"
    sol = solve_horizon(scaled_costs(qp, k))
    assert sol.status == "converged"
    assert relative(sol.x, reference.x) < 1e-6
    assert relative(sol.u, reference.u) < 1e-6


def test_small_cost_scale_gives_the_same_plan_in_hill_units(forty_stage):
    qp = scaled_costs(forty_stage[0], 1e-6)
    si = solve_horizon(qp)
    scaling = HillScaling(length=100.0, n=N_500)
    hill = solve_horizon(scaling.stage(qp))
    assert si.status == hill.status == "converged"
    assert relative(scaling.state_si(hill.x), si.x) < 1e-6
    assert relative(scaling.accel_si(hill.u), si.u) < 1e-6


def disturbed_qp(d_y):
    """Ten 20 s stages from rest at y = -150 m to y_N >= 100 m under a constant along-track push.

    The push d_y (m/s^2) enters as the affine term c_k = B_k (0, d_y, 0) of an exact held step.
    """
    c_n = np.zeros((1, NX))
    c_n[0, 1] = 1.0
    qp = approach_qp(10, terminal=(c_n, np.array([100.0])))
    return dataclasses.replace(qp, c=qp.B @ np.array([0.0, d_y, 0.0]))


def test_affine_terms_decide_an_infeasibility_certificate():
    # Without the push the box reaches y_N >= 100 m; a -0.015 m/s^2 push leaves 0.005 m/s^2 of
    # along-track authority, about 100 m in 200 s, so the row is out of reach only through c.
    assert solve_horizon(disturbed_qp(0.0)).status == "converged"
    qp = disturbed_qp(-0.015)
    sol = solve_horizon(qp)
    assert sol.status == "infeasible_certified"
    _, _, e_mat, f, g_mat, h_vec = assemble(qp)
    # f carries -c_k in the dynamics rows, so this Farkas check includes the affine terms.
    assert np.any(f[NX:] != 0.0)
    lam = np.asarray(sol.certificate, dtype=float).ravel()[: len(h_vec)]
    assert np.all(lam >= 0.0) and lam.sum() > 0.0
    nu, *_ = np.linalg.lstsq(e_mat.T, -g_mat.T @ lam, rcond=None)
    scale = np.abs(lam).sum() * max(1.0, np.abs(g_mat).max())
    assert np.linalg.norm(g_mat.T @ lam + e_mat.T @ nu) <= 1e-7 * scale
    assert h_vec @ lam + f @ nu > 1e-6 * np.abs(lam).sum() * np.abs(h_vec).max()
