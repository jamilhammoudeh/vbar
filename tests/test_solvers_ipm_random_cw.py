import dataclasses
from fractions import Fraction

import numpy as np
import pytest

from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.solvers import HillScaling, StageQP, solve_horizon

# About 450 km altitude.
N_MEAN = 1.1e-3
NX, NU = 6, 3
NZ = NX + NU


def cw_horizon(n, dt):
    # One step repeated: the seeds below were drawn this way, and a vectorized sine can differ
    # from the scalar one in the last bit, which these solves are sensitive to.
    a, b = cw_discrete(N_MEAN, dt)
    return np.repeat(a[None], n, axis=0), np.repeat(b[None], n, axis=0)


def proximity_problem(rng):
    """A proximity-operation horizon in SI units that zero thrust meets strictly.

    Exact CW, 5 to 60 stages of 1 to 100 s, a thrust box that zero thrust meets, up to two
    position halfspaces that the zero-thrust drift meets with 0.1 to 100 m to spare and up to one
    terminal row it meets with margin. The box binds wherever the quadratic cost asks for more
    thrust than it allows.
    """
    dt = float(rng.choice([1.0, 5.0, 10.0, 20.0, 60.0, 100.0]))
    n = int(rng.integers(5, 61))
    a, b = cw_horizon(n, dt)
    x0 = np.concatenate(
        [
            rng.uniform(-1, 1, 3) * 10.0 ** rng.uniform(1, 3.3),
            rng.uniform(-1, 1, 3) * 10.0 ** rng.uniform(-2, 0),
        ]
    )
    hold = np.concatenate([rng.uniform(-1, 1, 3) * 10.0 ** rng.uniform(0, 2.5), np.zeros(3)])
    q_pos, q_vel = 10.0 ** rng.uniform(-4, 0), 10.0 ** rng.uniform(-2, 2)
    r = 10.0 ** rng.uniform(0, 6)
    q_x = np.diag([q_pos] * 3 + [q_vel] * 3)
    cost = np.zeros((NZ, NZ))
    cost[:NX, :NX] = q_x
    cost[NX:, NX:] = r * np.eye(NU)
    grad = np.zeros(NZ)
    grad[:NX] = -q_x @ hold
    p_n = q_x * float(rng.choice([1.0, 10.0, 100.0]))
    a_max = 10.0 ** rng.uniform(-3.5, -1.5)
    drift = np.empty((n + 1, NX))
    drift[0] = x0
    for k in range(n):
        drift[k + 1] = a[k] @ drift[k]
    m_pos = int(rng.integers(0, 3))
    c = np.zeros((n, 6 + m_pos, NZ))
    e = np.zeros((n, 6 + m_pos))
    c[:, :3, NX:] = np.eye(NU)
    c[:, 3:6, NX:] = -np.eye(NU)
    e[:, :6] = -a_max
    for j in range(m_pos):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        margin = 10.0 ** rng.uniform(-1, 2)
        c[:, 6 + j, :3] = normal
        e[:, 6 + j] = np.min(drift[:n, :3] @ normal) - margin
    c_n = e_n = None
    if rng.integers(0, 2):
        d = rng.normal(size=NX)
        c_n = d[None, :]
        e_n = np.array([d @ drift[n] - abs(d @ drift[n]) * 0.1 - 1.0])
    return StageQP(
        x0=x0,
        A=a,
        B=b,
        c=np.zeros((n, NX)),
        Q=np.repeat(cost[None], n, axis=0),
        q=np.repeat(grad[None], n, axis=0),
        P=p_n,
        p=-p_n @ hold,
        C=c,
        e=e,
        C_N=c_n,
        e_N=e_n,
    )


def random_cw_problem(rng):
    """CW dynamics with random convex costs, random rows and random units.

    1 to 15 stages of 1 ms to 1e4 s, positive semidefinite state costs, input Hessians with
    condition numbers up to 1e8, up to four stage rows and two terminal rows that a random plan
    meets, then states, inputs and costs rescaled by random factors (up to 1e3 per axis and 1e12
    on the costs), which change no minimiser.
    """
    rng.integers(0, 6)  # the problem family drawn by the sweep these seeds come from
    n = int(rng.integers(1, 16))
    a, b = cw_horizon(n, 10.0 ** rng.uniform(-3, 4))
    c_aff = rng.normal(size=(n, NX)) * (0.0 if rng.random() < 0.5 else rng.uniform(0, 1))
    q = np.zeros((n, NZ, NZ))
    for k in range(n):
        root = rng.normal(size=(NX, int(rng.integers(0, NX + 1))))
        r_cond = 10.0 ** rng.uniform(0, 8)
        rot, _ = np.linalg.qr(rng.normal(size=(NU, NU)))
        q[k, :NX, :NX] = root @ root.T
        q[k, NX:, NX:] = rot @ np.diag(np.geomspace(1.0, 1.0 / r_cond, NU)) @ rot.T
        q[k] = 0.5 * (q[k] + q[k].T)
    q_lin = rng.normal(size=(n, NZ))
    root = rng.normal(size=(NX, NX))
    p_n = root @ root.T
    p_lin = rng.normal(size=NX)
    x0 = rng.normal(size=NX) * 10.0 ** rng.uniform(-2, 2)
    u_ref = rng.normal(size=(n, NU)) * 10.0 ** rng.uniform(-2, 1)
    x_ref = np.empty((n + 1, NX))
    x_ref[0] = x0
    for k in range(n):
        x_ref[k + 1] = a[k] @ x_ref[k] + b[k] @ u_ref[k] + c_aff[k]
    m = int(rng.integers(0, 5))
    c = e = None
    if m:
        c = rng.normal(size=(n, m, NZ))
        if rng.random() < 0.5:
            c[:, 0, :NX] = 0.0
        if rng.random() < 0.3:
            c *= rng.random(size=c.shape) < 0.5
        values = np.einsum("kij,kj->ki", c, np.concatenate([x_ref[:-1], u_ref], axis=1))
        slack = np.abs(rng.normal(size=(n, m))) * np.abs(values).max()
        e = values - slack * 10.0 ** rng.uniform(-3, 0, size=(n, m))
    c_n = e_n = None
    m_n = int(rng.integers(0, 3))
    if m_n:
        c_n = rng.normal(size=(m_n, NX))
        values = c_n @ x_ref[-1]
        e_n = values - np.abs(rng.normal(size=m_n)) * np.abs(values).max() * 10.0 ** rng.uniform(
            -3, 0
        )
    d_x = 10.0 ** rng.uniform(-3, 3, NX) if rng.random() < 0.6 else np.ones(NX)
    d_u = 10.0 ** rng.uniform(-3, 3, NU) if rng.random() < 0.6 else np.ones(NU)
    k_cost = 10.0 ** rng.uniform(-12, 12)
    # New units x' = d_x x and u' = d_u u; z = t z'.
    t = np.concatenate([1.0 / d_x, 1.0 / d_u])
    return StageQP(
        x0=x0 * d_x,
        A=d_x[None, :, None] * a / d_x[None, None, :],
        B=d_x[None, :, None] * b / d_u[None, None, :],
        c=c_aff * d_x,
        Q=k_cost * q * t[None, :, None] * t[None, None, :],
        q=k_cost * q_lin * t,
        P=k_cost * p_n / d_x[:, None] / d_x[None, :],
        p=k_cost * p_lin / d_x,
        C=None if c is None else c * t[None, None, :],
        e=e,
        C_N=None if c_n is None else c_n / d_x[None, :],
        e_N=e_n,
    )


def flat_rows(qp):
    """Stage rows (N, m, nz) and terminal rows (mN, nx) with their bounds, empty when absent."""
    n = qp.B.shape[0]
    c = np.zeros((n, 0, NZ)) if qp.C is None else qp.C
    e = np.zeros((n, 0)) if qp.e is None else qp.e
    c_n = np.zeros((0, NX)) if qp.C_N is None else qp.C_N
    e_n = np.zeros(0) if qp.e_N is None else qp.e_N
    return c, e, c_n, e_n


def kkt_check(qp, x, u, lam):
    """Scale-free KKT residuals of a plan and its row multipliers, built without the solver.

    The dynamics multipliers are not taken from the solver or from a least-squares fit: the
    stationarity conditions in x_N, x_{N-1}, ..., x_1 fix them one stage at a time, so only the
    input components of stationarity remain. That recursion carries rounding from stage to stage,
    so stationarity is judged over the horizon, its largest component against the largest sum of
    term magnitudes. Each dynamics component and row is divided by the sum of the magnitudes of
    the terms it adds up, and each product lam_i |C_i z - e_i| by the sum of the magnitudes of
    the objective's terms, so no test depends on the units.
    """
    n = qp.B.shape[0]
    c, e, c_n, e_n = flat_rows(qp)
    m = c.shape[1]
    z = np.concatenate([x[:-1], u], axis=1)
    lam_z, lam_n = lam[: n * m].reshape(n, m), lam[n * m :]
    grad = np.einsum("kij,kj->ki", qp.Q, z) + qp.q - np.einsum("kij,ki->kj", c, lam_z)
    size = (
        np.einsum("kij,kj->ki", np.abs(qp.Q), np.abs(z))
        + np.abs(qp.q)
        + np.einsum("kij,ki->kj", np.abs(c), np.abs(lam_z))
    )
    # y[k] belongs to x_{k+1} = A_k x_k + B_k u_k + c_k.
    y = np.empty((n, NX))
    y[-1] = qp.P @ x[-1] + qp.p - c_n.T @ lam_n
    for k in range(n - 1, 0, -1):
        y[k - 1] = grad[k, :NX] + qp.A[k].T @ y[k]
    stationarity = grad[:, NX:] + np.einsum("kji,kj->ki", qp.B, y)
    stationarity_size = size[:, NX:] + np.einsum("kji,kj->ki", np.abs(qp.B), np.abs(y))
    dynamics = (
        np.einsum("kij,kj->ki", qp.A, x[:-1]) + np.einsum("kij,kj->ki", qp.B, u) + qp.c - x[1:]
    )
    dynamics_size = (
        np.einsum("kij,kj->ki", np.abs(qp.A), np.abs(x[:-1]))
        + np.einsum("kij,kj->ki", np.abs(qp.B), np.abs(u))
        + np.abs(qp.c)
        + np.abs(x[1:])
    )
    rows = np.concatenate([np.einsum("kij,kj->ki", c, z).ravel() - e.ravel(), c_n @ x[-1] - e_n])
    row_size = np.concatenate(
        [
            np.einsum("kij,kj->ki", np.abs(c), np.abs(z)).ravel() + np.abs(e).ravel(),
            np.abs(c_n) @ np.abs(x[-1]) + np.abs(e_n),
        ]
    )
    objective_size = (
        0.5 * np.einsum("ki,kij,kj->", np.abs(z), np.abs(qp.Q), np.abs(z))
        + np.sum(np.abs(qp.q * z))
        + 0.5 * np.abs(x[-1]) @ np.abs(qp.P) @ np.abs(x[-1])
        + np.sum(np.abs(qp.p * x[-1]))
    )
    return {
        "stationarity": float(np.max(np.abs(stationarity)) / np.max(stationarity_size)),
        "dynamics": float(np.max(np.abs(dynamics) / dynamics_size)),
        "initial_state": float(np.max(np.abs(x[0] - qp.x0))),
        "violation": float(np.max(-rows / row_size, initial=0.0)),
        "negative_multiplier": float(np.max(-lam, initial=0.0)),
        "complementarity": float(np.max(lam * np.abs(rows), initial=0.0) / objective_size),
        "active_rows": int(np.sum(rows <= 1e-6 * row_size)),
    }


def dense_problem(qp):
    """min 1/2 w^T H w + g^T w, E w = f, G w >= h over w = (x_0, u_0, x_1, ..., u_{N-1}, x_N)."""
    n = qp.B.shape[0]
    nw = n * NZ + NX
    h_mat = np.zeros((nw, nw))
    g = np.zeros(nw)
    e_mat = np.zeros(((n + 1) * NX, nw))
    f = np.zeros((n + 1) * NX)
    e_mat[:NX, :NX] = np.eye(NX)
    f[:NX] = qp.x0
    c, e, c_n, e_n = flat_rows(qp)
    m = c.shape[1]
    g_mat = np.zeros((n * m + c_n.shape[0], nw))
    for k in range(n):
        stage = slice(k * NZ, (k + 1) * NZ)
        h_mat[stage, stage] = qp.Q[k]
        g[stage] = qp.q[k]
        dyn = slice((k + 1) * NX, (k + 2) * NX)
        e_mat[dyn, k * NZ : k * NZ + NX] = qp.A[k]
        e_mat[dyn, k * NZ + NX : (k + 1) * NZ] = qp.B[k]
        e_mat[dyn, (k + 1) * NZ : (k + 1) * NZ + NX] = -np.eye(NX)
        f[dyn] = -qp.c[k]
        g_mat[k * m : (k + 1) * m, stage] = c[k]
    h_mat[n * NZ :, n * NZ :] = qp.P
    g[n * NZ :] = qp.p
    g_mat[n * m :, n * NZ :] = c_n
    return h_mat, g, e_mat, f, g_mat, np.concatenate([e.ravel(), e_n])


def exact_solve(k, rhs):
    """Gauss-Jordan elimination in rational arithmetic on the float entries, so no rounding."""
    size = len(rhs)
    rows = [
        [Fraction(float(v)) for v in row] + [Fraction(float(r))]
        for row, r in zip(k, rhs, strict=True)
    ]
    for col in range(size):
        pivot = next(i for i in range(col, size) if rows[i][col] != 0)
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for i in range(size):
            if i != col and rows[i][col] != 0:
                ratio = rows[i][col] / rows[col][col]
                rows[i] = [a - ratio * b for a, b in zip(rows[i], rows[col], strict=True)]
    return [rows[i][size] / rows[i][i] for i in range(size)]


def exact_active_set_optimum(qp, active):
    """Plan and row multipliers solving the KKT system with the `active` rows as equalities.

    Returns w, lam and the exact row values G w - h. With lam >= 0 and every row value >= 0 they
    are the exact optimum of this convex problem, given its floating-point data.
    """
    h_mat, g, e_mat, f, g_mat, h_vec = dense_problem(qp)
    nw, ne, na = h_mat.shape[0], e_mat.shape[0], len(active)
    g_act = g_mat[active]
    k = np.block(
        [
            [h_mat, e_mat.T, -g_act.T],
            [e_mat, np.zeros((ne, ne)), np.zeros((ne, na))],
            [g_act, np.zeros((na, ne)), np.zeros((na, na))],
        ]
    )
    solution = exact_solve(k, np.concatenate([-g, f, h_vec[active]]))
    w = solution[:nw]
    lam = [Fraction(0)] * len(h_vec)
    for i, value in zip(active, solution[nw + ne :], strict=True):
        lam[i] = value
    rows = [
        sum(Fraction(float(g_mat[i, j])) * w[j] for j in np.flatnonzero(g_mat[i]))
        - Fraction(float(h_vec[i]))
        for i in range(len(h_vec))
    ]
    return np.array([float(v) for v in w]), lam, rows


def plan_vector(qp, x, u):
    return np.concatenate([np.concatenate([x[:-1], u], axis=1).ravel(), x[-1]])


@pytest.fixture(scope="module")
def strongly_active_case():
    # One 2710 s stage, |B| up to 4.7e6: row 1 is active with multiplier 1.1e8, rows 0, 2 and 3
    # are inactive with relative slacks 0.98, 0.999 and 0.997 at the optimum.
    qp = random_cw_problem(np.random.default_rng([22, 2235]))
    w, lam, rows = exact_active_set_optimum(qp, [1])
    assert all(v >= 0 for v in lam) and all(v >= 0 for v in rows)
    return qp, solve_horizon(qp), w, np.array([float(v) for v in lam])


def test_converged_multipliers_equal_the_exact_optimum_multipliers(strongly_active_case):
    qp, sol, w_ref, lam_ref = strongly_active_case
    assert sol.status == "converged"
    h_mat, g, _, _, g_mat, _ = dense_problem(qp)
    # A multiplier is judged against the larger of its exact value and the one at which its row's
    # force matches the cost gradient terms in the free component it moves most easily (x_0 is
    # fixed): an inactive row's multiplier must not push the plan against the cost.
    cost_terms = np.abs(h_mat) @ np.abs(w_ref) + np.abs(g)
    g_free = np.abs(g_mat[:, NX:])
    balancing = np.min(
        np.divide(cost_terms[NX:], g_free, out=np.full(g_free.shape, np.inf), where=g_free > 0),
        axis=1,
    )
    error = np.abs(sol.multipliers - lam_ref) / np.maximum(lam_ref, balancing)
    assert np.all(error <= 1e-6), error


def test_converged_plan_equals_the_exact_optimum(strongly_active_case):
    qp, sol, w_ref, _ = strongly_active_case
    assert sol.status == "converged"
    w = plan_vector(qp, sol.x, sol.u)
    u, u_ref = w[NX:NZ], w_ref[NX:NZ]
    assert np.max(np.abs(u - u_ref)) <= 1e-6 * np.max(np.abs(u_ref))
    assert np.max(np.abs(w - w_ref)) <= 1e-6 * np.max(np.abs(w_ref))


def same_solve(a, b, to_si_x=lambda x: x, to_si_u=lambda u: u):
    """Mismatches between two solves of one problem in two unit systems, as strings."""
    found = []
    if a.status != b.status:
        found.append(f"status {a.status} != {b.status}")
    if a.iterations != b.iterations:
        found.append(f"iterations {a.iterations} != {b.iterations}")
    if a.status == b.status == "converged":
        for name, va, vb in (("x", a.x, to_si_x(b.x)), ("u", a.u, to_si_u(b.u))):
            gap = np.max(np.abs(va - vb)) / np.max(np.abs(va))
            if gap > 1e-6:
                found.append(f"{name} differs by {gap:.1e}")
    return found


# Feasible proximity problems on which the SI and Hill-unit solves end differently.
@pytest.mark.parametrize("case", [114, 283, 757, 930])
def test_hill_units_give_the_same_status_iterations_and_plan(case):
    qp = proximity_problem(np.random.default_rng([31, case]))
    scaling = HillScaling(length=float(np.max(np.abs(qp.x0[:3]))), n=N_MEAN)
    si = solve_horizon(qp)
    hill = solve_horizon(scaling.stage(qp))
    assert same_solve(si, hill, scaling.state_si, scaling.accel_si) == []


def test_cost_scale_changes_neither_status_iterations_nor_plan():
    # docs/solvers.md: every cost multiplied by k takes the same steps; tested there for k from
    # 1e-9 to 1e6.
    qp = random_cw_problem(np.random.default_rng([21, 1302]))
    reference = solve_horizon(qp)
    assert reference.status == "converged"
    found = {}
    for k in (1e-9, 1e-6, 1e-3, 1e3, 1e6):
        scaled = dataclasses.replace(qp, Q=k * qp.Q, q=k * qp.q, P=k * qp.P, p=k * qp.p)
        mismatch = same_solve(reference, solve_horizon(scaled))
        if mismatch:
            found[k] = mismatch
    assert found == {}


def approach_with_reach_row(km):
    """The docs' 10-stage approach with y_N >= -110 m, which the zero-thrust start violates.

    Lengths in m when km is False, else in km (states km and km/s, inputs km/s^2, rows in km),
    with the costs rescaled so every plan has the same objective.
    """
    n, a_max = 10, 0.02
    a, b = cw_discrete(mean_motion(EARTH.radius + 500e3), np.full(n, 20.0))
    hold = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    cost = np.zeros((NZ, NZ))
    cost[:NX, :NX] = np.diag([1e-2] * 3 + [1.0] * 3)
    cost[NX:, NX:] = 1e2 * np.eye(NU)
    grad = np.zeros(NZ)
    grad[:NX] = -cost[:NX, :NX] @ hold
    terminal = np.diag([1.0] * 3 + [100.0] * 3)
    box = np.zeros((n, 6, NZ))
    box[:, :3, NX:] = np.eye(NU)
    box[:, 3:, NX:] = -np.eye(NU)
    reach = np.zeros((1, NX))
    reach[0, 1] = 1.0
    length = 1e-3 if km else 1.0
    return StageQP(
        x0=np.array([0.0, -150.0, 0.0, 0.0, 0.0, 0.0]) * length,
        A=a,
        B=b,
        c=np.zeros((n, NX)),
        Q=np.repeat(cost[None], n, axis=0) / length**2,
        q=np.repeat(grad[None], n, axis=0) / length,
        P=terminal / length**2,
        p=-terminal @ hold / length,
        C=box,
        e=np.full((n, 6), -a_max * length),
        C_N=reach,
        e_N=np.array([-110.0 * length]),
    )


def test_kilometres_give_the_same_status_iterations_and_plan_as_metres():
    metres = solve_horizon(approach_with_reach_row(km=False))
    kilometres = solve_horizon(approach_with_reach_row(km=True))
    assert metres.status == "converged"
    assert same_solve(metres, kilometres, lambda x: 1e3 * x, lambda u: 1e3 * u) == []


def assert_converges_to_a_kkt_point(qp):
    sol = solve_horizon(qp)
    assert sol.status == "converged", (sol.status, sol.iterations)
    check = kkt_check(qp, sol.x, sol.u, np.asarray(sol.multipliers, dtype=float))
    # tol_feas and tol_gap are 1e-8; these bounds leave a factor 100 for the different measures.
    assert check["stationarity"] <= 1e-6, check
    assert check["dynamics"] <= 1e-6, check
    assert check["violation"] <= 1e-6, check
    assert check["negative_multiplier"] == 0.0, check
    assert check["complementarity"] <= 1e-6, check
    return check


# Feasible proximity problems on which the solve ends in solver_failed.
@pytest.mark.parametrize("case", [14, 52, 78, 114, 1087])
def test_feasible_proximity_problem_converges_to_a_kkt_point(case):
    assert_converges_to_a_kkt_point(proximity_problem(np.random.default_rng([31, case])))


@pytest.mark.slow
def test_feasible_proximity_problems_all_converge_to_kkt_points():
    failures, active = [], 0
    for case in range(300):
        qp = proximity_problem(np.random.default_rng([31, case]))
        try:
            active += assert_converges_to_a_kkt_point(qp)["active_rows"] > 0
        except AssertionError as error:
            failures.append((case, str(error)[:80]))
    assert failures == []
    assert active > 100
