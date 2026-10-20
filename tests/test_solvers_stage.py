from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import solve_discrete_are
from scipy.sparse.linalg import spsolve

from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.solvers import HillScaling, StageQP, riccati_solve
from vbar.solvers.stage import _factor

N_500 = mean_motion(EARTH.radius + 500e3)
NX, NU = 6, 3
NZ = NX + NU


def stage_cost(qx, r, cross=None):
    """Stage Hessian on z = (x, u) from the state and input blocks and an input-state cross term."""
    q = np.zeros((NZ, NZ))
    q[:NX, :NX] = qx
    q[NX:, NX:] = r
    if cross is not None:
        q[NX:, :NX] = cross
        q[:NX, NX:] = cross.T
    return q


def cw_stage_qp(dts, seed=3):
    """Time-varying exact CW horizon with affine terms and small input-state cross terms.

    Weights are those of the cheap-control case in SI units, where an unsymmetrized
    recursion loses accuracy: Q = diag(1e-2 x3, 1 x3), R = 1e3 I.
    """
    rng = np.random.default_rng(seed)
    dts = np.asarray(dts, dtype=float)
    a, b = cw_discrete(N_500, dts)
    n = len(dts)
    qx = np.diag([1e-2] * 3 + [1.0] * 3)
    c = np.zeros((n, NX))
    q = np.zeros((n, NZ, NZ))
    q_lin = np.zeros((n, NZ))
    for k in range(n):
        c[k] = rng.normal(size=NX) * 1e-2
        q[k] = stage_cost(qx, 1e3 * np.eye(NU), rng.normal(size=(NU, NX)) * 1e-4)
        q_lin[k, :NX] = rng.normal(size=NX) * 1e-2
        q_lin[k, NX:] = rng.normal(size=NU) * 1e-2
    return StageQP(
        x0=np.array([200.0, -500.0, 30.0, 0.1, 0.05, -0.02]),
        A=a,
        B=b,
        c=c,
        Q=q,
        q=q_lin,
        P=np.diag([1.0] * 3 + [10.0] * 3),
        p=rng.normal(size=NX),
        C=None,
        e=None,
        C_N=None,
        e_N=None,
    )


def dense_kkt(qp):
    """Solve the stage KKT system assembled densely, independent of any recursion.

    Variables (x_0, u_0, x_1, u_1, ..., x_N), with x_0 = x0 and the dynamics as equality rows;
    a sparse LU of the whole system, no elimination of states or inputs.
    Returns x (N+1, nx), u (N, nu) and the dynamics duals y (N, nx) of the Lagrangian
    f + sum_k y_k^T (A_k x_k + B_k u_k + c_k - x_{k+1}).
    """
    n, nx, nu = qp.B.shape
    nz = nx + nu
    nw = n * nz + nx
    h = np.zeros((nw, nw))
    g = np.zeros(nw)
    for k in range(n):
        h[k * nz : (k + 1) * nz, k * nz : (k + 1) * nz] = qp.Q[k]
        g[k * nz : (k + 1) * nz] = qp.q[k]
    h[n * nz :, n * nz :] = qp.P
    g[n * nz :] = qp.p
    neq = (n + 1) * nx
    e = np.zeros((neq, nw))
    f = np.zeros(neq)
    e[:nx, :nx] = np.eye(nx)
    f[:nx] = qp.x0
    for k in range(n):
        rows = slice((k + 1) * nx, (k + 2) * nx)
        e[rows, k * nz : k * nz + nx] = qp.A[k]
        e[rows, k * nz + nx : (k + 1) * nz] = qp.B[k]
        e[rows, (k + 1) * nz : (k + 1) * nz + nx] = -np.eye(nx)
        f[rows] = -qp.c[k]
    kkt = sparse.bmat([[sparse.csc_array(h), sparse.csc_array(e.T)], [sparse.csc_array(e), None]])
    sol = spsolve(kkt.tocsc(), np.concatenate([-g, f]))
    w, nu_eq = sol[:nw], sol[nw:]
    x = np.array([w[k * nz : k * nz + nx] for k in range(n + 1)])
    u = np.array([w[k * nz + nx : (k + 1) * nz] for k in range(n)])
    return x, u, nu_eq[nx:].reshape(n, nx)


def unsymmetrized_riccati(qp):
    """The backward recursion of Rawlings, Mayne and Diehl eqs. (8.51) to (8.54) without
    re-symmetrizing P, so rounding asymmetry in P is carried from stage to stage."""
    n = qp.B.shape[0]
    p_mat, p_vec = qp.P, qp.p
    gains = []
    for k in range(n - 1, -1, -1):
        a, b, c = qp.A[k], qp.B[k], qp.c[k]
        qxx, qux, quu = qp.Q[k][:NX, :NX], qp.Q[k][NX:, :NX], qp.Q[k][NX:, NX:]
        qx, qu = qp.q[k][:NX], qp.q[k][NX:]
        huu = quu + b.T @ p_mat @ b
        hux = qux + b.T @ p_mat @ a
        pb = p_mat @ c + p_vec
        hu = qu + b.T @ pb
        gains.append((-np.linalg.solve(huu, hux), -np.linalg.solve(huu, hu)))
        p_vec = qx + a.T @ pb - hux.T @ np.linalg.solve(huu, hu)
        p_mat = qxx + a.T @ p_mat @ a - hux.T @ np.linalg.solve(huu, hux)
    gains.reverse()
    x, us = qp.x0, []
    for k, (gain, ff) in enumerate(gains):
        u = gain @ x + ff
        us.append(u)
        x = qp.A[k] @ x + qp.B[k] @ u + qp.c[k]
    return np.array(us)


def relative(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


DTS_40 = [10.0, 15.0, 20.0] * 13 + [10.0]


def test_riccati_equals_dense_kkt_on_time_varying_cw():
    qp = cw_stage_qp(DTS_40)
    x, u, y = riccati_solve(qp)
    x_ref, u_ref, y_ref = dense_kkt(qp)
    assert x.shape == (41, NX) and u.shape == (40, NU) and y.shape == (40, NX)
    # The symmetrized recursion and the dense solve agree to about 1e-15; 1e-10 leaves room for
    # BLAS differences between platforms.
    assert relative(u, u_ref) < 1e-10
    assert relative(x, x_ref) < 1e-10
    assert np.array_equal(x[0], qp.x0)


def test_riccati_duals_are_the_dynamics_multipliers():
    qp = cw_stage_qp(DTS_40)
    _, _, y = riccati_solve(qp)
    _, _, y_ref = dense_kkt(qp)
    # Both use the Lagrangian f + sum_k y_k^T (A_k x_k + B_k u_k + c_k - x_{k+1}).
    assert relative(y, y_ref) < 1e-9


def test_lu_solve_of_the_asymmetric_stage_system_is_off_by_more_than_ten_percent():
    # The same problem as the dense comparison, by the recursion that neither re-symmetrizes P
    # nor reads one triangle of the input block: rounding asymmetry in P, solved by LU, moves u by
    # 33% (6% to 33% over other seeds of the same construction), so the agreement to 1e-10 above
    # is a property of the solver and not of the problem.
    qp = cw_stage_qp(DTS_40)
    _, u_ref, _ = dense_kkt(qp)
    assert relative(unsymmetrized_riccati(qp), u_ref) > 0.1


def test_cost_to_go_matrices_are_exactly_symmetric_at_every_stage():
    # Without re-symmetrization the computed P_k drift to about 6e-14 relative asymmetry on this
    # problem; (P + P^T) / 2 is symmetric bit for bit.
    qp = cw_stage_qp(DTS_40)
    cost_to_go = _factor(qp, None, None).cost_to_go
    assert cost_to_go.shape == (41, NX, NX)
    assert all(np.array_equal(p_k, p_k.T) for p_k in cost_to_go)


def test_first_move_equals_lqr_gain_when_terminal_cost_is_the_dare_solution():
    a, b = cw_discrete(N_500, 10.0)
    qx = np.diag([1e-2] * 3 + [1.0] * 3)
    r = 1e3 * np.eye(NU)
    p_inf = solve_discrete_are(a, b, qx, r)
    p_inf = 0.5 * (p_inf + p_inf.T)
    gain = np.linalg.solve(r + b.T @ p_inf @ b, b.T @ p_inf @ a)
    n = 20
    x0 = np.array([50.0, -800.0, 10.0, 0.02, -0.01, 0.005])
    qp = StageQP(
        x0=x0,
        A=np.repeat(a[None], n, axis=0),
        B=np.repeat(b[None], n, axis=0),
        c=np.zeros((n, NX)),
        Q=np.repeat(stage_cost(qx, r)[None], n, axis=0),
        q=np.zeros((n, NZ)),
        P=p_inf,
        p=np.zeros(NX),
        C=None,
        e=None,
        C_N=None,
        e_N=None,
    )
    _, u, _ = riccati_solve(qp)
    lqr = -gain @ x0
    assert np.max(np.abs(u[0] - lqr)) < 1e-10 * np.max(np.abs(lqr))
    # With the DARE solution as terminal cost every stage applies the same gain.
    _, u_ref, _ = dense_kkt(qp)
    assert relative(u, u_ref) < 1e-10


def test_single_stage_equals_dense_kkt():
    qp = cw_stage_qp([20.0], seed=5)
    x, u, y = riccati_solve(qp)
    x_ref, u_ref, _ = dense_kkt(qp)
    assert x.shape == (2, NX) and u.shape == (1, NU) and y.shape == (1, NX)
    assert relative(u, u_ref) < 1e-10
    assert relative(x, x_ref) < 1e-10


def condensed_hessian(a, b, qx, r, n):
    """Hessian in u alone after eliminating the states, the dense form the module avoids."""
    impulse = [b]
    for _ in range(n - 1):
        impulse.append(a @ impulse[-1])
    lag = np.subtract.outer(np.arange(n), np.arange(n))
    blocks = np.array(impulse)[np.maximum(lag, 0)] * (lag >= 0)[:, :, None, None]
    su = blocks.transpose(0, 2, 1, 3).reshape(NX * n, NU * n)
    return su.T @ np.kron(np.eye(n), qx) @ su + np.kron(np.eye(n), r)


def test_two_hundred_stages_match_sparse_kkt_where_condensing_is_ill_conditioned():
    n = 200
    a, b = cw_discrete(N_500, 60.0)
    qx = np.diag([1e-2] * 3 + [1.0] * 3)
    r = 1e-2 * np.eye(NU)
    # About 1.7e11 for CW at 60 s steps and N = 200: the condensed problem would lose about
    # eleven digits, the stage-wise solve none.
    eig = np.linalg.eigvalsh(condensed_hessian(a, b, qx, r, n))
    assert eig[-1] / eig[0] > 1e10
    qp = StageQP(
        x0=np.ones(NX),
        A=np.repeat(a[None], n, axis=0),
        B=np.repeat(b[None], n, axis=0),
        c=np.zeros((n, NX)),
        Q=np.repeat(stage_cost(qx, r)[None], n, axis=0),
        q=np.zeros((n, NZ)),
        P=qx,
        p=np.zeros(NX),
        C=None,
        e=None,
        C_N=None,
        e_N=None,
    )
    x, u, _ = riccati_solve(qp)
    x_ref, u_ref, _ = dense_kkt(qp)
    assert x.shape == (201, NX) and u.shape == (200, NU)
    assert relative(u, u_ref) < 1e-9
    assert relative(x, x_ref) < 1e-9


def objective(qp, x, u):
    z = np.concatenate([x[:-1], u], axis=1)
    stage = 0.5 * np.einsum("ki,kij,kj->", z, qp.Q, z) + np.sum(qp.q * z)
    return stage + 0.5 * x[-1] @ qp.P @ x[-1] + qp.p @ x[-1]


def test_problem_in_hill_units_has_the_si_solution_objective_rows_and_multipliers():
    n = 40
    box = np.zeros((n, 2 * NU, NZ))
    box[:, :NU, NX:] = np.eye(NU)
    box[:, NU:, NX:] = -np.eye(NU)
    terminal = np.zeros((1, NX))
    terminal[0, 1] = 1.0
    qp = replace(
        cw_stage_qp(DTS_40),
        C=box,
        e=np.full((n, 2 * NU), -0.02),
        C_N=terminal,
        e_N=np.array([-150.0]),
    )
    scaling = HillScaling(length=100.0, n=N_500)
    scaled = scaling.stage(qp)
    x, u, y = riccati_solve(qp)
    xs, us, ys = riccati_solve(scaled)
    # Two rounding paths through the same well-conditioned solve; 1e-9 relative is far above
    # either one's rounding.
    assert relative(scaling.state_si(xs), x) < 1e-9
    assert relative(scaling.accel_si(us), u) < 1e-9
    assert objective(scaled, xs, us) == pytest.approx(objective(qp, x, u), rel=1e-9)
    z = np.concatenate([x[:-1], u], axis=1)
    zs = np.concatenate([xs[:-1], us], axis=1)
    rows = np.einsum("kij,kj->ki", qp.C, z) - qp.e
    rows_scaled = np.einsum("kij,kj->ki", scaled.C, zs) - scaled.e
    assert relative(rows_scaled, rows) < 1e-9
    assert relative(scaled.C_N @ xs[-1] - scaled.e_N, qp.C_N @ x[-1] - qp.e_N) < 1e-9
    # y_k^T times the dynamics residual is unit-free, so y in Hill units is D^-1 y_si with
    # D = diag(1/L x3, 1/(n L) x3): multipliers scale by the state units, inverted.
    state_units = np.array([100.0] * 3 + [N_500 * 100.0] * 3)
    assert relative(ys, y * state_units) < 1e-9


def barrier_terms(qp, seed=7):
    """Symmetric positive semidefinite Hessian terms and gradients like an interior-point step's."""
    rng = np.random.default_rng(seed)
    n = qp.B.shape[0]
    rows = rng.normal(size=(n, 4, NZ))
    weight = rng.uniform(0.1, 10.0, size=(n, 4))
    d_q = np.einsum("kri,kr,krj->kij", rows, weight, rows)
    d_q = 0.5 * (d_q + np.swapaxes(d_q, 1, 2))
    root = rng.normal(size=(NX, NX))
    d_p = root @ root.T
    d_p = 0.5 * (d_p + d_p.T)
    return d_q, rng.normal(size=(n, NZ)), d_p, rng.normal(size=NX)


def test_extra_terms_equal_the_problem_with_them_added():
    qp = cw_stage_qp(DTS_40)
    d_q, d_qv, d_p, d_pv = barrier_terms(qp)
    x, u, y = riccati_solve(qp, extra=(d_q, d_qv, d_p, d_pv))
    rebuilt = replace(qp, Q=qp.Q + d_q, q=qp.q + d_qv, P=qp.P + d_p, p=qp.p + d_pv)
    x_ref, u_ref, y_ref = riccati_solve(rebuilt)
    # The same recursion on sums formed in a different order; 1e-10 covers that rounding.
    assert relative(u, u_ref) < 1e-10
    assert relative(x, x_ref) < 1e-10
    assert relative(y, y_ref) < 1e-10


def test_rhs_replaces_the_affine_terms_and_start():
    qp = cw_stage_qp(DTS_40)
    rng = np.random.default_rng(8)
    c = rng.normal(size=qp.c.shape)
    x0 = np.array([-40.0, 300.0, 5.0, -0.01, 0.02, 0.0])
    x, u, y = riccati_solve(qp, rhs=(c, x0))
    x_ref, u_ref, y_ref = riccati_solve(replace(qp, c=c, x0=x0))
    assert relative(u, u_ref) < 1e-10
    assert relative(x, x_ref) < 1e-10
    assert relative(y, y_ref) < 1e-10
    assert np.array_equal(x[0], x0)


def test_extra_and_rhs_are_used_for_one_solve_only():
    qp = cw_stage_qp(DTS_40)
    before = riccati_solve(qp)
    riccati_solve(qp, extra=barrier_terms(qp), rhs=(np.ones(qp.c.shape), np.ones(NX)))
    after = riccati_solve(qp)
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))


def wrong_terms(qp):
    d_q, d_qv, d_p, d_pv = barrier_terms(qp)
    n = qp.B.shape[0]
    return [
        dict(extra=(d_q, d_qv, d_p)),
        dict(extra=(d_q[:-1], d_qv, d_p, d_pv)),
        dict(extra=(d_q, d_qv[:, :NX], d_p, d_pv)),
        dict(extra=(d_q, d_qv, d_p[:3, :3], d_pv)),
        dict(extra=(d_q, d_qv, d_p, d_pv[:3])),
        dict(rhs=(qp.c,)),
        dict(rhs=(qp.c[: n - 1], qp.x0)),
        dict(rhs=(qp.c, qp.x0[:3])),
    ]


@pytest.mark.parametrize("case", range(8))
def test_extra_and_rhs_of_the_wrong_shape_raise_value_error(case):
    qp = cw_stage_qp([20.0, 20.0, 20.0])
    with pytest.raises(ValueError):
        riccati_solve(qp, **wrong_terms(qp)[case])


def valid_fields(n=3):
    a, b = cw_discrete(N_500, np.full(n, 20.0))
    return dict(
        x0=np.zeros(NX),
        A=a,
        B=b,
        c=np.zeros((n, NX)),
        Q=np.repeat(np.eye(NZ)[None], n, axis=0),
        q=np.zeros((n, NZ)),
        P=np.eye(NX),
        p=np.zeros(NX),
        C=None,
        e=None,
        C_N=None,
        e_N=None,
    )


def test_valid_fields_build_a_problem():
    StageQP(**valid_fields())


@pytest.mark.parametrize(
    "field, value",
    [
        ("x0", np.zeros(5)),
        ("A", np.zeros((2, NX, NX))),
        ("B", np.zeros((3, NX, 2))),
        ("c", np.zeros((3, 4))),
        ("Q", np.zeros((3, NX, NX))),
        ("q", np.zeros((3, NX))),
        ("P", np.eye(5)),
        ("p", np.zeros(5)),
    ],
)
def test_stage_qp_rejects_mismatched_shapes(field, value):
    fields = valid_fields()
    fields[field] = value
    with pytest.raises(ValueError):
        StageQP(**fields)


def test_stage_qp_rejects_an_asymmetric_stage_hessian():
    fields = valid_fields()
    q = fields["Q"].copy()
    q[1, 0, 7] = 0.5
    fields["Q"] = q
    with pytest.raises(ValueError):
        StageQP(**fields)


def test_stage_qp_rejects_an_asymmetric_terminal_cost():
    fields = valid_fields()
    p = np.eye(NX)
    p[0, 1] = 0.5
    fields["P"] = p
    with pytest.raises(ValueError):
        StageQP(**fields)


def test_stage_qp_rejects_an_empty_horizon():
    fields = valid_fields()
    fields.update(
        A=np.zeros((0, NX, NX)),
        B=np.zeros((0, NX, NU)),
        c=np.zeros((0, NX)),
        Q=np.zeros((0, NZ, NZ)),
        q=np.zeros((0, NZ)),
    )
    with pytest.raises(ValueError):
        StageQP(**fields)


@pytest.mark.parametrize(
    "rows",
    [
        dict(C=np.zeros((3, 2, NZ)), e=None),
        dict(C=np.zeros((3, 2, NZ)), e=np.zeros((3, 4))),
        dict(C=np.zeros((3, 2, NX)), e=np.zeros((3, 2))),
        dict(C_N=np.zeros((2, NX)), e_N=None),
        dict(C_N=np.zeros((2, NZ)), e_N=np.zeros(2)),
    ],
)
def test_stage_qp_rejects_inconsistent_rows(rows):
    fields = valid_fields()
    fields.update(rows)
    with pytest.raises(ValueError):
        StageQP(**fields)


def test_riccati_rejects_an_input_block_that_is_not_positive_definite():
    fields = valid_fields()
    q = fields["Q"].copy()
    q[:, NX:, NX:] = -1e6 * np.eye(NU)
    fields["Q"] = q
    # Negative enough that adding B^T P B for 20 s steps (about 4e4) leaves it indefinite;
    # validation may refuse it already, otherwise the factorization must.
    with pytest.raises(ValueError):
        riccati_solve(StageQP(**fields))
