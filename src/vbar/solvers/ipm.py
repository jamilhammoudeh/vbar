"""Primal-dual interior-point method for stage-structured horizon QPs.

The rows C_k z_k >= e_k carry slacks s > 0 and multipliers lam > 0, and the method starts from any
point with s, lam > 0: neither the dynamics nor the rows need to hold at the start (Boyd and
Vandenberghe, *Convex Optimization*, sec. 11.7, Algorithm 11.2). Each Newton system is reduced by
eliminating the slack and multiplier steps, which adds C_k^T diag(lam_k / s_k) C_k to each stage
Hessian, and is then solved by the Riccati recursion of `riccati_solve`, so an iteration costs one
backward sweep of the Hessians, linear in the horizon. Mehrotra's predictor-corrector step (Nocedal
and Wright, *Numerical Optimization*, 2nd ed., sec. 16.6) reuses that sweep for a second right-hand
side and keeps the iteration count low on long horizons.

An iterate of this infeasible-start method satisfies the dynamics only at convergence, so it is
not a plan before then and none is returned (Rawlings, Mayne and Diehl, *Model Predictive
Control*, 2nd ed., sec. 6.1.2). When the primal residual stops falling, the sum-of-infeasibilities
LP over the horizon (Boyd and Vandenberghe, sec. 11.4.1) is solved, and its dual is accepted as a
Farkas certificate of infeasibility (sec. 5.8.3) only after it is verified in numpy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .stage import StageQP, _Factor, _factor, _sweep

HORIZON_STATUSES = ("converged", "iteration_cap", "infeasible_certified", "solver_failed")

_STEP_TO_BOUNDARY = 0.99
# The primal residual stalls when ten iterations have not halved it.
_STALL_ITERATIONS = 10
_STALL_RATIO = 0.5
# Backtracks before a direction is given up; at beta = 0.5 the last trial step is below 1e-18.
_MAX_BACKTRACKS = 60
# The corrected step may be off the descent direction of the residual norm, so it gets one
# backtrack before the plain step replaces it; more cost iterations on long horizons.
_CORRECTOR_BACKTRACKS = 1
# A component whose resistance to a row's force is below this fraction of what the row's scale
# gives, or a row whose terms are below this fraction of the largest row's, counts as unused:
# only rounding is left of its terms.
_UNUSED = math.sqrt(np.finfo(float).eps)
# A Farkas certificate must cancel to this fraction of the magnitude of the terms it sums.
_FARKAS_TOL = 1e-9
# Rows whose certificate multiplier exceeds this fraction of the largest one are its support.
_SUPPORT_TOL = 1e-9


@dataclass(frozen=True)
class IPMSettings:
    """Stopping and step parameters of `solve_horizon`.

    Parameters
    ----------
    tol_feas : float
        Primal and dual residuals, in the infinity norm relative to the largest term each one
        sums, below which the iterate counts as feasible.
    tol_gap : float
        Complementarity below which the iterate counts as optimal: every row has a slack below
        this fraction of its terms sum_j |C_ij z_j| + |e_i|, or a multiplier below this fraction
        of the one that balances the largest gradient term.
    max_iter : int
        Newton steps before the solve stops with status "iteration_cap".
    mu : float
        Barrier reduction of the fallback step, t = mu (rows) / s^T lam (Boyd and Vandenberghe,
        sec. 11.7.3), taken when the predictor-corrector step fails the line search.
    alpha : float
        Sufficient decrease of the residual norm in the backtracking line search, in (0, 0.5).
    beta : float
        Step reduction per backtrack, in (0, 1).

    Raises
    ------
    ValueError
        If a parameter is outside its range.
    """

    tol_feas: float = 1e-8
    tol_gap: float = 1e-8
    max_iter: int = 60
    mu: float = 10.0
    alpha: float = 0.01
    beta: float = 0.5

    def __post_init__(self):
        for name in ("tol_feas", "tol_gap"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if not (isinstance(self.max_iter, int) and self.max_iter >= 0):
            raise ValueError("max_iter must be a nonnegative integer")
        if not self.mu > 1.0:
            raise ValueError("mu must be greater than 1")
        if not 0.0 < self.alpha < 0.5:
            raise ValueError("alpha must be in (0, 0.5)")
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must be in (0, 1)")


@dataclass(frozen=True)
class HorizonSolution:
    """The result of `solve_horizon`.

    Attributes
    ----------
    x : ndarray, shape (N + 1, nx), or None
        States; None unless `status` is "converged".
    u : ndarray, shape (N, nu), or None
        Inputs; None unless `status` is "converged".
    status : str
        "converged", "iteration_cap", "infeasible_certified" or "solver_failed".
    iterations : int
        Newton steps taken.
    objective : float
        Objective at the solution; nan unless converged.
    residual_primal : float
        Infinity norm of the dynamics and slack residuals at the last iterate.
    residual_dual : float
        Infinity norm of the stationarity residual at the last iterate.
    gap : float
        s^T lam at the last iterate.
    multipliers : ndarray, shape (N m + mN,), or None
        Row multipliers, >= 0, stage rows in stage order and then the terminal rows; None unless
        converged.
    conflicting : tuple of (int, int)
        (stage, row) pairs in the support of the certificate, stage N for terminal rows; () unless
        the status is "infeasible_certified".
    certificate : ndarray, shape (N m + mN,), or None
        Farkas multipliers lam >= 0 over the rows, ordered as `multipliers`, for which no plan
        can meet every row; None unless the status is "infeasible_certified".
    elapsed_s : float
        Wall-clock time of the solve (s).
    """

    x: np.ndarray | None
    u: np.ndarray | None
    status: str
    iterations: int
    objective: float
    residual_primal: float
    residual_dual: float
    gap: float
    multipliers: np.ndarray | None
    conflicting: tuple[tuple[int, int], ...] = field(default=())
    certificate: np.ndarray | None = None
    elapsed_s: float = 0.0


def _relative(value: np.ndarray, size: np.ndarray) -> np.ndarray:
    """|value| / size componentwise, 0 where the size is 0 (the terms, and so the value, are 0)."""
    value, size = np.broadcast_arrays(np.abs(value), size)
    return np.divide(value, size, out=np.zeros(value.shape), where=size > 0.0)


class _Rows:
    """Stage and terminal rows of a problem as one flat vector, stages first."""

    def __init__(self, qp: StageQP):
        n, nx, nu = qp.B.shape
        nz = nx + nu
        self.c = np.zeros((n, 0, nz)) if qp.C is None else qp.C
        self.c_n = np.zeros((0, nx)) if qp.C_N is None else qp.C_N
        e = np.zeros((n, 0)) if qp.e is None else qp.e
        e_n = np.zeros(0) if qp.e_N is None else qp.e_N
        self.n, self.m, self.m_n = n, self.c.shape[1], self.c_n.shape[0]
        self.e = np.concatenate([e.ravel(), e_n])
        self.total = self.e.size

    def force(
        self,
        size_z: np.ndarray,
        size_n: np.ndarray,
        curvature_z: np.ndarray,
        curvature_n: np.ndarray,
        terms: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        """How far a unit multiplier moves the components a row enters, relative to the row.

        A force C_ij on component j is resisted by the cost and dynamics terms of j (`size`) and
        by its curvature Q_jj against the row's own scale in z_j, |C_i| |z| / |C_ij|. Where both
        vanish, on an axis the plan does not use, the row's `scale` (its terms with |e_i|) is
        used instead. The largest ratio over j is returned.
        """
        split = self.n * self.m
        c_abs, c_n_abs = np.abs(self.c), np.abs(self.c_n)
        stage = self._force(
            c_abs,
            size_z[:, None, :],
            curvature_z[:, None, :],
            terms[:split].reshape(self.n, self.m)[:, :, None],
            scale[:split].reshape(self.n, self.m)[:, :, None],
        )
        terminal = self._force(
            c_n_abs, size_n[None, :], curvature_n[None, :], terms[split:, None], scale[split:, None]
        )
        return np.concatenate(
            [np.max(stage, axis=2, initial=0.0).ravel(), np.max(terminal, axis=1, initial=0.0)]
        )

    @staticmethod
    def _force(c_abs, size, curvature, terms, bound) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            resisted = c_abs * size + curvature * terms
        unused = curvature * bound
        resisted = np.where(resisted > _UNUSED * unused, resisted, unused)
        with np.errstate(divide="ignore"):
            return np.divide(c_abs**2, resisted, out=np.zeros(c_abs.shape), where=c_abs > 0.0)

    def terms(self, z: np.ndarray, x_n: np.ndarray) -> np.ndarray:
        """Size of each row's variable terms, sum_j |C_ij z_j|."""
        stage = np.einsum("kij,kj->ki", np.abs(self.c), np.abs(z)).ravel()
        return np.concatenate([stage, np.abs(self.c_n) @ np.abs(x_n)])

    def apply(self, z: np.ndarray, x_n: np.ndarray) -> np.ndarray:
        """Row values C_k z_k and C_N x_N."""
        stage = np.einsum("kij,kj->ki", self.c, z).ravel()
        return np.concatenate([stage, self.c_n @ x_n])

    def transpose_abs(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """|C_k|^T |v_k| per stage and |C_N|^T |v_N|."""
        split = self.n * self.m
        w = np.abs(v)
        stage = np.einsum("kij,ki->kj", np.abs(self.c), w[:split].reshape(self.n, self.m))
        return stage, np.abs(self.c_n).T @ w[split:]

    def transpose(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """C_k^T v_k per stage and C_N^T v_N."""
        split = self.n * self.m
        stage = np.einsum("kij,ki->kj", self.c, v[:split].reshape(self.n, self.m))
        return stage, self.c_n.T @ v[split:]

    def weighted(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """C_k^T diag(w_k) C_k per stage and C_N^T diag(w_N) C_N."""
        split = self.n * self.m
        w_stage = w[:split].reshape(self.n, self.m)
        stage = np.einsum("kij,ki,kil->kjl", self.c, w_stage, self.c)
        return stage, self.c_n.T @ (w[split:, None] * self.c_n)


@dataclass
class _Iterate:
    x: np.ndarray  # (N + 1, nx)
    u: np.ndarray  # (N, nu)
    y: np.ndarray  # (N, nx), dynamics multipliers
    s: np.ndarray  # (rows,)
    lam: np.ndarray  # (rows,)

    def moved(self, step: _Iterate, alpha: float) -> _Iterate:
        return _Iterate(
            self.x + alpha * step.x,
            self.u + alpha * step.u,
            self.y + alpha * step.y,
            self.s + alpha * step.s,
            self.lam + alpha * step.lam,
        )


@dataclass(frozen=True)
class _Residuals:
    grad: np.ndarray  # (N, nx + nu), stationarity in z_k, zero in x_0
    grad_n: np.ndarray  # (nx,), stationarity in x_N
    x0: np.ndarray  # (nx,), x_0 - x0
    dynamics: np.ndarray  # (N, nx), A_k x_k + B_k u_k + c_k - x_{k+1}
    slack: np.ndarray  # (rows,), C z - e - s
    # Sizes of the terms each component sums, laid out as `dual` and `primal`.
    dual_size: np.ndarray
    primal_size: np.ndarray
    complementarity: float  # max over rows of min(slack, force) relative to the row's terms
    mean_product: float  # s^T lam / rows

    @property
    def dual(self) -> np.ndarray:
        return np.concatenate([self.grad.ravel(), self.grad_n])

    @property
    def primal(self) -> np.ndarray:
        return np.concatenate([self.x0, self.dynamics.ravel(), self.slack])

    @property
    def dual_norm(self) -> float:
        return float(np.max(np.abs(self.dual), initial=0.0))

    @property
    def primal_norm(self) -> float:
        return float(np.max(np.abs(self.primal), initial=0.0))

    @property
    def dual_relative(self) -> float:
        return float(np.max(_relative(self.dual, self.dual_size), initial=0.0))

    @property
    def primal_relative(self) -> float:
        return float(np.max(_relative(self.primal, self.primal_size), initial=0.0))


def _largest(*terms: np.ndarray) -> float:
    return max(float(np.max(np.abs(term), initial=0.0)) for term in terms)


def _residuals(qp: StageQP, rows: _Rows, it: _Iterate) -> _Residuals:
    nx = qp.A.shape[1]
    z = np.concatenate([it.x[:-1], it.u], axis=1)
    x_n = it.x[-1]
    ct_lam, ct_lam_n = rows.transpose(it.lam)
    hess_z = np.einsum("kij,kj->ki", qp.Q, z)
    hess_n = qp.P @ x_n
    a_y = np.einsum("kji,kj->ki", qp.A, it.y)
    b_y = np.einsum("kji,kj->ki", qp.B, it.y)
    grad = hess_z + qp.q - ct_lam
    grad[:, :nx] += a_y
    grad[1:, :nx] -= it.y[:-1]
    grad[:, nx:] += b_y
    # The multiplier of x_0 = x0 is free, so stationarity in x_0 always holds.
    grad[0, :nx] = 0.0
    grad_n = hess_n + qp.p - ct_lam_n - it.y[-1]
    dynamics = (
        np.einsum("kij,kj->ki", qp.A, it.x[:-1])
        + np.einsum("kij,kj->ki", qp.B, it.u)
        + qp.c
        - it.x[1:]
    )
    slack = rows.apply(z, x_n) - rows.e - it.s
    # Each component is measured against the magnitudes of the terms it sums, so the tests do not
    # depend on the units of the costs, states, inputs or rows.
    abs_y = np.abs(it.y)
    size_z = np.einsum("kij,kj->ki", np.abs(qp.Q), np.abs(z)) + np.abs(qp.q)
    size_z[:, :nx] += np.einsum("kji,kj->ki", np.abs(qp.A), abs_y)
    size_z[1:, :nx] += abs_y[:-1]
    size_z[:, nx:] += np.einsum("kji,kj->ki", np.abs(qp.B), abs_y)
    size_n = np.abs(qp.P) @ np.abs(x_n) + np.abs(qp.p) + abs_y[-1]
    abs_lam_z, abs_lam_n = rows.transpose_abs(it.lam)
    dual_size = np.concatenate([(size_z + abs_lam_z).ravel(), size_n + abs_lam_n])
    dynamics_size = (
        np.einsum("kij,kj->ki", np.abs(qp.A), np.abs(it.x[:-1]))
        + np.einsum("kij,kj->ki", np.abs(qp.B), np.abs(it.u))
        + np.abs(qp.c)
        + np.abs(it.x[1:])
    )
    terms = rows.terms(z, x_n)
    magnitude = terms + np.abs(rows.e)
    # A row whose terms and bound are only rounding (u_z <= 0 where the plan never thrusts
    # cross-track) has no scale of its own, so its slack and force are judged against the
    # largest row's.
    largest_row = _largest(magnitude)
    row_scale = np.where(magnitude > _UNUSED * largest_row, magnitude, largest_row)
    primal_size = np.concatenate(
        [np.abs(it.x[0]) + np.abs(qp.x0), dynamics_size.ravel(), magnitude + it.s]
    )
    # Each row must have either a slack that is small against its own terms (it is active) or a
    # multiplier whose force moves the components it enters little against the row's scale (it
    # is inactive). A single bound on the products would not do: multipliers of active rows
    # span orders of magnitude, and a product that is small overall leaves a weakly active row
    # visibly short of its bound. x_0 is fixed, so a force on it moves nothing.
    size_z[0, :nx] = np.inf
    curvature_z = np.diagonal(qp.Q, axis1=1, axis2=2)
    force = rows.force(size_z, size_n, curvature_z, np.diagonal(qp.P), terms, row_scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        complementarity = _largest(np.minimum(it.s / row_scale, it.lam * force))
    return _Residuals(
        grad,
        grad_n,
        it.x[0] - qp.x0,
        dynamics,
        slack,
        dual_size,
        primal_size,
        complementarity,
        float(it.s @ it.lam) / max(rows.total, 1),
    )


def _merit(res: _Residuals, it: _Iterate, target: float, units: _Residuals) -> float:
    """Norm of the residual of the perturbed KKT conditions with lam s = target.

    Each component is divided by its size in `units`, held fixed over a line search, so the norm
    does not depend on the units of the problem and the Newton step is still a descent direction.
    """
    dual = _relative(res.dual, units.dual_size)
    primal = _relative(res.primal, units.primal_size)
    centrality = (it.lam * it.s - target) / (units.mean_product or 1.0)
    return float(np.sqrt(dual @ dual + primal @ primal + centrality @ centrality))


def _direction(
    qp: StageQP,
    rows: _Rows,
    factor: _Factor,
    it: _Iterate,
    res: _Residuals,
    target: float,
    product: np.ndarray,
) -> _Iterate:
    """Newton step toward lam s = target - product, from the reduced system.

    With ds = C dz + r_s and dlam = -(lam s - target + product + lam ds) / s, the step dz
    minimizes a quadratic with the Hessian plus C^T diag(lam / s) C, subject to the dynamics
    linearized around the iterate. Its right-hand sides are residuals, which vanish at the
    solution, so rounding in the solve shrinks with the step.
    """
    pull = it.lam - (target - product) / it.s + (it.lam / it.s) * res.slack
    ct_pull, ct_pull_n = rows.transpose(pull)
    dx, du, dy = _sweep(
        qp, factor, res.grad + ct_pull, res.grad_n + ct_pull_n, res.dynamics, -res.x0
    )
    ds = rows.apply(np.concatenate([dx[:-1], du], axis=1), dx[-1]) + res.slack
    dlam = -(it.lam * it.s - target + product + it.lam * ds) / it.s
    return _Iterate(dx, du, dy, ds, dlam)


def _max_step(it: _Iterate, step: _Iterate) -> float:
    """Largest alpha keeping s + alpha ds and lam + alpha dlam nonnegative (inf if unbounded)."""
    values = np.concatenate([it.s, it.lam])
    deltas = np.concatenate([step.s, step.lam])
    falling = deltas < 0.0
    if not falling.any():
        return math.inf
    return float(np.min(-values[falling] / deltas[falling]))


def _line_search(
    qp: StageQP,
    rows: _Rows,
    it: _Iterate,
    res: _Residuals,
    step: _Iterate,
    target: float,
    settings: IPMSettings,
    backtracks: int,
) -> tuple[_Iterate, _Residuals] | None:
    """Backtracking on the residual norm from 0.99 of the step to the boundary (sec. 11.7.3)."""
    alpha = min(1.0, _STEP_TO_BOUNDARY * _max_step(it, step))
    merit = _merit(res, it, target, res)
    for _ in range(backtracks + 1):
        trial = it.moved(step, alpha)
        trial_res = _residuals(qp, rows, trial)
        trial_merit = _merit(trial_res, trial, target, res)
        if trial_merit < merit and trial_merit <= (1.0 - settings.alpha * alpha) * merit:
            return trial, trial_res
        alpha *= settings.beta
    return None


def _objective(qp: StageQP, x: np.ndarray, u: np.ndarray) -> float:
    z = np.concatenate([x[:-1], u], axis=1)
    stage = np.einsum("ki,kij,kj->", z, qp.Q, z) / 2.0 + np.sum(qp.q * z)
    return float(stage + x[-1] @ qp.P @ x[-1] / 2.0 + qp.p @ x[-1])


def _start(qp: StageQP, rows: _Rows, warm: tuple[np.ndarray, np.ndarray] | None) -> _Iterate:
    """The starting iterate: the given plan or the zero-input rollout, with tight slacks.

    An infeasible-start step removes residuals in proportion to its length, so residuals far
    above the problem's scale (an initial state hundreds of metres off, or slacks of 1 on rows
    whose values are 1e-2) hold every step to a small fraction of its length while the rows
    bind. The rollout meets x_0 = x0 and the dynamics, and a row that holds starts with its
    own value as slack; a violated row starts at 1.
    """
    n, nx, nu = qp.B.shape
    if warm is None:
        u = np.zeros((n, nu))
        x = np.empty((n + 1, nx))
        x[0] = qp.x0
        for k in range(n):
            x[k + 1] = qp.A[k] @ x[k] + qp.c[k]
    else:
        x, u = (np.array(v, dtype=float) for v in warm)
        if x.shape != (n + 1, nx) or u.shape != (n, nu):
            raise ValueError(f"warm start must have shapes {(n + 1, nx)} and {(n, nu)}")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(u))):
            raise ValueError("warm start must be finite")
    z = np.concatenate([x[:-1], u], axis=1)
    value = rows.apply(z, x[-1]) - rows.e
    s = np.where(value > 0.0, value, 1.0)
    # Products lam_i s_i start equal, at the size of the objective's terms per row: the start
    # then does not depend on the units of the states and inputs, and multiplying every cost by k
    # multiplies the starting multipliers by k.
    quadratic = np.abs(np.einsum("ki,kij,kj->k", z, qp.Q, z)).sum() + abs(x[-1] @ qp.P @ x[-1])
    linear = np.abs(qp.q * z).sum() + np.abs(qp.p * x[-1]).sum()
    mu = (quadratic / 2.0 + linear) / max(rows.total, 1)
    lam = mu / s if mu > 0.0 else np.ones(rows.total)
    return _Iterate(x, u, np.zeros((n, nx)), s, lam)


def _stage_blocks(
    qp: StageQP, rows: _Rows
) -> tuple[sparse.coo_array, np.ndarray, sparse.coo_array]:
    """Dynamics E w = f and rows G w over w = (x_0, u_0, x_1, ..., x_N), as sparse matrices."""
    n, nx, nu = qp.B.shape
    nz = nx + nu
    nw = n * nz + nx
    k = np.arange(n)[:, None, None]
    i = np.arange(nx)[None, :, None]
    j = np.arange(nz)[None, None, :]
    eye_r = np.arange(nx)
    dyn_r = np.broadcast_to((k + 1) * nx + i, (n, nx, nz))
    dyn_c = np.broadcast_to(k * nz + j, (n, nx, nz))
    next_r = ((np.arange(n)[:, None] + 1) * nx + eye_r).ravel()
    next_c = ((np.arange(n)[:, None] + 1) * nz + eye_r).ravel()
    e_mat = sparse.coo_array(
        (
            np.concatenate(
                [np.ones(nx), np.concatenate([qp.A, qp.B], axis=2).ravel(), -np.ones(n * nx)]
            ),
            (
                np.concatenate([eye_r, dyn_r.ravel(), next_r]),
                np.concatenate([eye_r, dyn_c.ravel(), next_c]),
            ),
        ),
        shape=((n + 1) * nx, nw),
    )
    f = np.concatenate([qp.x0, -qp.c.ravel()])
    m, m_n = rows.m, rows.m_n
    ri = np.arange(m)[None, :, None]
    stage_r = np.broadcast_to(k * m + ri, (n, m, nz))
    stage_c = np.broadcast_to(k * nz + j, (n, m, nz))
    term_r = np.broadcast_to(n * m + np.arange(m_n)[:, None], (m_n, nx))
    term_c = np.broadcast_to(n * nz + eye_r[None, :], (m_n, nx))
    g_mat = sparse.coo_array(
        (
            np.concatenate([rows.c.ravel(), rows.c_n.ravel()]),
            (
                np.concatenate([stage_r.ravel(), term_r.ravel()]),
                np.concatenate([stage_c.ravel(), term_c.ravel()]),
            ),
        ),
        shape=(rows.total, nw),
    )
    return e_mat, f, g_mat


def _certify(qp: StageQP, rows: _Rows, lam: np.ndarray) -> bool:
    """Whether lam >= 0 is a Farkas certificate that no plan meets every row.

    The dynamics multipliers nu are not taken from the LP: they are the unique values that cancel
    G^T lam + E^T nu in every state component, found by a backward recursion, so only the input
    components and the sign of h^T lam + f^T nu remain to check.
    """
    n, nx, _ = qp.B.shape
    if not (np.all(lam >= 0.0) and lam.sum() > 0.0):
        return False
    ct_lam, ct_lam_n = rows.transpose(lam)
    split = n * rows.m
    abs_stage = np.einsum("kij,ki->kj", np.abs(rows.c), lam[:split].reshape(n, rows.m))
    # nu[k] multiplies the dynamics row of stage k; nu_x0 the row x_0 = x0.
    nu = np.empty((n, nx))
    nu[-1] = ct_lam_n
    for k in range(n - 1, 0, -1):
        nu[k - 1] = ct_lam[k, :nx] + qp.A[k].T @ nu[k]
    nu_x0 = -(ct_lam[0, :nx] + qp.A[0].T @ nu[0])
    input_residual = ct_lam[:, nx:] + np.einsum("kji,kj->ki", qp.B, nu)
    input_scale = abs_stage[:, nx:] + np.einsum("kji,kj->ki", np.abs(qp.B), np.abs(nu))
    if np.any(np.abs(input_residual) > _FARKAS_TOL * np.max(input_scale)):
        return False
    margin = rows.e @ lam + qp.x0 @ nu_x0 - np.sum(qp.c * nu)
    margin_scale = np.abs(rows.e) @ lam + np.abs(qp.x0) @ np.abs(nu_x0) + np.sum(np.abs(qp.c * nu))
    return bool(margin > _FARKAS_TOL * margin_scale)


def _phase_one(qp: StageQP, rows: _Rows) -> np.ndarray | None:
    """Certificate from the sum-of-infeasibilities LP over the horizon, or None.

    min sum t subject to G w + t >= h, t >= 0 and the dynamics E w = f (Boyd and Vandenberghe,
    sec. 11.4.1, eq. 11.20); its dual is max h^T lam + f^T nu subject to G^T lam + E^T nu = 0,
    0 <= lam <= 1, which is a Farkas certificate when its value is positive.
    """
    if rows.total == 0:
        return None
    e_mat, f, g_mat = _stage_blocks(qp, rows)
    nw = e_mat.shape[1]
    m = rows.total
    res = linprog(
        np.concatenate([np.zeros(nw), np.ones(m)]),
        A_ub=sparse.hstack([-g_mat, -sparse.eye_array(m)], format="csc"),
        b_ub=-rows.e,
        A_eq=sparse.hstack([e_mat, sparse.coo_array((e_mat.shape[0], m))], format="csc"),
        b_eq=f,
        bounds=[(None, None)] * nw + [(0.0, None)] * m,
        method="highs",
    )
    if res.status != 0:
        return None
    lam = np.maximum(-np.asarray(res.ineqlin.marginals, dtype=float), 0.0)
    return lam if _certify(qp, rows, lam) else None


def _result(
    status: str,
    iterations: int,
    it: _Iterate,
    res: _Residuals,
    started: float,
    plan: bool = False,
    objective: float = math.nan,
    multipliers: np.ndarray | None = None,
    conflicting: tuple[tuple[int, int], ...] = (),
    certificate: np.ndarray | None = None,
) -> HorizonSolution:
    return HorizonSolution(
        x=it.x if plan else None,
        u=it.u if plan else None,
        status=status,
        iterations=iterations,
        objective=objective,
        residual_primal=res.primal_norm,
        residual_dual=res.dual_norm,
        gap=float(it.s @ it.lam),
        multipliers=multipliers,
        conflicting=conflicting,
        certificate=certificate,
        elapsed_s=time.perf_counter() - started,
    )


def _infeasibility(
    qp: StageQP, rows: _Rows, it: _Iterate, res: _Residuals, iterations: int, started: float
) -> HorizonSolution:
    """The infeasible_certified result if phase I gives a verified certificate, else failed."""
    lam = _phase_one(qp, rows)
    if lam is None:
        return _result("solver_failed", iterations, it, res, started)
    support = np.flatnonzero(lam > _SUPPORT_TOL * np.max(lam))
    split = rows.n * rows.m
    conflicting = tuple(
        (int(i) // rows.m, int(i) % rows.m) if i < split else (rows.n, int(i) - split)
        for i in support
    )
    return _result(
        "infeasible_certified",
        iterations,
        it,
        res,
        started,
        conflicting=conflicting,
        certificate=lam,
    )


def solve_horizon(
    qp: StageQP,
    settings: IPMSettings | None = None,
    warm: tuple[np.ndarray, np.ndarray] | None = None,
) -> HorizonSolution:
    """Solve a stage-structured horizon QP by a primal-dual interior-point method.

    Boyd and Vandenberghe, *Convex Optimization*, sec. 11.7, Algorithm 11.2, with Mehrotra's
    predictor-corrector step (Nocedal and Wright, *Numerical Optimization*, 2nd ed., sec. 16.6)
    and the Newton systems solved by the Riccati recursion of Rawlings, Mayne and Diehl, *Model
    Predictive Control*, 2nd ed., sec. 8.8.3. A corrected step that fails the line search on the
    residual norm is replaced by the plain step toward t = mu (rows) / s^T lam, which is a
    descent direction for that norm. Deterministic: the same input gives bitwise the same output.

    Parameters
    ----------
    qp : StageQP
        The problem.
    settings : IPMSettings, optional
        Tolerances and step parameters; the defaults when None.
    warm : (x, u), optional
        Starting states (N + 1, nx) and inputs (N, nu), which need not satisfy the dynamics or
        the rows; the rollout with zero inputs from x0 when None. Slacks start at C z - e on
        rows that hold and at 1 on the others, and multipliers at the value whose force balances the
        largest cost gradient term.

    Returns
    -------
    HorizonSolution
        "converged" with the plan and the row multipliers; "iteration_cap" after
        `settings.max_iter` steps, with no plan; "infeasible_certified" with a verified Farkas
        certificate when the primal residual stalled and the phase I LP proves that no plan
        meets every row; "solver_failed" when it stalled otherwise or broke down numerically.

    Raises
    ------
    ValueError
        If `warm` has the wrong shapes or an input block of the problem is not positive definite.
    """
    started = time.perf_counter()
    settings = IPMSettings() if settings is None else settings
    rows = _Rows(qp)
    it = _start(qp, rows, warm)
    # Definiteness is a property of the problem; failures after this check are numerical.
    _factor(qp, None, None)
    m = max(rows.total, 1)
    res = _residuals(qp, rows, it)
    history: list[float] = []
    iteration = 0
    while True:
        gap = float(it.s @ it.lam)
        if not (np.isfinite(res.primal_norm) and np.isfinite(res.dual_norm) and np.isfinite(gap)):
            return _infeasibility(qp, rows, it, res, iteration, started)
        if (
            res.primal_relative <= settings.tol_feas
            and res.dual_relative <= settings.tol_feas
            and res.complementarity <= settings.tol_gap
        ):
            return _result(
                "converged",
                iteration,
                it,
                res,
                started,
                plan=True,
                objective=_objective(qp, it.x, it.u),
                multipliers=it.lam.copy(),
            )
        if iteration == settings.max_iter:
            return _result("iteration_cap", iteration, it, res, started)
        history.append(res.primal_relative)
        if (
            res.primal_relative > settings.tol_feas
            and len(history) > _STALL_ITERATIONS
            and history[-1] > _STALL_RATIO * history[-1 - _STALL_ITERATIONS]
        ):
            return _infeasibility(qp, rows, it, res, iteration, started)
        try:
            factor = _factor(qp, *rows.weighted(it.lam / it.s))
        except ValueError:
            return _result("solver_failed", iteration, it, res, started)
        none = np.zeros(rows.total)
        mean = gap / m
        affine = _direction(qp, rows, factor, it, res, 0.0, none)
        trial = it.moved(affine, min(1.0, _max_step(it, affine)))
        sigma = (float(trial.s @ trial.lam) / m / mean) ** 3 if rows.total else 0.0
        target = sigma * mean
        step = _direction(qp, rows, factor, it, res, target, affine.s * affine.lam)
        found = _line_search(qp, rows, it, res, step, target, settings, _CORRECTOR_BACKTRACKS)
        if found is None:
            target = mean / settings.mu
            step = _direction(qp, rows, factor, it, res, target, none)
            found = _line_search(qp, rows, it, res, step, target, settings, _MAX_BACKTRACKS)
        iteration += 1
        if found is None:
            return _result("solver_failed", iteration, it, res, started)
        it, res = found
