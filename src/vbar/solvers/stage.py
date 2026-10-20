"""Stage-structured horizon QPs and their equality-constrained solve by a Riccati recursion.

A horizon problem keeps the states as variables and the dynamics as equality rows, so its KKT
system is banded with one block per stage. The backward Riccati recursion of Rawlings, Mayne and
Diehl, *Model Predictive Control*, 2nd ed., sec. 8.8.3, eqs. (8.51) to (8.54), factors it stage by
stage at a cost linear in the horizon, and a forward rollout recovers states, inputs and the
dynamics multipliers. The inputs are never condensed into one dense problem, whose Hessian grows
ill-conditioned with the horizon on unstable or slowly decaying dynamics (sec. 8.8.4).

The recursion propagates a symmetric cost-to-go matrix P_k, which rounding leaves slightly
asymmetric. Solving each stage's asymmetric input system by LU lets that asymmetry grow from stage
to stage, and on cheap-control problems in SI units the inputs end up wrong by tens of percent.
Here the input block is factored by Cholesky, which reads one triangle, and P_k is re-symmetrized
at every stage, so P_k stays exactly symmetric.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

# Hessians whose asymmetry exceeds this fraction of their largest entry are refused.
_SYM_TOL = 1e-12


def _readonly(x: np.ndarray) -> np.ndarray:
    x.flags.writeable = False
    return x


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    x = np.array(value, dtype=float)
    if x.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite")
    return x


def _symmetric(x: np.ndarray, name: str) -> np.ndarray:
    asymmetry = np.abs(x - np.swapaxes(x, -1, -2))
    if asymmetry.size and asymmetry.max() > _SYM_TOL * np.abs(x).max():
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (x + np.swapaxes(x, -1, -2))


@dataclass(frozen=True)
class StageQP:
    """A convex QP over a horizon of N stages with linear dynamics and stage rows.

    min sum_k 1/2 z_k^T Q_k z_k + q_k^T z_k + 1/2 x_N^T P x_N + p^T x_N over z_k = (x_k, u_k),
    subject to x_0 = x0, x_{k+1} = A_k x_k + B_k u_k + c_k, rows C_k z_k >= e_k for k < N and
    terminal rows C_N x_N >= e_N.

    Parameters
    ----------
    x0 : array_like, shape (nx,)
        Initial state, in the problem's state units.
    A : array_like, shape (N, nx, nx)
        State transition matrices.
    B : array_like, shape (N, nx, nu)
        Input matrices.
    c : array_like, shape (N, nx)
        Affine terms of the dynamics.
    Q : array_like, shape (N, nx + nu, nx + nu)
        Stage Hessians on z_k = (x_k, u_k), symmetric positive semidefinite with a positive
        definite input block.
    q : array_like, shape (N, nx + nu)
        Stage gradients.
    P : array_like, shape (nx, nx)
        Terminal Hessian, symmetric positive semidefinite.
    p : array_like, shape (nx,)
        Terminal gradient.
    C : array_like, shape (N, m, nx + nu), or None
        Stage row matrices; None for no stage rows.
    e : array_like, shape (N, m), or None
        Stage row bounds; None exactly when `C` is None.
    C_N : array_like, shape (mN, nx), or None
        Terminal row matrix; None for no terminal rows.
    e_N : array_like, shape (mN,), or None
        Terminal row bounds; None exactly when `C_N` is None.

    Raises
    ------
    ValueError
        If N < 1, a shape disagrees, a value is not finite, or `Q` or `P` is not symmetric to
        1e-12 of its largest entry.

    Notes
    -----
    Hessians are stored symmetrized. Definiteness is checked where it is used, by the Cholesky
    factorization in `riccati_solve`.
    """

    x0: np.ndarray
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    Q: np.ndarray
    q: np.ndarray
    P: np.ndarray
    p: np.ndarray
    C: np.ndarray | None
    e: np.ndarray | None
    C_N: np.ndarray | None
    e_N: np.ndarray | None

    def __post_init__(self):
        b_shape = np.shape(self.B)
        if len(b_shape) != 3:
            raise ValueError("B must have shape (N, nx, nu)")
        n, nx, nu = b_shape
        if n < 1:
            raise ValueError("the horizon needs at least one stage")
        nz = nx + nu
        fields = {
            "x0": _array(self.x0, (nx,), "x0"),
            "A": _array(self.A, (n, nx, nx), "A"),
            "B": _array(self.B, (n, nx, nu), "B"),
            "c": _array(self.c, (n, nx), "c"),
            "Q": _symmetric(_array(self.Q, (n, nz, nz), "Q"), "Q"),
            "q": _array(self.q, (n, nz), "q"),
            "P": _symmetric(_array(self.P, (nx, nx), "P"), "P"),
            "p": _array(self.p, (nx,), "p"),
        }
        if (self.C is None) != (self.e is None):
            raise ValueError("C and e must both be given or both be None")
        if self.C is not None:
            m = np.shape(self.C)[1] if np.ndim(self.C) == 3 else -1
            fields["C"] = _array(self.C, (n, m, nz), "C")
            fields["e"] = _array(self.e, (n, m), "e")
        if (self.C_N is None) != (self.e_N is None):
            raise ValueError("C_N and e_N must both be given or both be None")
        if self.C_N is not None:
            m_n = np.shape(self.C_N)[0] if np.ndim(self.C_N) == 2 else -1
            fields["C_N"] = _array(self.C_N, (m_n, nx), "C_N")
            fields["e_N"] = _array(self.e_N, (m_n,), "e_N")
        for name, value in fields.items():
            object.__setattr__(self, name, _readonly(value))


@dataclass(frozen=True)
class _Factor:
    """Backward sweep of the Hessian part: per-stage input-block Cholesky factors and gains."""

    chol: list[tuple[np.ndarray, bool]]
    gain: np.ndarray  # (N, nu, nx), u_k = gain_k x_k + feedforward_k
    cost_to_go: np.ndarray  # (N + 1, nx, nx), P_k, with P_N the terminal Hessian


def _factor(qp: StageQP, d_q: np.ndarray | None, d_p: np.ndarray | None) -> _Factor:
    """Riccati recursion on the Hessians, eqs. (8.52) and (8.53), with P_k symmetrized."""
    n, nx, nu = qp.B.shape
    hess = qp.Q if d_q is None else qp.Q + d_q
    p_mat = qp.P if d_p is None else qp.P + d_p
    cost_to_go = np.empty((n + 1, nx, nx))
    cost_to_go[n] = p_mat
    gain = np.empty((n, nu, nx))
    chol: list[tuple[np.ndarray, bool]] = [(np.empty(0), True)] * n
    for k in range(n - 1, -1, -1):
        a, b = qp.A[k], qp.B[k]
        pa, pb = p_mat @ a, p_mat @ b
        h_xx = hess[k, :nx, :nx] + a.T @ pa
        h_ux = hess[k, nx:, :nx] + b.T @ pa
        h_uu = hess[k, nx:, nx:] + b.T @ pb
        if not np.all(np.isfinite(h_uu)):
            raise ValueError(f"stage {k} Hessian is not finite")
        try:
            factor = cho_factor(h_uu, lower=True, check_finite=False)
        except LinAlgError:
            raise ValueError(f"input block of stage {k} is not positive definite") from None
        chol[k] = factor
        gain[k] = -cho_solve(factor, h_ux, check_finite=False)
        p_mat = h_xx + h_ux.T @ gain[k]
        p_mat = 0.5 * (p_mat + p_mat.T)
        cost_to_go[k] = p_mat
    return _Factor(chol, gain, cost_to_go)


def _sweep(
    qp: StageQP, factor: _Factor, q: np.ndarray, p: np.ndarray, c: np.ndarray, x0: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient recursion, eq. (8.54), and the forward rollout for given linear terms."""
    n, nx, nu = qp.B.shape
    p_vec = np.empty((n + 1, nx))
    p_vec[n] = p
    feedforward = np.empty((n, nu))
    for k in range(n - 1, -1, -1):
        a, b = qp.A[k], qp.B[k]
        grad_next = factor.cost_to_go[k + 1] @ c[k] + p_vec[k + 1]
        h_u = q[k, nx:] + b.T @ grad_next
        feedforward[k] = -cho_solve(factor.chol[k], h_u, check_finite=False)
        p_vec[k] = q[k, :nx] + a.T @ grad_next + factor.gain[k].T @ h_u
    x = np.empty((n + 1, nx))
    u = np.empty((n, nu))
    y = np.empty((n, nx))
    x[0] = x0
    for k in range(n):
        u[k] = factor.gain[k] @ x[k] + feedforward[k]
        x[k + 1] = qp.A[k] @ x[k] + qp.B[k] @ u[k] + c[k]
        y[k] = factor.cost_to_go[k + 1] @ x[k + 1] + p_vec[k + 1]
    return x, u, y


def riccati_solve(
    qp: StageQP,
    extra: Sequence[np.ndarray] | None = None,
    rhs: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the horizon problem without its rows by a backward Riccati sweep and a rollout.

    The rows `C`, `e`, `C_N`, `e_N` are ignored. The cost is O(N (nx + nu)^3); the inputs are not
    condensed (Rawlings, Mayne and Diehl, *Model Predictive Control*, 2nd ed., sec. 8.8.3,
    eqs. (8.51) to (8.54); Boyd and Vandenberghe, *Convex Optimization*, sec. 10.4.3).

    Parameters
    ----------
    qp : StageQP
        The problem.
    extra : sequence of four arrays, optional
        Terms (dQ, dq, dP, dp) of shapes (N, nx + nu, nx + nu), (N, nx + nu), (nx, nx) and (nx,)
        added to Q, q, P and p for this solve, as the barrier terms of an interior-point step.
        dQ and dP must be symmetric positive semidefinite.
    rhs : sequence of two arrays, optional
        (c, x0) of shapes (N, nx) and (nx,) used in place of the problem's affine terms and
        initial state, as the residuals of a Newton step.

    Returns
    -------
    x : ndarray, shape (N + 1, nx)
        States, with x[0] equal to the initial state.
    u : ndarray, shape (N, nu)
        Inputs.
    y : ndarray, shape (N, nx)
        Multipliers of the dynamics in the Lagrangian
        f + sum_k y_k^T (A_k x_k + B_k u_k + c_k - x_{k+1}); y_k is the gradient of the cost to
        go at x_{k+1}.

    Raises
    ------
    ValueError
        If an input block of the recursion is not positive definite, or `extra` or `rhs` has
        the wrong shapes.
    """
    n, nx, nu = qp.B.shape
    nz = nx + nu
    c, x0 = qp.c, qp.x0
    if rhs is not None:
        if len(rhs) != 2:
            raise ValueError("rhs must be (c, x0)")
        c, x0 = _array(rhs[0], (n, nx), "c"), _array(rhs[1], (nx,), "x0")
    if extra is None:
        return _sweep(qp, _factor(qp, None, None), qp.q, qp.p, c, x0)
    if len(extra) != 4:
        raise ValueError("extra must be (dQ, dq, dP, dp)")
    shapes = ((n, nz, nz), (n, nz), (nx, nx), (nx,))
    d_q, d_qv, d_p, d_pv = (
        _array(term, shape, name)
        for term, shape, name in zip(extra, shapes, ("dQ", "dq", "dP", "dp"), strict=True)
    )
    return _sweep(qp, _factor(qp, d_q, d_p), qp.q + d_qv, qp.p + d_pv, c, x0)
