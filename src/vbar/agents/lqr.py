"""Discrete LQR design on the zero-order-hold CW model.

Gains follow u = -K x with K = (R + Gamma' P Gamma)^-1 (Gamma' P Phi + S'), P the stabilizing
solution of the discrete Riccati equation with cross term S (Rawlings, Mayne and Diehl, Model
Predictive Control, 2nd ed., sections 1.2.5, 1.5.1 and 8.5.2; Alfriend et al., Spacecraft
Formation Flying, chapter 3, whose printed gain lacks the (R + Gamma' P Gamma)^-1 factor).
Every matrix is in SI units of the Hill frame unless stated.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike
from scipy.linalg import expm, solve_discrete_are

# The scaled Riccati residual sits at 1e-14 for steps from 1e-6 s to 1e4 s; a solve that lost
# the stabilizing solution (an unstabilizable pair gives 1e-5) is far above this.
RICCATI_TOL = 1e-10
# A negative eigenvalue of the sampled stage cost beyond this fraction of its largest is a loss
# of positive semidefiniteness in the exponential rather than rounding, and is clipped.
COST_EIGENVALUE_TOL = 1e-12


def _cost_scales(q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Jacobi scales that give the diagonals of Q and R unit size; zero weights keep SI."""
    scales = []
    for weight in (q, r):
        diagonal = np.diag(weight)
        scale = np.ones(len(diagonal))
        weighted = diagonal > 0.0
        scale[weighted] = 1.0 / np.sqrt(diagonal[weighted])
        scales.append(scale)
    return scales[0], scales[1]


def dlqr(
    phi: ArrayLike, gamma: ArrayLike, q: ArrayLike, r: ArrayLike, s: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Infinite-horizon discrete LQR gain and cost matrix.

    Minimizes the sum of x' Q x + u' R u + 2 x' S u over x_{k+1} = Phi x_k + Gamma u_k.

    Parameters
    ----------
    phi : array_like, shape (n, n)
        State transition matrix of one step.
    gamma : array_like, shape (n, m)
        Input matrix of one step; for the CW model the input is a Hill acceleration (m/s^2).
    q : array_like, shape (n, n)
        State weight, symmetric positive semidefinite.
    r : array_like, shape (m, m)
        Input weight, symmetric positive definite.
    s : array_like, shape (n, m), optional
        Cross weight; zero when omitted.

    Returns
    -------
    k : numpy.ndarray, shape (m, n)
        Feedback gain, u = -K x.
    p : numpy.ndarray, shape (n, n)
        Symmetric cost matrix, x' P x the optimal cost from x.

    Raises
    ------
    ValueError
        If the Riccati solve leaves a residual above 1e-10 of P (an unstabilizable pair, or a
        pair the solver cannot resolve), or the closed loop Phi - Gamma K is not stable.
    """
    phi = np.asarray(phi, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    s = np.zeros(gamma.shape) if s is None else np.asarray(s, dtype=float)
    # The weights span many decades in SI units (1e-2 on positions, 1e4 on velocities, 1e3 on
    # accelerations), which the solver's own balancing does not remove: the residual of the
    # plain solve reaches 1e-8 at long steps. In the scaled coordinates it stays at rounding.
    dx, du = _cost_scales(q, r)
    phi_s = phi * dx[None, :] / dx[:, None]
    gamma_s = gamma * du[None, :] / dx[:, None]
    q_s = q * dx[:, None] * dx[None, :]
    r_s = r * du[:, None] * du[None, :]
    s_s = s * dx[:, None] * du[None, :]
    try:
        p_s = solve_discrete_are(phi_s, gamma_s, q_s, r_s, s=s_s)
    except LinAlgError as e:
        raise ValueError("riccati solution did not converge") from e
    p_s = 0.5 * (p_s + p_s.T)
    k_s = np.linalg.solve(r_s + gamma_s.T @ p_s @ gamma_s, gamma_s.T @ p_s @ phi_s + s_s.T)
    residual = q_s + phi_s.T @ p_s @ phi_s - (phi_s.T @ p_s @ gamma_s + s_s) @ k_s - p_s
    if not np.max(np.abs(residual)) <= RICCATI_TOL * np.max(np.abs(p_s)):
        raise ValueError("riccati solution did not converge")
    p = p_s / dx[:, None] / dx[None, :]
    p = 0.5 * (p + p.T)
    k = np.linalg.solve(r + gamma.T @ p @ gamma, gamma.T @ p @ phi + s.T)
    if not np.max(np.abs(np.linalg.eigvals(phi - gamma @ k))) < 1.0:
        raise ValueError("lqr closed loop is not stable")
    return k, p


def sampled_data_cost(
    a: ArrayLike, b: ArrayLike, qc: ArrayLike, rc: ArrayLike, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact stage cost of a zero-order hold step for a continuous quadratic cost.

    Integrates x(t)' Qc x(t) + u' Rc u over one step of x' = A x + B u with u held, so that the
    discrete problem with (Qd, Rd, Sd) has the continuous optimum at the samples (Van Loan,
    1978; Rawlings, Mayne and Diehl section 8.5.2).

    Parameters
    ----------
    a : array_like, shape (n, n)
        Continuous system matrix (1/s on the velocity rows).
    b : array_like, shape (n, m)
        Continuous input matrix.
    qc : array_like, shape (n, n)
        Continuous state weight, symmetric positive semidefinite (per second).
    rc : array_like, shape (m, m)
        Continuous input weight, symmetric positive definite (per second).
    dt : float
        Step length (s).

    Returns
    -------
    qd : numpy.ndarray, shape (n, n)
        Discrete state weight.
    rd : numpy.ndarray, shape (m, m)
        Discrete input weight.
    sd : numpy.ndarray, shape (n, m)
        Discrete cross weight, the stage cost being x' Qd x + u' Rd u + 2 x' Sd u.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    qc = np.asarray(qc, dtype=float)
    rc = np.asarray(rc, dtype=float)
    n, m = b.shape
    dx, du = _cost_scales(qc, rc)
    d = np.concatenate([dx, du])
    # Van Loan's block matrix on z = (x, u) with u' = 0, in the cost-scaled coordinates so the
    # exponential mixes entries of one size; the cost block is scaled back at the end.
    az = np.zeros((n + m, n + m))
    az[:n, :n] = a
    az[:n, n:] = b
    az = az * d[None, :] / d[:, None]
    qz = np.zeros((n + m, n + m))
    qz[:n, :n] = qc
    qz[n:, n:] = rc
    qz = qz * d[:, None] * d[None, :]
    big = np.zeros((2 * (n + m), 2 * (n + m)))
    big[: n + m, : n + m] = -az.T
    big[: n + m, n + m :] = qz
    big[n + m :, n + m :] = az
    e = expm(big * dt)
    g = e[n + m :, n + m :].T @ e[: n + m, n + m :]
    g = 0.5 * (g + g.T)
    w, v = np.linalg.eigh(g)
    # Reconstructing from eigenvectors perturbs the smallest block at 1e-9 of itself, which
    # breaks known answers, so it is done only when the block is indefinite beyond rounding.
    if w[0] < -COST_EIGENVALUE_TOL * w[-1]:
        g = (v * np.maximum(w, 0.0)) @ v.T
    g = g / d[:, None] / d[None, :]
    g = 0.5 * (g + g.T)
    return g[:n, :n], g[n:, n:], g[:n, n:]


def delay_augmented(phi: ArrayLike, gamma: ArrayLike, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Model with the input delayed by `d` whole steps.

    The state becomes z_k = (x_k, u_{k-d}, ..., u_{k-1}) and the input u_k, so that
    x_{k+1} = Phi x_k + Gamma u_{k-d} (Rawlings, Mayne and Diehl section 1.2.5).

    Parameters
    ----------
    phi : array_like, shape (n, n)
        State transition matrix of one step.
    gamma : array_like, shape (n, m)
        Input matrix of one step.
    d : int
        Delay in steps, at least 0.

    Returns
    -------
    phi_a : numpy.ndarray, shape (n + d m, n + d m)
        Transition matrix on z.
    gamma_a : numpy.ndarray, shape (n + d m, m)
        Input matrix on z; with `d` = 0 the pair is the plain model.

    Raises
    ------
    ValueError
        If `d` is negative.
    """
    phi = np.asarray(phi, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if d < 0:
        raise ValueError("delay must not be negative")
    n, m = gamma.shape
    phi_a = np.zeros((n + d * m, n + d * m))
    gamma_a = np.zeros((n + d * m, m))
    phi_a[:n, :n] = phi
    if d == 0:
        gamma_a[:] = gamma
        return phi_a, gamma_a
    phi_a[:n, n : n + m] = gamma
    # Each stored command moves one slot toward the plant; the newest enters at the end.
    for j in range(d - 1):
        lo = n + j * m
        phi_a[lo : lo + m, lo + m : lo + 2 * m] = np.eye(m)
    gamma_a[n + (d - 1) * m :] = np.eye(m)
    return phi_a, gamma_a


def rate_augmented(
    phi: ArrayLike, gamma: ArrayLike, q: ArrayLike, r: ArrayLike, s: ArrayLike, rd: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Model and cost with the previous input as a state and a weight on the input change.

    The state becomes z_k = (x_k, u_{k-1}) and the stage cost gains (u_k - u_{k-1})' Rd
    (u_k - u_{k-1}) (Rawlings, Mayne and Diehl exercise 1.25).

    Parameters
    ----------
    phi : array_like, shape (n, n)
        State transition matrix of one step.
    gamma : array_like, shape (n, m)
        Input matrix of one step.
    q : array_like, shape (n, n)
        State weight.
    r : array_like, shape (m, m)
        Input weight.
    s : array_like, shape (n, m)
        Cross weight.
    rd : array_like, shape (m, m)
        Weight on the input change, symmetric positive semidefinite.

    Returns
    -------
    phi_r, gamma_r, q_r, r_r, s_r : numpy.ndarray
        The model and the weights on z and u_k, shapes (n + m, n + m), (n + m, m),
        (n + m, n + m), (m, m) and (n + m, m), with the same stage cost as before plus the
        change term.
    """
    phi = np.asarray(phi, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    rd = np.asarray(rd, dtype=float)
    n, m = gamma.shape
    phi_r = np.zeros((n + m, n + m))
    phi_r[:n, :n] = phi
    gamma_r = np.vstack([gamma, np.eye(m)])
    q_r = np.zeros((n + m, n + m))
    q_r[:n, :n] = q
    q_r[n:, n:] = rd
    # (u - u_prev)' Rd (u - u_prev) expands into u' Rd u, u_prev' Rd u_prev and -2 u_prev' Rd u.
    return phi_r, gamma_r, q_r, r + rd, np.vstack([s, -rd])


def steady_state_input(phi: ArrayLike, gamma: ArrayLike, xs: ArrayLike) -> np.ndarray:
    """Constant input that holds the state `xs` fixed under one step of the model.

    Solves Gamma us = (I - Phi) xs in the least-squares sense (Rawlings, Mayne and Diehl
    section 1.5.1); for a CW hold point at rest that is -3 n^2 x on the radial axis and
    n^2 z on the normal axis.

    Parameters
    ----------
    phi : array_like, shape (n, n)
        State transition matrix of one step.
    gamma : array_like, shape (n, m)
        Input matrix of one step.
    xs : array_like, shape (n,)
        State to hold (m, m/s in the Hill frame).

    Returns
    -------
    numpy.ndarray, shape (m,)
        The holding input, a Hill acceleration (m/s^2) for the CW model.

    Raises
    ------
    ValueError
        If no constant input holds `xs`, such as a hold point with a velocity.
    """
    phi = np.asarray(phi, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    xs = np.asarray(xs, dtype=float)
    drift = (np.eye(len(xs)) - phi) @ xs
    us = np.linalg.lstsq(gamma, drift, rcond=None)[0]
    mismatch = np.max(np.abs(gamma @ us - drift))
    size = np.max(np.abs(drift)) + np.max(np.abs(gamma)) * np.max(np.abs(us))
    if mismatch > 1e-9 * size:
        raise ValueError("hold point is not a steady state")
    return us
