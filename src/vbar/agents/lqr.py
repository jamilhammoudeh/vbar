"""Discrete LQR design on the zero-order-hold CW model, and the LQR baseline agent.

Gains follow u = -K x with K = (R + Gamma' P Gamma)^-1 (Gamma' P Phi + S'), P the stabilizing
solution of the discrete Riccati equation with cross term S (Rawlings, Mayne and Diehl, Model
Predictive Control, 2nd ed., sections 1.2.5, 1.5.1 and 8.5.2; Alfriend et al., Spacecraft
Formation Flying, chapter 3, whose printed gain lacks the (R + Gamma' P Gamma)^-1 factor).
Every matrix is in SI units of the Hill frame unless stated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike
from scipy.linalg import expm, solve_discrete_are

from ..dynamics import cw_discrete, cw_system
from .base import AgentView, action_scale, hold_point

# The scaled Riccati residual sits at 1e-14 for steps from 1e-6 s to 1e4 s; a solve that lost
# the stabilizing solution (an unstabilizable pair gives 1e-5) is far above this.
RICCATI_TOL = 1e-10
# A negative eigenvalue of the sampled stage cost beyond this fraction of its largest is a loss
# of positive semidefiniteness in the exponential rather than rounding, and is clipped.
COST_EIGENVALUE_TOL = 1e-12
# The closed-form CW step and the exponential of the CW system agree to 1e-15 at ordinary steps;
# a larger gap means the sampled cost would be built for another model than the one flown.
MODEL_TOL = 1e-12
# Waits are whole multiples of dt up to the rounding of the environment's clock.
ALIGNMENT_TOL = 1e-6
STAGE_COSTS = ("sampled", "discrete")


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


@dataclass(frozen=True, kw_only=True)
class LQRConfig:
    """Weights of the LQR baseline.

    The continuous cost is the integral over time of |r / pos_scale|^2 + |v / vel_scale|^2 +
    |u / accel_scale|^2, with r and v the deviation from the hold point (Hill frame) and u the
    commanded acceleration.

    Parameters
    ----------
    pos_scale_m : float
        Position error that costs 1 (m).
    vel_scale_mps : float
        Velocity error that costs 1 (m/s). When pos_scale_m / vel_scale_mps is long next to
        the step it is the time constant of the slowest closed-loop mode, 333 s at the
        defaults, short enough to reach and hold the approach-rbar point within its 3600 s;
        0.01 m/s gives 1000 s and leaves its starts 6 to 10 m out.
    accel_scale_mps2 : float or None
        Command that costs 1 (m/s^2); None uses the view's `max_accel`, so the weights follow
        the scenario's thrust.
    stage_cost : {"sampled", "discrete"}
        "sampled" minimizes the continuous cost exactly over zero-order-hold steps, with the
        stage cost by Van Loan's method, so the law does not depend on `dt`; "discrete" sums the
        same quadratic once per step.
    rate_weight : float
        Weight on |(u_k - u_{k-1}) / accel_scale|^2 per step; 0 turns it off.

    Raises
    ------
    ValueError
        If a scale is not positive, `stage_cost` is not one of the two, or `rate_weight` is
        negative.
    """

    pos_scale_m: float = 10.0
    vel_scale_mps: float = 0.03
    accel_scale_mps2: float | None = None
    stage_cost: str = "sampled"
    rate_weight: float = 0.0

    def __post_init__(self):
        if not self.pos_scale_m > 0.0:
            raise ValueError("pos_scale_m must be positive")
        if not self.vel_scale_mps > 0.0:
            raise ValueError("vel_scale_mps must be positive")
        if self.accel_scale_mps2 is not None and not self.accel_scale_mps2 > 0.0:
            raise ValueError("accel_scale_mps2 must be positive")
        if self.stage_cost not in STAGE_COSTS:
            raise ValueError(f"stage_cost must be one of {STAGE_COSTS}")
        if not self.rate_weight >= 0.0:
            raise ValueError("rate_weight must not be negative")

    @classmethod
    def alfriend(cls, n: float) -> LQRConfig:
        """Scales that give Q = diag(1, 1, 1, n^-2, n^-2, n^-2) and R = I / n^4 in SI units.

        A metre, a metre per radian of orbit and a metre per radian squared cost the same
        (Alfriend et al., Spacecraft Formation Flying, section 10.1.2), so the gains scale with
        the orbit and carry over between bodies.

        Parameters
        ----------
        n : float
            Mean motion of the target orbit (rad/s).

        Returns
        -------
        LQRConfig
            Config with these scales and the other fields at their defaults.
        """
        return cls(pos_scale_m=1.0, vel_scale_mps=n, accel_scale_mps2=n**2)


def _pending_slots(names: tuple[str, ...]) -> list[list[int]]:
    """Observation indices of (wait, x, y, z) of each pending-command slot, in slot order."""
    count = sum(1 for name in names if re.fullmatch(r"pending_\d+_t", name))
    return [[names.index(f"pending_{j}_{c}") for c in "txyz"] for j in range(count)]


def _check_zero_order_hold(
    a: np.ndarray, b: np.ndarray, phi: np.ndarray, gamma: np.ndarray, dt: float
) -> None:
    """Raise if expm([[A, B], [0, 0]] dt) differs from the closed-form step (phi, gamma)."""
    big = np.zeros((9, 9))
    big[:6, :6] = a
    big[:6, 6:] = b
    e = expm(big * dt)
    for block, ref in ((e[:6, :6], phi), (e[:6, 6:], gamma)):
        if not np.max(np.abs(block - ref)) <= MODEL_TOL * np.max(np.abs(ref)):
            raise ValueError("cw step and system matrices disagree")


@dataclass(frozen=True)
class RegulatorModel:
    """Regulator design on the CW model, augmented for delay and rate, about the hold point.

    The augmented state is z = (x, u_{k-d}, ..., u_{k-1}) with d pending-command slots, or
    z = (x, u_{k-1}) when a rate weight is set and there is no delay, or z = x. Commands are
    Hill accelerations (m/s^2); x is the Hill state (m, m/s). The arrays are read-only, so one
    model can be shared between controllers; copy one before changing it.

    Parameters
    ----------
    phi : numpy.ndarray, shape (nz, nz)
        Transition matrix of one step on z.
    gamma : numpy.ndarray, shape (nz, 3)
        Input matrix of one step on z.
    q : numpy.ndarray, shape (nz, nz)
        Stage weight on z - zs.
    r : numpy.ndarray, shape (3, 3)
        Stage weight on u - us.
    s : numpy.ndarray, shape (nz, 3)
        Stage cross weight, the stage cost being z' Q z + u' R u + 2 z' S u on the deviations.
    k : numpy.ndarray, shape (3, nz)
        LQR gain, u = us - K (z - zs).
    p : numpy.ndarray, shape (nz, nz)
        Symmetric DARE solution, (z - zs)' P (z - zs) the optimal cost from z.
    xs : numpy.ndarray, shape (6,)
        Hold point at rest (m, m/s).
    us : numpy.ndarray, shape (3,)
        Constant input that holds xs (m/s^2).
    zs : numpy.ndarray, shape (nz,)
        xs followed by one copy of us per augmented command.
    slots : list of list of int
        Observation indices of (wait, x, y, z) of each pending-command slot, oldest first.
    own_last : bool
        Whether z ends with the controller's own last command (rate weight without delay).
    """

    phi: np.ndarray
    gamma: np.ndarray
    q: np.ndarray
    r: np.ndarray
    s: np.ndarray
    k: np.ndarray
    p: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    zs: np.ndarray
    slots: list[list[int]]
    own_last: bool


def regulator_model(view: AgentView, config: LQRConfig) -> RegulatorModel:
    """Design the LQR regulator for the view's orbit, step, thrust and delay.

    The design shared by `LQRAgent` and the predictive controllers, so that they agree wherever
    no constraint binds.

    Parameters
    ----------
    view : AgentView
        Scenario description; uses `n`, `dt`, `max_accel`, `thrust_limit`, `action_mode`,
        `mass_kg`, `goal["hold_point"]` (Hill frame, m) and the observation names.
    config : LQRConfig
        Weights, stage-cost convention and rate weight.

    Returns
    -------
    RegulatorModel
        The augmented model, stage cost, gain, DARE solution and hold point.

    Raises
    ------
    ValueError
        If the goal is not a hold or a berthing or has no hold point, the action mode takes no
        continuous command, the thrust layout is unknown, the hold point needs more than
        `max_accel` of steady thrust, or the design fails (see `dlqr`).
    """
    hold = hold_point(view, "LQR agent")
    action_scale(view)
    if view.thrust_limit not in ("box", "ball"):
        raise ValueError(f"unknown thrust limit {view.thrust_limit!r}")
    accel_scale = view.max_accel if config.accel_scale_mps2 is None else config.accel_scale_mps2
    phi, gamma = cw_discrete(view.n, view.dt)
    qc = np.diag([config.pos_scale_m**-2] * 3 + [config.vel_scale_mps**-2] * 3)
    rc = np.eye(3) / accel_scale**2
    if config.stage_cost == "sampled":
        a, b = cw_system(view.n)
        _check_zero_order_hold(a, b, phi, gamma, view.dt)
        q, r, s = sampled_data_cost(a, b, qc, rc, view.dt)
    else:
        q, r, s = qc, rc, np.zeros((6, 3))

    slots = _pending_slots(view.observation)
    d = len(slots)
    phi_z, gamma_z = delay_augmented(phi, gamma, d)
    q_z = np.zeros(phi_z.shape)
    q_z[:6, :6] = q
    s_z = np.zeros(gamma_z.shape)
    if d:
        # The cross term pairs x_k with the input acting on it, u_{k-d}, and the input weight
        # moves to u_k: the infinite sum changes only by the already committed inputs.
        q_z[:6, 6:9] = s
        q_z[6:9, :6] = s.T
    else:
        s_z[:] = s
    r_z = r
    own_last = config.rate_weight > 0.0 and d == 0
    if config.rate_weight > 0.0:
        rd = config.rate_weight * np.eye(3) / accel_scale**2
        if own_last:
            phi_z, gamma_z, q_z, r_z, s_z = rate_augmented(phi_z, gamma_z, q_z, r_z, s_z, rd)
        else:
            # u_{k-1} is already the newest pending slot.
            q_z[-3:, -3:] += rd
            s_z[-3:] -= rd
            r_z = r + rd
    k, p = dlqr(phi_z, gamma_z, q_z, r_z, s_z)

    xs = np.concatenate([hold, np.zeros(3)])
    us = steady_state_input(phi, gamma, xs)
    if view.thrust_limit == "box":
        size = float(np.max(np.abs(us)))
    else:
        size = float(np.linalg.norm(us))
    if size > view.max_accel:
        raise ValueError(
            f"hold point needs {size:.2e} m/s^2 of steady thrust, above max_accel {view.max_accel}"
        )
    zs = np.concatenate([xs, np.tile(us, (len(phi_z) - 6) // 3)])
    for array in (phi_z, gamma_z, q_z, r_z, s_z, k, p, xs, us, zs):
        array.flags.writeable = False
    return RegulatorModel(
        phi=phi_z,
        gamma=gamma_z,
        q=q_z,
        r=r_z,
        s=s_z,
        k=k,
        p=p,
        xs=xs,
        us=us,
        zs=zs,
        slots=slots,
        own_last=own_last,
    )


class LQRAgent:
    """Discrete LQR regulator to the hold point on the CW model, with target-input feedforward.

    The command is u = us - K (z - zs) (Hill frame, m/s^2): z is the observed state followed
    by the commands not yet acting on it, zs the hold point at rest followed by copies of us, and
    us the constant input that holds it (Rawlings, Mayne and Diehl, Model Predictive Control,
    2nd ed., sections 1.2.5 and 1.5.1). Each pending-command slot adds one step of delay to the
    model; with a rate weight and no delay the agent keeps its own last command as a state. The
    command is clipped per axis (box layout) or scaled onto the ball (ball layout) at
    `max_accel`, then issued in the view's action mode. Constraints are not handled.

    Parameters
    ----------
    config : LQRConfig, optional
        Weights; the defaults of `LQRConfig` when omitted.
    name : str
        Name in results.
    """

    def __init__(self, config: LQRConfig | None = None, name: str = "lqr"):
        self.config = LQRConfig() if config is None else config
        self.name = name

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Design the gain for the view's orbit, step, thrust and delay.

        Parameters
        ----------
        view : AgentView
            Scenario description; uses `n`, `dt`, `max_accel`, `thrust_limit`, `action_mode`,
            `mass_kg`, `goal["hold_point"]` (Hill frame, m) and the observation names.
        rng : numpy.random.Generator, optional
            Unused; the law is deterministic.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, the action mode takes
            no continuous command, the thrust layout is unknown, the hold point needs more than
            `max_accel` of steady thrust, or the design fails (see `dlqr`).
        """
        model = regulator_model(view, self.config)
        self.scale = action_scale(view)
        self.view = view
        self.slots = model.slots
        self.own_last = model.own_last
        self.k = model.k
        self.us = model.us
        self.zs = model.zs
        self.last = np.zeros(3)

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Command toward the hold point.

        Parameters
        ----------
        obs : numpy.ndarray
            Observation in the order of the view's names: the Hill state (m, m/s) first, then
            the pending-command slots if enabled.
        info : dict
            Unused.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Command in the view's action mode.

        Raises
        ------
        ValueError
            If a pending command's wait is not a whole number of steps.
        """
        view = self.view
        obs = np.asarray(obs, dtype=float)
        parts = [obs[:6]]
        if self.slots:
            pending = obs[self.slots]
            steps = pending[:, 0] / view.dt
            if np.any(np.abs(steps - np.round(steps)) > ALIGNMENT_TOL):
                raise ValueError("pending commands are not aligned to the decision step")
            # Each command is echoed in the action units of its issue, which are the units this
            # agent sent it in, so max_accel / scale gives back the acceleration it asked for.
            parts.append((pending[:, 1:] * (view.max_accel / self.scale)).ravel())
        elif self.own_last:
            parts.append(self.last)
        u = self.us - self.k @ (np.concatenate(parts) - self.zs)
        if view.thrust_limit == "box":
            u = np.clip(u, -view.max_accel, view.max_accel)
        else:
            size = np.linalg.norm(u)
            if size > view.max_accel:
                u = u * (view.max_accel / size)
        self.last = u
        return u / view.max_accel * self.scale
