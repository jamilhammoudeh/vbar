"""Steady-state Kalman filter design, bias detectability, and the LQG baseline agent.

The filter is x_{k+1} = Phi x_k + w_k, y_k = C x_k + v_k with w ~ N(0, Qd) and v ~ N(0, Rv); its
predicted covariance solves the filter Riccati equation, the dual of the LQR one, and the updated
covariance follows the Joseph form (Crassidis and Junkins, Optimal Estimation of Dynamic Systems,
2nd ed., sections 3.3.2 and 3.3.4). Constant biases and disturbances are tested by Lemma 1.8 of
Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike
from scipy.linalg import matrix_balance, solve_discrete_are

from ..dynamics import cw_discrete
from .base import AgentView, action_scale, hold_point
from .lqr import LQRAgent, LQRConfig, _pending_slots

# The scaled filter Riccati residual stays below 1e-13 of P on the CW model at steps from 1 s to
# 1e3 s and position sigmas from 1e-3 m to 1 m; a solve that lost the stabilizing solution is far
# above this.
RICCATI_TOL = 1e-10
# A Jordan block of size two on the unit circle, as the along-track drift of the CW model, is
# computed with eigenvalues off by the square root of the rounding (1.5e-8), so modes this close
# to the circle are tested as if on it.
UNIT_CIRCLE_TOL = 1e-6
# Rank threshold of the balanced and equilibrated PBH and Lemma 1.8 matrices. On the CW model at
# steps from 1 s to 1e4 s a missing mode or bias leaves the deciding singular value below 2e-13 of
# the largest and a present one above 4e-4. Below 1e-3 s every mode crowds at 1 and the test
# cannot separate them.
RANK_TOL = 1e-8
# A process covariance eigenvalue below this fraction of the largest is negative beyond rounding.
COVARIANCE_TOL = 1e-12
STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")
# A constant along-track position bias is missing: a constant y offset is itself a CW
# equilibrium, so it cannot be told from the state (`bias_detectable`).
BIAS_STATES = ("x", "z", "vx", "vy", "vz")


def _balanced(phi: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Phi and G after the diagonal similarity that balances Phi."""
    phi_b, (scale, _) = matrix_balance(phi, permute=False, separate=True)
    return phi_b, g / scale[:, None]


def _full_rank(m: np.ndarray, rank: int) -> bool:
    """Whether `m` has rank `rank` after equilibrating its rows and columns."""
    for _ in range(2):
        row = np.max(np.abs(m), axis=1)
        m = m / np.where(row > 0.0, row, 1.0)[:, None]
        col = np.max(np.abs(m), axis=0)
        m = m / np.where(col > 0.0, col, 1.0)[None, :]
    sv = np.linalg.svd(m, compute_uv=False)
    return len(sv) >= rank and bool(sv[rank - 1] > RANK_TOL * sv[0])


def _stabilizable(phi: np.ndarray, g: np.ndarray) -> bool:
    """PBH test: rank [lambda I - Phi, G] = n at every eigenvalue on or outside the unit circle."""
    phi_b, g_b = _balanced(phi, g)
    n = len(phi_b)
    eigenvalues = np.linalg.eigvals(phi_b)
    for lam in eigenvalues[np.abs(eigenvalues) >= 1.0 - UNIT_CIRCLE_TOL]:
        if not _full_rank(np.hstack([lam * np.eye(n) - phi_b, g_b]), n):
            return False
    return True


def _measurement_scales(c: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Scale of each state: the least sigma of a measurement that sees it, else 1 (SI)."""
    scale = np.ones(c.shape[1])
    for i in range(c.shape[1]):
        seen = c[:, i] != 0.0
        if np.any(seen):
            scale[i] = np.min(sigma[seen] / np.abs(c[seen, i]))
    return scale


def kalman_gain(
    phi: ArrayLike, c: ArrayLike, qd: ArrayLike, rv: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steady-state Kalman gain and covariances of a discrete linear model.

    The update is xhat^+ = xhat^- + L (y - C xhat^-) and the prediction xhat^- = Phi xhat^+
    plus the known input.

    Parameters
    ----------
    phi : array_like, shape (n, n)
        State transition matrix of one step.
    c : array_like, shape (p, n)
        Measurement matrix.
    qd : array_like, shape (n, n)
        Process noise covariance added per step, symmetric positive semidefinite; for a white
        acceleration of sigma a (m/s^2) held over each step on the CW model, a^2 Gamma Gamma'.
    rv : array_like, shape (p, p)
        Measurement noise covariance, symmetric positive definite (m^2 and m^2/s^2 for a Hill
        position and velocity).

    Returns
    -------
    l : numpy.ndarray, shape (n, p)
        Filter gain.
    p_minus : numpy.ndarray, shape (n, n)
        Steady-state covariance before the update.
    p_plus : numpy.ndarray, shape (n, n)
        Steady-state covariance after the update, by the Joseph form
        (I - L C) P^- (I - L C)' + L Rv L'.

    Raises
    ------
    ValueError
        If `qd` is zero, if it is not positive semidefinite, if `rv` is not positive definite,
        if the process noise misses a mode on or outside the unit circle or a measurement
        misses one (PBH test), or if the Riccati solve leaves a residual above 1e-10 of P.
    """
    phi = np.asarray(phi, dtype=float)
    c = np.asarray(c, dtype=float)
    qd = np.asarray(qd, dtype=float)
    rv = np.asarray(rv, dtype=float)
    w = np.linalg.eigvalsh(qd)
    if not w[-1] > 0.0:
        raise ValueError("process noise must be positive: with none the filter gain collapses")
    if w[0] < -COVARIANCE_TOL * w[-1]:
        raise ValueError("process noise covariance must be positive semidefinite")
    if not np.linalg.eigvalsh(rv)[0] > 0.0:
        raise ValueError("measurement noise covariance must be positive definite")
    # Without noise on a mode that does not decay the steady-state gain on it is zero, and the
    # filter never corrects an error there. Qd has the range of its square root, and testing it
    # directly keeps the exact zeros an eigendecomposition would blur.
    if not _stabilizable(phi, qd):
        raise ValueError("process noise does not excite every mode that does not decay")
    if not _stabilizable(phi.T, c.T):
        raise ValueError("measurements do not see every mode that does not decay")
    # Covariances span many decades in SI units (1e-6 m^2 against 1e-12 m^2/s^2 at cheap
    # sigmas), and the unscaled solve is 1e-7 off in P there. Scaling each state by the sigma of
    # the measurement that sees it puts the steady state at order one.
    sigma = np.sqrt(np.diag(rv))
    t = _measurement_scales(c, sigma)
    phi_s = phi * t[None, :] / t[:, None]
    c_s = c * t[None, :] / sigma[:, None]
    qd_s = qd / t[:, None] / t[None, :]
    rv_s = rv / sigma[:, None] / sigma[None, :]
    try:
        p_s = solve_discrete_are(phi_s.T, c_s.T, qd_s, rv_s)
    except LinAlgError as e:
        raise ValueError("riccati solution did not converge") from e
    p_s = 0.5 * (p_s + p_s.T)
    innovation = c_s @ p_s @ c_s.T + rv_s
    l_s = np.linalg.solve(innovation, c_s @ p_s).T
    a = np.eye(len(phi)) - l_s @ c_s
    p_plus_s = a @ p_s @ a.T + l_s @ rv_s @ l_s.T
    p_plus_s = 0.5 * (p_plus_s + p_plus_s.T)
    residual = phi_s @ p_plus_s @ phi_s.T + qd_s - p_s
    if not np.max(np.abs(residual)) <= RICCATI_TOL * np.max(np.abs(p_s)):
        raise ValueError("riccati solution did not converge")
    p_minus = p_s * t[:, None] * t[None, :]
    p_plus = p_plus_s * t[:, None] * t[None, :]
    gain = l_s * t[:, None] / sigma[None, :]
    return gain, 0.5 * (p_minus + p_minus.T), 0.5 * (p_plus + p_plus.T)


def bias_detectable(phi: ArrayLike, c: ArrayLike, bd: ArrayLike, cd: ArrayLike) -> bool:
    """Whether a model augmented with constant disturbances is detectable.

    The augmented model is x_{k+1} = Phi x_k + Bd d_k, d_{k+1} = d_k, y_k = C x_k + Cd d_k. By
    Lemma 1.8 and eq 1.44 of Rawlings, Mayne and Diehl (Model Predictive Control, 2nd ed.) it
    is detectable if and only if (Phi, C) is detectable and
    rank [[I - Phi, -Bd], [C, Cd]] = n + nd. On the CW model with the full Hill state measured,
    a constant along-track position bias fails the rank condition (rank 6 of 7): a constant y
    offset is itself an equilibrium, so it cannot be told from the state. Biases on x, z or a
    velocity channel, and constant accelerations on any axis, pass it; biases on all six
    channels give rank 11 of 12.

    Parameters
    ----------
    phi : array_like, shape (n, n)
        State transition matrix of one step.
    c : array_like, shape (p, n)
        Measurement matrix.
    bd : array_like, shape (n, nd)
        Effect of each disturbance on the next state (for a constant acceleration on the CW
        model, a column of Gamma).
    cd : array_like, shape (p, nd)
        Effect of each disturbance on the measurement (for a measurement bias, a unit column).

    Returns
    -------
    bool
        True if the augmented model is detectable.
    """
    phi = np.asarray(phi, dtype=float)
    c = np.asarray(c, dtype=float)
    bd = np.asarray(bd, dtype=float)
    cd = np.asarray(cd, dtype=float)
    n, nd = bd.shape
    if not _stabilizable(phi.T, c.T):
        return False
    lemma = np.block([[np.eye(n) - phi, -bd], [c, cd]])
    return _full_rank(lemma, n + nd)


@dataclass(frozen=True, kw_only=True)
class LQGConfig:
    """Declared sensor law, process noise and regulator weights of the LQG baseline.

    The sigmas and the process noise are declared configuration, set by whoever runs the agent
    from the public scenario parameters; the agent never reads them from the environment.

    Parameters
    ----------
    sigma_pos_m : float
        Declared 1-sigma white noise of each measured Hill position (m).
    sigma_vel_mps : float
        Declared 1-sigma white noise of each measured Hill velocity (m/s).
    process_sigma_mps2 : float
        1-sigma of a white acceleration held over each step on every Hill axis (m/s^2); it must
        be positive when the agent is reset.
    bias_states : tuple of str
        Measurement channels with a constant bias to estimate, from ("x", "z", "vx", "vy",
        "vz"); each adds one filter state.
    bias_walk : float
        Per-step 1-sigma random walk of each bias, as a fraction of its channel's sigma; it
        must be positive when the agent is reset with bias states.
    regulator : LQRConfig
        Weights of the LQR law applied to the estimate.

    Raises
    ------
    ValueError
        If a sigma is not positive, the process noise or the walk is negative, or a bias state
        is "y" (along-track position, not detectable on the CW model), unknown or repeated.
    """

    sigma_pos_m: float = 1.0
    sigma_vel_mps: float = 1e-3
    process_sigma_mps2: float = 1e-6
    bias_states: tuple[str, ...] = ()
    bias_walk: float = 1e-3
    regulator: LQRConfig = field(default_factory=LQRConfig)

    def __post_init__(self):
        if not self.sigma_pos_m > 0.0:
            raise ValueError("sigma_pos_m must be positive")
        if not self.sigma_vel_mps > 0.0:
            raise ValueError("sigma_vel_mps must be positive")
        if not self.process_sigma_mps2 >= 0.0:
            raise ValueError("process_sigma_mps2 must not be negative")
        if not self.bias_walk >= 0.0:
            raise ValueError("bias_walk must not be negative")
        object.__setattr__(self, "bias_states", tuple(self.bias_states))
        for name in self.bias_states:
            if name == "y":
                raise ValueError("along-track position bias is not detectable on the CW model")
            if name not in BIAS_STATES:
                raise ValueError(f"unknown bias state {name!r}, expected one of {BIAS_STATES}")
        if len(set(self.bias_states)) != len(self.bias_states):
            raise ValueError("bias states must not repeat")


class LQGAgent:
    """Steady-state Kalman filter on the CW model feeding the LQR hold-point law.

    The filter state is the Hill state (m, m/s) followed by the declared measurement biases. At
    each decision the agent updates its prediction with the measurement,
    xhat^+ = xhat^- + L (y - C xhat^-), applies the `LQRAgent` law to the state estimate, and
    predicts xhat^- = Phi xhat^+ + Gamma u, u the command that acts over the coming step: the
    oldest pending command when the observation carries the pending-commands channel, else the
    command just issued. The first prediction is the first measurement with zero biases. L and
    the covariance are the steady-state ones of `kalman_gain`. A baseline, never an oracle: the
    sensor law is declared in `LQGConfig`, not read from the environment.

    Parameters
    ----------
    config : LQGConfig, optional
        Declared sensor law, process noise and regulator weights; the defaults of `LQGConfig`
        when omitted.
    name : str
        Name in results.

    Attributes
    ----------
    estimate : numpy.ndarray, shape (6 + nd,)
        State and bias estimate after the last update (m, m/s); NaN before the first act. For
        tests only; nothing scores it.
    covariance : numpy.ndarray, shape (6 + nd, 6 + nd)
        Steady-state covariance after an update, by the Joseph form. For tests only.
    """

    def __init__(self, config: LQGConfig | None = None, name: str = "lqg"):
        self.config = LQGConfig() if config is None else config
        self.name = name

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Design the filter and the regulator for the view's orbit, step, thrust and delay.

        Parameters
        ----------
        view : AgentView
            Scenario description; uses `n`, `dt`, `max_accel`, `action_mode`, the observation
            names and what `LQRAgent.reset` reads.
        rng : numpy.random.Generator, optional
            Unused; the law is deterministic.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, if the process noise
            is zero, or the bias walk is zero with bias states (the steady-state gain on a state
            without process noise is zero), if the action mode takes no continuous command, or
            if the filter or regulator design fails (see `kalman_gain` and `LQRAgent.reset`).
        """
        # The regulator checks the goal too; checking here names this agent in the refusal.
        hold_point(view, "LQG agent")
        cfg = self.config
        if not cfg.process_sigma_mps2 > 0.0 or (cfg.bias_states and not cfg.bias_walk > 0.0):
            raise ValueError("process noise must be positive: with none the filter gain collapses")
        self.regulator = LQRAgent(cfg.regulator)
        self.regulator.reset(view, rng)
        self.accel_per_unit = view.max_accel / action_scale(view)
        self.slots = _pending_slots(view.observation)

        phi, gamma = cw_discrete(view.n, view.dt)
        nd = len(cfg.bias_states)
        sigma = np.array([cfg.sigma_pos_m] * 3 + [cfg.sigma_vel_mps] * 3)
        self.phi = np.eye(6 + nd)
        self.phi[:6, :6] = phi
        self.gamma = np.vstack([gamma, np.zeros((nd, 3))])
        self.c = np.hstack([np.eye(6), np.zeros((6, nd))])
        qd = np.zeros((6 + nd, 6 + nd))
        qd[:6, :6] = cfg.process_sigma_mps2**2 * gamma @ gamma.T
        for j, name in enumerate(cfg.bias_states):
            channel = STATE_NAMES.index(name)
            self.c[channel, 6 + j] = 1.0
            qd[6 + j, 6 + j] = (cfg.bias_walk * sigma[channel]) ** 2
        self.gain, _, self.covariance = kalman_gain(self.phi, self.c, qd, np.diag(sigma**2))
        self.estimate = np.full(6 + nd, np.nan)
        self.predicted: np.ndarray | None = None

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Update the estimate with the measurement and command toward the hold point.

        Parameters
        ----------
        obs : numpy.ndarray
            Observation in the order of the view's names: the measured Hill state (m, m/s)
            first, then the pending-command slots if enabled.
        info : dict
            Passed to the regulator, which does not read it.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Command in the view's action mode.

        Raises
        ------
        ValueError
            If a pending command's wait is not a whole number of steps.
        """
        obs = np.asarray(obs, dtype=float)
        y = obs[:6]
        if self.predicted is None:
            self.predicted = np.concatenate([y, np.zeros(len(self.phi) - 6)])
        self.estimate = self.predicted + self.gain @ (y - self.c @ self.predicted)
        filtered = obs.copy()
        filtered[:6] = self.estimate[:6]
        command = self.regulator.act(filtered, info)
        # The channel echoes each command in the action units the agent issued it in.
        if self.slots:
            in_effect = obs[self.slots[0][1:]] * self.accel_per_unit
        else:
            in_effect = command * self.accel_per_unit
        self.predicted = self.phi @ self.estimate + self.gamma @ in_effect
        return command
