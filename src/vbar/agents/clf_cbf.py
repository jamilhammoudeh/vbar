"""CLF-CBF QP controller: a relaxed Lyapunov row toward the hold point and a hard keep-out barrier.

Each decision solves min 1/2 |u - us|^2 + (w / 2) delta^2 over the command u and the relaxation
delta, subject to the control Lyapunov row dV/dt <= -decay V + delta, the exponential keep-out
barrier as a hard row, and the thrust limit (Ames, Coogan, Egerstedt, Notomista, Sreenath and
Tabuada, "Control barrier functions: theory and applications", ECC 2019, sections II.C and IV).
V = e'Pe on the deviation e from the hold point at rest, with P the continuous Riccati solution of
the regulator weights on the CW model. The relaxation is dualized, so the QP is a scalar root
search over exact projections onto the barrier row and the thrust limit (`vbar.solvers.project`).
Hill frame, SI units. The guarantees of both rows are for continuous time; the command is held
over each step, and the held step is certified and logged, not enforced.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike
from scipy.linalg import eigh, solve_continuous_are
from scipy.optimize import brentq

from ..constraints import KeepOut, segment_minimum
from ..dynamics import cw_discrete, cw_system
from ..solvers import ActuatorSet, Projection, Rows, project
from .base import AgentView, action_scale, hold_point
from .lqr import RICCATI_TOL, LQRConfig, regulator_model

# Keep-out samples over the held step for its certificate, as the environment checks its
# constraints.
VIOLATION_SAMPLES = 20
# The relaxation's multiplier is found to a few ulps. Under a box limit the dual residual is
# piecewise linear in it, so Brent's method ends after a few projections.
MULTIPLIER_RTOL = 4.0 * np.finfo(float).eps


@dataclass(frozen=True, kw_only=True)
class ClfCbfConfig:
    """Settings of the CLF-CBF QP controller.

    Parameters
    ----------
    regulator : LQRConfig
        Weights of the control Lyapunov function: its position, velocity and acceleration scales
        give the continuous weights; the rate weight must be 0 and the stage-cost convention is
        not used.
    poles : tuple of 2 float
        Rates p1, p2 (1/s) of the exponential barrier h'' + (p1 + p2) h' + p1 p2 h >= 0, whose
        characteristic roots are -p1 and -p2; finite and positive.
    buffer_m : float
        Distance (m) added to the keep-out radius for the barrier, finite and not negative.
    clf_weight : float
        Weight w of the relaxation in (w / 2) delta^2, in (m/s^2)^2 per (V/s)^2; finite and
        positive. Larger values hold the Lyapunov decay more stiffly.

    Raises
    ------
    ValueError
        If `poles` is not two finite positive rates, `buffer_m` is negative or not finite, or
        `clf_weight` is not finite and positive.
    """

    regulator: LQRConfig = field(default_factory=LQRConfig)
    poles: tuple[float, ...] = (0.02, 0.05)
    buffer_m: float = 1.0
    clf_weight: float = 1e-2

    def __post_init__(self):
        poles = tuple(float(p) for p in self.poles)
        if len(poles) != 2 or not all(math.isfinite(p) and p > 0.0 for p in poles):
            raise ValueError("poles must be two finite positive rates")
        if not (math.isfinite(self.buffer_m) and self.buffer_m >= 0.0):
            raise ValueError("buffer_m must be finite and not negative")
        if not (math.isfinite(self.clf_weight) and self.clf_weight > 0.0):
            raise ValueError("clf_weight must be finite and positive")
        object.__setattr__(self, "poles", poles)


@dataclass(frozen=True)
class ClfDesign:
    """The control Lyapunov function V = e'Pe, e = x - xs, of one view and regulator.

    The arrays are read-only.

    Attributes
    ----------
    a : numpy.ndarray, shape (6, 6)
        Continuous CW system matrix (1/s, 1/s^2), Hill frame.
    p : numpy.ndarray, shape (6, 6)
        Continuous Riccati solution of the regulator weights, symmetric positive definite, so V
        is in units of the weighted cost per second.
    decay : float
        Rate (1/s) the continuous LQR meets at every deviation: dV/dt <= -decay V.
    xs : numpy.ndarray, shape (6,)
        The hold point at rest (m, m/s).
    us : numpy.ndarray, shape (3,)
        The constant input that holds it (m/s^2).
    """

    a: np.ndarray
    p: np.ndarray
    decay: float
    xs: np.ndarray
    us: np.ndarray


def clf_design(view: AgentView, regulator: LQRConfig) -> ClfDesign:
    """The control Lyapunov function of the continuous LQR to the view's hold point.

    P solves A'P + PA - P B Rc^-1 B'P + Qc = 0 on the CW model with Qc = diag(pos_scale^-2 I,
    vel_scale^-2 I) and Rc = accel_scale^-2 I. Under the continuous LQR, dV/dt = -e'Me with
    M = Qc + P B Rc^-1 B'P, so the largest rate met at every deviation is the smallest
    eigenvalue of the pencil M e = lambda P e; lambda_min(M) / lambda_max(P) also holds but is
    looser.

    Parameters
    ----------
    view : AgentView
        Scenario description; uses `n`, `dt`, `max_accel`, `thrust_limit`, `action_mode`,
        `mass_kg`, `goal["hold_point"]` (Hill frame, m) and the observation names.
    regulator : LQRConfig
        Position, velocity and acceleration scales of the weights.

    Returns
    -------
    ClfDesign
        P, the decay rate, the hold point at rest and its hold input.

    Raises
    ------
    ValueError
        If the goal is not a hold or a berthing or has no hold point, the action mode takes no
        continuous command, the thrust layout is unknown, the hold point needs more than
        `max_accel` of steady thrust, the regulator has a rate weight, the observation carries
        pending commands, or the Riccati solve leaves a residual above 1e-10 of its terms.
    """
    hold_point(view, "CLF-CBF agent")
    if regulator.rate_weight > 0.0:
        raise ValueError("the CLF-CBF agent takes no rate weight")
    model = regulator_model(view, regulator)
    if model.slots:
        raise ValueError("the CLF-CBF agent does not model pending commands")
    accel_scale = (
        view.max_accel if regulator.accel_scale_mps2 is None else regulator.accel_scale_mps2
    )
    a, b = cw_system(view.n)
    # Solved in units of the weight scales, where Qc and Rc are identities: in SI the weights
    # span decades and the plain solve leaves residuals near 1e-9 for scales such as 1 km and
    # 1 mm/s. The decay rate is unchanged by the congruence.
    d = np.array([regulator.pos_scale_m] * 3 + [regulator.vel_scale_mps] * 3)
    a_s = a * d[None, :] / d[:, None]
    b_s = b * accel_scale / d[:, None]
    try:
        p_s = solve_continuous_are(a_s, b_s, np.eye(6), np.eye(3))
    except LinAlgError as e:
        raise ValueError("riccati solution did not converge") from e
    p_s = 0.5 * (p_s + p_s.T)
    pb = p_s @ b_s
    residual = a_s.T @ p_s + p_s @ a_s - pb @ pb.T + np.eye(6)
    if not np.max(np.abs(residual)) <= RICCATI_TOL * np.max(np.abs(p_s)):
        raise ValueError("riccati solution did not converge")
    decay = float(eigh(np.eye(6) + pb @ pb.T, p_s, eigvals_only=True).min())
    p = p_s / d[:, None] / d[None, :]
    xs = model.xs.copy()
    # The hold point at rest is an equilibrium of both models, so the step's hold input is the
    # continuous one.
    us = model.us.copy()
    for array in (a, p, xs, us):
        array.flags.writeable = False
    return ClfDesign(a=a, p=p, decay=decay, xs=xs, us=us)


def clf_row(design: ClfDesign, x: ArrayLike) -> tuple[np.ndarray, float]:
    """The control Lyapunov row g . u <= bound + delta at the state `x`.

    dV/dt = 2 e'PAe + g . (u - us) with g = 2 (Pe) over the velocity rows, so the row
    dV/dt <= -decay V + delta reads g . u <= g . us - decay V - 2 e'PAe + delta.

    Parameters
    ----------
    design : ClfDesign
        The control Lyapunov function.
    x : array_like, shape (6,)
        Hill state (m, m/s).

    Returns
    -------
    g : numpy.ndarray, shape (3,)
        Row normal, in units of V/s per m/s^2.
    bound : float
        Right-hand side (V/s); both are zero at the hold point at rest.
    """
    e = np.asarray(x, dtype=float)[:6] - design.xs
    pe = design.p @ e
    g = 2.0 * pe[3:]
    bound = float(g @ design.us - design.decay * (e @ pe) - 2.0 * pe @ (design.a @ e))
    return g, bound


@dataclass(frozen=True)
class ClfCbfSolution:
    """The result of `clf_cbf_qp`.

    Attributes
    ----------
    u : numpy.ndarray, shape (3,)
        The command, in the units of the rows and the actuator set.
    delta : float
        Relaxation of the Lyapunov row, >= 0, in the row's units. When the hard rows conflict,
        the row's violation at `u`.
    multiplier : float
        Multiplier of the Lyapunov row, w delta; 0 when the hard rows conflict.
    status : str
        Status of `projection`: "feasible", "infeasible_certified" or "solver_failed".
    projection : Projection
        The projection that gave `u`, with its active rows.
    """

    u: np.ndarray
    delta: float
    multiplier: float
    status: str
    projection: Projection


class _ProjectionFailed(Exception):
    def __init__(self, projection: Projection):
        self.projection = projection


def clf_cbf_qp(
    u_ref: ArrayLike,
    g: ArrayLike,
    bound: float,
    rows: Rows,
    actuator: ActuatorSet | None,
    clf_weight: float,
) -> ClfCbfSolution:
    """min 1/2 |u - u_ref|^2 + (w / 2) delta^2 s.t. g . u <= bound + delta, the rows, the limit.

    With a multiplier lambda >= 0 on the relaxed row, stationarity in delta gives delta =
    lambda / w, and the minimizer over u is the projection u(lambda) of u_ref - lambda g onto the
    rows and the limit. lambda solves lambda = w max(0, g . u(lambda) - bound); the right side
    does not increase with lambda (a projection is monotone), so the root is unique and lies in
    [0, w (g . u(0) - bound)], where it is found by Brent's method. When the rows conflict with
    the limit the command is the least-violation point of `project` from u_ref, and the
    relaxation is dropped.

    Parameters
    ----------
    u_ref : array_like, shape (3,)
        Reference command.
    g : array_like, shape (3,)
        Normal of the relaxed row, per command unit.
    bound : float
        Right-hand side of the relaxed row.
    rows : Rows
        Hard rows a_i . u >= b_i.
    actuator : ActuatorSet or None
        Box or ball limit on u; none when None.
    clf_weight : float
        Weight w of the relaxation, finite and positive.

    Returns
    -------
    ClfCbfSolution
        The command, the relaxation, its multiplier and the projection that gave the command.

    Raises
    ------
    ValueError
        If `g` is not three finite numbers, `bound` is not finite, `clf_weight` is not finite and
        positive, or `project` refuses `u_ref`.
    """
    g = np.asarray(g, dtype=float)
    bound = float(bound)
    if g.shape != (3,) or not np.all(np.isfinite(g)):
        raise ValueError("g must be three finite numbers")
    if not math.isfinite(bound):
        raise ValueError("bound must be finite")
    if not (math.isfinite(clf_weight) and clf_weight > 0.0):
        raise ValueError("clf_weight must be finite and positive")
    u_ref = np.asarray(u_ref, dtype=float)
    start = project(u_ref, rows, actuator)
    if start.status != "feasible":
        delta = max(0.0, float(g @ start.u) - bound)
        return ClfCbfSolution(start.u, delta, 0.0, start.status, start)
    excess = float(g @ start.u) - bound
    if excess <= 0.0:
        return ClfCbfSolution(start.u, 0.0, 0.0, "feasible", start)

    # Brent's method evaluates both ends again and returns a point it evaluated; each projection
    # is kept so none is repeated.
    projections = {0.0: start}

    def residual(lam: float) -> float:
        if lam not in projections:
            found = project(u_ref - lam * g, rows, actuator)
            if found.status != "feasible":
                raise _ProjectionFailed(found)
            projections[lam] = found
        return lam - clf_weight * max(0.0, float(g @ projections[lam].u) - bound)

    high = clf_weight * excess
    try:
        lam = (
            high
            if residual(high) <= 0.0
            else brentq(residual, 0.0, high, xtol=1e-300, rtol=MULTIPLIER_RTOL)
        )
        residual(lam)
    except _ProjectionFailed as failed:
        found = failed.projection
        return ClfCbfSolution(
            found.u, max(0.0, float(g @ found.u) - bound), 0.0, found.status, found
        )
    found = projections[lam]
    return ClfCbfSolution(found.u, lam / clf_weight, lam, found.status, found)


def _barrier_row(
    x: np.ndarray, radius: float, poles: tuple[float, ...], a: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """r_hat . u >= b for h = rho - radius with h'' + (p1 + p2) h' + p1 p2 h >= 0; None at rho 0.

    rho'' = (|v|^2 - rho'^2) / rho + r_hat . (a_cw(x) + u), with a_cw the CW acceleration of the
    state, so the barrier condition is affine in u.
    """
    r, v = x[:3], x[3:6]
    rho = float(np.linalg.norm(r))
    if rho == 0.0:
        return None
    r_hat = r / rho
    rho_dot = float(r_hat @ v)
    p1, p2 = poles
    b = (
        -(v @ v - rho_dot**2) / rho
        - r_hat @ (a @ x)[3:]
        - (p1 + p2) * rho_dot
        - p1 * p2 * (rho - radius)
    )
    return r_hat, float(b)


class ClfCbfAgent:
    """CLF-CBF QP controller to the hold point with the keep-out sphere as a hard barrier.

    Each decision solves `clf_cbf_qp` from the hold input us: the Lyapunov row of `clf_design`,
    relaxed with weight `clf_weight`, and the exponential barrier on the range to the target with
    radius the keep-out radius plus `buffer_m`, under the view's box or ball limit (Hill frame,
    m/s^2). A barrier set in conflict with the limit gives the least-violation command of
    `project`, and a failed projection gives the command it returned; both are logged by status.
    The command is held over the step, where neither row's continuous-time guarantee applies, so
    the keep-out margin of the held step on the CW model is certified (`segment_minimum` over 20
    samples) and logged, not enforced. Without a keep-out constraint there is no barrier row.

    `log` holds one entry per decision and `fallbacks` the count of decisions by level (0 a
    feasible QP, 1 the least-violation command of a conflicting set, 2 the command of a failed
    projection). Both are the agent's own bookkeeping, which the recorder and metrics never read
    as scores.

    Parameters
    ----------
    config : ClfCbfConfig, optional
        Weights, poles, buffer and relaxation weight; the defaults of `ClfCbfConfig` when
        omitted.
    name : str
        Name in results.
    """

    def __init__(self, config: ClfCbfConfig | None = None, name: str = "clf-cbf"):
        self.config = ClfCbfConfig() if config is None else config
        self.name = name
        self.log: list[dict] = []
        self.fallbacks = [0, 0, 0]

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Design the Lyapunov function and set the barrier for the view.

        Parameters
        ----------
        view : AgentView
            Scenario description; uses `n`, `dt`, `max_accel`, `thrust_limit`, `action_mode`,
            `mass_kg`, `goal["hold_point"]` (Hill frame, m), the keep-out constraints and the
            observation names.
        rng : numpy.random.Generator, optional
            Unused; the law is deterministic.

        Raises
        ------
        ValueError
            If the goal is not a hold or a berthing or has no hold point, the action mode takes
            no continuous command, the thrust layout is unknown, the hold point needs more than
            `max_accel` of steady thrust, the regulator has a rate weight, the observation
            carries pending commands, or the Riccati solve fails.
        """
        self.scale = action_scale(view)
        self.design = clf_design(view, self.config.regulator)
        self.view = view
        self.actuator = ActuatorSet(view.max_accel, view.thrust_limit)
        radii = [float(c["radius"]) for c in view.constraints if c["type"] == "keep_out"]
        # Every keep-out sphere is centred on the target, so the largest is the only one that
        # binds.
        self.keep_out = KeepOut(max(radii)) if radii else None
        self.taus = np.linspace(0.0, view.dt, VIOLATION_SAMPLES)
        self.samples = cw_discrete(view.n, self.taus)
        self.log = []
        self.fallbacks = [0, 0, 0]

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Command of the CLF-CBF QP at the observed state.

        Parameters
        ----------
        obs : numpy.ndarray
            Observation whose first six entries are the Hill state (m, m/s).
        info : dict
            Agent-safe information; only `time_s` (s since the episode start) is read.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Command in the view's action mode.
        """
        view, cfg, design = self.view, self.config, self.design
        t = float(info["time_s"])
        x = np.asarray(obs, dtype=float)[:6].copy()
        start = time.perf_counter()
        row = None
        if self.keep_out is not None:
            row = _barrier_row(x, self.keep_out.radius + cfg.buffer_m, cfg.poles, design.a)
        if row is None:
            rows = Rows(np.zeros((0, 3)), np.zeros(0), ())
        else:
            rows = Rows(row[0][None, :], np.array([row[1]]), ("keep_out",))
        g, bound = clf_row(design, x)
        solution = clf_cbf_qp(design.us, g, bound, rows, self.actuator, cfg.clf_weight)
        u = solution.u
        level = {"feasible": 0, "infeasible_certified": 1}.get(solution.status, 2)
        self.fallbacks[level] += 1
        held = None if self.keep_out is None else self._held_bound(self.keep_out, x, u)
        self.log.append(
            {
                "time_s": t,
                "status": solution.status,
                "level": level,
                "cbf_active": "keep_out" in solution.projection.active,
                "delta": solution.delta,
                "multiplier": solution.multiplier,
                "held_bound": held,
                "elapsed_s": time.perf_counter() - start,
            }
        )
        return u / view.max_accel * self.scale

    def _held_bound(self, keep_out: KeepOut, x: np.ndarray, u: np.ndarray) -> float:
        """Certified lower bound on the keep-out margin (m) over the step with `u` held, on CW."""
        n = self.view.n
        phis, gammas = self.samples
        states = phis @ x + gammas @ u

        def state_at(tau: float) -> np.ndarray:
            phi, gamma = cw_discrete(n, tau)
            return phi @ x + gamma @ u

        found = segment_minimum(keep_out, state_at, self.taus, states, float(np.linalg.norm(u)), n)
        return found.lower_bound
