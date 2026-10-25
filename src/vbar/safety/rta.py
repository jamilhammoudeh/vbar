"""The four run time assurance filters of Dunlap et al. for docking under a speed limit.

Dunlap, Hibbard, Mote and Hobbs, "Comparing Run Time Assurance Approaches for Safe Spacecraft
Docking", IEEE Control Systems Letters 6, 2022 (arXiv:2110.00447v1), compare explicit and
implicit switching and optimization filters on the Clohessy-Wiltshire equations integrated by
Euler steps of 1 s. The allowable set C_A is the distance-dependent speed limit
|v| <= nu0 + nu1 |r| and the per-axis limit |v_i| <= v_max (eq. 14 to 19). The implicit filters
check, or constrain, a 5 s rollout of a backup controller that tracks the nearest point of a grid
of closed natural motion trajectories (sec. III.D and III.E).

The implicit optimization rows use eq. 11 with its bracket corrected to
f(x) + g(x) u - f(x) - g(x) u_b(x), the time derivative of the constraint along the backup flow
(Chen, Jankovic, Santillo and Ames, "Backup Control Barrier Functions: Formulation and
Comparative Study", arXiv:2104.11332v1, sec. III.A); the printed bracket evaluates the closed
loop at the backup point instead. Values the paper does not give come from the authors' code and
are marked so in `AfrlSetup`. None of these filters counts toward certification: Lemma 1 does not
hold at the paper's nu1 = 4 n under a 1 N limit on 12 kg, the plant is an Euler model, and the
implicit filters look 5 s ahead.
"""

from __future__ import annotations

import functools
import math
import numbers
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

from ..agents.base import CONTINUOUS_MODES, AgentView, action_scale, command_fraction
from ..dynamics.cw import cw_system
from ..dynamics.models import EulerCWDynamics
from ..solvers.qp import MAX_ROWS, ActuatorSet, Projection, Rows, project
from . import FilterResult, InfeasibilityProof, run_filter
from .cbf import FilterDeclaration

SCENARIO_ID = "dunlap2021-euler"
# The registered scenarios that fly `AfrlSetup.paper()` and `AfrlSetup.authors_code()`.
_PAPER_ID = "vbar:docking-afrl-paper:euler1:box:dv:v1"
_CODE_ID = _PAPER_ID + ":code"
CONSTRAINT_NAMES = ("speed_limit", "vx_limit", "vy_limit", "vz_limit")
_N = 0.001027
_STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")
# Below this speed (m/s) the gradient of |v| is undefined and the speed-limit row is dropped.
_SPEED_FLOOR = 1e-9
# Changes below this fraction of a full command are rounding, not intervention.
_MOVED = 1e-6
# A view's parameters must equal the setup's to this relative difference.
_MATCH = 1e-12
# Lemma 3 asks c <= v_max / n; with b up to v_max / (2 n) that is c / b <= 2.
_C_OVER_B = 2.0
# Grid points within this fraction of the Lemma 3 bound on (c / b)^2 lie on it in exact
# arithmetic: theta1 = pi/4 at theta2 = 0 gives exactly 4, which sin and cos round to either
# side of it depending on the platform's math library.
_ON_BOUND = 1e-12
# Leak (1/s) on the integrators of the backup's augmented LQR design. The authors' augmented
# system is not stabilizable, since the velocity integral repeats the position, and the Riccati
# equation has no solution without it. An assumption: K_1 moves by under 2e-4 of its largest
# entry between leaks of 1e-6 and 1e-8 1/s.
_INTEGRATOR_LEAK = 1e-6


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def _class_k(gains, name: str) -> tuple[float, float]:
    a1, a3 = (float(g) for g in gains)
    if not (math.isfinite(a1) and math.isfinite(a3) and a1 >= 0.0 and a3 >= 0.0 and a1 + a3 > 0):
        raise ValueError(f"{name} must be two finite nonnegative gains, not both zero")
    return a1, a3


@dataclass(frozen=True, kw_only=True)
class AfrlSetup:
    """The docking problem of Dunlap et al. with an Euler plant.

    Parameters
    ----------
    nu1 : float
        Slope of the speed limit, 1/s, positive. The paper uses 4 n (Assumption 2); the authors'
        code uses 2.15 n.
    mass_kg : float
        Deputy mass, kg (Assumption 2).
    u_max : float
        Thrust limit per Hill axis, N (Assumption 2).
    n : float
        Mean motion of the chief, rad/s (Assumption 2).
    nu0 : float
        Speed allowed at the chief, m/s, nonnegative (Assumption 2).
    v_max : float
        Per-axis velocity limit, m/s (Assumption 2).
    r_max : float
        Largest range of Lemma 1, m (Assumption 2).
    dt : float
        Euler step and command hold, s (sec. IV).
    horizon_s : float
        Length of the implicit filters' backup rollouts, s, a whole number of steps (sec. IV).
    start_range, start_speed : float
        Range (m) and speed (m/s) of the paper's start (sec. IV); it gives no directions.
    alpha_speed : tuple of float
        (a1, a3) of the class-K function alpha(h) = a1 h + a3 h^3 of the speed limit, 1/s and
        s/m^2; from the authors' code.
    alpha_velocity : tuple of float
        (a1, a3) of each velocity limit, 1/s and s^3/m^4; from the authors' code.
    nmt_grid : tuple of int
        Grid sizes over theta1, theta2 and psi of the backup's trajectories; from the authors'
        code.
    backup_q, backup_r : float
        Backup LQR weights Q = q I12 on the state (m, m/s) augmented with the integral of the
        tracking error (m s, m), and R = r I3 on the force (N); from the authors' code.
    track_radius : float
        Position error below which the backup's desired point starts to move, m; from the
        authors' code.
    primary_q, primary_r : float
        LQR weights Q = q I6 on the state and R = r I3 on the force (N) of the paper's primary
        controller; from the authors' code.
    stop_radius, stop_time : float
        A run ends at |r| <= stop_radius (m) or at stop_time (s); from the authors' code.

    Raises
    ------
    ValueError
        If a value is not finite or out of range, the horizon is not a whole number of steps,
        or a class-K pair is negative or zero.
    """

    nu1: float
    mass_kg: float = 12.0
    u_max: float = 1.0
    n: float = _N
    nu0: float = 0.2
    v_max: float = 10.0
    r_max: float = 10_000.0
    dt: float = 1.0
    horizon_s: float = 5.0
    start_range: float = 9850.0
    start_speed: float = 0.866
    alpha_speed: tuple[float, float] = (0.05, 0.1)
    alpha_velocity: tuple[float, float] = (0.0005, 0.001)
    nmt_grid: tuple[int, int, int] = (20, 20, 20)
    backup_q: float = 1e-5
    backup_r: float = 1e7
    track_radius: float = 1.5
    primary_q: float = 0.05
    primary_r: float = 1000.0
    stop_radius: float = 1.0
    stop_time: float = 5000.0

    def __post_init__(self):
        for name in (
            "nu1",
            "mass_kg",
            "u_max",
            "n",
            "v_max",
            "r_max",
            "dt",
            "horizon_s",
            "start_range",
            "backup_q",
            "backup_r",
            "track_radius",
            "primary_q",
            "primary_r",
            "stop_radius",
            "stop_time",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("nu0", "start_speed"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "alpha_speed", _class_k(self.alpha_speed, "alpha_speed"))
        object.__setattr__(self, "alpha_velocity", _class_k(self.alpha_velocity, "alpha_velocity"))
        grid = tuple(self.nmt_grid)
        if len(grid) != 3 or not all(isinstance(k, int) and k > 0 for k in grid):
            raise ValueError("nmt_grid must be three positive integers")
        object.__setattr__(self, "nmt_grid", grid)
        steps = self.horizon_s / self.dt
        if abs(steps - round(steps)) > 1e-9 * steps:
            raise ValueError("horizon_s must be a whole number of steps dt")

    @classmethod
    def paper(cls) -> AfrlSetup:
        """The paper's values, nu1 = 4 n (Assumption 2)."""
        return cls(nu1=4.0 * _N)

    @classmethod
    def authors_code(cls) -> AfrlSetup:
        """The paper's values with the authors' code's slope nu1 = 2.15 n."""
        return cls(nu1=2.15 * _N)

    @property
    def a_max(self) -> float:
        """The acceleration limit per axis, u_max / m, m/s^2."""
        return self.u_max / self.mass_kg

    @property
    def horizon_steps(self) -> int:
        """Euler steps in a backup rollout."""
        return round(self.horizon_s / self.dt)

    @property
    def lemma1_accel(self) -> float:
        """Right side of Lemma 1, (3 n^2 + 2 n nu1 + nu1^2) R_max + (2 n + nu1) nu0, m/s^2.

        The paper compares it with u_max in N; it is an acceleration and needs u_max / m.
        """
        n, nu1 = self.n, self.nu1
        return (3.0 * n * n + 2.0 * n * nu1 + nu1 * nu1) * self.r_max + (2.0 * n + nu1) * self.nu0

    @property
    def code_lemma_accel(self) -> float:
        """The authors' code's check on nu1, m/s^2, to be at most u_max / m.

        v_max (nu1 sqrt 3 + 3 n^2 sqrt 3 / nu1 + 2 n) - 3 n^2 nu0 / nu1.
        """
        n, nu1, root3 = self.n, self.nu1, math.sqrt(3.0)
        return (
            self.v_max * (nu1 * root3 + 3.0 * n * n * root3 / nu1 + 2.0 * n)
            - 3.0 * n * n * self.nu0 / nu1
        )

    def view(self) -> AgentView:
        """The public description the filters reset from.

        Returns
        -------
        AgentView
            Scenario "vbar:docking-afrl-paper:euler1:box:dv:v1" for `paper()`, the same with
            ":code" for `authors_code()`, and "dunlap2021-euler" (no dynamics part) for any other
            setup; commands in m/s^2 under a box of u_max / m, one "speed_limit" constraint with
            v0 = nu0 and v1 = nu1, and stop_time / dt steps.
        """
        if self == AfrlSetup.paper():
            scenario_id = _PAPER_ID
        elif self == AfrlSetup.authors_code():
            scenario_id = _CODE_ID
        else:
            scenario_id = SCENARIO_ID
        return AgentView(
            scenario_id=scenario_id,
            dt=self.dt,
            max_steps=math.ceil(self.stop_time / self.dt - 1e-9),
            n=self.n,
            action_mode="accel",
            max_accel=self.a_max,
            thrust_limit="box",
            mass_kg=self.mass_kg,
            constraints=({"type": "speed_limit", "v0": self.nu0, "v1": self.nu1},),
        )


class AfrlConstraints:
    """The allowable set: phi1 = nu0 + nu1 |r| - |v| and phi2..4 = v_max^2 - v_i^2.

    Dunlap et al., eq. 16 and 19, Hill frame (m, m/s); phi1 is in m/s and the others in
    m^2/s^2. `alpha` is the class-K function a1 h + a3 h^3 of each, with the setup's gains.

    Parameters
    ----------
    setup : AfrlSetup
        The speed limit, velocity limit and class-K gains.
    """

    names = CONSTRAINT_NAMES

    def __init__(self, setup: AfrlSetup) -> None:
        self.nu0, self.nu1, self.v_max = setup.nu0, setup.nu1, setup.v_max
        self._a1 = np.array([setup.alpha_speed[0], *[setup.alpha_velocity[0]] * 3])
        self._a3 = np.array([setup.alpha_speed[1], *[setup.alpha_velocity[1]] * 3])

    def values(self, x) -> np.ndarray:
        """phi1..phi4 at Hill-frame states `x` of shape (..., 6); returns (..., 4)."""
        x = np.asarray(x, dtype=float)
        rho = np.linalg.norm(x[..., :3], axis=-1)
        speed = np.linalg.norm(x[..., 3:6], axis=-1)
        phi1 = self.nu0 + self.nu1 * rho - speed
        return np.concatenate([phi1[..., None], self.v_max**2 - x[..., 3:6] ** 2], axis=-1)

    def contains(self, x) -> np.ndarray:
        """Whether states `x` of shape (..., 6) meet all four constraints; shape (...)."""
        return np.all(self.values(x) >= 0.0, axis=-1)

    def gradients(self, x) -> np.ndarray:
        """Gradients of phi1..phi4 at one state, shape (4, 6).

        At r = 0 the position part of grad phi1 is set to zero and at v = 0 its velocity part;
        the barrier rows drop phi1 below a speed of 1e-9 m/s.
        """
        x = np.asarray(x, dtype=float)
        r, v = x[:3], x[3:6]
        rho, speed = float(np.linalg.norm(r)), float(np.linalg.norm(v))
        grad = np.zeros((4, 6))
        if rho > 0.0:
            grad[0, :3] = self.nu1 * r / rho
        if speed > 0.0:
            grad[0, 3:] = -v / speed
        grad[[1, 2, 3], [3, 4, 5]] = -2.0 * v
        return grad

    def alpha(self, h) -> np.ndarray:
        """alpha(h) = a1 h + a3 h^3 per constraint for `h` of shape (..., 4)."""
        h = np.asarray(h, dtype=float)
        return h * (self._a1 + self._a3 * h * h)


@functools.lru_cache(maxsize=8)
def _unit_orbits(n: float, nu1: float, grid: tuple[int, int, int]) -> np.ndarray:
    """Grid points of sec. III.D with b = 1 that meet Lemma 3."""
    n1, n2, n3 = grid
    theta1 = np.linspace(0.0, 0.5 * np.pi, n1 + 1)[1:]
    theta2 = np.linspace(0.0, 0.5 * np.pi, n2 + 1)[:-1]
    psi = 2.0 * np.pi * np.arange(n3) / n3
    t1, t2, ps = (a.ravel() for a in np.meshgrid(theta1, theta2, psi, indexing="ij"))
    c2 = (np.tan(t2) ** 2 + 4.0 * np.cos(t1) ** 2) / np.sin(t1) ** 2
    bound = min((nu1 / n) ** 2 - 4.0, _C_OVER_B**2)
    keep = c2 <= bound * (1.0 + _ON_BOUND)
    # A point kept on the bound is moved onto it, so every kept point meets Lemma 3.
    t1, t2, ps, c = t1[keep], t2[keep], ps[keep], np.sqrt(np.minimum(c2[keep], bound))
    # The paper's sign of the phase; the arctangent of two arguments gives pi/2 at theta2 = 0.
    nu = np.arctan2(2.0 * np.cos(t1), np.tan(t2)) - ps
    points = np.stack(
        [
            np.sin(nu),
            2.0 * np.cos(nu),
            c * np.sin(ps),
            n * np.cos(nu),
            -2.0 * n * np.sin(nu),
            n * c * np.cos(ps),
        ],
        axis=-1,
    )
    points.flags.writeable = False
    return points


def _lqr_gain(setup: AfrlSetup, q: float, r: float) -> np.ndarray:
    """LQR gain of Q = q I6 and R = r I3 on the force (N), in m/s^2 per state unit."""
    a, b = cw_system(setup.n)
    b_force = b / setup.mass_kg
    p = solve_continuous_are(a, b_force, q * np.eye(6), r * np.eye(3))
    return b_force.T @ p / r / setup.mass_kg


def _backup_gain(setup: AfrlSetup) -> np.ndarray:
    """K_1 of the LQR on the state augmented with the integral of the tracking error.

    The augmented system is x' = A x + B_F u, z' = x - leak z, with Q = q I12 and R = r I3 on the
    force (N), as the authors design their backup; K_1, the block on x, is returned in m/s^2 per
    state unit and the integral block is dropped.
    """
    a, b = cw_system(setup.n)
    a_aug = np.block([[a, np.zeros((6, 6))], [np.eye(6), -_INTEGRATOR_LEAK * np.eye(6)]])
    b_aug = np.vstack([b / setup.mass_kg, np.zeros((6, 3))])
    p = solve_continuous_are(a_aug, b_aug, setup.backup_q * np.eye(12), setup.backup_r * np.eye(3))
    return (b_aug.T @ p / setup.backup_r)[:, :6] / setup.mass_kg


@dataclass(frozen=True)
class BackupRollout:
    """An Euler rollout of `NmtBackup` from one state.

    Attributes
    ----------
    states : ndarray, shape (k + 1, 6)
        phi_0 = x, ..., phi_k, Hill frame (m, m/s).
    targets : ndarray, shape (k + 1, 6)
        The desired state at each point.
    commands : ndarray, shape (k, 3)
        The backup's acceleration at phi_0 .. phi_{k-1}, m/s^2, on the box.
    sensitivities : ndarray, shape (k + 1, 6, 6)
        S_j = d phi_j / d x, the Jacobian of the rollout with respect to its start.
    target_jacobians : ndarray, shape (k + 1, 6, 6)
        T_j, the Jacobian of the target at point j with respect to the start.
    """

    states: np.ndarray
    targets: np.ndarray
    commands: np.ndarray
    sensitivities: np.ndarray
    target_jacobians: np.ndarray


class NmtBackup:
    """LQR tracking of the nearest point on a grid of closed natural motion trajectories.

    Parameters
    ----------
    setup : AfrlSetup
        Dynamics, limits, grid, LQR weights and track radius.

    Raises
    ------
    ValueError
        If no grid point meets Lemma 3, for example with nu1 <= 2 n.

    Notes
    -----
    Dunlap et al., sec. III.D and III.E. A grid point (theta1, theta2, psi) of sec. III.D is
    kept when (tan^2 theta2 + 4 cos^2 theta1) / sin^2 theta1 <= min((nu1 / n)^2 - 4, 4), which is
    Lemma 3 for every b up to v_max / (2 n); it gives nu1 |r| >= |v| at every phase of the
    orbit, not only at the point its proof evaluates. The grid takes theta1 at the upper ends
    and theta2 at the lower ends of equal intervals of [0, pi/2], and psi at equal steps of
    [0, 2 pi).

    At a state x the backup takes b = min(sqrt(x^2 + (y/2)^2), v_max / (2 n)) and targets the
    kept point b p nearest the position. It commands u_b = clip(K (x_des - x)), with K the block
    K_1 on the state of the authors' LQR on the state augmented with the integral of the tracking
    error (a leak of 1e-6 1/s on the integrators makes it solvable), divided by the mass (m/s^2
    per m and per m/s). The integral term itself is left out. Along a rollout the
    target is held, and moves by Euler free motion once the position comes within the track
    radius. The target is chosen again at every rollout start, so the backup has no memory
    between decisions. `points` holds the kept unit orbits (b = 1), Hill frame.
    """

    def __init__(self, setup: AfrlSetup) -> None:
        self.n, self.dt, self.a_max = setup.n, setup.dt, setup.a_max
        self.track_radius = setup.track_radius
        self.b_cap = setup.v_max / (2.0 * setup.n)
        self.points = _unit_orbits(setup.n, setup.nu1, setup.nmt_grid)
        if len(self.points) == 0:
            raise ValueError("no grid point meets Lemma 3 at this nu1")
        self._a, self._b = cw_system(setup.n)
        self.gain = _backup_gain(setup)
        self._drift_step = np.eye(6) + self.dt * self._a

    def target(self, x) -> tuple[np.ndarray, np.ndarray]:
        """The desired state chosen at `x` and its Jacobian with respect to `x`.

        Returns
        -------
        ndarray, shape (6,)
            b p for the kept unit orbit point p nearest the position, m and m/s.
        ndarray, shape (6, 6)
            d(b p) / dx: p times the gradient of b, zero where b is capped.
        """
        x = np.asarray(x, dtype=float)
        b_uncapped = math.hypot(x[0], 0.5 * x[1])
        b = min(b_uncapped, self.b_cap)
        i = int(np.argmin(np.sum((b * self.points[:, :3] - x[:3]) ** 2, axis=1)))
        jac = np.zeros((6, 6))
        if 0.0 < b_uncapped < self.b_cap:
            jac[:, 0] = self.points[i] * (x[0] / b_uncapped)
            jac[:, 1] = self.points[i] * (0.25 * x[1] / b_uncapped)
        return b * self.points[i], jac

    def command(self, x) -> np.ndarray:
        """u_b(x), the backup's acceleration at `x`, m/s^2 per Hill axis, on the box."""
        x = np.asarray(x, dtype=float)
        desired, _ = self.target(x)
        return np.clip(self.gain @ (desired - x), -self.a_max, self.a_max)

    def rollout(self, x, steps: int) -> BackupRollout:
        """Euler steps of the closed loop from `x` with their sensitivities.

        S_{j+1} = S_j + dt (A S_j + B D_j K (T_j - S_j)), S_0 = I, with D_j zero on the
        saturated axes and T_j the target's Jacobian, which follows the target's motion.

        Parameters
        ----------
        x : array_like, shape (6,)
            Start state, Hill frame (m, m/s).
        steps : int
            Euler steps, nonnegative.

        Returns
        -------
        BackupRollout
            The states, targets, commands, sensitivities and target Jacobians.
        """
        s = np.array(x, dtype=float)
        desired, jac = self.target(s)
        sens = np.eye(6)
        states, targets, sensitivities, jacobians = [s], [desired], [sens], [jac]
        commands = []
        moving = False
        for _ in range(steps):
            moving = moving or float(np.linalg.norm(s[:3] - desired[:3])) < self.track_radius
            raw = self.gain @ (desired - s)
            u = np.clip(raw, -self.a_max, self.a_max)
            feedback = self.gain * (np.abs(raw) < self.a_max)[:, None]
            sens = sens + self.dt * (self._a @ sens + self._b @ (feedback @ (jac - sens)))
            s = s + self.dt * (self._a @ s + self._b @ u)
            if moving:
                desired, jac = self._drift_step @ desired, self._drift_step @ jac
            states.append(s)
            targets.append(desired)
            sensitivities.append(sens)
            jacobians.append(jac)
            commands.append(u)
        return BackupRollout(
            np.array(states),
            np.array(targets),
            np.reshape(commands, (-1, 3)),
            np.array(sensitivities),
            np.array(jacobians),
        )


def _matches(given: float, own: float) -> bool:
    return math.isclose(float(given), own, rel_tol=_MATCH, abs_tol=0.0)


def _check_view(view: AgentView, setup: AfrlSetup) -> None:
    """Refuse a view the filters do not model."""
    if view.action_mode not in CONTINUOUS_MODES:
        raise ValueError(f"filters need a continuous action mode, not {view.action_mode!r}")
    if view.thrust_limit != "box":
        raise ValueError(f"the docking filters need a box thrust limit, not {view.thrust_limit!r}")
    if view.frame != "hill":
        raise ValueError(f"filters need the hill frame, not {view.frame!r}")
    if tuple(view.observation[:6]) != _STATE_NAMES:
        raise ValueError("the observation must start with x, y, z, vx, vy, vz")
    if any(name.startswith("pending_") for name in view.observation):
        raise ValueError("pending commands are not supported: the filter cannot predict delay")
    parts = view.scenario_id.split(":")
    dynamics = parts[2] if len(parts) > 2 else None
    if dynamics not in (None, "cw", "euler1"):
        raise ValueError(f"dynamics {dynamics!r} is neither cw nor euler1")
    # The euler1 plant takes 1 s steps; the filters predict with Euler steps of the setup's dt.
    if dynamics == "euler1" and not _matches(setup.dt, EulerCWDynamics.step_s):
        raise ValueError(
            f"the euler1 plant steps {EulerCWDynamics.step_s!r} s, not the setup's dt {setup.dt!r}"
        )
    limits = []
    for spec in view.constraints:
        if spec["type"] != "speed_limit":
            raise ValueError(f"cannot enforce constraint {spec['type']!r}")
        limits.append(spec)
    if len(limits) != 1:
        raise ValueError("the scenario must declare one speed_limit")
    for name, given, own in (
        ("n", view.n, setup.n),
        ("dt", view.dt, setup.dt),
        ("max_accel", view.max_accel, setup.a_max),
        ("speed limit v0", limits[0]["v0"], setup.nu0),
        ("speed limit v1", limits[0]["v1"], setup.nu1),
    ):
        if not _matches(given, own):
            raise ValueError(f"the view's {name} {given!r} differs from the setup's {own!r}")


def _state(obs) -> np.ndarray:
    x = np.array(obs, dtype=float)[:6]
    if x.shape != (6,) or not np.all(np.isfinite(x)):
        raise ValueError("the observation must start with six finite state entries")
    return x


def _binding(rows: Rows, a_max: float) -> Rows:
    """The rows some command in the box can break; the others hold for every command."""
    keep = rows.b > -a_max * np.sum(np.abs(rows.a), axis=1)
    return Rows(rows.a[keep], rows.b[keep], tuple(np.array(rows.names, dtype=object)[keep]))


def _violation(rows: Rows, u: np.ndarray) -> float:
    if len(rows.b) == 0:
        return 0.0
    return max(0.0, float(np.max(rows.b - rows.a @ u)))


def _proof(found: Projection, rows: Rows, a_max: float) -> InfeasibilityProof | None:
    """The Farkas certificate of `project` on the rows it names, or on all rows if needed."""
    if found.certificate is None:
        return None
    lam = np.asarray(found.certificate, dtype=float)[: len(rows.b)]
    named = np.array([name in found.conflicting for name in rows.names], dtype=bool)
    for mask in (named, np.ones(len(rows.b), dtype=bool)):
        if not mask.any():
            continue
        proof = InfeasibilityProof(
            normals=rows.a[mask],
            bounds=rows.b[mask],
            multipliers=lam[mask],
            layout="box",
            a_max=a_max,
            names=tuple(np.array(rows.names, dtype=object)[mask]),
        )
        if proof.verify():
            return proof
    return None


class _RtaFilter:
    """Reset checks, command conversions and results shared by the four filters."""

    name: str
    _backup_text: str | None = None

    def __init__(self, *, setup: AfrlSetup) -> None:
        if not isinstance(setup, AfrlSetup):
            raise TypeError("setup must be an AfrlSetup")
        self.setup = setup
        self.constraints = AfrlConstraints(setup)
        self._a, self._b = cw_system(setup.n)
        self._actuator = ActuatorSet(setup.a_max, "box")
        self._view: AgentView | None = None
        self._declaration: FilterDeclaration | None = None

    @property
    def declaration(self) -> FilterDeclaration:
        """The filter's declaration for the view of the last `reset`.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        """
        if self._declaration is None:
            raise RuntimeError("reset the filter first")
        return self._declaration

    def reset(self, view: AgentView) -> None:
        """Check the view against the setup and declare the filter.

        Parameters
        ----------
        view : AgentView
            A continuous action mode, a box thrust limit, the Hill frame, no pending commands,
            no dynamics part or "cw", one "speed_limit" and no other constraint, and n, dt,
            max_accel and the speed limit equal to the setup's.

        Raises
        ------
        ValueError
            If the view breaks any of those conditions.
        """
        self._view = None
        self._declaration = None
        _check_view(view, self.setup)
        self._scale = action_scale(view)
        self._view = view
        self._declaration = FilterDeclaration(
            name=self.name,
            safe_set=self._safe_set_text(),
            barriers=(self.constraints,),
            poles=None,
            gamma=None,
            design_dt=self.setup.dt,
            layout="box",
            a_max=self.setup.a_max,
            buffer_m=0.0,
            tightening={"navigation": None, "execution": 0.0, "model_error": None},
            backup=self._backup_text,
            certificate=None,
            counts_toward_certification=False,
        )

    def _safe_set_text(self) -> str:
        return "C_A: |v| <= nu0 + nu1 |r| and |v_i| <= v_max on every Hill axis"

    def contains(self, x) -> bool:
        """Return whether the Hill-frame state `x` (m, m/s) lies in the declared safe set."""
        return bool(self.constraints.contains(np.asarray(x, dtype=float)))

    def _started(self) -> AgentView:
        if self._view is None:
            raise RuntimeError("reset the filter first")
        return self._view

    def _clipped(self, action) -> np.ndarray:
        """The command as acceleration, m/s^2, clipped onto the box."""
        view = self._started()
        u = command_fraction(view, action) * view.max_accel
        if u.shape != (3,) or not np.all(np.isfinite(u)):
            raise ValueError("the action must be three finite numbers")
        return np.clip(u, -self.setup.a_max, self.setup.a_max)

    def _euler(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x + self.setup.dt * (self._a @ x + self._b @ u)

    def _result(
        self,
        u: np.ndarray,
        u_c: np.ndarray,
        t0: float,
        *,
        recovering: bool,
        active: bool,
        fallback_level: int,
        status: str = "feasible",
        slack: float = 0.0,
        conflicting: tuple[str, ...] = (),
        proof: InfeasibilityProof | None = None,
        certified: bool = False,
    ) -> FilterResult:
        a_max = self.setup.a_max
        u = np.clip(u, -a_max, a_max)
        return FilterResult(
            u / a_max * self._scale,
            intervened=float(np.linalg.norm(u - u_c)) > _MOVED * a_max,
            status=status,
            slack=slack,
            conflicting=tuple(conflicting),
            compute_s=time.perf_counter() - t0,
            active=active,
            recovering=recovering,
            fallback_level=fallback_level,
            certificate="finite-horizon" if certified else None,
            certificate_horizon=self.setup.horizon_s if certified else None,
            proof=proof,
        )

    def _project(
        self,
        u_c: np.ndarray,
        rows: Rows,
        t0: float,
        *,
        recovering: bool,
        backup: np.ndarray | None = None,
        certify: Callable[[np.ndarray], bool] | None = None,
    ) -> FilterResult:
        """The projection of the clipped command onto the rows under the box, as a result."""
        kept = _binding(rows, self.setup.a_max)
        if len(kept.b) == 0:
            found = None
        else:
            found = project(u_c, kept, self._actuator)
        if found is None or found.status == "feasible":
            active = found is not None and bool(np.any(found.multipliers > 0.0))
            # With no active row the projection of the clipped command is the command itself.
            u = found.u if found is not None and active else u_c
            certified = certify is not None and certify(u)
            return self._result(
                u,
                u_c,
                t0,
                recovering=recovering,
                active=active,
                fallback_level=0,
                certified=certified,
            )
        if found.status == "infeasible_certified":
            return self._result(
                found.u,
                u_c,
                t0,
                recovering=recovering,
                active=True,
                fallback_level=0,
                status="infeasible_certified",
                slack=float(found.phase1_margin),
                conflicting=found.conflicting,
                proof=_proof(found, kept, self.setup.a_max),
            )
        u = found.u if backup is None else backup
        return self._result(
            u,
            u_c,
            t0,
            recovering=recovering,
            active=True,
            fallback_level=0 if backup is None else 2,
            status="solver_failed",
            slack=_violation(kept, u),
        )


class ExplicitSwitchingFilter(_RtaFilter):
    """Switching filter on the allowable set with a one-step boundary backup.

    Parameters
    ----------
    setup : AfrlSetup
        The problem; reset requires a view that matches it.

    Raises
    ------
    TypeError
        If `setup` is not an `AfrlSetup`.

    Notes
    -----
    Dunlap et al., eq. 4 with C_S = C_A, and sec. III.E. The clipped command passes when the
    Euler prediction x1 = x + (A x + B u) dt lies in C_A. Otherwise the backup sets each violated
    constraint to zero at x1 by inverting the Euler step at x: a velocity component above v_max
    is put on +-v_max, then, if the speed limit is still violated, the predicted velocity is
    scaled along itself to nu0 + nu1 |r1|, so the speed limit has the last word. The paper does
    not say how the speed is reduced. The backup is clipped onto the box, where it may no longer
    reach the boundary. `recovering` is True outside C_A; `certificate` is None.
    """

    name = "rta-explicit-switching"
    _backup_text = "one-step command onto the boundary of each violated constraint"

    def filter(self, obs: np.ndarray, action: np.ndarray) -> FilterResult:
        """Return the command, or the boundary backup when the next state leaves C_A.

        Parameters
        ----------
        obs : np.ndarray
            The observation; its first six entries are the Hill-frame state, m and m/s.
        action : np.ndarray
            The agent's command in the view's action mode.

        Returns
        -------
        FilterResult
            `fallback_level` 0 for the command and 2 for the backup, `active` when the backup
            is used.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        ValueError
            If the observation or the action is malformed.
        """
        t0 = time.perf_counter()
        self._started()
        x = _state(obs)
        u_c = self._clipped(action)
        recovering = not self.contains(x)
        x1 = self._euler(x, u_c)
        if self.constraints.contains(x1):
            return self._result(u_c, u_c, t0, recovering=recovering, active=False, fallback_level=0)
        setup = self.setup
        v1 = x1[3:].copy()
        over = np.abs(v1) > setup.v_max
        v1[over] = np.sign(v1[over]) * setup.v_max
        cap = setup.nu0 + setup.nu1 * float(np.linalg.norm(x1[:3]))
        speed = float(np.linalg.norm(v1))
        if speed > cap:
            v1 *= cap / speed
        u_b = u_c + (v1 - x1[3:]) / setup.dt
        return self._result(u_b, u_c, t0, recovering=recovering, active=True, fallback_level=2)


class ExplicitOptimizationFilter(_RtaFilter):
    """Projection onto the barrier rows of the allowable set.

    Parameters
    ----------
    setup : AfrlSetup
        The problem and the class-K gains; reset requires a view that matches it.

    Raises
    ------
    TypeError
        If `setup` is not an `AfrlSetup`.

    Notes
    -----
    Dunlap et al., eq. 7 and 10: the clipped command is projected with `vbar.solvers.project`
    onto grad phi_i(x) (A x + B u) + alpha(phi_i(x)) >= 0 for the four constraints under the box.
    Rows the box meets for every command (b_i <= -a_max |a_i|_1) are dropped first, which does
    not change the projection. The speed-limit row is dropped below a speed of 1e-9 m/s.
    Infeasible rows give the least-violation command with an `InfeasibilityProof`. `recovering`
    is True outside C_A; `certificate` is None.
    """

    name = "rta-explicit-optimization"

    def _safe_set_text(self) -> str:
        return super()._safe_set_text() + _alpha_text(self.setup)

    def rows(self, x) -> Rows:
        """The barrier rows a_i . u >= b_i at the Hill-frame state `x`, u in m/s^2.

        Returns
        -------
        Rows
            Named by `AfrlConstraints.names`, in their units per second.
        """
        x = _state(x)
        grad = self.constraints.gradients(x)
        alpha = self.constraints.alpha(self.constraints.values(x))
        drift = self._a @ x
        keep = [i for i in range(4) if i > 0 or np.linalg.norm(x[3:]) >= _SPEED_FLOOR]
        a = grad[keep] @ self._b
        b = -(grad[keep] @ drift + alpha[keep])
        return Rows(a, b, tuple(CONSTRAINT_NAMES[i] for i in keep))

    def filter(self, obs: np.ndarray, action: np.ndarray) -> FilterResult:
        """Return the command nearest the agent's that meets the barrier rows in the box.

        Parameters
        ----------
        obs : np.ndarray
            The observation; its first six entries are the Hill-frame state, m and m/s.
        action : np.ndarray
            The agent's command in the view's action mode.

        Returns
        -------
        FilterResult
            "feasible", or "infeasible_certified" with the least-violation command and its
            proof, or "solver_failed" with `project`'s command.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        ValueError
            If the observation or the action is malformed.
        """
        t0 = time.perf_counter()
        self._started()
        x = _state(obs)
        u_c = self._clipped(action)
        return self._project(u_c, self.rows(x), t0, recovering=not self.contains(x))


def _alpha_text(setup: AfrlSetup) -> str:
    (s1, s3), (v1, v3) = setup.alpha_speed, setup.alpha_velocity
    return (
        f"; class-K alpha(h) = {s1:g} h + {s3:g} h^3 on the speed limit and "
        f"{v1:g} h + {v3:g} h^3 on each velocity limit"
    )


class _ImplicitFilter(_RtaFilter):
    """The backup and the implicit safe set shared by the implicit filters."""

    _backup_text = "LQR tracking of the nearest closed natural motion trajectory"

    def __init__(self, *, setup: AfrlSetup) -> None:
        super().__init__(setup=setup)
        self.backup = NmtBackup(setup)

    def _safe_set_text(self) -> str:
        return (
            f"states whose {self.setup.horizon_s:g} s Euler rollout under the backup stays in "
            "C_A at every step, C_A: |v| <= nu0 + nu1 |r| and |v_i| <= v_max"
        )

    def contains(self, x) -> bool:
        """Return whether the backup rollout from `x` stays in C_A at every Euler step."""
        roll = self.backup.rollout(_state(x), self.setup.horizon_steps)
        return bool(self.constraints.contains(roll.states).all())

    def _horizon_holds(self, x: np.ndarray, u: np.ndarray) -> bool:
        """One step under `u`, then the backup's rollout, all in C_A at the samples."""
        roll = self.backup.rollout(self._euler(x, u), self.setup.horizon_steps)
        return bool(self.constraints.contains(roll.states).all())


class ImplicitSwitchingFilter(_ImplicitFilter):
    """Switching filter that checks one step under the command and the backup's rollout.

    Parameters
    ----------
    setup : AfrlSetup
        The problem and the backup; reset requires a view that matches it.

    Raises
    ------
    TypeError
        If `setup` is not an `AfrlSetup`.

    Notes
    -----
    Dunlap et al., eq. 4 with the implicit safe set of eq. 6, sampled: the clipped command passes
    when its Euler step x1 and every point of the backup's rollout from x1 over the horizon lie
    in C_A, and is replaced by u_b(x) otherwise. All horizon points are checked, not only x1. A
    decision whose applied command passes that check carries the certificate "finite-horizon"
    over the horizon (s): a claim on the Euler samples only. `recovering` is True when the
    backup's rollout from x leaves C_A.
    """

    name = "rta-implicit-switching"

    def filter(self, obs: np.ndarray, action: np.ndarray) -> FilterResult:
        """Return the command, or the backup's command when the horizon leaves C_A.

        Parameters
        ----------
        obs : np.ndarray
            The observation; its first six entries are the Hill-frame state, m and m/s.
        action : np.ndarray
            The agent's command in the view's action mode.

        Returns
        -------
        FilterResult
            `fallback_level` 0 for the command and 2 for the backup, `active` when the backup
            is used.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        ValueError
            If the observation or the action is malformed.
        """
        t0 = time.perf_counter()
        self._started()
        x = _state(obs)
        u_c = self._clipped(action)
        recovering = not self.contains(x)
        if self._horizon_holds(x, u_c):
            return self._result(
                u_c, u_c, t0, recovering=recovering, active=False, fallback_level=0, certified=True
            )
        u_b = self.backup.command(x)
        return self._result(
            u_b,
            u_c,
            t0,
            recovering=recovering,
            active=True,
            fallback_level=2,
            certified=self._horizon_holds(x, u_b),
        )


class ImplicitOptimizationFilter(_ImplicitFilter):
    """Projection onto barrier rows on the backup's rollout, eq. 11 with the corrected bracket.

    Parameters
    ----------
    setup : AfrlSetup
        The problem and the backup; reset requires a view that matches it.

    Raises
    ------
    TypeError
        If `setup` is not an `AfrlSetup`.
    ValueError
        If the horizon gives more than 32 rows (more than 7 steps).

    Notes
    -----
    The rows are grad phi_i(phi_j) S_j B (u - u_b(x)) + alpha(phi_i(phi_j)) >= 0 for every
    constraint i and every rollout point phi_j, j = 0 .. k, with S_j from
    `NmtBackup.rollout`. At u = u_b(x) every row reduces to alpha(phi_i(phi_j)) >= 0, so the
    projection is feasible whenever the rollout from x stays in C_A. There is no row on the
    rollout's last point beyond C_A and no margin between samples, as in the paper. Rows the box
    meets for every command are dropped first. A feasible decision whose command passes the
    check of `ImplicitSwitchingFilter` carries the certificate "finite-horizon". Infeasible rows
    give the least-violation command with an `InfeasibilityProof`; a failed projection gives
    u_b(x).
    """

    name = "rta-implicit-optimization"

    def __init__(self, *, setup: AfrlSetup) -> None:
        if 4 * (setup.horizon_steps + 1) > MAX_ROWS:
            raise ValueError(f"a horizon of {setup.horizon_steps} steps gives more than 32 rows")
        super().__init__(setup=setup)

    def _safe_set_text(self) -> str:
        return super()._safe_set_text() + _alpha_text(self.setup)

    def rows(self, x) -> Rows:
        """The barrier rows a . u >= b on the backup's rollout from the Hill-frame state `x`.

        Returns
        -------
        Rows
            Named "<constraint>@<j>" for rollout point j, u in m/s^2.
        """
        return self._rollout_rows(_state(x))[0]

    def _rollout_rows(self, x: np.ndarray) -> tuple[Rows, np.ndarray, bool]:
        """The rows, u_b(x) and whether the rollout from `x` stays in C_A."""
        roll = self.backup.rollout(x, self.setup.horizon_steps)
        u_b = roll.commands[0] if len(roll.commands) else self.backup.command(x)
        values = self.constraints.values(roll.states)
        normals, sides, names = [], [], []
        for j, (s, sens) in enumerate(zip(roll.states, roll.sensitivities, strict=True)):
            w = self.constraints.gradients(s) @ sens @ self._b
            alpha = self.constraints.alpha(values[j])
            for i, name in enumerate(CONSTRAINT_NAMES):
                if i == 0 and np.linalg.norm(s[3:]) < _SPEED_FLOOR:
                    continue
                normals.append(w[i])
                sides.append(float(w[i] @ u_b) - alpha[i])
                names.append(f"{name}@{j}")
        rows = Rows(np.reshape(normals, (-1, 3)), np.array(sides), tuple(names))
        return rows, u_b, bool(np.all(values >= 0.0))

    def filter(self, obs: np.ndarray, action: np.ndarray) -> FilterResult:
        """Return the command nearest the agent's that meets the rollout rows in the box.

        Parameters
        ----------
        obs : np.ndarray
            The observation; its first six entries are the Hill-frame state, m and m/s.
        action : np.ndarray
            The agent's command in the view's action mode.

        Returns
        -------
        FilterResult
            "feasible", or "infeasible_certified" with the least-violation command and its
            proof, or "solver_failed" with u_b(x).

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        ValueError
            If the observation or the action is malformed.
        """
        t0 = time.perf_counter()
        self._started()
        x = _state(obs)
        u_c = self._clipped(action)
        rows, u_b, inside = self._rollout_rows(x)
        return self._project(
            u_c,
            rows,
            t0,
            recovering=not inside,
            backup=u_b,
            certify=lambda u: self._horizon_holds(x, u),
        )


def start_state(setup: AfrlSetup, seed: int) -> np.ndarray:
    """A start at the paper's range and speed in seeded directions.

    The paper gives |r| = 9.85 km and |v| = 0.866 m/s but no directions (sec. IV). Here the
    position is uniform on the sphere of radius `start_range` and the velocity, of size
    `start_speed`, points in an independent uniform direction.

    Parameters
    ----------
    setup : AfrlSetup
        Supplies the range (m) and speed (m/s).
    seed : int
        Seed of the draw, in [0, 2**128).

    Returns
    -------
    ndarray, shape (6,)
        The Hill-frame start state, m and m/s.

    Raises
    ------
    ValueError
        If `seed` is not an integer in [0, 2**128).
    """
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed < 2**128:
        raise ValueError("seed must be an integer in [0, 2**128)")
    rng = np.random.default_rng(int(seed))
    r, v = rng.normal(size=3), rng.normal(size=3)
    return np.concatenate(
        [setup.start_range * r / np.linalg.norm(r), setup.start_speed * v / np.linalg.norm(v)]
    )


class _PaperLqr:
    """The paper's primary controller (sec. IV), not a V-BAR baseline.

    Unconstrained LQR to the origin with the authors' code's weights, clipped to the box.
    """

    def __init__(self, setup: AfrlSetup) -> None:
        self.gain = _lqr_gain(setup, setup.primary_q, setup.primary_r)
        self.a_max = setup.a_max

    def command(self, x: np.ndarray) -> np.ndarray:
        return np.clip(-self.gain @ x, -self.a_max, self.a_max)


@dataclass(frozen=True)
class Reproduction:
    """One run of `reproduce`.

    Attributes
    ----------
    states : ndarray, shape (k + 1, 6)
        The Euler plant's states, Hill frame (m, m/s), start included.
    primary : ndarray, shape (k, 3)
        The paper's controller's commands, m/s^2.
    applied : ndarray, shape (k, 3)
        The filtered commands the plant received, m/s^2.
    backup : ndarray, shape (k,)
        Whether each decision used the filter's backup (`fallback_level` 2).
    results : tuple of FilterResult
        The filter's decisions, with `compute_s` measured by `run_filter`.
    filter_s : float
        The filter's total wall time over the run, s.
    """

    states: np.ndarray
    primary: np.ndarray
    applied: np.ndarray
    backup: np.ndarray
    results: tuple[FilterResult, ...]
    filter_s: float


def reproduce(safety_filter, setup: AfrlSetup, x0) -> Reproduction:
    """Fly the paper's docking run with a filter around the paper's controller.

    The plant is `EulerCWDynamics` with step dt, x + (A x + B u) dt on the Clohessy-Wiltshire
    equations, with perfect state knowledge. Each step the paper's LQR primary (sec. IV; weights
    and clipping from the authors' code, not a V-BAR baseline) gives a command, `run_filter`
    passes it to the filter and times it, and the returned command, clipped to the box, is
    applied. The run stops once |r| <= `stop_radius` or after `stop_time`.

    Parameters
    ----------
    safety_filter : SafetyFilter
        Reset here with `setup.view()`; `vbar.safety.PassThroughFilter` flies the primary alone.
    setup : AfrlSetup
        The problem.
    x0 : array_like, shape (6,)
        The start, Hill frame (m, m/s).

    Returns
    -------
    Reproduction
        States, commands, backup flags, results and the filter's wall time.

    Raises
    ------
    ValueError
        If `x0` is not six finite numbers, or the filter refuses the view.
    """
    if np.shape(x0) != (6,):
        raise ValueError("x0 must be six finite numbers")
    x = _state(x0)
    view = setup.view()
    safety_filter.reset(view)
    primary = _PaperLqr(setup)
    plant = EulerCWDynamics(setup.n, step_s=setup.dt)
    plant.reset(x)
    states, commands, applied, backup, results = [x], [], [], [], []
    filter_s = 0.0
    for _ in range(view.max_steps):
        if float(np.linalg.norm(x[:3])) <= setup.stop_radius:
            break
        u_des = primary.command(x)
        result, elapsed = run_filter(safety_filter, x, u_des)
        u = np.clip(np.asarray(result.action, dtype=float), -setup.a_max, setup.a_max)
        x, _ = plant.advance(u, setup.dt)
        filter_s += elapsed
        states.append(x)
        commands.append(u_des)
        applied.append(u)
        backup.append(result.fallback_level == 2)
        results.append(result)
    return Reproduction(
        np.array(states),
        np.reshape(commands, (-1, 3)),
        np.reshape(applied, (-1, 3)),
        np.array(backup, dtype=bool),
        tuple(results),
        filter_s,
    )


__all__ = [
    "CONSTRAINT_NAMES",
    "SCENARIO_ID",
    "AfrlConstraints",
    "AfrlSetup",
    "BackupRollout",
    "ExplicitOptimizationFilter",
    "ExplicitSwitchingFilter",
    "ImplicitOptimizationFilter",
    "ImplicitSwitchingFilter",
    "NmtBackup",
    "Reproduction",
    "reproduce",
    "start_state",
]
