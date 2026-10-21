"""Two-impulse Clohessy-Wiltshire targeting in the Hill frame.

The transfer from r0 to rf in a time tf solves the position rows of the CW transition matrix,
rf = phi_rr r0 + phi_rv v0, for the departure velocity v0 (Fehse, Automated Rendezvous and
Docking of Spacecraft, section 3.3.2, eqs 3.41 to 3.43). The matrix phi_rv is singular in plane
at n tf = 2 k pi and at the other roots of 8 cos x + 3 x sin x = 8, and across the track at
n tf = k pi (Alfriend, Vadali, Gurfil, How and Breger, Spacecraft Formation Flying, p. 92).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..dynamics import cw_discrete
from ..metrics.efficiency import FuelBounds
from .base import AgentView, action_scale
from .planning import impulse_action, predicted_margin

SINGULAR_SET = "n tf = 2 k pi, the roots of 8 cos x + 3 x sin x = 8, or k pi across the track"


def _finite(values: ArrayLike, size: int, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.shape != (size,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be {size} finite numbers")
    return out


def conditioning(n: float, tf: float) -> tuple[float, float]:
    """Sensitivity measures of CW targeting that vanish on its singular set.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    tf : float
        Transfer time (s).

    Returns
    -------
    g : float
        In-plane measure (8 - 8 cos x - 3 x sin x) / x^2 with x = n tf, the in-plane
        determinant of phi_rv scaled to 1 as x goes to 0. Zero at x = 2 k pi and at the other
        roots of 8 cos x + 3 x sin x = 8 (2.81346 pi, 4.89060 pi, ...); it lies in
        [-0.283, 1.628], with its peak near x = 2.96, and is not a normalized condition number.
    h : float
        Cross-track measure sin(x) / x, in [-0.218, 1], zero at x = k pi.

    Raises
    ------
    ValueError
        If `n` or `tf` is not a positive finite number.
    """
    if not (math.isfinite(n) and n > 0.0):
        raise ValueError("n must be a positive finite number")
    if not (math.isfinite(tf) and tf > 0.0):
        raise ValueError("tf must be a positive finite number")
    x = n * tf
    # 8 (1 - cos x) = 16 sin^2(x / 2) and the quotients are formed before squaring, so neither
    # the cancellation of 1 - cos x nor the underflow of x^2 reaches g at short transfers.
    half = math.sin(0.5 * x) / x
    sinc = math.sin(x) / x
    return 16.0 * half * half - 3.0 * sinc, sinc


def cw_lambert(
    n: float, r0: ArrayLike, rf: ArrayLike, tf: float, *, tol: float = 1e-2
) -> tuple[np.ndarray, np.ndarray]:
    """Velocities that carry a coasting chaser from `r0` to `rf` in `tf` under CW motion.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    r0, rf : array_like, shape (3,)
        Departure and arrival positions in the Hill frame (m).
    tf : float
        Transfer time (s).
    tol : float
        Transfers where either `conditioning` measure is below `tol` in magnitude are refused.

    Returns
    -------
    v0 : numpy.ndarray, shape (3,)
        Required departure velocity in the Hill frame (m/s).
    vf : numpy.ndarray, shape (3,)
        Arrival velocity in the Hill frame (m/s).

    Raises
    ------
    ValueError
        If `n` or `tf` is not a positive finite number, `tf` is within `tol` of the singular
        set, or `r0` or `rf` is not three finite numbers.
    """
    g, h = conditioning(n, tf)
    if abs(g) < tol or abs(h) < tol:
        raise ValueError(f"transfer time is near the singular set ({SINGULAR_SET})")
    r0, rf = _finite(r0, 3, "r0"), _finite(rf, 3, "rf")
    # The ZOH closed form keeps 1 - cos(n tf) free of cancellation, which the textbook matrix
    # loses at short transfers.
    phi, _ = cw_discrete(n, tf)
    phi_rr, phi_rv = phi[:3, :3], phi[:3, 3:]
    phi_vr, phi_vv = phi[3:, :3], phi[3:, 3:]
    v0 = np.linalg.solve(phi_rv, rf - phi_rr @ r0)
    return v0, phi_vr @ r0 + phi_vv @ v0


def two_impulse(
    n: float, state: ArrayLike, target: ArrayLike, tf: float, *, tol: float = 1e-2
) -> tuple[np.ndarray, np.ndarray]:
    """Two impulses that bring a chaser to rest at `target` after a coast of `tf`.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    state : array_like, shape (6,)
        Chaser state [x, y, z, vx, vy, vz] in the Hill frame (m, m/s).
    target : array_like, shape (3,)
        Arrival position in the Hill frame (m).
    tf : float
        Transfer time (s).
    tol : float
        Singular-set tolerance passed to `cw_lambert`.

    Returns
    -------
    dv1 : numpy.ndarray, shape (3,)
        Departure impulse in the Hill frame (m/s).
    dv2 : numpy.ndarray, shape (3,)
        Arrival impulse in the Hill frame (m/s), the negative of the arrival velocity.

    Raises
    ------
    ValueError
        If `n` or `tf` is not a positive finite number, `tf` is within `tol` of the singular
        set, or `state` or `target` is not six or three finite numbers.
    """
    state = _finite(state, 6, "state")
    v0, vf = cw_lambert(n, state[:3], target, tf, tol=tol)
    return v0 - state[3:6], -vf


MODES = ("retarget", "open_loop", "track")


def _delivered(view: AgentView, action: np.ndarray) -> np.ndarray:
    """Impulse (Hill, m/s) that one step of `action` delivers at the initial mass."""
    step_dv = view.max_accel * view.dt
    if view.action_mode == "impulse_coast":
        return action[:3] * step_dv
    return action / action_scale(view) * step_dv


def _burn_steps(view: AgentView, dv: np.ndarray) -> list[np.ndarray]:
    """Per-step impulses (Hill, m/s) in which `impulse_action` delivers `dv`."""
    steps = []
    left = np.array(dv, dtype=float)
    while True:
        action, saturated = impulse_action(view, left, 0.0)
        if not saturated:
            steps.append(left)
            return steps
        step = _delivered(view, action)
        steps.append(step)
        left = left - step


class TwoImpulseAgent:
    """Two-impulse CW transfer to rest at the hold point, then station keeping there.

    The transfer takes `n_tf` / n seconds from the first observation. "retarget" solves
    `two_impulse` again at every decision for the time left and fires its first impulse, then,
    fewer than `t_min_steps` steps before the arrival, cancels the measured velocity until that
    burn is delivered. "open_loop" plans both impulses at the first decision and flies them at
    their steps, coasting in between. "track", for the continuous action modes only, flies the
    open-loop plan and adds the feedback -wn^2 (r - r_nom) - 2 zeta wn (v - v_nom) toward the
    nominal state propagated from the first observation through the plan (Fehse, Automated
    Rendezvous and Docking of Spacecraft, section 4.4.1). After the arrival every mode holds
    the hold point with its hover acceleration (-3 n^2 x, 0, n^2 z), "track" adding the same
    feedback toward it. Impulses are executed by `planning.impulse_action`, so one larger than
    a step can deliver is spread over several steps.

    `log` holds one dict per decision with the time "t" (s), the "phase" and whether the
    command "saturated". The decision that ends the departure burn of "open_loop" or "track"
    commits the coast to the arrival burn and adds "coast_s" (s), the keep-out margin predicted
    over that coast, "predicted_margin", and its certified lower bound,
    "predicted_margin_bound" (m, from `planning.predicted_margin`). A predicted violation is
    logged and the plan flown as is.

    Parameters
    ----------
    n_tf : float
        Transfer angle n tf (rad). At the default 1 rad both `conditioning` measures exceed
        0.8 and, from rest on the V-bar, the first command points toward the goal on every
        axis.
    mode : str
        "retarget", "open_loop" or "track".
    t_min_steps : int
        Steps before the arrival below which "retarget" stops solving and brakes, since the
        first impulse grows as the error over the shrinking time left.
    singular_tol : float
        A transfer angle whose `conditioning` measures fall below this in magnitude is refused
        at reset; in "retarget" a decision whose time left falls there coasts instead.
    wn : float
        Natural frequency of the tracking feedback (rad/s).
    zeta : float
        Damping ratio of the tracking feedback.
    name : str
        Name used in results.

    Raises
    ------
    ValueError
        If `mode` is unknown, `n_tf`, `singular_tol`, `wn` or `zeta` is not positive and finite,
        or `t_min_steps` is not a positive integer.
    """

    def __init__(
        self,
        *,
        n_tf: float = 1.0,
        mode: str = "retarget",
        t_min_steps: int = 2,
        singular_tol: float = 1e-2,
        wn: float = 0.01,
        zeta: float = 1.0,
        name: str = "two-impulse",
    ):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        positive = {"n_tf": n_tf, "singular_tol": singular_tol, "wn": wn, "zeta": zeta}
        for label, value in positive.items():
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{label} must be positive and finite")
        if isinstance(t_min_steps, bool) or not isinstance(t_min_steps, int) or t_min_steps < 1:
            raise ValueError("t_min_steps must be a positive integer")
        self.n_tf = n_tf
        self.mode = mode
        self.t_min_steps = t_min_steps
        self.singular_tol = singular_tol
        self.kp = wn**2
        self.kd = 2.0 * zeta * wn
        self.name = name
        self.log: list[dict[str, Any]] = []

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Start an episode with the scenario's public description.

        Raises
        ------
        ValueError
            If the action mode is "disc27" or unknown, "track" is asked for in
            "impulse_coast", thrust actions come without `mass_kg`, or `n_tf` lies within
            `singular_tol` of the singular set.
        """
        if view.action_mode == "disc27":
            raise ValueError("disc27 actions cannot deliver a planned impulse")
        if view.action_mode != "impulse_coast":
            action_scale(view)
        elif self.mode == "track":
            raise ValueError("track mode needs a continuous action mode")
        g, h = conditioning(view.n, self.n_tf / view.n)
        if abs(g) < self.singular_tol or abs(h) < self.singular_tol:
            raise ValueError(f"n_tf = {self.n_tf} is near the singular set ({SINGULAR_SET})")
        n = view.n
        self.view = view
        self.hold = np.array(view.goal["hold_point"], dtype=float)
        self.hover = np.array([-3.0 * n**2 * self.hold[0], 0.0, n**2 * self.hold[2]])
        self.phi, self.gamma = cw_discrete(n, view.dt)
        self.log = []
        self.t0: float | None = None
        self.braking = False
        self.arrived = False

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """The action for this decision in the view's action mode; appends one `log` entry."""
        view = self.view
        t = float(info["time_s"])
        state = np.asarray(obs[:6], dtype=float)
        if self.t0 is None:
            self.t0 = t
            self._plan(state, t)
        entry: dict[str, Any] = {"t": t}
        if self.mode == "retarget":
            dv, coast_s = self._retarget(state, t, entry), 0.0
        else:
            dv, coast_s = self._follow(state, t, round((t - self.t0) / view.dt), entry)
        action, saturated = impulse_action(
            view, dv, coast_s if view.action_mode == "impulse_coast" else 0.0
        )
        entry["saturated"] = saturated
        if entry["phase"] == "arrival" and not saturated:
            self.arrived = True
        if coast_s > 0.0:
            after = self.phi @ state + self.gamma @ (_delivered(view, action) / view.dt)
            margin, bound = predicted_margin(view, after, coast_s)
            entry.update(coast_s=coast_s, predicted_margin=margin, predicted_margin_bound=bound)
        self.log.append(entry)
        return action

    def _plan(self, state: np.ndarray, t: float) -> None:
        """Arrival time, and for "open_loop" and "track" the per-step plan and its nominal."""
        view = self.view
        tf = self.n_tf / view.n
        self.t_arrival = t + tf
        if self.mode == "retarget":
            return
        dv1, dv2 = two_impulse(view.n, state, self.hold, tf, tol=self.singular_tol)
        departure = _burn_steps(view, dv1)
        self.departure_steps = len(departure)
        self.coast_steps = max(round(tf / view.dt) - len(departure), 0)
        self.schedule = [*departure, *[np.zeros(3)] * self.coast_steps, *_burn_steps(view, dv2)]
        # The nominal as executed: each step's impulse held as a constant acceleration.
        nominal = [state]
        for dv in self.schedule:
            nominal.append(self.phi @ nominal[-1] + self.gamma @ (dv / view.dt))
        self.nominal = np.array(nominal)

    def _feedback(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Tracking impulse over one step toward `reference` (Hill, m and m/s)."""
        accel = -self.kp * (state[:3] - reference[:3]) - self.kd * (state[3:] - reference[3:])
        return accel * self.view.dt

    def _hold(self, state: np.ndarray) -> np.ndarray:
        dv = self.hover * self.view.dt
        if self.mode == "track":
            dv = dv + self._feedback(state, np.r_[self.hold, 0.0, 0.0, 0.0])
        return dv

    def _retarget(self, state: np.ndarray, t: float, entry: dict[str, Any]) -> np.ndarray:
        view = self.view
        if self.arrived:
            entry["phase"] = "hold"
            return self._hold(state)
        time_left = self.t_arrival - t
        if not self.braking and round(time_left / view.dt) >= self.t_min_steps:
            g, h = conditioning(view.n, time_left)
            if abs(g) < self.singular_tol or abs(h) < self.singular_tol:
                # The chaser is already on a transfer; coasting crosses the singular time.
                entry["phase"] = "singular"
                return np.zeros(3)
            entry["phase"] = "transfer"
            dv1, _ = two_impulse(view.n, state, self.hold, time_left, tol=self.singular_tol)
            return dv1
        # A saturated arrival burn carries on, re-measured, until it is delivered.
        self.braking = True
        entry["phase"] = "arrival"
        return -state[3:6]

    def _follow(
        self, state: np.ndarray, t: float, k: int, entry: dict[str, Any]
    ) -> tuple[np.ndarray, float]:
        """Step `k` of the plan, and the coast it commits (s)."""
        view = self.view
        if k >= len(self.schedule):
            entry["phase"] = "hold"
            return self._hold(state), 0.0
        dv = self.schedule[k]
        if self.mode == "track":
            dv = dv + self._feedback(state, self.nominal[k])
        if k < self.departure_steps:
            entry["phase"] = "departure"
        elif k < self.departure_steps + self.coast_steps:
            entry["phase"] = "coast"
        else:
            entry["phase"] = "arrival"
        if k != self.departure_steps - 1:
            return dv, 0.0
        # The coast the environment flies: whole steps, cut at the end of the episode.
        steps_left = max(view.max_steps - round(t / view.dt) - 1, 0)
        return dv, min(self.coast_steps, steps_left) * view.dt


class TwoImpulseBound:
    """Fuel bounds from the cheapest two-impulse CW transfer to rest at the hold point.

    The upper bound is the least |dv1| + |dv2| of `two_impulse` over a grid of transfer
    angles n tf, in the 1-norm for box thrusters and the 2-norm for a ball. Transfer times
    within 1e-2 of the singular set (`conditioning`) and longer than the episode,
    (max_steps - 1) dt, are skipped. Nothing better than the trivial lower bound 0 is
    certified, so the bound gives the upper optimality ratio only. The rung is "cw": the upper
    bound is a feasible cost for impulsive burns under exact CW motion, and finite burns or
    other dynamics move the cost either way.

    Parameters
    ----------
    n_tf_grid : sequence of float or None
        Transfer angles n tf (rad) to search; None takes `grid_points` angles evenly spaced
        from 0.05 rad to 5 rad or the episode length, whichever is shorter.
    grid_points : int
        Size of the default grid, at least 2.

    Raises
    ------
    ValueError
        If `grid_points` is below 2 or `n_tf_grid` holds a value that is not positive and
        finite.
    """

    def __init__(self, *, n_tf_grid: ArrayLike | None = None, grid_points: int = 400):
        if grid_points < 2:
            raise ValueError("grid_points must be at least 2")
        if n_tf_grid is not None:
            n_tf_grid = np.asarray(n_tf_grid, dtype=float).ravel()
            if not np.all(np.isfinite(n_tf_grid) & (n_tf_grid > 0.0)):
                raise ValueError("n_tf_grid must hold positive finite angles")
        self.n_tf_grid = n_tf_grid
        self.grid_points = grid_points

    def bounds(self, view: AgentView, start: np.ndarray) -> FuelBounds:
        """Bounds on the least delta-v (m/s) from `start` to rest at the view's hold point.

        Parameters
        ----------
        view : AgentView
            Public scenario description; its mean motion `n` (rad/s), `dt` (s), `max_steps`,
            thrust layout and hold point are used.
        start : array_like, shape (6,)
            Start state [x, y, z, vx, vy, vz] in the Hill frame (m, m/s).

        Returns
        -------
        FuelBounds
            Lower bound 0, the least two-impulse cost as the upper bound, norm "l1" (box) or
            "l2" (ball), rung "cw".

        Raises
        ------
        ValueError
            If the thrust layout is unknown, `start` is not six finite numbers, or no grid
            angle fits the episode away from the singular set.
        """
        norms = {"box": ("l1", 1), "ball": ("l2", 2)}
        if view.thrust_limit not in norms:
            raise ValueError(f"unknown thrust limit {view.thrust_limit!r}")
        norm, order = norms[view.thrust_limit]
        start = _finite(start, 6, "start")
        hold = np.array(view.goal["hold_point"], dtype=float)
        longest = view.n * (view.max_steps - 1) * view.dt
        if self.n_tf_grid is None:
            grid = np.linspace(0.05, min(5.0, longest), self.grid_points)
        else:
            grid = self.n_tf_grid
        costs = []
        for n_tf in grid:
            g, h = conditioning(view.n, n_tf / view.n)
            if n_tf > longest or abs(g) < 1e-2 or abs(h) < 1e-2:
                continue
            dv1, dv2 = two_impulse(view.n, start, hold, n_tf / view.n)
            costs.append(np.linalg.norm(dv1, order) + np.linalg.norm(dv2, order))
        if not costs:
            raise ValueError("no transfer time on the grid fits the episode")
        return FuelBounds(lower=0.0, upper=float(min(costs)), norm=norm, rung="cw")
