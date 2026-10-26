"""Piecewise-affine MPC: the constrained hold-point regulator solved offline over a state box.

The condensed regulator min 1/2 U'HU + (F x)'U subject to G U <= w + S x is a multiparametric
quadratic program in the state x. On each set of active rows its minimizer is affine in x, and the
states where that set is optimal form a polyhedral critical region (Bemporad, Morari, Dua and
Pistikopoulos, "The explicit linear quadratic regulator for constrained systems", Automatica
38(1), 2002; Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed., sections 7.2 to 7.4).
The regions are enumerated once, and each decision looks the state up and applies its law.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve, qr
from scipy.optimize import linprog

from .base import AgentView, action_scale, hold_point
from .lqr import LQRConfig, regulator_model

# A region is kept when its largest inscribed ball, in coordinates where the parameter box is
# [-1, 1] on every axis, has a radius above this. Active sets that are optimal only on a lower-
# dimensional set (two rows meeting at a point) have radius zero up to the LP's rounding.
CHEBYSHEV_TOL = 1e-9
# A state belongs to a region when every row holds to this fraction of the row's magnitudes, so
# a state on a shared facet is found despite the rounding of the rows (about 1e-16 of them). The
# laws are continuous, so the region found on a facet gives the same command as its neighbour.
LOOKUP_TOL = 1e-12
# Hill indices of the decoupled CW subsystems: in-plane (x, y, vx, vy) driven by (ax, ay), and
# cross-track (z, vz) driven by az.
SUBSYSTEMS = (((0, 1, 3, 4), (0, 1)), ((2, 5), (2,)))


@dataclass(frozen=True)
class Region:
    """One critical region of a multiparametric QP and its affine law.

    The arrays are read-only, so regions can be shared between controllers.

    Parameters
    ----------
    active : tuple of int
        Indices of the rows of G that hold with equality, ascending.
    gain : numpy.ndarray, shape (nu, nx)
        Gain of the minimizer, U = gain x + offset, in the units of U per unit of x.
    offset : numpy.ndarray, shape (nu,)
        Constant term of the minimizer, in the units of U.
    a : numpy.ndarray, shape (m, nx)
        Rows of the region a x <= b: the inactive rows of G, the multipliers of the active rows,
        and the parameter box, in that order.
    b : numpy.ndarray, shape (m,)
        Right-hand side of the region.
    """

    active: tuple[int, ...]
    gain: np.ndarray
    offset: np.ndarray
    a: np.ndarray
    b: np.ndarray


def mpqp_regions(
    h: ArrayLike,
    f: ArrayLike,
    g: ArrayLike,
    w: ArrayLike,
    s: ArrayLike,
    bounds: Sequence[tuple[float, float]],
) -> tuple[Region, ...]:
    """Critical regions of min 1/2 U'HU + (F x)'U subject to G U <= w + S x over a box of x.

    Active sets are explored breadth first from the empty set, and a set is extended by one row
    only when its own region is not empty (Bemporad et al., 2002). For each set the law comes
    from the KKT conditions with the active rows solved for as many inputs as there are rows and
    the cost minimized over the remaining free inputs, so an input fixed at a bound takes the
    bound exactly. Sets whose rows are linearly dependent are skipped (no unique multipliers),
    and a region is kept when its Chebyshev ball, in coordinates scaled to the box, has a radius
    above 1e-9.

    Parameters
    ----------
    h : array_like, shape (nu, nu)
        Hessian of the cost in U, symmetric positive definite.
    f : array_like, shape (nu, nx)
        Linear term of the cost per unit of the parameter x.
    g : array_like, shape (m, nu)
        Constraint rows on U.
    w : array_like, shape (m,)
        Constant right-hand side of the constraints.
    s : array_like, shape (m, nx)
        Parameter term of the right-hand side.
    bounds : sequence of (float, float)
        Lower and upper bound of each entry of x; the box over which regions are built.

    Returns
    -------
    tuple of Region
        The regions in the order found: by number of active rows, then by discovery.

    Raises
    ------
    ValueError
        If the shapes disagree, an input is not finite, a bound is not an interval of positive
        width, or H is not positive definite.
    RuntimeError
        If a Chebyshev LP ends without an optimum or a proof of infeasibility.
    """
    h = np.asarray(h, dtype=float)
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)
    w = np.asarray(w, dtype=float)
    s = np.asarray(s, dtype=float)
    box = np.asarray(bounds, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")
    nu = h.shape[0]
    if f.ndim != 2 or f.shape[0] != nu:
        raise ValueError("f must have one row per input")
    nx = f.shape[1]
    if g.ndim != 2 or g.shape[1] != nu:
        raise ValueError("g must have one column per input")
    m = g.shape[0]
    if w.shape != (m,) or s.shape != (m, nx):
        raise ValueError("w and s must have one row per row of g")
    if box.shape != (nx, 2):
        raise ValueError("bounds must hold one (low, high) pair per parameter")
    if not all(np.all(np.isfinite(arr)) for arr in (h, f, g, w, s, box)):
        raise ValueError("problem data must be finite")
    if not np.all(box[:, 1] > box[:, 0]):
        raise ValueError("each bound must have high above low")
    try:
        np.linalg.cholesky(h)
    except np.linalg.LinAlgError as e:
        raise ValueError("h must be positive definite") from e

    regions = []
    frontier: list[tuple[int, ...]] = [()]
    seen: set[tuple[int, ...]] = {()}
    while frontier:
        children = []
        for active in frontier:
            region = _region(h, f, g, w, s, box, active)
            if region is None:
                continue
            regions.append(region)
            # More active rows than inputs cannot be linearly independent.
            if len(active) == nu:
                continue
            for i in range(m):
                child = tuple(sorted((*active, i)))
                if i not in active and child not in seen:
                    seen.add(child)
                    children.append(child)
        frontier = children
    return tuple(regions)


def _region(
    h: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
    s: np.ndarray,
    box: np.ndarray,
    active: tuple[int, ...],
) -> Region | None:
    """The region and law of one active set, or None if its rows are dependent or it is empty."""
    nu, nx = f.shape
    act = list(active)
    g_a = g[act]
    if act and np.linalg.matrix_rank(g_a) < len(act):
        return None
    # U = E U_free + c + C x satisfies the active rows for any U_free. The active rows are solved
    # for the inputs QR pivoting picks; for bound rows these are the bounded inputs, which then
    # take their bound exactly.
    if act:
        basic = np.sort(qr(g_a, pivoting=True, mode="r")[1][: len(act)])
    else:
        basic = np.array([], dtype=int)
    free = np.setdiff1d(np.arange(nu), basic)
    g_b = g_a[:, basic]
    e = np.zeros((nu, len(free)))
    e[free, np.arange(len(free))] = 1.0
    c = np.zeros(nu)
    c_x = np.zeros((nu, nx))
    if act:
        e[basic] = -np.linalg.solve(g_b, g_a[:, free])
        c[basic] = np.linalg.solve(g_b, w[act])
        c_x[basic] = np.linalg.solve(g_b, s[act])
    gain, offset = c_x, c
    if len(free):
        reduced = cho_factor(e.T @ h @ e)
        gain = c_x - e @ cho_solve(reduced, e.T @ (h @ c_x + f))
        offset = c - e @ cho_solve(reduced, e.T @ (h @ c))

    rows, rhs = [], []
    inactive = [i for i in range(len(w)) if i not in active]
    if inactive:
        rows.append(g[inactive] @ gain - s[inactive])
        rhs.append(w[inactive] - g[inactive] @ offset)
    if act:
        # Stationarity on the solved-for inputs, H U + F x + G_A' lam = 0, gives the multipliers.
        lam_x = -np.linalg.solve(g_b.T, (h @ gain + f)[basic])
        lam_c = -np.linalg.solve(g_b.T, (h @ offset)[basic])
        rows.append(-lam_x)
        rhs.append(lam_c)
    rows.append(np.vstack([np.eye(nx), -np.eye(nx)]))
    rhs.append(np.concatenate([box[:, 1], -box[:, 0]]))
    a = np.vstack(rows)
    b = np.concatenate(rhs)
    if not _chebyshev_radius(a, b, box) > CHEBYSHEV_TOL:
        return None
    for array in (gain, offset, a, b):
        array.flags.writeable = False
    return Region(active=active, gain=gain, offset=offset, a=a, b=b)


def _chebyshev_radius(a: np.ndarray, b: np.ndarray, box: np.ndarray) -> float:
    """Radius of the largest ball in {x : a x <= b}, with the box scaled to [-1, 1]; -1 if empty."""
    centre = box.mean(axis=1)
    half = (box[:, 1] - box[:, 0]) / 2
    a_s = a * half
    b_s = b - a @ centre
    nx = a.shape[1]
    c = np.r_[np.zeros(nx), -1.0]
    a_ub = np.column_stack([a_s, np.linalg.norm(a_s, axis=1)])
    bounds = [(None, None)] * nx + [(0.0, None)]
    lp = linprog(c, A_ub=a_ub, b_ub=b_s, bounds=bounds, method="highs")
    if lp.status not in (0, 2):
        # The oldest supported HiGHS reports an unknown status on some degenerate cells.
        lp = linprog(c, A_ub=a_ub, b_ub=b_s, bounds=bounds, method="highs-ipm")
    if lp.status == 2:
        return -1.0
    if lp.status != 0:
        raise RuntimeError(f"chebyshev LP failed: {lp.message}")
    return float(-lp.fun)


def lookup(regions: Sequence[Region], x: ArrayLike) -> tuple[int | None, int]:
    """First region that contains the parameter, by sequential search.

    Parameters
    ----------
    regions : sequence of Region
        Regions from `mpqp_regions`.
    x : array_like, shape (nx,)
        Parameter, in the units of the regions' columns.

    Returns
    -------
    index : int or None
        Position of the first region whose rows hold at x to 1e-12 of their magnitudes; None
        when no region contains x, as outside the box.
    visits : int
        Regions tested, index + 1 when found and the region count otherwise.
    """
    x = np.asarray(x, dtype=float)
    size = np.abs(x)
    for i, region in enumerate(regions):
        slack = LOOKUP_TOL * (np.abs(region.b) + np.abs(region.a) @ size)
        if np.all(region.a @ x <= region.b + slack):
            return i, i + 1
    return None, len(regions)


def _condensed(
    phi: np.ndarray,
    gamma: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    p: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """H and F of sum (x'Qx + u'Ru + 2x'Su) + x_N'P x_N over the moves, as 1/2 U'HU + (F x0)'U."""
    nx, nu = gamma.shape
    # Rows of block k give x_k = Phi^k x_0 + sum_(j<k) Phi^(k-1-j) Gamma u_j, for k = 0, ..., N.
    s_x = np.zeros(((horizon + 1) * nx, nx))
    s_u = np.zeros(((horizon + 1) * nx, horizon * nu))
    power = np.eye(nx)
    for k in range(horizon + 1):
        s_x[k * nx : (k + 1) * nx] = power
        power = phi @ power
    for k in range(1, horizon + 1):
        rows, earlier = slice(k * nx, (k + 1) * nx), slice((k - 1) * nx, k * nx)
        s_u[rows, (k - 1) * nu : k * nu] = gamma
        s_u[rows, : (k - 1) * nu] = phi @ s_u[earlier, : (k - 1) * nu]
    q_bar = np.zeros(((horizon + 1) * nx,) * 2)
    s_bar = np.zeros(((horizon + 1) * nx, horizon * nu))
    for k in range(horizon):
        q_bar[k * nx : (k + 1) * nx, k * nx : (k + 1) * nx] = q
        s_bar[k * nx : (k + 1) * nx, k * nu : (k + 1) * nu] = s
    q_bar[-nx:, -nx:] = p
    cross = s_u.T @ s_bar
    h = 2.0 * (s_u.T @ q_bar @ s_u + np.kron(np.eye(horizon), r) + cross + cross.T)
    f = 2.0 * (s_u.T @ q_bar @ s_x + s_bar.T @ s_x)
    return 0.5 * (h + h.T), f


@dataclass(frozen=True, kw_only=True)
class PiecewiseAffineConfig:
    """Settings of the piecewise-affine MPC.

    Parameters
    ----------
    regulator : LQRConfig
        Weights and stage-cost convention, shared with `LQRAgent`; the rate weight must be 0.
    horizon : int
        Moves per plan, at least 1; the terminal cost is the LQR cost-to-go.
    box : tuple of 6 float
        Half widths of the state box around the hold point over which regions are built, Hill
        order (m, m, m, m/s, m/s, m/s); outside it the agent applies the clipped LQR law.

    Raises
    ------
    ValueError
        If `horizon` is not a positive integer or `box` does not hold six positive finite half
        widths.
    """

    regulator: LQRConfig = field(default_factory=LQRConfig)
    horizon: int = 2
    box: tuple[float, ...] = (600.0, 800.0, 700.0, 1.0, 1.0, 1.0)

    def __post_init__(self):
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise ValueError("horizon must be an integer")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        box = tuple(float(v) for v in self.box)
        if len(box) != 6 or not all(math.isfinite(v) and v > 0.0 for v in box):
            raise ValueError("box must hold six positive finite half widths")
        object.__setattr__(self, "box", box)


class PiecewiseAffineMPCAgent:
    """Hold-point MPC on the CW model, solved offline into piecewise-affine laws.

    At reset the condensed box-constrained regulator of each decoupled CW subsystem (in-plane
    and cross-track) is solved as a multiparametric QP over the state box, with the stage cost
    and terminal cost of `regulator_model`, so the first move equals the LQR command
    us - K (x - xs) wherever no input limit binds over the horizon. Each decision looks up the
    deviation from the hold point in each subsystem's regions and applies the region's first
    move plus the hold input us (Hill frame, m/s^2). Outside the box the command is the LQR law
    clipped to the box. Box thrust limits only; keep-out and other path constraints are not
    modelled.

    `log` holds one entry per decision and `fallbacks` the count of decisions by level (0 a
    region's law, 2 the clipped LQR law outside the box; 1 is unused). Both are the agent's own
    bookkeeping, which the recorder and metrics never read as scores.

    Parameters
    ----------
    config : PiecewiseAffineConfig, optional
        Weights, horizon and box; the defaults of `PiecewiseAffineConfig` when omitted.
    name : str
        Name in results.
    """

    def __init__(self, config: PiecewiseAffineConfig | None = None, name: str = "pwa-mpc"):
        self.config = PiecewiseAffineConfig() if config is None else config
        self.name = name
        self.log: list[dict] = []
        self.fallbacks = [0, 0, 0]

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Build the regions for the view's orbit, step, thrust and hold point.

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
            no continuous command, the thrust limit is not a box, the regulator has a rate
            weight, the observation carries pending commands (no delay model), the hold point
            needs more than `max_accel` of steady thrust, or the LQR design fails.
        """
        hold_point(view, "piecewise-affine MPC agent")
        self.scale = action_scale(view)
        if view.thrust_limit == "ball":
            raise ValueError(
                "a ball thrust limit couples the axes; the piecewise-affine MPC takes a box limit"
            )
        if view.thrust_limit != "box":
            raise ValueError(f"unknown thrust limit {view.thrust_limit!r}")
        cfg = self.config
        if cfg.regulator.rate_weight > 0.0:
            raise ValueError("the piecewise-affine MPC takes no rate weight")
        model = regulator_model(view, cfg.regulator)
        if model.slots:
            raise ValueError("the piecewise-affine MPC does not model pending commands")
        self.view = view
        self.xs = model.xs
        self.us = model.us
        self.k = model.k
        a_max = view.max_accel
        self.subsystems = []
        for xi, ui in SUBSYSTEMS:
            ix, iu = list(xi), list(ui)
            h, f = _condensed(
                model.phi[np.ix_(ix, ix)],
                model.gamma[np.ix_(ix, iu)],
                model.q[np.ix_(ix, ix)],
                model.r[np.ix_(iu, iu)],
                model.s[np.ix_(ix, iu)],
                model.p[np.ix_(ix, ix)],
                cfg.horizon,
            )
            nu = h.shape[0]
            # Limits on the deviation u - us of every move: u - us <= a_max - us and
            # -(u - us) <= a_max + us.
            us = np.tile(model.us[iu], cfg.horizon)
            g = np.vstack([np.eye(nu), -np.eye(nu)])
            w = np.concatenate([a_max - us, a_max + us])
            s = np.zeros((2 * nu, len(ix)))
            bounds = [(-cfg.box[j], cfg.box[j]) for j in ix]
            self.subsystems.append((ix, iu, mpqp_regions(h, f, g, w, s, bounds)))
        self.log = []
        self.fallbacks = [0, 0, 0]

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Command of the region that contains the state, or the clipped LQR law outside the box.

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
        view = self.view
        t = float(info["time_s"])
        dx = np.asarray(obs, dtype=float)[:6] - self.xs
        start = time.perf_counter()
        du = np.zeros(3)
        found: list[int] = []
        visits = 0
        for ix, iu, regions in self.subsystems:
            index, count = lookup(regions, dx[ix])
            visits += count
            if index is None:
                break
            found.append(index)
            region = regions[index]
            du[iu] = region.gain[: len(iu)] @ dx[ix] + region.offset[: len(iu)]
        level = 0 if len(found) == len(self.subsystems) else 2
        if level == 2:
            du = -self.k @ dx
        # A region's law meets the limits up to the rounding of its solve; the clip keeps the
        # command inside the action space.
        u = np.clip(self.us + du, -view.max_accel, view.max_accel)
        self.fallbacks[level] += 1
        self.log.append(
            {
                "time_s": t,
                "level": level,
                "regions": tuple(found) if level == 0 else None,
                "visits": visits,
                "elapsed_s": time.perf_counter() - start,
                "fallback": level == 2,
            }
        )
        return u / view.max_accel * self.scale
