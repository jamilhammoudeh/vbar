"""Reachable sets of affine loops on the exact Clohessy-Wiltshire step.

With the command held for each step the CW motion is linear, so a loop x+ = A x + B w + c with
the start state in a zonotope and w in a zonotope or a thrust limit has zonotope reachable sets,
and their support functions are exact (Kochenderfer, Katz, Corso and Moss, *Algorithms for
Validation*, sec. 8.3 and 8.4). Here they give a certified keep-out clearance of a whole set of
states and a cover of a box of start states by parts whose drift is certified clear or enters,
an invariance test for closed loops, a minimum number of steps with a separating
direction that proves fewer steps infeasible, and a bound on the gravity a CW model leaves out.

Supports are computed by propagating the direction backward through the loop, so no set grows
and nothing is approximated; explicit sets, needed for containment and clearance, are formed
from the closed-loop matrix before any set operation, and reduced only on request.
"""

from __future__ import annotations

import heapq
import itertools
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, lsq_linear
from scipy.spatial import ConvexHull

from .constraints import KeepOut, interval_bounds
from .dynamics.cw import cw_discrete
from .solvers.qp import ActuatorSet

# contains() compares supports on every facet of the outer zonotope; above this many facets the
# enumeration is refused rather than left to run for minutes.
MAX_FACETS = 4096
# vertices() enumerates 2^m corners.
_MAX_VERTEX_GENERATORS = 16
# Rank and containment tests at the rounding of the products involved.
_RANK_TOL = 1e-10
_CONTAIN_TOL = 1e-12
# A minimum-time certificate must beat the support by this fraction of its terms' size.
_GAP_TOL = 1e-9
# The batched nearest-point search: iterations, then exact least squares where the bounds
# still differ by more than this fraction of the distance (or of 1 m).
_NEAREST_ITERATIONS = 300
_NEAREST_TOL = 1e-6


def _array(value, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=float)
    if array.ndim != len(shape) or any(
        size is not None and size != got for size, got in zip(shape, array.shape, strict=True)
    ):
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.flags.writeable = False
    return array


def _count(value, name: str, least: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    count = operator.index(value)
    if count < least:
        raise ValueError(f"{name} must be at least {least}")
    return count


@dataclass(frozen=True)
class Zonotope:
    """The set {c + G beta : |beta_j| <= 1}.

    Parameters
    ----------
    center : array_like, shape (n,)
        The center c, in the units of the state it describes.
    generators : array_like, shape (n, m)
        The generators G as columns; m may be 0.

    Raises
    ------
    ValueError
        If the arrays are not finite or their shapes disagree.

    Notes
    -----
    The support function is h(d) = d . c + sum_j |d . g_j| (Kochenderfer et al., sec. 8.4.1).
    The arrays are read-only copies.
    """

    center: np.ndarray
    generators: np.ndarray

    def __post_init__(self):
        center = _array(self.center, (None,), "center")
        generators = np.array(self.generators, dtype=float)
        if generators.size == 0:
            generators = np.zeros((center.size, 0))
        generators = _array(generators, (center.size, None), "generators")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "generators", generators)

    @classmethod
    def box(cls, center, half_widths) -> Zonotope:
        """The box |x_i - c_i| <= r_i, one generator per nonzero half-width.

        Raises
        ------
        ValueError
            If a half-width is negative or the shapes disagree.
        """
        c = _array(center, (None,), "center")
        r = _array(half_widths, (c.size,), "half_widths")
        if np.any(r < 0.0):
            raise ValueError("half_widths must not be negative")
        return cls(c, np.diag(r)[:, r > 0.0])

    @classmethod
    def point(cls, x) -> Zonotope:
        """The single point `x`."""
        return cls(x, np.zeros((len(np.atleast_1d(x)), 0)))

    @property
    def dim(self) -> int:
        return int(self.center.size)

    def support(self, d) -> float:
        """Return max over the set of d . x."""
        d = _array(d, (self.dim,), "d")
        return float(d @ self.center + np.sum(np.abs(self.generators.T @ d)))

    def support_point(self, d) -> np.ndarray:
        """A point of the set where d . x is largest."""
        d = _array(d, (self.dim,), "d")
        return self.center + self.generators @ np.sign(self.generators.T @ d)

    def map(self, matrix, offset=None) -> Zonotope:
        """The image {M x + b : x in the set}; M is (k, n), b is (k,) or None."""
        m = _array(matrix, (None, self.dim), "matrix")
        center = m @ self.center
        if offset is not None:
            center = center + _array(offset, (m.shape[0],), "offset")
        return Zonotope(center, m @ self.generators)

    def __add__(self, other) -> Zonotope:
        """Minkowski sum with a zonotope, or translation by a vector."""
        if isinstance(other, Zonotope):
            if other.dim != self.dim:
                raise ValueError("zonotopes of different dimension")
            return Zonotope(
                self.center + other.center, np.hstack([self.generators, other.generators])
            )
        return Zonotope(self.center + _array(other, (self.dim,), "offset"), self.generators)

    __radd__ = __add__

    def interval_hull(self) -> tuple[np.ndarray, np.ndarray]:
        """The smallest box containing the set, as (low, high)."""
        r = np.sum(np.abs(self.generators), axis=1)
        return self.center - r, self.center + r

    def reduce(self, order: float) -> Zonotope:
        """An outer zonotope with at most `order` times n generators.

        Keeps the generators that are furthest from axis-aligned, ranked by |g|_1 - |g|_inf,
        and covers the rest by their interval hull, n axis-aligned generators (Girard's
        reduction, HSCC 2005, as stated in arXiv:2209.09321). The result
        contains the set; the set is returned unchanged when it has few enough generators.

        Raises
        ------
        ValueError
            If `order` is below 1.
        """
        order = float(order)
        if not order >= 1.0:
            raise ValueError("order must be at least 1")
        n, m = self.generators.shape
        limit = int(math.floor(order * n))
        if m <= limit:
            return self
        g = self.generators
        score = np.sum(np.abs(g), axis=0) - np.max(np.abs(g), axis=0)
        ranked = np.argsort(score, kind="stable")
        boxed, kept = ranked[: m - (limit - n)], ranked[m - (limit - n) :]
        hull = np.sum(np.abs(g[:, boxed]), axis=1)
        return Zonotope(self.center, np.hstack([g[:, np.sort(kept)], np.diag(hull)]))

    def vertices(self) -> np.ndarray:
        """The vertices of a full-dimensional zonotope with at most 16 generators.

        Raises
        ------
        ValueError
            If the set is not full-dimensional or has more than 16 generators.
        """
        n, m = self.generators.shape
        if m > _MAX_VERTEX_GENERATORS:
            raise ValueError(f"vertices of more than {_MAX_VERTEX_GENERATORS} generators")
        if _rank(self.generators) < n:
            raise ValueError("the zonotope is not full-dimensional")
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=m)))
        corners = self.center + signs @ self.generators.T
        if n == 1:
            return np.array([[corners.min()], [corners.max()]])
        return corners[ConvexHull(corners).vertices]


@dataclass(frozen=True)
class Polytope:
    """The bounded set {x : H x <= g}.

    Parameters
    ----------
    H : array_like, shape (k, n)
        Facet normals as rows.
    g : array_like, shape (k,)
        Right-hand sides.

    Raises
    ------
    ValueError
        If the arrays are not finite or their shapes disagree.
    """

    H: np.ndarray
    g: np.ndarray

    def __post_init__(self):
        H = _array(self.H, (None, None), "H")
        object.__setattr__(self, "H", H)
        object.__setattr__(self, "g", _array(self.g, (H.shape[0],), "g"))

    @property
    def dim(self) -> int:
        return int(self.H.shape[1])

    def support(self, d) -> float:
        """Return max over the set of d . x, by a linear program.

        Raises
        ------
        ValueError
            If the set is empty or unbounded along `d`.
        """
        return _lp_support([(self, _array(d, (self.dim,), "d"))])

    def support_point(self, d) -> np.ndarray:
        """A point of the set where d . x is largest, by a linear program.

        Raises
        ------
        ValueError
            If the set is empty or unbounded along `d`.
        """
        d = _array(d, (self.dim,), "d")
        found = linprog(-d, A_ub=self.H, b_ub=self.g, bounds=[(None, None)] * self.dim)
        if found.status == 2:
            raise ValueError("the polytope is empty")
        if found.status == 3:
            raise ValueError("the polytope is unbounded along the direction")
        if found.status != 0:
            raise RuntimeError(f"the support program failed: {found.message}")
        return np.asarray(found.x, dtype=float)


SetLike = Zonotope | Polytope | ActuatorSet


def _as_set(value, dim: int, name: str) -> SetLike:
    """A zonotope, polytope or thrust limit; an array is the point it names."""
    if isinstance(value, Zonotope | Polytope):
        if value.dim != dim:
            raise ValueError(f"{name} has dimension {value.dim}, expected {dim}")
        return value
    if isinstance(value, ActuatorSet):
        if dim != 3:
            raise ValueError(f"{name} is a thrust limit, which needs 3 inputs, not {dim}")
        return value
    return Zonotope.point(_array(value, (dim,), name))


def _support_of(s: SetLike, d: np.ndarray) -> float:
    if isinstance(s, ActuatorSet):
        return s.a_max * float(np.sum(np.abs(d)) if s.layout == "box" else np.linalg.norm(d))
    return s.support(d)


def _lp_support(terms: list[tuple[SetLike, np.ndarray]]) -> float:
    """sum_i h_{S_i}(d_i) as one linear program over a point of every set."""
    cost: list[np.ndarray] = []
    bounds: list[tuple[float | None, float | None]] = []
    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    constant = 0.0
    width = 0
    for s, d in terms:
        if isinstance(s, Polytope):
            rows.append((width, s.H, s.g))
            cost.append(d)
            bounds += [(None, None)] * s.dim
            width += s.dim
        elif isinstance(s, Zonotope):
            constant += float(d @ s.center)
            cost.append(s.generators.T @ d)
            bounds += [(-1.0, 1.0)] * s.generators.shape[1]
            width += s.generators.shape[1]
        elif s.layout == "box":
            cost.append(d)
            bounds += [(-s.a_max, s.a_max)] * 3
            width += 3
        else:
            # A ball is not polyhedral; its support is closed form and separable.
            constant += s.a_max * float(np.linalg.norm(d))
    if width == 0:
        return constant
    a_ub = np.zeros((sum(h.shape[0] for _, h, _ in rows), width))
    b_ub = np.zeros(a_ub.shape[0])
    top = 0
    for start, h, g in rows:
        a_ub[top : top + h.shape[0], start : start + h.shape[1]] = h
        b_ub[top : top + h.shape[0]] = g
        top += h.shape[0]
    found = linprog(
        -np.concatenate(cost),
        A_ub=a_ub if rows else None,
        b_ub=b_ub if rows else None,
        bounds=bounds,
        method="highs",
    )
    if found.status == 2:
        raise ValueError("the polytope is empty")
    if found.status == 3:
        raise ValueError("the polytope is unbounded along the direction")
    if found.status != 0:
        raise RuntimeError(f"the support program failed: {found.message}")
    return constant - float(found.fun)


@dataclass(frozen=True)
class LinearLoop:
    """The affine loop x+ = A x + B w + c, one step per decision.

    Parameters
    ----------
    A : array_like, shape (n, n)
        The closed-loop transition matrix.
    B : array_like, shape (n, m)
        How the disturbance w enters.
    c : array_like, shape (n,)
        The constant term.

    Raises
    ------
    ValueError
        If the arrays are not finite or their shapes disagree.
    """

    A: np.ndarray
    B: np.ndarray
    c: np.ndarray

    def __post_init__(self):
        A = _array(self.A, (None, None), "A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", _array(self.B, (A.shape[0], None), "B"))
        object.__setattr__(self, "c", _array(self.c, (A.shape[0],), "c"))

    @property
    def dim(self) -> int:
        return int(self.A.shape[0])


@dataclass(frozen=True)
class CwLoop(LinearLoop):
    """A `LinearLoop` built by `cw_loop`, which also knows the state inside each step.

    `n` is the mean motion (rad/s), `dt` the step (s), `gain` the feedback K (3, 6) or None for
    free drift, and `hold` the state the feedback acts around (m, m/s).
    """

    n: float = 0.0
    dt: float = 0.0
    gain: np.ndarray | None = None
    hold: np.ndarray | None = None

    def within(self, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(A_s, B_s, c_s) of the state `s` s into a step, 0 <= s <= dt."""
        phi, gamma = cw_discrete(self.n, s)
        if self.gain is None:
            return phi, gamma, np.zeros(6)
        assert self.hold is not None
        k = self.gain
        return phi - gamma @ k, -gamma @ k, gamma @ k @ self.hold


def cw_loop(n: float, dt: float, gain=None, hold=None) -> CwLoop:
    """The CW step as a loop, open or closed by state feedback u = -K (x_hat - hold).

    Parameters
    ----------
    n : float
        Mean motion, rad/s, positive.
    dt : float
        Hold time of each command, s, positive.
    gain : array_like, shape (3, 6), optional
        The feedback K, m/s^2 per m and per m/s. Without it the loop is free drift with the
        thrust (m/s^2) as its disturbance: A = phi, B = gamma.
    hold : array_like, shape (6,), optional
        The state the feedback regulates to (m, m/s); zero by default. Needs `gain`.

    Returns
    -------
    CwLoop
        With feedback, A = phi - gamma K, the closed-loop matrix formed before any set
        operation, B = -gamma K for a navigation error w on the state, and c = gamma K hold.

    Raises
    ------
    ValueError
        If `n` or `dt` is not finite and positive, a shape is wrong, or `hold` comes without
        `gain`.
    """
    n, dt = float(n), float(dt)
    if not (math.isfinite(n) and n > 0.0 and math.isfinite(dt) and dt > 0.0):
        raise ValueError("n and dt must be finite and positive")
    phi, gamma = cw_discrete(n, dt)
    if gain is None:
        if hold is not None:
            raise ValueError("hold needs a gain")
        return CwLoop(phi, gamma, np.zeros(6), n=n, dt=dt)
    k = _array(gain, (3, 6), "gain")
    x_hold = np.zeros(6) if hold is None else _array(hold, (6,), "hold")
    return CwLoop(phi - gamma @ k, -gamma @ k, gamma @ k @ x_hold, n=n, dt=dt, gain=k, hold=x_hold)


def support(loop: LinearLoop, x0, w, d, steps: int) -> float:
    """Return the support of the reachable set after `steps` steps along `d`.

    Parameters
    ----------
    loop : LinearLoop
        The loop x+ = A x + B w + c.
    x0 : Zonotope, Polytope or array_like
        The start set, or a start state.
    w : Zonotope, Polytope, ActuatorSet, array_like or None
        The disturbance set, held to any value in it at each step independently; None for no
        disturbance.
    d : array_like, shape (n,)
        The direction.
    steps : int
        Steps, nonnegative.

    Returns
    -------
    float
        max d . x_N over every start in `x0` and every disturbance sequence in `w`.

    Notes
    -----
    h(d) = h_X0((A^T)^N d) + sum_j [h_W(B^T (A^T)^j d) + (A^T)^j d . c], the support of the
    Minkowski sum of eq. 8.20 of Kochenderfer et al., computed by propagating the direction
    backward, so it is exact for any horizon. When a set is a `Polytope` the sum is one linear
    program over the start and every disturbance (eq. 8.27).
    """
    dim = loop.dim
    start = _as_set(x0, dim, "x0")
    disturbance = None if w is None else _as_set(w, loop.B.shape[1], "w")
    direction = _array(d, (dim,), "d")
    steps = _count(steps, "steps", 0)
    terms: list[tuple[SetLike, np.ndarray]] = []
    constant = 0.0
    lam = direction
    for _ in range(steps):
        if disturbance is not None:
            terms.append((disturbance, loop.B.T @ lam))
        constant += float(lam @ loop.c)
        lam = loop.A.T @ lam
    terms.append((start, lam))
    if any(isinstance(s, Polytope) for s, _ in terms):
        return constant + _lp_support(terms)
    return constant + sum(_support_of(s, d) for s, d in terms)


def _nearest(c: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounds on the distance from the origin to the zonotopes c_i + g_i beta, |beta| <= 1.

    `c` is (s, k) and `g` (s, k, m). Returns (lower, upper), each (s,). The upper bound is the
    norm of a point of the set; the lower bound is lam . c - sum_j |lam . g_j| with lam the
    unit residual of that point, the separating hyperplane, less the allowance for its rounding
    (`_lower_allowance`), so both hold however far the search is from its optimum. The point
    comes from an accelerated projected gradient on all sets at once, and sets whose bounds
    still differ by more than 1e-6 of the distance are solved again by bounded least squares.
    """
    lower, upper = _searched(c, g)
    for i in np.flatnonzero(_loose(lower, upper)).tolist():
        _refine(c, g, lower, upper, i)
    return lower, upper


def _searched(c: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The bounds of `_nearest` from the gradient search alone."""
    count, _, m = g.shape
    if m == 0:
        norm = np.linalg.norm(c, axis=1)
        return norm - _lower_allowance(c, g), norm.copy()
    step = 1.0 / np.maximum(np.linalg.norm(g, ord=2, axis=(1, 2)) ** 2, 1e-300)
    beta = np.zeros((count, m))
    ahead = beta
    t = 1.0
    for _ in range(_NEAREST_ITERATIONS):
        residual = c + np.einsum("skm,sm->sk", g, ahead)
        moved = np.clip(ahead - step[:, None] * np.einsum("skm,sk->sm", g, residual), -1.0, 1.0)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        ahead = moved + (t - 1.0) / t_next * (moved - beta)
        beta, t = moved, t_next
    return _bounds_at(c, g, beta)


def _lower_allowance(c: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Rounding (m) a lower bound of `_bounds_at` and the norm of a point of the set can carry.

    lam . c - sum_j |lam . g_j| and the norm of a point of the set are each computed within
    (m + 5) eps of |c| + sum_j |g_j| (Higham, Accuracy and Stability of Numerical Algorithms,
    sec. 3.1), so the bound less this never sits above the exact distance, and a solve cannot
    bring an upper bound below the bound less this either.
    """
    return 4.0 * (g.shape[2] + 8) * np.finfo(float).eps * _largest_norms(c, g)


def _lower_margin(lower: np.ndarray, largest: np.ndarray, radius: float) -> np.ndarray:
    """Lower bounds on |p| - radius from `lower` on |p|, less the rounding of the margin.

    `largest` is |c| + sum_j |g_j| of each set. The subtraction and the sums that combine two
    margins in `_sample_bounds` round at the size of the larger of `largest` and `radius`, at
    most 2 eps of it in all. `_lower_allowance` exceeds the (m + 5) eps `largest` that the bound
    needs by at least 27 eps `largest`, which holds that while radius <= 13 `largest`; beyond it
    the margin is lowered by 4 eps (radius - 13 `largest`), which with the unused part is at
    least 2 eps radius.
    """
    return lower - radius - 4.0 * np.finfo(float).eps * np.maximum(radius - 13.0 * largest, 0.0)


def _loose(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return upper - lower > _NEAREST_TOL * np.maximum(upper, 1.0)


def _refine(c: np.ndarray, g: np.ndarray, lower: np.ndarray, upper: np.ndarray, i: int) -> None:
    """Tighten the bounds of set `i` in place by bounded least squares."""
    exact = np.clip(lsq_linear(g[i], -c[i], bounds=(-1.0, 1.0), method="bvls").x, -1.0, 1.0)
    low, up = _bounds_at(c[i : i + 1], g[i : i + 1], exact[None])
    lower[i] = max(lower[i], low[0])
    upper[i] = min(upper[i], up[0])


def _bounds_at(c: np.ndarray, g: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = c + np.einsum("skm,sm->sk", g, beta)
    upper = np.linalg.norm(residual, axis=1)
    lam = residual / np.where(upper > 0.0, upper, 1.0)[:, None]
    lower = np.einsum("sk,sk->s", lam, c) - np.sum(np.abs(np.einsum("skm,sk->sm", g, lam)), axis=1)
    lower -= _lower_allowance(c, g)
    # A set that reaches the origin has distance 0, which no direction can beat.
    return np.where(upper > 0.0, np.maximum(lower, 0.0), 0.0), upper


def ball_distance(z: Zonotope, radius: float) -> tuple[float, float]:
    """Bounds on min over the set of |p| - radius, for a zonotope of positions p.

    Parameters
    ----------
    z : Zonotope
        Positions, m.
    radius : float
        The ball's radius, m, nonnegative.

    Returns
    -------
    lower : float
        A certified lower bound, m: with r a point of the set near the origin (bounded least
        squares) and lam = r / |r|, every point p of the set has |p| >= lam . p >=
        lam . c - sum_j |lam . g_j|; a separating hyperplane, so the bound holds whatever the
        solver's accuracy, less an allowance of (4 m + 32) eps (|c| + sum_j |g_j|) for the
        rounding of the products and, for a radius beyond 13 (|c| + sum_j |g_j|), one of 4 eps
        times the excess for the rounding of the margin. -radius when the set reaches the
        origin.
    upper : float
        |r| - radius, m, the margin of a point of the set.
    """
    radius = float(radius)
    if not (math.isfinite(radius) and radius >= 0.0):
        raise ValueError("radius must be finite and nonnegative")
    lower, upper = _nearest(z.center[None], z.generators[None])
    largest = _largest_norms(z.center[None], z.generators[None])
    return float(_lower_margin(lower, largest, radius)[0]), float(upper[0]) - radius


@dataclass(frozen=True)
class Clearance:
    """Keep-out margin of a set of trajectories.

    `found` is the lowest margin of a point of the sets found at the samples (m), at time `t`
    (s); `certified` a lower bound on the margin of every trajectory at every time, between
    samples included (m). `clear` is whether the certified margin is at least zero.
    """

    found: float
    certified: float
    t: float

    @property
    def clear(self) -> bool:
        return self.certified >= 0.0


def _largest_norms(c: np.ndarray, g: np.ndarray) -> np.ndarray:
    """|c_i| + sum_j |g_ij|, the largest norm over each zonotope; `c` (s, k), `g` (s, k, m)."""
    return np.linalg.norm(c, axis=1) + np.sum(np.linalg.norm(g, axis=1), axis=1)


def _sample_bounds(
    c: np.ndarray, g: np.ndarray, h: float, thrust: float | np.ndarray, n: float, radius: float
) -> tuple[np.ndarray, float]:
    """Upper sample margins (m) and the certified margin over every interval (m).

    The sets are states c_i + G_i beta at consecutive samples `h` s apart, `c` (s, 6) and `g`
    (s, 6, m); `thrust` bounds the thrust (m/s^2) over every interval, or over each, (s - 1,).

    The lowest upper margin, its first index and the certified margin equal those of `_nearest`
    on every sample. A loose sample is solved by bounded least squares only while it can still
    set one of them: while its search lower margin, which carries the rounding allowance of both
    bounds and of the margin (`_lower_margin`), is at or below the lowest upper margin, or an
    interval it ends has a lower two-line bound than every interval between solved or tight
    samples. A solve only raises a lower bound and lowers an upper one, so a skipped sample can
    neither take the lowest upper margin, which it exceeds strictly, nor lower the certified
    one. Skipped samples keep their search bounds, which hold.
    """
    position, spread = np.ascontiguousarray(c[:, :3]), np.ascontiguousarray(g[:, :3])
    lower, upper = _searched(position, spread)
    largest = np.zeros((c.shape[0], 6))
    largest[:, 0] = _largest_norms(c[:, :3], g[:, :3])
    largest[:, 3] = _largest_norms(c[:, 3:], g[:, 3:])
    # interval_bounds reads only the norms of the sample states, here the largest of each set.
    speed, _ = interval_bounds(largest, h, thrust, n)
    pending = _loose(lower, upper)
    while np.any(pending):
        low = _lower_margin(lower, largest[:, 0], radius)
        lines = 0.5 * (low[:-1] + low[1:] - speed * h)
        unsolved = pending[:-1] | pending[1:]
        settled = np.min(lines[~unsolved], initial=math.inf)
        # A tie with the lowest upper margin could move its first index, so a sample is skipped
        # only when strictly above it; the certified margin is a value, and a tie leaves it.
        needed = pending & (low <= np.min(upper - radius))
        below = unsolved & (lines < settled)
        needed[:-1] |= below & pending[:-1]
        needed[1:] |= below & pending[1:]
        if not np.any(needed):
            break
        candidates = np.flatnonzero(needed)
        i = int(candidates[np.argmin(lower[candidates])])
        _refine(position, spread, lower, upper, i)
        pending[i] = False
    lower, upper = _lower_margin(lower, largest[:, 0], radius), upper - radius
    # The nearest distance to the ball changes no faster than the fastest trajectory.
    certified = float(np.min(0.5 * (lower[:-1] + lower[1:] - speed * h)))
    return upper, certified


def _sampled_clearance(
    c: np.ndarray, g: np.ndarray, h: float, thrust: float, n: float, radius: float, t0: float
) -> tuple[float, float, float]:
    """(found, certified, t) over consecutive samples `h` s apart starting at `t0`."""
    upper, certified = _sample_bounds(c, g, h, thrust, n, radius)
    k = int(np.argmin(upper))
    return float(upper[k]), certified, t0 + k * h


def drift_clearance(n: float, x0, radius: float, duration: float, samples: int) -> Clearance:
    """Keep-out clearance of the free drift of every state in a set.

    Parameters
    ----------
    n : float
        Mean motion, rad/s, positive.
    x0 : Zonotope or array_like
        Start states, Hill frame, m and m/s.
    radius : float
        Keep-out radius, m.
    duration : float
        Drift time, s, positive.
    samples : int
        Samples, endpoints included, at least 2.

    Returns
    -------
    Clearance
        The sampled minimum of the upper bound of `ball_distance` and the two-line bound
        0.5 (l_i + l_{i+1} - v h) over each interval, l the certified sample margins and v a
        bound on the speed of every trajectory over the interval. -inf when the interval is
        too long for that speed bound.
    """
    n, duration = float(n), float(duration)
    if not (math.isfinite(n) and n > 0.0 and math.isfinite(duration) and duration > 0.0):
        raise ValueError("n and duration must be finite and positive")
    samples = _count(samples, "samples", 2)
    start = _as_set(x0, 6, "x0")
    if not isinstance(start, Zonotope):
        raise ValueError("x0 must be a zonotope or a state")
    times = np.linspace(0.0, duration, samples)
    phis, _ = cw_discrete(n, times)
    found, certified, t = _sampled_clearance(
        phis @ start.center,
        phis @ start.generators,
        float(times[1] - times[0]),
        0.0,
        n,
        radius,
        0.0,
    )
    return Clearance(found, certified, t)


@dataclass(frozen=True)
class DriftCover:
    """A box of start states split by whether the free drift from each part clears a keep-out.

    Attributes
    ----------
    certified : float
        Fraction of the start box whose every state drifts clear at every time, between samples
        included.
    excluded : float
        Fraction from which every drift enters the keep-out by more than its 1 mm resolution.
    undecided : float
        The rest: boxes at the depth limit, boxes left when the budget ran out, and boxes that
        stopped at a counterexample of their center.
    counterexamples : tuple of (ndarray, float, float)
        Box centers in the declared set whose drift enters the keep-out by more than 1 mm: the
        state (Hill frame, m and m/s), its lowest sampled margin (m) and that sample's time (s).
    certified_boxes : ndarray, shape (k, 2, 6)
        Lower and upper corners of the certified boxes, m and m/s; read-only.
    undecided_boxes : ndarray, shape (k, 2, 6)
        Lower and upper corners of the undecided boxes, m and m/s; read-only.
    boxes : int
        Boxes checked.

    Notes
    -----
    Volume counts only the axes on which the start box has width, so a box halved s times holds
    2^-s of it and the three fractions add up to 1.
    """

    certified: float
    excluded: float
    undecided: float
    counterexamples: tuple[tuple[np.ndarray, float, float], ...]
    certified_boxes: np.ndarray
    undecided_boxes: np.ndarray
    boxes: int


def _corners(boxes: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    corners = np.array(boxes, dtype=float).reshape(-1, 2, 6)
    corners.flags.writeable = False
    return corners


def drift_cover(
    n: float,
    lo,
    hi,
    radius: float,
    duration: float,
    samples: int,
    contains: Callable[[np.ndarray], bool],
    max_boxes: int,
    depth: int,
    *,
    accel=None,
    burn_time: float = 0.0,
) -> DriftCover:
    """Split a box of start states until the drift of each part is certified or enters.

    Parameters
    ----------
    n : float
        Mean motion, rad/s, positive.
    lo, hi : array_like, shape (6,)
        Corners of the start box, Hill frame, m and m/s, with lo <= hi.
    radius : float
        Keep-out radius, m, nonnegative.
    duration : float
        Drift time, s, positive.
    samples : int
        Samples of each drift, endpoints included, at least 2.
    contains : callable
        Whether a state (ndarray, shape (6,), Hill frame, m and m/s) lies in the declared safe
        set; called on box centers only.
    max_boxes : int
        Most boxes to check, at least 1.
    depth : int
        Most halvings of each axis, nonnegative.
    accel : array_like, shape (3,), optional
        A collision avoidance burn, the thrust acceleration held in the Hill frame from the
        start for `burn_time`, m/s^2. Without it the motion is free drift throughout.
    burn_time : float, optional
        How long `accel` is held, s, finite and nonnegative; free drift follows it.

    Returns
    -------
    DriftCover
        The certified, excluded and undecided fractions, the counterexamples, and the
        certified and undecided boxes.

    Raises
    ------
    ValueError
        If a parameter is out of range, a corner is not six finite numbers, lo exceeds hi,
        `accel` is not three finite numbers, or a positive `burn_time` comes without `accel`.

    Notes
    -----
    The state at time t is Phi(t) x0 + Phi(t - t_b) Gamma(t_b) a once the burn of length t_b
    has ended and Phi(t) x0 + Gamma(t) a before, affine in x0 with the free-drift matrix, so a
    burn moves each sample set's center and leaves its generators as they are. Intervals that
    start before the burn ends add |a| to the speed bound of the certified margin.

    Boxes are checked largest first. With c_k + G_k beta the positions of a box's drift at
    sample k, a box is

    1. certified if the certified margin of `drift_clearance` is at least 0;
    2. excluded if |c_k| + sum_j |g_kj| <= R - 1 mm at some sample: by the triangle inequality
       every drift from it then enters. Its center is a counterexample if `contains` accepts it;
    3. undecided with a counterexample if `contains` accepts its center and the center's lowest
       sampled margin, exact for that state at that time, is below -1 mm;
    4. otherwise halved along the axis whose generator moves the position most, |Phi_r(t*) g_j|
       largest, at the sample t* where |c_k| - sum_j |g_kj| is lowest. Axes already halved
       `depth` times are not split again, and a box with none left is undecided.

    Boxes still queued after `max_boxes` checks are undecided. A center whose sampled margin is
    negative already rules out step 1, so the nearest-point search runs only for the others.
    """
    n, duration, radius = float(n), float(duration), float(radius)
    if not (math.isfinite(n) and n > 0.0 and math.isfinite(duration) and duration > 0.0):
        raise ValueError("n and duration must be finite and positive")
    if not (math.isfinite(radius) and radius >= 0.0):
        raise ValueError("radius must be finite and nonnegative")
    low, high = _array(lo, (6,), "lo"), _array(hi, (6,), "hi")
    if np.any(low > high):
        raise ValueError("lo must not exceed hi")
    samples = _count(samples, "samples", 2)
    max_boxes = _count(max_boxes, "max_boxes", 1)
    depth = _count(depth, "depth", 0)
    burn_time = float(burn_time)
    if not (math.isfinite(burn_time) and burn_time >= 0.0):
        raise ValueError("burn_time must be finite and nonnegative")
    if accel is None:
        if burn_time > 0.0:
            raise ValueError("burn_time needs accel")
        a = np.zeros(3)
    else:
        a = _array(accel, (3,), "accel")
    times = np.linspace(0.0, duration, samples)
    h = float(times[1] - times[0])
    phis, _ = cw_discrete(n, times)
    # Phi(t - t_b) Gamma(t_b) a rather than Gamma(t) a - Gamma(t - t_b) a, which cancels.
    coast, _ = cw_discrete(n, np.maximum(times - burn_time, 0.0))
    _, held = cw_discrete(n, np.minimum(times, burn_time))
    from_burn = np.einsum("tij,tj->ti", coast, held @ a)
    thrust = np.where(times[:-1] < burn_time, float(np.linalg.norm(a)), 0.0)
    axes = np.flatnonzero(high > low)
    resolution = KeepOut.resolution
    order = itertools.count()
    queue = [(0, next(order), low, high, np.zeros(axes.size, dtype=int))]
    certified = excluded = undecided = 0.0
    certified_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    undecided_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    counterexamples: list[tuple[np.ndarray, float, float]] = []
    checked = 0
    while queue and checked < max_boxes:
        splits, _, box_lo, box_hi, halvings = heapq.heappop(queue)
        checked += 1
        volume = 0.5**splits
        center = 0.5 * (box_lo + box_hi)
        center.flags.writeable = False
        c = phis @ center + from_burn
        g = phis[:, :, axes] * (0.5 * (box_hi - box_lo))[axes]
        rho = np.linalg.norm(c[:, :3], axis=1)
        spread = np.sum(np.linalg.norm(g[:, :3], axis=1), axis=1)
        k = int(np.argmin(rho))
        margin = float(rho[k] - radius)
        found = (center, margin, float(times[k]))
        if np.min(rho + spread) <= radius - resolution:
            excluded += volume
            if contains(center.copy()):
                counterexamples.append(found)
            continue
        if margin >= 0.0 and _sample_bounds(c, g, h, thrust, n, radius)[1] >= 0.0:
            certified += volume
            certified_boxes.append((box_lo, box_hi))
            continue
        open_axes = halvings < depth
        entered = margin < -resolution and bool(contains(center.copy()))
        if entered:
            counterexamples.append(found)
        if entered or not np.any(open_axes):
            undecided += volume
            undecided_boxes.append((box_lo, box_hi))
            continue
        worst = int(np.argmin(rho - spread))
        reach = np.linalg.norm(g[worst, :3], axis=0)
        j = int(np.argmax(np.where(open_axes, reach, -1.0)))
        axis = int(axes[j])
        more = halvings.copy()
        more[j] += 1
        lower_hi, upper_lo = box_hi.copy(), box_lo.copy()
        lower_hi[axis] = upper_lo[axis] = center[axis]
        heapq.heappush(queue, (splits + 1, next(order), box_lo, lower_hi, more))
        heapq.heappush(queue, (splits + 1, next(order), upper_lo, box_hi, more))
    for splits, _, box_lo, box_hi, _ in queue:
        undecided += 0.5**splits
        undecided_boxes.append((box_lo, box_hi))
    return DriftCover(
        certified,
        excluded,
        undecided,
        tuple(counterexamples),
        _corners(certified_boxes),
        _corners(undecided_boxes),
        checked,
    )


def _explicit(value, dim: int, name: str) -> Zonotope:
    s = _as_set(value, dim, name)
    if isinstance(s, Zonotope):
        return s
    if isinstance(s, ActuatorSet) and s.layout == "box":
        return Zonotope.box(np.zeros(3), np.full(3, s.a_max))
    raise ValueError(f"{name} must be a zonotope or a box thrust limit for an explicit set")


def _step(loop: LinearLoop, z: Zonotope, w: Zonotope | None, order: float | None) -> Zonotope:
    """reduce(A z + B w + c): the next set, fed back as the next input."""
    nxt = z.map(loop.A, loop.c)
    if w is not None:
        nxt = nxt + w.map(loop.B)
    return nxt if order is None else nxt.reduce(order)


def propagate(loop: LinearLoop, x0, w, steps: int, order: float | None = None) -> list[Zonotope]:
    """The reachable sets R_0 to R_steps as zonotopes.

    Each set is the image of the previous output, A R_k + B W + c, so the sets are exact
    without `order` and outer sets with it (reduced to `order` times n generators after every
    step). `w` is a zonotope, a box thrust limit or None.
    """
    steps = _count(steps, "steps", 0)
    z = _explicit(x0, loop.dim, "x0")
    disturbance = None if w is None else _explicit(w, loop.B.shape[1], "w")
    sets = [z]
    for _ in range(steps):
        z = _step(loop, z, disturbance, order)
        sets.append(z)
    return sets


def _rank(g: np.ndarray) -> int:
    if g.size == 0:
        return 0
    s = np.linalg.svd(g, compute_uv=False)
    return int(np.sum(s > _RANK_TOL * s[0]))


def _facet_normals(z: Zonotope) -> np.ndarray:
    """Unit normals of the facets of a full-dimensional zonotope, each sign once."""
    n, m = z.generators.shape
    if _rank(z.generators) < n:
        raise ValueError("the outer zonotope is not full-dimensional")
    if n == 1:
        return np.ones((1, 1))
    if 2 * math.comb(m, n - 1) > MAX_FACETS:
        raise ValueError(f"the outer zonotope has more than {MAX_FACETS} facets")
    normals = []
    for subset in itertools.combinations(range(m), n - 1):
        u, s, _ = np.linalg.svd(z.generators[:, subset], full_matrices=True)
        if s[-1] > _RANK_TOL * s[0]:
            normals.append(u[:, -1])
    return np.array(normals)


def contains(outer: Zonotope, inner) -> bool:
    """Return whether the zonotope `inner` (or a point) lies inside the zonotope `outer`.

    Exact: a full-dimensional zonotope is the intersection of its facet halfspaces, so `inner`
    lies inside it iff its support is at most the outer one along every facet normal, compared
    to 1e-12 of the outer set's extent along the normal.

    Raises
    ------
    ValueError
        If `outer` is not full-dimensional or has more than `MAX_FACETS` facets.
    """
    inside = _explicit(inner, outer.dim, "inner")
    for normal in _facet_normals(outer):
        for d in (normal, -normal):
            extent = abs(float(d @ outer.center)) + float(np.sum(np.abs(outer.generators.T @ d)))
            if inside.support(d) > outer.support(d) + _CONTAIN_TOL * max(extent, 1.0):
                return False
    return True


def invariant_step(loop: LinearLoop, x0, w, max_steps: int, order: float = 2.0) -> int | None:
    """The first d <= max_steps whose image A R_{d-1} + B W + c lies inside R_{d-1}, else None.

    R_{d-1} is the carried set: the previous image reduced to `order` times n generators, an
    outer set of the exact reachable set. When it contains its own image every later state stays
    in it (Kochenderfer et al., sec. 8.2), so the loop is safe for all time only if that carried
    set, not just the exact sets up to d, is clear of the avoid set. The image is tested exactly,
    and the reduction keeps the facet count of `contains` bounded.
    """
    max_steps = _count(max_steps, "max_steps", 1)
    z = _explicit(x0, loop.dim, "x0").reduce(order)
    disturbance = None if w is None else _explicit(w, loop.B.shape[1], "w")
    for d in range(1, max_steps + 1):
        image = _step(loop, z, disturbance, None)
        if contains(z, image):
            return d
        z = image.reduce(order)
    return None


def _thrust_bound(loop: CwLoop, z: Zonotope, w: Zonotope | None, actuator) -> float:
    """A bound on |u|_2 over the step from the set `z`, m/s^2; refuses a saturating loop."""
    if loop.gain is None:
        if w is None:
            return 0.0
        return float(np.linalg.norm(w.center) + np.sum(np.linalg.norm(w.generators, axis=0)))
    assert loop.hold is not None
    if actuator is None:
        raise ValueError("a closed loop needs its actuator to check saturation")
    error = z if w is None else z + w
    u = error.map(-loop.gain, loop.gain @ loop.hold)
    a_max = actuator.a_max * (1.0 + _CONTAIN_TOL)
    norm_bound = float(np.linalg.norm(u.center) + np.sum(np.linalg.norm(u.generators, axis=0)))
    if actuator.layout == "box":
        low, high = u.interval_hull()
        saturates = bool(np.any(np.maximum(-low, high) > a_max))
    else:
        saturates = norm_bound > a_max
    if saturates:
        raise ValueError("the loop saturates on the set")
    return norm_bound


def loop_clearance(
    loop: CwLoop,
    x0,
    w,
    radius: float,
    steps: int,
    samples_per_step: int,
    actuator: ActuatorSet | None = None,
    order: float | None = None,
) -> Clearance:
    """Keep-out clearance of every trajectory of a CW loop over `steps` steps.

    Parameters
    ----------
    loop : CwLoop
        From `cw_loop`.
    x0 : Zonotope or array_like
        Start states, Hill frame, m and m/s.
    w : Zonotope, ActuatorSet or None
        Navigation error (m, m/s) with a gain, thrust (m/s^2) without one.
    radius : float
        Keep-out radius, m.
    steps : int
        Steps, at least 1.
    samples_per_step : int
        Samples in each step, endpoints included, at least 2.
    actuator : ActuatorSet, optional
        The thrust limit; needed with a gain.
    order : float, optional
        Reduce each step's set to this many times 6 generators.

    Returns
    -------
    Clearance
        As `drift_clearance`, over the held steps; the speed bound includes the largest
        command over each step's set.

    Raises
    ------
    ValueError
        If the feedback command exceeds the thrust limit anywhere on a step's set ("the loop
        saturates on the set"): the loop is then not linear and its sets would be wrong.
    """
    if not isinstance(loop, CwLoop):
        raise ValueError("loop_clearance needs a loop from cw_loop")
    steps = _count(steps, "steps", 1)
    samples = _count(samples_per_step, "samples_per_step", 2)
    z = _explicit(x0, 6, "x0")
    disturbance = None if w is None else _explicit(w, loop.B.shape[1], "w")
    taus = np.linspace(0.0, loop.dt, samples)
    inner = [loop.within(float(s)) for s in taus]
    h = float(taus[1] - taus[0])
    best = Clearance(math.inf, math.inf, 0.0)
    for k in range(steps):
        thrust = _thrust_bound(loop, z, disturbance, actuator)
        sets = []
        for a_s, b_s, c_s in inner:
            at = z.map(a_s, c_s)
            sets.append(at if disturbance is None else at + disturbance.map(b_s))
        found, certified, t = _sampled_clearance(
            np.array([z.center for z in sets]),
            np.array([z.generators for z in sets]),
            h,
            thrust,
            loop.n,
            radius,
            k * loop.dt,
        )
        best = Clearance(
            min(best.found, found),
            min(best.certified, certified),
            t if found < best.found else best.t,
        )
        z = _step(loop, z, disturbance, order)
    return best


@dataclass(frozen=True)
class MinTime:
    """The fewest steps to reach a state under a box thrust limit, and why not fewer.

    `steps` is the first feasible horizon (None if none up to n_max), `plan` its commands
    (steps, 3) in m/s^2 and `residual` the miss of the plan on the exact CW steps (|.|_inf over
    m and m/s). `nu` separates the target from the reachable set one step shorter (None when
    `steps` is 1 or no certificate was found), and `gap` = nu . (x_f - phi^N x0) - rho(nu), its
    margin recomputed in closed form; a positive gap proves that horizon infeasible.
    """

    steps: int | None
    plan: np.ndarray | None
    residual: float | None
    nu: np.ndarray | None
    gap: float | None


def _horizon(
    phi: np.ndarray, gamma: np.ndarray, x0: np.ndarray, xf: np.ndarray, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """(M, b): x_N = phi^N x0 + M u reaches xf iff M u = b, u the stacked commands."""
    blocks = [np.linalg.matrix_power(phi, steps - 1 - k) @ gamma for k in range(steps)]
    return np.hstack(blocks), xf - np.linalg.matrix_power(phi, steps) @ x0


def _certificate(m: np.ndarray, b: np.ndarray, a_max: float) -> tuple[np.ndarray | None, float]:
    """The direction nu in [-1, 1]^n that best separates b from a_max-box image of m."""
    n, width = m.shape
    cost = np.r_[-b, a_max * np.ones(width)]
    a_ub = np.block([[m.T, -np.eye(width)], [-m.T, -np.eye(width)]])
    found = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.zeros(2 * width),
        bounds=[(-1.0, 1.0)] * n + [(0.0, None)] * width,
        method="highs",
    )
    if found.status != 0:
        return None, -math.inf
    nu = found.x[:n]
    rho = a_max * float(np.sum(np.abs(m.T @ nu)))
    gap = float(nu @ b) - rho
    # Recomputed from nu alone: only a margin above the rounding of its terms is a proof.
    if gap > _GAP_TOL * (abs(float(nu @ b)) + rho):
        return nu, gap
    return None, gap


def min_time_steps(phi, gamma, x0, xf, a_max: float, n_max: int, layout: str = "box") -> MinTime:
    """The fewest held steps from `x0` to `xf` under a per-axis thrust box.

    Parameters
    ----------
    phi, gamma : array_like, shapes (6, 6) and (6, 3)
        The exact one-step CW matrices, `cw_discrete(n, dt)`.
    x0, xf : array_like, shape (6,)
        Start and target, Hill frame, m and m/s.
    a_max : float
        Per-axis thrust limit, m/s^2, positive.
    n_max : int
        The longest horizon tried, at least 1.
    layout : str
        "box"; "ball" raises.

    Returns
    -------
    MinTime
        The first horizon whose linear program is feasible, its plan, and a separating
        direction for the horizon one shorter: x_f lies outside the reachable set R_{N-1} iff
        some nu has nu . (x_f - phi^{N-1} x0) above the support of R_{N-1} (Kochenderfer et al.,
        eq. 8.24 and 8.27).

    Raises
    ------
    ValueError
        For a ball limit (its certificate needs a cone program) or bad inputs.
    """
    if layout == "ball":
        raise ValueError("minimum time under a ball limit needs a cone program")
    if layout != "box":
        raise ValueError(f"unknown layout {layout!r}")
    phi = _array(phi, (6, 6), "phi")
    gamma = _array(gamma, (6, 3), "gamma")
    start = _array(x0, (6,), "x0")
    target = _array(xf, (6,), "xf")
    a_max = float(a_max)
    if not (math.isfinite(a_max) and a_max > 0.0):
        raise ValueError("a_max must be finite and positive")
    n_max = _count(n_max, "n_max", 1)
    for steps in range(1, n_max + 1):
        m, b = _horizon(phi, gamma, start, target, steps)
        found = linprog(
            np.zeros(3 * steps),
            A_eq=m,
            b_eq=b,
            bounds=[(-a_max, a_max)] * (3 * steps),
            method="highs",
        )
        if found.status == 0:
            plan = np.clip(found.x, -a_max, a_max)
            residual = float(np.max(np.abs(m @ plan - b)))
            if steps == 1:
                return MinTime(steps, plan.reshape(steps, 3), residual, None, None)
            m_short, b_short = _horizon(phi, gamma, start, target, steps - 1)
            nu, gap = _certificate(m_short, b_short, a_max)
            return MinTime(steps, plan.reshape(steps, 3), residual, nu, gap)
    nu, gap = _certificate(m, b, a_max)
    return MinTime(None, None, None, nu, gap)


def gravity_remainder_bound(mu: float, r: float, rho: float) -> float:
    """Bound on point-mass gravity minus its linearization at a separation `rho`, m/s^2.

    Parameters
    ----------
    mu : float
        Gravitational parameter, m^3/s^2, positive.
    r : float
        Distance of the reference point from the body's center, m, positive.
    rho : float
        Separation from the reference point, m, 0 <= rho < r.

    Returns
    -------
    float
        3 mu rho^2 / (r - rho)^4: the second derivative of -mu s / |s|^3 along any unit
        direction is at most 6 mu / |s|^4, its value along the radius, and |s| >= r - rho on
        the segment, so Taylor's theorem with remainder (Kochenderfer et al., sec. 9.3) bounds
        |g(r + rho) - g(r) - G rho| by half of it times rho^2.

    Raises
    ------
    ValueError
        If a value is not finite or rho is not in [0, r).
    """
    mu, r, rho = float(mu), float(r), float(rho)
    if not all(math.isfinite(v) for v in (mu, r, rho)) or mu <= 0.0 or r <= 0.0:
        raise ValueError("mu and r must be finite and positive")
    if not 0.0 <= rho < r:
        raise ValueError("rho must lie in [0, r)")
    return 3.0 * mu * rho * rho / (r - rho) ** 4


__all__ = [
    "MAX_FACETS",
    "Clearance",
    "CwLoop",
    "DriftCover",
    "LinearLoop",
    "MinTime",
    "Polytope",
    "Zonotope",
    "ball_distance",
    "contains",
    "cw_loop",
    "drift_clearance",
    "drift_cover",
    "gravity_remainder_bound",
    "invariant_step",
    "loop_clearance",
    "min_time_steps",
    "propagate",
    "support",
]
