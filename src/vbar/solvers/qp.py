"""Exact projection of a three-axis command onto linear rows under a thrust limit.

The problem is min 1/2 (u - u_nom)^T H (u - u_nom) subject to rows a_i . u >= b_i and a box or
ball limit on u, with H diagonal and positive. In three dimensions at most three linearly
independent rows are active at the solution, so every subset of at most three rows is tried: each
gives an equality-constrained QP whose KKT system is solved directly, and the subset whose point
satisfies every row with nonnegative multipliers is the solution (Boyd and Vandenberghe, *Convex
Optimization*, sec. 5.5.3, eq. 5.49). With linear rows the KKT conditions are necessary and
sufficient whenever the rows are feasible, so a subset always wins then. Box faces are rows. The
ball |u| <= a_max is dualized: for a fixed multiplier mu the problem is the rows-only QP with
H + mu I, and |u(mu)| is nonincreasing in mu, so mu is found by Brent's method, a bisection
safeguarded by secant steps, on a_max / |u(mu)| - 1.

When no subset wins, the sum-of-infeasibilities phase I LP (sec. 11.4.1, eq. 11.20) gives the
least-violation command and a dual vector lam >= 0 that is checked in numpy as a Farkas
certificate (sec. 5.8.3) before infeasibility is reported.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, linprog

MAX_ROWS = 32
ACTUATOR_LAYOUTS = ("box", "ball")
PROJECTION_STATUSES = ("feasible", "infeasible_certified", "solver_failed")
BOX_FACES = ("box+x", "box-x", "box+y", "box-y", "box+z", "box-z")
# Face "box+x" is the row -u_x >= -a_max, "box-x" the row u_x >= -a_max, and so on.
_FACE_NORMALS = np.array(
    [[-1.0, 0, 0], [1.0, 0, 0], [0, -1.0, 0], [0, 1.0, 0], [0, 0, -1.0], [0, 0, 1.0]]
)
_FACE_NORMALS.flags.writeable = False

# Candidates are accepted when rows and multipliers hold to this fraction of the problem's
# scale; rounding in the 6x6 solves is near 1e-15 of it.
_TOL = 1e-10
# Unit row normals are independent when the smallest singular value of the subset exceeds this.
_RANK_TOL = 1e-12
# The ball radius is matched to this fraction of a_max.
_RADIUS_TOL = 1e-12
_MAX_ITERATIONS = 200
# The ball multiplier is located to this fraction of the smallest weight, or 4 ulp of itself;
# u then moves by about the same fraction of its size on every axis.
_MU_TOL = 1e-14
# Without an actuator set, a Farkas certificate's A^T lam must vanish to this fraction of |lam|_1.
_FARKAS_TOL = 1e-9


@dataclass(frozen=True)
class Rows:
    """Linear rows a_i . u >= b_i on a three-axis command.

    Parameters
    ----------
    a : array_like, shape (k, 3)
        Row normals, in row units per command unit. A zero row holds when b_i <= 0 and
        conflicts on its own otherwise.
    b : array_like, shape (k,)
        Right-hand sides, in row units.
    names : tuple of str, length k
        Row names for reports, unique and different from the actuator names.

    Raises
    ------
    ValueError
        If the shapes disagree, a value is not finite, a name repeats or is an actuator name, or
        there are more than 32 rows.
    """

    a: np.ndarray
    b: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self):
        a = np.array(self.a, dtype=float)
        b = np.array(self.b, dtype=float)
        names = tuple(self.names)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError("a must have shape (k, 3)")
        k = a.shape[0]
        if k > MAX_ROWS:
            raise ValueError(f"at most {MAX_ROWS} rows, got {k}")
        if b.shape != (k,):
            raise ValueError("b must have shape (k,)")
        if len(names) != k or not all(isinstance(name, str) for name in names):
            raise ValueError("names must be k strings")
        if len(set(names)) != k or set(names) & {*BOX_FACES, "ball"}:
            raise ValueError("row names must be unique and not actuator names")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("rows must be finite")
        a.flags.writeable = False
        b.flags.writeable = False
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "names", names)


@dataclass(frozen=True)
class ActuatorSet:
    """The thrust limit: |u_i| <= a_max per axis ("box") or |u|_2 <= a_max ("ball").

    Parameters
    ----------
    a_max : float
        The limit in command units, finite and positive.
    layout : str
        "box" or "ball".

    Raises
    ------
    ValueError
        If `a_max` is not finite and positive or `layout` is unknown.
    """

    a_max: float
    layout: str

    def __post_init__(self):
        a_max = float(self.a_max)
        if not (math.isfinite(a_max) and a_max > 0.0):
            raise ValueError("a_max must be finite and positive")
        if self.layout not in ACTUATOR_LAYOUTS:
            raise ValueError(f"unknown layout {self.layout!r}, expected {ACTUATOR_LAYOUTS}")
        object.__setattr__(self, "a_max", a_max)


@dataclass(frozen=True)
class Projection:
    """The result of `project`.

    Attributes
    ----------
    u : ndarray, shape (3,)
        The projected command. When infeasible, the command nearest u_nom among those in the
        actuator set that violate no row more than the phase I point does.
    status : str
        "feasible", "infeasible_certified" (proven by `certificate`) or "solver_failed" (no
        answer and no proof).
    active : tuple of str
        Rows with a positive multiplier, then the active box faces ("box+x" to "box-z") or
        "ball".
    multipliers : ndarray, shape (k,)
        Row multipliers, >= 0, in objective units per row unit; zeros unless feasible. Inf on
        the active rows when the rows touch the ball at a single point, where finite multipliers
        do not exist in general.
    objective : float
        1/2 (u - u_nom)^T H (u - u_nom) at `u`.
    conflicting : tuple of str
        Rows, box faces and "ball" whose certificate entry exceeds 1e-9 of its largest; () unless
        "infeasible_certified".
    certificate : ndarray or None
        lam >= 0 over the rows, followed for a box by its six faces, with b^T lam above the
        largest lam^T A u over the actuator set. With a box (faces included) or no actuator set
        this is A^T lam = 0 and b^T lam > 0; with a ball, b^T lam > a_max |A^T lam|_2. None
        unless "infeasible_certified".
    phase1_margin : float
        When feasible, the largest row violation (b_i - a_i . u) / |a_i| at `u`, in command
        units, <= 0 up to rounding and -inf with no rows; otherwise the sum of those violations
        that are positive, the sum of infeasibilities at `u`.
    """

    u: np.ndarray
    status: str
    active: tuple[str, ...]
    multipliers: np.ndarray
    objective: float
    conflicting: tuple[str, ...]
    certificate: np.ndarray | None
    phase1_margin: float


@lru_cache(maxsize=MAX_ROWS + len(BOX_FACES) + 1)
def _subsets(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Every subset of at most three of m rows, padded with index m, and its membership mask."""
    combos = [c for size in range(4) for c in itertools.combinations(range(m), size)]
    idx = np.full((len(combos), 3), m, dtype=np.intp)
    used = np.zeros((len(combos), 3), dtype=bool)
    for j, combo in enumerate(combos):
        idx[j, : len(combo)] = combo
        used[j, : len(combo)] = True
    idx.flags.writeable = False
    used.flags.writeable = False
    return idx, used


class _NoCandidate(Exception):
    """No active set satisfied the optimality conditions."""


class _ActiveSets:
    """Candidate active sets of unit rows, with their independence decided once."""

    def __init__(self, a_hat: np.ndarray, b_hat: np.ndarray, scale: float):
        self.a_hat = a_hat
        self.b_hat = b_hat
        self.scale = scale
        self.idx, self.used = _subsets(len(a_hat))
        self.a_s = np.vstack([a_hat, np.zeros((1, 3))])[self.idx]
        singular = np.linalg.svd(self.a_s, compute_uv=False)
        self.independent = np.sum(singular > _RANK_TOL, axis=1) == np.sum(self.used, axis=1)
        self.b_s = np.where(self.used, np.append(b_hat, 0.0)[self.idx], 0.0)

    def tolerance(self, u: np.ndarray) -> float:
        return _TOL * max(self.scale, float(np.max(np.abs(u))))

    def solve(self, w: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """The optimum and its unit-row multipliers under weight diag(w), or None if none wins."""
        n = len(self.idx)
        kkt = np.zeros((n, 6, 6))
        kkt[:, [0, 1, 2], [0, 1, 2]] = w
        kkt[:, :3, 3:] = -np.swapaxes(self.a_s, 1, 2)
        kkt[:, 3:, :3] = self.a_s
        # Unused multiplier slots get lam_j = 0 from an identity row.
        kkt[:, [3, 4, 5], [3, 4, 5]] = ~self.used
        rhs = np.zeros((n, 6))
        rhs[:, :3] = w * u0
        rhs[:, 3:] = self.b_s
        kkt[~self.independent] = np.eye(6)
        rhs[~self.independent] = 0.0
        sol = _solve_each(kkt, rhs)
        u, lam = sol[:, :3], sol[:, 3:]
        # The solve's error grows with |u0|; projecting u back onto the active rows from its
        # small residual makes the active rows hold to rounding of |u| instead.
        gram = self.a_s @ np.swapaxes(self.a_s, 1, 2)
        gram[:, [0, 1, 2], [0, 1, 2]] += ~self.used
        gram[~self.independent] = np.eye(3)
        residual = np.einsum("nij,nj->ni", self.a_s, u) - self.b_s
        correction = _solve_each(gram, residual)
        u = u - np.einsum("nji,nj->ni", self.a_s, correction)
        tol_u = _TOL * np.maximum(self.scale, np.max(np.abs(u), axis=1))
        primal = np.all(u @ self.a_hat.T - self.b_hat >= -tol_u[:, None], axis=1)
        # Multipliers carry H (u - u0), whose rounding grows with |u0|; the rows' tolerance must
        # not, or a large nominal command would loosen them.
        tol_lam = np.maximum(tol_u, _TOL * float(np.max(np.abs(u0)))) * np.max(w)
        dual = np.all((lam >= -tol_lam[:, None]) | ~self.used, axis=1)
        ok = self.independent & primal & dual & np.all(np.isfinite(sol), axis=1)
        if not np.any(ok):
            return None
        objective = 0.5 * np.sum(w * (u - u0) ** 2, axis=1)
        j = int(np.argmin(np.where(ok, objective, np.inf)))
        lam_rows = np.zeros(len(self.a_hat))
        lam_rows[self.idx[j, self.used[j]]] = np.maximum(lam[j, self.used[j]], 0.0)
        return u[j], lam_rows


def _solve_each(m: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve the stacked systems m x = rhs, by least squares for one that is exactly singular.

    Elimination squares a subset's smallest singular value, so an independent subset of nearly
    parallel rows (a row 2e-12 off a box face normal, say) can round to an exactly singular
    matrix. Whether it does depends on the rounding, not on a threshold that a well-posed subset
    would not also cross, so only the system that fails is solved again. Its candidate is then
    judged by the optimality checks like any other.
    """
    try:
        return np.linalg.solve(m, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        x = np.empty_like(rhs)
        for k in range(len(m)):
            try:
                x[k] = np.linalg.solve(m[k], rhs[k])
            except np.linalg.LinAlgError:
                x[k] = np.linalg.lstsq(m[k], rhs[k], rcond=None)[0]
        return x


def _vector3(value, name: str) -> np.ndarray:
    vector = np.array(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be three finite numbers")
    return vector


def _frozen(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    x.flags.writeable = False
    return x


def _onto_actuator(u: np.ndarray, actuator: ActuatorSet | None) -> np.ndarray:
    if actuator is None:
        return u
    if actuator.layout == "box":
        return np.clip(u, -actuator.a_max, actuator.a_max)
    radius = float(np.linalg.norm(u))
    return u * (actuator.a_max / radius) if radius > actuator.a_max else u


def project(
    u_nom,
    rows: Rows,
    actuator: ActuatorSet | None = None,
    weight: np.ndarray | None = None,
) -> Projection:
    """Project a command onto linear rows under a thrust limit, exactly.

    Parameters
    ----------
    u_nom : array_like, shape (3,)
        The nominal command, in the frame and units of the rows and the actuator set.
    rows : Rows
        Rows a_i . u >= b_i, at most 32.
    actuator : ActuatorSet, optional
        Box or ball limit on u; none by default.
    weight : array_like, shape (3,), optional
        Diagonal of H, positive; the identity by default.

    Returns
    -------
    Projection
        The command, its status, active rows and multipliers, and the certificate when
        infeasible. Rows hold to 1e-10 of the largest of a_max, |b_i| / |a_i| and |u|_inf,
        multipliers to 1e-10 of that or |u_nom|_inf, so the result does not depend on the units
        and a large nominal command does not loosen the rows; the ball radius is met to 1e-12
        a_max.

    Raises
    ------
    ValueError
        If `u_nom` or `weight` is not three finite numbers or `weight` is not positive.
    """
    u_nom = _vector3(u_nom, "u_nom")
    w = np.ones(3) if weight is None else _vector3(weight, "weight")
    if not np.all(w > 0.0):
        raise ValueError("weight must be positive")
    norms = np.linalg.norm(rows.a, axis=1)
    unit = np.where(norms > 0.0, norms, 1.0)
    a_max = 0.0 if actuator is None else actuator.a_max
    scale = max(
        a_max,
        float(np.max(np.abs(rows.b / unit), initial=0.0)),
        float(np.finfo(float).tiny),
    )
    problem = _Problem(rows, unit, w, u_nom, actuator, scale)
    if actuator is not None and actuator.layout == "ball":
        return problem.ball(actuator.a_max)
    return problem.polyhedral()


class _Problem:
    """One call of `project`, with the rows scaled to unit normals."""

    def __init__(
        self,
        rows: Rows,
        unit: np.ndarray,
        w: np.ndarray,
        u_nom: np.ndarray,
        actuator: ActuatorSet | None,
        scale: float,
    ):
        self.names = rows.names
        self.unit = unit
        self.a_hat = rows.a / unit[:, None]
        self.b_hat = rows.b / unit
        self.w = w
        self.u_nom = u_nom
        self.actuator = actuator
        self.scale = scale
        self.k = len(unit)

    def _with_faces(self, b_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.actuator is None or self.actuator.layout != "box":
            return self.a_hat, b_hat
        faces = np.full(len(BOX_FACES), -self.actuator.a_max)
        return np.vstack([self.a_hat, _FACE_NORMALS]), np.append(b_hat, faces)

    def _objective(self, u: np.ndarray) -> float:
        return float(0.5 * np.sum(self.w * (u - self.u_nom) ** 2))

    def _feasible(
        self, u: np.ndarray, lam_hat: np.ndarray, actuator_active: tuple[str, ...], tol: float
    ) -> Projection:
        on = lam_hat[: self.k] > tol * np.max(self.w)
        margin = float(np.max(self.b_hat - self.a_hat @ u)) if self.k else -math.inf
        return Projection(
            u=_frozen(u),
            status="feasible",
            active=tuple(np.array(self.names, dtype=object)[on]) + actuator_active,
            multipliers=_frozen(np.where(on, lam_hat[: self.k] / self.unit, 0.0)),
            objective=self._objective(u),
            conflicting=(),
            certificate=None,
            phase1_margin=margin,
        )

    def _unsolved(
        self,
        u: np.ndarray,
        status: str,
        conflicting: tuple[str, ...] = (),
        certificate: np.ndarray | None = None,
    ) -> Projection:
        violation = np.maximum(self.b_hat - self.a_hat @ u, 0.0)
        return Projection(
            u=_frozen(u),
            status=status,
            active=(),
            multipliers=_frozen(np.zeros(self.k)),
            objective=self._objective(u),
            conflicting=conflicting,
            certificate=None if certificate is None else _frozen(certificate),
            phase1_margin=float(np.sum(violation)),
        )

    def polyhedral(self) -> Projection:
        sets = _ActiveSets(*self._with_faces(self.b_hat), self.scale)
        found = sets.solve(self.w, self.u_nom)
        if found is None:
            return self._phase_one()
        u, lam_hat = found
        tol = sets.tolerance(u)
        faces: tuple[str, ...] = ()
        if self.actuator is not None:
            on = lam_hat[self.k :] > tol * np.max(self.w)
            faces = tuple(np.array(BOX_FACES, dtype=object)[on])
        return self._feasible(u, lam_hat, faces, tol)

    def ball(self, a_max: float) -> Projection:
        sets = _ActiveSets(self.a_hat, self.b_hat, self.scale)
        found = sets.solve(self.w, self.u_nom)
        if found is None:
            return self._phase_one()
        u, lam_hat = found
        if np.linalg.norm(u) <= a_max * (1.0 + _RADIUS_TOL):
            return self._feasible(u, lam_hat, (), sets.tolerance(u))
        nearest = sets.solve(np.ones(3), np.zeros(3))
        if nearest is None:
            return self._unsolved(_onto_actuator(self.u_nom, self.actuator), "solver_failed")
        u_min, lam_min = nearest
        r_min = float(np.linalg.norm(u_min))
        if r_min > a_max * (1.0 + _RADIUS_TOL):
            # u_min = A^T lam_min and b^T lam_min = |u_min|^2 by strong duality, so every u in
            # the ball has lam_min^T A u <= a_max |u_min| < b^T lam_min.
            return self._certify(lam_min, u_min * (a_max / r_min))
        if r_min >= a_max * (1.0 - _RADIUS_TOL):
            # The rows meet the ball at one point (Slater's condition fails).
            projection = self._feasible(u_min, lam_min, ("ball",), sets.tolerance(u_min))
            infinite = np.where(projection.multipliers > 0.0, math.inf, 0.0)
            return replace(projection, multipliers=_frozen(infinite))
        return self._ball_multiplier(sets, a_max, float(np.linalg.norm(u)))

    def _ball_multiplier(self, sets: _ActiveSets, a_max: float, radius: float) -> Projection:
        """The rows-only optimum under H + mu I whose norm is a_max, found by a search on mu."""
        solutions: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        def excess(mu: float) -> float:
            # a_max / |u(mu)| - 1 rises with mu, linearly when no row is active and H = h I.
            found = sets.solve(self.w + mu, self.w * self.u_nom / (self.w + mu))
            if found is None:
                raise _NoCandidate
            solutions[mu] = found
            return a_max / float(np.linalg.norm(found[0])) - 1.0

        w_max = float(np.max(self.w))
        lo, hi = 0.0, w_max * radius / a_max
        try:
            while excess(hi) < 0.0:
                lo, hi = hi, 2.0 * hi
                if not math.isfinite(hi):
                    raise _NoCandidate
            mu = brentq(
                excess, lo, hi, xtol=_MU_TOL * float(np.min(self.w)), maxiter=_MAX_ITERATIONS
            )
            excess(mu)
        except (_NoCandidate, RuntimeError):
            return self._unsolved(_onto_actuator(self.u_nom, self.actuator), "solver_failed")
        u, lam_hat = solutions[mu]
        if abs(np.linalg.norm(u) - a_max) > _RADIUS_TOL * a_max:
            return self._unsolved(_onto_actuator(u, self.actuator), "solver_failed")
        return self._feasible(u, lam_hat, ("ball",), sets.tolerance(u))

    def _phase_one(self) -> Projection:
        fallback = _onto_actuator(self.u_nom, self.actuator)
        if self.k == 0:
            return self._unsolved(fallback, "solver_failed")
        # A ball is replaced by its bounding box here, which keeps any certificate valid for it.
        bound = None if self.actuator is None else self.actuator.a_max
        res = linprog(
            np.concatenate([np.zeros(3), np.ones(self.k)]),
            A_ub=np.hstack([-self.a_hat, -np.eye(self.k)]),
            b_ub=-self.b_hat,
            bounds=[(None if bound is None else -bound, bound)] * 3 + [(0.0, None)] * self.k,
            method="highs",
        )
        if res.status != 0:
            return self._unsolved(fallback, "solver_failed")
        u_lp = _onto_actuator(np.asarray(res.x[:3], dtype=float), self.actuator)
        lam_hat = np.maximum(-np.asarray(res.ineqlin.marginals, dtype=float), 0.0)
        return self._certify(lam_hat, self._least_violation(u_lp))

    def _least_violation(self, u_lp: np.ndarray) -> np.ndarray:
        """The command nearest u_nom that violates no row by more than `u_lp` does."""
        relaxed = np.minimum(self.b_hat, self.a_hat @ u_lp)
        found = _ActiveSets(*self._with_faces(relaxed), self.scale).solve(self.w, self.u_nom)
        if found is None:
            return u_lp
        return _onto_actuator(found[0], self.actuator)

    def _certify(self, lam_hat: np.ndarray, u: np.ndarray) -> Projection:
        """`u` as infeasible_certified if `lam_hat` verifies as a certificate, else failed."""
        total = float(np.sum(lam_hat))
        if not total > 0.0:
            return self._unsolved(u, "solver_failed")
        w_vec = self.a_hat.T @ lam_hat
        if self.actuator is None:
            if np.max(np.abs(w_vec)) > _FARKAS_TOL * total:
                return self._unsolved(u, "solver_failed")
            support = 0.0
        elif self.actuator.layout == "box":
            support = self.actuator.a_max * float(np.sum(np.abs(w_vec)))
        else:
            support = self.actuator.a_max * float(np.linalg.norm(w_vec))
        if not float(self.b_hat @ lam_hat) - support > _TOL * self.scale * total:
            return self._unsolved(u, "solver_failed")
        # The support of the certificate; positive phase I slacks depend on the LP vertex chosen.
        threshold = _FARKAS_TOL * float(np.max(lam_hat))
        names = tuple(np.array(self.names, dtype=object)[lam_hat > threshold])
        certificate = lam_hat / self.unit
        if self.actuator is not None and self.actuator.layout == "box":
            faces = np.stack([np.maximum(w_vec, 0.0), np.maximum(-w_vec, 0.0)], axis=1).ravel()
            names += tuple(np.array(BOX_FACES, dtype=object)[faces > threshold])
            certificate = np.concatenate([certificate, faces])
        elif self.actuator is not None and np.linalg.norm(w_vec) > threshold:
            names += ("ball",)
        return self._unsolved(u, "infeasible_certified", names, certificate)
