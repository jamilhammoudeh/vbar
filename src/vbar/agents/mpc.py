"""Model predictive control on the zero-order-hold CW model: the terminal set.

The terminal set is the maximal output admissible set of a stable closed loop, computed by the
algorithm of Gilbert and Tan, "Linear systems with state and control constraints: the theory and
application of maximal output admissible sets", IEEE Transactions on Automatic Control 36(9),
1991, as used for terminal sets in Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed.,
section 2.5.4 and exercise 2.5.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linprog

# Within 1e-12 of the unit circle the determination index grows past any cap (2.3e12 steps per
# decade of decay), and a closed loop with a mode on the circle has no finite description.
STABILITY_TOL = 1e-12
# A row whose maximum over the set equals its bound comes back from the LP above the bound by
# rounding, which would add rows the set does not need. The margin is relative to the bound.
REDUNDANCY_TOL = 1e-9


def admissible_set(
    f: ArrayLike, h: ArrayLike, bound: ArrayLike, *, t_max: int = 200
) -> tuple[np.ndarray, int]:
    """Maximal admissible set of x+ = F x under the output rows |h_i x| <= bound_i.

    The set {x : |h_i F^t x| <= bound_i for all i and all t >= 0} is described by the rows
    h F^t for t = 0, ..., t*, where t* is the first t at which every row of h F^(t+1) is
    redundant over the rows so far (Gilbert and Tan, 1991, algorithm 3.2). Redundancy is
    checked by one linear program per candidate row, in coordinates where each row is divided
    by its bound, so the result does not depend on the units of the bound.

    Parameters
    ----------
    f : array_like, shape (n, n)
        Closed-loop transition matrix of one step, spectral radius below 1.
    h : array_like, shape (p, n)
        Output rows, in the units of `bound` per unit of state.
    bound : array_like, shape (p,)
        Positive bound on |h_i x| for each output row.
    t_max : int, optional
        Largest determination index accepted.

    Returns
    -------
    rows : numpy.ndarray, shape ((t* + 1) p, n)
        The rows h, h F, ..., h F^t* stacked by time; row j has the bound bound[j mod p].
    t_star : int
        Determination index t*.

    Raises
    ------
    ValueError
        If the shapes disagree, f or h is not finite, a bound is not positive and finite, a
        row of h is zero, t_max is negative, the spectral radius of F is 1 - 1e-12 or more,
        or t* exceeds t_max. A pair (F, h) unobservable only up to rounding is refused at the
        cap; write it in exactly unobservable coordinates instead.
    RuntimeError
        If a redundancy LP ends without an optimum or a proof of unboundedness, or the set
        cannot be certified: for each row of h F^(t*+1), multipliers on independent returned
        rows, solved in exact rational arithmetic, must bound its maximum over the set by its
        bound times 1 + 1e-9. A row with no exact multipliers lies outside the span of the
        returned rows, where the set is unbounded; it is kept and the iteration goes on, as
        after an unbounded LP.
    """
    f = np.asarray(f, dtype=float)
    h = np.asarray(h, dtype=float)
    bound = np.asarray(bound, dtype=float)
    if f.ndim != 2 or f.shape[0] != f.shape[1]:
        raise ValueError("f must be a square matrix")
    if h.ndim != 2 or h.shape[1] != f.shape[0]:
        raise ValueError("h must have one column per state of f")
    if bound.shape != (h.shape[0],):
        raise ValueError("bound must have one entry per row of h")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(h)):
        raise ValueError("f and h must be finite")
    if not np.all(np.isfinite(bound) & (bound > 0.0)):
        raise ValueError("bound must be positive and finite")
    if not np.all(np.any(h != 0.0, axis=1)):
        raise ValueError("h has a row of zeros")
    if t_max < 0:
        raise ValueError("t_max must be nonnegative")
    if not np.max(np.abs(np.linalg.eigvals(f))) < 1.0 - STABILITY_TOL:
        raise ValueError("terminal set needs a stable closed loop")

    rows = [h]
    scaled = [h / bound[:, None]]
    for t in range(t_max + 1):
        current = np.vstack(scaled)
        maximizers = _maximizers(scaled[-1] @ f, current)
        if maximizers is not None:
            stacked = np.vstack(rows)
            limits = np.tile(bound, t + 1)
            checks = list(zip(rows[-1] @ f, bound, maximizers, strict=True))
            if all(_certified(row, limit, stacked, limits, x) for row, limit, x in checks):
                return stacked, t
            if all(_multipliers(row, stacked, limits, x) is not None for row, _, x in checks):
                raise RuntimeError("terminal set could not be certified")
        rows.append(rows[-1] @ f)
        scaled.append(scaled[-1] @ f)
    raise ValueError("terminal set is not finitely determined within t_max steps")


def _maximizers(candidates: np.ndarray, rows: np.ndarray) -> list[np.ndarray] | None:
    """Maximizer over {x : |rows x| <= 1} of each candidate, or None once one is not redundant."""
    # Gains in SI units span many decades between position and velocity columns, and rows decay
    # as F^t. Columns are scaled to unit maximum and every row, candidate included, to unit
    # norm, with the right-hand sides scaled to match.
    column = np.max(np.abs(rows), axis=0)
    column[column == 0.0] = 1.0
    a = rows / column
    size = np.linalg.norm(a, axis=1)
    a = a[size > 0.0] / size[size > 0.0, None]
    b = 1.0 / size[size > 0.0]
    # The set is symmetric about the origin, so the maximum of -row . x equals that of row . x
    # and one LP covers both signs.
    a_ub = np.vstack([a, -a])
    b_ub = np.concatenate([b, b])
    maximizers = []
    for row in candidates:
        c = row / column
        c_size = np.linalg.norm(c)
        if c_size == 0.0:
            maximizers.append(np.zeros(len(row)))
            continue
        # HiGHS presolve on these nearly parallel rows has reported an unbounded program as
        # infeasible and stopped 5e-5 below the optimum of another, so it is off. The origin
        # is feasible, so a reported infeasibility is a numerical failure, retried by interior
        # point.
        lp = linprog(
            -c / c_size,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=(None, None),
            method="highs-ds",
            options={"presolve": False},
        )
        if lp.status == 2:
            lp = linprog(-c / c_size, A_ub=a_ub, b_ub=b_ub, bounds=(None, None), method="highs-ipm")
        if lp.status == 3:
            return None
        if lp.status != 0:
            raise RuntimeError(f"terminal set LP failed: {lp.message}")
        if -lp.fun * c_size > 1.0 + REDUNDANCY_TOL:
            return None
        maximizers.append(lp.x / column)
    return maximizers


def _certified(
    row: np.ndarray, limit: float, rows: np.ndarray, limits: np.ndarray, x: np.ndarray
) -> bool:
    """Whether |row . z| <= limit (1 + REDUNDANCY_TOL) is proved over {z : |rows_j z| <= limits_j}.

    Any multipliers lam with sum_j lam_j rows_j = row bound the maximum by sum_j |lam_j|
    limits_j. False when the row has no exact multipliers on the rows.
    """
    found = _multipliers(row, rows, limits, x)
    if found is None:
        return False
    basis, lam = found
    proved = sum(
        (abs(lam_j) * Fraction(float(limits[j])) for lam_j, j in zip(lam, basis, strict=True)),
        Fraction(0),
    )
    return proved <= Fraction(float(limit)) * (1 + Fraction(REDUNDANCY_TOL))


def _multipliers(
    row: np.ndarray, rows: np.ndarray, limits: np.ndarray, x: np.ndarray
) -> tuple[list[int], list[Fraction]] | None:
    """Exact multipliers expressing `row` on independent rows; None off the rows' span.

    The rows are taken by their activity at the LP maximizer `x` up to a maximal independent
    set, and the multipliers solved in rational arithmetic on the float entries, so a bound
    built on them does not rest on the LP's tolerances.
    """
    activity = np.abs(rows @ x) / limits
    # Rank is judged with unit columns, since position gains can sit 1e-9 below velocity gains;
    # scaling columns leaves the multipliers unchanged.
    column = np.max(np.abs(rows), axis=0)
    column[column == 0.0] = 1.0
    unit = rows / column
    basis: list[int] = []
    for j in np.argsort(-activity, kind="stable"):
        if np.linalg.matrix_rank(unit[[*basis, j]]) > len(basis):
            basis.append(int(j))
            if len(basis) == rows.shape[1]:
                break
    lam = _exact_solve(rows[basis].T, row)
    return None if lam is None else (basis, lam)


def _exact_solve(m: np.ndarray, v: np.ndarray) -> list[Fraction] | None:
    """Exact solution y of m y = v, m of full column rank, in rational arithmetic.

    None when the system has no exact solution or m is rank deficient in exact arithmetic.
    """
    unknowns = m.shape[1]
    aug = [
        [Fraction(float(e)) for e in m_row] + [Fraction(float(v_i))]
        for m_row, v_i in zip(m, v, strict=True)
    ]
    for k in range(unknowns):
        pivot = next((i for i in range(k, len(aug)) if aug[i][k] != 0), None)
        if pivot is None:
            return None
        aug[k], aug[pivot] = aug[pivot], aug[k]
        for i in range(len(aug)):
            if i != k and aug[i][k] != 0:
                ratio = aug[i][k] / aug[k][k]
                aug[i] = [e - ratio * e_k for e, e_k in zip(aug[i], aug[k], strict=True)]
    # Equations past the pivots have no coefficients left; a nonzero right side is a residual.
    if any(aug[i][unknowns] != 0 for i in range(unknowns, len(aug))):
        return None
    return [aug[k][unknowns] / aug[k][k] for k in range(unknowns)]
