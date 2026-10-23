"""Model predictive control on the zero-order-hold CW model: the terminal set.

The terminal set is the maximal output admissible set of a stable closed loop, computed by the
algorithm of Gilbert and Tan, "Linear systems with state and control constraints: the theory and
application of maximal output admissible sets", IEEE Transactions on Automatic Control 36(9),
1991, as used for terminal sets in Rawlings, Mayne and Diehl, Model Predictive Control, 2nd ed.,
section 2.5.4 and exercise 2.5.
"""

from __future__ import annotations

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
        or t* exceeds t_max.
    RuntimeError
        If a redundancy LP ends without an optimum or a proof of unboundedness.
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

    # Rows over their bounds, then the state rescaled so the largest entry is 1: every LP has
    # right-hand side 1 and entries of one size whatever the units of h and bound.
    g = h / bound[:, None]
    g = g / np.max(np.abs(g))
    rows = [h]
    scaled = [g]
    for t in range(t_max + 1):
        a_ub = np.vstack(scaled)
        a_ub = np.vstack([a_ub, -a_ub])
        candidates = scaled[-1] @ f
        if all(_is_redundant(row, a_ub) for row in candidates):
            return np.vstack(rows), t
        rows.append(rows[-1] @ f)
        scaled.append(candidates)
    raise ValueError("terminal set is not finitely determined within t_max steps")


def _is_redundant(row: np.ndarray, a_ub: np.ndarray) -> bool:
    """Whether |row . w| <= 1 holds over {w : a_ub w <= 1}, up to the relative margin."""
    # The set is symmetric about the origin, so the maximum of -row . w equals that of row . w
    # and one LP covers both signs.
    lp = linprog(
        -row,
        A_ub=a_ub,
        b_ub=np.ones(len(a_ub)),
        bounds=(None, None),
        method="highs",
    )
    if lp.status == 3:
        return False
    if lp.status != 0:
        raise RuntimeError(f"terminal set LP failed: {lp.message}")
    return bool(-lp.fun <= 1.0 + REDUNDANCY_TOL)
