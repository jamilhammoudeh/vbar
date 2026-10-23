"""Exact constrained-LQ value on the zero-order-hold CW model, and its Riccati lower bound.

With z = x - xs the deviation from a hold point at rest, u = a - us the deviation from its
holding input and the stage cost z' Q z + u' R u + 2 z' S u, every step-held input history costs
at least z0' P z0, P the stabilizing solution of the discrete Riccati equation. The box-constrained
optimum V_N over N steps with terminal cost z_N' P z_N equals the constrained infinite-horizon
optimum whenever its predicted terminal state lies in the maximal admissible set of the LQR loop,
from which the LQR never meets the box (Rawlings, Mayne and Diehl, Model Predictive Control, 2nd
ed., section 2.5.4, proposition 2.18 and the notes of chapter 2; the steady-state target of
section 1.5.1; Gilbert and Tan, 1991).
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve

from ..agents.lqr import ALIGNMENT_TOL, COST_EIGENVALUE_TOL, dlqr, steady_state_input
from ..agents.mpc import admissible_set
from ..dynamics import cw_discrete
from ..rewards import QuadraticCost

HORIZONS = (5, 10, 20, 40, 80, 120)
# The active-set solve ends on the exact minimizer of its last free set, so the gradient sits at
# rounding, below 1e-16 of its terms for horizons of 5 to 120 steps; a solve stopped short of the
# optimum is far above this.
KKT_TOL = 1e-9


@dataclass(frozen=True)
class LQBound:
    """Riccati lower bound and exact constrained value of the LQ cost from one start.

    Parameters
    ----------
    lower : float
        z0' P z0, a lower bound on the cost of every input history held over each step, within
        the thrust box or not (units of the stage cost).
    value : float
        Least cost of an input history held over each step within the thrust box; NaN when no
        horizon tried ended in the admissible set (`valid` False).
    valid : bool
        Whether `value` is the exact constrained optimum.
    horizon : int or None
        The first horizon (steps) whose predicted terminal state lay in the admissible set.
    rung : str
        `Episode.dynamics` name of the model both numbers are exact for.
    """

    lower: float
    value: float
    valid: bool
    horizon: int | None
    rung: str = "cw"


def _factor(w: np.ndarray) -> np.ndarray:
    """L with L L' = W for a symmetric positive semidefinite W, from eigh in Jacobi scaling."""
    diagonal = np.diag(w)
    scale = np.ones(len(w))
    scale[diagonal > 0.0] = 1.0 / np.sqrt(diagonal[diagonal > 0.0])
    scaled = w * scale[:, None] * scale[None, :]
    lam, vectors = np.linalg.eigh(0.5 * (scaled + scaled.T))
    if lam[0] < -COST_EIGENVALUE_TOL * lam[-1]:
        raise ValueError("weights are not positive semidefinite")
    return vectors * np.sqrt(np.clip(lam, 0.0, None)) / scale[:, None]


class LQOracle:
    """Exact constrained infinite-horizon LQ cost to a hold point on the CW model.

    The stage cost z' Q z + u' R u + 2 z' S u is charged on the deviation z = x - xs from the
    hold point xs = (hold, 0) and on u = a - us, us the input that holds xs; the thrust box
    |a_i| <= a_max becomes -a_max - us_i <= u_i <= a_max - us_i. For each horizon N in turn the
    box-constrained optimum V_N with terminal cost z_N' P z_N is solved by a primal active-set
    method on the condensed inputs. It is accepted when it meets the KKT conditions to 1e-9 of
    the gradient's terms and its predicted z_N lies in the admissible set of the LQR loop; V_N
    is then the exact value. That set is the one of the symmetric box |u_i| <= a_max - |us_i|,
    which lies inside the shifted box, so a terminal state it accepts lies in the admissible set
    of the shifted box too.

    Parameters
    ----------
    n : float
        Mean motion of the target orbit (rad/s).
    dt : float
        Step (s); inputs are held from each step start to the next.
    q : array_like, shape (6, 6)
        State weight on z (Hill frame, per m^2 and per (m/s)^2), symmetric positive semidefinite.
    r : array_like, shape (3, 3)
        Input weight on u (per (m/s^2)^2), symmetric positive definite.
    s : array_like, shape (6, 3), optional
        Cross weight; zero when omitted.
    a_max : float
        Thrust limit per Hill axis (m/s^2).
    layout : {"box"}
        Thrust limit layout; only the per-axis box is supported.
    hold : array_like, shape (3,)
        Hold point (m, Hill frame), at rest.
    horizons : sequence of int
        Horizons (steps) tried in order.
    delay : float
        Actuation delay (s), a whole number of steps. The inputs as applied stay held over each
        step, so the bound and the value are those of the applied inputs.

    Raises
    ------
    ValueError
        If a number is not finite or not positive where required, a weight has the wrong shape
        or is not positive semidefinite, the layout is not "box", the delay is not a whole
        number of steps, a horizon is below 1, the hold point needs at least `a_max` of steady
        thrust on an axis, or the LQR design fails (see `dlqr` and `admissible_set`).
    """

    def __init__(
        self,
        *,
        n: float,
        dt: float,
        q: ArrayLike,
        r: ArrayLike,
        s: ArrayLike | None = None,
        a_max: float,
        layout: str = "box",
        hold: ArrayLike = (0.0, 0.0, 0.0),
        horizons: tuple[int, ...] = HORIZONS,
        delay: float = 0.0,
    ):
        if layout != "box":
            raise ValueError(f"the LQ oracle supports box thrusters only, not {layout!r}")
        for name, number in (("n", n), ("dt", dt), ("a_max", a_max)):
            if not (math.isfinite(number) and number > 0.0):
                raise ValueError(f"{name} must be positive and finite")
        if not (math.isfinite(delay) and delay >= 0.0):
            raise ValueError("delay must be nonnegative and finite")
        if abs(delay / dt - round(delay / dt)) > ALIGNMENT_TOL:
            raise ValueError("delay must be a whole number of steps")
        self.horizons = tuple(operator.index(steps) for steps in horizons)
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be one or more positive step counts")
        q = np.asarray(q, dtype=float)
        r = np.asarray(r, dtype=float)
        s = np.zeros((6, 3)) if s is None else np.asarray(s, dtype=float)
        if q.shape != (6, 6) or r.shape != (3, 3) or s.shape != (6, 3):
            raise ValueError("q must be 6x6, r 3x3 and s 6x3")
        if not all(np.all(np.isfinite(w)) for w in (q, r, s)):
            raise ValueError("weights must be finite")
        hold = np.asarray(hold, dtype=float)
        if hold.shape != (3,) or not np.all(np.isfinite(hold)):
            raise ValueError("hold must be 3 finite numbers")

        phi, gamma = cw_discrete(n, dt)
        self._k, self._p = dlqr(phi, gamma, q, r, s)
        self._xs = np.concatenate([hold, np.zeros(3)])
        us = steady_state_input(phi, gamma, self._xs)
        headroom = a_max - np.abs(us)
        if not np.all(headroom > 0.0):
            raise ValueError(
                f"hold point needs {np.max(np.abs(us)):.2e} m/s^2 of steady thrust, at or above "
                f"a_max {a_max}"
            )
        self._rows, _ = admissible_set(phi - gamma @ self._k, self._k, headroom)
        self._limits = np.tile(headroom, len(self._rows) // 3)
        self._lo = (-a_max - us) / a_max
        self._hi = (a_max - us) / a_max

        # Residual rows of the least-squares form: L_x' z_i + L_u' u_i per stage and L_P' z_N at
        # the end, with W = L L'. The value and the KKT check are taken from the residual, a sum
        # of squares, rather than from the expanded quadratic, which cancels.
        stage = _factor(np.block([[q, s], [s.T, r]]))
        self._terminal = _factor(self._p).T
        steps = max(self.horizons)
        # Closed-form transitions over i steps, so long horizons accumulate no product rounding.
        self._powers = cw_discrete(n, dt * np.arange(steps + 1))[0]
        # z_i = Phi^i z0 + forced[i] u, with u stacked by step and scaled by a_max.
        self._forced = np.zeros((steps + 1, 6, 3 * steps))
        response = a_max * self._powers[:steps] @ gamma
        for i in range(1, steps + 1):
            self._forced[i, :, : 3 * i] = np.concatenate(response[i - 1 :: -1], axis=1)
        self._stage = np.einsum("xk,ixc->ikc", stage[:6], self._forced[:steps])
        for i in range(steps):
            self._stage[i, :, 3 * i : 3 * i + 3] += a_max * stage[6:].T
        self._stage_free = np.einsum("xk,ixy->iky", stage[:6], self._powers[:steps])

    @classmethod
    def from_cost(
        cls, cost: QuadraticCost, *, dt: float, a_max: float, layout: str = "box"
    ) -> LQOracle:
        """Oracle for the exact step cost of a `QuadraticCost`, held at its reference.

        Q, R and S are the blocks of `cost.matrix(dt, None)`, the cost rate integrated over one
        step with the acceleration held. The cost is charged on a, not on a - us, so its sum is
        finite only at a reference that needs no steady thrust: a point at rest on the V-bar.

        Parameters
        ----------
        cost : QuadraticCost
            Undiscounted cost rate; its reference is the hold point.
        dt : float
            Step (s).
        a_max : float
            Thrust limit per Hill axis (m/s^2).
        layout : {"box"}
            Thrust limit layout.

        Returns
        -------
        LQOracle
            The oracle with the default horizons and no delay.

        Raises
        ------
        ValueError
            If the reference moves, needs steady thrust to hold, or the oracle refuses the
            weights, step, limit or layout.
        """
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be positive and finite")
        reference = cost.reference
        if np.any(reference[3:] != 0.0):
            raise ValueError("the reference must be a point at rest")
        phi, gamma = cw_discrete(cost.n, dt)
        if np.any(steady_state_input(phi, gamma, reference) != 0.0):
            raise ValueError(
                "a cost on the acceleration has no finite sum at a hold that needs steady thrust"
            )
        w = cost.matrix(dt, None)
        return cls(
            n=cost.n,
            dt=dt,
            q=w[:6, :6],
            r=w[6:9, 6:9],
            s=w[:6, 6:9],
            a_max=a_max,
            layout=layout,
            hold=reference[:3],
        )

    def __call__(self, x0: ArrayLike) -> LQBound:
        """Lower bound and, when the admissible-set check passes, the exact value from x0.

        Parameters
        ----------
        x0 : array_like, shape (6,)
            Start state (m, m/s, Hill frame).

        Returns
        -------
        LQBound
            The bound, the value at the first horizon that passes, and the validity flag.

        Raises
        ------
        ValueError
            If x0 is not 6 finite numbers.
        """
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (6,) or not np.all(np.isfinite(x0)):
            raise ValueError("x0 must be 6 finite numbers")
        z0 = x0 - self._xs
        lower = float(z0 @ self._p @ z0)
        for steps in self.horizons:
            solved = self._solve(z0, steps)
            if solved is None:
                continue
            value, z_n = solved
            if np.max(np.abs(self._rows @ z_n) / self._limits) <= 1.0:
                return LQBound(lower=lower, value=value, valid=True, horizon=steps)
        return LQBound(lower=lower, value=math.nan, valid=False, horizon=None)

    def _solve(self, z0: np.ndarray, steps: int) -> tuple[float, np.ndarray] | None:
        """V_N and the predicted z_N, or None when no solution meets the KKT conditions."""
        columns = 3 * steps
        rows = np.concatenate(
            [
                self._stage[:steps, :, :columns].reshape(-1, columns),
                self._terminal @ self._forced[steps, :, :columns],
            ]
        )
        target = -np.concatenate(
            [(self._stage_free[:steps] @ z0).ravel(), self._terminal @ self._powers[steps] @ z0]
        )
        lo, hi = np.tile(self._lo, steps), np.tile(self._hi, steps)
        # Normwise size of the gradient's terms over the box: the solve spreads rounding over the
        # coupled inputs, so an input that should be exactly zero carries a gradient of the size
        # of the others' rounding.
        reach = np.maximum(np.abs(lo), np.abs(hi))
        gradient_size = float(np.max(np.abs(rows).T @ (np.abs(rows) @ reach + np.abs(target))))
        slack = KKT_TOL * gradient_size
        w = _box_qp(rows.T @ rows, -rows.T @ target, lo, hi, slack)
        if w is None:
            return None
        residual = rows @ w - target
        gradient = rows.T @ residual
        free = (w > lo) & (w < hi)
        if not (
            np.all(np.abs(gradient[free]) <= slack)
            and np.all(gradient[w == lo] >= -slack)
            and np.all(gradient[w == hi] <= slack)
        ):
            return None
        z_n = self._powers[steps] @ z0 + self._forced[steps, :, :columns] @ w
        return float(residual @ residual), z_n


def _box_qp(
    h: np.ndarray, f: np.ndarray, lo: np.ndarray, hi: np.ndarray, slack: float
) -> np.ndarray | None:
    """Minimizer of w' H w / 2 + f' w over lo <= w <= hi, H positive definite.

    Primal active-set method: from the clipped unconstrained minimizer, each round solves for
    the free inputs with the others on their bounds, steps as far toward that point as the box
    allows, and frees the bounded input whose gradient points most into the box by more than
    `slack`. None when the round limit is reached.
    """
    w = np.clip(cho_solve(cho_factor(h), -f), lo, hi)
    bounded = (w == lo) | (w == hi)
    for _ in range(3 * len(w) + 10):
        free = ~bounded
        goal = w.copy()
        if np.any(free):
            pull = f[free] + h[np.ix_(free, bounded)] @ w[bounded]
            goal[free] = cho_solve(cho_factor(h[np.ix_(free, free)]), -pull)
        step = goal - w
        reach = np.full(len(w), np.inf)
        down, up = free & (step < 0.0), free & (step > 0.0)
        reach[down] = (lo[down] - w[down]) / step[down]
        reach[up] = (hi[up] - w[up]) / step[up]
        blocking = int(np.argmin(reach))
        if reach[blocking] < 1.0:
            # The clip keeps inputs that reach their bound in the same step inside the box.
            w = np.clip(w + reach[blocking] * step, lo, hi)
            w[blocking] = lo[blocking] if step[blocking] < 0.0 else hi[blocking]
            bounded[blocking] = True
            continue
        w = goal
        gradient = h @ w + f
        pull_in = np.zeros(len(w))
        pull_in[bounded & (w == lo)] = -gradient[bounded & (w == lo)]
        pull_in[bounded & (w == hi)] = gradient[bounded & (w == hi)]
        worst = int(np.argmax(pull_in))
        if pull_in[worst] <= slack:
            return w
        bounded[worst] = False
    return None
