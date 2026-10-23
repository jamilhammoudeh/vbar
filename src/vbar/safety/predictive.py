"""Predictive safety filter: horizon plans applied only once certified on the exact held steps.

Each decision proposes a plan of `horizon` held commands by a quadratic program linearized around
the previous plan, then checks the plan on the exact Clohessy-Wiltshire solution: every thrusting
step by `held_segment_bound`, and the free drift after the last thrust by the terminal free-drift
barrier. The program only proposes; a plan that fails the check is never applied.

The filter follows the predictive safety filter of Wabersich and Zeilinger, "Linear model
predictive safety certification for learning-based control", CDC 2018, and "A predictive safety
filter for learning-based control of constrained nonlinear dynamical systems", Automatica 129,
2021: a plan whose first input stays close to the learning input, with the stored plan replayed
when the program has no solution, and recursive feasibility only with an invariant terminal set.
Their cost is a norm of the first input's deviation; here it is squared, so the program is a QP.
The terminal set is the free-drift set of `PassiveBarrier(terminal=True)`, invariant under zero
thrust. The check of each plan after the solve is this module's own; in their filter,
feasibility of the program is the certificate.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..agents.base import AgentView, command_fraction
from ..constraints import KeepOut
from ..dynamics.cw import CWDiscretizer, cw_discrete
from ..solvers.ipm import IPMSettings, solve_horizon
from ..solvers.stage import StageQP
from . import FilterResult
from .barriers import PassiveBarrier
from .cbf import (
    _HELD_SAMPLES,
    FilterDeclaration,
    _HeldCommandFilter,
    _nonnegative,
    _positive,
    _state,
)
from .sampled import held_segment_bound

# The drift grid of BackupFilter's free-drift barrier, so the two declare the same set.
_GRID = 5.0
# Margin of the far-part terminal rows (m): it keeps the plan's end state receding, where a
# centre standing still on its threshold would let rounding decide membership.
_FAR_MARGIN = KeepOut.resolution
# Fraction of the input box's half-width the warm start is held within.
_WARM_INSIDE = 0.99
# Free-drift invariants as rows on (x, y, z, vx / n, vy / n, vz / n): the centre offsets
# x_c = 4 x + 2 vy / n and y_c = y - 2 vx / n, and the in-plane ellipse vector (p, q) with
# p = 3 x + 2 vy / n and q = vx / n, as in `PassiveBarrier`.
_INVARIANTS = np.array(
    [
        [4.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0, -2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ]
)


class PredictiveFilter(_HeldCommandFilter):
    """Horizon plans toward the agent's command, applied only when certified for all time.

    Parameters
    ----------
    horizon : int
        Number of held steps N in a plan, at least 1.
    row_samples : int
        Samples per step, endpoints included, at which the program bounds the range from the
        target; at least 2.
    eps : float
        Weight of the plan's later inputs against the deviation of the first, positive.
    settings : IPMSettings, optional
        Tolerances and iteration budget of `solve_horizon`; its defaults when None.
    delay_s : float
        Actuation delay, s, finite and nonnegative; the view must echo exactly
        ceil(delay_s / dt) pending commands.
    ignore : sequence of str
        Constraint types of the scenario not to enforce; any other type than keep_out makes
        reset raise, since the free drift may leave it.

    Raises
    ------
    ValueError
        If a parameter is out of range.

    Notes
    -----
    The program over the Hill state (m, m/s) and the Hill-frame acceleration (m/s^2) minimizes
    |u_0 - u_c|^2 + eps sum_k |u_k|^2, u_c the clipped agent command, plus
    eps sum_k (|n^2 (r_k - r'_k)|^2 + |n (v_k - v'_k)|^2) on the deviation from the reference
    (r', v'), under the exact CW step, a box of half-width a_max on each input (a_max / sqrt 3
    inside a ball limit), keep-out rows and terminal rows. The state term moves the plan by a
    negligible amount; without it, rows on states that no cost reaches never settle as inactive
    in the solver's complementarity test. The reference is the previous plan shifted by one
    step, or zero thrust without one. The keep-out rows are its tangent planes
    n . r >= min(R + b, |r'|) at `row_samples` times in each step; b covers the dip of the path
    between samples, (v h)^2 / (8 R) + a h^2 / 8 with v and a bounds on the speed and
    acceleration near the reference. The minimum with |r'|, which keeps the reference feasible,
    is taken only for a certified reference: the shifted plan after a certified decision, or
    zero thrust with h_p >= 0; any other reference gets R + b. A reference that ends in the far
    part of the terminal set is kept there by its polyhedral inner bound: y_c receding by 1 mm
    and |y_c| - 2 (|p| + |q|) >= R + 1 mm, with x_c, y_c and the in-plane ellipse vector (p, q) of
    `PassiveBarrier`. Any other end state gets h_p linearized by the envelope gradient of
    `PassiveBarrier.gradient`, or, where the drift minimum has none (a closed relative orbit, or
    a minimum at either end of the drift horizon), by the gradient of the end range. The warm
    start is the reference held 1% inside the input box.

    A plan is certified when `held_segment_bound` over 20 samples is nonnegative on every step
    up to the last nonzero input, and h_p >= 0 after it. h_p uses the terminal set, which free
    drift keeps, so a certified plan keeps the keep-out for all time with zero thrust after it,
    and its shift by one step, with zero thrust appended, is certified from the next state.
    Each decision applies the first input of: the clipped agent command followed by zero
    thrust; the solution, when converged and certified (level 0); the stored plan shifted and
    re-certified, accepting a held-step shortfall of the keep-out resolution for rounding
    (level 1); zero thrust when h_p >= 0 (level 2). Without error sets the filter is therefore
    recursively feasible. The solver's own infeasibility certificate concerns the linearized
    program only, so the filter never reports "infeasible_certified". Zero thrust is certified
    wherever h_p >= 0, so a decision with nothing certified lies outside the declared set and is
    a recovering step; the filter never reports "solver_failed".

    With a delay, the commands already issued act first: the state delay_s after the
    observation is predicted through the echoed commands and the plan starts there.
    """

    _pending_allowed = True

    def __init__(
        self,
        *,
        horizon: int = 10,
        row_samples: int = 5,
        eps: float = 1e-3,
        settings: IPMSettings | None = None,
        delay_s: float = 0.0,
        ignore=(),
    ) -> None:
        super().__init__(execution=0.0, model_error=None, ignore=ignore)
        if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
            raise ValueError("horizon must be an integer of at least 1")
        if isinstance(row_samples, bool) or int(row_samples) != row_samples or row_samples < 2:
            raise ValueError("row_samples must be an integer of at least 2")
        if settings is not None and not isinstance(settings, IPMSettings):
            raise ValueError("settings must be an IPMSettings")
        self.horizon = int(horizon)
        self.row_samples = int(row_samples)
        self.eps = _positive(eps, "eps")
        self.settings = settings
        self.delay_s = _nonnegative(delay_s, "delay_s")
        self.name = f"predictive-n{self.horizon}"

    def reset(self, view: AgentView) -> None:
        """Check the view, its pending command slots, and build the terminal barrier.

        Raises
        ------
        ValueError
            If the action mode, frame, observation, dynamics or constraints are unsupported, the
            scenario has no keep_out, or the view does not echo ceil(delay_s / dt) pending
            commands.
        """
        self._view = None
        self._declaration = None
        parts = view.scenario_id.split(":")
        if (parts[2] if len(parts) > 2 else None) != "cw":
            raise ValueError("the predictive filter needs cw dynamics")
        self._reset_view(view)
        n, dt, radius = self._n, self._dt, self._radius
        slots = 0
        while f"pending_{slots}_t" in view.observation:
            slots += 1
        needed = math.ceil(self.delay_s / dt - 1e-9)
        if slots != needed:
            raise ValueError(
                f"a delay of {self.delay_s:g} s at dt {dt:g} s needs {needed} pending command "
                f"slots, the view has {slots}"
            )
        names = view.observation
        self._slots = np.array(
            [
                [names.index(f"pending_{k}_{axis}") for axis in ("t", "x", "y", "z")]
                for k in range(slots)
            ],
            dtype=int,
        ).reshape(slots, 4)
        self._cw = CWDiscretizer(n)
        self._invariants = _INVARIANTS / np.array([1.0, 1.0, 1.0, n, n, n])
        # The deviation from the reference as the accelerations n^2 dr and n dv, weighted like
        # the later inputs.
        self._state_weight = self.eps * np.diag([n**4] * 3 + [n**2] * 3)
        self._phi, self._gamma_u = cw_discrete(n, dt)
        # The drift samples after a step must be samples of the drift before it, so the grid
        # divides dt and the terminal set is kept by a drift step.
        self._barrier = PassiveBarrier(
            radius=radius,
            n=n,
            grid=dt / math.ceil(dt / _GRID),
            t_cap=20.0 * math.pi / n,
            terminal=True,
        )
        phis, gammas = cw_discrete(n, np.linspace(0.0, dt, self.row_samples))
        # Sample 0 of a step is the last of the step before, or the current state.
        self._row_phi, self._row_gamma = phis[1:], gammas[1:]
        self._row_h = dt / (self.row_samples - 1)
        layout, a_max = self._actuator.layout, self._actuator.a_max
        self._a_reach = a_max * (math.sqrt(3.0) if layout == "box" else 1.0)
        self._half = a_max if layout == "box" else a_max / math.sqrt(3.0)
        thrust = np.zeros((6, 9))
        for axis in range(3):
            thrust[2 * axis, 6 + axis] = 1.0
            thrust[2 * axis + 1, 6 + axis] = -1.0
        self._thrust_rows = thrust
        self._thrust_bound = np.full(6, -self._half)
        self._plan: np.ndarray | None = None
        self._certified = False
        self._view = view
        self._declaration = FilterDeclaration(
            name=self.name,
            safe_set="free drift certified outside the keep-out for all time",
            barriers=(self._barrier,),
            poles=None,
            gamma=None,
            design_dt=dt,
            layout=layout,
            a_max=a_max,
            buffer_m=0.0,
            tightening={"navigation": None, "execution": 0.0, "model_error": None},
            backup="zero thrust",
            certificate="nominal",
            counts_toward_certification=True,
            ignored=self.ignore,
        )

    def contains(self, x) -> bool:
        """Return whether the Hill-frame state `x` (m, m/s) lies in the terminal free-drift set.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        """
        self._started()
        return self._barrier.contains(np.asarray(x, dtype=float))

    def predict(self, obs) -> np.ndarray:
        """Return the Hill state delay_s after the observation, through the echoed commands.

        Parameters
        ----------
        obs : np.ndarray
            The observation; its first six entries are the Hill-frame state, m and m/s, and its
            pending command slots give each command's wait (s) and value in the action mode.

        Returns
        -------
        np.ndarray, shape (6,)
            The state, m and m/s, on the CW model with each echoed command held from its wait
            to the next one's, the last up to delay_s, as acceleration at the initial mass.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        ValueError
            If the state or the slots are not finite, or the waits do not rise within delay_s.
        """
        view = self._started()
        x = _state(obs)
        if len(self._slots) == 0:
            return x
        slots = np.asarray(obs, dtype=float)[self._slots]
        if not np.all(np.isfinite(slots)):
            raise ValueError("the pending command slots must be finite")
        waits = np.append(slots[:, 0], self.delay_s)
        if waits[0] < 0.0 or np.any(np.diff(waits) < 0.0):
            raise ValueError("pending commands must take effect in order within delay_s")
        commands = command_fraction(view, slots[:, 1:]) * view.max_accel
        for duration, u in zip(np.diff(waits), commands, strict=True):
            if duration > 0.0:
                phi, gamma = self._cw.matrices(float(duration))
                x = phi @ x + gamma @ u
        return x

    def filter(self, obs: np.ndarray, action: np.ndarray) -> FilterResult:
        """Return the first input of the first certified plan, toward the agent's command.

        Parameters
        ----------
        obs : np.ndarray
            The observation; its first six entries are the Hill-frame state, m and m/s.
        action : np.ndarray
            The agent's command in the view's action mode.

        Returns
        -------
        FilterResult
            A certified agent command inside the thrust limit comes back as the agent's own
            array at level 0 with `active` False. Otherwise `fallback_level` is 0 for the
            certified solution, 1 for the stored plan shifted, 2 for zero thrust, and
            `iterations` is the solver's count. A certified decision has `certificate`
            "nominal" with horizon inf. When nothing is certified, which happens only outside
            the declared set, the converged solution, else the shifted plan, else zero thrust is
            returned as "feasible" with `recovering` True, its predicted shortfall (m) as
            `slack` and no certificate.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        ValueError
            If the observation or the action is malformed.
        """
        t0 = time.perf_counter()
        view = self._started()
        x = self.predict(obs)
        u_c = self._clipped(view, action)
        # Clipping that changed nothing leaves the agent's array to pass through bit for bit.
        own = (
            np.asarray(action, dtype=float)
            if np.array_equal(u_c, command_fraction(view, action) * view.max_accel)
            else None
        )
        if self._monitor_only(x):
            self._plan, self._certified = None, False
            return FilterResult(
                self._action(u_c) if own is None else own,
                intervened=False,
                compute_s=time.perf_counter() - t0,
            )
        steps = self.horizon
        candidate = np.zeros((steps, 3))
        candidate[0] = u_c
        held, end = self._margins(x, candidate)
        if held >= 0.0 and end >= 0.0:
            return self._result(candidate, u_c, t0, level=0, iterations=None, own=own)

        drift = np.zeros((steps, 3))
        drift_kept: bool | None = None
        if self._plan is None:
            shifted = None
            reference = drift
            drift_kept = float(self._barrier.value(x)) >= 0.0
            certified_reference = drift_kept
        else:
            shifted = np.vstack([self._plan[1:], np.zeros((1, 3))])
            reference = shifted
            certified_reference = self._certified
        # The solver starts each row's slack at its value at the warm start, so inputs on a face
        # of the box would start that row's multiplier near mu / 0; 1% inside, they do not.
        warm = np.clip(reference, -_WARM_INSIDE * self._half, _WARM_INSIDE * self._half)
        solution = solve_horizon(
            self._problem(x, u_c, reference, certified=certified_reference),
            self.settings,
            warm=(self._rollout(x, warm), warm),
        )
        iterations = int(solution.iterations)
        solved = None
        if solution.status == "converged" and solution.u is not None:
            solved = np.array([self._onto_limit(u) for u in np.asarray(solution.u, dtype=float)])
            held, end = self._margins(x, solved)
            if held >= 0.0 and end >= 0.0:
                return self._result(solved, u_c, t0, level=0, iterations=iterations)
        if shifted is not None:
            held, end = self._margins(x, shifted)
            # The stored plan was certified from the predicted state; re-rolled from the state
            # now, it differs by rounding, which can cost the held bound up to the resolution.
            if held >= -KeepOut.resolution and end >= 0.0:
                return self._result(shifted, u_c, t0, level=1, iterations=iterations)
        if drift_kept is None:
            drift_kept = float(self._barrier.value(x)) >= 0.0
        if drift_kept:
            return self._result(drift, u_c, t0, level=2, iterations=iterations)

        if solved is not None:
            plan, level = solved, 0
        elif shifted is not None:
            plan, level = shifted, 1
        else:
            plan, level = drift, 2
        # Zero thrust is certified wherever h_p >= 0, so nothing certified means the state is
        # outside the declared set: a recovering step, never "solver_failed".
        held, end = self._margins(x, plan, stop=False)
        return self._result(
            plan,
            u_c,
            t0,
            level=level,
            iterations=iterations,
            certified=False,
            recovering=True,
            slack=max(0.0, -held, -end),
        )

    def _rollout(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """States (N + 1, 6) of the plan `u` (N, 3) from `x` on the exact CW step."""
        xs = np.empty((len(u) + 1, 6))
        xs[0] = x
        for k, u_k in enumerate(u):
            xs[k + 1] = self._phi @ xs[k] + self._gamma_u @ u_k
        return xs

    def _margins(self, x: np.ndarray, u: np.ndarray, stop: bool = True) -> tuple[float, float]:
        """The lowest held-step bound up to the last thrust and h_p after it, m.

        With `stop`, a plan short by more than the keep-out resolution on a held step fails
        every check, so the rest is skipped and h_p is returned as -inf.
        """
        xs = self._rollout(x, u)
        thrusting = np.flatnonzero(np.any(u != 0.0, axis=1))
        last = int(thrusting[-1]) + 1 if thrusting.size else 0
        held = math.inf
        for k in range(last):
            bound = held_segment_bound(xs[k], u[k], self._dt, _HELD_SAMPLES, self._radius, self._n)
            held = min(held, bound)
            if stop and held < -KeepOut.resolution:
                return held, -math.inf
        return held, float(self._barrier.value(xs[last]))

    def _problem(
        self, x: np.ndarray, u_c: np.ndarray, reference: np.ndarray, certified: bool = True
    ) -> StageQP:
        """The horizon program from `x`, linearized around the plan `reference` (N, 3).

        The keep-out rows pass through a reference closer than R + b only when `certified`
        says the reference is a certified plan, which they then keep feasible.
        """
        steps, n, radius = self.horizon, self._n, self._radius
        ref_x = self._rollout(x, reference)
        samples = np.einsum("jab,kb->kja", self._row_phi, ref_x[:-1]) + np.einsum(
            "jab,kb->kja", self._row_gamma, reference
        )
        r_ref, v_ref = samples[..., :3], samples[..., 3:]
        rho = np.linalg.norm(r_ref, axis=-1)
        speed = np.linalg.norm(v_ref, axis=-1) + self._a_reach * self._dt
        accel = self._a_reach + 3.0 * n * n * rho + 2.0 * n * speed
        h = self._row_h
        buffer = (speed * h) ** 2 / (8.0 * radius) + accel * h * h / 8.0
        # A reference through the target's centre has no tangent plane; its rows are left out.
        through = rho == 0.0
        normal = r_ref / np.where(through, 1.0, rho)[..., None]
        keep_out = np.concatenate(
            [
                np.einsum("kja,jab->kjb", normal, self._row_phi[:, :3]),
                np.einsum("kja,jab->kjb", normal, self._row_gamma[:, :3]),
            ],
            axis=-1,
        )
        keep_bound = np.where(
            through, -1.0, np.minimum(radius + buffer, rho) if certified else radius + buffer
        )
        rows = np.concatenate([np.broadcast_to(self._thrust_rows, (steps, 6, 9)), keep_out], axis=1)
        bounds = np.concatenate(
            [np.broadcast_to(self._thrust_bound, (steps, 6)), keep_bound], axis=1
        )
        terminal, terminal_bound = self._terminal_rows(ref_x[-1])
        hessian = np.zeros((steps, 9, 9))
        hessian[:, :6, :6] = self._state_weight
        hessian[:, 6:, 6:] = self.eps * np.eye(3)
        hessian[0, 6:, 6:] += np.eye(3)
        gradient = np.zeros((steps, 9))
        gradient[:, :6] = -ref_x[:-1] @ self._state_weight
        gradient[0, 6:] = -u_c
        return StageQP(
            x0=x,
            A=np.broadcast_to(self._phi, (steps, 6, 6)),
            B=np.broadcast_to(self._gamma_u, (steps, 6, 3)),
            c=np.zeros((steps, 6)),
            Q=hessian,
            q=gradient,
            P=self._state_weight,
            p=-self._state_weight @ ref_x[-1],
            C=rows,
            e=bounds,
            C_N=terminal,
            e_N=terminal_bound,
        )

    def _terminal_rows(self, x_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rows C_N x_N >= e_N, linearized at `x_end`, that keep the end state in the set.

        An end state in the far part of the terminal set is kept there by the far rows. Any
        other gets h_p linearized by the envelope gradient, or, where the drift minimum has no
        gradient, by the gradient of the range at the end state, which is the minimum when the
        drift recedes from the start.
        """
        radius = self._radius
        x_c_row, y_c_row, p_row, q_row = self._invariants
        x_c, y_c, p, q = self._invariants @ x_end
        side = 1.0 if y_c >= 0.0 else -1.0
        if x_c * y_c <= 0.0 and abs(y_c) - 2.0 * math.hypot(p, q) - radius >= 0.0:
            # The far part, |y_c| - 2 b - R >= 0 with y_c receding, bounded inside by
            # b <= |p| + |q| and a 1 mm margin on each row.
            rows = [-side * x_c_row]
            for sp in (1.0, -1.0):
                for sq in (1.0, -1.0):
                    rows.append(side * y_c_row - 2.0 * (sp * p_row + sq * q_row))
            return np.array(rows), np.array([_FAR_MARGIN, *([radius + _FAR_MARGIN] * 4)])
        barrier = self._barrier
        try:
            grad = barrier.gradient(x_end)
        except ValueError:
            rho = float(np.linalg.norm(x_end[:3]))
            if rho == 0.0:
                return np.zeros((0, 6)), np.zeros(0)
            grad = np.concatenate([x_end[:3] / rho, np.zeros(3)])
        value = float(barrier.value(x_end))
        return grad[None, :], np.array([float(grad @ x_end) - value])

    def _result(
        self,
        plan: np.ndarray,
        u_c: np.ndarray,
        t0: float,
        *,
        level: int,
        iterations: int | None,
        own: np.ndarray | None = None,
        certified: bool = True,
        status: str = "feasible",
        recovering: bool = False,
        slack: float = 0.0,
    ) -> FilterResult:
        """Store `plan` as applied and report its first input."""
        self._plan, self._certified = plan, certified
        u = plan[0]
        return FilterResult(
            self._action(u) if own is None else own,
            intervened=self._moved(u, u_c),
            status=status,
            slack=slack,
            compute_s=time.perf_counter() - t0,
            # Every answer but the agent's own command comes after a solve.
            active=iterations is not None,
            recovering=recovering,
            fallback_level=level,
            certificate="nominal" if certified else None,
            certificate_horizon=math.inf if certified else None,
            iterations=iterations,
        )


__all__ = ["PredictiveFilter"]
