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
from ..metrics.passive import cam_direction
from ..metrics.settings import MetricSettings
from ..reach import Zonotope, cw_loop, drift_clearance, loop_clearance
from ..solvers.ipm import IPMSettings, solve_horizon
from ..solvers.stage import StageQP
from . import FilterResult
from .barriers import BrakingBarrier, CamBarrier, PassiveBarrier, braking_deceleration
from .cbf import (
    _HELD_SAMPLES,
    FilterDeclaration,
    _HeldCommandFilter,
    _nonnegative,
    _positive,
    _state,
)
from .sampled import held_segment_bound
from .tightening import BallError, BoxError, EllipsoidError, ErrorSet, error_zonotope

PREDICTIVE_MODES = ("terminal", "finite-horizon", "soft")
PREDICTIVE_BACKUPS = ("passive", "cam")
_NAME_TAGS = {"terminal": "", "finite-horizon": "fh-", "soft": "soft-"}
# Largest distance of a delay from a whole number of steps, in steps, as `delay_margin` takes.
_STEP_TOL = 1e-9
# Margin of the far-part terminal rows (m): it keeps the plan's end state receding, where a
# centre standing still on its threshold would let rounding decide membership.
_FAR_MARGIN = KeepOut.resolution
# A keep-out row through a certified reference sits this fraction of its range below it. On the
# row itself the reference meets it only to rounding, and where the reference brakes at the
# corner of the box the row favours, that corner is the stage's only plan: rounding then decides
# whether the program is feasible at all, and the solver's convergence with it.
_BELOW_REFERENCE = 1e-9
# Fraction of the input box's half-width the warm start is held within.
_WARM_INSIDE = 0.99
# Weights of the soft program's keep-out and terminal slacks, per metre of enlargement. An exact
# penalty needs each above the sum of the hard program's multipliers on the rows its slack
# relaxes.
_ALPHA = 1.0
_BETA = 1.0
# Slacks start this far (m) above the shortfall of the warm start, so their rows start inside.
_WARM_SLACK = KeepOut.resolution
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
    mode : str
        "terminal" certifies plans for all time through the terminal free-drift set;
        "finite-horizon" certifies the free drift after the last thrust up to `drift_horizon`
        only; "soft" relaxes the program's keep-out and terminal rows by penalized slacks, so
        the program always has a solution, and certifies as "terminal" does.
    drift_horizon : float, optional
        The finite-horizon mode's drift horizon, s, positive; ten orbits, 20 pi / n, by
        default. Refused in the other modes.
    alpha : float, optional
        The soft mode's weight on each step's keep-out slack, per m, positive; 1 by default.
        Refused in the other modes.
    beta : float, optional
        The soft mode's weight on the terminal slack, per m, positive; 1 by default. Refused in
        the other modes.
    backup : str
        "passive" declares the free-drift set, kept by zero thrust; "cam" declares the
        CAM-then-drift set of `BackupFilter(barrier="cam")`, kept by its CAM, and adds "cam-"
        to the name. Refused as "cam" in the finite-horizon mode.
    brake_v_cap : float
        The largest speed of the braking region, m/s, positive.
    rho_cap : float
        The largest range of the braking region, m, above the keep-out radius.
    navigation : BoxError, BallError or EllipsoidError, optional
        Bound on the error of the observed state, m and m/s; none by default.
    execution : float
        Bound on the execution error of each held command, m/s^2, nonnegative, taken as the
        box W of that half-width on each Hill axis, which holds the ball of that radius.
    grid : float
        Spacing of the free-drift samples, s, positive; shortened to dt / k for the smallest
        whole k that reaches it, except under error sets.
    exceedance : float, optional
        The episode-level probability, in (0, 1), that the untruncated error models leave the
        truncation the error sets were derived from; given with `per_draw_level` and an error
        set, and reported in the declaration only.
    per_draw_level : float, optional
        The truncation level of each Gaussian draw of those models, in sigma, finite and
        positive; given with `exceedance`, and reported in the declaration only.
    ignore : sequence of str
        Constraint types of the scenario not to enforce; any other type than keep_out makes
        reset raise, since the free drift may leave it.

    Raises
    ------
    ValueError
        If a parameter is out of range, the mode or the backup is unknown, a parameter is
        given to a mode that does not use it, error sets are given with the cam backup or
        the finite-horizon mode, or `exceedance` and `per_draw_level` are not given together
        with an error set.

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
    n . r >= min(R + b, (1 - 1e-9) |r'|) at `row_samples` times in each step; b covers the dip of
    the path between samples, (v h)^2 / (8 R) + a h^2 / 8 with v and a bounds on the speed and
    acceleration near the reference. The minimum with (1 - 1e-9) |r'|, which keeps the reference
    strictly feasible whatever the rounding, is taken only for a certified reference: the
    shifted plan after a certified decision, or zero thrust with h_p >= 0; any other reference
    gets R + b. A reference that ends in the far
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

    When nothing is certified and the state lies in the region of the braking barrier
    `BrakingBarrier` with a_b from `braking_deceleration` at `brake_v_cap` and `rho_cap`, with
    h_b >= 0, full outward braking is applied at level 2: h_b does not fall under it in that
    region. The decision is recovering and has no certificate. Without a positive a_b there is
    no braking level. With backup="cam", level 2 is `CamBarrier.backup` wherever the
    CAM-then-drift margin is nonnegative, certified "nominal", and the stored plan is that
    backup rolled out until the free drift is certified, so the next decision without a
    solution replays it at level 1. Plans are still certified through the free-drift set.

    The finite-horizon mode checks the drift after the last thrust with
    `PassiveBarrier(terminal=False, t_cap=drift_horizon)` and linearizes it by its gradient
    row alone. A certified decision has certificate "finite-horizon" with horizon
    drift_horizon. Free drift does not keep that set, so a decision at a state outside it is
    recovering even when its plan is certified.

    The soft mode adds to each step's input a slack s_k >= 0 (m) that enlarges every keep-out
    row of the step, and a slack s_T >= 0 on the terminal rows, which act on the last step
    through x_N = Phi x_{N-1} + Gamma u_{N-1}. The cost gains alpha sum s_k + beta s_T and the
    term eps n^4 |s|^2 / 2 that keeps the input block definite. The penalty is exact (Boyd and
    Vandenberghe, *Convex Optimization*, ex. 5.16): with alpha and beta above the sums of the
    hard program's multipliers on the rows each slack relaxes, the soft solution is the hard
    one wherever the hard program has a solution. A soft solution that fails the check takes
    the place of the converged solution in the recovering answers. Neither the finite-horizon
    nor the soft mode counts toward certification.

    With a delay, the commands already issued act first: the state delay_s after the
    observation is predicted through the echoed commands and the plan starts there. With
    delay_s > dt the latest echoed command waits delay_s - dt from the second decision on, which
    is checked; with delay_s <= dt every wait is zero and the value cannot be checked.

    Error sets make the filter bounded. The navigation set E enters as the zonotope of
    `tightening.error_zonotope`, exact for a box and an outer bound otherwise, and the
    execution set W as the box of half-width `execution`. No feedback acts on the errors over a
    plan (K = 0), so the error of the plan's state x_k lies in
    Z_k = Phi^k Z_0 + sum_{j<k} Phi^j Gamma W, with Z_0 = Phi(delay_s) E plus the error the
    pending commands build up, sum_{j<delay_s/dt} Phi^j Gamma W (Kochenderfer et al.,
    *Algorithms for Validation*, eq. 8.20). Each keep-out row a_x . x_k + a_u . u_k >= b of
    the program is raised by the support h_{Z_k}(a_x) + h_W(a_u), and each terminal row
    a . x_N >= b by h_{Z_N}(a); the terminal rows linearize `PassiveBarrier` without
    `terminal`, capped at t_cap = 4 pi / n. The check of a plan bounds each step up to the
    last nonzero input by `vbar.reach.loop_clearance` of x_k + Z_k under u_k + W, and the free
    drift after it by `vbar.reach.drift_clearance` of the end state's set over t_cap, on
    ceil(t_cap / grid) + 1 samples; zero thrust is taken as exact. `contains(x)` is that drift
    clearance of x + E, and a decision tests the predicted state with Z_0, which is x + E
    without a delay. A certified decision has certificate "bounded-W", or "chance-epsilon" for
    an ellipsoid with a positive exceedance, with horizon t_cap. Zero thrust at level 2 passes
    the same check or is not certified; a decision at a state outside the set is recovering.
    Free drift does not keep the set, so recursive feasibility is not claimed.
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
        mode: str = "terminal",
        drift_horizon: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        backup: str = "passive",
        brake_v_cap: float = 1.0,
        rho_cap: float = 2000.0,
        navigation: ErrorSet | None = None,
        execution: float = 0.0,
        grid: float = 5.0,
        exceedance: float | None = None,
        per_draw_level: float | None = None,
        ignore=(),
    ) -> None:
        super().__init__(
            execution=execution,
            model_error=None,
            ignore=ignore,
            navigation=navigation,
            exceedance=exceedance,
            per_draw_level=per_draw_level,
        )
        if navigation is not None and not isinstance(
            navigation, (BoxError, BallError, EllipsoidError)
        ):
            raise ValueError("navigation must be a BoxError, BallError or EllipsoidError")
        self.navigation = navigation
        if self.bounded and backup == "cam":
            raise ValueError("the cam backup makes no claim under error sets")
        if self.bounded and mode == "finite-horizon":
            raise ValueError("error sets need the terminal or the soft mode")
        if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
            raise ValueError("horizon must be an integer of at least 1")
        if isinstance(row_samples, bool) or int(row_samples) != row_samples or row_samples < 2:
            raise ValueError("row_samples must be an integer of at least 2")
        if settings is not None and not isinstance(settings, IPMSettings):
            raise ValueError("settings must be an IPMSettings")
        if mode not in PREDICTIVE_MODES:
            raise ValueError(f"unknown mode {mode!r}, expected {PREDICTIVE_MODES}")
        if backup not in PREDICTIVE_BACKUPS:
            raise ValueError(f"unknown backup {backup!r}, expected {PREDICTIVE_BACKUPS}")
        if drift_horizon is not None and mode != "finite-horizon":
            raise ValueError("drift_horizon is a parameter of the finite-horizon mode")
        if (alpha is not None or beta is not None) and mode != "soft":
            raise ValueError("alpha and beta are parameters of the soft mode")
        if backup == "cam" and mode == "finite-horizon":
            raise ValueError("the cam backup declares a set for all time, not up to a horizon")
        self.horizon = int(horizon)
        self.row_samples = int(row_samples)
        self.eps = _positive(eps, "eps")
        self.settings = settings
        self.delay_s = _nonnegative(delay_s, "delay_s")
        self.mode = mode
        self.drift_horizon = (
            None if drift_horizon is None else _positive(drift_horizon, "drift_horizon")
        )
        self.alpha: float | None = None
        self.beta: float | None = None
        if mode == "soft":
            self.alpha = _ALPHA if alpha is None else _positive(alpha, "alpha")
            self.beta = _BETA if beta is None else _positive(beta, "beta")
        self.backup = backup
        self.brake_v_cap = _positive(brake_v_cap, "brake_v_cap")
        self.rho_cap = _positive(rho_cap, "rho_cap")
        self.grid = _positive(grid, "grid")
        cam = "cam-" if backup == "cam" else ""
        self.name = f"predictive-{_NAME_TAGS[mode]}{cam}n{self.horizon}"

    def reset(self, view: AgentView) -> None:
        """Check the view, its pending command slots, and build the barriers.

        Raises
        ------
        ValueError
            If the action mode, frame, observation, dynamics or constraints are unsupported, the
            scenario has no keep_out, the view does not echo ceil(delay_s / dt) pending
            commands, rho_cap does not exceed the keep-out radius, drift_horizon is shorter
            than the drift grid, the cam backup is asked for a goal with no CAM direction, or
            an execution error comes with a delay that is not a whole number of steps.
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
        self._decisions = 0
        self._cw = CWDiscretizer(n)
        self._invariants = _INVARIANTS / np.array([1.0, 1.0, 1.0, n, n, n])
        # The deviation from the reference as the accelerations n^2 dr and n dv, weighted like
        # the later inputs.
        self._state_weight = self.eps * np.diag([n**4] * 3 + [n**2] * 3)
        self._phi, self._gamma_u = cw_discrete(n, dt)
        # The drift samples after a step must be samples of the drift before it, so the grid
        # divides dt and the terminal set is kept by a drift step.
        grid = dt / math.ceil(dt / self.grid)
        self._errors: list[np.ndarray] | None = None
        if self.bounded:
            t_cap = 4.0 * math.pi / n
            self._drift = PassiveBarrier(radius=radius, n=n, grid=self.grid, t_cap=t_cap)
            self._drift_samples = math.ceil(t_cap / self.grid) + 1
            self._loop = cw_loop(n, dt)
            self._navigation_set = np.zeros((6, 0))
            if self.navigation is not None:
                self._navigation_set = error_zonotope(self.navigation).generators
            self._errors = self._error_sets()
        elif self.mode == "finite-horizon":
            self._drift = PassiveBarrier(radius=radius, n=n, grid=grid, t_cap=self.drift_horizon)
        else:
            self._drift = PassiveBarrier(
                radius=radius, n=n, grid=grid, t_cap=20.0 * math.pi / n, terminal=True
            )
        layout, a_max = self._actuator.layout, self._actuator.a_max
        a_b = braking_deceleration(
            a_max=a_max,
            layout=layout,
            v_cap=self.brake_v_cap,
            dt=dt,
            radius=radius,
            rho_cap=self.rho_cap,
            n=n,
        )
        self._braking: BrakingBarrier | None = None
        if a_b > 0.0:
            self._braking = BrakingBarrier(
                radius=radius, a_b=a_b, rho_cap=self.rho_cap, v_cap=self.brake_v_cap
            )
        self._cam = self._cam_barrier(view, grid) if self.backup == "cam" else None
        phis, gammas = cw_discrete(n, np.linspace(0.0, dt, self.row_samples))
        # Sample 0 of a step is the last of the step before, or the current state.
        self._row_phi, self._row_gamma = phis[1:], gammas[1:]
        self._row_h = dt / (self.row_samples - 1)
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
        if self._cam is not None:
            safe_set = (
                f"a CAM of at most {self._cam.steps} steps, then free drift, certified "
                "outside the keep-out for all time"
            )
            backup = "zero thrust, else the CAM"
        elif self.bounded:
            safe_set = (
                "free drift of every state within the declared errors of the estimate, "
                "propagated without feedback (K = 0), certified outside the keep-out up to "
                "the drift horizon"
            )
            backup = "zero thrust"
        elif self.mode == "finite-horizon":
            safe_set = "free drift certified outside the keep-out up to the drift horizon"
            backup = "zero thrust"
        else:
            safe_set = "free drift certified outside the keep-out for all time"
            backup = "zero thrust"
        if self._braking is not None:
            backup += ", else full outward braking outside the set"
        self._declaration = FilterDeclaration(
            name=self.name,
            safe_set=safe_set,
            barriers=(self._declared,),
            poles=None,
            gamma=None,
            design_dt=dt,
            layout=layout,
            a_max=a_max,
            buffer_m=0.0,
            tightening={
                "navigation": self.navigation,
                "execution": self.execution,
                "model_error": None,
            },
            backup=backup,
            certificate=self._kind,
            counts_toward_certification=self.mode == "terminal",
            ignored=self.ignore,
            exceedance=self.exceedance,
            per_draw_level=self.per_draw_level,
        )

    @property
    def bounded(self) -> bool:
        """Whether error sets are declared, which tightens the program and every check."""
        return self.navigation is not None or self.execution > 0.0

    @property
    def _declared(self) -> PassiveBarrier | CamBarrier:
        """The barrier of the declared set."""
        return self._cam if self._cam is not None else self._drift

    @property
    def _kind(self) -> str:
        """The kind of guarantee of a certified decision."""
        if self.bounded:
            # An ellipsoid left with a stated probability bounds the error only with that chance.
            if isinstance(self.navigation, EllipsoidError) and self.navigation.exceedance > 0.0:
                return "chance-epsilon"
            return "bounded-W"
        return "finite-horizon" if self.mode == "finite-horizon" else "nominal"

    def _error_sets(self) -> list[np.ndarray]:
        """Generators of Z_0 to Z_N, the error of each plan state about its prediction."""
        n, dt, da = self._n, self._dt, self.execution
        ratio = self.delay_s / dt
        steps = round(ratio)
        if da > 0.0 and abs(ratio - steps) > _STEP_TOL:
            raise ValueError("an execution error needs a delay of a whole number of steps dt")
        pushed = [da * self._gamma_u] if da > 0.0 else []
        window, power = [], np.eye(6)
        for _ in range(steps):
            window += [power @ g for g in pushed]
            power = self._phi @ power
        z = np.hstack([cw_discrete(n, self.delay_s)[0] @ self._navigation_set, *window])
        sets = [z]
        for _ in range(self.horizon):
            z = np.hstack([self._phi @ z, *pushed])
            sets.append(z)
        return sets

    def _cam_barrier(self, view: AgentView, grid: float) -> CamBarrier:
        """The CAM-then-drift barrier that `BackupFilter(barrier="cam")` builds for the view."""
        try:
            direction = cam_direction(view.goal)
        except ValueError as error:
            raise ValueError("the scenario's goal declares no CAM direction") from error
        a_max, dt = self._actuator.a_max, self._dt
        # A quotient that rounds just above a whole number of steps is that number.
        steps = max(1, math.ceil(MetricSettings().cam_dv / (a_max * dt) - 1e-9))
        return CamBarrier(
            radius=self._radius,
            n=self._n,
            dt=dt,
            accel=tuple(a_max * direction),
            steps=steps,
            grid=grid,
            samples=_HELD_SAMPLES,
        )

    def contains(self, x) -> bool:
        """Return whether the Hill-frame state `x` (m, m/s) lies in the declared set.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        """
        self._started()
        x = np.asarray(x, dtype=float)
        if self._errors is not None:
            return self._clearance(x, self._navigation_set) >= 0.0
        return self._declared.contains(x)

    def contains_start(self, x) -> bool:
        """Return whether every estimate x + e, e in the navigation set, lies in the set.

        A start the audit draws for the true state `x` (m, m/s) passes when the filter's first
        estimate is in its declared set whatever the navigation error. Every estimate's set
        x + e + Z_E lies in x + 2 Z_E, so this is the drift clearance of x + 2 Z_E; without a
        navigation set it is `contains(x)`.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        """
        self._started()
        x = np.asarray(x, dtype=float)
        if self.navigation is None:
            return self.contains(x)
        return self._clearance(x, 2.0 * self._navigation_set) >= 0.0

    def _clearance(self, x: np.ndarray, generators: np.ndarray) -> float:
        """The certified free-drift clearance up to t_cap of every state in x + G beta, m."""
        found = drift_clearance(
            self._n,
            Zonotope(x, generators),
            self._radius,
            float(self._drift.t_cap),  # type: ignore[arg-type]
            self._drift_samples,
        )
        return found.certified

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

    def _check_wait(self, obs) -> None:
        """Count the decision; from the second, the latest echoed wait must be delay_s - dt."""
        self._decisions += 1
        if self._decisions == 1 or not self.delay_s > self._dt:
            return
        wait = float(np.asarray(obs, dtype=float)[self._slots[-1, 0]])
        expected = self.delay_s - self._dt
        if not abs(wait - expected) <= 1e-6:
            raise ValueError(
                f"the latest pending command waits {wait:g} s; a delay of {self.delay_s:g} s "
                f"at dt {self._dt:g} s gives {expected:g} s"
            )

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
            certified solution, 1 for the stored plan shifted, 2 for zero thrust, the CAM or
            braking, and `iterations` is the solver's count. A certified decision has
            `certificate` "nominal" with horizon inf, in the finite-horizon mode
            "finite-horizon" with horizon drift_horizon, and under error sets "bounded-W", or
            "chance-epsilon", with horizon 4 pi / n. When nothing is certified, which
            happens only outside the declared set, braking in its region, else the converged
            solution, else the shifted plan, else zero thrust is returned as "feasible" with
            `recovering` True, its predicted shortfall (m) as `slack` and no certificate.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        ValueError
            If the observation or the action is malformed, or the latest echoed command's wait
            disagrees with delay_s.
        """
        t0 = time.perf_counter()
        view = self._started()
        # predict checks the observation's shape, which the wait check relies on.
        x = self.predict(obs)
        self._check_wait(obs)
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
        # Free drift does not keep the finite-horizon set or the set under error sets, so a
        # certified plan can start outside them.
        drift_kept: bool | None = None
        if self._errors is not None:
            drift_kept = self._end_value(x, 0) >= 0.0
            outside = not drift_kept
        else:
            outside = self.mode == "finite-horizon" and not self._drift.contains(x)
        steps = self.horizon
        candidate = np.zeros((steps, 3))
        candidate[0] = u_c
        held, end = self._margins(x, candidate)
        if held >= 0.0 and end >= 0.0:
            return self._result(
                candidate, u_c, t0, level=0, iterations=None, own=own, recovering=outside
            )

        drift = np.zeros((steps, 3))
        if self._plan is None:
            shifted = None
            reference = drift
            if drift_kept is None:
                drift_kept = self._end_value(x, 0) >= 0.0
            certified_reference = drift_kept
        else:
            shifted = np.vstack([self._plan[1:], np.zeros((1, 3))])
            reference = shifted
            certified_reference = self._certified
        # The solver starts each row's slack at its value at the warm start, so inputs on a face
        # of the box would start that row's multiplier near mu / 0; 1% inside, they do not.
        warm = np.clip(reference, -_WARM_INSIDE * self._half, _WARM_INSIDE * self._half)
        problem = self._problem(x, u_c, reference, certified=certified_reference)
        start = (self._rollout(x, warm), warm)
        if self.mode == "soft":
            problem, start = self._softened(problem, start)
        solution = solve_horizon(problem, self.settings, warm=start)
        iterations = int(solution.iterations)
        solved = None
        if solution.status == "converged" and solution.u is not None:
            inputs = np.asarray(solution.u, dtype=float)[:, :3]
            solved = np.array([self._onto_limit(u) for u in inputs])
            held, end = self._margins(x, solved)
            if held >= 0.0 and end >= 0.0:
                return self._result(
                    solved, u_c, t0, level=0, iterations=iterations, recovering=outside
                )
        if shifted is not None:
            held, end = self._margins(x, shifted)
            # The stored plan was certified from the predicted state; re-rolled from the state
            # now, it differs by rounding, which can cost the held bound up to the resolution.
            if held >= -KeepOut.resolution and end >= 0.0:
                return self._result(
                    shifted, u_c, t0, level=1, iterations=iterations, recovering=outside
                )
        if drift_kept is None:
            drift_kept = self._end_value(x, 0) >= 0.0
        if drift_kept:
            return self._result(drift, u_c, t0, level=2, iterations=iterations)
        if self._cam is not None and float(self._cam.value(x)) >= 0.0:
            return self._result(self._cam_plan(x), u_c, t0, level=2, iterations=iterations)

        # Zero thrust, or the CAM, is certified throughout the declared set, so nothing
        # certified means the state is outside it: a recovering step, never "solver_failed".
        braking = self._braking
        if braking is not None and np.any(x[:3] != 0.0) and braking.contains(x):
            plan, level = drift.copy(), 2
            plan[0] = braking.backup(x, self._actuator)
        elif solved is not None:
            plan, level = solved, 0
        elif shifted is not None:
            plan, level = shifted, 1
        else:
            plan, level = drift, 2
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

    def _cam_plan(self, x: np.ndarray) -> np.ndarray:
        """The CAM backup rolled out from `x` until the free drift is certified, then zeros."""
        cam = self._cam
        assert cam is not None
        plan = np.zeros((self.horizon, 3))
        for k in range(min(cam.steps, self.horizon)):
            u = cam.backup(x, self._actuator)
            if not np.any(u):
                break
            plan[k] = u
            x = self._phi @ x + self._gamma_u @ u
        return plan

    def _rollout(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """States (N + 1, 6) of the plan `u` (N, 3) from `x` on the exact CW step."""
        xs = np.empty((len(u) + 1, 6))
        xs[0] = x
        for k, u_k in enumerate(u):
            xs[k + 1] = self._phi @ xs[k] + self._gamma_u @ u_k
        return xs

    def _margins(self, x: np.ndarray, u: np.ndarray, stop: bool = True) -> tuple[float, float]:
        """The lowest held-step bound up to the last thrust and h_p after it, over the errors, m.

        With `stop`, a plan short by more than the keep-out resolution on a held step fails
        every check, so the rest is skipped and h_p is returned as -inf.
        """
        xs = self._rollout(x, u)
        thrusting = np.flatnonzero(np.any(u != 0.0, axis=1))
        last = int(thrusting[-1]) + 1 if thrusting.size else 0
        held = math.inf
        for k in range(last):
            held = min(held, self._held(xs[k], u[k], k))
            if stop and held < -KeepOut.resolution:
                return held, -math.inf
        return held, self._end_value(xs[last], last)

    def _held(self, x: np.ndarray, u: np.ndarray, k: int) -> float:
        """A lower bound on the keep-out margin over step k of a plan, from `x` under `u`, m."""
        if self._errors is None:
            return held_segment_bound(x, u, self._dt, _HELD_SAMPLES, self._radius, self._n)
        found = loop_clearance(
            self._loop,
            Zonotope(x, self._errors[k]),
            Zonotope.box(u, np.full(3, self.execution)),
            self._radius,
            1,
            _HELD_SAMPLES,
        )
        return found.certified

    def _end_value(self, x: np.ndarray, k: int) -> float:
        """h_p at the plan's state `x` after k steps, or with error sets its set clearance, m."""
        if self._errors is None:
            return float(self._drift.value(x))
        return self._clearance(x, self._errors[k])

    def _problem(
        self, x: np.ndarray, u_c: np.ndarray, reference: np.ndarray, certified: bool = True
    ) -> StageQP:
        """The horizon program from `x`, linearized around the plan `reference` (N, 3).

        The keep-out rows pass just below a reference closer than R + b only when `certified`
        says the reference is a certified plan, which they then keep strictly feasible.
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
            through,
            -1.0,
            np.minimum(radius + buffer, (1.0 - _BELOW_REFERENCE) * rho)
            if certified
            else radius + buffer,
        )
        terminal, terminal_bound = self._terminal_rows(ref_x[-1])
        if self._errors is not None:
            # Each row is raised by the support of the error of its sample, h_Zk(a_x) + h_W(a_u).
            keep_bound = keep_bound + np.array(
                [np.abs(keep_out[k, :, :6] @ self._errors[k]).sum(axis=1) for k in range(steps)]
            )
            keep_bound += self.execution * np.abs(keep_out[..., 6:]).sum(axis=-1)
            terminal_bound = terminal_bound + np.abs(terminal @ self._errors[-1]).sum(axis=1)
        rows = np.concatenate([np.broadcast_to(self._thrust_rows, (steps, 6, 9)), keep_out], axis=1)
        bounds = np.concatenate(
            [np.broadcast_to(self._thrust_bound, (steps, 6)), keep_bound], axis=1
        )
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

    def _softened(
        self, qp: StageQP, warm: tuple[np.ndarray, np.ndarray]
    ) -> tuple[StageQP, tuple[np.ndarray, np.ndarray]]:
        """The program `qp` with the slack inputs (s_k, s_T) on its keep-out and terminal rows.

        Each stage's input becomes (u, s_k, s_T). The terminal rows move to the last stage
        through its dynamics, with s_T, and leave rows that always hold on the others, where
        s_T meets only its bound and its weight and settles at zero. The warm start's slacks
        sit just above its shortfall on the rows they relax.
        """
        assert self.alpha is not None and self.beta is not None
        steps, nx = self.horizon, 6
        nz = nx + 3
        rows, bounds = np.asarray(qp.C), np.asarray(qp.e)
        thrusts, m = len(self._thrust_rows), rows.shape[1]
        terminal = np.zeros((0, nx)) if qp.C_N is None else np.asarray(qp.C_N)
        terminal_bound = np.zeros(0) if qp.e_N is None else np.asarray(qp.e_N)

        soft_rows = np.zeros((steps, m + 2 + len(terminal), nz + 2))
        soft_bounds = np.zeros((steps, m + 2 + len(terminal)))
        soft_rows[:, :m, :nz] = rows
        soft_rows[:, thrusts:m, nz] = 1.0
        soft_bounds[:, :m] = bounds
        soft_rows[:, m, nz] = 1.0
        soft_rows[:, m + 1, nz + 1] = 1.0
        soft_bounds[:, m + 2 :] = -1.0
        soft_rows[-1, m + 2 :, :nx] = terminal @ qp.A[-1]
        soft_rows[-1, m + 2 :, nx:nz] = terminal @ qp.B[-1]
        soft_rows[-1, m + 2 :, nz + 1] = 1.0
        soft_bounds[-1, m + 2 :] = terminal_bound

        hessian = np.zeros((steps, nz + 2, nz + 2))
        hessian[:, :nz, :nz] = qp.Q
        hessian[:, nz, nz] = hessian[:, nz + 1, nz + 1] = self.eps * self._n**4
        gradient = np.zeros((steps, nz + 2))
        gradient[:, :nz] = qp.q
        gradient[:, nz] = self.alpha
        gradient[:, nz + 1] = self.beta
        inputs = np.zeros((steps, nx, 5))
        inputs[:, :, :3] = qp.B

        x_warm, u_warm = warm
        z = np.concatenate([x_warm[:-1], u_warm], axis=1)
        short = bounds[:, thrusts:] - np.einsum("kij,kj->ki", rows[:, thrusts:], z)
        slacks = np.full((steps, 2), _WARM_SLACK)
        slacks[:, 0] += np.maximum(0.0, short.max(axis=1, initial=0.0))
        if len(terminal):
            slacks[-1, 1] += max(0.0, float(np.max(terminal_bound - terminal @ x_warm[-1])))
        soft = StageQP(
            x0=qp.x0,
            A=qp.A,
            B=inputs,
            c=qp.c,
            Q=hessian,
            q=gradient,
            P=qp.P,
            p=qp.p,
            C=soft_rows,
            e=soft_bounds,
            C_N=None,
            e_N=None,
        )
        return soft, (x_warm, np.concatenate([u_warm, slacks], axis=1))

    def _terminal_rows(self, x_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rows C_N x_N >= e_N, linearized at `x_end`, that keep the end state in the set.

        An end state in the far part of the terminal set is kept there by the far rows. Any
        other gets h_p linearized by the envelope gradient, or, where the drift minimum has no
        gradient, by the gradient of the range at the end state, which is the minimum when the
        drift recedes from the start. The finite-horizon set has no far part.
        """
        radius = self._radius
        x_c_row, y_c_row, p_row, q_row = self._invariants
        x_c, y_c, p, q = self._invariants @ x_end
        side = 1.0 if y_c >= 0.0 else -1.0
        far = x_c * y_c <= 0.0 and abs(y_c) - 2.0 * math.hypot(p, q) - radius >= 0.0
        if self._drift.terminal and far:
            # The far part, |y_c| - 2 b - R >= 0 with y_c receding, bounded inside by
            # b <= |p| + |q| and a 1 mm margin on each row.
            rows = [-side * x_c_row]
            for sp in (1.0, -1.0):
                for sq in (1.0, -1.0):
                    rows.append(side * y_c_row - 2.0 * (sp * p_row + sq * q_row))
            return np.array(rows), np.array([_FAR_MARGIN, *([radius + _FAR_MARGIN] * 4)])
        barrier = self._drift
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
        horizon = None
        if certified:
            horizon = math.inf
            if not self._drift.terminal:
                horizon = float(self._drift.t_cap)  # type: ignore[arg-type]
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
            certificate=self._kind if certified else None,
            certificate_horizon=horizon,
            iterations=iterations,
        )


__all__ = ["PREDICTIVE_BACKUPS", "PREDICTIVE_MODES", "PredictiveFilter"]
