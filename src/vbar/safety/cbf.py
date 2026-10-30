"""Control barrier function safety filters for the keep-out sphere under sample and hold.

Every filter here turns the agent's command into an acceleration (m/s^2, Hill frame), clips it
onto the thrust limit, changes it only as far as its barriers require, and returns it in the
view's action mode. Clipping is what the spacecraft does anyway, so it never counts as an
intervention.

`CbfFilter` projects the command onto the exponential barrier rows of the keep-out sphere and
of declared approach cones and first-order speed rows (Ames et al., *Control Barrier Functions:
Theory and Applications*, ECC 2019, sec. II and IV). The rows are continuous-time conditions,
so the filter is designed for the held step: its poles obey max(p) dt <= 0.6, the keep-out
radius carries a buffer for riding the boundary, and it makes no certified claim.
`BackupFilter` checks each held step on the exact Clohessy-Wiltshire solution and searches
toward a backup command, free drift, a collision avoidance manoeuvre then free drift, or
braking along and across the line of sight, whose safety is the barrier's own theorem (sec. III,
Theorem 4).
`SoftCbfFilter` penalizes the rows instead of enforcing them and is never a certified filter.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ..agents.base import CONTINUOUS_MODES, AgentView, action_scale, command_fraction
from ..constraints import ApproachCone, KeepOut
from ..dynamics.cw import cw_discrete
from ..metrics.passive import cam_direction
from ..metrics.settings import MetricSettings
from ..reach import Zonotope, drift_clearance
from ..solvers.qp import ActuatorSet, Projection, Rows, project
from . import CERTIFICATE_KINDS, FilterResult, InfeasibilityProof
from .barriers import (
    BrakingBarrier,
    CamBarrier,
    ClosingSpeedCbf,
    ConeEcbf,
    KeepOutEcbf,
    PassiveBarrier,
    SpeedCbf,
    _a_cw,
    braking_deceleration,
)
from .projections import ball, box
from .sampled import held_segment_bound, line_search
from .tightening import (
    EllipsoidError,
    ErrorSet,
    ecbf_bias,
    error_zonotope,
    first_order_bias,
    keep_out_margin,
)

_STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")
# Largest max(p) dt and gamma dt: the double integrator under hold loses the start set at 0.618
# and the barrier between samples at 0.787, and CW head-on matches it to 0.3%.
_POLE_RULE = 0.6
# The products p dt and gamma dt are compared with this relative slack, so a value set to the
# limit on purpose, such as 0.06 1/s at 10 s, is not refused by its rounding.
_RULE_SLACK = 1e-12
# Riding depth over 3 n^2 v_cap dt / (p1 p2): measured at most 0.94 over tangential directions
# and steps of 1 to 20 s; the buffer covers it with 20% to spare.
_RIDING_FACTOR = 1.2
# The speed may exceed v_cap between samples; the safe set's thrust budget assumes up to 10%.
_SPEED_OVERSHOOT = 1.1
# Changes below this fraction of a full command are rounding, not intervention.
_MOVED = 1e-6
# Samples of the held step, endpoints included, for the certified in-step bound.
_HELD_SAMPLES = 20
_SOFT_MAX_ROWS = 4
# Cutting planes for the end-of-step speed cap: at most this many, each the tangent of the cap
# tightened by this fraction, so the loop ends in a few cuts below the cap itself.
_MAX_CUTS = 16
_CUT_TOL = 1e-3
# Bisection steps on the keep-out row's tightening, as for the line search of the backups.
_TIGHTEN_STEPS = 12
_SPEED_TYPES = ("speed_limit", "closing_speed_limit")
BACKUP_BARRIERS = ("passive", "braking", "cam")
_BACKUP_NAMES = {"passive": "cbf-passive", "braking": "cbf-brake", "cam": "cbf-cam"}
BACKUP_CANDIDATES = ("clipped", "projection")


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value >= 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _exceedance(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError("exceedance must be in (0, 1)")
    return value


def _draw_level(value: float | None) -> float | None:
    return None if value is None else _positive(value, "per_draw_level")


def _truncation(
    exceedance: float | None, per_draw_level: float | None, navigation, execution: float
) -> tuple[float | None, float | None]:
    """Check p and b of a declared truncation: both or neither, and only with an error set."""
    p, b = _exceedance(exceedance), _draw_level(per_draw_level)
    if (p is None) != (b is None):
        raise ValueError("exceedance and per_draw_level are given together")
    # They state how the error sets were derived, so without a set they claim nothing.
    if p is not None and navigation is None and not execution > 0.0:
        raise ValueError("exceedance and per_draw_level need a declared error set")
    return p, b


def _pole_pair(poles) -> tuple[float, float]:
    values = tuple(poles)
    if len(values) != 2 or any(np.iscomplexobj(p) for p in values):
        raise ValueError("poles must be two real positive values")
    p1, p2 = (float(p) for p in values)
    if not all(math.isfinite(p) and p > 0.0 for p in (p1, p2)):
        raise ValueError("poles must be two real positive values")
    return p1, p2


def _pole_label(poles: tuple[float, float]) -> str:
    p1, p2 = poles
    return f"{p1:g}" if p1 == p2 else f"{p1:g}-{p2:g}"


def _numbered(kind: str, taken: int) -> str:
    """The row name of a constraint of type `kind` after `taken` others of that type."""
    return kind if taken == 0 else f"{kind}#{taken}"


def _check_rule(rate: float, dt: float, name: str) -> None:
    if rate * dt > _POLE_RULE * (1.0 + _RULE_SLACK):
        raise ValueError(
            f"{name} {rate:g} 1/s is too fast for dt {dt:g} s; the largest allowed is "
            f"{_POLE_RULE / dt:.6g} 1/s"
        )


@dataclass(frozen=True, kw_only=True)
class FilterDeclaration:
    """What a filter enforces, for which design and with what kind of guarantee.

    Parameters
    ----------
    name : str
        The filter's name.
    safe_set : str
        The declared safe set the filter's claims start from, in words; the filter's
        `contains(x)` tests membership of a Hill-frame state.
    barriers : tuple
        The barrier objects the filter evaluates, keep-out first.
    poles : tuple of float or None
        The exponential barrier's poles p1, p2, 1/s; None for a filter without one.
    gamma : float or None
        The class-K gain of the first-order rows, 1/s; None when the filter has none.
    design_dt : float
        The hold time of a command the filter was designed for, s.
    layout : str
        The thrust limit, "box" or "ball".
    a_max : float
        The thrust limit per axis (box) or in norm (ball), m/s^2.
    buffer_m : float
        The amount added to the keep-out radius, m.
    tightening : mapping
        The declared error sets: "navigation" (an error set on the state or None), "execution"
        (da_max, m/s^2) and "model_error" (m/s^2 or None).
    backup : str or None
        The backup command, "zero thrust", "braking along and across the line of sight, CW
        acceleration cancelled" or "zero thrust, else the CAM"; None without one.
    certificate : str or None
        The kind of guarantee of a step decided inside the safe set, one of `CERTIFICATE_KINDS`,
        or None when the filter makes no certified claim.
    counts_toward_certification : bool
        Whether the filter's runs may be counted as certified results.
    ignored : tuple of str
        Constraint types of the scenario the filter was told not to enforce.
    exceedance : float or None
        The episode-level probability p, in (0, 1), that the untruncated error models leave
        the truncation the declared sets were derived from; None when no truncation is
        declared.
    per_draw_level : float or None
        The truncation level b of each Gaussian draw of those models, in sigma, finite and
        positive; None when no truncation is declared.

    Raises
    ------
    ValueError
        If `certificate` is neither None nor one of `CERTIFICATE_KINDS`, `exceedance` is
        outside (0, 1), `per_draw_level` is not finite and positive, or the two are not given
        together with a navigation set or an execution bound above 0 in `tightening`.
    """

    name: str
    safe_set: str
    barriers: tuple[Any, ...]
    poles: tuple[float, float] | None
    gamma: float | None
    design_dt: float
    layout: str
    a_max: float
    buffer_m: float
    tightening: Mapping[str, Any]
    backup: str | None
    certificate: str | None
    counts_toward_certification: bool
    ignored: tuple[str, ...] = ()
    exceedance: float | None = None
    per_draw_level: float | None = None

    def __post_init__(self):
        if self.certificate is not None and (
            type(self.certificate) is not str or self.certificate not in CERTIFICATE_KINDS
        ):
            raise ValueError(
                f"unknown certificate {self.certificate!r}, expected {CERTIFICATE_KINDS}"
            )
        object.__setattr__(self, "tightening", MappingProxyType(dict(self.tightening)))
        p, b = _truncation(
            self.exceedance,
            self.per_draw_level,
            self.tightening.get("navigation"),
            self.tightening.get("execution") or 0.0,
        )
        object.__setattr__(self, "exceedance", p)
        object.__setattr__(self, "per_draw_level", b)


class _HeldCommandFilter:
    """Reset checks and the command conversions shared by the filters."""

    name: str
    _enforced: tuple[str, ...] = ("keep_out",)
    # A filter that predicts the state through the echoed commands checks the slots itself.
    _pending_allowed = False

    def __init__(
        self,
        *,
        execution: float,
        model_error: float | None,
        ignore,
        navigation: ErrorSet | None = None,
        exceedance: float | None = None,
        per_draw_level: float | None = None,
    ) -> None:
        self.execution = _nonnegative(execution, "execution")
        self.model_error = None if model_error is None else _nonnegative(model_error, "model_error")
        self.exceedance, self.per_draw_level = _truncation(
            exceedance, per_draw_level, navigation, self.execution
        )
        ignored = tuple(ignore)
        if not all(isinstance(kind, str) for kind in ignored):
            raise ValueError("ignore must list constraint type names")
        if "keep_out" in ignored:
            raise ValueError("keep_out cannot be ignored")
        self.ignore = ignored
        self._view: AgentView | None = None
        self._declaration: FilterDeclaration | None = None

    @property
    def da_max(self) -> float:
        """The declared acceleration error, execution plus model error, m/s^2."""
        return self.execution + (self.model_error or 0.0)

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

    def _reset_view(self, view: AgentView) -> list[Mapping[str, Any]]:
        """Check the view and keep what every filter needs; return the enforced constraints."""
        if view.action_mode not in CONTINUOUS_MODES:
            raise ValueError(f"filters need a continuous action mode, not {view.action_mode!r}")
        if view.frame != "hill":
            raise ValueError(f"filters need the hill frame, not {view.frame!r}")
        if tuple(view.observation[:6]) != _STATE_NAMES:
            raise ValueError("the observation must start with x, y, z, vx, vy, vz")
        if not self._pending_allowed and any(
            name.startswith("pending_") for name in view.observation
        ):
            raise ValueError("pending commands are not supported: the filter cannot predict delay")
        parts = view.scenario_id.split(":")
        dynamics = parts[2] if len(parts) > 2 else None
        if dynamics != "cw" and self.model_error is None:
            raise ValueError(f"dynamics {dynamics!r} is not cw; declare model_error")
        enforced = []
        for spec in view.constraints:
            kind = spec["type"]
            if kind in self.ignore:
                continue
            if kind not in self._enforced:
                raise ValueError(f"cannot enforce constraint {kind!r}; list it in ignore=")
            enforced.append(spec)
        radii = [float(spec["radius"]) for spec in enforced if spec["type"] == "keep_out"]
        if not radii:
            raise ValueError("the scenario has no keep_out constraint")
        self._view = view
        self._radius = max(radii)
        self._n = _positive(view.n, "n")
        self._dt = _positive(view.dt, "dt")
        self._actuator = ActuatorSet(view.max_accel, view.thrust_limit)
        self._scale = action_scale(view)
        self._commit = float(view.goal.get("commit_radius", 0.0))
        return enforced

    def _started(self) -> AgentView:
        if self._view is None:
            raise RuntimeError("reset the filter first")
        return self._view

    def _clipped(self, view: AgentView, action) -> np.ndarray:
        """The command as acceleration at the initial mass, m/s^2, on the thrust limit."""
        u = command_fraction(view, action) * view.max_accel
        if u.shape != (3,) or not np.all(np.isfinite(u)):
            raise ValueError("the action must be three finite numbers")
        return self._onto_limit(u)

    def _onto_limit(self, u: np.ndarray) -> np.ndarray:
        """`u` on the thrust limit; solvers meet it only to their tolerance, 1e-10 of a_max."""
        a_max = self._actuator.a_max
        return box(u, a_max) if self._actuator.layout == "box" else ball(u, a_max)

    def _action(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(u, dtype=float) / self._actuator.a_max * self._scale

    def _moved(self, u: np.ndarray, u_c: np.ndarray) -> bool:
        return float(np.linalg.norm(u - u_c)) > _MOVED * self._actuator.a_max

    def _monitor_only(self, x: np.ndarray) -> bool:
        """Inside the goal's commit radius the filter only watches."""
        return float(np.linalg.norm(x[:3])) <= self._commit

    def _monitor(self, u_c: np.ndarray, t0: float) -> FilterResult:
        return FilterResult(self._action(u_c), intervened=False, compute_s=time.perf_counter() - t0)


def _state(obs) -> np.ndarray:
    x = np.array(obs, dtype=float)[:6]
    if x.shape != (6,) or not np.all(np.isfinite(x)):
        raise ValueError("the observation must start with six finite state entries")
    return x


class CbfFilter(_HeldCommandFilter):
    """Exponential keep-out barrier and speed rows, checked on the exact held step.

    Parameters
    ----------
    poles : tuple of float
        The keep-out barrier's poles p1, p2, 1/s, real and positive; reset requires
        max(p1, p2) dt <= 0.6.
    gamma : float
        The class-K gain of the speed rows, 1/s, positive; reset requires gamma dt <= 0.6.
    v_cap : float
        The filter's own speed cap, m/s, positive; it bounds the riding buffer and the safe set.
    execution : float
        Bound on the execution error of a held command, m/s^2, nonnegative.
    model_error : float, optional
        Bound on the acceleration the CW model misses, m/s^2; required on other dynamics.
    navigation : None
        Navigation error sets are not supported by this filter (its rows are nonlinear in the
        state); anything but None raises.
    exceedance : float, optional
        The episode-level probability, in (0, 1), that the untruncated error models leave the
        truncation `execution` was derived from; given with `per_draw_level` and an `execution`
        above 0, and reported in the declaration only.
    per_draw_level : float, optional
        The truncation level of each Gaussian draw of those models, in sigma, finite and
        positive; given with `exceedance`, and reported in the declaration only.
    ignore : sequence of str
        Constraint types of the scenario not to enforce; every other type except keep_out,
        speed_limit, closing_speed_limit and approach_cone makes reset raise.

    Raises
    ------
    ValueError
        If a parameter is out of range, `navigation` is given, or `exceedance` and
        `per_draw_level` are not given together with an `execution` above 0; at reset, if the
        declared errors leave a speed row with v0 at or below zero.

    Notes
    -----
    The rows are the keep-out barrier h = rho - R_f with R_f = R + buffer,
    buffer = 1.2 * 3 n^2 v_cap dt / (p1 p2) + da_max / (p1 p2), the speed cap
    s = v_cap - |v| and a first-order row for each declared speed limit (Ames et al., sec. II,
    Def. 2, and sec. IV, Def. 7). The declared safe set is the barrier's start set at R_f with
    |v| <= v_cap and |a_cw(x)| + 1.1 (2 max(p) + v1) v_cap + da_max <= a_max, v1 the largest
    declared speed slope: without errors the command -a_cw + 2 max(p) c rhat - v1 c vhat, c the
    closing speed, meets every row inside the inscribed ball of the thrust limit, so the
    projection is feasible under box and ball. Each speed row lowered for the declared errors
    asks up to da_max more along its normal, which the da_max term reserves.

    Under a persistent acceleration error the first-order row s' >= -gamma s settles at
    s = -da_max / gamma (`tightening.first_order_bias`), so every speed row, the cap's and each
    declared limit's, enforces v0 lowered by that depth; the safe set keeps the limits as
    declared. Each declared approach cone adds the row of a `ConeEcbf` with the keep-out's poles
    wherever the state lies within the cone's active range of its apex, and the safe set adds the
    cone's start set there. The cone turns on at its active range, so no declared set is
    invariant across it: with a cone the safe set says "cone activation untreated" and the
    filter does not count toward certification. The held-step check covers the keep-out only;
    the cone rows rest on the pole rule.

    The rows are continuous-time conditions, and a command held for dt can break both: thrust
    across the velocity raises the speed at second order in dt, and the turn of rhat within the
    step changes rhat . u. Each decision therefore also caps the speed at the end of the held
    step on the exact CW solution, by cutting planes added to the projection, and checks the
    keep-out on the held step with `held_segment_bound` and the end range against R_f, both for
    every execution error up to da_max. A command that fails the check is replaced by the
    projection with the keep-out row raised by the least amount, found by bisection, that
    passes. If no raise passes, the most outward feasible command is returned with its predicted
    shortfall (m) as slack: as "solver_failed" inside the declared set when it is short by more
    than the keep-out resolution, and as "feasible" otherwise, so outside the set the status is
    "feasible" or "infeasible_certified". An "infeasible_certified" step carries the
    `InfeasibilityProof` of its rows. Recursive feasibility under hold is not proven, so
    `certificate` is None.
    """

    _enforced = ("keep_out", *_SPEED_TYPES, "approach_cone")
    _counts = True

    def __init__(
        self,
        *,
        poles=(0.03, 0.03),
        gamma: float = 0.03,
        v_cap: float = 0.2,
        execution: float = 0.0,
        model_error: float | None = None,
        navigation: ErrorSet | None = None,
        exceedance: float | None = None,
        per_draw_level: float | None = None,
        ignore=(),
    ) -> None:
        if navigation is not None:
            raise ValueError("CbfFilter does not tighten for navigation error; use BackupFilter")
        super().__init__(
            execution=execution,
            model_error=model_error,
            ignore=ignore,
            exceedance=exceedance,
            per_draw_level=per_draw_level,
        )
        self.poles = _pole_pair(poles)
        self.gamma = _positive(gamma, "gamma")
        self.v_cap = _positive(v_cap, "v_cap")
        self.name = f"cbf-ecbf-p{_pole_label(self.poles)}"

    def reset(self, view: AgentView) -> None:
        """Check the view against the design rules and build the rows.

        Raises
        ------
        ValueError
            If the action mode, frame, observation, dynamics or constraints are unsupported, the
            scenario has no keep_out, or max(poles) dt or gamma dt exceeds 0.6.
        """
        self._view = None
        self._declaration = None
        enforced = self._reset_view(view)
        dt = self._dt
        _check_rule(max(self.poles), dt, "pole")
        _check_rule(self.gamma, dt, "gamma")
        p1, p2 = self.poles
        n = self._n
        self.buffer_m = _RIDING_FACTOR * 3.0 * n * n * self.v_cap * dt / (p1 * p2) + ecbf_bias(
            self.da_max, self.poles
        )
        self._keep_out = KeepOutEcbf(radius=self._radius + self.buffer_m, poles=self.poles)
        depth = first_order_bias(self.da_max, self.gamma)
        speed_rows: list[tuple[str, SpeedCbf | ClosingSpeedCbf]] = [
            ("speed_cap", self._lowered(SpeedCbf, self.v_cap, 0.0, depth, "v_cap"))
        ]
        limits: list[SpeedCbf | ClosingSpeedCbf] = []
        cone_rows: list[tuple[str, ConeEcbf]] = []
        for spec in enforced:
            kind = spec["type"]
            if kind == "approach_cone":
                cone = ApproachCone(**{k: v for k, v in spec.items() if k != "type"})
                barrier = ConeEcbf(
                    half_angle=cone.half_angle,
                    axis=cone.axis,
                    apex=cone.apex,
                    active_range=cone.active_range,
                    poles=self.poles,
                )
                cone_rows.append((_numbered(kind, len(cone_rows)), barrier))
                continue
            if kind not in _SPEED_TYPES:
                continue
            barrier_type = SpeedCbf if kind == "speed_limit" else ClosingSpeedCbf
            v0, v1 = float(spec["v0"]), float(spec["v1"])
            limits.append(barrier_type(v0=v0, v1=v1, gamma=self.gamma))
            name = _numbered(kind, sum(row.split("#")[0] == kind for row, _ in speed_rows))
            speed_rows.append((name, self._lowered(barrier_type, v0, v1, depth, name)))
        self._speed_rows = tuple(speed_rows)
        self._limits = tuple(limits)
        self._cone_rows = tuple(cone_rows)
        self._v1 = max((b.v1 for b in limits), default=0.0)
        self._check_rows(1 + len(speed_rows) + len(cone_rows))
        phi, gamma_u = cw_discrete(n, dt)
        self._phi, self._gamma_u = phi, gamma_u
        # Largest position and velocity errors at the end of the step from the execution error.
        self._end_r = self.da_max * float(np.linalg.norm(gamma_u[:3], 2))
        self._end_v = self.da_max * float(np.linalg.norm(gamma_u[3:], 2))
        self._view = view
        safe_set = (
            "keep-out barrier start set at R + buffer, |v| <= v_cap, declared speed limits "
            "met, |a_cw| + 1.1 (2 max(p) + v1) v_cap + da_max <= a_max"
        )
        if cone_rows:
            safe_set += ", cone start sets within their active ranges; cone activation untreated"
        self._declaration = FilterDeclaration(
            name=self.name,
            safe_set=safe_set,
            barriers=(self._keep_out, *(b for _, b in speed_rows), *(b for _, b in cone_rows)),
            poles=self.poles,
            gamma=self.gamma,
            design_dt=dt,
            layout=self._actuator.layout,
            a_max=self._actuator.a_max,
            buffer_m=self.buffer_m,
            tightening={
                "navigation": None,
                "execution": self.execution,
                "model_error": self.model_error,
            },
            backup=None,
            certificate=None,
            counts_toward_certification=self._counts and not cone_rows,
            ignored=self.ignore,
            exceedance=self.exceedance,
            per_draw_level=self.per_draw_level,
        )

    def _lowered(
        self,
        barrier_type: type[SpeedCbf] | type[ClosingSpeedCbf],
        v0: float,
        v1: float,
        depth: float,
        name: str,
    ) -> SpeedCbf | ClosingSpeedCbf:
        """The speed barrier with v0 (m/s) lowered by the riding depth `depth`, m/s."""
        if not v0 - depth > 0.0:
            raise ValueError(
                f"the declared errors use up {name}: v0 {v0:g} m/s, riding depth {depth:g} m/s"
            )
        return barrier_type(v0=v0 - depth, v1=v1, gamma=self.gamma)

    def _check_rows(self, count: int) -> None:
        """Every row count is supported by the exact projection."""

    def contains(self, x) -> bool:
        """Return whether the Hill-frame state `x` (m, m/s) lies in the declared safe set.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        """
        self._started()
        x = np.asarray(x, dtype=float)
        if not self._keep_out.contains(x):
            return False
        if float(np.linalg.norm(x[3:])) > self.v_cap:
            return False
        if not all(barrier.contains(x) for barrier in self._limits):
            return False
        for _, cone in self._cone_rows:
            dist = float(np.linalg.norm(x[:3] - np.array(cone.apex)))
            # The cone's gradient is undefined at its apex, which the start set leaves out.
            if dist <= cone.active_range and (dist == 0.0 or not cone.contains(x)):
                return False
        budget = (2.0 * max(self.poles) + self._v1) * _SPEED_OVERSHOOT * self.v_cap + self.da_max
        return float(np.linalg.norm(_a_cw(x, self._n))) + budget <= self._actuator.a_max

    def contains_start(self, x) -> bool:
        """Return whether every estimate of the true Hill-frame state `x` lies in the safe set.

        The filter declares no navigation set, so the estimate is `x` and this is `contains`.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        """
        return self.contains(x)

    def _rows(self, x: np.ndarray) -> Rows:
        a_cw = _a_cw(x, self._n)
        normals, sides, names = [], [], []
        for name, barrier in (("keep_out", self._keep_out), *self._speed_rows, *self._cone_rows):
            row = barrier.row(x, a_cw)
            if row is None:
                continue
            normals.append(row[0])
            sides.append(row[1])
            names.append(name)
        return Rows(np.reshape(normals, (-1, 3)), np.array(sides, dtype=float), tuple(names))

    def filter(self, obs: np.ndarray, action: np.ndarray) -> FilterResult:
        """Return the command nearest the agent's that meets the rows under the thrust limit.

        Parameters
        ----------
        obs : np.ndarray
            The observation; its first six entries are the Hill-frame state, m and m/s.
        action : np.ndarray
            The agent's command in the view's action mode.

        Returns
        -------
        FilterResult
            The filtered action in the view's action mode with its status. `recovering` is True
            outside the declared safe set; `certificate` is None.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        ValueError
            If the observation or the action is malformed.
        """
        t0 = time.perf_counter()
        view = self._started()
        x = _state(obs)
        u_c = self._clipped(view, action)
        if self._monitor_only(x):
            return self._monitor(u_c, t0)
        rows = self._rows(x)
        recovering = not self.contains(x)
        return self._decide(x, u_c, rows, recovering, t0)

    def _decide(
        self, x: np.ndarray, u_c: np.ndarray, rows: Rows, recovering: bool, t0: float
    ) -> FilterResult:
        found, excess, used = self._solve(x, u_c, rows, 0.0)
        if found.status == "feasible" and excess > 0.0:
            # The cuts ran out above the cap: a limit of the filter inside its set, and outside
            # it the predicted excess (m/s) the step accepts while recovering.
            return self._result(
                found.u,
                u_c,
                t0,
                recovering,
                active=True,
                status="feasible" if recovering else "solver_failed",
                slack=excess,
            )
        if found.status == "feasible":
            shortfall = self._shortfall(x)
            if shortfall(found.u) == 0.0:
                active = _row_active(found)
                # With no active row the projection of the clipped command is the command itself.
                u = found.u if active else u_c
                return self._result(u, u_c, t0, recovering, active=active)
            return self._tightened(x, u_c, rows, recovering, t0, shortfall)
        if found.status == "infeasible_certified":
            return self._result(
                found.u,
                u_c,
                t0,
                recovering,
                active=True,
                status="infeasible_certified",
                slack=float(found.phase1_margin),
                conflicting=found.conflicting,
                proof=self._proof(found, used),
            )
        # No answer and no proof: fall back to the command that does most for the keep-out row.
        u = self._support(rows.a[0])
        return self._result(
            u,
            u_c,
            t0,
            recovering,
            active=True,
            status="solver_failed",
            slack=_violation(rows, u),
            fallback_level=2,
        )

    def _solve(
        self, x: np.ndarray, u_c: np.ndarray, rows: Rows, tighten: float
    ) -> tuple[Projection, float, Rows]:
        """Project onto the rows, the keep-out row raised by `tighten` (m/s^2), and the cap on
        the speed at the end of the held step; return the projection, the end speed's excess
        over the cap (m/s), nonzero only when the cuts ran out, and the rows projected onto.

        |v(dt)| <= v is convex in u but not linear, so it is imposed by cutting planes: each
        projection that ends above the cap adds the tangent g(u_k) + grad g . (u - u_k) <= 0 of
        g(u) = |phi_v x + gamma_v u| - v at v = 0.999 times the cap, which every command meeting
        that tightened cap also meets. An infeasible set of cuts therefore proves the cap
        tightened by 0.1% infeasible with the rows, not the cap itself.
        """
        a, b, names = rows.a, rows.b.copy(), rows.names
        b[0] += tighten
        speed_now = float(np.linalg.norm(x[3:]))
        # Above the cap the step may not speed up; the end speed is also lowered by the
        # execution error's share.
        limit = max(self.v_cap - self._end_v, speed_now)
        drift_v = self._phi[3:] @ x
        gamma_v = self._gamma_u[3:]
        cut_a: list[np.ndarray] = []
        cut_b: list[float] = []
        while True:
            used = Rows(
                np.vstack([a, *cut_a]) if cut_a else a,
                np.concatenate([b, cut_b]),
                names + tuple(f"speed_end#{i}" for i in range(len(cut_a))),
            )
            found = project(u_c, used, self._actuator)
            if found.status != "feasible":
                return found, 0.0, used
            v_end = drift_v + gamma_v @ found.u
            speed = float(np.linalg.norm(v_end))
            if speed <= limit:
                return found, 0.0, used
            if len(cut_a) == _MAX_CUTS:
                return found, speed - limit, used
            # Cutting a little inside the cap ends the loop in a few cuts; a stop above the cap
            # would raise the next step's limit and ratchet the speed up.
            grad = gamma_v.T @ (v_end / speed)
            cut_a.append(-grad)
            cut_b.append(speed - limit * (1.0 - _CUT_TOL) - float(grad @ found.u))

    def _shortfall(self, x: np.ndarray):
        """The predicted keep-out shortfall (m) of a held command, 0 when it passes.

        A command passes when the certified bound on the keep-out margin over the held step
        reaches min(0, rho - R) and the end range reaches min(0, rho - R_f), both for every
        execution error up to da_max.
        """
        rho = float(np.linalg.norm(x[:3]))
        floor_s = min(0.0, rho - self._radius)
        floor_f = min(0.0, rho - self._keep_out.radius)
        drift_r = self._phi[:3] @ x
        gamma_r = self._gamma_u[:3]

        def shortfall(u: np.ndarray) -> float:
            held = held_segment_bound(
                x, u, self._dt, _HELD_SAMPLES, self._radius, self._n, da_max=self.da_max
            )
            end = float(np.linalg.norm(drift_r + gamma_r @ u)) - self._keep_out.radius
            return max(0.0, floor_s - held, floor_f - (end - self._end_r))

        return shortfall

    def _tightened(
        self, x: np.ndarray, u_c: np.ndarray, rows: Rows, recovering: bool, t0: float, shortfall
    ) -> FilterResult:
        """Raise the keep-out row by the least amount that passes the held-step check.

        Raising the row only shrinks the feasible set, so the largest feasible raise is found by
        bisection first, up to the row's support over the thrust limit; the least passing raise
        below it follows by a second bisection, which assumes the check passes on an interval.
        Without a passing raise the most outward feasible command is returned with its
        predicted shortfall (m) as slack.
        """
        w = rows.a[0]
        reach = self._actuator.a_max * float(
            np.sum(np.abs(w)) if self._actuator.layout == "box" else np.linalg.norm(w)
        )
        top = max(0.0, reach - float(rows.b[0]))

        def solved(tighten: float) -> Projection | None:
            """The projection at this raise, or None when it is infeasible or over the cap."""
            found, excess, _ = self._solve(x, u_c, rows, tighten)
            return found if found.status == "feasible" and excess == 0.0 else None

        best = solved(top)
        if best is None:
            lo, hi = 0.0, top
            best = solved(0.0)
            # The unraised rows were solved within the cap before this method was called.
            assert best is not None
            for _ in range(_TIGHTEN_STEPS):
                mid = 0.5 * (lo + hi)
                trial = solved(mid)
                if trial is not None:
                    lo, best = mid, trial
                else:
                    hi = mid
            top = lo
        missing = shortfall(best.u)
        if missing > 0.0:
            # Outside the declared set, or short by less than the keep-out resolution (the slack
            # of the certified bound at the boundary), this is the predicted shortfall the step
            # accepts, not a defect.
            defect = not recovering and missing > KeepOut.resolution
            return self._result(
                best.u,
                u_c,
                t0,
                recovering,
                active=True,
                status="solver_failed" if defect else "feasible",
                slack=missing,
                fallback_level=2,
            )
        lo, hi = 0.0, top
        for _ in range(_TIGHTEN_STEPS):
            mid = 0.5 * (lo + hi)
            trial = solved(mid)
            if trial is not None and shortfall(trial.u) == 0.0:
                hi, best = mid, trial
            else:
                lo = mid
        return self._result(best.u, u_c, t0, recovering, active=True)

    def _proof(self, found: Projection, used: Rows) -> InfeasibilityProof | None:
        """The Farkas certificate of `project` as a proof on the rows it names, in m/s^2.

        The rows are already in SI (normals per m/s^2, right-hand sides in m/s^2), and the
        certificate's leading entries are the multipliers of those rows as given. The proof keeps
        the rows `project` names as conflicting; when dropping the others' negligible multipliers
        breaks the margin, it keeps every row.
        """
        if found.certificate is None:
            return None
        lam = np.asarray(found.certificate, dtype=float)[: len(used.b)]
        keep = np.array([name in found.conflicting for name in used.names], dtype=bool)
        for mask in (keep, np.ones(len(used.b), dtype=bool)):
            if not mask.any():
                continue
            proof = InfeasibilityProof(
                normals=used.a[mask],
                bounds=used.b[mask],
                multipliers=lam[mask],
                layout=self._actuator.layout,
                a_max=self._actuator.a_max,
                names=tuple(np.array(used.names, dtype=object)[mask]),
            )
            if proof.verify():
                return proof
        return None

    def _support(self, w: np.ndarray) -> np.ndarray:
        """The command in the thrust limit that maximizes w . u, m/s^2."""
        a_max = self._actuator.a_max
        if self._actuator.layout == "box":
            return a_max * np.sign(w)
        return a_max * w / float(np.linalg.norm(w))

    def _result(
        self,
        u: np.ndarray,
        u_c: np.ndarray,
        t0: float,
        recovering: bool,
        *,
        active: bool,
        status: str = "feasible",
        slack: float = 0.0,
        conflicting: tuple[str, ...] = (),
        fallback_level: int = 0,
        proof: InfeasibilityProof | None = None,
    ) -> FilterResult:
        u = self._onto_limit(u)
        return FilterResult(
            self._action(u),
            intervened=self._moved(u, u_c),
            status=status,
            slack=slack,
            conflicting=tuple(conflicting),
            compute_s=time.perf_counter() - t0,
            active=active,
            recovering=recovering,
            fallback_level=fallback_level,
            proof=proof,
        )


def _row_active(found: Projection) -> bool:
    return bool(np.any(found.multipliers > 0.0))


def _violation(rows: Rows, u: np.ndarray) -> float:
    """The largest row violation b_i - a_i . u at `u`, m/s^2, or 0."""
    if len(rows.b) == 0:
        return 0.0
    return max(0.0, float(np.max(rows.b - rows.a @ u)))


class SoftCbfFilter(CbfFilter):
    """The rows of `CbfFilter` as an exact penalty instead of hard constraints.

    Parameters
    ----------
    alpha : float
        The penalty weight on each row's violation, m/s^2, positive. Above the sum of the hard
        projection's multipliers the filter returns the hard projection whenever it exists.
    poles, gamma, v_cap, execution, model_error, navigation, exceedance, per_draw_level, ignore
        As for `CbfFilter`.

    Raises
    ------
    ValueError
        If a parameter is out of range, or at reset if the scenario gives more than four rows.

    Notes
    -----
    The filter solves min 1/2 |u - u_nom|^2 + alpha sum_i max(0, b_i - a_i . u) over the thrust
    limit (Boyd and Vandenberghe, *Convex Optimization*, exercise 5.16). The problem is strongly
    convex, so its solution is unique, and at it the rows split into a set V weighted by alpha
    and the rest, held as hard rows: the solution is the projection of u_nom + alpha sum_V a_i
    onto the rows outside V. Every V of the k <= 4 rows is tried and the candidate with the
    lowest penalized objective is returned. A violated row is reported in `slack` (the summed
    violation, m/s^2) and `conflicting`; the status is "feasible", since the penalized problem
    always has a solution. A soft barrier has no invariance guarantee, so the filter never counts
    toward certification.
    """

    _counts = False

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        poles=(0.03, 0.03),
        gamma: float = 0.03,
        v_cap: float = 0.2,
        execution: float = 0.0,
        model_error: float | None = None,
        navigation: ErrorSet | None = None,
        exceedance: float | None = None,
        per_draw_level: float | None = None,
        ignore=(),
    ) -> None:
        super().__init__(
            poles=poles,
            gamma=gamma,
            v_cap=v_cap,
            execution=execution,
            model_error=model_error,
            navigation=navigation,
            exceedance=exceedance,
            per_draw_level=per_draw_level,
            ignore=ignore,
        )
        self.alpha = _positive(alpha, "alpha")
        self.name = f"cbf-ecbf-soft-p{_pole_label(self.poles)}"

    def _check_rows(self, count: int) -> None:
        if count > _SOFT_MAX_ROWS:
            raise ValueError(f"the soft filter takes at most {_SOFT_MAX_ROWS} rows, got {count}")

    def _decide(
        self, x: np.ndarray, u_c: np.ndarray, rows: Rows, recovering: bool, t0: float
    ) -> FilterResult:
        k = len(rows.b)
        a_max = self._actuator.a_max
        scale = max(a_max, float(np.max(np.abs(rows.b), initial=0.0)))
        best: tuple[float, np.ndarray, bool] | None = None
        for weighted in itertools.product((False, True), repeat=k):
            mask = np.array(weighted, dtype=bool)
            target = u_c + self.alpha * rows.a[mask].sum(axis=0)
            hard = Rows(rows.a[~mask], rows.b[~mask], tuple(np.array(rows.names)[~mask]))
            found = project(target, hard, self._actuator)
            u = found.u
            excess = np.maximum(rows.b - rows.a @ u, 0.0)
            objective = 0.5 * float(np.sum((u - u_c) ** 2)) + self.alpha * float(excess.sum())
            # Candidates that reach the same point differ only by rounding; the first one, with
            # the fewest weighted rows, is kept.
            if best is None or objective < best[0] - 1e-12 * scale * scale:
                active = bool(mask.any()) or (found.status == "feasible" and _row_active(found))
                best = (objective, u, active)
        assert best is not None
        _, u, active = best
        excess = rows.b - rows.a @ u
        violated = excess > 1e-10 * scale
        return self._result(
            u,
            u_c,
            t0,
            recovering,
            active=active,
            slack=float(np.sum(excess[violated])),
            conflicting=tuple(np.array(rows.names)[violated]),
        )


class BackupFilter(_HeldCommandFilter):
    """Held-step check on the exact CW solution with a line search toward a backup command.

    Parameters
    ----------
    barrier : str
        "passive" (free drift, backup zero thrust), "braking" (backup braking along and across
        the line of sight; see `BrakingBarrier`) or "cam" (backup zero thrust where free drift
        is certified, else the CAM; see `CamBarrier`).
    candidate : str
        "clipped" tries the clipped command first; "projection", with the passive barrier only,
        first projects it onto the first-order row grad h_p . f(x, u) >= -gamma h_p, which makes
        the correction smaller.
    gamma : float
        The class-K gain of the projection candidate's row, 1/s, positive.
    a_b : float, optional
        The braking barrier's braking acceleration, m/s^2, positive and at most
        `braking_deceleration` at the view's thrust limit and dt. By default that value minus
        da_max.
    rho_cap : float
        The largest range of the braking set, m.
    v_cap : float
        The speed cap of the braking set, m/s.
    grid : float
        Spacing of the free-drift samples, s; without error sets it is shortened to dt / k for
        the smallest whole k that reaches it.
    t_cap : float, optional
        The longest free-drift horizon, s; ten orbits by default, within which the drift must
        reach the terminal set, or two orbits when error sets are declared, and then the horizon
        of every certificate.
    samples : int
        Samples of the held step for the certified in-step bound, endpoints included.
    iterations : int
        Bisection steps of the line search.
    navigation : BoxError, BallError or EllipsoidError, optional
        Bound on the error of the observed state, m and m/s; with the passive barrier it makes
        the barrier set-valued.
    execution : float
        Bound on the execution error of a held command, m/s^2, nonnegative.
    model_error : float, optional
        Bound on the acceleration the CW model misses, m/s^2; required on other dynamics.
    exceedance : float, optional
        The episode-level probability, in (0, 1), that the untruncated error models leave the
        truncation the error sets were derived from; given with `per_draw_level` and an error
        set, and reported in the declaration only.
    per_draw_level : float, optional
        The truncation level of each Gaussian draw of those models, in sigma, finite and
        positive; given with `exceedance`, and reported in the declaration only.
    ignore : sequence of str
        Constraint types of the scenario not to enforce; any other type than keep_out makes
        reset raise, since the backup may leave it.

    Raises
    ------
    ValueError
        If a parameter is out of range, `samples` or `iterations` is not a finite integer, the
        projection candidate is asked for a barrier other than the passive one, the cam barrier
        is asked for with error sets, or `exceedance` and `per_draw_level` are not given together
        with an error set.

    Notes
    -----
    A command u is accepted when (i) the certified lower bound on the keep-out margin over the
    held step is at least floor_s, (ii) the barrier value at the end of the step is at least
    floor_b and, for the braking barrier, (iii) the end state keeps h_o >= min(0, h_o(x)) and
    |v| <= max(v_cap, |v(x)|), all tightened for the declared error sets but the speed. The
    floors are 0 inside the barrier's set and min(0, current value) outside it, so a recovering
    step does not get worse. The line search goes from the candidate toward the backup beta,
    which is not checked inside the set: without error sets the free-drift set certifies the
    drift for all time and is kept by a drift step, since the grid divides dt (sec. III,
    Theorem 4 with beta = 0), and the braking backup keeps h_b >= 0, h_o >= 0 and |v| <= v_cap
    over every held step (`BrakingBarrier`). Without error sets the filter is therefore
    recursively feasible under box and ball limits, at any dt for free drift and at every dt
    `braking_deceleration` admits for braking; the free-drift backup is returned inside the set
    without the check, and a braking backup that fails it there is reported as "solver_failed"
    with no certificate. With error sets the free-drift set is certified up to t_cap only and
    recursive feasibility is not claimed. The certificate is "bounded-W" with error sets and
    "chance-epsilon" when the navigation set is an ellipsoid with a positive exceedance, since
    the error leaves it with that probability per draw.

    With a navigation set E and the passive barrier the set is that of the estimate x_hat whose
    whole error set drifts clear: `vbar.reach.drift_clearance` of the zonotope x_hat + Z_E, with
    Z_E from `tightening.error_zonotope` (exact for a box, an outer bound otherwise), certified
    >= 0 over t_cap on samples at most `grid` s apart. The end of a held step u is checked by the
    same clearance of Phi (x_hat + Z_E) + Gamma u + Gamma da_max B, B the unit box, which holds
    every true end state under the declared errors; the held step itself keeps the scalar margin
    of `held_segment_bound`. With an execution error alone the end check subtracts the largest
    range error of the drift instead.

    The cam barrier holds the CAM a_c of magnitude a_max along the direction
    `vbar.metrics.passive.cam_direction` gives for the view's goal (along-track away from a V-bar
    approach, radially away with a fifth retrograde from an R-bar one) for
    m = ceil(0.3 m/s / (a_max dt)) whole steps, the 0.3 m/s CAM of the passive-safety metrics.
    Its set is kept by its backup (`CamBarrier`), so it too returns the backup inside the set
    without the check and is recursively feasible at any dt under box and ball limits. Its
    claim is nominal only, so it refuses error sets.
    """

    def __init__(
        self,
        *,
        barrier: str,
        candidate: str = "clipped",
        gamma: float = 0.05,
        a_b: float | None = None,
        rho_cap: float = 2000.0,
        v_cap: float = 0.3,
        grid: float = 5.0,
        t_cap: float | None = None,
        samples: int = _HELD_SAMPLES,
        iterations: int = 12,
        navigation: ErrorSet | None = None,
        execution: float = 0.0,
        model_error: float | None = None,
        exceedance: float | None = None,
        per_draw_level: float | None = None,
        ignore=(),
    ) -> None:
        if barrier not in BACKUP_BARRIERS:
            raise ValueError(f"unknown barrier {barrier!r}, expected {BACKUP_BARRIERS}")
        if candidate not in BACKUP_CANDIDATES:
            raise ValueError(f"unknown candidate {candidate!r}, expected {BACKUP_CANDIDATES}")
        if candidate == "projection" and barrier != "passive":
            raise ValueError("the projection candidate needs the passive barrier")
        super().__init__(
            execution=execution,
            model_error=model_error,
            ignore=ignore,
            navigation=navigation,
            exceedance=exceedance,
            per_draw_level=per_draw_level,
        )
        if (
            isinstance(samples, bool)
            or not math.isfinite(samples)
            or int(samples) != samples
            or samples < 2
        ):
            raise ValueError("samples must be an integer of at least 2")
        if (
            isinstance(iterations, bool)
            or not math.isfinite(iterations)
            or int(iterations) != iterations
            or iterations < 0
        ):
            raise ValueError("iterations must be a nonnegative integer")
        self.barrier = barrier
        self.candidate = candidate
        self.gamma = _positive(gamma, "gamma")
        self.a_b = None if a_b is None else _positive(a_b, "a_b")
        self.rho_cap = _positive(rho_cap, "rho_cap")
        self.v_cap = _positive(v_cap, "v_cap")
        self.grid = _positive(grid, "grid")
        self.t_cap = None if t_cap is None else _positive(t_cap, "t_cap")
        self.samples = int(samples)
        self.iterations = int(iterations)
        self.navigation = navigation
        if barrier == "cam" and self.bounded:
            raise ValueError("the cam barrier makes no claim under error sets")
        self.name = _BACKUP_NAMES[barrier]

    @property
    def bounded(self) -> bool:
        """Whether error sets are declared, which makes every check tightened."""
        return self.navigation is not None or self.da_max > 0.0

    @property
    def _certificate(self) -> str:
        """The kind of guarantee of a step decided inside the set."""
        # An ellipsoid left with a stated probability bounds the error only with that chance.
        if isinstance(self.navigation, EllipsoidError) and self.navigation.exceedance > 0.0:
            return "chance-epsilon"
        return "bounded-W" if self.bounded else "nominal"

    def reset(self, view: AgentView) -> None:
        """Check the view and build the barrier for its dt and thrust limit.

        Raises
        ------
        ValueError
            If the action mode, frame, observation, dynamics or constraints are unsupported, the
            scenario has no keep_out, no braking barrier exists for the view (a_b <= 0), an
            explicit a_b is above `braking_deceleration`, or the cam barrier is asked for a goal
            with no approach direction.
        """
        self._view = None
        self._declaration = None
        self._reset_view(view)
        n, dt, radius = self._n, self._dt, self._radius
        phi, gamma = cw_discrete(n, dt)
        self._phi, self._gamma_u = phi, gamma
        # Generators of the navigation set about the estimate and of the end-of-step set about
        # its prediction, for the set-valued free-drift barrier; None for a point barrier.
        self._start_set: np.ndarray | None = None
        self._end_set: np.ndarray | None = None
        self._barrier: PassiveBarrier | BrakingBarrier | CamBarrier
        if self.barrier == "cam":
            self._barrier = self._cam_barrier(view)
            self._end_tightening = 0.0
            backup = "zero thrust, else the CAM"
            safe_set = (
                f"a CAM of at most {self._barrier.steps} steps, then free drift, certified "
                "outside the keep-out for all time"
            )
        elif self.barrier == "passive":
            t_cap = self.t_cap
            if t_cap is None:
                t_cap = (4.0 if self.bounded else 20.0) * math.pi / n
            if self.navigation is not None:
                self._barrier = PassiveBarrier(radius=radius, n=n, grid=self.grid, t_cap=t_cap)
                self._end_tightening = 0.0
                self._start_set = error_zonotope(self.navigation).generators
                end = [phi @ self._start_set]
                if self.da_max > 0.0:
                    end.append(self.da_max * gamma)
                self._end_set = np.hstack(end)
                self._drift_samples = math.ceil(t_cap / self.grid) + 1
                safe_set = (
                    "free drift of every state within the navigation error of the estimate "
                    "certified outside the keep-out up to the drift horizon"
                )
            elif self.bounded:
                self._barrier = PassiveBarrier(radius=radius, n=n, grid=self.grid, t_cap=t_cap)
                self._end_tightening = self._drift_tightening(phi, gamma)
                safe_set = "certified free drift outside the keep-out up to the drift horizon"
            else:
                # The drift samples after a step must be samples of the drift before it, so the
                # grid divides dt and the set is kept by the step.
                grid = dt / math.ceil(dt / self.grid)
                self._barrier = PassiveBarrier(
                    radius=radius, n=n, grid=grid, t_cap=t_cap, terminal=True
                )
                self._end_tightening = 0.0
                safe_set = "free drift certified outside the keep-out for all time"
            # The projection row is the first-order condition on the drift minimum, so it takes
            # the capped value, whose gradient `gradient` gives; the terminal value lies below it.
            self._row_barrier = self._barrier
            if self.candidate == "projection" and not self.bounded:
                self._row_barrier = PassiveBarrier(
                    radius=radius, n=n, grid=self._barrier.grid, t_cap=t_cap
                )
            backup = "zero thrust"
        else:
            a_max = self._actuator.a_max
            derived = braking_deceleration(
                a_max=a_max, v_cap=self.v_cap, dt=dt, radius=radius, rho_cap=self.rho_cap, n=n
            )
            a_b = self.a_b
            if a_b is None:
                a_b = derived - self.da_max
                if not a_b > 0.0:
                    raise ValueError(
                        f"no braking barrier at dt {dt:g} s: a_b = {a_b:.3g} m/s^2; "
                        "lower v_cap, rho_cap or dt"
                    )
            elif a_b > derived:
                raise ValueError(
                    f"a_b above what the braking backup delivers at dt {dt:g} s, "
                    f"{derived:.4g} m/s^2"
                )
            self._barrier = BrakingBarrier(
                radius=radius,
                a_b=a_b,
                rho_cap=self.rho_cap,
                v_cap=self.v_cap,
                n=n,
                dt=dt,
                a_max=a_max,
            )
            self._end_deviation = self._deviation(phi[:3], phi[3:], gamma[:3], gamma[3:])
            backup = "braking along and across the line of sight, CW acceleration cancelled"
            safe_set = "braking barriers h_b >= 0 and h_o >= 0 with |v| <= v_cap"
        self._view = view
        self._declaration = FilterDeclaration(
            name=self.name,
            safe_set=safe_set,
            barriers=(self._barrier,),
            poles=None,
            gamma=self.gamma if self.candidate == "projection" else None,
            design_dt=dt,
            layout=self._actuator.layout,
            a_max=self._actuator.a_max,
            buffer_m=0.0,
            tightening={
                "navigation": self.navigation,
                "execution": self.execution,
                "model_error": self.model_error,
            },
            backup=backup,
            certificate=self._certificate,
            counts_toward_certification=True,
            ignored=self.ignore,
            exceedance=self.exceedance,
            per_draw_level=self.per_draw_level,
        )

    def _cam_barrier(self, view: AgentView) -> CamBarrier:
        """The CAM-then-drift barrier for the view's goal, thrust limit and dt."""
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
            grid=dt / math.ceil(dt / self.grid),
            t_cap=self.t_cap,
            samples=self.samples,
        )

    def _deviation(self, phi_r, phi_v, gamma_r, gamma_v) -> tuple[float, float]:
        """Bounds on the position (m) and velocity (m/s) error of a predicted state."""
        da = self.da_max
        g_r = da * float(np.linalg.norm(gamma_r, 2))
        g_v = da * float(np.linalg.norm(gamma_v, 2))
        if self.navigation is not None:
            g_r += keep_out_margin(phi_r, self.navigation)
            g_v += keep_out_margin(phi_v, self.navigation)
        return g_r, g_v

    def _drift_tightening(self, phi: np.ndarray, gamma: np.ndarray) -> float:
        """The largest range error of the free drift from the end of the step, m.

        The error at the end of the step is phi e + gamma da; the drift carries it by its own
        transition matrices, sampled every `grid` s, and between samples the error's position
        grows at most at its speed, bounded as in `held_segment_bound`.
        """
        n, h = self._n, self.grid
        barrier = self._barrier
        assert isinstance(barrier, PassiveBarrier) and barrier.t_cap is not None
        tau = h * np.arange(math.ceil(barrier.t_cap / h) + 1)
        drift = cw_discrete(n, tau)[0]
        g_r = np.empty(len(tau))
        g_v = np.empty(len(tau))
        for i, phi_tau in enumerate(drift):
            through = phi_tau @ phi
            pushed = phi_tau @ gamma
            g_r[i], g_v[i] = self._deviation(through[:3], through[3:], pushed[:3], pushed[3:])
        denominator = 1.0 - n * h - 0.75 * n * n * h * h
        w = (float(g_v.max()) + 0.5 * h * 3.0 * n * n * float(g_r.max())) / denominator
        return float(np.max(0.5 * (g_r[:-1] + g_r[1:] + w * h)))

    def contains(self, x) -> bool:
        """Return whether the Hill-frame state `x` (m, m/s) lies in the barrier's set.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        """
        self._started()
        x = np.asarray(x, dtype=float)
        if self._start_set is not None:
            return self._set_value(x, self._start_set) >= 0.0
        return self._barrier.contains(x)

    def contains_start(self, x) -> bool:
        """Return whether every estimate x + e, e in the navigation set, lies in the set.

        A start the audit draws for the true state `x` (m, m/s) passes when the filter's first
        estimate is in its declared set whatever the navigation error. With the free-drift
        barrier every estimate's set x + e + Z_E lies in x + 2 Z_E, so this is the clearance of
        x + 2 Z_E. With the braking barrier every x + e is in the set when |v| + g_v <= v_cap
        and the lower bounds on h_b and h_o of states within g_r and g_v of x are nonnegative,
        g_r and g_v the largest position and velocity errors in the set. Without a navigation
        set this is `contains(x)`.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        """
        self._started()
        x = np.asarray(x, dtype=float)
        if self.navigation is None:
            return self.contains(x)
        if self._start_set is not None:
            return self._set_value(x, 2.0 * self._start_set) >= 0.0
        barrier = self._barrier
        assert isinstance(barrier, BrakingBarrier)
        axes = np.eye(6)
        g_r = keep_out_margin(axes[:3], self.navigation)
        g_v = keep_out_margin(axes[3:], self.navigation)
        if float(np.linalg.norm(x[3:])) + g_v > barrier.v_cap:
            return False
        return self._braking_bound(x, g_r, g_v) >= 0.0 and self._outer_bound(x, g_r, g_v) >= 0.0

    def _set_value(self, center: np.ndarray, generators: np.ndarray) -> float:
        """The certified free-drift clearance of every state in the zonotope, m."""
        barrier = self._barrier
        assert isinstance(barrier, PassiveBarrier) and barrier.t_cap is not None
        found = drift_clearance(
            self._n, Zonotope(center, generators), self._radius, barrier.t_cap, self._drift_samples
        )
        return found.certified

    def _end_value(self, x1: np.ndarray) -> float:
        """A lower bound on the barrier value at the true end state of the step, m."""
        barrier = self._barrier
        if self._end_set is not None:
            return self._set_value(x1, self._end_set)
        if isinstance(barrier, (PassiveBarrier, CamBarrier)):
            return float(barrier.value(x1)) - self._end_tightening
        return self._braking_bound(x1, *self._end_deviation)

    def _braking_bound(self, x: np.ndarray, g_r: float, g_v: float) -> float:
        """A lower bound on h_b over the states within g_r (m) and g_v (m/s) of `x`, m."""
        barrier = self._barrier
        assert isinstance(barrier, BrakingBarrier)
        if g_r == 0.0 and g_v == 0.0:
            return barrier.value(x)
        rho = float(np.linalg.norm(x[:3]))
        if rho <= g_r:
            return -math.inf
        speed = float(np.linalg.norm(x[3:]))
        # rhat moves by at most 2 |dr| / rho, so rho_dot moves by at most this much.
        closing = max(0.0, -float(x[:3] @ x[3:]) / rho) + g_v + 2.0 * g_r * (speed + g_v) / rho
        return rho - g_r - barrier.radius - closing * closing / (2.0 * barrier.a_b)

    def _outer_bound(self, x: np.ndarray, g_r: float, g_v: float) -> float:
        """A lower bound on h_o over the states within g_r (m) and g_v (m/s) of `x`, m."""
        barrier = self._barrier
        assert isinstance(barrier, BrakingBarrier)
        if g_r == 0.0 and g_v == 0.0:
            return barrier.outer_value(x)
        rho = float(np.linalg.norm(x[:3]))
        if rho <= g_r:
            return -math.inf
        speed = float(np.linalg.norm(x[3:]))
        opening = max(0.0, float(x[:3] @ x[3:]) / rho) + g_v + 2.0 * g_r * (speed + g_v) / rho
        return barrier.rho_cap - rho - g_r - opening * opening / (2.0 * barrier.a_b)

    def _kept(self, x1: np.ndarray, v_top: float, floor_o: float) -> bool:
        """Whether the end state `x1` keeps the speed cap and the outer barrier of the set."""
        if float(np.linalg.norm(x1[3:])) > v_top:
            return False
        return self._outer_bound(x1, *self._end_deviation) >= floor_o

    def _margins(
        self, x: np.ndarray, u: np.ndarray, floor_s: float = -math.inf
    ) -> tuple[float, float]:
        """The held-step bound and, when it reaches `floor_s`, the end value; m."""
        held = held_segment_bound(
            x,
            u,
            self._dt,
            self.samples,
            self._radius,
            self._n,
            error=self.navigation,
            da_max=self.da_max,
        )
        if held < floor_s:
            return held, -math.inf
        return held, self._end_value(self._phi @ x + self._gamma_u @ u)

    def _candidate(self, x: np.ndarray, u_c: np.ndarray, h0: float) -> tuple[np.ndarray, bool]:
        """The command the line search starts from and whether a row moved it."""
        if self.candidate == "clipped":
            return u_c, False
        drift = self._row_barrier
        try:
            grad = drift.gradient(x)
        except ValueError:
            # No unique drift minimiser (closed orbit or a minimum at an end): no smooth row.
            return u_c, False
        if drift is not self._barrier:
            h0 = float(drift.value(x))
        f = np.concatenate([x[3:], _a_cw(x, self._n)])
        rows = Rows(grad[None, 3:], np.array([-self.gamma * h0 - float(grad @ f)]), ("passive",))
        found = project(u_c, rows, self._actuator)
        if found.status != "feasible" or not _row_active(found):
            return u_c, False
        return found.u, True

    def filter(self, obs: np.ndarray, action: np.ndarray) -> FilterResult:
        """Return the command nearest the candidate, toward the backup, that passes the checks.

        Parameters
        ----------
        obs : np.ndarray
            The observation; its first six entries are the Hill-frame state, m and m/s.
        action : np.ndarray
            The agent's command in the view's action mode.

        Returns
        -------
        FilterResult
            The filtered action in the view's action mode. `iterations` counts the checks of the
            line search, `fallback_level` is 2 when the backup itself is returned, and
            `certificate` with `certificate_horizon` state the guarantee of a step decided
            inside the set (None when recovering). When recovering, `slack` is the predicted
            violation of the returned action, m.

        Raises
        ------
        RuntimeError
            Before the first `reset`.
        ValueError
            If the observation or the action is malformed.
        """
        t0 = time.perf_counter()
        view = self._started()
        x = _state(obs)
        u_c = self._clipped(view, action)
        if self._monitor_only(x):
            return self._monitor(u_c, t0)
        barrier = self._barrier
        h0 = float(barrier.value(x))
        # The end check of a set-valued barrier is a set clearance, so its floor is one too.
        h_set = h0 if self._start_set is None else self._set_value(x, self._start_set)
        if isinstance(barrier, BrakingBarrier):
            inside = barrier.contains(x)
        else:
            inside = h_set >= 0.0
        floor_s = min(0.0, float(np.linalg.norm(x[:3])) - self._radius)
        floor_b = min(0.0, h_set)

        region: tuple[float, float] | None = None
        if isinstance(barrier, BrakingBarrier):
            # A step ends in the braking set, or no further out of it when it has left; the
            # backup keeps the set, so the next step has a backup again.
            v_top = max(barrier.v_cap, float(np.linalg.norm(x[3:])))
            region = (v_top, min(0.0, barrier.outer_value(x)))

        def accept(u: np.ndarray) -> bool:
            if region is not None and not self._kept(self._phi @ x + self._gamma_u @ u, *region):
                return False
            held, end = self._margins(x, u, floor_s)
            return held >= floor_s and end >= floor_b

        start, moved_by_row = self._candidate(x, u_c, h0)
        beta = barrier.backup(x, self._actuator)
        found = line_search(accept, start, beta, self.iterations)
        status, slack, certified = "feasible", 0.0, True
        # Inside the terminal free-drift set the drift step is covered by h(x) and ends in the
        # set again, and inside the CAM set the backup step is covered by the level of h(x) it
        # starts, so the backup needs no check there.
        kept = inside and (
            isinstance(barrier, CamBarrier)
            or (isinstance(barrier, PassiveBarrier) and barrier.terminal)
        )
        if (found.theta == 0.0 and not kept) or not inside:
            held, end = self._margins(x, found.u)
            shortfall = max(0.0, floor_s - held, floor_b - end)
            # The speed cap and the outer barrier are not keep-out distances, so a backup step
            # that leaves them gets no slack.
            left = region is not None and not self._kept(
                self._phi @ x + self._gamma_u @ found.u, *region
            )
            # A backup short of its own check by less than the keep-out resolution is the
            # slack of the certified bound at the boundary, not a defect of the barrier.
            if inside and not self.bounded and (left or shortfall > KeepOut.resolution):
                status, slack, certified = "solver_failed", shortfall, False
            elif not inside:
                slack = shortfall
            elif self.bounded and (left or shortfall > 0.0):
                # With error sets the backup is safe by no theorem: one that fails the tightened
                # check gets no certificate, and its shortfall is reported.
                slack, certified = shortfall, False
        certificate, horizon = None, None
        if inside and certified:
            certificate = self._certificate
            horizon = math.inf
            if isinstance(barrier, PassiveBarrier) and not barrier.terminal:
                horizon = float(barrier.t_cap)  # type: ignore[arg-type]
        u = self._onto_limit(found.u)
        return FilterResult(
            self._action(u),
            intervened=self._moved(u, u_c),
            status=status,
            slack=slack,
            compute_s=time.perf_counter() - t0,
            active=found.theta < 1.0 or moved_by_row,
            recovering=not inside,
            fallback_level=2 if found.theta == 0.0 else 0,
            certificate=certificate,
            certificate_horizon=horizon,
            iterations=found.checks,
        )


__all__ = [
    "BACKUP_BARRIERS",
    "BACKUP_CANDIDATES",
    "BackupFilter",
    "CbfFilter",
    "FilterDeclaration",
    "SoftCbfFilter",
]
