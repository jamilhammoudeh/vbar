"""Audit of a safety filter over seeded starts inside, or outside, its declared safe set.

Each start is flown with the filter around a nominal agent through `record_episode`, at the
scenario's own dt, with perfect navigation and no execution error by default, and every step is
judged on the true trajectory. A hard constraint's step is certified when its in-step lower bound
(the larger of the first-order and second-order bounds of `segment_minimum`) is at least minus
the constraint's resolution. A step whose bound misses that is re-flown from its recorded segment
starts and delivered accelerations on the exact Clohessy-Wiltshire solution with 4, 16 and 64
times the sample intervals, until the bound certifies the resolution or a refined minimum falls
below it (a violation); a step neither proves is uncertified and never counts as a pass.

Under `errors="bounded"` the scenario's navigation and execution error models are truncated at
the filter's declared per-draw level and its declared sets must hold them, so the audit judges a
claim of zero violations; under `errors="gaussian"` the models are untruncated and the audit
reports each constraint's rate of violating runs against the declared episode exceedance.

From starts outside the set the audit reports every decision that was not "feasible", whether an
"infeasible_certified" one carries an infeasibility proof that passes its own check, and the
time until the true state is back in the set.

`falsify_filter` flies the same runs from starts chosen by a cross-entropy search over the
declared safe set (`vbar.stress.cross_entropy`) that minimizes the certified in-step bound, and
judges the worst start found as the audit judges any run.

`cam_rates` measures how often a switching filter's collision avoidance manoeuvre (CAM) fires
without need, fails to save a run that needed it, or causes a violation, from runs flown with the
filter on and off; it reads the CAM from the recorded commands, never from the filter's report.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import itertools
import math
import operator
import secrets
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ..agents.pd import PDAgent
from ..constraints import Constraint, segment_minimum
from ..dynamics.cw import cw_states
from ..metrics.episode import Episode, FilterPair, record_episode
from ..metrics.safety import lowest_margin
from ..metrics.summary import clopper_pearson
from ..scenarios.base import EnvConfig
from ..scenarios.spec import per_draw_level
from ..spacecraft import ExecutionError, GaussianNav, PerfectNav
from ..stress.falsify import SearchResult, cross_entropy
from . import FilteredAgent
from .tightening import BallError, BoxError, EllipsoidError, declared_errors, episode_draws

AUDIT_MODES = ("inside", "outside")
ERROR_MODES = ("none", "bounded", "gaussian")
# Multiples of the environment's sample intervals at which an uncertified step is re-flown.
REFINEMENTS = (4, 16, 64)
# Rejection sampling gives up after this many draws per start asked for.
_DRAWS_PER_START = 1000
_SEED_BITS = 128
_MIN_SEED_BITS = 64
# Spawn key of the falsifier's episode and agent seeds, apart from the search's own streams.
_RUN_SEEDS_KEY = 1 << 32
FALSIFY_VERDICTS = ("violation", "uncertified", "infeasible", "solver_failed", "none_found")


@dataclass(frozen=True)
class Refinement:
    """One step whose certified bound on a hard constraint missed minus its resolution.

    `bound` and `margin` are the environment's certified bound and refined minimum, in the
    constraint's units; `refined_bound` and `refined_margin` come from the last re-fly, made with
    `factor` times the environment's sample intervals. `outcome` is "certified", "violation"
    (a refined minimum below minus the resolution) or "uncertified" (the split ran out).
    """

    step: int
    constraint: str
    bound: float
    margin: float
    refined_bound: float
    refined_margin: float
    factor: int
    outcome: str


@dataclass(frozen=True)
class FlaggedDecision:
    """A decision whose status was not "feasible".

    `t` is the decision time, s from the episode start; `inside` whether the true state there
    was in the declared safe set; `proof_verified` whether the result's infeasibility proof
    passed its own `verify()`, None when the result carries no proof.
    """

    step: int
    t: float
    status: str
    conflicting: tuple[str, ...]
    recovering: bool | None
    inside: bool
    proof_verified: bool | None


@dataclass(frozen=True)
class RunAudit:
    """The audit of one start.

    `start` is the true start state (m, m/s, Hill); `seed` and `agent_seed` the episode's and the
    nominal agent's seeds; `violations` and `uncertified` count steps; `worst_bound` maps each
    hard constraint to its lowest certified in-step bound over the run after re-flying;
    `reentry_s` is the first decision time (s) at which the true state is back in the declared
    set, inf if it never is, and None for starts drawn inside the set. `uncovered` counts the
    decisions without a certificate, less those that command zero thrust and end before the
    horizon of the last certified decision, measured from that decision's time; `violated` maps
    each hard constraint to the steps that violate it.
    """

    start: np.ndarray
    seed: int
    agent_seed: int
    steps: int
    violations: int
    uncertified: int
    refinements: tuple[Refinement, ...]
    flagged: tuple[FlaggedDecision, ...]
    worst_bound: Mapping[str, float]
    reentry_s: float | None
    uncovered: int = 0
    violated: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "worst_bound", MappingProxyType(dict(self.worst_bound)))
        object.__setattr__(self, "violated", MappingProxyType(dict(self.violated)))

    @property
    def infeasible(self) -> int:
        """Decisions with status "infeasible_certified"."""
        return sum(d.status == "infeasible_certified" for d in self.flagged)

    @property
    def solver_failed(self) -> int:
        """Decisions with status "solver_failed"."""
        return sum(d.status == "solver_failed" for d in self.flagged)


@dataclass(frozen=True)
class ExceedanceRate:
    """The share of runs that violate a constraint, against the level the filter declares.

    `count` of `trials` runs violate, `rate` = count / trials, `interval` is its 95%
    Clopper-Pearson interval and `level` the declared episode exceedance it is read against.
    """

    count: int
    trials: int
    rate: float
    interval: tuple[float, float]
    level: float


@dataclass(frozen=True)
class FilterAudit:
    """The audit of one filter around one nominal agent.

    `mode` is "inside" (starts drawn in the declared safe set) or "outside"; `seed` the 128-bit
    root seed every start and episode seed is drawn from; `box` the (low, high) corners of the
    box of Hill states (m, m/s) the starts were drawn from; `drawn` the draws made and `rejected`
    those on the wrong side of the set. `errors` is "none", "bounded" or "gaussian";
    under the last two `exceedance` (p) and `per_draw_level` (b, sigma) are the filter's declared
    truncation and `draws` (D) the Gaussian draws of an episode, all None under "none".
    `covered` names the hard constraints the filter enforces.
    """

    filter_name: str
    nominal_name: str
    mode: str
    declaration: Any
    seed: int
    box: tuple[tuple[float, ...], tuple[float, ...]]
    drawn: int
    rejected: int
    runs: tuple[RunAudit, ...]
    errors: str = "none"
    exceedance: float | None = None
    per_draw_level: float | None = None
    draws: int | None = None
    covered: tuple[str, ...] = ()

    @property
    def steps(self) -> int:
        return sum(run.steps for run in self.runs)

    @property
    def violations(self) -> int:
        return sum(run.violations for run in self.runs)

    @property
    def uncertified(self) -> int:
        return sum(run.uncertified for run in self.runs)

    @property
    def uncovered(self) -> int:
        """Decisions without a certificate that no earlier one covers."""
        return sum(run.uncovered for run in self.runs)

    @property
    def refined(self) -> int:
        """Steps re-flown with finer samples."""
        return sum(len({r.step for r in run.refinements}) for run in self.runs)

    @property
    def infeasible(self) -> int:
        return sum(run.infeasible for run in self.runs)

    @property
    def solver_failed(self) -> int:
        return sum(run.solver_failed for run in self.runs)

    @property
    def proofless(self) -> int:
        """ "infeasible_certified" decisions that carry no proof."""
        return sum(
            d.status == "infeasible_certified" and d.proof_verified is None
            for run in self.runs
            for d in run.flagged
        )

    @property
    def proofs_rejected(self) -> int:
        """ "infeasible_certified" decisions whose proof fails its check."""
        return sum(d.proof_verified is False for run in self.runs for d in run.flagged)

    @property
    def reentered(self) -> int:
        """Runs from outside the set that were back in it before the episode ended."""
        return sum(run.reentry_s is not None and math.isfinite(run.reentry_s) for run in self.runs)

    @property
    def longest_reentry_s(self) -> float | None:
        """The longest re-entry time over the runs (s), inf if a run never re-entered."""
        times = [run.reentry_s for run in self.runs if run.reentry_s is not None]
        return max(times) if times else None

    @property
    def worst_bound(self) -> dict[str, float]:
        """Each hard constraint's lowest certified in-step bound over every run."""
        worst: dict[str, float] = {}
        for run in self.runs:
            for name, bound in run.worst_bound.items():
                worst[name] = min(worst.get(name, math.inf), bound)
        return worst

    @property
    def passed(self) -> bool | None:
        """Inside: no violation, no uncertified step, no infeasible and no "solver_failed"
        decision, and under "bounded" errors no uncovered decision. Outside: no
        "solver_failed" decision, and every "infeasible_certified" one carries a proof that
        verifies. None under "gaussian" errors, which report `exceedance_rates` instead."""
        if self.errors == "gaussian":
            return None
        if self.mode == "inside":
            return (
                self.violations == 0
                and self.uncertified == 0
                and self.infeasible == 0
                and self.solver_failed == 0
                and (self.errors != "bounded" or self.uncovered == 0)
            )
        return self.solver_failed == 0 and self.proofless == 0 and self.proofs_rejected == 0

    @property
    def exceedance_rates(self) -> dict[str, ExceedanceRate]:
        """Under "gaussian" errors, the rate of runs violating each covered constraint against
        p, and of runs violating any of them against min(1, p times their number); empty
        otherwise."""
        if self.errors != "gaussian" or self.exceedance is None or not self.runs:
            return {}
        p = self.exceedance
        levels = dict.fromkeys(self.covered, p)
        # A union bound over the constraints, each violated with probability at most p.
        levels["any"] = min(1.0, len(self.covered) * p)
        trials = len(self.runs)
        rates = {}
        for name, level in levels.items():
            names = self.covered if name == "any" else (name,)
            count = sum(any(run.violated.get(c, 0) > 0 for c in names) for run in self.runs)
            rates[name] = ExceedanceRate(
                count, trials, count / trials, clopper_pearson(count, trials), level
            )
        return rates


def _root_seed(seed: int | None) -> int:
    if seed is None:
        return secrets.randbits(_SEED_BITS)
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    value = operator.index(seed)
    # A seed below 2**64 can be found by trying candidates (scored runs draw 128 bits).
    if not 1 << _MIN_SEED_BITS <= value < 1 << _SEED_BITS:
        raise ValueError("seed must lie in [2**64, 2**128)")
    return value


def _draw_seed(rng: np.random.Generator) -> int:
    return int.from_bytes(rng.bytes(_SEED_BITS // 8), "little")


def _box(box, cfg) -> tuple[np.ndarray, np.ndarray]:
    if box is None:
        ranges = (*getattr(cfg, "start_pos_range", ()), *getattr(cfg, "start_vel_range", ()))
        if len(ranges) != 6:
            raise ValueError("the scenario has no start ranges; pass box=(low, high)")
        low, high = zip(*ranges, strict=True)
    else:
        low, high = box
    lo = np.asarray(low, dtype=float)
    hi = np.asarray(high, dtype=float)
    if lo.shape != (6,) or hi.shape != (6,):
        raise ValueError("box corners must be six numbers each")
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(lo <= hi)):
        raise ValueError("box corners must be finite with low <= high")
    return lo, hi


def _starts(
    contains: Callable[[np.ndarray], bool],
    count: int,
    inside: bool,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], int]:
    """`count` states drawn uniformly from the box on the asked side of the set; draws made."""
    starts: list[np.ndarray] = []
    drawn = 0
    limit = _DRAWS_PER_START * count
    while len(starts) < count:
        if drawn == limit:
            side = "inside" if inside else "outside"
            raise ValueError(
                f"found {len(starts)} of {count} starts {side} the set in {drawn} draws"
            )
        x = rng.uniform(lo, hi)
        drawn += 1
        if bool(contains(x)) == inside:
            starts.append(x)
    return starts, drawn


def _refly(
    constraint: Constraint,
    record,
    n: float,
    dt: float,
    samples: int,
    factor: int,
) -> tuple[float, float]:
    """The certified bound and refined minimum of one step re-flown with `factor` times the
    environment's sample intervals on the exact CW solution."""
    trace = record.segment
    bound = margin = math.inf
    for (t0, duration, accel), start in zip(trace.segments, trace.starts, strict=True):
        count = max(2, round(samples * duration / dt))
        tau = np.linspace(0.0, duration, factor * (count - 1) + 1)
        a = np.asarray(accel, dtype=float)
        states = cw_states(
            n, tau, np.broadcast_to(start, (len(tau), 6)), np.broadcast_to(a, (len(tau), 3))
        )

        def state_at(s: float, start=start, a=a) -> np.ndarray:
            return cw_states(n, np.array([s]), start[None], a[None])[0]

        found = segment_minimum(constraint, state_at, tau, states, float(np.linalg.norm(a)), n, t0)
        bound = min(bound, found.lower_bound)
        margin = min(margin, found.margin)
    return bound, margin


def _judge_steps(
    episode: Episode, n: float, dt: float, samples: int
) -> tuple[int, int, list[Refinement], dict[str, float], dict[str, int]]:
    """Violations, uncertified steps, re-flown steps, the worst bounds and the violations of
    each hard constraint of one run."""
    hard = [c for c in episode.constraints if c.name in episode.hard]
    worst = {c.name: math.inf for c in hard}
    by_constraint = dict.fromkeys(worst, 0)
    violations = uncertified = 0
    refinements: list[Refinement] = []
    for k, record in enumerate(episode.records):
        violated = unproven = False
        for c in hard:
            found = record.segment.minima[c.name]
            floor = -c.resolution
            if found.lower_bound >= floor:
                worst[c.name] = min(worst[c.name], found.lower_bound)
                continue
            if found.margin < floor:
                violated = True
                by_constraint[c.name] += 1
                worst[c.name] = min(worst[c.name], found.lower_bound)
                continue
            # With no refinement left the step keeps the environment's own samples.
            outcome = "uncertified"
            refined_bound, refined_margin, factor = found.lower_bound, found.margin, 1
            for factor in REFINEMENTS:
                refined_bound, refined_margin = _refly(c, record, n, dt, samples, factor)
                if refined_margin < floor:
                    outcome = "violation"
                    break
                if refined_bound >= floor:
                    outcome = "certified"
                    break
            refinements.append(
                Refinement(
                    k,
                    c.name,
                    found.lower_bound,
                    found.margin,
                    refined_bound,
                    refined_margin,
                    factor,
                    outcome,
                )
            )
            worst[c.name] = min(worst[c.name], refined_bound)
            by_constraint[c.name] += outcome == "violation"
            violated |= outcome == "violation"
            unproven |= outcome == "uncertified"
        violations += violated
        uncertified += unproven
    return violations, uncertified, refinements, worst, by_constraint


def _flagged(episode: Episode, contains: Callable[[np.ndarray], bool]) -> list[FlaggedDecision]:
    flagged = []
    for k, record in enumerate(episode.records):
        result = record.filter_result
        if result is None or result.status == "feasible":
            continue
        x = episode.start if k == 0 else episode.records[k - 1].true_state
        # Read by name: results from filters that predate the proof field carry none.
        proof = getattr(result, "proof", None)
        verified = None
        if result.status == "infeasible_certified" and proof is not None:
            verified = bool(proof.verify())
        flagged.append(
            FlaggedDecision(
                k,
                0.0 if k == 0 else float(episode.records[k - 1].t),
                result.status,
                tuple(result.conflicting),
                result.recovering,
                bool(contains(x)),
                verified,
            )
        )
    return flagged


def _uncovered(episode: Episode) -> int:
    """Decisions without a certificate, except zero-thrust ones that end before the last
    certified decision's time plus its horizon."""
    count = 0
    covered_until = -math.inf
    t = 0.0
    for record in episode.records:
        result = record.filter_result
        if result is not None and result.certificate is not None:
            covered_until = t + result.certificate_horizon
        # A coast draws no execution error, so the last certificate still covers it up to its
        # horizon; an uncertified burn is never covered.
        elif np.any(record.filtered) or record.t > covered_until:
            count += 1
        t = float(record.t)
    return count


def _reentry(episode: Episode, contains: Callable[[np.ndarray], bool]) -> float:
    for record in episode.records:
        if contains(record.true_state):
            return float(record.t)
    return math.inf


def _run(safety_filter, nominal, env_type, cfg, steps: int, inside: bool, job) -> dict[str, Any]:
    """Fly and judge one start, `job` (start, episode seed, agent seed); return the fields of
    its `RunAudit`, plain so that they pickle."""
    return _flown(safety_filter, nominal, env_type, cfg, steps, inside, job)[1]


def _flown(
    safety_filter, nominal, env_type, cfg, steps: int, inside: bool, job
) -> tuple[Episode, dict[str, Any]]:
    """The episode of one start and the fields of its `RunAudit`."""
    x0, episode_seed, agent_seed = job
    flown = env_type(dataclasses.replace(cfg, start_state=tuple(x0), max_steps=steps))
    episode = record_episode(flown, FilteredAgent(nominal, safety_filter), episode_seed, agent_seed)
    violations, uncertified, refinements, worst, violated = _judge_steps(
        episode, flown.n, cfg.dt, cfg.violation_samples
    )
    contains = safety_filter.contains
    return episode, {
        "start": episode.start,
        "seed": episode_seed,
        "agent_seed": agent_seed,
        "steps": len(episode.records),
        "violations": violations,
        "uncertified": uncertified,
        "refinements": tuple(refinements),
        "flagged": tuple(_flagged(episode, contains)),
        "worst_bound": worst,
        "reentry_s": None if inside else _reentry(episode, contains),
        "uncovered": _uncovered(episode),
        "violated": violated,
    }


@dataclass(frozen=True)
class _Setup:
    """What `audit_filter` and `falsify_filter` share after checking their arguments."""

    base: Any
    cfg: EnvConfig
    steps: int
    flown_filter: Any
    declaration: Any
    lo: np.ndarray
    hi: np.ndarray
    root: int
    draws: int | None
    covered: tuple[str, ...]


def _contains_box(declared, box: BoxError) -> bool:
    """Whether the error set `declared` holds the box; both are convex, so its corners decide."""
    match declared:
        case BoxError():
            return declared.r_pos >= box.r_pos and declared.r_vel >= box.r_vel
        case BallError():
            corner = math.sqrt(3.0)
            return declared.r_pos >= corner * box.r_pos and declared.r_vel >= corner * box.r_vel
        case EllipsoidError():
            half = np.repeat([box.r_pos, box.r_vel], 3)
            corners = np.array(list(itertools.product((-1.0, 1.0), repeat=6))).T * half[:, None]
            # The least-norm z with P z = e; a corner off the range of a flat P is outside.
            z = np.linalg.lstsq(declared.P, corners, rcond=None)[0]
            reach = np.linalg.norm(declared.P @ z - corners, axis=0)
            scale = 1e-12 * max(1.0, float(np.abs(corners).max()))
            return bool(np.all(reach <= scale) and np.all(np.linalg.norm(z, axis=0) <= 1.0))
    return False


def _error_models(cfg: EnvConfig, errors: str) -> None:
    """Check the scenario's navigation and execution models for the error mode."""
    sensor, execution = cfg.sensor, cfg.execution
    if errors == "none":
        if not isinstance(sensor, PerfectNav):
            raise ValueError("the audit needs perfect navigation")
        if execution is not None:
            raise ValueError("the audit needs no execution error")
        return
    if not isinstance(sensor, (PerfectNav, GaussianNav)):
        raise ValueError(f"{errors} errors have no declared set for {type(sensor).__name__}")
    if execution is not None and not isinstance(execution, ExecutionError):
        raise ValueError(f"{errors} errors have no declared set for {type(execution).__name__}")
    for label, model in (("navigation", sensor), ("execution", execution)):
        if not isinstance(model, (GaussianNav, ExecutionError)):
            continue
        if errors == "bounded" and model.bound is None:
            raise ValueError(f"bounded errors need a truncated {label} model")
        if errors == "gaussian" and model.bound is not None:
            raise ValueError(f"gaussian errors need an untruncated {label} model")


def _truncation(cfg: EnvConfig, declaration, errors: str, steps: int) -> int:
    """Check the filter's declared truncation against the scenario's models; return D."""
    p, b = declaration.exceedance, declaration.per_draw_level
    if p is None or b is None:
        raise ValueError(f"{errors} errors need a declared exceedance and per_draw_level")
    # `_error_models` has passed only these types; perfect navigation adds no set and no draw.
    sensor = cfg.sensor if isinstance(cfg.sensor, GaussianNav) else None
    execution = cfg.execution
    if errors == "bounded":
        for label, model in (("navigation", sensor), ("execution", execution)):
            if isinstance(model, (GaussianNav, ExecutionError)) and model.bound != b:
                raise ValueError(
                    f"the {label} model is truncated at {model.bound:g} sigma, the filter "
                    f"declares {b:g}"
                )
    draws = episode_draws(sensor, execution, steps=steps)
    # A b below the level for p and D would make the episode exceedance larger than p.
    if draws and b < per_draw_level(p, draws):
        raise ValueError(
            f"per_draw_level {b:g} is below {per_draw_level(p, draws):g}, the level for "
            f"exceedance {p:g} over {draws} draws"
        )
    navigation, da_max = declared_errors(
        sensor, execution, a_max=cfg.max_accel, layout=cfg.thrust_limit, level=b
    )
    declared = declaration.tightening.get("navigation")
    if navigation is not None:
        if declared is None:
            raise ValueError("the filter declares no navigation set")
        if not _contains_box(declared, navigation):
            raise ValueError(f"the declared navigation set does not hold {navigation}")
    if (declaration.tightening.get("execution") or 0.0) < da_max:
        raise ValueError(f"the declared execution bound is below {da_max:.6g} m/s^2")
    return draws


def _setup(safety_filter, env, box, seed, max_steps, workers, errors="none") -> _Setup:
    """Check the scenario and the filter as the audit needs them; reset the filter."""
    base = getattr(env, "unwrapped", env)
    cfg: EnvConfig = base.cfg
    if cfg.dynamics != "cw":
        raise ValueError(f"the audit needs cw dynamics, not {cfg.dynamics!r}")
    _error_models(cfg, errors)
    if cfg.delay_s != 0.0:
        raise ValueError("the audit needs no actuation delay")
    if cfg.min_impulse != 0.0:
        raise ValueError("the audit needs no minimum impulse bit")
    if cfg.drive != "accel":
        raise ValueError("the audit needs the accel drive")
    if "start_state" not in {f.name for f in dataclasses.fields(cfg)}:
        raise ValueError("the scenario takes no start_state")
    steps = cfg.max_steps if max_steps is None else operator.index(max_steps)
    if steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if isinstance(workers, bool) or operator.index(workers) < 1:
        raise ValueError("workers must be a positive integer")
    # Read from the class: the declaration of a filter not yet reset raises.
    kind = type(safety_filter)
    if not (hasattr(kind, "declaration") and hasattr(kind, "contains")):
        raise TypeError("the filter declares no safe set")
    if errors != "none" and not hasattr(kind, "contains_start"):
        raise TypeError("the filter declares no start set under navigation error")
    flown_filter = safety_filter
    if workers > 1:
        # A reset filter holds its view, whose read-only mappings do not pickle.
        try:
            flown_filter = copy.deepcopy(safety_filter)
        except TypeError:
            raise ValueError("workers above 1 need a filter that has not been reset") from None
    safety_filter.reset(base.agent_view())
    declaration = safety_filter.declaration
    if not declaration.counts_toward_certification:
        raise ValueError(
            f"{safety_filter.name} does not count toward certification; safe set: "
            f"{declaration.safe_set}"
        )
    draws = None if errors == "none" else _truncation(cfg, declaration, errors, steps)
    soft = set(getattr(base, "soft_constraints", ()))
    covered = tuple(
        c.name for c in base.constraints if c.name not in soft and c.name not in declaration.ignored
    )
    lo, hi = _box(box, cfg)
    return _Setup(
        base, cfg, steps, flown_filter, declaration, lo, hi, _root_seed(seed), draws, covered
    )


def audit_filter(
    safety_filter,
    env,
    *,
    nominal=None,
    starts: int = 1000,
    mode: str = "inside",
    box=None,
    seed: int | None = None,
    max_steps: int | None = None,
    workers: int = 1,
    errors: str = "none",
) -> FilterAudit:
    """Fly a safety filter from seeded starts drawn on one side of its declared safe set.

    Parameters
    ----------
    safety_filter : CbfFilter, BackupFilter or PredictiveFilter
        The filter; it must have a `declaration` that counts toward certification and a
        `contains(x)` method, which decides membership of its declared safe set for the start
        draws, the flagged decisions and re-entry (the declaration itself has no membership
        test). Under "bounded" and "gaussian" errors it must also have `contains_start(x)`,
        which draws the starts instead, and declare `exceedance` and `per_draw_level`. It is
        reset on every episode.
    env : BaseEnv
        The scenario, with "cw" dynamics, no actuation delay, no minimum impulse bit, the
        "accel" drive and a `start_state` field in its config; its sensor and execution error
        as `errors` asks. Each start is flown in a fresh environment built from its class and
        config with `start_state` set.
    nominal : Agent, optional
        The agent whose commands the filter judges; `PDAgent()` by default.
    starts : int
        Starts to fly, positive.
    mode : str
        "inside" draws starts in the declared safe set, "outside" outside it.
    box : tuple of array_like, optional
        (low, high) corners of the box of Hill states (m, m/s) starts are drawn from uniformly,
        each six numbers; by default the scenario's start position and velocity ranges.
    seed : int, optional
        Root seed in [2**64, 2**128); drawn (128 bits) when not given. The starts and, per start, an
        episode seed and a different agent seed, each 128 bits, are drawn from it.
    max_steps : int, optional
        Step limit of each episode; the scenario's by default.
    workers : int
        Processes the runs are flown in, positive; above 1 the filter must not have been reset
        yet, and it, the nominal and the environment's class and config must pickle. The report
        is the same for every value.
    errors : str
        "none": a `PerfectNav` sensor and no execution error. "bounded": a `GaussianNav` or
        `PerfectNav` sensor and an `ExecutionError` or none, each `GaussianNav` and
        `ExecutionError` truncated at the filter's `per_draw_level` b, which may not lie below
        `per_draw_level(exceedance, D)`, D from `tightening.episode_draws` over the episode's
        steps, and declared sets that hold those `tightening.declared_errors` derives at b.
        "gaussian": the same with the models untruncated. Both draw starts with
        `contains_start` and need `mode="inside"`.

    Returns
    -------
    FilterAudit
        Per-run and total counts of violations, uncertified steps, re-flown steps,
        infeasible and failed decisions with the check of each infeasibility proof,
        uncovered decisions, and, for outside starts, re-entry times; under "gaussian" errors
        the rate of violating runs per covered constraint in `exceedance_rates`.

    Raises
    ------
    ValueError
        If the dynamics are not "cw", the sensor or execution model does not fit `errors`, a
        delay, a minimum impulse bit or the thrust drive is set, the scenario takes no
        `start_state`, the filter does not count toward certification, its declared
        truncation or sets do not fit the models, a parameter is out of range, or the box
        holds too few starts on the asked side.
    TypeError
        If the filter declares no safe set, or no start set under errors.

    Notes
    -----
    Steps are re-flown with the accelerations the spacecraft delivered, execution error
    included, so the judged trajectory is the true one. A decision without a certificate is
    covered when it commands zero thrust and its step ends before the time of the last
    certified decision plus that decision's horizon: a coast draws no execution error.
    Under "bounded" errors an uncovered decision fails the audit.
    """
    if mode not in AUDIT_MODES:
        raise ValueError(f"unknown mode {mode!r}, expected {AUDIT_MODES}")
    if errors not in ERROR_MODES:
        raise ValueError(f"unknown errors {errors!r}, expected {ERROR_MODES}")
    if errors != "none" and mode != "inside":
        raise ValueError(f"{errors} errors need mode='inside'")
    if isinstance(starts, bool) or operator.index(starts) < 1:
        raise ValueError("starts must be a positive integer")
    setup = _setup(safety_filter, env, box, seed, max_steps, workers, errors)
    base, cfg, steps, flown_filter = setup.base, setup.cfg, setup.steps, setup.flown_filter
    declaration, lo, hi, root = setup.declaration, setup.lo, setup.hi, setup.root
    sampler_seq, seeds_seq = np.random.SeedSequence(root).spawn(2)
    # Under navigation error the filter first sees an estimate, so every estimate of a start
    # must lie in its set.
    contains = safety_filter.contains if errors == "none" else safety_filter.contains_start
    inside = mode == "inside"
    xs, drawn = _starts(
        contains, operator.index(starts), inside, lo, hi, np.random.default_rng(sampler_seq)
    )
    seeds = np.random.default_rng(seeds_seq)
    jobs = []
    for x0 in xs:
        episode_seed = _draw_seed(seeds)
        agent_seed = _draw_seed(seeds)
        while agent_seed == episode_seed:
            agent_seed = _draw_seed(seeds)
        jobs.append((x0, episode_seed, agent_seed))
    nominal = PDAgent() if nominal is None else nominal
    fly = functools.partial(_run, flown_filter, nominal, type(base), cfg, steps, inside)
    if workers == 1:
        flown = [fly(job) for job in jobs]
    else:
        # Every start and seed is drawn above, so the runs, and the report, do not depend on
        # how they are shared out.
        with ProcessPoolExecutor(workers) as pool:
            flown = list(pool.map(fly, jobs, chunksize=max(1, len(jobs) // (4 * workers))))
    runs = [RunAudit(**fields) for fields in flown]
    return FilterAudit(
        filter_name=safety_filter.name,
        nominal_name=getattr(nominal, "name", type(nominal).__name__),
        mode=mode,
        declaration=declaration,
        seed=root,
        box=(tuple(lo.tolist()), tuple(hi.tolist())),
        drawn=drawn,
        rejected=drawn - len(xs),
        runs=tuple(runs),
        errors=errors,
        exceedance=None if errors == "none" else declaration.exceedance,
        per_draw_level=None if errors == "none" else declaration.per_draw_level,
        draws=setup.draws,
        covered=setup.covered,
    )


@dataclass(frozen=True)
class FilterFalsification:
    """The worst start a falsification search found for one filter around one nominal agent.

    `verdict` comes from the run of that start, judged as the audit judges any run: "violation",
    "uncertified" (a step neither certified nor shown violated), "infeasible" (a decision with
    status "infeasible_certified"), "solver_failed", or "none_found". "none_found" is the best
    verdict there is: the search samples, and a run without a finding shows nothing about the
    starts it did not fly. `score` is the run's certified robustness, the lowest in-step lower
    bound of any hard constraint in units of its resolution (below -1 is a certified bound
    deeper than the resolution); `start` the start (m, m/s, Hill) and `states` the true states of
    its run, re-flown; `run` its `RunAudit`; `evaluations` the runs flown by the search;
    `rejected` the draws outside the declared safe set, never flown; `dispersion` the search's
    average dispersion over the box; `seed` the 128-bit root seed; `box` the (low, high) corners
    searched; `search` the search's own record.
    """

    filter_name: str
    nominal_name: str
    declaration: Any
    verdict: str
    start: np.ndarray
    states: np.ndarray
    run: RunAudit
    score: float
    evaluations: int
    rejected: int
    dispersion: float
    seed: int
    box: tuple[tuple[float, ...], tuple[float, ...]]
    search: SearchResult


def _robustness(fly, resolutions: Mapping[str, float], x0: np.ndarray, seeds) -> float:
    """The certified robustness of the run from `x0`: min over hard constraints of the lowest
    in-step bound over the constraint's resolution."""
    fields = fly((np.asarray(x0, dtype=float), *seeds))
    bounds = fields["worst_bound"]
    return min((bound / resolutions[name] for name, bound in bounds.items()), default=math.inf)


def _verdict(run: RunAudit) -> str:
    if run.violations:
        return "violation"
    if run.uncertified:
        return "uncertified"
    if run.infeasible:
        return "infeasible"
    if run.solver_failed:
        return "solver_failed"
    return "none_found"


def falsify_filter(
    safety_filter,
    env,
    *,
    nominal=None,
    budget: int = 1000,
    box=None,
    seed: int | None = None,
    max_steps: int | None = None,
    workers: int = 1,
    population: int = 50,
    chains: int = 2,
) -> FilterFalsification:
    """Search the declared safe set for the start whose run comes closest to failing.

    Parameters
    ----------
    safety_filter : CbfFilter or BackupFilter
        The filter, as for `audit_filter`: it must declare a safe set, have `contains(x)`, and
        count toward certification.
    env : BaseEnv
        The scenario, under the same conditions as for `audit_filter`.
    nominal : Agent, optional
        The agent whose commands the filter judges; `PDAgent()` by default.
    budget : int
        Runs to fly, at least 1.
    box : tuple of array_like, optional
        (low, high) corners of the box of Hill states (m, m/s) searched; by default the scenario's
        start ranges. Points outside the declared safe set are rejected and never flown.
    seed : int, optional
        Root seed in [2**64, 2**128); drawn (128 bits) when not given. The search's streams and one
        episode seed and a different agent seed, shared by every run, are drawn from it.
    max_steps : int, optional
        Step limit of each episode; the scenario's by default.
    workers : int
        Processes the runs are flown in; the result is the same for every value.
    population, chains : int
        Candidates per generation and independent chains of `vbar.stress.cross_entropy`.

    Returns
    -------
    FilterFalsification
        The worst start found, its re-flown run and verdict, the search's evaluations, rejections
        and dispersion, and the seed.

    Raises
    ------
    ValueError
        As `audit_filter`, and if the box holds too few starts in the declared safe set.
    TypeError
        If the filter declares no safe set.

    Notes
    -----
    The search minimizes the certified robustness of each run (Kochenderfer, Katz, Corso and
    Moss, *Algorithms for Validation*, ch. 4), with every run sharing one episode seed and one
    agent seed, so the score depends on the start alone. The verdict is taken from the run of the
    worst start, never from the score.
    """
    setup = _setup(safety_filter, env, box, seed, max_steps, workers)
    base, cfg, steps, root = setup.base, setup.cfg, setup.steps, setup.root
    seeds_rng = np.random.default_rng(np.random.SeedSequence(root, spawn_key=(_RUN_SEEDS_KEY,)))
    episode_seed = _draw_seed(seeds_rng)
    agent_seed = _draw_seed(seeds_rng)
    while agent_seed == episode_seed:
        agent_seed = _draw_seed(seeds_rng)
    nominal = PDAgent() if nominal is None else nominal
    fly = functools.partial(_run, setup.flown_filter, nominal, type(base), cfg, steps, True)
    resolutions = {c.name: float(c.resolution) for c in base.constraints}
    score = functools.partial(_robustness, fly, resolutions, seeds=(episode_seed, agent_seed))
    search = cross_entropy(
        score,
        setup.lo,
        setup.hi,
        accept=safety_filter.contains,
        budget=budget,
        population=population,
        chains=chains,
        seed=root,
        workers=workers,
    )
    episode, fields = _flown(
        safety_filter, nominal, type(base), cfg, steps, True, (search.x, episode_seed, agent_seed)
    )
    run = RunAudit(**fields)
    return FilterFalsification(
        filter_name=safety_filter.name,
        nominal_name=getattr(nominal, "name", type(nominal).__name__),
        declaration=setup.declaration,
        verdict=_verdict(run),
        start=episode.start,
        states=episode.states,
        run=run,
        score=search.score,
        evaluations=search.evaluations,
        rejected=search.rejected,
        dispersion=search.dispersion,
        seed=root,
        box=(tuple(setup.lo.tolist()), tuple(setup.hi.tolist())),
        search=search,
    )


@dataclass(frozen=True)
class CamRate:
    """One CAM rate over paired runs.

    `count` of `trials` pairs, `rate` = count / trials and `interval` its 95% Clopper-Pearson
    interval; with no trials the rate and both ends of the interval are NaN.
    """

    count: int
    trials: int
    rate: float
    interval: tuple[float, float]


@dataclass(frozen=True)
class CamRates:
    """The false-CAM, missed-CAM and CAM-induced rates of one filter, as in `cam_rates`."""

    false_cam: CamRate
    missed_cam: CamRate
    cam_induced: CamRate


def _rate(count: int, trials: int) -> CamRate:
    if trials == 0:
        return CamRate(0, 0, math.nan, (math.nan, math.nan))
    return CamRate(count, trials, count / trials, clopper_pearson(count, trials))


def _violates(episode: Episode) -> bool:
    """Whether the run's S1 on the keep-out, its lowest margin and not the bound, is below minus
    the keep-out resolution (1 mm)."""
    keep_out = episode.constraint("keep_out")
    if keep_out is None:
        raise ValueError("the run has no keep_out constraint")
    return lowest_margin(episode, keep_out)[0] < -keep_out.resolution


def _fired(episode: Episode) -> bool:
    """Whether some applied command differs from the agent's recorded one."""
    return any(not np.array_equal(rec.filtered, rec.command) for rec in episode.records)


def cam_rates(pairs: Sequence[FilterPair]) -> CamRates:
    """The CAM rates of a switching filter from runs flown with it on and off.

    Parameters
    ----------
    pairs : sequence of FilterPair
        Seeds flown by `vbar.metrics.episode.record_pair`, each with the filter on and off.

    Returns
    -------
    CamRates
        `false_cam`: pairs where the CAM fired, among those that did not need one.
        `missed_cam`: pairs where the CAM did not fire or the on run violates, among those that
        needed one. `cam_induced`: pairs whose on run violates, among those that did not need a
        CAM. Each with its count, trials, rate and 95% Clopper-Pearson interval.

    Raises
    ------
    ValueError
        If a run has no keep_out constraint.

    Notes
    -----
    A run violates when its S1 on the keep-out (its lowest margin, not the certified bound) is
    below -1 mm, the keep-out resolution; a pair needed a CAM when its off run violates. The CAM
    fired when some step of the on run applied a command different from the agent's recorded
    one, so any change counts, a change the filter did not report included, and a reported
    intervention that changed nothing does not.
    """
    flags = [(_violates(pair.off), _fired(pair.on), _violates(pair.on)) for pair in pairs]
    needed = [(fired, bad) for need, fired, bad in flags if need]
    not_needed = [(fired, bad) for need, fired, bad in flags if not need]
    return CamRates(
        false_cam=_rate(sum(fired for fired, _ in not_needed), len(not_needed)),
        missed_cam=_rate(sum(not fired or bad for fired, bad in needed), len(needed)),
        cam_induced=_rate(sum(bad for _, bad in not_needed), len(not_needed)),
    )


__all__ = [
    "AUDIT_MODES",
    "ERROR_MODES",
    "FALSIFY_VERDICTS",
    "REFINEMENTS",
    "CamRate",
    "CamRates",
    "ExceedanceRate",
    "FilterAudit",
    "FilterFalsification",
    "FlaggedDecision",
    "Refinement",
    "RunAudit",
    "audit_filter",
    "cam_rates",
    "falsify_filter",
]
