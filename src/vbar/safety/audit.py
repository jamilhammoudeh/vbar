"""Audit of a safety filter over seeded starts inside, or outside, its declared safe set.

Each start is flown with the filter around a nominal agent through `record_episode`, at the
scenario's own dt, with perfect navigation and no execution error, and every step is judged on
the true trajectory. A hard constraint's step is certified when its in-step lower bound (the
larger of the first-order and second-order bounds of `segment_minimum`) is at least minus the
constraint's resolution. A step whose bound misses that is re-flown from its recorded segment
starts and accelerations on the exact Clohessy-Wiltshire solution with 4, 16 and 64 times the
sample intervals, until the bound certifies the resolution or a refined minimum falls below it
(a violation); a step neither proves is uncertified and never counts as a pass.

From starts outside the set the audit reports every decision that was not "feasible", whether an
"infeasible_certified" one carries an infeasibility proof that passes its own check, and the
time until the true state is back in the set.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import math
import operator
import secrets
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ..agents.pd import PDAgent
from ..constraints import Constraint, segment_minimum
from ..dynamics.cw import cw_states
from ..metrics.episode import Episode, record_episode
from ..scenarios.base import EnvConfig
from ..spacecraft import PerfectNav
from . import FilteredAgent

AUDIT_MODES = ("inside", "outside")
# Multiples of the environment's sample intervals at which an uncertified step is re-flown.
REFINEMENTS = (4, 16, 64)
# Rejection sampling gives up after this many draws per start asked for.
_DRAWS_PER_START = 1000
_SEED_BITS = 128
_MIN_SEED_BITS = 64


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
    set, inf if it never is, and None for starts drawn inside the set.
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

    def __post_init__(self):
        object.__setattr__(self, "worst_bound", MappingProxyType(dict(self.worst_bound)))

    @property
    def infeasible(self) -> int:
        """Decisions with status "infeasible_certified"."""
        return sum(d.status == "infeasible_certified" for d in self.flagged)

    @property
    def solver_failed(self) -> int:
        """Decisions with status "solver_failed"."""
        return sum(d.status == "solver_failed" for d in self.flagged)


@dataclass(frozen=True)
class FilterAudit:
    """The audit of one filter around one nominal agent.

    `mode` is "inside" (starts drawn in the declared safe set) or "outside"; `seed` the 128-bit
    root seed every start and episode seed is drawn from; `box` the (low, high) corners of the
    box of Hill states (m, m/s) the starts were drawn from; `drawn` the draws made and `rejected`
    those on the wrong side of the set.
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
    def passed(self) -> bool:
        """Inside: no violation, no uncertified step, no infeasible and no "solver_failed"
        decision. Outside: no "solver_failed" decision, and every "infeasible_certified" one
        carries a proof that verifies."""
        if self.mode == "inside":
            return (
                self.violations == 0
                and self.uncertified == 0
                and self.infeasible == 0
                and self.solver_failed == 0
            )
        return self.solver_failed == 0 and self.proofless == 0 and self.proofs_rejected == 0


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
) -> tuple[int, int, list[Refinement], dict[str, float]]:
    """Violations, uncertified steps, re-flown steps and the worst bounds of one run."""
    hard = [c for c in episode.constraints if c.name in episode.hard]
    worst = {c.name: math.inf for c in hard}
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
            violated |= outcome == "violation"
            unproven |= outcome == "uncertified"
        violations += violated
        uncertified += unproven
    return violations, uncertified, refinements, worst


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


def _reentry(episode: Episode, contains: Callable[[np.ndarray], bool]) -> float:
    for record in episode.records:
        if contains(record.true_state):
            return float(record.t)
    return math.inf


def _run(safety_filter, nominal, env_type, cfg, steps: int, inside: bool, job) -> dict[str, Any]:
    """Fly and judge one start, `job` (start, episode seed, agent seed); return the fields of
    its `RunAudit`, plain so that they pickle."""
    x0, episode_seed, agent_seed = job
    flown = env_type(dataclasses.replace(cfg, start_state=tuple(x0), max_steps=steps))
    episode = record_episode(flown, FilteredAgent(nominal, safety_filter), episode_seed, agent_seed)
    violations, uncertified, refinements, worst = _judge_steps(
        episode, flown.n, cfg.dt, cfg.violation_samples
    )
    contains = safety_filter.contains
    return {
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
    }


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
) -> FilterAudit:
    """Fly a safety filter from seeded starts drawn on one side of its declared safe set.

    Parameters
    ----------
    safety_filter : CbfFilter or BackupFilter
        The filter; it must have a `declaration` that counts toward certification and a
        `contains(x)` method, which decides membership of its declared safe set for the start
        draws, the flagged decisions and re-entry (the declaration itself has no membership
        test). It is reset on every episode.
    env : BaseEnv
        The scenario, with "cw" dynamics, a `PerfectNav` sensor, no execution error, no
        actuation delay, no minimum impulse bit, the "accel" drive and a `start_state` field
        in its config. Each start is flown in a fresh environment built
        from its class and config with `start_state` set.
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

    Returns
    -------
    FilterAudit
        Per-run and total counts of violations, uncertified steps, re-flown steps,
        infeasible and failed decisions with the check of each infeasibility proof, and, for
        outside starts, re-entry times.

    Raises
    ------
    ValueError
        If the dynamics are not "cw", navigation is not perfect, an execution error, a delay, a
        minimum impulse bit or the thrust drive is set, the scenario takes no `start_state`, the
        filter does not count toward certification, a parameter is out of range, or the box
        holds too few starts on the asked side.
    TypeError
        If the filter declares no safe set.
    """
    base = getattr(env, "unwrapped", env)
    cfg: EnvConfig = base.cfg
    if cfg.dynamics != "cw":
        raise ValueError(f"the audit needs cw dynamics, not {cfg.dynamics!r}")
    if not isinstance(cfg.sensor, PerfectNav):
        raise ValueError("the audit needs perfect navigation")
    if cfg.execution is not None:
        raise ValueError("the audit needs no execution error")
    if cfg.delay_s != 0.0:
        raise ValueError("the audit needs no actuation delay")
    if cfg.min_impulse != 0.0:
        raise ValueError("the audit needs no minimum impulse bit")
    if cfg.drive != "accel":
        raise ValueError("the audit needs the accel drive")
    if "start_state" not in {f.name for f in dataclasses.fields(cfg)}:
        raise ValueError("the scenario takes no start_state")
    if mode not in AUDIT_MODES:
        raise ValueError(f"unknown mode {mode!r}, expected {AUDIT_MODES}")
    if isinstance(starts, bool) or operator.index(starts) < 1:
        raise ValueError("starts must be a positive integer")
    steps = cfg.max_steps if max_steps is None else operator.index(max_steps)
    if steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if isinstance(workers, bool) or operator.index(workers) < 1:
        raise ValueError("workers must be a positive integer")
    # Read from the class: the declaration of a filter not yet reset raises.
    kind = type(safety_filter)
    if not (hasattr(kind, "declaration") and hasattr(kind, "contains")):
        raise TypeError("the filter declares no safe set")
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
        raise ValueError(f"{safety_filter.name} does not count toward certification (soft filter)")
    lo, hi = _box(box, cfg)
    root = _root_seed(seed)
    sampler_seq, seeds_seq = np.random.SeedSequence(root).spawn(2)
    contains = safety_filter.contains
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
    )


__all__ = [
    "AUDIT_MODES",
    "REFINEMENTS",
    "FilterAudit",
    "FlaggedDecision",
    "Refinement",
    "RunAudit",
    "audit_filter",
]
