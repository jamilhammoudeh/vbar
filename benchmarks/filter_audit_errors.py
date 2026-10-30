"""Audit of the certified keep-out filters under navigation and execution errors.

Run with `python benchmarks/filter_audit_errors.py [--starts 1000] [--budget 1000] [--runs 1000]
[--workers N] [--seed S] [--grid G] [--parts audit falsify gaussian] [--filters NAME ...]`.
Every run is on approach-vbar (500 km, keep-out 50 m, box 0.02 m/s^2, dt 10 s, 360 steps), flown
by `vbar.safety.audit.audit_filter` and `falsify_filter` as in benchmarks/filter_audit.py. The full
run takes days, so it is meant to be started as one process per filter and part, each with its
runs shared over `--workers` processes (all cores by default).

Error models: the navigation of the `:nav1` variant (`GaussianNav`, 1 m and 1 mm/s per axis, no
bias) and an `ExecutionError` with scale sigma 0.01 and noise sigma 0.005 of a full command. No
catalog scenario declares an execution error, so the latter is a test setting, not a model of a
real thruster. The episode-level exceedance is p = 1e-3, and each filter's per-draw level is
b = `per_draw_level(p, D)`, D = `tightening.episode_draws` of its models over 360 steps. Each
filter declares the sets `tightening.declared_errors` derives from the model parameters at b.

Filters: `BackupFilter` with the free-drift barrier and `PredictiveFilter` in its terminal mode,
under navigation and execution error, named "free drift" and "predictive"; `CbfFilter` at poles
0.06 1/s (gamma 0.06 1/s, v_cap 0.1 m/s) and 0.03 1/s (gamma 0.03 1/s, v_cap 0.2 m/s), named
"ecbf p 0.06" and "ecbf p 0.03", under execution error alone, since its rows take no navigation
set: no claim under navigation error. The braking and cam backup filters make no certified claim
under errors and are only listed. `--filters` picks some of them, all by default.
`--grid` sets the free-drift sample spacing (s) the free-drift and predictive filters declare,
their own default (5 s) when not given.

Parts, each picked by `--parts`:

1. "audit": under truncated models (`errors="bounded"`), `--starts` starts from the scenario's
   start box with the PD nominal and from the box near the keep-out with the inward nominal.
   The script asserts no violation past the keep-out resolution (1 mm), no uncertified step, no
   infeasible, "solver_failed" or uncovered decision.
2. "falsify": a cross-entropy search over the start, the navigation offsets and the execution
   direction inside the declared sets (`errors="bounded"`), near the keep-out, with the PD and
   the inward nominal and `--budget` runs each. The script asserts "none_found" with a certified
   robustness of at least -1 resolution; that is the best a search can give, not a proof.

`CbfFilter` issues no certificate, so every decision of its runs is uncovered. Its lines are
labelled "uncertified", report the uncovered count without gating it, and pass on no violation
past 1 mm and no infeasible or "solver_failed" decision; they never count as a certified pass.
The lines of the other filters are labelled "certified".
3. "gaussian": under untruncated models (`errors="gaussian"`), `--runs` starts near the keep-out
   with the inward nominal, reporting for the keep-out and for any covered constraint the count
   of violating runs, its 95% Clopper-Pearson interval and the level p it is read against. The
   rate is reported, never passed or failed.

Each finished audit or search prints one flushed line: the filter, its error kinds, the part,
the nominal and box, p, b and D, the violations past 1 mm, uncertified steps, infeasible,
"solver_failed" and uncovered decisions, and the verdict, then the root seed and the time.
Each audit draws its own 128-bit root seed; `--seed` uses one given seed for all of them.
"""

import argparse
import os
import secrets
import time

from filter_audit import NEAR, Inward

from vbar import ApproachConfig, ApproachEnv
from vbar.safety.audit import audit_filter, falsify_filter
from vbar.safety.cbf import BackupFilter, CbfFilter
from vbar.safety.predictive import PredictiveFilter
from vbar.safety.tightening import declared_errors, episode_draws
from vbar.scenarios import get, per_draw_level
from vbar.spacecraft import ExecutionError, GaussianNav, PerfectNav

EXCEEDANCE = 1e-3
STARTS = 1000
BUDGET = 1000
GAUSSIAN_RUNS = 1000
STEPS = ApproachConfig().max_steps
_NAV1 = get("approach-vbar:nav1").config["sensor"]
NAV1 = {
    "sigma_pos": float(_NAV1.sigma[0]),
    "sigma_vel": float(_NAV1.sigma[3]),
    "bias_pos": float(_NAV1.bias_sigma[0]),
    "bias_vel": float(_NAV1.bias_sigma[3]),
}
# A test setting, not a realism claim: no catalog scenario declares an execution error.
BURN = {"scale_sigma": 0.01, "noise_sigma": 0.005}
FILTERS = {
    "free drift": lambda **sets: BackupFilter(barrier="passive", **sets),
    "predictive": lambda **sets: PredictiveFilter(mode="terminal", **sets),
    "ecbf p 0.06": lambda **sets: CbfFilter(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1, **sets),
    "ecbf p 0.03": lambda **sets: CbfFilter(poles=(0.03, 0.03), gamma=0.03, v_cap=0.2, **sets),
}
# CbfFilter's rows are nonlinear in the state and take no navigation set.
EXECUTION_ONLY = ("ecbf p 0.06", "ecbf p 0.03")
# The filters whose declared set is checked by free drift sampled `grid` s apart.
GRIDDED = ("free drift", "predictive")
# Filters that are listed and never audited: a declaration under error sets certifies nothing.
NO_CLAIM = {
    "braking": "no certified claim under errors",
    "cam": "no claim under errors",
}
PARTS = ("audit", "falsify", "gaussian")
# (label, box, nominal) of each bounded audit; a None box is the scenario's start box and a None
# nominal the PD agent.
AUDITS = (("scenario starts", None, None), ("near the keep-out", NEAR, Inward()))


def models(name, bound=None):
    """(sensor, execution) of the filter `name`'s runs, truncated at `bound` sigma when given."""
    sensor = PerfectNav() if name in EXECUTION_ONLY else GaussianNav(**NAV1, bound=bound)
    return sensor, ExecutionError(**BURN, bound=bound)


def truncation(name):
    """(b, D): the per-draw level for p over the D Gaussian draws of an episode of `name`."""
    draws = episode_draws(*models(name), steps=STEPS)
    return per_draw_level(EXCEEDANCE, draws), draws


def declared(name, level, grid=None):
    """The filter `name` declaring the sets derived from its models at `level`, not reset.

    `grid` (s) is the free-drift sample spacing of the filters in `GRIDDED`; their own default
    when None.
    """
    cfg = ApproachConfig()
    navigation, da_max = declared_errors(
        *models(name), a_max=cfg.max_accel, layout=cfg.thrust_limit, level=level
    )
    sets = {"navigation": navigation, "execution": da_max}
    if grid is not None and name in GRIDDED:
        sets["grid"] = grid
    return FILTERS[name](**sets, exceedance=EXCEEDANCE, per_draw_level=level)


def scenario(name, level, truncated):
    sensor, execution = models(name, level if truncated else None)
    return ApproachEnv(ApproachConfig(sensor=sensor, execution=execution))


def kinds(name):
    return "execution" if name in EXECUTION_ONLY else "nav+exec"


def claim(name):
    if name in NO_CLAIM:
        return "no claim"
    return "uncertified" if name in EXECUTION_ONLY else "certified"


def passed(name, found):
    """Whether an audit, or the run a search found, passes for the filter `name`.

    `CbfFilter` issues no certificate, so every decision of its runs is uncovered: its lines are
    judged on violations past 1 mm, infeasible and "solver_failed" decisions alone, and never
    count as a certified pass. The other filters fail on any uncovered decision too.
    """
    if name in EXECUTION_ONLY:
        return found.violations == 0 and found.infeasible == 0 and found.solver_failed == 0
    return bool(found.passed)


def counts(found):
    return (
        f"violations past 1 mm {found.violations} uncertified {found.uncertified} "
        f"infeasible {found.infeasible} solver_failed {found.solver_failed} "
        f"uncovered {found.uncovered}"
    )


def declared_grid(declaration):
    """The free-drift sample spacing (s) of the declared barrier, "-" for a barrier without one."""
    grids = [barrier.grid for barrier in declaration.barriers if hasattr(barrier, "grid")]
    return f"{grids[0]:g} s" if grids else "-"


def head(name, part, nominal, label, found):
    """The start of an audit's line: who, which part, the declared grid, and the audit's own
    p, b and D."""
    return (
        f"{name:12s} {kinds(name):9s} {claim(name):11s} {part:8s} {nominal:7s} {label:18s} "
        f"grid {declared_grid(found.declaration)} "
        f"p {found.exceedance:g} b {found.per_draw_level:.4f} D {found.draws}"
    )


def root(args):
    return secrets.randbits(128) if args.seed is None else args.seed


def audit(name, args, failed):
    level, _ = truncation(name)
    env = scenario(name, level, truncated=True)
    for label, box, nominal in AUDITS:
        t0 = time.perf_counter()
        found = audit_filter(
            declared(name, level, args.grid),
            env,
            nominal=nominal,
            starts=args.starts,
            box=box,
            seed=root(args),
            workers=args.workers,
            errors="bounded",
        )
        ok = passed(name, found)
        level_line = head(name, "audit", found.nominal_name, label, found)
        print(
            f"{level_line} starts {len(found.runs)} drawn {found.drawn} {counts(found)} "
            f"{'pass' if ok else 'FAIL'} worst {1e3 * found.worst_bound['keep_out']:.3f} mm "
            f"seed {found.seed:#034x} {time.perf_counter() - t0:.0f} s",
            flush=True,
        )
        if not ok:
            failed.append(f"{name}, {found.nominal_name}, audit, {label}")


def falsify(name, args, failed):
    level, draws = truncation(name)
    env = scenario(name, level, truncated=True)
    for nominal in (None, Inward()):
        t0 = time.perf_counter()
        found = falsify_filter(
            declared(name, level, args.grid),
            env,
            nominal=nominal,
            budget=args.budget,
            box=NEAR,
            seed=root(args),
            workers=args.workers,
            errors="bounded",
        )
        if name in EXECUTION_ONLY:
            ok = passed(name, found.run)
        else:
            ok = found.verdict == "none_found" and found.score >= -1.0
        declaration = found.declaration
        level_line = (
            f"{name:12s} {kinds(name):9s} {claim(name):11s} falsify  {found.nominal_name:7s} "
            f"near the keep-out  grid {declared_grid(declaration)} p {declaration.exceedance:g} "
            f"b {declaration.per_draw_level:.4f} D {draws}"
        )
        print(
            f"{level_line} {counts(found.run)} {found.verdict} {'pass' if ok else 'FAIL'} "
            f"score {found.score:.1f} evaluations {found.evaluations} rejected {found.rejected} "
            f"dispersion {found.dispersion:.3f} seed {found.seed:#034x} "
            f"{time.perf_counter() - t0:.0f} s",
            flush=True,
        )
        if not ok:
            failed.append(f"{name}, {found.nominal_name}, falsify")


def gaussian(name, args):
    level, _ = truncation(name)
    t0 = time.perf_counter()
    found = audit_filter(
        declared(name, level, args.grid),
        scenario(name, level, truncated=False),
        nominal=Inward(),
        starts=args.runs,
        box=NEAR,
        seed=root(args),
        workers=args.workers,
        errors="gaussian",
    )
    rates = " ".join(
        f"{constraint} {rate.count}/{rate.trials} CP [{rate.interval[0]:.4f}, "
        f"{rate.interval[1]:.4f}] against {rate.level:g}"
        for constraint, rate in found.exceedance_rates.items()
    )
    level_line = head(name, "gaussian", found.nominal_name, "near the keep-out", found)
    print(
        f"{level_line} {counts(found)} reported {rates} seed {found.seed:#034x} "
        f"{time.perf_counter() - t0:.0f} s",
        flush=True,
    )


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=STARTS)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--runs", type=int, default=GAUSSIAN_RUNS)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=None)
    parser.add_argument("--grid", type=float, default=None)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS)
    choices = (*FILTERS, *NO_CLAIM)
    parser.add_argument("--filters", nargs="+", choices=choices, default=choices)
    return parser


def main():
    args = parser().parse_args()
    print(
        f"navigation :nav1 {NAV1}, execution {BURN} (test setting), "
        f"exceedance {EXCEEDANCE:g}, {STEPS} steps",
        flush=True,
    )
    for name, claim in NO_CLAIM.items():
        if name in args.filters:
            print(f"{name:12s} {claim}", flush=True)
    names = [name for name in FILTERS if name in args.filters]
    failed: list[str] = []
    t_all = time.perf_counter()
    for part in (part for part in PARTS if part in args.parts):
        for name in names:
            if part == "audit":
                audit(name, args, failed)
            elif part == "falsify":
                falsify(name, args, failed)
            else:
                gaussian(name, args)
    print(f"total {time.perf_counter() - t_all:.0f} s", flush=True)
    assert not failed, f"failed: {failed}"


if __name__ == "__main__":
    main()
