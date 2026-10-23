"""Audit of the certified keep-out filters over seeded starts inside and outside their safe sets.

Run with `python benchmarks/filter_audit.py [--starts 1000] [--budget 2000] [--workers N]
[--seed S] [--boxes 20000] [--parts audit|falsify|both|cover] [--filters NAME ...]`. Every
audit is on approach-vbar (500 km, keep-out 50 m, box 0.02 m/s^2, dt 10 s, 360 steps) with
perfect navigation and no execution error, flown by `vbar.safety.audit.audit_filter`; the full
run takes hours on one core, so the runs are shared over `--workers` processes (all cores by
default).

Filters: `CbfFilter` at the pole rule's limit (poles 0.06 1/s, gamma 0.06 1/s, v_cap 0.1 m/s)
and below it (poles 0.03 1/s, gamma 0.03 1/s, v_cap 0.2 m/s), `BackupFilter` with the
free-drift, the braking and the cam barrier, and `PredictiveFilter` with its defaults, named
"ecbf p 0.06", "ecbf p 0.03", "free drift", "braking", "cam" and "predictive"; `--filters`
picks some of them, all by default. Each one is audited

1. from starts inside its declared safe set drawn from the scenario's start ranges, with the PD
   nominal and with a nominal that pushes at full thrust straight at the target;
2. from starts inside the set near the keep-out (52 to 150 m along V-bar, speeds up to
   0.1 m/s per axis), with the inward nominal;
3. from starts outside the set near the keep-out (speeds up to 0.5 m/s per axis), with the PD
   nominal, reporting every infeasible decision and the time to re-enter the set.

The script asserts that every audit passes: inside the set no violation deeper than the
keep-out resolution (1 mm), no step left uncertified after re-flying it with up to 64 times the
samples, and no infeasible decision; outside it no "solver_failed" decision and an infeasibility
proof that verifies on every "infeasible_certified" one. Each audit draws its own 128-bit root
seed and prints it; `--seed` uses one given seed for all of them.

Then each filter is searched with `vbar.safety.audit.falsify_filter` for the start in its
declared safe set near the keep-out whose run comes closest to failing, with the PD and the
inward nominal and `--budget` runs each (2000 by default), and the script asserts the verdict
"none_found" with a certified robustness of at least -1 resolution. That verdict is the best a
search can give; it is not a proof that no failing start exists.

`--parts cover` runs neither and checks declared sets without sampling: `vbar.reach.drift_cover`
splits the scenario's start box and the box near the keep-out into boxes whose motion over the
audited run and one step more (3610 s) is certified clear of the keep-out, boxes from which
every such motion enters it, and undecided boxes, asking the set at each box center. For the
free-drift filter the motion is free drift and the set its declared one. The cam filter's set
is the union of the levels h^j >= 0 of `CamBarrier`, j = 0 to m, each claiming j held CAM steps
and then free drift; level j >= 1 is covered with that CAM as `accel` held for j dt, and level 0
is the free-drift set. The predictive filter's set is the free-drift set too. The script asserts
both equalities and covers each set once, the free-drift set when "free drift" is selected.
Each box is first halved once along every axis and the 64 parts are covered in `--workers`
processes with an equal share of `--boxes`, so the result does not depend on the worker count.
The script asserts no counterexample, a center in the set whose motion enters by more than
1 mm. It prints the certified, excluded and undecided volume fractions and the share of
`--starts` starts, drawn from the set as the inside audit draws them, that lie in certified
boxes, and marks a region below 99% with "coverage below 99%" without failing it. The share
describes the box method and its budget and is never a safety claim: the undecided region is
covered only by the sampled audit and the falsification search. Each cover draws its own
128-bit root seed for those starts and prints it; `--seed` uses one given seed for all.
"""

import argparse
import functools
import itertools
import os
import secrets
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from vbar import ApproachEnv
from vbar.agents.base import action_scale
from vbar.dynamics import cw_discrete
from vbar.reach import drift_cover
from vbar.safety.audit import audit_filter, falsify_filter
from vbar.safety.barriers import PassiveBarrier
from vbar.safety.cbf import BackupFilter, CbfFilter
from vbar.safety.predictive import PredictiveFilter
from vbar.safety.sampled import held_segment_bound

NEAR = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))
FAST = ((-80.0, 52.0, -30.0, -0.5, -0.5, -0.5), (80.0, 150.0, 30.0, 0.5, 0.5, 0.5))
FILTERS = {
    "ecbf p 0.06": lambda: CbfFilter(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1),
    "ecbf p 0.03": lambda: CbfFilter(poles=(0.03, 0.03), gamma=0.03, v_cap=0.2),
    "free drift": lambda: BackupFilter(barrier="passive"),
    "braking": lambda: BackupFilter(barrier="braking"),
    "cam": lambda: BackupFilter(barrier="cam"),
    "predictive": lambda: PredictiveFilter(),
}
# Most halvings of each axis in the drift cover; in the full run the box budget stops it first.
COVER_DEPTH = 12
# Drift samples 1 s apart; the certified bound covers the time between them.
COVER_SPACING = 1.0
COVERED_SHARE = 0.99


class Inward:
    """Full command straight at the target."""

    name = "inward"

    def reset(self, view, rng=None):
        self.scale = action_scale(view)

    def act(self, obs, info):
        return -obs[:3] / np.linalg.norm(obs[:3]) * self.scale


def audits():
    """(label, mode, box, nominal) for each audit of a filter; None is the PD nominal."""
    return (
        ("scenario starts", "inside", None, None),
        ("scenario starts", "inside", None, Inward()),
        ("near the keep-out", "inside", NEAR, Inward()),
        ("outside, near", "outside", FAST, None),
    )


def line(name, label, found, seconds):
    worst = found.worst_bound["keep_out"]
    entry = found.longest_reentry_s
    reentry = "" if entry is None else f" re-entered {found.reentered}, longest {entry:.0f} s"
    return (
        f"{name:12s} {found.nominal_name:7s} {label:18s} {found.mode:7s} "
        f"starts {len(found.runs)} drawn {found.drawn} rejected {found.rejected} "
        f"steps {found.steps} violations {found.violations} uncertified {found.uncertified} "
        f"re-flown {found.refined} infeasible {found.infeasible} "
        f"failed {found.solver_failed} proofless {found.proofless} "
        f"proofs rejected {found.proofs_rejected} worst {1e3 * worst:.3f} mm{reentry} "
        f"seed {found.seed:#034x} {seconds:.0f} s {'pass' if found.passed else 'FAIL'}"
    )


def falsify_line(name, found, seconds):
    return (
        f"{name:12s} {found.nominal_name:7s} falsify            {found.verdict:13s} "
        f"score {found.score:.1f} evaluations {found.evaluations} rejected {found.rejected} "
        f"dispersion {found.dispersion:.3f} seed {found.seed:#034x} {seconds:.0f} s"
    )


def falsify(env, args, failed):
    for name in args.filters:
        make = FILTERS[name]
        for nominal in (None, Inward()):
            t0 = time.perf_counter()
            found = falsify_filter(
                make(),
                env,
                nominal=nominal,
                budget=args.budget,
                box=NEAR,
                seed=secrets.randbits(128) if args.seed is None else args.seed,
                workers=args.workers,
            )
            print(falsify_line(name, found, time.perf_counter() - t0), flush=True)
            if found.verdict != "none_found" or found.score < -1.0:
                failed.append(f"{name}, {found.nominal_name}, falsify")


@functools.cache
def declared(name):
    """The filter `name` reset on approach-vbar, built once per process."""
    safety_filter = FILTERS[name]()
    safety_filter.reset(ApproachEnv().agent_view())
    return safety_filter


def barrier(name):
    """The one barrier the filter `name` declares."""
    (found,) = declared(name).declaration.barriers
    return found


def cam_drift():
    """The free-drift set h^0 of the cam filter's `CamBarrier`, rebuilt from its fields."""
    cam = barrier("cam")
    return PassiveBarrier(radius=cam.radius, n=cam.n, grid=cam.grid, t_cap=cam.t_cap, terminal=True)


@functools.cache
def arc(name, steps):
    """(contains, accel, burn_time) of the arc whose clearance the set of `name` claims.

    For "free drift" the arc is free drift and the set the filter's own. For "cam" it is `steps`
    held CAM steps, then free drift, and the set is the level h^steps >= 0 of `CamBarrier`: the
    filter's set is the union of the levels 0 to m, and only on its own level does a state claim
    that arc.
    """
    if name == "free drift":
        return declared(name).contains, None, 0.0
    cam = barrier("cam")
    drift = cam_drift()
    phi, gamma = cw_discrete(cam.n, cam.dt)
    accel = np.array(cam.accel)
    # The successor as CamBarrier forms it, so the level is the one the filter evaluates.
    push = gamma @ accel

    def contains(x):
        for _ in range(steps):
            if held_segment_bound(x, accel, cam.dt, cam.samples, cam.radius, cam.n) < 0.0:
                return False
            x = phi @ x + push
        return bool(drift.value(x) >= 0.0)

    return contains, accel, steps * cam.dt


def halves(lo, hi):
    """The 2^k boxes made by halving each of the k axes on which [lo, hi] has width."""
    mid = 0.5 * (lo + hi)
    axes = np.flatnonzero(hi > lo)
    pieces = []
    for upper in itertools.product((False, True), repeat=axes.size):
        piece_lo, piece_hi = lo.copy(), hi.copy()
        for axis, up in zip(axes, upper, strict=True):
            (piece_lo if up else piece_hi)[axis] = mid[axis]
        pieces.append((piece_lo, piece_hi))
    return pieces


def cover_piece(n, radius, duration, max_boxes, name, steps, piece):
    lo, hi = piece
    samples = round(duration / COVER_SPACING) + 1
    # The piece is already one halving deep on every axis.
    depth = COVER_DEPTH - 1
    contains, accel, burn_time = arc(name, steps)
    return drift_cover(
        n,
        lo,
        hi,
        radius,
        duration,
        samples,
        contains,
        max_boxes,
        depth,
        accel=accel,
        burn_time=burn_time,
    )


def in_set_starts(contains, count, lo, hi, root):
    """`count` states of the declared set, drawn from the box as `audit_filter` draws its
    inside starts from the same root seed."""
    rng = np.random.default_rng(np.random.SeedSequence(root).spawn(2)[0])
    starts = []
    while len(starts) < count:
        x = rng.uniform(lo, hi)
        if contains(x):
            starts.append(x)
    return np.array(starts)


def cover_arcs(filters):
    """(name, steps) of each cover the selected filters need, every set covered once."""
    free = barrier("free drift")
    arcs = []
    if "predictive" in filters:
        assert barrier("predictive") == free, "the predictive set is not the free-drift set"
    if "free drift" in filters:
        arcs.append(("free drift", 0))
    if "cam" in filters:
        assert cam_drift() == free, "the cam level 0 is not the free-drift set"
        arcs += [("cam", steps) for steps in range(1, barrier("cam").steps + 1)]
    return arcs


def cover(env, args, failed):
    cfg = env.cfg
    start_box = tuple(zip(*cfg.start_pos_range, *cfg.start_vel_range, strict=True))
    duration = (cfg.max_steps + 1) * cfg.dt
    for (name, steps), (label, box) in itertools.product(
        cover_arcs(args.filters), (("scenario starts", start_box), ("near the keep-out", NEAR))
    ):
        t0 = time.perf_counter()
        lo, hi = (np.asarray(corner, dtype=float) for corner in box)
        pieces = halves(lo, hi)
        piece = functools.partial(
            cover_piece,
            env.n,
            cfg.keep_out_radius,
            duration,
            max(1, args.boxes // len(pieces)),
            name,
            steps,
        )
        if args.workers == 1:
            covers = [piece(p) for p in pieces]
        else:
            # The pieces and their budgets are fixed before the pool starts, so the cover does
            # not depend on how they are shared out.
            with ProcessPoolExecutor(args.workers) as pool:
                covers = list(pool.map(piece, pieces))
        root = secrets.randbits(128) if args.seed is None else args.seed
        starts = in_set_starts(arc(name, steps)[0], args.starts, lo, hi, root)
        boxes_lo, boxes_hi = np.concatenate([c.certified_boxes for c in covers]).transpose(1, 0, 2)
        covered = sum(
            bool(np.any(np.all((boxes_lo <= x) & (x <= boxes_hi), axis=1))) for x in starts
        )
        counterexamples = [found for c in covers for found in c.counterexamples]
        passed = not counterexamples
        short = (
            "" if covered >= COVERED_SHARE * len(starts) else f" coverage below {COVERED_SHARE:.0%}"
        )
        motion = "drift" if steps == 0 else f"cam {steps}"
        print(
            f"{name:12s} {motion:7s} {label:18s} cover   "
            f"certified {sum(c.certified for c in covers) / len(pieces):.5f} "
            f"excluded {sum(c.excluded for c in covers) / len(pieces):.5f} "
            f"undecided {sum(c.undecided for c in covers) / len(pieces):.5f} "
            f"boxes {sum(c.boxes for c in covers)} counterexamples {len(counterexamples)} "
            f"starts in certified boxes {covered} of {len(starts)} "
            f"({100.0 * covered / len(starts):.1f}%) seed {root:#034x} "
            f"{time.perf_counter() - t0:.0f} s {'pass' if passed else 'FAIL'}{short}",
            flush=True,
        )
        for x, margin, t in counterexamples:
            print(f"  counterexample {np.array2string(x, precision=6)} {margin:.4f} m at {t:.0f} s")
        if not passed:
            failed.append(f"{name}, {motion}, cover, {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=None)
    parser.add_argument("--boxes", type=int, default=20000)
    parser.add_argument("--parts", choices=("audit", "falsify", "both", "cover"), default="both")
    parser.add_argument("--filters", nargs="+", choices=tuple(FILTERS), default=tuple(FILTERS))
    args = parser.parse_args()
    args.filters = [name for name in FILTERS if name in args.filters]
    env = ApproachEnv()
    failed: list[str] = []
    t_all = time.perf_counter()
    for name in args.filters if args.parts in ("audit", "both") else ():
        make = FILTERS[name]
        for label, mode, box, nominal in audits():
            t0 = time.perf_counter()
            found = audit_filter(
                make(),
                env,
                nominal=nominal,
                starts=args.starts,
                mode=mode,
                box=box,
                seed=secrets.randbits(128) if args.seed is None else args.seed,
                workers=args.workers,
            )
            print(line(name, label, found, time.perf_counter() - t0), flush=True)
            if not found.passed:
                failed.append(f"{name}, {found.nominal_name}, {label}")
    if args.parts in ("falsify", "both"):
        falsify(env, args, failed)
    if args.parts == "cover":
        cover(env, args, failed)
    print(f"total {time.perf_counter() - t_all:.0f} s")
    assert not failed, f"failed: {failed}"


if __name__ == "__main__":
    main()
