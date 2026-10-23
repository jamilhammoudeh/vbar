"""Audit of the certified keep-out filters over seeded starts inside and outside their safe sets.

Run with `python benchmarks/filter_audit.py [--starts 1000] [--budget 2000] [--workers N]
[--seed S] [--boxes 20000] [--parts audit|falsify|both|cover]`. Every
audit is on approach-vbar (500 km, keep-out 50 m, box 0.02 m/s^2, dt 10 s, 360 steps) with
perfect navigation and no execution error, flown by `vbar.safety.audit.audit_filter`; the full
run takes hours on one core, so the runs are shared over `--workers` processes (all cores by
default).

Filters: `CbfFilter` at the pole rule's limit (poles 0.06 1/s, gamma 0.06 1/s, v_cap 0.1 m/s)
and below it (poles 0.03 1/s, gamma 0.03 1/s, v_cap 0.2 m/s), and `BackupFilter` with the
free-drift and the braking barrier. Each one is audited

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

`--parts cover` runs neither and checks the free-drift filter's declared set without sampling:
`vbar.reach.drift_cover` splits the scenario's start box and the box near the keep-out into
boxes whose free drift over the audited run and one step more (3610 s) is certified clear of
the keep-out, boxes from which every drift enters it, and undecided boxes, asking the declared
set at each box center. Each box is first halved once along every axis and the 64 parts are
covered in `--workers` processes with an equal share of `--boxes`, so the result does not
depend on the worker count. The script asserts no counterexample, a center in the declared set
whose drift enters by more than 1 mm, and that at least 99% of `--starts` starts drawn from the
declared set as the inside audit draws them lie in certified boxes. Each of the two boxes draws
its own 128-bit root seed for those starts and prints it; `--seed` uses one given seed for both.
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
from vbar.reach import drift_cover
from vbar.safety.audit import audit_filter, falsify_filter
from vbar.safety.cbf import BackupFilter, CbfFilter

NEAR = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))
FAST = ((-80.0, 52.0, -30.0, -0.5, -0.5, -0.5), (80.0, 150.0, 30.0, 0.5, 0.5, 0.5))
FILTERS = {
    "ecbf p 0.06": lambda: CbfFilter(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1),
    "ecbf p 0.03": lambda: CbfFilter(poles=(0.03, 0.03), gamma=0.03, v_cap=0.2),
    "free drift": lambda: BackupFilter(barrier="passive"),
    "braking": lambda: BackupFilter(barrier="braking"),
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
    for name, make in FILTERS.items():
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
def free_drift_contains():
    """`contains` of the free-drift filter reset on approach-vbar, built once per process."""
    declared = BackupFilter(barrier="passive")
    declared.reset(ApproachEnv().agent_view())
    return declared.contains


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


def cover_piece(n, radius, duration, max_boxes, piece):
    lo, hi = piece
    samples = round(duration / COVER_SPACING) + 1
    # The piece is already one halving deep on every axis.
    depth = COVER_DEPTH - 1
    return drift_cover(
        n, lo, hi, radius, duration, samples, free_drift_contains(), max_boxes, depth
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


def cover(env, args, failed):
    cfg = env.cfg
    start_box = tuple(zip(*cfg.start_pos_range, *cfg.start_vel_range, strict=True))
    duration = (cfg.max_steps + 1) * cfg.dt
    for label, box in (("scenario starts", start_box), ("near the keep-out", NEAR)):
        t0 = time.perf_counter()
        lo, hi = (np.asarray(corner, dtype=float) for corner in box)
        pieces = halves(lo, hi)
        piece = functools.partial(
            cover_piece,
            env.n,
            cfg.keep_out_radius,
            duration,
            max(1, args.boxes // len(pieces)),
        )
        if args.workers == 1:
            covers = [piece(p) for p in pieces]
        else:
            # The pieces and their budgets are fixed before the pool starts, so the cover does
            # not depend on how they are shared out.
            with ProcessPoolExecutor(args.workers) as pool:
                covers = list(pool.map(piece, pieces))
        root = secrets.randbits(128) if args.seed is None else args.seed
        starts = in_set_starts(free_drift_contains(), args.starts, lo, hi, root)
        boxes_lo, boxes_hi = np.concatenate([c.certified_boxes for c in covers]).transpose(1, 0, 2)
        covered = sum(
            bool(np.any(np.all((boxes_lo <= x) & (x <= boxes_hi), axis=1))) for x in starts
        )
        counterexamples = [found for c in covers for found in c.counterexamples]
        passed = not counterexamples and covered >= COVERED_SHARE * len(starts)
        print(
            f"{'free drift':12s} {'drift':7s} {label:18s} cover   "
            f"certified {sum(c.certified for c in covers) / len(pieces):.5f} "
            f"excluded {sum(c.excluded for c in covers) / len(pieces):.5f} "
            f"undecided {sum(c.undecided for c in covers) / len(pieces):.5f} "
            f"boxes {sum(c.boxes for c in covers)} counterexamples {len(counterexamples)} "
            f"starts in certified boxes {covered} of {len(starts)} "
            f"({100.0 * covered / len(starts):.1f}%) seed {root:#034x} "
            f"{time.perf_counter() - t0:.0f} s {'pass' if passed else 'FAIL'}",
            flush=True,
        )
        for x, margin, t in counterexamples:
            print(f"  counterexample {np.array2string(x, precision=6)} {margin:.4f} m at {t:.0f} s")
        if not passed:
            failed.append(f"free drift, cover, {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=None)
    parser.add_argument("--boxes", type=int, default=20000)
    parser.add_argument("--parts", choices=("audit", "falsify", "both", "cover"), default="both")
    args = parser.parse_args()
    env = ApproachEnv()
    failed: list[str] = []
    t_all = time.perf_counter()
    for name, make in FILTERS.items() if args.parts in ("audit", "both") else ():
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
