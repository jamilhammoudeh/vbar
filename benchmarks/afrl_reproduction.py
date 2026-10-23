"""Reproduction of the four run time assurance filters of Dunlap et al. on the Euler docking plant.

Run with `python benchmarks/afrl_reproduction.py [--starts 100] [--workers N] [--seed S]`.
Dunlap, Hibbard, Mote and Hobbs, "Comparing Run Time Assurance Approaches for Safe Spacecraft
Docking" (IEEE Control Systems Letters 6, 2022, arXiv:2110.00447v1). Every start is flown by
`vbar.safety.rta.reproduce` with the paper's LQR primary alone and inside each of the four
filters, for nu1 = 2.15 n (the authors' code) and nu1 = 4 n (the paper). Starts follow
`start_state`: 9.85 km in a uniform direction and 0.866 m/s in an independent uniform direction,
each from its own 128-bit seed spawned from the printed root seed. The same starts are used for
every filter and both slopes.

A sample violates when phi1 < -1e-5 m/s or some |v_i| > v_max + 1e-5 m/s, the resolution of the
speed limit; samples are the Euler steps. At 2.15 n the script asserts:

1. no filter has a violating sample in any run; every filter intervenes in at least 95 runs; the
   primary alone violates in at least 95 runs;
2. explicit optimization keeps min phi1 >= 1e-3 m/s in every run; implicit switching has a median
   of at least 10 switches between command and backup per run, and at least 5 times the median of
   explicit switching; implicit and explicit optimization stay within 98.5 m and 0.02 m/s of each
   other, over their common samples, in at least 95 runs.

At 4 n the same counts are printed and not asserted. Two more start cases from the authors' code
(github.com/kyle-dunlap/rta-for-spacecraft-docking) are flown at 2.15 n, the code's slope, and
reported, never asserted: its standard start, (-5800, 6690, 4700) m at rest (main.py:40), and its
random rule, position uniform in [-5000, 5000] m and velocity uniform in [-1, 1] m/s per axis
(main.py:34-37), drawn from the same run seeds.

The compute table gives each filter's mean and standard deviation of per-run filter time over the
mean of explicit switching, beside Table I of the paper (1.00, 3.25, 11.31, 13.33; SD 0.03, 0.13,
1.05, 0.52); the paper's hardware and code are unknown, and runs shared over `--workers`
processes time each other's load. The script prints every result before it fails, and exits with
status 1 if any assertion fails.
"""

import argparse
import os
import secrets
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from vbar.safety import PassThroughFilter
from vbar.safety.rta import (
    AfrlConstraints,
    AfrlSetup,
    ExplicitOptimizationFilter,
    ExplicitSwitchingFilter,
    ImplicitOptimizationFilter,
    ImplicitSwitchingFilter,
    reproduce,
    start_state,
)

SETUPS = {"2.15n": AfrlSetup.authors_code, "4n": AfrlSetup.paper}
FILTERS = {
    "explicit switching": ExplicitSwitchingFilter,
    "explicit optimization": ExplicitOptimizationFilter,
    "implicit switching": ImplicitSwitchingFilter,
    "implicit optimization": ImplicitOptimizationFilter,
}
PRIMARY = "primary alone"
TABLE_I = {
    "explicit switching": (1.00, 0.03),
    "explicit optimization": (3.25, 0.13),
    "implicit switching": (11.31, 1.05),
    "implicit optimization": (13.33, 0.52),
}
# The speed limit's resolution, m/s.
DEPTH = 1e-5
PAPER_RULE = "paper rule"
AUTHORS_START = "authors' code start"
AUTHORS_RANDOM = "authors' random rule"
# The authors' code, main.py:40: m and m/s, Hill frame.
AUTHORS_STATE = np.array([-5800.0, 6690.0, 4700.0, 0.0, 0.0, 0.0])


def run_seeds(root, count):
    """Independent 128-bit seeds spawned from the root seed."""
    children = np.random.SeedSequence(root).spawn(count)
    return [int.from_bytes(c.generate_state(4, np.uint32).tobytes(), "little") for c in children]


def authors_random_start(seed):
    """The authors' code's random start, main.py:34-37: uniform per axis in a box."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.uniform(-5000.0, 5000.0, 3), rng.uniform(-1.0, 1.0, 3)])


def start(case, setup, seed):
    if case == AUTHORS_START:
        return AUTHORS_STATE
    if case == AUTHORS_RANDOM:
        return authors_random_start(seed)
    return start_state(setup, seed)


def fly(task):
    """Fly one start and return what the criteria need."""
    case, variant, name, seed = task
    setup = SETUPS[variant]()
    flown = PassThroughFilter() if name == PRIMARY else FILTERS[name](setup=setup)
    run = reproduce(flown, setup, start(case, setup, seed))
    phi1 = AfrlConstraints(setup).values(run.states)[:, 0]
    excess = float(np.max(np.abs(run.states[:, 3:]))) - setup.v_max
    return {
        "case": case,
        "variant": variant,
        "name": name,
        "seed": seed,
        "violated": bool(np.min(phi1) < -DEPTH or excess > DEPTH),
        "min_phi1": float(np.min(phi1)),
        "excess": excess,
        "intervened": any(r.intervened for r in run.results),
        "switches": int(np.sum(run.backup[1:] != run.backup[:-1])),
        "steps": len(run.results),
        "docked": bool(np.linalg.norm(run.states[-1, :3]) <= setup.stop_radius),
        "filter_s": run.filter_s,
        "states": run.states if "optimization" in name else None,
    }


def apart(a, b):
    """Largest position (m) and velocity (m/s) difference over the common samples."""
    k = min(len(a), len(b))
    dr = np.linalg.norm(a[:k, :3] - b[:k, :3], axis=1)
    dv = np.linalg.norm(a[:k, 3:] - b[:k, 3:], axis=1)
    return float(np.max(dr)), float(np.max(dv))


def report(case, variant, runs):
    """Print one case's counts and table; return the criteria as (label, passed)."""
    by = {name: [r for r in runs if r["name"] == name] for name in (*FILTERS, PRIMARY)}
    starts = len(by[PRIMARY])
    print(f"\n{case}, nu1 = {variant}, {starts} starts")
    print(
        f"{'':24s} {'violating':>9s} {'intervene':>9s} {'min phi1':>10s} {'v excess':>10s}"
        f" {'med switch':>10s} {'docked':>6s} {'mean steps':>10s}"
    )
    for name, rows in by.items():
        print(
            f"{name:24s} {sum(r['violated'] for r in rows):9d}"
            f" {sum(r['intervened'] for r in rows):9d}"
            f" {min(r['min_phi1'] for r in rows):10.3e} {max(r['excess'] for r in rows):10.3e}"
            f" {np.median([r['switches'] for r in rows]):10.1f}"
            f" {sum(r['docked'] for r in rows):6d} {np.mean([r['steps'] for r in rows]):10.1f}"
        )
    keeps = sum(r["min_phi1"] >= 1e-3 for r in by["explicit optimization"])
    implicit = np.median([r["switches"] for r in by["implicit switching"]])
    explicit = np.median([r["switches"] for r in by["explicit switching"]])
    pairs = [
        apart(a["states"], b["states"])
        for a, b in zip(by["implicit optimization"], by["explicit optimization"], strict=True)
    ]
    close = sum(dr <= 98.5 and dv <= 0.02 for dr, dv in pairs)
    print(f"explicit optimization runs with min phi1 >= 1e-3 m/s: {keeps}")
    print(f"median switches: implicit {implicit:g}, explicit {explicit:g}")
    print(
        f"optimization filters within 98.5 m and 0.02 m/s: {close} runs; largest "
        f"{max(dr for dr, _ in pairs):.3f} m and {max(dv for _, dv in pairs):.4f} m/s"
    )
    if case == PAPER_RULE:
        base = np.mean([r["filter_s"] for r in by["explicit switching"]])
        print(f"{'compute, over explicit switching':34s} {'mean':>6s} {'SD':>6s}   Table I (SD)")
        for name, (paper_mean, paper_sd) in TABLE_I.items():
            times = np.array([r["filter_s"] for r in by[name]]) / base
            print(
                f"{name:34s} {times.mean():6.2f} {times.std(ddof=1):6.2f}"
                f"   {paper_mean:.2f} ({paper_sd:.2f})"
            )
    checks = [
        (f"{name}: no violating run", not any(r["violated"] for r in by[name])) for name in FILTERS
    ]
    checks += [
        (f"{name}: intervenes in >= 95 runs", sum(r["intervened"] for r in by[name]) >= 95)
        for name in FILTERS
    ]
    checks += [
        ("primary alone: violates in >= 95 runs", sum(r["violated"] for r in by[PRIMARY]) >= 95),
        (
            "explicit optimization: min phi1 >= 1e-3 m/s in every run",
            keeps == len(by["explicit optimization"]),
        ),
        ("implicit switching: median switches >= 10", implicit >= 10),
        ("implicit switching: median >= 5 x explicit switching", implicit >= 5 * explicit),
        ("optimization filters within 98.5 m and 0.02 m/s in >= 95 runs", close >= 95),
    ]
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=100)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    root = args.seed
    while root is None or (args.seed is None and root < 2**64):
        root = secrets.randbits(128)
    if not 2**64 <= root < 2**128:
        raise SystemExit("--seed must lie in [2**64, 2**128)")
    print(f"root seed {root}")
    seeds = run_seeds(root, args.starts)
    names = (*FILTERS, PRIMARY)
    cases = [(PAPER_RULE, variant, seeds) for variant in SETUPS]
    cases += [(AUTHORS_START, "2.15n", [None]), (AUTHORS_RANDOM, "2.15n", seeds)]
    tasks = [
        (case, variant, name, seed)
        for case, variant, case_seeds in cases
        for name in names
        for seed in case_seeds
    ]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(fly, tasks))
    print(f"flown {len(runs)} runs in {time.perf_counter() - t0:.0f} s on {args.workers} workers")
    failed = []
    for case, variant, _ in cases:
        flown = [r for r in runs if r["case"] == case and r["variant"] == variant]
        checks = report(case, variant, flown)
        if case == PAPER_RULE and variant == "2.15n":
            for label, passed in checks:
                print(f"{'PASS' if passed else 'FAIL'}  {label}")
                if not passed:
                    failed.append(label)
    if failed:
        raise SystemExit(f"{len(failed)} criteria failed at nu1 = 2.15 n")


if __name__ == "__main__":
    main()
