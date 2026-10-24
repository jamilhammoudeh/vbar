"""Comparison of the predictive filter with the barrier filter and the free-drift backup filter.

Run with `python benchmarks/predictive_comparison.py [--parts table|states|both] [--starts 40]
[--states 50] [--inside 25] [--duration 2000] [--workers N]` from the repository root. The
filters are `PredictiveFilter()`, `CbfFilter()` and `BackupFilter(barrier="passive")` with their
defaults, on approach-vbar (500 km, keep-out 50 m, dt 10 s, box 0.02 m/s^2) with perfect
navigation. The checks are those of `tests/test_safety_comparison.py`, whose tests run them on
one start and one state:

1. The table: `--starts` approach-vbar starts inside all three declared sets, in the seed-1903
   order of `benchmarks/filter_known_answers.py`, flown for `--duration` s with the PD nominal and
   the reckless one, full thrust toward the velocity -r / 100 s, which enters the keep-out
   unfiltered, each through each filter and without it.
   Per filter and nominal it prints F1 to F6, F5f, F5l, F11 and F11h and the runs with a
   violation, a certified in-step keep-out bound below -1 mm. The free-drift filters must
   certify "nominal" on every run with no violation.
2. The states: `--states` fast tangential passes, drawn from a fixed 128-bit root seed, where the
   keep-out row of `CbfFilter` asks more than any command in the box can give (Ames et al.,
   *Control Barrier Functions: Theory and Applications*, ECC 2019, sec. IV.B). On each,
   `CbfFilter` must answer "infeasible_certified" with a verified proof; on at least `--inside`
   of them, inside the free-drift sets of both other filters, the predictive and backup filters
   must be certified at the first decision and fly the reckless nominal for `--duration` s without
   a violation.

The table runs as one process per filter and nominal and the states as one more process, at
most `--workers` at a time. Each table process draws the same starts and flies them with the same
seeds as one call over all filters, so the table does not depend on the worker count; F6 and F6p
are wall times and do.
"""

import argparse
import importlib
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

STARTS = 40
DURATION = 2000.0
STATES = 50
INSIDE = 25

COLUMNS = (
    ("F1", "intervention_rate", "{:.3f}"),
    ("F1c", "clip_rate", "{:.3f}"),
    ("F2", "intervention_size", "{:.2e}"),
    ("F2n", "intervention_size_normalized", "{:.3f}"),
    ("F3", "safety_dv_cost", "{:+.4f}"),
    ("F4", "filter_dependence", "{:+.3f}"),
    ("F5", "infeasible_steps", "{:.0f}"),
    ("F5b", "solver_failures", "{:.0f}"),
    ("F6", "filter_compute_median", "{:.4f}"),
    ("F6p", "filter_compute_p95", "{:.4f}"),
    ("F5f", "fallback_count", "{:.0f}"),
    ("F5l", "fallback_level_max", "{:.0f}"),
    ("F11h", "certificate_horizon_min", "{:.0f}"),
    ("viol", "violations", "{:d}"),
    ("runs", "runs", "{:d}"),
)


def from_tests(name):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    return importlib.import_module(name)


def table_part(label, nominal, starts, duration):
    """The table row of one filter and nominal."""
    checks = from_tests("test_safety_comparison")
    table = checks.check_comparison_table(starts, duration, filters=(label,), nominals=(nominal,))
    return table[label][nominal]


def states_part(states, inside, duration):
    """The number of CBF-infeasible passes inside the free-drift sets, after checking them all."""
    return from_tests("test_safety_comparison").check_cbf_infeasible_states(
        states, inside, duration
    )


def cell(fmt, value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return fmt.format(value)


def print_table(table, checks):
    print(f"{'filter':12s}{'nominal':10s}" + "".join(f"{h:>10s}" for h, _, _ in COLUMNS) + "  F11")
    for label in checks.FILTERS:
        for nominal in checks.NOMINALS:
            row = table[label][nominal]
            cells = "".join(f"{cell(fmt, row[name]):>10s}" for _, name, fmt in COLUMNS)
            print(f"{label:12s}{nominal:10s}{cells}  {','.join(row['filter_certificates']) or '-'}")


def check_table(table, checks):
    for label in checks.CERTIFIED:
        for nominal, row in table[label].items():
            assert row["filter_certificates"] == ("nominal",), f"{label} {nominal}"
            assert row["certificate_horizon_min"] == math.inf, f"{label} {nominal}"
            assert row["violations"] == 0, f"{label} {nominal}: {row['violations']} violations"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", choices=("table", "states", "both"), default="both")
    parser.add_argument("--starts", type=int, default=STARTS)
    parser.add_argument("--states", type=int, default=STATES)
    parser.add_argument("--inside", type=int, default=INSIDE)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    checks = from_tests("test_safety_comparison")
    t_all = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        if args.parts in ("states", "both"):
            print(f"passes: root seed {checks.PASS_SEED:#034x}", flush=True)
            found = pool.submit(states_part, args.states, args.inside, args.duration)
        if args.parts in ("table", "both"):
            print(f"starts: seed {checks.START_SEED}, batches of {checks.START_BATCH}", flush=True)
            jobs = {
                (label, nominal): pool.submit(
                    table_part, label, nominal, args.starts, args.duration
                )
                for label in checks.FILTERS
                for nominal in checks.NOMINALS
            }
            table = {label: {} for label in checks.FILTERS}
            for (label, nominal), job in jobs.items():
                table[label][nominal] = job.result()
            print(f"table: {args.starts} starts, {args.duration:.0f} s")
            print_table(table, checks)
            check_table(table, checks)
            print(f"free-drift filters certified on every run, no violation ({args.starts} starts)")
        if args.parts in ("states", "both"):
            print(
                f"states: cbf infeasible_certified with a verified proof on {args.states} passes; "
                f"{found.result()} inside the free-drift sets, certified at decision 0 and clear "
                f"for {args.duration:.0f} s"
            )
    print(f"total {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
