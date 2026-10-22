"""Full tables of the keep-out bound checks that `tests/test_constraints.py` runs on a prefix.

Run with `python benchmarks/keep_out_bound_tables.py` from the repository root. The tables are
the seeded flybys and random segments of the tests (200 dense, 200 thrusting, 200 coarse and 320
random), whose tests run the first 100 of each:

1. The first-order bound alone leaves a median gap above 0.1 m on the dense table.
2. The certified bound is never below the first-order bound, on every segment of every table.
3. A doubled curvature bound stays below the exact minimum on every flyby segment and leaves a
   dense gap above the 0.03 m ceiling of the tightness table.
"""

import importlib
import sys
import time
from pathlib import Path


def from_tests(name):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    return importlib.import_module(name)


if __name__ == "__main__":
    tables = from_tests("test_constraints")
    start = time.perf_counter()
    tables.check_first_order_bound_misses_the_tightness_table(None)
    print(f"first-order bound misses the dense table ({time.perf_counter() - start:.1f} s)")
    for table in tables.ALL_TABLES:
        start = time.perf_counter()
        tables.check_certified_bound_is_never_below_the_first_order_bound(table, None)
        size = len(tables.cases(table))
        elapsed = time.perf_counter() - start
        print(f"never below first order, {table:9s}: {size} segments ({elapsed:.1f} s)")
    curvature_bound = tables.KeepOut.curvature_bound
    tables.KeepOut.curvature_bound = lambda self, *args: 2.0 * curvature_bound(self, *args)
    try:
        start = time.perf_counter()
        tables.check_loosened_bound_is_sound_and_outside_the_tightness_table(None)
        print(f"doubled curvature sound and caught ({time.perf_counter() - start:.1f} s)")
    finally:
        tables.KeepOut.curvature_bound = curvature_bound
