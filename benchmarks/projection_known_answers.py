"""Full counts of the projection checks that `tests/test_solvers_qp.py` runs on a prefix.

Run with `python benchmarks/projection_known_answers.py` from the repository root. Each check
draws the same seeded instances as its test, which runs the first 100 of them:

1. 500 one-row instances per layout against the support-function feasibility rule and a
   bisection on the dual (box and ball, a_max 0.02 m/s^2).
2. 200 instances of 2 to 6 rows per layout against SLSQP, with the active rows tight.
"""

import importlib
import sys
import time
from pathlib import Path


def from_tests(name):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    return importlib.import_module(name)


if __name__ == "__main__":
    qp_tests = from_tests("test_solvers_qp")
    for layout, norm_ord in (("box", 1), ("ball", 2)):
        start = time.perf_counter()
        qp_tests.check_one_row_against_dual_search(layout, norm_ord, 500)
        print(f"one row, {layout:4s}: 500 instances agree ({time.perf_counter() - start:.1f} s)")
    for layout in ("box", "ball"):
        start = time.perf_counter()
        qp_tests.check_several_rows_against_slsqp(layout, 200)
        print(f"SLSQP,   {layout:4s}: 200 instances agree ({time.perf_counter() - start:.1f} s)")
