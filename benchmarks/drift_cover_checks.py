"""Full counts of the drift cover checks that `tests/test_reach_cover.py` runs on a prefix.

Run with `python benchmarks/drift_cover_checks.py` from the repository root. The tests run three
of the five `drift_clearance` cases and one certified box of 16 states; this runs:

1. All five cases, the 2001- and 4001-sample ones included: the stacked `drift_clearance` equals
   the computation with every sample set built as its own zonotope to 1e-12 of the 50 m scale,
   and equals the values the per-sample `drift_clearance` returned before the sample sets were
   stacked, stored below.
2. 50 certified boxes from covers of boxes straddling the passive limit at the hold: 64 drawn
   states of each drift clear at 1 s steps, and every counterexample flown again enters by more
   than 1 mm.
"""

import importlib
import sys
import time
from pathlib import Path

import pytest

# (found, certified, t) in m, m and s from the per-sample drift_clearance.
PER_SAMPLE_VALUES = {
    "point": (50.0, 49.96177392817392, 0.0),
    "three_metres": (-50.0, -50.10710461955714, 10439.962594459055),
    "sixty_centimetres": (-6.157414466231167, -6.20876771853695, 10933.859682940805),
    "safe_velocities": (0.027334749667822678, 0.015136524539036737, 10184.498583175391),
    "unsafe_velocities": (-0.04329103482808705, -0.152336295996808, 10190.175561203918),
}


def from_tests(name):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    return importlib.import_module(name)


if __name__ == "__main__":
    checks = from_tests("test_reach_cover")
    for case, (half_widths, samples) in checks.CLEARANCE_CASES.items():
        start = time.perf_counter()
        checks.check_stacked_drift_clearance_equals_the_per_sample_sets(case)
        x0 = checks.Zonotope.box(checks.HOLD, half_widths)
        clearance = checks.drift_clearance(checks.N, x0, checks.R, checks.TWO_ORBITS, samples)
        found, certified, t = PER_SAMPLE_VALUES[case]
        assert clearance.found == pytest.approx(found, rel=1e-12, abs=1e-12 * checks.R)
        assert clearance.certified == pytest.approx(certified, rel=1e-12, abs=1e-12 * checks.R)
        assert clearance.t == pytest.approx(t, rel=1e-12)
        elapsed = time.perf_counter() - start
        print(f"stacked equals per-sample, {case:17s}: {samples} samples ({elapsed:.1f} s)")
    start = time.perf_counter()
    checks.check_certified_boxes_drift_clear_and_counterexamples_enter(50, 64)
    elapsed = time.perf_counter() - start
    print(f"50 certified boxes of 64 states clear, counterexamples enter ({elapsed:.1f} s)")
