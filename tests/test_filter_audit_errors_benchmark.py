"""The sizes and the filter list of benchmarks/filter_audit_errors.py, without running it."""

import importlib.util
import pathlib
import sys
import types

import pytest

from vbar.scenarios import get, per_draw_level
from vbar.spacecraft import GaussianNav, PerfectNav

BENCHMARKS = pathlib.Path(__file__).resolve().parents[1] / "benchmarks"


@pytest.fixture(scope="module")
def bench():
    # The script imports its boxes and nominal from filter_audit.py next to it.
    sys.path.insert(0, str(BENCHMARKS))
    try:
        spec = importlib.util.spec_from_file_location(
            "filter_audit_errors", BENCHMARKS / "filter_audit_errors.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(BENCHMARKS))


def test_full_run_sizes_are_1000_starts_budget_and_runs_at_exceedance_1e_3(bench):
    assert (bench.STARTS, bench.BUDGET, bench.GAUSSIAN_RUNS) == (1000, 1000, 1000)
    assert bench.EXCEEDANCE == 1e-3
    args = bench.parser().parse_args([])
    assert (args.starts, args.budget, args.runs) == (1000, 1000, 1000)
    assert set(args.parts) == {"audit", "falsify", "gaussian"}


def test_covered_filters_and_their_error_kinds(bench):
    assert set(bench.FILTERS) == {
        "free drift",
        "braking",
        "predictive",
        "ecbf p 0.06",
        "ecbf p 0.03",
    }
    assert set(bench.EXECUTION_ONLY) == {"ecbf p 0.06", "ecbf p 0.03"}
    assert bench.NO_CLAIM == {"cam": "no claim under errors"}
    assert set(bench.parser().parse_args([]).filters) == {*bench.FILTERS, "cam"}


def test_navigation_is_the_nav1_sensor_truncated(bench):
    nav1 = get("approach-vbar:nav1").config["sensor"]
    sensor, execution = bench.models("braking", bound=5.0)
    assert isinstance(sensor, GaussianNav)
    assert list(sensor.sigma) == list(nav1.sigma)
    assert list(sensor.bias_sigma) == list(nav1.bias_sigma)
    assert sensor.bound == execution.bound == 5.0
    assert (execution.scale_sigma, execution.noise_sigma) == (0.01, 0.005)


def test_ecbf_runs_without_navigation_error(bench):
    sensor, _ = bench.models("ecbf p 0.06")
    assert isinstance(sensor, PerfectNav)


def test_draw_counts_over_360_steps(bench):
    # Per step 6 navigation draws and 3 + 1 execution draws; the ecbf lines have no navigation.
    assert bench.truncation("braking") == (per_draw_level(1e-3, 3600), 3600)
    assert bench.truncation("ecbf p 0.03") == (per_draw_level(1e-3, 1440), 1440)


def audited(**counts):
    """An audit's counts, all zero but those given, with the audit's own pass rule."""
    fields = {"violations": 0, "uncertified": 0, "infeasible": 0, "solver_failed": 0}
    fields.update(counts)
    fields.setdefault("uncovered", 0)
    clean = not any(fields.values())
    return types.SimpleNamespace(passed=clean, **fields)


@pytest.mark.parametrize("name", ["ecbf p 0.06", "ecbf p 0.03"])
def test_uncertified_line_passes_with_every_decision_uncovered(bench, name):
    # CbfFilter issues no certificate, so all 360 decisions of each of 5 runs are uncovered.
    assert bench.claim(name) == "uncertified"
    assert bench.passed(name, audited(uncovered=1800))
    for count in ("violations", "infeasible", "solver_failed"):
        assert not bench.passed(name, audited(uncovered=1800, **{count: 1}))


@pytest.mark.parametrize("name", ["free drift", "braking", "predictive"])
def test_certified_line_fails_on_one_uncovered_decision(bench, name):
    assert bench.claim(name) == "certified"
    assert bench.passed(name, audited())
    assert not bench.passed(name, audited(uncovered=1))


def searched(verdict, score, **counts):
    """A falsification's worst run, its counts given as for `audited`, with `verdict`."""
    declaration = types.SimpleNamespace(barriers=(), exceedance=1e-3, per_draw_level=1e-6)
    return types.SimpleNamespace(
        verdict=verdict,
        score=score,
        run=audited(**counts),
        declaration=declaration,
        nominal_name="pd",
        evaluations=1,
        rejected=0,
        dispersion=0.0,
        seed=0,
    )


# The verdicts falsify_filter gives each run: a clean run, a run with uncovered decisions (every
# CbfFilter run) and a run violating the keep-out by 50 resolutions.
SEARCHES = {
    "none_found": lambda: searched("none_found", 0.0),
    "uncovered": lambda: searched("uncovered", 0.0, uncovered=1800),
    "violation": lambda: searched("violation", -50.0, violations=1, uncovered=1800),
}


@pytest.mark.parametrize(
    ("name", "verdict", "ok"),
    [
        ("braking", "none_found", True),
        ("braking", "uncovered", False),
        ("braking", "violation", False),
        ("ecbf p 0.06", "none_found", True),
        ("ecbf p 0.06", "uncovered", True),
        ("ecbf p 0.06", "violation", False),
    ],
)
def test_falsify_line_fails_uncovered_only_for_a_certified_filter(
    bench, monkeypatch, name, verdict, ok
):
    found = SEARCHES[verdict]()
    monkeypatch.setattr(bench, "truncation", lambda name: (1e-6, 3600))
    monkeypatch.setattr(bench, "scenario", lambda *a, **k: None)
    monkeypatch.setattr(bench, "declared", lambda *a, **k: None)
    monkeypatch.setattr(bench, "falsify_filter", lambda *a, **k: found)
    args = bench.parser().parse_args(["--seed", "0", "--workers", "1"])
    failed = []
    bench.falsify(name, args, failed)
    # One line per nominal: the PD agent and the inward one.
    assert failed == ([] if ok else [f"{name}, pd, falsify"] * 2)


@pytest.mark.parametrize("name", ["free drift", "predictive"])
@pytest.mark.parametrize(("argv", "grid"), [([], 5.0), (["--grid", "30"], 30.0)])
def test_grid_option_reaches_the_declared_drift_barrier(bench, name, argv, grid):
    args = bench.parser().parse_args(argv)
    level, _ = bench.truncation(name)
    filt = bench.declared(name, level, args.grid)
    filt.reset(bench.scenario(name, level, truncated=True).agent_view())
    (barrier,) = filt.declaration.barriers
    assert barrier.grid == grid
    assert bench.declared_grid(filt.declaration) == f"{grid:g} s"
