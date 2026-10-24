"""Comparison of the predictive filter with the barrier filter and the free-drift backup filter.

Approach-vbar at 500 km, keep-out 50 m, dt 10 s, box 0.02 m/s^2, perfect navigation. The filters
are `PredictiveFilter()`, `CbfFilter()` and `BackupFilter(barrier="passive")` with their
defaults. A violation is a certified in-step keep-out bound below -1 mm, the keep-out resolution.

The interface these tests assume:

- `benchmarks/predictive_comparison.py` imports this module and runs its two checks at the full
  sizes, which it holds as module constants: `STARTS = 40` starts and `DURATION = 2000.0` s for
  `check_comparison_table`, and `STATES = 50` CBF-infeasible states, of which at least
  `INSIDE = 25` lie in the free-drift set, for `check_cbf_infeasible_states`. Importing it runs
  nothing.
- `check_comparison_table` flies each start with a PD and a reckless nominal through each filter
  and without it by `record_pair`, and gives per filter and nominal F1, F1c, F2, F2n, F3, F4, F5,
  F5b, F6, F6p, F5f, F5l, F11, F11h and the count of runs with a violation. The reckless nominal
  is full thrust toward the velocity -r / 100 s, not straight at the target, which misses the
  keep-out from these starts.
- `check_cbf_infeasible_states` draws fast tangential passes from a recorded 128-bit seed and
  keeps those where the keep-out row of `CbfFilter` has b > a_max |rhat|_1, so no command in the
  box meets it (the single-row feasibility condition under a box limit, Ames et al. 2019,
  sec. IV). On each,
  `CbfFilter` must answer "infeasible_certified" with a verified proof; on those inside the
  free-drift sets of both other filters, the predictive and backup filters must be certified at
  the first decision and fly the reckless nominal for the given duration without a violation.
"""

import functools
import importlib.util
import math
import time
from pathlib import Path

import numpy as np

from vbar.agents.base import action_scale
from vbar.agents.pd import PDAgent
from vbar.constraints import KeepOut
from vbar.metrics.episode import record_pair
from vbar.metrics.filter import filter_behaviour, filter_metrics, paired_filter_metrics
from vbar.safety import FilteredAgent
from vbar.safety.barriers import KeepOutEcbf
from vbar.safety.cbf import BackupFilter, CbfFilter
from vbar.safety.predictive import PredictiveFilter
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.spacecraft import PerfectNav

A_MAX = 0.02
DT = 10.0
N = ApproachEnv().n
DEPTH = KeepOut.resolution
FAR = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])
# A fixed 128-bit root seed for the fast passes, so the drawn states are the same on every run.
PASS_SEED = 0xD826767B1484EC8344C8A468532CC35E
# The seed and draw order of the approach-vbar starts of benchmarks/filter_known_answers.py.
START_SEED = 1903
START_BATCH = 400

FILTERS = {
    "predictive": PredictiveFilter,
    "cbf": CbfFilter,
    "passive": lambda: BackupFilter(barrier="passive"),
}
CERTIFIED = ("predictive", "passive")
FIELDS = (
    "intervention_rate",
    "clip_rate",
    "intervention_size",
    "intervention_size_normalized",
    "safety_dv_cost",
    "filter_dependence",
    "infeasible_steps",
    "solver_failures",
    "filter_compute_median",
    "filter_compute_p95",
    "fallback_count",
    "fallback_level_max",
    "filter_certificates",
    "certificate_horizon_min",
    "violations",
    "runs",
)


class Reckless:
    """Full command toward the velocity -r / 100 s, which closes on the target head-on.

    A full command straight at the target leaves the start's lateral velocity uncorrected and
    passes the comparison starts about 60 m out, so no filter acts; this one cancels it and
    enters the keep-out unfiltered within 400 s from each of the first three starts.
    """

    name = "reckless"

    def reset(self, view, rng=None):
        self.scale = action_scale(view)

    def act(self, obs, info):
        toward = -(obs[:3] / 100.0 + obs[3:6])
        return toward / np.linalg.norm(toward) * self.scale


NOMINALS = {"pd": PDAgent, "reckless": Reckless}


def approach(start, steps):
    start = tuple(float(s) for s in start)
    return ApproachEnv(
        ApproachConfig(dt=DT, max_steps=steps, start_state=start, sensor=PerfectNav())
    )


def ready(make):
    filt = make()
    filt.reset(approach(FAR, 1).agent_view())
    return filt


def a_cw(x):
    return np.array([3 * N * N * x[0] + 2 * N * x[4], -2 * N * x[3], -N * N * x[2]])


def violated(ep):
    return any(rec.segment.minima["keep_out"].lower_bound < -DEPTH for rec in ep.records)


# Part 1: the comparison table.


def comparison_starts(count):
    """The first `count` approach-vbar starts, in seed-1903 order, inside all three declared sets.

    The starts are drawn as table 3 of benchmarks/filter_known_answers.py draws them, in batches
    of 400 over the scenario's start ranges, without that table's five gradient-check states.
    """
    filters = [ready(make) for make in FILTERS.values()]
    rng = np.random.default_rng(START_SEED)
    starts = []
    while len(starts) < count:
        batch = np.column_stack(
            [
                rng.uniform(-200, 200, START_BATCH),
                rng.uniform(200, 600, START_BATCH),
                rng.uniform(-50, 50, START_BATCH),
                rng.uniform(-0.1, 0.1, (START_BATCH, 3)),
            ]
        )
        starts += [x for x in batch if all(f.contains(x) for f in filters)]
    return starts[:count]


def reported(values):
    return [v for v in values if not math.isnan(v)]


def table_row(pairs):
    """One filter and nominal: the episode metrics over the on runs and the paired F3 and F4.

    Rates and sizes are means over the runs that report them, counts are sums, F5l the deepest
    level and F11h the shortest horizon; F6 and F6p are taken over every decision of every run.
    """
    on = [pair.on for pair in pairs]
    per_run = [filter_metrics(ep) | filter_behaviour(ep) for ep in on]
    paired = paired_filter_metrics(on, [pair.off for pair in pairs])
    compute = [
        rec.filter_result.compute_s
        for ep in on
        for rec in ep.records
        if rec.filter_result is not None
    ]

    def mean(name):
        values = reported([m[name] for m in per_run])
        return float(np.mean(values)) if values else math.nan

    def total(name):
        values = reported([m[name] for m in per_run])
        return float(np.sum(values)) if values else math.nan

    levels = reported([m["fallback_level_max"] for m in per_run])
    horizons = reported([m["certificate_horizon_min"] for m in per_run])
    return {
        "intervention_rate": mean("intervention_rate"),
        "clip_rate": mean("clip_rate"),
        "intervention_size": mean("intervention_size"),
        "intervention_size_normalized": mean("intervention_size_normalized"),
        "safety_dv_cost": paired["safety_dv_cost"],
        "filter_dependence": paired["filter_dependence"],
        "infeasible_steps": total("infeasible_steps"),
        "solver_failures": total("solver_failures"),
        "filter_compute_median": float(np.median(compute)),
        "filter_compute_p95": float(np.percentile(compute, 95)),
        "fallback_count": total("fallback_count"),
        "fallback_level_max": max(levels) if levels else math.nan,
        "filter_certificates": tuple(
            sorted({k for m in per_run for k in m["filter_certificates"]})
        ),
        "certificate_horizon_min": min(horizons) if horizons else math.nan,
        "violations": sum(violated(ep) for ep in on),
        "runs": len(on),
    }


def check_comparison_table(starts, duration, filters=tuple(FILTERS), nominals=tuple(NOMINALS)):
    """Fly `starts` comparison starts for `duration` s, each nominal through each filter.

    `filters` and `nominals` name the keys of `FILTERS` and `NOMINALS` to fly, all by default.
    Returns {filter: {nominal: row}}, each row holding `FIELDS`. Each start k is one seed pair,
    episode seed 2k and agent seed 2k + 1, flown with the filter and without it by `record_pair`,
    so a subset gives the rows the full table gives.
    """
    steps = round(duration / DT)
    chosen = comparison_starts(starts)
    table = {}
    for label in filters:
        make = FILTERS[label]
        table[label] = {}
        for nominal in nominals:
            agent = NOMINALS[nominal]
            pairs = [
                record_pair(approach(x0, steps), agent, make(), seed=2 * k, agent_seed=2 * k + 1)
                for k, x0 in enumerate(chosen)
            ]
            table[label][nominal] = table_row(pairs)
    return table


# Part 2: states where the barrier filter is infeasible under the box.


def fast_pass(rng):
    """A Hill state (m, m/s) 60 to 120 m out, at 1 to 3 m/s along a line missing by 52 to 60 m."""
    rhat = rng.normal(size=3)
    rhat /= np.linalg.norm(rhat)
    rho = rng.uniform(60.0, 120.0)
    # Sine of the angle between the velocity and the line to the target: miss distance / range.
    sine = rng.uniform(52.0, 60.0) / rho
    across = rng.normal(size=3)
    across -= (across @ rhat) * rhat
    across /= np.linalg.norm(across)
    v = rng.uniform(1.0, 3.0) * (-math.sqrt(1.0 - sine * sine) * rhat + sine * across)
    return np.concatenate([rho * rhat, v])


def keep_out_row_infeasible(cbf, x):
    """Whether the keep-out row a . u >= b has b > a_max |a|_1: no u in the box meets it."""
    barrier = next(b for b in cbf.declaration.barriers if isinstance(b, KeepOutEcbf))
    a, b = barrier.row(x, a_cw(x))
    return b > A_MAX * float(np.sum(np.abs(a)))


def fly_reckless(filt, start, duration):
    """The first decision's result and the lowest certified keep-out bound (m) of the flight."""
    steps = round(duration / DT)
    env = approach(start, steps)
    pilot = FilteredAgent(Reckless(), filt)
    obs, _ = env.reset(seed=0)
    pilot.reset(env.agent_view(), np.random.default_rng(0))
    first, worst = None, math.inf
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(pilot.act(obs, env.agent_info()))
        if first is None:
            first = pilot.last_result
        worst = min(worst, env.record.segment.minima["keep_out"].lower_bound)
        if terminated or truncated:
            break
    return first, worst


def check_cbf_infeasible_states(states, inside, duration):
    """Check the first `states` CBF-infeasible passes; at least `inside` lie in the drift sets.

    Returns the number of the passes inside the free-drift sets of both other filters.
    """
    cbf = ready(CbfFilter)
    drift_sets = [ready(FILTERS[label]) for label in CERTIFIED]
    view = approach(FAR, 1).agent_view()
    rng = np.random.default_rng(PASS_SEED)
    drawn = found = 0
    while drawn < states:
        x = fast_pass(rng)
        if not keep_out_row_infeasible(cbf, x):
            continue
        drawn += 1
        toward = -x[:3] / np.linalg.norm(x[:3]) * action_scale(view)
        result = cbf.filter(x.copy(), toward)
        assert result.status == "infeasible_certified", f"cbf at {x}: {result.status}"
        assert result.proof is not None and result.proof.verify(), f"cbf at {x}: no proof"
        assert "keep_out" in result.proof.names
        if not all(f.contains(x) for f in drift_sets):
            continue
        found += 1
        for label in CERTIFIED:
            first, worst = fly_reckless(FILTERS[label](), x, duration)
            assert first.status == "feasible" and first.recovering is False, f"{label} at {x}"
            assert first.certificate == "nominal", f"{label} at {x}: {first.certificate}"
            assert first.certificate_horizon == math.inf
            assert worst >= -DEPTH, f"{label} at {x}: keep-out bound {worst:.4f} m"
    assert found >= inside, f"{found} of {states} CBF-infeasible passes in the drift sets"
    return found


@functools.cache
def short_table():
    # By 150 s the reckless nominal closes at 2.7 m/s and each drift filter has acted once.
    return check_comparison_table(1, 150.0)


def test_a_pass_where_the_barrier_row_is_infeasible_is_certified_by_the_drift_filters():
    t0 = time.process_time()
    assert check_cbf_infeasible_states(1, 1, 100.0) == 1
    assert time.process_time() - t0 < 2.0


def test_a_pass_outside_the_drift_sets_is_not_counted_as_inside():
    # The second CBF-infeasible pass of PASS_SEED is not certified clear in free drift.
    assert check_cbf_infeasible_states(2, 1, 100.0) == 1


def test_the_reckless_nominal_makes_every_filter_act_on_the_comparison_start():
    for rows in short_table().values():
        assert rows["reckless"]["intervention_rate"] > 0.0


def test_comparison_row_holds_every_filter_metric_and_the_certificate_columns():
    table = short_table()
    assert set(table) == set(FILTERS)
    for rows in table.values():
        assert set(rows) == set(NOMINALS)
        for row in rows.values():
            assert set(row) == set(FIELDS)
            assert row["runs"] == 1
            assert isinstance(row["filter_certificates"], tuple)
            assert row["filter_compute_p95"] >= row["filter_compute_median"] > 0.0


def test_drift_filters_claim_nominal_safety_and_keep_it_on_the_comparison_starts():
    table = short_table()
    for label in CERTIFIED:
        for row in table[label].values():
            assert row["filter_certificates"] == ("nominal",)
            assert row["certificate_horizon_min"] == math.inf
            assert row["violations"] == 0
    for row in table["cbf"].values():
        assert row["filter_certificates"] == ()
        assert math.isnan(row["certificate_horizon_min"])


def test_benchmark_runs_both_checks_at_the_comparison_sizes():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "predictive_comparison.py"
    assert path.is_file(), "benchmarks/predictive_comparison.py is missing"
    spec = importlib.util.spec_from_file_location("predictive_comparison", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    assert benchmark.STARTS == 40
    assert benchmark.DURATION == 2000.0
    assert benchmark.STATES >= 50
    assert benchmark.INSIDE >= 25
    source = path.read_text(encoding="utf-8")
    assert "check_comparison_table" in source
    assert "check_cbf_infeasible_states" in source
