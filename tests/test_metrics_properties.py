"""Properties every metric must have: convergence, invariance and reproducibility.

The golden fixtures in `tests/golden/metrics_*.json` freeze `evaluate` on three episodes; after an
intended change to a metric, run `python tests/test_metrics_properties.py` to regenerate them and
review the diff.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from metric_cases import FLYBY, ConstantAgent, episode

from vbar import ApproachConfig, ApproachEnv
from vbar.agents import PDAgent
from vbar.constraints import ApproachCone, ClosingSpeedLimit, SpeedLimit
from vbar.dynamics import EARTH, mean_motion
from vbar.metrics import MetricSettings, evaluate, passive_metrics, record_episode

GOLDEN = Path(__file__).parent / "golden"
N = mean_motion(EARTH.radius + 500e3)
GUARDS = (
    SpeedLimit(0.2, 4 * N),
    ClosingSpeedLimit(0.2, 4 * N),
    ApproachCone(np.radians(10.0), active_range=120.0),
)
SETTINGS = MetricSettings(plume_range=60.0, partial_burn_samples=2)
# Wall-clock quantities differ between runs by nature.
WALL_CLOCK = {"latency_median", "latency_p95", "compute"}
# Minima are refined to a resolution: which intervals get refined, and where a bracketing search
# stops, can change with the last bits of a sample, so these agree to that resolution (1 mm for
# ranges, used for every constraint margin; 1e-6 for angles and ratios), not to rounding.
RESOLVED = {
    "min_margin": 1e-3,
    "violation_depth": 1e-3,
    "passive_safety_margin": 1e-3,
    "stuck_thruster_margin": 1e-3,
    "cam_margin": 1e-3,
    "plume_margin": 1e-6,
    "closing_speed_ratio": 1e-6,
}


def cases():
    return {
        "metrics_pd": lambda: episode(
            PDAgent(), start_state=(-150.0, -400.0, 20.0, 0.0, 0.0, 0.0), max_steps=80
        ),
        "metrics_pd_through_target": lambda: episode(
            PDAgent(), start_state=(0.0, 400.0, 0.0, 0.0, 0.0, 0.0), max_steps=50, extra=GUARDS
        ),
        "metrics_flyby": lambda: episode(extra=GUARDS),
    }


def scored(ep):
    return {k: v for k, v in evaluate(ep, SETTINGS).items() if k not in WALL_CLOCK}


def assert_same(actual, expected, key=""):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), key
        for name in expected:
            assert_same(actual[name], expected[name], key)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected, key
    elif isinstance(expected, list | tuple):
        assert list(actual) == list(expected), key
    elif isinstance(expected, int) and isinstance(actual, int):
        assert actual == expected, key
    elif math.isnan(expected):
        assert math.isnan(actual), key
    else:
        floor = RESOLVED.get(key, 1e-12)
        assert actual == pytest.approx(expected, rel=1e-9, abs=floor), key


def compare(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        assert_same(actual[key], expected[key], key)


@pytest.mark.parametrize("name", list(cases()))
def test_metrics_reproduce_the_golden_values(name):
    expected = json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))
    compare(json.loads(json.dumps(scored(cases()[name]()))), expected)


def test_continuous_time_metrics_do_not_depend_on_the_decision_step():
    # A constant command flies the same trajectory whatever the step; every metric defined in
    # continuous time must agree. Those defined at step boundaries are checked separately, and
    # the certified bound S1b tightens with the sample spacing by design.
    boundary = {
        "min_margin_bound",
        "time_to_complete",
        "burn_count",
        "min_time_to_keep_out",
        "passive_safety_margin",
        "passive_safe_fraction",
        "passive_forever_fraction",
        "passive_threat_time",
        "stuck_thruster_margin",
        "cam_margin",
        "unprotected_time",
        "plume_impulse",
    }
    agent = ConstantAgent([0.05, -0.1, 0.02])
    runs = [
        episode(agent, extra=GUARDS, dt=dt, max_steps=int(200 / dt)) for dt in (20.0, 10.0, 5.0)
    ]
    found = [{k: v for k, v in scored(ep).items() if k not in boundary} for ep in runs]
    for other in found[1:]:
        compare(other, found[0])


def test_passive_safety_converges_as_decision_steps_nest():
    # Halving the step adds states to the set the minimum is taken over, so S4 can only fall,
    # and by less each time as the boundaries fill the trajectory.
    margins = []
    for dt in (40.0, 20.0, 10.0, 5.0):
        ep = episode(start_state=FLYBY, dt=dt, max_steps=int(200 / dt))
        margins.append(
            passive_metrics(ep, MetricSettings(partial_burn_samples=0))["passive_safety_margin"]
        )
    steps = -np.diff(margins)
    assert np.all(steps >= -1e-3)
    assert steps[-1] <= steps[0]


def test_all_metrics_are_unchanged_by_mirroring_the_orbit_normal():
    start = np.array([-150.0, -400.0, 20.0, 0.01, 0.0, -0.02])
    mirror = np.array([1, 1, -1, 1, 1, -1])
    a = scored(episode(PDAgent(), start_state=tuple(start), max_steps=40, extra=GUARDS))
    b = scored(episode(PDAgent(), start_state=tuple(start * mirror), max_steps=40, extra=GUARDS))
    compare(b, a)


def test_scaling_lengths_scales_lengths_and_keeps_ratios():
    # Clohessy-Wiltshire motion is linear: doubling every length, the thrust and the abort impulse
    # doubles the trajectory and leaves times, fractions and the order of events alone.
    def fly(k):
        config = {
            "start_state": tuple(k * np.array([-150.0, -400.0, 20.0, 0.0, 0.0, 0.0])),
            "hold_point": (0.0, -100.0 * k, 0.0),
            "capture_radius": 5.0 * k,
            "capture_speed": 0.05 * k,
            "keep_out_radius": 50.0 * k,
            "max_accel": 0.02 * k,
            "max_steps": 80,
        }
        return evaluate(episode(PDAgent(), **config), MetricSettings(cam_dv=0.3 * k))

    one, two = fly(1.0), fly(2.0)
    for name in ("terminal_pos_error", "delta_v", "passive_safety_margin", "cam_margin"):
        assert two[name] == pytest.approx(2.0 * one[name], rel=1e-6, abs=2e-3)
    for name in ("success", "time_to_complete", "progress", "duty_cycle", "passive_safe_fraction"):
        assert two[name] == pytest.approx(one[name], rel=1e-9)
    assert two["min_margin"]["keep_out"] == pytest.approx(
        2.0 * one["min_margin"]["keep_out"], abs=2e-3
    )


def test_metrics_do_not_depend_on_evaluation_order_or_environment_reuse():
    env = ApproachEnv(ApproachConfig(max_steps=20))
    reused = [record_episode(env, PDAgent(), seed=s, agent_seed=0) for s in (3, 4)]
    fresh = [
        record_episode(ApproachEnv(ApproachConfig(max_steps=20)), PDAgent(), seed=s, agent_seed=0)
        for s in (4, 3)
    ]
    forward = [scored(ep) for ep in reused]
    backward = [scored(ep) for ep in fresh][::-1]
    for a, b in zip(forward, backward, strict=True):
        compare(a, b)


if __name__ == "__main__":
    for name, make in cases().items():
        text = json.dumps(scored(make()), indent=1, sort_keys=True)
        (GOLDEN / f"{name}.json").write_text(text + "\n", encoding="utf-8", newline="\n")
        print("wrote", name)
