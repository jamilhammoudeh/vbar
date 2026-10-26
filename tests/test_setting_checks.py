"""Settings that would hide a violation or change a score silently are refused."""

import math

import numpy as np
import pytest

from vbar.constraints import ClosingSpeedLimit, KeepOut, SpeedLimit
from vbar.metrics import MetricSettings
from vbar.scenarios import ApproachConfig, EnvConfig, make

NUMBERS = ["dt", "max_accel", "isp", "mass_kg", "altitude_m", "min_impulse", "delay_s"]


@pytest.mark.parametrize("field", NUMBERS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0, "10"])
def test_every_number_of_the_shared_config_is_checked(field, value):
    with pytest.raises(ValueError, match=field):
        EnvConfig(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_steps": 0},
        {"max_steps": -5},
        {"max_steps": math.nan},
        {"max_steps": 10.0},
        {"violation_samples": -3},
        {"inclination": math.nan},
        {"propellant_kg": 600.0},
        {"propellant_kg": -1.0},
        {"dt": 0.0},
    ],
)
def test_counts_and_limits_of_the_shared_config_are_checked(overrides):
    with pytest.raises(ValueError):
        EnvConfig(**overrides)
    with pytest.raises(ValueError):
        make("approach-vbar", **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"keep_out_radius": math.nan},
        {"keep_out_radius": -50.0},
        {"capture_radius": math.nan},
        {"capture_speed": -0.05},
        {"hold_point": (0.0, math.nan, 0.0)},
        {"hold_point": (0.0, -100.0)},
    ],
)
def test_the_approach_config_checks_its_goal_and_keep_out(overrides):
    # A NaN or negative keep-out radius reported no violation for a run passing 2 m from the target.
    with pytest.raises(ValueError):
        ApproachConfig(**overrides)


def test_constraints_refuse_limits_that_would_hide_a_violation():
    for build in (
        lambda: KeepOut(math.nan),
        lambda: KeepOut(-50.0),
        lambda: SpeedLimit(math.nan, 0.004),
        lambda: SpeedLimit(0.2, -math.inf),
        lambda: ClosingSpeedLimit(0.2, math.inf),
    ):
        with pytest.raises(ValueError):
            build()
    assert KeepOut(50.0).margin(np.zeros(6)) == -50.0
    # Radius 0 is the range itself, which the coast tool's closest approach uses.
    assert KeepOut(0.0).margin(np.array([3.0, 4.0, 0, 0, 0, 0])) == 5.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"collision_radius": -5.0},
        {"collision_radius": math.nan},
        {"drift_horizon": 0.0},
        {"cam_horizon": math.inf},
        {"plume_range": -1.0},
        {"plume_sigma": 4.0},
        {"plume_half_angle": 0.0},
        {"plume_target_area": 0.0},
        {"cam_impulses": ((0.3, 0.0),)},
        {"cam_impulses": ((math.nan, 0.0, 0.0),)},
        {"composite_weights": {"keep_out": -0.7}},
        {"oscillation_threshold": math.nan},
        {"stall_speed": -0.01},
        {"partial_burn_samples": -1},
        {"partial_burn_samples": 2.5},
        {"drift_samples_per_orbit": 4},
        {"stuck_time": math.inf},
    ],
)
def test_metric_settings_refuse_values_that_would_change_a_score_silently(overrides):
    # A collision radius of -5 m turned a passive safe fraction of 0.41 into 1.0.
    with pytest.raises(ValueError):
        MetricSettings(**overrides)
