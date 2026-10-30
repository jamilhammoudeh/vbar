"""Baseline settings that would run an agent on a model or a gain it cannot use are refused.

Each configuration is checked when it is built, not at the first reset or act: a NaN altitude gave
a Lyapunov agent that reset and then commanded NaN, an infinite position scale gave an LQR agent
with no position feedback, and a NaN iteration cap gave a fuel MPC agent that silently fell back
to zero thrust. A number out of range is a ValueError; a count of the wrong type may be a
TypeError instead.
"""

import math

import pytest

from vbar.agents import DeclaredModel, LQGConfig, LQRConfig, TwoImpulseBound
from vbar.agents.mpc_fuel import FuelMPCConfig

NAN, INF = math.nan, math.inf
WRONG_TYPE = (TypeError, ValueError)


def label(value):
    return str(value) if isinstance(value, dict) else None


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"altitude_m": NAN}, ValueError),
        ({"altitude_m": INF}, ValueError),
        ({"altitude_m": -INF}, ValueError),
        ({"altitude_m": -1.0}, ValueError),
        ({"altitude_m": "500000"}, WRONG_TYPE),
        ({"inclination": NAN}, ValueError),
        ({"inclination": INF}, ValueError),
        ({"inclination": -INF}, ValueError),
        ({"inclination": "0.9"}, WRONG_TYPE),
        ({"dynamics": "kepler"}, ValueError),
        ({"dynamics": "twobody_j3"}, ValueError),
        ({"body": "vulcan"}, ValueError),
    ],
    ids=label,
)
def test_a_declared_model_refuses_an_orbit_or_dynamics_it_cannot_build(overrides, error):
    with pytest.raises(error):
        DeclaredModel(**overrides)


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"max_iter": NAN}, WRONG_TYPE),
        ({"max_iter": INF}, WRONG_TYPE),
        ({"max_iter": 2.5}, WRONG_TYPE),
        ({"max_iter": "100"}, WRONG_TYPE),
        ({"max_iter": 0}, ValueError),
        ({"max_iter": -1}, ValueError),
        ({"tie_break": INF}, ValueError),
        ({"tie_break": NAN}, ValueError),
        ({"tie_break": 0.0}, ValueError),
    ],
    ids=label,
)
def test_the_fuel_mpc_config_refuses_an_iteration_cap_that_is_not_a_whole_number(overrides, error):
    with pytest.raises(error):
        FuelMPCConfig(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"pos_scale_m": INF},
        {"pos_scale_m": NAN},
        {"pos_scale_m": 0.0},
        {"pos_scale_m": -10.0},
        {"vel_scale_mps": INF},
        {"vel_scale_mps": NAN},
        {"vel_scale_mps": 0.0},
        {"vel_scale_mps": -0.03},
        {"accel_scale_mps2": INF},
        {"accel_scale_mps2": NAN},
        {"accel_scale_mps2": 0.0},
        {"accel_scale_mps2": -0.02},
        {"rate_weight": INF},
        {"rate_weight": NAN},
        {"rate_weight": -1.0},
    ],
    ids=label,
)
def test_the_lqr_config_refuses_a_scale_or_weight_that_removes_feedback(overrides):
    with pytest.raises(ValueError):
        LQRConfig(**overrides)


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"grid_points": NAN}, WRONG_TYPE),
        ({"grid_points": INF}, WRONG_TYPE),
        ({"grid_points": 2.5}, WRONG_TYPE),
        ({"grid_points": "400"}, WRONG_TYPE),
        ({"grid_points": 1}, ValueError),
        ({"grid_points": 0}, ValueError),
    ],
    ids=label,
)
def test_the_two_impulse_bound_refuses_a_grid_that_is_not_a_whole_number_of_points(
    overrides, error
):
    with pytest.raises(error):
        TwoImpulseBound(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sigma_pos_m": INF},
        {"sigma_pos_m": NAN},
        {"sigma_pos_m": 0.0},
        {"sigma_vel_mps": INF},
        {"sigma_vel_mps": NAN},
        {"sigma_vel_mps": -1e-3},
        {"process_sigma_mps2": INF},
        {"process_sigma_mps2": NAN},
        {"process_sigma_mps2": -1e-6},
        {"bias_walk": INF},
        {"bias_walk": NAN},
        {"bias_walk": -1e-3},
        {"bias_states": ("x",), "bias_walk": INF},
    ],
    ids=label,
)
def test_the_lqg_config_refuses_noise_levels_that_are_not_finite_and_positive(overrides):
    with pytest.raises(ValueError):
        LQGConfig(**overrides)


@pytest.mark.parametrize(
    "build, settings",
    [
        (DeclaredModel, {"altitude_m": 1.0}),
        (DeclaredModel, {"inclination": -3.0}),
        (DeclaredModel, {"dynamics": "twobody_j2", "body": "Earth"}),
        (FuelMPCConfig, {"max_iter": 1}),
        (LQRConfig, {"pos_scale_m": 1e-6, "vel_scale_mps": 1e-9, "accel_scale_mps2": 1e-9}),
        (LQRConfig, {"rate_weight": 1e6}),
        (TwoImpulseBound, {"grid_points": 2}),
        # Zero process noise or walk is refused when the agent is reset, not here (test_lqg_agent).
        (LQGConfig, {"process_sigma_mps2": 0.0}),
        (LQGConfig, {"bias_states": ("x",), "bias_walk": 0.0}),
        (LQGConfig, {"sigma_pos_m": 1e-6, "sigma_vel_mps": 1e-9, "bias_walk": 1e3}),
    ],
    ids=label,
)
def test_a_setting_just_inside_each_limit_is_still_accepted(build, settings):
    assert build(**settings) is not None
