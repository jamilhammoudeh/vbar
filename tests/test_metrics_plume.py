import math

import numpy as np
import pytest
from metric_cases import ConstantAgent, episode
from scipy.integrate import quad

from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.metrics import MetricSettings, exhausts, plume_metrics, plume_normalization

N = mean_motion(EARTH.radius + 500e3)
SIGMA = math.radians(13.0)  # Fehse 4.2.4: typical 1-sigma plume half-angle
PLUME = MetricSettings(plume_range=50.0, partial_burn_samples=0)


def test_plume_pressure_integrates_to_the_thrust_over_a_unit_sphere():
    # Midpoint rule on a fine latitude-longitude grid, independent of the quadrature in the code.
    theta = (np.arange(4000) + 0.5) * np.pi / 4000
    sphere = (
        np.sum(np.exp(-0.5 * (theta / SIGMA) ** 2) * np.sin(theta)) * (np.pi / 4000) * 2 * np.pi
    )
    assert plume_normalization(SIGMA) == pytest.approx(sphere, rel=1e-6)
    # A narrow plume integrates like a flat Gaussian: 2 pi sigma^2.
    assert plume_normalization(0.01) == pytest.approx(2 * np.pi * 0.01**2, rel=1e-4)


def test_box_thrusters_fire_per_axis_and_a_ball_engine_along_the_command():
    box = exhausts(np.array([0.01, -0.02, 0.0]), "box")
    assert [tuple(d) for d, _ in box] == [(-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    assert [a for _, a in box] == [0.01, 0.02]
    (ball,) = exhausts(np.array([0.0, 0.03, 0.04]), "ball")
    assert np.allclose(ball[0], [0.0, -0.6, -0.8]) and ball[1] == pytest.approx(0.05)


def test_braking_toward_the_target_gives_the_closed_form_plume_impulse():
    # At rest 30 m behind, thrusting away from the target (-y): the exhaust points at it.
    start = np.array([0.0, -30.0, 0.0, 0.0, 0.0, 0.0])
    accel = np.array([0.0, -0.02, 0.0])
    ep = episode(ConstantAgent([0.0, -1.0, 0.0]), start_state=tuple(start), max_steps=1)
    m = plume_metrics(ep, PLUME)

    def density(t):
        phi, gamma = cw_discrete(N, t)
        r = (phi @ start + gamma @ accel)[:3]
        theta = math.atan2(np.linalg.norm(np.cross([0, 1, 0], -r)), -r[1])
        return math.exp(-0.5 * (theta / SIGMA) ** 2) / (r @ r)

    exposure, _ = quad(density, 0.0, 10.0, epsrel=1e-12)
    expected = 500.0 * 0.02 * 1.0 / plume_normalization(SIGMA) * exposure
    assert m["plume_impulse"] == pytest.approx(expected, rel=1e-8)
    assert m["plume_time"] == pytest.approx(10.0)
    assert m["plume_margin"] == pytest.approx(-math.radians(39.0), abs=1e-6)


def test_thrusting_toward_the_target_puts_the_exhaust_behind():
    ep = episode(
        ConstantAgent([0.0, 1.0, 0.0]), start_state=(0.0, -30.0, 0.0, 0, 0, 0), max_steps=1
    )
    m = plume_metrics(ep, PLUME)
    assert m["plume_time"] == 0.0
    # The along-track thrust couples into radial motion, so the line of sight tilts a little.
    phi, gamma = cw_discrete(N, np.linspace(0.0, 10.0, 10001))
    r = (phi @ np.array([0.0, -30.0, 0.0, 0, 0, 0]) + gamma @ np.array([0.0, 0.02, 0.0]))[:, :3]
    angles = np.arctan2(
        np.linalg.norm(np.cross([0.0, -1.0, 0.0], -r), axis=1), -r @ [0.0, -1.0, 0.0]
    )
    assert m["plume_margin"] == pytest.approx(angles.min() - math.radians(39.0), abs=1e-6)
    assert np.pi - angles.min() < 1e-3
    # exp(-pi^2 / (2 sigma^2)) is about 3e-42: no measurable force.
    assert m["plume_impulse"] < 1e-40


def test_thrust_outside_the_plume_range_does_not_count_and_no_range_means_no_metric():
    far = episode(
        ConstantAgent([0.0, -1.0, 0.0]), start_state=(0.0, -80.0, 0.0, 0, 0, 0), max_steps=1
    )
    m = plume_metrics(far, PLUME)
    assert m["plume_impulse"] == 0.0 and m["plume_time"] == 0.0 and m["plume_margin"] == math.inf
    assert math.isnan(plume_metrics(far)["plume_impulse"])


def test_the_scenarios_plume_range_is_used_unless_the_settings_give_one():
    start = (0.0, -30.0, 0.0, 0.0, 0.0, 0.0)
    declared = episode(
        ConstantAgent([0.0, -1.0, 0.0]), start_state=start, max_steps=1, goal={"plume_range": 50.0}
    )
    plain = episode(ConstantAgent([0.0, -1.0, 0.0]), start_state=start, max_steps=1)
    no_samples = MetricSettings(partial_burn_samples=0)
    assert plume_metrics(declared, no_samples)["plume_impulse"] == pytest.approx(
        plume_metrics(plain, PLUME)["plume_impulse"], rel=1e-12
    )
    overridden = MetricSettings(plume_range=10.0, partial_burn_samples=0)
    assert plume_metrics(declared, overridden)["plume_impulse"] == 0.0


def test_entering_the_plume_range_mid_step_counts_only_the_time_inside():
    # Closing at 1 m/s from 55 m while braking: inside 50 m after about 5 s.
    start = (0.0, -55.0, 0.0, 0.0, 1.0, 0.0)
    ep = episode(ConstantAgent([0.0, -1.0, 0.0]), start_state=start, max_steps=1)
    m = plume_metrics(ep, PLUME)
    phi, gamma = cw_discrete(N, np.linspace(0.0, 10.0, 100001))
    states = phi @ np.array(start) + gamma @ np.array([0.0, -0.02, 0.0])
    inside = np.linalg.norm(states[:, :3], axis=1) < 50.0
    assert m["plume_time"] == pytest.approx(10.0 * inside.mean(), abs=2e-4)


def test_plume_impulse_does_not_depend_on_the_decision_step():
    start = (0.0, -45.0, 3.0, 0.0, 0.3, 0.0)
    a = plume_metrics(
        episode(ConstantAgent([0.2, -0.6, 0.1]), start_state=start, max_steps=4), PLUME
    )
    b = plume_metrics(
        episode(ConstantAgent([0.2, -0.6, 0.1]), start_state=start, max_steps=16, dt=2.5), PLUME
    )
    # The thrust uses the mass at the start of each step; shorter steps track the falling mass
    # more closely, a 1e-5 relative change over 40 s at 0.02 m/s^2.
    assert b["plume_impulse"] == pytest.approx(a["plume_impulse"], rel=1e-4)
    assert b["plume_time"] == pytest.approx(a["plume_time"], abs=1e-6)
    assert b["plume_margin"] == pytest.approx(a["plume_margin"], abs=1e-6)
