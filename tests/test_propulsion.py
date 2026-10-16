import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from vbar.conventions import G0
from vbar.dynamics import EARTH, CWDynamics, cw_rhs, mean_motion
from vbar.spacecraft import ExecutionError, Spacecraft


@pytest.mark.parametrize("drive", ["accel", "thrust"])
@pytest.mark.parametrize("layout", ["box", "ball"])
def test_propellant_use_follows_the_rocket_equation(layout, drive):
    rng = np.random.default_rng(0)
    craft = Spacecraft(
        max_accel=0.02, layout=layout, isp=220.0, mass=500.0, propellant=400.0, drive=drive
    )
    dv = propellant = 0.0
    for _ in range(360):
        act = craft.apply(rng.uniform(-1.0, 1.0, 3), 10.0)
        dv += act.dv_l1 if layout == "box" else act.dv_l2
        propellant += act.propellant_kg
    expected = 500.0 * -math.expm1(-dv / (220.0 * G0))
    assert propellant == pytest.approx(expected, rel=1e-9)
    assert craft.mass == pytest.approx(500.0 - expected, rel=1e-12)


def test_box_thrusters_spend_propellant_on_each_axis():
    box = Spacecraft(layout="box").apply([0.6, 0.6, 0.0], 10.0)
    ball = Spacecraft(layout="ball").apply([0.6, 0.6, 0.0], 10.0)
    assert box.dv_l1 == pytest.approx(0.24) and box.dv_l2 == pytest.approx(0.12 * math.sqrt(2))
    assert ball.dv_l1 == box.dv_l1 and ball.dv_l2 == box.dv_l2
    assert box.propellant_kg / ball.propellant_kg == pytest.approx(math.sqrt(2), rel=1e-4)


def test_burn_stops_when_the_propellant_runs_out():
    craft = Spacecraft(max_accel=0.02, layout="ball", isp=220.0, mass=500.0, propellant=0.01)
    exhaust = 220.0 * G0
    burn = exhaust / 0.02 * math.log(500.0 / 499.99)
    act = craft.apply([1.0, 0.0, 0.0], 10.0)
    (t1, a1), (t2, a2) = act.segments
    assert t1 == pytest.approx(burn, rel=1e-12) and t1 + t2 == pytest.approx(10.0, rel=1e-15)
    assert np.array_equal(a1, [0.02, 0.0, 0.0]) and np.array_equal(a2, np.zeros(3))
    assert act.empty and act.propellant_kg == pytest.approx(0.01, rel=1e-12)
    assert craft.mass == 499.99 and craft.propellant == 0.0
    later = craft.apply([1.0, 0.0, 0.0], 10.0)
    assert later.empty and later.propellant_kg == 0.0 and later.dv_l2 == 0.0


def test_default_propellant_never_runs_out():
    craft = Spacecraft(max_accel=0.02, layout="box", mass=500.0)
    for _ in range(1000):
        act = craft.apply([1.0, 1.0, 1.0], 100.0)
        assert not act.empty and len(act.segments) == 1
    # 6000 m/s of 1-norm delta-v at 2157 m/s exhaust speed leaves e^-2.78 of the mass.
    assert craft.mass == pytest.approx(500.0 * math.exp(-6000.0 / (220.0 * G0)), rel=1e-12)


def test_coasting_spends_nothing():
    craft = Spacecraft()
    act = craft.apply(np.zeros(3), 10.0)
    assert act.propellant_kg == 0.0 and act.dv_l1 == 0.0 and craft.mass == 500.0
    ((duration, accel),) = act.segments
    assert duration == 10.0 and np.array_equal(accel, np.zeros(3))


def test_reset_refills_the_tanks():
    craft = Spacecraft(propellant=5.0)
    craft.apply([1.0, 0.0, 0.0], 600.0)
    craft.reset()
    assert craft.mass == 500.0 and craft.propellant == 5.0


def test_invalid_spacecraft_is_rejected():
    with pytest.raises(ValueError, match="layout"):
        Spacecraft(layout="cone")
    with pytest.raises(ValueError, match="propellant"):
        Spacecraft(mass=500.0, propellant=600.0)


def test_box_limits_each_axis_and_ball_keeps_direction():
    box = Spacecraft(layout="box")
    u, saturated, _ = box.limit([2.0, -0.5, -3.0], 10.0)
    assert np.array_equal(u, [1.0, -0.5, -1.0]) and saturated
    ball = Spacecraft(layout="ball")
    u, saturated, _ = ball.limit([2.0, -0.5, -3.0], 10.0)
    assert np.linalg.norm(u) == pytest.approx(1.0) and saturated
    assert np.allclose(u / np.linalg.norm(u), np.array([2.0, -0.5, -3.0]) / np.sqrt(13.25))
    u, saturated, _ = ball.limit([0.6, 0.0, 0.7], 10.0)
    assert np.array_equal(u, [0.6, 0.0, 0.7]) and not saturated


def test_impulses_below_the_minimum_bit_are_dropped():
    # A 0.7 mm/s minimum, as on PRISMA (Alfriend et al., Spacecraft Formation Flying, p. 272),
    # is 0.35 N s for 500 kg; at 0.02 m/s^2 for 10 s a full command gives 200 mm/s.
    craft = Spacecraft(max_accel=0.02, layout="box", mass=500.0, min_impulse=0.35)
    u, saturated, below = craft.limit([0.003, -0.004, 0.5], 10.0)
    assert np.array_equal(u, [0.0, -0.004, 0.5]) and below and not saturated
    act = craft.apply([0.003, 2.0, 0.0], 10.0)
    assert act.below_mib and act.saturated
    assert np.array_equal(act.segments[0][1], [0.0, 0.02, 0.0])
    ball = Spacecraft(max_accel=0.02, layout="ball", mass=500.0, min_impulse=0.35)
    act = ball.apply([0.002, 0.002, 0.0], 10.0)
    assert act.below_mib and act.dv_l2 == 0.0 and act.propellant_kg == 0.0


def applied_profile(actuations):
    """(start time, duration, acceleration) of every segment, in order."""
    t, profile = 0.0, []
    for act in actuations:
        for duration, accel in act.segments:
            profile.append((t, duration, accel))
            t += duration
    return profile


def test_delay_inside_a_step_splits_it_at_the_switch_time():
    craft = Spacecraft(max_accel=0.02, delay=2.5)
    first = craft.apply([1.0, 0.0, 0.0], 10.0)
    second = craft.apply([0.0, -1.0, 0.0], 10.0)
    assert [d for d, _ in first.segments] == [2.5, 7.5]
    assert np.array_equal(first.segments[0][1], np.zeros(3))
    assert np.array_equal(first.segments[1][1], [0.02, 0.0, 0.0])
    assert [d for d, _ in second.segments] == [2.5, 7.5]
    assert np.array_equal(second.segments[0][1], [0.02, 0.0, 0.0])
    assert np.array_equal(second.segments[1][1], [0.0, -0.02, 0.0])


@pytest.mark.parametrize("delay", [0.0, 10.0, 25.0, 33.3])
def test_delay_shifts_the_applied_commands_in_time(delay):
    rng = np.random.default_rng(1)
    commands = rng.uniform(-1.0, 1.0, (12, 3))
    craft = Spacecraft(max_accel=0.02, delay=delay)
    profile = applied_profile([craft.apply(u, 10.0) for u in commands])
    for t, duration, accel in profile:
        midpoint = t + 0.5 * duration - delay
        expected = np.zeros(3) if midpoint < 0.0 else commands[int(midpoint // 10.0)] * 0.02
        assert np.array_equal(accel, expected)
    assert sum(duration for _, duration, _ in profile) == pytest.approx(120.0, rel=1e-15)


def test_pending_lists_commands_not_yet_in_effect():
    craft = Spacecraft(delay=25.0)
    craft.apply([1.0, 0.0, 0.0], 10.0)
    craft.apply([0.0, 1.0, 0.0], 10.0)
    pending = craft.pending()
    assert [t for t, _ in pending] == pytest.approx([5.0, 15.0])
    assert np.array_equal(pending[0][1], [1.0, 0.0, 0.0])
    craft.apply([0.0, 0.0, 1.0], 10.0)
    assert [t for t, _ in craft.pending()] == pytest.approx([5.0, 15.0])


def test_delayed_burns_spend_propellant_by_the_rocket_equation():
    craft = Spacecraft(max_accel=0.02, layout="box", delay=7.0)
    rng = np.random.default_rng(2)
    acts = [craft.apply(rng.uniform(-1.0, 1.0, 3), 10.0) for _ in range(50)]
    dv = sum(a.dv_l1 for a in acts)
    propellant = sum(a.propellant_kg for a in acts)
    assert propellant == pytest.approx(500.0 * -math.expm1(-dv / (220.0 * G0)), rel=1e-9)


def test_constant_thrust_delta_v_matches_the_finite_burn_formula():
    # Montenbruck and Gill, Satellite Orbits, eq 3.129: dv = -(F / mdot) ln(1 - mdot t / m0); a
    # 10 N, Isp 350 s station-keeping thruster (their table 3.13) on 1000 kg for 600 s.
    craft = Spacecraft(
        max_accel=0.01, layout="ball", isp=350.0, mass=1000.0, propellant=500.0, drive="thrust"
    )
    act = craft.apply([1.0, 0.0, 0.0], 600.0)
    flow = 10.0 / (350.0 * G0)
    assert act.dv_l2 == pytest.approx(
        -(10.0 / flow) * math.log1p(-flow * 600.0 / 1000.0), rel=1e-12
    )
    assert act.propellant_kg == pytest.approx(flow * 600.0, rel=1e-12)


def test_constant_thrust_acceleration_rises_as_propellant_burns():
    craft = Spacecraft(max_accel=0.02, layout="ball", mass=500.0, propellant=100.0, drive="thrust")
    accels = []
    for _ in range(5):
        act = craft.apply([1.0, 0.0, 0.0], 600.0)
        accels.append(act.dv_l2 / 600.0)
    assert np.all(np.diff(accels) > 0.0)
    mean_mass = 500.0 - act.propellant_kg * 4.5
    assert accels[-1] == pytest.approx(10.0 / mean_mass, rel=1e-4)


def test_constant_thrust_stops_at_the_dry_mass():
    craft = Spacecraft(max_accel=0.02, layout="ball", mass=500.0, propellant=0.01, drive="thrust")
    act = craft.apply([1.0, 0.0, 0.0], 10.0)
    burn = 0.01 / (10.0 / (220.0 * G0))
    assert sum(d for d, _ in act.segments[:-1]) == pytest.approx(burn, rel=1e-12)
    assert np.array_equal(act.segments[-1][1], np.zeros(3)) and act.empty
    assert act.dv_l2 == pytest.approx(220.0 * G0 * math.log(500.0 / 499.99), rel=1e-9)
    assert craft.mass == 499.99


def rising_thrust(t, y, n, thrust, flow):
    """CW state and mass under constant thrust (N) and mass flow (kg/s)."""
    return np.r_[cw_rhs(t, y[:6], n, thrust / y[6]), -flow]


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_constant_thrust_segments_follow_the_exact_rising_acceleration(layout):
    # Reference: DOP853 on the CW equations with a = F / m(t) and the mass equation.
    n = mean_motion(EARTH.radius + 500e3)
    craft = Spacecraft(max_accel=0.02, layout=layout, mass=500.0, propellant=400.0, drive="thrust")
    model = CWDynamics(n)
    state = np.array([100.0, -300.0, 20.0, 0.1, 0.0, -0.05])
    model.reset(state)
    exact = np.r_[state, 500.0]
    for u in np.random.default_rng(0).uniform(-1.0, 1.0, (int(2 * np.pi / n // 10.0), 3)):
        for duration, accel in craft.apply(u, 10.0).segments:
            model.advance(accel, duration)
        thrust = craft.limit(u, 10.0)[0] * craft.max_thrust
        size = np.sum(np.abs(thrust)) if layout == "box" else np.linalg.norm(thrust)
        exact = solve_ivp(
            rising_thrust,
            (0.0, 10.0),
            exact,
            method="DOP853",
            args=(n, thrust, size / (220.0 * G0)),
            rtol=1e-13,
            atol=1e-12,
        ).y[:, -1]
    final, _ = model.advance(np.zeros(3), 0.0)
    # One mean-acceleration segment per step would be off by 1e-3 m here.
    assert np.max(np.abs(final[:3] - exact[:3])) < 1e-7
    assert np.max(np.abs(final[3:] - exact[3:6])) < 1e-10
    assert craft.mass == pytest.approx(exact[6], rel=1e-12)


def test_constant_thrust_needs_a_dry_mass():
    with pytest.raises(ValueError, match="constant thrust"):
        Spacecraft(drive="thrust")
    with pytest.raises(ValueError, match="drive"):
        Spacecraft(drive="pulsed")


def delivered(craft, command, count):
    """Delivered commands (fractions) for `count` burns of `command`, ball layout, no delay."""
    return np.array(
        [craft.apply(command, 10.0).segments[0][1] / craft.max_accel for _ in range(count)]
    )


def test_execution_error_has_the_specified_mean_and_covariance():
    # Each check fails a correct model with probability 1e-6; the seed is fixed.
    z, count = 4.89, 20000
    error = ExecutionError(
        scale_bias=0.05, scale_sigma=0.02, noise_sigma=0.01, misalignment=(0.0, 0.0, 0.03)
    )
    craft = Spacecraft(layout="ball", execution=error)
    craft.reset(np.random.default_rng(0))
    command = np.array([0.6, -0.3, 0.2])
    out = delivered(craft, command, count)
    rotated = Rotation.from_rotvec([0.0, 0.0, 0.03]).apply(command)
    cov = 0.02**2 * np.outer(rotated, rotated) + 0.01**2 * np.eye(3)
    sigma = np.sqrt(np.diag(cov))
    assert np.all(np.abs(out.mean(axis=0) - 1.05 * rotated) < z * sigma / np.sqrt(count))
    ratio = (count - 1) * np.diag(np.cov(out, rowvar=False)) / np.diag(cov)
    assert np.all((ratio > chi2.ppf(5e-7, count - 1)) & (ratio < chi2.ppf(1 - 5e-7, count - 1)))
    corr = np.corrcoef(out, rowvar=False)
    assert np.all(np.abs(corr - cov / np.outer(sigma, sigma)) < z / np.sqrt(count))


def test_one_degree_misalignment_matches_published_cross_axis_values():
    error = ExecutionError(misalignment=(0.0, 0.0, np.radians(1.0)))
    craft = Spacecraft(max_accel=0.02, layout="ball", execution=error)
    accel = craft.apply([1.0, 0.0, 0.0], 10.0).segments[0][1]
    # Fehse, Automated Rendezvous and Docking, p. 97: 1 deg on a 1 m/s burn puts 0.017452 m/s
    # across it. At 0.02 m/s^2 that is 3.5e-4 m/s^2, about what holding 100 m below the target
    # takes (3 n^2 x = 3.7e-4 m/s^2 at 500 km).
    assert accel[1] / np.linalg.norm(accel) == pytest.approx(0.017452, abs=5e-7)
    assert accel[1] == pytest.approx(3.49e-4, abs=5e-7)
    assert np.linalg.norm(accel) == pytest.approx(0.02, rel=1e-15)


def test_execution_error_leaves_coasting_alone_and_stays_hidden():
    error = ExecutionError(scale_bias=0.1, scale_sigma=0.05, noise_sigma=0.02)
    craft = Spacecraft(delay=25.0, execution=error)
    craft.reset(np.random.default_rng(1))
    assert np.array_equal(craft.apply(np.zeros(3), 10.0).segments[0][1], np.zeros(3))
    craft.apply([0.5, 0.0, 0.0], 10.0)
    assert np.array_equal(craft.pending()[-1][1], [0.5, 0.0, 0.0])


def test_propellant_is_spent_on_the_delivered_thrust():
    craft = Spacecraft(layout="ball", execution=ExecutionError(scale_bias=0.1))
    act = craft.apply([0.5, 0.0, 0.0], 10.0)
    assert act.dv_l2 == pytest.approx(1.1 * 0.5 * 0.02 * 10.0, rel=1e-14)
    assert act.propellant_kg == pytest.approx(500.0 * -math.expm1(-act.dv_l2 / (220.0 * G0)))
