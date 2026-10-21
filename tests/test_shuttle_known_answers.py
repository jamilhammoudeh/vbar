"""Known answers from Goodman, History of Space Shuttle Rendezvous (JSC-63400 Rev 3, 2011),
and Fehse, Automated Rendezvous and Docking of Spacecraft (2003), behind the scenario catalogue.

Shuttle tables are in the LVLH frame of Goodman's appendix G (+X along the V-bar, +Z toward
Earth) in ft and ft/s; the checks run in Hill axes and SI. n is computed from the WGS 84 radius
and the stated altitude (185 n.mi. for the ISS flights).
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from vbar.constraints import ClosingSpeedLimit, GateTable, PassiveSpeed
from vbar.dynamics import EARTH, cw_rhs, cw_stm, mean_motion
from vbar.scenarios import PRESETS
from vbar.scenarios.catalog import FT, GATES, N_ISS, hold_oracle
from vbar.scenarios.inspection import fibonacci_sphere

T_ISS = 2.0 * math.pi / N_ISS


def circle_thrust(radius, theta, omega, n):
    """Thrust that holds a circle of `radius` at angle `theta` and rate `omega` in Hill axes."""
    k = omega**2 + 2.0 * n * omega
    return -radius * np.array([math.cos(theta) * (k + 3.0 * n**2), math.sin(theta) * k, 0.0])


def test_the_twice_orbital_rate_circle_needs_only_the_rbar_hover_force():
    # Goodman p. 137 (derived): at omega = -2n the along-track thrust vanishes and the radial
    # thrust is -3 n^2 x; propagating the closed loop from +R-bar keeps the range to 1e-9 m and
    # reaches +V-bar after T/8.
    n, radius = N_ISS, 600.0 * FT
    for theta in (math.pi, 0.75 * math.pi, 0.5 * math.pi):
        a = circle_thrust(radius, theta, -2.0 * n, n)
        assert a[1] == pytest.approx(0.0, abs=1e-15)
        assert a[0] == pytest.approx(-3.0 * n**2 * radius * math.cos(theta))
    start = np.array([-radius, 0.0, 0.0, 0.0, 2.0 * n * radius, 0.0])
    sol = solve_ivp(
        lambda t, s: cw_rhs(t, s, n, np.array([-3.0 * n**2 * s[0], 0.0, 0.0])),
        (0.0, T_ISS / 8.0),
        start,
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        dense_output=True,
    )
    path = sol.sol(np.linspace(0.0, T_ISS / 8.0, 200))
    assert np.max(np.abs(np.linalg.norm(path[:3], axis=0) - radius)) < 1e-9
    end = sol.y[:, -1]
    assert math.degrees(math.atan2(end[1], end[0])) == pytest.approx(90.0, abs=1e-6)


def test_the_transition_and_pitch_manoeuvre_durations():
    # Table 25.8 p. 233: 575 ft to the +V-bar at 310 ft took 690 s, T/8 at 185 n.mi. is 685.4 s;
    # the pitch manoeuvre of 360 deg at 0.75 deg/s lasted 480 s (p. 240).
    assert T_ISS / 8.0 == pytest.approx(685.4, abs=0.1)
    assert abs(T_ISS / 8.0 - 690.0) / 690.0 < 0.02
    assert 360.0 / 0.75 == 480.0


def test_thrust_loss_on_the_twice_rate_arc_never_comes_closer_than_the_circle():
    # Derived (supports p. 137): on the clockwise twice-orbital-rate circle vy + 2 n x = 0, so the
    # free ellipse after thrust loss has its least range at the current radius and its largest at
    # 2R, 4.15R, 6.20R and 7.00R from 180, 150, 120 and 90 deg.
    n, radius = N_ISS, 600.0 * FT
    for degrees, far in ((180, 2.0), (150, 4.15), (120, 6.20), (90, 7.00)):
        theta = math.radians(degrees)
        r = radius * np.array([math.cos(theta), math.sin(theta), 0.0])
        v = -2.0 * n * radius * np.array([-math.sin(theta), math.cos(theta), 0.0])
        state = np.concatenate([r, v])
        assert state[4] + 2.0 * n * state[0] == pytest.approx(0.0, abs=1e-12)
        times = np.linspace(0.0, T_ISS, 4001)
        ranges = [np.linalg.norm((cw_stm(n, t) @ state)[:3]) for t in times]
        assert min(ranges) == pytest.approx(radius, rel=1e-3)
        assert max(ranges) == pytest.approx(far * radius, rel=2e-3)


def test_fly_around_delta_v_at_twice_and_once_orbital_rate():
    # Derived from the p. 137 propellant claim: with impulsive start and stop from rest at a
    # constant radius, the 2n arc costs 5.500, 7.000, 8.500 and 10.000 nR for 90 to 360 deg
    # against 4.422, 6.844, 9.266 and 11.688 nR at the inertial rate n; break-even near 200 deg.
    n, radius = N_ISS, 600.0 * FT
    expected = {
        90: (5.500, 4.422),
        180: (7.000, 6.844),
        270: (8.500, 9.266),
        360: (10.000, 11.688),
    }
    for degrees, (fast, slow) in expected.items():
        sweep = math.radians(degrees)
        for omega, value in ((-2.0 * n, fast), (-n, slow)):

            def magnitude(theta, omega=omega):
                return float(np.linalg.norm(circle_thrust(radius, theta, omega, n)))

            continuous = quad(magnitude, math.pi - sweep, math.pi, limit=200)[0] / abs(omega)
            total = (continuous + 2.0 * abs(omega) * radius) / (n * radius)
            assert total == pytest.approx(value, abs=1e-3)


def test_the_sts130_final_approach_rates():
    # Table 25.8 p. 233: 300 to 100 ft in 1000 s is 0.200 ft/s, 10 ft to contact in 100 s is
    # 0.10 ft/s; the gate table of fig. 25.22 holds those rates at 320 and 30 ft.
    assert (300.0 - 100.0) / 1000.0 == pytest.approx(0.200)
    assert 10.0 / 100.0 == pytest.approx(0.10)
    gates = dict(GATES["shuttle-vbar-iss"])
    assert gates[320.0 * FT] == pytest.approx(0.20 * FT)
    assert gates[30.0 * FT] == pytest.approx(0.07 * FT)


def test_gate_time_constants_are_1000_s_inside_500_ft_on_the_hst_profiles():
    # Table 24.3 p. 209: R / Rdot is 1000 s at the 500, 400, 300, 200 and 100 ft gates (120 ft
    # entries 600 s); the dual-coelliptic gates of fig. 9.3 sit at 200 s out to 1000 ft and 100 s
    # at 400 to 200 ft.
    for r, v in GATES["shuttle-rbar-orbt"]:
        if r <= 500.0 * FT and abs(r - 120.0 * FT) > 1e-9:
            assert r / v == pytest.approx(1000.0)
        elif abs(r - 120.0 * FT) < 1e-9:
            assert r / v == pytest.approx(600.0)
    for r, v in GATES["dual-coelliptic"]:
        assert r / v == pytest.approx(200.0 if r >= 1000.0 * FT else 100.0)


def test_the_linear_closing_limit_is_three_to_eleven_times_the_shuttle_gates():
    # Table 24.3 ORBT column against 0.2 + 4 n rho at 500 km: ratios 3.2 (2000 ft), 3.4 (1000),
    # 4.1 (600), 5.7 (500), 7.7 (200), 11.0 (100 ft); at 35 ft the gate is 0 and the law allows
    # 0.247 m/s.
    n = mean_motion(EARTH.radius + 500e3)
    law = ClosingSpeedLimit(0.2, 4.0 * n)
    gates = dict(GATES["shuttle-rbar-orbt"])
    for feet, ratio in ((2000, 3.2), (1000, 3.4), (600, 4.1), (500, 5.7), (200, 7.7), (100, 11.0)):
        rho = feet * FT
        assert float(law.allowed_speed(rho)) / gates[rho] == pytest.approx(ratio, abs=0.1)
    assert float(law.allowed_speed(35.0 * FT)) == pytest.approx(0.247, abs=1e-3)


def test_the_measured_shuttle_iss_contact_speed_only_fits_the_apds_preset():
    # p. 244: 0.113 ft/s = 0.0344 m/s, outside the ATV, Soyuz and Hermes windows and inside the
    # Shuttle APDS window of 0.025 to 0.05 m/s (Fehse appendix C, lower end ours).
    contact = 0.113 * FT
    assert contact == pytest.approx(0.0344, abs=1e-4)
    inside = {name: lo <= contact <= hi for name, (lo, hi) in PRESETS.items()}
    assert inside == {"probe-drogue-atv": False, "impact": False, "low-impact": False, "apds": True}


def test_rbar_free_drift_through_the_pitch_manoeuvre():
    # Derived (supports pp. 140 to 141 and 241): 600 ft below on the +R-bar closing at 0.3 ft/s,
    # unforced for the 267 s of the pitch between 80 and 280 deg, the range stays within 581 to
    # 605 ft and the closing speed that returns to the start radius is 0.318 ft/s.
    n = N_ISS
    t_free = (360.0 - 160.0) / 0.75
    assert t_free == pytest.approx(266.67, abs=0.01)
    start = np.array([-600.0 * FT, 0.0, 0.0, 0.3 * FT, 0.0, 0.0])
    times = np.linspace(0.0, t_free, 400)
    ranges = np.array([np.linalg.norm((cw_stm(n, t) @ start)[:3]) for t in times]) / FT
    assert 581.0 <= ranges.min() and ranges.max() <= 605.0 + 1.0
    end = cw_stm(n, t_free) @ start
    assert -end[3] / FT == pytest.approx(0.33, abs=0.01)
    c, s = math.cos(n * t_free), math.sin(n * t_free)
    returning = 3.0 * n * start[0] * (c - 1.0) / s
    assert returning / FT == pytest.approx(0.318, abs=0.01)


def test_flown_vbar_rates_sit_inside_the_passive_speed_envelope():
    # Fig. 25.22 p. 242 and Table 25.8: the envelope 2.094 n d over the flown rate is 3.84 at
    # 320 ft, 1.76 at 110 ft, 1.80 at 75 ft and 1.03 at 30 ft, so the flown profile is passively
    # safe everywhere.
    envelope = PassiveSpeed("vbar", 0.0, N_ISS)
    table = {320: (0.20, 3.84), 110: (0.15, 1.76), 75: (0.10, 1.80), 30: (0.07, 1.03)}
    for feet, (rate, ratio) in table.items():
        allowed = float(envelope.speed_envelope(feet * FT))
        assert allowed / (rate * FT) == pytest.approx(ratio, abs=0.01)
    gates = GateTable(GATES["shuttle-vbar-iss"])
    for r, v in GATES["shuttle-vbar-iss"]:
        assert float(gates.allowed_speed(r)) == pytest.approx(v)


def test_fehse_two_impulse_and_forced_transfers_at_400_km():
    # Fehse eqs 3.29 to 3.32 and 3.44 to 3.48 for a 2700 m V-bar transfer at 400 km: tangential
    # two-impulse 0.3241 m/s over one orbit, radial 1.5273 m/s over half an orbit, forced straight
    # line 7.082 m/s over one orbit (the book's "44x" is a misprint; the ratio is 21.85).
    n = mean_motion(EARTH.radius + 400e3)
    oracle = hold_oracle(n, 2700.0)
    assert oracle["tangential_two_impulse_mps"] == pytest.approx(0.3241, abs=5e-5)
    assert oracle["radial_two_impulse_mps"] == pytest.approx(1.5273, abs=5e-5)
    assert oracle["forced_straight_line_one_orbit_mps"] == pytest.approx(7.082, abs=5e-4)
    ratio = oracle["forced_straight_line_one_orbit_mps"] / oracle["tangential_two_impulse_mps"]
    assert ratio == pytest.approx(21.85, abs=0.01)


def test_fehse_fly_around_costs_to_a_point_500_m_below():
    # Fehse eqs 3.33 and 3.38 to 3.39 at 400 km: the tangential fly-around to R-bar costs
    # 2 n dz = 1.1314 m/s, the radial one 3 n dz = 1.697 m/s, from 1178.1 m and 1000 m behind.
    n = mean_motion(EARTH.radius + 400e3)
    dz = 500.0
    assert 2.0 * n * dz == pytest.approx(1.1314, abs=5e-4)
    assert 3.0 * n * dz == pytest.approx(1.697, abs=5e-4)
    assert 3.0 * math.pi * dz / 4.0 == pytest.approx(1178.1, abs=0.1)
    # The tangential transfer: a retrograde burn of n dz / 4 from the V-bar 1178.1 m behind
    # reaches (-dz, 0) after half an orbit with along-track speed 7 n dz / 4.
    start = np.array([0.0, -3.0 * math.pi * dz / 4.0, 0.0, 0.0, -n * dz / 4.0, 0.0])
    end = cw_stm(n, math.pi / n) @ start
    assert end[:2] == pytest.approx([-dz, 0.0], abs=1e-6)
    assert end[4] == pytest.approx(7.0 * n * dz / 4.0, rel=1e-9)


def test_fehse_berthing_box_drift_from_rest():
    # Fehse 5.3.1 p. 119: released at rest 15 m below the V-bar, the chaser drifts 0.414 m
    # radially in 120 s (the book says about 0.4 m); in 60 s it is a quarter of that.
    n = mean_motion(EARTH.radius + 400e3)
    start = np.array([-15.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    after = cw_stm(n, 120.0) @ start
    assert abs(after[0] - start[0]) == pytest.approx(0.414, abs=2e-3)
    assert abs((cw_stm(n, 60.0) @ start)[0] - start[0]) == pytest.approx(0.414 / 4.0, abs=2e-3)


def test_fibonacci_lattice_is_even_and_on_the_unit_sphere():
    points = fibonacci_sphere(100)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0)
    # Nearest-neighbour spacing varies by less than a factor of two over the lattice.
    gaps = []
    for k, p in enumerate(points):
        others = np.delete(points, k, axis=0)
        gaps.append(np.min(np.linalg.norm(others - p, axis=1)))
    assert max(gaps) / min(gaps) < 2.0
