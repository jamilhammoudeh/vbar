import math

import numpy as np
import pytest

from vbar.dynamics import KERBIN
from vbar.games import SCENARIOS, PursuitEvasionEnv
from vbar.games.pursuit import INITIAL_CONDITIONS

MU = KERBIN.mu
COAST = [0.0, 0.0, 0.0, 10.0]

# The ORBIT blocks of KSPDG's pe1 mission saves (`ksp_files/saves/missions/pe1_i*/*.sfs`,
# v0.15.0): SMA (m), ECC, INC, LPE, LAN (deg), MNA (rad), pursuer then evader.
SAVES = {
    "i1": (
        (724999.90000000002, 0.034483000636100769, 0.0001, 180.0, 0.0, 2.7743753789701864),
        (750000.0, 0.0, 0.0001, 180.0, 0.0, 2.7925268031909272),
    ),
    "i2": (
        (750000.0, 0.0, 1.0, 0.0, 0.0, 0.034906585039886591),
        (750000.0, 0.0, 0.0001, 0.0, 0.0, 0.034906585039886591),
    ),
    "i3": (
        (750000.0, 0.0, 0.0, 0.0, 0.0, 5.9306287982767323),
        (750000.0, 0.0, 0.0001, 0.0, 0.0, 5.9341194567807207),
    ),
    "i4": (
        (754500.1875, 0.0072900000959634781, 1.0, 355.0, 350.0, 6.1959188445798699),
        (750000.0, 0.0, 0.0001, 0.0, 0.0, 5.9341194567807207),
    ),
}


def state(sma, ecc, inc, lpe, lan, mna):
    """Position and velocity by the perifocal frame and three rotations, written out here."""
    big_e = mna
    for _ in range(50):
        big_e -= (big_e - ecc * math.sin(big_e) - mna) / (1.0 - ecc * math.cos(big_e))
    nu = 2.0 * math.atan2(
        math.sqrt(1.0 + ecc) * math.sin(big_e / 2), math.sqrt(1.0 - ecc) * math.cos(big_e / 2)
    )
    p = sma * (1.0 - ecc**2)
    radius = p / (1.0 + ecc * math.cos(nu))
    r_pf = radius * np.array([math.cos(nu), math.sin(nu), 0.0])
    v_pf = math.sqrt(MU / p) * np.array([-math.sin(nu), ecc + math.cos(nu), 0.0])

    def rz(a):
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

    def rx(a):
        return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

    rotation = rz(math.radians(lan)) @ rx(math.radians(inc)) @ rz(math.radians(lpe))
    return rotation @ r_pf, rotation @ v_pf


@pytest.mark.parametrize("initial", sorted(SAVES))
def test_each_start_puts_the_vessels_on_the_orbits_of_its_mission_save(initial):
    assert set(INITIAL_CONDITIONS) == set(SAVES)
    observation = np.array(PursuitEvasionEnv(initial=initial).reset()[0])
    for elements, offset in zip(SAVES[initial], (3, 9), strict=True):
        r, v = state(*elements)
        # Positions near 7.5e5 m, velocities near 2.2e3 m/s: agreement to rounding.
        assert np.allclose(observation[offset : offset + 3], r, rtol=0, atol=1e-6)
        assert np.allclose(observation[offset + 3 : offset + 6], v, rtol=0, atol=1e-9)


def coast_until(initial, timeout):
    env = PursuitEvasionEnv(initial=initial, episode_timeout=timeout, capture_dist=1e-3)
    env.reset()
    done = False
    while not done:
        _, _, done, info = env.step(COAST)
    return info


def test_i1_brings_the_vessels_together_where_the_ellipse_touches_the_circle():
    pursuer = SAVES["i1"][0]
    # The pursuer's apoapsis is the evader's radius, to the save's digits.
    assert pursuer[0] * (1.0 + pursuer[1]) == pytest.approx(750000.0, abs=0.1)
    info = coast_until("i1", 130.0)
    # Unpowered, the two pass within 0.2 m of each other at 121.7 s, at the contact point.
    assert info["closest_approach"] < 0.2
    assert info["closest_approach_time"] == pytest.approx(121.7, abs=0.1)


def test_i2_starts_457_m_apart_and_meets_again_at_the_other_node():
    r = 750000.0
    u = 0.034906585039886591
    # Two great circles at relative inclination di, both at argument of latitude u: the angle
    # between the vessels has sin(theta / 2) = sin(u) sin(di / 2), and the range is the chord.
    chord = 2.0 * r * math.sin(u) * math.sin(math.radians(1.0 - 0.0001) / 2.0)
    start = PursuitEvasionEnv(initial="i2").reset()[1]["closest_approach"]
    assert start == pytest.approx(chord, rel=1e-12)
    assert start == pytest.approx(456.79, abs=0.01)
    # Both circles cross at the line of nodes; the vessels reach the far node together after
    # (180 - 2) / 360 of a period.
    period = 2.0 * math.pi * math.sqrt(r**3 / MU)
    info = coast_until("i2", 1100.0)
    assert info["closest_approach"] < 0.1
    assert info["closest_approach_time"] == pytest.approx(178.0 / 360.0 * period, abs=0.05)
    assert 178.0 / 360.0 * period == pytest.approx(1073.75, abs=0.01)


def test_i4_puts_the_pursuer_on_an_inclined_ellipse_of_another_period():
    observation = np.array(PursuitEvasionEnv(initial="i4").reset()[0])
    r, v = observation[3:6], observation[6:9]
    energy = v @ v / 2.0 - MU / np.linalg.norm(r)
    h = np.cross(r, v)
    sma = -MU / (2.0 * energy)
    ecc = math.sqrt(1.0 + 2.0 * energy * (h @ h) / MU**2)
    assert sma == pytest.approx(754500.1875, rel=1e-12)
    assert ecc == pytest.approx(0.0072900000959634781, rel=1e-9)
    assert math.degrees(math.acos(h[2] / np.linalg.norm(h))) == pytest.approx(1.0, rel=1e-9)
    assert (sma / 750000.0) ** 1.5 == pytest.approx(1.009, abs=1e-3)


@pytest.mark.parametrize(("initial", "metres"), [("i1", 4510.0), ("i2", 456.8), ("i4", 2660.0)])
def test_the_start_ranges_quoted_in_the_docs(initial, metres):
    info = PursuitEvasionEnv(initial=initial).reset()[1]
    assert info["closest_approach"] == pytest.approx(metres, abs=1.0)


@pytest.mark.parametrize("key", sorted(SCENARIOS))
def test_every_scenario_class_flies_its_start_and_its_evader(key):
    _, evader, start = key.split("-")
    env = SCENARIOS[key]()
    assert env.initial == start and env.scenario == key
    assert type(env.evader).__name__ == {"e1": "PassiveEvader", "e3": "NormalEscapeEvader"}[evader]
    assert env.reset()[0] == PursuitEvasionEnv(initial=start).reset()[0]


def test_the_e3_classes_take_kspdg_evasion_arguments():
    env = SCENARIOS["pe1-e3-i2"](evade_dist=300.0, evade_throttle=0.5)
    assert env.evader.evade_distance == 300.0 and env.evader.throttle == 0.5
    with pytest.raises(ValueError):
        SCENARIOS["pe1-e3-i2"](evade_throttle=2.0)
