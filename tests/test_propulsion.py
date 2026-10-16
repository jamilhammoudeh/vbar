import math

import numpy as np
import pytest

from vbar.conventions import G0
from vbar.spacecraft import Spacecraft


@pytest.mark.parametrize("layout", ["box", "ball"])
def test_propellant_use_follows_the_rocket_equation(layout):
    rng = np.random.default_rng(0)
    craft = Spacecraft(max_accel=0.02, layout=layout, isp=220.0, mass=500.0)
    dv = propellant = 0.0
    for _ in range(360):
        act = craft.apply(rng.uniform(-1.0, 1.0, 3), 10.0)
        dv += act.dv_l1 if layout == "box" else act.dv_l2
        propellant += act.propellant_kg
    expected = 500.0 * -math.expm1(-dv / (220.0 * G0))
    assert propellant == pytest.approx(expected, rel=1e-9)
    assert craft.mass == pytest.approx(500.0 - expected, rel=1e-12)


def test_box_thrusters_spend_propellant_on_each_axis():
    box = Spacecraft(layout="box").apply([1.0, 1.0, 0.0], 10.0)
    ball = Spacecraft(layout="ball").apply([1.0, 1.0, 0.0], 10.0)
    assert box.dv_l1 == pytest.approx(0.4) and box.dv_l2 == pytest.approx(0.2 * math.sqrt(2))
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
