import numpy as np
import pytest
from scipy.integrate import quad

from vbar.conventions import inertial_to_hill
from vbar.dynamics.bodies import AU, EARTH, SOLAR_PRESSURE, SUN_RADIUS, Body
from vbar.dynamics.models import TwoBodyDynamics
from vbar.dynamics.perturbations import (
    Drag,
    FlatPlatePressure,
    HarrisPriester,
    Plate,
    SolarPressure,
    conical_shadow,
    cylindrical_shadow,
    geodetic_height,
)

SUN = np.array([1.0, 0.0, 0.0])
ZERO = np.zeros(3)
# The Earth without rotation or oblateness, so the air is at rest and speeds are orbital.
STILL_EARTH = Body("earth", EARTH.mu, EARTH.radius)
V_400 = np.sqrt(EARTH.mu / (EARTH.radius + 400e3))


def constant_density(rho):
    return lambda r: np.full(np.shape(r)[:-1], rho)


def circle(radius, samples):
    u = np.linspace(0.0, 2 * np.pi, samples, endpoint=False)
    return radius * np.stack([np.cos(u), np.sin(u), np.zeros_like(u)], axis=-1)


def boundary(lit, lo, hi):
    """Bisect for the switch of the predicate `lit` between `lo` (False) and `hi` (True)."""
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if lit(mid) else (mid, hi)
    return 0.5 * (lo + hi)


@pytest.mark.parametrize("beta_deg, fraction", [(0.0, 0.378), (30.0, 0.358), (60.0, 0.231)])
def test_cylindrical_eclipse_fraction_at_500_km(beta_deg, fraction):
    # Share of a circular 500 km orbit inside the shadow cylinder, from its geometry
    # (Montenbruck and Gill, Satellite Orbits, sec. 3.4.1).
    beta = np.radians(beta_deg)
    sun = [np.cos(beta), 0.0, np.sin(beta)]
    shadow = 1.0 - cylindrical_shadow(circle(EARTH.radius + 500e3, 100_000), sun).mean()
    assert shadow == pytest.approx(fraction, abs=5e-4)


def test_conical_shadow_cones_have_published_half_angles_and_vertex():
    # Montenbruck and Gill sec. 3.4.1: umbra half-cone 0.264 deg, penumbra 0.269 deg, umbra
    # vertex 1.384e6 km behind the Earth.
    def umbra_edge(d):
        return boundary(lambda y: conical_shadow([-d, y, 0.0], SUN) > 0.0, 0.0, 2 * EARTH.radius)

    def penumbra_edge(d):
        return boundary(lambda y: conical_shadow([-d, y, 0.0], SUN) == 1.0, 0.0, 2 * EARTH.radius)

    d1, d2 = 1e8, 5e8
    umbra = np.degrees(np.arctan((umbra_edge(d1) - umbra_edge(d2)) / (d2 - d1)))
    penumbra = np.degrees(np.arctan((penumbra_edge(d2) - penumbra_edge(d1)) / (d2 - d1)))
    assert umbra == pytest.approx(0.264, abs=5e-4)
    assert penumbra == pytest.approx(0.269, abs=5e-4)
    vertex = boundary(lambda d: conical_shadow([-d, 0.0, 0.0], SUN) > 0.0, 1e9, 2e9)
    assert vertex == pytest.approx(1.384e9, abs=5e5)


def test_conical_shadow_equals_numerically_integrated_disk_overlap():
    r_orbit = EARTH.radius + 500e3
    # A point on the orbit half way through the penumbra.
    u = boundary(
        lambda u: conical_shadow(r_orbit * np.array([-np.cos(u), np.sin(u), 0.0]), SUN) > 0.5,
        0.0,
        np.pi / 2,
    )
    r = r_orbit * np.array([-np.cos(u), np.sin(u), 0.0])
    to_sun = AU * SUN - r
    a = np.arcsin(SUN_RADIUS / np.linalg.norm(to_sun))
    b = np.arcsin(EARTH.radius / np.linalg.norm(r))
    c = np.arccos(-r @ to_sun / (np.linalg.norm(r) * np.linalg.norm(to_sun)))

    def chord_overlap(x):
        half_sun = np.sqrt(max(a * a - x * x, 0.0))
        half_body = np.sqrt(max(b * b - (x - c) ** 2, 0.0))
        return 2 * min(half_sun, half_body) if half_body > 0.0 else 0.0

    overlap, _ = quad(chord_overlap, -a, a, points=[c - b], epsabs=1e-16, epsrel=1e-12, limit=200)
    assert conical_shadow(r, SUN) == pytest.approx(1 - overlap / (np.pi * a * a), abs=1e-9)


def test_conical_shadow_is_total_inside_and_annular_beyond_the_umbra():
    assert conical_shadow([-EARTH.radius - 500e3, 0.0, 0.0], SUN) == 0.0
    assert conical_shadow([EARTH.radius + 500e3, 0.0, 0.0], SUN) == 1.0
    beyond = conical_shadow([-3e9, 0.0, 0.0], SUN)
    b = np.arcsin(EARTH.radius / 3e9)
    a = np.arcsin(SUN_RADIUS / (AU + 3e9))
    assert beyond == pytest.approx(1 - (b / a) ** 2, rel=1e-12)


@pytest.mark.parametrize("beta_deg", [0.0, 45.0])
def test_conical_and_cylindrical_shadow_agree_on_average_over_a_leo_orbit(beta_deg):
    # The penumbra is about 8 s wide at 500 km; it straddles the cylinder's edge, so the mean
    # occulted fraction matches the cylinder to a few parts in 1e5.
    beta = np.radians(beta_deg)
    sun = [np.cos(beta), 0.0, np.sin(beta)]
    points = circle(EARTH.radius + 500e3, 200_000)
    conical = conical_shadow(points, sun).mean()
    assert conical == pytest.approx(cylindrical_shadow(points, sun).mean(), abs=5e-5)


def test_geodetic_height_reproduces_published_example():
    # Montenbruck and Gill, Satellite Orbits, exercise 5.3 (WGS 84): h = -63.667 m.
    h = geodetic_height([1917032.190, 6029782.349, -801376.113])
    assert h == pytest.approx(-63.667, abs=5e-4)


def test_geodetic_height_at_the_equator_and_poles():
    polar_radius = EARTH.radius * (1 - EARTH.flattening)
    points = [[EARTH.radius + 4e5, 0.0, 0.0], [0.0, 0.0, polar_radius + 4e5], [0, 0, -polar_radius]]
    assert geodetic_height(points) == pytest.approx([4e5, 4e5, 0.0], abs=1e-8)


def on_equator(height, angle):
    r = EARTH.radius + height
    return [r * np.cos(angle), r * np.sin(angle), 0.0]


@pytest.mark.parametrize(
    "height_km, rho_min, rho_max",
    # Montenbruck and Gill, table 3.8, in g/km^3.
    [(400, 2.249, 7.492), (500, 0.3916, 2.042), (600, 0.08070, 0.6390)],
)
def test_harris_priester_density_at_bulge_apex_and_antipode(height_km, rho_min, rho_max):
    density = HarrisPriester(SUN, exponent=2)
    # With the Sun on the equator the apex is on the equator, 30 deg east of the Sun.
    apex = np.radians(30.0)
    assert density(on_equator(1e3 * height_km, apex)) == pytest.approx(1e-12 * rho_max, rel=1e-12)
    assert density(on_equator(1e3 * height_km, apex + np.pi)) == pytest.approx(
        1e-12 * rho_min, rel=1e-12
    )


def test_harris_priester_density_is_exponential_between_table_heights():
    density = HarrisPriester(SUN, exponent=2)
    apex = np.radians(30.0)
    middle = density(on_equator(410e3, apex))
    assert middle == pytest.approx(np.sqrt(7.492 * 5.684) * 1e-12, rel=1e-12)


@pytest.mark.parametrize("exponent", [2, 6])
def test_harris_priester_bulge_narrows_with_the_exponent(exponent):
    density = HarrisPriester(SUN, exponent=exponent)
    side = density(on_equator(500e3, np.radians(30.0 + 90.0)))
    blend = 0.5 ** (exponent / 2)
    assert side == pytest.approx(1e-12 * (0.3916 + (2.042 - 0.3916) * blend), rel=1e-12)


def test_harris_priester_density_ends_at_1000_km_and_refuses_below_100_km():
    density = HarrisPriester(SUN, exponent=2)
    assert density(on_equator(1001e3, 0.0)) == 0.0
    with pytest.raises(ValueError, match="100 km"):
        density(on_equator(99e3, 0.0))


def test_drag_acceleration_matches_published_value():
    # Fehse, Automated Rendezvous and Docking of Spacecraft, eq 4.3: 1e-11 kg/m^3, circular
    # speed at 400 km, ballistic coefficient 215 kg/m^2; the book rounds to 1.38e-6 m/s^2.
    drag = Drag(constant_density(1e-11), 215.0, 215.0, STILL_EARTH)
    v = np.array([0.0, V_400, 0.0])
    accel = drag.acceleration([EARTH.radius + 400e3, 0.0, 0.0], v, 215.0)
    assert np.linalg.norm(accel) == pytest.approx(1.3676e-6, rel=5e-5)
    assert np.allclose(np.cross(accel, v), 0.0, atol=1e-20) and accel @ v < 0


def test_drag_acts_against_the_velocity_relative_to_the_rotating_air():
    # Prograde on the equator the air moves with the spacecraft at w r.
    r = EARTH.radius + 400e3
    drag = Drag(constant_density(1e-11), 100.0, 100.0)
    still = Drag(constant_density(1e-11), 100.0, 100.0, STILL_EARTH)
    v = [0.0, V_400, 0.0]
    ratio = drag.acceleration([r, 0, 0], v, 100.0)[1] / still.acceleration([r, 0, 0], v, 100.0)[1]
    assert ratio == pytest.approx((1 - EARTH.spin_rate * r / V_400) ** 2, rel=1e-14)


def test_differential_drag_drift_matches_published_values():
    # Fehse eq 4.3 and fig 4.3: a chaser with ballistic coefficient 470 kg/m^2 and 1.5 times
    # the target's, from the target's position, at 1e-11 kg/m^3 for three orbits at 400 km.
    # The values solve CW with that constant differential acceleration (the book's text, about
    # 70 m behind and 5 m above, does not follow from its numbers); flown here through the
    # nonlinear model, where the target also decays 9 m per orbit.
    drag = Drag(constant_density(1e-11), 470.0 / 1.5, 470.0, STILL_EARTH)
    m = TwoBodyDynamics(STILL_EARTH, 400e3, perturbations=[drag])
    m.reset(np.zeros(6))
    final, _ = m.advance(ZERO, 3 * 2 * np.pi / m.n)
    assert final[0] == pytest.approx(9.213, rel=1e-3)
    assert final[1] == pytest.approx(-130.24, rel=1e-3)


def test_differential_drag_brings_a_retrograde_start_back_after_9_08_orbits():
    # Fehse fig 4.15: -0.06 m/s along-track, 9.4e-12 kg/m^3, chaser ballistic coefficient
    # 470 kg/m^2 and 5 times the target's; back after "about ten" orbits, 9.08 in CW with the
    # constant differential acceleration.
    drag = Drag(constant_density(9.4e-12), 470.0 / 5, 470.0, STILL_EARTH)
    m = TwoBodyDynamics(STILL_EARTH, 400e3, perturbations=[drag])
    m.reset([0.0, 0.0, 0.0, 0.0, -0.06, 0.0])
    period = 2 * np.pi / m.n
    _, samples = m.advance(ZERO, 10 * period, n_samples=1001)
    y = samples[:, 1]
    k = np.flatnonzero((y[:-1] > 0) & (y[1:] <= 0))
    assert k.size == 1
    assert 10 * k[0] / 1000 == pytest.approx(9.08, abs=0.01)


def test_solar_pressure_pushes_away_from_the_sun_at_the_published_pressure():
    # Montenbruck and Gill eq 3.69: 4.56e-6 N/m^2 at 1 AU.
    srp = SolarPressure(SUN, 0.013, 0.013)
    r = np.array([0.0, EARTH.radius + 500e3, 0.0])
    lit = srp.acceleration(r, 0.013)
    to_sun = AU * SUN - r
    expected = -SOLAR_PRESSURE * 0.013 * (AU / np.linalg.norm(to_sun)) ** 2
    assert np.allclose(lit, expected * to_sun / np.linalg.norm(to_sun), rtol=1e-14, atol=0.0)
    assert np.linalg.norm(lit) == pytest.approx(SOLAR_PRESSURE * 0.013, rel=1e-8)
    assert np.all(srp.acceleration([-EARTH.radius - 500e3, 0.0, 0.0], 0.013) == 0.0)


def test_equal_spacecraft_feel_no_differential_force():
    hp = HarrisPriester(SUN, exponent=2)
    forces = [Drag(hp, 50.0, 50.0), SolarPressure(SUN, 0.02, 0.02)]
    r = np.array([EARTH.radius + 500e3, 1e5, 2e5])
    v = np.array([-100.0, 7.6e3, 50.0])
    for force in forces:
        target, relative = force.accelerations(r, v, ZERO, ZERO)
        assert np.linalg.norm(target) > 1e-8
        assert np.all(relative == 0.0)
    # Identical spacecraft starting together stay together.
    m = TwoBodyDynamics(inclination=0.9, j2=True, perturbations=forces)
    m.reset(np.zeros(6))
    assert np.all(m.advance(ZERO, 3000.0)[0] == 0.0)


def test_perturbations_of_zero_strength_change_nothing():
    plain = TwoBodyDynamics(inclination=0.9, j2=True)
    forces = [Drag(HarrisPriester(SUN, 2), np.inf, np.inf), SolarPressure(SUN, 0.0, 0.0)]
    perturbed = TwoBodyDynamics(inclination=0.9, j2=True, perturbations=forces)
    assert perturbed.name == "twobody_j2_drag_srp"
    s0 = [150.0, -400.0, 30.0, 0.05, 0.1, -0.02]
    plain.reset(s0)
    perturbed.reset(s0)
    accel = [1e-3, 0.0, -1e-3]
    assert np.array_equal(plain.advance(accel, 600.0)[0], perturbed.advance(accel, 600.0)[0])


def test_hill_velocity_with_solar_pressure_needs_the_full_frame_rate():
    # A large out-of-plane solar force on the target turns its orbit plane; the Hill velocity
    # must follow the frame's exact rate, as under J2.
    sun = [0.3, -0.4, 0.866]
    m = TwoBodyDynamics(inclination=0.9, perturbations=[SolarPressure(sun, 2.0, 0.5)])
    m.reset([200.0, -1000.0, 300.0, 0.05, 0.1, -0.2])
    h = 1.0
    m.advance(ZERO, 500.0 - 2 * h)
    positions = [m.state_in_last_segment(500.0 - 2 * h)[:3]]
    for k in range(4):
        state, _ = m.advance(ZERO, h)
        positions.append(state[:3])
        if k == 1:
            rel, target, chaser = state, m.target_state(), m.chaser_state()
    rate = (positions[0] - 8 * positions[1] + 8 * positions[3] - positions[4]) / (12 * h)
    assert np.max(np.abs(rel[3:] - rate)) < 1e-9
    keplerian = inertial_to_hill(
        target[:3], target[3:], chaser[:3] - target[:3], chaser[3:] - target[3:]
    )
    assert np.max(np.abs(keplerian[3:] - rate)) > 1e-8


def sunward_plate_acceleration(plate, sun=SUN):
    """Acceleration of `plate` on a spacecraft at noon, where Hill x points at the Sun."""
    r = np.array([EARTH.radius + 500e3, 0.0, 0.0])
    v = np.array([0.0, 7.6e3, 0.0])
    model = FlatPlatePressure(sun, [plate], [plate])
    return model.accelerations(r, v, ZERO, ZERO)[0]


@pytest.mark.parametrize(
    "specular, diffuse, c_r", [(0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (0.0, 1.0, 5.0 / 3.0)]
)
def test_sun_facing_plate_equals_a_cannonball(specular, diffuse, c_r):
    plate = Plate(0.01, (1.0, 0.0, 0.0), specular, diffuse)
    accel = sunward_plate_acceleration(plate)
    cannonball = SolarPressure(SUN, c_r * 0.01, 0.0).acceleration(
        [EARTH.radius + 500e3, 0, 0], c_r * 0.01
    )
    assert np.allclose(accel, cannonball, rtol=1e-14, atol=1e-24)
    # A plate is lit on both sides.
    flipped = sunward_plate_acceleration(Plate(0.01, (-1.0, 0.0, 0.0), specular, diffuse))
    assert np.array_equal(accel, flipped)


def test_oblique_mirror_is_pushed_along_its_normal_by_cos_squared():
    # Montenbruck and Gill eq 3.72: a specular plate at incidence t feels 2 P (A/m) cos^2 t
    # along its normal, away from the Sun.
    t = np.radians(35.0)
    plate = Plate(0.02, (np.cos(t), np.sin(t), 0.0), specular=1.0)
    accel = sunward_plate_acceleration(plate)
    flux = (AU / (AU - EARTH.radius - 500e3)) ** 2
    normal = np.array([np.cos(t), np.sin(t), 0.0])
    expected = -2 * SOLAR_PRESSURE * flux * 0.02 * np.cos(t) ** 2 * normal
    assert np.allclose(accel, expected, rtol=1e-12, atol=1e-22)


def test_flat_plates_act_through_the_two_body_model():
    plates = FlatPlatePressure(
        [0.3, -0.4, 0.866],
        [Plate(0.5, (1.0, 0.0, 0.0), 0.3, 0.2)],
        [Plate(0.02, (0.0, 0.0, 1.0), 0.1, 0.1), Plate(0.01, (0.0, 1.0, 0.0))],
    )
    m = TwoBodyDynamics(inclination=0.9, perturbations=[plates])
    plain = TwoBodyDynamics(inclination=0.9)
    assert m.name == "twobody_srp_plate"
    for model in (m, plain):
        model.reset([100.0, -300.0, 20.0, 0.0, 0.0, 0.0])
    gap = m.advance(ZERO, 2000.0)[0] - plain.advance(ZERO, 2000.0)[0]
    # The target is pushed harder than the chaser: about 1e-6 m/s^2 of differential force
    # displaces the chaser by about 0.5 a t^2 = 2 m in 2000 s.
    assert 0.5 < np.linalg.norm(gap[:3]) < 5.0
