import numpy as np
import pytest
from scipy.integrate import quad

from vbar.dynamics.bodies import AU, EARTH, SUN_RADIUS
from vbar.dynamics.perturbations import (
    HarrisPriester,
    conical_shadow,
    cylindrical_shadow,
    geodetic_height,
)

SUN = np.array([1.0, 0.0, 0.0])


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
