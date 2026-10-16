import numpy as np
import pytest
from scipy.integrate import quad

from vbar.dynamics.bodies import AU, EARTH, SUN_RADIUS
from vbar.dynamics.perturbations import conical_shadow, cylindrical_shadow

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
