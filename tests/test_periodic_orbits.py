"""Reference-orbit generator tests. Values marked "computed" were obtained with Koon, Lo, Marsden
and Ross's methods and constants (Dynamical Systems, the Three-Body Problem and Space Mission
Design) and are reproduced here independently; the others are printed in the book."""

import numpy as np
import pytest

from vbar.dynamics import cr3bp
from vbar.dynamics import periodic_orbits as po
from vbar.dynamics.bodies import EARTH_MOON_KLMR, SUN_EARTH_RICHARDSON

MU = EARTH_MOON_KLMR.mu
NRHO = np.array([0.9873849199042153, 0.0, 0.008369210996156513, 0.0, 1.6744665364075588, 0.0])
NRHO_PERIOD = 1.5088887454509277


@pytest.fixture(scope="module")
def small_l2_halo():
    # Computed: Earth-Moon L2, Richardson m = 1 seed with Az = 8,000 km at 385,000 km.
    seed, _ = po.richardson_halo(MU, 2, 8000 / 385000)
    return seed, po.correct_halo(seed, MU, fixed="z")


def test_richardson_coefficients_reproduce_the_books_sun_earth_l1_values():
    # Book secs. 6.4 and 6.6 with b22 = +3 c3 wp / d1 and the corrected a32; the book's values
    # are reproduced to 4e-7. Sec. 6.4's l1 = -15.9650314 is a misprint of sec. 6.6's.
    h = po.richardson_coefficients(SUN_EARTH_RICHARDSON.mu, 1)
    printed = {
        "s1": -0.8246608317,
        "s2": 0.1210985938,
        "l1": -15.96560314,
        "l2": 1.740900800,
        "delta": 0.29221444425,
    }
    for name, value in printed.items():
        assert getattr(h, name) == pytest.approx(value, rel=1e-6)
    # With the book's printed minus sign on b22, s2 and l2 come out 0.0721 and 1.314.
    assert h.b22 > 0


def test_isee3_halo_amplitudes_and_period():
    # Book sec. 6.6: Az = 110,000 km about Sun-Earth L1 gives Ax about 206,000 km, Ay about
    # 665,000 km, period 177.73 days, and the smallest halo Ax about 200,000 km. Computed:
    # 205,838 km, 664,708 km, 177.72 days (with the 3.156e7 s year), 202,608 km.
    system = SUN_EARTH_RICHARDSON
    h = po.richardson_coefficients(system.mu, 1)
    unit = h.gamma * system.length / 1e3
    az = 110000.0 / unit
    ax = np.sqrt(-(h.l2 * az**2 + h.delta) / h.l1)
    assert ax * unit == pytest.approx(205838, abs=1)
    assert h.k * ax * unit == pytest.approx(664708, abs=1)
    assert np.sqrt(-h.delta / h.l1) * unit == pytest.approx(202608, abs=1)
    _, period = po.richardson_halo(system.mu, 1, 110000e3 / system.length)
    assert period * system.time / 86400 == pytest.approx(177.72, abs=5e-3)
    assert period * system.time / 86400 == pytest.approx(177.73, abs=0.02)


def test_planar_lyapunov_corrector_reproduces_computed_orbit():
    # Computed, Earth-Moon L1: x0 = xL1 - 1e-3, linear seed vy0 = 0.0083722414; converges in 3
    # iterations to vy0 = 0.0084348541, period 2.69179537 (linear 2.69158482); monodromy has
    # lambda1 = 2674 and a real pair with product 1.
    x_l1 = cr3bp.lagrange_points(MU)[0, 0]
    _, wp, _, kappa = cr3bp.libration_frequencies(cr3bp.collinear_coefficient(MU, 1, 2))
    seed = np.array([x_l1 - 1e-3, 0.0, 0.0, 0.0, 1e-3 * wp * kappa, 0.0])
    assert seed[4] == pytest.approx(0.0083722414, abs=1e-10)
    assert 2 * np.pi / wp == pytest.approx(2.69158482, abs=1e-8)
    orbit = po.correct_lyapunov(seed, MU)
    assert orbit.iterations == 3
    assert orbit.state[4] == pytest.approx(0.0084348541, abs=1e-10)
    assert orbit.period == pytest.approx(2.69179537, abs=1e-8)
    real = np.sort(np.abs(np.linalg.eigvals(po.monodromy(orbit.state, orbit.period, MU))))
    assert real[-1] == pytest.approx(2674, rel=1e-3)
    assert real[0] * real[-1] == pytest.approx(1.0, rel=1e-6)


def test_halo_corrector_from_richardson_seed(small_l2_halo):
    # Computed: the seed's z0 = 0.0181144 (+6,974 km at the Moon-side crossing); holding it,
    # the corrector converges in 4 iterations to x0 = 1.1179888291, vy0 = 0.1829734174,
    # period 3.41028833 (14.83 days), lambda1 about 1155.
    seed, orbit = small_l2_halo
    assert seed[2] == pytest.approx(0.0181144, abs=5e-8)
    assert orbit.iterations == 4
    assert orbit.state[2] == seed[2]
    assert orbit.state[[0, 4]] == pytest.approx([1.1179888291, 0.1829734174], abs=1e-9)
    assert orbit.period == pytest.approx(3.41028833, abs=1e-8)
    assert orbit.period * EARTH_MOON_KLMR.time / 86400 == pytest.approx(14.83, abs=5e-3)
    eig = np.abs(np.linalg.eigvals(po.monodromy(orbit.state, orbit.period, MU)))
    assert eig.max() == pytest.approx(1155, rel=2e-3)
    # The m = 1 branch at Earth-Moon L2 reaches farthest below the plane, on the far side.
    _, far, _ = po.half_period_crossing(orbit.state, MU)
    assert far[2] < -abs(orbit.state[2])


def test_nrho_monodromy_and_floquet_modes():
    # Computed: eigenvalues -2.176942, -0.45936, 0.684704 +- 0.728821i and a pair at 1;
    # stability index -1.318, mildly unstable with a sign flip each revolution.
    eig, modes = po.floquet_modes(NRHO, NRHO_PERIOD, MU, [0.0, 0.4, NRHO_PERIOD])
    real = np.sort(eig[np.abs(eig.imag) < 1e-8].real)
    assert real[0] == pytest.approx(-2.176942, abs=1e-6)
    assert real[1] == pytest.approx(-0.45936, abs=1e-5)
    complex_pair = eig[np.abs(eig.imag) > 1e-8]
    assert complex_pair.real == pytest.approx([0.684704] * 2, abs=1e-6)
    assert np.abs(complex_pair.imag) == pytest.approx([0.728821] * 2, abs=1e-6)
    # The pair at 1 is a Jordan block, which rounding splits by about the square root of eps.
    unit = real[2:]
    assert unit.prod() == pytest.approx(1.0, abs=1e-6)
    assert unit == pytest.approx([1.0, 1.0], abs=1e-2)
    index = po.stability_indices(po.monodromy(NRHO, NRHO_PERIOD, MU))
    assert index.real.min() == pytest.approx(-1.318, abs=5e-4)
    # The modes repeat every period; the flow direction is the mode at 1.
    assert np.allclose(modes[-1], modes[0], rtol=0, atol=1e-7 * np.abs(modes).max())
    monodromy = po.monodromy(NRHO, NRHO_PERIOD, MU)
    flow = cr3bp.rhs(0.0, NRHO, MU)
    assert np.max(np.abs(monodromy @ flow - flow)) < 1e-7 * np.max(np.abs(flow))
    assert np.linalg.det(monodromy) == pytest.approx(1.0, abs=1e-8)
