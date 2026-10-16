"""CR3BP model tests. Values marked "computed" were obtained with Koon, Lo, Marsden and Ross's
methods and constants (Dynamical Systems, the Three-Body Problem and Space Mission Design) and are
reproduced here independently; the others are printed in the book."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from vbar.dynamics import cr3bp
from vbar.dynamics.bodies import (
    EARTH_MOON,
    EARTH_MOON_KLMR,
    SUN_EARTH_RICHARDSON,
    SUN_JUPITER_KLMR,
    Body,
    ThreeBodySystem,
)
from vbar.dynamics.cw import cw_discrete, cw_system
from vbar.dynamics.models import TwoBodyDynamics

MU = EARTH_MOON_KLMR.mu
EPS = np.finfo(float).eps
# Computed: 9:2 synodic-resonant L2 near-rectilinear halo orbit with its apolune below the
# plane, at its perilune-side crossing of the x-z plane, and its period.
NRHO = np.array([0.9873849199042153, 0.0, 0.008369210996156513, 0.0, 1.6744665364075588, 0.0])
NRHO_PERIOD = 1.5088887454509277


def propagate(state, t, mu=MU, rtol=1e-13):
    sol = solve_ivp(cr3bp.rhs, (0.0, t), state, method="DOP853", rtol=rtol, atol=1e-15, args=(mu,))
    return sol.y[:, -1]


def test_book_constant_sets_are_consistent():
    # Table 2.2.1: Earth-Moon velocity unit 1.025 km/s.
    assert EARTH_MOON_KLMR.velocity == pytest.approx(1.025e3, rel=1e-3)
    assert SUN_JUPITER_KLMR.velocity == pytest.approx(13.102e3, rel=1e-3)
    assert SUN_EARTH_RICHARDSON.velocity == pytest.approx(29.784e3, rel=1e-3)
    # The current Earth-Moon set: sidereal month 27.32 days, from Kepler's third law with the
    # 384,400 km mean distance (the Moon's orbit is not circular).
    assert EARTH_MOON.period / 86400 == pytest.approx(27.3, abs=0.1)
    assert EARTH_MOON.acceleration == pytest.approx(
        EARTH_MOON.length * (2 * np.pi / EARTH_MOON.period) ** 2
    )


def test_earth_moon_lagrange_points_and_jacobi_constants():
    # Computed, mu = 0.01215: x of L1, L2, L3 and C (Szebehely) at L1 to L4; the book's
    # convention adds mu (1 - mu) = 0.012002.
    points = cr3bp.lagrange_points(MU)
    assert points[:3, 0] == pytest.approx([0.8369180073, 1.1556799131, -1.0050624018], abs=1e-10)
    states = np.c_[points, np.zeros((5, 3))]
    c = cr3bp.jacobi_constant(states, MU)
    expected = [3.1883357175, 3.1721558389, 3.0121465654, 2.9879976225, 2.9879976225]
    assert c == pytest.approx(expected, abs=1e-10)
    assert c + MU * (1 - MU) == pytest.approx(
        [3.2003380950, 3.1841582164, 3.0241489429, 3, 3], abs=1e-10
    )
    points = cr3bp.lagrange_points(0.012150585)
    assert points[:2, 0] == pytest.approx([0.8369151288, 1.1556821631], abs=1e-10)


def test_sun_jupiter_l1():
    # Book sec. 2.5: Hill radius 6.825e-2, L1 at 0.9324; computed gamma1 = 0.0666765476.
    mu = SUN_JUPITER_KLMR.mu
    x1 = cr3bp.lagrange_points(mu)[0, 0]
    assert (mu / 3) ** (1 / 3) == pytest.approx(6.825e-2, abs=5e-6)
    assert x1 == pytest.approx(0.9323697524, abs=1e-10)
    assert 1 - mu - x1 == pytest.approx(0.0666765476, abs=1e-10)
    # Book sec. 4.4: energies E = -C_book / 2 of L1, L2, L3, computed -1.5198545, -1.5192186,
    # -1.5009532 (printed (E2, E3) about (-1.519, -1.501)).
    points = np.c_[cr3bp.lagrange_points(mu)[:3], np.zeros((3, 3))]
    energy = -(cr3bp.jacobi_constant(points, mu) + mu * (1 - mu)) / 2
    assert energy == pytest.approx([-1.5198545, -1.5192186, -1.5009532], abs=5e-8)


@pytest.mark.parametrize(
    "point, c2, lam, wp, wv, kappa",
    [
        (1, 5.1475733476, 2.9320486823, 2.3343813158, 2.2688264252, 3.5864926414),
        (2, 3.1904366096, 2.1586796525, 1.8626489826, 1.7861793330, 2.9126084820),
    ],
)
def test_earth_moon_collinear_linear_constants(point, c2, lam, wp, wv, kappa):
    # Computed with the book's eqs 2.7.1 and 6.4.1 to 6.4.3, mu = 0.01215.
    assert cr3bp.collinear_coefficient(MU, point, 2) == pytest.approx(c2, abs=1e-10)
    assert cr3bp.libration_frequencies(c2) == pytest.approx((lam, wp, wv, kappa), abs=1e-9)
    if point == 2:
        assert cr3bp.collinear_coefficient(MU, 2, 3) == pytest.approx(-2.6593411074, abs=1e-10)
        assert cr3bp.collinear_coefficient(MU, 2, 4) == pytest.approx(2.5830168833, abs=1e-10)


def test_sun_earth_l1_constants_reproduce_the_books_printed_values():
    # Book secs. 6.3 and 6.4, with Richardson's mass ratio.
    mu = SUN_EARTH_RICHARDSON.mu
    assert 1 - mu - cr3bp.lagrange_points(mu)[0, 0] == pytest.approx(1.001090475e-2, abs=5e-12)
    c = [cr3bp.collinear_coefficient(mu, 1, n) for n in (2, 3, 4)]
    assert c == pytest.approx([4.0610735668, 3.0200105081, 3.0305378797], abs=1e-10)
    _, wp, wv, kappa = cr3bp.libration_frequencies(c[0])
    assert (wp, wv, kappa) == pytest.approx((2.086453455, 2.0152105515, 3.2292680962), abs=1e-9)


def test_l4_loses_linear_stability_at_the_routh_ratio():
    # Book fig. 2.5.1: spectrally stable below mu = 0.038521; exactly (1 - sqrt(23/27)) / 2.
    routh = (1 - math.sqrt(23 / 27)) / 2
    assert routh == pytest.approx(0.038521, abs=5e-7)
    for mu, stable in ((routh * 0.999, True), (routh * 1.001, False)):
        l4 = cr3bp.lagrange_points(mu)[3]
        growth = np.max(np.linalg.eigvals(cr3bp.jacobian(l4, mu)).real)
        assert (growth < 1e-6) == stable


def test_jacobian_at_the_secondary_is_cw_when_its_mass_vanishes():
    # Book eqs 2.6.1 to 2.6.4: at mu = 0 the linearization about (1, 0, 0) is CW with n = 1.
    a, _ = cw_system(1.0)
    assert np.array_equal(cr3bp.jacobian([1.0, 0.0, 0.0], 0.0), a)


def test_nrho_closes_after_one_period():
    # Computed: closes to 1.4e-10 at DOP853 rtol 1e-13; Jacobi constant 3.0466675644;
    # perilune 3,227 km and apolune 71,274 km from the Moon at 385,000 km; apolune state
    # (1.021853, 0, -0.18197747, 0, -0.1028871, 0).
    assert np.max(np.abs(propagate(NRHO, NRHO_PERIOD) - NRHO)) < 3e-10
    assert cr3bp.jacobi_constant(NRHO, MU) == pytest.approx(3.0466675644, abs=1e-10)
    apolune = propagate(NRHO, NRHO_PERIOD / 2)
    assert apolune == pytest.approx([1.021853, 0, -0.18197747, 0, -0.1028871, 0], abs=1e-6)
    sol = solve_ivp(
        cr3bp.rhs,
        (0, NRHO_PERIOD),
        NRHO,
        "DOP853",
        rtol=1e-13,
        atol=1e-15,
        args=(MU,),
        dense_output=True,
    )
    moon = np.array([1 - MU, 0.0, 0.0])
    radius = np.linalg.norm(sol.sol(np.linspace(0, NRHO_PERIOD, 20001))[:3].T - moon, axis=1)
    assert radius.min() * 385000 == pytest.approx(3227, abs=1)
    assert radius.max() * 385000 == pytest.approx(71274, abs=1)


def inertial_propagation(state, t, mu):
    """Propagate in the barycentric inertial frame with the primaries on circles, then rotate
    the result back into the synodic frame at time t (book eqs 2.3.1 to 2.3.5)."""
    r, v = state[:3], state[3:]
    inertial0 = np.r_[r, v + np.cross([0.0, 0.0, 1.0], r)]

    def rhs(tau, s):
        c, s_ = math.cos(tau), math.sin(tau)
        accel = np.zeros(3)
        for mass, p in cr3bp.primaries(mu):
            q = np.array([p[0] * c, p[0] * s_, 0.0])
            d = s[:3] - q
            accel -= mass * d / np.linalg.norm(d) ** 3
        return np.r_[s[3:], accel]

    end = solve_ivp(rhs, (0, t), inertial0, "DOP853", rtol=1e-13, atol=1e-15).y[:, -1]
    c, s_ = math.cos(t), math.sin(t)
    rot = np.array([[c, s_, 0.0], [-s_, c, 0.0], [0.0, 0.0, 1.0]])
    r1 = rot @ end[:3]
    return np.r_[r1, rot @ end[3:] - np.cross([0.0, 0.0, 1.0], r1)]


def test_synodic_equations_match_an_inertial_frame_integration():
    # The only check that sees the Coriolis sign (book sec. 2.3): a model with it flipped
    # conserves the Jacobi constant just as well but is 0.038 off after half a time unit.
    synodic = propagate(NRHO, 0.5)
    assert np.max(np.abs(synodic - inertial_propagation(NRHO, 0.5, MU))) < 1e-11

    def flipped(t, s, mu):
        a = cr3bp.acceleration(s[:3], s[3:], mu)
        a[:2] -= 4 * np.array([s[4], -s[3]])
        return np.r_[s[3:], a]

    wrong = solve_ivp(flipped, (0, 0.5), NRHO, "DOP853", rtol=1e-13, atol=1e-15, args=(MU,))
    assert np.max(np.abs(wrong.y[:, -1] - synodic)) > 1e-2
    jacobi = cr3bp.jacobi_constant(np.stack([NRHO, wrong.y[:, -1]]), MU)
    assert jacobi[1] == pytest.approx(jacobi[0], abs=1e-12)


def test_mirror_image_runs_backward():
    # Book eq 3.3.1 in space: (x, -y, z, -vx, vy, -vz) at -t mirrors the state at t.
    state = np.array([1.05, 0.02, 0.03, 0.01, -0.2, 0.05])
    mirror = np.array([1, -1, 1, -1, 1, -1])
    forward = propagate(state, 0.7)
    backward = propagate(mirror * state, -0.7)
    assert np.allclose(backward, mirror * forward, rtol=0, atol=1e-12)


SYSTEM = EARTH_MOON_KLMR
OFFSET = np.array([800.0, -1500.0, 400.0, 0.05, 0.02, -0.03])


def absolute_difference(target, rel, accel, duration):
    """Chaser minus target, each propagated on its own, in m and m/s."""
    scale = np.repeat([SYSTEM.length, SYSTEM.velocity], 3)
    a = np.asarray(accel) / SYSTEM.acceleration

    def thrusted(t, s):
        return np.r_[s[3:], cr3bp.acceleration(s[:3], s[3:], SYSTEM.mu) + a]

    tau = duration / SYSTEM.time
    chaser = solve_ivp(thrusted, (0, tau), target + rel / scale, "DOP853", rtol=1e-13, atol=1e-16)
    return (chaser.y[:, -1] - propagate(target, tau, SYSTEM.mu)) * scale


@pytest.mark.parametrize("accel", [np.zeros(3), np.array([2e-5, -1e-5, 5e-6])])
def test_relative_model_matches_differenced_absolute_propagations(accel):
    m = cr3bp.CR3BPDynamics(SYSTEM, NRHO)
    m.reset(OFFSET)
    final, samples = m.advance(accel, 86400.0, n_samples=5)
    expected = absolute_difference(NRHO, OFFSET, accel, 86400.0)
    # Each absolute state carries about 1e-13 of 385,000 km; the difference inherits it.
    assert np.max(np.abs(final[:3] - expected[:3])) < 2e-4
    assert np.max(np.abs(final[3:] - expected[3:])) < 1e-9
    assert np.array_equal(samples[-1], final)
    assert np.allclose(m.state_in_last_segment(43200.0), samples[2], rtol=0, atol=1e-9)


def test_relative_model_conserves_jacobi_and_thrust_does_its_work():
    # Book sec. 2.3: dC/dt = -2 v . a, so a held acceleration changes C by -2 a . dr.
    m = cr3bp.CR3BPDynamics(SYSTEM, NRHO)
    m.reset(OFFSET)
    start = m.chaser_state()
    m.advance(np.zeros(3), 2 * 86400.0)
    coasted = m.chaser_state()
    assert cr3bp.jacobi_constant(coasted, SYSTEM.mu) == pytest.approx(
        cr3bp.jacobi_constant(start, SYSTEM.mu), abs=1e-12
    )
    accel = np.array([3e-4, -2e-4, 1e-4])
    m.advance(accel, 3 * 3600.0)
    burned = m.chaser_state()
    work = -2 * (accel / SYSTEM.acceleration) @ (burned[:3] - coasted[:3])
    change = cr3bp.jacobi_constant(burned, SYSTEM.mu) - cr3bp.jacobi_constant(coasted, SYSTEM.mu)
    assert change == pytest.approx(work, rel=1e-9)


def test_massless_secondary_reduces_to_two_body_relative_motion():
    # With mu = 0 a target at rest on the unit circle is on a circular orbit about the larger
    # primary; at (0, 1, 0) the Hill axes are the synodic y, -x and z.
    system = ThreeBodySystem("test", 1, 0.0, 4e8, 2.3e6)
    m = cr3bp.CR3BPDynamics(system, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    gm = system.length**3 / system.time**2
    twobody = TwoBodyDynamics(Body("primary", gm, 0.0), altitude=system.length, rtol=1e-12)
    to_hill = np.kron(np.eye(2), np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    m.reset(OFFSET * 10)
    twobody.reset(to_hill @ OFFSET * 10)
    accel = np.array([1e-5, 2e-5, -1e-5])
    synodic, _ = m.advance(accel, 2 * 86400.0)
    hill, _ = twobody.advance(to_hill[:3, :3] @ accel, 2 * 86400.0)
    assert m.n == pytest.approx(twobody.n, rel=1e-15)
    assert np.max(np.abs(to_hill @ synodic - hill)[:3]) < 1e-6
    assert np.max(np.abs(to_hill @ synodic - hill)[3:]) < 1e-11


@pytest.mark.parametrize("dt", [1e-3, 1.0, 600.0, 86400.0, 2.4e6])
def test_libration_hold_with_c2_one_is_cw(dt):
    # Book eq 6.4.1 at c2 = 1 is CW; this includes CW's defective zero eigenvalue and short
    # steps, where both closed forms avoid cancellation.
    n = 2.66e-6
    phi, gamma = cr3bp.libration_discrete(n, 1.0, dt)
    phi_cw, gamma_cw = cw_discrete(n, dt)
    assert np.max(np.abs(phi - phi_cw)) < 4 * EPS * np.max(np.abs(phi_cw))
    assert np.max(np.abs(gamma - gamma_cw)) < 8 * EPS * np.max(np.abs(gamma_cw))


@pytest.mark.parametrize("dt", [10.0, 3600.0, 86400.0])
def test_libration_hold_matches_the_matrix_exponential(dt):
    # Van Loan's block exponential as an independent oracle, at Earth-Moon L1.
    n, c2 = 2.66e-6, cr3bp.collinear_coefficient(MU, 1, 2)
    block = np.zeros((9, 9))
    block[0:3, 3:6] = block[3:6, 6:9] = np.eye(3)
    block[3, 0], block[4, 1], block[5, 2] = (1 + 2 * c2) * n**2, (1 - c2) * n**2, -c2 * n**2
    block[3, 4], block[4, 3] = 2 * n, -2 * n
    exact = expm(block * dt)
    phi, gamma = cr3bp.libration_discrete(n, c2, dt)
    assert np.max(np.abs(phi - exact[:6, :6])) < 1e-14 * np.max(np.abs(phi))
    assert np.max(np.abs(gamma - exact[:6, 6:])) < 1e-14 * np.max(np.abs(gamma))


def test_libration_model_error_against_cr3bp_grows_as_range_squared():
    model = cr3bp.LibrationPointDynamics.at_point(SYSTEM, 1)
    truth = cr3bp.CR3BPDynamics(SYSTEM, np.r_[cr3bp.lagrange_points(SYSTEM.mu)[0], np.zeros(3)])
    assert model.n == truth.n
    u = np.array([0.3, -0.8, 0.5]) / np.linalg.norm([0.3, -0.8, 0.5])
    errors = []
    for rho in (10.0, 100.0, 1000.0):
        for m in (model, truth):
            m.reset(np.r_[rho * u, 1e-5 * rho * u[[1, 2, 0]]])
        errors.append(
            np.max(
                np.abs(
                    model.advance(np.zeros(3), 6 * 3600.0)[0][:3]
                    - truth.advance(np.zeros(3), 6 * 3600.0)[0][:3]
                )
            )
        )
    assert errors[0] < 1e-7
    assert errors[1] / errors[0] == pytest.approx(100.0, rel=1e-3)
    assert errors[2] / errors[1] == pytest.approx(100.0, rel=1e-3)


def test_libration_saddle_e_folds_in_a_day_and_a_half_at_earth_moon_l1():
    # Computed from the eigenvalues of book secs. 2.7 and 6.4: e-folding time 1 / lambda is
    # 1.5 days at Earth-Moon L1 and 2 days at L2.
    for point, days in ((1, 1.5), (2, 2.0)):
        lam, _, _, _ = cr3bp.libration_frequencies(cr3bp.collinear_coefficient(MU, point, 2))
        assert EARTH_MOON_KLMR.time / lam / 86400 == pytest.approx(days, abs=0.1)


@pytest.fixture(scope="module")
def nrho_linear():
    # Integrating the period once is most of the cost; every test resets the state it uses.
    return cr3bp.PeriodicLinearDynamics(SYSTEM, NRHO, NRHO_PERIOD)


def nrho_linear_reference(t0, t1, offset, accel):
    """Integrate the linear equations along the NRHO from nondimensional t0 to t1, restarting
    the target on the periodic orbit at each period boundary as the periodic model does."""

    def linear(t, y, a):
        return np.r_[cr3bp.rhs(t, y[:6], MU), cr3bp.jacobian(y[:3], MU) @ y[6:] + np.r_[0, 0, 0, a]]

    t, d = t0, offset
    k = math.floor(t0 / NRHO_PERIOD)
    target = propagate(NRHO, t0 - k * NRHO_PERIOD) if t0 > k * NRHO_PERIOD else NRHO
    while t < t1:
        stop = min(t1, (k + 1) * NRHO_PERIOD)
        end = solve_ivp(
            linear, (t, stop), np.r_[target, d], "DOP853", rtol=1e-13, atol=1e-18, args=(accel,)
        ).y[:, -1]
        t, d, target, k = stop, end[6:], NRHO, k + 1
    return d


@pytest.mark.parametrize(
    "start_days, days, tol", [(0.0, 0.007, 1e-12), (3.0, 1.0, 1e-12), (5.0, 2.0, 1e-7)]
)
def test_periodic_linear_model_matches_integrating_the_linear_equations(
    nrho_linear, start_days, days, tol
):
    # The last case runs through perilune and into the next period.
    m = nrho_linear
    accel = np.array([2e-5, -1e-5, 5e-6])
    phi, gamma = m.matrices(start_days * 86400.0, days * 86400.0)
    got = phi @ OFFSET + gamma @ accel
    scale = np.repeat([SYSTEM.length, SYSTEM.velocity], 3)
    t0 = start_days * 86400.0 / SYSTEM.time
    t1 = t0 + days * 86400.0 / SYSTEM.time
    expected = nrho_linear_reference(t0, t1, OFFSET / scale, accel / SYSTEM.acceleration) * scale
    assert np.max(np.abs(got - expected)) < tol * np.max(np.abs(expected))


def test_periodic_linear_model_reuses_the_monodromy_matrix(nrho_linear):
    m = nrho_linear
    period = NRHO_PERIOD * SYSTEM.time
    scale = np.repeat([SYSTEM.length, SYSTEM.velocity], 3)
    one, _ = m.matrices(0.0, period)
    two, _ = m.matrices(0.0, 2 * period)
    assert np.allclose(one * scale[None, :] / scale[:, None], m.monodromy, rtol=0, atol=1e-12)
    assert np.max(np.abs(two - one @ one)) < 1e-12 * np.max(np.abs(two))
    m.reset(OFFSET)
    before, _ = m.advance(np.zeros(3), 0.3 * period)
    forward, samples = m.advance([1e-5, 0.0, 0.0], 0.9 * period, n_samples=4)
    back, _ = m.advance([1e-5, 0.0, 0.0], -0.9 * period)
    assert np.allclose(samples[-1], forward, rtol=1e-14, atol=0)
    assert np.max(np.abs(back - before)) < 1e-10 * np.max(np.abs(forward))


def test_periodic_linear_model_error_against_cr3bp_grows_as_range_squared(nrho_linear):
    # A passive chaser offset from perilune, one day.
    model = nrho_linear
    truth = cr3bp.CR3BPDynamics(SYSTEM, NRHO)
    errors = []
    for rho in (10.0, 100.0, 1000.0):
        for m in (model, truth):
            m.reset([rho, 0.0, 0.0, 0.0, 0.0, 0.0])
        linear, _ = model.advance(np.zeros(3), 86400.0)
        nonlinear, _ = truth.advance(np.zeros(3), 86400.0)
        errors.append(np.linalg.norm(linear[:3] - nonlinear[:3]))
    assert errors[1] / errors[0] == pytest.approx(100.0, rel=2e-3)
    assert errors[2] / errors[1] == pytest.approx(100.0, rel=2e-3)


def test_cr3bp_transition_matrices_keep_the_rotating_symplectic_form():
    # Book secs. 4.3 and 6.7: symplectic in canonical coordinates. In [r, v] the invariant
    # form is J = [[2 W, I], [-I, 0]]; the standard one fails on correct code.
    def variational(t, y):
        phi = y[6:].reshape(6, 6)
        return np.r_[cr3bp.rhs(t, y[:6], MU), (cr3bp.jacobian(y[:3], MU) @ phi).ravel()]

    phi = (
        solve_ivp(
            variational, (0, 0.7), np.r_[NRHO, np.eye(6).ravel()], "DOP853", rtol=1e-13, atol=1e-15
        )
        .y[6:, -1]
        .reshape(6, 6)
    )
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rotating = np.block([[2 * w, np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])
    standard = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])
    size = np.max(np.abs(phi)) ** 2
    assert np.max(np.abs(phi.T @ rotating @ phi - rotating)) < 1e-15 * size
    assert np.max(np.abs(phi.T @ standard @ phi - standard)) > 1.0
    assert np.max(np.abs(cr3bp.transition_inverse(phi) @ phi - np.eye(6))) < 1e-15 * size
    assert np.linalg.det(phi) == pytest.approx(1.0, abs=1e-8)
