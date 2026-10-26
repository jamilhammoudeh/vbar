"""Torque-free rigid-body attitude: the closed form against known answers and an integrator.

The independent check integrates Euler's equations with the quaternion kinematics
q_dot = 1/2 q (x) [w, 0] (DOP853, relative tolerance 1e-12), every case in one stacked system on
its own time scale, so the module pays for one integration of a few seconds.
"""

import functools
import itertools
import math
from unittest import mock

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from vbar.dynamics.rigid_body import GravityGradientMotion, TorqueFreeMotion, principal_axes


def integrated(bodies):
    """Attitudes R_n_b and body rates of every (moments, omega, attitude, span) body, each at 61
    evenly spaced times over its own span, from one integration of all of them on s = t / span.
    """
    moments = np.array([b[0] for b in bodies], dtype=float)
    spans = np.array([b[3] for b in bodies], dtype=float)
    count = len(bodies)

    def rhs(s, y):
        y = y.reshape(count, 7)
        x, yy, z, q0 = y[:, :4].T
        w = y[:, 4:]
        wx, wy, wz = w.T
        dq = 0.5 * np.stack(
            [
                q0 * wx + yy * wz - z * wy,
                q0 * wy + z * wx - x * wz,
                q0 * wz + x * wy - yy * wx,
                -x * wx - yy * wy - z * wz,
            ],
            axis=1,
        )
        dw = np.cross(moments * w, w) / moments
        return (np.concatenate([dq, dw], axis=1) * spans[:, None]).ravel()

    y0 = np.concatenate([np.r_[b[2], b[1]] for b in bodies])
    grid = np.linspace(0.0, 1.0, 61)
    sol = solve_ivp(rhs, (0.0, 1.0), y0, method="DOP853", t_eval=grid, rtol=1e-12, atol=1e-14)
    y = sol.y.T.reshape(len(grid), count, 7)
    return [
        (grid * span, Rotation.from_quat(y[:, k, :4]).as_matrix(), y[:, k, 4:])
        for k, span in enumerate(spans)
    ]


def angle_between(a, b):
    """Rotation angle (rad) between stacked rotation matrices."""
    return np.linalg.norm(Rotation.from_matrix(np.swapaxes(a, -1, -2) @ b).as_rotvec(), axis=-1)


def test_principal_axes_match_schaub_example_and_are_right_handed():
    # Schaub and Junkins, example 4.2: principal inertias 7.04892, 2.30798 and 2.64310 kg m^2.
    inertia = np.array([[3.0, 1.0, 1.0], [1.0, 5.0, 2.0], [1.0, 2.0, 4.0]])
    moments, axes = principal_axes(inertia)
    assert moments == pytest.approx([2.30798, 2.64310, 7.04892], abs=5e-6)
    assert np.linalg.det(axes) == pytest.approx(1.0)
    assert axes.T @ inertia @ axes == pytest.approx(np.diag(moments), abs=1e-12)
    # Whatever sign an eigensolver gives each eigenvector, the axes come out right-handed: every
    # sign pattern of the solver's own result is fed back through a stand-in solver.
    swapped = np.diag([2.0, 1.0, 3.0])
    values, vectors = np.linalg.eigh(swapped)
    for signs in itertools.product((1.0, -1.0), repeat=3):
        with mock.patch.object(np.linalg, "eigh", return_value=(values, vectors * signs)):
            moments, axes = principal_axes(swapped)
        assert moments == pytest.approx([1.0, 2.0, 3.0])
        assert np.linalg.det(axes) == pytest.approx(1.0)
        assert axes.T @ swapped @ axes == pytest.approx(np.diag(moments), abs=1e-15)
    with pytest.raises(ValueError):
        principal_axes([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        principal_axes(-np.eye(3))


def test_axisymmetric_body_rate_turns_at_the_body_precession_rate():
    # Schaub and Junkins, eq. 4.42: with I1 = I2 = I_T the transverse rate turns at
    # (I3 / I_T - 1) w3 in the body while w3 stays constant.
    i_t, i3 = 100.0, 300.0
    w0 = np.array([0.03, -0.01, 0.02])
    motion = TorqueFreeMotion((i_t, i_t, i3), w0)
    times = np.linspace(0.0, 2000.0, 401)
    rate = (i3 / i_t - 1.0) * w0[2]
    turned = np.einsum(
        "tij,j->ti", Rotation.from_rotvec(np.outer(rate * times, [0, 0, 1])).as_matrix(), w0
    )
    assert motion.omega(times) == pytest.approx(turned, rel=1e-10, abs=1e-14)


def test_energy_and_inertial_momentum_are_conserved_with_schaubs_numbers():
    # Schaub and Junkins, problem 4.9: I = diag(210, 200, 118) kg m^2 and
    # w0 = (0.2, 0.15, -0.18) rad/s give T = 8.3616 J and |H| = 55.8134 N m s.
    moments = np.array([210.0, 200.0, 118.0])
    motion = TorqueFreeMotion(
        moments, (0.2, 0.15, -0.18), Rotation.from_rotvec([0.3, -1, 2]).as_quat()
    )
    times = np.linspace(0.0, 30.0, 301)
    w = motion.omega(times)
    energy = 0.5 * np.sum(moments * w**2, axis=-1)
    momentum = np.einsum("tij,tj->ti", motion.matrix(times), moments * w)
    assert energy == pytest.approx(8.3616, rel=1e-10)
    assert np.linalg.norm(momentum, axis=-1) == pytest.approx(55.8134, abs=5e-5)
    assert momentum == pytest.approx(np.broadcast_to(motion.momentum, momentum.shape), rel=1e-10)


def test_toutatis_long_axis_tumble_matches_scheeres():
    # Scheeres, section 3.2 and table 8.1 (Toutatis): Ix/Iz = 0.31335, Iy/Iz = 0.94471,
    # I_D/Iz = 0.51983, w_l = 1.8548e-5 rad/s give k^2 = 0.037658 and a body-rate period of
    # 5.4253 d, and the 3-1-3 precession angle advances on average at 2.3331e-5 rad/s. The
    # node of the minor axis used here winds once less per period about H, so its mean rate
    # is lower by 2 pi / P; the rotation per period about H is the same.
    motion = TorqueFreeMotion.from_tumble((0.31335, 0.94471, 1.0), 0.51983, 1.8548e-5)
    assert motion.long_axis
    assert motion.m == pytest.approx(0.037658, abs=5e-7)
    assert motion.period / 86400.0 == pytest.approx(5.4253, abs=5e-5)
    period = motion.period
    phi = motion.precession(np.array([0.0, 40.0 * period]))
    assert (phi[1] - phi[0]) / (40.0 * period) + 2.0 * math.pi / period == pytest.approx(
        2.3331e-5, rel=1e-4
    )
    # A whole period returns the body rate and turns the body about H by that angle.
    start, end = motion.matrix(np.array([1234.0, 1234.0 + period]))
    turn = Rotation.from_matrix(end @ start.T).as_rotvec()
    h = motion.momentum / motion.h
    assert np.cross(turn, h) == pytest.approx(np.zeros(3), abs=1e-8)
    assert motion.omega(1234.0 + period) == pytest.approx(motion.omega(1234.0), rel=1e-9)


def test_small_nutation_period_tends_to_the_principal_spin_formula():
    # Scheeres, eqs. 3.84 to 3.87: near a spin about the major axis the body-rate period tends to
    # sqrt(Ix Iy / ((Iz - Iy)(Iz - Ix))) 2 pi / w_l, and the modulus to zero.
    ix, iy, w_l = 0.31335, 0.94471, 1.8548e-5
    motion = TorqueFreeMotion.from_tumble((ix, iy, 1.0), 1.0 - 1e-9, w_l)
    expected = math.sqrt(ix * iy / ((1.0 - iy) * (1.0 - ix))) * 2.0 * math.pi / w_l
    assert motion.period == pytest.approx(expected, rel=1e-8)
    assert motion.m < 1e-7
    near = TorqueFreeMotion.from_tumble((ix, iy, 1.0), iy * (1.0 + 1e-6), w_l)
    assert 1.0 - near.m < 1e-4


def _near_separatrix(factor):
    """A body rate of the (300, 200, 100) kg m^2 body with D = factor I2."""
    motion = TorqueFreeMotion.from_tumble((300.0, 200.0, 100.0), 200.0 * factor, 0.03, 0.1)
    return tuple(motion.omega0)


CASES = {
    "long-axis": ((210.0, 200.0, 118.0), (0.2, 0.15, -0.18)),
    "short-axis": ((300.0, 200.0, 100.0), (0.05, 0.01, 0.02)),
    "upper-stage": ((1000.0, 4200.0, 4700.0), (0.01, 0.02, 0.03)),
    "oblate": ((100.0, 100.0, 300.0), (0.03, -0.01, 0.02)),
    "prolate": ((100.0, 300.0, 300.0), (0.03, -0.01, 0.02)),
    # D one part in a thousand either side of the intermediate moment: m = 0.996 on both sides.
    "near-separatrix-short": ((300.0, 200.0, 100.0), _near_separatrix(1.001)),
    "near-separatrix-long": ((300.0, 200.0, 100.0), _near_separatrix(0.999)),
    "small-nutation": ((100.0, 200.0, 300.0), (1e-5, -2e-5, 0.05)),
    "spin-intermediate": ((100.0, 200.0, 300.0), (0.0, 0.05, 0.0)),
    "spin-sphere": ((100.0, 100.0, 100.0), (0.01, -0.02, 0.03)),
    "at-rest": ((100.0, 200.0, 300.0), (0.0, 0.0, 0.0)),
}


ATTITUDE = Rotation.from_rotvec([0.4, -0.7, 1.9]).as_quat()


def _sweep():
    rng = np.random.default_rng(2026)
    return [
        (rng.uniform(50.0, 500.0, 3), rng.normal(0.0, 0.05, 3), Rotation.random(random_state=rng))
        for _ in range(12)
    ]


@functools.cache
def integrated_cases():
    """Each named case over three body periods (600 s for a spin) and the sweep over two."""
    bodies = []
    for moments, omega in CASES.values():
        motion = TorqueFreeMotion(moments, omega, ATTITUDE)
        span = 3.0 * motion.period if math.isfinite(motion.period) else 600.0
        bodies.append((moments, omega, ATTITUDE, span))
    for moments, omega, attitude in _sweep():
        motion = TorqueFreeMotion(moments, omega, attitude.as_quat())
        bodies.append((moments, omega, attitude.as_quat(), 2.0 * motion.period))
    return bodies, integrated(bodies)


@pytest.mark.parametrize("name", CASES)
def test_closed_form_attitude_matches_integrated_eulers_equations(name):
    bodies, runs = integrated_cases()
    k = list(CASES).index(name)
    moments, omega, attitude, _ = bodies[k]
    times, r, w = runs[k]
    motion = TorqueFreeMotion(moments, omega, attitude)
    scale = max(float(np.linalg.norm(omega)), 1e-12)
    # The integrator's own error over three periods is below 1e-10 rad and 1e-10 relative.
    assert np.max(angle_between(motion.matrix(times), r)) < 1e-8
    assert np.max(np.linalg.norm(motion.omega(times) - w, axis=-1)) / scale < 1e-9
    # The bounds hold along the integrated motion and are within a factor 2 of its extremes.
    rates = np.linalg.norm(w, axis=-1)
    assert np.all(rates <= motion.rate_bound() * (1.0 + 1e-12))
    assert rates.max() >= 0.5 * motion.rate_bound()
    moments = np.asarray(moments, dtype=float)
    spin_up = np.linalg.norm(np.cross(moments * w, w) / moments, axis=-1)
    assert np.all(spin_up <= motion.acceleration_bound() * (1.0 + 1e-9) + 1e-18)
    assert spin_up.max() >= 0.5 * motion.acceleration_bound()
    inertial = motion.angular_velocity(times)
    assert inertial == pytest.approx(np.einsum("tij,tj->ti", r, w), abs=1e-9 * scale)


def test_a_sweep_of_random_bodies_and_rates_matches_the_integrator():
    bodies, runs = integrated_cases()
    worst = 0.0
    for (moments, omega, attitude, _), (times, r, _) in list(zip(bodies, runs, strict=True))[
        len(CASES) :
    ]:
        motion = TorqueFreeMotion(moments, omega, attitude)
        worst = max(worst, float(np.max(angle_between(motion.matrix(times), r))))
    assert worst < 1e-8


def test_tumble_parameters_set_the_momentum_energy_and_phase():
    moments = (1000.0, 4200.0, 4700.0)
    direction = np.array([0.3, -0.5, 0.8])
    for d in (1500.0, 3000.0, 4500.0, 4700.0):
        motion = TorqueFreeMotion.from_tumble(moments, d, 0.02, 0.3, direction, 1.1)
        assert motion.effective_inertia == pytest.approx(d, rel=1e-12)
        assert motion.h / d == pytest.approx(0.02, rel=1e-12)
        assert motion.momentum / motion.h == pytest.approx(direction / np.linalg.norm(direction))
    # Starting a quarter period later is the same tumble a quarter period on, up to the turn of
    # the frame about H.
    early = TorqueFreeMotion.from_tumble(moments, 3000.0, 0.02, 0.0)
    late = TorqueFreeMotion.from_tumble(moments, 3000.0, 0.02, 0.25)
    assert late.omega(0.0) == pytest.approx(early.omega(0.25 * early.period), rel=1e-9)
    with pytest.raises(ValueError):
        TorqueFreeMotion.from_tumble(moments, 4200.0, 0.02)
    with pytest.raises(ValueError):
        TorqueFreeMotion.from_tumble(moments, 900.0, 0.02)


def test_the_exact_separatrix_and_bad_inputs_are_refused():
    # With I = (1, 5, 9) and w = (3, 0, 1), H^2 = 90 and 2 T = 18, so D = 5 = I2 exactly.
    with pytest.raises(ValueError):
        TorqueFreeMotion((1.0, 5.0, 9.0), (3.0, 0.0, 1.0))
    assert not TorqueFreeMotion((1.0, 5.0, 9.0), (3.0, 0.0, 1.001)).long_axis
    with pytest.raises(ValueError):
        TorqueFreeMotion((1.0, 0.0, 3.0), (0.1, 0.0, 0.0))
    with pytest.raises(ValueError):
        TorqueFreeMotion((1.0, 2.0, 3.0), (0.1, 0.0), (0, 0, 0, 1))


def test_without_an_orbit_the_gravity_gradient_motion_is_the_closed_form():
    # At n = 0 the torque vanishes, so the integrated motion is the torque-free closed form: a
    # check of the integration against the independent solution.
    moments, omega = (1000.0, 4200.0, 4700.0), (0.004, -0.006, 0.017)
    attitude = Rotation.from_rotvec([0.3, 0.9, -1.2]).as_quat()
    numeric = GravityGradientMotion(moments, omega, attitude, 0.0)
    closed = TorqueFreeMotion(moments, omega, attitude)
    times = np.linspace(0.0, 1000.0, 41)
    assert np.max(angle_between(numeric.matrix(times), closed.matrix(times))) < 1e-8
    assert numeric.omega(times) == pytest.approx(closed.omega(times), abs=1e-10)
    with pytest.raises(ValueError):
        numeric.matrix(-1.0)


def test_the_gravity_gradient_motion_keeps_its_jacobi_integral_and_bounds():
    # On a circular orbit the attitude keeps J = 1/2 w_r . I w_r + 3/2 n^2 o . I o
    # - 1/2 n^2 z . I z; the rate bounds it gives hold along the motion.
    n = 1.1e-3
    moments = np.array([1000.0, 4200.0, 4700.0])
    motion = GravityGradientMotion(
        moments, (0.002, -0.003, 0.006), Rotation.from_rotvec([1.0, -0.5, 0.2]).as_quat(), n
    )
    times = np.linspace(0.0, 1800.0, 181)
    r_nb, w = motion.matrix(times), motion.omega(times)
    values = [motion._jacobi(t, r, v) for t, r, v in zip(times, r_nb, w, strict=True)]
    assert np.array(values) == pytest.approx(motion.jacobi, rel=1e-10)
    normal = np.einsum("tji,j->ti", r_nb, (0.0, 0.0, 1.0))
    relative = np.linalg.norm(w - n * normal, axis=-1)
    assert np.all(relative <= motion.relative_rate_bound())
    spin_up = np.diff(w - n * normal, axis=0) / np.diff(times)[:, None]
    largest = np.max(np.linalg.norm(spin_up, axis=-1))
    # The componentwise bound is 7 times the largest spin-up of this tumble; one much looser would
    # brief the capture point as far quicker to speed up than it is.
    assert largest <= motion.relative_acceleration_bound() <= 10.0 * largest
    # The same time asked for alone or after others gives the same attitude: whole pieces.
    for t in (1500.0, 3000.0):
        alone = GravityGradientMotion(
            moments, (0.002, -0.003, 0.006), Rotation.from_rotvec([1.0, -0.5, 0.2]).as_quat(), n
        )
        assert np.array_equal(alone.matrix(t), motion.matrix(t))
    with pytest.raises(ValueError):
        motion.matrix(np.inf)


def test_the_relative_rate_bound_is_reached_on_the_long_axis_radial():
    # Long axis radial and largest axis on the orbit normal is the potential's minimum, and a
    # relative rate about the smallest axis holds all of J - V_min, so the bound is equality.
    n = 1.1e-3
    motion = GravityGradientMotion((1000.0, 4200.0, 4700.0), (0.003, 0.0, n), (0, 0, 0, 1), n)
    assert motion.relative_rate_bound() == pytest.approx(0.003, rel=1e-12)


@pytest.mark.parametrize("pitch", [0.2, 2.0])
def test_a_small_pitch_librates_at_schaubs_gravity_gradient_frequency(pitch):
    # Schaub and Junkins, eq. 4.158, in V-BAR axes (largest moment about the orbit normal z,
    # smallest along the radial x): a body held on the Hill axes and pitched by a small angle
    # librates about z at n sqrt(3 (I_yy - I_xx) / I_zz). The torque goes as sin 2 theta, a
    # pendulum in 2 theta, so the period is longer by (2 theta_0)^2 / 16 at amplitude theta_0.
    # The next term of the series, 11/3072 (2 theta_0)^4, is 8.5e-8 at 2 deg, inside 1e-6.
    n = 1.1e-3
    moments = np.array([1000.0, 4200.0, 4700.0])
    delta = math.radians(pitch)
    motion = GravityGradientMotion(
        moments, (0.0, 0.0, n), Rotation.from_rotvec([0.0, 0.0, delta]).as_quat(), n
    )
    linear = 2.0 * math.pi / (n * math.sqrt(3.0 * (moments[1] - moments[0]) / moments[2]))
    times = np.linspace(0.0, 2.2 * linear, 22001)
    r_hb = np.einsum("tij,tjk->tik", _turn(-n * times), motion.matrix(times))
    angle = np.arctan2(r_hb[:, 1, 0], r_hb[:, 0, 0])
    down = np.flatnonzero((angle[:-1] > 0.0) & (angle[1:] <= 0.0))
    crossings = [brentq(lambda s: _pitch(motion, n, s), times[k], times[k + 1]) for k in down]
    period = crossings[1] - crossings[0]
    assert period == pytest.approx(linear * (1.0 + (2.0 * delta) ** 2 / 16.0), rel=1e-6)
    assert np.max(np.abs(angle)) == pytest.approx(delta, rel=1e-6)


def _turn(angle):
    return Rotation.from_rotvec(np.outer(angle, (0.0, 0.0, 1.0))).as_matrix()


def _pitch(motion, n, t):
    r_hb = _turn(np.array([-n * t]))[0] @ motion.matrix(t)
    return math.atan2(r_hb[1, 0], r_hb[0, 0])
