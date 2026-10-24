"""Set-valued passive safety: the free-drift barrier of a whole navigation error set.

The interface these tests assume:

- `vbar.safety.tightening.error_zonotope(error)` returns a `vbar.reach.Zonotope` centered at the
  zero error of the six Hill-frame state entries (m, m/s): a `BoxError` exactly, a `BallError` as
  the box of half-widths r_pos and r_vel (an outer bound), an `EllipsoidError` with generators P
  (an outer bound).
- `BackupFilter(barrier="passive", navigation=E)` with G_E its generators decides membership,
  in `contains` and in `filter`, by
  `drift_clearance(n, Zonotope(x_hat, G_E), R, t_cap, samples).certified >= 0`, and checks the
  end of a held step u by the same clearance of
  `Zonotope(Phi x_hat + Gamma u, [Phi G_E, Gamma da_max I])`, with (Phi, Gamma) the CW step of
  dt. The drift samples are `grid` s apart or closer over [0, t_cap]: samples =
  ceil(t_cap / grid) + 1. A decision whose backup fails the end check has no certificate and
  its slack is the shortfall, -certified at an end inside the set.

Setup after Kochenderfer et al., Algorithms for Validation, eq. 8.29 with the keep-out ball (our
case, values recomputed on CW): 500 km, keep-out 50 m, the V-bar hold (0, -100, 0) m at rest,
two orbits (t_cap = 4 pi / n, the filter's default under error sets), cases (a) to (d) below.
Grid 30 s: at 60 s the certified bound of the interval speeds loses 1.3 m, which puts
the (0.6 m, 0.3 mm/s) box at -7.43 m and the 1.38 mm/s box at -0.15 m.
"""

import math

import numpy as np
import pytest
from test_safety_cbf import N, view

from vbar.dynamics import cw_discrete
from vbar.reach import Zonotope, drift_clearance
from vbar.safety.cbf import BackupFilter
from vbar.safety.tightening import BallError, BoxError, EllipsoidError, halfspace_margin

R = 50.0
DT = 10.0
HOLD = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
T_CAP = 4.0 * math.pi / N
GRID = 30.0
SAMPLES = math.ceil(T_CAP / GRID) + 1


def error_zonotope(error):
    from vbar.safety.tightening import error_zonotope as build

    return build(error)


def passive(error, grid=GRID, **kw):
    filt = BackupFilter(barrier="passive", grid=grid, navigation=error, **kw)
    filt.reset(view(DT, start=HOLD))
    return filt


def set_clearance(x, error, grid=GRID):
    samples = math.ceil(T_CAP / grid) + 1
    gens = error_zonotope(error).generators
    return drift_clearance(N, Zonotope(x, gens), R, T_CAP, samples)


DIRECTIONS = np.random.default_rng(1122).normal(size=(40, 6))


def test_box_error_zonotope_has_the_box_support_exactly():
    error = BoxError(r_pos=0.6, r_vel=3e-4)
    z = error_zonotope(error)
    assert z.dim == 6
    np.testing.assert_array_equal(z.center, np.zeros(6))
    for d in DIRECTIONS:
        assert z.support(d) == pytest.approx(halfspace_margin(d, error), rel=1e-12)


def test_ball_error_zonotope_is_the_box_of_the_radii():
    ball = error_zonotope(BallError(r_pos=0.6, r_vel=3e-4))
    box = BoxError(r_pos=0.6, r_vel=3e-4)
    for d in DIRECTIONS:
        assert ball.support(d) == pytest.approx(halfspace_margin(d, box), rel=1e-12)
        assert ball.support(d) >= halfspace_margin(d, BallError(r_pos=0.6, r_vel=3e-4))


def test_ellipsoid_error_zonotope_has_the_shape_matrix_as_generators():
    rng = np.random.default_rng(18)
    shape = rng.normal(size=(6, 6)) * np.array([0.5] * 3 + [5e-4] * 3)[:, None]
    error = EllipsoidError(P=shape)
    z = error_zonotope(error)
    for d in DIRECTIONS:
        assert z.support(d) == pytest.approx(np.sum(np.abs(shape.T @ d)), rel=1e-12)
        assert z.support(d) >= halfspace_margin(d, error) * (1.0 - 1e-12)


# Case (a): a 3-sigma box of (3 m, 3 mm/s) about the hold.


def test_point_estimate_at_the_hold_drifts_50_m_clear():
    assert drift_clearance(N, HOLD, R, T_CAP, SAMPLES).found == pytest.approx(50.0, abs=1e-3)


def test_large_navigation_box_at_the_hold_is_outside_the_set():
    # A state of the box drifts through the target (found -50.000 m at every grid), so the
    # certified bound is at least 50 m below zero whatever the grid. The 120 s grid keeps one
    # clearance of the box near 0.1 s (certified -68.5 m) and the point barrier at +47.97 m.
    error = BoxError(r_pos=3.0, r_vel=3e-3)
    # The candidate and the backup are both zero, so bisection would only repeat the check.
    filt = passive(error, grid=120.0, iterations=0)
    samples = math.ceil(T_CAP / 120.0) + 1
    assert drift_clearance(N, HOLD, R, T_CAP, samples).found == pytest.approx(50.0, abs=1e-3)
    assert filt.declaration.barriers[0].value(HOLD) > 0.0
    assert filt.contains(HOLD) is False
    result = filt.filter(HOLD.copy(), np.zeros(3))
    assert result.recovering is True and result.certificate is None


# Case (b): the (0.6 m, 0.3 mm/s) box, found minimum -6.157 m.


def test_set_margin_of_the_small_box_lies_at_most_half_a_metre_below_the_found_minimum():
    error = BoxError(r_pos=0.6, r_vel=3e-4)
    found = set_clearance(HOLD, error)
    assert found.found == pytest.approx(-6.157, abs=1e-3)
    assert -6.657 <= found.certified <= -6.157
    assert passive(error).contains(HOLD) is False


# Case (c): velocity-only boxes; the largest one certified clear is 1.4148 mm/s, between the
# 1.4140 mm/s and 1.4160 mm/s boxes of the reachability checks.


def test_velocity_box_below_the_threshold_is_certified_for_two_orbits():
    filt = passive(BoxError(r_pos=0.0, r_vel=1.38e-3))
    assert filt.contains(HOLD) is True
    result = filt.filter(HOLD.copy(), np.zeros(3))
    assert result.recovering is False
    assert result.certificate == "bounded-W"
    assert result.certificate_horizon == pytest.approx(T_CAP, rel=1e-12)


def test_velocity_box_above_the_threshold_is_outside_the_set():
    filt = passive(BoxError(r_pos=0.0, r_vel=1.45e-3))
    assert filt.contains(HOLD) is False
    result = filt.filter(HOLD.copy(), np.zeros(3))
    assert result.recovering is True and result.certificate is None


# Case (d): the end check carries the execution error through Gamma da I.


def test_execution_error_at_the_end_of_the_step_withdraws_the_certificate():
    # The 1.38 mm/s box drifts 0.854 m clear; 1e-5 m/s^2 held for one 10 s step costs 3.5 m.
    error = BoxError(r_pos=0.0, r_vel=1.38e-3)
    da = 1e-5
    phi, gamma = cw_discrete(N, DT)
    gens = error_zonotope(error).generators
    end = drift_clearance(
        N, Zonotope(phi @ HOLD, np.hstack([phi @ gens, da * gamma])), R, T_CAP, SAMPLES
    )
    without = drift_clearance(N, Zonotope(phi @ HOLD, phi @ gens), R, T_CAP, SAMPLES)
    assert end.certified < -2.0 and without.certified > 0.5
    # The candidate is the backup, so bisection adds nothing but clearance evaluations.
    filt = passive(error, execution=da, iterations=0)
    assert filt.contains(HOLD) is True
    result = filt.filter(HOLD.copy(), np.zeros(3))
    assert result.recovering is False
    assert result.certificate is None
    assert result.slack == pytest.approx(-end.certified, abs=0.02)
