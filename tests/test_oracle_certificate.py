"""Primer-vector certificates of impulsive plans and the indirect finite-thrust optimum on CW.

An impulsive plan that reaches its target is fuel-optimal in the actuator norm when some
multiplier nu gives a primer p(t) = B^T Phi(t_f - t)^T nu that equals the burn direction at every
burn (the unit vector for one engine, the sign of each burning axis for box thrusters) and stays
within 1 in the dual norm (l2 for one engine, l_inf for box thrusters) over the whole interval.
For linear dynamics these conditions are sufficient as well as necessary (D. F. Lawden, Optimal
Trajectories for Space Navigation, 1963; Longuski, Guzman and Prussing, Optimal Control with
Aerospace Applications, 2014, section 10.5), and the plan's cost then equals the impulsive lower
bound D(nu) / sup |p|_*, reported as `lower` (0 when negative). The multiplier is fitted to the
plan's own burns; among those that fit, the certificate keeps one of least primer peak. `reason`
names the first condition that fails: "miss" (the plan does not reach the target), "direction"
(no multiplier puts the primer along every burn), "primer_max" (the primer of the fitted
multiplier exceeds 1 + 1e-9); it is empty when the plan is certified.

With an acceleration limit the optimum switches between full thrust and coasting where the primer
crosses 1 (bang-off-bang); the indirect solve finds the multiplier whose switching control lands
on the target. Its cost must agree with the direct LP optima on refined step grids, and the
Hamiltonian H = |a| - nu^T Phi(t_f - t) (A x + B a), with costate -Phi(t_f - t)^T nu, must stay
constant along it (Longuski, Guzman and Prussing, chapters 4 and 10).

Values marked "independent computation" were computed with `cw_discrete`, scipy's HiGHS, Brent
root finding and quadrature in a separate implementation, not with the module under test.
"""

import functools
import math
from typing import NamedTuple

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.optimize import brentq, minimize_scalar

from vbar.agents.base import AgentView
from vbar.agents.cw_targeting import two_impulse
from vbar.dynamics import cw_discrete, cw_system, mean_motion
from vbar.dynamics.cw import R_EARTH
from vbar.oracles import certificate
from vbar.oracles.fuel import FuelOracle
from vbar.oracles.primer import dual_bound


class Plan(NamedTuple):
    n: float
    start: np.ndarray
    impulses: list
    target: np.ndarray
    tf: float


def primer_at(n, nu, tf, t):
    """Primer (..., 3) of one terminal multiplier at times t before tf, from the ZOH closed form."""
    phi, _ = cw_discrete(n, tf - np.asarray(t, dtype=float))
    return np.swapaxes(phi[..., :, 3:6], -1, -2) @ nu


def dual_size(p, norm):
    return np.abs(p).max(axis=-1) if norm == "l1" else np.linalg.norm(p, axis=-1)


def peak(n, nu, tf, norm):
    """sup over [0, tf] of |p|_*: 20001 samples, then a bounded search at every local maximum."""
    t = np.linspace(0.0, tf, 20001)
    f = dual_size(primer_at(n, nu, tf, t), norm)
    best = f.max()
    for k in np.flatnonzero((f[1:-1] >= f[:-2]) & (f[1:-1] >= f[2:])) + 1:
        found = minimize_scalar(
            lambda s: -dual_size(primer_at(n, nu, tf, s), norm),
            bounds=(t[k - 1], t[k + 1]),
            method="bounded",
            options={"xatol": 1e-14 * tf},
        )
        best = max(best, -found.fun)
    return best


def fly(n, start, impulses, tf):
    x, now = np.array(start, dtype=float), 0.0
    for t, dv in impulses:
        x = cw_discrete(n, t - now)[0] @ x
        x[3:] += dv
        now = t
    return cw_discrete(n, tf - now)[0] @ x


def plan_to(n, start, impulses, tf):
    """A plan whose target is wherever its burns take the start."""
    impulses = [(t, np.asarray(dv, dtype=float)) for t, dv in impulses]
    start = np.asarray(start, dtype=float)
    return Plan(n, start, impulses, fly(n, start, impulses, tf), tf)


def certify(plan, norm):
    cert = certificate.certify_impulsive(
        plan.n, plan.start, plan.impulses, plan.target, tf=plan.tf, norm=norm
    )
    assert isinstance(cert, certificate.PrimerCertificate)
    assert_consistent(cert, plan, norm)
    return cert


def assert_consistent(cert, plan, norm):
    """Cost in the actuator norm, a sound and tight primer peak, and lower = D(nu) / peak."""
    order = 1 if norm == "l1" else 2
    cost = sum(np.linalg.norm(dv, ord=order) for _, dv in plan.impulses)
    assert cert.cost == pytest.approx(cost, rel=1e-12)
    nu = np.asarray(cert.nu, dtype=float)
    assert nu.shape == (6,)
    seen = peak(plan.n, nu, plan.tf, norm)
    # A certified supremum is never below a sampled value; near the threshold it must also be
    # tight, or optimal plans cannot pass at 1 + 1e-9.
    assert seen <= cert.primer_max * (1 + 1e-12)
    if seen <= 1 + 1e-6:
        assert cert.primer_max <= seen + 1e-9
    d = nu @ (plan.target - cw_discrete(plan.n, plan.tf)[0] @ plan.start)
    expected = max(0.0, d / cert.primer_max) if cert.primer_max > 0 else 0.0
    assert cert.lower == pytest.approx(expected, rel=1e-10, abs=1e-18)
    assert cert.optimal == (cert.reason == "")


def assert_certified(cert):
    assert cert.optimal
    assert cert.reason == ""
    # The primer is 1 at a burn, so its peak is at least 1.
    assert 1.0 - 1e-12 <= cert.primer_max <= 1.0 + 1e-9
    assert cert.cost / (1.0 + 1e-9) <= cert.lower <= cert.cost * (1 + 1e-12)


# Fehse, Automated Rendezvous and Docking of Spacecraft, 2003, eqs 3.29 to 3.32 and 3.37: from
# 500 m behind to the target point in one orbit with two tangential burns of n dy / (6 pi), or in
# half an orbit with two radial burns of n dy / 4; 500 km circular orbit (WGS 84).
N_500 = mean_motion(R_EARTH + 500e3)
DY = 500.0
FEHSE_START = np.array([0.0, -DY, 0.0, 0.0, 0.0, 0.0])


def fehse(transfer, scale_second=1.0):
    if transfer == "tangential":
        tf = 2 * math.pi / N_500
        dv = N_500 * DY / (6 * math.pi)
        impulses = [(0.0, [0.0, -dv, 0.0]), (tf, [0.0, scale_second * dv, 0.0])]
    else:
        tf = math.pi / N_500
        dv = N_500 * DY / 4
        impulses = [(0.0, [-dv, 0.0, 0.0]), (tf, [-scale_second * dv, 0.0, 0.0])]
    impulses = [(t, np.array(v)) for t, v in impulses]
    return Plan(N_500, FEHSE_START, impulses, np.zeros(6), tf)


@pytest.mark.parametrize(
    "transfer,optimum",
    [("tangential", N_500 * DY / (3 * math.pi)), ("radial", N_500 * DY / 2)],
)
def test_fehse_transfers_certify_as_box_optimal(transfer, optimum):
    # Certifying multipliers exist in closed form (independent computation): tangential
    # nu = (2 n, n / (3 pi), 0, -2 / (3 pi), 1, 0) with primer (-2 / (3 pi), 1 - n tau / pi, 0);
    # radial nu = (3 pi n / 8, n / 2, 0, -1, 0, 0). On the tangential transfer any certifying
    # multiplier has p_x <= -1 / (6 pi) at the burns, or p_y leaves [-1, 1] just after the first
    # one, so the 2-norm of the primer there is at least sqrt(1 + 1 / (6 pi)^2) = 1.0014:
    # measured in the wrong dual norm this optimum would be rejected.
    plan = fehse(transfer)
    cert = certify(plan, "l1")
    assert_certified(cert)
    assert cert.cost == pytest.approx(optimum, rel=1e-12)
    if transfer == "tangential":
        assert peak(plan.n, np.asarray(cert.nu), plan.tf, "l2") >= 1.0014


def test_radial_transfer_certifies_for_a_single_engine():
    # Both burns lie along x, so the 2-norm cost is n dy / 2 too, and the least-squares
    # multiplier keeps |p|_2 <= 1 (independent computation).
    cert = certify(fehse("radial"), "l2")
    assert_certified(cert)
    assert cert.cost == pytest.approx(N_500 * DY / 2, rel=1e-12)


def test_plan_that_misses_the_target_is_not_certified():
    cert = certify(fehse("radial", scale_second=1.01), "l1")
    assert not cert.optimal
    assert cert.reason == "miss"


def final_state_size(plan):
    """Sum of the 2-norms of the free drift, the target and each burn's kick at tf, velocities
    over n: the scale the 1e-9 terminal-miss tolerance is relative to."""
    hill = np.r_[np.ones(3), np.full(3, 1 / plan.n)]
    drift = cw_discrete(plan.n, plan.tf)[0] @ plan.start
    kicks = [cw_discrete(plan.n, plan.tf - t)[0][:, 3:6] @ dv for t, dv in plan.impulses]
    return sum(np.linalg.norm(hill * x) for x in [drift, plan.target, *kicks])


@pytest.mark.parametrize("fraction,misses", [(0.5, False), (2.0, True)])
def test_target_is_reached_within_1e_9_of_the_final_state_size(fraction, misses):
    plan = fehse("radial")
    offset = fraction * 1e-9 * final_state_size(plan)
    cert = certify(plan._replace(target=plan.target + [offset, 0, 0, 0, 0, 0]), "l1")
    assert cert.optimal is not misses
    assert cert.reason == ("miss" if misses else "")


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_empty_plan_at_its_target_is_optimal_at_zero_cost(norm):
    # A V-bar point at rest is an equilibrium of CW motion, so holding it needs no burn.
    hold = np.array([0.0, -DY, 0.0, 0.0, 0.0, 0.0])
    cert = certificate.certify_impulsive(N_500, hold, [], hold, tf=2 * math.pi / N_500, norm=norm)
    assert cert.optimal
    assert cert.reason == ""
    assert cert.cost == 0.0
    assert cert.lower == 0.0
    assert cert.primer_max <= 1.0


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_empty_plan_away_from_its_target_misses(norm):
    cert = certificate.certify_impulsive(
        N_500, FEHSE_START, [], np.zeros(6), tf=2 * math.pi / N_500, norm=norm
    )
    assert not cert.optimal
    assert cert.reason == "miss"
    assert cert.cost == 0.0


# Alfriend, Vadali, Gurfil, How and Breger, Spacecraft Formation Flying, 2010, example 5.2, in
# normalized units n = 1: chaser at (0.01, 0.02, 0.015, 0.001, 0.001, 0.001) to the origin at
# rest. The least two-impulse 2-norm cost is 0.03572 at n tf = 4.6404 (the book prints 0.0375).
X0_ALFRIEND = np.array([0.01, 0.02, 0.015, 0.001, 0.001, 0.001])
TF_TWO = 4.6404


def alfriend_two_impulse():
    dv1, dv2 = two_impulse(1.0, X0_ALFRIEND, np.zeros(3), TF_TWO)
    return Plan(1.0, X0_ALFRIEND, [(0.0, dv1), (TF_TWO, dv2)], np.zeros(6), TF_TWO)


def test_cost_is_the_actuator_norm_of_the_impulses():
    plan = alfriend_two_impulse()
    # Sum of per-axis magnitudes for box thrusters, of vector magnitudes for one engine
    # (independent computation); they differ by 30%.
    assert certify(plan, "l1").cost == pytest.approx(0.046425488331972686, rel=1e-12)
    assert certify(plan, "l2").cost == pytest.approx(0.03572017826320445, rel=1e-12)


@pytest.mark.parametrize("norm,peak_value", [("l1", 5.592851507167275), ("l2", 1.9216888534910461)])
def test_best_two_impulse_plan_of_alfriend_example_5_2_is_not_optimal(norm, peak_value):
    # Six burning components fix all six multipliers, so the primer is unique; it equals the
    # burn direction at both burns and peaks at the value given, inside the transfer
    # (independent computation). A check at the burns alone would pass this plan.
    cert = certify(alfriend_two_impulse(), norm)
    assert not cert.optimal
    assert cert.reason == "primer_max"
    assert cert.primer_max >= peak_value * (1 - 1e-12)
    # Either optimum at this horizon is at most 0.0377879, the 1-norm (so also an upper bound
    # on the 2-norm) cost of a 2000-interval grid plan (independent computation).
    assert cert.lower <= 0.037787933549941956


def test_a_verified_ball_plan_beats_the_two_impulse_plan_the_certificate_rejects():
    view = AgentView(
        scenario_id="certificate-test",
        dt=TF_TWO / 40,
        max_steps=40,
        n=1.0,
        action_mode="accel",
        max_accel=10.0,
        thrust_limit="ball",
        goal={"hold_point": (0.0, 0.0, 0.0), "capture_radius": 1e-6, "capture_speed": 1e-6},
    )
    sol = FuelOracle().solve(view, X0_ALFRIEND, arrival_step=40)
    two = 0.03572017826320445
    # The oracle's 40-step plan costs 0.0339611, 4.9% below the two-impulse plan, and its
    # impulsive lower bound is 0.0318867 (independent computation).
    assert sol.upper <= 0.96 * two
    assert sol.impulsive_lower <= sol.finite_lower <= sol.upper


def alfriend_optimum():
    """Impulsive 1-norm optimum over n T = 5: y burns at 0 and at an interior peak of |p_y|, one
    z burn at the node pi - atan(15), an x burn at T (independent computation: a 2000-interval
    grid LP refined by Newton on Lawden's conditions)."""
    impulses = [
        (0.0, np.array([0.0, -0.01885662135425249, 0.0])),
        (math.pi - math.atan(15.0), np.array([0.0, 0.0, 1e-3 * math.sqrt(226.0)])),
        (3.2742233593582735, np.array([0.0, -0.0021433786457475, 0.0])),
        (5.0, np.array([-0.00152685064478429, 0.0, 0.0])),
    ]
    return Plan(1.0, X0_ALFRIEND, impulses, np.zeros(6), 5.0)


def test_multi_impulse_optimum_of_alfriend_example_5_2_certifies():
    cert = certify(alfriend_optimum(), "l1")
    assert_certified(cert)
    assert cert.cost == pytest.approx(0.0375601, rel=1e-5)
    # Four burning components leave two multipliers free; the certifying one makes |p_y| and
    # |p_z| stationary at the interior burns. Its primer at T is the velocity multiplier
    # (-1, 0.707, -0.976), of 2-norm 1.566: a 2-norm check would reject this optimum.
    assert np.allclose(cert.nu[3:], [-1.0, 0.70713, -0.97567], rtol=0.0, atol=1e-4)
    assert np.linalg.norm(cert.nu[3:]) == pytest.approx(1.566, abs=5e-4)
    # On the coasts H = -nu^T Phi(T - t) A x is constant, and continuous across the interior
    # burns because the primer is stationary there (spread 4e-14 of |H|, independent
    # computation).
    assert cert.hamiltonian_spread <= 1e-8


def test_ball_certificate_measures_the_primer_in_the_2_norm():
    # Burns along (cos 30, 0, sin 30) at 0 and (cos 150, 0, sin 150) at 1.8 from rest, n = 1: the
    # six fitted multipliers are unique and give a primer whose largest component never exceeds
    # cos 30 = 0.866 while its 2-norm reaches 1.1606 (independent computation).
    u1 = np.array([math.cos(math.radians(30)), 0.0, math.sin(math.radians(30))])
    u2 = np.array([math.cos(math.radians(150)), 0.0, math.sin(math.radians(150))])
    plan = plan_to(1.0, np.zeros(6), [(0.0, 1e-3 * u1), (1.8, 1e-3 * u2)], 1.8)
    cert = certify(plan, "l2")
    assert not cert.optimal
    assert cert.reason == "primer_max"
    assert cert.primer_max >= 1.1606146098375734 * (1 - 1e-12)
    for t, u in ((0.0, u1), (1.8, u2)):
        assert np.allclose(primer_at(1.0, cert.nu, 1.8, t), u, rtol=0.0, atol=1e-9)
    assert peak(1.0, np.asarray(cert.nu), 1.8, "l1") <= 0.8661


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_primer_peak_between_burns_is_found(norm):
    # Two +z burns w apart around t_m, n = 1: the fitted primer is A cos(t - t_m) with
    # A cos(w / 2) = 1, here A = 1 + 1e-6. It is exactly 1 at both burns and above 1 only on the
    # 2.8e-3 between them; t_m is placed off any uniform grid, where the largest of 1001
    # samples is 1 + 9.994e-7 (independent computation).
    t_m, a_peak = 0.6180339887498949, 1.0 + 1e-6
    w = 2 * math.acos(1 / a_peak)
    burn = np.array([0.0, 0.0, 1e-3])
    plan = plan_to(1.0, np.zeros(6), [(t_m - w / 2, burn), (t_m + w / 2, burn)], 1.0)
    cert = certify(plan, norm)
    assert not cert.optimal
    assert cert.reason == "primer_max"
    assert a_peak - 1e-12 <= cert.primer_max <= a_peak + 1e-9


def test_unique_primer_peaking_2e_8_above_1_between_two_burns_is_not_optimal():
    # The multiplier was solved for so that its primer has unit 2-norm at 0 and tf and an
    # interior maximum of 1 + 2e-8, and the burns point along it. Two one-engine burns fix all
    # six multipliers (the burn rows have condition number 134), so no other primer fits and
    # Lawden's conditions fail. Solving the six burn equations directly and refining every local
    # maximum of 20001 samples puts the peak at 1 + 1.99999e-8 at n t = 1.48946 (independent
    # computation). The excess is 20 times the 1e-9 tolerance, yet a cell bound that drops its
    # third-derivative term misses it and reports a peak of 1.
    n, tf = 0.12091963607008353, 33.29470181714913
    start = np.array(
        [
            114.46408693320411,
            -38.01398958868518,
            86.23542379672526,
            -2.138641925943051,
            8.409198614867128,
            6.397263537332022,
        ]
    )
    impulses = [
        (0.0, np.array([-0.035947240006155685, -0.08113470191017998, 0.04609724592517518])),
        (tf, np.array([0.03517826419830042, -0.01517959214602565, -0.058583869027872285])),
    ]
    target = np.array(
        [
            914.439057596794,
            -4320.4236748516605,
            -95.8711488728202,
            -43.58978589011124,
            -185.1524803148231,
            3.9238811432769447,
        ]
    )
    cert = certify(Plan(n, start, impulses, target, tf), "l2")
    assert not cert.optimal
    assert cert.reason == "primer_max"
    assert cert.primer_max >= 1 + 1.9999e-8
    assert cert.lower < cert.cost


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_three_close_burns_along_z_fail_on_direction(norm):
    # Three +z burns 0.01 apart, n = 1: p_z = A cos(t - phi) reaches 1 at most twice in less
    # than a period, so no primer is +1 at all three. The least-squares fit misses by
    # w^2 / 3 = 3.3e-5, well below 1e-3 but far above the 1e-9 the direction check allows.
    burn = np.array([0.0, 0.0, 1e-3])
    plan = plan_to(1.0, np.zeros(6), [(0.3, burn), (0.31, burn), (0.32, burn)], 1.0)
    cert = certify(plan, norm)
    assert not cert.optimal
    assert cert.reason == "direction"
    assert cert.lower < cert.cost


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_burn_against_the_primer_fails_the_certificate(norm):
    # n = 1, moving at vz = -1e-3 through the origin: one +1e-3 burn now stops it (the optimum).
    # The plan burns +2e-3 now and +1e-3 half a period later, three times that cost. No primer
    # is +1 at both burns, since p_z(pi) = -p_z(0); nu_vz = -1 keeps |p| <= 1 with |p| = 1 at
    # both burns but points against the second, so a magnitude check alone would pass it.
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1e-3])
    impulses = [(0.0, np.array([0.0, 0.0, 2e-3])), (math.pi, np.array([0.0, 0.0, 1e-3]))]
    cert = certify(Plan(1.0, start, impulses, np.zeros(6), math.pi), norm)
    assert not cert.optimal
    assert cert.reason == "direction"
    assert cert.cost == pytest.approx(3e-3, rel=1e-12)
    assert 0.0 <= cert.lower <= 1e-3 * (1 + 1e-12)


def test_invalid_impulsive_inputs_raise():
    plan = fehse("radial")
    with pytest.raises(ValueError):
        certificate.certify_impulsive(
            plan.n, plan.start, plan.impulses, plan.target, tf=plan.tf, norm="l3"
        )
    late = [(0.0, plan.impulses[0][1]), (1.5 * plan.tf, plan.impulses[1][1])]
    with pytest.raises(ValueError):
        certificate.certify_impulsive(plan.n, plan.start, late, plan.target, tf=plan.tf, norm="l1")


# 2000 m behind the target, 2.2 rad of a 1.1e-3 rad/s orbit: a radial burn of 1e200 m/s throws
# the chaser about 3e203 m along V-bar, so the miss and the size of the final state's terms both
# overflow to inf, and in "l2" so does the burn's norm.
OVERFLOW_START = np.array([0.0, -2000.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_plan_with_an_overflowing_miss_raises(norm):
    with pytest.raises(ValueError):
        certificate.certify_impulsive(
            1.1e-3, OVERFLOW_START, [(0.0, [1e200, 0.0, 0.0])], np.zeros(6), tf=2000.0, norm=norm
        )


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_plan_with_an_overflowing_later_burn_raises(norm):
    impulses = [(0.0, [0.1, 0.0, 0.0]), (1000.0, [1e200, 0.0, 0.0])]
    with pytest.raises(ValueError):
        certificate.certify_impulsive(
            1.1e-3, OVERFLOW_START, impulses, np.zeros(6), tf=2000.0, norm=norm
        )


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_plan_whose_cost_overflows_raises(norm):
    # Each component is finite, but 1.5e308 + 1.5e308 and its 2-norm exceed the float maximum.
    with pytest.raises(ValueError):
        certificate.certify_impulsive(
            1.1e-3,
            OVERFLOW_START,
            [(0.0, [1.5e308, 1.5e308, 0.0])],
            np.zeros(6),
            tf=2000.0,
            norm=norm,
        )


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_coast_whose_propagated_state_overflows_raises(norm):
    # A finite start 1e306 m above the target drifts back along V-bar by 6 (n tf - sin n tf) x0,
    # about 6.6e309 m after n tf = 1100 rad, past the float maximum of 1.8e308.
    start = np.array([1e306, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        certificate.certify_impulsive(1.1e-3, start, [], np.zeros(6), tf=1e6, norm=norm)


# A computed case, not a book example: half an orbit of a 6778 km circle from (-200, -1500, 50) m
# at rest to (0, -100, 0) m at rest, |a_i| <= 2e-3 m/s^2 (box) or |a|_2 <= 2e-3 m/s^2 (one
# engine). The step-grid optima are LPs and their equality multipliers give lower bounds by LP
# duality (Boyd and Vandenberghe, Convex Optimization, 2004, sections 4.3 and 5.1.3).
N_HALF = mean_motion(6778e3)
T_HALF = math.pi / N_HALF
X0_HALF = np.array([-200.0, -1500.0, 50.0, 0.0, 0.0, 0.0])
GOAL_HALF = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
A_HALF = 2e-3
# Box ZOH optima on 160 and 640 steps, feasible plans and so upper bounds on the continuous
# optimum, and the equality multipliers of the 160-step LP (independent computation).
P160 = 0.8949323268656184
P640 = 0.8947292762009076
NU160 = np.array(
    [
        3.5524098253677471e-03,
        5.5370862201840898e-04,
        1.1314736570322344e-03,
        -1.2533601856102836e00,
        1.0196633907830843e00,
        0.0,
    ]
)
# Continuous box optimum by the switching control of the converged multiplier: four burn arcs,
# terminal miss 4e-14 of |d|, equal to D(nu) - a E(nu) to 1e-13 (independent computation).
J_BOX = 0.8947124818260699
# One engine: outer-LP multipliers of a 40-step, 400-direction polyhedral problem, the exact
# 2-norm cost of a 40-step plan with |u_k|_2 <= a, and the continuous optimum by the switching
# control a p / |p|_2 (four arcs, miss 2e-12 of |d|, independent computation).
NU_BALL = np.array(
    [
        0.0019899957123063593,
        0.0005028098178603568,
        0.0010684558619793439,
        -1.2080758403822702,
        0.28813086168409263,
        0.03691134065725177,
    ]
)
UB_BALL = 0.8324776
J_BALL = 0.8289720621930146
# Velocities over n, so the miss compares metres with metres.
HILL_SCALE = np.r_[np.ones(3), np.full(3, 1 / N_HALF)]


def half_view(layout):
    return AgentView(
        scenario_id="certificate-test",
        dt=T_HALF,
        max_steps=1,
        n=N_HALF,
        action_mode="accel",
        max_accel=A_HALF,
        thrust_limit=layout,
        goal={"hold_point": tuple(GOAL_HALF[:3]), "capture_radius": 0.0, "capture_speed": 0.0},
    )


def crossings(value, tf):
    """Intervals of [0, tf] where value(t) > 1, ends found by Brent on the sampled sign changes."""
    t = np.linspace(0.0, tf, 20001)
    f = value(t) - 1.0
    edges = [
        brentq(lambda s: value(np.array([s]))[0] - 1.0, t[k], t[k + 1], xtol=1e-14 * tf)
        for k in np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:]))
    ]
    if f[0] > 0:
        edges.insert(0, 0.0)
    if f[-1] > 0:
        edges.append(tf)
    return list(zip(edges[::2], edges[1::2], strict=True))


def box_arcs(nu):
    """(t0, t1, axis, sign) of the box switching control a sign(p_i) where |p_i| > 1."""
    arcs = []
    for i in range(3):
        for t0, t1 in crossings(
            lambda t, i=i: np.abs(primer_at(N_HALF, nu, T_HALF, t)[..., i]), T_HALF
        ):
            sign = np.sign(primer_at(N_HALF, nu, T_HALF, 0.5 * (t0 + t1))[i])
            arcs.append((t0, t1, i, sign))
    return arcs


def box_samples(nu):
    """Times, states and accelerations inside each constant piece of the switching control,
    propagated exactly between switches; and the final state."""
    arcs = box_arcs(nu)
    cuts = sorted({0.0, T_HALF, *(t for arc in arcs for t in arc[:2])})
    x = X0_HALF.copy()
    times, states, accels = [], [], []
    for t0, t1 in zip(cuts[:-1], cuts[1:], strict=True):
        u = np.zeros(3)
        for s0, s1, i, sign in arcs:
            if s0 <= 0.5 * (t0 + t1) <= s1:
                u[i] = A_HALF * sign
        for s in np.linspace(t0, t1, 5)[1:-1]:
            phi, gamma = cw_discrete(N_HALF, s - t0)
            times.append(s)
            states.append(phi @ x + gamma @ u)
            accels.append(u)
        phi, gamma = cw_discrete(N_HALF, t1 - t0)
        x = phi @ x + gamma @ u
    return np.array(times), np.array(states), np.array(accels), x, arcs


@functools.cache
def box_indirect():
    cost, nu = certificate.indirect_optimum(
        N_HALF, X0_HALF, GOAL_HALF, T_HALF, accel=A_HALF, norm="l1", nu0=NU160
    )
    return cost, np.asarray(nu, dtype=float)


def miss_ratio(x_final):
    d = GOAL_HALF - cw_discrete(N_HALF, T_HALF)[0] @ X0_HALF
    return np.linalg.norm((x_final - GOAL_HALF) * HILL_SCALE) / np.linalg.norm(d * HILL_SCALE)


def test_indirect_box_optimum_agrees_with_the_direct_lp_optima():
    cost, nu = box_indirect()
    # Held steps cost more than the continuous optimum, by 2.457e-4 at 160 steps and 1.877e-5
    # at 640, 13 times less for steps 4 times shorter, near second order (independent
    # computation). The tolerances cover the 1e-7 allowed on the cost below.
    assert (P160 - cost) / P160 == pytest.approx(2.457e-4, rel=2e-3)
    assert (P640 - cost) / P640 == pytest.approx(1.877e-5, rel=1e-2)
    assert cost == pytest.approx(J_BOX, rel=1e-7)
    *_, x_final, arcs = box_samples(nu)
    assert miss_ratio(x_final) <= 1e-6
    # The cost is the 1-norm of the flown acceleration: a times the burn time summed over axes.
    assert cost == pytest.approx(A_HALF * sum(t1 - t0 for t0, t1, _, _ in arcs), rel=1e-9)
    # Any multiplier gives a certified lower bound; at the optimum the cell rule at 20000 cells
    # sits 9.7e-4 below the cost (independent computation).
    view = half_view("box")
    lower = dual_bound(N_HALF, view, X0_HALF, nu[None], [T_HALF], norm="l1", cells=20_000)
    assert cost * (1 - 2e-3) <= lower <= cost * (1 + 1e-12)


def test_indirect_ball_optimum_lies_inside_the_lp_bracket():
    cost, nu = certificate.indirect_optimum(
        N_HALF, X0_HALF, GOAL_HALF, T_HALF, accel=A_HALF, norm="l2", nu0=NU_BALL
    )
    nu = np.asarray(nu, dtype=float)
    assert cost == pytest.approx(J_BALL, rel=1e-6)
    # The cell rule at 20000 cells sits 1.6e-3 below the cost (independent computation).
    view = half_view("ball")
    lower = dual_bound(N_HALF, view, X0_HALF, nu[None], [T_HALF], norm="l2", cells=20_000)
    assert cost * (1 - 3e-3) <= lower <= cost * (1 + 1e-12)
    assert cost <= UB_BALL
    # Fly the switching control a p / |p|_2 where |p|_2 > 1; its cost is a times the burn time.
    arcs = crossings(lambda t: np.linalg.norm(primer_at(N_HALF, nu, T_HALF, t), axis=-1), T_HALF)

    def reach(s):
        p = primer_at(N_HALF, nu, T_HALF, s)
        return cw_discrete(N_HALF, T_HALF - s)[0][:, 3:6] @ (A_HALF * p / np.linalg.norm(p))

    x_final = cw_discrete(N_HALF, T_HALF)[0] @ X0_HALF
    for t0, t1 in arcs:
        x_final = x_final + quad_vec(reach, t0, t1, epsabs=1e-14, epsrel=1e-12)[0]
    assert miss_ratio(x_final) <= 1e-6
    assert cost == pytest.approx(A_HALF * sum(t1 - t0 for t0, t1 in arcs), rel=1e-9)


def test_hamiltonian_is_constant_along_the_indirect_optimum():
    _, nu = box_indirect()
    times, states, accels, *_ = box_samples(nu)
    a_mat, b_mat = cw_system(N_HALF)
    phi = cw_discrete(N_HALF, T_HALF - times)[0]
    costate = -np.einsum("i,kij->kj", nu, phi)
    terms = np.stack(
        [
            np.abs(accels).sum(axis=1),
            np.einsum("kj,ji,ki->k", costate, a_mat, states),
            np.einsum("kj,ji,ki->k", costate, b_mat, accels),
        ]
    )
    largest = np.abs(terms).max()
    h = certificate.hamiltonian(N_HALF, nu, T_HALF, times, states, accels, norm="l1")
    assert np.allclose(h, terms.sum(axis=0), rtol=0.0, atol=1e-12 * largest)
    # Between switches the two time derivatives cancel, and the switching terms vanish where
    # |p_i| = 1; the spread is at rounding level, 4e-15 of the largest term.
    assert np.ptp(h) <= 1e-8 * largest
    # With the costate sign flipped H becomes 2 |a|_1 - H, so the bang-off-bang steps in |a|_1
    # no longer cancel: spread 1.7715 times the largest term (independent computation).
    flipped = certificate.hamiltonian(N_HALF, -nu, T_HALF, times, states, accels, norm="l1")
    assert np.ptp(flipped) / largest == pytest.approx(1.7715, abs=1e-3)


def test_invalid_indirect_inputs_raise():
    with pytest.raises(ValueError):
        certificate.indirect_optimum(
            N_HALF, X0_HALF, GOAL_HALF, T_HALF, accel=A_HALF, norm="l3", nu0=NU160
        )
    with pytest.raises(ValueError):
        certificate.indirect_optimum(
            N_HALF, X0_HALF, GOAL_HALF, T_HALF, accel=0.0, norm="l1", nu0=NU160
        )


@pytest.mark.parametrize("n", [0.0, -1.1e-3])
def test_hamiltonian_rejects_a_mean_motion_that_is_not_positive(n):
    with pytest.raises(ValueError):
        certificate.hamiltonian(
            n, np.ones(6), 2000.0, [0.0, 1000.0], np.ones((2, 6)), np.ones((2, 3)), norm="l1"
        )


@pytest.mark.parametrize(
    "name,index,bad",
    [
        ("tf", None, np.nan),
        ("tf", None, np.inf),
        ("nu", 2, np.nan),
        ("t", 1, np.nan),
        ("t", 1, np.inf),
        ("x", (1, 4), np.nan),
        ("a", (0, 2), np.inf),
    ],
)
def test_hamiltonian_rejects_inputs_that_are_not_finite(name, index, bad):
    args = {
        "nu": np.ones(6),
        "tf": 2000.0,
        "t": np.array([0.0, 1000.0]),
        "x": np.ones((2, 6)),
        "a": np.ones((2, 3)),
    }
    if index is None:
        args[name] = bad
    else:
        args[name][index] = bad
    with pytest.raises(ValueError):
        certificate.hamiltonian(1.1e-3, **args, norm="l1")


def test_indirect_solve_raises_when_the_target_is_out_of_reach():
    # n = 1 over one radian at 1e-3 m/s^2 per axis: every entry of the CW velocity-to-position
    # block is at most tau for n tau <= 1, so no thrust history moves the chaser more than
    # 1.5e-3 m along any axis, far short of the 1 m asked for.
    with pytest.raises(RuntimeError):
        certificate.indirect_optimum(
            1.0, np.zeros(6), [1.0, 0, 0, 0, 0, 0], 1.0, accel=1e-3, norm="l1", nu0=np.zeros(6)
        )
