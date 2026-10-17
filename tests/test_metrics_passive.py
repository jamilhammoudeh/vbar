import math

import numpy as np
import pytest
from metric_cases import ConstantAgent, episode, pd_episode
from scipy.optimize import brentq, minimize_scalar

from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.metrics import (
    MetricSettings,
    TravellingEllipse,
    cam_impulses,
    escape_arc,
    passive_forever,
    passive_metrics,
)

N = mean_motion(EARTH.radius + 500e3)
N400 = mean_motion(EARTH.radius + 400e3)
T400 = 2 * np.pi / N400


def dense_drift_minimum(start, n, horizon, radius, accel=(0, 0, 0), burn=0.0, kick=(0, 0, 0)):
    """Least range minus radius over an escape arc: 20001 closed-form samples, then Brent."""
    start, accel = np.asarray(start, float), np.asarray(accel, float)

    def rho(t):
        return np.linalg.norm(
            escape_arc(n, start, accel, burn, kick, np.atleast_1d(t))[:, :3], axis=1
        )

    times = np.linspace(0.0, burn + horizon, 20001)
    ranges = rho(times)
    k = int(np.argmin(ranges))
    window = (times[max(k - 1, 0)], times[min(k + 1, len(times) - 1)])
    found = minimize_scalar(
        lambda t: rho(t)[0], bounds=window, method="bounded", options={"xatol": 1e-9}
    )
    return min(found.fun, ranges[k]) - radius


def test_escape_arc_reproduces_fehse_rbar_cam_with_a_retrograde_part():
    # Fehse figs 4.19 to 4.20: 15 m below, 0.5 m/s away and 0.10 m/s retrograde.
    states = escape_arc(N400, [-15.0, 0, 0, 0, 0, 0], np.zeros(3), 0.0, [-0.5, -0.1, 0.0], [T400])
    assert states[0, 1] == pytest.approx(2231.6, abs=0.05)


@pytest.mark.parametrize(("depth", "advance"), [(20.0, 753.98), (50.0, 1884.96), (100.0, 3769.91)])
def test_radial_cam_advance_after_one_orbit_is_twelve_pi_times_the_depth(depth, advance):
    # Fehse figs 4.17 to 4.18: independent of the impulse size.
    for dv in (0.3, 0.7):
        states = escape_arc(N400, [-depth, 0, 0, 0, 0, 0], np.zeros(3), 0.0, [-dv, 0, 0], [T400])
        assert states[0, 1] == pytest.approx(advance, abs=0.01)


def test_minus_vbar_cam_passes_below_the_target():
    # Fehse 4.5, fig 4.16: from 20 m behind, 0.3 m/s further back. It passes under the target at
    # x = -396.220 m, the root of y = 0 at 400 km; with Fehse's r = 6766 km it is -395.209 m.
    def at(t):
        return escape_arc(N400, [0, -20.0, 0, 0, 0, 0], np.zeros(3), 0.0, [0, -0.3, 0], [t])[0]

    crossing = brentq(lambda t: at(t)[1], 1000.0, 1300.0, xtol=1e-12)
    assert at(crossing)[0] == pytest.approx(-396.220, abs=1e-3)
    assert at(T400)[1] == pytest.approx(4978.3, abs=0.05)


def test_travelling_ellipse_matches_propagation_with_the_corrected_fehse_sign():
    # Fehse appendix A.1.5 case in Hill axes: x_F = 10, z_F = 20, xd_F = 0.03,
    # zd_F = 0.05 is x = -20, y = 10, vx = -0.05, vy = 0.03.
    state = np.array([-20.0, 10.0, 0.0, -0.05, 0.03, 0.0])
    ellipse = TravellingEllipse.from_state(state, N400)
    for t in (300.0, 1500.0, 4000.0):
        exact = (cw_discrete(N400, t)[0] @ state)[:3]
        assert np.allclose(ellipse.position(t), exact, atol=1e-9)
    # Fehse's printed A.69, x_F = x_c + 2 b cos(nt + phi), is 176 m off at 300 s.
    x0, z0, xd0, zd0 = 10.0, 20.0, 0.03, 0.05
    zc = 4 * z0 - 2 * xd0 / N400
    b = math.hypot(z0 - zc, zd0 / N400)
    phi = math.atan2(z0 - zc, zd0 / N400)
    xc = x0 + 2 * zd0 / N400 + 1.5 * zc * N400 * 300.0
    exact_along = (cw_discrete(N400, 300.0)[0] @ state)[1]
    assert xc - 2 * b * math.cos(N400 * 300.0 + phi) == pytest.approx(exact_along, abs=1e-9)
    assert abs(xc + 2 * b * math.cos(N400 * 300.0 + phi) - exact_along) > 170.0
    assert ellipse.semi_axis == pytest.approx(b, rel=1e-12)
    assert ellipse.drift_rate == pytest.approx(1.5 * zc * N400, rel=1e-12)


def test_a_drift_line_through_the_target_is_not_safe_forever_and_says_when():
    # 10 m below on a parallel orbit drifts forward at 1.5 n 10 m/s with no oscillation.
    state = np.array([-10.0, -1000.0, 0.0, 0.0, 15.0 * N, 0.0])
    verdict = passive_forever(state, N, 50.0)
    entry = (1000.0 - math.sqrt(50.0**2 - 10.0**2)) / (15.0 * N)
    assert not verdict.safe
    assert verdict.time_to_threat == pytest.approx(entry, rel=2e-3)


def test_the_same_line_drifting_away_is_safe_forever():
    state = np.array([10.0, -1000.0, 0.0, 0.0, -15.0 * N, 0.0])
    assert passive_forever(state, N, 50.0).safe


def test_a_relative_orbit_whose_tube_clears_the_target_is_safe_forever():
    # Radial and cross-track oscillations a quarter period apart never pass through zero together.
    state = np.array([0.0, -200.0, 80.0, 60.0 * N, 0.0, 0.0])
    ellipse = TravellingEllipse.from_state(state, N)
    assert ellipse.drift_rate == 0.0
    assert ellipse.cross_section_clearance() == pytest.approx(60.0, rel=1e-6)
    assert passive_forever(state, N, 50.0).safe


def test_rest_on_the_vbar_is_safe_with_its_range_as_the_margin():
    ep = episode(start_state=(0.0, -100.0, 0.0, 0.0, 0.0, 0.0))
    m = passive_metrics(ep)
    assert m["passive_safety_margin"] == pytest.approx(50.0, abs=1e-9)
    assert m["passive_safe_fraction"] == 1.0 and m["passive_forever_fraction"] == 1.0
    assert m["passive_threat_time"] == math.inf
    assert m["unprotected_time"] == 0.0


def test_passive_margin_matches_a_dense_closed_form_minimum_over_two_orbits():
    ep = episode()
    horizon = 4 * np.pi / N
    expected = min(dense_drift_minimum(x, N, horizon, 50.0) for x in ep.states)
    m = passive_metrics(ep)
    assert m["passive_safety_margin"] == pytest.approx(expected, abs=1e-3)
    unsafe = [dense_drift_minimum(x, N, horizon, 50.0) < 0.0 for x in ep.states]
    assert m["passive_safe_fraction"] == pytest.approx(1.0 - np.mean(unsafe))


def test_passive_metrics_do_not_depend_on_drift_sampling():
    ep = pd_episode(max_steps=40)
    coarse = passive_metrics(ep, MetricSettings(drift_samples_per_orbit=50))
    fine = passive_metrics(ep, MetricSettings(drift_samples_per_orbit=800))
    for name, value in coarse.items():
        assert fine[name] == pytest.approx(value, abs=1e-3)


def test_states_inside_the_exclusion_zone_are_not_scored():
    ep = episode(start_state=(0.0, -100.0, 0.0, 0.0, 0.0, 0.0))
    m = passive_metrics(ep, MetricSettings(passive_exclusion=200.0, commit_radius=200.0))
    assert math.isnan(m["passive_safety_margin"]) and math.isnan(m["cam_margin"])
    assert m["unprotected_time"] == 0.0


def test_a_stuck_thruster_from_the_hold_point_matches_the_closed_form():
    # Holding at rest while the last command, full thrust toward the target, stays on for 30 s.
    ep = episode(
        ConstantAgent([0.0, 1.0, 0.0]), start_state=(0.0, -300.0, 0.0, 0, 0, 0), max_steps=1
    )
    horizon = 4 * np.pi / N
    end = ep.states[-1]
    stuck = dense_drift_minimum(end, N, horizon, 50.0, accel=(0.0, 0.02, 0.0), burn=30.0)
    m = passive_metrics(ep, MetricSettings(partial_burn_samples=0))
    assert m["stuck_thruster_margin"] == pytest.approx(stuck, abs=1e-3)
    assert m["stuck_thruster_margin"] < m["passive_safety_margin"]


def test_partial_burns_are_checked_inside_the_step():
    # One full-thrust step toward the target from 300 m: a burn cut at any point drifts less far
    # than the full burn, so the full burn sets the margin; the partial points can only lower it.
    ep = episode(
        ConstantAgent([0.0, 1.0, 0.0]), start_state=(0.0, -300.0, 0.0, 0, 0, 0), max_steps=1
    )
    without = passive_metrics(ep, MetricSettings(partial_burn_samples=0))["stuck_thruster_margin"]
    with_points = passive_metrics(ep, MetricSettings(partial_burn_samples=6))[
        "stuck_thruster_margin"
    ]
    assert with_points <= without


def test_default_cam_is_opposite_a_vbar_approach_and_radially_away_on_rbar():
    settings = MetricSettings(cam_dv=0.5)
    vbar = episode(start_state=(0.0, -300.0, 0.0, 0, 0, 0), max_steps=1)
    assert np.allclose(cam_impulses(vbar, settings), [[0.0, -0.5, 0.0]])
    rbar = episode(start_state=(-300.0, 0.0, 0.0, 0, 0, 0), max_steps=1, hold_point=(-100.0, 0, 0))
    # Fehse 5.7: away radially with a fifth of it retrograde.
    assert np.allclose(
        cam_impulses(rbar, settings), [[-0.5 * 5 / math.sqrt(26), -0.5 / math.sqrt(26), 0.0]]
    )


def test_cam_margin_matches_the_closed_form_and_unprotected_time_counts_bad_steps():
    ep = episode()
    kick = np.array([0.0, -0.3, 0.0])
    horizon = 4 * np.pi / N
    per_state = [dense_drift_minimum(x, N, horizon, 50.0, kick=kick) for x in ep.states]
    passive = [dense_drift_minimum(x, N, horizon, 50.0) for x in ep.states]
    m = passive_metrics(ep)
    assert m["cam_margin"] == pytest.approx(min(per_state), abs=1e-3)
    exposed = [(c < 0) and (p < 0) for c, p in zip(per_state[:-1], passive[:-1], strict=True)]
    assert m["unprotected_time"] == pytest.approx(10.0 * sum(exposed))


def test_passive_metrics_are_unchanged_by_mirroring_the_orbit_normal():
    start = np.array([20.0, 300.0, 30.0, 0.01, -0.6, -0.03])
    a = passive_metrics(episode(start_state=tuple(start)))
    b = passive_metrics(episode(start_state=tuple(start * [1, 1, -1, 1, 1, -1])))
    for name, value in a.items():
        assert b[name] == pytest.approx(value, abs=1e-6)


def test_the_two_body_model_is_used_for_two_body_episodes():
    # Coasting from 150 m behind at 0.2 m/s; over 1000 s the two-body drift differs from
    # Clohessy-Wiltshire by the rho^2 / r nonlinearity, about 1e-3 m here.
    start = (0.0, -150.0, 5.0, 0.0, 0.2, 0.0)
    settings = MetricSettings(drift_horizon=1000.0, cam_horizon=1000.0, partial_burn_samples=0)
    cw = passive_metrics(episode(start_state=start, max_steps=1), settings)
    tb = passive_metrics(episode(start_state=start, max_steps=1, dynamics="twobody"), settings)
    for name in ("passive_safety_margin", "stuck_thruster_margin", "cam_margin"):
        assert tb[name] == pytest.approx(cw[name], abs=1e-2)
        assert tb[name] != cw[name]
    assert math.isnan(tb["passive_forever_fraction"])


def test_arcs_whose_sign_the_samples_cannot_settle_are_refined():
    # 4 m/s through the target, with only 8 drift samples per orbit: the samples miss the pass.
    start = (0.0, 400.0, 0.0, 0.0, -4.0, 0.0)
    ep = episode(start_state=start, max_steps=1)
    coarse = passive_metrics(ep, MetricSettings(drift_samples_per_orbit=8, partial_burn_samples=0))
    horizon = 4 * np.pi / N
    expected = min(dense_drift_minimum(x, N, horizon, 50.0) for x in ep.states)
    assert coarse["passive_safety_margin"] == pytest.approx(expected, abs=1e-3)
    assert coarse["passive_safe_fraction"] == 0.0


def test_a_receding_ellipse_within_twice_its_semi_axis_can_still_pass_the_target():
    # Its centre moves away and is farther than b + R along-track, but the along-track swing is
    # 2 b, so the passage must still be checked; this one comes within 16 m.
    state = np.array([17.5165, 343.5107, 0.0, 0.1038, -0.0409, 0.0])
    ellipse = TravellingEllipse.from_state(state, N)
    b = ellipse.semi_axis
    assert ellipse.center_y * ellipse.drift_rate > 0.0
    assert b + 50.0 < abs(ellipse.center_y) < 2 * b + 50.0
    phi, _ = cw_discrete(N, np.linspace(0.0, 6 * np.pi / N, 30001))
    assert np.linalg.norm((phi @ state)[:, :3], axis=1).min() < 20.0
    assert not passive_forever(state, N, 50.0).safe


def test_a_burn_cut_short_midway_can_be_the_worst_case():
    # Both ends of this one-step burn drift clear by 196 m; a cut inside it does not.
    start = (5.55, -258.95, 0.0, 0.1448, -0.0668, 0.0)
    ep = episode(ConstantAgent([-0.239, 0.819, 0.0]), start_state=start, max_steps=1)
    ends = MetricSettings(partial_burn_samples=0, stuck_time=0.0)
    inside = MetricSettings(partial_burn_samples=8, stuck_time=0.0)
    without = passive_metrics(ep, ends)["stuck_thruster_margin"]
    with_points = passive_metrics(ep, inside)["stuck_thruster_margin"]
    accel = np.array([-0.239, 0.819, 0.0]) * 0.02
    times = np.linspace(0.0, 10.0, 201)
    cuts = escape_arc(N, np.array(start), accel, 10.0, np.zeros(3), times)
    lowest = min(dense_drift_minimum(x, N, 4 * np.pi / N, 50.0) for x in cuts)
    assert with_points < without - 100.0
    assert with_points >= lowest - 1e-3


@pytest.mark.parametrize(("swept", "widest"), [(90, 7.00), (120, 6.20), (150, 4.15), (180, 2.00)])
def test_thrust_loss_on_the_twice_orbital_rate_circle_never_comes_closer_than_its_radius(
    swept, widest
):
    # Goodman p. 137: the circle of radius R flown clockwise at 2n from below the target needs only
    # the radial hover thrust, and lies on the no-drift manifold vy + 2 n x = 0.
    n, radius = mean_motion(EARTH.radius + 185.0 * 1852.0), 100.0
    phi = np.pi - np.radians(swept)
    start = radius * np.array(
        [np.cos(phi), np.sin(phi), 0.0, 2 * n * np.sin(phi), -2 * n * np.cos(phi), 0.0]
    )
    assert start[4] + 2 * n * start[0] == pytest.approx(0.0, abs=1e-15)
    states = escape_arc(
        n, start, np.zeros(3), 0.0, np.zeros(3), np.linspace(0, 2 * np.pi / n, 20001)
    )
    ranges = np.linalg.norm(states[:, :3], axis=1) / radius
    assert ranges.min() == pytest.approx(1.0, abs=1e-3)
    assert ranges.max() == pytest.approx(widest, abs=1e-2)
    assert passive_forever(start, n, 0.999 * radius).safe
    assert not passive_forever(start, n, 1.001 * radius).safe
