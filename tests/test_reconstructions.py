"""Mission reconstructions: their starts and faults, and what reference controllers do on them.

The whole-episode claims are behind the `slow` marker; they fly validation seeds.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from scenario_references import Differenced, Glide

from vbar.dynamics import cw_stm
from vbar.scenarios import make, seeds

NAMES = ("recon-progress-m34", "recon-ets7", "recon-dart")


class Drift:
    name = "drift"

    def reset(self, view, rng=None):
        pass

    def act(self, obs, info):
        return np.zeros(3)


def fly(name, agent, seed):
    """Success, closest approach (m) and whether the keep-out sphere was entered."""
    env = make(name)
    obs, info = env.reset(seed=seed)
    agent.reset(env.agent_view())
    closest = float(np.linalg.norm(info["true_state"][:3]))
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(agent.act(obs, env.agent_info()))
        closest = min(closest, float(np.linalg.norm(info["true_state"][:3])))
    return env._success(), closest, info["min_margins"]["keep_out"] < 0.0


@pytest.mark.parametrize("name", [*NAMES, *(n + ":nominal" for n in NAMES)])
def test_reconstructions_follow_the_gymnasium_api(name):
    check_env(make(name), skip_render_check=True)


def test_progress_start_is_the_reported_range_and_closing_rate_on_an_arc_through_the_aim_point():
    env = make("recon-progress-m34")
    for seed in seeds("recon-progress-m34", "validation", 20):
        _, info = env.reset(seed=seed)
        x0, draw = info["true_state"], env.draw
        rho = np.linalg.norm(x0[:3])
        assert 6000.0 <= rho <= 7000.0 and 300.0 <= abs(x0[2]) <= 500.0
        assert x0[0] > 0.0 and x0[1] < 0.0
        scatter = np.asarray(draw["velocity_scatter"])
        on_arc = np.concatenate([x0[:3], x0[3:] - scatter])
        assert -(on_arc[:3] @ on_arc[3:]) / rho == pytest.approx(draw["closure_speed"], rel=1e-9)
        assert 5.0 <= draw["closure_speed"] <= 6.5
        aim = cw_stm(env.n, env.coast_s) @ on_arc
        expected = np.array([0.0, -1000.0, 0.0]) + np.asarray(draw["aim_offset"])
        expected[2] = x0[2]
        # The coast solves a 3x3 system; 1e-6 m is far above its rounding at 7 km.
        np.testing.assert_allclose(aim[:3], expected, atol=1e-6)
        assert 600.0 < env.coast_s < 1800.0


def test_progress_start_is_not_solved_from_the_spec_and_one_reading():
    # The attack that worked on a fixed 400 m offset: read the range off the out-of-plane
    # component of the line of sight, then fit the arc through the nominal aim point to the
    # line-of-sight rate. With the start spread out it misses by a large fraction of the range
    # and by metres per second.
    env = make("recon-progress-m34")
    misses = []
    for seed in seeds("recon-progress-m34", "validation", 10):
        obs, info = env.reset(seed=seed)
        u = obs[:3] / 6500.0
        rho = 400.0 / abs(u[2])
        p0 = rho * u
        aim = np.array([0.0, -1000.0, np.copysign(400.0, u[2])])

        def velocity(coast, p0=p0, aim=aim):
            phi = cw_stm(env.n, coast)
            return np.linalg.solve(phi[:3, 3:], aim - phi[:3, :3] @ p0)

        across = obs[3:6] * rho / 6500.0
        coasts = np.linspace(10.0, 0.45 * 2.0 * np.pi / env.n, 2000)
        fit = [np.linalg.norm(velocity(c) - (velocity(c) @ u) * u - across) for c in coasts]
        v0 = velocity(coasts[int(np.argmin(fit))])
        truth = info["true_state"]
        misses.append(
            (
                np.linalg.norm(p0 - truth[:3]) / np.linalg.norm(truth[:3]),
                np.linalg.norm(v0 - truth[3:]),
            )
        )
    misses = np.array(misses)
    assert np.sum(misses[:, 0] > 0.02) >= 8 and np.median(misses[:, 1]) > 0.1


def test_progress_reading_carries_no_range():
    env = make("recon-progress-m34")
    obs, info = env.reset(seed=1)
    truth = info["true_state"]
    assert np.linalg.norm(obs[:3]) == pytest.approx(6500.0, rel=1e-12)
    np.testing.assert_allclose(obs[:3] / 6500.0, truth[:3] / np.linalg.norm(truth[:3]), rtol=1e-12)
    # The range rate is gone from the velocity reading.
    assert abs(obs[:3] @ obs[3:]) < 1e-9 * 6500.0


def test_dart_bias_lasts_until_the_waypoint_sphere_is_entered():
    env = make("recon-dart", training=True)
    obs, info = env.reset(seed=3, options={"start_state": [0.0, -210.0, 0.0, 0.0, 0.5, 0.0]})
    assert info["faults_active"] == ("velocity_bias",)
    assert 0.4 <= np.linalg.norm(obs[3:] - info["true_state"][3:]) <= 0.8
    # 10 s at 0.5 m/s carries the chaser 5 m along the V-bar, into the 6.3 m sphere at -200 m.
    obs, _, _, _, info = env.step(np.zeros(3))
    assert info["faults_active"] == ()
    np.testing.assert_array_equal(obs, info["true_state"])


def test_dart_bias_stays_when_the_sphere_is_missed():
    env = make("recon-dart", training=True)
    _, info = env.reset(seed=3, options={"start_state": [7.0, -210.0, 0.0, 0.0, 0.5, 0.0]})
    *_, info = env.step(np.zeros(3))
    assert info["faults_active"] == ("velocity_bias",)


def test_ets7_fault_takes_the_braking_side_only():
    env = make("recon-ets7")
    env.reset(seed=2)
    fault = env.faults[0]
    fault.active = True
    braking = fault.on_command(0.0, np.array([0.1, 0.5, -0.2]))
    closing = fault.on_command(0.0, np.array([0.1, -0.5, 0.0]))
    np.testing.assert_array_equal(braking, [0.1, 0.0, -0.2])
    np.testing.assert_array_equal(closing, [0.1, -0.5, 0.0])


def progress_reference():
    return Glide([(0.0, -200.0, 0.0)], v_max=6.0, gain=0.005, k_v=0.05, tol=10.0)


def dart_reference():
    return Glide([(0.0, -15.24, 0.0)], v_max=0.5, gain=0.005, k_v=0.05, tol=2.0)


def ets7_reference():
    return Glide([(0.0, 2.0, 0.0)], v_max=0.1, gain=0.001, k_v=0.02, wn=0.008, tol=0.2)


@pytest.mark.slow
def test_progress_free_drift_never_hits_the_station():
    for seed in seeds("recon-progress-m34", "validation", 20):
        _, closest, hit = fly("recon-progress-m34", Drift(), seed)
        assert not hit and closest > 100.0


@pytest.mark.slow
def test_progress_reference_holds_with_range_and_mostly_hits_without_it():
    split = list(seeds("recon-progress-m34", "validation", 20))
    assert all(fly("recon-progress-m34:nominal", progress_reference(), s)[0] for s in split)
    hits = [fly("recon-progress-m34", progress_reference(), s)[2] for s in split]
    assert sum(hits) >= len(split) // 2


@pytest.mark.slow
def test_dart_bias_defeats_a_controller_that_trusts_the_velocity_reading():
    split = list(seeds("recon-dart", "validation", 10))
    assert all(fly("recon-dart:nominal", dart_reference(), s)[0] for s in split)
    assert not any(fly("recon-dart", dart_reference(), s)[0] for s in split)
    assert all(fly("recon-dart", Differenced(dart_reference()), s)[0] for s in split)


@pytest.mark.slow
def test_ets7_slow_approach_holds_with_the_fault_at_less_margin():
    for seed in seeds("recon-ets7", "validation", 5):
        ok, clean, _ = fly("recon-ets7:nominal", ets7_reference(), seed)
        ok_faulty, faulty, _ = fly("recon-ets7", ets7_reference(), seed)
        assert ok and ok_faulty and faulty < clean
