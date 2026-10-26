"""Comparison tasks rebuilt from published benchmarks: every number checked against its paper.

AFRL docking: Hamilton, Dunlap and Hobbs, "Investigating the Impact of Choice on Deep
Reinforcement Learning for Space Controls", 2024, arXiv 2405.12355, section III.B. AFRL inspection:
van Wijk, Dunlap, Majji and Hobbs, "Deep Reinforcement Learning for Autonomous Spacecraft
Inspection using Illumination", AAS 23-342, 2023, arXiv 2308.02743. SpaceGym: Supreeth and Mehta,
"SpaceGym: A Gymnasium-Based Benchmark for Deep Reinforcement Learning in Spacecraft Proximity
Operations", CVPR 2026 Workshops, sections 3.1 and 3.2.
"""

import math

import numpy as np
import pytest

from vbar import run_episode
from vbar.agents import DriftAgent
from vbar.dynamics import MU_EARTH, R_EARTH, cw_discrete
from vbar.scenarios import SCENARIOS, make, seeds
from vbar.scenarios.targets import sun_at

AFRL_N = 0.001027


def test_the_afrl_docking_task_has_its_papers_constants():
    env = make("afrl-docking")
    cfg, p = env.cfg, env.params
    assert env.n == pytest.approx(AFRL_N, rel=1e-12)
    # 12 kg with 1 N per axis, 1 s steps for 2000 steps, docking within 10 m below 0.2 m/s.
    assert (cfg.mass_kg, cfg.max_accel * cfg.mass_kg) == pytest.approx((12.0, 1.0))
    assert (cfg.dt, cfg.max_steps) == (1.0, 2000)
    assert (p.capture_radius, p.capture_speed, p.out_of_bounds) == (10.0, 0.2, 800.0)
    limit = next(c for c in env.constraints if c.name == "speed_limit")
    # nu0 + nu1 (rho - dd) with nu0 = 0.2 m/s, nu1 = 2 n and dd = 10 m (eq. 5).
    for rho in (10.0, 150.0):
        allowed = limit.v0 + limit.v1 * rho
        assert allowed == pytest.approx(0.2 + 2.0 * AFRL_N * (rho - 10.0), rel=1e-12)
    assert "speed_limit" in env.soft_constraints


def test_afrl_docking_starts_100_to_150_m_out_below_08_of_the_speed_limit():
    spec = SCENARIOS["afrl-docking"]
    ranges, ratios = [], []
    for seed in seeds("afrl-docking", "validation", 200):
        draw = spec.episode(seed)
        rho = float(np.linalg.norm(draw["start_pos"]))
        ranges.append(rho)
        ratios.append(np.linalg.norm(draw["start_vel"]) / (0.2 + 2.0 * AFRL_N * (rho - 10.0)))
    assert 100.0 <= min(ranges) and max(ranges) <= 150.0
    assert 0.0 <= min(ratios) and max(ratios) <= 0.8 and max(ratios) > 0.7


@pytest.mark.parametrize("speed, success", [(0.15, True), (0.3, False)])
def test_arriving_inside_the_docking_radius_ends_the_episode_docked_or_crashed(speed, success):
    env = make("afrl-docking", training=True)
    env.reset(seed=0, options={"start_state": (0.0, -10.1, 0.0, 0.0, speed, 0.0)})
    _, _, terminated, truncated, info = env.step(np.zeros(3))
    assert terminated and not truncated and info["success"] is success


def test_leaving_the_bounds_ends_the_episode_as_a_failure():
    env = make("afrl-docking", training=True)
    env.reset(seed=0, options={"start_state": (0.0, 799.5, 0.0, 0.0, 1.0, 0.0)})
    _, _, terminated, _, info = env.step(np.zeros(3))
    assert terminated and not info["success"]


def test_leaving_the_inspection_bounds_ends_the_episode_as_a_failure():
    env = make("afrl-inspection", training=True)
    env.reset(seed=0, options={"start_state": (0.0, 799.5, 0.0, 0.0, 1.0, 0.0)})
    _, _, terminated, _, info = env.step(np.zeros(3))
    assert terminated and not info["success"]


def test_exact_propagation_and_the_afrl_euler_steps_differ_by_under_2_m():
    # The AFRL environments step the Clohessy-Wiltshire equations with forward Euler at 1 s
    # (arXiv 2207.04117, eqs. 11 and 12); V-BAR propagates them exactly. Over the task's 2000 s
    # of free drift from its start set the two stay within 1.5 m (1.47 m measured over 50
    # validation seeds), inside the 10 m radius.
    n = AFRL_N
    continuous = np.zeros((6, 6))
    continuous[:3, 3:] = np.eye(3)
    continuous[3, 0], continuous[3, 4], continuous[4, 3] = 3 * n * n, 2 * n, -2 * n
    continuous[5, 2] = -n * n
    euler = np.eye(6) + continuous
    exact, _ = cw_discrete(n, 1.0)
    spec = SCENARIOS["afrl-docking"]
    worst = 0.0
    for seed in seeds("afrl-docking", "validation", 20):
        draw = spec.episode(seed)
        a = b = np.concatenate([draw["start_pos"], draw["start_vel"]])
        for _ in range(2000):
            a, b = euler @ a, exact @ b
            worst = max(worst, float(np.linalg.norm(a[:3] - b[:3])))
    assert 0.1 < worst < 1.5


def test_every_point_of_the_afrl_chief_is_first_lit_after_half_a_sun_orbit():
    # arXiv 2308.02743: "the minimum amount of time the task can be completed in is approximately
    # 3060 seconds", pi / n = 3059 s. The Sun turns at -n in the orbit plane, so the lit
    # hemisphere has swept every in-plane direction only after half a turn.
    env = make("afrl-inspection")
    env.reset(seed=4)
    times = np.arange(0.0, 6000.0, 10.0)
    sun = sun_at(env.sun0, env.n, times)
    lit = np.sum(env.normals(times) * sun[:, None, :], axis=-1) > 0.0
    covered = np.cumsum(lit, axis=0).astype(bool).mean(axis=1)
    first = times[np.argmax(covered >= 1.0)]
    assert first == pytest.approx(math.pi / AFRL_N, abs=20.0)
    assert covered[times.searchsorted(0.9 * math.pi / AFRL_N)] < 1.0


def test_the_afrl_inspection_task_has_its_papers_constants():
    env = make("afrl-inspection")
    p, cfg = env.params, env.cfg
    assert (cfg.dt, cfg.max_steps) == (10.0, 1224)
    # Two Sun orbits: 4 pi / n = 12236 s, 1223.6 steps of 10 s.
    assert cfg.max_steps * cfg.dt == pytest.approx(4.0 * math.pi / AFRL_N, abs=10.0)
    assert (cfg.mass_kg, cfg.max_accel * cfg.mass_kg) == pytest.approx((12.0, 1.0))
    assert (p.points, p.target_radius, p.keep_out, p.coverage) == (100, 10.0, 15.0, 1.0)
    assert p.out_of_bounds == 800.0 and not p.shadow
    env.reset(seed=1)
    assert np.all(env.target_lit(np.linspace(0.0, 12000.0, 50)) == 1.0)
    assert env.draw["beta"] == 0.0
    for seed in seeds("afrl-inspection", "validation", 50):
        draw = SCENARIOS["afrl-inspection"].episode(seed)
        assert 50.0 <= np.linalg.norm(draw["start_pos"]) <= 100.0
        assert np.linalg.norm(draw["start_vel"]) <= 0.3


@pytest.mark.parametrize(
    "name, steps, radius, speed, start, bound",
    [
        ("spacegym-docking", 500, 1.0, 0.1, (50.0, 100.0, 0.5), 200.0),
        ("spacegym-rendezvous", 1000, 10.0, 0.5, (800.0, 1200.0, 1.0), 2000.0),
    ],
)
def test_the_spacegym_tasks_have_their_papers_constants(name, steps, radius, speed, start, bound):
    env = make(name)
    cfg, p = env.cfg, env.params
    # a = 6778 km "yielding n = 0.00113 rad/s"; 500 kg with 10 N per axis; a 50 kg fuel budget.
    assert R_EARTH + cfg.altitude_m == pytest.approx(6778e3, rel=1e-12)
    assert env.n == pytest.approx(math.sqrt(MU_EARTH / 6778e3**3), rel=1e-12)
    assert round(env.n, 5) == 0.00113
    assert (cfg.mass_kg, cfg.max_accel * cfg.mass_kg, cfg.propellant_kg) == (500.0, 10.0, 50.0)
    assert (cfg.dt, cfg.max_steps) == (1.0, steps)
    assert (p.capture_radius, p.capture_speed, p.out_of_bounds) == (radius, speed, bound)
    low, high, top = start
    for seed in seeds(name, "validation", 50):
        draw = SCENARIOS[name].episode(seed)
        assert low <= np.linalg.norm(draw["start_pos"]) <= high
        assert np.linalg.norm(draw["start_vel"]) <= top


def test_spacegym_rendezvous_fails_on_entering_its_keep_out_sphere():
    env = make("spacegym-rendezvous", training=True)
    env.reset(seed=0, options={"start_state": (0.0, 5.5, 0.0, 0.0, -1.0, 0.0)})
    _, _, terminated, _, info = env.step(np.zeros(3))
    assert env.params.keep_out == 5.0 and terminated and not info["success"]


@pytest.mark.parametrize(
    "name", ["afrl-docking", "afrl-inspection", "spacegym-docking", "spacegym-rendezvous"]
)
def test_free_drift_fails_the_comparison_tasks(name):
    env = make(name)
    seed, agent_seed = seeds(name, "validation", 1).pairs()[0]
    run_episode(env, DriftAgent(), seed=seed, agent_seed=agent_seed)
    assert not env._info()["success"]


def test_the_comparison_task_draws_and_bounds_refuse_bad_values():
    from dataclasses import replace

    from vbar.scenarios import AngleShell, LimitedSpeed

    hold = make("afrl-docking").params
    inspect = make("afrl-inspection").params
    for bad in (
        lambda: AngleShell(-1.0, 2.0),
        lambda: AngleShell(3.0, 2.0),
        lambda: AngleShell(1.0, math.inf),
        lambda: LimitedSpeed(0.5, 0.2, 0.2, 0.002),
        lambda: LimitedSpeed(0.0, 0.8, 0.0, 0.002),
        lambda: LimitedSpeed(0.0, 0.8, 0.2, -0.002),
        lambda: LimitedSpeed(0.0, 0.8, 0.2, math.inf),
        lambda: LimitedSpeed(0.0, 0.8, 0.2, 0.002, offset=math.nan),
        lambda: replace(hold, out_of_bounds=0.0),
        lambda: replace(hold, out_of_bounds=math.inf),
        lambda: replace(hold, end_on_arrival=True, hold_time=10.0),
        lambda: replace(inspect, out_of_bounds=-5.0),
        lambda: replace(inspect, out_of_bounds=math.nan),
    ):
        with pytest.raises(ValueError):
            bad()
