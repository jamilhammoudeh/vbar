import numpy as np
import pytest

from vbar.conventions import G0
from vbar.games import (
    CONTROL_RANGE,
    PE1_E2_I2_Env,
    PE1_E2_I3_Env,
    PE1_E2_I4_Env,
    RandomEvader,
    Segment,
    fly,
)

COAST = [0.0, 0.0, 0.0, 1.0]
R_E, V_E = np.array([7.5e5, 0.0, 0.0]), np.array([0.0, 2170.0, 0.0])
NEAR, FAR = R_E - [0.0, 500.0, 0.0], R_E - [0.0, 3000.0, 0.0]


class Coast:
    def get_action(self, observation):
        return COAST


def test_the_draws_follow_kspdg_e2_inside_the_control_range():
    bot = RandomEvader()
    rng = np.random.default_rng(2**90 + 1)
    segment, durations, on, throttles = None, [], 0, []
    t = 0.0
    for _ in range(4000):
        segment = bot.segment(t, NEAR, V_E, R_E, V_E, segment, None, rng)
        durations.append(segment.until - t)
        if any(segment.throttle):
            on += 1
            throttles.extend(segment.throttle)
        assert segment.pointing == "prograde" and segment.triggers == ()
        t = segment.until
    durations = np.array(durations)
    # Uniform on (0, 2) s: mean 1 s, standard error 0.58 / sqrt(4000) = 0.009 s.
    assert 0.0 < durations.min() and durations.max() < 2.0
    assert durations.mean() == pytest.approx(1.0, abs=0.04)
    # Throttles on half the time (standard error 0.008), each uniform on [-1, 1].
    assert on / 4000 == pytest.approx(0.5, abs=0.035)
    assert np.mean(throttles) == pytest.approx(0.0, abs=0.04)
    assert np.mean(np.abs(throttles)) == pytest.approx(0.5, abs=0.02)


def test_outside_the_control_range_the_last_throttles_are_kept():
    bot = RandomEvader()
    rng = np.random.default_rng(3)
    last = Segment(throttle=(0.3, -0.2, 0.9), until=5.0)
    held = bot.segment(5.0, FAR, V_E, R_E, V_E, last, None, rng)
    assert held.throttle == last.throttle and held.until == np.inf
    ((g, falling),) = held.triggers
    assert falling == -1 and g(FAR, V_E, R_E, V_E) == pytest.approx(3000.0 - CONTROL_RANGE)
    # Back inside through the control range, it draws at once.
    drawn = bot.segment(9.0, FAR, V_E, R_E, V_E, held, 0, rng)
    assert 9.0 < drawn.until < 11.0 and drawn.triggers == ()
    assert bot.segment(0.0, FAR, V_E, R_E, V_E, None, None, rng).throttle == (0.0, 0.0, 0.0)


def test_the_random_evader_does_nothing_while_the_pursuer_stays_out_of_range():
    # The I3 coast never comes nearer than 2618 m.
    info = fly(PE1_E2_I3_Env(), Coast())
    assert info["evader_fuel_usage"] == 0.0


def test_the_random_evader_starts_drawing_when_the_pursuer_comes_inside_its_range():
    # The I4 coast starts 2660 m apart and closes to 999.5 m at 63 s.
    env = PE1_E2_I4_Env(seed=2**100 + 3)
    env.reset()
    fuel, done = [], False
    while not done:
        observation, _, done, info = env.step(COAST)
        distance = float(np.linalg.norm(np.subtract(observation[9:12], observation[3:6])))
        fuel.append((distance, info["evader_fuel_usage"]))
    entered = next(i for i, (distance, _) in enumerate(fuel) if distance < CONTROL_RANGE)
    assert all(used == 0.0 for _, used in fuel[: entered - 1])
    assert fuel[-1][1] > 0.0


def test_the_random_evader_replays_by_seed_and_differs_between_seeds():
    runs = [
        fly(PE1_E2_I2_Env(episode_timeout=20.0), Coast(), seed=seed)
        for seed in (2**100, 2**100, 2**100 + 1)
    ]
    assert runs[0] == {**runs[1], "wall_s": runs[0]["wall_s"]}
    assert runs[0]["evader_fuel_usage"] != runs[2]["evader_fuel_usage"]
    # I2 starts 457 m apart, inside the control range: the bot burns from the start.
    assert runs[0]["evader_fuel_usage"] > 0.0


def test_lateral_draws_use_each_axis_flow():
    env = PE1_E2_I2_Env(seed=11)
    env.reset()
    segment = env._segment
    assert all(segment.throttle)  # this seed's first draw fires on all three axes
    fuel = env.step([0.0, 0.0, 0.0, min(segment.until, 0.1)])[3]["evader_fuel_usage"]
    limits = np.array([8000.0, 4000.0, 4000.0])
    expected = np.abs(segment.throttle) @ limits / (240.0 * G0) * min(segment.until, 0.1)
    assert fuel == pytest.approx(expected, rel=1e-12)


def first_evader_states(seed, steps=3):
    env = PE1_E2_I2_Env(seed=seed)
    env.reset()
    return [env.step(COAST)[0][9:15] for _ in range(steps)]


def test_an_agent_cannot_find_a_drawn_seed_by_flying_twins():
    """A cheat agent replays candidate seeds and matches the evader's first seconds."""

    def attack(observed, candidates):
        return [k for k in candidates if first_evader_states(k) == observed]

    # A small seed falls to enumeration in a fraction of a second...
    assert attack(first_evader_states(7), range(16)) == [7]
    # ...a drawn 128-bit seed does not: none of the same number of candidates matches. A seed
    # whose evader idles through the window leaves the fixed start, which about 3 in 1000 seeds
    # share (seed 1 among them), so it carries no seed information and is drawn again.
    while True:
        env = PE1_E2_I2_Env()
        env.reset()
        steps = [env.step(COAST) for _ in range(3)]
        if steps[-1][3]["evader_fuel_usage"] > 0.0:
            break
    observed = [step[0][9:15] for step in steps]
    assert env.seed >= 2**64
    assert attack(observed, range(16)) == []


def test_an_idle_evader_window_matches_every_idle_seed():
    # Why the twin attack redraws idle seeds: their windows are identical.
    assert first_evader_states(1) == first_evader_states(148)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_duration": 0.0},
        {"max_duration": float("inf")},
        {"max_duration": 1e-15},
        {"max_duration": 0.999e-3},
        {"control_range": -1.0},
    ],
)
def test_bad_settings_of_the_random_evader_are_refused(kwargs):
    with pytest.raises(ValueError):
        RandomEvader(**kwargs)


def test_the_shortest_hold_flies_without_stalling():
    env = PE1_E2_I2_Env(evader=RandomEvader(max_duration=1e-3), seed=5, episode_timeout=0.05)
    info = fly(env, Coast())
    assert info["is_episode_done"] and info["evader_fuel_usage"] > 0.0
