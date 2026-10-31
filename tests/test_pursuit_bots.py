import math
from dataclasses import dataclass

import numpy as np
import pytest

from vbar.conventions import G0
from vbar.games import (
    CONTROL_RANGE,
    AgentEnvRunner,
    NormalEscapeEvader,
    PE1_E3_I3_Env,
    PursuitEvasionEnv,
    PursuitEvasionGymEnv,
    Segment,
    axes_along,
    fly,
    pursuit,
)
from vbar.games.pursuers import VelocityMatchingPursuer

FLOW = 8000.0 / (240.0 * G0)  # kg/s of the eight forward nozzles at full throttle
COAST = [0.0, 0.0, 0.0, 1.0]


@dataclass(frozen=True)
class Scripted:
    """Flies `burn` until `stop` s, then coasts; the timed switch is a segment's `until`."""

    burn: tuple = (1.0, 0.0, 0.0)
    pointing: str = "prograde"
    stop: float = 2.5
    name: str = "scripted"

    def segment(self, t, r_p, v_p, r_e, v_e, previous, cause, rng):
        if t < self.stop:
            return Segment(throttle=self.burn, pointing=self.pointing, until=self.stop)
        return Segment()


@dataclass(frozen=True)
class Random:
    """Draws a throttle each second from the episode's stream."""

    name: str = "random"

    def segment(self, t, r_p, v_p, r_e, v_e, previous, cause, rng):
        return Segment(throttle=tuple(2.0 * rng.random(3) - 1.0), until=t + 1.0)


def test_the_rework_flies_the_i3_episodes_of_release_0_4_0():
    # Recorded with vbar 0.4.0 (VelocityMatchingPursuer against PE1_E3_I3_Env); the agreement
    # is to rounding, because the evader's thrust is now a body-axis product, not a vector.
    info = AgentEnvRunner(VelocityMatchingPursuer(), PE1_E3_I3_Env, None).run()["agent_env_results"]
    assert info["steps"] == 480
    assert info["closest_approach"] == pytest.approx(19.579560013801114, rel=1e-9)
    assert info["evader_fuel_usage"] == pytest.approx(310.11758571997507, rel=1e-9)
    assert info["pursuer_fuel_usage"] == pytest.approx(466.18239932614324, rel=1e-9)
    assert info["weighted_score"] == pytest.approx(128.04753542533712, rel=1e-9)


def test_a_timed_segment_ends_at_its_time_inside_a_step():
    env = PursuitEvasionEnv(evader=Scripted())
    env.reset()
    env.step(COAST)
    env.step(COAST)
    info = env.step(COAST)[3]
    # The burn stops at 2.5 s, inside the third 1 s step: fuel is 2.5 s at full forward flow.
    assert info["evader_fuel_usage"] == pytest.approx(2.5 * FLOW, rel=1e-12)
    assert env.step(COAST)[3]["evader_fuel_usage"] == info["evader_fuel_usage"]


def test_lateral_throttles_draw_each_axis_from_its_own_nozzles():
    env = PursuitEvasionEnv(evader=Scripted(burn=(1.0, -1.0, 0.5), stop=1.0))
    env.reset()
    info = env.step(COAST)[3]
    # 8000 N forward, 4000 N left and 2000 N down, each at 240 s.
    assert info["evader_fuel_usage"] == pytest.approx(14000.0 / (240.0 * G0), rel=1e-12)


@pytest.mark.parametrize("pointing", ["prograde", "normal"])
def test_a_forward_burn_follows_the_pointing_and_the_rocket_equation(pointing):
    coast = PursuitEvasionEnv()
    burn = PursuitEvasionEnv(evader=Scripted(pointing=pointing, stop=1.0))
    start = np.array(coast.reset()[0])
    burn.reset()
    dv = np.array(burn.step(COAST)[0][12:15]) - np.array(coast.step(COAST)[0][12:15])
    r_e, v_e = start[9:12], start[12:15]
    forward = v_e if pointing == "prograde" else np.cross(r_e, v_e)
    wet = 6685.0
    expected = 240.0 * G0 * math.log(wet / (wet - FLOW))
    assert np.linalg.norm(dv) == pytest.approx(expected, rel=1e-5)
    # The direction turns with the orbit by about a milliradian in the second.
    assert dv @ forward / (np.linalg.norm(dv) * np.linalg.norm(forward)) > 1.0 - 1e-5


def test_axes_along_keep_down_toward_the_body():
    r = np.array([7.5e5, 0.0, 0.0])
    axes = axes_along(r, [0.0, 2170.0, 0.0])
    assert np.allclose(axes, [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], atol=1e-15)
    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-15)
    with pytest.raises(ValueError):
        axes_along(r, [0.0, 0.0, 0.0])


def test_the_same_seed_flies_the_same_episode_and_another_seed_differs():
    runs = {}
    for name, seed in (("a", 2**100 + 7), ("b", 2**100 + 7), ("c", 2**100 + 8)):
        env = PursuitEvasionEnv(evader=Random())
        env.reset(seed=seed)
        for _ in range(5):
            observation = env.step(COAST)[0]
        runs[name] = observation
    assert runs["a"] == runs["b"]
    assert runs["a"] != runs["c"]


def test_the_constructor_seed_holds_for_every_reset_and_none_draws_a_fresh_one():
    fixed = PursuitEvasionEnv(evader=Random(), seed=5)
    fixed.reset()
    first = fixed.step(COAST)[0]
    fixed.reset()
    assert fixed.step(COAST)[0] == first and fixed.seed == 5
    fresh = PursuitEvasionEnv(evader=Random())
    seeds = set()
    for _ in range(3):
        fresh.reset()
        seeds.add(fresh.seed)
    # Drawn from the operating system: 128 bits, so three draws never repeat in practice.
    assert len(seeds) == 3 and all(0 <= seed < 2**128 for seed in seeds)
    assert max(seeds) > 2**64


@pytest.mark.parametrize("seed", [-1, 2.0, True, "1", np.float64(3.0)])
def test_a_bad_reset_seed_is_refused_before_the_episode_changes(seed):
    env = PursuitEvasionEnv(evader=Random(), seed=3)
    env.reset()
    env.step(COAST)
    before = (env.get_observation(), env.get_info(), env.seed)
    with pytest.raises(ValueError):
        env.reset(seed=seed)
    assert (env.get_observation(), env.get_info(), env.seed) == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"throttle": (1.5, 0.0, 0.0)},
        {"throttle": (float("nan"), 0.0, 0.0)},
        {"throttle": (1.0, 0.0)},
        {"pointing": "retrograde"},
        {"until": float("nan")},
        {"triggers": ((lambda *states: 0.0, 0),)},
        {"triggers": (("not callable", 1),)},
    ],
)
def test_a_bad_segment_is_refused(kwargs):
    with pytest.raises(ValueError):
        Segment(**kwargs)


@dataclass(frozen=True)
class Looping:
    name: str = "looping"

    def segment(self, t, r_p, v_p, r_e, v_e, previous, cause, rng):
        return Segment(until=t)


@dataclass(frozen=True)
class Wrong:
    name: str = "wrong"

    def segment(self, t, r_p, v_p, r_e, v_e, previous, cause, rng):
        return {"throttle": (1.0, 0.0, 0.0)}


def test_a_bot_that_never_lets_time_pass_is_stopped_not_left_hanging():
    env = PursuitEvasionEnv(evader=Looping())
    env.reset()
    with pytest.raises(RuntimeError, match="no time passing"):
        env.step(COAST)


def test_a_bot_must_return_a_segment():
    with pytest.raises(TypeError, match="Segment"):
        PursuitEvasionEnv(evader=Wrong()).reset()


@dataclass(frozen=True)
class Rearming:
    """Sets each trigger a nanometre inside the current range, so every segment is about 1e-10 s.

    Re-arming the trigger at the crossing just found loops only when rounding leaves the range
    on the near side of the root, which varies by platform; a fixed nanometre does not.
    """

    name: str = "rearming"

    def segment(self, t, r_p, v_p, r_e, v_e, previous, cause, rng):
        near = float(np.linalg.norm(r_e - r_p)) - 1e-9

        def g(r_p, v_p, r_e, v_e):
            return float(np.linalg.norm(r_e - r_p)) - near

        return Segment(triggers=((g, -1),))


def test_a_bot_whose_segments_last_almost_no_time_is_stopped_not_left_hanging(monkeypatch):
    # Each segment costs a solve, so the cap is lowered from 1000 to 20 to keep the test short;
    # burning toward the evader, the range falls at metres per second from the start.
    monkeypatch.setattr(pursuit, "_MAX_SHORT_SEGMENTS", 20)
    env = PursuitEvasionEnv(evader=Rearming())
    env.reset()
    with pytest.raises(RuntimeError, match="no time passing"):
        env.step([1.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_each_body_axis_throttle_pushes_along_that_axis(axis):
    burn = [0.0, 0.0, 0.0]
    burn[axis] = 1.0
    coast = PursuitEvasionEnv()
    start = np.array(coast.reset()[0])
    fired = PursuitEvasionEnv(evader=Scripted(burn=tuple(burn), stop=0.1))
    fired.reset()
    step = [0.0, 0.0, 0.0, 0.1]
    dv = np.array(fired.step(step)[0][12:15]) - np.array(coast.step(step)[0][12:15])
    expected = axes_along(start[9:12], start[12:15])[axis]
    thrust = 8000.0 if axis == 0 else 4000.0
    # The axes turn with the orbit by 0.3 mrad in the 0.1 s: a cosine within 1e-6 of 1 (the
    # other axes give 0).
    assert dv / np.linalg.norm(dv) @ expected > 1.0 - 1e-6
    assert np.linalg.norm(dv) == pytest.approx(thrust * 0.1 / 6685.0, rel=1e-4)


def test_the_gymnasium_seed_seeds_the_evader():
    env = PursuitEvasionGymEnv("pe1-e1-i3", evader=Random())
    action = {"burn_vec": np.array([0.0, 0.0, 0.0, 1.0]), "ref_frame": 0, "vec_type": 0}
    runs = []
    for seed in (0, 0, 1):
        env.reset(seed=seed)
        runs.append(env.step(action)[0])
    assert np.array_equal(runs[0], runs[1]) and not np.array_equal(runs[0], runs[2])
    # Unseeded resets draw their seeds from np_random, so they follow from a seeded one.
    drawn = []
    for _ in range(2):
        env.reset(seed=3)
        drawn.append([env.reset() and env.game.seed, env.reset() and env.game.seed])
    assert drawn[0] == drawn[1] and drawn[0][0] != drawn[0][1]


class Coaster:
    def get_action(self, observation):
        return COAST


def test_fly_passes_its_seed_to_the_reset():
    runs = [
        fly(PursuitEvasionEnv(evader=Random(), episode_timeout=5.0), Coaster(), seed=seed)
        for seed in (9, 9, 10)
    ]
    assert runs[0]["evader_fuel_usage"] == runs[1]["evader_fuel_usage"]
    assert runs[0]["evader_fuel_usage"] != runs[2]["evader_fuel_usage"]


@pytest.mark.parametrize("evade_distance", [CONTROL_RANGE, 2500.0])
def test_the_normal_escape_evader_acts_only_inside_the_control_range(evade_distance):
    # An evasion distance at or beyond the 2200 m control range: KSPDG's e3 does nothing until
    # the pursuer is inside 2200 m and, once out again, changes nothing, so it keeps burning.
    env = PursuitEvasionEnv(evader=NormalEscapeEvader(evade_distance=evade_distance))
    env.reset()
    rows, done = [], False
    while not done:
        throttle = 1.0 if env.t < 20.0 - 1e-9 else 0.0
        observation, _, done, info = env.step([throttle, 0.0, 0.0, 1.0])
        distance = float(np.linalg.norm(np.subtract(observation[9:12], observation[3:6])))
        rows.append((observation[0], distance, info["evader_fuel_usage"]))
    entered = next(i for i, row in enumerate(rows) if row[1] < CONTROL_RANGE)
    # Nothing between 2500 m and 2200 m, where the old rule burned.
    assert rows[0][1] > 2500.0 and all(fuel == 0.0 for _, _, fuel in rows[: entered - 1])
    left = next(i for i, row in enumerate(rows) if i > entered and row[1] > CONTROL_RANGE)
    assert rows[-1][1] > 10 * CONTROL_RANGE and left < len(rows) - 100
    # One burn from the crossing (inside the step that ends below 2200 m) to the end.
    burning = rows[-1][2] / FLOW
    t_in = rows[entered][0]
    assert 240.0 - t_in < burning < 240.0 - t_in + 1.0


def test_the_normal_escape_evader_switches_on_which_crossing_fired():
    bot = NormalEscapeEvader(evade_distance=500.0)
    rng = np.random.default_rng(0)
    r_e, v_e = np.array([7.5e5, 0.0, 0.0]), np.array([0.0, 2170.0, 0.0])
    at = {d: r_e - [0.0, d, 0.0] for d in (300.0, 1000.0, 3000.0)}
    outside = bot.segment(0.0, at[3000.0], v_e, r_e, v_e, None, None, rng)
    assert outside.throttle == (0.0, 0.0, 0.0) and outside.triggers == ((bot._control, -1),)
    coast = bot.segment(1.0, at[1000.0], v_e, r_e, v_e, outside, 0, rng)
    assert coast.throttle == (0.0, 0.0, 0.0)
    assert coast.triggers == ((bot._evade, -1), (bot._control, 1))
    burn = bot.segment(2.0, at[300.0], v_e, r_e, v_e, coast, 0, rng)
    assert burn.throttle == (1.0, 0.0, 0.0)
    assert bot.segment(3.0, at[1000.0], v_e, r_e, v_e, burn, 0, rng).throttle == coast.throttle
    # Leaving through the control range keeps the last command: thrust after a burn.
    assert bot.segment(3.0, at[3000.0], v_e, r_e, v_e, burn, 1, rng).throttle == burn.throttle
    assert bot.segment(3.0, at[3000.0], v_e, r_e, v_e, coast, 1, rng).throttle == coast.throttle
    # Entering when the evasion distance is the wider one starts a burn at once.
    wide = NormalEscapeEvader(evade_distance=2500.0)
    held = wide.segment(0.0, at[3000.0], v_e, r_e, v_e, None, None, rng)
    assert wide.segment(1.0, at[3000.0], v_e, r_e, v_e, held, 0, rng).throttle == burn.throttle
    with pytest.raises(ValueError):
        NormalEscapeEvader(control_range=0.0)
