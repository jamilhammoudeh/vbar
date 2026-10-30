import math
from abc import ABC, abstractmethod

import numpy as np
import pytest

from vbar.conventions import G0
from vbar.dynamics import KERBIN
from vbar.dynamics.kepler import propagate_kepler
from vbar.games import (
    OBSERVATION,
    PE1_VESSEL,
    AgentEnvRunner,
    BaseAgent,
    NormalEscapeEvader,
    PursuitEvasionEnv,
    Vessel,
    body_axes,
    fly,
    ntw_axes,
    parse_action,
    weighted_score,
)
from vbar.games.pursuers import LineOfSightPursuer, VelocityMatchingPursuer
from vbar.games.pursuit import I3_EVADER, I3_PURSUER

FLOW = 8000.0 / (240.0 * G0)  # kg/s of the eight forward nozzles at full throttle


def relative_range(observation):
    return float(np.linalg.norm(np.subtract(observation[9:12], observation[3:6])))


class KSPDGBaseAgent(ABC):
    """Stands in for KSPDG's base class, with its public signature (base_agent.py)."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    @abstractmethod
    def get_action(self, observation):
        pass


class NaivePursuitAgent(KSPDGBaseAgent):
    """Written as KSPDG's example agent is: full forward throttle for one second."""

    def get_action(self, observation):
        return {"burn_vec": [1.0, 0, 0, 1.0], "vec_type": 0, "ref_frame": 0}


class Scripted(BaseAgent):
    def __init__(self, actions):
        super().__init__()
        self.actions = list(actions)
        self.seen = []

    def get_action(self, observation):
        self.seen.append(observation)
        return self.actions.pop(0) if self.actions else None


def test_ntw_axes_are_orthonormal_right_handed_with_t_along_velocity():
    r, v = I3_PURSUER.state(KERBIN.mu)
    axes = ntw_axes(r, v)
    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-15)
    assert np.isclose(np.linalg.det(axes), 1.0)
    assert np.allclose(axes[1], v / np.linalg.norm(v))
    # On a circular orbit N is radial and outward.
    assert np.allclose(axes[0], r / np.linalg.norm(r), atol=1e-12)


def test_ntw_axes_refuse_a_radial_velocity():
    with pytest.raises(ValueError, match="NTW"):
        ntw_axes([7e5, 0.0, 0.0], [10.0, 0.0, 0.0])


def test_body_axes_point_forward_at_the_target_with_down_toward_the_body():
    r_p, _ = I3_PURSUER.state(KERBIN.mu)
    r_e, _ = I3_EVADER.state(KERBIN.mu)
    axes = body_axes(r_p, r_e)
    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-15)
    assert np.isclose(np.linalg.det(axes), 1.0)
    assert np.allclose(axes[0], (r_e - r_p) / np.linalg.norm(r_e - r_p), rtol=0, atol=1e-15)
    # Down is the part of the nadir direction across the line of sight.
    nadir = -r_p / np.linalg.norm(r_p)
    across = nadir - (nadir @ axes[0]) * axes[0]
    assert np.allclose(axes[2], across / np.linalg.norm(across), atol=1e-12)


@pytest.mark.parametrize("target", [[7.5e5 - 100.0, 0.0, 0.0], [0.0, 0.0, 9e5]])
def test_body_axes_stay_defined_with_the_target_straight_below(target):
    r = [7.5e5, 0.0, 0.0] if target[2] == 0.0 else [0.0, 0.0, 7.5e5]
    axes = body_axes(r, target)
    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-15)
    assert np.isclose(np.linalg.det(axes), 1.0)


def test_body_axes_refuse_a_target_at_the_vessel():
    with pytest.raises(ValueError, match="target"):
        body_axes([7.5e5, 0.0, 0.0], [7.5e5, 0.0, 0.0])


def test_vessel_thrust_limits_are_eight_and_four_nozzles_each_way():
    thrust = PE1_VESSEL.thrust(np.array([2.0, -1.0, 0.5]))
    assert np.allclose(thrust, [8000.0, -4000.0, 2000.0])
    assert np.allclose(PE1_VESSEL.throttle(np.array([-8000.0, 2000.0, -4000.0])), [-1, 0.5, -1])
    assert math.isclose(PE1_VESSEL.mass_flow(np.array([8000.0, 0.0, -4000.0])), 1.5 * FLOW)
    assert PE1_VESSEL.wet_mass == 1885.0 + 1200 * 4.0


def test_i3_start_puts_both_vessels_on_one_circular_orbit_2618_m_apart():
    observation, info = PursuitEvasionEnv().reset()
    assert len(observation) == len(OBSERVATION) == 15
    assert all(type(x) is float for x in observation)
    assert observation[:3] == [0.0, 6685.0, 1200.0]
    speed = math.sqrt(KERBIN.mu / 750e3)
    assert np.isclose(np.linalg.norm(observation[6:9]), speed, rtol=1e-14)
    assert np.isclose(np.linalg.norm(observation[12:15]), speed, rtol=1e-14)
    # The mean anomalies in the I3 mission save differ by 0.2 deg; the arc a dM is 2617.994 m,
    # the chord 1.3 mm shorter and the evader's 1e-4 deg tilt adds 0.04 mm.
    assert relative_range(observation) == pytest.approx(
        750e3 * (5.9341194567807207 - 5.9306287982767323), abs=2e-3
    )
    assert info["closest_approach"] == pytest.approx(2617.99, abs=0.01)
    assert info["closest_approach_time"] == 0.0


def test_a_coast_follows_kepler_propagation():
    env = PursuitEvasionEnv()
    env.reset()
    for duration in (0.5, 10.0, 3.7, 10.0, 10.0):
        observation = env.step([0.0, 0.0, 0.0, duration])[0]
    assert observation[0] == pytest.approx(34.2, abs=1e-12)
    for orbit, at in ((I3_PURSUER, 3), (I3_EVADER, 9)):
        r, v = propagate_kepler(*orbit.state(KERBIN.mu), observation[0], KERBIN.mu)
        # Both paths are accurate to near 1e-12 of the 7.5e5 m radius.
        assert np.allclose(observation[at : at + 3], r, rtol=0, atol=1e-6)
        assert np.allclose(observation[at + 3 : at + 6], v, rtol=0, atol=1e-9)


def test_a_full_forward_burn_follows_the_rocket_equation():
    burning, coasting = PursuitEvasionEnv(), PursuitEvasionEnv()
    start = burning.reset()[0]
    coasting.reset()
    after = burning.step([1.0, 0.0, 0.0, 1.0])[0]
    twin = coasting.step([0.0, 0.0, 0.0, 1.0])[0]
    used = 6685.0 - after[1]
    assert used == pytest.approx(FLOW, rel=1e-12)
    assert 1200.0 - after[2] == pytest.approx(FLOW / 4.0, rel=1e-12)
    dv = np.subtract(after[6:9], twin[6:9])
    expected = 240.0 * G0 * math.log(6685.0 / after[1])
    # Gravity differs by 1e-6 between the twins' paths and the line of sight turns slightly.
    assert np.linalg.norm(dv) == pytest.approx(expected, rel=1e-5)
    line_of_sight = np.subtract(start[9:12], start[3:6])
    cosine = dv @ line_of_sight / np.linalg.norm(dv) / np.linalg.norm(line_of_sight)
    # The line of sight between co-orbiting vessels turns with the orbit, at n rad/s, so the
    # burn's mean direction is within n * 1 s of the starting line of sight.
    n = math.sqrt(KERBIN.mu / 750e3**3)
    assert 0.1 * n < math.acos(cosine) < n


def test_one_command_in_three_frames_flies_the_same_burn():
    env = PursuitEvasionEnv()
    observation = env.reset()[0]
    r_p, v_p = np.array(observation[3:6]), np.array(observation[6:9])
    body = body_axes(r_p, observation[9:12])
    direction = np.array([0.6, -0.3, 0.2])
    commands = [
        {"burn_vec": [*direction, 2.0], "ref_frame": 0},
        {"burn_vec": [*(body.T @ direction), 2.0], "ref_frame": 1},
        {"burn_vec": [*(ntw_axes(r_p, v_p) @ body.T @ direction), 2.0], "ref_frame": 2},
        {"burn_vec": [*(direction * [8000.0, 4000.0, 4000.0]), 2.0], "ref_frame": 0, "vec_type": 1},
    ]
    ends = []
    for command in commands:
        env.reset()
        ends.append(np.array(env.step(command)[0]))
    for end in ends[1:]:
        # Frame conversions round at 1e-16 of the command; positions near 7.5e5 m.
        assert np.allclose(end, ends[0], rtol=1e-12, atol=1e-9)


def test_throttle_beyond_one_saturates_and_a_list_is_body_frame_throttle():
    env = PursuitEvasionEnv()
    runs = []
    for action in ([3.0, 0.0, 0.0, 1.0], {"burn_vec": [1.0, 0.0, 0.0, 1.0], "ref_frame": 0}):
        env.reset()
        runs.append(env.step(action)[0])
    assert runs[0] == runs[1]
    env.reset()
    thrust = env.step({"burn_vec": [16000.0, 0.0, 0.0, 1.0], "ref_frame": 0, "vec_type": 1})[0]
    assert thrust == runs[0]


def test_a_short_burn_coasts_to_the_shortest_step():
    env = PursuitEvasionEnv(min_step=0.1)
    env.reset()
    observation = env.step([1.0, 0.0, 0.0, 0.04])[0]
    assert observation[0] == pytest.approx(0.1, abs=1e-15)
    assert 6685.0 - observation[1] == pytest.approx(0.04 * FLOW, rel=1e-12)
    assert env.step([1.0, 0.0, 0.0, 0.0])[0][1] == observation[1]


def test_the_episode_ends_at_the_timeout_with_the_score_as_reward():
    env = PursuitEvasionEnv(episode_timeout=25.0)
    env.reset()
    for _ in range(2):
        observation, reward, done, info = env.step([0.0, 0.0, 0.0, 10.0])
        assert not done and reward == 0.0
    observation, reward, done, info = env.step([0.0, 0.0, 0.0, 10.0])
    assert done and info["is_episode_done"] and not env.captured
    assert observation[0] == 25.0
    assert reward == -info["weighted_score"]
    with pytest.raises(RuntimeError, match="ended"):
        env.step([0.0, 0.0, 0.0, 1.0])


def test_capture_stops_the_step_at_the_capture_distance():
    env = PursuitEvasionEnv(capture_dist=2000.0)
    env.reset()
    done = False
    while not done:
        observation, reward, done, info = env.step([1.0, 0.0, 0.0, 10.0])
    assert env.captured
    # The event is located to the integrator's tolerance, far below a millimetre.
    assert relative_range(observation) == pytest.approx(2000.0, abs=1e-6)
    assert info["closest_approach"] == pytest.approx(2000.0, abs=1e-6)
    assert observation[0] % 10.0 > 0.0
    assert reward == -info["weighted_score"]


def test_an_episode_that_starts_inside_the_capture_distance_ends_on_the_first_step():
    env = PursuitEvasionEnv(capture_dist=3000.0)
    env.reset()
    observation, _, done, _ = env.step([0.0, 0.0, 0.0, 1.0])
    assert done and env.captured and observation[0] == 0.0


def test_the_closest_approach_is_found_between_steps():
    # Forty seconds of forward thrust, then a coast past the evader, flown in 10 s steps and in
    # 0.25 s steps with the same body-frame throttle.
    runs = []
    for step in (10.0, 0.25):
        env = PursuitEvasionEnv(episode_timeout=90.0, min_step=0.05)
        env.reset()
        ranges, done = [], False
        while not done:
            throttle = 1.0 if env.t < 40.0 - 1e-9 else 0.0
            observation, _, done, info = env.step([throttle, 0.0, 0.0, step])
            ranges.append(relative_range(observation))
        runs.append((ranges, info))
    (coarse, info), (fine, fine_info) = runs
    # The pass falls inside a 10 s step, 20 m below the nearest step end.
    assert info["closest_approach"] < min(coarse) - 10.0
    assert info["closest_approach"] <= min(fine)
    # The fine samples come within 0.125 s of the pass at about 50 m/s.
    assert info["closest_approach"] == pytest.approx(min(fine), abs=0.125 * 50.0)
    assert info["closest_approach"] == pytest.approx(fine_info["closest_approach"], abs=1e-6)
    assert info["closest_approach_time"] == pytest.approx(
        fine_info["closest_approach_time"], abs=1e-6
    )
    assert 40.0 < info["closest_approach_time"] < 90.0


def test_weighted_score_matches_the_published_formula():
    assert weighted_score(10.0, 2.0, 100.0, 50.0) == pytest.approx(1.0 + 1.0 + 10**1.25 + 0.5)
    assert weighted_score(0.0, 0.0, 0.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "action",
    [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, float("nan")],
        [float("inf"), 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.0],
        "fire",
        None,
        {"burn_vec": [1.0, 0.0, 0.0, 1.0]},
        {"burn_vec": [1.0, 0.0, 0.0, 1.0], "ref_frame": 3},
        {"burn_vec": [1.0, 0.0, 0.0, 1.0], "ref_frame": True},
        {"burn_vec": [1.0, 0.0, 0.0, 1.0], "ref_frame": 1.0},
        {"burn_vec": [1.0, 0.0, 0.0, 1.0], "ref_frame": 0, "vec_type": 2},
        {"burn_vec": [[1.0, 0.0], [0.0, 1.0]], "ref_frame": 0},
    ],
)
def test_a_malformed_action_is_refused_and_changes_nothing(action):
    env = PursuitEvasionEnv()
    before = env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.get_observation() == before[0]
    assert env.get_info() == before[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episode_timeout": 0.0},
        {"episode_timeout": float("nan")},
        {"capture_dist": -1.0},
        {"min_step": float("inf")},
        {"min_step": True},
        {"initial": "i9"},
    ],
)
def test_bad_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        PursuitEvasionEnv(**kwargs)


def test_an_environment_must_be_reset_first():
    with pytest.raises(RuntimeError, match="reset"):
        PursuitEvasionEnv().step([0.0, 0.0, 0.0, 1.0])


def test_parse_action_reads_both_forms():
    vector, duration, frame, kind = parse_action({"burn_vec": [1, 2, 3, 4], "ref_frame": 2})
    assert vector.tolist() == [1.0, 2.0, 3.0] and duration == 4.0 and (frame, kind) == (2, 0)
    assert parse_action((0.5, 0, 0, 1))[2:] == (0, 0)


def test_the_normal_escape_evader_thrusts_along_its_orbit_normal_inside_its_range():
    bot = NormalEscapeEvader()
    r_e, v_e = I3_EVADER.state(KERBIN.mu)
    near, far = r_e - [0.0, 150.0, 0.0], r_e - [0.0, 250.0, 0.0]
    assert bot.mode(far, v_e, r_e, v_e) == 0
    assert bot.mode(near, v_e, r_e, v_e) == 1
    assert not bot.force(0, near, v_e, r_e, v_e).any()
    h = np.cross(r_e, v_e)
    assert np.allclose(bot.force(1, near, v_e, r_e, v_e), 8000.0 * h / np.linalg.norm(h))
    with pytest.raises(ValueError):
        NormalEscapeEvader(evade_distance=0.0)
    with pytest.raises(ValueError):
        NormalEscapeEvader(thrust=float("nan"))


def test_the_evader_starts_escaping_when_the_pursuer_crosses_its_range():
    env = PursuitEvasionEnv(evader=NormalEscapeEvader(evade_distance=2500.0), episode_timeout=30.0)
    env.reset()
    crossed = None
    done = False
    while not done:
        before = env.get_observation()
        observation, _, done, info = env.step([1.0, 0.0, 0.0, 5.0])
        if crossed is None and relative_range(observation) < 2500.0:
            crossed = before[0]
    assert crossed is not None and crossed > 0.0
    # The evader burns at full forward thrust from the crossing on; the crossing lies in the
    # step that ends below the range, so its fuel use is bounded by the step's two ends.
    evading = info["evader_fuel_usage"] / FLOW
    assert 30.0 - crossed - 5.0 < evading < 30.0 - crossed
    r_e, v_e = np.array(observation[9:12]), np.array(observation[12:15])
    assert abs(np.cross(r_e, v_e)[2]) > 0.0 and abs(v_e[2]) > 1.0


def test_the_passive_evader_never_uses_fuel_and_runs_repeat_exactly():
    infos = [
        AgentEnvRunner(
            LineOfSightPursuer(),
            PursuitEvasionEnv,
            {"evader": NormalEscapeEvader(), "episode_timeout": 40.0},
        ).run()
        for _ in range(2)
    ]
    first, second = (run["agent_env_results"] for run in infos)
    for key in first:
        if key != "wall_s":
            assert first[key] == second[key]
    passive = AgentEnvRunner(
        LineOfSightPursuer(), PursuitEvasionEnv, {"episode_timeout": 40.0}
    ).run()
    assert passive["agent_env_results"]["evader_fuel_usage"] == 0.0


def test_an_agent_written_for_the_kspdg_interface_runs_unchanged():
    agent = NaivePursuitAgent()
    runner = AgentEnvRunner(
        agent=agent,
        env_cls=PursuitEvasionEnv,
        env_kwargs={"episode_timeout": 20.0},
        runner_timeout=100,
        debug=True,
        enable_telemetry=False,
    )
    results = runner.run()["agent_env_results"]
    for key in (
        "is_episode_done",
        "closest_approach",
        "closest_approach_time",
        "closest_approach_speed",
        "closest_approach_pursuer_fuel_usage",
        "pursuer_fuel_usage",
        "evader_fuel_usage",
        "weighted_score",
    ):
        assert key in results
    assert results["is_episode_done"] and results["steps"] == 20
    assert results["pursuer_fuel_usage"] == pytest.approx(20 * FLOW, rel=1e-12)


def test_telemetry_is_refused_with_a_warning_not_an_error():
    with pytest.warns(UserWarning, match="telemetry"):
        AgentEnvRunner(NaivePursuitAgent(), PursuitEvasionEnv, None, enable_telemetry=True)
    with pytest.raises(TypeError, match="get_action"):
        AgentEnvRunner(object(), PursuitEvasionEnv, None)


def test_none_repeats_the_last_action_and_coasts_before_the_first():
    steady = Scripted([[1.0, 0.0, 0.0, 1.0]] * 5)
    held = Scripted([[1.0, 0.0, 0.0, 1.0]])
    late = Scripted([None, None, [1.0, 0.0, 0.0, 1.0]])
    runs = [fly(PursuitEvasionEnv(episode_timeout=5.0), agent) for agent in (steady, held, late)]
    assert runs[0]["pursuer_fuel_usage"] == runs[1]["pursuer_fuel_usage"] == pytest.approx(5 * FLOW)
    assert runs[0]["closest_approach"] == runs[1]["closest_approach"]
    # Two coasts of one shortest step (0.1 s) before the first action.
    assert runs[2]["pursuer_fuel_usage"] == pytest.approx(4.8 * FLOW, rel=1e-12)


def test_the_runner_stops_at_its_wall_clock_timeout():
    results = fly(PursuitEvasionEnv(), LineOfSightPursuer(), runner_timeout=1e-9)
    assert results["steps"] == 1 and not results["is_episode_done"]


def test_the_agent_receives_a_fresh_list_and_nothing_else():
    agent = Scripted([[1.0, 0.0, 0.0, 1.0]] * 3)
    env = PursuitEvasionEnv(episode_timeout=3.0)
    fly(env, agent)
    first = agent.seen[0]
    assert type(first) is list and len(first) == 15
    snapshot = env.get_observation()
    for observation in agent.seen:
        observation[3:15] = [0.0] * 12
    assert env.get_observation() == snapshot
    assert len({id(observation) for observation in agent.seen}) == len(agent.seen)


def test_the_velocity_matching_pursuer_captures_the_passive_evader_faster_than_real_time():
    results = fly(PursuitEvasionEnv(), VelocityMatchingPursuer())
    assert results["closest_approach"] == pytest.approx(5.0, abs=1e-6)
    # The simulated minutes take seconds even on a loaded machine.
    assert results["wall_s"] < results["closest_approach_time"]


def test_an_empty_tank_stops_the_thrust_at_that_instant():
    vessel = Vessel(monopropellant=1.0)
    env = PursuitEvasionEnv(vessel=vessel)
    start = env.reset()[0]
    twin = PursuitEvasionEnv(vessel=vessel)
    twin.reset()
    end = env.step([1.0, 0.0, 0.0, 3.0])[0]
    coast = twin.step([0.0, 0.0, 0.0, 3.0])[0]
    assert end[1] == pytest.approx(vessel.dry_mass, abs=1e-9) and end[2] == pytest.approx(
        0.0, abs=1e-9
    )
    # The 4 kg last 4 / 3.399 s; the rest of the step is a coast. The twins drift about 4 m
    # apart, so gravity changes their velocity difference by about n^2 * 4 m * 3 s = 1e-4 m/s.
    dv = np.linalg.norm(np.subtract(end[6:9], coast[6:9]))
    assert dv == pytest.approx(240.0 * G0 * math.log(start[1] / vessel.dry_mass), rel=1e-4)
    assert env.step([1.0, 0.0, 0.0, 1.0])[0][1] == end[1]


@pytest.mark.parametrize(
    "build",
    [
        lambda: Vessel(dry_mass=0.0),
        lambda: Vessel(monopropellant=-1.0),
        lambda: Vessel(isp=float("nan")),
        lambda: Vessel(max_thrust_positive=(1.0, 1.0)),
        lambda: Vessel(max_thrust_negative=(1.0, -1.0, 1.0)),
        lambda: LineOfSightPursuer(burn=-1.0),
        lambda: VelocityMatchingPursuer(speed=0.0),
        lambda: VelocityMatchingPursuer(gain=float("inf")),
        lambda: NormalEscapeEvader(evade_distance=-5.0),
    ],
)
def test_bad_vessels_bots_and_pursuers_are_refused(build):
    with pytest.raises(ValueError):
        build()


def test_the_evader_stops_thrusting_when_the_pursuer_leaves_its_range():
    env = PursuitEvasionEnv(evader=NormalEscapeEvader(evade_distance=1000.0), episode_timeout=150.0)
    env.reset()
    rows, done = [], False
    while not done:
        # Forty seconds toward the evader, then a coast through its range and out again.
        throttle = 1.0 if env.t < 40.0 - 1e-9 else 0.0
        observation, _, done, info = env.step([throttle, 0.0, 0.0, 1.0])
        rows.append((observation[0], relative_range(observation), info["evader_fuel_usage"]))
    inside = [t for t, distance, _ in rows if distance < 1000.0]
    left = next(i for i, (t, distance, _) in enumerate(rows) if t > inside[-1])
    assert rows[left][1] > 1000.0 and rows[-1][1] > 1000.0
    # No burn after the range rises back through 1000 m: the mass stays exactly as it was.
    assert all(fuel == rows[left][2] for _, _, fuel in rows[left:])
    # The burn spans the crossings, each located inside a 1 s step.
    burning = rows[left][2] / FLOW
    assert inside[-1] - inside[0] < burning < inside[-1] - inside[0] + 2.0


def test_the_evader_stops_when_its_tank_runs_dry():
    vessel = Vessel(monopropellant=1.0)
    env = PursuitEvasionEnv(evader=NormalEscapeEvader(evade_distance=3000.0), vessel=vessel)
    env.reset()
    info = env.step([0.0, 0.0, 0.0, 3.0])[3]
    assert info["evader_fuel_usage"] == pytest.approx(4.0, abs=1e-9)
    assert info["pursuer_fuel_usage"] == 0.0
    assert env.step([0.0, 0.0, 0.0, 3.0])[3]["evader_fuel_usage"] == info["evader_fuel_usage"]
