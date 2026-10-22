"""Mutation wrapper: deliberately broken versions of any continuous-command agent.

Each kind changes the observation before the inner agent or its command after, in fractions of a
full command. The table tests check that each metric family detects its mutant and, where two
severities of one kind exist, ranks them in order. Filter metrics judge a safety filter rather
than the agent, so no agent mutant is ranked by them.
"""

import numpy as np
import pytest
from metric_cases import episode

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import PDAgent
from vbar.agents.base import AgentView, command_fraction
from vbar.agents.lqr import LQRAgent
from vbar.agents.mutants import MUTANT_KINDS, Mutant
from vbar.agents.sign_check import sign_check
from vbar.dynamics import EARTH, mean_motion
from vbar.metrics import efficiency_metrics, mission_metrics, safety_metrics

N = mean_motion(EARTH.radius + 500e3)
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
BEHIND = (0.0, -400.0, 30.0, 0.0, 0.0, 0.0)
GOAL = {"hold_point": (0.0, -100.0, 0.0), "capture_radius": 5.0, "capture_speed": 0.05}
INFO = {"time_s": 0.0}


def hand_view(mode="normalized"):
    """An AgentView built without an environment."""
    return AgentView(
        scenario_id="hand-made",
        dt=DT,
        max_steps=360,
        n=N,
        action_mode=mode,
        max_accel=A_MAX,
        thrust_limit="box",
        mass_kg=500.0,
        goal=GOAL,
        constraints=({"type": "keep_out", "radius": 50.0},),
    )


def observations(count=6):
    """Observations near the hold point, where PD commands stay far inside the limit."""
    rng = np.random.default_rng(4)
    hold = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    return [hold + rng.normal(size=6) * [1.0, 1.0, 1.0, 0.05, 0.05, 0.05] for _ in range(count)]


def fractions(agent, obs_list, mode="normalized", rng=None):
    view = hand_view(mode)
    agent.reset(view, rng)
    return np.array([command_fraction(view, agent.act(o.copy(), dict(INFO))) for o in obs_list])


def pd_fractions(obs_list):
    return fractions(PDAgent(), obs_list)


def close(a, b):
    # A few operations on fractions of order 1e-2: rounding stays below 1e-12 of the largest.
    return np.max(np.abs(a - b)) <= 1e-12 * max(np.max(np.abs(b)), 1e-3)


def test_kinds_are_the_published_set():
    assert MUTANT_KINDS == (
        "position_bias",
        "velocity_bias",
        "delay",
        "sign_flip",
        "dither",
        "chatter",
        "radial_sign",
        "no_coriolis",
        "bang_bang",
    )


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        Mutant(PDAgent(), "gain_scale", 1.0)


@pytest.mark.parametrize(
    ("kind", "axis", "severity"), [("position_bias", 0, 2.0), ("velocity_bias", 3, 0.01)]
)
def test_navigation_bias_shifts_what_the_inner_agent_sees(kind, axis, severity):
    obs = observations()
    shifted = [o.copy() for o in obs]
    for o in shifted:
        o[axis] += severity
    assert close(fractions(Mutant(PDAgent(), kind, severity), obs), pd_fractions(shifted))


def test_delay_feeds_the_observation_from_severity_steps_before():
    obs = observations()
    late = [obs[max(0, k - 2)] for k in range(len(obs))]
    assert close(fractions(Mutant(PDAgent(), "delay", 2), obs), pd_fractions(late))


def test_sign_flip_negates_the_command_and_radial_sign_only_its_radial_axis():
    obs = observations()
    pd = pd_fractions(obs)
    assert close(fractions(Mutant(PDAgent(), "sign_flip", 1), obs), -pd)
    radial = pd * np.array([-1.0, 1.0, 1.0])
    assert close(fractions(Mutant(PDAgent(), "radial_sign", 1), obs), radial)


def test_bang_bang_issues_the_severity_with_the_command_sign_on_each_axis():
    obs = observations()
    expected = 0.3 * np.sign(pd_fractions(obs))
    assert np.array_equal(fractions(Mutant(PDAgent(), "bang_bang", 0.3), obs), expected)


def test_no_coriolis_adds_back_the_coriolis_acceleration():
    obs = observations()
    coriolis = np.array([[2 * N * o[4], -2 * N * o[3], 0.0] for o in obs]) / A_MAX
    got = fractions(Mutant(PDAgent(), "no_coriolis", 0.5), obs)
    assert close(got, pd_fractions(obs) + 0.5 * coriolis)


def test_chatter_alternates_by_its_amplitude_on_the_radial_axis():
    obs = observations()
    change = fractions(Mutant(PDAgent(), "chatter", 0.02), obs) - pd_fractions(obs)
    assert np.max(np.abs(change[:, 1:])) == 0.0
    assert np.allclose(np.abs(change[:, 0]), 0.02, rtol=1e-12, atol=0.0)
    assert np.all(np.sign(change[1:, 0]) == -np.sign(change[:-1, 0]))


def test_dither_levels_are_scaled_copies_of_one_noise_sequence():
    obs = observations()
    pd = pd_fractions(obs)
    steps = [
        (fractions(Mutant(PDAgent(), "dither", s), obs, rng=np.random.default_rng(9)) - pd) / s
        for s in (0.01, 0.03)
    ]
    assert np.min(np.abs(steps[0])) > 0.0
    # Removing the PD command from each level leaves rounding near 1e-15 of the draws.
    assert np.max(np.abs(steps[1] - steps[0])) <= 1e-12 * np.max(np.abs(steps[0]))


def test_dither_without_a_generator_draws_from_its_noise_seed():
    obs = observations()
    same = [fractions(Mutant(PDAgent(), "dither", 0.05, noise_seed=3), obs) for _ in range(2)]
    other = fractions(Mutant(PDAgent(), "dither", 0.05, noise_seed=4), obs)
    assert np.array_equal(same[0], same[1])
    assert not np.array_equal(same[0], other)


def dither_actions(seed, agent_seed, steps=5):
    env = ApproachEnv(ApproachConfig(start_state=BEHIND, max_steps=steps))
    agent = Mutant(PDAgent(), "dither", 0.05)
    return run_episode(env, agent, seed=seed, agent_seed=agent_seed)["actions"]


def test_dither_follows_the_agent_seed_and_not_the_episode_seed():
    assert np.array_equal(dither_actions(0, 12), dither_actions(0, 12))
    # The start state is fixed and navigation exact, so only the draws can differ.
    assert np.array_equal(dither_actions(0, 12)[0], dither_actions(5, 12)[0])
    assert not np.array_equal(dither_actions(0, 12)[0], dither_actions(0, 13)[0])


@pytest.mark.parametrize("kind", MUTANT_KINDS)
def test_commands_are_equal_in_fractions_in_every_continuous_mode(kind):
    obs = observations()
    ref = fractions(Mutant(PDAgent(), kind, 0.5), obs, rng=np.random.default_rng(9))
    for mode in ("accel", "thrust"):
        got = fractions(Mutant(PDAgent(), kind, 0.5), obs, mode, rng=np.random.default_rng(9))
        # One division and one multiplication by the action scale per command.
        assert np.max(np.abs(got - ref)) <= 1e-12 * np.max(np.abs(ref))


class Fixed:
    """Accepts any view and returns the same array object at every decision."""

    name = "fixed"

    def __init__(self):
        self.command = np.array([0.2, -0.1, 0.05])

    def reset(self, view, rng=None):
        pass

    def act(self, obs, info):
        return self.command


@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_discrete_and_impulsive_action_modes_are_refused_by_the_wrapper(mode):
    with pytest.raises(ValueError, match="takes no continuous command"):
        Mutant(Fixed(), "sign_flip", 1).reset(hand_view(mode), np.random.default_rng(0))


@pytest.mark.parametrize("kind", MUTANT_KINDS)
def test_neither_the_observation_nor_the_inner_command_is_changed_in_place(kind):
    inner = Fixed()
    agent = Mutant(inner, kind, 0.5)
    agent.reset(hand_view(), np.random.default_rng(0))
    for obs in observations(4):
        given = obs.copy()
        agent.act(given, dict(INFO))
        assert np.array_equal(given, obs)
    assert np.array_equal(inner.command, [0.2, -0.1, 0.05])


A_START = (60.0, -300.0, -20.0, 0.01, 0.0, 0.0)


def actions(agent, start):
    env = ApproachEnv(ApproachConfig(start_state=start, max_steps=40))
    return run_episode(env, agent, seed=11, agent_seed=12)["actions"]


@pytest.mark.parametrize(("kind", "severity"), [("delay", 3), ("chatter", 0.1), ("dither", 0.1)])
def test_reset_wipes_the_previous_episode(kind, severity):
    agent = Mutant(PDAgent(), kind, severity)
    actions(agent, A_START)
    assert np.array_equal(
        actions(agent, BEHIND), actions(Mutant(PDAgent(), kind, severity), BEHIND)
    )


class ReadLog(dict):
    """An info mapping that records which keys are read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


@pytest.mark.parametrize("kind", MUTANT_KINDS)
def test_wrapper_reads_only_the_time(kind):
    agent = Mutant(PDAgent(), kind, 0.5)
    agent.reset(hand_view(), np.random.default_rng(0))
    for obs in observations(3):
        info = ReadLog(time_s=0.0, true_state=np.full(6, 1e6), dv_used_mps=1e6)
        agent.act(obs, info)
        assert info.read <= {"time_s"}


def run(agent, steps=200):
    return episode(agent, start_state=BEHIND, max_steps=steps)


def increasing(values):
    return all(a < b for a, b in zip(values, values[1:], strict=False))


def test_mission_metrics_catch_a_flipped_radial_command():
    pd = mission_metrics(run(PDAgent()))
    flipped = mission_metrics(run(Mutant(PDAgent(), "radial_sign", 1)))
    # An independent computation gives 74523 m against the PD's 4.677 m.
    assert pd["success"] and not flipped["success"]
    assert flipped["terminal_pos_error"] > pd["terminal_pos_error"]


def test_dropping_the_coriolis_cancellation_is_detected():
    # Detection only: the Coriolis term does no work, so no ordering is proven, and the terminal
    # error is not monotone in the severity (4.677, 4.810, 4.748 m at 0, 0.5, 1.0 in an
    # independent computation). Capture comes at 670 s against 620 s.
    pd = run(PDAgent())
    mutant = run(Mutant(PDAgent(), "no_coriolis", 1.0))
    assert mission_metrics(mutant)["time_to_complete"] > mission_metrics(pd)["time_to_complete"]
    assert efficiency_metrics(mutant)["delta_v"] > efficiency_metrics(pd)["delta_v"]


def test_bang_bang_raises_chatter_and_delta_v_with_its_amplitude():
    eps = [run(PDAgent())] + [run(Mutant(PDAgent(), "bang_bang", s)) for s in (0.1, 0.3, 1.0)]
    found = [efficiency_metrics(ep) for ep in eps]
    # An independent computation: chatter 0.0032 (PD) against 0.144, 0.226, 0.230 per s; delta_v
    # 2.427, 2.696, 9.958, 32.50 m/s.
    assert all(m["chatter"] > found[0]["chatter"] for m in found[1:])
    assert increasing([m["delta_v"] for m in found])


def test_dither_lowers_the_keep_out_margin_with_its_size():
    margins = [
        safety_metrics(run(Mutant(PDAgent(), "dither", s)))["min_margin"]["keep_out"]
        for s in (0.0, 0.15, 0.45)
    ]
    # An independent computation gives 54.66, 36.89, 13.85 m at agent seed 1, strictly
    # decreasing at agent seeds 1 to 8 with the smallest step 2.2 m. A dither of 0.05 is left
    # out: it moves the margin by +0.2 to -3.3 m depending on the draw, inside the seed noise.
    # Three 200-step episodes with their safety metrics take about 2.2 s; fewer steps would end
    # before the closest pass (the 0.45 margin is 39.45, 18.16 and 13.85 m at 100, 150, 200 steps).
    assert increasing(margins[::-1])


def test_a_long_decision_delay_enters_the_keep_out_zone():
    assert safety_metrics(run(PDAgent()))["min_margin"]["keep_out"] > 0.0
    # An independent computation gives -5.04 m at six steps of delay.
    assert safety_metrics(run(Mutant(PDAgent(), "delay", 6)))["min_margin"]["keep_out"] < 0.0


def test_sign_check_fails_flipped_axes_and_passes_the_other_kinds():
    failures = {
        kind: sign_check(Mutant(PDAgent(), kind, s), ApproachConfig()).failures
        for kind, s in (
            ("sign_flip", 1),
            ("radial_sign", 1),
            ("no_coriolis", 1.0),
            ("bang_bang", 0.3),
        )
    }
    assert sign_check(PDAgent(), ApproachConfig()).passed
    assert set(failures["sign_flip"]) == {"+x", "-x", "+y", "-y", "+z", "-z"}
    assert set(failures["radial_sign"]) == {"+x", "-x"}
    # From rest the velocity is zero and the command keeps its sign.
    assert failures["no_coriolis"] == () and failures["bang_bang"] == ()


def test_wrapper_breaks_an_lqr_agent_that_succeeds():
    # The LQR at default weights captures from 50 m behind the hold point at 2310 s (an
    # independent computation). Two full episodes and their metrics take about 1.3 s, most of it
    # the flipped run's far trajectory.
    near = (0.0, -150.0, 0.0, 0.0, 0.0, 0.0)
    assert mission_metrics(episode(LQRAgent(), start_state=near, max_steps=240))["success"]
    flipped = episode(Mutant(LQRAgent(), "sign_flip", 1), start_state=near, max_steps=240)
    assert not mission_metrics(flipped)["success"]
    assert mission_metrics(flipped)["progress"] == 0.0
