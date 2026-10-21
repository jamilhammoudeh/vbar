"""LQR agent regulating to the hold point on the exact zero-order-hold CW model.

The hold-point target input and the one-step actuation delay follow Rawlings, Mayne and Diehl,
Model Predictive Control, 2nd ed., chapter 1 (sections 1.2.5 and 1.5.1, exercise 1.61); the
weight on the command change follows exercise 1.25. Every number is for a 500 km circular orbit
of the WGS 84 Earth, a 10 s step and a 0.02 m/s^2 box limit per axis unless stated.
"""

import dataclasses

import numpy as np
import pytest
from metric_cases import episode

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents import lqr
from vbar.agents.base import AgentView, agent_generator, command_fraction
from vbar.agents.lqr import LQRAgent, LQRConfig
from vbar.agents.sign_check import sign_check
from vbar.dynamics import EARTH, cw_discrete, cw_system, mean_motion
from vbar.metrics import completion_time, mission_metrics

N = mean_motion(EARTH.radius + 500e3)
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
BEHIND = (0.0, -400.0, 30.0, 0.0, 0.0, 0.0)
V_BAR_HOLD = (0.0, -100.0, 0.0)
R_BAR_HOLD = (-100.0, 0.0, 0.0)
STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")
INFO = {"time_s": 0.0}


def pending_names(slots):
    return tuple(f"pending_{k}_{c}" for k in range(slots) for c in ("t", "x", "y", "z"))


def hand_view(hold=V_BAR_HOLD, mode="normalized", limit="box", slots=0):
    """An AgentView built without an environment."""
    return AgentView(
        scenario_id="hand-made",
        dt=DT,
        max_steps=360,
        n=N,
        action_mode=mode,
        max_accel=A_MAX,
        thrust_limit=limit,
        mass_kg=500.0,
        goal={"hold_point": hold, "capture_radius": 5.0, "capture_speed": 0.05},
        constraints=({"type": "keep_out", "radius": 50.0},),
        observation=STATE_NAMES + pending_names(slots),
    )


def weights(pos_scale, vel_scale, stage_cost):
    """Q, R, S of the stated stage cost, the control scale being the view's max_accel."""
    qc = np.diag([pos_scale**-2] * 3 + [vel_scale**-2] * 3)
    rc = np.eye(3) / A_MAX**2
    if stage_cost == "discrete":
        return qc, rc, np.zeros((6, 3))
    a, b = cw_system(N)
    return lqr.sampled_data_cost(a, b, qc, rc, DT)


def acted(agent, view, obs):
    agent.reset(view, np.random.default_rng(0))
    return command_fraction(view, agent.act(np.asarray(obs, dtype=float), dict(INFO)))


def test_config_is_frozen_and_keyword_only():
    with pytest.raises(TypeError):
        LQRConfig(10.0)
    config = LQRConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.pos_scale_m = 1.0
    assert (config.pos_scale_m, config.vel_scale_mps, config.accel_scale_mps2) == (10.0, 0.01, None)
    assert (config.stage_cost, config.rate_weight) == ("sampled", 0.0)


def test_alfriend_weights_make_one_radian_of_orbit_cost_one_unit():
    config = LQRConfig.alfriend(N)
    assert config.pos_scale_m == 1.0
    assert config.vel_scale_mps == N
    assert config.accel_scale_mps2 == N**2


@pytest.mark.parametrize("stage_cost", ["discrete", "sampled"])
@pytest.mark.parametrize("hold", [V_BAR_HOLD, R_BAR_HOLD])
def test_command_is_the_lqr_law_with_target_input(stage_cost, hold):
    phi, gamma = cw_discrete(N, DT)
    q, r, s = weights(10.0, 0.01, stage_cost)
    k, _ = lqr.dlqr(phi, gamma, q, r, s)
    xs = np.array([*hold, 0.0, 0.0, 0.0])
    us = lqr.steady_state_input(phi, gamma, xs)
    view = hand_view(hold=hold)
    rng = np.random.default_rng(5)
    for _ in range(20):
        # Within 1 m and 1 mm/s of the hold point the law stays below the box limit.
        x = xs + rng.uniform(-1.0, 1.0, 6) * np.array([1.0] * 3 + [1e-3] * 3)
        expected = (us - k @ (x - xs)) / A_MAX
        assert np.max(np.abs(expected)) < 1.0
        command = acted(LQRAgent(LQRConfig(stage_cost=stage_cost)), view, x)
        # The same Riccati solution reached through the agent's own assembly of the matrices;
        # its residual is at 1e-14 of P.
        assert np.max(np.abs(command - expected)) <= 1e-9 * np.max(np.abs(expected))


@pytest.mark.parametrize("stage_cost", ["discrete", "sampled"])
def test_commands_are_equal_in_fractions_in_every_continuous_mode(stage_cost):
    views = {
        mode: ApproachEnv(ApproachConfig(action_mode=mode)).agent_view()
        for mode in ("normalized", "accel", "thrust")
    }
    rng = np.random.default_rng(3)
    # Near the hold point the law is linear; far from it the commands saturate.
    for spread in (1.0, 1e3):
        for _ in range(10):
            obs = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
            obs += rng.normal(size=6) * spread * np.array([1.0] * 3 + [1e-3] * 3)
            fractions = {
                mode: acted(LQRAgent(LQRConfig(stage_cost=stage_cost)), view, obs)
                for mode, view in views.items()
            }
            ref = fractions["normalized"]
            for mode in ("accel", "thrust"):
                assert np.max(np.abs(fractions[mode] - ref)) <= 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_discrete_and_impulsive_action_modes_are_refused_at_reset(mode):
    view = ApproachEnv(ApproachConfig(action_mode=mode)).agent_view()
    with pytest.raises(ValueError, match="takes no continuous command"):
        LQRAgent().reset(view, np.random.default_rng(0))


def test_box_commands_are_clipped_per_axis_and_ball_commands_keep_their_direction():
    xs = np.array([*V_BAR_HOLD, 0.0, 0.0, 0.0])
    delta = np.array([30.0, -40.0, 20.0, 0.01, 0.02, -0.01])
    config = LQRConfig(stage_cost="discrete")
    # At the V-bar hold the target input is zero, so the law is linear in the deviation.
    near = acted(LQRAgent(config), hand_view(limit="ball"), xs + 1e-3 * delta)
    assert np.linalg.norm(near) < 1.0
    far_box = acted(LQRAgent(config), hand_view(limit="box"), xs + 1e3 * delta)
    assert np.max(np.abs(far_box)) == pytest.approx(1.0, abs=1e-15)
    far_ball = acted(LQRAgent(config), hand_view(limit="ball"), xs + 1e3 * delta)
    assert np.linalg.norm(far_ball) == pytest.approx(1.0, rel=1e-12)
    assert np.max(np.abs(far_ball - near / np.linalg.norm(near))) <= 1e-9


def test_hold_point_that_needs_more_than_full_thrust_is_refused():
    # 3 n^2 10 km = 3.67e-2 m/s^2, above the 2e-2 m/s^2 limit.
    with pytest.raises(ValueError, match="above max_accel"):
        LQRAgent().reset(hand_view(hold=(-1e4, 0.0, 0.0)), np.random.default_rng(0))


R_BAR_GOAL = {"hold_point": R_BAR_HOLD, "capture_radius": 5.0, "capture_speed": 0.05}


def r_bar_hold_episode(agent, steps):
    # The custom goal keeps the episode running past capture to its step limit.
    start = (*R_BAR_HOLD, 0.0, 0.0, 0.0)
    return episode(
        agent, start_state=start, hold_point=R_BAR_HOLD, max_steps=steps, goal=R_BAR_GOAL
    )


@pytest.mark.parametrize("stage_cost", ["discrete", "sampled"])
def test_target_input_holds_the_r_bar_point_to_rounding(stage_cost):
    ep = r_bar_hold_episode(LQRAgent(LQRConfig(stage_cost=stage_cost)), 360)
    # An independent computation gives 3.0e-13 m (discrete) and 9.9e-14 m (sampled) after
    # 360 steps; the bound sits six decades above.
    assert mission_metrics(ep)["terminal_pos_error"] < 1e-6


def test_regulating_without_target_input_settles_at_the_known_offset(monkeypatch):
    monkeypatch.setattr(lqr, "steady_state_input", lambda phi, gamma, xs: np.zeros(3))
    ep = r_bar_hold_episode(LQRAgent(LQRConfig(stage_cost="discrete")), 1080)
    # Fixed point of the closed loop: (-3.702, 0.041, 0) m from the hold point. Its slowest
    # mode needs about three hours: an independent computation gives 3.601 m at 360 steps and
    # 3.7026 m from 1080 steps on.
    assert mission_metrics(ep)["terminal_pos_error"] == pytest.approx(3.702, abs=1e-2)


class UndelayedLQR:
    """The same gains designed as if each command acted at once; ignores the pending channel."""

    name = "undelayed-lqr"

    def __init__(self, pos_scale, vel_scale):
        self.q = np.diag([pos_scale**-2] * 3 + [vel_scale**-2] * 3)

    def reset(self, view, rng=None):
        phi, gamma = cw_discrete(view.n, view.dt)
        self.k, _ = lqr.dlqr(phi, gamma, self.q, np.eye(3) / view.max_accel**2)
        self.xs = np.array([*view.goal["hold_point"], 0.0, 0.0, 0.0])
        self.us = lqr.steady_state_input(phi, gamma, self.xs)
        self.max_accel = view.max_accel

    def act(self, obs, info):
        u = self.us - self.k @ (obs[:6] - self.xs)
        return np.clip(u / self.max_accel, -1.0, 1.0)


DELAYED = {"delay_s": 10.0, "observation_channels": ("pending_commands",)}


def test_delay_augmented_design_captures_where_the_undelayed_design_does_not():
    # Scales (1 m, 1 m/s): the undelayed design's loop with a one-step delay has spectral
    # radius 1.3921. An independent computation captures at 1360 s with the augmented design
    # and leaves the undelayed one 280 m from the hold point after 300 steps.
    config = LQRConfig(pos_scale_m=1.0, vel_scale_mps=1.0, stage_cost="discrete")
    augmented = episode(LQRAgent(config), start_state=BEHIND, max_steps=300, **DELAYED)
    assert completion_time(augmented) is not None
    undelayed = episode(UndelayedLQR(1.0, 1.0), start_state=BEHIND, max_steps=300, **DELAYED)
    assert completion_time(undelayed) is None


def test_delay_that_is_not_a_whole_number_of_steps_is_refused():
    channels = DELAYED["observation_channels"]
    env = ApproachEnv(
        ApproachConfig(start_state=BEHIND, delay_s=15.0, observation_channels=channels)
    )
    obs, _ = env.reset(seed=0)
    agent = LQRAgent()
    agent.reset(env.agent_view(), agent_generator(1))
    # Nothing is pending at the first decision, so every wait is zero; the first command's
    # 5 s wait appears at the second.
    with pytest.raises(ValueError, match="not aligned to the decision step"):
        for _ in range(2):
            obs, *_ = env.step(agent.act(obs, env.agent_info()))


def test_pending_wait_off_the_step_grid_is_refused_on_a_hand_made_observation():
    view = hand_view(slots=2)
    agent = LQRAgent()
    agent.reset(view, np.random.default_rng(0))
    state = [0.0, -150.0, 0.0, 0.0, 0.0, 0.0]
    aligned = np.array([*state, 0.0, 0.1, 0.0, 0.0, 10.0, 0.2, 0.0, 0.0])
    assert agent.act(aligned, dict(INFO)).shape == (3,)
    off_grid = np.array([*state, 0.0, 0.1, 0.0, 0.0, 5.0, 0.2, 0.0, 0.0])
    with pytest.raises(ValueError, match="not aligned to the decision step"):
        agent.act(off_grid, dict(INFO))


def test_one_step_delay_command_is_the_augmented_lqr_law_in_every_mode():
    phi, gamma = cw_discrete(N, DT)
    q, r, _ = weights(10.0, 0.01, "discrete")
    phi_a, gamma_a = lqr.delay_augmented(phi, gamma, 1)
    q_a = np.zeros((9, 9))
    q_a[:6, :6] = q
    k_a, _ = lqr.dlqr(phi_a, gamma_a, q_a, r)
    xs = np.array([*V_BAR_HOLD, 0.0, 0.0, 0.0])
    x = xs + np.array([0.4, -0.7, 0.3, 2e-4, -1e-4, 3e-4])
    in_effect = np.array([0.03, -0.02, 0.01])  # fractions of a full command
    expected = -k_a @ np.concatenate([x - xs, in_effect * A_MAX]) / A_MAX
    assert np.max(np.abs(expected)) < 1.0
    config = LQRConfig(stage_cost="discrete")
    for mode, scale in (("normalized", 1.0), ("accel", A_MAX), ("thrust", A_MAX * 500.0)):
        obs = np.concatenate([x, [0.0], in_effect * scale])
        command = acted(LQRAgent(config), hand_view(mode=mode, slots=1), obs)
        # Same Riccati solution assembled by the agent; residual at 1e-14 of P.
        assert np.max(np.abs(command - expected)) <= 1e-9 * np.max(np.abs(expected))


def test_rate_weight_makes_the_command_depend_on_the_previous_one():
    view = hand_view()
    obs = np.array([0.5, -100.4, 0.2, 1e-4, 0.0, -1e-4])
    for rate, same in ((0.0, True), (0.1, False)):
        agent = LQRAgent(LQRConfig(rate_weight=rate))
        agent.reset(view, np.random.default_rng(0))
        first = agent.act(obs, dict(INFO))
        second = agent.act(obs, dict(INFO))
        assert np.array_equal(first, second) is same


def test_rate_weight_lowers_the_summed_command_change():
    changes = []
    for rate in (0.0, 0.1):
        ep = episode(LQRAgent(LQRConfig(rate_weight=rate)), start_state=BEHIND, max_steps=200)
        u = np.array([rec.command for rec in ep.records])
        changes.append(np.sum(np.diff(u, axis=0) ** 2))
    # An independent computation gives 0.88132 without and 0.88086 with the weight
    # (fractions squared); the gap is 5e-4 relative, far above rounding.
    assert changes[1] < changes[0]


@pytest.mark.parametrize(
    "config",
    [
        LQRConfig(stage_cost="discrete"),
        LQRConfig(stage_cost="sampled"),
        LQRConfig(rate_weight=0.1),
    ],
)
@pytest.mark.parametrize("fields", [{}, {"action_mode": "accel"}, DELAYED])
def test_lqr_agent_passes_the_sign_check(config, fields):
    assert sign_check(LQRAgent(config), ApproachConfig(**fields)).passed


A_START = (60.0, -300.0, -20.0, 0.01, 0.0, 0.0)
B_START = BEHIND


def actions(agent, start, **fields):
    env = ApproachEnv(ApproachConfig(start_state=start, max_steps=60, **fields))
    return run_episode(env, agent, seed=11, agent_seed=12)["actions"]


def test_reset_wipes_the_previous_episode():
    config = LQRConfig(rate_weight=0.1)
    agent = LQRAgent(config)
    actions(agent, A_START, **DELAYED)
    reused = actions(agent, B_START)
    fresh = actions(LQRAgent(config), B_START)
    assert np.array_equal(reused, fresh)


def test_two_agents_from_one_config_on_interleaved_environments_match_separate_runs():
    config = LQRConfig(rate_weight=0.1)
    starts = (A_START, B_START)
    envs = [ApproachEnv(ApproachConfig(start_state=s, max_steps=60)) for s in starts]
    agents = [LQRAgent(config), LQRAgent(config)]
    obs, logs = [], [[], []]
    for env, agent in zip(envs, agents, strict=True):
        o, _ = env.reset(seed=11)
        agent.reset(env.agent_view(), agent_generator(12))
        obs.append(o)
    for _ in range(60):
        for i, (env, agent) in enumerate(zip(envs, agents, strict=True)):
            a = agent.act(obs[i], env.agent_info())
            logs[i].append(a)
            obs[i], *_ = env.step(a)
    for i, start in enumerate(starts):
        env = ApproachEnv(ApproachConfig(start_state=start, max_steps=60))
        o, _ = env.reset(seed=11)
        agent = LQRAgent(config)
        agent.reset(env.agent_view(), agent_generator(12))
        alone = []
        for _ in range(60):
            a = agent.act(o, env.agent_info())
            alone.append(a)
            o, *_ = env.step(a)
        assert np.array_equal(np.array(logs[i]), np.array(alone))


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


@pytest.mark.parametrize("slots", [0, 1])
def test_agent_runs_on_a_hand_made_view_and_reads_only_the_time(slots):
    view = hand_view(slots=slots)
    obs = np.array([5.0, -180.0, 3.0, 0.01, 0.0, -0.002] + [0.0, 0.1, -0.1, 0.0] * slots)
    agent = LQRAgent(LQRConfig(rate_weight=0.1))
    agent.reset(view)
    plain = agent.act(obs, {"time_s": 0.0})
    agent.reset(view)
    poisoned = ReadLog(time_s=0.0, true_state=np.full(6, 1e6), dv_used_mps=1e6)
    command = agent.act(obs, poisoned)
    assert poisoned.read <= {"time_s"}
    assert np.array_equal(command, plain)
    assert command.shape == (3,) and np.all(np.isfinite(command))
