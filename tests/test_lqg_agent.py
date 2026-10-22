"""LQG agent: a steady-state Kalman filter feeding the LQR hold-point regulator.

The filter and its Joseph-form covariance update follow Crassidis and Junkins, Optimal Estimation
of Dynamic Systems, 2nd ed., sections 3.3.2 and 3.3.4; the update adds the gain times the
innovation y - C xhat (Alfriend et al., Spacecraft Formation Flying, print a minus sign in their
filter update; the printed sign is a misprint). Bias detectability is Lemma 1.8 of Rawlings,
Mayne and Diehl, Model Predictive Control, 2nd ed. Every number is for a 500 km circular orbit
of the WGS 84 Earth, a 10 s step and a 0.02 m/s^2 box limit per axis unless stated.
"""

import dataclasses
import functools

import numpy as np
import pytest
from metric_cases import GuardedApproach, episode

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.base import AgentView, agent_generator, command_fraction
from vbar.agents.lqg import LQGAgent, LQGConfig, kalman_gain
from vbar.agents.lqr import LQRAgent, LQRConfig
from vbar.agents.sign_check import sign_check
from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.metrics import completion_time
from vbar.spacecraft import GaussianNav, PerfectNav

N = mean_motion(EARTH.radius + 500e3)
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
BEHIND = (0.0, -400.0, 30.0, 0.0, 0.0, 0.0)
V_BAR_HOLD = (0.0, -100.0, 0.0)
GOAL = {"hold_point": V_BAR_HOLD, "capture_radius": 5.0, "capture_speed": 0.05}
STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")
INFO = {"time_s": 0.0}
ORBIT = 568  # steps of 10 s; the period at 500 km is 5677 s
DELAYED = {"delay_s": 10.0, "observation_channels": ("pending_commands",)}


def pending_names(slots):
    return tuple(f"pending_{k}_{c}" for k in range(slots) for c in ("t", "x", "y", "z"))


def hand_view(mode="normalized", slots=0):
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
        observation=STATE_NAMES + pending_names(slots),
    )


def filtered_run(config, sensor, steps, seed):
    """Fly the agent past capture; per decision, the estimate's position error and its NEES.

    The test reads the true state from the environment; the agent never sees it.
    """
    env = GuardedApproach(
        ApproachConfig(start_state=BEHIND, max_steps=steps, sensor=sensor), goal=GOAL
    )
    obs, _ = env.reset(seed=seed)
    agent = LQGAgent(config)
    agent.reset(env.agent_view(), agent_generator(seed + 1))
    errors, nees = [], []
    for _ in range(steps):
        command = agent.act(obs, env.agent_info())
        e = agent.estimate[:6] - env.state
        errors.append(np.linalg.norm(e[:3]))
        nees.append(e @ np.linalg.solve(agent.covariance[:6, :6], e))
        obs, *_ = env.step(command)
    return agent, env, np.array(errors), np.array(nees)


def test_config_is_frozen_and_keyword_only():
    with pytest.raises(TypeError):
        LQGConfig(1.0)
    config = LQGConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.sigma_pos_m = 2.0
    assert (config.sigma_pos_m, config.sigma_vel_mps, config.process_sigma_mps2) == (
        1.0,
        1e-3,
        1e-6,
    )
    assert (config.bias_states, config.bias_walk, config.regulator) == ((), 1e-3, LQRConfig())
    assert LQGAgent().name == "lqg"


def filter_model(bias_states, sigma_pos=1.0, sigma_vel=1e-3, process=1e-6, walk=1e-3):
    """Phi, Gamma, C, Qd and Rv of the filter with measurement biases on the named channels."""
    phi, gamma = cw_discrete(N, DT)
    nd = len(bias_states)
    sigma = np.array([sigma_pos] * 3 + [sigma_vel] * 3)
    phi_a = np.eye(6 + nd)
    phi_a[:6, :6] = phi
    gamma_a = np.vstack([gamma, np.zeros((nd, 3))])
    c_a = np.hstack([np.eye(6), np.zeros((6, nd))])
    qd_a = np.zeros((6 + nd, 6 + nd))
    qd_a[:6, :6] = process**2 * gamma @ gamma.T
    for j, name in enumerate(bias_states):
        channel = STATE_NAMES.index(name)
        c_a[channel, 6 + j] = 1.0
        qd_a[6 + j, 6 + j] = (walk * sigma[channel]) ** 2
    return phi_a, gamma_a, c_a, qd_a, np.diag(sigma**2)


@pytest.mark.parametrize("bias_states", [(), ("x",), ("vz",)])
def test_update_adds_the_gain_times_the_innovation(bias_states):
    phi_a, gamma_a, c_a, qd_a, rv = filter_model(bias_states)
    gain, _, p_plus = kalman_gain(phi_a, c_a, qd_a, rv)
    agent = LQGAgent(LQGConfig(bias_states=bias_states))
    agent.reset(hand_view(), np.random.default_rng(0))
    y0 = np.array([3.0, -180.0, 4.0, 0.02, -0.01, 0.005])
    u0 = agent.act(y0, dict(INFO)) * A_MAX
    start = np.concatenate([y0, np.zeros(len(bias_states))])
    # The filter starts at the first measurement with zero biases, so its innovation is zero.
    assert np.max(np.abs(agent.estimate - start)) <= 1e-12 * np.max(np.abs(start))
    y1 = np.array([5.0, -175.0, 1.0, 0.01, 0.03, -0.02])
    agent.act(y1, dict(INFO))
    predicted = phi_a @ start + gamma_a @ u0
    expected = predicted + gain @ (y1 - c_a @ predicted)
    # Each entry in units of its channel's sigma, so positions and velocities weigh alike; the
    # same Riccati solution reached through the agent's own assembly differs at rounding level.
    sigma = np.sqrt(np.diag(rv))
    scale = np.concatenate([sigma, [sigma[STATE_NAMES.index(name)] for name in bias_states]])
    off = (agent.estimate - expected) / scale
    assert np.max(np.abs(off)) <= 1e-9 * np.max(np.abs(expected / scale))
    assert np.max(np.abs(agent.covariance - p_plus)) <= 1e-9 * np.max(np.abs(p_plus))


@functools.cache
def three_orbits(seed):
    """Position errors and NEES of the default filter over three orbits, shared by two tests."""
    # 1704 steps, about 2.5 s: the NEES window is the last two orbits and the first orbit is
    # left for the filter to settle from the 400 m start, so fewer orbits drop the claim.
    _, _, errors, nees = filtered_run(LQGConfig(), GaussianNav(1.0, 1e-3), 3 * ORBIT, seed)
    return errors, nees


@pytest.mark.parametrize("seed", [0, 4])
def test_estimate_error_over_the_last_orbit_is_well_inside_the_sensor_noise(seed):
    errors, _ = three_orbits(seed)
    rms = np.sqrt(np.mean(errors[-ORBIT:] ** 2))
    # An independent computation gives 0.111 m (seed 0) and 0.208 m (seed 4); the raw fix is
    # off by 1.73 m RMS (three axes of 1 m), and with the innovation sign flipped the estimate
    # diverges to 1e23 m.
    assert rms < 1.0


@pytest.mark.parametrize("seed", [0, 4])
def test_normalized_estimation_error_is_at_most_its_chi_square_mean(seed):
    _, nees = three_orbits(seed)
    last = nees[-2 * ORBIT :]
    # Six states: the mean NEES of a consistent filter is 6 with standard deviation
    # sqrt(12 / N). The true process noise is zero, so this filter is conservative: an
    # independent computation gives 3.79 (seed 0) and 4.62 (seed 4).
    assert np.mean(last) <= 6.0 + 3.0 * np.sqrt(12.0 / len(last))


def test_radial_measurement_bias_is_estimated_within_half_a_metre():
    # At the default walk of 1e-3 the steady-state gain recovers the bias with a time constant
    # of about 1.5 orbits (0.80 of it after three); at 1e-2 an independent computation leaves
    # 0.109 m of a drawn 7.22 m bias after three orbits (seed 0). 1704 steps, about 2.5 s; the
    # claim is the estimate after three orbits.
    config = LQGConfig(bias_states=("x",), bias_walk=1e-2)
    agent, env, _, _ = filtered_run(config, GaussianNav(1.0, 1e-3, bias_pos=5.0), 3 * ORBIT, 0)
    assert agent.estimate.shape == (7,)
    assert abs(agent.estimate[6] - env.sensor.bias[0]) < 0.5


def test_along_track_position_bias_is_refused():
    # A constant y offset is itself a CW equilibrium, so it cannot be told from the state.
    with pytest.raises(ValueError, match="along-track position bias is not detectable"):
        LQGAgent(LQGConfig(bias_states=("y",)))


def test_unknown_bias_channel_is_refused():
    with pytest.raises(ValueError):
        LQGAgent(LQGConfig(bias_states=("w",)))


@pytest.mark.parametrize(
    "config",
    [LQGConfig(process_sigma_mps2=0.0), LQGConfig(bias_states=("x",), bias_walk=0.0)],
)
def test_zero_process_noise_is_refused(config):
    with pytest.raises(ValueError, match="process noise must be positive"):
        LQGAgent(config).reset(hand_view(), np.random.default_rng(0))


@pytest.mark.parametrize("stage_cost", ["discrete", "sampled"])
def test_filter_that_trusts_exact_measurements_flies_the_lqr_commands(stage_cost):
    regulator = LQRConfig(stage_cost=stage_cost)
    config = LQGConfig(sigma_pos_m=1e-9, sigma_vel_mps=1e-9, regulator=regulator)
    env = ApproachEnv(ApproachConfig(start_state=BEHIND, max_steps=200, sensor=PerfectNav()))
    lqg = run_episode(env, LQGAgent(config), seed=0, agent_seed=1)["actions"]
    lqr = run_episode(env, LQRAgent(regulator), seed=0, agent_seed=1)["actions"]
    assert lqg.shape == lqr.shape
    assert np.max(np.abs(lqg[1:] - lqr[1:])) <= 1e-9 * np.max(np.abs(lqr[1:]))


FAST = LQRConfig(pos_scale_m=1.0, vel_scale_mps=1.0, stage_cost="discrete")


@pytest.mark.parametrize("sensor", [PerfectNav(), GaussianNav(1.0, 1e-3)])
def test_predicting_with_the_command_in_effect_captures_under_a_one_step_delay(sensor):
    # An independent computation captures at 1360 s (exact navigation) and 1450 s (seed 0);
    # predicting with the command just issued instead never captures in 300 steps.
    agent = LQGAgent(LQGConfig(regulator=FAST))
    ep = episode(agent, start_state=BEHIND, max_steps=300, sensor=sensor, **DELAYED)
    assert completion_time(ep) is not None


def test_delay_that_is_not_a_whole_number_of_steps_is_refused():
    channels = DELAYED["observation_channels"]
    env = ApproachEnv(
        ApproachConfig(start_state=BEHIND, delay_s=15.0, observation_channels=channels)
    )
    obs, _ = env.reset(seed=0)
    agent = LQGAgent()
    agent.reset(env.agent_view(), agent_generator(1))
    # Nothing is pending at the first decision; the first command's 5 s wait appears at the
    # second.
    with pytest.raises(ValueError, match="not aligned to the decision step"):
        for _ in range(2):
            obs, *_ = env.step(agent.act(obs, env.agent_info()))


@pytest.mark.parametrize("slots", [0, 1])
def test_commands_are_equal_in_fractions_in_every_continuous_mode(slots):
    rng = np.random.default_rng(3)
    states = [
        np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
        + rng.normal(size=6) * [5, 5, 5, 5e-3, 5e-3, 5e-3]
        for _ in range(6)
    ]
    in_effect = np.array([0.03, -0.02, 0.01])  # fractions of a full command
    fractions = {}
    for mode, scale in (("normalized", 1.0), ("accel", A_MAX), ("thrust", A_MAX * 500.0)):
        view = hand_view(mode=mode, slots=slots)
        agent = LQGAgent(LQGConfig(bias_states=("x",)))
        agent.reset(view, np.random.default_rng(0))
        pending = np.concatenate([[0.0], in_effect * scale]) if slots else np.zeros(0)
        fractions[mode] = np.array(
            [
                command_fraction(view, agent.act(np.concatenate([x, pending]), dict(INFO)))
                for x in states
            ]
        )
    ref = fractions["normalized"]
    for mode in ("accel", "thrust"):
        # One division and one multiplication by the action scale per command.
        assert np.max(np.abs(fractions[mode] - ref)) <= 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_discrete_and_impulsive_action_modes_are_refused_at_reset(mode):
    view = ApproachEnv(ApproachConfig(action_mode=mode)).agent_view()
    with pytest.raises(ValueError, match="takes no continuous command"):
        LQGAgent().reset(view, np.random.default_rng(0))


@pytest.mark.parametrize("fields", [{}, {"action_mode": "accel"}, DELAYED])
def test_lqg_agent_passes_the_sign_check(fields):
    assert sign_check(LQGAgent(), ApproachConfig(**fields)).passed


A_START = (60.0, -300.0, -20.0, 0.01, 0.0, 0.0)
NOISY = {"sensor": GaussianNav(1.0, 1e-3)}


def actions(agent, start, **fields):
    env = ApproachEnv(ApproachConfig(start_state=start, max_steps=60, **fields))
    return run_episode(env, agent, seed=11, agent_seed=12)["actions"]


def test_reset_wipes_the_previous_episode():
    config = LQGConfig(bias_states=("x",))
    agent = LQGAgent(config)
    actions(agent, A_START, **DELAYED, **NOISY)
    reused = actions(agent, BEHIND, **NOISY)
    fresh = actions(LQGAgent(config), BEHIND, **NOISY)
    assert np.array_equal(reused, fresh)


def test_two_agents_from_one_config_on_interleaved_environments_match_separate_runs():
    config = LQGConfig(bias_states=("x",))
    starts = (A_START, BEHIND)

    def env_for(start):
        return ApproachEnv(ApproachConfig(start_state=start, max_steps=60, **NOISY))

    envs = [env_for(s) for s in starts]
    agents = [LQGAgent(config), LQGAgent(config)]
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
        alone = actions(LQGAgent(config), start, **NOISY)
        assert np.array_equal(np.array(logs[i]), alone)


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
    agent = LQGAgent(LQGConfig(bias_states=("x",)))
    agent.reset(view)
    plain = [agent.act(obs, {"time_s": 10.0 * k}) for k in range(3)]
    agent.reset(view)
    poisoned = []
    for k in range(3):
        info = ReadLog(time_s=10.0 * k, true_state=np.full(6, 1e6), dv_used_mps=1e6)
        poisoned.append(agent.act(obs, info))
        assert info.read <= {"time_s"}
    assert np.array_equal(np.array(poisoned), np.array(plain))
    assert np.all(np.isfinite(plain)) and np.array(plain).shape == (3, 3)
