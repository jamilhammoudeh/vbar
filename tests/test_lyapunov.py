"""CW tracking and Cartesian Lyapunov agents.

The tracking law is Alfriend et al., Spacecraft Formation Flying, eq 10.8; the Cartesian law is
Schaub and Junkins, Analytical Mechanics of Space Systems, section 13.7, eq 13.229, on a declared
two-body model. Every number is for a 500 km circular orbit of the WGS 84 Earth, a 10 s step and a
0.02 m/s^2 box limit per axis unless stated.
"""

import dataclasses

import gymnasium
import numpy as np
import pytest
from metric_cases import episode

from vbar import ApproachConfig, ApproachEnv, run_episode
from vbar.agents.base import AgentView, agent_generator, command_fraction
from vbar.agents.lyapunov import CWTrackingAgent, DeclaredModel, LyapunovAgent, LyapunovConfig
from vbar.agents.pd import PDAgent
from vbar.agents.sign_check import sign_check
from vbar.dynamics import EARTH, cw_discrete, cw_stm, cw_system, mean_motion
from vbar.dynamics.models import TwoBodyDynamics, relative_acceleration
from vbar.metrics import mission_metrics
from vbar.scenarios.base import EnvConfig

R_TARGET = EARTH.radius + 500e3
N = mean_motion(R_TARGET)
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
INCLINATION = np.radians(51.6)
BEHIND = (0.0, -400.0, 30.0, 0.0, 0.0, 0.0)
V_BAR_HOLD = (0.0, -100.0, 0.0)
R_BAR_HOLD = (-100.0, 0.0, 0.0)
INFO = {"time_s": 0.0}


def hand_view(hold=V_BAR_HOLD, mode="normalized", limit="box", dt=DT, max_accel=A_MAX):
    """An AgentView built without an environment."""
    return AgentView(
        scenario_id="hand-made",
        dt=dt,
        max_steps=360,
        n=N,
        action_mode=mode,
        max_accel=max_accel,
        thrust_limit=limit,
        mass_kg=500.0,
        goal={"hold_point": hold, "capture_radius": 5.0, "capture_speed": 0.05},
        constraints=({"type": "keep_out", "radius": 50.0},),
    )


def acted(agent, view, obs, time_s=0.0):
    agent.reset(view, np.random.default_rng(0))
    return command_fraction(view, agent.act(np.asarray(obs, dtype=float), {"time_s": time_s}))


def cw_terms(r, v):
    """The CW accelerations 3 n^2 x + 2 n vy, -2 n vx, -n^2 z (m/s^2)."""
    return np.array([3 * N**2 * r[0] + 2 * N * v[1], -2 * N * v[0], -(N**2) * r[2]])


def random_observations(rng, count, hold=V_BAR_HOLD):
    """Half within 1 m and 1 mm/s of the hold point (linear law), half 1000 times farther."""
    spreads = np.where(np.arange(count) < count // 2, 1.0, 1e3)
    obs = np.tile([*hold, 0.0, 0.0, 0.0], (count, 1))
    return obs + rng.normal(size=(count, 6)) * spreads[:, None] * np.array([1.0] * 3 + [1e-3] * 3)


def declared(dynamics, **fields):
    return LyapunovConfig(model=DeclaredModel(dynamics=dynamics, inclination=INCLINATION, **fields))


def test_tracking_law_at_rest_on_the_hold_point_is_the_cw_cancelling_pd_law():
    view = hand_view()
    rng = np.random.default_rng(1)
    for k in (2.0, 0.7):
        for obs in random_observations(rng, 50):
            expected = acted(PDAgent(wn=N, zeta=k / 2), view, obs)
            command = acted(CWTrackingAgent(k=k), view, obs)
            assert np.max(np.abs(command - expected)) <= 1e-12 * np.max(np.abs(expected))


def natural_reference(x0):
    """A free CW trajectory as (r_d, v_d, a_d), the acceleration being A x_d."""
    a, _ = cw_system(N)

    def reference(t):
        x_d = cw_stm(N, t) @ x0
        return x_d[:3], x_d[3:], (a @ x_d)[3:]

    return reference


def test_tracking_a_natural_cw_trajectory_with_zero_error_needs_no_thrust():
    view = hand_view(mode="accel")
    rng = np.random.default_rng(2)
    for _ in range(50):
        x0 = np.concatenate([rng.uniform(-500.0, 500.0, 3), rng.uniform(-0.5, 0.5, 3)])
        t = rng.uniform(0.0, 6000.0)
        x_d = cw_stm(N, t) @ x0
        agent = CWTrackingAgent(k=2.0, reference=natural_reference(x0))
        agent.reset(view, np.random.default_rng(0))
        u = agent.act(x_d.copy(), {"time_s": t})
        # The feedforward cancels the CW terms term by term; an independent computation of the
        # same sum is exactly zero in float64. A wrong sign on the Coriolis terms leaves 4 n |v|.
        assert np.linalg.norm(u) < 1e-12 * N**2 * np.linalg.norm(x_d[:3])


def test_tracking_command_feeds_forward_the_reference_and_damps_the_velocity_error():
    view = hand_view(mode="accel", max_accel=1.0)
    rng = np.random.default_rng(3)
    k = 2.0
    for _ in range(50):
        r_d, v_d, a_d = rng.uniform(-500.0, 500.0, 3), rng.uniform(-0.5, 0.5, 3), rng.normal(size=3)
        a_d *= 1e-4
        x = np.concatenate([r_d + rng.uniform(-50.0, 50.0, 3), v_d + rng.uniform(-0.05, 0.05, 3)])
        e, e_dot = x[:3] - r_d, x[3:] - v_d
        terms = (a_d, cw_terms(x[:3], x[3:]), N**2 * e, k * N * e_dot)
        expected = terms[0] - terms[1] - terms[2] - terms[3]
        agent = CWTrackingAgent(k=k, reference=lambda t, ref=(r_d, v_d, a_d): ref)
        agent.reset(view, np.random.default_rng(0))
        u = agent.act(x, {"time_s": 300.0})
        # Same sum in another order: rounding at 1e-16 of the largest term.
        scale = max(np.linalg.norm(term) for term in terms)
        assert np.max(np.abs(u - expected)) <= 1e-12 * scale


def test_tracking_error_from_rest_decays_critically_damped_over_one_orbit():
    dt = 1.0
    phi, gamma = cw_discrete(N, dt)
    view = hand_view(mode="accel", dt=dt)
    hold = np.array(V_BAR_HOLD)
    x = np.concatenate([hold + [10.0, -10.0, 5.0], np.zeros(3)])
    agent = CWTrackingAgent(k=2.0)
    agent.reset(view, np.random.default_rng(0))
    steps = round(2 * np.pi / N / dt)
    norms = [np.linalg.norm(x[:3] - hold)]
    for i in range(steps):
        x = phi @ x + gamma @ agent.act(x.copy(), {"time_s": i * dt})
        norms.append(np.linalg.norm(x[:3] - hold))
    norms = np.array(norms)
    assert np.all(np.diff(norms) <= 0.0)
    # Continuous e'' + 2 n e' + n^2 e = 0 from rest gives 15 (1 + 2 pi) exp(-2 pi) = 0.20401 m
    # after one orbit; holding each command for 1 s lowers it to 0.2032639 m (an independent
    # computation); the same law in another order agrees far inside 1e-6. A dropped Coriolis
    # term or another k moves it by percent. The 5677 steps take about 0.5 s.
    assert norms[steps] == pytest.approx(0.2032639, rel=1e-6)


@pytest.mark.parametrize(
    "make", [lambda: CWTrackingAgent(), lambda: LyapunovAgent(LyapunovConfig())]
)
def test_commands_are_equal_in_fractions_in_every_continuous_mode(make):
    views = {
        mode: ApproachEnv(ApproachConfig(action_mode=mode)).agent_view()
        for mode in ("normalized", "accel", "thrust")
    }
    for obs in random_observations(np.random.default_rng(4), 20):
        fractions = {mode: acted(make(), view, obs) for mode, view in views.items()}
        ref = fractions["normalized"]
        for mode in ("accel", "thrust"):
            # One scale factor applied and removed: rounding at 1e-16.
            assert np.max(np.abs(fractions[mode] - ref)) <= 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("make", [CWTrackingAgent, LyapunovAgent])
@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_discrete_and_impulsive_action_modes_are_refused_at_reset(make, mode):
    view = ApproachEnv(ApproachConfig(action_mode=mode)).agent_view()
    with pytest.raises(ValueError, match="takes no continuous command"):
        make().reset(view, np.random.default_rng(0))


@pytest.mark.parametrize("make", [CWTrackingAgent, LyapunovAgent])
def test_box_commands_are_clipped_per_axis_and_ball_commands_keep_their_direction(make):
    xs = np.array([*V_BAR_HOLD, 0.0, 0.0, 0.0])
    delta = np.array([30.0, -40.0, 20.0, 0.01, 0.02, -0.01])
    # At the V-bar hold the CW terms of the hold point vanish, so both laws are linear in the
    # deviation and the far command is the near one scaled by 1e6 before clipping.
    near = acted(make(), hand_view(limit="ball"), xs + 1e-3 * delta)
    assert np.linalg.norm(near) < 1.0
    far_box = acted(make(), hand_view(limit="box"), xs + 1e3 * delta)
    assert np.max(np.abs(far_box)) == pytest.approx(1.0, abs=1e-15)
    far_ball = acted(make(), hand_view(limit="ball"), xs + 1e3 * delta)
    assert np.linalg.norm(far_ball) == pytest.approx(1.0, rel=1e-12)
    assert np.max(np.abs(far_ball - near / np.linalg.norm(near))) <= 1e-9


def test_config_and_declared_model_are_frozen_and_keyword_only():
    with pytest.raises(TypeError):
        LyapunovConfig(1e-4)
    with pytest.raises(TypeError):
        DeclaredModel("cw")
    config = LyapunovConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.k1 = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.model.dynamics = "twobody"
    assert (config.k1, config.k2, config.model) == (1e-4, 0.02, DeclaredModel())


def test_declared_model_holds_only_the_public_scenario_parameters_with_their_defaults():
    names = tuple(f.name for f in dataclasses.fields(DeclaredModel))
    assert names == ("dynamics", "altitude_m", "inclination", "body")
    model, env_config = DeclaredModel(), EnvConfig()
    for name in names:
        assert getattr(model, name) == getattr(env_config, name)


def test_unknown_declared_dynamics_is_refused():
    with pytest.raises(ValueError, match="unknown dynamics"):
        agent = LyapunovAgent(LyapunovConfig(model=DeclaredModel(dynamics="twobody_drag")))
        agent.reset(hand_view(), np.random.default_rng(0))


def test_lyapunov_law_on_the_declared_cw_model_is_the_tracking_law():
    view = hand_view()
    k2 = 0.02
    config = LyapunovConfig(k1=N**2, k2=k2, model=DeclaredModel(dynamics="cw"))
    for obs in random_observations(np.random.default_rng(5), 100):
        expected = acted(CWTrackingAgent(k=k2 / N), view, obs)
        command = acted(LyapunovAgent(config), view, obs)
        assert np.max(np.abs(command - expected)) <= 1e-12 * np.max(np.abs(expected))


def test_default_lyapunov_law_is_the_default_pd_law():
    view = hand_view()
    for obs in random_observations(np.random.default_rng(6), 100):
        expected = acted(PDAgent(), view, obs)
        command = acted(LyapunovAgent(), view, obs)
        assert np.max(np.abs(command - expected)) <= 1e-12 * np.max(np.abs(expected))


@pytest.mark.parametrize("dynamics", ["twobody", "twobody_j2"])
def test_closed_loop_acceleration_is_the_linear_error_law_on_the_declared_model(dynamics):
    j2 = dynamics == "twobody_j2"
    config = declared(dynamics)
    # A 1 m/s^2 limit keeps every command unclipped at 1 km and 0.1 m/s.
    view = hand_view(mode="accel", max_accel=1.0)
    agent = LyapunovAgent(config)
    agent.reset(view, np.random.default_rng(0))
    target = TwoBodyDynamics(inclination=INCLINATION, j2=j2)
    rng = np.random.default_rng(7)
    hold = np.array(V_BAR_HOLD)
    for t in (0.0, 300.0, 1500.0, 3000.0, 5000.0):
        target.reset(np.zeros(6))
        if t > 0.0:
            target.advance(np.zeros(3), t)
        x = np.concatenate([rng.uniform(-1e3, 1e3, 3), rng.uniform(-0.1, 0.1, 3)])
        f = relative_acceleration(x, target.target_state(), EARTH, j2=j2)
        u = agent.act(x.copy(), {"time_s": float(t)})
        residual = u + f + config.k1 * (x[:3] - hold) + config.k2 * x[3:]
        # The target advanced in other steps and in one segment agree to 2.4e-12 of |f| (an
        # independent computation); the CW terms miss f by at least 9.5e-5 of |f| on these
        # states, and f taken at the hold point or with the wrong sign misses by order |f|.
        assert np.linalg.norm(residual) <= 1e-9 * np.linalg.norm(f)


def test_declared_altitude_sets_the_model_whatever_the_view_says():
    # At rest on its hold point the command is -f alone; n^2 is 4.5% larger at 400 km.
    hold = (300.0, -500.0, 40.0)
    view = hand_view(hold=hold, mode="accel")
    obs = np.array([*hold, 0.0, 0.0, 0.0])
    at_500 = acted(LyapunovAgent(declared("twobody")), view, obs)
    at_400 = acted(LyapunovAgent(declared("twobody", altitude_m=400e3)), view, obs)
    assert np.max(np.abs(at_500 - at_400)) > 1e-2 * np.max(np.abs(at_500))


R_BAR_GOAL = {"hold_point": R_BAR_HOLD, "capture_radius": 5.0, "capture_speed": 0.05}


def r_bar_hold_error(config, dynamics, steps):
    # The custom goal keeps the episode running past capture to its step limit.
    ep = episode(
        LyapunovAgent(config),
        start_state=(*R_BAR_HOLD, 0.0, 0.0, 0.0),
        hold_point=R_BAR_HOLD,
        max_steps=steps,
        goal=R_BAR_GOAL,
        dynamics=dynamics,
        inclination=INCLINATION,
    )
    return mission_metrics(ep)["terminal_pos_error"]


def test_matched_two_body_model_holds_the_r_bar_point_to_rounding():
    # 30 steps on the two-body rung take about 0.5 s. An independent computation gives 4.3e-14 m
    # here and 1.6e-13 m after three orbits; the cw model leaves 5.3e-5 m.
    assert r_bar_hold_error(declared("twobody"), "twobody", 30) < 1e-6


def test_cw_model_on_two_body_motion_holds_off_by_the_second_order_gravity_remainder():
    # The CW model drops -3 n^2 x^2 / r of the radial gravity difference, which the position
    # gain balances at 3 n^2 x^2 / (r k1) = 5.3429e-5 m. After 80 steps the critically damped
    # transient is down to 3e-3 of it and the third-order terms are 1.5e-5 of it; an
    # independent computation gives 5.325e-5 m. 80 steps take about 1 s on this rung.
    expected = 3 * N**2 * R_BAR_HOLD[0] ** 2 / (R_TARGET * 1e-4)
    error = r_bar_hold_error(declared("cw"), "twobody", 80)
    assert error == pytest.approx(expected, rel=1e-2)


def test_matched_j2_model_holds_the_r_bar_point_a_hundred_times_closer_than_the_cw_model():
    # An independent computation after 30 steps (about 0.5 s each): matched 5.3e-5 m, set by
    # holding a J2 acceleration that turns at 2n for 10 s (6.3e-5, 3.2e-5, 1.3e-5 m at
    # dt = 10, 5, 2 s); declared cw 6.0e-3 m, dominated by J2.
    assert r_bar_hold_error(declared("twobody_j2"), "twobody_j2", 30) < 2e-4
    assert r_bar_hold_error(declared("cw"), "twobody_j2", 30) > 2e-3


def test_lyapunov_agent_completes_the_approach_on_the_j2_rung():
    # Captures at 620 s in an independent computation; the run takes about 1.5 s.
    ep = episode(
        LyapunovAgent(declared("twobody_j2")),
        start_state=BEHIND,
        max_steps=200,
        dynamics="twobody_j2",
        inclination=INCLINATION,
    )
    assert mission_metrics(ep)["success"]


@pytest.mark.parametrize(
    "make, fields",
    [
        (CWTrackingAgent, {}),
        (CWTrackingAgent, {"action_mode": "accel"}),
        (LyapunovAgent, {}),
        (LyapunovAgent, {"action_mode": "thrust"}),
        (
            lambda: LyapunovAgent(declared("twobody_j2")),
            {"dynamics": "twobody_j2", "inclination": INCLINATION},
        ),
    ],
)
def test_agents_pass_the_sign_check(make, fields):
    assert sign_check(make(), ApproachConfig(**fields)).passed


A_START = (60.0, -300.0, -20.0, 0.01, 0.0, 0.0)
TWO_BODY = {"dynamics": "twobody", "inclination": INCLINATION}


def actions(agent, start, steps):
    env = ApproachEnv(ApproachConfig(start_state=start, max_steps=steps, **TWO_BODY))
    return run_episode(env, agent, seed=11, agent_seed=12)["actions"]


def test_reset_wipes_the_declared_target_orbit():
    # Episodes of 10 to 15 steps on the two-body rung, about 0.7 s in all.
    agent = LyapunovAgent(declared("twobody"))
    actions(agent, A_START, 15)
    reused = actions(agent, BEHIND, 10)
    fresh = actions(LyapunovAgent(declared("twobody")), BEHIND, 10)
    assert np.array_equal(reused, fresh)


def test_two_agents_from_one_config_on_interleaved_environments_match_separate_runs():
    # Four 10-step episodes on the two-body rung, about 1 s in all.
    config = declared("twobody")
    starts = (A_START, BEHIND)
    envs = [ApproachEnv(ApproachConfig(start_state=s, max_steps=10, **TWO_BODY)) for s in starts]
    agents = [LyapunovAgent(config), LyapunovAgent(config)]
    obs, logs = [], [[], []]
    for env, agent in zip(envs, agents, strict=True):
        o, _ = env.reset(seed=11)
        agent.reset(env.agent_view(), agent_generator(12))
        obs.append(o)
    for _ in range(10):
        for i, (env, agent) in enumerate(zip(envs, agents, strict=True)):
            a = agent.act(obs[i], env.agent_info())
            logs[i].append(a)
            obs[i], *_ = env.step(a)
    for i, start in enumerate(starts):
        assert np.array_equal(np.array(logs[i]), actions(LyapunovAgent(config), start, 10))


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


@pytest.mark.parametrize(
    "make",
    [
        lambda: CWTrackingAgent(),
        lambda: CWTrackingAgent(reference=natural_reference(np.array([0, -150.0, 0, 0, 0, 0]))),
        lambda: LyapunovAgent(),
        lambda: LyapunovAgent(declared("twobody_j2")),
    ],
)
def test_agent_runs_on_a_hand_made_view_and_reads_only_the_time(make):
    view = hand_view()
    obs = np.array([5.0, -180.0, 3.0, 0.01, 0.0, -0.002])
    agent = make()
    agent.reset(view)
    plain = agent.act(obs.copy(), {"time_s": 120.0})
    agent = make()
    agent.reset(view)
    poisoned = ReadLog(time_s=120.0, true_state=np.full(6, 1e6), dv_used_mps=1e6)
    command = agent.act(obs.copy(), poisoned)
    assert poisoned.read <= {"time_s"}
    assert np.array_equal(command, plain)
    assert command.shape == (3,) and np.all(np.isfinite(command))


def held_objects(root, depth=4):
    """Every object reachable from `root` through attributes and containers, to `depth`."""
    seen, frontier = {id(root): root}, [root]
    for _ in range(depth):
        nxt = []
        for obj in frontier:
            if isinstance(obj, dict):
                children = list(obj.values())
            elif isinstance(obj, list | tuple | set | frozenset):
                children = list(obj)
            elif hasattr(obj, "__dict__") and not isinstance(obj, type):
                children = list(vars(obj).values())
            else:
                children = []
            for child in children:
                if id(child) not in seen and not isinstance(child, np.ndarray):
                    seen[id(child)] = child
                    nxt.append(child)
        frontier = nxt
    return list(seen.values())


def test_lyapunov_agent_holds_no_environment_object_and_replays_from_the_view_alone():
    config = declared("twobody_j2")
    env = ApproachEnv(
        ApproachConfig(
            start_state=BEHIND, max_steps=20, dynamics="twobody_j2", inclination=INCLINATION
        )
    )
    o, _ = env.reset(seed=11)
    agent = LyapunovAgent(config)
    agent.reset(env.agent_view(), agent_generator(12))
    log = []
    for _ in range(20):
        info = env.agent_info()
        a = agent.act(o.copy(), info)
        log.append((o.copy(), info["time_s"], a))
        o, *_ = env.step(a)
    held = held_objects(agent)
    assert not any(isinstance(obj, gymnasium.Env) for obj in held)
    assert not any(obj is env.dynamics or obj is env.cfg for obj in held)
    # A fresh agent given only a copy of the public view, the observations and the times
    # reproduces every command; the run takes about 1 s.
    view = dataclasses.replace(env.agent_view())
    replay = LyapunovAgent(config)
    replay.reset(view)
    for obs, time_s, a in log:
        assert np.array_equal(replay.act(obs, {"time_s": time_s}), a)
