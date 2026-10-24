"""The regulator model shared by the LQR agent and the predictive controllers.

`regulator_model` holds the design that `LQRAgent.reset` used to build inline: the stage cost
(sampled by Van Loan or discrete), the delay-augmented model from the pending-command slots, the
rate augmentation and the hold point (xs, us) (Rawlings, Mayne and Diehl, Model Predictive
Control, 2nd ed., sections 1.2.5, 1.5.1 and 8.5.2, exercise 1.25). Every number is for a 500 km
circular orbit of the WGS 84 Earth, a 10 s step and a 0.02 m/s^2 box limit per axis.
"""

import dataclasses
import re

import numpy as np
import pytest

from vbar import ApproachConfig, ApproachEnv
from vbar.agents import lqr
from vbar.agents.base import AgentView, action_scale, agent_generator, hold_point
from vbar.agents.lqr import LQRAgent, LQRConfig, RegulatorModel, regulator_model
from vbar.dynamics import EARTH, cw_discrete, cw_system, mean_motion

N = mean_motion(EARTH.radius + 500e3)
DT = 10.0
A_MAX = 0.02  # m/s^2 per axis
V_BAR_HOLD = (0.0, -100.0, 0.0)
R_BAR_HOLD = (-100.0, 0.0, 0.0)
STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")
# The recorded commands and the stage-cost values below are for scales of 10 m and 0.01 m/s.
DISCRETE = LQRConfig(stage_cost="discrete", vel_scale_mps=0.01)
SAMPLED = LQRConfig(stage_cost="sampled", vel_scale_mps=0.01)
RATE = LQRConfig(rate_weight=0.1, vel_scale_mps=0.01)


def pending_names(slots):
    return tuple(f"pending_{k}_{c}" for k in range(slots) for c in ("t", "x", "y", "z"))


def hand_view(hold=V_BAR_HOLD, mode="normalized", limit="box", slots=0, task=None):
    """An AgentView built without an environment."""
    goal = {"hold_point": hold, "capture_radius": 5.0, "capture_speed": 0.05}
    if task is not None:
        goal["task"] = task
    return AgentView(
        scenario_id="hand-made",
        dt=DT,
        max_steps=360,
        n=N,
        action_mode=mode,
        max_accel=A_MAX,
        thrust_limit=limit,
        mass_kg=500.0,
        goal=goal,
        constraints=({"type": "keep_out", "radius": 50.0},),
        observation=STATE_NAMES + pending_names(slots),
    )


class FrozenLQRAgent:
    """`LQRAgent` as committed before the design moved into `regulator_model`, kept verbatim.

    The reference for the refactor: the same operations in the same order, so the commands must
    agree bit for bit on any machine.
    """

    name = "frozen-lqr"

    def __init__(self, config):
        self.config = config

    def reset(self, view, rng=None):
        hold = hold_point(view, "LQR agent")
        cfg = self.config
        self.scale = action_scale(view)
        if view.thrust_limit not in ("box", "ball"):
            raise ValueError(f"unknown thrust limit {view.thrust_limit!r}")
        self.view = view
        accel_scale = view.max_accel if cfg.accel_scale_mps2 is None else cfg.accel_scale_mps2
        phi, gamma = cw_discrete(view.n, view.dt)
        qc = np.diag([cfg.pos_scale_m**-2] * 3 + [cfg.vel_scale_mps**-2] * 3)
        rc = np.eye(3) / accel_scale**2
        if cfg.stage_cost == "sampled":
            a, b = cw_system(view.n)
            lqr._check_zero_order_hold(a, b, phi, gamma, view.dt)
            q, r, s = lqr.sampled_data_cost(a, b, qc, rc, view.dt)
        else:
            q, r, s = qc, rc, np.zeros((6, 3))

        names = view.observation
        count = sum(1 for name in names if re.fullmatch(r"pending_\d+_t", name))
        self.slots = [[names.index(f"pending_{j}_{c}") for c in "txyz"] for j in range(count)]
        d = len(self.slots)
        phi_z, gamma_z = lqr.delay_augmented(phi, gamma, d)
        q_z = np.zeros(phi_z.shape)
        q_z[:6, :6] = q
        s_z = np.zeros(gamma_z.shape)
        if d:
            q_z[:6, 6:9] = s
            q_z[6:9, :6] = s.T
        else:
            s_z[:] = s
        r_z = r
        self.own_last = cfg.rate_weight > 0.0 and d == 0
        if cfg.rate_weight > 0.0:
            rd = cfg.rate_weight * np.eye(3) / accel_scale**2
            if self.own_last:
                phi_z, gamma_z, q_z, r_z, s_z = lqr.rate_augmented(
                    phi_z, gamma_z, q_z, r_z, s_z, rd
                )
            else:
                q_z[-3:, -3:] += rd
                s_z[-3:] -= rd
                r_z = r + rd
        self.k, _ = lqr.dlqr(phi_z, gamma_z, q_z, r_z, s_z)

        xs = np.concatenate([hold, np.zeros(3)])
        self.us = lqr.steady_state_input(phi, gamma, xs)
        if view.thrust_limit == "box":
            size = float(np.max(np.abs(self.us)))
        else:
            size = float(np.linalg.norm(self.us))
        if size > view.max_accel:
            raise ValueError(
                f"hold point needs {size:.2e} m/s^2 of steady thrust, above max_accel "
                f"{view.max_accel}"
            )
        self.zs = np.concatenate([xs, np.tile(self.us, (len(phi_z) - 6) // 3)])
        self.last = np.zeros(3)

    def act(self, obs, info):
        view = self.view
        obs = np.asarray(obs, dtype=float)
        parts = [obs[:6]]
        if self.slots:
            pending = obs[self.slots]
            steps = pending[:, 0] / view.dt
            if np.any(np.abs(steps - np.round(steps)) > lqr.ALIGNMENT_TOL):
                raise ValueError("pending commands are not aligned to the decision step")
            parts.append((pending[:, 1:] * (view.max_accel / self.scale)).ravel())
        elif self.own_last:
            parts.append(self.last)
        u = self.us - self.k @ (np.concatenate(parts) - self.zs)
        if view.thrust_limit == "box":
            u = np.clip(u, -view.max_accel, view.max_accel)
        else:
            size = np.linalg.norm(u)
            if size > view.max_accel:
                u = u * (view.max_accel / size)
        self.last = u
        return u / view.max_accel * self.scale


# Starts 20 to 60 m from the V-bar hold: the commands stay below the box limit for most of the
# 30 steps, so the gains, not the clipping, set them.
STARTS = (
    (12.0, -130.0, -6.0, 0.0, 0.0, 0.0),
    (-20.0, -70.0, 8.0, 0.005, -0.004, 0.002),
    (5.0, -160.0, 15.0, -0.01, 0.02, 0.0),
)
DELAY_10 = {"delay_s": 10.0, "observation_channels": ("pending_commands",)}
DELAY_20 = {"delay_s": 20.0, "observation_channels": ("pending_commands",)}
CASES = {
    "discrete": (DISCRETE, {}),
    "sampled": (SAMPLED, {}),
    "delay": (SAMPLED, DELAY_10),
    "rate": (RATE, {}),
    "delay_rate": (RATE, DELAY_10),
    "two_step_delay": (SAMPLED, DELAY_20),
}
STEPS = 30


def flown_commands(agent, start, mode="normalized", **fields):
    env = ApproachEnv(
        ApproachConfig(start_state=start, max_steps=STEPS, action_mode=mode, **fields)
    )
    obs, _ = env.reset(seed=11)
    agent.reset(env.agent_view(), agent_generator(12))
    commands = []
    done = False
    while not done:
        action = agent.act(obs, env.agent_info())
        commands.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return np.array(commands)


@pytest.mark.parametrize("mode", ["normalized", "accel", "thrust"])
@pytest.mark.parametrize("start", STARTS)
@pytest.mark.parametrize("case", list(CASES))
def test_lqr_commands_are_unchanged_by_the_shared_model(case, start, mode):
    # Regression check: the design moved, the arithmetic did not.
    config, fields = CASES[case]
    flown = flown_commands(LQRAgent(config), start, mode, **fields)
    frozen = flown_commands(FrozenLQRAgent(config), start, mode, **fields)
    assert flown.shape == (STEPS, 3)
    assert np.array_equal(flown, frozen)


# First and last normalized commands of each case and start, recorded from the committed agent
# before the design moved. Riccati solves round differently across BLAS builds, so these are
# compared at 1e-9 of the largest entry; the bitwise check is against the frozen copy above.
RECORDED = {
    "discrete": (
        (
            (-0.06340520209005997, 0.14821498039333678, 0.029409619167228777),
            (-0.003685614142538113, -0.0021162078168279926, -0.0004994816156054062),
        ),
        (
            (0.07972867482782779, -0.12744983087449485, -0.049237415862776086),
            (0.004482115484365028, 0.0027759370652265204, 0.000666812117642489),
        ),
        (
            (0.019992090987790118, 0.1966809835324299, -0.07352404791807195),
            (-0.005456281388033171, -0.0026509403919704622, 0.0012487040390135075),
        ),
    ),
    "sampled": (
        (
            (-0.07914622709061078, 0.18755082022600936, 0.03727684194259295),
            (-0.003677793950270338, -0.0021118167296123444, -0.0004984263867955784),
        ),
        (
            (0.0994041086640025, -0.16153929270064007, -0.06234968302596284),
            (0.004474321448316655, 0.002771331188339255, 0.0006650552670817936),
        ),
        (
            (0.02653805113405774, 0.24913019485999377, -0.09319210485648238),
            (-0.0054502924171915225, -0.0026487640929521213, 0.001246065966988908),
        ),
    ),
    "delay": (
        (
            (-0.08194899396056037, 0.18760590035464605, 0.03680979424420256),
            (-0.0036781457293870575, -0.0021121068804876757, -0.000498377971945584),
        ),
        (
            (0.10432105082716128, -0.16069063613482962, -0.061850431748906275),
            (0.00446660674175807, 0.002766214729898889, 0.0006666520715056775),
        ),
        (
            (0.02319730196647464, 0.24655255170360893, -0.09202448561050641),
            (-0.005424131452558215, -0.0026333793823896286, 0.0012459449298639381),
        ),
    ),
    "rate": (
        (
            (-0.07913286819431703, 0.18751783903833155, 0.037270254873332576),
            (-0.0036777979208128526, -0.0021118190131666796, -0.0004984269209750147),
        ),
        (
            (0.09938739183274747, -0.16151074932902906, -0.06233869573859345),
            (0.004474325392239773, 0.002771333584954583, 0.0006650561585552233),
        ),
        (
            (0.026532744471403594, 0.24908625624151864, -0.09317563718333143),
            (-0.005450295442236845, -0.0026487652641235277, 0.0012460673024375203),
        ),
    ),
    "delay_rate": (
        (
            (-0.08193514654599103, 0.1875729128306261, 0.03680328859113773),
            (-0.003678149748372972, -0.0021121092038706548, -0.0004983784995057332),
        ),
        (
            (0.10430347660895145, -0.16066224743096397, -0.06183953119503764),
            (0.004466610757353478, 0.0027662171904025215, 0.0006666529559229952),
        ),
        (
            (0.023192579219061406, 0.24650906911809445, -0.09200822147784433),
            (-0.005424134498165675, -0.002633380584342142, 0.0012459462487643346),
        ),
    ),
    "two_step_delay": (
        (
            (-0.08477936212562158, 0.18772307022955578, 0.03633823750395712),
            (-0.003678948306540372, -0.0021127661607553156, -0.0004982685079324217),
        ),
        (
            (0.10929338774397879, -0.1599511047861569, -0.06134360405934279),
            (0.0044596432077943565, 0.0027617469379665523, 0.0006681672139112189),
        ),
        (
            (0.01980964153716296, 0.24404946340327036, -0.0908455937598928),
            (-0.005398429541936923, -0.002618437721091469, 0.0012456712698310514),
        ),
    ),
}


@pytest.mark.parametrize("start_index", range(len(STARTS)))
@pytest.mark.parametrize("case", list(CASES))
def test_lqr_commands_match_those_recorded_before_the_change(case, start_index):
    config, fields = CASES[case]
    flown = flown_commands(LQRAgent(config), STARTS[start_index], **fields)
    expected = np.array(RECORDED[case][start_index])
    got = flown[[0, -1]]
    assert np.max(np.abs(got - expected)) <= 1e-9 * np.max(np.abs(expected))


def test_lqr_agent_designs_through_the_shared_model(monkeypatch):
    calls = []

    def spy(view, config):
        calls.append((view, config))
        return regulator_model(view, config)

    monkeypatch.setattr(lqr, "regulator_model", spy)
    view = hand_view()
    LQRAgent(SAMPLED).reset(view, np.random.default_rng(0))
    assert calls == [(view, SAMPLED)]


def dare_residual(model):
    """Largest entry of the Riccati residual of (phi, gamma, q, r, s) at P, over max |P|."""
    phi, gamma, p = model.phi, model.gamma, model.p
    gain = np.linalg.solve(model.r + gamma.T @ p @ gamma, gamma.T @ p @ phi + model.s.T)
    residual = model.q + phi.T @ p @ phi - (phi.T @ p @ gamma + model.s) @ gain - p
    return np.max(np.abs(residual)) / np.max(np.abs(p))


MODEL_CASES = {
    "discrete": (DISCRETE, 0, 0),
    "sampled": (SAMPLED, 0, 0),
    "delay": (SAMPLED, 1, 0),
    "rate": (RATE, 0, 3),
    "delay_rate": (RATE, 1, 0),
    "two_step_delay": (SAMPLED, 2, 0),
}


@pytest.mark.parametrize("case", list(MODEL_CASES))
def test_model_fields_have_the_augmented_shapes(case):
    config, d, own = MODEL_CASES[case]
    model = regulator_model(hand_view(slots=d), config)
    assert isinstance(model, RegulatorModel)
    nz = 6 + 3 * d + own
    assert model.phi.shape == (nz, nz)
    assert model.gamma.shape == (nz, 3)
    assert model.q.shape == (nz, nz)
    assert model.r.shape == (3, 3)
    assert model.s.shape == (nz, 3)
    assert model.k.shape == (3, nz)
    assert model.p.shape == (nz, nz)
    assert model.xs.shape == (6,)
    assert model.us.shape == (3,)
    assert model.zs.shape == (nz,)
    assert len(model.slots) == d
    assert model.own_last is (own > 0)


@pytest.mark.parametrize("case", list(MODEL_CASES))
def test_p_solves_the_riccati_equation_and_gives_the_gain(case):
    config, d, _ = MODEL_CASES[case]
    model = regulator_model(hand_view(slots=d), config)
    # The design's own check holds the scaled residual at 1e-10; the SI residual sits near 1e-14.
    assert dare_residual(model) <= 1e-10
    assert np.array_equal(model.p, model.p.T)
    gain = np.linalg.solve(
        model.r + model.gamma.T @ model.p @ model.gamma,
        model.gamma.T @ model.p @ model.phi + model.s.T,
    )
    assert np.max(np.abs(model.k - gain)) <= 1e-10 * np.max(np.abs(model.k))
    assert np.max(np.abs(np.linalg.eigvals(model.phi - model.gamma @ model.k))) < 1.0


def test_discrete_stage_cost_is_the_weights_per_step_in_si():
    model = regulator_model(hand_view(), DISCRETE)
    # Scales 10 m and 0.01 m/s, command scale max_accel: Q = diag(1e-2 x3, 1e4 x3), R = 2500 I.
    assert np.array_equal(model.q, np.diag([1e-2] * 3 + [1e4] * 3))
    assert np.array_equal(model.r, np.eye(3) / A_MAX**2)
    assert np.array_equal(model.s, np.zeros((6, 3)))
    phi, gamma = cw_discrete(N, DT)
    assert np.array_equal(model.phi, phi)
    assert np.array_equal(model.gamma, gamma)


def test_sampled_stage_cost_is_van_loans():
    model = regulator_model(hand_view(), SAMPLED)
    a, b = cw_system(N)
    qc = np.diag([1e-2] * 3 + [1e4] * 3)
    q, r, s = lqr.sampled_data_cost(a, b, qc, np.eye(3) / A_MAX**2, DT)
    assert np.array_equal(model.q, q)
    assert np.array_equal(model.r, r)
    assert np.array_equal(model.s, s)


def test_one_step_delay_pairs_the_cross_term_with_the_acting_command():
    model = regulator_model(hand_view(slots=1), SAMPLED)
    a, b = cw_system(N)
    qc = np.diag([1e-2] * 3 + [1e4] * 3)
    q, r, s = lqr.sampled_data_cost(a, b, qc, np.eye(3) / A_MAX**2, DT)
    assert np.array_equal(model.q[:6, :6], q)
    assert np.array_equal(model.q[:6, 6:9], s)
    assert np.array_equal(model.q[6:9, :6], s.T)
    assert np.array_equal(model.s, np.zeros((9, 3)))
    assert np.array_equal(model.r, r)
    phi, gamma = cw_discrete(N, DT)
    assert np.array_equal(model.phi[:6, 6:9], gamma)
    assert np.array_equal(model.gamma[6:9], np.eye(3))


def test_rate_weight_adds_the_change_cost_on_the_newest_command():
    a_rate = 0.1 / A_MAX**2
    own = regulator_model(hand_view(), LQRConfig(stage_cost="discrete", rate_weight=0.1))
    assert np.array_equal(own.q[6:9, 6:9], a_rate * np.eye(3))
    assert np.array_equal(own.s[6:9], -a_rate * np.eye(3))
    assert np.array_equal(own.r, (1.0 / A_MAX**2 + a_rate) * np.eye(3))
    # With a delay the newest pending slot is the last command; no extra state is added.
    delayed = regulator_model(hand_view(slots=1), LQRConfig(stage_cost="discrete", rate_weight=0.1))
    assert np.array_equal(delayed.q[6:9, 6:9], a_rate * np.eye(3))
    assert np.array_equal(delayed.s[6:9], -a_rate * np.eye(3))


def test_r_bar_hold_input_is_three_n_squared_times_the_offset():
    model = regulator_model(hand_view(hold=R_BAR_HOLD, slots=2), SAMPLED)
    assert np.array_equal(model.xs, np.array([*R_BAR_HOLD, 0.0, 0.0, 0.0]))
    # 3 n^2 100 m = 3.6749087912e-4 m/s^2 radial at 500 km.
    assert model.us[0] == pytest.approx(3.6749087912e-4, rel=1e-9)
    assert np.max(np.abs(model.us[1:])) <= 1e-12 * model.us[0]
    # The pending slots hold commands in m/s^2; at the hold each one is us.
    assert np.array_equal(model.zs, np.concatenate([model.xs, model.us, model.us]))


def test_command_scale_from_the_config_overrides_max_accel():
    model = regulator_model(hand_view(), LQRConfig(stage_cost="discrete", accel_scale_mps2=0.005))
    assert np.array_equal(model.r, np.eye(3) / 0.005**2)


def test_slots_index_the_pending_entries_of_the_observation():
    view = hand_view(slots=2)
    model = regulator_model(view, SAMPLED)
    names = [[view.observation[i] for i in slot] for slot in model.slots]
    assert names == [[f"pending_{j}_{c}" for c in "txyz"] for j in range(2)]


def both(view, config=SAMPLED):
    """Each entry point that must refuse the same views."""
    return (
        lambda: LQRAgent(config).reset(view, np.random.default_rng(0)),
        lambda: regulator_model(view, config),
    )


@pytest.mark.parametrize("call", range(2))
@pytest.mark.parametrize("task", ["dock", "inspect", "depart"])
def test_tasks_other_than_a_hold_are_refused(task, call):
    with pytest.raises(ValueError, match=f"flies to a hold point and cannot fly the {task} task"):
        both(hand_view(task=task))[call]()


@pytest.mark.parametrize("call", range(2))
def test_goal_without_a_hold_point_is_refused(call):
    view = dataclasses.replace(hand_view(), goal={"capture_radius": 5.0})
    with pytest.raises(ValueError, match="this goal has none"):
        both(view)[call]()


@pytest.mark.parametrize("call", range(2))
@pytest.mark.parametrize("mode", ["disc27", "impulse_coast"])
def test_discrete_and_impulsive_action_modes_are_refused(mode, call):
    with pytest.raises(ValueError, match="takes no continuous command"):
        both(hand_view(mode=mode))[call]()


@pytest.mark.parametrize("call", range(2))
@pytest.mark.parametrize(
    ("goal", "message"),
    [
        ({"task": "dock", "hold_point": V_BAR_HOLD}, "cannot fly the dock task"),
        ({"capture_radius": 5.0}, "this goal has none"),
    ],
)
def test_goal_is_refused_before_the_action_mode(goal, message, call):
    view = dataclasses.replace(hand_view(mode="disc27"), goal=goal)
    with pytest.raises(ValueError, match=message):
        both(view)[call]()


@pytest.mark.parametrize("call", range(2))
def test_unknown_thrust_layout_is_refused(call):
    with pytest.raises(ValueError, match="unknown thrust limit 'cone'"):
        both(hand_view(limit="cone"))[call]()


@pytest.mark.parametrize("call", range(2))
@pytest.mark.parametrize("limit", ["box", "ball"])
def test_hold_point_needing_more_than_max_accel_is_refused(limit, call):
    # 3 n^2 x reaches 0.02 m/s^2 at x = 5442 m: 5400 m holds, 5500 m does not.
    both(hand_view(hold=(-5400.0, 0.0, 0.0), limit=limit))[call]()
    with pytest.raises(ValueError, match=r"hold point needs 2\.02e-02 m/s\^2 .* above max_accel"):
        both(hand_view(hold=(-5500.0, 0.0, 0.0), limit=limit))[call]()


@pytest.mark.parametrize("call", range(2))
def test_off_grid_delay_is_accepted_at_design_and_refused_when_acting(call):
    # The view carries the slot count, not the delay: 15 s gives two slots, and the off-grid
    # wait shows up only in the observation, where the agent refuses it.
    env = ApproachEnv(
        ApproachConfig(
            start_state=STARTS[0], delay_s=15.0, **{"observation_channels": ("pending_commands",)}
        )
    )
    env.reset(seed=0)
    both(env.agent_view())[call]()
    assert len(regulator_model(env.agent_view(), SAMPLED).slots) == 2


def test_off_grid_pending_wait_is_still_refused_by_the_agent():
    agent = LQRAgent(SAMPLED)
    agent.reset(hand_view(slots=2), np.random.default_rng(0))
    state = [0.0, -150.0, 0.0, 0.0, 0.0, 0.0]
    off_grid = np.array([*state, 0.0, 0.1, 0.0, 0.0, 5.0, 0.2, 0.0, 0.0])
    with pytest.raises(ValueError, match="not aligned to the decision step"):
        agent.act(off_grid, {"time_s": 0.0})


ARRAY_FIELDS = ("phi", "gamma", "q", "r", "s", "k", "p", "xs", "us", "zs")


@pytest.mark.parametrize("case", list(MODEL_CASES))
def test_two_calls_on_one_view_agree_and_share_nothing(case):
    config, d, _ = MODEL_CASES[case]
    view = hand_view(slots=d)
    first = regulator_model(view, config)
    second = regulator_model(view, config)
    for name in ARRAY_FIELDS:
        a, b = getattr(first, name), getattr(second, name)
        assert np.array_equal(a, b), name
        assert not np.shares_memory(a, b), name
    assert first.slots == second.slots
    assert first.slots is not second.slots
    assert first.own_last is second.own_last
    # Writing into one model's arrays leaves the other and later designs alone.
    for name in ARRAY_FIELDS:
        array = getattr(first, name)
        array.flags.writeable = True
        array[...] = np.nan
    first.slots.append([0, 0, 0, 0])
    third = regulator_model(view, config)
    for name in ARRAY_FIELDS:
        assert np.array_equal(getattr(third, name), getattr(second, name)), name
    assert third.slots == second.slots


@pytest.mark.parametrize("case", list(MODEL_CASES))
def test_model_arrays_are_read_only(case):
    config, d, _ = MODEL_CASES[case]
    model = regulator_model(hand_view(slots=d), config)
    for name in ARRAY_FIELDS:
        array = getattr(model, name)
        assert not array.flags.writeable, name
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0.0


def test_model_is_frozen():
    model = regulator_model(hand_view(), SAMPLED)
    with pytest.raises(dataclasses.FrozenInstanceError):
        model.k = np.zeros((3, 6))


def test_agent_does_not_share_arrays_with_another_agent_on_the_same_view():
    view = hand_view(slots=1)
    agents = [LQRAgent(RATE), LQRAgent(RATE)]
    for agent in agents:
        agent.reset(view, np.random.default_rng(0))
    obs = np.array([3.0, -110.0, 2.0, 1e-3, 0.0, -1e-3, 0.0, 0.1, -0.1, 0.0])
    alone = LQRAgent(RATE)
    alone.reset(view, np.random.default_rng(0))
    expected = [alone.act(obs, {"time_s": 0.0}) for _ in range(3)]
    for step in range(3):
        for agent in agents:
            assert np.array_equal(agent.act(obs, {"time_s": 0.0}), expected[step])
