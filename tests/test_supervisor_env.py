"""The supervisor learning wrapper: an environment whose actions are operator commands.

`SupervisorEnv` lets a learner pick one of the seven supervisory commands at each consult of a
supervised agent and runs the inner environment until the next consult. The checks compare it
with a `SupervisedAgent` flown on the same seeds, which is built from the same pieces but stepped
by hand here, so the expected observations, masks, rewards and discounts do not come from the
code under test. The observation layout is the observation, the position relative to the rung
being flown to, a one-hot phase, the rung index over the number of rungs and the command mask.
"""

import dataclasses
import math

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from supervision_support import Cycle, Flight, Session

from vbar.agents import agent_generator
from vbar.agents.learned_supervisor import PolicySupervisor, supervisor_features
from vbar.agents.supervision import PHASES, Command, SupervisedAgent, SupervisionStep
from vbar.agents.tracker import SetpointTracker
from vbar.rewards import Reward
from vbar.scenarios import make, seeds
from vbar.training import SupervisorEnv

GO, HOLD = Command.GO, Command.HOLD
COMMANDS = len(Command)
OBSERVATION = 6
LENGTH = OBSERVATION + 3 + len(PHASES) + 1 + COMMANDS

PAIRS = seeds("approach-vbar", "validation", 2).pairs()
SEED, AGENT_SEED = PAIRS[0]

# Discounting over 200 s, so the 60 s between consults is worth about 0.74, and a time and a
# fuel term, so every step has a reward and the burns differ between flights.
REWARD = Reward(tau=200.0, time_weight=0.01, fuel_weight=5.0)
STEPS = 120
DT = 10.0
ORDERS = [int(c) for c in Cycle.ORDERS]


def inner(**config):
    return make("approach-vbar", **{"max_steps": STEPS, "reward": REWARD, **config})


def parts(features):
    """The named pieces of a feature vector, counted from the end."""
    f = np.asarray(features)
    return {
        "obs": f[: -(3 + len(PHASES) + 1 + COMMANDS)],
        "gap": f[-(3 + len(PHASES) + 1 + COMMANDS) : -(len(PHASES) + 1 + COMMANDS)],
        "phase": f[-(len(PHASES) + 1 + COMMANDS) : -(1 + COMMANDS)],
        "rung": f[-(1 + COMMANDS)],
        "mask": f[-COMMANDS:],
    }


def expected(situation):
    """The features of a situation, written out from its fields."""
    one_hot = np.zeros(len(PHASES))
    one_hot[PHASES.index(situation.phase)] = 1.0
    rung = np.asarray(situation.ladder[situation.rung].position, dtype=float)
    return np.concatenate(
        [
            situation.obs,
            situation.obs[:3] - rung,
            one_hot,
            [situation.rung / len(situation.ladder)],
            np.array(situation.mask, dtype=float),
        ]
    )


class Replay:
    """Gives the listed commands in turn, one per consult."""

    name = "replay"

    def __init__(self, commands):
        self.commands = commands
        self.seen = []

    def reset(self, view, ladder, rng=None):
        self.seen = []

    def decide(self, situation):
        self.seen.append(situation)
        return Command(self.commands[(len(self.seen) - 1) % len(self.commands)])


class Twin:
    """A supervised agent flying the wrapper's flight, cut into the wrapper's steps.

    `segments[k]` holds the inner rewards and discounts from the k-th consult to the step before
    the next, the true state at the end of the segment, and `supervisor.seen[k]` the situation
    the k-th command answered.
    """

    def __init__(self, commands, seed=SEED, tracker=None, period_s=60.0, **config):
        self.env = inner(**config)
        self.supervisor = Replay(commands)
        agent = SupervisedAgent(self.supervisor, tracker=tracker, period_s=period_s)
        obs, _ = self.env.reset(seed=seed)
        agent.reset(self.env.agent_view(), agent_generator(AGENT_SEED))
        self.segments = []
        self.view = self.env.agent_view()
        done = False
        while not done:
            action = agent.act(obs, self.env.agent_info())
            if agent.last_step.consulted:
                self.segments.append({"rewards": [], "discounts": []})
            obs, reward, terminated, truncated, info = self.env.step(action)
            segment = self.segments[-1]
            segment["rewards"].append(reward)
            segment["discounts"].append(info["discount"])
            segment["state"] = self.env.state.copy()
            done = terminated or truncated
        self.situations = self.supervisor.seen


def fly(env, commands, seed=SEED):
    """Steps the wrapper through the commands; returns the reset and every step's results."""
    first = env.reset(seed=seed)
    results = []
    for command in commands:
        results.append(env.step(command))
        if results[-1][2] or results[-1][3]:
            break
    return first, results


# The spaces and the layout


def test_the_wrapper_passes_the_gymnasium_environment_checker():
    check_env(SupervisorEnv(make("approach-vbar")), skip_render_check=True)


def test_the_action_space_is_one_of_the_seven_commands():
    env = SupervisorEnv(make("approach-vbar"))
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == COMMANDS == 7
    assert env.action_space.start == 0


def test_the_observation_space_is_a_flat_float_box_that_holds_the_observations():
    env = SupervisorEnv(inner())
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (LENGTH,)
    assert np.issubdtype(env.observation_space.dtype, np.floating)
    obs, _ = env.reset(seed=SEED)
    assert obs.shape == (LENGTH,) and np.all(np.isfinite(obs))
    assert env.observation_space.contains(obs)


def test_the_features_are_the_observation_the_rung_gap_the_phase_the_rung_and_the_mask():
    s = Session(inner().agent_view())
    s.step((0.0, -300.0, 0.0, 0.0, 0.0, 0.0))
    features = supervisor_features(s.sit, s.view)
    assert features.shape == (LENGTH,)
    pieces = parts(features)
    # Rung 0 of approach-vbar is at (0, -200, 0) m.
    assert np.allclose(pieces["obs"], [0.0, -300.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(pieces["gap"], [0.0, -100.0, 0.0])
    assert np.array_equal(pieces["phase"], [1, 0, 0, 0, 0, 0, 0])
    assert pieces["rung"] == 0.0
    assert np.array_equal(pieces["mask"], [1, 0, 1, 0, 1, 1, 0])
    assert np.allclose(features, expected(s.sit))


def test_the_features_follow_the_phase_and_the_rung_after_an_arrival_and_a_go():
    s = Session(inner().agent_view())
    rest = (0.0, -200.0, 0.0, 0.0, 0.0, 0.0)
    s.step((0.0, -300.0, 0.0, 0.0, 0.0, 0.0))
    s.step(rest)
    pieces = parts(supervisor_features(s.sit, s.view))
    assert np.array_equal(pieces["phase"], [0, 1, 0, 0, 0, 0, 0])
    assert pieces["mask"][GO] == 1.0
    s.step(rest, GO)
    s.step((0.0, -150.0, 0.0, 0.0, 0.0, 0.0))
    features = supervisor_features(s.sit, s.view)
    pieces = parts(features)
    assert s.sit.rung == 1 and s.sit.phase == "transfer"
    assert pieces["rung"] == 0.5
    assert np.allclose(pieces["gap"], [0.0, -50.0, 0.0])
    assert np.allclose(features, expected(s.sit))


def test_the_features_hold_nothing_of_the_info_beyond_what_the_situation_was_given():
    s = Session(inner().agent_view())
    s.step((0.0, -300.0, 0.0, 0.0, 0.0, 0.0))
    s.step((0.0, -299.0, 0.0, 0.0, 0.0, 0.0))
    junk = {"time_s": s.sit.t, "true_state": np.full(6, 123.456), "dv_used_mps": 9.87}
    other = dataclasses.replace(s.sit, info=junk)
    assert np.array_equal(supervisor_features(s.sit, s.view), supervisor_features(other, s.view))


# Masks


def test_the_masks_are_the_situations_masks_at_every_decision_as_bool_and_as_int8():
    twin = Twin(ORDERS)
    masks = [np.array(s.mask) for s in twin.situations]
    assert len(masks) > 5
    env = SupervisorEnv(inner())
    _, info = env.reset(seed=SEED)
    for k, mask in enumerate(masks):
        assert env.action_masks().dtype == np.bool_ and env.action_masks().shape == (COMMANDS,)
        assert np.array_equal(env.action_masks(), mask), k
        assert info["action_mask"].dtype == np.int8
        assert np.array_equal(info["action_mask"], mask.astype(np.int8)), k
        if k + 1 < len(masks):
            _, _, _, _, info = env.step(ORDERS[k % len(ORDERS)])


def test_continue_is_always_valid_and_reconfigure_never():
    env = SupervisorEnv(inner())
    env.reset(seed=SEED)
    for _ in range(8):
        mask = env.action_masks()
        assert mask[Command.CONTINUE] and not mask[Command.RECONFIGURE]
        env.step(int(Command.CONTINUE))


# Steps, rewards and discounts


def test_a_step_runs_the_inner_environment_for_one_period_and_gives_the_time_reached():
    env = SupervisorEnv(inner())
    env.reset(seed=SEED)
    _, _, terminated, truncated, info = env.step(int(Command.CONTINUE))
    assert not (terminated or truncated)
    assert env.unwrapped.steps == 6 and info["time_s"] == 60.0


@pytest.mark.parametrize(("period", "inner_steps"), [(10.0, 1), (30.0, 3), (120.0, 12)])
def test_the_period_sets_the_inner_steps_of_a_step_while_nothing_changes(period, inner_steps):
    env = SupervisorEnv(inner(), period_s=period)
    env.reset(seed=SEED)
    env.step(int(Command.CONTINUE))
    assert env.unwrapped.steps == inner_steps


def test_an_episode_of_continue_commands_ends_truncated_at_the_inner_step_limit():
    env = SupervisorEnv(inner())
    env.reset(seed=SEED)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(int(Command.CONTINUE))
    assert truncated and not terminated
    assert env.unwrapped.steps == STEPS


def test_a_step_ends_at_the_next_consult_whether_by_the_period_or_by_a_phase_change():
    twin = Twin(ORDERS)
    lengths = [len(seg["rewards"]) for seg in twin.segments]
    # A HOLD accepted at t = 0 and the later arrivals cut some steps short of the period.
    assert min(lengths) < 6 and max(lengths) == 6
    env = SupervisorEnv(inner())
    env.reset(seed=SEED)
    for k, segment in enumerate(twin.segments):
        _, _, terminated, truncated, info = env.step(ORDERS[k % len(ORDERS)])
        assert env.unwrapped.steps == sum(len(s["rewards"]) for s in twin.segments[: k + 1]), k
        assert info["time_s"] == DT * env.unwrapped.steps
        assert np.array_equal(env.unwrapped.state, segment["state"]), k
    assert truncated or terminated


def test_the_reward_is_the_discounted_sum_of_the_inner_rewards_and_the_discount_their_product():
    twin = Twin(ORDERS)
    env = SupervisorEnv(inner())
    env.reset(seed=SEED)
    gaps = []
    for k, segment in enumerate(twin.segments):
        _, reward, _, _, info = env.step(ORDERS[k % len(ORDERS)])
        rewards, discounts = np.array(segment["rewards"]), np.array(segment["discounts"])
        before = np.concatenate([[1.0], np.cumprod(discounts)[:-1]])
        assert reward == pytest.approx(float(np.sum(before * rewards)), rel=1e-9, abs=1e-12), k
        assert info["discount"] == pytest.approx(float(np.prod(discounts)), rel=1e-12), k
        assert info["discount"] == pytest.approx(math.exp(-DT * len(rewards) / 200.0)), k
        gaps.append(abs(reward - float(np.sum(rewards))))
    assert max(gaps) > 1e-3


@pytest.mark.parametrize("command", list(Command))
def test_a_refused_order_costs_the_refused_penalty_and_an_accepted_one_does_not(command):
    plain = SupervisorEnv(inner())
    penalised = SupervisorEnv(inner(), refused_penalty=2.5)
    plain.reset(seed=SEED)
    penalised.reset(seed=SEED)
    refused = not plain.action_masks()[command]
    _, base, _, _, _ = plain.step(int(command))
    _, reward, _, _, _ = penalised.step(int(command))
    assert reward == pytest.approx(base - 2.5 if refused else base, abs=1e-12)


def test_the_penalty_is_paid_once_for_each_refused_order_and_never_for_a_valid_one():
    plain = SupervisorEnv(inner())
    penalised = SupervisorEnv(inner(), refused_penalty=1.5)
    plain.reset(seed=SEED)
    penalised.reset(seed=SEED)
    orders = [Command.RECONFIGURE, Command.CONTINUE, Command.RECONFIGURE, Command.GO]
    paid = []
    for command in orders:
        refused = not plain.action_masks()[command]
        base = plain.step(int(command))[1]
        reward = penalised.step(int(command))[1]
        paid.append(round(base - reward, 9))
        assert refused == (paid[-1] == 1.5)
    assert paid[:3] == [1.5, 0.0, 1.5]


def test_the_reward_hook_is_called_for_each_step_with_the_supervision_step_and_added():
    calls = []

    def hook(step):
        calls.append(step)
        return 0.25

    plain = SupervisorEnv(inner())
    hooked = SupervisorEnv(inner(), reward_hook=hook)
    plain.reset(seed=SEED)
    hooked.reset(seed=SEED)
    for command in ORDERS:
        base = plain.step(command)[1]
        reward = hooked.step(command)[1]
        assert reward == pytest.approx(base + 0.25, abs=1e-12)
    assert len(calls) == len(ORDERS)
    assert all(isinstance(step, SupervisionStep) for step in calls)


def test_a_tracker_given_to_the_wrapper_flies_the_commands():
    slow = SetpointTracker(v_max=0.1)
    env = SupervisorEnv(inner(), tracker=slow)
    default = SupervisorEnv(inner())
    env.reset(seed=SEED)
    default.reset(seed=SEED)
    env.step(int(Command.CONTINUE))
    default.step(int(Command.CONTINUE))
    twin = Twin([Command.CONTINUE], tracker=slow)
    first = twin.segments[0]["state"]
    assert len(twin.segments[0]["rewards"]) == 6
    assert np.array_equal(env.unwrapped.state, first)
    assert not np.array_equal(default.unwrapped.state, first)


# Information the features are made from


def scripted_observation(t):
    """The same track whatever the true state: closing on rung 0, resting on it, then on rung 1."""
    if t < 200.0:
        return np.array([0.0, -300.0 + 0.5 * t, 0.0, 0.0, 0.5, 0.0])
    if t < 500.0:
        return np.array([0.0, -200.0, 0.0, 0.0, 0.0, 0.0])
    if t < 700.0:
        return np.array([0.0, -200.0 + 0.5 * (t - 500.0), 0.0, 0.0, 0.5, 0.0])
    return np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])


class Scripted:
    """A sensor that reports one observation sequence whatever the true state."""

    name = "scripted"

    def reset(self, rng):
        pass

    def observe(self, true_state, t):
        return scripted_observation(t)


def scripted_run(seed, steps=90):
    env = SupervisorEnv(make("approach-vbar", max_steps=steps, sensor=Scripted(), reward=REWARD))
    obs, _ = env.reset(seed=seed)
    start = env.unwrapped.state[:3].copy()
    seen = [(obs.copy(), env.action_masks().copy())]
    done = False
    while not done:
        command = int(GO) if env.action_masks()[GO] else int(Command.CONTINUE)
        obs, _, terminated, truncated, _ = env.step(command)
        seen.append((obs.copy(), env.action_masks().copy()))
        done = terminated or truncated
    return start, seen


def test_runs_from_true_starts_more_than_10_m_apart_give_identical_observations_and_masks():
    start_a, seen_a = scripted_run(PAIRS[0][0])
    start_b, seen_b = scripted_run(PAIRS[1][0])
    assert np.linalg.norm(start_a - start_b) > 10.0
    assert len(seen_a) == len(seen_b) > 5
    for (obs_a, mask_a), (obs_b, mask_b) in zip(seen_a, seen_b, strict=True):
        assert np.array_equal(obs_a, obs_b)
        assert np.array_equal(mask_a, mask_b)
    phases = {PHASES[int(np.argmax(parts(obs)["phase"]))] for obs, _ in seen_a}
    assert phases >= {"transfer", "holding"}


def test_the_wrapper_observations_are_the_features_of_the_situation_at_each_consult():
    twin = Twin(ORDERS)
    env = SupervisorEnv(inner())
    obs, _ = env.reset(seed=SEED)
    observed = [obs]
    for k in range(len(twin.situations) - 1):
        observed.append(env.step(ORDERS[k % len(ORDERS)])[0])
    assert len(observed) > 5
    for k, (obs, situation) in enumerate(zip(observed, twin.situations, strict=False)):
        assert np.allclose(obs, expected(situation), rtol=1e-6, atol=1e-4), k


# The learned supervisor


def go_when_valid(features):
    return int(GO) if parts(features)["mask"][GO] > 0.5 else int(Command.CONTINUE)


def from_phase_rung_and_mask(features):
    """Depends on discrete parts of the features only, so it gives the same command whether it
    reads them as float32 or float64; most of its commands are refused."""
    pieces = parts(features)
    phase = int(np.argmax(pieces["phase"]))
    return (3 * phase + 5 * int(round(float(pieces["rung"]) * 2)) + int(pieces["mask"].sum())) % 7


def test_a_policy_supervisor_gives_the_policys_command_on_the_features_of_the_situation():
    s = Session(inner().agent_view())
    s.step((0.0, -300.0, 0.0, 0.0, 0.0, 0.0))
    got = []

    def policy(features):
        got.append(np.array(features))
        return 2

    supervisor = PolicySupervisor(policy)
    supervisor.reset(s.view, s.supervisor.ladder, None)
    answer = supervisor.decide(s.sit)
    assert isinstance(answer, Command) and answer == HOLD
    assert len(got) == 1
    assert np.array_equal(got[0], supervisor_features(s.sit, s.view))


@pytest.mark.parametrize("policy", [go_when_valid, from_phase_rung_and_mask])
def test_a_policy_supervisor_flies_the_trajectory_of_the_wrapper_with_the_same_policy(policy):
    # Long enough for the chaser to reach the first rung and be sent on to the second.
    env = SupervisorEnv(inner(max_steps=300))
    obs, _ = env.reset(seed=SEED)
    ends = []
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        ends.append((env.unwrapped.steps, env.unwrapped.state.copy()))
        done = terminated or truncated
    agent = SupervisedAgent(PolicySupervisor(policy))
    flight = Flight("approach-vbar", agent, SEED, AGENT_SEED, max_steps=300, reward=REWARD)
    flight.run(300)
    assert len(ends) > 20
    for steps, state in ends:
        assert np.array_equal(flight.truths[steps], state), steps
    assert ends[-1][0] == len(flight.actions)
    assert any(r[3] is False for r in flight.records) == (policy is from_phase_rung_and_mask)
    assert any(r[3] is True and "CONTINUE" not in r[2] for r in flight.records)


# The constructor


@pytest.mark.parametrize(
    "period",
    [float("nan"), float("inf"), -60.0, 0.0, "60", None, True, 5.0, 65.0],
)
def test_a_period_that_is_not_a_positive_whole_number_of_steps_raises_value_error(period):
    with pytest.raises(ValueError):
        SupervisorEnv(inner(), period_s=period)


@pytest.mark.parametrize(
    "penalty", [float("nan"), float("inf"), float("-inf"), -1.0, "1", None, True]
)
def test_a_refused_penalty_that_is_not_a_finite_number_not_below_zero_raises_value_error(penalty):
    with pytest.raises(ValueError):
        SupervisorEnv(inner(), refused_penalty=penalty)


@pytest.mark.parametrize("action_mode", ["disc27", "impulse_coast"])
def test_an_action_mode_that_is_not_continuous_raises_value_error_naming_it(action_mode):
    with pytest.raises(ValueError, match=action_mode):
        SupervisorEnv(make("approach-vbar", action_mode=action_mode))


def test_a_whole_number_period_and_an_integer_penalty_are_accepted():
    env = SupervisorEnv(inner(), period_s=120.0, refused_penalty=2)
    env.reset(seed=SEED)
    env.step(int(Command.CONTINUE))
    assert env.unwrapped.steps == 12
