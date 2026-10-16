"""Rewards on real episodes of the approach scenario (Sutton and Barto review, KAT-1 to KAT-3)."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from scipy.linalg import solve_discrete_are

from vbar import ApproachConfig, ApproachEnv
from vbar.agents import PDAgent
from vbar.dynamics import cw_discrete
from vbar.rewards import QuadraticCost, Reward

HOLD = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
THROUGH = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)
CLEAR = (-150.0, -400.0, 0.0, 0.0, 0.0, 0.0)


def rollout(env, agent, seed=0):
    """Rewards, discounts, final info, terminal flag and true states of one episode."""
    obs, info = env.reset(seed=seed)
    agent.reset(env.agent_view(), np.random.default_rng(0))
    rewards, discounts, states = [], [], [info["true_state"]]
    terminated = truncated = False
    while not (terminated or truncated):
        obs, r, terminated, truncated, info = env.step(agent.act(obs, env.agent_info()))
        rewards.append(r)
        discounts.append(info.get("discount", 1.0))
        states.append(info["true_state"])
    return np.array(rewards), np.array(discounts), info, terminated, np.array(states)


def episode_return(rewards, discounts):
    """Discounted return and the product of all step discounts."""
    weights = np.r_[1.0, np.cumprod(discounts)[:-1]]
    return float(np.sum(weights * rewards)), float(np.prod(discounts))


class Schedule:
    """Open-loop: `command` until `t_off` s, then coasting."""

    name = "schedule"

    def __init__(self, command, t_off):
        self.command, self.t_off = np.array(command, dtype=float), t_off

    def reset(self, view, rng=None):
        pass

    def act(self, obs, info):
        return self.command if info["time_s"] < self.t_off - 1e-9 else np.zeros(3)


class Waypoints:
    """PD flight through waypoints, moving on within 10 m of each, then to the hold point."""

    name = "waypoints"

    def __init__(self, points):
        self.points = [np.array(p, dtype=float) for p in points]

    def reset(self, view, rng=None):
        self.pd = PDAgent()
        self.pd.reset(view)
        self.route = [*self.points, self.pd.hold.copy()]
        self.k = 0

    def act(self, obs, info):
        if self.k < len(self.points) and np.linalg.norm(obs[:3] - self.route[self.k]) < 10.0:
            self.k += 1
        self.pd.hold = self.route[self.k]
        return self.pd.act(obs, info)


@pytest.mark.parametrize("tau", [None, 3600.0])
def test_one_trajectory_earns_the_same_return_at_every_decision_interval(tau):
    # Sutton and Barto 14.4 (KAT-2): an open-loop burn that carries the chaser through the
    # keep-out sphere, flown at 10, 5 and 2.5 s steps.
    n = ApproachEnv().n
    reward = Reward(
        tau=tau,
        fuel_weight=3.0,
        time_weight=1e-2,
        violation_weights={"keep_out": 0.2},
        entry_weights={"keep_out": 4.0},
        quadratic=QuadraticCost(n, np.diag([1e-4] * 3 + [1.0] * 3), 10.0 * np.eye(3), HOLD),
    )
    returns = []
    for dt in (10.0, 5.0, 2.5):
        cfg = ApproachConfig(
            start_state=(3.0, 150.0, -2.0, 0.0, -1.0, 0.0),
            capture_radius=0.0,
            dt=dt,
            max_steps=round(400.0 / dt),
            reward=reward,
        )
        rewards, discounts, info, *_ = rollout(ApproachEnv(cfg), Schedule([0.3, -0.2, 0.1], 100.0))
        assert info["keep_out_violated"]
        returns.append(episode_return(rewards, discounts)[0])
    # Violation boundaries are roots found to 1e-9 s; every other term is exact.
    assert returns[1] == pytest.approx(returns[0], rel=1e-10)
    assert returns[2] == pytest.approx(returns[0], rel=1e-10)


@pytest.mark.parametrize("start", [CLEAR, THROUGH])
def test_progress_shaping_changes_the_return_by_minus_the_first_potential(start):
    # KAT-1: both flights end in capture, a true terminal, one after cutting through the zone.
    shaped = Reward(tau=3600.0, fuel_weight=1.0, progress_weight=0.01)
    plain = Reward(tau=3600.0, fuel_weight=1.0)
    with_shaping = rollout(ApproachEnv(ApproachConfig(start_state=start, reward=shaped)), PDAgent())
    without = rollout(ApproachEnv(ApproachConfig(start_state=start, reward=plain)), PDAgent())
    assert with_shaping[3]
    gap = episode_return(*with_shaping[:2])[0] - episode_return(*without[:2])[0]
    d0 = np.linalg.norm(np.array(start[:3]) - HOLD[:3])
    assert gap == pytest.approx(0.01 * d0, rel=1e-12)


def test_truncated_episode_keeps_the_discounted_last_potential():
    cfg = ApproachConfig(
        start_state=THROUGH, max_steps=30, reward=Reward(tau=600.0, progress_weight=0.01)
    )
    rewards, discounts, info, terminated, _ = rollout(ApproachEnv(cfg), PDAgent())
    total, discount = episode_return(rewards, discounts)
    assert not terminated
    assert total == pytest.approx(0.01 * (500.0 - discount * info["dist_to_hold_m"]), rel=1e-12)


def test_success_bonus_is_withheld_after_a_keep_out_violation():
    bonus = Reward(success_bonus=10.0)
    rewards, _, info, terminated, _ = rollout(
        ApproachEnv(ApproachConfig(start_state=THROUGH, reward=bonus)), PDAgent()
    )
    assert info["captured"] and info["keep_out_violated"] and terminated
    assert np.all(rewards == 0.0)
    rewards, _, info, _, _ = rollout(
        ApproachEnv(ApproachConfig(start_state=CLEAR, reward=bonus)), PDAgent()
    )
    assert info["captured"] and not info["keep_out_violated"]
    assert rewards[-1] == 10.0 and np.all(rewards[:-1] == 0.0)


def test_going_around_the_target_earns_more_than_cutting_through():
    # Sutton review C2: the legacy reward pays the capture bonus after cutting through the
    # keep-out zone; under the aligned reward the flight around it has the higher return.
    aligned = Reward(
        tau=3600.0,
        success_bonus=10.0,
        fuel_weight=1.0,
        time_weight=1e-3,
        violation_weights={"keep_out": 0.1},
        entry_weights={"keep_out": 5.0},
    )
    cfg = ApproachConfig(start_state=THROUGH, reward=aligned)
    through = rollout(ApproachEnv(cfg), PDAgent())
    around = rollout(ApproachEnv(cfg), Waypoints([(120.0, 150.0, 0.0), (120.0, -100.0, 0.0)]))
    assert through[2]["keep_out_violated"] and through[2]["captured"]
    assert not around[2]["keep_out_violated"] and around[2]["captured"]
    assert episode_return(*around[:2])[0] > episode_return(*through[:2])[0] + 5.0
    legacy = rollout(ApproachEnv(ApproachConfig(start_state=THROUGH)), PDAgent())
    assert legacy[0][-1] > 9.0


def test_discounted_lqr_value_is_the_return_of_its_own_policy():
    # Sutton review KAT-3: under the exactly discretized quadratic cost the N-step return of
    # u = -K x plus the bootstrap Gamma_N V(x_N) equals V(x_0) = -x_0^T P x_0, with P from the
    # Riccati equation for (sqrt(gamma) A_d, sqrt(gamma) B_d) and the cost's cross term.
    dt, tau = 10.0, 3600.0
    n = ApproachEnv().n
    q, r = np.diag([1e-4] * 3 + [1.0] * 3), 100.0 * np.eye(3)
    w = QuadraticCost(n, q, r).matrix(dt, tau)
    ad, bd = cw_discrete(n, dt)
    sa, sb = np.sqrt(np.exp(-dt / tau)) * ad, np.sqrt(np.exp(-dt / tau)) * bd
    p = solve_discrete_are(sa, sb, w[:6, :6], w[6:9, 6:9], s=w[:6, 6:9])
    k = np.linalg.solve(w[6:9, 6:9] + sb.T @ p @ sb, sb.T @ p @ sa + w[:6, 6:9].T)
    e0 = np.array([5.0, -3.0, 1.0, 0.01, -0.005, 0.002])

    class Lqr:
        name = "lqr"

        def reset(self, view, rng=None):
            pass

        def act(self, obs, info):
            return -k @ (obs - HOLD) / 0.02

    cfg = ApproachConfig(
        start_state=tuple(HOLD + e0),
        capture_radius=0.0,
        max_steps=5,
        reward=Reward(tau=tau, quadratic=QuadraticCost(n, q, r, HOLD)),
    )
    rewards, discounts, _, _, states = rollout(ApproachEnv(cfg), Lqr())
    total, discount = episode_return(rewards, discounts)
    e_end = states[-1] - HOLD
    value = -e0 @ p @ e0
    assert total - discount * e_end @ p @ e_end == pytest.approx(value, rel=1e-9)
    # Without the bootstrap the gap is the discounted value of the last state, 9% here.
    assert (value - total) / value == pytest.approx(discount * e_end @ p @ e_end / -value)
    assert abs(value - total) > 0.05 * abs(value)


def test_environment_reports_the_step_discount_with_a_reward():
    env = ApproachEnv(ApproachConfig(dt=7.0, reward=Reward(tau=700.0)))
    env.reset(seed=0)
    assert env.step(np.zeros(3))[-1]["discount"] == pytest.approx(np.exp(-0.01), rel=1e-15)
    env = ApproachEnv(ApproachConfig())
    env.reset(seed=0)
    assert "discount" not in env.step(np.zeros(3))[-1]


def test_environment_with_a_reward_follows_the_gymnasium_api():
    reward = Reward(tau=3600.0, success_bonus=10.0, fuel_weight=1.0, progress_weight=0.01)
    check_env(ApproachEnv(ApproachConfig(reward=reward)), skip_render_check=True)
