"""Episodes with known answers for the metric tests."""

import numpy as np
from scipy.optimize import brentq

from vbar import ApproachConfig, ApproachEnv
from vbar.agents import DriftAgent, PDAgent
from vbar.constraints import KeepOut
from vbar.dynamics import cw_discrete, cw_stm
from vbar.metrics import record_episode

# Passes about 16 m from the target in 100 s of free drift, through the 50 m keep-out sphere.
FLYBY = (0.0, 150.0, 0.0, 0.0, -1.5, 0.0)


class ConstantAgent:
    """Holds one normalized command."""

    name = "constant"

    def __init__(self, command):
        self.command = np.asarray(command, dtype=float)

    def reset(self, view, rng=None):
        pass

    def act(self, obs, info):
        return self.command.copy()


class GuardedApproach(ApproachEnv):
    """The approach scenario with extra constraints after its keep-out sphere.

    `goal` replaces the public goal metrics read; with one, the episode runs to its step limit.
    """

    def __init__(self, config, extra=(), soft=(), goal=None):
        self.extra = tuple(extra)
        self.soft_constraints = tuple(soft)
        self.goal = goal
        super().__init__(config)

    def _constraints(self):
        return (KeepOut(self.cfg.keep_out_radius), *self.extra)

    def _goal(self):
        return super()._goal() if self.goal is None else dict(self.goal)

    def _evaluate(self, record):
        reward, terminated, extra = super()._evaluate(record)
        return reward, terminated and self.goal is None, extra


# Docking defaults: port 2 m behind the target, ATV-class probe-drogue window 0.05 to 0.10 m/s
# (Fehse appendix C), lateral 0.1 m and 0.03 m/s inside his 7.1.1 ranges.
DOCKING_GOAL = {
    "port": (0.0, -2.0, 0.0),
    "port_axis": (0.0, -1.0, 0.0),
    "lateral_limit": 0.1,
    "lateral_speed_limit": 0.03,
    "axial_speed_window": (0.05, 0.10),
}


def episode(agent=None, seed=0, agent_seed=1, extra=(), soft=(), goal=None, **config):
    config.setdefault("start_state", FLYBY)
    config.setdefault("max_steps", 20)
    env = GuardedApproach(ApproachConfig(**config), extra, soft, goal)
    return record_episode(env, agent or DriftAgent(), seed=seed, agent_seed=agent_seed)


def pd_episode(**config):
    config.setdefault("start_state", (0.0, -400.0, 30.0, 0.0, 0.0, 0.0))
    config.setdefault("max_steps", 200)
    return episode(PDAgent(), **config)


def drift_closest_approach(start, n, end):
    """Least free-drift range over [0, end], from the closed-form STM and the root of r . v."""
    start = np.asarray(start, dtype=float)

    def state(t):
        return cw_stm(n, t) @ start

    grid = np.linspace(0.0, end, 20001)
    ranges = np.linalg.norm((cw_discrete(n, grid)[0] @ start)[:, :3], axis=1)
    k = int(np.argmin(ranges))
    if k in (0, len(grid) - 1):
        return ranges[k]
    t = brentq(lambda t: state(t)[:3] @ state(t)[3:], grid[k - 1], grid[k + 1], xtol=1e-13)
    return np.linalg.norm(state(t)[:3])


def drift_range_crossings(start, n, radius, end):
    """Times where the free-drift range crosses `radius`, from the closed-form STM."""
    start = np.asarray(start, dtype=float)

    def f(t):
        return np.linalg.norm((cw_stm(n, t) @ start)[:3]) - radius

    grid = np.linspace(0.0, end, 20001)
    values = np.linalg.norm((cw_discrete(n, grid)[0] @ start)[:, :3], axis=1) - radius
    signs = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
    return [brentq(f, grid[i], grid[i + 1], xtol=1e-13) for i in signs]
