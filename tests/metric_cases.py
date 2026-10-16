"""Episodes with known answers for the metric tests."""

import numpy as np
from scipy.optimize import brentq

from vbar import ApproachConfig, ApproachEnv
from vbar.agents import DriftAgent, PDAgent
from vbar.dynamics import cw_stm
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


def episode(agent=None, seed=0, agent_seed=1, **config):
    config.setdefault("start_state", FLYBY)
    config.setdefault("max_steps", 20)
    env = ApproachEnv(ApproachConfig(**config))
    return record_episode(env, agent or DriftAgent(), seed=seed, agent_seed=agent_seed)


def pd_episode(**config):
    config.setdefault("start_state", (0.0, -400.0, 30.0, 0.0, 0.0, 0.0))
    config.setdefault("max_steps", 200)
    return episode(PDAgent(), **config)


def drift_range_crossings(start, n, radius, end):
    """Times where the free-drift range crosses `radius`, from the closed-form STM."""
    start = np.asarray(start, dtype=float)

    def f(t):
        return np.linalg.norm((cw_stm(n, t) @ start)[:3]) - radius

    grid = np.linspace(0.0, end, 20001)
    values = np.array([f(t) for t in grid])
    signs = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
    return [brentq(f, grid[i], grid[i + 1], xtol=1e-13) for i in signs]
