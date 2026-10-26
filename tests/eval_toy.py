"""A domain that is not spacecraft rendezvous: a planar double integrator with a keep-out disk.

It exists to show that the harness (runner, provenance, statistics and report) depends only on
the task interface. The vehicle starts on a circle of radius 10 m and must reach the origin's
neighbourhood (0.5 m, 0.1 m/s) within 400 steps of 0.1 s without entering a disk of radius 1 m
centred at (5, 0) m.
"""

from dataclasses import dataclass, field

import numpy as np

from vbar.scenarios.seeds import SeedSplit

DT = 0.1
STEPS = 400
OBSTACLE = np.array([5.0, 0.0])


class PlanarEnv:
    def __init__(self, scenario):
        self.scenario = scenario
        self.max_accel = 1.0 if scenario == "planar-strong" else 0.5

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        angle = rng.uniform(-np.pi, np.pi)
        self.state = np.array([10.0 * np.cos(angle), 10.0 * np.sin(angle), 0.0, 0.0])
        return self.state.copy()

    def step(self, accel):
        a = np.clip(np.asarray(accel, dtype=float), -self.max_accel, self.max_accel)
        self.state = self.state + DT * np.array([*self.state[2:], *a])
        return self.state.copy()


@dataclass
class PlanarEpisode:
    states: list = field(default_factory=list)


class Brake:
    """Proportional-derivative to the origin, ignoring the obstacle."""

    name = "brake"

    def __init__(self, kp=1.0, kd=1.6):
        self.kp, self.kd = kp, kd

    def act(self, state):
        return -self.kp * state[:2] - self.kd * state[2:]


@dataclass(frozen=True)
class PlanarTask:
    name: str = "planar"

    def make(self, scenario):
        return PlanarEnv(scenario)

    def record(self, env, agent, seed, agent_seed):
        episode = PlanarEpisode([env.reset(seed)])
        for _ in range(STEPS):
            episode.states.append(env.step(agent.act(episode.states[-1])))
        return episode

    def score(self, env, episode):
        states = np.array(episode.states)
        clearance = np.linalg.norm(states[:, :2] - OBSTACLE, axis=1) - 1.0
        final = states[-1]
        reached = bool(np.linalg.norm(final[:2]) < 0.5 and np.linalg.norm(final[2:]) < 0.1)
        return {
            "success": reached and bool(clearance.min() >= 0.0),
            "min_clearance": float(clearance.min()),
            "final_distance": float(np.linalg.norm(final[:2])),
            "failure_label": None if reached else "timeout",
        }

    def identity(self, env):
        return {"scenario_id": f"planar:{env.scenario}:v1"}


SUMMARIES = {"success": "rate", "min_clearance": "lower_tail", "final_distance": "iqm"}


@dataclass(frozen=True)
class PlanarEntry:
    scenario: str
    split: str = "validation"
    count: int = 20

    def seeds(self):
        return SeedSplit(self.scenario, self.split, self.count)


@dataclass(frozen=True)
class PlanarSuite:
    """A suite of the toy domain, with the interface the harness reads from a `Suite`."""

    entries: tuple = (PlanarEntry("planar-weak"), PlanarEntry("planar-strong"))
    id: str = "planar-v1"

    def runs(self, identifiers=None):
        return [(identifiers[e.scenario], s, a) for e in self.entries for s, a in e.seeds().pairs()]

    def require_complete(self, results, identifiers=None):
        seen = {(r["scenario_id"], int(r["seed"])) for r in results}
        missing = [run for run in self.runs(identifiers) if run[:2] not in seen]
        if missing:
            raise ValueError(f"{len(missing)} runs are missing")
