"""Vectorized approach environment on the exact Clohessy-Wiltshire path.

`num_envs` copies of the approach scenario step together: propulsion, propagation, the
constraint minimum within each step, rewards and termination are array operations over the
batch. Each copy keeps its own `ApproachEnv` for what happens once per episode or needs its own
random stream (reset, start state, sensor), so a copy flies the episode a single environment
with the same seed would. Ended copies reset in the same step (Gymnasium's same-step
autoreset): the returned observation is the new episode's, and the ended episode's last
observation and info are in `info["final_obs"]` and `info["final_info"]`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import vector
from gymnasium.vector.utils import batch_space

from ..agents.base import DISC27, AgentView
from ..constraints import SegmentMinimum, segment_minima, violation_intervals
from ..dynamics.cw import CWDiscretizer, cw_states
from ..rewards import Transition
from ..spacecraft import Actuation, PerfectNav
from .approach import ApproachConfig, ApproachEnv
from .base import SegmentTrace


class ApproachVectorEnv(vector.VectorEnv):
    """`num_envs` approach episodes stepped as one batch, with Gymnasium's vector API.

    Supports the "cw" dynamics with the constant-acceleration drive, both thruster layouts, the
    minimum impulse bit, propellant limits, every sensor, the "normalized", "accel", "thrust"
    and "disc27" action modes, the time and propellant observation channels and library
    rewards. Actuation delay, the constant-thrust drive, execution errors and the
    "impulse_coast" mode are refused. `reset(seed=s)` seeds copy i with s + i; a list gives each
    copy its own seed. Info entries are arrays over the copies.
    """

    def __init__(self, config: ApproachConfig | None = None, num_envs: int = 8):
        cfg = config or ApproachConfig()
        refused = {
            "dynamics other than cw": cfg.dynamics != "cw",
            "actuation delay": cfg.delay_s != 0.0,
            "the constant-thrust drive": cfg.drive != "accel",
            "execution errors": cfg.execution is not None,
            "the impulse_coast mode": cfg.action_mode == "impulse_coast",
        }
        for what, used in refused.items():
            if used:
                raise ValueError(f"the vector environment does not support {what}")
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.cfg = cfg
        self.envs = [ApproachEnv(cfg) for _ in range(num_envs)]
        first = self.envs[0]
        self.num_envs = num_envs
        self.single_observation_space = first.observation_space
        self.single_action_space = first.action_space
        self.observation_space = batch_space(first.observation_space, num_envs)
        self.action_space = batch_space(first.action_space, num_envs)
        self.metadata: dict[str, Any] = {"render_modes": []}
        if hasattr(vector, "AutoresetMode"):
            self.metadata["autoreset_mode"] = vector.AutoresetMode.SAME_STEP
        self.render_mode = None
        self.spec = None
        self.closed = False
        self.viewer = None
        self.n = first.n
        self.constraints = first.constraints
        self.hold_point = first.hold_point
        craft = first.spacecraft
        self.dry_mass, self.initial_mass = craft.dry_mass, craft.initial_mass
        self.exhaust_speed = craft.exhaust_speed
        count = max(2, round(cfg.violation_samples))
        self.times = np.linspace(0.0, cfg.dt, count)
        disc = CWDiscretizer(self.n)
        self.phi, self.gamma = disc.matrices(cfg.dt)
        self.phis, self.gammas = disc.sample_matrices(cfg.dt, count)
        self.perfect = isinstance(cfg.sensor, PerfectNav)
        self.state = np.zeros((num_envs, 6))

    @property
    def scenario_id(self) -> str:
        return self.envs[0].scenario_id

    def agent_view(self) -> AgentView:
        return self.envs[0].agent_view()

    def close(self, **kwargs) -> None:
        self.closed = True

    def reset(self, *, seed: int | list[int | None] | None = None, options=None):
        if seed is None or isinstance(seed, int | np.integer):
            seeds = [None if seed is None else int(seed) + i for i in range(self.num_envs)]
        else:
            seeds = list(seed)
            if len(seeds) != self.num_envs:
                raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        n, names = self.num_envs, [c.name for c in self.constraints]
        self.t, self.steps = np.zeros(n), np.zeros(n, dtype=int)
        self.mass = np.full(n, self.initial_mass)
        self.dv_l2, self.dv_l1, self.propellant_used = np.zeros(n), np.zeros(n), np.zeros(n)
        self.min_margins = {name: np.zeros(n) for name in names}
        self.min_bounds = {name: np.zeros(n) for name in names}
        self.inside = {name: np.zeros(n, dtype=bool) for name in names}
        self.violated, self.captured = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        self.violation_steps = np.zeros(n, dtype=int)
        self.dist_to_hold = np.zeros(n)
        self.obs = np.zeros((n, len(self.envs[0].observation_names)))
        for i in range(n):
            self.obs[i] = self.envs[i].reset(seed=seeds[i], options=options)[0]
            self._load(i)
        return self.obs.copy(), self._info()

    def _load(self, i: int) -> None:
        """Take copy i's state from its freshly reset environment."""
        env = self.envs[i]
        self.state[i] = env.state
        self.t[i], self.steps[i], self.mass[i] = 0.0, 0, env.spacecraft.mass
        self.dv_l2[i] = self.dv_l1[i] = self.propellant_used[i] = 0.0
        for c in self.constraints:
            self.min_margins[c.name][i] = env.min_margins[c.name]
            self.min_bounds[c.name][i] = env.min_margin_bounds[c.name]
            self.inside[c.name][i] = False
        self.violated[i], self.captured[i] = env.violated, False
        self.violation_steps[i] = env.violation_steps
        self.dist_to_hold[i] = env.dist_to_hold

    def step(self, actions):
        cfg = self.cfg
        accel, tau, used = self._burn(self._limit(self._commands(actions)))
        start, t0 = self.state.copy(), self.t.copy()
        # Rows whose propellant runs out inside the step fly two segments through their own
        # environment; every other row is one segment of the whole step.
        runout = tau < cfg.dt
        traces = {int(i): self._fly_one(int(i), accel[i], tau[i]) for i in np.flatnonzero(runout)}
        samples = np.einsum("mij,nj->nmi", self.phis, start)
        samples += np.einsum("mij,nj->nmi", self.gammas, accel)
        self.state = start @ self.phi.T + accel @ self.gamma.T
        thrust = np.linalg.norm(accel, axis=1)
        state_at = self._state_at(start, accel)
        minima = {}
        for c in self.constraints:
            minima[c.name] = segment_minima(c, state_at, self.times, samples, thrust, self.n, t0)
        for i, trace in traces.items():
            self.state[i] = self.envs[i].state
            for name, found in trace.minima.items():
                for column, value in enumerate((found.t, found.margin, found.lower_bound)):
                    minima[name][column][i] = value
        dv = accel * tau[:, None]
        step_dv_l2 = np.linalg.norm(dv, axis=1)
        self.mass = np.where(runout, self.dry_mass, self.mass - used)
        self.t = self.t + cfg.dt
        self.steps = self.steps + 1
        self.dv_l2 += step_dv_l2
        self.dv_l1 += np.sum(np.abs(dv), axis=1)
        self.propellant_used += used
        for name, (_, margin, bound) in minima.items():
            self.min_margins[name] = np.minimum(self.min_margins[name], margin)
            self.min_bounds[name] = np.minimum(self.min_bounds[name], bound)
        self.obs = self._observe()
        rewards, terminated = self._evaluate(step_dv_l2, minima)
        truncated = self.steps >= cfg.max_steps
        info = self._info()
        if cfg.reward is not None:
            spans = self._violations(samples, start, accel, t0, minima, traces)
            transition = self._transition(start, t0, accel, tau, spans, traces, terminated)
            rewards = np.asarray(cfg.reward(transition), dtype=float)
            info["discount"] = cfg.reward.discount(np.full(self.num_envs, cfg.dt))
        ended = terminated | truncated
        if ended.any():
            final_obs = np.where(ended[:, None], self.obs, np.nan)
            final_info = info
            for i in np.flatnonzero(ended):
                self.obs[i] = self.envs[i].reset()[0]
                self._load(int(i))
            info = {**self._info(), **{k: v for k, v in final_info.items() if k == "discount"}}
            info.update(
                final_obs=final_obs,
                _final_obs=ended.copy(),
                final_info=final_info,
                _final_info=ended.copy(),
            )
        return self.obs.copy(), rewards, terminated, truncated, info

    def _commands(self, actions) -> np.ndarray:
        """Each row's action as fractions of a full command at the current mass, (N, 3)."""
        if self.cfg.action_mode == "disc27":
            index = np.asarray(actions).astype(int).reshape(self.num_envs)
            if np.any((index < 0) | (index >= len(DISC27))):
                raise ValueError("disc27 actions are 0 to 26")
            return DISC27[index]
        commands = np.array(actions, dtype=float).reshape(self.num_envs, 3)
        max_force = self.cfg.max_accel * self.mass
        if self.cfg.action_mode == "accel":
            return commands / (max_force / self.mass)[:, None]
        if self.cfg.action_mode == "thrust":
            return commands / max_force[:, None]
        return commands

    def _limit(self, commands: np.ndarray) -> np.ndarray:
        """The thruster limits and minimum impulse bit of `Spacecraft.limit`, row by row."""
        if self.cfg.thrust_limit == "box":
            limited = np.clip(commands, -1.0, 1.0)
            impulse = np.abs(limited)
        else:
            length = np.linalg.norm(commands, axis=1, keepdims=True)
            limited = np.where(length > 1.0, commands / np.maximum(length, 1.0), commands)
            impulse = np.repeat(np.minimum(length, 1.0), 3, axis=1)
        impulse = impulse * (self.cfg.max_accel * self.mass * self.cfg.dt)[:, None]
        small = (impulse > 0.0) & (impulse < self.cfg.min_impulse)
        return np.where(small, 0.0, limited)

    def _burn(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each row's acceleration, burn time and propellant used, as `Spacecraft._burn`."""
        cfg, ve = self.cfg, self.exhaust_speed
        burning = np.any(u != 0.0, axis=1) & (self.mass > self.dry_mass)
        accel = np.where(burning[:, None], u * cfg.max_accel, 0.0)
        if cfg.thrust_limit == "box":
            rate = np.sum(np.abs(accel), axis=1)
        else:
            rate = np.linalg.norm(accel, axis=1)
        tau = np.full(self.num_envs, cfg.dt)
        if self.dry_mass > 0.0:
            # Burn time until the propellant is gone, from m0 exp(-rate t / ve) = m_dry.
            with np.errstate(divide="ignore", invalid="ignore"):
                left = ve / rate * np.log(self.mass / self.dry_mass)
            tau = np.where(burning, np.minimum(cfg.dt, left), cfg.dt)
        used = np.where(burning, -self.mass * np.expm1(-rate * tau / ve), 0.0)
        used = np.where(tau < cfg.dt, self.mass - self.dry_mass, used)
        return accel, tau, used

    def _fly_one(self, i: int, accel: np.ndarray, tau: float) -> SegmentTrace:
        """A step in which copy i's propellant runs out, flown by its own environment."""
        env = self.envs[i]
        env.state = self.state[i].copy()
        env.dynamics.reset(env.state)
        env.t = float(self.t[i])
        segments = ((float(tau), accel), (self.cfg.dt - float(tau), np.zeros(3)))
        return env._fly(Actuation(segments, 0.0, 0.0, 0.0))

    def _state_at(self, start: np.ndarray, accel: np.ndarray):
        """state_at(rows, tau): the states `tau` s into the step of the copies `rows`."""

        def state_at(rows: np.ndarray, tau: np.ndarray) -> np.ndarray:
            return cw_states(self.n, tau, start[rows], accel[rows])

        return state_at

    def _observe(self) -> np.ndarray:
        if self.perfect:
            parts = [self.state.copy()]
        else:
            observed = [
                env.sensor.observe(x, t)
                for env, x, t in zip(self.envs, self.state, self.t, strict=True)
            ]
            parts = [np.array(observed)]
        channels = self.cfg.observation_channels
        if "t_remaining" in channels:
            parts.append(((self.cfg.max_steps - self.steps) * self.cfg.dt)[:, None])
        if "propellant_fraction" in channels:
            left = (self.mass - self.dry_mass) / (self.initial_mass - self.dry_mass)
            parts.append(left[:, None])
        return np.concatenate(parts, axis=1)

    def _evaluate(self, step_dv: np.ndarray, minima) -> tuple[np.ndarray, np.ndarray]:
        """The approach scenario's legacy reward and termination, row by row."""
        cfg = self.cfg
        previous = self.dist_to_hold
        self.dist_to_hold = np.linalg.norm(self.state[:, :3] - self.hold_point, axis=1)
        entered = minima["keep_out"][1] < 0.0
        self.violated |= entered
        self.violation_steps += entered
        speed = np.linalg.norm(self.state[:, 3:], axis=1)
        self.captured = (self.dist_to_hold < cfg.capture_radius) & (speed < cfg.capture_speed)
        reward = (previous - self.dist_to_hold) / 100.0 - cfg.reward_dv_weight * step_dv
        reward = reward - entered + 10.0 * self.captured
        terminated = self.captured | (entered & cfg.terminate_on_violation)
        return reward, terminated

    def _violations(self, samples, start, accel, t0, minima, traces):
        """Each constraint's violation intervals (episode times) for every row."""
        spans: dict[str, list[tuple[tuple[float, float], ...]]] = {}
        for c in self.constraints:
            t_min, margin, bound = minima[c.name]
            rows: list[tuple[tuple[float, float], ...]] = [()] * self.num_envs
            for i in np.flatnonzero(margin < 0.0):
                if int(i) in traces:
                    rows[i] = traces[int(i)].violations[c.name]
                    continue
                at = self._state_at(start[i : i + 1], accel[i : i + 1])

                def state_at(tau: float, at=at) -> np.ndarray:
                    return at(np.zeros(1, dtype=int), np.array([tau]))[0]

                found = SegmentMinimum(t_min[i], margin[i], bound[i])
                rows[i] = violation_intervals(c, state_at, self.times, samples[i], found, t0[i])
            spans[c.name] = rows
        return spans

    def _transition(self, start, t0, accel, tau, spans, traces, terminated) -> Transition:
        """The batch's step as rewards see it; updates the violation-entry state."""
        n, dt = self.num_envs, self.cfg.dt
        width = 2 if traces else 1
        segments = np.zeros((n, width, 5))
        segments[:, 0, 1], segments[:, 0, 2:5] = tau, accel
        starts = np.repeat(start[:, None, :], width, axis=1)
        if width == 2:
            segments[:, 1, 0] = tau
            segments[:, 1, 1] = dt - tau
            for i, trace in traces.items():
                starts[i] = trace.starts
        order = 1 if self.cfg.thrust_limit == "box" else 2
        fuel_rates = np.linalg.norm(segments[:, :, 2:5], ord=order, axis=2)
        violations, entries = {}, {}
        for name, rows in spans.items():
            depth = max([len(r) for r in rows] + [0])
            offsets = np.zeros((n, depth, 2))
            begun = np.full((n, depth), np.inf)
            for i, r in enumerate(rows):
                if not r:
                    continue
                offsets[i, : len(r)] = np.array(r) - t0[i]
                new = [
                    a for a, _ in offsets[i, : len(r)] if not (a == 0.0 and self.inside[name][i])
                ]
                begun[i, : len(new)] = new
            for i in range(n):
                closed = len(rows[i]) and offsets[i, len(rows[i]) - 1, 1] >= dt - 1e-9
                self.inside[name][i] = bool(closed)
            violations[name], entries[name] = offsets, begun
        return Transition(
            duration=np.full(n, dt),
            segments=segments,
            starts=starts,
            fuel_rates=fuel_rates,
            violations=violations,
            entries=entries,
            goal_distance=np.linalg.norm(start[:, :3] - self.hold_point, axis=1),
            next_goal_distance=self.dist_to_hold.copy(),
            success=self.captured & ~self.violated,
            terminal=terminated,
        )

    def _info(self) -> dict[str, Any]:
        radius = self.cfg.keep_out_radius
        return {
            "time_s": self.t.copy(),
            "true_state": self.state.copy(),
            "dv_used_mps": self.dv_l2.copy(),
            "dv_axes_mps": self.dv_l1.copy(),
            "propellant_used_kg": self.propellant_used.copy(),
            "mass_kg": self.mass.copy(),
            "min_margins": {k: v.copy() for k, v in self.min_margins.items()},
            "min_margin_bounds": {k: v.copy() for k, v in self.min_bounds.items()},
            "dist_to_hold_m": np.linalg.norm(self.state[:, :3] - self.hold_point, axis=1),
            "range_to_target_m": np.linalg.norm(self.state[:, :3], axis=1),
            "speed_mps": np.linalg.norm(self.state[:, 3:], axis=1),
            "min_range_to_target_m": self.min_margins["keep_out"] + radius,
            "keep_out_violated": self.violated.copy(),
            "violation_steps": self.violation_steps.copy(),
            "captured": self.captured.copy(),
        }
