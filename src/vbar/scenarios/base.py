"""Base environment: the step loop every scenario shares.

A scenario supplies its start-state distribution, constraints, goal and success test; the base
environment composes the dynamics model, the spacecraft and the navigation sensor. Each step
holds the agent's command for `dt` s: the spacecraft turns it into segments of constant
acceleration (actuation delay and propellant run-out are boundaries), the dynamics model
propagates each segment, and each constraint's minimum margin over each segment is found in
continuous time. The agent sees only the sensor's output and `agent_info()`; the true state goes
to the `info` dict and the step's full `record` for metrics and logging.

The episode seed drives Gymnasium's generator, which samples the start state; the sensor,
actuator, target and agent streams are spawned from the same seed sequence, so changing one
model does not change another's random draws.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..agents.base import AgentView
from ..constraints import Constraint, SegmentMinimum, segment_minimum
from ..dynamics import make_dynamics
from ..spacecraft import Actuation, PerfectNav, Sensor, Spacecraft

ACTION_MODES = ("normalized", "accel")
STREAMS = ("sensor", "actuator", "target", "agent")


@dataclass
class EnvConfig:
    """Parameters every scenario shares (SI units, angles in rad).

    `action_mode` "normalized" takes fractions of `max_accel` per Hill axis, "accel" takes
    m/s^2. `thrust_limit` is the thruster layout ("box" or "ball"). `violation_samples` is the
    number of constraint samples per step before refinement.
    """

    dt: float = 10.0
    max_steps: int = 360
    dynamics: str = "cw"
    body: str = "earth"
    altitude_m: float = 500e3
    inclination: float = 0.0
    action_mode: str = "normalized"
    max_accel: float = 0.02
    thrust_limit: str = "box"
    isp: float = 220.0
    mass_kg: float = 500.0
    propellant_kg: float | None = None
    min_impulse: float = 0.0
    delay_s: float = 0.0
    sensor: Sensor = field(default_factory=PerfectNav)
    violation_samples: int = 20


@dataclass(frozen=True)
class SegmentTrace:
    """One step in continuous time: its constant-acceleration segments, samples and minima.

    `segments` are (start time s, duration s, Hill acceleration m/s^2); `times` (s) and
    `states` are the samples of all segments in order; `minima` is each constraint's lowest
    margin over the step.
    """

    segments: tuple[tuple[float, float, np.ndarray], ...]
    times: np.ndarray
    states: np.ndarray
    minima: Mapping[str, SegmentMinimum]


@dataclass(frozen=True)
class StepRecord:
    """Everything that happened in one step, for metrics; agents never see it."""

    t: float
    true_state: np.ndarray
    obs: np.ndarray
    command: np.ndarray
    filtered: np.ndarray
    actuation: Actuation
    segment: SegmentTrace
    filter_result: Any = None
    agent_latency_s: float = 0.0


def public_spec(constraint: Constraint) -> dict[str, Any]:
    """A constraint's type and parameters, as agents may see them."""
    params: dict[str, Any] = {}
    if is_dataclass(constraint):
        params = {f.name: getattr(constraint, f.name) for f in fields(constraint)}
    params.pop("name", None)
    return {"type": constraint.name, **params}


class BaseEnv(gym.Env):
    """Gymnasium environment shared by all scenarios; subclasses fill in the scenario hooks."""

    metadata = {"render_modes": []}
    scenario_name = "base"
    record: StepRecord | None = None

    def __init__(self, config: EnvConfig):
        super().__init__()
        if config.action_mode not in ACTION_MODES:
            raise ValueError(f"unknown action mode {config.action_mode!r}, expected {ACTION_MODES}")
        self.cfg = config
        self.dynamics = make_dynamics(
            config.dynamics, config.body, config.altitude_m, config.inclination
        )
        self.n = self.dynamics.n
        self.spacecraft = Spacecraft(
            max_accel=config.max_accel,
            layout=config.thrust_limit,
            isp=config.isp,
            mass=config.mass_kg,
            propellant=config.propellant_kg,
            min_impulse=config.min_impulse,
            delay=config.delay_s,
        )
        self.sensor = config.sensor
        self.constraints = tuple(self._constraints())
        high = 1.0 if config.action_mode == "normalized" else config.max_accel
        self.action_space = spaces.Box(-high, high, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-1e7, 1e7, shape=(6,), dtype=np.float64)
        self.state = np.zeros(6)
        self.t = 0.0

    # Scenario hooks.

    def _sample_start(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def _constraints(self) -> tuple[Constraint, ...]:
        return ()

    def _goal(self) -> dict[str, Any]:
        return {}

    def _evaluate(self, record: StepRecord) -> tuple[float, bool, dict[str, Any]]:
        """Reward, whether the episode ends, and scenario entries for `info`."""
        return 0.0, False, {}

    # Public interface.

    @property
    def scenario_id(self) -> str:
        return f"vbar:{self.scenario_name}:{self.cfg.dynamics}:{self.cfg.thrust_limit}:dv:v0"

    def agent_view(self) -> AgentView:
        return AgentView(
            scenario_id=self.scenario_id,
            dt=self.cfg.dt,
            max_steps=self.cfg.max_steps,
            n=self.n,
            action_mode=self.cfg.action_mode,
            max_accel=self.cfg.max_accel,
            thrust_limit=self.cfg.thrust_limit,
            goal=self._goal(),
            constraints=tuple(public_spec(c) for c in self.constraints),
        )

    def agent_info(self) -> dict[str, Any]:
        """The part of `info` agents may see."""
        return {"time_s": self.t}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        children = self.np_random.bit_generator.seed_seq.spawn(len(STREAMS))
        self.rngs = {
            name: np.random.default_rng(s) for name, s in zip(STREAMS, children, strict=True)
        }
        self.state = np.asarray(self._sample_start(self.np_random), dtype=float)
        self.dynamics.reset(self.state)
        self.spacecraft.reset()
        self.sensor.reset(self.rngs["sensor"])
        self.t = 0.0
        self.steps = 0
        self.dv_l2 = self.dv_l1 = self.propellant_used = 0.0
        self.min_margins = {c.name: float(c.margin(self.state, 0.0)) for c in self.constraints}
        self.obs = self.sensor.observe(self.state, self.t)
        self.record = None
        return self.obs.copy(), self._info()

    def step(self, action):
        command = np.asarray(action, dtype=float)
        fraction = command if self.cfg.action_mode == "normalized" else command / self.cfg.max_accel
        actuation = self.spacecraft.apply(fraction, self.cfg.dt)
        trace = self._fly(actuation)
        self.t += self.cfg.dt
        self.steps += 1
        self.dv_l2 += actuation.dv_l2
        self.dv_l1 += actuation.dv_l1
        self.propellant_used += actuation.propellant_kg
        for name, found in trace.minima.items():
            self.min_margins[name] = min(self.min_margins[name], found.margin)
        self.obs = self.sensor.observe(self.state, self.t)
        record = StepRecord(
            self.t, self.state.copy(), self.obs.copy(), command, command, actuation, trace
        )
        reward, terminated, extra = self._evaluate(record)
        truncated = self.steps >= self.cfg.max_steps
        self.record = record
        info = self._info()
        info.update(extra)
        return self.obs.copy(), float(reward), terminated, truncated, info

    def _fly(self, actuation: Actuation) -> SegmentTrace:
        """Propagate each constant-acceleration segment and check constraints within it."""
        t0 = self.t
        segments, times, states = [], [], []
        minima: dict[str, SegmentMinimum] = {}
        for duration, accel in actuation.segments:
            if duration <= 0.0:
                continue
            count = max(2, round(self.cfg.violation_samples * duration / self.cfg.dt))
            self.state, samples = self.dynamics.advance(accel, duration, n_samples=count)
            assert samples is not None
            tau = np.linspace(0.0, duration, count)
            for c in self.constraints:
                found = segment_minimum(c, self.dynamics.state_in_last_segment, tau, samples, t0)
                if c.name not in minima or found.margin < minima[c.name].margin:
                    minima[c.name] = found
            segments.append((t0, duration, accel))
            times.append(t0 + tau)
            states.append(samples)
            t0 += duration
        return SegmentTrace(tuple(segments), np.concatenate(times), np.concatenate(states), minima)

    def _info(self) -> dict[str, Any]:
        return {
            "time_s": self.t,
            "true_state": self.state.copy(),
            "dv_used_mps": self.dv_l2,
            "dv_axes_mps": self.dv_l1,
            "propellant_used_kg": self.propellant_used,
            "mass_kg": self.spacecraft.mass,
            "min_margins": dict(self.min_margins),
        }
