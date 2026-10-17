"""Base environment: the step loop every scenario shares.

A scenario supplies its start-state distribution, constraints, goal and success test; the base
environment composes the dynamics model, the spacecraft and the navigation sensor. Each step
holds the agent's command for `dt` s: the spacecraft turns it into segments of constant
acceleration (actuation delay and propellant run-out are boundaries), the dynamics model
propagates each segment, and each constraint's minimum margin over each segment is found in
continuous time. The agent sees only the sensor's output and `agent_info()`; the true state goes
to the `info` dict and the step's full `record` for metrics and logging.

The episode seed drives Gymnasium's generator, which samples the start state; the sensor,
actuator and target streams are spawned from the same seed sequence, so changing one model does
not change another's random draws. The agent's stream is seeded separately by the runner, so
nothing an agent holds is derived from the episode seed.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..agents.base import DISC27, AgentView
from ..constraints import Constraint, SegmentMinimum, segment_minimum, violation_intervals
from ..dynamics import make_dynamics
from ..rewards import Reward, Transition
from ..spacecraft import Actuation, ExecutionError, PerfectNav, Sensor, Spacecraft

ACTION_MODES = ("normalized", "accel", "thrust", "disc27", "impulse_coast")
SCALAR_CHANNELS = ("t_remaining", "propellant_fraction")
CHANNELS = (*SCALAR_CHANNELS, "pending_commands")
STREAMS = ("sensor", "actuator", "target")


@dataclass(kw_only=True)
class EnvConfig:
    """Parameters every scenario shares (SI units, angles in rad).

    `action_mode` "normalized" takes fractions of `max_accel` per Hill axis, "accel" takes
    m/s^2, "thrust" newtons and "disc27" an index into the 27 bang-off-bang commands
    `vbar.agents.DISC27`. "impulse_coast" takes (fx, fy, fz, coast s): fractions of a full
    command held for one step, then a coast rounded to whole steps, all in one decision.
    `thrust_limit` is the thruster layout ("box" or "ball") and `drive` holds either the
    acceleration ("accel") or the thrust ("thrust") over a burn.
    `violation_samples` is the number of constraint samples per step before refinement.

    `observation_channels` appends, in this order, any of: "t_remaining" (s), "propellant_fraction"
    (propellant left over propellant at start) and "pending_commands", which lists the commands
    that will still act after this step, the one in effect first and then those not yet in effect,
    each as (seconds until it takes effect, command in the action mode's units); it has
    ceil(delay_s / dt) slots, zero-filled at the front before the first commands.

    `reward` replaces the scenario's legacy reward with a `vbar.rewards.Reward`; `info` then
    holds each step's `discount`. `training` allows the training-only options of `reset`
    (start states injected or drawn from wider ranges); scored runs refuse such environments.
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
    drive: str = "accel"
    isp: float = 220.0
    mass_kg: float = 500.0
    propellant_kg: float | None = None
    min_impulse: float = 0.0
    delay_s: float = 0.0
    execution: ExecutionError | None = None
    sensor: Sensor = field(default_factory=PerfectNav)
    violation_samples: int = 20
    observation_channels: tuple[str, ...] = ()
    reward: Reward | None = None
    training: bool = False


@dataclass(frozen=True)
class SegmentTrace:
    """One step in continuous time: its constant-acceleration segments, samples and minima.

    `segments` are (start time s, duration s, Hill acceleration m/s^2); `times` (s) and
    `states` are the samples of all segments in order; `minima` is each constraint's lowest
    margin over the step; `violations` holds, for each constraint, the (start, end) times (s)
    of the step's intervals with a negative margin, merged across segment boundaries; `starts`
    is the true state at the start of each segment.
    """

    segments: tuple[tuple[float, float, np.ndarray], ...]
    times: np.ndarray
    states: np.ndarray
    minima: Mapping[str, SegmentMinimum]
    violations: Mapping[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)
    starts: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))


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


def _joined(first: Actuation, second: Actuation) -> Actuation:
    """Two consecutive actuations as one."""
    return Actuation(
        first.segments + second.segments,
        first.dv_l2 + second.dv_l2,
        first.dv_l1 + second.dv_l1,
        first.propellant_kg + second.propellant_kg,
        first.saturated or second.saturated,
        first.below_mib or second.below_mib,
        first.empty or second.empty,
    )


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
    # Constraints scored but not counted as failures; every other constraint is hard.
    soft_constraints: tuple[str, ...] = ()
    # Keys `reset(options=...)` accepts when the environment is built for training.
    reset_options: tuple[str, ...] = ("start_state",)

    def __init__(self, config: EnvConfig):
        super().__init__()
        if not config.dt > 0.0:
            raise ValueError("dt must be positive")
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
            drive=config.drive,
            execution=config.execution,
        )
        # Sensors hold episode state (stream, bias, correlated error), so environments built
        # from one config must not share one.
        self.sensor = copy.deepcopy(config.sensor)
        self.constraints = tuple(self._constraints())
        unknown = set(config.observation_channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown observation channels {sorted(unknown)}, expected {CHANNELS}")
        self.pending_slots = int(np.ceil(config.delay_s / config.dt - 1e-9))
        self.observation_names = self._observation_names()
        self.action_space: spaces.Space
        if config.action_mode == "disc27":
            self.action_space = spaces.Discrete(len(DISC27))
        elif config.action_mode == "impulse_coast":
            low = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
            top = np.array([1.0, 1.0, 1.0, config.max_steps * config.dt], dtype=np.float32)
            self.action_space = spaces.Box(low, top, dtype=np.float32)
        else:
            high = {"normalized": 1.0, "accel": config.max_accel}.get(
                config.action_mode, config.max_accel * config.mass_kg
            )
            self.action_space = spaces.Box(-high, high, shape=(3,), dtype=np.float32)
        shape = (len(self.observation_names),)
        self.observation_space = spaces.Box(-1e7, 1e7, shape=shape, dtype=np.float64)
        self.state = np.zeros(6)
        self.t = 0.0

    def _sample_start(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def _constraints(self) -> tuple[Constraint, ...]:
        return ()

    def _goal(self) -> dict[str, Any]:
        return {}

    def _evaluate(self, record: StepRecord) -> tuple[float, bool, dict[str, Any]]:
        """Legacy reward, whether the episode ends, and scenario entries for `info`."""
        return 0.0, False, {}

    def _goal_distance(self, state: np.ndarray) -> float:
        """Distance (m) from the state to the goal, for progress shaping."""
        return 0.0

    def _success(self) -> bool:
        """Whether the success predicate holds now with no hard constraint violated (M1)."""
        return False

    @property
    def scenario_id(self) -> str:
        base = f"vbar:{self.scenario_name}:{self.cfg.dynamics}:{self.cfg.thrust_limit}:dv:v0"
        return base + (":disc27" if self.cfg.action_mode == "disc27" else "")

    def agent_view(self) -> AgentView:
        return AgentView(
            scenario_id=self.scenario_id,
            dt=self.cfg.dt,
            max_steps=self.cfg.max_steps,
            n=self.n,
            action_mode=self.cfg.action_mode,
            max_accel=self.cfg.max_accel,
            thrust_limit=self.cfg.thrust_limit,
            mass_kg=self.cfg.mass_kg,
            goal=self._goal(),
            constraints=tuple(public_spec(c) for c in self.constraints),
            observation=self.observation_names,
        )

    def agent_info(self) -> dict[str, Any]:
        """The part of `info` agents may see."""
        return {"time_s": self.t}

    def reset(self, *, seed=None, options=None):
        """Start an episode; `options` are training-only and last for this episode alone.

        "start_state" sets the start state; scenarios may accept more (`reset_options`).
        """
        self.options = dict(options or {})
        if self.options and not self.cfg.training:
            raise ValueError(
                "reset options are for training; build the environment with training=True"
            )
        unknown = set(self.options) - set(self.reset_options)
        if unknown:
            raise ValueError(
                f"unknown reset options {sorted(unknown)}, expected {self.reset_options}"
            )
        super().reset(seed=seed)
        children = self.np_random.bit_generator.seed_seq.spawn(len(STREAMS))
        self.rngs = {
            name: np.random.default_rng(s) for name, s in zip(STREAMS, children, strict=True)
        }
        if "start_state" in self.options:
            self.state = np.array(self.options["start_state"], dtype=float).reshape(6)
        else:
            self.state = np.asarray(self._sample_start(self.np_random), dtype=float)
        self.dynamics.reset(self.state)
        self.spacecraft.reset(self.rngs["actuator"])
        self.sensor.reset(self.rngs["sensor"])
        self.t = 0.0
        self.steps = 0
        self.dv_l2 = self.dv_l1 = self.propellant_used = 0.0
        self.min_margins = {c.name: float(c.margin(self.state, 0.0)) for c in self.constraints}
        self.min_margin_bounds = dict(self.min_margins)
        self.inside = {c.name: False for c in self.constraints}
        self.obs = self._observe()
        self.record = None
        return self.obs.copy(), self._info()

    def step(self, action):
        coast_steps = 0
        if self.cfg.action_mode == "disc27":
            command = DISC27[self._index(action)].copy()
        else:
            command = np.array(action, dtype=float)
        if self.cfg.action_mode == "impulse_coast":
            coast_steps = self._coast_steps(command)
            fraction = command[:3]
        else:
            fraction = command / self._action_scale()
        actuation = self.spacecraft.apply(fraction, self.cfg.dt)
        if coast_steps:
            coast = self.spacecraft.apply(np.zeros(3), coast_steps * self.cfg.dt)
            actuation = _joined(actuation, coast)
        trace = self._fly(actuation)
        self.t += (1 + coast_steps) * self.cfg.dt
        self.steps += 1 + coast_steps
        self.dv_l2 += actuation.dv_l2
        self.dv_l1 += actuation.dv_l1
        self.propellant_used += actuation.propellant_kg
        for name, found in trace.minima.items():
            self.min_margins[name] = min(self.min_margins[name], found.margin)
            self.min_margin_bounds[name] = min(self.min_margin_bounds[name], found.lower_bound)
        self.obs = self._observe()
        record = StepRecord(
            self.t, self.state.copy(), self.obs.copy(), command, command, actuation, trace
        )
        reward, terminated, extra = self._evaluate(record)
        truncated = self.steps >= self.cfg.max_steps
        self.record = record
        info = self._info()
        info.update(extra)
        if self.cfg.reward is not None:
            transition = self.transition(record, terminated)
            reward = float(self.cfg.reward(transition))
            info["discount"] = float(self.cfg.reward.discount(transition.duration))
        return self.obs.copy(), float(reward), terminated, truncated, info

    def transition(self, record: StepRecord, terminated: bool) -> Transition:
        """The step as rewards see it; call once per step, since it tracks violation entries."""
        trace = record.segment
        start = trace.segments[0][0]
        duration = sum(d for _, d, _ in trace.segments)
        rows = np.array([[t - start, d, *a] for t, d, a in trace.segments]).reshape(-1, 5)
        accels = rows[:, 2:5]
        norm_order = 1 if self.cfg.thrust_limit == "box" else 2
        violations, entries = {}, {}
        for name, spans in trace.violations.items():
            offsets = np.array(spans, dtype=float).reshape(-1, 2) - start
            # An interval open at the step's start continues one from the previous step.
            new = [a for a, _ in offsets if not (a == 0.0 and self.inside[name])]
            violations[name] = offsets
            entries[name] = np.array(new, dtype=float)
            self.inside[name] = bool(len(offsets)) and offsets[-1, 1] >= duration - 1e-9
        return Transition(
            duration=duration,
            segments=rows,
            starts=trace.starts,
            fuel_rates=np.linalg.norm(accels, ord=norm_order, axis=1),
            violations=violations,
            entries=entries,
            goal_distance=self._goal_distance(trace.starts[0]),
            next_goal_distance=self._goal_distance(record.true_state),
            success=self._success(),
            terminal=terminated,
        )

    def _fly(self, actuation: Actuation) -> SegmentTrace:
        """Propagate each constant-acceleration segment and check constraints within it.

        Each segment is its own `advance` call, so no integrator steps across a thrust change.
        Discontinuities inside a dynamics model's own forces (shadow entry and exit) are the
        model's to split.
        """
        t0 = self.t
        segments, times, states = [], [], []
        minima: dict[str, SegmentMinimum] = {}
        violations: dict[str, list[tuple[float, float]]] = {c.name: [] for c in self.constraints}
        starts = []
        for duration, accel in actuation.segments:
            if duration <= 0.0:
                continue
            starts.append(self.state.copy())
            count = max(2, round(self.cfg.violation_samples * duration / self.cfg.dt))
            self.state, sampled = self.dynamics.advance(accel, duration, n_samples=count)
            samples = cast(np.ndarray, sampled)
            tau = np.linspace(0.0, duration, count)
            thrust = float(np.linalg.norm(accel))
            for c in self.constraints:
                state_at = self.dynamics.state_in_last_segment
                found = segment_minimum(c, state_at, tau, samples, thrust, self.n, t0)
                for start, end in violation_intervals(c, state_at, tau, samples, found, t0):
                    merged = violations[c.name]
                    # An interval that runs across a segment boundary is one interval.
                    if merged and start - merged[-1][1] < 1e-9:
                        start = merged.pop()[0]
                    merged.append((start, end))
                known = minima.get(c.name)
                if known is not None:
                    lowest = found if found.margin < known.margin else known
                    bound = min(found.lower_bound, known.lower_bound)
                    found = SegmentMinimum(lowest.t, lowest.margin, bound)
                minima[c.name] = found
            segments.append((t0, duration, accel))
            times.append(t0 + tau)
            states.append(samples)
            t0 += duration
        return SegmentTrace(
            tuple(segments),
            np.concatenate(times),
            np.concatenate(states),
            minima,
            {name: tuple(spans) for name, spans in violations.items()},
            np.array(starts),
        )

    def _observation_names(self) -> tuple[str, ...]:
        names = ["x", "y", "z", "vx", "vy", "vz"]
        channels = self.cfg.observation_channels
        names += [name for name in SCALAR_CHANNELS if name in channels]
        if "pending_commands" in channels:
            for k in range(self.pending_slots):
                names += [f"pending_{k}_t", f"pending_{k}_x", f"pending_{k}_y", f"pending_{k}_z"]
        return tuple(names)

    def _observe(self) -> np.ndarray:
        """Sensor output followed by the enabled channels."""
        parts = [self.sensor.observe(self.state, self.t)]
        channels = self.cfg.observation_channels
        if "t_remaining" in channels:
            parts.append(np.array([(self.cfg.max_steps - self.steps) * self.cfg.dt]))
        if "propellant_fraction" in channels:
            craft = self.spacecraft
            parts.append(np.array([craft.propellant / (craft.initial_mass - craft.dry_mass)]))
        if "pending_commands" in channels:
            acting = [(0.0, self.spacecraft.in_effect()), *self.spacecraft.pending()]
            slots = np.zeros((self.pending_slots, 4))
            # Padding goes first, so the entries stay in order of the time they take effect.
            recent = acting[max(len(acting) - self.pending_slots, 0) :]
            for k, (wait, u) in enumerate(recent[::-1]):
                slots[-1 - k] = [wait, *(u * self._action_scale())]
            parts.append(slots.ravel())
        return np.concatenate(parts)

    def _coast_steps(self, command: np.ndarray) -> int:
        """Whole steps of coast an impulse_coast action asks for, cut at the time limit."""
        if command.shape != (4,) or not np.all(np.isfinite(command)):
            raise ValueError("impulse_coast actions are four finite numbers (fx, fy, fz, coast s)")
        left = max(self.cfg.max_steps - self.steps - 1, 0)
        return int(np.clip(round(float(command[3]) / self.cfg.dt), 0, left))

    def _index(self, action) -> int:
        index = int(np.asarray(action).item())
        if not 0 <= index < len(DISC27):
            raise ValueError(f"disc27 actions are 0 to 26, got {index}")
        return index

    def _action_scale(self) -> float:
        """Action-mode units per fraction of a full command."""
        if self.cfg.action_mode == "accel":
            return self.spacecraft.max_force / self.spacecraft.mass
        if self.cfg.action_mode == "thrust":
            return self.spacecraft.max_force
        return 1.0

    def _info(self) -> dict[str, Any]:
        return {
            "time_s": self.t,
            "true_state": self.state.copy(),
            "dv_used_mps": self.dv_l2,
            "dv_axes_mps": self.dv_l1,
            "propellant_used_kg": self.propellant_used,
            "mass_kg": self.spacecraft.mass,
            "min_margins": dict(self.min_margins),
            "min_margin_bounds": dict(self.min_margin_bounds),
        }
