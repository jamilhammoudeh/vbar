"""Base environment: the step loop every scenario shares.

A scenario supplies its start-state distribution, constraints, goal and success test; the base
environment composes the dynamics model, the spacecraft and the navigation sensor. Each step
holds the agent's command for `dt` s: the spacecraft turns it into segments of constant
acceleration (actuation delay and propellant run-out are boundaries), the dynamics model
propagates each segment, and each constraint's minimum margin over each segment is found in
continuous time. The exception is the forward-Euler rung `euler1`, a discrete plant with no state
between its steps: its constraints are judged at the nodes alone. The agent sees only the
sensor's output and `agent_info()`; the true state goes to the `info` dict and the step's full
`record` for metrics and logging.

The episode seed drives Gymnasium's generator, which samples the start state; the sensor,
actuator and target streams are spawned from the same seed sequence, so changing one model does
not change another's random draws. The agent's stream is seeded separately by the runner, so
nothing an agent holds is derived from the episode seed.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..agents.base import DISC27, AgentView
from ..constraints import (
    Constraint,
    SegmentMinimum,
    node_minimum,
    node_violations,
    segment_minimum,
    violation_intervals,
)
from ..dynamics import make_dynamics
from ..faults import Fault
from ..rewards import Reward, Transition
from ..spacecraft import Actuation, ExecutionModel, PerfectNav, Sensor, Spacecraft

ACTION_MODES = ("normalized", "accel", "thrust", "disc27", "impulse_coast")
# An event time this far (s) past a segment's end still counts as the end: root searches land a
# rounding step either side of it.
_EVENT_TOL = 1e-9
SCALAR_CHANNELS = ("t_remaining", "propellant_fraction")
CHANNELS = (*SCALAR_CHANNELS, "pending_commands")
STREAMS = ("sensor", "actuator", "target")
# Spawn-key branch of the fault stream; spawned children count up from 0 and never reach it.
_FAULT_KEY = 0x6661756C


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
    each as (seconds until it takes effect, command as issued in the action mode's units, limited
    by the thruster layout at the action-space bound); it has ceil(delay_s / dt) slots,
    zero-filled at the front before the first commands.

    `faults` are injected into the episode (`vbar.faults`): each acts on the command as issued
    or on the navigation reading while its activation says so, and `info["faults_active"]` names
    those acting after each step.

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
    execution: ExecutionModel | None = None
    sensor: Sensor = field(default_factory=PerfectNav)
    violation_samples: int = 20
    observation_channels: tuple[str, ...] = ()
    faults: tuple[Fault, ...] = ()
    reward: Reward | None = None
    training: bool = False

    def __post_init__(self):
        # A NaN or negative entry would not fail loudly: a NaN step count removes the limit
        # and a NaN keep-out reports no violation, so every number is checked here.
        positive = ("dt", "max_accel", "isp", "mass_kg", "altitude_m")
        for name in positive:
            check_number(name, getattr(self, name), low=0.0, strict=True)
        for name in ("min_impulse", "delay_s"):
            check_number(name, getattr(self, name), low=0.0)
        check_number("inclination", self.inclination)
        if self.propellant_kg is not None:
            check_number("propellant_kg", self.propellant_kg, low=0.0)
            if self.propellant_kg > self.mass_kg:
                raise ValueError("propellant_kg must not exceed mass_kg")
        if not isinstance(self.faults, tuple):
            raise ValueError("faults must be a tuple")
        for fault in self.faults:
            hooks = ("reset", "update", "on_command", "on_observation")
            if not all(callable(getattr(fault, hook, None)) for hook in hooks):
                raise ValueError(f"{type(fault).__name__} is not a fault: it needs {hooks}")
            if not isinstance(getattr(fault, "name", None), str) or not hasattr(fault, "active"):
                raise ValueError(f"{type(fault).__name__} is not a fault: it needs name and active")
        # One object listed twice would stay one object in each environment's copy and act twice.
        if len({id(fault) for fault in self.faults}) != len(self.faults):
            raise ValueError("faults must be distinct objects")
        check_count("max_steps", self.max_steps, 1)
        check_count("violation_samples", self.violation_samples, 2)


def check_number(name: str, value: Any, low: float = -math.inf, strict: bool = False) -> float:
    """`value` as a finite float not below `low` (above it when `strict`).

    Parameters
    ----------
    name : str
        The setting's name, for the message.
    value : Any
        The value.
    low : float
        Lower limit.
    strict : bool
        Whether `low` itself is refused.

    Returns
    -------
    float
        The value.

    Raises
    ------
    ValueError
        If the value is not a finite real number at or above (or above) `low`.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"{name} must be a number, not {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or number < low or (strict and number == low):
        side = "above" if strict else "at least"
        raise ValueError(f"{name} must be finite and {side} {low:g}, not {value!r}")
    return number


def check_count(name: str, value: Any, low: int) -> int:
    """`value` as a whole number of at least `low`.

    Parameters
    ----------
    name : str
        The setting's name, for the message.
    value : Any
        The value.
    low : int
        Least allowed value.

    Returns
    -------
    int
        The value.

    Raises
    ------
    ValueError
        If the value is not an integer of at least `low`.
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < low:
        raise ValueError(f"{name} must be an integer of at least {low}, not {value!r}")
    return int(value)


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


def _cut(
    actuation: Actuation, flown: float, mass_before: float, craft: Spacecraft, layout: str
) -> Actuation:
    """The part of `actuation` flown in its first `flown` s, with the rocket equation redone."""
    pieces: list[tuple[float, np.ndarray]] = []
    left = flown
    for duration, accel in actuation.segments:
        if left <= 0.0:
            break
        pieces.append((min(duration, left), accel))
        left -= duration
    dv_l2 = sum(float(np.linalg.norm(a)) * d for d, a in pieces)
    dv_l1 = sum(float(np.sum(np.abs(a))) * d for d, a in pieces)
    spent = dv_l1 if layout == "box" else dv_l2
    burned = mass_before * -math.expm1(-spent / craft.exhaust_speed)
    propellant = min(burned, actuation.propellant_kg)
    return Actuation(
        tuple(pieces),
        dv_l2,
        dv_l1,
        propellant,
        actuation.saturated,
        actuation.below_mib,
        actuation.empty,
    )


def continuous_command(action: Any, mode: str) -> np.ndarray:
    """A copy of a continuous-mode action, checked before anything is flown.

    Parameters
    ----------
    action : array_like
        The action: three numbers, or four (with the coast in s) for "impulse_coast".
    mode : str
        The action mode.

    Returns
    -------
    numpy.ndarray
        The action as floats.

    Raises
    ------
    ValueError
        If the action has another shape or a non-finite entry.
    """
    command = np.array(action, dtype=float)
    if mode == "impulse_coast":
        if command.shape != (4,) or not np.all(np.isfinite(command)):
            raise ValueError("impulse_coast actions are four finite numbers (fx, fy, fz, coast s)")
        return command
    if command.shape != (3,):
        raise ValueError(f"{mode} actions have shape (3,), not {command.shape}")
    if not np.all(np.isfinite(command)):
        raise ValueError(f"{mode} actions must be finite")
    return command


def public_spec(constraint: Constraint) -> dict[str, Any]:
    """A constraint's type and parameters, as agents may see them.

    The parameters are copies, so an agent changing them leaves the environment's constraint as
    it was, whatever types the constraint holds. Fields a constraint names in `private_fields`
    (the target mode a moving constraint follows, which may differ from the briefing) are left
    out.
    """
    params: dict[str, Any] = {}
    hidden = set(getattr(constraint, "private_fields", ()))
    if is_dataclass(constraint):
        params = {
            f.name: copy.deepcopy(getattr(constraint, f.name))
            for f in fields(constraint)
            if f.name not in hidden
        }
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
        step_s = getattr(self.dynamics, "step_s", None)
        if step_s is not None and abs(config.dt / step_s - round(config.dt / step_s)) > 1e-9:
            raise ValueError(f"dt must be a whole number of {step_s} s steps on {config.dynamics}")
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
        self.faults = copy.deepcopy(config.faults)
        self.constraints = tuple(self._constraints())
        unknown = set(config.observation_channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown observation channels {sorted(unknown)}, expected {CHANNELS}")
        self.pending_slots = int(np.ceil(config.delay_s / config.dt - 1e-9))
        self.observation_names = self._observation_names()
        # Action units of a full command at the initial mass, the bound the agent's view gives.
        self.action_bound = {
            "accel": config.max_accel,
            "thrust": config.max_accel * config.mass_kg,
        }.get(config.action_mode, 1.0)
        self.action_space: spaces.Space
        if config.action_mode == "disc27":
            self.action_space = spaces.Discrete(len(DISC27))
        elif config.action_mode == "impulse_coast":
            low = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
            top = np.array([1.0, 1.0, 1.0, config.max_steps * config.dt], dtype=np.float32)
            self.action_space = spaces.Box(low, top, dtype=np.float32)
        else:
            high = self.action_bound
            self.action_space = spaces.Box(-high, high, shape=(3,), dtype=np.float32)
        shape = (len(self.observation_names),)
        self.observation_space = spaces.Box(-1e7, 1e7, shape=shape, dtype=np.float64)
        self.state = np.zeros(6)
        self.t = 0.0
        self.issued: list[np.ndarray] = []

    def _sample_start(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def _draw_episode(self, rng: np.random.Generator) -> None:
        """Draw the episode's dispersed parameters, before the start state is chosen."""

    def _limit_command(self, fraction: np.ndarray) -> np.ndarray:
        """The command the scenario lets through now (thrust inhibit windows)."""
        return fraction

    def _split_actuation(self, actuation: Actuation) -> Actuation:
        """Cut the step's segments where the scenario's own forces change (target burns)."""
        return actuation

    def _event_time(self, tau: np.ndarray, samples: np.ndarray, t0: float) -> float | None:
        """Time (s from the segment's start) at which an event ends the step early, or None.

        `tau` and `samples` are the segment's sampled times and states; the state at any
        time in it is `self.dynamics.state_in_last_segment`.
        """
        return None

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

    def task_outcome(self) -> dict[str, Any] | None:
        """The environment's verdict on a goal the metrics cannot rebuild from the trajectory.

        Returns
        -------
        dict or None
            "met" (bool, the goal reached, constraints aside), "time_s" (s, when, or None) and
            "progress" (fraction); None for goals the metrics judge from the states alone.
        """
        return None

    @property
    def scenario_id(self) -> str:
        base = f"vbar:{self.scenario_name}:{self.cfg.dynamics}:{self.cfg.thrust_limit}:dv:v0"
        return base + (":disc27" if self.cfg.action_mode == "disc27" else "")

    def _briefed_id(self) -> str:
        """The identifier agents are given; it may hide what the briefing hides."""
        return self.scenario_id

    def _briefed_constraints(self) -> tuple[Constraint, ...]:
        """The constraints as the briefing describes them, for their public specs."""
        return tuple(self.constraints)

    def agent_view(self) -> AgentView:
        return AgentView(
            scenario_id=self._briefed_id(),
            dt=self.cfg.dt,
            max_steps=self.cfg.max_steps,
            n=self.n,
            action_mode=self.cfg.action_mode,
            max_accel=self.cfg.max_accel,
            thrust_limit=self.cfg.thrust_limit,
            mass_kg=self.cfg.mass_kg,
            goal=self._goal(),
            constraints=tuple(public_spec(c) for c in self._briefed_constraints()),
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
        self._draw_episode(self.np_random)
        if "start_state" in self.options:
            self.state = np.array(self.options["start_state"], dtype=float).reshape(6)
        else:
            self.state = np.asarray(self._sample_start(self.np_random), dtype=float)
        self.dynamics.reset(self.state)
        self.spacecraft.reset(self.rngs["actuator"])
        self.issued = []
        self.sensor.reset(self.rngs["sensor"])
        # The fault stream is derived, not spawned, so the seed sequence's counter moves exactly as
        # it does without faults and adding a fault changes no other model's draws, on seeded and
        # unseeded resets alike. Its key sits under a branch the counter never reaches, and the
        # counter value keeps each reset's faults apart. Each fault draws from its own child, so
        # appending one leaves the others' draws alone.
        seq = self.np_random.bit_generator.seed_seq
        fault_root = np.random.SeedSequence(
            seq.entropy, spawn_key=(*seq.spawn_key, _FAULT_KEY, seq.n_children_spawned)
        )
        fault_seqs = fault_root.spawn(len(self.faults))
        for fault, seq in zip(self.faults, fault_seqs, strict=True):
            fault.reset(np.random.default_rng(seq))
            fault.update(0.0, self.state[None])
        self.t = 0.0
        self.steps = 0
        self.dv_l2 = self.dv_l1 = self.propellant_used = 0.0
        self.min_margins = {c.name: float(c.margin(self.state, 0.0)) for c in self.constraints}
        self.min_margin_bounds = dict(self.min_margins)
        self.inside = {c.name: False for c in self.constraints}
        self.obs = self._observe()
        self.record = None
        return self.obs.copy(), self._info()

    def step(self, action, *, latency_s: float = 0.0):
        """Fly one decision; `latency_s` delays this command's effect (real-time runs).

        An action of the wrong shape or with a non-finite entry raises `ValueError` before any
        state changes.
        """
        coast_steps = 0
        if self.cfg.action_mode == "disc27":
            command = DISC27[self._index(action)].copy()
        else:
            command = continuous_command(action, self.cfg.action_mode)
        if self.cfg.action_mode == "impulse_coast":
            coast_steps = self._coast_steps(command)
            fraction = burn = command[:3]
        else:
            burn = command
            fraction = command / self._action_scale()
        fraction = self._limit_command(fraction)
        for fault in self.faults:
            fraction = fault.on_command(self.t, fraction)
        mass_before = self.spacecraft.mass
        actuation = self.spacecraft.apply(fraction, self.cfg.dt, latency_s)
        # The echo is recorded only once `apply` has accepted the command, so a rejected step
        # leaves the record aligned with the queue.
        self.issued.append(self._echo(burn))
        if coast_steps:
            coast = self.spacecraft.apply(np.zeros(3), coast_steps * self.cfg.dt, latency_s)
            self.issued.append(np.zeros(3))
            actuation = _joined(actuation, coast)
        actuation = self._split_actuation(actuation)
        trace, flown = self._fly(actuation)
        if flown is None:
            self.t += (1 + coast_steps) * self.cfg.dt
        else:
            # An event ended the step early: only the part flown counts, and the propellant the
            # spacecraft already spent on the rest is given back.
            actuation = _cut(actuation, flown, mass_before, self.spacecraft, self.cfg.thrust_limit)
            self.spacecraft.mass = mass_before - actuation.propellant_kg
            self.t += flown
        self.steps += 1 + coast_steps
        for fault in self.faults:
            fault.update(self.t, trace.states)
        self.dv_l2 += actuation.dv_l2
        self.dv_l1 += actuation.dv_l1
        self.propellant_used += actuation.propellant_kg
        for name, found in trace.minima.items():
            self.min_margins[name] = min(self.min_margins[name], found.margin)
            self.min_margin_bounds[name] = min(self.min_margin_bounds[name], found.lower_bound)
        self.obs = self._observe()
        record = StepRecord(
            self.t,
            self.state.copy(),
            self.obs.copy(),
            command,
            command,
            actuation,
            trace,
            agent_latency_s=latency_s,
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

    def _fly(self, actuation: Actuation) -> tuple[SegmentTrace, float | None]:
        """Propagate each constant-acceleration segment and check constraints within it.

        Each segment is its own `advance` call, so no integrator steps across a thrust change.
        Discontinuities inside a dynamics model's own forces (shadow entry and exit) are the
        model's to split. A discrete plant (one with `step_s`) is checked at its nodes only.
        Returns the trace and, when `_event_time` ended the step early, the time flown (s); None
        otherwise.
        """
        t0 = self.t
        discrete = getattr(self.dynamics, "step_s", None) is not None
        segments, times, states = [], [], []
        minima: dict[str, SegmentMinimum] = {}
        violations: dict[str, list[tuple[float, float]]] = {c.name: [] for c in self.constraints}
        starts = []
        flown: float | None = None
        for duration, accel in actuation.segments:
            if duration <= 0.0:
                continue
            starts.append(self.state.copy())
            count = max(2, round(self.cfg.violation_samples * duration / self.cfg.dt))
            self.state, sampled = self.dynamics.advance(accel, duration, n_samples=count)
            samples = cast(np.ndarray, sampled)
            # A discrete plant returns its nodes, however many samples were asked for.
            tau = np.linspace(0.0, duration, len(samples))
            stop = self._event_time(tau, samples, t0)
            if stop is not None:
                if not 0.0 < stop <= duration + _EVENT_TOL:
                    raise ValueError("an event time must lie inside the segment")
                # An event at the segment's end (a root search may land a rounding step past
                # it) ends the step there without truncating the segment.
                if stop < duration:
                    keep = int(np.searchsorted(tau, stop))
                    tau = np.append(tau[:keep], stop)
                    samples = np.vstack([samples[:keep], self.dynamics.state_in_last_segment(stop)])
                    self.state = samples[-1].copy()
                    duration = stop
                flown = t0 - self.t + duration
            thrust = float(np.linalg.norm(accel))
            for c in self.constraints:
                if discrete:
                    found = node_minimum(c, tau, samples, t0)
                    spans = node_violations(c, tau, samples, t0)
                else:
                    state_at = self.dynamics.state_in_last_segment
                    found = segment_minimum(c, state_at, tau, samples, thrust, self.n, t0)
                    spans = violation_intervals(
                        c, state_at, tau, samples, found, t0, thrust, self.n
                    )
                for start, end in spans:
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
            if flown is not None:
                break
        trace = SegmentTrace(
            tuple(segments),
            np.concatenate(times),
            np.concatenate(states),
            minima,
            {name: tuple(spans) for name, spans in violations.items()},
            np.array(starts),
        )
        return trace, flown

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
        reading = self.sensor.observe(self.state, self.t)
        for fault in self.faults:
            reading = fault.on_observation(self.t, reading)
        parts = [reading]
        channels = self.cfg.observation_channels
        if "t_remaining" in channels:
            parts.append(np.array([(self.cfg.max_steps - self.steps) * self.cfg.dt]))
        if "propellant_fraction" in channels:
            craft = self.spacecraft
            parts.append(np.array([craft.propellant / (craft.initial_mass - craft.dry_mass)]))
        if "pending_commands" in channels:
            waits = [0.0] + [wait for wait, _ in self.spacecraft.pending()]
            # Commands enter the spacecraft's queue in issue order, so the last echoes match the
            # queued entries; the zero in effect before the first command has none.
            del self.issued[: -len(waits)]
            echoes = [np.zeros(3)] * (len(waits) - len(self.issued)) + self.issued
            slots = np.zeros((self.pending_slots, 4))
            # Padding goes first, so the entries stay in order of the time they take effect.
            start = max(len(waits) - self.pending_slots, 0)
            recent = list(zip(waits[start:], echoes[start:], strict=True))
            for k, (wait, echo) in enumerate(recent[::-1]):
                slots[-1 - k] = [wait, *echo]
            parts.append(slots.ravel())
        return np.concatenate(parts)

    def _echo(self, command: np.ndarray) -> np.ndarray:
        """The issued command limited by the thruster layout at the action-space bound.

        The true limit sits at the action scale now, which in some modes moves with the mass, and
        the minimum impulse bit is decided on the mass too; limiting at the public bound instead
        keeps the echo a function of the command and the agent's view alone.
        """
        u = command / self.action_bound
        if self.cfg.thrust_limit == "box":
            u = np.clip(u, -1.0, 1.0)
        else:
            length = float(np.linalg.norm(u))
            if length > 1.0:
                u = u / length
        return u * self.action_bound

    def _coast_steps(self, command: np.ndarray) -> int:
        """Whole steps of coast an impulse_coast action asks for, cut at the time limit."""
        left = max(self.cfg.max_steps - self.steps - 1, 0)
        return int(np.clip(round(float(command[3]) / self.cfg.dt), 0, left))

    def _index(self, action) -> int:
        value = np.asarray(action).item()
        if not (float(value).is_integer() and 0 <= value < len(DISC27)):
            raise ValueError(f"disc27 actions are the integers 0 to 26, got {value}")
        return int(value)

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
        } | ({"faults_active": self._faults_active()} if self.faults else {})

    def _faults_active(self) -> tuple[str, ...]:
        return tuple(fault.name for fault in self.faults if fault.active)
