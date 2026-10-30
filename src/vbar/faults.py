"""Faults injected into an episode: when each one acts, and what it does to commands or readings.

A fault pairs an effect (a thruster that stops firing, a sensor bias, lost range) with an
activation: a scheduled time window, a condition on the true state that starts or ends it, or
an intermittent draw. The environment decides activity at each decision boundary from the step
just flown: `update` sees the step's sampled true states, and the new activity applies to the
observation made at the end of that step and to the next command. Command faults act on the
command as issued, before the actuation delay. Activations read the true state; agents see a
fault only through what it does to their observations and their spacecraft.

Each environment deep-copies the faults of its configuration, since activations hold episode
state (a latched trigger, the fault's random stream).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "Activation",
    "After",
    "Always",
    "Condition",
    "Fault",
    "InSphere",
    "Intermittent",
    "RangeBelow",
    "RangeLoss",
    "SensorBias",
    "ThrusterOff",
    "Until",
    "Window",
]


def _finite(name: str, value: float, low: float = -math.inf, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"{name} must be a number, not {type(value).__name__}")
    number = float(value)
    if math.isnan(number) or number < low or (strict and number == low):
        side = "above" if strict else "at least"
        raise ValueError(f"{name} must be {side} {low:g}, not {value!r}")
    return number


def _vector3(name: str, value: ArrayLike) -> np.ndarray:
    vector = np.array(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be three finite numbers")
    return vector


class Condition(Protocol):
    """A test on the true trajectory of one step."""

    def met(self, states: np.ndarray) -> bool:
        """Whether the condition holds anywhere along `states` (k, 6), Hill frame, m and m/s."""
        ...


@dataclass(frozen=True, kw_only=True)
class RangeBelow:
    """The chaser's distance from the target falls below `range_m` (m) at a sampled state."""

    range_m: float

    def __post_init__(self) -> None:
        _finite("range_m", self.range_m, low=0.0, strict=True)

    def met(self, states: np.ndarray) -> bool:
        return bool(np.any(np.linalg.norm(states[:, :3], axis=1) < self.range_m))


@dataclass(frozen=True, kw_only=True)
class InSphere:
    """The path enters the sphere of `radius_m` (m) about `center_m` (Hill frame, m).

    The path between samples is taken as straight, so a pass through the sphere between two
    samples counts. The arc departs from its chord by at most a h^2 / 8 for an acceleration a
    over a sample interval h; the Coriolis term 2 n v at 1 m/s and n = 1.1e-3 rad/s over the
    default 0.53 s interval gives 8e-5 m.
    """

    center_m: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_m", tuple(_vector3("center_m", self.center_m)))
        _finite("radius_m", self.radius_m, low=0.0, strict=True)

    def met(self, states: np.ndarray) -> bool:
        p = states[:, :3] - np.asarray(self.center_m)
        if len(p) == 1:
            return bool(np.linalg.norm(p[0]) < self.radius_m)
        a, b = p[:-1], p[1:]
        d = b - a
        length2 = np.sum(d * d, axis=1)
        s = np.clip(-np.sum(a * d, axis=1) / np.where(length2 > 0.0, length2, 1.0), 0.0, 1.0)
        closest = a + s[:, None] * d
        return bool(np.any(np.linalg.norm(closest, axis=1) < self.radius_m))


class Activation(Protocol):
    """Decides, once per decision boundary, whether a fault acts until the next one."""

    def reset(self, rng: np.random.Generator) -> None: ...

    def update(self, t: float, states: np.ndarray) -> bool:
        """Activity after a step ending at `t` (s) whose sampled true states are `states`."""
        ...


@dataclass(kw_only=True)
class Always:
    """Active for the whole episode."""

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def update(self, t: float, states: np.ndarray) -> bool:
        return True


@dataclass(kw_only=True)
class Window:
    """Active at decision times in [`start_s`, `end_s`) (s since the episode began)."""

    start_s: float = 0.0
    end_s: float = math.inf

    def __post_init__(self) -> None:
        _finite("start_s", self.start_s, low=0.0)
        _finite("end_s", self.end_s)
        if not self.end_s > self.start_s:
            raise ValueError("end_s must be after start_s")

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def update(self, t: float, states: np.ndarray) -> bool:
        return self.start_s <= t < self.end_s


@dataclass(kw_only=True)
class After:
    """Active from the first step in which `condition` holds, for the rest of the episode."""

    condition: Condition
    _met: bool = field(default=False, init=False, repr=False)

    def reset(self, rng: np.random.Generator) -> None:
        self._met = False

    def update(self, t: float, states: np.ndarray) -> bool:
        self._met = self._met or self.condition.met(states)
        return self._met


@dataclass(kw_only=True)
class Until:
    """Active until the first step in which `condition` holds, inactive after it."""

    condition: Condition
    _met: bool = field(default=False, init=False, repr=False)

    def reset(self, rng: np.random.Generator) -> None:
        self._met = False

    def update(self, t: float, states: np.ndarray) -> bool:
        self._met = self._met or self.condition.met(states)
        return not self._met


@dataclass(kw_only=True)
class Intermittent:
    """Active at each decision with `probability` while `within` is active.

    Each decision draws once from the fault's own stream, so the pattern is reproducible from
    the episode seed and independent of every other fault and model.
    """

    probability: float
    within: Activation = field(default_factory=Always)
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _finite("probability", self.probability, low=0.0)
        if self.probability > 1.0:
            raise ValueError("probability must be at most 1")

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.within.reset(rng)

    def update(self, t: float, states: np.ndarray) -> bool:
        if self._rng is None:
            raise RuntimeError("reset the activation before updating it")
        # One draw per decision whatever `within` says, so a change of window does not shift
        # the pattern of later draws.
        draw = self._rng.random()
        return self.within.update(t, states) and draw < self.probability


class Fault(Protocol):
    """A fault an environment applies to commands and readings while it is active.

    `name` labels it in `info["faults_active"]`.
    """

    name: str
    active: bool

    def reset(self, rng: np.random.Generator) -> None: ...

    def update(self, t: float, states: np.ndarray) -> None: ...

    def on_command(self, t: float, command: np.ndarray) -> np.ndarray: ...

    def on_observation(self, t: float, obs: np.ndarray) -> np.ndarray: ...


@dataclass(kw_only=True, eq=False)
class _Base:
    when: Activation = field(default_factory=Always)
    active: bool = field(default=False, init=False)

    def reset(self, rng: np.random.Generator) -> None:
        self.when.reset(rng)
        self.active = False

    def update(self, t: float, states: np.ndarray) -> None:
        self.active = self.when.update(t, states)

    def on_command(self, t: float, command: np.ndarray) -> np.ndarray:
        return command

    def on_observation(self, t: float, obs: np.ndarray) -> np.ndarray:
        return obs


@dataclass(kw_only=True, eq=False)
class ThrusterOff(_Base):
    """The thrusters that push along `direction` (+1 or -1) of Hill `axis` (0, 1, 2) stop firing.

    While active, that sign of the axis's command fraction is delivered as zero and the other
    sign is untouched; no propellant is spent on the lost part.

    Parameters
    ----------
    axis : int
        Hill axis: 0 radial (x), 1 along-track (y), 2 orbit normal (z).
    direction : int
        +1 for the thrusters that push along the axis, -1 for those that push against it.
    when : Activation
        When the fault acts; `Always` by default.
    """

    axis: int
    direction: int
    name: str = "thruster_off"

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")

    def on_command(self, t: float, command: np.ndarray) -> np.ndarray:
        if not self.active or command[self.axis] * self.direction <= 0.0:
            return command
        delivered = command.copy()
        delivered[self.axis] = 0.0
        return delivered


@dataclass(kw_only=True, eq=False)
class SensorBias(_Base):
    """A constant bias on the navigation reading, of a magnitude drawn once per episode.

    The position bias's magnitude is uniform on `position_m` (low, high) in m and the velocity
    bias's on `velocity_mps` in m/s, each along a direction uniform on the sphere, all drawn
    from the fault's stream at reset. A fixed bias is a range with equal ends.

    Parameters
    ----------
    position_m : tuple of float
        Least and greatest position-bias magnitude, m.
    velocity_mps : tuple of float
        Least and greatest velocity-bias magnitude, m/s.
    when : Activation
        When the fault acts; `Always` by default.
    """

    position_m: tuple[float, float] = (0.0, 0.0)
    velocity_mps: tuple[float, float] = (0.0, 0.0)
    name: str = "sensor_bias"
    _bias: np.ndarray = field(default_factory=lambda: np.zeros(6), init=False, repr=False)

    def __post_init__(self) -> None:
        for label in ("position_m", "velocity_mps"):
            low, high = getattr(self, label)
            _finite(f"{label} low", low, low=0.0)
            _finite(f"{label} high", high, low=0.0)
            if high < low or math.isinf(high):
                raise ValueError(f"{label} must be a finite (low, high) pair with low <= high")

    def reset(self, rng: np.random.Generator) -> None:
        super().reset(rng)
        parts = []
        for low, high in (self.position_m, self.velocity_mps):
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            parts.append(rng.uniform(low, high) * direction)
        self._bias = np.concatenate(parts)

    def on_observation(self, t: float, obs: np.ndarray) -> np.ndarray:
        return obs + self._bias if self.active else obs


@dataclass(kw_only=True, eq=False)
class RangeLoss(_Base):
    """Range and range rate are lost; only the line of sight and its rate remain.

    The reading becomes the measured line of sight scaled to `assumed_range_m` and its rate
    scaled the same way: p' = R p / |p| and v' = R (v - (p . v) p / |p|^2) / |p|, so the
    observation keeps its layout while carrying no range information. A reading at zero range
    passes unchanged.

    Parameters
    ----------
    assumed_range_m : float
        The range the reading is scaled to, m.
    when : Activation
        When the fault acts; `Always` by default.
    """

    assumed_range_m: float
    name: str = "range_loss"

    def __post_init__(self) -> None:
        _finite("assumed_range_m", self.assumed_range_m, low=0.0, strict=True)
        if math.isinf(self.assumed_range_m):
            raise ValueError("assumed_range_m must be finite")

    def on_observation(self, t: float, obs: np.ndarray) -> np.ndarray:
        if not self.active:
            return obs
        p, v = obs[:3], obs[3:6]
        rho = float(np.linalg.norm(p))
        if rho == 0.0:
            return obs
        scale = self.assumed_range_m / rho
        reading = obs.copy()
        reading[:3] = scale * p
        reading[3:6] = scale * (v - (p @ v) / rho**2 * p)
        return reading
