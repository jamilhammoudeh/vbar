"""Logical scenario specs: parameter distributions that a seed turns into one concrete episode.

A `ScenarioSpec` is data: the environment kind that flies it, the shared `EnvConfig` fields, the
kind's own parameters, the dispersions drawn once per episode, the target's attitude mode, tags,
the operational rules as temporal-logic formulas and the observation-completeness statement.
`draw(rng)` turns the dispersions into one episode's parameters, and `episode(seed)` does the
same from a seed, through the generator Gymnasium's `reset(seed=...)` would build, so the
environment and the spec agree on every draw.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from gymnasium.utils import seeding
from scipy.special import erfcinv

from .targets import Passive, TargetMode


class Dispersion(Protocol):
    """A typed distribution over one episode parameter (SI units, angles in rad).

    Every method returns values in the parameter's own units: a float for a scalar parameter,
    an array for a vector one (Hill axes for positions and velocities).
    """

    def sample(self, rng: np.random.Generator) -> Any:
        """One draw of the parameter.

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator; draws are made in a fixed order from it.

        Returns
        -------
        Any
            The drawn value, in the parameter's units.
        """
        ...

    def nominal(self) -> Any:
        """The value with no dispersion, for audits and the stress lab's centre point.

        Returns
        -------
        Any
            The nominal value, in the parameter's units.
        """
        ...

    def bounds(self) -> tuple[Any, Any]:
        """The support, (low, high), per component where the value is a vector.

        Returns
        -------
        tuple
            (low, high) in the parameter's units; infinite where the support is unbounded.
        """
        ...


def _floats(value, name: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in np.atleast_1d(np.asarray(value, dtype=float)))
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite")
    return out


def _unwrap(value: tuple[float, ...], scalar: bool):
    return value[0] if scalar else np.array(value)


@dataclass(frozen=True)
class Fixed:
    """A parameter that does not vary between episodes.

    Parameters
    ----------
    value : Any
        The parameter's value in its own SI units (m, m/s, s, rad); vectors in Hill axes.
    """

    value: Any

    def sample(self, rng: np.random.Generator) -> Any:
        """The fixed value; `rng` is not drawn from.

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator, unused.

        Returns
        -------
        Any
            `value`.
        """
        return self.value

    def nominal(self) -> Any:
        """The fixed value.

        Returns
        -------
        Any
            `value`.
        """
        return self.value

    def bounds(self) -> tuple[Any, Any]:
        """The support, a single point.

        Returns
        -------
        tuple
            (`value`, `value`).
        """
        return self.value, self.value


@dataclass(frozen=True)
class Uniform:
    """Uniform on [low, high], per component when `low` and `high` are sequences.

    Parameters
    ----------
    low : float or sequence of float
        Lower end of the support, in the parameter's SI units (m, m/s, s, rad).
    high : float or sequence of float
        Upper end of the support, same units and length as `low`.

    Raises
    ------
    ValueError
        If an end is not finite, the lengths differ or `high` is below `low`.
    """

    low: Any
    high: Any
    scalar: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        low, high = _floats(self.low, "low"), _floats(self.high, "high")
        if len(low) != len(high):
            raise ValueError("low and high must have the same length")
        if any(b < a for a, b in zip(low, high, strict=True)):
            raise ValueError("high must not be below low")
        object.__setattr__(self, "scalar", np.ndim(self.low) == 0)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def sample(self, rng: np.random.Generator) -> Any:
        """One uniform draw.

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator.

        Returns
        -------
        float or numpy.ndarray
            The draw, a float when `low` is a scalar.
        """
        draw = rng.uniform(self.low, self.high)
        return float(draw[0]) if self.scalar else draw

    def nominal(self) -> Any:
        """The midpoint of the support.

        Returns
        -------
        float or numpy.ndarray
            (low + high) / 2, per component.
        """
        mid = tuple(0.5 * (a + b) for a, b in zip(self.low, self.high, strict=True))
        return _unwrap(mid, self.scalar)

    def bounds(self) -> tuple[Any, Any]:
        """The support.

        Returns
        -------
        tuple
            (`low`, `high`).
        """
        return _unwrap(self.low, self.scalar), _unwrap(self.high, self.scalar)


@dataclass(frozen=True)
class Normal:
    """Gaussian with `mean` and `sigma`, drawn again while outside `bound` sigma when set.

    Per component when `mean` and `sigma` are sequences. The truncation is by rejection, so the
    density inside the bound keeps its shape. For a disturbance model the bound is derived from
    an episode-level exceedance probability with `per_draw_level`.

    Parameters
    ----------
    mean : float or sequence of float
        Mean, in the parameter's SI units (m, m/s, s, rad).
    sigma : float or sequence of float
        Standard deviation, same units and length as `mean`.
    bound : float, optional
        Truncation level in units of sigma; None for no truncation.

    Raises
    ------
    ValueError
        If a value is not finite, the lengths differ, a sigma is negative or `bound` is not
        positive.
    """

    mean: Any
    sigma: Any
    bound: float | None = None
    scalar: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        mean, sigma = _floats(self.mean, "mean"), _floats(self.sigma, "sigma")
        if len(mean) != len(sigma):
            raise ValueError("mean and sigma must have the same length")
        if any(s < 0.0 for s in sigma):
            raise ValueError("sigma must not be negative")
        if self.bound is not None and not self.bound > 0.0:
            raise ValueError("bound must be positive, in units of sigma")
        object.__setattr__(self, "scalar", np.ndim(self.mean) == 0)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "sigma", sigma)

    def sample(self, rng: np.random.Generator) -> Any:
        """One draw, redrawn per component while outside `bound` sigma.

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator.

        Returns
        -------
        float or numpy.ndarray
            The draw, a float when `mean` is a scalar.
        """
        draw = rng.standard_normal(len(self.mean))
        if self.bound is not None:
            outside = np.abs(draw) > self.bound
            while np.any(outside):
                draw[outside] = rng.standard_normal(int(np.sum(outside)))
                outside = np.abs(draw) > self.bound
        value = np.array(self.mean) + np.array(self.sigma) * draw
        return float(value[0]) if self.scalar else value

    def nominal(self) -> Any:
        """The mean.

        Returns
        -------
        float or numpy.ndarray
            `mean`.
        """
        return _unwrap(self.mean, self.scalar)

    def bounds(self) -> tuple[Any, Any]:
        """The support, mean -/+ bound sigma.

        Returns
        -------
        tuple
            (low, high) per component; infinite when `bound` is None.
        """
        reach = math.inf if self.bound is None else self.bound
        low = tuple(m - reach * s for m, s in zip(self.mean, self.sigma, strict=True))
        high = tuple(m + reach * s for m, s in zip(self.mean, self.sigma, strict=True))
        return _unwrap(low, self.scalar), _unwrap(high, self.scalar)


@dataclass(frozen=True)
class Choice:
    """One of `options`, with `weights` (default equal).

    Parameters
    ----------
    options : tuple
        The values to choose from, in the parameter's units.
    weights : tuple of float, optional
        Relative weights, one per option, normalised to sum to 1; None for equal weights.

    Raises
    ------
    ValueError
        If `options` is empty or `weights` is not one non-negative number per option with a
        positive sum.
    """

    options: tuple[Any, ...]
    weights: tuple[float, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "options", tuple(self.options))
        if not self.options:
            raise ValueError("options must not be empty")
        if self.weights is not None:
            weights = _floats(self.weights, "weights")
            if len(weights) != len(self.options) or min(weights) < 0.0 or sum(weights) <= 0.0:
                raise ValueError("weights must be one non-negative number per option")
            object.__setattr__(self, "weights", tuple(w / sum(weights) for w in weights))

    def sample(self, rng: np.random.Generator) -> Any:
        """One option, drawn with `weights`.

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator.

        Returns
        -------
        Any
            The chosen option.
        """
        return self.options[int(rng.choice(len(self.options), p=self.weights))]

    def nominal(self) -> Any:
        """The most likely option, the first one when weights are equal.

        Returns
        -------
        Any
            The nominal option.
        """
        if self.weights is None:
            return self.options[0]
        return self.options[int(np.argmax(self.weights))]

    def bounds(self) -> tuple[Any, Any]:
        """The first and last options.

        Returns
        -------
        tuple
            (options[0], options[-1]).
        """
        return self.options[0], self.options[-1]


@dataclass(frozen=True)
class Shell:
    """A vector of length uniform on [low, high] in a direction uniform on the sphere.

    Parameters
    ----------
    low : float
        Least length, in the vector's units (m for a position, m/s for a velocity).
    high : float
        Greatest length, same units, finite.

    Raises
    ------
    ValueError
        If the lengths are not 0 <= low <= high < inf.
    """

    low: float
    high: float

    def __post_init__(self):
        if not 0.0 <= self.low <= self.high < math.inf:
            raise ValueError("need 0 <= low <= high")

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """One vector in Hill axes.

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator.

        Returns
        -------
        numpy.ndarray
            Shape (3,), Hill axes.
        """
        direction = rng.standard_normal(3)
        while not np.linalg.norm(direction) > 0.0:
            direction = rng.standard_normal(3)
        return rng.uniform(self.low, self.high) * direction / np.linalg.norm(direction)

    def nominal(self) -> np.ndarray:
        """The mid length along -V-bar.

        Returns
        -------
        numpy.ndarray
            (0, -(low + high) / 2, 0), Hill axes.
        """
        return np.array([0.0, -0.5 * (self.low + self.high), 0.0])

    def bounds(self) -> tuple[Any, Any]:
        """The bounding box of the shell.

        Returns
        -------
        tuple of numpy.ndarray
            (-high, high) in each Hill component.
        """
        return np.full(3, -self.high), np.full(3, self.high)


@dataclass(frozen=True)
class DriftingVelocity:
    """A start velocity uniform in a box, drawn again until free drift leaves along track.

    In the Clohessy-Wiltshire model a start (x0, vy0) drifts along track by
    6 pi (2 x0 + vy0 / n) per orbit (Fehse, Automated Rendezvous and Docking of Spacecraft,
    2003, eq. 3.22); a draw is kept only when |2 x0 + vy0 / n| is at least `min_offset`, so a
    start never sits on a slowly drifting relative orbit about the target, from which free drift
    alone could complete a task. The position is the draw named `position`, made before this one.

    Parameters
    ----------
    low, high : sequence of float
        Corners of the velocity box (m/s), Hill axes.
    min_offset : float
        Least |2 x0 + vy0 / n| (m), positive.
    n : float
        Mean motion (rad/s), positive.
    position : str
        Name of the position draw.

    Raises
    ------
    ValueError
        If the box cannot reach `min_offset` from x0 = 0, or `n` or `min_offset` is not positive;
        at draw time, if it cannot reach `min_offset` from the drawn position.
    """

    low: tuple[float, float, float]
    high: tuple[float, float, float]
    min_offset: float
    n: float
    position: str = "start_pos"

    def __post_init__(self):
        object.__setattr__(self, "low", _floats(self.low, "low"))
        object.__setattr__(self, "high", _floats(self.high, "high"))
        if len(self.low) != 3 or len(self.high) != 3:
            raise ValueError("low and high are 3-vectors")
        if not (self.n > 0.0 and self.min_offset > 0.0):
            raise ValueError("n and min_offset must be positive")
        if max(abs(self.low[1]), abs(self.high[1])) / self.n <= self.min_offset:
            raise ValueError("the along-track speeds cannot reach min_offset")

    def offset(self, position: Any, velocity: Any) -> float:
        """2 x0 + vy0 / n (m), the along-track drift per orbit over 6 pi.

        Parameters
        ----------
        position, velocity : array_like
            Start position (m) and velocity (m/s), Hill axes.

        Returns
        -------
        float
            The offset.
        """
        return 2.0 * float(np.asarray(position)[0]) + float(np.asarray(velocity)[1]) / self.n

    def sample_given(self, rng: np.random.Generator, drawn: Mapping[str, Any]) -> np.ndarray:
        """One velocity for the position already drawn.

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator.
        drawn : mapping
            The draws made before this one; holds `position`.

        Returns
        -------
        numpy.ndarray
            Velocity (m/s), shape (3,), Hill axes.
        """
        position = drawn[self.position]
        # The offset is monotone in vy0, so its extremes over the box sit at the corners; with
        # neither past min_offset no draw is ever kept.
        corners = (self.low[1], self.high[1])
        reach = max(abs(self.offset(position, (0.0, vy, 0.0))) for vy in corners)
        if reach <= self.min_offset:
            raise ValueError(
                f"no velocity in the box reaches min_offset from x0 = {float(position[0]):g} m"
            )
        while True:
            velocity = rng.uniform(self.low, self.high)
            if abs(self.offset(position, velocity)) >= self.min_offset:
                return velocity

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """A velocity for a start at the origin's radial level (x0 = 0).

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator.

        Returns
        -------
        numpy.ndarray
            Velocity (m/s), shape (3,).
        """
        return self.sample_given(rng, {self.position: np.zeros(3)})

    def nominal(self) -> np.ndarray:
        """The box centre with the along-track speed giving `min_offset` at x0 = 0.

        Returns
        -------
        numpy.ndarray
            Velocity (m/s), shape (3,).
        """
        mid = np.array([0.5 * (a + b) for a, b in zip(self.low, self.high, strict=True)])
        speed = self.min_offset * self.n
        mid[1] = speed if self.high[1] >= speed else self.low[1]
        return mid

    def bounds(self) -> tuple[Any, Any]:
        """The velocity box.

        Returns
        -------
        tuple of numpy.ndarray
            (`low`, `high`).
        """
        return np.array(self.low), np.array(self.high)


@dataclass(frozen=True)
class AngleShell:
    """A vector of uniform length, uniform azimuth and uniform elevation.

    The length is uniform on [low, high], the azimuth on [0, 2 pi) and the elevation from the
    orbit plane on [-pi/2, pi/2]. Equal steps in elevation are not equal areas, so directions
    crowd toward the poles; this is the draw some published benchmarks state (radius, azimuth
    and elevation ranges), kept for reproducing them. `Shell` draws directions uniform on the
    sphere.

    Parameters
    ----------
    low, high : float
        Least and greatest length, in the vector's units, 0 <= low <= high < inf.

    Raises
    ------
    ValueError
        If the lengths are not 0 <= low <= high < inf.
    """

    low: float
    high: float

    def __post_init__(self):
        if not 0.0 <= self.low <= self.high < math.inf:
            raise ValueError("need 0 <= low <= high")

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """One vector in Hill axes, elevation measured from the orbit plane.

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator.

        Returns
        -------
        numpy.ndarray
            Shape (3,).
        """
        length = rng.uniform(self.low, self.high)
        azimuth = rng.uniform(0.0, 2.0 * math.pi)
        elevation = rng.uniform(-0.5 * math.pi, 0.5 * math.pi)
        c = math.cos(elevation)
        return length * np.array(
            [c * math.cos(azimuth), c * math.sin(azimuth), math.sin(elevation)]
        )

    def nominal(self) -> np.ndarray:
        """The mid length along -V-bar.

        Returns
        -------
        numpy.ndarray
            (0, -(low + high) / 2, 0).
        """
        return np.array([0.0, -0.5 * (self.low + self.high), 0.0])

    def bounds(self) -> tuple[Any, Any]:
        """The bounding box.

        Returns
        -------
        tuple of numpy.ndarray
            (-high, high) in each component.
        """
        return np.full(3, -self.high), np.full(3, self.high)


@dataclass(frozen=True)
class LimitedSpeed:
    """A start velocity whose speed is a uniform fraction of a distance-dependent limit.

    The direction is uniform on the sphere and the speed a fraction, uniform on [low, high], of
    v0 + v1 (rho - offset) at the position drawn before it.

    Parameters
    ----------
    low, high : float
        Least and greatest fraction of the limit, 0 <= low <= high.
    v0 : float
        Limit at `offset` (m/s), positive.
    v1 : float
        Slope of the limit (1/s), not negative.
    offset : float
        Range at which the limit is `v0` (m).
    position : str
        Name of the position draw.

    Raises
    ------
    ValueError
        If the fractions are not 0 <= low <= high, `v0` is not positive, `v1` is negative, or a
        value is not finite.
    """

    low: float
    high: float
    v0: float
    v1: float
    offset: float = 0.0
    position: str = "start_pos"

    def __post_init__(self):
        if not 0.0 <= self.low <= self.high < math.inf:
            raise ValueError("need 0 <= low <= high")
        if not (math.inf > self.v0 > 0.0 and math.inf > self.v1 >= 0.0):
            raise ValueError("v0 must be positive and v1 not negative, both finite")
        if not math.isfinite(self.offset):
            raise ValueError("offset must be finite")

    def limit(self, position: Any) -> float:
        """The speed limit (m/s) at a position (m).

        Parameters
        ----------
        position : array_like
            Position (m), Hill axes.

        Returns
        -------
        float
            v0 + v1 (rho - offset), at least 0.
        """
        rho = float(np.linalg.norm(np.asarray(position, dtype=float)))
        return max(self.v0 + self.v1 * (rho - self.offset), 0.0)

    def sample_given(self, rng: np.random.Generator, drawn: Mapping[str, Any]) -> np.ndarray:
        """One velocity for the position already drawn.

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator.
        drawn : mapping
            The draws made before this one; holds `position`.

        Returns
        -------
        numpy.ndarray
            Velocity (m/s), shape (3,), Hill axes.
        """
        direction = rng.standard_normal(3)
        while not np.linalg.norm(direction) > 0.0:
            direction = rng.standard_normal(3)
        speed = rng.uniform(self.low, self.high) * self.limit(drawn[self.position])
        return speed * direction / np.linalg.norm(direction)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """A velocity for a start at range `offset`.

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator.

        Returns
        -------
        numpy.ndarray
            Velocity (m/s), shape (3,).
        """
        return self.sample_given(rng, {self.position: np.array([self.offset, 0.0, 0.0])})

    def nominal(self) -> np.ndarray:
        """Rest.

        Returns
        -------
        numpy.ndarray
            Zeros, shape (3,).
        """
        return np.zeros(3)

    def bounds(self) -> tuple[Any, Any]:
        """Unbounded: the limit grows with the range.

        Returns
        -------
        tuple of numpy.ndarray
            (-inf, inf) in each component.
        """
        return np.full(3, -math.inf), np.full(3, math.inf)


def per_draw_level(episode_probability: float, draws: int) -> float:
    """Truncation level (in sigma) per Gaussian draw for an episode-level exceedance probability.

    An episode with `draws` independent standard normal draws leaves the bounded model when any
    draw lies outside the level; the per-draw tail 1 - (1 - p)^(1 / draws) follows from the
    union over draws (Kochenderfer, Katz, Corso and Moss, Algorithms for Validation, sec. 8.1).
    At 3 sigma per draw, 94.6% of 1080-draw episodes have at least one draw outside; an
    episode-level 1e-3 needs 4.91 sigma for 1080 draws and 5.04 for 2160.

    Parameters
    ----------
    episode_probability : float
        Probability that an episode has any draw outside the level, in (0, 1).
    draws : int
        Number of independent standard normal draws per episode, at least 1.

    Returns
    -------
    float
        The per-draw truncation level, in units of sigma.

    Raises
    ------
    ValueError
        If `episode_probability` is not in (0, 1) or `draws` is below 1.
    """
    if not 0.0 < episode_probability < 1.0:
        raise ValueError("episode_probability must be between 0 and 1")
    if draws < 1:
        raise ValueError("draws must be at least 1")
    tail = -math.expm1(math.log1p(-episode_probability) / draws)
    return float(math.sqrt(2.0) * erfcinv(tail))


@dataclass(frozen=True)
class Observation:
    """What the agent sees and what it would need for the task to be Markov.

    `horizon` is "indefinite" (the time cap is a truncation; the default) or "finite" (time to
    go is observed through `t_remaining` and the cap is a terminal state). `hidden` names the
    quantities the observation does not carry; `note` says what an agent needs to make the task
    Markov in its observation.

    Parameters
    ----------
    horizon : {"indefinite", "finite"}
        Whether the time cap is a truncation or a terminal state.
    hidden : tuple of str
        Names of the quantities the observation does not carry.
    note : str
        What an agent needs to make the task Markov in its observation.

    Raises
    ------
    ValueError
        If `horizon` is neither "indefinite" nor "finite".
    """

    horizon: str = "indefinite"
    hidden: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self):
        if self.horizon not in ("indefinite", "finite"):
            raise ValueError("horizon must be 'indefinite' or 'finite'")


def _frozen(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True, kw_only=True)
class ScenarioSpec:
    """A logical scenario: everything but the seed.

    `entry_point` is the environment class, or its "module:Class" path; it takes
    `(spec, config)`. `config` holds `EnvConfig` fields, `parameters` the kind's own, and
    `dispersions` the per-episode draws in the order they are made. `variants` map a suffix
    such as "nav1" or "iss" to overrides of those three mappings. `cost` is the ID's cost
    field. `oracle` holds reference figures with their sources and `audit` the audit
    record. `rules` are operational rules as temporal-logic formulas.

    Parameters
    ----------
    name : str
        Scenario name, without colons, slashes or spaces.
    version : int
        Version of the scenario's behaviour, not negative.
    entry_point : str or type
        The `ScenarioEnv` subclass, or its "module:Class" path.
    summary : str
        One-sentence description of the task.
    config : mapping
        `EnvConfig` field values (SI units).
    parameters : mapping
        The kind's own parameters (m, m/s, s, rad; Hill axes, body axes where the kind says).
    dispersions : mapping of str to Dispersion
        Per-episode draws, in the order they are made.
    target : TargetMode
        The target's attitude mode; `Passive` by default.
    tags : tuple of str
        Labels for grouping results.
    rules : tuple of str
        Operational rules as temporal-logic formulas.
    observation : Observation
        The observation-completeness statement.
    oracle : mapping
        Reference figures with their sources.
    audit : str
        The scenario audit record.
    cost : str
        The identifier's cost field.
    variants : mapping of str to mapping
        Suffix to overrides of "config", "parameters", "dispersions", "target" and "tags".
    applied : tuple of str
        Names of the variants already applied, in order.

    Raises
    ------
    ValueError
        If the name holds a colon, slash or space, the version is negative, or a variant has a
        bad name or overrides an unknown field.
    """

    name: str
    version: int
    entry_point: str | type
    summary: str
    config: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    dispersions: Mapping[str, Dispersion] = field(default_factory=dict)
    target: TargetMode = field(default_factory=Passive)
    tags: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()
    observation: Observation = field(default_factory=Observation)
    oracle: Mapping[str, Any] = field(default_factory=dict)
    audit: str = ""
    cost: str = "dv"
    variants: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    applied: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name or any(c in self.name for c in ":/ "):
            raise ValueError("a scenario name has no colons, slashes or spaces")
        if self.version < 0:
            raise ValueError("version must not be negative")
        for label in ("config", "parameters", "dispersions", "oracle"):
            object.__setattr__(self, label, _frozen(getattr(self, label)))
        variants = {k: _frozen(v) for k, v in dict(self.variants).items()}
        for key, overrides in variants.items():
            unknown = set(overrides) - {"config", "parameters", "dispersions", "target", "tags"}
            if unknown or any(c in key for c in ":/ "):
                raise ValueError(f"variant {key!r} overrides unknown fields {sorted(unknown)}")
        object.__setattr__(self, "variants", MappingProxyType(variants))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "applied", tuple(self.applied))

    def variant(self, *names: str) -> ScenarioSpec:
        """This spec with the named variants' overrides applied, in order.

        Parameters
        ----------
        *names : str
            Variant names, applied left to right.

        Returns
        -------
        ScenarioSpec
            A new spec; mappings are merged, the target replaced and tags appended.

        Raises
        ------
        KeyError
            If a name is not a variant of the spec.
        """
        spec = self
        for name in names:
            if name not in spec.variants:
                raise KeyError(f"scenario {spec.name!r} has no variant {name!r}")
            over = spec.variants[name]
            spec = replace(
                spec,
                config={**spec.config, **over.get("config", {})},
                parameters={**spec.parameters, **over.get("parameters", {})},
                dispersions={**spec.dispersions, **over.get("dispersions", {})},
                target=over.get("target", spec.target),
                tags=tuple(dict.fromkeys((*spec.tags, *over.get("tags", ())))),
                applied=(*spec.applied, name),
            )
        return spec

    def draw(self, rng: np.random.Generator) -> dict[str, Any]:
        """One episode's parameters, drawn in the spec's order.

        Parameters
        ----------
        rng : numpy.random.Generator
            The episode's generator.

        Returns
        -------
        dict
            Dispersion name to drawn value, in each parameter's SI units.
        """
        drawn: dict[str, Any] = {}
        for name, dispersion in self.dispersions.items():
            # A draw conditioned on earlier ones reads them; the order of the mapping is the
            # order of the draws.
            given = getattr(dispersion, "sample_given", None)
            drawn[name] = dispersion.sample(rng) if given is None else given(rng, drawn)
        return drawn

    def nominal(self) -> dict[str, Any]:
        """Every dispersed parameter at its nominal value.

        Returns
        -------
        dict
            Dispersion name to nominal value, in each parameter's SI units.
        """
        return {name: d.nominal() for name, d in self.dispersions.items()}

    def episode(self, seed: int) -> dict[str, Any]:
        """The parameters `reset(seed=seed)` draws for this spec.

        Parameters
        ----------
        seed : int
            The episode seed.

        Returns
        -------
        dict
            Dispersion name to drawn value, in each parameter's SI units.
        """
        rng, _ = seeding.np_random(seed)
        return self.draw(rng)


__all__ = [
    "Choice",
    "Dispersion",
    "AngleShell",
    "DriftingVelocity",
    "LimitedSpeed",
    "Fixed",
    "Normal",
    "Observation",
    "ScenarioSpec",
    "Shell",
    "Uniform",
    "per_draw_level",
]
