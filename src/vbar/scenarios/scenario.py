"""The environment a scenario spec builds, and the identifier that names what it flies.

The identifier is `vbar:{scenario}:{dynamics}:{actuator}:{cost}:v{n}`, followed by the names
of the variants applied and then one `field=value` token for every `EnvConfig` field that
differs from the spec's declared configuration, so two environments with the same identifier
and seed fly the same episode; the reward and the training flag change no trajectory and are
left out. Named sensors keep their track names (`nav1`, `nav2`).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, ClassVar, cast

import numpy as np

from ..dynamics import TwoBodyDynamics, get_body
from ..spacecraft import Actuation, GaussianNav, GaussMarkovNav, PerfectNav, RangeScaledNav
from .base import BaseEnv, EnvConfig
from .spec import ScenarioSpec
from .targets import ManeuveringDynamics, burn_boundaries, split_at

_SKIPPED_FIELDS = ("reward", "training")
# Tolerance on the inhibit window edges: the environment's clock is a sum of steps.
_EDGE_TOL = 1e-6


@dataclass(frozen=True, kw_only=True)
class ScenarioParams:
    """Parameters every scenario kind shares (SI units).

    `thrust_inhibit` lists (start s, end s) windows in which the scenario zeroes every command,
    standing in for an attitude manoeuvre or a procedural inhibit; their edges must fall on
    step boundaries. `soft` names the constraints scored but not counted as failures.

    Parameters
    ----------
    thrust_inhibit : tuple of tuple of float
        (start s, end s) windows with 0 <= start < end.
    soft : tuple of str
        Names of the soft constraints.

    Raises
    ------
    ValueError
        If a window is not 0 <= start < end.
    """

    thrust_inhibit: tuple[tuple[float, float], ...] = ()
    soft: tuple[str, ...] = ()

    def __post_init__(self):
        windows = tuple((float(a), float(b)) for a, b in self.thrust_inhibit)
        if any(not 0.0 <= a < b for a, b in windows):
            raise ValueError("thrust_inhibit windows are (start, end) with 0 <= start < end")
        object.__setattr__(self, "thrust_inhibit", windows)
        object.__setattr__(self, "soft", tuple(self.soft))


def _number(value: float) -> str:
    text = format(float(value), ".10g")
    return text[:-2] if text.endswith(".0") else text


def label(value: Any) -> str:
    """A short deterministic text for a configuration value, for the identifier.

    Parameters
    ----------
    value : Any
        A configuration value: None, a number, a string, a sequence, a navigation model or a
        dataclass.

    Returns
    -------
    str
        The label; numbers to 10 significant digits, named sensors by their track names.
    """
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float | np.floating | np.integer):
        return _number(float(value))
    if isinstance(value, str):
        return value
    if isinstance(value, tuple | list):
        return "+".join(label(v) for v in value)
    if isinstance(value, PerfectNav):
        return "perfect"
    if isinstance(value, GaussianNav):
        sp, sv, bp, bv = value.sigma[0], value.sigma[3], value.bias_sigma[0], value.bias_sigma[3]
        bound = getattr(value, "bound", None)
        if (sp, sv, bp, bv, bound) == (1.0, 1e-3, 0.0, 0.0, None):
            return "nav1"
        return f"gaussian({label((sp, sv, bp, bv))}{'' if bound is None else ',b' + label(bound)})"
    if isinstance(value, RangeScaledNav):
        items = (
            value.k_range,
            value.k_lateral,
            value.floor,
            value.c_range,
            value.k_vel,
            value.floor_vel,
        )
        bound = getattr(value, "bound", None)
        return f"range_scaled({label(items)}{'' if bound is None else ',b' + label(bound)})"
    if isinstance(value, GaussMarkovNav):
        sp, sv = math.sqrt(value.cov[0, 0]), math.sqrt(value.cov[1, 1])
        zeta = -value.system[1, 1] / (2.0 * sv / sp)
        if np.allclose((sp, sv, zeta), (0.04, 4e-4, 1.0), rtol=1e-9, atol=0.0):
            return "nav2"
        return f"gauss_markov({label((sp, sv, zeta))})"
    tag = getattr(value, "label", None)
    if isinstance(tag, str):
        return tag
    if is_dataclass(value) and not isinstance(value, type):
        inner = ",".join(label(getattr(value, f.name)) for f in fields(value))
        return f"{type(value).__name__.lower()}({inner})"
    return type(value).__name__.lower()


def config_tokens(cfg: EnvConfig, declared: EnvConfig) -> tuple[str, ...]:
    """`field=value` for every result-changing field of `cfg` that differs from `declared`.

    Parameters
    ----------
    cfg : EnvConfig
        The configuration the environment flies.
    declared : EnvConfig
        The spec's declared configuration.

    Returns
    -------
    tuple of str
        Tokens in `EnvConfig` field order; the reward and training fields are left out.
    """
    tokens = []
    for f in fields(EnvConfig):
        if f.name in _SKIPPED_FIELDS:
            continue
        mine, theirs = label(getattr(cfg, f.name)), label(getattr(declared, f.name))
        if mine != theirs:
            tokens.append(f"{f.name}={mine}")
    return tuple(tokens)


def dynamics_token(cfg: EnvConfig) -> str:
    """The identifier's dynamics field; the J2 rung names the inclination it depends on.

    Parameters
    ----------
    cfg : EnvConfig
        The environment's configuration.

    Returns
    -------
    str
        `cfg.dynamics`, with `-i<deg>` appended for "twobody_j2".
    """
    if cfg.dynamics == "twobody_j2":
        return f"twobody_j2-i{_number(math.degrees(cfg.inclination))}"
    return cfg.dynamics


def spec_fingerprint(spec: ScenarioSpec) -> str:
    """A short hash of everything in a spec that changes what an episode flies.

    Parameters
    ----------
    spec : ScenarioSpec
        The spec, with its applied variants.

    Returns
    -------
    str
        The first 12 hex digits of a SHA-256 over the kind, configuration labels, parameters,
        dispersions, target mode, cost and version.
    """
    kind = getattr(spec.entry_point, "__qualname__", spec.entry_point)
    config = tuple(sorted((key, label(value)) for key, value in spec.config.items()))
    parameters = tuple(sorted((key, repr(value)) for key, value in spec.parameters.items()))
    dispersions = tuple((key, repr(value)) for key, value in spec.dispersions.items())
    content = repr(
        (kind, config, parameters, dispersions, repr(spec.target), spec.cost, spec.version)
    )
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def scenario_id(spec: ScenarioSpec, cfg: EnvConfig) -> str:
    """The identifier of the environment `spec` builds with `cfg`.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec, with its applied variants.
    cfg : EnvConfig
        The configuration the environment flies.

    Returns
    -------
    str
        `vbar:{scenario}:{dynamics}:{actuator}:{cost}:v{n}`, the applied variants and the
        configuration tokens, joined by colons; a spec that differs from the registered spec of
        its name and variants adds `spec=<fingerprint>`, so a changed spec never borrows the
        catalogue's identifier.
    """
    from .registry import SCENARIOS

    known = SCENARIOS.get(spec.name)
    try:
        registered = None if known is None else known.variant(*spec.applied)
    except KeyError:
        registered = None
    fingerprint = spec_fingerprint(spec)
    changed = registered is None or spec_fingerprint(registered) != fingerprint
    declared = EnvConfig(**spec.config)
    parts = [
        "vbar",
        spec.name,
        dynamics_token(cfg),
        cfg.thrust_limit,
        spec.cost,
        f"v{spec.version}",
        *spec.applied,
        *config_tokens(cfg, declared),
    ]
    if changed:
        parts.append(f"spec={fingerprint}")
    return ":".join(parts)


def build_config(spec: ScenarioSpec, config: Mapping[str, Any] | None = None) -> EnvConfig:
    """The spec's `EnvConfig` with `config` overrides applied.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec.
    config : mapping, optional
        `EnvConfig` field overrides (SI units).

    Returns
    -------
    EnvConfig
        The configuration.

    Raises
    ------
    TypeError
        If `config` names a field `EnvConfig` does not have.
    """
    declared = EnvConfig(**spec.config)
    return replace(declared, **dict(config or {})) if config else declared


class ScenarioEnv(BaseEnv):
    """A scenario environment driven by a `ScenarioSpec`.

    Subclasses (the scenario kinds) set `params_type`, read `self.params` and `self.draw` (the
    episode's dispersed parameters, nominal before the first reset) and implement the hooks of
    `BaseEnv`. The spec's target mode gives time-varying geometry, scheduled burns and the
    public description agents get in the goal under "target".

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec.
    config : mapping, optional
        `EnvConfig` field overrides (SI units).

    Raises
    ------
    ValueError
        If a `thrust_inhibit` edge is not a whole number of steps.
    """

    params_type: ClassVar[type] = ScenarioParams
    scenario_name = "scenario"

    def __init__(self, spec: ScenarioSpec, config: Mapping[str, Any] | None = None):
        self.scenario = spec
        self.params = self.params_type(**spec.parameters)
        self.target = spec.target
        self.draw: dict[str, Any] = spec.nominal()
        self.soft_constraints = tuple(self.params.soft)
        self.scenario_name = spec.name
        super().__init__(build_config(spec, config))
        self.burn_edges = burn_boundaries(self.target)
        if self.burn_edges:
            self.dynamics = ManeuveringDynamics(self.dynamics, self.target)
        for a, b in self.params.thrust_inhibit:
            if not self._on_grid(a) or not self._on_grid(b):
                raise ValueError("thrust_inhibit edges must be whole steps")

    def _on_grid(self, t: float) -> bool:
        return abs(t / self.cfg.dt - round(t / self.cfg.dt)) < 1e-9

    @property
    def scenario_id(self) -> str:
        """The identifier of what this environment flies (`scenario_id`)."""
        return scenario_id(self.scenario, self.cfg)

    def _draw_episode(self, rng: np.random.Generator) -> None:
        self.draw = self.scenario.draw(rng)
        # Draws named target_<field> set that field of the target mode (a pitch, a phase).
        fields_drawn = {k[7:]: v for k, v in self.draw.items() if k.startswith("target_")}
        if fields_drawn:
            try:
                self.target = replace(cast(Any, self.scenario.target), **fields_drawn)
            except TypeError as error:
                raise ValueError(
                    f"target draws {sorted(fields_drawn)} need a mode with those fields, "
                    f"not {type(self.scenario.target).__name__}"
                ) from error
        u0 = self.draw.get("argument_of_latitude")
        if u0 is not None and self.cfg.dynamics == "twobody_j2":
            # The J2 drift depends on where in its orbit the target starts (Alfriend et al.,
            # ch. 8): the mean argument of latitude is a dispersed parameter.
            body = get_body(self.cfg.body)
            a = body.radius + self.cfg.altitude_m
            model = TwoBodyDynamics(
                body, j2=True, elements=[a, 0.0, self.cfg.inclination, 0.0, 0.0, float(u0)]
            )
            self.dynamics = ManeuveringDynamics(model, self.target) if self.burn_edges else model
        elif self.burn_edges and fields_drawn:
            inner = getattr(self.dynamics, "inner", self.dynamics)
            self.dynamics = ManeuveringDynamics(inner, self.target)
        # Constraints may hold the episode's draws (a GO time, the target mode, the Sun).
        self.constraints = tuple(self._constraints())

    def _sample_start(self, rng: np.random.Generator) -> np.ndarray:
        pos = np.asarray(self.draw["start_pos"], dtype=float).reshape(3)
        vel = np.asarray(self.draw["start_vel"], dtype=float).reshape(3)
        return np.concatenate([pos, vel])

    def _limit_command(self, fraction: np.ndarray) -> np.ndarray:
        for a, b in self.params.thrust_inhibit:
            if a - _EDGE_TOL <= self.t < b - _EDGE_TOL:
                return np.zeros_like(fraction)
        return fraction

    def _split_actuation(self, actuation: Actuation) -> Actuation:
        if not self.burn_edges:
            return actuation
        pieces = split_at(actuation.segments, self.t, self.burn_edges)
        return replace(actuation, segments=pieces)

    def _goal(self) -> dict[str, Any]:
        goal: dict[str, Any] = {"target": self.target.public(self.n)}
        if self.params.thrust_inhibit:
            # Agents know when their commands will be zeroed, as a crew knows the manoeuvre plan.
            goal["thrust_inhibit"] = self.params.thrust_inhibit
        return goal

    def inhibited(self) -> bool:
        """Whether commands are being zeroed now.

        Returns
        -------
        bool
            True inside a `thrust_inhibit` window.
        """
        return any(a - _EDGE_TOL <= self.t < b - _EDGE_TOL for a, b in self.params.thrust_inhibit)


__all__ = [
    "ScenarioEnv",
    "ScenarioParams",
    "build_config",
    "config_tokens",
    "dynamics_token",
    "label",
    "scenario_id",
    "spec_fingerprint",
]
