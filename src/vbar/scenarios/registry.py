"""The scenario catalogue: specs by name, identifiers, Gymnasium registration and plugins.

`register(spec)` adds a spec and registers it and each of its variants with Gymnasium as
`vbar/<Name>-v<n>` (`vbar/ApproachVbar-v1`, `vbar/ApproachVbarIss-v1`). `make` builds an
environment from a name, a name with variant suffixes (`approach-vbar:iss:nav1`) or a full
identifier. Third-party packages add scenarios, agents and filters through the entry-point
groups `vbar.scenarios`, `vbar.agents` and `vbar.filters`.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Iterable, Mapping
from importlib.metadata import entry_points
from typing import Any

import gymnasium as gym

from .scenario import ScenarioEnv
from .spec import ScenarioSpec

log = logging.getLogger(__name__)

SCENARIOS: dict[str, ScenarioSpec] = {}
PLUGIN_GROUPS = ("vbar.scenarios", "vbar.agents", "vbar.filters")
_ID = re.compile(r"^vbar:([^:]+):([^:]+):([^:]+):([^:]+):v(\d+)((?::[^:]+)*)$")


def camel(name: str) -> str:
    """A hyphen- or underscore-separated name in CamelCase.

    Parameters
    ----------
    name : str
        Name such as "approach-vbar".

    Returns
    -------
    str
        CamelCase name such as "ApproachVbar".
    """
    return "".join(part.capitalize() for part in re.split(r"[-_]", name) if part)


def gymnasium_id(spec: ScenarioSpec, variants: Iterable[str] = ()) -> str:
    """The Gymnasium identifier of a spec with the given variants applied.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec.
    variants : iterable of str
        Variant names, in order.

    Returns
    -------
    str
        `vbar/<Name><Variants>-v<n>`.
    """
    return f"vbar/{camel(spec.name)}{''.join(camel(v) for v in variants)}-v{spec.version}"


def entry_class(spec: ScenarioSpec) -> type[ScenarioEnv]:
    """The environment class a spec names.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec.

    Returns
    -------
    type
        The `ScenarioEnv` subclass of `spec.entry_point`, imported when given as a path.

    Raises
    ------
    ValueError
        If a string entry point is not "module:Class".
    TypeError
        If the entry point is not a `ScenarioEnv` subclass.
    """
    target = spec.entry_point
    if isinstance(target, str):
        module, _, attr = target.partition(":")
        if not attr:
            raise ValueError(f"entry_point {target!r} must be 'module:Class'")
        target = getattr(importlib.import_module(module), attr)
    if not (isinstance(target, type) and issubclass(target, ScenarioEnv)):
        raise TypeError(f"entry_point of {spec.name!r} is not a ScenarioEnv subclass")
    return target


def register(spec: ScenarioSpec, gymnasium: bool = True) -> ScenarioSpec:
    """Add a spec to the catalogue; a name already present must be the same spec.

    Parameters
    ----------
    spec : ScenarioSpec
        The scenario spec.
    gymnasium : bool
        Whether to register the spec and each variant with Gymnasium.

    Returns
    -------
    ScenarioSpec
        `spec`.

    Raises
    ------
    ValueError
        If a different spec is registered under the same name.
    """
    known = SCENARIOS.get(spec.name)
    if known is not None and known != spec:
        raise ValueError(f"scenario {spec.name!r} is already registered with a different spec")
    SCENARIOS[spec.name] = spec
    if gymnasium:
        for variants in [(), *((v,) for v in spec.variants)]:
            gym_id = gymnasium_id(spec, variants)
            if gym_id not in gym.registry:
                gym.register(
                    id=gym_id,
                    entry_point="vbar.scenarios.registry:_make_registered",
                    kwargs={"name": spec.name, "variants": variants},
                )
    return spec


def get(name: str) -> ScenarioSpec:
    """The spec of `name`, with any `:variant` suffixes applied.

    Parameters
    ----------
    name : str
        Scenario name, optionally followed by `:variant` suffixes.

    Returns
    -------
    ScenarioSpec
        The spec with the variants applied in order.

    Raises
    ------
    KeyError
        If the scenario or a variant is unknown.
    """
    base, *variants = name.split(":")
    if base not in SCENARIOS:
        raise KeyError(f"unknown scenario {base!r}; known: {sorted(SCENARIOS)}")
    return SCENARIOS[base].variant(*variants)


def parse_id(identifier: str) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    """Split an identifier into its scenario name, variant names and dynamics override.

    Parameters
    ----------
    identifier : str
        A full identifier, `vbar:{scenario}:{dynamics}:{actuator}:{cost}:v{n}[:variant...]`.

    Returns
    -------
    tuple
        The scenario name, the variant names, and the `EnvConfig` overrides "dynamics",
        "thrust_limit" and, for the J2 rung, "inclination" (rad).

    Raises
    ------
    ValueError
        If the identifier is malformed, names another version or cost, carries a
        configuration token or names an unknown variant.
    KeyError
        If the scenario is unknown.
    """
    match = _ID.match(identifier)
    if match is None:
        raise ValueError(f"not a scenario identifier: {identifier!r}")
    name, dynamics, limit, cost, version, tail = match.groups()
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}")
    spec = SCENARIOS[name]
    if int(version) != spec.version or cost != spec.cost:
        raise ValueError(
            f"{identifier!r} names version {version} and cost {cost!r}, this "
            f"release has v{spec.version} and {spec.cost!r}"
        )
    variants, overrides = [], {}
    for token in tail.split(":")[1:]:
        if token in spec.variants:
            variants.append(token)
        elif "=" in token:
            raise ValueError(
                f"configuration token {token!r} is not rebuilt from an identifier; "
                "pass the field to make() instead"
            )
        else:
            raise ValueError(f"unknown variant {token!r} of {name!r}")
    kind, _, inclination = dynamics.partition("-i")
    overrides["dynamics"] = kind
    if inclination:
        overrides["inclination"] = float(inclination) * 3.141592653589793 / 180.0
    overrides["thrust_limit"] = limit
    return name, tuple(variants), overrides


def make(name: str, **config: Any) -> ScenarioEnv:
    """Build the environment of a scenario name, `name:variant...` or a full identifier.

    `config` overrides `EnvConfig` fields and goes into the identifier.

    Parameters
    ----------
    name : str
        Scenario name, `name:variant...` or a full identifier.
    **config : Any
        `EnvConfig` field overrides (SI units).

    Returns
    -------
    ScenarioEnv
        The environment.

    Raises
    ------
    KeyError
        If the scenario or a variant is unknown.
    ValueError
        If an identifier is malformed or names another version or cost.
    """
    if name.startswith("vbar:"):
        base, variants, overrides = parse_id(name)
        spec = SCENARIOS[base].variant(*variants)
        declared = {k: spec.config.get(k) for k in overrides}
        config = {**{k: v for k, v in overrides.items() if declared[k] != v}, **config}
    else:
        spec = get(name)
    return entry_class(spec)(spec, config)


def _make_registered(name: str, variants: tuple[str, ...] = (), **config: Any) -> ScenarioEnv:
    spec = SCENARIOS[name].variant(*variants)
    return entry_class(spec)(spec, config)


def by_tag(tag: str) -> list[str]:
    """Names of the registered scenarios carrying `tag`.

    Parameters
    ----------
    tag : str
        The tag.

    Returns
    -------
    list of str
        Sorted scenario names; variant tags are not included.
    """
    return sorted(name for name, spec in SCENARIOS.items() if tag in spec.tags)


def tags_of(identifier: str) -> tuple[str, ...]:
    """The tags of a scenario identifier or name, variants included.

    Parameters
    ----------
    identifier : str
        Full identifier, or a scenario name with any `:variant` suffixes.

    Returns
    -------
    tuple of str
        The tags.

    Raises
    ------
    KeyError
        If the scenario or a variant is unknown.
    ValueError
        If an identifier is malformed.
    """
    if identifier.startswith("vbar:"):
        name, variants, _ = parse_id(identifier)
        return SCENARIOS[name].variant(*variants).tags
    return get(identifier).tags


def group_by_tag(results: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Results (mappings with a `scenario_id`) grouped under each tag of their scenario.

    Parameters
    ----------
    results : iterable of mapping
        Episode results, each with "scenario_id".

    Returns
    -------
    dict
        Tag to the results of scenarios carrying it; a result appears under each of its tags.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        for tag in tags_of(result["scenario_id"]):
            groups.setdefault(tag, []).append(result)
    return groups


def plugins(group: str) -> dict[str, Any]:
    """Objects advertised under an entry-point group, by entry-point name.

    An entry point that fails to load is logged as a warning and left out.

    Parameters
    ----------
    group : str
        One of `PLUGIN_GROUPS`.

    Returns
    -------
    dict
        Entry-point name to loaded object.

    Raises
    ------
    ValueError
        If `group` is not one of `PLUGIN_GROUPS`.
    """
    if group not in PLUGIN_GROUPS:
        raise ValueError(f"unknown plugin group {group!r}, expected one of {PLUGIN_GROUPS}")
    found: dict[str, Any] = {}
    for point in entry_points(group=group):
        try:
            found[point.name] = point.load()
        except Exception as error:  # noqa: BLE001 - one broken plugin must not hide the rest
            log.warning("plugin %s in %s failed to load: %s", point.name, group, error)
    return found


def load_plugins() -> list[str]:
    """Register every scenario spec third-party packages advertise; returns their names.

    An entry point may give a `ScenarioSpec` or a callable returning specs.

    Returns
    -------
    list of str
        Names of the specs registered.

    Raises
    ------
    TypeError
        If a plugin gives something other than scenario specs.
    ValueError
        If a plugin's spec clashes with a registered one.
    """
    added = []
    for name, obj in plugins("vbar.scenarios").items():
        specs = obj() if callable(obj) and not isinstance(obj, ScenarioSpec) else obj
        for spec in [specs] if isinstance(specs, ScenarioSpec) else list(specs):
            if not isinstance(spec, ScenarioSpec):
                raise TypeError(f"plugin {name!r} gave {type(spec).__name__}, not a ScenarioSpec")
            register(spec)
            added.append(spec.name)
    return added


__all__ = [
    "PLUGIN_GROUPS",
    "SCENARIOS",
    "by_tag",
    "camel",
    "entry_class",
    "get",
    "group_by_tag",
    "gymnasium_id",
    "load_plugins",
    "make",
    "parse_id",
    "plugins",
    "register",
    "tags_of",
]
