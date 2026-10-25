from . import catalog as _catalog  # noqa: F401  (registers the built-in scenarios and suites)
from .approach import ApproachConfig, ApproachEnv
from .base import BaseEnv, EnvConfig, SegmentTrace, StepRecord
from .berthing import BerthingEnv, BerthingParams
from .departure import DepartureEnv, DepartureParams
from .docking import PRESETS, DockingEnv, DockingParams
from .hold import HoldParams, HoldPointEnv
from .inspection import InspectionEnv, InspectionParams
from .registry import (
    SCENARIOS,
    by_tag,
    get,
    group_by_tag,
    gymnasium_id,
    load_plugins,
    make,
    parse_id,
    plugins,
    register,
    tags_of,
)
from .scenario import ScenarioEnv, ScenarioParams, scenario_id
from .seeds import SPLITS, SeedSplit, seeds
from .spec import (
    AngleShell,
    Choice,
    DriftingVelocity,
    Fixed,
    LimitedSpeed,
    Normal,
    Observation,
    ScenarioSpec,
    Shell,
    Uniform,
    per_draw_level,
)
from .suites import SUITES, Suite, SuiteEntry, register_suite
from .targets import (
    FixedOffset,
    InertialPointing,
    Maneuvering,
    Misbriefed,
    Passive,
    TargetMode,
    TorqueFree,
    Tumbling,
)
from .vector import ApproachVectorEnv

__all__ = [
    "PRESETS",
    "SCENARIOS",
    "SPLITS",
    "SUITES",
    "AngleShell",
    "ApproachConfig",
    "ApproachEnv",
    "ApproachVectorEnv",
    "BaseEnv",
    "BerthingEnv",
    "BerthingParams",
    "Choice",
    "DepartureEnv",
    "DepartureParams",
    "DockingEnv",
    "DockingParams",
    "DriftingVelocity",
    "EnvConfig",
    "Fixed",
    "FixedOffset",
    "HoldParams",
    "HoldPointEnv",
    "InertialPointing",
    "InspectionEnv",
    "InspectionParams",
    "LimitedSpeed",
    "Maneuvering",
    "Misbriefed",
    "Normal",
    "Observation",
    "Passive",
    "ScenarioEnv",
    "ScenarioParams",
    "ScenarioSpec",
    "SeedSplit",
    "SegmentTrace",
    "Shell",
    "StepRecord",
    "Suite",
    "SuiteEntry",
    "TargetMode",
    "TorqueFree",
    "Tumbling",
    "Uniform",
    "by_tag",
    "get",
    "group_by_tag",
    "gymnasium_id",
    "load_plugins",
    "make",
    "parse_id",
    "per_draw_level",
    "plugins",
    "register",
    "register_suite",
    "scenario_id",
    "seeds",
    "tags_of",
]
