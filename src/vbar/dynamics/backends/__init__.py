"""Independent propagators that fly the same states as V-BAR's two-body rung, for agreement checks.

Each backend propagates a target and a chaser from inertial states with the central body's point
mass and, optionally, its J2 term, using the constants of `vbar.dynamics.bodies`, so a difference
from V-BAR comes from the integrator or the force model's implementation rather than from the
constants. Backends are optional extras: `vbar[basilisk]` and `vbar[orekit]`.
"""

from .common import Burn, hill_states, relative_inertial

__all__ = ["Burn", "hill_states", "relative_inertial"]
