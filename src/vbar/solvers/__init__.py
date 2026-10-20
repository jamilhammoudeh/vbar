"""Solvers for the safety filters and the horizon problems.

`HillScaling` gives nondimensional units for Hill-frame problems. `project` projects a three-axis
command onto linear rows under a box or ball thrust limit, exactly, with multipliers and a
certificate of infeasibility. `StageQP` holds a horizon problem with the dynamics as equality
rows and `riccati_solve` solves it without its rows at a cost linear in the horizon.
"""

from .qp import ActuatorSet, Projection, Rows, project
from .scaling import HillScaling
from .stage import StageQP, riccati_solve

__all__ = [
    "ActuatorSet",
    "HillScaling",
    "Projection",
    "Rows",
    "StageQP",
    "project",
    "riccati_solve",
]
