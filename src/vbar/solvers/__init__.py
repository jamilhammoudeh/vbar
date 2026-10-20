"""Solvers for the safety filters and the horizon problems.

`HillScaling` gives nondimensional units for Hill-frame problems. `project` projects a three-axis
command onto linear rows under a box or ball thrust limit, exactly, with multipliers and a
certificate of infeasibility.
"""

from .qp import ActuatorSet, Projection, Rows, project
from .scaling import HillScaling

__all__ = ["ActuatorSet", "HillScaling", "Projection", "Rows", "project"]
