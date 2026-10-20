"""Solvers for the safety filters and the horizon problems.

`HillScaling` gives nondimensional units for Hill-frame problems.
"""

from .scaling import HillScaling

__all__ = ["HillScaling"]
