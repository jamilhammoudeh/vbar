"""Reference states shared by the dynamics tests."""

import numpy as np

# Computed with Koon, Lo, Marsden and Ross's Earth-Moon constants: the 9:2 synodic-resonant L2
# near-rectilinear halo orbit with its apolune below the plane, at its perilune-side crossing of
# the x-z plane, and its nondimensional period.
NRHO = np.array([0.9873849199042153, 0.0, 0.008369210996156513, 0.0, 1.6744665364075588, 0.0])
NRHO_PERIOD = 1.5088887454509277


def pco_state(rho, alpha, n):
    """CW state on a projected circular orbit of radius `rho` (m) at phase `alpha` (rad)."""
    s, c = np.sin(alpha), np.cos(alpha)
    return [0.5 * rho * s, rho * c, rho * s, 0.5 * rho * n * c, -rho * n * s, rho * n * c]
