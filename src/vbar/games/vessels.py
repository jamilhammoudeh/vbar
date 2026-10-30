"""The vessels of KSPDG's pe1 problem: mass, propellant and RCS thrust limits.

Values come from KSPDG's public pe1 definition (github.com/mit-ll/spacegym-kspdg, v0.15.0,
`src/kspdg/pe1/pe1_base.py` and the mission save files): 1000 N per RCS nozzle at 240 s specific
impulse, eight nozzles along the forward axis each way and four along each lateral axis each
way, and three tanks of 400 units of monopropellant on a 1885 kg dry vessel. The thrust model
is KSPDG's declared one, not the game's part physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..conventions import G0

NOZZLE_THRUST = 1000.0  # N
RCS_ISP = 240.0  # s
# KSP's monopropellant density, 0.004 t per unit; kRPC reports amounts in units.
MONOPROPELLANT_KG = 4.0


def _number(value: Any, name: str, *, zero: bool = False) -> float:
    """`value` as a float, refused unless it is a real number, finite and positive (or zero)."""
    if isinstance(value, bool | np.bool_) or not isinstance(
        value, int | float | np.integer | np.floating
    ):
        raise ValueError(f"{name} must be a number")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(number) or number < 0.0 or (number == 0.0 and not zero):
        raise ValueError(f"{name} must be {'non-negative' if zero else 'positive'} and finite")
    return number


@dataclass(frozen=True, kw_only=True)
class Vessel:
    """A vessel's dry mass, monopropellant load and RCS thrust limits along its body axes.

    Parameters
    ----------
    dry_mass : float
        Mass without propellant (kg).
    monopropellant : float
        Monopropellant at the start (units of `MONOPROPELLANT_KG` kg).
    max_thrust_positive, max_thrust_negative : tuple of float
        Largest thrust (N) along +forward, +right, +down and along -forward, -right, -down.
    isp : float
        Specific impulse of the thrusters (s).
    """

    dry_mass: float = 1885.0
    monopropellant: float = 1200.0
    max_thrust_positive: tuple[float, float, float] = (
        8 * NOZZLE_THRUST,
        4 * NOZZLE_THRUST,
        4 * NOZZLE_THRUST,
    )
    max_thrust_negative: tuple[float, float, float] = (
        8 * NOZZLE_THRUST,
        4 * NOZZLE_THRUST,
        4 * NOZZLE_THRUST,
    )
    isp: float = RCS_ISP

    def __post_init__(self):
        _number(self.dry_mass, "dry_mass")
        _number(self.isp, "isp")
        _number(self.monopropellant, "monopropellant", zero=True)
        for name in ("max_thrust_positive", "max_thrust_negative"):
            limits = getattr(self, name)
            if not isinstance(limits, tuple | list) or len(limits) != 3:
                raise ValueError(f"{name} must be three thrust limits")
            for limit in limits:
                _number(limit, name, zero=True)

    @property
    def wet_mass(self) -> float:
        """Mass with full tanks (kg)."""
        return self.dry_mass + self.monopropellant * MONOPROPELLANT_KG

    def thrust(self, throttle: np.ndarray) -> np.ndarray:
        """Body-axis thrust (N) of per-axis throttles, each clipped to [-1, 1]."""
        throttle = np.clip(throttle, -1.0, 1.0)
        limit = np.where(throttle >= 0.0, self.max_thrust_positive, self.max_thrust_negative)
        return throttle * limit

    def throttle(self, thrust: np.ndarray) -> np.ndarray:
        """Per-axis throttles of a body-axis thrust (N), before clipping."""
        limit = np.where(thrust >= 0.0, self.max_thrust_positive, self.max_thrust_negative)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(limit > 0.0, thrust / limit, 0.0)

    def mass_flow(self, thrust: np.ndarray) -> float:
        """Propellant mass flow (kg/s) of a body-axis thrust; each axis has its own nozzles."""
        return float(np.abs(thrust).sum()) / (self.isp * G0)


PE1_VESSEL = Vessel()

__all__ = ["MONOPROPELLANT_KG", "NOZZLE_THRUST", "PE1_VESSEL", "RCS_ISP", "Vessel"]
