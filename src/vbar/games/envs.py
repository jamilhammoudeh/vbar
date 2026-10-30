"""The pe1 scenarios as KSPDG-format classes.

`PE1_E1_I3_Env` and `PE1_E3_I3_Env` carry KSPDG's environment class names, so a KSPDG agent
script needs only its imports changed. They are V-BAR versions on two-body dynamics, and their
evaders are V-BAR's bots: results are not comparable with KSPDG's.
"""

from __future__ import annotations

from typing import Any

from .bots import NormalEscapeEvader, PassiveEvader
from .pursuit import PursuitEvasionEnv


class PE1_E1_I3_Env(PursuitEvasionEnv):  # noqa: N801  KSPDG's class name
    """pe1 from the I3 start against V-BAR's passive evader (KSPDG's e1 is passive)."""

    scenario = "pe1-e1-i3"

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("evader", PassiveEvader())
        super().__init__(initial="i3", **kwargs)


class PE1_E3_I3_Env(PursuitEvasionEnv):  # noqa: N801  KSPDG's class name
    """pe1 from the I3 start against V-BAR's `NormalEscapeEvader`, not KSPDG's e3 bot."""

    scenario = "pe1-e3-i3"

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("evader", NormalEscapeEvader())
        super().__init__(initial="i3", **kwargs)


SCENARIOS: dict[str, type[PursuitEvasionEnv]] = {
    "pe1-e1-i3": PE1_E1_I3_Env,
    "pe1-e3-i3": PE1_E3_I3_Env,
}
__all__ = ["PE1_E1_I3_Env", "PE1_E3_I3_Env", "SCENARIOS"]
