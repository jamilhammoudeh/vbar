"""Tools an LLM agent may call: Clohessy-Wiltshire predictions from its own observation.

A tool sees only the observation the agent received and its public `AgentView`, so it can do
arithmetic for the model but never reveal the truth. Results are rounded to the precision a
model can use, which also keeps transcripts short.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constraints import KeepOut, segment_minimum
from ..dynamics import CWDynamics, cw_discrete
from .base import AgentView

# A transfer whose position-from-velocity block is this ill-conditioned is refused. CW targeting
# is singular out of plane where sin nT = 0, every half orbit, and in plane at the roots of
# 8 (1 - cos nT) = 3 nT sin nT, whole orbits among them (Alfriend et al., Spacecraft Formation
# Flying, p. 92).
_MAX_CONDITION = 1e8
# Longest prediction offered, in orbits.
_MAX_ORBITS = 3.0


@dataclass(frozen=True)
class Tool:
    """A function the model may call: name, description and JSON schema of its arguments.

    `run(obs, view, arguments)` returns a JSON-serializable dict; invalid arguments give
    {"error": ...}.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    run: Callable[[np.ndarray, AgentView, Mapping[str, Any]], dict[str, Any]]

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _number(arguments: Mapping[str, Any], key: str) -> float:
    value = float(arguments[key])
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def predict_coast(obs: np.ndarray, view: AgentView, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """State after coasting `seconds` from the observed state, and the closest range on the way."""
    seconds = _number(arguments, "seconds")
    longest = _MAX_ORBITS * 2.0 * np.pi / view.n
    if not 0.0 < seconds <= longest:
        return {"error": f"seconds must be in (0, {longest:.0f}]"}
    model = CWDynamics(view.n)
    model.reset(obs[:6])
    count = int(min(2001, max(2, seconds + 1)))
    end, samples = model.advance(np.zeros(3), seconds, n_samples=count)
    times = np.linspace(0.0, seconds, count)
    closest = segment_minimum(
        KeepOut(0.0), model.state_in_last_segment, times, np.asarray(samples), 0.0, view.n
    )
    return {
        "position_m": np.round(end[:3], 2).tolist(),
        "velocity_mps": np.round(end[3:], 4).tolist(),
        "closest_range_m": round(closest.margin, 2),
        "closest_at_s": round(closest.t, 1),
    }


def transfer_burn(obs: np.ndarray, view: AgentView, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Two-impulse CW transfer from the observed state to rest at a point in `seconds`."""
    target = np.array([_number(arguments, k) for k in ("x", "y", "z")])
    seconds = _number(arguments, "seconds")
    if not seconds > 0.0:
        return {"error": "seconds must be positive"}
    phi, _ = cw_discrete(view.n, seconds)
    rr, rv, vr, vv = phi[:3, :3], phi[:3, 3:], phi[3:, :3], phi[3:, 3:]
    if np.linalg.cond(rv) > _MAX_CONDITION:
        return {"error": "no transfer of this duration: the targeting problem is singular"}
    r0, v0 = obs[:3], obs[3:6]
    v_depart = np.linalg.solve(rv, target - rr @ r0)
    first, second = v_depart - v0, -(vr @ r0 + vv @ v_depart)
    return {
        "first_burn_mps": np.round(first, 4).tolist(),
        "second_burn_mps": np.round(second, 4).tolist(),
        "total_mps": round(float(np.linalg.norm(first) + np.linalg.norm(second)), 4),
        "first_burn_steps_at_full_thrust": round(
            float(np.max(np.abs(first)) / (view.max_accel * view.dt)), 2
        ),
    }


PREDICT_COAST = Tool(
    "predict_coast",
    "Predict the position and velocity after coasting (no thrust) for a number of seconds from "
    "the current observed state, and the closest distance to the target on the way.",
    {
        "type": "object",
        "properties": {"seconds": {"type": "number", "description": "coast duration, s"}},
        "required": ["seconds"],
    },
    predict_coast,
)

TRANSFER_BURN = Tool(
    "transfer_burn",
    "Compute the two velocity changes (impulses) that move the spacecraft from its current "
    "observed state to rest at a point (x, y, z, in meters) in a given number of seconds.",
    {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
            "seconds": {"type": "number", "description": "transfer duration, s"},
        },
        "required": ["x", "y", "z", "seconds"],
    },
    transfer_burn,
)

CW_TOOLS = (PREDICT_COAST, TRANSFER_BURN)


def call_tool(tools: Mapping[str, Tool], name: str, obs, view, arguments) -> dict[str, Any]:
    """Run tool `name`; unknown tools and bad arguments come back as {"error": ...}."""
    tool = tools.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return tool.run(np.asarray(obs, dtype=float), view, dict(arguments))
    except (KeyError, TypeError, ValueError) as error:
        return {"error": f"bad arguments for {name}: {error}"}
