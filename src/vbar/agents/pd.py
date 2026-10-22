"""Classical baseline: proportional-derivative (PD) controller with CW cancellation.

Idea: the CW equations add "orbital" accelerations (3n^2 x + 2n vy, -2n vx,
-n^2 z). We cancel them, then add a PD law that pulls the chaser to the hold
point like a damped spring:

    a = -(CW terms) - Kp * (position - hold_point) - Kd * velocity

With Kp = wn^2 and Kd = 2 * zeta * wn, the error behaves like a damped
oscillator with natural frequency wn and damping ratio zeta.

The controller ignores the keep-out zone, so it flies straight through it when
the target lies between the start and the hold point. It serves as the
unfiltered baseline.
"""

from __future__ import annotations

import numpy as np

from .base import AgentView, action_scale, hold_point


class PDAgent:
    """PD law toward the hold point, issued in the view's action mode.

    In thrust mode it asks for the thrust its acceleration needs at the initial mass.
    """

    def __init__(self, wn: float = 0.01, zeta: float = 1.0, name: str = "pd"):
        self.wn = wn
        self.zeta = zeta
        self.name = name

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        self.hold = hold_point(view, "PD agent")
        self.n = view.n
        self.max_accel = view.max_accel
        self.scale = action_scale(view)

    def act(self, obs, info):
        x, y, z, vx, vy, vz = obs[:6]
        n = self.n
        cw = np.array([3 * n**2 * x + 2 * n * vy, -2 * n * vx, -(n**2) * z])
        kp, kd = self.wn**2, 2 * self.zeta * self.wn
        a = -cw - kp * (obs[:3] - self.hold) - kd * obs[3:6]
        return np.clip(a / self.max_accel, -1.0, 1.0) * self.scale
