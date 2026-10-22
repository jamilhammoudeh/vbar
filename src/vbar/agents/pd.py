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

    The command is clipped per axis (box layout) or scaled onto the ball keeping its direction
    (ball layout) at `max_accel`. In thrust mode it asks for the thrust its acceleration needs
    at the initial mass.
    """

    def __init__(self, wn: float = 0.01, zeta: float = 1.0, name: str = "pd"):
        self.wn = wn
        self.zeta = zeta
        self.name = name

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Read the orbit, hold point, thrust limit and action mode from the view.

        Parameters
        ----------
        view : AgentView
            Scenario description; uses `n`, `goal["hold_point"]` (Hill frame, m), `max_accel`
            (m/s^2), `thrust_limit` and the action mode.
        rng : numpy.random.Generator, optional
            Unused; the law is deterministic.

        Raises
        ------
        ValueError
            If the goal is not a hold or berthing task or has no hold point, the thrust layout
            is neither "box" nor "ball", or the action mode takes no continuous command.
        """
        self.hold = hold_point(view, "PD agent")
        if view.thrust_limit not in ("box", "ball"):
            raise ValueError(f"unknown thrust limit {view.thrust_limit!r}")
        self.n = view.n
        self.max_accel = view.max_accel
        self.thrust_limit = view.thrust_limit
        self.scale = action_scale(view)

    def act(self, obs, info):
        x, y, z, vx, vy, vz = obs[:6]
        n = self.n
        cw = np.array([3 * n**2 * x + 2 * n * vy, -2 * n * vx, -(n**2) * z])
        kp, kd = self.wn**2, 2 * self.zeta * self.wn
        a = -cw - kp * (obs[:3] - self.hold) - kd * obs[3:6]
        u = a / self.max_accel
        if self.thrust_limit == "ball":
            size = np.linalg.norm(u)
            if size > 1.0:
                u = u / size
        else:
            u = np.clip(u, -1.0, 1.0)
        return u * self.scale
