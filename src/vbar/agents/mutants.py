"""Deliberately broken versions of any continuous-command agent, for testing the metrics.

`Mutant` wraps any agent and changes the observation before it or its command after, in fractions
of a full command, with one defect of a given severity. A metric that is responsible for a defect
should move in the bad direction when the defect is present and, where the kind has an ordered
severity, rank the severities in order.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Agent, AgentView, action_scale, command_fraction

MUTANT_KINDS = (
    "position_bias",
    "velocity_bias",
    "delay",
    "sign_flip",
    "dither",
    "chatter",
    "radial_sign",
    "no_coriolis",
    "bang_bang",
)


class Mutant:
    """An agent with one defect of a given severity.

    The kinds, with u the inner agent's command in fractions of a full command (Hill frame) and
    s the severity:

    - "position_bias": s (m) is added to the observed radial position x. Raises the terminal
      position error and lowers the composite score as s grows.
    - "velocity_bias": s (m/s) is added to the observed radial velocity vx. Same metrics.
    - "delay": the inner agent sees the observation of s decisions before, the first one until
      that many exist; a fractional delay severity is rounded down to whole steps. Raises
      delta-v with the delay; at six 10 s steps the approach from behind enters the keep-out
      zone.
    - "sign_flip": the command is -u; s is not used. Fails the mission with no progress and
      fails the sign check on all six directions.
    - "dither": s times a standard normal draw per axis is added to u, clipped to [-1, 1]. Raises
      delta-v and lowers the keep-out margin as s grows; the levels are scaled copies of one
      sequence of draws.
    - "chatter": -s, +s, -s, ... is added to the radial component of u, clipped to [-1, 1].
      Raises the chatter metric with s.
    - "radial_sign": the radial component of u is negated; s is not used. Fails the mission, and
      the sign check on +x and -x.
    - "no_coriolis": s (2 n vy, -2 n vx, 0) / max_accel is added to u, clipped to [-1, 1], with
      the observed velocity, so s = 1 undoes a law's cancellation of the Coriolis term.
      Detection only: capture is later and delta-v higher, but the Coriolis force does no work,
      so no severity ordering holds.
    - "bang_bang": the command is s sign(u) per axis. Raises the chatter metric above the inner
      agent's at every level and delta-v with s.

    The observation and the inner agent's command are copied, never changed in place. Only the
    continuous action modes are supported.

    Parameters
    ----------
    agent : Agent
        The agent to break; reset and called through the `Agent` protocol only.
    kind : str
        One of `MUTANT_KINDS`.
    severity : float
        Size of the defect, in the units of its kind above.
    noise_seed : int
        Seed of the dither draws when `reset` gets no generator.

    Raises
    ------
    ValueError
        If `kind` is unknown, `severity` is not finite, or a delay is negative.
    """

    def __init__(self, agent: Agent, kind: str, severity: float, noise_seed: int = 7):
        if kind not in MUTANT_KINDS:
            raise ValueError(f"unknown mutant kind {kind!r}, expected one of {MUTANT_KINDS}")
        if not math.isfinite(severity):
            raise ValueError("severity must be finite")
        if kind == "delay" and severity < 0:
            raise ValueError("delay must not be negative")
        self.agent = agent
        self.kind = kind
        self.severity = severity
        self.noise_seed = noise_seed
        self.name = f"{agent.name}:{kind}:{severity}"

    def reset(self, view: AgentView, rng: np.random.Generator | None = None) -> None:
        """Reset the inner agent and the defect's own state.

        Parameters
        ----------
        view : AgentView
            Scenario description, passed to the inner agent; the wrapper uses `n`,
            `max_accel` and `action_mode`.
        rng : numpy.random.Generator, optional
            Passed to the inner agent unchanged; the dither draws come from a child spawned
            from it, or from `default_rng(noise_seed)` when None.

        Raises
        ------
        ValueError
            If the action mode takes no continuous command.
        """
        self.scale = action_scale(view)
        self.view = view
        self.agent.reset(view, rng)
        # A spawned child leaves the parent's stream untouched, so a random inner agent draws
        # what it would unwrapped and the dither does not depend on how much it draws.
        self.noise = np.random.default_rng(self.noise_seed) if rng is None else rng.spawn(1)[0]
        self.history: list[np.ndarray] = []
        self.k = 0

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """The inner agent's command with the defect applied.

        Parameters
        ----------
        obs : numpy.ndarray
            Observation in the order of the view's names, the Hill state (m, m/s) first.
        info : dict
            Passed to the inner agent.

        Returns
        -------
        numpy.ndarray, shape (3,)
            Command in the view's action mode.
        """
        obs = np.array(obs, dtype=float)
        s = self.severity
        self.k += 1
        if self.kind == "position_bias":
            obs[0] += s
        elif self.kind == "velocity_bias":
            obs[3] += s
        elif self.kind == "delay":
            self.history.append(obs)
            obs = self.history[max(0, len(self.history) - 1 - int(s))].copy()
        # command_fraction divides, so u is a new array and the inner command stays untouched.
        u = command_fraction(self.view, self.agent.act(obs, info))
        if self.kind == "sign_flip":
            u = -u
        elif self.kind == "dither":
            u = np.clip(u + s * self.noise.standard_normal(3), -1.0, 1.0)
        elif self.kind == "chatter":
            u = np.clip(u + np.array([s * (-1) ** self.k, 0.0, 0.0]), -1.0, 1.0)
        elif self.kind == "radial_sign":
            u = u * np.array([-1.0, 1.0, 1.0])
        elif self.kind == "no_coriolis":
            n = self.view.n
            coriolis = np.array([2 * n * obs[4], -2 * n * obs[3], 0.0])
            u = np.clip(u + s * coriolis / self.view.max_accel, -1.0, 1.0)
        elif self.kind == "bang_bang":
            u = s * np.sign(u)
        return u * self.scale
