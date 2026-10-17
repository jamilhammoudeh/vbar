"""Deliberately broken versions of the PD baseline, for checking that metrics detect defects."""

import numpy as np

from vbar.agents import PDAgent


class Mutant(PDAgent):
    """PD with one defect of a given severity.

    `kind` is "position_bias" (m added to the observed position along x), "velocity_bias" (m/s
    added to the observed velocity along x), "delay" (acts on the observation `severity` steps
    old), "sign_flip" (feedback with the wrong sign), "dither" (adds `severity` times a fixed
    sequence of standard normal draws, so levels are scaled copies of one noise law) or
    "chatter" (adds +/- `severity` on x, alternating every step).
    """

    def __init__(self, kind, severity, noise_seed=7):
        super().__init__(name=f"{kind}:{severity}")
        self.kind, self.severity, self.noise_seed = kind, severity, noise_seed

    def reset(self, view, rng=None):
        super().reset(view, rng)
        self.history = []
        self.noise = np.random.default_rng(self.noise_seed)
        self.k = 0

    def act(self, obs, info):
        obs = np.array(obs, dtype=float)
        self.k += 1
        if self.kind == "position_bias":
            obs[0] += self.severity
        elif self.kind == "velocity_bias":
            obs[3] += self.severity
        elif self.kind == "delay":
            self.history.append(obs)
            obs = self.history[max(0, len(self.history) - 1 - int(self.severity))]
        u = super().act(obs, info)
        if self.kind == "sign_flip":
            u = -u
        elif self.kind == "dither":
            u = np.clip(u + self.severity * self.noise.standard_normal(3), -1.0, 1.0)
        elif self.kind == "chatter":
            u = np.clip(u + np.array([self.severity * (-1) ** self.k, 0.0, 0.0]), -1.0, 1.0)
        return u
