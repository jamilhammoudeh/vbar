"""A learned supervisor: a policy that picks a supervisory command from numeric features.

The features are made from a `Situation` alone, so a policy sees what a rule or language model
supervisor sees and nothing of the true state. `vbar.training.SupervisorEnv` hands the same vector
to a learner as its observation, so a trained policy flies the trajectory it was trained on when
it is wrapped in a `PolicySupervisor` and a `SupervisedAgent`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .base import AgentView
from .supervision import PHASES, Command, Rung, Situation


def supervisor_features(situation: Situation, view: AgentView) -> np.ndarray:
    """The feature vector a learned supervisor acts on.

    The layout is the observation, the observed position relative to the rung being flown to, a
    one-hot of the phase, the rung index over the number of rungs, and the command mask. The rung
    is taken at its t = 0 position; an inspection orbit rung turns with the Sun, so its gap is
    only exact at t = 0.

    Parameters
    ----------
    situation : Situation
        The consult; only its observation, phase, rung, ladder and mask are read.
    view : AgentView
        The scenario's public description; its observation channels give the observation's length.

    Returns
    -------
    numpy.ndarray
        Shape (len(observation) + 3 + len(PHASES) + 1 + len(Command),), float64. The gap is in m
        (Hill axes), the one-hot and mask are 0 or 1, the rung is in [0, 1).

    Raises
    ------
    ValueError
        If the observation does not have the view's channels.
    """
    obs = np.asarray(situation.obs, dtype=float)
    if obs.shape != (len(view.observation),):
        raise ValueError(f"expected {len(view.observation)} observation entries, got {obs.shape}")
    gap = obs[:3] - np.asarray(situation.ladder[situation.rung].position, dtype=float)
    phase = np.zeros(len(PHASES))
    phase[PHASES.index(situation.phase)] = 1.0
    return np.concatenate(
        [
            obs,
            gap,
            phase,
            [situation.rung / len(situation.ladder)],
            np.asarray(situation.mask, dtype=float),
        ]
    )


class PolicySupervisor:
    """A supervisor that gives the command a policy picks on the features of the situation.

    Parameters
    ----------
    policy : callable
        Maps the `supervisor_features` vector to the integer value of a `Command`. It is shared,
        not copied, between the agents built on this supervisor.

    Raises
    ------
    ValueError
        If `policy` is not callable. Calling it raises ValueError for a value that is not a
        `Command`.
    """

    name = "policy"

    def __init__(self, policy: Callable[[np.ndarray], int]):
        if not callable(policy):
            raise ValueError("policy must be callable")
        self.policy = policy
        self._view: AgentView | None = None

    def reset(
        self, view: AgentView, ladder: tuple[Rung, ...], rng: np.random.Generator | None = None
    ) -> None:
        """Start an episode with the public view."""
        self._view = view

    def decide(self, situation: Situation) -> Command:
        """The command the policy picks for the situation."""
        if self._view is None:
            raise RuntimeError("reset the supervisor before deciding")
        return Command(int(self.policy(supervisor_features(situation, self._view))))
