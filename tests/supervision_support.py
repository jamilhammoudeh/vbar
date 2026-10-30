"""Stand-in supervisors and stepping harnesses for the supervision and tracker tests.

`Session` feeds a supervised agent observations chosen by the test, one step at a time, so a
mask or a phase can be checked on a constructed state. `Flight` flies a scenario environment
under an agent and keeps the agent's step records.
"""

from __future__ import annotations

import itertools

import numpy as np

from vbar.agents import agent_generator
from vbar.agents.supervision import Command, SupervisedAgent
from vbar.scenarios import make


class Script:
    """Records every situation and answers `next` once, then CONTINUE.

    `next` may be a command, an order, a value the agent must refuse, or a function building one
    (called inside `decide`, so an error in building it surfaces inside the agent's `act`).
    """

    name = "script"

    def __init__(self):
        self.next = Command.CONTINUE
        self.seen = []
        self.ladder = None

    def reset(self, view, ladder, rng=None):
        self.ladder = ladder
        self.seen = []

    def decide(self, situation):
        self.seen.append(situation)
        answer, self.next = self.next, Command.CONTINUE
        return answer() if callable(answer) else answer


class Always:
    """Gives one command at every consult."""

    name = "always"

    def __init__(self, command):
        self.command = command
        self.seen = []

    def reset(self, view, ladder, rng=None):
        self.seen = []

    def decide(self, situation):
        self.seen.append(situation)
        return self.command


class GoWhenPossible:
    """GO whenever the mask allows it, CONTINUE otherwise."""

    name = "go-when-possible"

    def __init__(self):
        self.seen = []

    def reset(self, view, ladder, rng=None):
        self.seen = []

    def decide(self, situation):
        self.seen.append(situation)
        return Command.GO if situation.mask[Command.GO] else Command.CONTINUE


class Cycle:
    """Gives a fixed cycle of commands, valid or not, one per consult."""

    name = "cycle"
    ORDERS = (
        Command.HOLD,
        Command.GO,
        Command.CONTINUE,
        Command.THRUST_INHIBIT,
        Command.GO,
        Command.RETREAT,
        Command.GO,
    )

    def __init__(self):
        self.seen = []
        self._orders = itertools.cycle(self.ORDERS)

    def reset(self, view, ladder, rng=None):
        self.seen = []
        self._orders = itertools.cycle(self.ORDERS)

    def decide(self, situation):
        self.seen.append(situation)
        return next(self._orders)


class Session:
    """A supervised agent stepped through observations the test chooses.

    The period defaults to one step, so every step consults the supervisor and the situation
    of a step is the latest one it saw.
    """

    def __init__(self, view, *, period_s=None, tracker=None, mode="ladder", supervisor=None):
        self.view = view
        self.supervisor = Script() if supervisor is None else supervisor
        period = view.dt if period_s is None else period_s
        self.agent = SupervisedAgent(self.supervisor, tracker=tracker, period_s=period, mode=mode)
        self.agent.reset(view, None)
        self.t = 0.0
        self.consulted = False
        self.sit = None
        self.action = None

    def step(self, obs, order=Command.CONTINUE, info=None):
        """Act once on `obs` at the next step time; returns the agent's step record."""
        self.supervisor.next = order
        before = len(self.supervisor.seen)
        self.action = self.agent.act(np.array(obs, dtype=float), {"time_s": self.t, **(info or {})})
        self.t += self.view.dt
        grew = len(self.supervisor.seen) - before
        assert grew in (0, 1), "the supervisor was consulted more than once in a step"
        self.consulted = grew == 1
        self.sit = self.supervisor.seen[-1] if self.supervisor.seen else None
        return self.agent.last_step

    def settle(self, obs, info=None):
        """Two steps on the same observation; returns the situation of the second."""
        self.step(obs, info=info)
        self.step(obs, info=info)
        assert self.consulted
        return self.sit


def plain_situation(situation):
    """A situation as comparable plain values."""
    return (
        float(situation.t),
        tuple(np.round(np.asarray(situation.obs, dtype=float), 9)),
        situation.phase,
        situation.rung,
        tuple(bool(m) for m in situation.mask),
        tuple(sorted((int(k), v) for k, v in situation.reasons.items())),
        situation.refused,
        tuple(sorted(dict(situation.info).items())),
    )


def plain_step(step):
    """A supervision step as comparable plain values."""
    return (
        float(step.t),
        step.consulted,
        repr(step.order),
        step.accepted,
        step.reason,
        step.phase,
        step.rung,
        tuple(bool(m) for m in step.mask),
        tuple(np.round(np.asarray(step.r_ref, dtype=float), 9)),
        tuple(np.round(np.asarray(step.v_ref, dtype=float), 9)),
    )


class Flight:
    """One episode of a scenario under an agent, advanced a step at a time."""

    def __init__(self, name, agent, seed, agent_seed, max_steps=None, **config):
        if max_steps is not None:
            config["max_steps"] = max_steps
        self.env = make(name, **config)
        self.agent = agent
        self.obs, self.info = self.env.reset(seed=seed)
        agent.reset(self.env.agent_view(), agent_generator(agent_seed))
        self.done = False
        self.records = []
        self.actions = []
        self.reprs = []
        self.truths = [np.array(self.info["true_state"], dtype=float)]

    def advance(self):
        action = self.agent.act(self.obs, self.env.agent_info())
        self.records.append(plain_step(self.agent.last_step))
        self.actions.append(np.array(action, dtype=float))
        self.reprs.append(repr(self.agent.last_step))
        self.obs, _, terminated, truncated, self.info = self.env.step(action)
        self.truths.append(np.array(self.info["true_state"], dtype=float))
        self.done = terminated or truncated

    def run(self, steps):
        for _ in range(steps):
            if self.done:
                break
            self.advance()
        return self
