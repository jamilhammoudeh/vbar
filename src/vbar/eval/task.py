"""The task interface the harness runs, and V-BAR's spacecraft task.

The runner, the statistics and the report know a task only through four calls: build an
environment for a scenario name, record an episode, score it, and say what identifies the run.
A domain other than spacecraft rendezvous plugs in by implementing them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class Task(Protocol):
    """What the harness needs from a domain.

    `make` builds a fresh environment for a scenario name, `record` flies one episode with the
    given seeds and returns whatever `score` reads, `score` returns the metrics by name (with a
    boolean "success" and, optionally, a "failure_label"), and `identity` the fields of the run
    the provenance hash covers beyond the seeds, agent and filter.
    """

    @property
    def name(self) -> str: ...

    def make(self, scenario: str) -> Any: ...

    def record(self, env: Any, agent: Any, seed: int, agent_seed: int) -> Any: ...

    def score(self, env: Any, episode: Any) -> Mapping[str, Any]: ...

    def identity(self, env: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SpaceTask:
    """V-BAR's rendezvous and proximity-operations scenarios, scored by `vbar.metrics.evaluate`.

    Parameters
    ----------
    name : str
        The task's name in result files.
    """

    name: str = "space"

    def make(self, scenario: str) -> Any:
        """The catalogue environment of `scenario` (a name with variants or an identifier).

        Parameters
        ----------
        scenario : str
            Scenario name, `name:variant...` or a full identifier.

        Returns
        -------
        ScenarioEnv
            A new environment.
        """
        from ..scenarios import make

        return make(scenario)

    def record(self, env: Any, agent: Any, seed: int, agent_seed: int) -> Any:
        """One scored episode (`vbar.metrics.record_episode`).

        Parameters
        ----------
        env : ScenarioEnv
            The environment.
        agent : Agent
            The agent, a `FilteredAgent` when a filter is on.
        seed, agent_seed : int
            Episode and agent seeds.

        Returns
        -------
        Episode
            The recorded episode.
        """
        from ..metrics import record_episode

        return record_episode(env, agent, seed=seed, agent_seed=agent_seed)

    def score(self, env: Any, episode: Any) -> Mapping[str, Any]:
        """Every metric of the episode with the scenario's declared settings.

        Parameters
        ----------
        env : ScenarioEnv
            The environment the episode was flown in.
        episode : Episode
            The recorded episode.

        Returns
        -------
        dict
            `vbar.metrics.evaluate` of the episode.
        """
        from ..metrics import evaluate

        return evaluate(episode, env.metric_settings())

    def identity(self, env: Any) -> Mapping[str, Any]:
        """The identifier flown and its dynamics model.

        Parameters
        ----------
        env : ScenarioEnv
            The environment.

        Returns
        -------
        dict
            "scenario_id" and "dynamics".
        """
        return {"scenario_id": env.scenario_id, "dynamics": env.cfg.dynamics}


__all__ = ["SpaceTask", "Task"]
