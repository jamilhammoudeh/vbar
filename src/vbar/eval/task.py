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

    def make(self, scenario: str) -> Any:
        """A fresh environment for a scenario name.

        Parameters
        ----------
        scenario : str
            Scenario name as a suite entry gives it.

        Returns
        -------
        Any
            An environment that `record`, `score` and `identity` accept.
        """

    def record(self, env: Any, agent: Any, seed: int, agent_seed: int) -> Any:
        """Fly one episode and return what `score` reads.

        Parameters
        ----------
        env : Any
            The environment `make` built.
        agent : Any
            The agent, wrapped by its filter when one is on.
        seed, agent_seed : int
            Episode and agent seeds.

        Returns
        -------
        Any
            The recorded episode.
        """

    def score(self, env: Any, episode: Any) -> Mapping[str, Any]:
        """The metrics of an episode by name.

        Parameters
        ----------
        env : Any
            The environment the episode was flown in.
        episode : Any
            What `record` returned.

        Returns
        -------
        mapping
            Metric name to value, with a boolean "success" and optionally a "failure_label".
        """

    def identity(self, env: Any) -> Mapping[str, Any]:
        """The fields of the run that the provenance hash covers beyond seeds, agent and filter.

        Parameters
        ----------
        env : Any
            The environment `make` built.

        Returns
        -------
        mapping
            JSON-safe name to value; "scenario_id" names the run in the results when present.
        """


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
