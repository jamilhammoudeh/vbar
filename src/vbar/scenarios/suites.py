"""Fixed, versioned suites: the scenario and seed sets a report must cover in full."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .registry import get, make
from .seeds import SeedSplit


@dataclass(frozen=True)
class SuiteEntry:
    """One scenario of a suite: its name (with variants), split and number of seeds.

    Parameters
    ----------
    scenario : str
        Scenario name, with any `:variant` suffixes.
    split : {"train", "validation", "test"}
        The seed split.
    count : int
        Number of seeds.
    """

    scenario: str
    split: str = "validation"
    count: int = 50

    def seeds(self) -> SeedSplit:
        """The entry's seeds.

        Returns
        -------
        SeedSplit
            `count` seeds of `scenario` in `split`.
        """
        return SeedSplit(self.scenario, self.split, self.count)

    @property
    def scenario_id(self) -> str:
        """The identifier of the environment the entry builds."""
        return make(self.scenario).scenario_id


@dataclass(frozen=True)
class Suite:
    """A named, versioned list of entries; results are complete only when every run is in.

    Parameters
    ----------
    name : str
        Suite name.
    version : int
        Suite version.
    entries : tuple of SuiteEntry
        The scenarios, each listed once.

    Raises
    ------
    KeyError
        If an entry names an unknown scenario or variant.
    ValueError
        If a scenario is listed twice.
    """

    name: str
    version: int
    entries: tuple[SuiteEntry, ...]

    def __post_init__(self):
        for entry in self.entries:
            get(entry.scenario)
        names = [entry.scenario for entry in self.entries]
        if len(set(names)) != len(names):
            raise ValueError("a suite lists each scenario once")

    @property
    def id(self) -> str:
        """The suite identifier, `<name>-v<version>`."""
        return f"{self.name}-v{self.version}"

    def runs(self) -> list[tuple[str, int, int]]:
        """(scenario_id, seed, agent_seed) for every episode of the suite.

        Returns
        -------
        list of tuple
            One (scenario_id, seed, agent_seed) per episode, entries in order.
        """
        out = []
        for entry in self.entries:
            scenario_id = entry.scenario_id
            out += [(scenario_id, s, a) for s, a in entry.seeds().pairs()]
        return out

    def missing(self, results: Iterable[Mapping[str, Any]]) -> list[tuple[str, int]]:
        """The (scenario_id, seed) runs absent from `results`; empty when a report is complete.

        Parameters
        ----------
        results : iterable of mapping
            Episode results, each with "scenario_id" and "seed".

        Returns
        -------
        list of tuple
            (scenario_id, seed) of every missing run, in suite order.

        Raises
        ------
        KeyError
            If a result lacks "scenario_id" or "seed".
        """
        seen = {(r["scenario_id"], int(r["seed"])) for r in results}
        return [(sid, seed) for sid, seed, _ in self.runs() if (sid, seed) not in seen]

    def require_complete(self, results: Iterable[Mapping[str, Any]]) -> None:
        """Refuse `results` that leave out any run of the suite.

        Parameters
        ----------
        results : iterable of mapping
            Episode results, each with "scenario_id" and "seed".

        Raises
        ------
        ValueError
            If a run is missing; the message counts them and names the first.
        KeyError
            If a result lacks "scenario_id" or "seed".
        """
        missing = self.missing(results)
        if missing:
            total = sum(entry.count for entry in self.entries)
            scenario_id, seed = missing[0]
            raise ValueError(
                f"{self.id} is reported in full or not at all: {len(missing)} of {total} runs "
                f"are missing, the first {scenario_id} seed {seed}"
            )

    def complete(self, results: Iterable[Mapping[str, Any]]) -> bool:
        """Whether `results` cover every run of the suite.

        Parameters
        ----------
        results : iterable of mapping
            Episode results, each with "scenario_id" and "seed".

        Returns
        -------
        bool
            True when no run is missing.
        """
        return not self.missing(results)


SUITES: dict[str, Suite] = {}


def register_suite(suite: Suite) -> Suite:
    """Add a suite to `SUITES`; an identifier already present must be the same suite.

    Parameters
    ----------
    suite : Suite
        The suite to register.

    Returns
    -------
    Suite
        `suite`.

    Raises
    ------
    ValueError
        If another suite with different entries holds the identifier.
    """
    if suite.id in SUITES and SUITES[suite.id] != suite:
        raise ValueError(f"suite {suite.id!r} is already registered with different entries")
    SUITES[suite.id] = suite
    return suite


__all__ = ["SUITES", "Suite", "SuiteEntry", "register_suite"]
