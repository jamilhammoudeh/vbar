"""Fixed, versioned suites: the scenario and seed sets a report must cover in full."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .registry import get, make
from .seeds import SeedSplit, seed_digest


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

    def runs(self, identifiers: Mapping[str, str] | None = None) -> list[tuple[str, int, int]]:
        """(scenario_id, seed, agent_seed) for every episode of the suite.

        Parameters
        ----------
        identifiers : mapping, optional
            Entry scenario to the identifier its runs fly under, where an evaluation builds the
            environment otherwise than `make`; the catalogue identifier by default.

        Returns
        -------
        list of tuple
            One (scenario_id, seed, agent_seed) per episode, entries in order.
        """
        out = []
        for entry in self.entries:
            scenario_id = (identifiers or {}).get(entry.scenario) or entry.scenario_id
            out += [(scenario_id, s, a) for s, a in entry.seeds().pairs()]
        return out

    def missing(
        self, results: Iterable[Mapping[str, Any]], identifiers: Mapping[str, str] | None = None
    ) -> list[tuple[str, int]]:
        """The (scenario_id, seed) runs absent from `results`; empty when a report is complete.

        A result belongs to a run by its "seed", or by its "seed_digest" where the seed is None,
        as in a published file of the test split (`match_runs`).

        Parameters
        ----------
        results : iterable of mapping
            Episode results, each with "scenario_id" and "seed", and "seed_digest" where "seed"
            may be None.
        identifiers : mapping, optional
            As in `runs`.

        Returns
        -------
        list of tuple
            (scenario_id, seed) of every missing run, in suite order.

        Raises
        ------
        KeyError
            If a result lacks "scenario_id" or "seed".
        """
        runs = self.runs(identifiers)
        return [runs[k][:2] for k in self._absent(results, identifiers)]

    def _absent(
        self, results: Iterable[Mapping[str, Any]], identifiers: Mapping[str, str] | None
    ) -> list[int]:
        """Positions in `runs` that no result belongs to."""
        found = set(match_runs(self, results, identifiers))
        return [k for k in range(sum(entry.count for entry in self.entries)) if k not in found]

    def require_complete(
        self, results: Iterable[Mapping[str, Any]], identifiers: Mapping[str, str] | None = None
    ) -> None:
        """Refuse `results` that leave out any run of the suite.

        Parameters
        ----------
        results : iterable of mapping
            Episode results, each with "scenario_id" and "seed", and "seed_digest" where "seed"
            may be None.
        identifiers : mapping, optional
            As in `runs`.

        Raises
        ------
        ValueError
            If a run is missing; the message counts them and names the first, by its seed, or by
            its index in the entry where the entry is on the test split, whose seeds stay secret.
        KeyError
            If a result lacks "scenario_id" or "seed".
        """
        absent = self._absent(results, identifiers)
        if absent:
            scenario_id, seed, _ = self.runs(identifiers)[absent[0]]
            entry, index = _slots(self)[absent[0]]
            first = f"index {index}" if entry.split == "test" else f"seed {seed}"
            total = sum(e.count for e in self.entries)
            raise ValueError(
                f"{self.id} is reported in full or not at all: {len(absent)} of {total} runs "
                f"are missing, the first {scenario_id} {first}"
            )

    def complete(self, results: Iterable[Mapping[str, Any]]) -> bool:
        """Whether `results` cover every run of the suite.

        Parameters
        ----------
        results : iterable of mapping
            Episode results, each with "scenario_id" and "seed", and "seed_digest" where "seed"
            may be None.

        Returns
        -------
        bool
            True when no run is missing.

        Raises
        ------
        KeyError
            If a result lacks "scenario_id" or "seed".
        """
        return not self._absent(results, None)


def _slots(suite) -> list[tuple[Any, int]]:
    """(entry, index in the entry) of every run, in the order of `suite.runs`."""
    return [(entry, k) for entry in suite.entries for k in range(entry.count)]


def match_runs(
    suite, results: Iterable[Mapping[str, Any]], identifiers: Mapping[str, str] | None = None
) -> list[int | None]:
    """The run of the suite each result belongs to.

    A result belongs to the run with its scenario identifier and its "seed", or else with its
    scenario identifier and a "seed_digest" equal to `seed_digest(split, seed, index)` of the
    run, which is how a published result of the test split, whose seed is None, is matched. A
    result that matches no run belongs to none.

    Parameters
    ----------
    suite : Suite
        A `Suite`, or any object with its `entries` (each with `split` and `count`) and `runs`.
    results : iterable of mapping
        Episode results, each with "scenario_id" and "seed", and "seed_digest" where "seed" may
        be None.
    identifiers : mapping, optional
        As in `Suite.runs`.

    Returns
    -------
    list of int or None
        For each result, in order, the position of its run in `suite.runs(identifiers)`, or None.

    Raises
    ------
    KeyError
        If a result lacks "scenario_id" or "seed".
    """
    by_seed: dict[tuple[str, int], int] = {}
    by_digest: dict[tuple[str, str | None], int] = {}
    runs = suite.runs(identifiers)
    for k, ((scenario_id, seed, _), (entry, index)) in enumerate(
        zip(runs, _slots(suite), strict=True)
    ):
        by_seed[(scenario_id, seed)] = k
        try:
            by_digest[(scenario_id, seed_digest(entry.split, seed, index))] = k
        except ValueError:
            # A suite of another domain may name splits that have no key; its results hold seeds.
            continue
    found: list[int | None] = []
    for result in results:
        scenario_id, seed = result["scenario_id"], result["seed"]
        position = None if seed is None else by_seed.get((scenario_id, int(seed)))
        if position is None:
            position = by_digest.get((scenario_id, result.get("seed_digest")))
        found.append(position)
    return found


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


__all__ = ["SUITES", "Suite", "SuiteEntry", "match_runs", "register_suite"]
