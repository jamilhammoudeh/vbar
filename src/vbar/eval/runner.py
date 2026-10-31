"""Run a suite: every (scenario, seed) episode, in parallel, resumable, each with its provenance.

An episode's result depends only on its scenario, its seeds, the agent and the filter, never on
the number of workers or the order the episodes run in: each episode builds its own environment,
agent and filter from their factories. With an output file the results are appended one by one,
and a rerun skips the runs the file already holds with the same provenance, so an interrupted
suite resumes where it stopped. An episode that raises is recorded with its error and counts as
a failure; it is never dropped or quietly run again.
"""

from __future__ import annotations

import importlib
import multiprocessing
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..safety import FilteredAgent
from ..scenarios.seeds import seed_digest
from .provenance import environment, provenance_fields, provenance_hash
from .results import SCHEMA_VERSION, read_results, write_results
from .stats import _whole
from .task import SpaceTask, Task

Factory = Callable[[], Any] | str


@dataclass(frozen=True)
class Job:
    """One episode to run: the scenario with its split, seed index and seeds.

    Parameters
    ----------
    scenario : str
        Scenario name with variants (what the task's `make` takes).
    split : str
        Seed split the seeds come from.
    index : int
        Index of the seeds in the split.
    seed, agent_seed : int
        Episode and agent seeds.
    suite : str, optional
        The suite identifier the job belongs to.

    Raises
    ------
    ValueError
        If `scenario` is empty or `scenario` or `split` is not text, or `index`, `seed` or
        `agent_seed` is not a whole number of at least 0.
    """

    scenario: str
    split: str
    index: int
    seed: int
    agent_seed: int
    suite: str | None = None

    def __post_init__(self):
        if not isinstance(self.scenario, str) or not self.scenario:
            raise ValueError("scenario must be a name")
        if not isinstance(self.split, str):
            raise ValueError("split must be a name")
        # Python integers, since gymnasium seeds its generator only from an int.
        for name in ("index", "seed", "agent_seed"):
            object.__setattr__(self, name, _whole(name, getattr(self, name), 0))


def resolve(factory: Factory | None) -> Callable[[], Any] | None:
    """A factory given as a callable or as "module:attribute", as a callable.

    Parameters
    ----------
    factory : callable, str or None
        A zero-argument callable (a class such as `PDAgent` is one) or its import path.

    Returns
    -------
    callable or None
        The factory.

    Raises
    ------
    ValueError
        If a string is not of the form "module:attribute".
    """
    if factory is None or callable(factory):
        return factory
    module, _, attribute = factory.partition(":")
    if not module or not attribute:
        raise ValueError(f"a factory path is 'module:attribute', not {factory!r}")
    target: Any = importlib.import_module(module)
    for part in attribute.split("."):
        target = getattr(target, part)
    return target


def suite_jobs(suite) -> list[Job]:
    """Every episode of a suite, entries in order and seeds in index order.

    Parameters
    ----------
    suite : Suite
        A `vbar.scenarios.Suite`.

    Returns
    -------
    list of Job
        One job per run of the suite.
    """
    jobs = []
    for entry in suite.entries:
        for index, (seed, agent_seed) in enumerate(entry.seeds().pairs()):
            jobs.append(Job(entry.scenario, entry.split, index, seed, agent_seed, suite.id))
    return jobs


def _digests(job: Job) -> tuple[str | None, str | None]:
    """Digests of the job's seeds; None where its split has no key to sign them with."""
    try:
        return (
            seed_digest(job.split, job.seed, job.index),
            seed_digest(job.split, job.agent_seed, job.index, "agent"),
        )
    except ValueError:
        return None, None


def _prepare(job: Job, task: Task, agent_factory, filter_factory) -> tuple[Any, Any, Any, str]:
    env = task.make(job.scenario)
    agent = agent_factory()
    safety_filter = None if filter_factory is None else filter_factory()
    fields = provenance_fields(
        task=task.name,
        identity=task.identity(env),
        seed=job.seed,
        agent_seed=job.agent_seed,
        agent=agent,
        safety_filter=safety_filter,
    )
    return env, agent, safety_filter, provenance_hash(fields)


def run_job(job: Job, task: Task, agent: Factory, safety_filter: Factory | None = None) -> dict:
    """Fly and score one episode; an exception is recorded, not raised.

    Parameters
    ----------
    job : Job
        The episode.
    task : Task
        The domain.
    agent : callable or str
        Agent factory.
    safety_filter : callable or str, optional
        Filter factory; the agent is wrapped in a `FilteredAgent` when given.

    Returns
    -------
    dict
        The result, with every field of the result schema. `seed_digest` and `agent_seed_digest`
        are None where the job's split has no key (a custom split name, or the test split without
        the hidden key); `write_results` refuses such a result of the test split.
    """
    agent_factory, filter_factory = resolve(agent), resolve(safety_filter)
    started = time.perf_counter()
    seed_hash, agent_seed_hash = _digests(job)
    result: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "task": task.name,
        "suite": job.suite,
        "scenario": job.scenario,
        "scenario_id": None,
        "split": job.split,
        "index": job.index,
        "seed": job.seed,
        "agent_seed": job.agent_seed,
        "seed_digest": seed_hash,
        "agent_seed_digest": agent_seed_hash,
        "agent": None,
        "filter": None,
        "metrics": {"success": False},
        "failure_label": "error",
        "error": None,
        "provenance": None,
        "environment": environment(),
        "wall_s": 0.0,
    }
    try:
        env, flier, guard, digest = _prepare(job, task, agent_factory, filter_factory)
        result.update(
            scenario_id=task.identity(env).get("scenario_id", job.scenario),
            agent=getattr(flier, "name", type(flier).__name__),
            filter=None if guard is None else getattr(guard, "name", type(guard).__name__),
            provenance=digest,
        )
        pilot = flier if guard is None else FilteredAgent(flier, guard)
        episode = task.record(env, pilot, job.seed, job.agent_seed)
        metrics = dict(task.score(env, episode))
        result.update(metrics=metrics, failure_label=metrics.pop("failure_label", None))
    except Exception as error:  # noqa: BLE001  the run is recorded as failed, never lost
        result["error"] = f"{type(error).__name__}: {error}"
    result["wall_s"] = time.perf_counter() - started
    return result


def _run_job_args(args: tuple) -> dict:
    return run_job(*args)


def run_jobs(
    jobs: Sequence[Job],
    agent: Factory,
    *,
    task: Task | None = None,
    safety_filter: Factory | None = None,
    workers: int = 1,
    out: str | Path | None = None,
) -> list[dict]:
    """Run episodes, in parallel when `workers` > 1, and return their results in job order.

    Parameters
    ----------
    jobs : sequence of Job
        The episodes.
    agent : callable or str
        Agent factory: a zero-argument callable or "module:attribute". With several workers it
        must be importable (a module-level class or function, or a path).
    task : Task, optional
        The domain; `SpaceTask()` by default.
    safety_filter : callable or str, optional
        Filter factory, or None for the filter off.
    workers : int
        Number of processes; 1 runs in this process.
    out : str or Path, optional
        JSON-lines file the results are appended to as they finish; runs it already holds with
        the same provenance are not flown again and their stored results are returned. Results of
        the test split are written without their seeds (`write_results`).

    Returns
    -------
    list of dict
        One result per job, in job order.

    Raises
    ------
    ValueError
        If `workers` is less than 1, or `out` is given and a result of the test split has no
        digests because the hidden key is unavailable, would publish the hidden key or one of its
        seeds in a field other than `error`, or `VBAR_HIDDEN_SEED_LOG` names `out` itself.
    OSError
        If `out` is given, the variable `VBAR_HIDDEN_SEED_LOG` is set and the log it names cannot
        be opened for appending.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    chosen: Task = task or SpaceTask()
    agent_factory, filter_factory = resolve(agent), resolve(safety_filter)
    done: dict[str, dict] = {}
    if out is not None and Path(out).exists():
        done = {r["provenance"]: r for r in read_results(out) if r["provenance"] is not None}
    pending, keys = [], []
    for job in jobs:
        key = _key(job, chosen, agent_factory, filter_factory) if done else None
        keys.append(key)
        if key is None or key not in done:
            pending.append(job)
    fresh: list[dict] = []
    args = [(job, chosen, agent, safety_filter) for job in pending]
    if workers == 1 or len(pending) <= 1:
        for item in args:
            fresh.append(_finish(run_job(*item), out))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            for result in pool.map(_run_job_args, args):
                fresh.append(_finish(result, out))
    flown = iter(fresh)
    return [done[key] if key is not None and key in done else next(flown) for key in keys]


def _key(job: Job, task: Task, agent_factory, filter_factory) -> str | None:
    try:
        return _prepare(job, task, agent_factory, filter_factory)[3]
    except Exception:  # noqa: BLE001  a job that cannot be built is flown, and fails, as itself
        return None


def _finish(result: dict, out: str | Path | None) -> dict:
    if out is not None:
        write_results(out, [result])
    return result


def run_suite(
    suite,
    agent: Factory,
    *,
    task: Task | None = None,
    safety_filter: Factory | None = None,
    workers: int = 1,
    out: str | Path | None = None,
) -> list[dict]:
    """Run every episode of a suite (`run_jobs` on `suite_jobs(suite)`).

    Parameters
    ----------
    suite : Suite
        A `vbar.scenarios.Suite`; its entries' seeds are drawn from their splits.
    agent : callable or str
        Agent factory.
    task : Task, optional
        The domain; `SpaceTask()` by default.
    safety_filter : callable or str, optional
        Filter factory, or None.
    workers : int
        Number of processes.
    out : str or Path, optional
        JSON-lines output file, appended to and resumed from.

    Returns
    -------
    list of dict
        One result per run of the suite, in suite order.
    """
    return run_jobs(
        suite_jobs(suite), agent, task=task, safety_filter=safety_filter, workers=workers, out=out
    )


def by_scenario(results: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    """Results grouped by scenario name, in first-seen order.

    Parameters
    ----------
    results : sequence of mapping
        Results.

    Returns
    -------
    dict
        Scenario name to its results.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        groups.setdefault(result["scenario"], []).append(result)
    return groups


__all__ = ["Job", "by_scenario", "resolve", "run_job", "run_jobs", "run_suite", "suite_jobs"]
