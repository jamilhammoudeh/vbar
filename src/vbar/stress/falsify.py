"""Cross-entropy search for the worst point of a box, the kernel of falsification.

A falsification search looks for a start that makes a system fail, by minimizing a robustness
score over the starts it is declared to handle (Kochenderfer, Katz, Corso and Moss, *Algorithms
for Validation*, ch. 4). Finding nothing is not a proof of safety: the search samples, and its
average dispersion (sec. 5.3.2) says how much of the box it left unvisited.

The search is the cross-entropy method (sec. 7.3) on the unit box, with independent normal
proposals per coordinate, a mean that moves toward the elites and a spread that never falls below
a floor: the full-covariance update without smoothing of alg. 7.5 collapses onto the first elites
found. Two or more chains search independently, so a second failure region is not abandoned for
the first, the population remedy of example 4.8.
"""

from __future__ import annotations

import math
import operator
import secrets
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

_SEED_BITS = 128
_MIN_SEED_BITS = 64
# Rejection sampling gives up after this many draws per candidate asked for.
_DRAWS_PER_CANDIDATE = 1000
# Uniform probe points for the dispersion of the evaluated points.
_PROBES = 4096
# The spread of a uniform draw on [0, 1], the spread before the first update.
_UNIFORM_SIGMA = math.sqrt(1.0 / 12.0)
# Elites per chain by default. With 50 candidates, over the test seed family, 6 elites reached the
# narrow well of the two-well problem within 1e-3 in 500 evaluations on 200 seeds of 200 (7 on 196,
# 10 on 185) and found the six-dimensional failure ball in 1000 evaluations on 100 of 100.
ELITES = 6


@dataclass(frozen=True)
class Generation:
    """One generation of one chain.

    Attributes
    ----------
    chain : int
        The chain, from 0.
    mean, sigma : ndarray, shape (d,)
        The proposal after the generation's update, on the unit box.
    best : float
        The lowest score of the generation.
    evaluations : int
        Candidates scored in the generation.
    """

    chain: int
    mean: np.ndarray
    sigma: np.ndarray
    best: float
    evaluations: int


@dataclass(frozen=True)
class SearchResult:
    """The outcome of `cross_entropy`.

    Attributes
    ----------
    x : ndarray, shape (d,)
        The point with the lowest score, in the box's units.
    score : float
        Its score.
    points : ndarray, shape (n, d)
        Every evaluated point, in the box's units, in evaluation order.
    scores : ndarray, shape (n,)
        Their scores.
    generations : tuple of Generation
        The history, chain by chain within each generation.
    evaluations : int
        Points scored, at most the budget.
    rejected : int
        Draws rejected for leaving the box or failing `accept`.
    dispersion : float
        Mean distance from 4096 uniform probe points of the unit box to the nearest evaluated
        point, in unit-box coordinates; smaller means the box was covered more evenly.
    seed : int
        The 128-bit root seed every stream was spawned from.
    """

    x: np.ndarray
    score: float
    points: np.ndarray
    scores: np.ndarray
    generations: tuple[Generation, ...]
    evaluations: int
    rejected: int
    dispersion: float
    seed: int


def root_seed(seed: int | None) -> int:
    """Return `seed`, or a drawn 128-bit seed when it is None.

    Raises
    ------
    ValueError
        If `seed` lies outside [2**64, 2**128): a smaller seed can be found by trying candidates.
    """
    if seed is None:
        return secrets.randbits(_SEED_BITS)
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    value = operator.index(seed)
    if not 1 << _MIN_SEED_BITS <= value < 1 << _SEED_BITS:
        raise ValueError("seed must lie in [2**64, 2**128)")
    return value


def _count(value, name: str, least: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    value = operator.index(value)
    if value < least:
        raise ValueError(f"{name} must be at least {least}")
    return value


def _draw(
    rng: np.random.Generator,
    size: int,
    mean: np.ndarray | None,
    sigma: np.ndarray,
    to_box: Callable[[np.ndarray], np.ndarray],
    accept: Callable[[np.ndarray], bool] | None,
) -> tuple[np.ndarray, int]:
    """`size` unit-box points, uniform when `mean` is None; the draws rejected."""
    d = len(sigma)
    kept: list[np.ndarray] = []
    rejected = 0
    limit = _DRAWS_PER_CANDIDATE * size
    while len(kept) < size:
        if rejected == limit:
            raise ValueError(f"found {len(kept)} of {size} acceptable points in {limit} draws")
        z = rng.uniform(0.0, 1.0, d) if mean is None else rng.normal(mean, sigma)
        if np.all((z >= 0.0) & (z <= 1.0)) and (accept is None or bool(accept(to_box(z)))):
            kept.append(z)
        else:
            rejected += 1
    return np.array(kept), rejected


def cross_entropy(
    score: Callable[[np.ndarray], float],
    lo,
    hi,
    *,
    accept: Callable[[np.ndarray], bool] | None = None,
    budget: int = 1000,
    population: int = 50,
    elites: int = ELITES,
    smoothing: float = 0.7,
    sigma_floor: float = 0.02,
    chains: int = 2,
    seed: int | None = None,
    workers: int = 1,
) -> SearchResult:
    """Minimize `score` over the points of the box [lo, hi] that `accept` passes.

    Parameters
    ----------
    score : callable
        score(x) -> float for a point x of the box; lower is closer to failure. It must pickle
        when `workers` is above 1.
    lo, hi : array_like, shape (d,)
        The box corners, finite, lo < hi.
    accept : callable, optional
        accept(x) -> bool; points it rejects are never scored. Called in this process.
    budget : int
        Points to score in all, at least 1.
    population : int
        Candidates per chain and generation, at least 2.
    elites : int
        Lowest-scoring candidates the update follows, from 1 to `population`.
    smoothing : float
        Weight a in (0, 1] of the new statistics: mean <- (1 - a) mean + a mean(elites),
        sigma <- max(floor, (1 - a) sigma + a std(elites)), per coordinate of the unit box.
    sigma_floor : float
        Smallest spread, unit-box coordinates, in (0, 1].
    chains : int
        Independent searches sharing the budget, at least 1.
    seed : int, optional
        Root seed in [2**64, 2**128); drawn (128 bits) when not given. Each chain's stream and
        the probe points are spawned from it.
    workers : int
        Processes the scores are computed in. All draws are made here and results are kept in
        order, so the result is the same for every value.

    Returns
    -------
    SearchResult
        The best point and score, every evaluation, the history, the rejections and the
        dispersion.

    Raises
    ------
    ValueError
        If a parameter is out of range, a score is NaN, or `accept` passes too few draws.

    Notes
    -----
    The first generation of each chain is uniform on the box, and its elites' mean and spread
    (floored) are the first proposal; later updates are smoothed. A chain's elites are the
    lowest-scoring points among its new generation and its previous elites. Draws outside the
    box or rejected by `accept` are drawn again, up to 1000 times per candidate, and counted.
    Generations run until the budget is spent; the last one may be smaller.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.ndim != 1 or lo.shape != hi.shape or lo.size == 0:
        raise ValueError("lo and hi must be vectors of the same length")
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(lo < hi)):
        raise ValueError("box corners must be finite with lo < hi")
    budget = _count(budget, "budget", 1)
    population = _count(population, "population", 2)
    elites = _count(elites, "elites", 1)
    if elites > population:
        raise ValueError("elites must not exceed population")
    chains = _count(chains, "chains", 1)
    workers = _count(workers, "workers", 1)
    smoothing = float(smoothing)
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must lie in (0, 1]")
    sigma_floor = float(sigma_floor)
    if not 0.0 < sigma_floor <= 1.0:
        raise ValueError("sigma_floor must lie in (0, 1]")
    root = root_seed(seed)
    d = lo.size
    width = hi - lo

    def to_box(z: np.ndarray) -> np.ndarray:
        return lo + z * width

    *chain_seqs, probe_seq = np.random.SeedSequence(root).spawn(chains + 1)
    rngs = [np.random.default_rng(seq) for seq in chain_seqs]
    means: list[np.ndarray | None] = [None] * chains
    sigmas = [np.full(d, _UNIFORM_SIGMA) for _ in range(chains)]
    kept_z = [np.empty((0, d)) for _ in range(chains)]
    kept_scores = [np.empty(0) for _ in range(chains)]
    points: list[np.ndarray] = []
    scores: list[float] = []
    history: list[Generation] = []
    rejected = 0
    pool = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        while len(scores) < budget:
            for c in range(chains):
                size = min(population, budget - len(scores))
                if size == 0:
                    break
                z, missed = _draw(rngs[c], size, means[c], sigmas[c], to_box, accept)
                rejected += missed
                xs = [to_box(row) for row in z]
                found = list(pool.map(score, xs)) if pool else [score(x) for x in xs]
                values = np.array([float(v) for v in found])
                if np.any(np.isnan(values)):
                    raise ValueError("score returned NaN")
                points.extend(xs)
                scores.extend(values.tolist())
                # The elites are the best of this generation and the chain's previous elites, so
                # a narrow well that gave only a few draws is not dropped for a broad one.
                pooled_z = np.vstack([kept_z[c], z])
                pooled_scores = np.concatenate([kept_scores[c], values])
                best = np.argsort(pooled_scores, kind="stable")[:elites]
                elite = pooled_z[best]
                kept_z[c], kept_scores[c] = elite, pooled_scores[best]
                current = means[c]
                # The uniform first generation has no proposal to smooth toward: its elites set
                # the first one.
                a = 1.0 if current is None else smoothing
                previous = elite.mean(axis=0) if current is None else current
                mean = (1.0 - a) * previous + a * elite.mean(axis=0)
                spread = elite.std(axis=0) if len(elite) > 1 else np.zeros(d)
                sigma = np.maximum(sigma_floor, (1.0 - a) * sigmas[c] + a * spread)
                means[c], sigmas[c] = mean, sigma
                history.append(Generation(c, mean.copy(), sigma.copy(), float(values.min()), size))
    finally:
        if pool is not None:
            pool.shutdown()
    all_points = np.array(points)
    all_scores = np.array(scores)
    k = int(np.argmin(all_scores))
    probes = np.random.default_rng(probe_seq).uniform(0.0, 1.0, (_PROBES, d))
    unit = (all_points - lo) / width
    nearest = np.full(_PROBES, np.inf)
    for row in unit:
        nearest = np.minimum(nearest, np.linalg.norm(probes - row, axis=1))
    return SearchResult(
        x=all_points[k].copy(),
        score=float(all_scores[k]),
        points=all_points,
        scores=all_scores,
        generations=tuple(history),
        evaluations=len(all_scores),
        rejected=rejected,
        dispersion=float(nearest.mean()),
        seed=root,
    )


__all__ = ["ELITES", "Generation", "SearchResult", "cross_entropy", "root_seed"]
