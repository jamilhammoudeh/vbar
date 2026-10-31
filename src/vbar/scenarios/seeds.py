"""Seed splits: train, validation and hidden test seeds derived from a key, scenario and index.

Every seed is 128 bits, so it cannot be guessed, but the train and validation keys are public
and fixed here, so those seeds can be listed: an agent can find its episode among them by its
first observation, or its index by hashing the public key with the agent label. Scored
comparisons use the test split, whose key is never in this repository and comes from the
environment variable `VBAR_HIDDEN_SEED_KEY` (the hex string the benchmark's maintainers keep).
The same scenario and index give the same seed on every platform and release.

A test seed that is published lets anyone rebuild that episode's start state, so a result file
holds `seed_digest` of each test seed in its place: a keyed hash that only the holder of the key
can check.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral

SPLITS = ("train", "validation", "test")
HIDDEN_KEY_VARIABLE = "VBAR_HIDDEN_SEED_KEY"
HIDDEN_LOG_VARIABLE = "VBAR_HIDDEN_SEED_LOG"
"""Names the private file that result writing appends the test split's seeds to.

The file belongs next to the hidden key, in the maintainers' private storage, never beside the
published results.
"""
_PUBLIC_KEYS = {"train": "vbar train seeds 1", "validation": "vbar validation seeds 1"}


def split_key(split: str) -> str:
    """The key material of a split; the test split needs the hidden key.

    Parameters
    ----------
    split : {"train", "validation", "test"}
        The split.

    Returns
    -------
    str
        The key text seeds are derived from.

    Raises
    ------
    ValueError
        If `split` is unknown, or it is "test" and the hidden key variable does not hold at
        least 32 hex digits. Surrounding white space and the case of the digits are ignored, so
        a key read from a file with its newline gives the same split as the bare key.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    if split != "test":
        return _PUBLIC_KEYS[split]
    key = os.environ.get(HIDDEN_KEY_VARIABLE, "").strip().lower()
    if len(key) < 32 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError(
            f"the test split needs the hidden key in {HIDDEN_KEY_VARIABLE} (at least 32 hex digits)"
        )
    return "vbar test seeds " + key


def derive_seed(key: str, scenario: str, index: int, stream: str = "episode") -> int:
    """A 128-bit seed from a key, a scenario name, an index and a stream label.

    Parameters
    ----------
    key : str
        The split's key material (`split_key`).
    scenario : str
        Scenario name, with any variant suffixes.
    index : int
        Episode index, not negative.
    stream : str
        Stream label: "episode" for the environment, "agent" for the agent.

    Returns
    -------
    int
        The first 16 bytes of a SHA-256 digest, big-endian.

    Raises
    ------
    ValueError
        If `index` is negative.
    """
    if index < 0:
        raise ValueError("index must not be negative")
    material = f"{key}|{scenario}|{stream}|{index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:16])


def seed_digest(split: str, seed: int, index: int, stream: str = "episode") -> str:
    """A keyed digest of a seed, the only form of a test seed a published file may hold.

    The HMAC-SHA256 of the stream label, the index and the seed under the split's key
    (`split_key`). Whoever holds the key recomputes it for a seed and so checks a published
    file; without the key a guessed seed cannot be tested against it, as it could against a plain
    hash.

    Parameters
    ----------
    split : {"train", "validation", "test"}
        The split whose key signs the digest.
    seed : int
        The seed to digest, not negative.
    index : int
        Episode index of the seed in its split, not negative.
    stream : str
        Stream label: "episode" for an episode seed, "agent" for an agent seed.

    Returns
    -------
    str
        64 lowercase hex digits.

    Raises
    ------
    ValueError
        If `split` is unknown, it is "test" and the hidden key is unavailable (`split_key`), or
        `seed` or `index` is not a whole number of at least 0.
    """
    for name, value in (("seed", seed), ("index", index)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a whole number of at least 0")
    message = f"vbar seed digest|{stream}|{int(index)}|{int(seed)}".encode()
    return hmac.digest(split_key(split).encode(), message, "sha256").hex()


@dataclass(frozen=True)
class SeedSplit:
    """The seeds of one scenario in one split: episode seeds and matching agent seeds.

    Iterating yields the `count` episode seeds in index order.

    Parameters
    ----------
    scenario : str
        Scenario name, with any variant suffixes.
    split : {"train", "validation", "test"}
        The split.
    count : int
        Number of episodes, at least 1.

    Raises
    ------
    ValueError
        If `count` is below 1 or the split's key is unavailable (`split_key`).
    """

    scenario: str
    split: str
    count: int

    def __post_init__(self):
        if self.count < 1:
            raise ValueError("count must be at least 1")
        split_key(self.split)

    def seed(self, index: int) -> int:
        """The episode seed of episode `index`.

        Parameters
        ----------
        index : int
            Episode index, not negative.

        Returns
        -------
        int
            A 128-bit seed.
        """
        return derive_seed(split_key(self.split), self.scenario, index)

    def agent_seed(self, index: int) -> int:
        """The agent stream's seed for episode `index`; never equal to the episode seed.

        Parameters
        ----------
        index : int
            Episode index, not negative.

        Returns
        -------
        int
            A 128-bit seed.
        """
        return derive_seed(split_key(self.split), self.scenario, index, "agent")

    def __iter__(self) -> Iterator[int]:
        return (self.seed(k) for k in range(self.count))

    def pairs(self) -> list[tuple[int, int]]:
        """(seed, agent_seed) for every episode of the split.

        Returns
        -------
        list of tuple of int
            One (seed, agent_seed) pair per episode, in index order.
        """
        return [(self.seed(k), self.agent_seed(k)) for k in range(self.count)]


def seeds(scenario: str, split: str = "train", count: int = 100) -> SeedSplit:
    """The first `count` seeds of `scenario` in `split`.

    Parameters
    ----------
    scenario : str
        Scenario name, with any variant suffixes.
    split : {"train", "validation", "test"}
        The split.
    count : int
        Number of episodes, at least 1.

    Returns
    -------
    SeedSplit
        The split's seeds.

    Raises
    ------
    ValueError
        If `count` is below 1 or the split's key is unavailable.
    """
    return SeedSplit(scenario, split, count)


__all__ = [
    "HIDDEN_KEY_VARIABLE",
    "HIDDEN_LOG_VARIABLE",
    "SPLITS",
    "SeedSplit",
    "derive_seed",
    "seed_digest",
    "seeds",
    "split_key",
]
