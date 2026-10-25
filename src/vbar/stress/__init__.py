"""Stress testing: searches for the starts and conditions under which a controller fails.

`cross_entropy` is the search kernel; `vbar.safety.audit.falsify_filter` applies it to a safety
filter's declared safe set. `ScriptedNav` and `ScriptedExecution` are errors fixed by a few
numbers in the unit box, scaled into declared error sets, for a search to steer.
"""

from .disturbances import NAV_BLOCKS, ScriptedExecution, ScriptedNav
from .falsify import ELITES, Generation, SearchResult, cross_entropy, root_seed

__all__ = [
    "ELITES",
    "NAV_BLOCKS",
    "Generation",
    "ScriptedExecution",
    "ScriptedNav",
    "SearchResult",
    "cross_entropy",
    "root_seed",
]
