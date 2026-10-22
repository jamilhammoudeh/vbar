"""Stress testing: searches for the starts and conditions under which a controller fails.

`cross_entropy` is the search kernel; `vbar.safety.audit.falsify_filter` applies it to a safety
filter's declared safe set.
"""

from .falsify import ELITES, Generation, SearchResult, cross_entropy, root_seed

__all__ = ["ELITES", "Generation", "SearchResult", "cross_entropy", "root_seed"]
