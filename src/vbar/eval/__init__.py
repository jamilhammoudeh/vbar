"""The evaluation harness: run suites, record results with provenance, and summarise them."""

from .provenance import combined_hash, environment, provenance_fields, provenance_hash
from .results import SCHEMA_VERSION, read_results, write_results
from .runner import Job, run_job, run_jobs, run_suite, suite_jobs
from .task import SpaceTask, Task

__all__ = [
    "SCHEMA_VERSION",
    "Job",
    "SpaceTask",
    "Task",
    "combined_hash",
    "environment",
    "provenance_fields",
    "provenance_hash",
    "read_results",
    "run_job",
    "run_jobs",
    "run_suite",
    "suite_jobs",
    "write_results",
]
