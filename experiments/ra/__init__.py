"""Reliability-adequacy (RA) experiment harness.

Phase-1 harness for chapter 3.  A *run* is a resolved config plus the zap
commit; it expands deterministically into *tasks* (one block solve, by one
method, for one (year, draw)).  Tasks write one JSON file each, atomically, so
a run is resumable and aggregation is a pure function of the task files.

Entry point: ``python -m experiments.ra.cli`` (from the zap repository root).
"""

__all__ = [
    "blocks",
    "config",
    "dispatch",
    "evaluate",
    "identity",
    "metrics",
    "paths",
    "runcard",
    "system",
    "tasks",
]
