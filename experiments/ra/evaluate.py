"""The evaluation-pipeline seam (thin in phase 1).

Evaluating a set of candidate designs is *the same task machinery* with a
``Design`` attached: one task per (design, block, method, draw), the same ledger,
the same aggregation.  Phase 1 evaluates only the as-built design, which is why
the operational benchmark is written as an evaluation of ``asbuilt``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import metrics as metrics_mod
from . import tasks as tasks_mod
from .system import Design

logger = logging.getLogger(__name__)


def evaluate_designs(
    designs: Sequence[Design],
    cfg: dict,
    run_dir: Path,
    *,
    force: bool = False,
    shard: str | None = None,
) -> pd.DataFrame:
    """Score every design on the same blocks and return the aggregated frame."""
    by_id = {d.design_id: d for d in designs}
    if len(by_id) != len(designs):
        raise ValueError("design_id must be unique across designs")

    all_tasks = tasks_mod.enumerate_tasks(cfg, design_ids=tuple(by_id))
    selected = tasks_mod.select_shard(all_tasks, shard)

    tasks_mod.run_tasks(selected, cfg, run_dir, force=force, designs=by_id)

    return metrics_mod.aggregate(run_dir)
