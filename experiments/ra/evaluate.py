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


def designs_from_run(run_dir: Path, system=None) -> list[Design]:
    """Read every ``designs/*.json`` a planning run wrote (the WP5 seam, 3.4).

    ``system`` is optional; when given, each design's recorded row names are
    asserted equal to the system's, so a design built on a different dataset
    fails loudly instead of silently mis-mapping.
    """
    from .planning.design import design_paths, read_design

    designs = [read_design(path, system=system) for path in design_paths(run_dir)]
    logger.info("read %d design(s) from %s", len(designs), Path(run_dir) / "designs")
    return designs


def evaluate_designs(
    designs: Sequence[Design],
    cfg: dict,
    run_dir: Path,
    *,
    force: bool = False,
    shard: str | None = None,
) -> pd.DataFrame:
    """Score every design on the same blocks and return the aggregated frame.

    ``cfg`` is a **dispatch-mode** config: evaluation is block dispatch of a
    design, whatever produced the design.
    """
    by_id = {d.design_id: d for d in designs}
    if len(by_id) != len(designs):
        raise ValueError("design_id must be unique across designs")

    all_tasks = tasks_mod.enumerate_tasks(cfg, design_ids=tuple(by_id))
    selected = tasks_mod.select_shard(all_tasks, shard)

    tasks_mod.run_tasks(selected, cfg, run_dir, force=force, designs=by_id)

    return metrics_mod.aggregate(run_dir)
