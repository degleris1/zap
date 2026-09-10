"""The three iteration tables of a planning run (spec section 3.4, D6).

``designs/<id>.history.json`` is the raw trajectory; it is a JSON document with
one list per tracker and is awkward to plot from.  This module normalises it
into three parquet tables under ``runs/<id>/iterations/``:

``iterations.parquet``
    one row per iteration -- the scalars (objective, gradient norms, step size,
    clip fraction, wall clock, carbon multiplier);
``iteration_blocks.parquet``
    one row per (iteration, sampled block) -- which subproblems the minibatch
    saw (P9's input).  Its ``day_of_year`` is derived from the block's absolute
    in-year start hour (``hour // 24 + 1``), which is a **UTC** hour for the
    CA2040 exports; the shipped window starts at hour 7 so blocks begin at
    Pacific midnight, so a block's ``day_of_year`` is the UTC day its first
    hour falls in, not the local one;
``iteration_capacity.parquet``
    one row per (iteration, parameter, carrier) -- the capacity trajectory
    (P3's input), written only when ``optimizer.save_param_history`` is on.

History lists have ``num_iterations + 1`` entries: index 0 is the state *before*
the first step.  That convention is kept here.

A method with no history (the single-level presets) writes **no** files at all;
a loader must treat that as "no trajectory", not as an error.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from ..persist import write_parquet_atomic

logger = logging.getLogger(__name__)

ITERATIONS_COLUMNS = (
    "task_id",
    "design_id",
    "run_id",
    "method",
    "preset",
    "draw",
    "outer_iteration",
    "iteration",
    "sampled_objective_raw",
    "rolling_objective_raw",
    "n_batch",
    "batch_hours",
    "sampled_objective_annual",
    "estimated_full_objective_annual",
    "grad_norm_l1",
    "grad_norm_l2",
    "proj_grad_norm_l1",
    "step_size",
    "clip",
    "clip_fraction",
    "step_norm_mw",
    "suboptimality",
    "time_s",
    "lambda_carbon",
    "annualization_factor",
)

ITERATION_BLOCKS_COLUMNS = (
    "task_id",
    "outer_iteration",
    "iteration",
    "subproblem_id",
    "block_start",
    "block_stop",
    "year",
    "day_of_year",
    "hours",
    "weight",
)

ITERATION_CAPACITY_COLUMNS = (
    "task_id",
    "outer_iteration",
    "iteration",
    "param",
    "device_class",
    "carrier",
    "capacity_mw",
)

_STR = pa.dictionary(pa.int32(), pa.string())


def iterations_schema() -> pa.Schema:
    fields = [
        pa.field("task_id", _STR),
        pa.field("design_id", _STR),
        pa.field("run_id", _STR),
        pa.field("method", _STR),
        pa.field("preset", _STR),
        pa.field("draw", pa.int32(), nullable=True),
        pa.field("outer_iteration", pa.int32()),
        pa.field("iteration", pa.int32()),
        pa.field("n_batch", pa.int32()),
        pa.field("batch_hours", pa.int32()),
    ]
    floats = [c for c in ITERATIONS_COLUMNS if c not in {f.name for f in fields}]
    by_name = {f.name: f for f in fields}
    by_name.update({c: pa.field(c, pa.float64()) for c in floats})
    return pa.schema([by_name[c] for c in ITERATIONS_COLUMNS])


def iteration_blocks_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("task_id", _STR),
            pa.field("outer_iteration", pa.int32()),
            pa.field("iteration", pa.int32()),
            pa.field("subproblem_id", pa.int32()),
            pa.field("block_start", pa.int32()),
            pa.field("block_stop", pa.int32()),
            pa.field("year", pa.int16()),
            pa.field("day_of_year", pa.int16()),
            pa.field("hours", pa.int32()),
            pa.field("weight", pa.float64()),
        ]
    )


def iteration_capacity_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("task_id", _STR),
            pa.field("outer_iteration", pa.int32()),
            pa.field("iteration", pa.int32()),
            pa.field("param", _STR),
            pa.field("device_class", _STR),
            pa.field("carrier", _STR),
            pa.field("capacity_mw", pa.float64()),
        ]
    )


TABLES = {
    "iterations": (ITERATIONS_COLUMNS, iterations_schema),
    "iteration_blocks": (ITERATION_BLOCKS_COLUMNS, iteration_blocks_schema),
    "iteration_capacity": (ITERATION_CAPACITY_COLUMNS, iteration_capacity_schema),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(history: dict, key: str, n: int) -> list[float]:
    values = list(history.get(key) or [])
    out = []
    for i in range(n):
        if i >= len(values) or values[i] is None:
            out.append(float("nan"))
            continue
        value = values[i]
        if hasattr(value, "item") and np.size(value) == 1:
            value = value.item()
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def outer_histories(result) -> list[tuple[int, dict]]:
    """``[(outer_iteration, history)]``: one entry unless dual ascent ran."""
    dual = (result.emissions or {}).get("dual_ascent") or {}
    histories = dual.get("outer_histories")
    if histories:
        return [(i, dict(h)) for i, h in enumerate(histories)]
    if result.history:
        return [(0, dict(result.history))]
    return []


def _lambda_for_outer(result, outer: int) -> float:
    emissions = result.emissions or {}
    dual = emissions.get("dual_ascent") or {}
    lambdas = dual.get("lambda_history") or []
    if outer < len(lambdas):
        return float(lambdas[outer])
    price = emissions.get("price")
    if price is None:
        price = ((result.meta or {}).get("emissions") or {}).get("price")
    return float(price or 0.0)


def _device_class_for_param(result, param: str) -> str:
    """``generator_capacity -> Generator`` via the recorded capacity blocks."""
    prefix = param.rsplit("_", 1)[0]
    for cls_name in result.capacities or {}:
        if str(cls_name).lower() == prefix:
            return str(cls_name)
    return prefix


def _block_year(result, start: int) -> tuple[int, int]:
    """``(weather year, in-year hour)`` of a loaded-window index."""
    years = [int(y) for y in (result.years or [])] or [0]
    total = int((result.annualization or {}).get("total_hours") or 0)
    per_year = total // len(years) if total and len(years) else 0
    window_start = int((result.window or {}).get("start") or 0)
    if per_year <= 0:
        return years[0], window_start + int(start)
    idx = min(int(start) // per_year, len(years) - 1)
    return years[idx], window_start + int(start) % per_year


def _param_snapshot(snapshot) -> dict[str, np.ndarray]:
    return {
        str(k): np.asarray(v, dtype=np.float64).reshape(-1) for k, v in (snapshot or {}).items()
    }


class _CarrierShim:
    """The minimum ``metrics.capacity_by_carrier`` needs, for one parameter."""

    def __init__(self, labels: dict, param: str, values: np.ndarray):
        self.meta = {"carrier_labels": labels}
        self.parameters = {param: values}


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------


def build_iteration_tables(result, *, task=None) -> dict[str, pd.DataFrame]:
    """The three normalised tables, as DataFrames with their declared columns."""
    from .. import metrics as metrics_mod

    task_id = str(getattr(task, "task_id", None) or result.design_id)
    draw = getattr(task, "draw", None)
    meta = result.meta or {}
    optimizer = dict(meta.get("optimizer") or {})
    step_size = float(optimizer.get("step_size", float("nan")))
    clip = float(optimizer.get("clip", float("nan")))
    save_params = bool(optimizer.get("save_param_history", True))
    labels = meta.get("carrier_labels") or {}
    blocks = [(int(a), int(b)) for a, b in ((result.selection or {}).get("blocks") or [])]
    annualization = result.annualization or {}
    af = float(annualization.get("annualization_factor") or float("nan"))

    # `design_selection: best_checkpointed` evaluates the objective over the
    # WHOLE block set at selected iterates; those are the only full-horizon
    # numbers a minibatch run has, so they land in the column reserved for them.
    checkpoints = {
        (int(c.get("outer_iteration") or 0), int(c["iteration"])): c
        for c in ((getattr(result, "objective", None) or {}).get("checkpoints") or [])
        if c.get("iteration") is not None
    }

    scalar_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    capacity_rows: list[dict[str, Any]] = []

    for outer, history in outer_histories(result):
        n = len(history.get("loss") or [])
        if n == 0:
            continue
        loss = _series(history, "loss", n)
        rolling = _series(history, "rolling_loss", n)
        grad_l1 = _series(history, "grad_norm", n)
        grad_l2 = _series(history, "grad_norm_l2", n)
        proj = _series(history, "proj_grad_norm", n)
        subopt = _series(history, "suboptimality", n)
        times = _series(history, "time", n)
        batches = list(history.get("batch") or [])
        params = list(history.get("param") or [])
        lam = _lambda_for_outer(result, outer)

        for i in range(n):
            batch = [int(b) for b in (batches[i] if i < len(batches) else [])]
            batch_hours = sum(blocks[b][1] - blocks[b][0] for b in batch if 0 <= b < len(blocks))
            # `StochasticPlanningProblem._get_batch_weights` already rescales a
            # minibatch by `total_weight / total_batch_weight`, so `history["loss"]`
            # is in the units of the WHOLE block set (`sampled_hours`), not of the
            # batch.  Annualising it is therefore the same `annualization_factor`
            # (= total_hours / sampled_hours) every other annual number uses;
            # dividing by `batch_hours` as well would double-scale whenever
            # `batch_size < num_subproblems`.
            annual = loss[i] * af if math.isfinite(af) else float("nan")
            g2 = grad_l2[i]
            if not (math.isfinite(g2) and math.isfinite(clip)):
                # A gradient we could not measure says nothing about the clip.
                clip_fraction = float("nan")
                step_norm = float("nan")
            elif g2 == 0.0:
                clip_fraction = 1.0  # a zero gradient is never clipped
                step_norm = 0.0
            else:
                clip_fraction = min(1.0, clip / g2)
                step_norm = step_size * min(g2, clip)
            scalar_rows.append(
                {
                    "task_id": task_id,
                    "design_id": str(result.design_id),
                    "run_id": str(result.run_id or ""),
                    "method": str(result.method),
                    "preset": str(result.preset),
                    "draw": draw,
                    "outer_iteration": outer,
                    "iteration": i,
                    "sampled_objective_raw": loss[i],
                    "rolling_objective_raw": rolling[i],
                    "n_batch": len(batch),
                    "batch_hours": int(batch_hours),
                    "sampled_objective_annual": annual,
                    # A full-block-set forward pass at this iterate, when one
                    # was taken (`optimizer.checkpoint_every`); NaN otherwise.
                    # P4's second series.
                    "estimated_full_objective_annual": float(
                        checkpoints[(outer, i)].get("objective_annual", float("nan"))
                    )
                    if (outer, i) in checkpoints
                    else float("nan"),
                    "grad_norm_l1": grad_l1[i],
                    "grad_norm_l2": g2,
                    "proj_grad_norm_l1": proj[i],
                    "step_size": step_size,
                    "clip": clip,
                    "clip_fraction": clip_fraction,
                    "step_norm_mw": step_norm,
                    "suboptimality": subopt[i],
                    "time_s": times[i],
                    "lambda_carbon": lam,
                    "annualization_factor": af,
                }
            )

            for sub in batch:
                if not (0 <= sub < len(blocks)):
                    continue
                start, stop = blocks[sub]
                year, in_year = _block_year(result, start)
                block_rows.append(
                    {
                        "task_id": task_id,
                        "outer_iteration": outer,
                        "iteration": i,
                        "subproblem_id": sub,
                        "block_start": start,
                        "block_stop": stop,
                        "year": year,
                        "day_of_year": in_year // 24 + 1,
                        "hours": stop - start,
                        # 1.0 today (D-W8 forbids non-uniform weights); the
                        # column exists so weighting research is a data change.
                        "weight": 1.0,
                    }
                )

            if save_params and i < len(params):
                for param, values in _param_snapshot(params[i]).items():
                    shim = _CarrierShim(labels, param, values)
                    for carrier, value in metrics_mod.capacity_by_carrier(shim).items():
                        capacity_rows.append(
                            {
                                "task_id": task_id,
                                "outer_iteration": outer,
                                "iteration": i,
                                "param": param,
                                "device_class": _device_class_for_param(result, param),
                                "carrier": str(carrier),
                                "capacity_mw": float(value),
                            }
                        )

    return {
        "iterations": _frame(scalar_rows, ITERATIONS_COLUMNS),
        "iteration_blocks": _frame(block_rows, ITERATION_BLOCKS_COLUMNS),
        "iteration_capacity": _frame(capacity_rows, ITERATION_CAPACITY_COLUMNS),
    }


def _frame(rows: Sequence[dict], columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=list(columns))
    if "draw" in frame.columns:
        frame["draw"] = pd.array(frame["draw"].tolist(), dtype="Int32")
    return frame


def table_paths(run_dir, name: str) -> list[Path]:
    """Every per-task file of one iteration table, plus the legacy single file.

    Files are written **per task** (``iterations/<task_id>.<name>.parquet``) so
    that two planning tasks -- two outage draws, or two SLURM shards sharing a
    run directory -- can never lose each other's rows to a read-modify-write
    race.  The pre-change whole-run file ``iterations/<name>.parquet`` is read
    only when *no* per-task file exists, so an old run directory keeps working
    and a re-run that leaves both behind is not counted twice.
    """
    directory = Path(run_dir) / "iterations"
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob(f"*.{name}.parquet"))
    if paths:
        # A run directory can hold both after a re-run (`--force`, or a retried
        # task): the legacy file is the *old* whole-run table and its rows are a
        # superset of some of the per-task ones, so reading both double-counts.
        # Per-task files are authoritative whenever any exists.
        return paths
    legacy = directory / f"{name}.parquet"
    return [legacy] if legacy.exists() else []


def has_table(run_dir, name: str) -> bool:
    return bool(table_paths(run_dir, name))


def read_table(run_dir, name: str) -> pd.DataFrame:
    """Concatenate every per-task file of one iteration table."""
    paths = table_paths(run_dir, name)
    if not paths:
        raise FileNotFoundError(f"no {name} parquet under {Path(run_dir) / 'iterations'}")
    frames = [pd.read_parquet(path) for path in paths]
    columns = TABLES[name][0]
    out = pd.concat(frames, ignore_index=True)
    return out.reindex(columns=list(columns))


def write_iteration_tables(result, run_dir, cfg: dict | None = None, *, task=None) -> list[Path]:
    """Write ``iterations/<task_id>.{iterations,iteration_blocks,iteration_capacity}.parquet``.

    Returns ``[]`` for a method with no history (the single-level presets):
    those runs get no iteration files at all.  One file per task and table, so
    concurrent tasks in one run directory never overwrite each other.
    """
    if cfg is not None and not (cfg.get("output") or {}).get("save_iterations", True):
        return []
    frames = build_iteration_tables(result, task=task)
    if frames["iterations"].empty:
        return []

    task_id = str(getattr(task, "task_id", None) or result.design_id)
    out_dir = Path(run_dir) / "iterations"
    written: list[Path] = []
    for name, frame in frames.items():
        if name == "iteration_capacity" and frame.empty:
            continue
        _columns, schema_fn = TABLES[name]
        path = out_dir / f"{task_id}.{name}.parquet"
        written.append(write_parquet_atomic(frame, path, schema_fn()))
    return written
