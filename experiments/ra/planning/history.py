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
    "free_grad_norm_l2",
    "proj_grad_norm_l1",
    "rule",
    "step_size",
    "lr_mw",
    "clip",
    "clip_fraction",
    "step_norm_mw",
    "step_norm_mw_actual",
    "step_norm_free_mw",
    "max_abs_step_mw",
    "n_free",
    "n_at_lower",
    "n_at_upper",
    "trust_radius_mw",
    "rho_actual_pred",
    "stationarity_max",
    "suboptimality",
    "time_s",
    "lambda_carbon",
    "annualization_factor",
)

ITERATION_GRADIENT_COLUMNS = (
    "task_id",
    "outer_iteration",
    "iteration",
    "param",
    "row",
    "name",
    "carrier",
    "capacity_mw",
    "gradient",
    "step_mw",
    "m_hat",
    "v_hat",
    "lower_mw",
    "upper_mw",
    "at_lower",
    "at_upper",
    "free",
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
        pa.field("rule", _STR),
        pa.field("n_free", pa.int32()),
        pa.field("n_at_lower", pa.int32()),
        pa.field("n_at_upper", pa.int32()),
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


def iteration_gradient_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("task_id", _STR),
            pa.field("outer_iteration", pa.int32()),
            pa.field("iteration", pa.int32()),
            pa.field("param", _STR),
            pa.field("row", pa.int32()),
            pa.field("name", _STR),
            pa.field("carrier", _STR),
            pa.field("capacity_mw", pa.float64()),
            pa.field("gradient", pa.float64()),
            pa.field("step_mw", pa.float64()),
            pa.field("m_hat", pa.float64()),
            pa.field("v_hat", pa.float64()),
            pa.field("lower_mw", pa.float64()),
            pa.field("upper_mw", pa.float64()),
            pa.field("at_lower", pa.bool_()),
            pa.field("at_upper", pa.bool_()),
            pa.field("free", pa.bool_()),
        ]
    )


TABLES = {
    "iterations": (ITERATIONS_COLUMNS, iterations_schema),
    "iteration_gradient": (ITERATION_GRADIENT_COLUMNS, iteration_gradient_schema),
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


def _diag_series(history: dict, key: str, n: int) -> list[float]:
    """One field of the per-iteration ``opt_diag`` mapping, as a float series.

    ``trackers.track_opt_diag`` records ``problem.algorithm.diagnostics()``,
    whose keys differ per step rule (``trust_radius_mw`` only exists for the
    trust region).  A missing key is NaN, not an error.
    """
    values = list(history.get("opt_diag") or [])
    out: list[float] = []
    for i in range(n):
        entry = values[i] if i < len(values) else None
        if not isinstance(entry, dict) or entry.get(key) is None:
            out.append(float("nan"))
            continue
        try:
            out.append(float(entry[key]))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _int_series(history: dict, key: str, n: int) -> list:
    """A history series as nullable ints (``None`` where it was not recorded)."""
    values = _series(history, key, n)
    return [None if not math.isfinite(v) else round(v) for v in values]


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


def _row_names(result) -> dict[str, list[str]]:
    """``{param: [row name]}`` from the recorded per-class capacity block."""
    out: dict[str, list[str]] = {}
    capacities = getattr(result, "capacities", None) or {}
    for param in getattr(result, "parameter_names", None) or {}:
        cls_name = _device_class_for_param(result, param)
        names = (capacities.get(cls_name) or {}).get("names")
        if names:
            out[param] = [str(n) for n in names]
    return out


def _row_bounds(result) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{param: (lower, upper)}`` in MW, flattened."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    uppers = getattr(result, "upper_bounds", None) or {}
    for param, lower in (getattr(result, "lower_bounds", None) or {}).items():
        upper = uppers.get(param)
        if upper is None:
            continue
        out[param] = (
            np.asarray(lower, dtype=float).reshape(-1),
            np.asarray(upper, dtype=float).reshape(-1),
        )
    return out


def _gradient_rows(
    *,
    task_id: str,
    outer: int,
    iteration: int,
    state: dict,
    previous: dict,
    grad: dict,
    moments,
    labels: dict,
    names: dict,
    bounds: dict,
) -> list[dict[str, Any]]:
    """One row per parameter row of one recorded iteration.

    Every number in the step-rule diagnosis (``memory/plans/
    2026-09-10-step-rule-spec.md`` section 1) had to be back-solved from the
    parameter history because this table did not exist: per-row gradients, the
    free share of the gradient norm, the realised per-row step.  It is written
    every ``optimizer.grad_history_every`` iterations.

    **The columns are not all on the same clock.**  The descent loop steps, then
    evaluates, then records once, so on row ``i``:

    * ``capacity_mw``, ``at_lower`` / ``at_upper`` / ``free``, ``gradient`` are
      all *at* iterate ``i`` -- ``gradient`` is ``dF/deta`` evaluated there, and
      it is the gradient that drives the step **to** iterate ``i + 1``;
    * ``step_mw`` is ``eta_i - eta_{i-1}``, the step that *arrived* at this
      iterate, i.e. one iteration behind ``gradient`` (NaN on ``iteration 0``,
      and on any recorded iteration whose predecessor was not recorded --
      ``grad_history_every > 1`` samples the parameter history, it does not
      resample it);
    * ``m_hat`` / ``v_hat`` are the step rule's moments **after** the update that
      produced iterate ``i``, so they are the moments that generated
      ``step_mw``, not the ones that will generate the next step.

    Reconstructing "gradient at i -> step to i+1" therefore means joining row
    ``i``'s ``gradient`` to row ``i + 1``'s ``step_mw``.
    """
    m_hat = (moments or {}).get("m_hat") or {}
    v_hat = (moments or {}).get("v_hat") or {}
    rows: list[dict[str, Any]] = []
    for param, values in grad.items():
        g = np.asarray(values, dtype=float).reshape(-1)
        eta = np.asarray(state.get(param, np.full(g.shape, np.nan)), dtype=float).reshape(-1)
        prev = previous.get(param)
        step = (
            eta - np.asarray(prev, dtype=float).reshape(-1)
            if prev is not None
            else np.full(g.shape, np.nan)
        )
        lower, upper = bounds.get(param, (None, None))
        m = np.asarray(m_hat.get(param, np.full(g.shape, np.nan)), dtype=float).reshape(-1)
        v = np.asarray(v_hat.get(param, np.full(g.shape, np.nan)), dtype=float).reshape(-1)
        row_names = names.get(param) or []
        carriers = labels.get(param) or []
        for j in range(g.size):
            lo = float(lower[j]) if lower is not None and j < lower.size else float("nan")
            hi = float(upper[j]) if upper is not None and j < upper.size else float("nan")
            value = float(eta[j]) if j < eta.size else float("nan")
            at_lower = bool(math.isfinite(lo) and math.isfinite(value) and value <= lo + 1e-9)
            at_upper = bool(
                math.isfinite(hi)
                and math.isfinite(value)
                and value >= hi - 1e-9
                and not at_lower
            )
            rows.append(
                {
                    "task_id": task_id,
                    "outer_iteration": int(outer),
                    "iteration": int(iteration),
                    "param": str(param),
                    "row": int(j),
                    "name": str(row_names[j]) if j < len(row_names) else str(j),
                    "carrier": str(carriers[j]) if j < len(carriers) else "",
                    "capacity_mw": value,
                    "gradient": float(g[j]),
                    "step_mw": float(step[j]) if j < step.size else float("nan"),
                    "m_hat": float(m[j]) if j < m.size else float("nan"),
                    "v_hat": float(v[j]) if j < v.size else float("nan"),
                    "lower_mw": lo,
                    "upper_mw": hi,
                    "at_lower": at_lower,
                    "at_upper": at_upper,
                    "free": not (at_lower or at_upper),
                }
            )
    return rows


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
    rule = str(optimizer.get("rule") or "gradient")
    grad_every = int(optimizer.get("grad_history_every") or 0)
    save_params = bool(optimizer.get("save_param_history", True))
    labels = meta.get("carrier_labels") or {}
    row_names = _row_names(result)
    bounds = _row_bounds(result)
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
    gradient_rows: list[dict[str, Any]] = []

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

        free_grad = _series(history, "free_grad_norm_l2", n)
        # `step_norm_mw_actual` is `||eta - eta_prev||_2` AFTER the box
        # projection: the step the design actually took.  The old
        # `step_norm_mw` was `step_size * min(||g||, clip)`, the step the rule
        # asked for before projection and before the frozen coordinates were
        # dropped -- 1,000 MW against a realised 13 MW on ca2040_z4.  The
        # column keeps its name and now carries the honest number; the
        # `_actual` alias is the name the step-rule spec uses.
        step_actual = _series(history, "step_norm_mw_actual", n)
        step_free = _series(history, "step_norm_free_mw", n)
        stationarity = _series(history, "stationarity_max", n)
        n_free = _int_series(history, "n_free", n)
        n_at_lower = _int_series(history, "n_at_lower", n)
        n_at_upper = _int_series(history, "n_at_upper", n)
        lr_mw = _diag_series(history, "lr_mw", n)
        max_abs_step = _diag_series(history, "max_abs_step_mw", n)
        trust_radius = _diag_series(history, "trust_radius_mw", n)
        rho = _diag_series(history, "rho_actual_pred", n)
        grads = list(history.get("grad_sampled") or [])
        moments = list(history.get("opt_moments") or [])

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
                    "free_grad_norm_l2": free_grad[i],
                    "proj_grad_norm_l1": proj[i],
                    "rule": rule,
                    "step_size": step_size,
                    "lr_mw": lr_mw[i],
                    "clip": clip,
                    "clip_fraction": clip_fraction,
                    # Post-projection, over every row; NaN-safe fallback to the
                    # pre-projection number only for a history recorded before
                    # the actual-step tracker existed.
                    "step_norm_mw": (
                        step_actual[i] if math.isfinite(step_actual[i]) else step_norm
                    ),
                    "step_norm_mw_actual": step_actual[i],
                    "step_norm_free_mw": step_free[i],
                    "max_abs_step_mw": max_abs_step[i],
                    "n_free": n_free[i],
                    "n_at_lower": n_at_lower[i],
                    "n_at_upper": n_at_upper[i],
                    "trust_radius_mw": trust_radius[i],
                    "rho_actual_pred": rho[i],
                    "stationarity_max": stationarity[i],
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

            snapshot = _param_snapshot(params[i]) if i < len(params) else {}
            grad_snapshot = grads[i] if i < len(grads) else None
            if grad_every > 0 and isinstance(grad_snapshot, dict):
                previous = _param_snapshot(params[i - 1]) if 0 < i < len(params) else {}
                gradient_rows.extend(
                    _gradient_rows(
                        task_id=task_id,
                        outer=outer,
                        iteration=i,
                        state=snapshot,
                        previous=previous,
                        grad=grad_snapshot,
                        moments=moments[i] if i < len(moments) else None,
                        labels=labels,
                        names=row_names,
                        bounds=bounds,
                    )
                )

            if save_params and i < len(params):
                for param, values in snapshot.items():
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
        "iteration_gradient": _frame(gradient_rows, ITERATION_GRADIENT_COLUMNS),
    }


#: Columns that are counts, not measurements: they carry ``pd.NA`` (never NaN)
#: where a history predates the tracker that records them, so the int32 schema
#: still accepts them.
NULLABLE_INT_COLUMNS = ("draw", "n_free", "n_at_lower", "n_at_upper")


def _frame(rows: Sequence[dict], columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=list(columns))
    for column in NULLABLE_INT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.array(
                [None if v is None or (isinstance(v, float) and math.isnan(v)) else int(v)
                 for v in frame[column].tolist()],
                dtype="Int32",
            )
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
    """Write ``iterations/<task_id>.<table>.parquet`` for every non-empty table.

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
        if name in ("iteration_capacity", "iteration_gradient") and frame.empty:
            continue
        _columns, schema_fn = TABLES[name]
        path = out_dir / f"{task_id}.{name}.parquet"
        written.append(write_parquet_atomic(frame, path, schema_fn()))
    return written
