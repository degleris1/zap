"""Metrics for one block solve, and aggregation over a run's task files.

``block_metrics`` takes a solved block and returns a plain dict of floats and
strings; everything downstream (the ledger, ``metrics.csv``, the run card) works
on that dict, so the aggregation path is testable without a solver.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Metrics that are sums over hours, and can therefore be added across blocks.
ADDITIVE_METRICS = (
    "operational_cost",
    "generation_cost",
    "voll_cost",
    "export_revenue",
    "unserved_energy_mwh",
    "lost_load_hours",
    "co2_tonnes",
    "curtailment_mwh",
    "imports_mwh",
    "exports_mwh",
)

SHORTFALL_TOL_MW = 1e-3


def _to_numpy(x):
    """Convert a (possibly torch) array-like to a float64 numpy array."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def numpyify(obj):
    """Recursively convert nested lists/tuples of tensors to numpy arrays."""
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        converted = [numpyify(o) for o in obj]
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
            return type(obj)(*converted)
        return converted
    return _to_numpy(obj)


def device_class(device) -> str:
    return type(device).__name__


def device_groups(devices: Sequence) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, device in enumerate(devices):
        groups.setdefault(device_class(device), []).append(i)
    return groups


def _carriers(index, cls_name: str, device, n_rows: int):
    """Per-row carrier labels, from the SystemIndex if present else the device.

    The index is authoritative, so a length mismatch means the index and the
    device list disagree about the system and every carrier-keyed metric would
    be silently misattributed: fail instead (verifier, item 6).
    """
    if index is not None:
        table = getattr(index, "carrier", None)
        if isinstance(table, dict) and cls_name in table:
            carriers = np.asarray(table[cls_name]).astype(str)
            if carriers.size != n_rows:
                raise ValueError(
                    f"SystemIndex.carrier[{cls_name!r}] has {carriers.size} rows but the "
                    f"{cls_name} device has {n_rows}; the index does not match the devices"
                )
            return carriers
    for attr in ("fuel_type", "carrier"):
        value = getattr(device, attr, None)
        if value is not None and np.size(value) == n_rows:
            return np.asarray(value).ravel().astype(str)
    name = getattr(device, "name", None)
    if name is not None and np.size(name) == n_rows:
        return np.asarray(name).ravel().astype(str)
    return None


def _row_sum(x) -> np.ndarray:
    """Sum over the time axis, returning one value per row."""
    return np.asarray(x, dtype=np.float64).sum(axis=1)


def block_metrics(loaded, devices: Sequence, outcome, block) -> dict[str, Any]:
    """Metrics of one solved block. ``loaded`` may be None (only the index is used)."""
    index = getattr(loaded, "index", None)
    network = getattr(loaded, "network", None)

    power = numpyify(outcome.power)
    angle = numpyify(outcome.angle)
    local_vars = numpyify(outcome.local_variables)
    groups = device_groups(devices)

    # Devices may include the Ground appended by PowerNetwork.dispatch.
    n = len(devices)
    power, angle, local_vars = power[:n], angle[:n], local_vars[:n]

    per_device_cost = []
    for device, p, a, u in zip(devices, power, angle, local_vars):
        try:
            per_device_cost.append(float(device.operation_cost(p, a, u, la=np)))
        except Exception:  # pragma: no cover - defensive
            logger.exception("operation_cost failed for %s", device_class(device))
            per_device_cost.append(float("nan"))

    if network is not None:
        operational_cost = float(network.operation_cost(devices, power, angle, local_vars, la=np))
    else:  # pragma: no cover - only when a caller has no network handle
        operational_cost = float(np.nansum(per_device_cost))

    metrics: dict[str, Any] = {
        "operational_cost": operational_cost,
        "generation_cost": float(sum(per_device_cost[i] for i in groups.get("Generator", []))),
        "voll_cost": float(sum(per_device_cost[i] for i in groups.get("Load", []))),
        "hours": int(block.hours),
    }

    # --- Unserved energy -----------------------------------------------------
    # Equivalent to zap.planning.operation_objectives.UnservedEnergyObjective
    # (ENS = load + power[0]) but computed in numpy and scaled by the injector's
    # nominal_capacity, which is 1.0 for every Load built by the WP1 reader.
    ens = 0.0
    lost_load_hours = 0
    for i in groups.get("Load", []):
        device = devices[i]
        demand = np.asarray(device.load, dtype=np.float64) * np.asarray(
            device.nominal_capacity, dtype=np.float64
        )
        shortfall = demand + power[i][0]
        ens += float(shortfall.sum())
        lost_load_hours += int(np.count_nonzero(shortfall > SHORTFALL_TOL_MW))
    metrics["unserved_energy_mwh"] = ens
    metrics["lost_load_hours"] = lost_load_hours

    # --- Emissions -----------------------------------------------------------
    co2 = 0.0
    for device, p in zip(devices, power):
        try:
            co2 += float(device.get_emissions(p, la=np))
        except Exception:
            logger.debug("get_emissions failed for %s", device_class(device), exc_info=True)
    metrics["co2_tonnes"] = co2

    # --- Generators: dispatch by carrier and VRE curtailment -----------------
    gen_by_carrier: dict[str, float] = {}
    curtailment = float("nan")
    curtailment_total = 0.0
    saw_vre = False
    for i in groups.get("Generator", []):
        device = devices[i]
        gen = _row_sum(power[i][0])
        carriers = _carriers(index, "Generator", device, gen.size)
        if carriers is not None:
            for carrier, value in zip(carriers, gen):
                gen_by_carrier[str(carrier)] = gen_by_carrier.get(str(carrier), 0.0) + float(value)

        vre_mask = getattr(index, "vre_mask", None) if index is not None else None
        if vre_mask is not None and np.size(vre_mask) == gen.size:
            available = np.asarray(device.nominal_capacity, dtype=np.float64) * np.asarray(
                device.dynamic_capacity, dtype=np.float64
            )
            available = np.broadcast_to(available, power[i][0].shape)
            curtail = (available - power[i][0])[np.asarray(vre_mask, dtype=bool), :]
            curtailment_total += float(np.maximum(curtail, 0.0).sum())
            saw_vre = True
    metrics["generation_mwh_by_carrier"] = json.dumps(gen_by_carrier, sort_keys=True)
    metrics["curtailment_mwh"] = curtailment_total if saw_vre else curtailment

    # --- Links: export revenue, imports, exports -----------------------------
    export_revenue = 0.0
    imports_mwh = 0.0
    exports_mwh = 0.0
    saw_links = False
    for cls_name in ("DirectedLine", "DCLine", "ACLine", "PowerLine"):
        for i in groups.get(cls_name, []):
            saw_links = True
            device = devices[i]
            linear_cost = np.asarray(getattr(device, "linear_cost", 0.0), dtype=np.float64)
            sink = power[i][1]
            source = power[i][0]
            if linear_cost.size:
                cost = np.broadcast_to(linear_cost, sink.shape) * sink
                negative_rows = np.broadcast_to(linear_cost, sink.shape).min(axis=1) < 0
                export_revenue += float(-cost[negative_rows, :].sum())
            carriers = _carriers(index, cls_name, device, sink.shape[0])
            if carriers is not None:
                is_import = np.array(["import" in c for c in carriers])
                is_export = np.array(["export" in c for c in carriers])
                imports_mwh += float(sink[is_import, :].sum())
                exports_mwh += float(-source[is_export, :].sum())
    metrics["export_revenue"] = export_revenue
    metrics["imports_mwh"] = imports_mwh if saw_links else float("nan")
    metrics["exports_mwh"] = exports_mwh if saw_links else float("nan")

    # --- Storage -------------------------------------------------------------
    discharge_total = 0.0
    energy_capacity = 0.0
    for i in groups.get("StorageUnit", []):
        device = devices[i]
        state = local_vars[i]
        if state is None:
            continue
        discharge = getattr(state, "discharge", None)
        if discharge is None and isinstance(state, (list, tuple)) and len(state) >= 3:
            discharge = state[2]
        if discharge is None:
            continue
        discharge_total += float(np.asarray(discharge, dtype=np.float64).sum())
        energy_capacity += float(
            np.sum(
                np.asarray(device.power_capacity, dtype=np.float64)
                * np.asarray(device.duration, dtype=np.float64)
            )
        )
    metrics["storage_cycles"] = (
        discharge_total / energy_capacity if energy_capacity > 0 else float("nan")
    )

    # --- Prices --------------------------------------------------------------
    prices = _to_numpy(outcome.prices)
    if prices is None or prices.size == 0:
        metrics["mean_price"] = float("nan")
        metrics["max_price"] = float("nan")
    else:
        weights = np.zeros(prices.shape)
        for i in groups.get("Load", []):
            device = devices[i]
            demand = np.asarray(device.load, dtype=np.float64) * np.asarray(
                device.nominal_capacity, dtype=np.float64
            )
            demand = np.broadcast_to(demand, power[i][0].shape)
            terminals = np.asarray(device.terminal).ravel()
            for row, node in enumerate(terminals):
                weights[node, :] += demand[row, :]
        total = weights.sum()
        metrics["mean_price"] = (
            float((prices * weights).sum() / total) if total > 0 else float(prices.mean())
        )
        metrics["max_price"] = float(prices.max())

    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def read_task_records(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted((Path(run_dir) / "tasks").glob("*.json")):
        try:
            with open(path, "r") as f:
                records.append(json.load(f))
        except json.JSONDecodeError:  # pragma: no cover - defensive
            logger.warning("skipping unreadable task file %s", path)
    return records


def records_to_frame(records: Iterable[dict]) -> pd.DataFrame:
    """One row per task: record fields, then metrics that do not collide.

    ``wall_clock_s`` is the whole task (including building the system);
    ``solve_wall_clock_s`` is the solver call alone. Both are kept.
    """
    rows = []
    for record in records:
        row = {k: v for k, v in record.items() if k != "metrics"}
        for key, value in (record.get("metrics") or {}).items():
            if key in row and row[key] is not None:
                continue  # the record-level field wins; see the docstring
            row[key] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("task_id").reset_index(drop=True)
    return frame


def aggregate(run_dir: Path) -> pd.DataFrame:
    """Read every task file into one row and write ``metrics.csv``."""
    run_dir = Path(run_dir)
    frame = records_to_frame(read_task_records(run_dir))
    frame.to_csv(run_dir / "metrics.csv", index=False)
    return frame


def deviation_vs_reference(
    df: pd.DataFrame, reference_bounds: tuple[int, int] | None = None
) -> pd.DataFrame:
    """Blocking error against the reference solve, on the reference window only.

    For every (method, block_size) pair the additive metrics are re-summed over
    exactly those blocks that lie *entirely* inside the reference window; blocks
    that straddle the boundary are excluded and counted.

    The comparison is only like-for-like if the retained blocks cover the whole
    reference window, so the covered hours are recorded per block size and a
    shortfall raises (verifier, item 1c): a misaligned reference window would
    otherwise silently compare, say, 504 h of blocks against a 672 h reference.
    """
    empty = pd.DataFrame(
        columns=[
            "method",
            "block_size",
            "metric",
            "value",
            "reference",
            "dev_abs",
            "dev_rel",
            "n_blocks",
            "n_blocks_excluded",
            "ref_window_hours",
            "ref_window_hours_covered",
        ]
    )
    if df is None or df.empty or "block_size" not in df.columns:
        return empty

    ok = df[df.get("status", "ok") == "ok"]
    ref_rows = ok[ok["block_size"].astype(str) == "reference"]
    if ref_rows.empty:
        return empty

    if reference_bounds is None:
        reference_bounds = (int(ref_rows["start"].min()), int(ref_rows["stop"].max()))
    lo, hi = reference_bounds

    metrics = [m for m in ADDITIVE_METRICS if m in ok.columns]
    out = []
    for (method, block_size), group in ok.groupby(["method", "block_size"], sort=True):
        if str(block_size) == "reference":
            continue
        inside = group[(group["start"] >= lo) & (group["stop"] <= hi)]
        straddling = group[
            (group["start"] < hi) & (group["stop"] > lo) & ~group.index.isin(inside.index)
        ]
        ref_for_method = ref_rows[ref_rows["method"] == method]
        if ref_for_method.empty:
            ref_for_method = ref_rows

        covered = int((inside["stop"] - inside["start"]).sum())
        window_hours = hi - lo
        if covered != window_hours:
            raise ValueError(
                f"blocks of size {block_size} cover {covered} h of the {window_hours} h "
                f"reference window [{lo}, {hi}) for method {method!r}: the window is not an "
                "exact number of blocks (or blocks are missing), so the blocking error would "
                "not be like-for-like"
            )

        for metric in metrics:
            value = float(pd.to_numeric(inside[metric], errors="coerce").sum())
            reference = float(pd.to_numeric(ref_for_method[metric], errors="coerce").sum())
            dev_abs = value - reference
            dev_rel = dev_abs / reference if reference not in (0.0,) else float("nan")
            out.append(
                {
                    "method": method,
                    "block_size": block_size,
                    "metric": metric,
                    "value": value,
                    "reference": reference,
                    "dev_abs": dev_abs,
                    "dev_rel": dev_rel,
                    "n_blocks": len(inside),
                    "n_blocks_excluded": len(straddling),
                    "ref_window_hours": window_hours,
                    "ref_window_hours_covered": covered,
                }
            )
    if not out:
        return empty
    return pd.DataFrame(out)
