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

from . import persist

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
    # Gross scaled demand of the block, the denominator of NEUE = EUE / annual
    # demand (FORMULATIONS 3.4).  Additive over blocks by construction, and the
    # weight `evaluate` uses to pool per-block mean prices (WP-E0 / spec E6).
    "demand_mwh",
)

SHORTFALL_TOL_MW = 1e-3

#: Metrics of a ``mode: plan`` task.  Every monetary / emissions number here is
#: **annual** (spec section 6, decision W-A): the raw objective is scaled by the
#: coverage of the block sample, so raw values are incomparable across
#: ``selection`` settings.  The raw ones are kept alongside, clearly named.
PLANNING_METRICS = (
    "objective_annual",
    "capex_annual",
    "opex_annual",
    "emissions_tonnes_annual",
    "total_capacity_mw",
    # Which iterate became the design and how far the descent loop got.
    # `runcard.summarize` coerces these to numbers, so only the numeric half of
    # the iterate-selection record lives here; the rule (`design_selection`) and
    # the stop reason (`stopped_by`) are strings and are shown per design in the
    # card's Objective table instead.
    "design_iteration",
    "num_iterations_completed",
)


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

    # Swallowed exceptions here hid `StorageUnit.operation_cost(state=None) -> 0.0`
    # on every ADMM row of the phase-1 benchmark; count and name them instead.
    failures: dict[str, str] = {}

    per_device_cost = []
    for device, p, a, u in zip(devices, power, angle, local_vars):
        try:
            per_device_cost.append(float(device.operation_cost(p, a, u, la=np)))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "operation_cost failed for %s: %s", device_class(device), exc, exc_info=True
            )
            failures[f"operation_cost:{device_class(device)}"] = f"{type(exc).__name__}: {exc}"
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
    # Factored into `persist.load_shortfall` so the ENS profile written for the
    # R5 heat map and this scalar can never disagree about what "unserved" means.
    shortfalls = persist.load_shortfall(devices, power, groups)
    # LOLH counts *hours* in which the system sheds anywhere, not (bus, hour)
    # pairs: sum the shortfall over load rows first, then count hours.
    system_shortfall = None
    # Gross demand of the block, i.e. `load * nominal_capacity` summed over rows
    # and hours -- the same quantity the price weights and the ENS profile use,
    # *after* `system.demand_scaling` (it is what the LP was asked to serve).
    demand = 0.0
    for entry in shortfalls:
        ens += float(entry.shortfall.sum())
        demand += float(np.asarray(entry.demand, dtype=np.float64).sum())
        hourly = np.asarray(entry.shortfall, dtype=np.float64).sum(axis=0)
        system_shortfall = hourly if system_shortfall is None else system_shortfall + hourly
    if system_shortfall is not None:
        lost_load_hours = int(np.count_nonzero(system_shortfall > SHORTFALL_TOL_MW))
    metrics["unserved_energy_mwh"] = ens
    metrics["lost_load_hours"] = lost_load_hours
    metrics["demand_mwh"] = demand if shortfalls else float("nan")

    # --- Emissions -----------------------------------------------------------
    co2 = 0.0
    for device, p in zip(devices, power):
        try:
            co2 += float(device.get_emissions(p, la=np))
        except Exception as exc:
            logger.warning(
                "get_emissions failed for %s: %s", device_class(device), exc, exc_info=True
            )
            failures[f"get_emissions:{device_class(device)}"] = f"{type(exc).__name__}: {exc}"
    metrics["co2_tonnes"] = co2

    # --- Generators: dispatch by carrier and VRE curtailment -----------------
    gen_by_carrier: dict[str, float] = {}
    curtailment = float("nan")
    curtailment_total = 0.0
    saw_vre = False
    # Per-hour available capacity, so `eval.parquet` has an exact minimum
    # available capacity per design without hourly data.  One membership
    # everywhere (decision 2026-09-09): in-state generators x weather x
    # outage/UCAP derate (imports excluded via `index.import_mask`) PLUS
    # storage power capacity x `power_availability`.  The storage term is not
    # SoC-limited.  Same membership as `available_capacity_mw` in
    # `persist.hourly_rows`.
    available_by_hour = np.zeros(int(block.hours), dtype=np.float64)
    saw_available = False
    for i in groups.get("Generator", []):
        device = devices[i]
        gen = _row_sum(power[i][0])
        carriers = _carriers(index, "Generator", device, gen.size)
        if carriers is not None:
            for carrier, value in zip(carriers, gen):
                gen_by_carrier[str(carrier)] = gen_by_carrier.get(str(carrier), 0.0) + float(value)

        gen_available = np.broadcast_to(
            np.asarray(device.nominal_capacity, dtype=np.float64)
            * np.asarray(device.dynamic_capacity, dtype=np.float64),
            power[i][0].shape,
        )
        in_state = np.ones(gen.size, dtype=bool)
        import_mask = getattr(index, "import_mask", None) if index is not None else None
        if import_mask is not None and np.size(import_mask) == gen.size:
            in_state = ~np.asarray(import_mask, dtype=bool)
        if gen_available.shape[1] == available_by_hour.size:
            available_by_hour += gen_available[in_state, :].sum(axis=0)
            saw_available = True

        vre_mask = getattr(index, "vre_mask", None) if index is not None else None
        if vre_mask is not None and np.size(vre_mask) == gen.size:
            available = np.asarray(device.nominal_capacity, dtype=np.float64) * np.asarray(
                device.dynamic_capacity, dtype=np.float64
            )
            available = np.broadcast_to(available, power[i][0].shape)
            curtail = (available - power[i][0])[np.asarray(vre_mask, dtype=bool), :]
            curtailment_total += float(np.maximum(curtail, 0.0).sum())
            saw_vre = True
    for i in groups.get("StorageUnit", []):
        device = devices[i]
        power_capacity = np.asarray(device.power_capacity, dtype=np.float64).reshape(-1, 1)
        availability = np.asarray(
            getattr(device, "power_availability", None)
            if getattr(device, "power_availability", None) is not None
            else 1.0,
            dtype=np.float64,
        )
        storage_available = power_capacity * np.atleast_2d(availability)
        if storage_available.shape[1] == 1:
            storage_available = np.broadcast_to(
                storage_available, (storage_available.shape[0], available_by_hour.size)
            )
        if storage_available.shape[1] == available_by_hour.size:
            available_by_hour += storage_available.sum(axis=0)
            saw_available = True

    metrics["generation_mwh_by_carrier"] = json.dumps(gen_by_carrier, sort_keys=True)
    metrics["curtailment_mwh"] = curtailment_total if saw_vre else curtailment
    metrics["available_mw_min"] = (
        float(available_by_hour.min()) if saw_available else float("nan")
    )
    metrics["available_mwh_total"] = (
        float(available_by_hour.sum()) if saw_available else float("nan")
    )

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
    start_energy = 0.0
    saw_energy = False
    for i in groups.get("StorageUnit", []):
        device = devices[i]
        state = local_vars[i]
        if state is None:
            continue
        discharge = getattr(state, "discharge", None)
        energy = getattr(state, "energy", None)
        if isinstance(state, (list, tuple)) and len(state) >= 3:
            energy = state[0] if energy is None else energy
            discharge = state[2] if discharge is None else discharge
        if discharge is None:
            continue
        discharge_total += float(np.asarray(discharge, dtype=np.float64).sum())
        energy_capacity += float(
            np.sum(
                np.asarray(device.power_capacity, dtype=np.float64)
                * np.asarray(device.duration, dtype=np.float64)
            )
        )
        if energy is not None:
            # Column 0 is the block's starting level of every unit; with an ADMM
            # window it is the first window's start.
            start_energy += float(np.asarray(energy, dtype=np.float64)[:, 0].sum())
            saw_energy = True
    storage_cycles = discharge_total / energy_capacity if energy_capacity > 0 else float("nan")
    metrics["storage_cycles"] = storage_cycles
    # `storage_cycles` counts cycles *per block*, so it is not comparable across
    # block sizes (24 h vs 168 h vs a full year); normalise it per day.
    metrics["storage_cycles_per_day"] = storage_cycles * 24.0 / float(block.hours)
    # Fleet energy-weighted starting state of charge, as a fraction of total
    # energy capacity: with `storage_soc_mode: fixed` this is exactly
    # `storage_init_soc`; with `cyclic_free` it is the level the block chose.
    metrics["storage_start_soc_frac"] = (
        start_energy / energy_capacity if (saw_energy and energy_capacity > 0) else float("nan")
    )

    # --- Prices --------------------------------------------------------------
    # Restricted to load-carrying buses: `ca2040_z4` prices the import bus
    # `p6_imports` (no load) at exactly the 60 $/MWh of its marginal unit and the
    # export buses at a degenerate -155 $/MWh, neither of which is a system price.
    prices = _to_numpy(outcome.prices)
    if prices is None or prices.size == 0:
        metrics["mean_price"] = float("nan")
        metrics["max_price"] = float("nan")
        metrics["max_price_all_buses"] = float("nan")
    else:
        weights = np.zeros(prices.shape)
        load_nodes: set[int] = set()
        for i in groups.get("Load", []):
            device = devices[i]
            demand = np.asarray(device.load, dtype=np.float64) * np.asarray(
                device.nominal_capacity, dtype=np.float64
            )
            demand = np.broadcast_to(demand, power[i][0].shape)
            terminals = np.asarray(device.terminal).ravel()
            for row, node in enumerate(terminals):
                weights[node, :] += demand[row, :]
                load_nodes.add(int(node))
        total = weights.sum()
        metrics["mean_price"] = (
            float((prices * weights).sum() / total) if total > 0 else float(prices.mean())
        )
        metrics["max_price_all_buses"] = float(prices.max())
        nodes = sorted(load_nodes)
        metrics["max_price"] = float(prices[nodes, :].max()) if nodes else float(prices.max())

    if failures:
        metrics["metric_failures"] = len(failures)
        metrics["metric_failure_detail"] = json.dumps(failures, sort_keys=True)
    else:
        metrics["metric_failures"] = 0

    return _to_physical_units(metrics, getattr(loaded, "meta", None) or {})


#: Metrics denominated in money (scaled by ``cost_unit * power_unit``), in power
#: or energy (``power_unit``), and in price (``cost_unit``). Everything else --
#: hours, counts, ratios such as ``storage_cycles`` -- is unit-invariant.
_MONEY_METRICS = ("operational_cost", "generation_cost", "voll_cost", "export_revenue")
_ENERGY_METRICS = (
    "unserved_energy_mwh",
    "demand_mwh",
    "curtailment_mwh",
    "imports_mwh",
    "exports_mwh",
    "available_mwh_total",
    # Set in `dispatch.solve_block_admm`, *after* `block_metrics` has already run
    # `_to_physical_units`, and scaled by `power_unit` there by hand. Listed here
    # so their unit is declared in one place; they cannot be double-scaled.
    "storage_energy_capacity_mwh",
    "admm_soc_residual_gate_mwh",
    "admm_sum_abs_imbalance_mwh",
    "admm_sum_abs_soc_residual_mwh",
)
#: Metrics denominated in power alone (MW), scaled by ``power_unit``.
#: `block_peak_load_mw` / `admm_imbalance_gate_mw` are likewise set (and scaled)
#: in `dispatch.solve_block_admm`; see the note in `_ENERGY_METRICS`.
_POWER_METRICS = ("available_mw_min", "block_peak_load_mw", "admm_imbalance_gate_mw")
#: Emission rates are divided by `cost_unit` too (`AbstractInjector.scale_costs`
#: scales them with costs so prices stay in $/MWh), so emissions carry both units.
_EMISSION_METRICS = ("co2_tonnes",)
#: Metrics denominated in price ($/MWh), scaled by ``cost_unit``.
#: The two ``admm_price_*`` names are produced in `dispatch.solve_block_admm`,
#: *after* `block_metrics` has run `_to_physical_units`, and are scaled by
#: `cost_unit` there by hand; they are listed here so their unit is declared in
#: one place and cannot be double-scaled.
_PRICE_METRICS = (
    "mean_price",
    "max_price",
    "max_price_all_buses",
    "admm_price_error_max_usd_per_mwh",
    "admm_price_movement_usd_per_mwh",
)


def _to_physical_units(metrics: dict[str, Any], meta: dict) -> dict[str, Any]:
    """Undo ``LoadOptions.power_unit`` / ``cost_unit`` so every card is in MW and $.

    The importer divides powers by ``power_unit`` and costs by ``cost_unit`` purely
    to condition the solve; a run card that reported those units would not be
    comparable across configs (benchmark review, section 3.4).
    """
    power_unit = float(meta.get("power_unit", 1.0) or 1.0)
    cost_unit = float(meta.get("cost_unit", 1.0) or 1.0)
    if power_unit == 1.0 and cost_unit == 1.0:
        return metrics

    for key in _MONEY_METRICS:
        if key in metrics:
            metrics[key] *= cost_unit * power_unit
    for key in _ENERGY_METRICS:
        if key in metrics:
            metrics[key] *= power_unit
    for key in _POWER_METRICS:
        if key in metrics:
            metrics[key] *= power_unit
    for key in _EMISSION_METRICS:
        if key in metrics:
            metrics[key] *= cost_unit * power_unit
    for key in _PRICE_METRICS:
        if key in metrics:
            metrics[key] *= cost_unit
    if "generation_mwh_by_carrier" in metrics:
        by_carrier = json.loads(metrics["generation_mwh_by_carrier"])
        metrics["generation_mwh_by_carrier"] = json.dumps(
            {k: v * power_unit for k, v in by_carrier.items()}, sort_keys=True
        )
    metrics["power_unit"] = power_unit
    metrics["cost_unit"] = cost_unit
    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def read_task_records(run_dir: Path) -> list[dict]:
    """Every task record of a run, **flattened to one record per block**.

    A ``execution.task_granularity: case`` task writes one file holding its 52
    per-block records under ``blocks`` (WP-E3).  Those elements already carry
    their own ``task_id`` / ``block_index`` / metrics, so yielding them in place
    of the case record keeps ``metrics.csv`` exactly per block and leaves every
    downstream consumer -- ``eval.parquet``, ``deviation_vs_reference``, the
    plots -- unchanged.  A case whose ``blocks`` list is empty contributes no
    rows: the case record itself has no block metrics and must never become one.
    """
    records = []
    for path in sorted((Path(run_dir) / "tasks").glob("*.json")):
        try:
            with open(path, "r") as f:
                record = json.load(f)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            logger.warning("skipping unreadable task file %s", path)
            continue
        blocks = record.get("blocks") if isinstance(record, dict) else None
        if isinstance(blocks, list):
            records.extend(b for b in blocks if isinstance(b, dict))
        else:
            records.append(record)
    _assert_unique_task_ids(records, run_dir)
    return records


def _assert_unique_task_ids(records: list[dict], run_dir: Path) -> None:
    """One row per block, always -- the backstop under the granularity guard.

    A run directory holding both a case file and its blocks' own files yields
    every block twice, which silently doubles every additive metric and puts
    ``coverage`` at 2 (verifier F1, 2026-09-09).  ``cli.adopt_task_granularity``
    stops that being *created*; this refuses to aggregate a directory where it
    somehow already happened, because a doubled headline is worse than no
    headline.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        task_id = str(record.get("task_id"))
        if task_id in seen:
            duplicates.append(task_id)
        seen.add(task_id)
    if duplicates:
        raise ValueError(
            f"{run_dir}: {len(duplicates)} task id(s) appear more than once in the ledger "
            f"(e.g. {sorted(set(duplicates))[:5]}). This directory holds task files of both "
            "granularities -- `case` records nest the same blocks that the per-block files "
            "already carry -- so every affected metric would be counted twice. Remove one "
            "set of files (the per-block ones, or the case ones) and aggregate again."
        )


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


def aggregate(run_dir: Path, cfg: dict | None = None) -> pd.DataFrame:
    """Read every task file into one row and write ``metrics.csv``.

    When ``cfg`` is given and ``output.combine_hourly`` is true, the per-task
    parquet directories (``hourly/``, ``ens_profile/``, ``admm_trace/``) are
    also concatenated into their sibling single files (D3) *first*, because the
    price-error decomposition below reads the combined hourly table; and when
    ``output.save_price_error`` is true, ``price_error.parquet`` is written and
    its per-block summary columns are merged onto ``metrics.csv``.
    """
    run_dir = Path(run_dir)
    frame = records_to_frame(read_task_records(run_dir))
    if cfg is not None:
        try:
            for path in persist.combine_outputs(run_dir, cfg):
                logger.info("combined %s", path)
        except Exception as exc:  # noqa: BLE001 - combination must not fail a run
            logger.warning("could not combine per-task parquet files: %s", exc)
        frame = _attach_price_error(run_dir, frame, cfg)
    frame.to_csv(run_dir / "metrics.csv", index=False)
    return frame


def _attach_price_error(run_dir: Path, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Write ``price_error.parquet`` and merge its summary columns onto ``frame``.

    Never raises: a diagnostic must not lose a run's ``metrics.csv``.
    """
    if not bool((cfg.get("output") or {}).get("save_price_error", True)):
        return frame
    try:
        rows = price_error_rows(run_dir, frame)
        if rows is None or rows.empty:
            logger.info(
                "no price_error.parquet: %s", (rows.attrs.get("skip_reason") if rows is not None
                                               else "no rows")
            )
            return frame
        path = persist.write_price_error(run_dir, rows)
        logger.info("wrote %s (%d bus-hours)", path, len(rows))
        summary = price_error_block_summary(rows)
        if summary.empty or "task_id" not in frame.columns:
            return frame
        keep = ["task_id"] + list(PRICE_ERROR_SUMMARY_COLUMNS)
        frame = frame.merge(summary[keep], on="task_id", how="left")
    except Exception as exc:  # noqa: BLE001 - see the docstring
        logger.warning("could not write the price-error decomposition: %s", exc, exc_info=True)
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


PRICE_ERROR_COLUMNS = (
    "method",
    "block_size",
    "year",
    "draw",
    "design_id",
    "block_index",
    "n_hours",
    "max_abs_price_error",
    "mean_abs_price_error",
    "reference_mean_price",
    "skip_reason",
)

#: The `hourly` quantity the dual-accuracy comparison reads.  `persist` writes it
#: for load buses only (D5), which is exactly the restriction `block_metrics`
#: applies to its price statistics: `ca2040_z4` prices the import bus and the
#: export buses degenerately and they are not system prices.
PRICE_QUANTITY = "price_usd_per_mwh"


def _empty_price_error(reason: str, columns=PRICE_ERROR_COLUMNS) -> pd.DataFrame:
    """A zero-row frame carrying ``reason``.

    The reason lives in ``frame.attrs["skip_reason"]`` -- a zero-row frame cannot
    carry a value *in* a column -- and the column is kept in the schema so a
    caller can concatenate skipped and non-skipped runs.  ``runcard`` prints the
    reason: silence in the dual-accuracy section has been mistaken for "no error"
    before.
    """
    frame = pd.DataFrame(columns=list(columns))
    frame.attrs["skip_reason"] = reason
    return frame


#: Per-block summary columns merged onto ``metrics.csv`` from the per-(bus, hour)
#: price-error decomposition. Load-bus max and RMS are the reportable numbers
#: (D5); the all-bus max is kept because it is a *constant* on ``ca2040_z4``
#: (a flat 56-61 $/MWh of degenerate import/export duals), which is the fact that
#: makes `history.price_error` useless as a stopping test in its current form.
PRICE_ERROR_SUMMARY_COLUMNS = (
    "price_error_load_max_usd_per_mwh",
    "price_error_load_rms_usd_per_mwh",
    "price_error_all_bus_max_usd_per_mwh",
    "price_error_n_bus_hours",
    # Against the LP on the SAME block: the pure ADMM dual error, with the
    # blocking effect divided out. These are the dual-accuracy numbers; the three
    # above are distances to the reference solve and are dominated by blocking.
    "price_error_load_max_vs_block_lp_usd_per_mwh",
    "price_error_load_rms_vs_block_lp_usd_per_mwh",
)


#: Sentinel for a null ``draw`` in a join key: pandas will not match NaN to NaN
#: in a merge, and an as-built run writes ``draw = None`` on every row.
_NO_DRAW = "__none__"


def _with_draw_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``_draw``: ``draw`` as a string, nulls collapsed to a sentinel.

    A merge never matches null to null, and an as-built run writes ``draw = None``
    on every row, so the key has to be a string. ``hourly.parquet`` types the
    column as a **nullable** ``Int32`` (``pa.int32(), nullable=True``), which
    refuses a string sentinel in place: cast to ``object`` before filling.
    """
    frame = frame.copy()
    if "draw" in frame.columns:
        draw = frame["draw"]
        frame["_draw"] = draw.astype("object").where(draw.notna(), _NO_DRAW).astype(str)
    else:
        frame["_draw"] = _NO_DRAW
    return frame


def _price_join_keys(blocks: pd.DataFrame, reference: pd.DataFrame) -> tuple[list, list]:
    """``(reference_key, same_block_key)`` for the price joins.

    ``hour`` is the hour **within a weather year**, so (bus, hour) alone collides
    across years and across outage draws: a two-year run would compare every
    year's block against the first year's reference. Both keys therefore carry
    ``year`` and, when the column is present, ``design_id``.

    ``draw`` is on the same-block key always (both sides are block rows of the
    same run), but on the reference key only when the reference side actually
    carries draws: a run may solve one draw-independent reference for a
    multi-draw sweep, and keying on ``draw`` there would join nothing.
    """
    ref_key = ["year", "bus", "hour"]
    block_key = ["block_size", "year", "bus", "hour"]
    if "design_id" in blocks.columns and "design_id" in reference.columns:
        ref_key.append("design_id")
        block_key.append("design_id")
    if "draw" in reference.columns and reference["draw"].notna().any():
        ref_key.append("_draw")
    block_key.append("_draw")
    return ref_key, block_key


def _price_rows(run_dir, df, reference_bounds=None):
    """``(block_prices, reference_prices, skip_reason, reference_key)``.

    ``reference_prices`` is one row per key of the ``block_size == "reference"``
    solve; ``block_prices`` is every non-reference block's price row. The frames
    are ``None`` when the comparison cannot be made, with the reason as the third
    element -- this never raises.
    """
    run_dir = Path(run_dir)
    try:
        if not persist.has_artefact(run_dir, "hourly"):
            return None, None, (
                "no hourly artefact: set `output.save_hourly: carrier_bus` to record prices"
            ), None
        hourly = persist.read_combined(run_dir, "hourly")
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail a run
        logger.warning("could not read the hourly artefact for the price error: %s", exc)
        return None, None, f"could not read the hourly artefact: {exc}", None

    if hourly is None or hourly.empty or "quantity" not in hourly.columns:
        return None, None, "the hourly artefact has no `quantity` column", None
    prices = hourly[hourly["quantity"].astype(str) == PRICE_QUANTITY].copy()
    if prices.empty:
        return None, None, (
            f"no `{PRICE_QUANTITY}` rows in the hourly artefact: add it to "
            "`output.hourly_quantities`"
        ), None

    for column in ("block_size", "method", "bus", "carrier", "task_id", "design_id"):
        if column in prices.columns:
            prices[column] = prices[column].astype(str)
    prices = _with_draw_key(prices)
    # `carrier` marks the load buses only when `output.price_all_buses` is on;
    # with the flag off every persisted price row *is* a load bus (D5).
    prices["is_load_bus"] = prices.get(
        "carrier", pd.Series("", index=prices.index)
    ).ne(persist.NON_LOAD_BUS_CARRIER)

    reference = prices[prices["block_size"] == "reference"]
    if reference.empty:
        return None, None, (
            "no `block_size == reference` rows: the run has no reference solve to "
            "compare prices against (`selection.reference: none`)"
        ), None
    # A run may hold a reference row for more than one method; the LP is the
    # reference by construction, so prefer it.
    if "method" in reference.columns and reference["method"].nunique() > 1:
        lp = reference[reference["method"] == "lp"]
        if not lp.empty:
            reference = lp

    blocks = prices[prices["block_size"] != "reference"]
    ref_key, _ = _price_join_keys(blocks, reference)
    reference = reference.drop_duplicates(subset=ref_key)[
        ref_key + ["value", "is_load_bus"]
    ].rename(columns={"value": "reference_price"})

    if reference_bounds is not None:
        lo, hi = reference_bounds
        reference = reference[(reference["hour"] >= lo) & (reference["hour"] < hi)]
        prices = prices[(prices["hour"] >= lo) & (prices["hour"] < hi)]
        blocks = prices[prices["block_size"] != "reference"]

    # Only tasks that succeeded: an `infeasible` ADMM row has no usable duals and
    # must not be averaged into a dual-accuracy table.
    if df is not None and not df.empty and {"task_id", "status"} <= set(df.columns):
        ok = set(df.loc[df["status"].astype(str) == "ok", "task_id"].astype(str))
        if "task_id" in blocks.columns:
            blocks = blocks[blocks["task_id"].isin(ok)]

    if blocks.empty:
        return None, None, "no non-reference block prices to compare", None
    return blocks, reference, None, ref_key


def price_error_rows(run_dir, df, reference_bounds=None) -> pd.DataFrame:
    """The per-(bus, hour) dual-error decomposition of every ADMM block.

    Columns: ``task_id, method, block_size, block_start_hour, bus, hour,
    is_load_bus, lp_price, admm_price, delta_price`` -- one row per bus and hour
    of every ADMM block that shares (bus, hour) with the reference LP.  Written
    to ``price_error.parquet`` by :func:`aggregate`; the block-level summary the
    run card shows is :func:`price_error_block_summary` of this frame.

    ADMM blocks only: the column names say ``lp_price`` / ``admm_price``, and the
    blocked-*LP* comparison against the reference is a blocking error, not a dual
    error (``deviation_vs_reference`` owns that).  Returns an empty frame -- never
    raises -- with the reason in ``attrs["skip_reason"]``.
    """
    blocks, reference, reason, ref_key = _price_rows(run_dir, df, reference_bounds)
    if blocks is None:
        return _empty_price_error(reason, columns=persist.PRICE_ERROR_COLUMNS)
    _, block_key = _price_join_keys(blocks, reference)

    admm = blocks[blocks["method"] == "admm"] if "method" in blocks.columns else blocks
    # The LP solved on the *same* block, if the run has one: the control that
    # separates the ADMM dual error from the blocking error. Keyed on the block's
    # full identity (block size, year, draw, design) -- `hour` alone repeats every
    # weather year and every outage draw.
    block_lp = (
        blocks[blocks["method"] == "lp"] if "method" in blocks.columns else blocks.iloc[:0]
    )
    block_lp = block_lp.drop_duplicates(subset=block_key)[
        block_key + ["value"]
    ].rename(columns={"value": "block_lp_price"})
    if admm.empty:
        return _empty_price_error(
            "no ADMM block prices: `price_error.parquet` compares ADMM duals against "
            "the reference LP",
            columns=persist.PRICE_ERROR_COLUMNS,
        )

    joined = admm.merge(reference.drop(columns=["is_load_bus"]), on=ref_key, how="inner")
    joined = joined.merge(block_lp, on=block_key, how="left")
    if joined.empty:
        return _empty_price_error(
            "no (bus, hour) pair is shared between the reference solve and any ADMM block",
            columns=persist.PRICE_ERROR_COLUMNS,
        )

    out = pd.DataFrame(
        {
            "task_id": joined["task_id"].astype(str),
            "method": joined["method"].astype(str),
            "block_size": joined["block_size"].astype(str),
            # The block's first hour, so a heat map can be cut per block without
            # joining back to metrics.csv.
            "block_start_hour": joined.groupby("task_id")["hour"].transform("min").astype(int),
            "bus": joined["bus"].astype(str),
            "hour": joined["hour"].astype(int),
            "is_load_bus": joined["is_load_bus"].astype(bool),
            "lp_price": pd.to_numeric(joined["reference_price"], errors="coerce"),
            "admm_price": pd.to_numeric(joined["value"], errors="coerce"),
            "block_lp_price": pd.to_numeric(joined["block_lp_price"], errors="coerce"),
        }
    )
    out["delta_price"] = out["admm_price"] - out["lp_price"]
    out["delta_price_vs_block_lp"] = out["admm_price"] - out["block_lp_price"]
    out = out.reindex(columns=list(persist.PRICE_ERROR_COLUMNS))
    return out.sort_values(["block_size", "block_start_hour", "bus", "hour"]).reset_index(
        drop=True
    )


def price_error_block_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-task summary of :func:`price_error_rows`: load-bus max / RMS, all-bus max.

    One row per ``task_id``, so it merges straight onto ``metrics.csv``.
    """
    columns = ["task_id", "method", "block_size", "block_start_hour"] + list(
        PRICE_ERROR_SUMMARY_COLUMNS
    )
    if rows is None or rows.empty:
        return pd.DataFrame(columns=columns)

    out = []
    for task_id, group in rows.groupby("task_id", sort=True):
        load = group[group["is_load_bus"]]
        delta = pd.to_numeric(load["delta_price"], errors="coerce").abs()
        all_bus = pd.to_numeric(group["delta_price"], errors="coerce").abs()
        block = pd.to_numeric(
            load.get("delta_price_vs_block_lp", pd.Series(dtype=float)), errors="coerce"
        ).abs().dropna()
        out.append(
            {
                "task_id": str(task_id),
                "method": str(group["method"].iloc[0]),
                "block_size": str(group["block_size"].iloc[0]),
                "block_start_hour": int(group["block_start_hour"].iloc[0]),
                "price_error_load_max_usd_per_mwh": float(delta.max())
                if not delta.empty
                else float("nan"),
                "price_error_load_rms_usd_per_mwh": float(np.sqrt((delta**2).mean()))
                if not delta.empty
                else float("nan"),
                "price_error_all_bus_max_usd_per_mwh": float(all_bus.max())
                if not all_bus.empty
                else float("nan"),
                "price_error_n_bus_hours": len(group),
                "price_error_load_max_vs_block_lp_usd_per_mwh": float(block.max())
                if not block.empty
                else float("nan"),
                "price_error_load_rms_vs_block_lp_usd_per_mwh": float(
                    np.sqrt((block**2).mean())
                )
                if not block.empty
                else float("nan"),
            }
        )
    return pd.DataFrame(out, columns=columns)


def price_error_vs_reference(run_dir, df, reference_bounds=None) -> pd.DataFrame:
    """Per (method, block_size, year, draw, block) max/mean |price - reference price|.

    Joins each block's ``price_usd_per_mwh`` rows from the combined ``hourly``
    parquet against the ``block_size == "reference"`` row's rows on
    (year, draw, bus, hour) -- ``hour`` is the hour *within a weather year*, so
    (bus, hour) alone collides across years and draws -- restricted to load buses
    (which is all ``persist`` writes, D5).  Returns an
    empty frame -- never raises -- when the reference row, the hourly artefact or
    the price quantity is absent, with the reason in ``attrs["skip_reason"]``.

    Computed here, at aggregate time from persisted hourly prices, rather than
    inside the solve: the reference block and the blocked solves are separate
    SLURM array tasks in an arbitrary order, so no solve can see the reference.
    """
    blocks, reference, reason, ref_key = _price_rows(run_dir, df, reference_bounds)
    if blocks is None:
        return _empty_price_error(reason)
    reference = reference.drop(columns=["is_load_bus"])
    # Load buses only, matching every other reported price statistic (D5).
    blocks = blocks[blocks["is_load_bus"]]
    if blocks.empty:
        return _empty_price_error("no load-bus block prices to compare")

    # One block per row: `block_index` repeats across weather years and outage
    # draws, so the grouping carries both -- exactly like the join key.
    key = [
        c
        for c in ("method", "block_size", "year", "draw", "design_id", "block_index")
        if c in blocks.columns
    ]
    rows = []
    # `observed=True`: the parquet's string columns are dictionary-encoded, so they
    # arrive as pandas Categoricals and the default would emit a row per unobserved
    # (method, block_size, ...) combination. `dropna=False` keeps `draw = None`.
    for values, group in blocks.groupby(key, sort=True, observed=True, dropna=False):
        joined = group.merge(reference, on=ref_key, how="inner")
        if joined.empty:
            continue
        error = (
            pd.to_numeric(joined["value"], errors="coerce")
            - pd.to_numeric(joined["reference_price"], errors="coerce")
        ).abs()
        row = dict(zip(key, values if isinstance(values, tuple) else (values,)))
        row.update(
            {
                "n_hours": int(joined["hour"].nunique()),
                "max_abs_price_error": float(error.max()),
                "mean_abs_price_error": float(error.mean()),
                "reference_mean_price": float(
                    pd.to_numeric(joined["reference_price"], errors="coerce").mean()
                ),
                "skip_reason": None,
            }
        )
        rows.append(row)
    if not rows:
        return _empty_price_error(
            "no (bus, hour) pair is shared between the reference solve and any block"
        )
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=[c for c in PRICE_ERROR_COLUMNS if c in frame.columns])


# ---------------------------------------------------------------------------
# Planning metrics (``mode: plan``)
# ---------------------------------------------------------------------------


def capacity_by_carrier(result) -> dict[str, float]:
    """Total designed capacity per carrier label, summed over parameters."""
    labels = (result.meta or {}).get("carrier_labels") or {}
    out: dict[str, float] = {}
    for param, values in (result.parameters or {}).items():
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        names = labels.get(param)
        if names is None or len(names) != values.size:
            names = [param] * values.size
        for carrier, value in zip(names, values):
            out[str(carrier)] = out.get(str(carrier), 0.0) + float(value)
    return out


def _total(mapping) -> float:
    return float(sum(np.asarray(v, dtype=np.float64).sum() for v in (mapping or {}).values()))


def planning_metrics(result) -> dict[str, Any]:
    """The metrics row of a planning task, from a ``PlanningResult``.

    Annual quantities lead (spec section 6); ``*_raw`` are kept because they are
    what ``design.json`` records and what the equivalence test compares.
    """
    objective = result.objective or {}
    annualization = result.annualization or {}
    selection = result.selection or {}
    initial = (result.meta or {}).get("initial_parameters")

    total_capacity = _total(result.parameters)
    metrics: dict[str, Any] = {
        "objective_annual": objective.get("annual"),
        "capex_annual": objective.get("capex_annual"),
        "opex_annual": objective.get("opex_annual"),
        "emissions_tonnes_annual": objective.get("emissions_tonnes_annual"),
        "objective_raw": objective.get("raw"),
        "capex_raw": objective.get("capex_raw"),
        "opex_raw": objective.get("opex_raw"),
        "emissions_tonnes_raw": objective.get("emissions_tonnes_raw"),
        "lower_bound_raw": objective.get("lower_bound_raw"),
        "optimality_gap": objective.get("optimality_gap"),
        "total_capacity_mw": total_capacity,
        "capacity_mw_by_carrier": json.dumps(capacity_by_carrier(result), sort_keys=True),
        "total_hours": annualization.get("total_hours"),
        "sampled_hours": annualization.get("sampled_hours"),
        "coverage": annualization.get("coverage"),
        "annualization_factor": annualization.get("annualization_factor"),
        "year_factor": annualization.get("year_factor"),
        "n_blocks": len(selection.get("blocks") or []),
        "selection_strategy": selection.get("strategy"),
        "selection_seed": selection.get("seed"),
        "planning_method": result.method,
        "planning_preset": result.preset,
        "planning_kind": result.kind,
        "emissions_mode": (result.emissions or {}).get("mode"),
        "emissions_cap_applied": (result.emissions or {}).get("cap_applied"),
    }

    # Which iterate the design came from and why the loop stopped (empty for the
    # single-level presets, which have neither).
    solver = result.solver or {}
    metrics["design_selection"] = objective.get("design_selection")
    metrics["design_iteration"] = objective.get("design_iteration")
    metrics["num_iterations_completed"] = solver.get("num_iterations_completed")
    metrics["stopped_by"] = solver.get("stopped_by")

    bounds = (result.meta or {}).get("bounds") or {}
    metrics["min_capacity_mw"] = bounds.get("min_capacity_mw")
    metrics["min_storage_mw"] = bounds.get("min_storage_mw")
    metrics["rows_raised_by_floor"] = json.dumps(
        bounds.get("rows_raised_by_floor") or {}, sort_keys=True
    )
    if initial:
        metrics["capacity_added_mw"] = total_capacity - _total(initial)
    return metrics
