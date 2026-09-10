"""Per-run persistence: the hourly store, the ENS profile, the ADMM trace and
``system_static.json``.

This module owns **every** schema constant and every writer for the artefacts a
run produces beyond ``metrics.csv`` / ``CARD.md`` / ``design.json``, so a plot
never reconstructs a column name by hand and a test can import the constants.
See ``memory/plans/2026-09-09-plots-spec.md`` (WP-P1).

Nothing here is on by default except the ENS profile (which is tiny): a full
year of ``ca2040_z4`` hourly data is ~3 M rows, and an evaluation campaign over
thousands of outage draws must not write it at all.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

HOURLY_SCHEMA_VERSION = 1
SYSTEM_STATIC_SCHEMA_VERSION = 1

#: Long-format columns of ``hourly/<task_id>.parquet``.  One row per
#: (hour, quantity, carrier, bus, name).
HOURLY_COLUMNS = (
    "task_id",
    "design_id",
    "method",
    "block_size",
    "year",
    "block_index",
    "draw",
    "hour",
    "quantity",
    "carrier",
    "bus",
    "name",
    "value",
    "unit",
)

#: The vocabulary of ``quantity``.  ``output.hourly_quantities`` is validated
#: against this tuple.
HOURLY_QUANTITIES = (
    # In-state Generator nominal x dynamic (imports excluded via
    # ``index.import_mask``) plus StorageUnit power x power_availability.
    "available_capacity_mw",
    "dispatch_mw",  # Generator injection, >= 0
    "load_mw",  # gross demand before shedding, >= 0
    "unserved_mw",  # demand + power[0] (the metrics.py definition), >= 0
    "curtailment_mw",  # VRE available - dispatch, clipped at 0
    "storage_charge_mw",  # >= 0
    "storage_discharge_mw",  # >= 0
    "storage_soc_mwh",  # state of charge at the start of the hour
    "line_flow_mw",  # DirectedLine sink power, bus0 -> bus1 positive
    "price_usd_per_mwh",  # LOAD BUSES ONLY (D5)
)

#: The physical unit of every quantity, written into the ``unit`` column.
QUANTITY_UNITS = {
    "available_capacity_mw": "MW",
    "dispatch_mw": "MW",
    "load_mw": "MW",
    "unserved_mw": "MW",
    "curtailment_mw": "MW",
    "storage_charge_mw": "MW",
    "storage_discharge_mw": "MW",
    "storage_soc_mwh": "MWh",
    "line_flow_mw": "MW",
    "price_usd_per_mwh": "$/MWh",
}

#: ``carrier`` marker on a ``price_usd_per_mwh`` row when ``output.price_all_buses``
#: is on.  With the flag off (the default) prices are written for load buses only
#: (D5) and the carrier stays empty, exactly as before; with it on every bus is
#: written and every consumer that wants the D5 restriction filters on
#: ``carrier != NON_LOAD_BUS_CARRIER``.  On ``ca2040_z4`` the extra buses are the
#: import bus and the export buses, whose duals are degenerate (a flat
#: 56-61 $/MWh) and are not system prices -- which is exactly why they must never
#: be pooled into a price statistic, and exactly why they are worth recording
#: separately when diagnosing ADMM duals.
LOAD_BUS_CARRIER = "load_bus"
NON_LOAD_BUS_CARRIER = "other_bus"

#: Columns of ``price_error.parquet``: the per-(bus, hour) decomposition of an
#: ADMM block's dual error, written by ``ra aggregate``.
#:
#: **Two references, and they answer different questions.** ``lp_price`` is the
#: *reference* solve's price (the whole window as one LP) and ``delta_price`` is
#: the distance to it -- but that distance is dominated by *blocking* error, not
#: by the solver: the blocked **LP** is just as far from the reference as ADMM is
#: (measured 2026-09-09 on ca2040_z4: 8.36 vs 8.30 $/MWh max on 24 h blocks).
#: ``block_lp_price`` is the LP solved on the *same block*, so
#: ``delta_price_vs_block_lp`` is the pure ADMM dual error -- the quantity a
#: price-based convergence criterion has to drive down, and the one the gradient
#: planner's duals live or die by. It is NaN when the run has no LP row for that
#: block.
PRICE_ERROR_COLUMNS = (
    "task_id",
    "method",
    "block_size",
    "block_start_hour",
    "bus",
    "hour",
    "is_load_bus",
    "lp_price",
    "admm_price",
    "delta_price",
    "block_lp_price",
    "delta_price_vs_block_lp",
)

#: Quantities whose all-zero rows are dropped.  Never drop a zero row of
#: ``dispatch_mw`` or ``available_capacity_mw``: a carrier that is off for a
#: whole week must still stack as zero.
ZERO_SUPPRESSED_QUANTITIES = ("unserved_mw", "curtailment_mw")

#: Columns of ``ens_profile/<task_id>.parquet``.
#:
#: ``day_of_year`` (1-based) and ``hour_of_day`` (0-based) are derived purely
#: from the absolute in-year hour index -- ``hour // 24 + 1`` and ``hour % 24``
#: -- which for the CA2040 exports is a **UTC** hour.  The shipped benchmark
#: window deliberately starts at hour 7 so that blocks begin at Pacific
#: midnight, so ``hour_of_day`` runs 7 hours ahead of local time and hour 0 of
#: a day is 17:00 the previous day in California.  Convert before labelling a
#: figure with a local clock hour.
ENS_PROFILE_COLUMNS = (
    "task_id",
    "design_id",
    "method",
    "block_size",
    "year",
    "draw",
    "hour",
    "day_of_year",
    "hour_of_day",
    "bus",
    "ens_mwh",
)

#: Columns of ``admm_trace/<task_id>.parquet``.
ADMM_TRACE_COLUMNS = (
    "task_id",
    "method",
    "block_size",
    "year",
    "block_index",
    "draw",
    "iteration",
    "objective",
    "primal_power",
    "primal_phase",
    "dual_power",
    "dual_phase",
    "primal_tol",
    "dual_tol",
    "price_error",
    "rho_power",
    "power_unit",
    "cost_unit",
)

def price_error_schema() -> pa.Schema:
    """The explicit arrow schema of ``price_error.parquet``."""
    return pa.schema(
        [
            pa.field("task_id", _STR),
            pa.field("method", _STR),
            pa.field("block_size", _STR),
            pa.field("block_start_hour", pa.int32()),
            pa.field("bus", _STR),
            pa.field("hour", pa.int32()),
            pa.field("is_load_bus", pa.bool_()),
            pa.field("lp_price", pa.float64()),
            pa.field("admm_price", pa.float64()),
            pa.field("delta_price", pa.float64()),
            pa.field("block_lp_price", pa.float64(), nullable=True),
            pa.field("delta_price_vs_block_lp", pa.float64(), nullable=True),
        ]
    )


def write_price_error(run_dir, frame: pd.DataFrame) -> Path | None:
    """Write ``price_error.parquet``; ``None`` for an empty frame."""
    if run_dir is None or frame is None or frame.empty:
        return None
    frame = frame.reindex(columns=list(PRICE_ERROR_COLUMNS))
    frame["block_start_hour"] = frame["block_start_hour"].astype("int32")
    frame["hour"] = frame["hour"].astype("int32")
    frame["is_load_bus"] = frame["is_load_bus"].astype(bool)
    for col in ("lp_price", "admm_price", "delta_price", "block_lp_price",
                "delta_price_vs_block_lp"):
        frame[col] = frame[col].astype("float64")
    for col in ("task_id", "method", "block_size", "bus"):
        frame[col] = frame[col].astype(str)
    return write_parquet_atomic(frame, Path(run_dir) / "price_error.parquet", price_error_schema())


#: Sub-directories that hold one parquet per task and are concatenated into a
#: single sibling file by ``ra aggregate`` (D3).
COMBINABLE_DIRS = ("hourly", "ens_profile", "admm_trace")

_STR = pa.dictionary(pa.int32(), pa.string())


def hourly_schema() -> pa.Schema:
    """The explicit arrow schema of ``hourly.parquet`` (dtypes are never inferred)."""
    return pa.schema(
        [
            pa.field("task_id", _STR),
            pa.field("design_id", _STR),
            pa.field("method", _STR),
            pa.field("block_size", _STR),
            pa.field("year", pa.int16()),
            pa.field("block_index", pa.int32()),
            pa.field("draw", pa.int32(), nullable=True),
            pa.field("hour", pa.int32()),
            pa.field("quantity", _STR),
            pa.field("carrier", _STR),
            pa.field("bus", _STR),
            pa.field("name", _STR),
            pa.field("value", pa.float64()),
            pa.field("unit", _STR),
        ]
    )


def ens_profile_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("task_id", _STR),
            pa.field("design_id", _STR),
            pa.field("method", _STR),
            pa.field("block_size", _STR),
            pa.field("year", pa.int16()),
            pa.field("draw", pa.int32(), nullable=True),
            pa.field("hour", pa.int32()),
            pa.field("day_of_year", pa.int16()),
            pa.field("hour_of_day", pa.int8()),
            pa.field("bus", _STR),
            pa.field("ens_mwh", pa.float64()),
        ]
    )


def admm_trace_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("task_id", _STR),
            pa.field("method", _STR),
            pa.field("block_size", _STR),
            pa.field("year", pa.int16()),
            pa.field("block_index", pa.int32()),
            pa.field("draw", pa.int32(), nullable=True),
            pa.field("iteration", pa.int32()),
            pa.field("objective", pa.float64()),
            pa.field("primal_power", pa.float64()),
            pa.field("primal_phase", pa.float64()),
            pa.field("dual_power", pa.float64()),
            pa.field("dual_phase", pa.float64()),
            pa.field("primal_tol", pa.float64()),
            pa.field("dual_tol", pa.float64()),
            pa.field("price_error", pa.float64(), nullable=True),
            pa.field("rho_power", pa.float64()),
            pa.field("power_unit", pa.float64()),
            pa.field("cost_unit", pa.float64()),
        ]
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def output_options(cfg: dict | None) -> dict:
    return dict((cfg or {}).get("output") or {})


def price_all_buses(cfg: dict | None) -> bool:
    """``output.price_all_buses``: write prices at every bus, not only load buses."""
    return bool(output_options(cfg).get("price_all_buses", False))


def hourly_mode(cfg: dict | None) -> str:
    return str(output_options(cfg).get("save_hourly", "none"))


def resolve_quantities(cfg: dict | None) -> tuple[str, ...]:
    """The quantities ``output.hourly_quantities`` asks for, in schema order."""
    requested = output_options(cfg).get("hourly_quantities", "all")
    if requested is None or (isinstance(requested, str) and str(requested) == "all"):
        return tuple(HOURLY_QUANTITIES)
    wanted = {str(q) for q in requested}
    return tuple(q for q in HOURLY_QUANTITIES if q in wanted)


def write_parquet_atomic(frame: pd.DataFrame, path: Path, schema: pa.Schema) -> Path:
    """Serialise, write ``<path>.tmp``, then ``os.replace`` it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def _unit_scales(loaded) -> tuple[float, float]:
    meta = getattr(loaded, "meta", None) or {}
    return (
        float(meta.get("power_unit", 1.0) or 1.0),
        float(meta.get("cost_unit", 1.0) or 1.0),
    )


def _as_2d(value, shape) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, dtype=np.float64), shape)


def bus_name_table(loaded, devices: Sequence) -> np.ndarray:
    """``node index -> bus name``, from the SystemIndex's per-row bus labels.

    ``SystemIndex.bus[cls]`` is the source-table bus column (``bus1`` for a
    ``DirectedLine``, which is also the terminal ``sink_terminal`` points at),
    so pairing it with the device's terminals recovers the name of every node
    that carries a device.  Nodes with no device keep a positional label.
    """
    index = getattr(loaded, "index", None)
    n_nodes = getattr(getattr(loaded, "network", None), "num_nodes", None)
    if n_nodes is None:
        n_nodes = 0
        if index is not None:
            for cls_name, i in index.device_index.items():
                if i < len(devices):
                    n_nodes = max(n_nodes, int(np.max(np.asarray(devices[i].terminals))) + 1)
    names = np.array([f"node{i}" for i in range(int(n_nodes))], dtype=object)
    if index is None:
        return names
    for cls_name, i in index.device_index.items():
        if i >= len(devices):
            continue
        buses = np.asarray(index.bus.get(cls_name, []), dtype=object).ravel()
        if buses.size == 0:
            continue
        device = devices[i]
        terminals = getattr(device, "sink_terminal", None)
        if terminals is None:
            terminals = getattr(device, "terminal", None)
        if terminals is None:
            continue
        terminals = np.asarray(terminals).ravel()
        if terminals.size != buses.size:
            continue
        for node, name in zip(terminals, buses):
            node = int(node)
            if 0 <= node < names.size:
                names[node] = str(name)
    return names


@dataclass(frozen=True)
class LoadShortfall:
    """One ``Load`` device's demand and unserved energy over a block."""

    device_index: int
    terminals: np.ndarray  # (rows,) node index of each load row
    demand: np.ndarray  # (rows, T) gross demand, solver units
    shortfall: np.ndarray  # (rows, T) demand + power[0], solver units


def load_shortfall(devices: Sequence, power, groups: dict[str, list[int]]) -> list[LoadShortfall]:
    """Per-``Load``-device demand and unserved energy, in *solver* units.

    ``metrics.block_metrics`` and :func:`build_ens_profile_frame` both consume
    this, so the two can never disagree about what "unserved" means.
    """
    out: list[LoadShortfall] = []
    for i in groups.get("Load", []):
        device = devices[i]
        demand = np.asarray(device.load, dtype=np.float64) * np.asarray(
            device.nominal_capacity, dtype=np.float64
        )
        injection = np.asarray(power[i][0], dtype=np.float64)
        demand = _as_2d(demand, injection.shape)
        out.append(
            LoadShortfall(
                device_index=i,
                terminals=np.asarray(device.terminal).ravel(),
                demand=demand,
                shortfall=demand + injection,
            )
        )
    return out


# ---------------------------------------------------------------------------
# hourly/<task_id>.parquet
# ---------------------------------------------------------------------------


def _long_rows(
    values: np.ndarray,
    *,
    carriers: Sequence[str],
    buses: Sequence[str],
    names: Sequence[str],
    quantity: str,
    hours: np.ndarray,
) -> dict[str, np.ndarray]:
    """Melt a ``(rows, T)`` matrix into the long-format column arrays."""
    n_rows, n_hours = values.shape
    return {
        "hour": np.tile(hours, n_rows),
        "quantity": np.repeat(quantity, n_rows * n_hours),
        "carrier": np.repeat(np.asarray(carriers, dtype=object), n_hours),
        "bus": np.repeat(np.asarray(buses, dtype=object), n_hours),
        "name": np.repeat(np.asarray(names, dtype=object), n_hours),
        "value": values.reshape(-1),
        "unit": np.repeat(QUANTITY_UNITS[quantity], n_rows * n_hours),
    }


def _aggregate_by_key(values: np.ndarray, keys: list[tuple[str, str]]):
    """Sum rows that share a ``(carrier, bus)`` key; returns (matrix, keys)."""
    order: dict[tuple[str, str], int] = {}
    for key in keys:
        if key not in order:
            order[key] = len(order)
    out = np.zeros((len(order), values.shape[1]), dtype=np.float64)
    for row, key in enumerate(keys):
        out[order[key], :] += values[row, :]
    return out, list(order)


def build_hourly_frame(
    loaded,
    devices: Sequence,
    outcome,
    block,
    *,
    task,
    quantities: Sequence[str] = HOURLY_QUANTITIES,
    all_buses: bool = False,
) -> pd.DataFrame:
    """The long-format hourly table of one solved block, in physical units.

    ``hour`` is the **absolute hour index inside the weather year**
    (``block.start + t``), so the blocks of one year concatenate into one
    series.  Values carry ``LoadOptions.power_unit`` / ``cost_unit`` undone, as
    ``metrics.block_metrics`` does.
    """
    from . import metrics as metrics_mod

    index = getattr(loaded, "index", None)
    power = metrics_mod.numpyify(outcome.power)
    local_vars = metrics_mod.numpyify(outcome.local_variables)
    prices = metrics_mod._to_numpy(outcome.prices)
    groups = metrics_mod.device_groups(devices)
    n = len(devices)
    power, local_vars = power[:n], local_vars[:n]

    power_unit, cost_unit = _unit_scales(loaded)
    hours = np.arange(block.start, block.start + int(block.hours), dtype=np.int64)
    wanted = [q for q in quantities if q in set(HOURLY_QUANTITIES)]
    bus_names = bus_name_table(loaded, devices)

    chunks: list[dict[str, np.ndarray]] = []

    def emit(values, keys, quantity, *, names=None, scale=power_unit):
        if values.size == 0:
            return
        values = np.asarray(values, dtype=np.float64) * scale
        if names is None:
            values, keys = _aggregate_by_key(values, keys)
            names = [""] * len(keys)
        if quantity in ZERO_SUPPRESSED_QUANTITIES:
            keep = [i for i in range(values.shape[0]) if np.count_nonzero(values[i, :])]
            if not keep:
                return
            values = values[keep, :]
            keys = [keys[i] for i in keep]
            names = [names[i] for i in keep]
        chunks.append(
            _long_rows(
                values,
                carriers=[k[0] for k in keys],
                buses=[k[1] for k in keys],
                names=names,
                quantity=quantity,
                hours=hours,
            )
        )

    # --- Generators -------------------------------------------------------
    for i in groups.get("Generator", []):
        device = devices[i]
        dispatch = np.asarray(power[i][0], dtype=np.float64)
        n_rows = dispatch.shape[0]
        carriers = metrics_mod._carriers(index, "Generator", device, n_rows)
        carriers = [""] * n_rows if carriers is None else [str(c) for c in carriers]
        buses = _row_buses(index, "Generator", device, n_rows, bus_names, "terminal")
        keys = list(zip(carriers, buses))
        available = _as_2d(
            np.asarray(device.nominal_capacity, dtype=np.float64)
            * np.asarray(device.dynamic_capacity, dtype=np.float64),
            dispatch.shape,
        )
        if "dispatch_mw" in wanted:
            emit(dispatch, keys, "dispatch_mw")
        if "available_capacity_mw" in wanted:
            # One membership everywhere (decision 2026-09-09): available
            # capacity is in-state generators only; import rows are excluded
            # here exactly as they are in ``metrics.available_mw_min``.
            import_mask = getattr(index, "import_mask", None) if index is not None else None
            if import_mask is not None and np.size(import_mask) == n_rows:
                in_state = ~np.asarray(import_mask, dtype=bool)
            else:
                in_state = np.ones(n_rows, dtype=bool)
            emit(
                available[in_state, :],
                [k for k, m in zip(keys, in_state) if m],
                "available_capacity_mw",
            )
        if "curtailment_mw" in wanted:
            vre = getattr(index, "vre_mask", None) if index is not None else None
            if vre is not None and np.size(vre) == n_rows:
                mask = np.asarray(vre, dtype=bool)
                curtail = np.maximum(available - dispatch, 0.0)[mask, :]
                emit(curtail, [k for k, m in zip(keys, mask) if m], "curtailment_mw")

    # --- Loads ------------------------------------------------------------
    shortfalls = load_shortfall(devices, power, groups)
    for entry in shortfalls:
        device = devices[entry.device_index]
        n_rows = entry.demand.shape[0]
        carriers = metrics_mod._carriers(index, "Load", device, n_rows)
        carriers = ["load"] * n_rows if carriers is None else [str(c) for c in carriers]
        buses = [str(bus_names[int(t)]) for t in entry.terminals]
        keys = list(zip(carriers, buses))
        if "load_mw" in wanted:
            emit(entry.demand, keys, "load_mw")
        if "unserved_mw" in wanted:
            emit(entry.shortfall, keys, "unserved_mw")

    # --- Storage ----------------------------------------------------------
    for i in groups.get("StorageUnit", []):
        device = devices[i]
        state = local_vars[i]
        n_rows = int(np.asarray(device.power_capacity).reshape(-1).size)
        carriers = metrics_mod._carriers(index, "StorageUnit", device, n_rows)
        carriers = [""] * n_rows if carriers is None else [str(c) for c in carriers]
        buses = _row_buses(index, "StorageUnit", device, n_rows, bus_names, "terminal")
        keys = list(zip(carriers, buses))
        if "available_capacity_mw" in wanted:
            available = _as_2d(
                np.asarray(device.power_capacity, dtype=np.float64)
                * np.asarray(device.power_availability, dtype=np.float64),
                (n_rows, int(block.hours)),
            )
            emit(available, keys, "available_capacity_mw")
        energy, charge, discharge = _storage_state(state)
        if charge is not None and "storage_charge_mw" in wanted:
            emit(charge, keys, "storage_charge_mw")
        if discharge is not None and "storage_discharge_mw" in wanted:
            emit(discharge, keys, "storage_discharge_mw")
        if energy is not None and "storage_soc_mwh" in wanted:
            # Column 0 is the level at the start of hour 0; the last column is
            # the level at the end of the block, which is not an hourly value.
            emit(np.asarray(energy, dtype=np.float64)[:, :-1], keys, "storage_soc_mwh")

    # --- Lines (device resolution: few rows, and O7 needs them) -----------
    if "line_flow_mw" in wanted:
        for cls_name in ("DirectedLine", "DCLine", "ACLine", "PowerLine"):
            for i in groups.get(cls_name, []):
                device = devices[i]
                flow = np.asarray(power[i][1], dtype=np.float64)
                n_rows = flow.shape[0]
                carriers = metrics_mod._carriers(index, cls_name, device, n_rows)
                carriers = [""] * n_rows if carriers is None else [str(c) for c in carriers]
                buses = _row_buses(index, cls_name, device, n_rows, bus_names, "sink_terminal")
                row_names = _row_names(index, cls_name, device, n_rows)
                emit(flow, list(zip(carriers, buses)), "line_flow_mw", names=row_names)

    # --- Prices (load buses only unless `output.price_all_buses`; D5) ------
    if "price_usd_per_mwh" in wanted and prices is not None and prices.size:
        load_nodes = load_bus_nodes(shortfalls)
        if all_buses:
            # Every bus, each row labelled by whether it carries load, so the D5
            # restriction is a filter rather than a lost distinction.
            nodes = list(range(int(np.asarray(prices).shape[0])))
            carriers = [
                LOAD_BUS_CARRIER if node in load_nodes else NON_LOAD_BUS_CARRIER
                for node in nodes
            ]
        else:
            nodes = sorted(load_nodes)
            carriers = ["" for _ in nodes]
        if nodes:
            values = np.asarray(prices, dtype=np.float64)[nodes, : int(block.hours)]
            keys = [(carrier, str(bus_names[node])) for carrier, node in zip(carriers, nodes)]
            emit(values, keys, "price_usd_per_mwh", names=[""] * len(nodes), scale=cost_unit)

    if not chunks:
        return _empty_hourly_frame()

    frame = pd.DataFrame({key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]})
    frame.insert(0, "task_id", str(task.task_id))
    frame.insert(1, "design_id", str(getattr(task, "design_id", "asbuilt")))
    frame.insert(2, "method", str(task.method))
    frame.insert(3, "block_size", str(task.block_size))
    frame.insert(4, "year", np.int16(block.year))
    frame.insert(5, "block_index", np.int32(block.index))
    draw = getattr(task, "draw", None)
    frame.insert(6, "draw", pd.array([draw] * len(frame), dtype="Int32"))
    frame["hour"] = frame["hour"].astype("int32")
    frame["value"] = frame["value"].astype("float64")
    for col in ("quantity", "carrier", "bus", "name", "unit"):
        frame[col] = frame[col].astype(str)
    return frame[list(HOURLY_COLUMNS)]


def _empty_hourly_frame() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="object") for c in HOURLY_COLUMNS})
    frame["year"] = frame["year"].astype("int16")
    frame["block_index"] = frame["block_index"].astype("int32")
    frame["draw"] = pd.array([], dtype="Int32")
    frame["hour"] = frame["hour"].astype("int32")
    frame["value"] = frame["value"].astype("float64")
    return frame


def _storage_state(state):
    """``(energy, charge, discharge)`` from a StorageUnitVariable or a 3-list."""
    if state is None:
        return None, None, None
    energy = getattr(state, "energy", None)
    charge = getattr(state, "charge", None)
    discharge = getattr(state, "discharge", None)
    if isinstance(state, (list, tuple)) and len(state) >= 3:
        energy = state[0] if energy is None else energy
        charge = state[1] if charge is None else charge
        discharge = state[2] if discharge is None else discharge
    return energy, charge, discharge


def _row_buses(index, cls_name, device, n_rows, bus_names, terminal_attr) -> list[str]:
    if index is not None:
        buses = np.asarray(index.bus.get(cls_name, []), dtype=object).ravel()
        if buses.size == n_rows:
            return [str(b) for b in buses]
    terminals = getattr(device, terminal_attr, None)
    if terminals is None:
        terminals = getattr(device, "terminal", None)
    if terminals is None:
        return [""] * n_rows
    terminals = np.asarray(terminals).ravel()
    return [str(bus_names[int(t)]) if int(t) < bus_names.size else "" for t in terminals]


def _row_names(index, cls_name, device, n_rows) -> list[str]:
    if index is not None and cls_name in getattr(index, "names", {}):
        names = [str(n) for n in index.names[cls_name]]
        if len(names) == n_rows:
            return names
    name = getattr(device, "name", None)
    if name is not None and np.size(name) == n_rows:
        return [str(n) for n in np.asarray(name).ravel()]
    return [f"{cls_name}_{i}" for i in range(n_rows)]


def load_bus_nodes(shortfalls: Sequence[LoadShortfall]) -> set[int]:
    """Nodes carrying non-zero demand in this block (D5)."""
    nodes: set[int] = set()
    for entry in shortfalls:
        for row, node in enumerate(entry.terminals):
            if np.any(entry.demand[row, :] != 0.0):
                nodes.add(int(node))
    return nodes


def write_hourly(run_dir, task, cfg: dict, loaded, devices, outcome, block) -> Path | None:
    """Write ``hourly/<task_id>.parquet``; ``None`` when the flag is off."""
    if run_dir is None or hourly_mode(cfg) == "none":
        return None
    quantities = resolve_quantities(cfg)
    if not quantities:
        return None
    frame = build_hourly_frame(
        loaded,
        devices,
        outcome,
        block,
        task=task,
        quantities=quantities,
        all_buses=price_all_buses(cfg),
    )
    path = Path(run_dir) / "hourly" / f"{task.task_id}.parquet"
    return write_parquet_atomic(frame, path, hourly_schema())


# ---------------------------------------------------------------------------
# ens_profile/<task_id>.parquet
# ---------------------------------------------------------------------------


def build_ens_profile_frame(loaded, devices, outcome, block, *, task) -> pd.DataFrame:
    """Every (hour, bus) with a shortfall above ``metrics.SHORTFALL_TOL_MW``.

    ``day_of_year`` / ``hour_of_day`` are UTC-hour based (see
    :data:`ENS_PROFILE_COLUMNS`), not local clock time.
    """
    from . import metrics as metrics_mod

    power = metrics_mod.numpyify(outcome.power)[: len(devices)]
    groups = metrics_mod.device_groups(devices)
    power_unit, _ = _unit_scales(loaded)
    bus_names = bus_name_table(loaded, devices)

    rows: dict[tuple[int, str], float] = {}
    for entry in load_shortfall(devices, power, groups):
        for row, node in enumerate(entry.terminals):
            values = entry.shortfall[row, :] * power_unit
            hit = np.nonzero(values > metrics_mod.SHORTFALL_TOL_MW)[0]
            bus = str(bus_names[int(node)]) if int(node) < bus_names.size else ""
            for t in hit:
                key = (int(block.start + t), bus)
                rows[key] = rows.get(key, 0.0) + float(values[t])

    if not rows:
        frame = pd.DataFrame({c: pd.Series(dtype="object") for c in ENS_PROFILE_COLUMNS})
    else:
        ordered = sorted(rows)
        frame = pd.DataFrame(
            {
                "hour": [k[0] for k in ordered],
                "bus": [k[1] for k in ordered],
                "ens_mwh": [rows[k] for k in ordered],
            }
        )
        frame["task_id"] = str(task.task_id)
        frame["design_id"] = str(getattr(task, "design_id", "asbuilt"))
        frame["method"] = str(task.method)
        frame["block_size"] = str(task.block_size)
        frame["year"] = np.int16(block.year)
        frame["draw"] = pd.array([getattr(task, "draw", None)] * len(frame), dtype="Int32")
        frame["day_of_year"] = (frame["hour"] // 24 + 1).astype("int16")
        frame["hour_of_day"] = (frame["hour"] % 24).astype("int8")

    frame = frame.reindex(columns=list(ENS_PROFILE_COLUMNS))
    if frame.empty:
        frame["year"] = frame["year"].astype("int16")
        frame["draw"] = pd.array([], dtype="Int32")
        frame["day_of_year"] = frame["day_of_year"].astype("int16")
        frame["hour_of_day"] = frame["hour_of_day"].astype("int8")
    frame["hour"] = frame["hour"].astype("int32")
    frame["ens_mwh"] = frame["ens_mwh"].astype("float64")
    return frame


def write_ens_profile(run_dir, task, cfg: dict, loaded, devices, outcome, block) -> Path | None:
    if run_dir is None or not bool(output_options(cfg).get("save_ens_profile", True)):
        return None
    frame = build_ens_profile_frame(loaded, devices, outcome, block, task=task)
    path = Path(run_dir) / "ens_profile" / f"{task.task_id}.parquet"
    return write_parquet_atomic(frame, path, ens_profile_schema())


# ---------------------------------------------------------------------------
# admm_trace/<task_id>.parquet
# ---------------------------------------------------------------------------


def _history_list(history, name: str) -> list:
    return list(getattr(history, name, None) or [])


def build_admm_trace_frame(history, solver, loaded, block, *, task, every: int) -> pd.DataFrame:
    """One row per kept ADMM iteration.

    ``objective`` is in **solver units** (the system is scaled by
    ``power_unit`` / ``cost_unit``); the two unit columns are written so a plot
    can convert.  ``rho_power`` is the solver's *final* rho written as a
    constant column -- ``history`` carries no per-iteration rho, so with
    ``adapt_rho: false`` (our default) it is exact and otherwise it is only the
    last value.  Per-iteration ``max_imbalance_mw`` is not available at all:
    only the final state is evaluated.
    """
    power = _history_list(history, "power")
    n = len(power)
    every = max(1, int(every))
    keep = [i for i in range(n) if i % every == 0]
    if n and (n - 1) not in keep:
        keep.append(n - 1)

    power_unit, cost_unit = _unit_scales(loaded)

    def column(name: str) -> list[float]:
        values = _history_list(history, name)
        return [float(values[i]) if i < len(values) and values[i] is not None else float("nan")
                for i in keep]

    frame = pd.DataFrame(
        {
            "iteration": np.asarray(keep, dtype="int32"),
            "objective": column("objective"),
            "primal_power": column("power"),
            "primal_phase": column("phase"),
            "dual_power": column("dual_power"),
            "dual_phase": column("dual_phase"),
            "primal_tol": column("primal_tol"),
            "dual_tol": column("dual_tol"),
            "price_error": column("price_error"),
        }
    )
    frame["task_id"] = str(task.task_id)
    frame["method"] = str(task.method)
    frame["block_size"] = str(task.block_size)
    frame["year"] = np.int16(block.year)
    frame["block_index"] = np.int32(block.index)
    frame["draw"] = pd.array([getattr(task, "draw", None)] * len(frame), dtype="Int32")
    frame["rho_power"] = float(getattr(solver, "rho_power", float("nan")) or float("nan"))
    frame["power_unit"] = power_unit
    frame["cost_unit"] = cost_unit
    return frame.reindex(columns=list(ADMM_TRACE_COLUMNS))


def write_admm_trace(run_dir, task, cfg: dict, loaded, block, history, solver) -> Path | None:
    every = int(output_options(cfg).get("admm_trace_every", 0) or 0)
    if run_dir is None or every <= 0:
        return None
    frame = build_admm_trace_frame(history, solver, loaded, block, task=task, every=every)
    path = Path(run_dir) / "admm_trace" / f"{task.task_id}.parquet"
    return write_parquet_atomic(frame, path, admm_trace_schema())


# ---------------------------------------------------------------------------
# system_static.json
# ---------------------------------------------------------------------------


def _clean(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.integer, int)):
        return int(value)
    text = str(value)
    return None if text in ("nan", "None", "") else text


def build_system_static(loaded, dataset_dir: Path) -> dict:
    """The static picture of the system a plot needs without reloading it."""
    from zap.importers.wy_store import read_static

    static = read_static(Path(dataset_dir))
    index = getattr(loaded, "index", None)

    load_buses: list[str] = []
    if index is not None:
        load_buses = sorted({str(b) for b in np.asarray(index.bus.get("Load", [])).ravel()})

    buses_df = static["buses"]
    buses = []
    for name, row in buses_df.iterrows():
        buses.append(
            {
                "name": str(name),
                "x": _clean(row.get("x")),
                "y": _clean(row.get("y")),
                "has_load": str(name) in set(load_buses),
            }
        )

    carriers = {}
    for name, row in static["carriers"].iterrows():
        carriers[str(name)] = {
            "color": _clean(row.get("color")),
            "nice_name": _clean(row.get("nice_name")),
            "co2_emissions": _clean(row.get("co2_emissions")),
        }

    lines = []
    for name, row in static["links"].iterrows():
        lines.append(
            {
                "name": str(name),
                "bus0": _clean(row.get("bus0")),
                "bus1": _clean(row.get("bus1")),
                "carrier": _clean(row.get("carrier")),
                "capacity_mw": _clean(row.get("p_nom")),
                "length_km": _clean(row.get("length")),
            }
        )

    storage = []
    for name, row in static["storage_units"].iterrows():
        p_nom = _clean(row.get("p_nom"))
        hours = _clean(row.get("max_hours"))
        storage.append(
            {
                "name": str(name),
                "bus": _clean(row.get("bus")),
                "carrier": _clean(row.get("carrier")),
                "power_capacity_mw": p_nom,
                "energy_capacity_mwh": (
                    None if p_nom is None or hours is None else float(p_nom) * float(hours)
                ),
            }
        )

    meta = getattr(loaded, "meta", None) or {}
    return {
        "schema_version": SYSTEM_STATIC_SCHEMA_VERSION,
        "dataset": meta.get("dataset"),
        "buses": buses,
        "load_buses": load_buses,
        "carriers": carriers,
        "lines": lines,
        "storage": storage,
    }


def write_system_static(run_dir, loaded, cfg: dict) -> Path | None:
    """Write ``system_static.json`` once per run dir (write-if-absent).

    ``cfg`` is needed only to resolve the dataset directory the static CSVs
    live in; ``loaded.meta["dataset"]`` is a bare name, which does not resolve
    for a dataset given as an absolute path.
    """
    if run_dir is None:
        return None
    path = Path(run_dir) / "system_static.json"
    if path.exists():
        return path
    from .system import dataset_path

    try:
        payload = build_system_static(loaded, dataset_path(cfg))
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        logger.warning("could not write system_static.json: %s", exc)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)
    return path


def read_system_static(run_dir) -> dict | None:
    path = Path(run_dir) / "system_static.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Combination (D3)
# ---------------------------------------------------------------------------


def combine_parquet_dir(run_dir, name: str) -> Path | None:
    """Concatenate ``<run_dir>/<name>/*.parquet`` into ``<run_dir>/<name>.parquet``."""
    run_dir = Path(run_dir)
    directory = run_dir / name
    if not directory.is_dir():
        return None
    parts = sorted(p for p in directory.glob("*.parquet"))
    if not parts:
        return None
    table = pa.concat_tables([pq.read_table(p) for p in parts], promote_options="default")
    out = run_dir / f"{name}.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, out)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return out


def combine_outputs(run_dir, cfg: dict | None) -> list[Path]:
    """Combine every per-task parquet directory this run wrote (D3)."""
    if not bool(output_options(cfg).get("combine_hourly", True)):
        return []
    written = []
    for name in COMBINABLE_DIRS:
        path = combine_parquet_dir(run_dir, name)
        if path is not None:
            written.append(path)
    return written


def read_combined(run_dir, name: str, **read_kwargs) -> pd.DataFrame:
    """Read ``<name>.parquet`` if present, else the ``<name>/`` directory."""
    run_dir = Path(run_dir)
    combined = run_dir / f"{name}.parquet"
    if combined.exists():
        return pq.read_table(combined, **read_kwargs).to_pandas()
    directory = run_dir / name
    parts = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
    if not parts:
        raise FileNotFoundError(f"no {name}.parquet and no {name}/*.parquet under {run_dir}")
    tables = [pq.read_table(p, **read_kwargs) for p in parts]
    return pa.concat_tables(tables, promote_options="default").to_pandas()


def has_artefact(run_dir, name: str) -> bool:
    run_dir = Path(run_dir)
    if (run_dir / f"{name}.parquet").exists():
        return True
    directory = run_dir / name
    return directory.is_dir() and any(directory.glob("*.parquet"))


def jsonable(value: Any) -> Any:  # pragma: no cover - tiny helper
    return json.loads(json.dumps(value, default=str))
