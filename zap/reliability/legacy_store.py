"""Read-only reader for the **superseded** offset-keyed ``outages.zarr`` (v1).

Kept for exactly one purpose: the WP-O5 statistical-equivalence study, which
compares the archived draws against ``slot-v1``. It is deleted, together with the
archived store, once that study passes (outage-pool spec R5 / section 5.2).

Nothing in the live code path may import this module. It carries its own copy of
the version-1 parameter dataclass, because ``outage_params.yaml`` is now version
2 and no longer has the pool-sizing keys the old pool was built from -- the v1
parameters are read back from the store's own ``attrs["params"]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LegacyCarrierParams:
    unit_size_mw: float
    forced_outage_rate: float
    mttr_h: float
    source: str = "legacy"

    @property
    def mttf_h(self) -> float:
        return self.mttr_h * (1.0 - self.forced_outage_rate) / self.forced_outage_rate

    @property
    def p_fail(self) -> float:
        return 1.0 / self.mttf_h

    @property
    def p_repair(self) -> float:
        return 1.0 / self.mttr_h


@dataclass(frozen=True)
class LegacyOutageParams:
    """Version-1 ``outage_params.yaml`` as recorded in a store's attributes."""

    version: int
    reviewed: bool
    pool_multiplier: float
    min_units_per_row: int
    min_pool_capacity_mw: float
    excluded_carriers: frozenset[str]
    carriers: Mapping[str, LegacyCarrierParams]

    @classmethod
    def from_dict(cls, raw: Mapping) -> LegacyOutageParams:
        return cls(
            version=int(raw["version"]),
            reviewed=bool(raw["reviewed"]),
            pool_multiplier=float(raw["pool_multiplier"]),
            min_units_per_row=int(raw["min_units_per_row"]),
            min_pool_capacity_mw=float(raw["min_pool_capacity_mw"]),
            excluded_carriers=frozenset(raw["excluded_carriers"]),
            carriers={
                k: LegacyCarrierParams(
                    unit_size_mw=float(v["unit_size_mw"]),
                    forced_outage_rate=float(v["forced_outage_rate"]),
                    mttr_h=float(v["mttr_h"]),
                    source=str(v.get("source", "legacy")),
                )
                for k, v in raw["carriers"].items()
            },
        )


@dataclass(frozen=True)
class UnitPool:
    """The v1 virtual-unit axis: one contiguous slice per source row."""

    table: pd.DataFrame
    row_offset: Mapping[str, int]
    row_units: Mapping[str, int]
    row_size: Mapping[str, float]

    @property
    def n_units(self) -> int:
        return len(self.table)


def open_store(path: Path):
    """Open an archived ``outages.zarr`` read-only; returns ``(path, zarr group)``."""
    import zarr

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no archived outage store at {path}")
    return path, zarr.open_group(str(path), mode="r")


def pool_from_store(root) -> UnitPool:
    """Rebuild the v1 ``UnitPool`` from the store's coordinate arrays."""
    table = pd.DataFrame(
        {
            "unit_id": [str(v) for v in root["unit_id"][:]],
            "component": [str(v) for v in root["unit_component"][:]],
            "row": [str(v) for v in root["unit_row"][:]],
            "carrier": [str(v) for v in root["unit_carrier"][:]],
            "bus": [str(v) for v in root["unit_bus"][:]],
            "slot": np.asarray(root["unit_slot"][:], dtype=int),
            "unit_size_mw": np.asarray(root["unit_size_mw"][:], dtype=float),
            "row_offset": np.asarray(root["unit_row_offset"][:], dtype=int),
            "row_units": np.asarray(root["unit_row_units"][:], dtype=int),
        }
    )
    first = table.drop_duplicates("row").set_index("row")
    return UnitPool(
        table=table,
        row_offset=first["row_offset"].astype(int).to_dict(),
        row_units=first["row_units"].astype(int).to_dict(),
        row_size=first["unit_size_mw"].astype(float).to_dict(),
    )


def legacy_row_weights(capacity: float, size: float, available_units: int, row: str) -> np.ndarray:
    """v1 ``_row_weights``, verbatim (including the pool bound)."""
    if capacity <= 0.0:
        return np.zeros(0, dtype=np.float64)
    n = max(1, math.ceil(capacity / size - 1e-9))
    if n > available_units:
        raise ValueError(
            f"row {row!r} needs {n} units for {capacity} MW at {size} MW/unit but the "
            f"pool only holds {available_units}"
        )
    w = np.ones(n, dtype=np.float64)
    w[-1] = (capacity - (n - 1) * size) / size
    return w


def legacy_row_availability(
    up: np.ndarray,
    pool: UnitPool,
    capacities: Mapping[str, float],
    rows: Sequence[str],
) -> np.ndarray:
    """v1 ``row_availability``: ``(n_units, n_hours)`` -> ``(n_hours, len(rows))``."""
    up = np.asarray(up)
    n_hours = up.shape[1]
    out = np.ones((n_hours, len(rows)), dtype=np.float64)
    for j, row in enumerate(rows):
        if row not in pool.row_offset:
            continue
        capacity = float(capacities[row])
        size = pool.row_size[row]
        w = legacy_row_weights(capacity, size, pool.row_units[row], row)
        if w.size == 0:
            continue
        offset = pool.row_offset[row]
        block = up[offset : offset + w.size, :].astype(np.float64)
        out[:, j] = (w @ block) / w.sum()
    return out


def read_chunk(root, year: int, draw: int, start: int = 0, stop: int | None = None) -> np.ndarray:
    """``(n_units, n_hours)`` uptime of one ``(year, draw)`` from the archived store."""
    years = [int(y) for y in np.asarray(root["weather_year"][:])]
    draws = [int(d) for d in np.asarray(root["draw"][:])]
    if int(year) not in years:
        raise KeyError(f"year {year} not in the store (have {years})")
    if int(draw) not in draws:
        raise KeyError(f"draw {draw} not in the store (have {draws})")
    yi, di = years.index(int(year)), draws.index(int(draw))
    if "done" in root and not bool(np.asarray(root["done"][yi, di])):
        raise ValueError(f"(year={year}, draw={draw}) was never generated")
    stop = int(root["available"].shape[2]) if stop is None else int(stop)
    return np.asarray(root["available"][yi, di, int(start) : stop, :], dtype=np.uint8).T
