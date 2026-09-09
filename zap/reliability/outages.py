"""Thermal / storage forced-outage pool, sampler, store, and UCAP table.

This module is deliberately independent of the rest of the CH3 harness: it reads
only ``<dataset_dir>/static/*.csv`` and writes only new files
(``outages.zarr``, ``outage_units.csv``, ``ucap.csv``).

Model
-----
Every source row of ``generators.csv`` / ``storage_units.csv`` whose carrier has
outage parameters is represented by a contiguous, disjoint slice of *virtual
units* on a global unit axis (spec D6: the key is the **source row**, never
``(carrier, bus)`` -- the import buses carry many rows of the same carrier and
bus, which would otherwise be perfectly correlated or double counted).

Each unit is an independent two-state Markov chain in discrete hourly time with
stationary availability ``1 - FOR``.

RNG contract (spec D7)
----------------------
One ``numpy.random.Generator`` per ``(weather_year, draw)``::

    rng = np.random.default_rng(
        np.random.SeedSequence(entropy=base_seed, spawn_key=(year, draw))
    )
    u = rng.random((n_units, n_hours))     # a single, C-order call

NumPy fills C-order, so unit ``u`` always consumes stream positions
``[u * n_hours, (u + 1) * n_hours)``. Appending units to the pool therefore
leaves every existing unit's draw bit-identical, and a chunk is reproducible in
isolation without paying for ``n_units`` ``SeedSequence`` spawns.

Caveats
-------
* Storage outages derate **power only**; the energy cap is untouched (spec D8).
* Hydro carries outage draws even though its ``p_max_pu`` profile already encodes
  availability, so the two derates compound (spec R6). ``excluded_carriers`` in
  ``outage_params.yaml`` makes a no-hydro sensitivity a one-line change.
* The storage UCAP produced here is a forced-outage derate, **not** an
  accreditation (ELCC). Label it that way on run cards (spec R5).
* The ``done`` array in the store is the **authoritative** resume ledger: it has
  one chunk per ``(year, draw)``, so concurrent shards never race on it. The root
  attribute ``completed`` is a convenience view recomputed from ``done`` at the
  end of each shard, and can lag while other shards are still running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import yaml
import zarr

GENERATOR_VERSION = 1
HOURS_PER_YEAR = 8760
DEFAULT_PARAMS_PATH = Path(__file__).parent / "outage_params.yaml"

UNIT_TABLE_COLUMNS = [
    "unit_id",
    "component",
    "row",
    "carrier",
    "bus",
    "slot",
    "unit_size_mw",
    "row_offset",
    "row_units",
]

# (component name, static csv file) in the order they enter the unit axis.
_COMPONENT_FILES = [("Generator", "generators.csv"), ("StorageUnit", "storage_units.csv")]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarrierOutageParams:
    """Two-state Markov outage parameters for one carrier."""

    unit_size_mw: float
    forced_outage_rate: float
    mttr_h: float
    source: str

    def __post_init__(self):
        if not (0.0 < self.forced_outage_rate < 1.0):
            raise ValueError(f"forced_outage_rate must be in (0, 1), got {self.forced_outage_rate}")
        if self.mttr_h < 1.0:
            raise ValueError(
                f"mttr_h must be >= 1 h so that p_repair = 1 / mttr_h is a probability, "
                f"got {self.mttr_h}"
            )
        if self.unit_size_mw <= 0.0:
            raise ValueError(f"unit_size_mw must be positive, got {self.unit_size_mw}")
        if not self.source:
            raise ValueError("every carrier entry must carry a `source` string")

    @property
    def mttf_h(self) -> float:
        """Mean time to failure, hours: MTTR (1 - FOR) / FOR."""
        return self.mttr_h * (1.0 - self.forced_outage_rate) / self.forced_outage_rate

    @property
    def p_fail(self) -> float:
        """Hourly up -> down transition probability, ``1 / MTTF``.

        Linear rates, per the contract (``memory/plans/2026-09-08-phase1.md``
        section 1.2). They make the chain's stationary unavailability exactly the
        forced outage rate::

            p_fail / (p_fail + p_repair) = MTTR / (MTTF + MTTR) = FOR

        The exponential form ``1 - exp(-1/MTTF)`` does *not*: it biases
        unavailability high by roughly 1 % of FOR, which is detectable at the
        sample sizes this store is generated at.
        """
        return 1.0 / self.mttf_h

    @property
    def p_repair(self) -> float:
        """Hourly down -> up transition probability, ``1 / MTTR``."""
        return 1.0 / self.mttr_h

    def to_dict(self) -> dict:
        return {
            "unit_size_mw": float(self.unit_size_mw),
            "forced_outage_rate": float(self.forced_outage_rate),
            "mttr_h": float(self.mttr_h),
            "source": str(self.source),
        }


@dataclass(frozen=True)
class OutageParams:
    """The full resolved contents of ``outage_params.yaml``."""

    version: int
    reviewed: bool
    pool_multiplier: float
    min_units_per_row: int
    min_pool_capacity_mw: float
    excluded_carriers: frozenset[str]
    carriers: Mapping[str, CarrierOutageParams]
    sha256: str

    def to_dict(self) -> dict:
        return {
            "version": int(self.version),
            "reviewed": bool(self.reviewed),
            "pool_multiplier": float(self.pool_multiplier),
            "min_units_per_row": int(self.min_units_per_row),
            "min_pool_capacity_mw": float(self.min_pool_capacity_mw),
            "excluded_carriers": sorted(self.excluded_carriers),
            "carriers": {k: v.to_dict() for k, v in sorted(self.carriers.items())},
        }

    @classmethod
    def from_dict(cls, raw: Mapping, sha256: str = "unknown") -> OutageParams:
        missing = {
            "version",
            "reviewed",
            "pool_multiplier",
            "min_units_per_row",
            "min_pool_capacity_mw",
            "excluded_carriers",
            "carriers",
        } - set(raw)
        if missing:
            raise KeyError(f"outage params missing required keys: {sorted(missing)}")

        carriers = {k: CarrierOutageParams(**v) for k, v in raw["carriers"].items()}
        excluded = frozenset(raw["excluded_carriers"])
        overlap = excluded & set(carriers)
        if overlap:
            raise ValueError(
                f"carriers listed both as parameterised and excluded: {sorted(overlap)}"
            )

        return cls(
            version=int(raw["version"]),
            reviewed=bool(raw["reviewed"]),
            pool_multiplier=float(raw["pool_multiplier"]),
            min_units_per_row=int(raw["min_units_per_row"]),
            min_pool_capacity_mw=float(raw["min_pool_capacity_mw"]),
            excluded_carriers=excluded,
            carriers=carriers,
            sha256=sha256,
        )


def load_outage_params(path: Path | None = None) -> OutageParams:
    """Load and validate ``outage_params.yaml`` (defaults to the checked-in file)."""
    path = Path(path) if path is not None else DEFAULT_PARAMS_PATH
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes)
    return OutageParams.from_dict(raw, sha256=hashlib.sha256(raw_bytes).hexdigest())


# ---------------------------------------------------------------------------
# Unit pool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitPool:
    """The virtual-unit axis: one contiguous slice per source row."""

    table: pd.DataFrame
    row_offset: Mapping[str, int]
    row_units: Mapping[str, int]
    row_size: Mapping[str, float]

    @property
    def n_units(self) -> int:
        return len(self.table)

    def p_fail_vector(self, params: OutageParams) -> np.ndarray:
        return np.array(
            [params.carriers[c].p_fail for c in self.table["carrier"]], dtype=np.float64
        )

    def p_repair_vector(self, params: OutageParams) -> np.ndarray:
        return np.array(
            [params.carriers[c].p_repair for c in self.table["carrier"]], dtype=np.float64
        )

    def for_vector(self, params: OutageParams) -> np.ndarray:
        return np.array(
            [params.carriers[c].forced_outage_rate for c in self.table["carrier"]],
            dtype=np.float64,
        )


def _read_static(dataset_dir: Path, filename: str) -> pd.DataFrame:
    path = Path(dataset_dir) / "static" / filename
    if not path.exists():
        raise FileNotFoundError(f"expected static component table at {path}")
    return pd.read_csv(path)


def build_unit_pool(dataset_dir: Path, params: OutageParams) -> UnitPool:
    """Build the deterministic virtual-unit pool for a dataset.

    Rows are ``generators.csv`` then ``storage_units.csv``, each in CSV order,
    keeping only rows whose carrier has outage parameters. A carrier that appears
    in the data but in neither ``carriers`` nor ``excluded_carriers`` is a hard
    error, so new datasets cannot silently skip units.
    """
    dataset_dir = Path(dataset_dir)
    records = []
    seen_rows: set[str] = set()

    for component, filename in _COMPONENT_FILES:
        df = _read_static(dataset_dir, filename)
        if len(df) == 0:
            continue

        unknown = sorted(
            set(df["carrier"].astype(str)) - set(params.carriers) - params.excluded_carriers
        )
        if unknown:
            raise KeyError(
                f"{filename}: carriers {unknown} are in neither `carriers` nor "
                "`excluded_carriers` of the outage parameters; add them explicitly"
            )

        for _, row in df.iterrows():
            carrier = str(row["carrier"])
            if carrier not in params.carriers:
                continue

            name = str(row["name"])
            if name in seen_rows:
                raise ValueError(f"duplicate component name {name!r} across static tables")
            seen_rows.add(name)

            cp = params.carriers[carrier]
            reference_mw = max(float(row["p_nom"]), params.min_pool_capacity_mw)
            n_units = max(
                params.min_units_per_row,
                math.ceil(params.pool_multiplier * reference_mw / cp.unit_size_mw),
            )
            records.append(
                {
                    "component": component,
                    "row": name,
                    "carrier": carrier,
                    "bus": str(row["bus"]),
                    "unit_size_mw": float(cp.unit_size_mw),
                    "n_units": n_units,
                }
            )

    row_offset: dict[str, int] = {}
    row_units: dict[str, int] = {}
    row_size: dict[str, float] = {}
    table_rows = []
    offset = 0
    for rec in records:
        row_offset[rec["row"]] = offset
        row_units[rec["row"]] = rec["n_units"]
        row_size[rec["row"]] = rec["unit_size_mw"]
        for slot in range(rec["n_units"]):
            table_rows.append(
                {
                    "unit_id": f"{rec['row']}#{slot}",
                    "component": rec["component"],
                    "row": rec["row"],
                    "carrier": rec["carrier"],
                    "bus": rec["bus"],
                    "slot": slot,
                    "unit_size_mw": rec["unit_size_mw"],
                    "row_offset": offset,
                    "row_units": rec["n_units"],
                }
            )
        offset += rec["n_units"]

    table = pd.DataFrame(table_rows, columns=UNIT_TABLE_COLUMNS)
    table.index = pd.RangeIndex(len(table))
    return UnitPool(table=table, row_offset=row_offset, row_units=row_units, row_size=row_size)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_chunk(
    pool: UnitPool,
    params: OutageParams,
    *,
    year: int,
    draw: int,
    n_hours: int,
    base_seed: int,
) -> np.ndarray:
    """Sample one ``(weather_year, draw)`` chunk.

    Returns ``(n_units, n_hours)`` ``uint8``, 1 = available. See the module
    docstring for the RNG contract (spec D7): the whole chunk's uniforms come
    from a single C-order ``rng.random`` call, so growing the pool by appending
    units leaves existing units bit-identical.
    """
    if n_hours < 1:
        raise ValueError(f"n_hours must be >= 1, got {n_hours}")

    n_units = pool.n_units
    rng = np.random.default_rng(
        np.random.SeedSequence(entropy=int(base_seed), spawn_key=(int(year), int(draw)))
    )
    u = rng.random((n_units, n_hours))

    pf = pool.p_fail_vector(params)
    pr = pool.p_repair_vector(params)
    f = pool.for_vector(params)

    out = np.empty((n_units, n_hours), dtype=np.uint8)
    state = u[:, 0] < (1.0 - f)
    out[:, 0] = state
    for t in range(1, n_hours):
        state = np.where(state, u[:, t] >= pf, u[:, t] < pr)
        out[:, t] = state
    return out


def _row_weights(capacity: float, size: float, available_units: int, row: str) -> np.ndarray:
    """Weights of the units backing ``capacity`` MW of a row with unit size ``size``."""
    if capacity <= 0.0:
        return np.zeros(0, dtype=np.float64)
    # max(1, ...) guards a capacity so small that the epsilon rounds n down to 0.
    n = max(1, math.ceil(capacity / size - 1e-9))
    if n > available_units:
        raise ValueError(
            f"row {row!r} needs {n} units for {capacity} MW at {size} MW/unit but the "
            f"pool only holds {available_units}; regenerate the pool with a larger "
            "pool_multiplier"
        )
    w = np.ones(n, dtype=np.float64)
    w[-1] = (capacity - (n - 1) * size) / size
    return w


def row_availability(
    up: np.ndarray,
    pool: UnitPool,
    capacities: pd.Series,
    rows: Sequence[str],
) -> np.ndarray:
    """Map unit availability onto source rows.

    ``up``: ``(n_units, n_hours)`` -> returns ``(n_hours, len(rows))`` float64 in
    ``[0, 1]``. For row ``r`` with capacity ``C`` and unit size ``s``::

        n = ceil(C / s)                       # n == 0 if C == 0 -> availability 1.0
        w = [1] * (n - 1) + [(C - (n - 1) s) / s]
        availability(t) = sum_k w_k up[offset + k, t] / sum_k w_k

    Raises ``ValueError`` if ``n`` exceeds the units the pool holds for that row
    (the design outgrew the pool). Rows that are absent from the pool -- VRE, or
    any carrier in ``excluded_carriers`` -- get availability 1.0 in every hour.
    """
    up = np.asarray(up)
    if up.ndim != 2:
        raise ValueError(f"`up` must be (n_units, n_hours), got shape {up.shape}")
    if up.shape[0] != pool.n_units:
        raise ValueError(f"`up` has {up.shape[0]} units but the pool has {pool.n_units}")

    n_hours = up.shape[1]
    out = np.ones((n_hours, len(rows)), dtype=np.float64)

    for j, row in enumerate(rows):
        if row not in pool.row_offset:
            continue
        capacity = float(capacities[row])
        size = pool.row_size[row]
        w = _row_weights(capacity, size, pool.row_units[row], row)
        if w.size == 0:
            continue
        offset = pool.row_offset[row]
        block = up[offset : offset + w.size, :].astype(np.float64)
        out[:, j] = (w @ block) / w.sum()

    return out


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _zap_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _vlen_str_array(group: zarr.Group, name: str, values: Sequence[str]) -> None:
    arr = group.create_dataset(
        name,
        shape=(len(values),),
        dtype=object,
        object_codec=numcodecs.VLenUTF8(),
        overwrite=True,
    )
    arr[:] = np.array([str(v) for v in values], dtype=object)


def default_store_path(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / "outages.zarr"


def default_units_csv(store_path: Path) -> Path:
    """The unit sidecar lives next to its store, so a sensitivity store built with
    ``--out`` elsewhere cannot clobber the canonical ``outage_units.csv``."""
    store_path = Path(store_path)
    stem = store_path.stem
    name = "outage_units.csv" if stem == "outages" else f"{stem}_units.csv"
    return store_path.parent / name


def default_ucap_csv(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / "ucap.csv"


def storage_projection(pool: UnitPool, years: Sequence[int], draws: int, hours: int) -> dict:
    raw = pool.n_units * len(years) * draws * hours
    return {
        "n_units": pool.n_units,
        "n_years": len(years),
        "n_draws": draws,
        "hours_per_year": hours,
        "raw_bytes": raw,
        "raw_gb": raw / 1e9,
    }


def init_store(
    dataset_dir: Path,
    *,
    years: Sequence[int],
    draws: int,
    base_seed: int,
    params: OutageParams,
    out_path: Path | None = None,
    hours_per_year: int = HOURS_PER_YEAR,
    chunk_hours: int = 168,
    overwrite: bool = False,
) -> Path:
    """Create the empty outage store, its coordinates, and ``outage_units.csv``.

    Separating this from :func:`generate` is what makes a SLURM array safe: the
    store exists before any shard starts, so shards never race on creation and,
    since chunking is ``(1, 1, chunk_hours, n_units)``, never share a chunk.
    """
    dataset_dir = Path(dataset_dir)
    out_path = Path(out_path) if out_path is not None else default_store_path(dataset_dir)
    years = [int(y) for y in years]
    if sorted(set(years)) != years:
        raise ValueError("`years` must be unique and ascending")
    if draws < 1:
        raise ValueError("`draws` must be >= 1")

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists; pass overwrite=True (--overwrite) to replace it"
        )

    pool = build_unit_pool(dataset_dir, params)
    n_units = pool.n_units
    if n_units == 0:
        raise ValueError("the unit pool is empty; check `carriers` in the outage parameters")

    root = zarr.open_group(str(out_path), mode="w")
    root.create_dataset(
        "available",
        shape=(len(years), draws, hours_per_year, n_units),
        chunks=(1, 1, min(chunk_hours, hours_per_year), n_units),
        dtype="uint8",
        fill_value=255,
        compressor=numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE),
        overwrite=True,
    )
    # Race-free resume ledger: one chunk per (year, draw), so shards never contend.
    root.create_dataset(
        "done",
        shape=(len(years), draws),
        chunks=(1, 1),
        dtype="uint8",
        fill_value=0,
        overwrite=True,
    )

    root.create_dataset("weather_year", shape=(len(years),), dtype="int32", overwrite=True)[:] = (
        np.array(years, dtype=np.int32)
    )
    root.create_dataset("draw", shape=(draws,), dtype="int32", overwrite=True)[:] = np.arange(
        draws, dtype=np.int32
    )
    root.create_dataset("hour", shape=(hours_per_year,), dtype="int32", overwrite=True)[:] = (
        np.arange(hours_per_year, dtype=np.int32)
    )

    table = pool.table
    for name, col in [
        ("unit_id", "unit_id"),
        ("unit_row", "row"),
        ("unit_carrier", "carrier"),
        ("unit_bus", "bus"),
        ("unit_component", "component"),
    ]:
        _vlen_str_array(root, name, table[col].tolist())
    for name, col in [
        ("unit_slot", "slot"),
        ("unit_row_offset", "row_offset"),
        ("unit_row_units", "row_units"),
    ]:
        root.create_dataset(name, shape=(n_units,), dtype="int32", overwrite=True)[:] = table[
            col
        ].to_numpy(np.int32)
    root.create_dataset("unit_size_mw", shape=(n_units,), dtype="float32", overwrite=True)[:] = (
        table["unit_size_mw"].to_numpy(np.float32)
    )

    root.attrs.update(
        {
            "generator_version": GENERATOR_VERSION,
            "dataset": dataset_dir.name,
            "created_utc": _utc_now(),
            "zap_commit": _zap_commit(),
            "base_seed": int(base_seed),
            "params_sha256": params.sha256,
            "params": params.to_dict(),
            "pool_multiplier": params.pool_multiplier,
            "min_units_per_row": params.min_units_per_row,
            "min_pool_capacity_mw": params.min_pool_capacity_mw,
            "weather_years": years,
            "n_draws": int(draws),
            "hours_per_year": int(hours_per_year),
            "n_units": int(n_units),
            "completed": [],
        }
    )

    units_csv = default_units_csv(out_path)
    table.to_csv(units_csv, index=False)
    return out_path


def _shard_jobs(years: Sequence[int], draws: int, chunk: tuple[int, int]) -> list[tuple[int, int]]:
    k, n = int(chunk[0]), int(chunk[1])
    if n < 1 or not (1 <= k <= n):
        raise ValueError(f"chunk must be (k, n) with 1 <= k <= n, got {chunk}")
    jobs = sorted((int(y), int(d)) for y in years for d in range(draws))
    # Contiguous, balanced split of the sorted job list.
    total = len(jobs)
    start = (total * (k - 1)) // n
    stop = (total * k) // n
    return jobs[start:stop]


def _assert_units_csv_matches(store_path: Path, root: zarr.Group, pool: UnitPool) -> None:
    csv_path = default_units_csv(store_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"missing sidecar {csv_path}; re-run `init`")
    sidecar = pd.read_csv(csv_path)
    if list(sidecar["unit_id"]) != list(np.asarray(root["unit_id"][:])):
        raise ValueError(f"{csv_path} does not match the store's unit coordinates")
    if list(sidecar["unit_id"]) != list(pool.table["unit_id"]):
        raise ValueError(
            f"{csv_path} does not match the pool rebuilt from static/*.csv and the "
            "store's parameters; the static tables or parameters changed"
        )


def generate(
    dataset_dir: Path,
    *,
    years: Sequence[int],
    draws: int,
    base_seed: int,
    params: OutageParams,
    out_path: Path | None = None,
    chunk: tuple[int, int] = (1, 1),
    overwrite: bool = False,
    init: bool = False,
    hours_per_year: int = HOURS_PER_YEAR,
    chunk_hours: int = 168,
    verbose: bool = True,
) -> Path:
    """Generate outage draws into ``outages.zarr``.

    ``chunk=(k, n)`` processes shard ``k`` (1-based) of ``n`` contiguous shards of
    the sorted ``(year, draw)`` job list -- the SLURM-array seam. The store must
    already exist (see :func:`init_store`) unless ``init=True``.

    Already-generated ``(year, draw)`` pairs are skipped unless ``overwrite`` is
    set, so a failed array job can simply be resubmitted.
    """
    dataset_dir = Path(dataset_dir)
    out_path = Path(out_path) if out_path is not None else default_store_path(dataset_dir)

    if init:
        init_store(
            dataset_dir,
            years=years,
            draws=draws,
            base_seed=base_seed,
            params=params,
            out_path=out_path,
            hours_per_year=hours_per_year,
            chunk_hours=chunk_hours,
            overwrite=overwrite,
        )
    elif not out_path.exists():
        raise FileNotFoundError(
            f"{out_path} does not exist; run the `init` subcommand first (or pass init=True)"
        )

    root = zarr.open_group(str(out_path), mode="r+")
    store_years = [int(y) for y in root.attrs["weather_years"]]
    if [int(y) for y in years] != store_years:
        raise ValueError(f"store was initialised for years {store_years}, asked for {list(years)}")
    if int(draws) != int(root.attrs["n_draws"]):
        raise ValueError(
            f"store was initialised for {root.attrs['n_draws']} draws, asked for {draws}"
        )
    if int(base_seed) != int(root.attrs["base_seed"]):
        raise ValueError(
            f"store was initialised with base_seed {root.attrs['base_seed']}, asked for {base_seed}"
        )
    if params.sha256 != root.attrs["params_sha256"]:
        raise ValueError(
            "outage parameters differ from the ones the store was initialised with "
            f"({params.sha256[:12]} != {str(root.attrs['params_sha256'])[:12]})"
        )

    hours = int(root.attrs["hours_per_year"])
    pool = build_unit_pool(dataset_dir, params)
    _assert_units_csv_matches(out_path, root, pool)

    year_ix = {y: i for i, y in enumerate(store_years)}
    jobs = _shard_jobs(store_years, draws, chunk)
    available = root["available"]
    done = root["done"]

    t0 = time.time()
    n_written = 0
    for year, draw in jobs:
        yi = year_ix[year]
        if done[yi, draw] and not overwrite:
            if verbose:
                print(f"  skip (done) year={year} draw={draw}")
            continue
        up = sample_chunk(pool, params, year=year, draw=draw, n_hours=hours, base_seed=base_seed)
        available[yi, draw, :, :] = up.T
        done[yi, draw] = 1
        n_written += 1
        if verbose:
            print(f"  wrote year={year} draw={draw} ({n_written}/{len(jobs)})")

    # `done` is the authoritative ledger (one chunk per job, so shards never race).
    # `attrs["completed"]` is a convenience view recomputed at the end of each shard
    # and can lag while other shards are still running.
    done_arr = np.asarray(done[:])
    completed = [[int(store_years[i]), int(j)] for i, j in zip(*np.nonzero(done_arr), strict=True)]
    root.attrs["completed"] = completed

    if verbose:
        elapsed = time.time() - t0
        proj = storage_projection(pool, store_years, draws, hours)
        size = _dir_size(out_path)
        print(
            f"shard {chunk[0]}/{chunk[1]}: wrote {n_written} (year, draw) chunks in "
            f"{elapsed:.1f} s; store {size / 1e6:.1f} MB on disk, projected raw "
            f"{proj['raw_gb']:.2f} GB, {len(completed)}/{len(store_years) * draws} complete"
        )
    return out_path


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# UCAP
# ---------------------------------------------------------------------------


def write_ucap(
    dataset_dir: Path,
    store_path: Path | None = None,
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """Derive ``ucap.csv`` from a generated outage store.

    ``ucap_empirical`` is the mean over every generated ``(year, draw, hour)`` of
    ``row_availability`` for the row at its current ``p_nom``. Because that map is
    linear in the unit uptimes, the mean is computed from per-unit uptime sums --
    the store is read once, streaming.
    """
    dataset_dir = Path(dataset_dir)
    store_path = Path(store_path) if store_path is not None else default_store_path(dataset_dir)
    out_csv = Path(out_csv) if out_csv is not None else default_ucap_csv(dataset_dir)
    if not store_path.exists():
        raise FileNotFoundError(f"no outage store at {store_path}; run `generate` first")

    root = zarr.open_group(str(store_path), mode="r")
    params = OutageParams.from_dict(root.attrs["params"], sha256=root.attrs["params_sha256"])
    pool = build_unit_pool(dataset_dir, params)
    if pool.n_units != int(root.attrs["n_units"]):
        raise ValueError("the pool rebuilt from static/*.csv does not match the store")

    hours = int(root.attrs["hours_per_year"])
    done = np.asarray(root["done"][:])
    completed = list(zip(*np.nonzero(done), strict=True))
    if not completed:
        raise ValueError(f"{store_path} has no completed (year, draw) chunks")

    available = root["available"]
    uptime = np.zeros(pool.n_units, dtype=np.float64)
    for yi, di in completed:
        uptime += np.asarray(available[yi, di, :, :], dtype=np.float64).sum(axis=0)
    n_samples = len(completed) * hours
    mean_uptime = uptime / n_samples

    # Capacities: the current p_nom of every pooled row.
    caps = {}
    for _, filename in _COMPONENT_FILES:
        df = _read_static(dataset_dir, filename)
        for _, r in df.iterrows():
            if str(r["name"]) in pool.row_offset:
                caps[str(r["name"])] = float(r["p_nom"])

    records = []
    for row, offset in pool.row_offset.items():
        meta = pool.table.iloc[offset]
        carrier = str(meta["carrier"])
        cp = params.carriers[carrier]
        capacity = caps[row]
        analytic = 1.0 - cp.forced_outage_rate
        w = _row_weights(capacity, pool.row_size[row], pool.row_units[row], row)
        if w.size == 0:
            empirical, samples = analytic, 0
        else:
            empirical = float((w @ mean_uptime[offset : offset + w.size]) / w.sum())
            samples = n_samples
        records.append(
            {
                "component": str(meta["component"]),
                "row": row,
                "carrier": carrier,
                "bus": str(meta["bus"]),
                "p_nom_mw": capacity,
                "unit_size_mw": pool.row_size[row],
                "n_units_row": pool.row_units[row],
                "forced_outage_rate": cp.forced_outage_rate,
                "ucap_analytic": analytic,
                "ucap_empirical": empirical,
                "n_samples": samples,
            }
        )

    df = pd.DataFrame.from_records(records)

    def _weighted(group: pd.DataFrame) -> float:
        w = group["p_nom_mw"].to_numpy(np.float64)
        v = group["ucap_empirical"].to_numpy(np.float64)
        return float(v.mean()) if w.sum() <= 0.0 else float((w * v).sum() / w.sum())

    agg = df.groupby(["carrier", "bus"], sort=False).apply(_weighted, include_groups=False)
    df["ucap_carrier_bus_empirical"] = [
        agg.loc[(c, b)] for c, b in zip(df["carrier"], df["bus"], strict=True)
    ]

    df = df[
        [
            "component",
            "row",
            "carrier",
            "bus",
            "p_nom_mw",
            "unit_size_mw",
            "n_units_row",
            "forced_outage_rate",
            "ucap_analytic",
            "ucap_empirical",
            "n_samples",
            "ucap_carrier_bus_empirical",
        ]
    ]
    df.to_csv(out_csv, index=False)
    print(
        f"wrote {out_csv} ({len(df)} rows) from {len(completed)} (year, draw) chunks, "
        f"{n_samples} hourly samples per row"
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_pool_summary(pool: UnitPool, params: OutageParams, years, draws, hours) -> None:
    proj = storage_projection(pool, years, draws, hours)
    by_carrier = pool.table.groupby("carrier").size().sort_values(ascending=False)
    print(f"unit pool: {pool.n_units} units over {len(pool.row_offset)} source rows")
    for carrier, n in by_carrier.items():
        cp = params.carriers[carrier]
        print(
            f"  {carrier:>22s}: {n:5d} units @ {cp.unit_size_mw:g} MW, "
            f"FOR={cp.forced_outage_rate:.3f}, MTTR={cp.mttr_h:g} h, "
            f"MTTF={cp.mttf_h:.0f} h"
        )
    print(
        f"projection: {proj['n_years']} years x {proj['n_draws']} draws x "
        f"{proj['hours_per_year']} h x {proj['n_units']} units = "
        f"{proj['raw_gb']:.2f} GB raw uint8 (expect strong compression)"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m zap.reliability.outages",
        description="Generate thermal/storage forced-outage draws and the UCAP table.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--dataset-dir", type=Path, required=True)
        sp.add_argument("--params", type=Path, default=None, help="outage_params.yaml override")
        sp.add_argument("--out", type=Path, default=None, help="store path")

    init_p = sub.add_parser("init", help="create the empty store and its coordinates")
    common(init_p)
    init_p.add_argument("--years", type=int, nargs="+", required=True)
    init_p.add_argument("--draws", type=int, required=True)
    init_p.add_argument("--seed", type=int, required=True)
    init_p.add_argument("--hours-per-year", type=int, default=HOURS_PER_YEAR)
    init_p.add_argument("--chunk-hours", type=int, default=168)
    init_p.add_argument("--overwrite", action="store_true")
    init_p.add_argument("--dry-run", action="store_true")

    gen_p = sub.add_parser("generate", help="fill one shard of (year, draw) jobs")
    common(gen_p)
    gen_p.add_argument("--chunk", type=str, default="1/1", help="shard k/n, 1-based")
    gen_p.add_argument("--years", type=int, nargs="+", default=None)
    gen_p.add_argument("--draws", type=int, default=None)
    gen_p.add_argument("--seed", type=int, default=None)
    gen_p.add_argument("--hours-per-year", type=int, default=HOURS_PER_YEAR)
    gen_p.add_argument("--chunk-hours", type=int, default=168)
    gen_p.add_argument("--init", action="store_true", help="create the store first (shard 1 only)")
    gen_p.add_argument("--overwrite", action="store_true")
    gen_p.add_argument("--dry-run", action="store_true")

    ucap_p = sub.add_parser("ucap", help="write ucap.csv from a generated store")
    common(ucap_p)
    ucap_p.add_argument("--out-csv", type=Path, default=None)

    pool_p = sub.add_parser("pool", help="print the unit pool without writing anything")
    common(pool_p)
    pool_p.add_argument("--years", type=int, nargs="+", default=[2020])
    pool_p.add_argument("--draws", type=int, default=1)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params = load_outage_params(args.params)
    if not params.reviewed:
        print(
            "WARNING: outage_params.yaml has `reviewed: false` -- the numbers are "
            "placeholders and must not be cited in a reported run.",
            file=sys.stderr,
        )

    if args.command == "pool":
        pool = build_unit_pool(args.dataset_dir, params)
        _print_pool_summary(pool, params, args.years, args.draws, HOURS_PER_YEAR)
        return 0

    if args.command == "ucap":
        write_ucap(args.dataset_dir, store_path=args.out, out_csv=args.out_csv)
        return 0

    if args.command == "init":
        pool = build_unit_pool(args.dataset_dir, params)
        _print_pool_summary(pool, params, args.years, args.draws, args.hours_per_year)
        if args.dry_run:
            print("--dry-run: nothing written")
            return 0
        path = init_store(
            args.dataset_dir,
            years=args.years,
            draws=args.draws,
            base_seed=args.seed,
            params=params,
            out_path=args.out,
            hours_per_year=args.hours_per_year,
            chunk_hours=args.chunk_hours,
            overwrite=args.overwrite,
        )
        print(f"initialised {path}")
        return 0

    if args.command == "generate":
        k, n = (int(x) for x in str(args.chunk).split("/"))
        store_path = args.out or default_store_path(args.dataset_dir)
        if store_path.exists() and not args.init:
            root = zarr.open_group(str(store_path), mode="r")
            years = [int(y) for y in root.attrs["weather_years"]]
            draws = int(root.attrs["n_draws"])
            seed = int(root.attrs["base_seed"])
        else:
            if args.years is None or args.draws is None or args.seed is None:
                raise SystemExit(
                    "no store yet: pass --init together with --years, --draws and --seed"
                )
            years, draws, seed = args.years, args.draws, args.seed

        if args.dry_run:
            pool = build_unit_pool(args.dataset_dir, params)
            _print_pool_summary(pool, params, years, draws, args.hours_per_year)
            print(f"--dry-run: shard {k}/{n} would write {len(_shard_jobs(years, draws, (k, n)))}")
            return 0

        t0 = time.time()
        path = generate(
            args.dataset_dir,
            years=years,
            draws=draws,
            base_seed=seed,
            params=params,
            out_path=args.out,
            chunk=(k, n),
            overwrite=args.overwrite,
            init=args.init,
            hours_per_year=args.hours_per_year,
            chunk_hours=args.chunk_hours,
        )
        raw = _dir_size(path)
        pool = build_unit_pool(args.dataset_dir, params)
        proj = storage_projection(pool, years, draws, args.hours_per_year)
        print(
            json.dumps(
                {
                    "store": str(path),
                    "bytes_on_disk": raw,
                    "raw_bytes": proj["raw_bytes"],
                    "compression_ratio": proj["raw_bytes"] / max(raw, 1),
                    "elapsed_s": round(time.time() - t0, 1),
                },
                indent=2,
            )
        )
        return 0

    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
