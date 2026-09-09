"""Weather store converter and reader for pypsa-usa weather-year exports.

This module replaces the per-run ``pypsa.Network`` round trip for the CH3
datasets (``data/ca2040_z4``, ``data/ca2040_county``).  It has two halves:

``convert_dataset``
    One-shot conversion of ``<dataset>/timeseries/*.parquet`` into a chunked
    zarr store ``<dataset>/weather.zarr`` with a ``(year, hour, column)`` layout,
    so a block solve reads a contiguous slice instead of re-parsing parquet.

``load_system``
    Builds ``zap`` devices *directly* from ``<dataset>/static/*.csv`` plus the
    weather store (decision D1 of the Phase-1 spec).  No ``pypsa.Network`` is
    constructed at run time.  ``zap/importers/pypsa.py`` remains the reference
    for field semantics; the mapping here is documented line by line in
    :func:`load_system` and its helpers and is tested against the CSV/parquet
    values directly (``zap/tests/test_wy_store.py``).

Key modelling decisions implemented here (see ``memory/plans/2026-09-08-phase1-spec.md``):

* D2 -- device order is ``[Generator, Load, DirectedLine, StorageUnit, ExportSink]``.
* D3 -- load shedding is ``Load.linear_cost = voll``; no slack device is added.
* D4 -- ``zap.devices.store.Store`` is not used; export stores become ``Injector`` sinks.
* D5 -- links become ``DirectedLine`` (directional, signed linear cost, efficiency).
* D9 -- ``peak_fraction`` demand scaling is clipped at 1.0 by default.
* D10 -- ``p_min_pu`` (minimum stable level) is ignored.
* D12 -- the converter refuses to overwrite an existing store and records provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numcodecs
import numpy as np
import pandas as pd
import zarr

from zap.devices.abstract import AbstractDevice
from zap.devices.injector import Generator, Injector, Load
from zap.devices.storage_unit import StorageUnit
from zap.network import PowerNetwork

logger = logging.getLogger(__name__)

# ``DirectedLine`` is delivered by WP3.  Import it lazily so that the converter
# half of this module is usable before that work package lands.
try:  # pragma: no cover - trivial import guard
    from zap.devices.transporter import DirectedLine
except ImportError:  # pragma: no cover
    DirectedLine = None


# ===========================================================================
# Constants
# ===========================================================================

CONVERTER_VERSION = 1
STORE_NAME = "weather.zarr"
UCAP_NAME = "ucap.csv"
OUTAGE_STORE_NAME = "outages.zarr"

DEFAULT_HOURS_PER_YEAR = 365 * 24
DEFAULT_CHUNK_HOURS = 168

#: If a chunk would be smaller than this many bytes, widen it to a whole year.
CHUNK_BYTE_FLOOR = 65536

#: Above this estimated in-memory size (float64) a parquet file is read one
#: weather year at a time instead of in one go.
MAX_PARQUET_READ_BYTES = 4 * 1024**3

#: Carriers treated as variable renewables (used for the curtailment metric).
VRE_CARRIERS = frozenset({"solar", "onwind", "offwind_floating"})

#: Fallback thermal-carrier list, used for ``SystemIndex.thermal_mask`` when
#: ``zap.reliability.outages`` (WP2) is not importable.  Kept in sync with
#: ``zap/reliability/outage_params.yaml``.
DEFAULT_THERMAL_CARRIERS = frozenset(
    {
        "CCGT",
        "CCGT-95CCS",
        "OCGT",
        "hydrogen_ct",
        "coal",
        "oil",
        "nuclear",
        "biomass",
        "waste",
        "geothermal",
        "hydro",
        "PHS",
        "battery",
        "4hr_battery_storage",
        "8hr_battery_storage",
    }
)

#: Columns of ``ucap.csv`` tried, in order, for the derate factor.
UCAP_COLUMN_PREFERENCE = ("ucap_empirical", "ucap_analytic")

STATIC_FILES = {
    "buses": "buses.csv",
    "generators": "generators.csv",
    "loads": "loads.csv",
    "links": "links.csv",
    "storage_units": "storage_units.csv",
    "stores": "stores.csv",
    "carriers": "carriers.csv",
}


# ===========================================================================
# Small helpers
# ===========================================================================


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    """Hex sha256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def zap_commit() -> str:
    """``git rev-parse HEAD`` of the zap repository, or ``"unknown"``."""
    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"
    if out.returncode != 0:  # pragma: no cover
        return "unknown"
    return out.stdout.strip() or "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def array_name_for(parquet_stem: str) -> str:
    """``generators_t_p_max_pu`` -> ``generators_p_max_pu``."""
    return parquet_stem.replace("_t_", "_", 1)


def chunk_hours_for(n_cols: int, chunk_hours: int, hours_per_year: int) -> int:
    """Chunk-height rule from the spec: widen tiny chunks to a whole year."""
    chunk_hours = min(int(chunk_hours), int(hours_per_year))
    if n_cols * chunk_hours * 4 < CHUNK_BYTE_FLOOR:
        return int(hours_per_year)
    return chunk_hours


def directory_size(path: Path) -> int:
    """Total size in bytes of every file below ``path``."""
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PB"  # pragma: no cover


def _blosc_shuffle():
    return numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)


def _vlen_str_array(group, name: str, values: Sequence[str]):
    arr = group.create_dataset(
        name,
        shape=(len(values),),
        chunks=(len(values),) if len(values) else (1,),
        dtype=object,
        object_codec=numcodecs.VLenUTF8(),
        overwrite=True,
    )
    if len(values):
        arr[:] = np.array([str(v) for v in values], dtype=object)
    return arr


# ===========================================================================
# Converter
# ===========================================================================


def _parquet_year_counts(path: Path) -> pd.Series:
    """Rows per weather year, read from the ``weather_year`` column only."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["weather_year"])
    years = pd.Series(table.column("weather_year").to_numpy())
    return years.value_counts().sort_index()


def _estimated_read_bytes(path: Path) -> int:
    import pyarrow.parquet as pq

    md = pq.ParquetFile(path).metadata
    return int(md.num_rows) * int(md.num_columns) * 8


def convert_dataset(
    dataset_dir: Path,
    out_path: Optional[Path] = None,
    *,
    years: Optional[Sequence[int]] = None,
    chunk_hours: int = DEFAULT_CHUNK_HOURS,
    overwrite: bool = False,
) -> Path:
    """Convert ``<dataset_dir>/timeseries/*.parquet`` into ``weather.zarr``.

    One zarr array per parquet file, shaped ``(n_years, hours_per_year, n_cols)``
    in float32.  Rows are ordered by weather year ascending; the hour axis is the
    parquet order within a year.  Years whose length differs from the dataset's
    modal year length (a leap year, or a truncated export) are *not* written --
    they are recorded in ``attrs["skipped_years"]`` with a reason.

    Memory: parquet files are read one at a time.  A file whose float64
    materialisation would exceed ``MAX_PARQUET_READ_BYTES`` is read one weather
    year at a time with a pyarrow predicate instead; the CH3 datasets are well
    under that ceiling (the county ``generators_t_p_max_pu`` is ~0.64 GB), so in
    practice each file is read once and peak RSS stays below ~1 GB.
    """
    dataset_dir = Path(dataset_dir)
    ts_dir = dataset_dir / "timeseries"
    if not ts_dir.is_dir():
        raise FileNotFoundError(f"No timeseries directory at {ts_dir}")

    parquets = sorted(ts_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files in {ts_dir}")

    out = Path(out_path) if out_path is not None else dataset_dir / STORE_NAME
    if out.exists():
        if not overwrite:
            raise FileExistsError(
                f"{out} already exists; pass overwrite=True (--overwrite) to replace it."
            )
        shutil.rmtree(out)

    t0 = time.time()

    # ---- Year inventory ---------------------------------------------------
    counts_by_file = {p: _parquet_year_counts(p) for p in parquets}
    reference = counts_by_file[parquets[0]]
    for p, counts in counts_by_file.items():
        if not counts.equals(reference):
            raise ValueError(
                f"{p.name} does not have the same (weather_year -> n_rows) inventory as "
                f"{parquets[0].name}; refusing to guess an alignment."
            )

    hours_per_year = int(reference.mode().iloc[0])
    if hours_per_year != DEFAULT_HOURS_PER_YEAR:
        logger.warning(
            "hours_per_year is %d, not %d; this is expected only for test fixtures.",
            hours_per_year,
            DEFAULT_HOURS_PER_YEAR,
        )

    skipped_years: dict[str, str] = {}
    available = []
    for year, n in reference.items():
        year = int(year)
        if int(n) != hours_per_year:
            skipped_years[str(year)] = f"{int(n)} hours, expected {hours_per_year}"
            continue
        available.append(year)

    if years is not None:
        requested = [int(y) for y in years]
        missing = [y for y in requested if y not in available]
        if missing:
            raise ValueError(
                f"Requested weather years {missing} are not usable in {dataset_dir}; "
                f"available: {available}, skipped: {skipped_years}"
            )
        kept_years = sorted(requested)
    else:
        kept_years = sorted(available)

    if not kept_years:
        raise ValueError(f"No usable weather years in {dataset_dir}: {skipped_years}")

    year_pos = {y: i for i, y in enumerate(kept_years)}
    n_years = len(kept_years)

    # ---- Store ------------------------------------------------------------
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(out), mode="w")

    array_names: list[str] = []
    source_files: dict[str, str] = {}
    timestep_iso: Optional[list[str]] = None
    summary: list[tuple[str, tuple, tuple]] = []

    for path in parquets:
        name = array_name_for(path.stem)
        array_names.append(name)
        source_files[str(path.relative_to(dataset_dir))] = sha256_file(path)

        per_year = _estimated_read_bytes(path) > MAX_PARQUET_READ_BYTES
        if per_year:
            logger.info("Reading %s one weather year at a time (large file).", path.name)
            frames = None
            columns = list(
                pd.read_parquet(path, filters=[("weather_year", "==", kept_years[0])]).columns
            )
        else:
            frames = pd.read_parquet(path)
            columns = list(frames.columns)

        n_cols = len(columns)
        ch = chunk_hours_for(n_cols, chunk_hours, hours_per_year)
        arr = root.create_dataset(
            name,
            shape=(n_years, hours_per_year, n_cols),
            chunks=(1, ch, max(n_cols, 1)),
            dtype="float32",
            compressor=_blosc_shuffle(),
            fill_value=float("nan"),
            overwrite=True,
        )
        arr.attrs["source_file"] = str(path.relative_to(dataset_dir))
        arr.attrs["columns_array"] = f"{name}__columns"
        _vlen_str_array(root, f"{name}__columns", columns)

        for year in kept_years:
            if frames is not None:
                block = frames.loc[year]
            else:
                block = pd.read_parquet(path, filters=[("weather_year", "==", year)])
                block = block.loc[year]
                if list(block.columns) != columns:
                    raise ValueError(f"Column order changed between years in {path.name}")

            if block.shape[0] != hours_per_year:
                raise AssertionError(
                    f"{path.name} year {year}: {block.shape[0]} hours, expected {hours_per_year}"
                )

            if timestep_iso is None:
                timestep_iso = [pd.Timestamp(t).isoformat() for t in block.index.to_numpy()]

            arr[year_pos[year], :, :] = block.to_numpy(dtype=np.float32)

        summary.append((name, arr.shape, arr.chunks))
        del frames

    assert timestep_iso is not None and len(timestep_iso) == hours_per_year

    # ---- Coordinates ------------------------------------------------------
    wy = root.create_dataset(
        "weather_year", shape=(n_years,), chunks=(n_years,), dtype="int32", overwrite=True
    )
    wy[:] = np.array(kept_years, dtype=np.int32)

    hour = root.create_dataset(
        "hour",
        shape=(hours_per_year,),
        chunks=(hours_per_year,),
        dtype="int32",
        overwrite=True,
    )
    hour[:] = np.arange(hours_per_year, dtype=np.int32)

    _vlen_str_array(root, "timestep_iso", timestep_iso)

    # ---- Provenance -------------------------------------------------------
    root.attrs.update(
        {
            "converter_version": CONVERTER_VERSION,
            "dataset": dataset_dir.name,
            "created_utc": _utc_now(),
            "zap_commit": zap_commit(),
            "weather_years": [int(y) for y in kept_years],
            "hours_per_year": int(hours_per_year),
            "arrays": array_names,
            "source_files": source_files,
            "chunk_hours": int(chunk_hours),
            "skipped_years": skipped_years,
            # The snapshots carry the *model* year (2040 for the CH3 exports);
            # `detect_model_year` reads this back for the retirement rule.
            "model_year": int(pd.Timestamp(str(timestep_iso[0])).year),
        }
    )

    elapsed = time.time() - t0
    size = directory_size(out)
    print(f"Wrote {out}")
    for name, shape, chunks in summary:
        print(f"  {name:<28} shape={tuple(shape)} chunks={tuple(chunks)} dtype=float32")
    print(f"  years: {kept_years[0]}..{kept_years[-1]} ({n_years}), hours/year: {hours_per_year}")
    if skipped_years:
        print(f"  skipped years: {skipped_years}")
    print(f"  on-disk size: {_human_bytes(size)} ({size} bytes)")
    print(f"  elapsed: {elapsed:.1f} s")

    return out


# ===========================================================================
# Reader
# ===========================================================================


class WeatherStore:
    """Read-only view of ``weather.zarr``."""

    def __init__(self, path: Path, root):
        self.path = Path(path)
        self.root = root
        self.attrs = dict(root.attrs)
        self.years = np.asarray(root["weather_year"][:], dtype=np.int32)
        self.hours_per_year = int(self.attrs["hours_per_year"])
        self.arrays = list(self.attrs["arrays"])
        self._year_pos = {int(y): i for i, y in enumerate(self.years)}
        self._columns: dict[str, pd.Index] = {}

    @classmethod
    def open(cls, dataset_dir: Path) -> "WeatherStore":
        """Open ``<dataset_dir>/weather.zarr`` (or a path to the store itself)."""
        path = Path(dataset_dir)
        if path.name != STORE_NAME and path.suffix != ".zarr":
            path = path / STORE_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"No weather store at {path}. Build it with "
                f"`python -m zap.importers.wy_store convert --dataset-dir {dataset_dir}`."
            )
        return cls(path, zarr.open_group(str(path), mode="r"))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"WeatherStore({self.path}, years={self.years[0]}..{self.years[-1]}, "
            f"arrays={self.arrays})"
        )

    def has(self, array: str) -> bool:
        return array in self.arrays

    def columns(self, array: str) -> pd.Index:
        if array not in self._columns:
            if array not in self.arrays:
                raise KeyError(f"{array} is not in {self.path} (have {self.arrays})")
            values = self.root[f"{array}__columns"][:]
            self._columns[array] = pd.Index([str(v) for v in values])
        return self._columns[array]

    def _year_index(self, year: int) -> int:
        try:
            return self._year_pos[int(year)]
        except KeyError:
            raise KeyError(
                f"Weather year {year} is not in {self.path} (have {list(self.years)})"
            ) from None

    def read(self, array: str, year: int, hours: slice) -> np.ndarray:
        """``(n_hours, n_cols)`` float64 slice of one weather year."""
        if array not in self.arrays:
            raise KeyError(f"{array} is not in {self.path} (have {self.arrays})")
        start, stop, step = hours.indices(self.hours_per_year)
        if step != 1:
            raise ValueError("WeatherStore.read only supports contiguous hour slices")
        if stop <= start:
            raise ValueError(f"Empty hour slice {hours}")
        block = self.root[array][self._year_index(year), start:stop, :]
        return np.asarray(block, dtype=np.float64)

    def read_year(self, array: str, year: int) -> np.ndarray:
        return self.read(array, year, slice(0, self.hours_per_year))


def open_weather_store(dataset_dir: Path) -> WeatherStore:
    """Open ``<dataset_dir>/weather.zarr`` read-only."""
    return WeatherStore.open(dataset_dir)


# ===========================================================================
# Load options / index
# ===========================================================================


@dataclass(frozen=True)
class HourWindow:
    start: int = 0
    stop: int = DEFAULT_HOURS_PER_YEAR

    def __post_init__(self):
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(f"Invalid hour window [{self.start}, {self.stop})")

    def __len__(self) -> int:
        return int(self.stop - self.start)

    @property
    def slice(self) -> slice:
        return slice(int(self.start), int(self.stop))


@dataclass(frozen=True)
class LoadOptions:
    years: tuple[int, ...] = (2020,)
    window: HourWindow = HourWindow()
    voll: float = 10_000.0
    demand_scaling: Literal["none", "fixed", "peak_fraction"] = "none"
    scale_load: float = 1.0
    peak_capacity_fraction: float = 0.85
    clip_scale_to_one: bool = True
    ucap_derate: bool = False
    outage_draw: Optional[int] = None
    #: Investment year the system represents.  ``None`` -> :func:`detect_model_year`.
    model_year: Optional[int] = None
    #: Zero the as-built capacity of rows whose ``build_year + lifetime <= model_year``.
    apply_lifetimes: bool = True
    link_losses: bool = True
    ignore_min_power: bool = True
    export_mode: Literal["sink", "drop"] = "sink"
    carbon_tax: float = 0.0
    power_unit: float = 1.0
    cost_unit: float = 1.0
    storage_init_soc: float = 0.5
    storage_final_soc: float = 0.5
    #: Storage boundary condition of a block. ``"fixed"`` pins start and end at
    #: ``storage_init_soc`` / ``storage_final_soc``; ``"cyclic_free"`` only ties
    #: them together (``energy[0] == energy[T]``) and ignores both levels.
    storage_soc_mode: Literal["fixed", "cyclic_free"] = "cyclic_free"  # default by decision 2026-09-09
    dtype: str = "float64"


@dataclass
class SystemIndex:
    device_index: dict[str, int] = field(default_factory=dict)
    names: dict[str, pd.Index] = field(default_factory=dict)
    carrier: dict[str, np.ndarray] = field(default_factory=dict)
    bus: dict[str, np.ndarray] = field(default_factory=dict)
    emission_rates: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vre_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    thermal_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    #: generators, bool: rows sitting on an import bus (``<zone>_imports``)
    import_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def get(self, devices: list, cls_name: str):
        """Return the device of class ``cls_name`` from a device list."""
        if cls_name not in self.device_index:
            raise KeyError(f"{cls_name} is not part of this system ({list(self.device_index)})")
        return devices[self.device_index[cls_name]]


@dataclass
class LoadedSystem:
    network: PowerNetwork
    devices: list[AbstractDevice]
    index: SystemIndex
    meta: dict


# ===========================================================================
# Static data
# ===========================================================================


def _read_static(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    static_dir = Path(dataset_dir) / "static"
    if not static_dir.is_dir():
        raise FileNotFoundError(f"No static directory at {static_dir}")

    out: dict[str, pd.DataFrame] = {}
    for key, fname in STATIC_FILES.items():
        path = static_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path, index_col=0)
        if df.index.has_duplicates:
            raise ValueError(f"Duplicate names in {path}")
        out[key] = df
    return out


#: Public alias: planning reads the static tables to recover PyPSA extendability.
read_static = _read_static


# ===========================================================================
# Asset lifetimes (retirements)
# ===========================================================================

#: Static columns the retirement rule needs.  A table missing either of them
#: carries no lifetime information and nothing in it can retire.
LIFETIME_COLUMNS = ("build_year", "lifetime")

#: Component -> static table key for the two tables that carry lifetimes.
RETIREMENT_TABLES = {"Generator": "generators", "StorageUnit": "storage_units"}


def has_lifetime_columns(static_df: pd.DataFrame) -> bool:
    """True when a static table carries both ``build_year`` and ``lifetime``."""
    return all(c in static_df.columns for c in LIFETIME_COLUMNS)


def retired_mask(static_df: pd.DataFrame, model_year: Optional[int]) -> np.ndarray:
    """Rows whose asset life has ended by ``model_year``.

    A row is retired iff ``lifetime`` is finite and positive and
    ``build_year + lifetime <= model_year``; an infinite, missing, or **zero**
    lifetime never retires.  ``lifetime == 0`` is a pypsa-usa missing-data
    marker (e.g. Hoover hydro, build_year 1942) and is read as infinite by
    decision (Kamran, 2026-09-09).  Otherwise this is the rule
    ``zap/importers/pypsa.py::get_active_assets`` applied.  The PyPSA
    ``active`` column is *not* consulted: pypsa-usa writes ``active == True``
    everywhere because activity is resolved per investment period at solve time.

    ``model_year is None`` or a table without :data:`LIFETIME_COLUMNS` returns an
    all-``False`` mask, so callers can use this unconditionally.
    """
    n = len(static_df)
    if model_year is None or n == 0 or not has_lifetime_columns(static_df):
        return np.zeros(n, dtype=bool)
    build = pd.to_numeric(static_df["build_year"], errors="coerce").to_numpy(dtype=np.float64)
    life = pd.to_numeric(static_df["lifetime"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(build) & np.isfinite(life) & (life > 0)
    out = np.zeros(n, dtype=bool)
    out[ok] = (build[ok] + life[ok]) <= float(model_year)
    return out


def detect_model_year(dataset_dir: Path) -> tuple[int, str]:
    """``(model_year, source)`` for a dataset: the year the snapshots are stamped.

    The CH3 exports are single-investment-period PyPSA networks whose snapshots
    carry the *model* year (2040) while the weather-year index carries the
    weather year, so the model year is read, in order, from

    1. ``weather.zarr`` ``attrs["model_year"]`` (written by newer conversions),
    2. the first ``weather.zarr`` ``timestep_iso`` stamp,
    3. ``meta/wy*.json`` -> ``scenario.planning_horizons``,
    4. the ``timestep`` index of a ``timeseries/*.parquet``.
    """
    dataset_dir = Path(dataset_dir)

    store_path = dataset_dir / STORE_NAME
    if store_path.exists():
        try:
            root = zarr.open_group(str(store_path), mode="r")
            attrs = dict(root.attrs)
        except (OSError, ValueError, KeyError):  # pragma: no cover - unreadable store
            root, attrs = None, {}
        if attrs.get("model_year") is not None:
            return int(attrs["model_year"]), "weather_store_attrs"
        if root is not None and "timestep_iso" in root:
            stamps = root["timestep_iso"][:1]
            if len(stamps):
                return int(pd.Timestamp(str(stamps[0])).year), "weather_store:timestep_iso"

    meta_dir = dataset_dir / "meta"
    if meta_dir.is_dir():
        for meta_path in sorted(meta_dir.glob("wy*.json")):
            try:
                raw = json.loads(meta_path.read_text())
            except (OSError, ValueError):  # pragma: no cover - unreadable meta
                continue
            horizons = (raw.get("scenario") or {}).get("planning_horizons")
            if horizons:
                return int(next(iter(horizons))), f"meta/{meta_path.name}:planning_horizons"

    ts_dir = dataset_dir / "timeseries"
    if ts_dir.is_dir():
        for path in sorted(ts_dir.glob("*.parquet")):
            index = pd.read_parquet(path, columns=[]).index
            if "timestep" not in (index.names or []):
                continue
            stamps = index.get_level_values("timestep")
            if len(stamps):
                return int(pd.Timestamp(stamps[0]).year), f"timeseries/{path.name}:timestep"

    raise ValueError(
        f"Cannot determine the model year of {dataset_dir}: no model_year/timestep_iso in "
        "the weather store, no scenario.planning_horizons in meta/wy*.json and no parquet "
        "timestep index. Pass LoadOptions.model_year explicitly."
    )


def apply_retirements(
    static: dict[str, pd.DataFrame], model_year: Optional[int]
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Zero the as-built capacity of every retired generator / storage row.

    Rows are *kept*: order, names and count are unchanged, so outage-pool
    offsets, :class:`SystemIndex`, ``design.json`` and ``Design.apply`` all keep
    working; only ``p_nom`` becomes 0.  Returns ``(static, summary)`` where
    ``static`` holds copies of the two touched tables.
    """
    out = dict(static)
    summary: dict[str, Any] = {
        "model_year": None if model_year is None else int(model_year),
        "retired_rows": {},
        "retired_capacity_mw": {},
        "retired_capacity_mw_by_carrier": {},
        "retired_names": {},
    }
    for component, key in RETIREMENT_TABLES.items():
        df = static.get(key)
        if df is None:
            continue
        mask = retired_mask(df, model_year)
        capacity = df["p_nom"].to_numpy(dtype=np.float64)
        by_carrier: dict[str, float] = {}
        if mask.any():
            df = df.copy()
            if "carrier" in df.columns:
                grouped = (
                    pd.Series(capacity[mask], index=df["carrier"].to_numpy()[mask])
                    .groupby(level=0)
                    .sum()
                    .sort_values(ascending=False)
                )
                by_carrier = {str(k): float(v) for k, v in grouped.items()}
            df.loc[mask, "p_nom"] = 0.0
            out[key] = df
        summary["retired_rows"][component] = int(mask.sum())
        summary["retired_capacity_mw"][component] = float(capacity[mask].sum())
        summary["retired_capacity_mw_by_carrier"][component] = by_carrier
        summary["retired_names"][component] = [str(n) for n in df.index[mask]]
    return out, summary


def _thermal_carriers() -> frozenset[str]:
    """Carriers covered by the outage pool (WP2), with a static fallback."""
    try:  # pragma: no cover - depends on WP2 landing
        from zap.reliability.outages import load_outage_params

        return frozenset(load_outage_params().carriers)
    except Exception:
        return DEFAULT_THERMAL_CARRIERS


def _terminals(series: pd.Series, bus_index: dict[str, int], what: str) -> np.ndarray:
    missing = sorted(set(series) - set(bus_index))
    if missing:
        raise KeyError(f"{what} reference buses not in buses.csv: {missing}")
    return series.map(bus_index).to_numpy(dtype=int)


def _concat_years(
    store: WeatherStore, array: str, years: Sequence[int], window: HourWindow
) -> np.ndarray:
    """``(n_hours, n_cols)``, years concatenated along the time axis in order."""
    blocks = [store.read(array, y, window.slice) for y in years]
    return np.concatenate(blocks, axis=0)


def _dense_from_store(
    store: WeatherStore,
    array: str,
    names: pd.Index,
    static_fallback: np.ndarray,
    years: Sequence[int],
    window: HourWindow,
    what: str,
) -> np.ndarray:
    """``(n_rows, n_hours)``: store columns where present, static scalar elsewhere."""
    n_hours = len(window) * len(years)
    dense = np.empty((len(names), n_hours), dtype=np.float64)

    if store.has(array):
        columns = store.columns(array)
        pos = {c: i for i, c in enumerate(columns)}
        col_of = np.array([pos.get(n, -1) for n in names], dtype=int)
        unused = sorted(set(columns) - set(names))
        if unused:
            logger.warning("%s columns not present in the static table: %s", array, unused)
    else:
        col_of = np.full(len(names), -1, dtype=int)

    have = col_of >= 0
    if have.any():
        data = _concat_years(store, array, years, window)
        if np.isnan(data).any():
            raise ValueError(
                f"{array} contains NaN in the selected window; the store may be incomplete."
            )
        dense[have, :] = data[:, col_of[have]].T
    if (~have).any():
        logger.info(
            "%s: %d %s rows have no column in %s; using the static value.",
            what,
            int((~have).sum()),
            what,
            array,
        )
        dense[~have, :] = np.asarray(static_fallback, dtype=np.float64)[~have, None]

    return dense


# ===========================================================================
# UCAP / outage composition
# ===========================================================================


def _ucap_factors(dataset_dir: Path, component: str, rows: pd.Index) -> tuple[np.ndarray, str]:
    """UCAP derate per source row (1.0 when the row is absent), plus the csv sha256."""
    path = Path(dataset_dir) / UCAP_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"ucap_derate=True but {path} does not exist. Build it with "
            f"`python -m zap.reliability.outages ucap --dataset-dir {dataset_dir}`."
        )
    table = pd.read_csv(path)
    for required in ("component", "row"):
        if required not in table.columns:
            raise ValueError(f"{path} has no '{required}' column")

    column = next((c for c in UCAP_COLUMN_PREFERENCE if c in table.columns), None)
    if column is None:
        raise ValueError(f"{path} has none of the columns {UCAP_COLUMN_PREFERENCE}")

    sub = table[table["component"] == component]
    factors = pd.Series(sub[column].to_numpy(), index=sub["row"].to_numpy())
    if factors.index.has_duplicates:
        raise ValueError(f"{path} has duplicate rows for component {component}")

    # Fall back to the analytic column where the preferred one is missing.
    if column != "ucap_analytic" and "ucap_analytic" in sub.columns:
        analytic = pd.Series(sub["ucap_analytic"].to_numpy(), index=sub["row"].to_numpy())
        factors = factors.fillna(analytic)

    values = factors.reindex(rows).to_numpy(dtype=np.float64)
    return np.where(np.isnan(values), 1.0, values), sha256_file(path)


def _open_outage_store(dataset_dir: Path):
    path = Path(dataset_dir) / OUTAGE_STORE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"outage_draw was requested but {path} does not exist. Build it with "
            f"`python -m zap.reliability.outages generate --dataset-dir {dataset_dir}`."
        )
    return path, zarr.open_group(str(path), mode="r")


def _unit_pool_from_store(root) -> Any:
    """Rebuild WP2's ``UnitPool`` from the outage store's coordinate arrays."""
    from zap.reliability.outages import UnitPool

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
    if first.index.has_duplicates:  # pragma: no cover - defensive
        raise ValueError("Outage pool row names are not unique")
    return UnitPool(
        table=table,
        row_offset=first["row_offset"].astype(int).to_dict(),
        row_units=first["row_units"].astype(int).to_dict(),
        row_size=first["unit_size_mw"].astype(float).to_dict(),
    )


def _outage_availability(
    dataset_dir: Path,
    draw: int,
    component: str,
    rows: pd.Index,
    capacities: pd.Series,
    years: Sequence[int],
    window: HourWindow,
) -> tuple[np.ndarray, dict]:
    """``(n_rows, n_hours)`` availability from ``outages.zarr`` via WP2."""
    from zap.reliability.outages import row_availability

    path, root = _open_outage_store(dataset_dir)
    attrs = dict(root.attrs)
    pool = _unit_pool_from_store(root)

    store_years = [int(y) for y in np.asarray(root["weather_year"][:])]
    draws = [int(d) for d in np.asarray(root["draw"][:])]
    if draw not in draws:
        raise KeyError(f"Draw {draw} is not in {path} (have {draws})")
    di = draws.index(draw)

    known = set(pool.row_offset)
    blocks = []
    for year in years:
        if int(year) not in store_years:
            raise KeyError(f"Weather year {year} is not in {path} (have {store_years})")
        yi = store_years.index(int(year))
        if "done" in root and not bool(np.asarray(root["done"][yi, di])):
            raise ValueError(
                f"Outage chunk (year={year}, draw={draw}) was never generated in {path}; "
                "run the outage generator for it before loading."
            )
        up = np.asarray(
            root["available"][yi, di, window.start : window.stop, :], dtype=np.uint8
        ).T  # (n_units, n_hours)
        if up.size and up.max() > 1:
            raise ValueError(
                f"Outage store {path} holds fill values (>1) for year={year}, draw={draw} "
                f"in hours [{window.start}, {window.stop}); the chunk is incomplete."
            )
        pooled = [r for r in rows if r in known]
        avail = np.ones((len(window), len(rows)), dtype=np.float64)
        if pooled:
            sub = row_availability(up, pool, capacities.loc[pooled], pooled)
            positions = [rows.get_loc(r) for r in pooled]
            avail[:, positions] = sub
        blocks.append(avail)

    stacked = np.concatenate(blocks, axis=0).T  # (n_rows, n_hours)
    del component  # component is implicit in the row names; kept for signature clarity
    return stacked, attrs


# ===========================================================================
# Device builders
# ===========================================================================


def _build_generators(
    static: dict[str, pd.DataFrame],
    store: WeatherStore,
    bus_index: dict[str, int],
    options: LoadOptions,
    outage_factor: Optional[np.ndarray],
    ucap_factor: Optional[np.ndarray],
) -> tuple[Generator, np.ndarray, np.ndarray]:
    gens = static["generators"]
    carriers = static["carriers"]

    weather = _dense_from_store(
        store,
        "generators_p_max_pu",
        gens.index,
        gens["p_max_pu"].to_numpy(dtype=np.float64),
        options.years,
        options.window,
        "generator",
    )

    factor = np.ones_like(weather)
    if ucap_factor is not None:
        factor = np.broadcast_to(ucap_factor[:, None], weather.shape).copy()
    elif outage_factor is not None:
        factor = outage_factor
    dynamic_capacity = weather * factor

    missing_carriers = sorted(set(gens["carrier"]) - set(carriers.index))
    if missing_carriers:
        raise KeyError(f"Generator carriers missing from carriers.csv: {missing_carriers}")

    efficiency = gens["efficiency"].to_numpy(dtype=np.float64)
    if np.any(efficiency <= 0):
        raise ValueError("Generators with non-positive efficiency; cannot form emission rates.")
    fuel_rate = carriers.loc[gens["carrier"].to_numpy(), "co2_emissions"].to_numpy(float)
    emission_rates = fuel_rate / efficiency

    linear_cost = (
        gens["marginal_cost"].to_numpy(dtype=np.float64) + options.carbon_tax * emission_rates
    )
    p_nom = gens["p_nom"].to_numpy(dtype=np.float64)

    dev = Generator(
        num_nodes=len(bus_index),
        name=gens.index,
        terminal=_terminals(gens["bus"], bus_index, "generators.csv"),
        nominal_capacity=p_nom,
        dynamic_capacity=dynamic_capacity,
        linear_cost=linear_cost,
        capital_cost=gens["capital_cost"].to_numpy(dtype=np.float64),
        min_nominal_capacity=p_nom.copy(),
        max_nominal_capacity=p_nom.copy(),
        emission_rates=emission_rates,
    )
    dev.fuel_type = gens["carrier"].to_numpy()
    return dev, weather, emission_rates


def _build_loads(
    static: dict[str, pd.DataFrame],
    store: WeatherStore,
    bus_index: dict[str, int],
    options: LoadOptions,
    applied_scale: float,
) -> Load:
    loads = static["loads"]
    profile = _dense_from_store(
        store,
        "loads_p_set",
        loads.index,
        loads["p_set"].to_numpy(dtype=np.float64),
        options.years,
        options.window,
        "load",
    )
    return Load(
        num_nodes=len(bus_index),
        name=loads.index,
        terminal=_terminals(loads["bus"], bus_index, "loads.csv"),
        load=profile * applied_scale,
        linear_cost=np.full(len(loads), float(options.voll)),
        quadratic_cost=None,
    )


def _build_links(
    static: dict[str, pd.DataFrame],
    store: WeatherStore,
    bus_index: dict[str, int],
    options: LoadOptions,
):
    if DirectedLine is None:
        raise ImportError(
            "zap.devices.transporter.DirectedLine is required to build this system "
            "(WP3 of the Phase-1 spec) but is not available."
        )

    links = static["links"]
    eff = links["efficiency"].to_numpy(dtype=np.float64)
    if not options.link_losses:
        eff = np.ones_like(eff)
    if np.any(eff <= 0):
        raise ValueError("Links with non-positive efficiency")

    # PyPSA charges marginal_cost on p0; zap prices power[1] = efficiency * flow,
    # so the per-unit cost of the sink-end injection is marginal_cost / efficiency.
    cost = (
        _dense_from_store(
            store,
            "links_marginal_cost",
            links.index,
            links["marginal_cost"].to_numpy(dtype=np.float64),
            options.years,
            options.window,
            "link",
        )
        / eff[:, None]
    )

    p_nom = links["p_nom"].to_numpy(dtype=np.float64)
    return DirectedLine(
        num_nodes=len(bus_index),
        name=links.index,
        source_terminal=_terminals(links["bus0"], bus_index, "links.csv bus0"),
        sink_terminal=_terminals(links["bus1"], bus_index, "links.csv bus1"),
        min_power=links["p_min_pu"].to_numpy(dtype=np.float64) * eff,
        max_power=links["p_max_pu"].to_numpy(dtype=np.float64) * eff,
        linear_cost=cost,
        efficiency=eff,
        nominal_capacity=p_nom,
        capital_cost=links["capital_cost"].to_numpy(dtype=np.float64),
        min_nominal_capacity=p_nom.copy(),
        max_nominal_capacity=p_nom.copy(),
    )


def _storage_supports_availability() -> bool:
    import inspect

    return "power_availability" in inspect.signature(StorageUnit.__init__).parameters


def _build_storage(
    static: dict[str, pd.DataFrame],
    bus_index: dict[str, int],
    options: LoadOptions,
    n_hours: int,
    outage_factor: Optional[np.ndarray],
    ucap_factor: Optional[np.ndarray],
) -> StorageUnit:
    units = static["storage_units"]
    p_nom = units["p_nom"].to_numpy(dtype=np.float64)

    availability: Optional[np.ndarray] = None
    if ucap_factor is not None:
        availability = np.broadcast_to(ucap_factor[:, None], (len(units), 1)).copy()
    elif outage_factor is not None:
        availability = outage_factor

    kwargs: dict[str, Any] = {}
    if availability is not None:
        if not _storage_supports_availability():
            raise NotImplementedError(
                "StorageUnit.power_availability (WP3.1) is required for ucap_derate / "
                "outage_draw on storage but is not available in this zap build."
            )
        kwargs["power_availability"] = availability
    elif _storage_supports_availability():
        kwargs["power_availability"] = np.ones((len(units), 1))

    del n_hours  # availability already carries the horizon when time varying

    return StorageUnit(
        num_nodes=len(bus_index),
        name=units.index,
        terminal=_terminals(units["bus"], bus_index, "storage_units.csv"),
        power_capacity=p_nom,
        duration=units["max_hours"].to_numpy(dtype=np.float64),
        charge_efficiency=units["efficiency_store"].to_numpy(dtype=np.float64),
        discharge_efficiency=units["efficiency_dispatch"].to_numpy(dtype=np.float64),
        # state_of_charge_initial is 0 for every row in these datasets; the plan
        # mandates cyclic 50 %, so we use the constant instead (spec 2.5).
        # Both are ignored when `storage_soc_mode == "cyclic_free"`.
        initial_soc=np.full(len(units), float(options.storage_init_soc)),
        final_soc=np.full(len(units), float(options.storage_final_soc)),
        soc_mode=str(options.storage_soc_mode),
        linear_cost=units["marginal_cost"].to_numpy(dtype=np.float64),
        capital_cost=units["capital_cost"].to_numpy(dtype=np.float64),
        min_power_capacity=p_nom.copy(),
        max_power_capacity=p_nom.copy(),
        **kwargs,
    )


def _build_export_sinks(
    static: dict[str, pd.DataFrame], bus_index: dict[str, int]
) -> Optional[Injector]:
    stores = static["stores"]
    exports = stores[stores["carrier"] == "exports"]
    if exports.empty:
        return None

    links = static["links"]
    incoming = links.groupby("bus1")["p_nom"].sum()
    capacity = incoming.reindex(exports["bus"].to_numpy()).to_numpy(dtype=np.float64)
    if np.any(np.isnan(capacity)) or np.any(capacity <= 0):
        bad = exports.index[np.isnan(capacity) | (capacity <= 0)].tolist()
        raise ValueError(f"Export stores with no incoming export-link capacity: {bad}")

    # The sink's capacity is constant, so its bounds and cost stay static
    # ``(n, 1)`` arrays: ``time_horizon`` is then 0 and the device survives
    # ``sample_time`` into a block of any length unchanged.
    n = len(exports)
    return Injector(
        num_nodes=len(bus_index),
        name=exports.index,
        terminal=_terminals(exports["bus"], bus_index, "stores.csv"),
        nominal_capacity=capacity,
        min_power=-np.ones(n),
        max_power=np.zeros(n),
        linear_cost=np.zeros(n),
    )


# ===========================================================================
# Demand scaling (D9)
# ===========================================================================


IMPORT_BUS_SUFFIX = "_imports"


def import_bus_mask(bus: Sequence[str]) -> np.ndarray:
    """Rows that sit on an import bus (bus name ending in ``_imports``)."""
    return np.array([str(b).endswith(IMPORT_BUS_SUFFIX) for b in bus], dtype=bool)


def available_capacity(
    system: Optional["LoadedSystem"] = None,
    *,
    capacity_mw: Optional[np.ndarray] = None,
    availability: Optional[np.ndarray] = None,
    import_mask: Optional[np.ndarray] = None,
    bus: Optional[Sequence[str]] = None,
    storage_capacity_mw: Optional[np.ndarray] = None,
    storage_bus: Optional[Sequence[str]] = None,
    include_storage: bool = False,
    include_imports: bool = False,
    per_bus: bool = False,
):
    """Hourly available capacity in MW.

    Default: in-state generators only -- import-bus rows excluded, storage
    excluded -- summed over the system, i.e. ``max_t`` of the return value is the
    headline ``peak_available_mw`` used by the D9 demand scaling.

    Parameters
    ----------
    system:
        A :class:`LoadedSystem`; every array below defaults to the corresponding
        quantity of that system.
    capacity_mw:
        Per generator row, MW.  Defaults to the system's ``p_nom``.  A designed
        system passes its ``p_nom_opt`` here.
    availability:
        ``(n_hours, n_gen)`` effective availability in ``[0, 1]``.  Defaults to
        the system's ``Generator.dynamic_capacity`` transposed, i.e. weather
        availability already multiplied by any outage draw or UCAP derate.
    per_bus:
        Return a ``(n_hours, n_bus)`` DataFrame with bus names as columns instead
        of an ``(n_hours,)`` array.
    """
    if system is not None:
        generator = system.index.get(system.devices, "Generator")
        if capacity_mw is None:
            capacity_mw = np.asarray(generator.nominal_capacity, dtype=np.float64).reshape(-1)
        if availability is None:
            availability = np.asarray(generator.dynamic_capacity, dtype=np.float64).T
        if import_mask is None:
            import_mask = system.index.import_mask
        if bus is None:
            bus = system.index.bus.get("Generator")
        if storage_capacity_mw is None and "StorageUnit" in system.index.device_index:
            storage = system.index.get(system.devices, "StorageUnit")
            storage_capacity_mw = np.asarray(storage.power_capacity, dtype=np.float64).reshape(-1)
            if storage_bus is None:
                storage_bus = system.index.bus.get("StorageUnit")

    if capacity_mw is None or availability is None:
        raise ValueError("available_capacity needs either a system or capacity/availability")

    capacity_mw = np.asarray(capacity_mw, dtype=np.float64).reshape(-1)
    availability = np.atleast_2d(np.asarray(availability, dtype=np.float64))
    if availability.shape[1] != capacity_mw.size:
        raise ValueError(
            f"availability has {availability.shape[1]} columns but there are "
            f"{capacity_mw.size} generator rows"
        )

    keep = np.ones(capacity_mw.size, dtype=bool)
    if import_mask is not None and not include_imports:
        keep &= ~np.asarray(import_mask, dtype=bool)

    contribution = availability[:, keep] * capacity_mw[keep]

    if not per_bus:
        total = contribution.sum(axis=1)
        if include_storage and storage_capacity_mw is not None:
            total = total + float(np.sum(storage_capacity_mw))
        return total

    if bus is None:
        raise ValueError("per_bus=True needs bus labels for the generator rows")
    frame = pd.DataFrame(contribution, columns=pd.Index(np.asarray(bus)[keep]))
    out = frame.T.groupby(level=0).sum().T
    if include_storage and storage_capacity_mw is not None:
        if storage_bus is None:
            raise ValueError("per_bus=True with storage needs bus labels for storage rows")
        extra = pd.Series(np.asarray(storage_capacity_mw), index=pd.Index(storage_bus))
        extra = extra.groupby(level=0).sum()
        for name, value in extra.items():
            if name not in out.columns:
                out[name] = 0.0
            out[name] = out[name] + value
    return out.sort_index(axis=1)


def peak_available_mw(system: Optional["LoadedSystem"] = None, **kwargs) -> float:
    """``max`` over hours of :func:`available_capacity` (never per-bus)."""
    kwargs.pop("per_bus", None)
    return float(np.max(available_capacity(system, per_bus=False, **kwargs)))


def _peak_metrics(
    static: dict[str, pd.DataFrame], store: WeatherStore, years: Sequence[int]
) -> dict[str, float]:
    """Peak demand and the three peak-available-capacity denominators.

    All three are computed over the *full* selected weather year(s) using weather
    availability only (no outage draw, no UCAP derate), per decision D9.  The
    headline ``peak_available_mw`` is in-state generators only.
    """
    full = HourWindow(0, store.hours_per_year)
    gens = static["generators"]

    weather = _dense_from_store(
        store,
        "generators_p_max_pu",
        gens.index,
        gens["p_max_pu"].to_numpy(dtype=np.float64),
        years,
        full,
        "generator",
    )
    p_nom = gens["p_nom"].to_numpy(dtype=np.float64)
    mask = import_bus_mask(gens["bus"].to_numpy())
    storage_p_nom = static["storage_units"]["p_nom"].to_numpy(dtype=np.float64)

    common = dict(capacity_mw=p_nom, availability=weather.T, import_mask=mask)
    in_state = available_capacity(**common)
    with_storage = available_capacity(
        **common, storage_capacity_mw=storage_p_nom, include_storage=True
    )
    with_imports = available_capacity(**common, include_imports=True)

    loads = static["loads"]
    profile = _dense_from_store(
        store,
        "loads_p_set",
        loads.index,
        loads["p_set"].to_numpy(dtype=np.float64),
        years,
        full,
        "load",
    )

    return {
        "peak_load_mw": float(profile.sum(axis=0).max()),
        "peak_available_mw": float(in_state.max()),
        "peak_available_incl_storage_mw": float(with_storage.max()),
        "peak_available_incl_imports_mw": float(with_imports.max()),
        "annual_demand_mwh": float(profile.sum()),
    }


def _demand_scale(options: LoadOptions, peak_load: float, peak_available: float):
    implied = float("nan")
    if peak_load > 0:
        implied = options.peak_capacity_fraction * peak_available / peak_load

    if options.demand_scaling == "none":
        applied = 1.0
    elif options.demand_scaling == "fixed":
        applied = float(options.scale_load)
    elif options.demand_scaling == "peak_fraction":
        if peak_load <= 0:
            raise ValueError("peak_fraction demand scaling requires a positive peak load")
        applied = min(1.0, implied) if options.clip_scale_to_one else implied
    else:
        raise ValueError(f"Unknown demand_scaling {options.demand_scaling!r}")

    return implied, float(applied)


# ===========================================================================
# load_system
# ===========================================================================


def load_system(dataset_dir: Path, options: Optional[LoadOptions] = None) -> LoadedSystem:
    """Build a ``zap`` system from a dataset directory and its weather store."""
    dataset_dir = Path(dataset_dir)
    options = options if options is not None else LoadOptions()

    if options.ucap_derate and options.outage_draw is not None:
        raise ValueError(
            "ucap_derate and outage_draw are mutually exclusive: a UCAP derate is a "
            "frozen approximation of the outage draws."
        )
    if not options.ignore_min_power:
        raise NotImplementedError(
            "ignore_min_power=False (unit-commitment minimum stable levels) is not "
            "implemented; see decision D10."
        )
    if options.export_mode not in ("sink", "drop"):
        raise ValueError(f"Unknown export_mode {options.export_mode!r}")
    if options.dtype != "float64":
        raise NotImplementedError("Only float64 systems are supported in phase 1")

    store = WeatherStore.open(dataset_dir)
    static = _read_static(dataset_dir)

    # ---- Asset lifetimes -------------------------------------------------
    # pypsa-usa exports carry `active == True` on every row because PyPSA
    # resolves activity per investment period at solve time; the retirement rule
    # lives in `build_year + lifetime` (see `retired_mask`).
    model_year: Optional[int] = None
    model_year_source = "disabled"
    if options.apply_lifetimes:
        if options.model_year is not None:
            model_year, model_year_source = int(options.model_year), "LoadOptions.model_year"
        else:
            model_year, model_year_source = detect_model_year(dataset_dir)
    static, retirements = apply_retirements(static, model_year)
    retirements["model_year_source"] = model_year_source
    retirements["apply_lifetimes"] = bool(options.apply_lifetimes)
    n_retired = sum(retirements["retired_rows"].values())
    if n_retired:
        by_carrier = retirements["retired_capacity_mw_by_carrier"]["Generator"]
        logger.info(
            "Lifetimes (model_year=%s from %s): retired %d rows / %.1f MW "
            "(generators %d rows %.1f MW, storage %d rows %.1f MW); generator carriers: %s",
            model_year,
            model_year_source,
            n_retired,
            sum(retirements["retired_capacity_mw"].values()),
            retirements["retired_rows"]["Generator"],
            retirements["retired_capacity_mw"]["Generator"],
            retirements["retired_rows"]["StorageUnit"],
            retirements["retired_capacity_mw"]["StorageUnit"],
            ", ".join(f"{k} {v:.1f} MW" for k, v in by_carrier.items()) or "none",
        )
    else:
        logger.info(
            "Lifetimes (model_year=%s from %s): no rows retired.", model_year, model_year_source
        )

    years = tuple(int(y) for y in options.years)
    if not years:
        raise ValueError("LoadOptions.years is empty")
    window = options.window
    if window.stop > store.hours_per_year:
        raise ValueError(
            f"Window {window} exceeds the store's {store.hours_per_year} hours per year"
        )
    n_hours = len(window) * len(years)

    buses = static["buses"].index
    bus_index = {b: i for i, b in enumerate(buses)}

    # ---- Reliability heuristics ------------------------------------------
    gen_ucap = storage_ucap = None
    gen_outage = storage_outage = None
    ucap_sha: Optional[str] = None
    outage_attrs: Optional[dict] = None

    if options.ucap_derate:
        gen_ucap, ucap_sha = _ucap_factors(dataset_dir, "Generator", static["generators"].index)
        storage_ucap, _ = _ucap_factors(dataset_dir, "StorageUnit", static["storage_units"].index)
    elif options.outage_draw is not None:
        gen_outage, outage_attrs = _outage_availability(
            dataset_dir,
            options.outage_draw,
            "Generator",
            static["generators"].index,
            static["generators"]["p_nom"],
            years,
            window,
        )
        storage_outage, _ = _outage_availability(
            dataset_dir,
            options.outage_draw,
            "StorageUnit",
            static["storage_units"].index,
            static["storage_units"]["p_nom"],
            years,
            window,
        )

    # ---- Demand scaling (D9) ---------------------------------------------
    peaks = _peak_metrics(static, store, years)
    peak_load = peaks["peak_load_mw"]
    peak_available = peaks["peak_available_mw"]
    implied_scale, applied_scale = _demand_scale(options, peak_load, peak_available)

    # ---- Devices, in the D2 order ----------------------------------------
    generator, _, emission_rates = _build_generators(
        static, store, bus_index, options, gen_outage, gen_ucap
    )
    load = _build_loads(static, store, bus_index, options, applied_scale)
    line = _build_links(static, store, bus_index, options)
    storage = _build_storage(static, bus_index, options, n_hours, storage_outage, storage_ucap)
    sink = _build_export_sinks(static, bus_index) if options.export_mode == "sink" else None

    ordered = [
        ("Generator", generator),
        ("Load", load),
        ("DirectedLine", line),
        ("StorageUnit", storage),
        ("ExportSink", sink),
    ]

    devices: list[AbstractDevice] = []
    index = SystemIndex()
    key_to_static = {
        "Generator": "generators",
        "Load": "loads",
        "DirectedLine": "links",
        "StorageUnit": "storage_units",
        "ExportSink": "stores",
    }
    for key, dev in ordered:
        if dev is None or dev.num_devices == 0:
            continue
        index.device_index[key] = len(devices)
        devices.append(dev)

        table = static[key_to_static[key]]
        if key == "ExportSink":
            table = table[table["carrier"] == "exports"]
        index.names[key] = pd.Index(table.index)
        index.carrier[key] = table["carrier"].to_numpy() if "carrier" in table else np.array([])
        bus_col = "bus1" if key == "DirectedLine" else "bus"
        index.bus[key] = table[bus_col].to_numpy()

    gen_carriers = static["generators"]["carrier"].to_numpy()
    index.emission_rates = emission_rates
    index.vre_mask = np.isin(gen_carriers, list(VRE_CARRIERS))
    index.thermal_mask = np.isin(gen_carriers, list(_thermal_carriers()))
    index.import_mask = import_bus_mask(static["generators"]["bus"].to_numpy())

    for dev in devices:
        dev.scale_costs(options.cost_unit)
        dev.scale_power(options.power_unit)

    network = PowerNetwork(len(buses))

    meta = {
        "dataset": dataset_dir.name,
        "years": list(years),
        "window_start": int(window.start),
        "window_stop": int(window.stop),
        "n_hours": int(n_hours),
        "peak_load_mw": peak_load,
        "peak_available_mw": peak_available,
        "peak_available_incl_storage_mw": peaks["peak_available_incl_storage_mw"],
        "peak_available_incl_imports_mw": peaks["peak_available_incl_imports_mw"],
        "annual_demand_mwh": peaks["annual_demand_mwh"],
        "implied_scale": implied_scale,
        "applied_scale": applied_scale,
        "voll": float(options.voll),
        "ucap_derate": bool(options.ucap_derate),
        "outage_draw": options.outage_draw,
        "model_year": model_year,
        "model_year_source": model_year_source,
        "apply_lifetimes": bool(options.apply_lifetimes),
        "retired_rows": retirements["retired_rows"],
        "retired_capacity_mw": retirements["retired_capacity_mw"],
        "retired_capacity_mw_by_carrier": retirements["retired_capacity_mw_by_carrier"],
        "retired_names": retirements["retired_names"],
        "link_losses": bool(options.link_losses),
        "storage_soc_mode": str(options.storage_soc_mode),
        "storage_init_soc": float(options.storage_init_soc),
        "storage_final_soc": float(options.storage_final_soc),
        "power_unit": float(options.power_unit),
        "cost_unit": float(options.cost_unit),
        "weather_store_attrs": store.attrs,
        "ucap_csv_sha256": ucap_sha,
        "outage_store_attrs": outage_attrs,
    }

    return LoadedSystem(network=network, devices=devices, index=index, meta=meta)


# ===========================================================================
# CLI
# ===========================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m zap.importers.wy_store")
    sub = parser.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="Convert a dataset's parquet timeseries to zarr.")
    conv.add_argument("--dataset-dir", required=True, type=Path)
    conv.add_argument("--out", type=Path, default=None)
    conv.add_argument("--years", type=int, nargs="*", default=None)
    conv.add_argument("--chunk-hours", type=int, default=DEFAULT_CHUNK_HOURS)
    conv.add_argument("--overwrite", action="store_true")

    info = sub.add_parser("info", help="Print the schema of an existing weather store.")
    info.add_argument("--dataset-dir", required=True, type=Path)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "convert":
        convert_dataset(
            args.dataset_dir,
            args.out,
            years=args.years,
            chunk_hours=args.chunk_hours,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "info":
        store = WeatherStore.open(args.dataset_dir)
        print(f"{store.path}")
        for key, value in store.attrs.items():
            if key == "source_files":
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        for name in store.arrays:
            arr = store.root[name]
            print(f"  {name}: shape={arr.shape} chunks={arr.chunks} dtype={arr.dtype}")
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
