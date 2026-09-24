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

* D2 -- device order is ``[Generator, Load, DirectedLine, StorageUnit, ExportSink]``,
  followed by ``[PerfectGenerator, PerfectLink]`` only when
  ``LoadOptions.perfect_capacity_mw > 0`` (perfect-capacity hub spec, 2026-09-23).
* D3 -- load shedding is ``Load.linear_cost = voll``; no slack device is added.
* D4 -- ``zap.devices.store.Store`` is not used; export stores become ``Injector`` sinks.
* D5 -- links become ``DirectedLine`` (directional, signed linear cost, efficiency).
* D9 -- ``peak_fraction`` demand scaling is clipped at 1.0 by default.
* D10 -- ``p_min_pu`` (minimum stable level) is ignored unless
  ``LoadOptions.commitment == "minimal"``, which builds the linearised
  unit-commitment device of ``memory/plans/2026-09-14-uc-minimal-impl-spec.md``
  (start-up cost + minimum stable level on the file's ``committable`` rows).
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
from typing import Any, Literal, Mapping, Optional, Sequence

import numcodecs
import numpy as np
import pandas as pd
import zarr

from zap.devices.abstract import AbstractDevice
from zap.devices.injector import Generator, Injector, Load
from zap.devices.perfect_capacity import (
    PERFECT_CARRIER,
    PERFECT_HUB_BUS,
    PERFECT_LINK_HEADROOM,
    perfect_capacity_devices,
    perfect_load_nodes,
)
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

DEFAULT_HOURS_PER_YEAR = 365 * 24
DEFAULT_CHUNK_HOURS = 168

#: If a chunk would be smaller than this many bytes, widen it to a whole year.
CHUNK_BYTE_FLOOR = 65536

#: Above this estimated in-memory size (float64) a parquet file is read one
#: weather year at a time instead of in one go.
MAX_PARQUET_READ_BYTES = 4 * 1024**3

#: Carriers treated as variable renewables (used for the curtailment metric).
VRE_CARRIERS = frozenset({"solar", "onwind", "offwind_floating"})

#: Thermal-carrier list used for ``SystemIndex.thermal_mask``.  zap ships no
#: parameter table any more (the numbers are CH3 policy), so this is the list,
#: not a fallback: kept in sync by hand with ``ch3/ra/configs/outage_params.yaml``.
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


# ===========================================================================
# The hour-of-day import-limit profile (import-limit profile spec 3.3)
# ===========================================================================

#: Hours of a non-leap model year, month by month.  The CA2040 exports stamp
#: their timesteps ``2040-01-01 00:00 ... 2040-12-31 23:00`` but carry **8760**
#: rows with a 28-day February, even though 2040 is a leap year, so a month
#: mapping must use this synthetic calendar and never ``pd.date_range("2040-")``.
NON_LEAP_MONTH_HOURS: tuple[int, ...] = (
    744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744,
)

#: UTC -> local offset of the CA2040 hour grid, in hours.  Fixed PDT, the same
#: offset ``ch3.ra.persist.LOCAL_DAY_START_UTC_HOUR`` uses for ``lole_days`` and
#: the shipped window ``[7, 8743)`` uses for its local-midnight block starts.
DEFAULT_LOCAL_OFFSET_HOURS = 7


def local_hour_of_day(hour, local_offset_hours: int = DEFAULT_LOCAL_OFFSET_HOURS) -> np.ndarray:
    """0-based local hour of day of an absolute in-year UTC hour: ``(h - off) % 24``.

    ``0`` is 00:00-01:00 local.  Hour-ENDING ``HE k`` of the CPUC tables is local
    hour index ``k - 1`` (``HE17`` = 16:00-17:00 local = index 16).
    """
    return np.mod(np.asarray(hour, dtype=np.int64) - int(local_offset_hours), 24)


def local_month(
    hour,
    local_offset_hours: int = DEFAULT_LOCAL_OFFSET_HOURS,
    hours_per_year: int = 8760,
) -> np.ndarray:
    """1-12 local calendar month of an absolute in-year UTC hour.

    Wraps, so hours ``0 .. off-1`` (which are the previous local year) read as
    December.  Only the non-leap calendar of :data:`NON_LEAP_MONTH_HOURS` is
    supported; anything else is refused rather than silently misaligned.
    """
    total = int(sum(NON_LEAP_MONTH_HOURS))
    if int(hours_per_year) != total:
        raise ValueError(
            f"local_month needs a {total}-hour non-leap year (NON_LEAP_MONTH_HOURS), "
            f"got hours_per_year={hours_per_year}: the month boundaries would be wrong"
        )
    local = np.mod(np.asarray(hour, dtype=np.int64) - int(local_offset_hours), total)
    edges = np.cumsum(np.asarray(NON_LEAP_MONTH_HOURS, dtype=np.int64))
    return (np.searchsorted(edges, local, side="right") + 1).astype(int)


def hourly_import_limit(
    *,
    base_mw: float,
    profile_mw: Optional[Sequence[float]],
    months: Optional[Sequence[int]],
    years: Sequence[int],
    window: "HourWindow",
    hours_per_year: int,
    local_offset_hours: int = DEFAULT_LOCAL_OFFSET_HOURS,
) -> np.ndarray:
    """The MW cap on the import group, expanded onto the loaded hour grid.

    ``profile_mw is None`` returns the 1-D ``(1,)`` base cap -- the flat case,
    which the caller passes straight through so that ``DirectedLine`` keeps a
    ``(G, 1)`` limit and the system is byte-identical to the pre-profile one.

    Otherwise the 24 values are indexed by **local hour of day** and tiled over
    ``years`` in ``options.years`` order, which is the year-major layout
    ``load_system`` builds and ``ch3.ra.dispatch.block_time_periods`` indexes:
    the returned array is ``(1, len(window) * len(years))``.

    ``months`` restricts the profile to those local calendar months; every other
    hour gets ``base_mw``.  ``months is None`` (the shipped default) needs no
    calendar at all, which is what lets a 48 h test fixture use the profile.
    """
    base = float(base_mw)
    if not np.isfinite(base) or base <= 0:
        raise ValueError(f"import limit base_mw must be finite and > 0, got {base_mw!r}")
    if profile_mw is None:
        return np.array([base], dtype=np.float64)

    profile = np.asarray(profile_mw, dtype=np.float64).reshape(-1)
    if profile.size != 24:
        raise ValueError(
            "the import-limit profile is indexed by local hour of day and must have "
            f"exactly 24 entries, got {profile.size}"
        )
    if not np.all(np.isfinite(profile)) or np.any(profile <= 0):
        raise ValueError(
            f"every import-limit profile entry must be finite and > 0, got {profile.tolist()}"
        )
    if np.any(profile > base):
        raise ValueError(
            f"the import-limit profile exceeds its base cap of {base} MW in hour(s) "
            f"{np.flatnonzero(profile > base).tolist()}: base_mw is the loosest hour"
        )

    year_list = tuple(int(y) for y in years)
    if not year_list:
        raise ValueError("hourly_import_limit needs at least one weather year")

    hours = np.arange(int(window.start), int(window.stop), dtype=np.int64)
    if hours.size == 0:
        raise ValueError(f"hourly_import_limit got an empty window {window}")
    limit = profile[local_hour_of_day(hours, local_offset_hours)]

    if months is not None:
        wanted = np.asarray(sorted({int(m) for m in months}), dtype=np.int64)
        if wanted.size == 0:
            raise ValueError("hourly_import_limit got an empty month list; use None for all")
        if np.any(wanted < 1) or np.any(wanted > 12):
            raise ValueError(f"import-limit months must lie in 1..12, got {wanted.tolist()}")
        inside = np.isin(local_month(hours, local_offset_hours, hours_per_year), wanted)
        limit = np.where(inside, limit, base)

    return np.tile(limit, len(year_list))[None, :].astype(np.float64)


def import_limit_array(options: "LoadOptions", hours_per_year: int) -> Optional[np.ndarray]:
    """:func:`hourly_import_limit` for a :class:`LoadOptions`, or ``None`` uncapped."""
    if options.import_limit_mw is None:
        if options.import_limit_profile_mw is not None:
            raise ValueError(
                "LoadOptions.import_limit_profile_mw is set but import_limit_mw is None: "
                "a profile is a shape on a base cap, and without the cap no group is "
                "built at all, so the profile would be silently ignored"
            )
        return None
    return hourly_import_limit(
        base_mw=float(options.import_limit_mw),
        profile_mw=options.import_limit_profile_mw,
        months=options.import_limit_months,
        years=options.years,
        window=options.window,
        hours_per_year=int(hours_per_year),
        local_offset_hours=int(options.import_limit_local_offset_hours),
    )


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
    #: Path to the caller's outage parameter table.  zap ships none: the numbers
    #: are CH3 policy and live at ``ch3/ra/configs/outage_params.yaml``.
    #: **Required whenever** ``outage_draw is not None`` **or** ``ucap_derate``;
    #: ignored otherwise.
    outage_params_path: Optional[str] = None
    #: Unit-key scheme of the forced-outage sampler (``zap.reliability.keys.SCHEMES``).
    #: Carried on the run card and refused on mismatch: draws are only
    #: *statistically* equivalent across schemes, never bit-reproducible.
    outage_scheme: str = "slot-v1"
    #: Base seed of the outage sampler. The scheme version, not the seed, marks a
    #: change of draw definition (spec section 5.3).
    outage_seed: int = 20260908
    #: Investment year the system represents.  ``None`` -> :func:`detect_model_year`.
    model_year: Optional[int] = None
    #: Zero the as-built capacity of rows whose ``build_year + lifetime <= model_year``.
    apply_lifetimes: bool = True
    #: Capacities of a *design* to build the system at, keyed by device-class name
    #: (``"Generator"``, ``"StorageUnit"``, ``"DirectedLine"``) with one value per
    #: source row, in static-table order.  Applied *after* the lifetime rule and
    #: *before* the outage / UCAP lookup, so the availability multipliers are
    #: weighted over the units backing the **designed** capacity: capacity built
    #: on a retired or greenfield row draws from the pool instead of being
    #: outage-free (WP-E1).  See :func:`apply_design_capacity`.
    design_capacity: Optional[Mapping[str, Sequence[float]]] = None
    #: Provenance only: the ``design_id`` the capacities came from.
    design_id: Optional[str] = None
    link_losses: bool = True
    #: MW cap on the **sum** of hourly delivered flow over every
    #: ``carrier == "imports"`` link -- the simultaneous-import interface limit
    #: (import-limit spec, 2026-09-14).  ``None`` = no cap, and the system built
    #: is then bit-identical to the one built before the axis existed.  It counts
    #: all flow on those links, i.e. the real plant PyPSA-USA places behind the
    #: import buses *plus* ``unspecified_imports``.
    import_limit_mw: Optional[float] = None
    #: The **hour-of-day shape** of that cap: 24 MW values indexed by local hour
    #: of day (``0`` = 00:00-01:00 local), the CPUC/SERVM simultaneous-import
    #: profile (import-limit profile spec P2/P5).  ``None`` = a flat cap, and the
    #: device then keeps a ``(G, 1)`` limit exactly as before the axis existed.
    #: zap receives **numbers, never a policy name**: the named table is CH3
    #: policy, like ``outage_params.yaml``.  A frozen dataclass, so this must be
    #: a tuple.
    import_limit_profile_mw: Optional[tuple[float, ...]] = None
    #: Local calendar months the profile applies in; every other hour gets
    #: :attr:`import_limit_mw`.  ``None`` = every month (the shipped default),
    #: which needs no calendar at all.
    import_limit_months: Optional[tuple[int, ...]] = None
    #: Carriers kept on import buses (bus name ends in ``_imports``).  ``None``
    #: keeps every row -- the historical system, bit-identical devices.  A tuple
    #: holds every *other* generator / storage row on an import bus at zero
    #: capacity (rows kept, order unchanged, like a lifetime retirement), and a
    #: design cannot put capacity back on them.  zap receives carrier names,
    #: never a policy name (the ``unspecified_only`` label is CH3 policy).
    #: See :func:`apply_import_bus_rule`.
    import_bus_keep_carriers: Optional[tuple[str, ...]] = None
    #: UTC -> local offset used by both of the above.  Fixed PDT; ch3 passes
    #: ``ch3.ra.persist.LOCAL_DAY_START_UTC_HOUR`` so that the profile's local
    #: midnight and ``lole_days``' local midnight can never drift apart.
    import_limit_local_offset_hours: int = DEFAULT_LOCAL_OFFSET_HOURS
    #: Retired by the minimal unit-commitment device (2026-09-14).  Kept for key
    #: stability and now inert; ``False`` is a ``ValueError`` pointing at
    #: :attr:`commitment`.
    ignore_min_power: bool = True
    #: ``"minimal"`` builds the LP-relaxed unit-commitment device: a start-up
    #: cost and a minimum stable level on the ``committable`` rows of
    #: ``static/generators.csv``.  ``"off"`` is the historical device.
    commitment: Literal["off", "minimal"] = "off"
    #: Boundary condition on the commitment of the *first* hour of a block.
    #: ``"cyclic_free"`` wraps (``c_{-1} = c_{T-1}``); ``"pypsa"`` starts the
    #: block uncommitted (``c_{-1} = 0``), which is what PyPSA's linearised UC
    #: does when ``up_time_before == 0`` and is only used for the acceptance test.
    commitment_mode: Literal["cyclic_free", "pypsa"] = "cyclic_free"
    export_mode: Literal["sink", "drop"] = "sink"
    carbon_tax: float = 0.0
    power_unit: float = 1.0
    cost_unit: float = 1.0
    storage_init_soc: float = 0.5
    storage_final_soc: float = 0.5
    #: Storage boundary condition of a block. ``"fixed"`` pins start and end at
    #: ``storage_init_soc`` / ``storage_final_soc``; ``"cyclic_free"`` only ties
    #: them together (``energy[0] == energy[T]``) and ignores both levels.
    storage_soc_mode: Literal["fixed", "cyclic_free"] = (
        "cyclic_free"  # default by decision 2026-09-09
    )
    #: A constant MW of **firm load** added to demand in every hour of the window,
    #: split across load buses pro-rata to each bus's window-peak demand
    #: (accreditation spec D15).  This is the denominator direction of a marginal
    #: ELCC: an ELCC point is "the perturbed design, evaluated with `firm_load_mw`
    #: of extra firm demand".  It is applied *after* demand scaling, so it is an
    #: absolute MW increment and not a fraction of anything.
    firm_load_mw: float = 0.0
    #: Perfect capacity (MW) at a hub bus appended after every dataset bus, joined to
    #: every load bus by a lossless, zero-cost, one-way link rated
    #: PERFECT_LINK_HEADROOM * P (perfect-capacity hub spec, 2026-09-23).  0.0 = no
    #: hub node and no device: the system is bit-identical to one built before the
    #: field existed.  Never outaged, never a design row, never a planning parameter.
    perfect_capacity_mw: float = 0.0
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


# ===========================================================================
# Design capacities (WP-E1)
# ===========================================================================

#: Device class -> static table whose ``p_nom`` a design overwrites.  Storage
#: energy follows from ``p_nom * max_hours`` (``StorageUnit.duration``), so the
#: designed ``e_nom`` needs no separate entry.  ``ExportSink`` is derived from
#: the export links' capacity and is not designable here.
DESIGN_CAPACITY_TABLES = {
    "Generator": "generators",
    "StorageUnit": "storage_units",
    "DirectedLine": "links",
}


def design_capacity_digest(
    design_capacity: Optional[Mapping[str, Sequence[float]]],
) -> Optional[str]:
    """A sha256 over the design's capacity vectors, for cache keys and provenance."""
    if not design_capacity:
        return None
    h = hashlib.sha256()
    for cls_name in sorted(design_capacity):
        values = np.asarray(design_capacity[cls_name], dtype=np.float64).reshape(-1)
        h.update(cls_name.encode())
        h.update(np.ascontiguousarray(values).tobytes())
    return h.hexdigest()


def apply_design_capacity(
    static: dict[str, pd.DataFrame],
    design_capacity: Optional[Mapping[str, Sequence[float]]],
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Overwrite the static tables' ``p_nom`` with a design's capacities.

    Called *after* :func:`apply_retirements` and *before* the outage / UCAP
    lookup in :func:`load_system`, so the system that is built **is** the
    designed system: a retired or greenfield row carrying designed capacity is a
    new build on that row and draws its availability from that row's slice of
    the outage pool, and an expanded row is derated over the units backing its
    *designed* capacity rather than its as-built capacity (WP-E1).

    Rows are kept and their order is unchanged; only ``p_nom`` moves.  Storage
    energy follows through ``max_hours``.  Returns ``(static, summary)`` where
    ``static`` holds copies of the touched tables.
    """
    summary: dict[str, Any] = {
        "applied": bool(design_capacity),
        "digest": design_capacity_digest(design_capacity),
        "classes": {},
    }
    if not design_capacity:
        return static, summary

    unknown = sorted(set(design_capacity) - set(DESIGN_CAPACITY_TABLES))
    if unknown:
        raise ValueError(
            f"design_capacity has unknown device classes {unknown}; "
            f"expected any of {sorted(DESIGN_CAPACITY_TABLES)}"
        )

    out = dict(static)
    for cls_name, values in design_capacity.items():
        key = DESIGN_CAPACITY_TABLES[cls_name]
        df = out[key]
        designed = np.asarray(values, dtype=np.float64).reshape(-1)
        if designed.size != len(df):
            raise ValueError(
                f"design_capacity[{cls_name!r}] has {designed.size} values but "
                f"static/{key}.csv has {len(df)} rows"
            )
        if not np.all(np.isfinite(designed)):
            raise ValueError(f"design_capacity[{cls_name!r}] has non-finite values")
        if np.any(designed < 0.0):
            bad = [str(n) for n in df.index[designed < 0.0]]
            raise ValueError(f"design_capacity[{cls_name!r}] has negative capacity on rows {bad}")

        previous = df["p_nom"].to_numpy(dtype=np.float64)
        df = df.copy()
        df["p_nom"] = designed
        out[key] = df

        changed = designed != previous
        summary["classes"][cls_name] = {
            "n_rows": int(designed.size),
            "n_rows_changed": int(changed.sum()),
            "capacity_mw": float(designed.sum()),
            "as_built_mw": float(previous.sum()),
            "capacity_added_mw": float((designed - previous).sum()),
            "n_rows_built_from_zero": int(np.sum((previous <= 0.0) & (designed > 0.0))),
        }
    return out, summary


def _thermal_carriers() -> frozenset[str]:
    """Carriers covered by the outage pool (WP2).

    Reads no parameter table: zap has none, and the caller's table is a
    per-load-options argument, while ``thermal_mask`` is a property of the
    carrier names alone.  :data:`DEFAULT_THERMAL_CARRIERS` is kept in sync with
    ``ch3/ra/configs/outage_params.yaml`` by hand.
    """
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


def _ucap_factors(
    dataset_dir: Path,
    component: str,
    rows: pd.Index,
    expect_digest: Optional[str] = None,
) -> tuple[np.ndarray, str]:
    """UCAP derate per source row (1.0 when the row is absent), plus the csv sha256.

    ``expect_digest`` is the caller's ``OutageParams.draw_digest()[:12]``.  A
    ``ucap.csv`` sampled under a *different* parameter table is a silently wrong
    derate -- the numbers look plausible and nothing else in the run says which
    table they came from -- so a mismatch (or a file predating the provenance
    columns) is a hard error asking for a regeneration (spec D7).  ``None``
    skips the check.
    """
    path = Path(dataset_dir) / UCAP_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"ucap_derate=True but {path} does not exist. Build it with "
            f"`ra build-store --dataset {dataset_dir} --ucap --years ... --draws ...`."
        )
    table = pd.read_csv(path)
    for required in ("component", "row"):
        if required not in table.columns:
            raise ValueError(f"{path} has no '{required}' column")

    if expect_digest is not None:
        if "params_digest" not in table.columns:
            raise ValueError(
                f"{path} predates the outage-parameter provenance columns "
                f"(`params_digest`, `params_version`), so it cannot be shown to match "
                f"the loaded parameter table (digest {expect_digest}). Regenerate it: "
                f"`ra build-store --dataset {dataset_dir} --ucap --years ... --draws ...`."
            )
        found = sorted({str(d) for d in table["params_digest"].dropna().unique()})
        if found != [str(expect_digest)]:
            raise ValueError(
                f"{path} was sampled under outage parameters {found}, but the loaded "
                f"table has digest {expect_digest}: the derate would be wrong. "
                f"Regenerate it: `ra build-store --dataset {dataset_dir} --ucap "
                f"--years ... --draws ...`."
            )

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


def _outage_row_specs(static: dict[str, pd.DataFrame], params_path: str):
    """``(row specs, params)`` of the dataset's pooled rows.

    The specs are structural -- component, carrier, bus and the per-group ordinal
    -- so they are a property of the static tables alone and never of a design,
    a capacity or a draw.  They are derived from the *post-design* tables only
    because those are the ones in hand; the design moves ``p_nom``, which the
    specs do not read.
    """
    from zap.reliability.keys import row_specs
    from zap.reliability.outages import load_outage_params

    params = load_outage_params(params_path)
    return row_specs(static, params), params


def _outage_availability(
    static: dict[str, pd.DataFrame],
    options: LoadOptions,
    component: str,
    rows: pd.Index,
    capacities: pd.Series,
    years: Sequence[int],
    window: HourWindow,
) -> tuple[np.ndarray, dict]:
    """``(n_rows, n_hours)`` availability, sampled on demand (outage-pool spec D3).

    The capacities handed in are the *designed* ones, so the availability of a
    row is weighted over the slots backing the capacity it actually carries: a
    greenfield or rebuilt row draws outages instead of being outage-free
    (WP-E1), and this is now structural rather than a weighting trick, because
    slot ``k``'s realisation does not depend on how many slots the row uses.
    """
    from zap.reliability.outages import row_availability, slot_count, unit_cache_info

    if options.outage_params_path is None:
        raise ValueError(
            "outage_draw is set but LoadOptions.outage_params_path is None: zap ships "
            "no outage parameter table. Pass the caller's table -- CH3 runs use "
            "`ch3/ra/configs/outage_params.yaml` (`ch3.ra.paths.outage_params_path()`), "
            "zap's own tests `zap/tests/fixtures/outage_params_test.yaml`."
        )
    specs, params = _outage_row_specs(static, options.outage_params_path)
    by_component = [s for s in specs if s.component == component]
    known = {s.name: s for s in by_component}
    wanted = [known[r] for r in rows if r in known]

    caps = {name: float(capacities[name]) for name in (s.name for s in wanted)}
    blocks = []
    for year in years:
        avail = np.ones((len(window), len(rows)), dtype=np.float64)
        if wanted:
            sub = row_availability(
                wanted,
                caps,
                params,
                year=int(year),
                draw=int(options.outage_draw),
                base_seed=int(options.outage_seed),
                scheme=str(options.outage_scheme),
                window=window,
            )
            positions = [rows.get_loc(s.name) for s in wanted]
            avail[:, positions] = sub
        blocks.append(avail)

    stacked = np.concatenate(blocks, axis=0).T  # (n_rows, n_hours)
    info = {
        "scheme": str(options.outage_scheme),
        "base_seed": int(options.outage_seed),
        "params_sha256": params.sha256,
        "params_digest": params.draw_digest()[:12],
        "params_version": int(params.version),
        "params_reviewed": bool(params.reviewed),
        "n_units": int(
            sum(slot_count(caps[s.name], params.carriers[s.carrier].unit_size_mw) for s in wanted)
        ),
        "cache": unit_cache_info(),
        # Which component this snapshot counted, so a merged record can say so.
        "components": [str(component)],
    }
    return stacked, info


def merge_outage_info(*infos: Optional[dict]) -> Optional[dict]:
    """Combine the per-component records of :func:`_outage_availability`.

    ``load_system`` samples generators and storage in two calls, and a system's
    slot count is the sum of the two -- reporting only the first under-counted
    every run card and every ``meta["outages"]`` reader (verifier, 2026-09-12:
    640 of 1,138 slots on ``ca2040_z4`` draw 3).

    The RNG identity (scheme, seed, parameters) must agree across the calls --
    they come from one ``LoadOptions`` and one parameter file -- and a
    disagreement is a bug, so it raises rather than being silently reconciled.
    The cache block is taken from the **last** record: ``unit_cache_info``'s
    hit/miss counters are cumulative over the process, so the later snapshot
    already contains the earlier call's activity, and ``units`` is the live
    size of the one shared cache, not a per-call quantity.
    """
    present = [dict(i) for i in infos if i]
    if not present:
        return None
    identity_keys = (
        "scheme",
        "base_seed",
        "params_sha256",
        "params_digest",
        "params_version",
    )
    first = present[0]
    for other in present[1:]:
        for key in identity_keys:
            if other.get(key) != first.get(key):
                raise ValueError(
                    f"outage info disagrees on {key!r} between components "
                    f"({first.get('components')} vs {other.get('components')}): "
                    f"{first.get(key)!r} != {other.get(key)!r}"
                )
    merged = dict(first)
    merged["n_units"] = int(sum(int(i.get("n_units", 0)) for i in present))
    merged["cache"] = present[-1].get("cache")
    merged["components"] = [c for i in present for c in (i.get("components") or [])]
    merged["n_units_by_component"] = {
        component: int(i.get("n_units", 0))
        for i in present
        for component in (i.get("components") or ["?"])
    }
    return merged


# ===========================================================================
# Device builders
# ===========================================================================


def commitment_fields(gens: pd.DataFrame, file_p_nom: np.ndarray) -> dict[str, np.ndarray]:
    """``committable`` / ``min_power_fraction`` / ``start_up_cost_per_mw`` (spec 3).

    ``k_g = start_up_cost_g / p_nom_file_g`` in **$ per MW started**, from the
    *file's* ``p_nom`` -- before the lifetime rule and before
    :func:`apply_design_capacity` -- so it is a constant of the row rather than a
    function of the fleet year or of the design being scored.  PyPSA charges a
    flat ``$`` per start; the two coincide whenever the row is committed at its
    file capacity, which is every operational run and (on ca2040_z4) every
    planning cell, since all eight committable+extendable rows have
    ``p_nom_max == p_nom``.

    ``p_min_pu`` on a **non-committable** row is deliberately ignored: zap's
    ``Generator.min_power`` is identically zero and PyPSA's
    ``p >= p_min_pu * p_nom`` agrees with it at ``p_nom == 0``, which is every
    such row on ca2040_z4.  It would diverge on an expansion run that built one,
    and that is a documented limitation.
    """
    committable = np.asarray(gens["committable"], dtype=bool)
    p_min_pu = gens["p_min_pu"].to_numpy(dtype=np.float64)
    start_up_cost = gens["start_up_cost"].to_numpy(dtype=np.float64)
    file_p_nom = np.asarray(file_p_nom, dtype=np.float64)

    # `p_nom_file == 0` => `k_g = 0`; the guard below catches the one case where
    # that would be a silent modelling error rather than a harmless zero.
    denominator = np.where(file_p_nom > 0.0, file_p_nom, 1.0)
    k = np.where(file_p_nom > 0.0, start_up_cost / denominator, 0.0)
    # A buildable committable row with `p_nom_file == 0` has no defined
    # $/MW-started.  Defensive: no ca2040_z4 committable row is like this.
    extendable = np.asarray(gens.get("p_nom_extendable", False), dtype=bool)
    bad = committable & extendable & (file_p_nom <= 0.0) & (start_up_cost > 0.0)
    if bad.any():
        raise ValueError(
            "committable, extendable generator rows with p_nom == 0 in the file and a "
            f"positive start_up_cost have no defined $/MW-started cost: {list(gens.index[bad])}"
        )

    return {
        "committable": committable,
        "min_power_fraction": np.where(committable, p_min_pu, 0.0),
        "start_up_cost_per_mw": np.where(committable, k, 0.0),
    }


def _build_generators(
    static: dict[str, pd.DataFrame],
    store: WeatherStore,
    bus_index: dict[str, int],
    options: LoadOptions,
    outage_factor: Optional[np.ndarray],
    ucap_factor: Optional[np.ndarray],
    file_p_nom: Optional[np.ndarray] = None,
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

    commitment: dict[str, Any] = {}
    if options.commitment == "minimal":
        if file_p_nom is None:
            raise ValueError(
                "commitment='minimal' needs the file's p_nom (before retirements and "
                "before the design) to form the $/MW-started cost"
            )
        commitment = commitment_fields(gens, file_p_nom)
        commitment["commitment_mode"] = str(options.commitment_mode)

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
        **commitment,
    )
    dev.fuel_type = gens["carrier"].to_numpy()
    return dev, weather, emission_rates


def bus_peak_shares(profile: np.ndarray, terminal: np.ndarray, n_nodes: int) -> np.ndarray:
    """``f_n``: each bus's share of the system's window-peak demand.

    The spatial direction of a firm-load increment (accreditation spec D15) and,
    by D5, the denominator direction of every accreditation factor.  A bus's peak
    is the maximum over the window of the **summed** demand of the load rows
    sitting on it, so two rows at one bus count once; the shares are normalised
    over buses, not rows.  Returns zeros when the system carries no load.
    """
    profile = np.atleast_2d(np.asarray(profile, dtype=np.float64))
    terminal = np.asarray(terminal).ravel().astype(int)
    by_bus = np.zeros((n_nodes, profile.shape[1]))
    for row, node in enumerate(terminal):
        by_bus[node, :] += profile[row, :]
    peaks = by_bus.max(axis=1) if profile.shape[1] else np.zeros(n_nodes)
    peaks = np.maximum(peaks, 0.0)
    total = float(peaks.sum())
    if total <= 0.0:
        return np.zeros(n_nodes)
    return peaks / total


def firm_load_allocation(
    profile: np.ndarray, terminal: np.ndarray, n_nodes: int, firm_load_mw: float
) -> np.ndarray:
    """Per-**row** MW of a ``firm_load_mw`` increment (accreditation spec D15).

    The increment is split across *buses* pro-rata to :func:`bus_peak_shares`, and
    a bus's share is then split across the load rows sitting on it pro-rata to
    their own window peaks.  With one load row per bus -- which is every CA2040
    dataset -- the second step is the identity and row ``r`` at bus ``n`` simply
    receives ``f_n * firm_load_mw``.
    """
    firm_load_mw = float(firm_load_mw)
    profile = np.atleast_2d(np.asarray(profile, dtype=np.float64))
    terminal = np.asarray(terminal).ravel().astype(int)
    if firm_load_mw == 0.0:
        return np.zeros(profile.shape[0])
    shares = bus_peak_shares(profile, terminal, n_nodes)
    if not shares.any():
        raise ValueError(
            f"LoadOptions.firm_load_mw is {firm_load_mw} but the system carries no load, "
            "so there is no window peak to split the increment over"
        )
    row_peaks = profile.max(axis=1) if profile.shape[1] else np.zeros(profile.shape[0])
    row_peaks = np.maximum(row_peaks, 0.0)
    bus_peaks = np.zeros(n_nodes)
    np.add.at(bus_peaks, terminal, row_peaks)
    out = np.zeros(profile.shape[0])
    for row, node in enumerate(terminal):
        if bus_peaks[node] <= 0.0:
            continue
        out[row] = firm_load_mw * shares[node] * row_peaks[row] / bus_peaks[node]
    return out


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
    terminal = _terminals(loads["bus"], bus_index, "loads.csv")
    profile = profile * applied_scale
    # The firm-load increment is absolute MW and goes on *after* demand scaling:
    # scaling is a property of the scenario, the increment is the ELCC probe.
    increment = firm_load_allocation(profile, terminal, len(bus_index), options.firm_load_mw)
    if increment.any():
        profile = profile + increment[:, None]
    return Load(
        num_nodes=len(bus_index),
        name=loads.index,
        terminal=terminal,
        load=profile,
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

    # The aggregate interface limit (import-limit spec 3.2), resolved before any
    # timeseries is read so that an unenforceable cap fails on the config, not
    # after a store pull.
    group_kwargs: dict[str, Any] = {}
    if options.import_limit_mw is not None:
        cap = float(options.import_limit_mw)
        mask = links["carrier"].to_numpy().astype(str) == IMPORT_LINK_CARRIER
        if not mask.any():
            raise ValueError(
                f"import_limit_mw={cap} was asked for but static/links.csv has no row with "
                f"carrier == {IMPORT_LINK_CARRIER!r}: there is nothing to cap "
                "(import-limit spec D7)"
            )
        interface_mw = float(links["p_nom"].to_numpy(dtype=np.float64)[mask].sum())
        if cap >= interface_mw:
            logger.warning(
                "import_limit_mw=%.6g is at or above the %d import link(s)' total p_nom "
                "(%.6g MW), so the cap cannot bind: legal as a sensitivity point, but this "
                "system is the uncapped one.",
                cap,
                int(mask.sum()),
                interface_mw,
            )
        # `(1,)` when there is no profile -- the flat case, bit-identical to the
        # pre-profile device -- and `(1, len(window) * len(years))` when there is
        # (import-limit profile spec 3.3).
        group_kwargs = {
            "group": np.where(mask, 0, -1).astype(int),
            "group_limit": import_limit_array(options, int(store.hours_per_year)),
            "group_name": np.array([IMPORT_LINK_CARRIER], dtype=object),
        }

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
        **group_kwargs,
    )


def import_link_capacity_mw(static: dict[str, pd.DataFrame]) -> float:
    """Total ``p_nom`` of the ``carrier == "imports"`` links, in MW."""
    links = static["links"]
    mask = links["carrier"].to_numpy().astype(str) == IMPORT_LINK_CARRIER
    return float(links["p_nom"].to_numpy(dtype=np.float64)[mask].sum())


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

#: ``static/links.csv`` carrier of the links that cross the state interface.
#: The rows :attr:`LoadOptions.import_limit_mw` caps as one group.
IMPORT_LINK_CARRIER = "imports"


def import_bus_mask(bus: Sequence[str]) -> np.ndarray:
    """Rows that sit on an import bus (bus name ending in ``_imports``)."""
    return np.array([str(b).endswith(IMPORT_BUS_SUFFIX) for b in bus], dtype=bool)


# ===========================================================================
# Import-bus resources (issue #35)
# ===========================================================================

#: Tables the import-bus rule acts on (links, loads and stores are never touched).
IMPORT_BUS_RULE_TABLES = {"Generator": "generators", "StorageUnit": "storage_units"}


def _normalized_keep_carriers(
    keep_carriers: Optional[Sequence[str]],
) -> Optional[tuple[str, ...]]:
    """``None`` stays ``None``; anything else becomes a non-empty tuple of str."""
    if keep_carriers is None:
        return None
    if isinstance(keep_carriers, str):
        raise TypeError(
            f"import_bus_keep_carriers must be a sequence of carrier names, not the string "
            f"{keep_carriers!r}"
        )
    keep = tuple(str(c) for c in keep_carriers)
    if not keep:
        raise ValueError(
            "import_bus_keep_carriers is an empty tuple, which would remove every row on "
            "every import bus (the generic imports included). Pass None to keep every row."
        )
    return keep


def import_bus_removed_mask(
    df: pd.DataFrame, keep_carriers: Optional[Sequence[str]]
) -> np.ndarray:
    """Rows on an import bus whose carrier is not in ``keep_carriers``; all-False for None.

    ``df`` is a static generator or storage table (it needs ``bus`` and
    ``carrier`` columns).  An empty ``keep_carriers`` is a ``ValueError``, as in
    :func:`apply_import_bus_rule`.
    """
    keep = _normalized_keep_carriers(keep_carriers)
    n = len(df)
    if keep is None or n == 0:
        return np.zeros(n, dtype=bool)
    on_import = import_bus_mask(df["bus"].to_numpy())
    if "carrier" in df.columns:
        carrier = df["carrier"].astype(str).to_numpy()
    else:  # pragma: no cover - every CH3 export carries a carrier column
        carrier = np.full(n, "", dtype=object)
    return on_import & ~np.isin(carrier, list(keep))


def apply_import_bus_rule(
    static: dict[str, pd.DataFrame], keep_carriers: Optional[Sequence[str]]
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Zero ``p_nom`` of every row an import bus is not allowed to keep.

    The rule of issue #35 (Kamran, 2026-09-23): an import bus carries only the
    carriers in ``keep_carriers`` -- for CH3, the generic ``unspecified_imports``
    -- and every other generator / storage row there is held at zero capacity.
    Rows are *kept* (order, names and count unchanged), exactly like a lifetime
    retirement (:func:`apply_retirements`), so outage-pool slot keys,
    :class:`SystemIndex`, ``design.json`` and the accreditation tables stay
    aligned.  ``keep_carriers is None`` touches nothing.

    Returns ``(static, summary)`` where ``static`` holds copies of the touched
    tables and ``summary`` is ``{"keep_carriers", "removed_rows",
    "removed_capacity_mw", "removed_capacity_mw_by_carrier", "removed_names"}``
    keyed by component, mirroring :func:`apply_retirements`.
    ``removed_capacity_mw`` is the capacity *this call* zeroed: post-lifetime on
    the scenario pass, the design's MW on the design pass.  ``removed_rows`` and
    ``removed_names`` count every masked row, including those already at 0.

    Raises ``ValueError`` for an empty tuple, or for a kept carrier present on
    no import bus in any of :data:`IMPORT_BUS_RULE_TABLES` (a typo guard: a
    misspelt ``unspecified_imports`` would otherwise remove every import).
    """
    keep = _normalized_keep_carriers(keep_carriers)
    out = dict(static)
    summary: dict[str, Any] = {
        "keep_carriers": None if keep is None else list(keep),
        "removed_rows": {},
        "removed_capacity_mw": {},
        "removed_capacity_mw_by_carrier": {},
        "removed_names": {},
    }
    if keep is not None:
        present: set[str] = set()
        for key in IMPORT_BUS_RULE_TABLES.values():
            df = static.get(key)
            if df is None or len(df) == 0 or "carrier" not in df.columns:
                continue
            on_import = import_bus_mask(df["bus"].to_numpy())
            present.update(str(c) for c in df["carrier"].to_numpy()[on_import])
        missing = sorted(set(keep) - present)
        if missing:
            raise ValueError(
                f"import_bus_keep_carriers names carrier(s) {missing} that sit on no import "
                f"bus (carriers present on import buses: {sorted(present)}); refusing a rule "
                "that would remove every import-bus row"
            )

    for component, key in IMPORT_BUS_RULE_TABLES.items():
        df = static.get(key)
        if df is None:
            continue
        mask = import_bus_removed_mask(df, keep)
        capacity = df["p_nom"].to_numpy(dtype=np.float64)
        by_carrier: dict[str, float] = {}
        if mask.any():
            if "carrier" in df.columns:
                grouped = (
                    pd.Series(capacity[mask], index=df["carrier"].astype(str).to_numpy()[mask])
                    .groupby(level=0)
                    .sum()
                    .sort_values(ascending=False)
                )
                by_carrier = {str(k): float(v) for k, v in grouped.items()}
            df = df.copy()
            df.loc[mask, "p_nom"] = 0.0
            out[key] = df
        summary["removed_rows"][component] = int(mask.sum())
        summary["removed_capacity_mw"][component] = float(capacity[mask].sum())
        summary["removed_capacity_mw_by_carrier"][component] = by_carrier
        summary["removed_names"][component] = [str(n) for n in df.index[mask]]
    return out, summary


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
    """Build a ``zap`` system from a dataset directory and its weather store.

    Order of the capacity rules, which matters (WP-E1): the lifetime
    retirements are applied first, then the import-bus rule
    (``options.import_bus_keep_carriers``, issue #35), then
    ``options.design_capacity`` overwrites ``p_nom`` and the import-bus rule is
    enforced again (a design cannot put capacity back on a removed row), and
    only then are the outage draw / UCAP factors looked up.  So a design is
    never patched onto a system that was already derated at as-built capacity
    -- the loaded system *is* the designed system.
    """
    dataset_dir = Path(dataset_dir)
    options = options if options is not None else LoadOptions()

    if options.ucap_derate and options.outage_draw is not None:
        raise ValueError(
            "ucap_derate and outage_draw are mutually exclusive: a UCAP derate is a "
            "frozen approximation of the outage draws."
        )
    if not options.ignore_min_power:
        raise ValueError(
            "LoadOptions.ignore_min_power is inert since the minimal unit-commitment "
            "device landed (2026-09-14); ask for minimum stable levels with "
            "commitment='minimal' instead."
        )
    if options.commitment not in ("off", "minimal"):
        raise ValueError(f"Unknown commitment {options.commitment!r}; expected 'off' or 'minimal'")
    if options.commitment_mode not in ("cyclic_free", "pypsa"):
        raise ValueError(
            f"Unknown commitment_mode {options.commitment_mode!r}; "
            "expected 'cyclic_free' or 'pypsa'"
        )
    if options.export_mode not in ("sink", "drop"):
        raise ValueError(f"Unknown export_mode {options.export_mode!r}")
    if options.dtype != "float64":
        raise NotImplementedError("Only float64 systems are supported in phase 1")
    perfect_mw = float(options.perfect_capacity_mw)
    if not np.isfinite(perfect_mw) or perfect_mw < 0.0:
        raise ValueError(
            f"LoadOptions.perfect_capacity_mw must be finite and >= 0 MW, got "
            f"{options.perfect_capacity_mw!r} (a firm-load increment is `firm_load_mw`)"
        )

    store = WeatherStore.open(dataset_dir)
    static = _read_static(dataset_dir)
    # The `$/MW started` denominator is the *file's* p_nom, taken here: before the
    # lifetime rule below and before `apply_design_capacity` (spec 1.4).
    file_generator_p_nom = static["generators"]["p_nom"].to_numpy(dtype=np.float64).copy()
    if options.commitment == "minimal":
        dynamic_p_min_pu = dataset_dir / "timeseries" / "generators_t_p_min_pu.parquet"
        if dynamic_p_min_pu.exists():
            raise NotImplementedError(
                f"{dynamic_p_min_pu} exists: `p_min_pu` is static in these exports and the "
                "minimal unit-commitment device reads it from static/generators.csv only"
            )

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

    # ---- Import-bus resources (issue #35) --------------------------------
    # After the lifetime rule and *before* the `as_built` snapshot, so the
    # scenario the demand-scaling denominators, the outage pool and the UCAP
    # lookup see is the ruled one.  `None` touches nothing.
    keep_carriers = options.import_bus_keep_carriers
    static, import_bus = apply_import_bus_rule(static, keep_carriers)
    if keep_carriers is not None:
        logger.info(
            "Import-bus rule (keep %s): %d generator rows / %.1f MW and %d storage rows / "
            "%.1f MW held at 0 (MW zeroed after the lifetime rule).",
            list(keep_carriers),
            import_bus["removed_rows"].get("Generator", 0),
            import_bus["removed_capacity_mw"].get("Generator", 0.0),
            import_bus["removed_rows"].get("StorageUnit", 0),
            import_bus["removed_capacity_mw"].get("StorageUnit", 0.0),
        )

    # ---- Design capacities (WP-E1) ---------------------------------------
    # The design is imposed on the static tables *here*, after the lifetime rule
    # and before the outage / UCAP lookup below, so every availability
    # multiplier is computed over the units backing the designed capacity.
    # `as_built` keeps the pre-design tables: the demand-scaling denominators
    # are a property of the scenario, not of the design being scored, and must
    # not move from one design to the next.
    as_built = static
    static, design_summary = apply_design_capacity(static, options.design_capacity)
    # Re-enforce the import-bus rule: a design (an older `design.json`, or one
    # planned under the other setting) cannot put capacity back on a removed
    # row.  zap does not refuse -- the policy gate is the caller's preflight --
    # but it says so and records the MW.
    static, import_bus_design = apply_import_bus_rule(static, keep_carriers)
    design_mw_removed = import_bus_design["removed_capacity_mw"]
    if any(v > 0.0 for v in design_mw_removed.values()):
        logger.warning(
            "Design %s puts capacity on import-bus rows the rule removes (keep %s); "
            "held at 0: %s MW.",
            options.design_id or "<unnamed>",
            list(keep_carriers or ()),
            {k: round(v, 3) for k, v in design_mw_removed.items() if v > 0.0},
        )
    if design_summary["applied"]:
        logger.info(
            "Design %s applied to the static tables (%s).",
            options.design_id or "<unnamed>",
            ", ".join(
                f"{cls}: {info['capacity_mw']:.1f} MW "
                f"({info['capacity_added_mw']:+.1f} MW on {info['n_rows_changed']} rows, "
                f"{info['n_rows_built_from_zero']} from zero)"
                for cls, info in design_summary["classes"].items()
            ),
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

    # The tightest hour of the import cap in *this* window, in MW, resolved here
    # (before `scale_power` divides the device's copy by `power_unit`) so that
    # the card and `meta` are always MW-denominated.  `None` when uncapped.
    _import_limit = import_limit_array(options, int(store.hours_per_year))
    import_limit_mw_min = None if _import_limit is None else float(np.min(_import_limit))

    buses = static["buses"].index
    bus_index = {b: i for i, b in enumerate(buses)}

    # ---- Perfect-capacity hub node (perfect-capacity hub spec 4.2) --------
    # The hub is appended as the *last* node and entered in `bus_index` before
    # any `_build_*` call, so every device's `num_nodes = len(bus_index)` is
    # n + 1; no static table references it, so every existing terminal is
    # unchanged.  P = 0 adds nothing at all.
    hub_node: int | None = None
    if perfect_mw > 0.0:
        if PERFECT_HUB_BUS in bus_index:
            raise ValueError(
                f"static/buses.csv has a bus named {PERFECT_HUB_BUS!r}, the name reserved for "
                "the perfect-capacity hub; rename it or build with perfect_capacity_mw = 0"
            )
        hub_node = len(buses)
        bus_index[PERFECT_HUB_BUS] = hub_node

    # ---- Reliability heuristics ------------------------------------------
    gen_ucap = storage_ucap = None
    gen_outage = storage_outage = None
    ucap_sha: Optional[str] = None
    outage_attrs: Optional[dict] = None

    params_digest: Optional[str] = None

    if options.ucap_derate:
        if options.outage_params_path is None:
            raise ValueError(
                "ucap_derate=True but LoadOptions.outage_params_path is None: zap ships "
                "no outage parameter table, and without it a stale `ucap.csv` cannot be "
                "detected. Pass the caller's table -- CH3 runs use "
                "`ch3/ra/configs/outage_params.yaml` "
                "(`ch3.ra.paths.outage_params_path()`)."
            )
        from zap.reliability.outages import load_outage_params

        params_digest = load_outage_params(options.outage_params_path).draw_digest()[:12]
        gen_ucap, ucap_sha = _ucap_factors(
            dataset_dir, "Generator", static["generators"].index, params_digest
        )
        storage_ucap, _ = _ucap_factors(
            dataset_dir, "StorageUnit", static["storage_units"].index, params_digest
        )
    elif options.outage_draw is not None:
        # The capacities handed to `_outage_availability` are the *designed*
        # ones (`static` is post-design): every row is derated over the slots
        # backing the capacity it actually carries.  Nothing can overflow -- the
        # virtual pool is unbounded (outage-pool spec D3.1).
        gen_outage, gen_outage_attrs = _outage_availability(
            static,
            options,
            "Generator",
            static["generators"].index,
            static["generators"]["p_nom"],
            years,
            window,
        )
        storage_outage, storage_outage_attrs = _outage_availability(
            static,
            options,
            "StorageUnit",
            static["storage_units"].index,
            static["storage_units"]["p_nom"],
            years,
            window,
        )
        # Both components, or the card under-reports the case: keeping only the
        # generator call's info reported 640 of 1,138 slots on z4 draw 3
        # (verifier, 2026-09-12).
        outage_attrs = merge_outage_info(gen_outage_attrs, storage_outage_attrs)
        params_digest = (outage_attrs or {}).get("params_digest")

    # ---- Demand scaling (D9) ---------------------------------------------
    # As-built capacities on purpose: the peak-available denominator (and hence
    # `peak_fraction` demand scaling) must be identical for every design scored
    # against this scenario.
    peaks = _peak_metrics(as_built, store, years)
    peak_load = peaks["peak_load_mw"]
    peak_available = peaks["peak_available_mw"]
    implied_scale, applied_scale = _demand_scale(options, peak_load, peak_available)

    # ---- Devices, in the D2 order ----------------------------------------
    generator, _, emission_rates = _build_generators(
        static, store, bus_index, options, gen_outage, gen_ucap, file_generator_p_nom
    )
    load = _build_loads(static, store, bus_index, options, applied_scale)
    line = _build_links(static, store, bus_index, options)
    storage = _build_storage(static, bus_index, options, n_hours, storage_outage, storage_ucap)
    sink = _build_export_sinks(static, bus_index) if options.export_mode == "sink" else None
    if sink is not None and design_summary["classes"].get("DirectedLine"):
        # The export sink's capacity is the sum of the incoming export links'
        # `p_nom`, so a design that expands those links also enlarges the sink --
        # which the phase-1 lesson says must instead be capped by the built store
        # `e_nom`.  Evaluation uses `export_mode: drop`, so this is a warning, not
        # a refusal; a designed-link run in `sink` mode is not trustworthy.
        logger.warning(
            "export_mode='sink' with a DirectedLine design: the export sink capacity follows "
            "the designed link p_nom, not the built export-store e_nom. Use export_mode='drop' "
            "for evaluation, or cap the sink by the built e_nom before trusting export revenue."
        )

    # ---- Perfect-capacity hub devices (spec 4.2 steps 2-3) --------------
    # The load nodes come from the built load profile: after demand scaling and
    # the firm-load increment (a positive increment never changes the support of
    # f_n).  The two devices go *after* ExportSink, so no existing
    # `device_index` value moves.
    perfect_gen = perfect_link = None
    perfect_info: dict | None = None
    hub_labels: dict[str, tuple] = {}
    if hub_node is not None:
        node_names = np.empty(len(bus_index), dtype=object)
        for bus_name, i in bus_index.items():
            node_names[i] = bus_name
        load_nodes = perfect_load_nodes(load.load, load.terminal, len(bus_index))
        load_buses = [str(node_names[i]) for i in load_nodes]
        perfect_gen, perfect_link = perfect_capacity_devices(
            num_nodes=len(bus_index),
            hub_node=hub_node,
            load_nodes=load_nodes,
            capacity_mw=perfect_mw,
            load_bus_names=load_buses,
        )
        hub_labels = {
            "PerfectGenerator": (
                pd.Index(list(perfect_gen.name)),
                np.array([PERFECT_CARRIER], dtype=object),
                np.array([PERFECT_HUB_BUS], dtype=object),
            ),
            "PerfectLink": (
                pd.Index(list(perfect_link.name)),
                np.full(len(load_buses), PERFECT_CARRIER, dtype=object),
                np.array(load_buses, dtype=object),
            ),
        }
        perfect_info = {
            "hub_node": int(hub_node),
            "hub_bus": PERFECT_HUB_BUS,
            "load_buses": load_buses,
            "load_nodes": [int(i) for i in load_nodes],
            "link_headroom": float(PERFECT_LINK_HEADROOM),
            "link_mw": float(PERFECT_LINK_HEADROOM * perfect_mw),
        }

    ordered = [
        ("Generator", generator),
        ("Load", load),
        ("DirectedLine", line),
        ("StorageUnit", storage),
        ("ExportSink", sink),
        ("PerfectGenerator", perfect_gen),
        ("PerfectLink", perfect_link),
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

        if key in hub_labels:
            index.names[key], index.carrier[key], index.bus[key] = hub_labels[key]
            continue
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

    # `len(bus_index)`, not `len(buses)`: the hub, when present, is the last node.
    network = PowerNetwork(len(bus_index))

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
        # Constant MW of firm load added in every hour, split across load buses
        # pro-rata to window peak (accreditation spec D15). 0.0 on every ordinary
        # run; non-zero only on an ELCC probe.
        "firm_load_mw": float(options.firm_load_mw),
        # Perfect capacity at the hub (perfect-capacity hub spec 4.2): always
        # present; `perfect_capacity` is None at P = 0, else the hub's node, the
        # load buses its links reach, and the link rating in MW.
        "perfect_capacity_mw": perfect_mw,
        "perfect_capacity": perfect_info,
        "voll": float(options.voll),
        "ucap_derate": bool(options.ucap_derate),
        "outage_draw": options.outage_draw,
        "outage_scheme": str(options.outage_scheme),
        "outage_seed": int(options.outage_seed),
        "model_year": model_year,
        "model_year_source": model_year_source,
        "apply_lifetimes": bool(options.apply_lifetimes),
        "retired_rows": retirements["retired_rows"],
        "retired_capacity_mw": retirements["retired_capacity_mw"],
        "retired_capacity_mw_by_carrier": retirements["retired_capacity_mw_by_carrier"],
        "retired_names": retirements["retired_names"],
        # The import-bus rule (issue #35): always present, zero counts when
        # `import_bus_keep_carriers is None`.  `removed_capacity_mw` is what the
        # scenario pass zeroed (post-lifetime); `design_mw_removed` is what the
        # re-enforcement after the design zeroed.
        "import_bus_resources": {**import_bus, "design_mw_removed": design_mw_removed},
        "design_id": options.design_id,
        "design_capacity_applied": bool(design_summary["applied"]),
        "design_capacity_digest": design_summary["digest"],
        "design_capacity_summary": design_summary["classes"],
        # The peak-load / peak-available metrics above are computed at the
        # *as-built* (post-retirement, pre-design) capacities so demand scaling
        # is identical across the designs scored on one scenario.
        "peaks_at": "as_built",
        "link_losses": bool(options.link_losses),
        # The simultaneous-import interface limit and what it is a limit on.
        # `peak_available_*_incl_imports_mw` above ignores it and is therefore an
        # overstatement whenever a cap is set (import-limit spec D8).
        "import_limit_mw": (
            None if options.import_limit_mw is None else float(options.import_limit_mw)
        ),
        # The hour-of-day shape of that cap and the months it applies in
        # (import-limit profile spec 3.3).  `import_limit_mw_min` is the
        # tightest hour actually present in *this* window -- the one number a
        # card reader wants, and equal to `import_limit_mw` when there is no
        # profile.
        "import_limit_profile_mw": (
            None
            if options.import_limit_profile_mw is None
            else [float(v) for v in options.import_limit_profile_mw]
        ),
        "import_limit_months": (
            None
            if options.import_limit_months is None
            else [int(m) for m in options.import_limit_months]
        ),
        "import_limit_local_offset_hours": int(options.import_limit_local_offset_hours),
        "import_limit_mw_min": import_limit_mw_min,
        "import_link_capacity_mw": import_link_capacity_mw(static),
        "storage_soc_mode": str(options.storage_soc_mode),
        "commitment": str(options.commitment),
        "commitment_mode": str(options.commitment_mode),
        "storage_init_soc": float(options.storage_init_soc),
        "storage_final_soc": float(options.storage_final_soc),
        "power_unit": float(options.power_unit),
        "cost_unit": float(options.cost_unit),
        "weather_store_attrs": store.attrs,
        "ucap_csv_sha256": ucap_sha,
        # 12-hex `OutageParams.draw_digest()` of the parameter table the derate or
        # the draws were taken under; None when the system carries neither.
        "outage_params_digest": params_digest,
        # `{scheme, base_seed, params_sha256, n_units, cache}` -- the forty bytes
        # that replaced `outages.zarr` as the canonical artefact (spec D2/D3.1).
        # None when the system carries no draw.
        "outages": outage_attrs,
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
