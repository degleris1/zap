"""Thermal / storage forced-outage sampling and the UCAP table.

This module is deliberately independent of the rest of the CH3 harness: it reads
only ``<dataset_dir>/static/*.csv`` and ``outage_params.yaml``, and writes only
``ucap.csv``.

Model
-----
Every source row of ``generators.csv`` / ``storage_units.csv`` whose carrier has
outage parameters is backed by *virtual units* ("slots") of that carrier's
``unit_size_mw``. A row at designed capacity ``C`` uses slots ``0 .. n-1`` with
``n = 0`` if ``C == 0`` else ``max(1, ceil(C / size))``; the last slot carries
the remainder as a weight. Each slot is an independent two-state Markov chain in
discrete hourly time with stationary availability ``1 - FOR``.

There is **no store**. Draws are generated on demand and the canonical artefact
is ``(base_seed, scheme, params sha256)`` -- forty bytes on a run card instead of
the ~1.1 M files a 23 y x 500-draw ``outages.zarr`` would have cost.

RNG contract (spec D1.2)
------------------------
A draw is a pure function of ``(base_seed, scheme, year, draw, key)`` and of
nothing else::

    uid = uint64(blake2b(f"{scheme}|{carrier}|{bus}|{ordinal}|{k}", digest_size=8))
    rng = default_rng(SeedSequence(entropy=(base_seed, year, draw, uid_hi, uid_lo)))
    u   = rng.random(n_hours)

Two consequences, both load-bearing:

* **Slot k's realisation does not depend on n.** Growing a row's capacity leaves
  every slot it already used bit-identical, so perturbed designs are
  common-random-number paired on the slots they share.
* **The chain always starts at hour 0 of the weather year** and is then sliced to
  the window, so a 168 h block's realisation is identical whether it is loaded
  alone or inside a full year. ``n_hours < 8760`` gives exactly the first
  ``n_hours`` hours of the full-year realisation, because the uniforms are a
  prefix of the same stream.

Caveats
-------
* Storage outages derate **power only**; the energy cap is untouched (spec D8).
* Hydro carries outage draws even though its ``p_max_pu`` profile already encodes
  availability, so the two derates compound (issue #16). ``excluded_carriers`` in
  ``outage_params.yaml`` makes a no-hydro sensitivity a one-line change.
* The storage UCAP produced here is a forced-outage derate, **not** an
  accreditation (ELCC). Label it that way on run cards.
* ``outage_params.yaml`` is still ``reviewed: false`` (issue #7).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from zap.reliability.keys import (
    COMPONENT_TABLES,
    DEFAULT_SCHEME,
    RowSpec,
    get_scheme,
    row_specs,
)

GENERATOR_VERSION = 2
HOURS_PER_YEAR = 8760
DEFAULT_BASE_SEED = 20260908
DEFAULT_PARAMS_PATH = Path(__file__).parent / "outage_params.yaml"

#: Units whose uniforms are materialised in one go. 512 x 8760 float64 is ~36 MB;
#: the recursion is vectorised across the chunk, so this is a memory knob only.
_BATCH_UNITS = 512

#: Cap on the in-process slot cache, in slots. 20,000 x 8,760 uint8 ~ 175 MB.
CACHE_UNITS_ENV = "CH3_OUTAGE_CACHE_UNITS"
DEFAULT_CACHE_UNITS = 20_000


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
        sample sizes this sampler is run at.
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
    """The full resolved contents of ``outage_params.yaml`` (version 2).

    Version 2 dropped ``pool_multiplier`` / ``min_units_per_row`` /
    ``min_pool_capacity_mw``: with on-demand, slot-keyed generation there is no
    pool to size and nothing that can overflow.
    """

    version: int
    reviewed: bool
    excluded_carriers: frozenset[str]
    carriers: Mapping[str, CarrierOutageParams]
    sha256: str

    #: Keys that version 1 carried and version 2 must not.
    RETIRED_KEYS = ("pool_multiplier", "min_units_per_row", "min_pool_capacity_mw")

    def to_dict(self) -> dict:
        return {
            "version": int(self.version),
            "reviewed": bool(self.reviewed),
            "excluded_carriers": sorted(self.excluded_carriers),
            "carriers": {k: v.to_dict() for k, v in sorted(self.carriers.items())},
        }

    def content_hash(self) -> str:
        """Digest of the *values* that drive the chain, not of the file they came from.

        ``sha256`` is the sha of ``outage_params.yaml``'s bytes: the right label
        for provenance, but only as good as whoever populated it --
        :meth:`from_dict` defaults it to ``"unknown"`` and callers (tests, a
        store's recorded attributes) pass literals. The slot cache keys on this
        instead, because two parameter sets that disagree on a single forced
        outage rate must never share a cached realisation.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, raw: Mapping, sha256: str = "unknown") -> OutageParams:
        missing = {"version", "reviewed", "excluded_carriers", "carriers"} - set(raw)
        if missing:
            raise KeyError(f"outage params missing required keys: {sorted(missing)}")
        stale = [k for k in cls.RETIRED_KEYS if k in raw]
        if stale:
            raise KeyError(
                f"outage params still carry the version-1 pool-sizing keys {stale}; "
                "the slot-keyed scheme has no pool to size -- remove them and set "
                "`version: 2`"
            )

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


def params_fingerprint(params: OutageParams, scheme: str, base_seed: int) -> dict:
    """The forty bytes that replace ``outages.zarr`` as the canonical artefact."""
    return {
        "scheme": str(scheme),
        "base_seed": int(base_seed),
        "params_sha256": str(params.sha256),
        "params_version": int(params.version),
        "reviewed": bool(params.reviewed),
    }


# ---------------------------------------------------------------------------
# Static tables and row specs
# ---------------------------------------------------------------------------


def _read_static(dataset_dir: Path, filename: str) -> pd.DataFrame:
    path = Path(dataset_dir) / "static" / filename
    if not path.exists():
        raise FileNotFoundError(f"expected static component table at {path}")
    return pd.read_csv(path)


def read_static_tables(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    """``{"generators": df, "storage_units": df}`` with ``name`` as a column."""
    return {key: _read_static(dataset_dir, filename) for _c, key, filename in COMPONENT_TABLES}


def resolve_model_year(dataset_dir: Path, model_year: int | None) -> int | None:
    """Model year for the retirement rule, auto-detected when not given.

    Returns ``None`` -- meaning "no lifetime information, nothing retires" --
    when neither static table carries ``build_year``/``lifetime`` (the hermetic
    test fixtures) or when the dataset has no snapshot stamps to read a year
    from.
    """
    if model_year is not None:
        return int(model_year)

    from zap.importers.wy_store import detect_model_year, has_lifetime_columns

    dataset_dir = Path(dataset_dir)
    lifetimes = False
    for _component, _key, filename in COMPONENT_TABLES:
        path = dataset_dir / "static" / filename
        if path.exists() and has_lifetime_columns(pd.read_csv(path, nrows=0)):
            lifetimes = True
    if not lifetimes:
        return None
    try:
        year, _ = detect_model_year(dataset_dir)
    except ValueError:
        return None
    return int(year)


def dataset_row_specs(
    dataset_dir: Path, params: OutageParams
) -> tuple[list[RowSpec], dict[str, pd.DataFrame]]:
    """Pooled rows of a dataset, plus the static tables they came from."""
    static = read_static_tables(dataset_dir)
    return row_specs(static, params), static


def active_capacities(
    dataset_dir: Path, params: OutageParams, model_year: int | None = None
) -> dict[str, float]:
    """As-built ``p_nom`` per pooled row, zeroed where the row retires by ``model_year``."""
    from zap.importers.wy_store import retired_mask

    model_year = resolve_model_year(dataset_dir, model_year)
    rows, static = dataset_row_specs(dataset_dir, params)
    pooled = {r.name for r in rows}
    caps: dict[str, float] = {}
    for _component, key, _filename in COMPONENT_TABLES:
        df = static[key]
        if len(df) == 0:
            continue
        retired = retired_mask(df, model_year)
        names = df["name"].astype(str) if "name" in df.columns else df.index.astype(str)
        for name, p_nom, gone in zip(names, df["p_nom"], retired, strict=True):
            if str(name) in pooled:
                caps[str(name)] = 0.0 if bool(gone) else float(p_nom)
    return caps


# ---------------------------------------------------------------------------
# Slot arithmetic
# ---------------------------------------------------------------------------


def slot_count(capacity: float, size: float) -> int:
    """Slots backing ``capacity`` MW at ``size`` MW per unit (0 when ``C == 0``).

    The epsilon keeps an exact multiple from rounding up to one unit too many;
    the ``max(1, ...)`` guards a capacity so small that it would round down to 0.
    """
    if capacity <= 0.0:
        return 0
    return max(1, math.ceil(capacity / size - 1e-9))


def _row_weights(capacity: float, size: float) -> np.ndarray:
    """Weights of the slots backing ``capacity`` MW of a row with unit size ``size``.

    ``[1, ..., 1, (C - (n-1) s) / s]``. Unlike version 1 there is no bound on
    ``n``: the virtual pool is unbounded, so nothing can overflow.
    """
    n = slot_count(capacity, size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    w[-1] = (capacity - (n - 1) * size) / size
    return w


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _as_vector(value, n: int, what: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    arr = arr.reshape(-1)
    if arr.size != n:
        raise ValueError(f"{what} has {arr.size} entries but there are {n} units")
    return arr


def sample_units(
    unit_ids,
    p_fail,
    p_repair,
    fo_rate,
    *,
    year: int,
    draw: int,
    base_seed: int,
    n_hours: int,
) -> np.ndarray:
    """``(n_units, n_hours)`` uint8 uptime, 1 = available.

    One ``SeedSequence`` per unit, seeded on ``(base_seed, year, draw, uid)``, so
    a unit's realisation depends on its key and on nothing else -- not on which
    other units were asked for, nor on how many. The hour recursion is vectorised
    across the batch; only the uniform fill is per unit.

    ``n_hours`` shorter than a full year returns exactly the first ``n_hours``
    hours of the full-year realisation (the uniforms are a stream prefix).
    """
    if n_hours < 1:
        raise ValueError(f"n_hours must be >= 1, got {n_hours}")
    ids = np.asarray(unit_ids, dtype=np.uint64).reshape(-1)
    n_units = ids.size
    out = np.empty((n_units, int(n_hours)), dtype=np.uint8)
    if n_units == 0:
        return out

    pf = _as_vector(p_fail, n_units, "p_fail")
    pr = _as_vector(p_repair, n_units, "p_repair")
    f = _as_vector(fo_rate, n_units, "fo_rate")

    base_seed, year, draw = int(base_seed), int(year), int(draw)
    mask32 = np.uint64(0xFFFFFFFF)
    for start in range(0, n_units, _BATCH_UNITS):
        stop = min(start + _BATCH_UNITS, n_units)
        u = np.empty((stop - start, int(n_hours)), dtype=np.float64)
        for i, uid in enumerate(ids[start:stop]):
            uid = np.uint64(uid)
            entropy = (
                base_seed,
                year,
                draw,
                int(uid >> np.uint64(32)),
                int(uid & mask32),
            )
            rng = np.random.default_rng(np.random.SeedSequence(entropy=entropy))
            u[i, :] = rng.random(int(n_hours))

        chunk = out[start:stop]
        # Stationary start: available with probability 1 - FOR.
        state = u[:, 0] < (1.0 - f[start:stop])
        chunk[:, 0] = state
        pf_c, pr_c = pf[start:stop], pr[start:stop]
        for t in range(1, int(n_hours)):
            state = np.where(state, u[:, t] >= pf_c, u[:, t] < pr_c)
            chunk[:, t] = state
    return out


# ---------------------------------------------------------------------------
# In-process slot cache (spec D3.2)
# ---------------------------------------------------------------------------

#: ``uid -> (n_hours,) uint8``, valid for exactly one
#: ``(scheme, base_seed, year, draw, n_hours, params content hash)`` and dropped
#: wholesale when that tuple changes.  The evaluation enumerates design-inner, so
#: consecutive cases share it and a perturbed design misses only on the slots it
#: added.
_UNIT_CACHE: OrderedDict[int, np.ndarray] = OrderedDict()
_UNIT_CACHE_KEY: tuple | None = None
_UNIT_CACHE_STATS: dict[str, Any] = {"hits": 0, "misses": 0}


def cache_capacity() -> int:
    """Slot cap of the in-process cache (``CH3_OUTAGE_CACHE_UNITS``)."""
    raw = os.environ.get(CACHE_UNITS_ENV)
    if raw is None:
        return DEFAULT_CACHE_UNITS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CACHE_UNITS
    return max(0, value)


def unit_cache_clear() -> None:
    """Drop the cache and reset its counters (tests, long runs)."""
    global _UNIT_CACHE_KEY
    _UNIT_CACHE.clear()
    _UNIT_CACHE_KEY = None
    _UNIT_CACHE_STATS.update({"hits": 0, "misses": 0})


def unit_cache_info() -> dict:
    """``{"hits", "misses", "units", "key"}`` of the slot cache."""
    return {
        "hits": int(_UNIT_CACHE_STATS["hits"]),
        "misses": int(_UNIT_CACHE_STATS["misses"]),
        "units": len(_UNIT_CACHE),
        "key": _UNIT_CACHE_KEY,
        "capacity": cache_capacity(),
    }


def _cached_uptime(
    ids: np.ndarray,
    carriers: Sequence[str],
    params: OutageParams,
    *,
    scheme: str,
    base_seed: int,
    year: int,
    draw: int,
    n_hours: int,
) -> dict[int, np.ndarray]:
    """Uptime rows for ``ids``, generating (in one batched call) only the misses.

    The cache key carries ``params.content_hash()`` as well as the RNG tuple: the
    uniforms depend only on ``(scheme, seed, year, draw, uid)``, but the *chain*
    they are pushed through depends on the carrier's ``p_fail`` / ``p_repair`` /
    FOR, so two parameter sets at the same ``(scheme, seed, year, draw)`` would
    otherwise serve each other's availability (verifier, 2026-09-12: a x5 forced
    outage rate read back 0.9769 instead of 0.8805).
    """
    global _UNIT_CACHE_KEY
    key = (
        str(scheme),
        int(base_seed),
        int(year),
        int(draw),
        int(n_hours),
        params.content_hash(),
    )
    if _UNIT_CACHE_KEY != key:
        _UNIT_CACHE.clear()
        _UNIT_CACHE_KEY = key

    wanted: dict[int, np.ndarray] = {}
    miss_ids: list[int] = []
    miss_carriers: list[str] = []
    for uid, carrier in zip(ids, carriers, strict=True):
        uid = int(uid)
        if uid in wanted:
            continue
        cached = _UNIT_CACHE.get(uid)
        if cached is not None:
            _UNIT_CACHE.move_to_end(uid)
            _UNIT_CACHE_STATS["hits"] += 1
            wanted[uid] = cached
        else:
            miss_ids.append(uid)
            miss_carriers.append(str(carrier))

    if miss_ids:
        _UNIT_CACHE_STATS["misses"] += len(miss_ids)
        cps = [params.carriers[c] for c in miss_carriers]
        up = sample_units(
            np.array(miss_ids, dtype=np.uint64),
            [cp.p_fail for cp in cps],
            [cp.p_repair for cp in cps],
            [cp.forced_outage_rate for cp in cps],
            year=year,
            draw=draw,
            base_seed=base_seed,
            n_hours=n_hours,
        )
        capacity = cache_capacity()
        for i, uid in enumerate(miss_ids):
            row = up[i]
            row.flags.writeable = False
            wanted[uid] = row
            if capacity:
                _UNIT_CACHE[uid] = row
                _UNIT_CACHE.move_to_end(uid)
        while capacity and len(_UNIT_CACHE) > capacity:
            _UNIT_CACHE.popitem(last=False)
    return wanted


# ---------------------------------------------------------------------------
# Row availability
# ---------------------------------------------------------------------------


def _window_bounds(window, n_hours: int) -> tuple[int, int]:
    if window is None:
        return 0, int(n_hours)
    start = getattr(window, "start", None)
    stop = getattr(window, "stop", None)
    if start is None or stop is None:
        start, stop = window  # a (start, stop) pair
    start, stop = int(start), int(stop)
    if start < 0 or stop <= start:
        raise ValueError(f"invalid hour window [{start}, {stop})")
    if stop > n_hours:
        raise ValueError(f"window [{start}, {stop}) exceeds {n_hours} hours per year")
    return start, stop


def row_availability(
    rows: Sequence[RowSpec],
    capacities: Mapping[str, float],
    params: OutageParams,
    *,
    year: int,
    draw: int,
    base_seed: int = DEFAULT_BASE_SEED,
    scheme: str = DEFAULT_SCHEME,
    window=None,
    hours_per_year: int = HOURS_PER_YEAR,
) -> np.ndarray:
    """``(n_hours, n_rows)`` availability in ``[0, 1]``, one column per row of ``rows``.

    For row ``r`` at capacity ``C`` with unit size ``s``::

        n = 0 if C == 0 else max(1, ceil(C / s))     # n == 0 -> availability 1.0
        w = [1] * (n - 1) + [(C - (n - 1) s) / s]
        availability(t) = sum_k w_k up[uid(r, k), t] / sum_k w_k

    The chain is simulated from hour 0 of the weather year and sliced to
    ``window``, so a block's realisation does not depend on how it was loaded.
    Rows with no slots (zero capacity) are available in every hour.
    """
    scheme_obj = get_scheme(scheme)
    start, stop = _window_bounds(window, hours_per_year)
    n_hours = stop - start
    out = np.ones((n_hours, len(rows)), dtype=np.float64)

    per_row: list[tuple[int, np.ndarray, np.ndarray]] = []
    all_ids: list[int] = []
    all_carriers: list[str] = []
    for j, row in enumerate(rows):
        cp = params.carriers.get(row.carrier)
        if cp is None:  # not pooled; availability stays 1.0
            continue
        capacity = float(capacities.get(row.name, 0.0))
        w = _row_weights(capacity, cp.unit_size_mw)
        if w.size == 0:
            continue
        ids = scheme_obj.unit_ids(row, int(w.size))
        per_row.append((j, ids, w))
        all_ids.extend(int(u) for u in ids)
        all_carriers.extend([row.carrier] * int(ids.size))

    if not per_row:
        return out

    uptime = _cached_uptime(
        np.array(all_ids, dtype=np.uint64),
        all_carriers,
        params,
        scheme=scheme,
        base_seed=base_seed,
        year=year,
        draw=draw,
        n_hours=hours_per_year,
    )

    for j, ids, w in per_row:
        block = np.stack([uptime[int(u)][start:stop] for u in ids]).astype(np.float64)
        out[:, j] = (w @ block) / w.sum()
    return out


def slot_counts(
    rows: Sequence[RowSpec], capacities: Mapping[str, float], params: OutageParams
) -> dict[str, int]:
    """``{row name: slots}`` at the given capacities -- the cost guard's input."""
    out: dict[str, int] = {}
    for row in rows:
        cp = params.carriers.get(row.carrier)
        if cp is None:
            continue
        out[row.name] = slot_count(float(capacities.get(row.name, 0.0)), cp.unit_size_mw)
    return out


# ---------------------------------------------------------------------------
# UCAP
# ---------------------------------------------------------------------------


def default_ucap_csv(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / "ucap.csv"


UCAP_COLUMNS = [
    "component",
    "row",
    "carrier",
    "bus",
    "ordinal",
    "p_nom_mw",
    "unit_size_mw",
    "n_units_row",
    "forced_outage_rate",
    "ucap_analytic",
    "ucap_empirical",
    "n_samples",
    "ucap_carrier_bus_empirical",
    "outage_scheme",
    "base_seed",
    "n_years",
    "n_draws",
]


def write_ucap(
    dataset_dir: Path,
    *,
    years: Sequence[int],
    draws: int,
    base_seed: int = DEFAULT_BASE_SEED,
    scheme: str = DEFAULT_SCHEME,
    hours_per_year: int = HOURS_PER_YEAR,
    out_csv: Path | None = None,
    params: OutageParams | None = None,
    model_year: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sample ``years x draws`` on demand and write ``ucap.csv``.

    ``ucap_empirical`` is the mean over every sampled ``(year, draw, hour)`` of
    the row's availability *at its active as-built ``p_nom``*.
    ``ucap_analytic = 1 - FOR`` is scheme-independent; the empirical column is
    an estimate of it and is not expected to move between schemes beyond its own
    Monte-Carlo error (spec D4).
    """
    dataset_dir = Path(dataset_dir)
    params = params if params is not None else load_outage_params()
    out_csv = Path(out_csv) if out_csv is not None else default_ucap_csv(dataset_dir)
    years = [int(y) for y in years]
    draws = int(draws)
    if not years:
        raise ValueError("`years` must not be empty")
    if draws < 1:
        raise ValueError("`draws` must be >= 1")

    rows, _static = dataset_row_specs(dataset_dir, params)
    if not rows:
        raise ValueError("no pooled rows; check `carriers` in the outage parameters")
    caps = active_capacities(dataset_dir, params, model_year)

    total = np.zeros(len(rows), dtype=np.float64)
    n_cases = 0
    t0 = time.time()
    for year in years:
        for draw in range(draws):
            avail = row_availability(
                rows,
                caps,
                params,
                year=year,
                draw=draw,
                base_seed=base_seed,
                scheme=scheme,
                window=(0, hours_per_year),
                hours_per_year=hours_per_year,
            )
            total += avail.mean(axis=0)
            n_cases += 1
            if verbose and n_cases % 10 == 0:
                print(f"  ucap: {n_cases}/{len(years) * draws} (year, draw) in {time.time() - t0:.1f} s")
    empirical = total / n_cases
    n_samples = n_cases * hours_per_year

    records = []
    for j, row in enumerate(rows):
        cp = params.carriers[row.carrier]
        capacity = float(caps.get(row.name, 0.0))
        n_slots = slot_count(capacity, cp.unit_size_mw)
        analytic = 1.0 - cp.forced_outage_rate
        records.append(
            {
                "component": row.component,
                "row": row.name,
                "carrier": row.carrier,
                "bus": row.bus,
                "ordinal": int(row.ordinal),
                "p_nom_mw": capacity,
                "unit_size_mw": float(cp.unit_size_mw),
                "n_units_row": int(n_slots),
                "forced_outage_rate": float(cp.forced_outage_rate),
                "ucap_analytic": analytic,
                # A row with no slots derates nothing; report the analytic value.
                "ucap_empirical": analytic if n_slots == 0 else float(empirical[j]),
                "n_samples": 0 if n_slots == 0 else n_samples,
                "outage_scheme": str(scheme),
                "base_seed": int(base_seed),
                "n_years": len(years),
                "n_draws": draws,
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

    df = df[UCAP_COLUMNS]
    df.to_csv(out_csv, index=False)
    if verbose:
        print(
            f"wrote {out_csv} ({len(df)} rows) from {n_cases} (year, draw) cases, "
            f"{n_samples} hourly samples per row, scheme {scheme}, seed {base_seed}"
        )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_row_summary(rows: Sequence[RowSpec], caps: Mapping[str, float], params: OutageParams):
    counts = slot_counts(rows, caps, params)
    by_carrier: dict[str, list[int]] = {}
    for row in rows:
        by_carrier.setdefault(row.carrier, []).append(counts.get(row.name, 0))
    print(f"pooled rows: {len(rows)}; slots at active as-built p_nom: {sum(counts.values())}")
    for carrier in sorted(by_carrier, key=lambda c: -sum(by_carrier[c])):
        cp = params.carriers[carrier]
        slots = by_carrier[carrier]
        print(
            f"  {carrier:>22s}: {len(slots):4d} rows, {sum(slots):6d} slots @ "
            f"{cp.unit_size_mw:g} MW, FOR={cp.forced_outage_rate:.3f}, "
            f"MTTR={cp.mttr_h:g} h, MTTF={cp.mttf_h:.0f} h"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m zap.reliability.outages",
        description="Slot-keyed forced-outage sampling and the UCAP table.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--dataset-dir", type=Path, required=True)
        sp.add_argument("--params", type=Path, default=None, help="outage_params.yaml override")
        sp.add_argument(
            "--scheme", type=str, default=DEFAULT_SCHEME, help="outage key scheme"
        )
        sp.add_argument(
            "--model-year",
            type=int,
            default=None,
            help="investment year for the lifetime rule (default: detect from the dataset)",
        )

    ucap_p = sub.add_parser("ucap", help="sample on demand and write ucap.csv")
    common(ucap_p)
    ucap_p.add_argument("--years", type=int, nargs="+", required=True)
    ucap_p.add_argument("--draws", type=int, required=True)
    ucap_p.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    ucap_p.add_argument("--hours-per-year", type=int, default=HOURS_PER_YEAR)
    ucap_p.add_argument("--out-csv", type=Path, default=None)

    rows_p = sub.add_parser("rows", help="print the pooled rows and their slot counts")
    common(rows_p)

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

    if args.command == "rows":
        rows, _ = dataset_row_specs(args.dataset_dir, params)
        caps = active_capacities(args.dataset_dir, params, args.model_year)
        print(f"model year: {resolve_model_year(args.dataset_dir, args.model_year)}")
        print(f"scheme: {args.scheme}")
        _print_row_summary(rows, caps, params)
        return 0

    if args.command == "ucap":
        write_ucap(
            args.dataset_dir,
            years=args.years,
            draws=args.draws,
            base_seed=args.seed,
            scheme=args.scheme,
            hours_per_year=args.hours_per_year,
            out_csv=args.out_csv,
            params=params,
            model_year=args.model_year,
        )
        return 0

    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
