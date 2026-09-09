"""A synthetic dataset with the same *shape* as a pypsa-usa weather-year export.

The fixture is hermetic: nothing here reads ``data/``.  It is shared by the WP1,
WP2, WP3 and WP4 test suites, so keep the component names stable.

Layout written under ``root``::

    static/{buses,generators,loads,links,storage_units,stores,carriers}.csv
    timeseries/{generators_t_p_max_pu,loads_t_p_set,links_t_marginal_cost}.parquet
    meta/wy<YYYY>.json

Time series are deterministic functions of the hour index, so any test can
recompute the expected value without reading the parquet back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# name, bus, carrier, p_nom, marginal_cost, efficiency, capital_cost, p_max_pu,
# in_parquet, build_year, lifetime
#
# The model year of the fixture is 2040 (the timeseries snapshots are stamped
# 2040), so ``z2 old CCGT`` (1990 + 30 = 2020 <= 2040) is retired by the
# lifetime rule and ``z1 legacy CCGT`` (infinite lifetime) never is.
TINY_MODEL_YEAR = 2040

TINY_GENERATORS = [
    ("z1 CCGT", "z1", "CCGT", 100.0, 30.0, 0.50, 29000.0, 1.0, True, 2030, 30.0),
    ("z1 solar", "z1", "solar", 80.0, 0.0, 1.00, 21000.0, 1.0, True, 2035, 25.0),
    ("z2 CCGT", "z2", "CCGT", 120.0, 35.0, 0.45, 29000.0, 1.0, True, 2030, 30.0),
    ("z2 onwind", "z2", "onwind", 60.0, 0.0, 1.00, 31000.0, 1.0, True, 2035, 25.0),
    # No column in the p_max_pu parquet: exercises the static-scalar fallback.
    (
        "z1_imports unspecified_imports",
        "z1_imports",
        "unspecified_imports",
        50.0,
        60.0,
        1.0,
        0.0,
        0.9,
        False,
        2030,
        30.0,
    ),
    ("z2 hydro", "z2", "hydro", 40.0, 1.0, 1.00, 0.0, 1.0, True, 2020, 40.0),
    # Retired by 2040: build_year + lifetime == 2020 <= 2040.
    ("z2 old CCGT", "z2", "CCGT", 500.0, 45.0, 0.45, 29000.0, 1.0, True, 1990, 30.0),
    # Infinite lifetime: never retires, however old it is.  Small and expensive
    # so it does not displace the CCGT fleet in the dispatch/emissions tests.
    ("z1 legacy CCGT", "z1", "CCGT", 20.0, 55.0, 0.40, 29000.0, 1.0, True, 1985, np.inf),
]

#: Generator rows the lifetime rule retires at :data:`TINY_MODEL_YEAR`.
TINY_RETIRED_GENERATORS = ["z2 old CCGT"]

# name, bus
TINY_LOADS = [("z1 AC", "z1"), ("z2 AC", "z2")]

# name, bus0, bus1, carrier, p_nom, efficiency, marginal_cost, in_cost_parquet
TINY_LINKS = [
    ("z1||z2_fwd", "z1", "z2", "AC", 50.0, 0.95, 0.0, False),
    ("z1||z2_rev", "z2", "z1", "AC", 40.0, 0.95, 0.0, False),
    ("z1_imports_link", "z1_imports", "z1", "imports", 50.0, 1.0, 0.0, False),
    ("z1_exports_link", "z1", "z1_exports", "exports", 60.0, 1.0, 0.0, True),
]

# name, bus, carrier, p_nom, max_hours, efficiency_store, efficiency_dispatch,
# capital_cost, build_year, lifetime
TINY_STORAGE = [
    ("z1 battery", "z1", "battery", 30.0, 4.0, 0.95, 0.95, 27491.0, 2035, 20.0),
    ("z2 PHS", "z2", "PHS", 20.0, 8.0, 0.90, 0.90, 0.0, 1970, np.inf),
]

TINY_BUSES = ["z1", "z2", "z1_imports", "z1_exports"]

#: ``(x, y)`` per bus, in the same order as :data:`TINY_BUSES`.
TINY_BUS_COORDS = [(-121.5, 38.6), (-118.2, 34.1), (-119.8, 36.7), (-117.2, 33.0)]

TINY_CARRIERS = {
    "CCGT": 0.181,
    "solar": 0.0,
    "onwind": 0.0,
    "hydro": 0.0,
    "unspecified_imports": 0.0,
    "AC": 0.0,
    "imports": 0.0,
    "exports": 0.0,
    "battery": 0.0,
    "PHS": 0.0,
}

#: PyPSA-USA ships a ``color`` column on ``carriers.csv``; mirror it here so
#: readers that expect it (``persist.build_system_static``) see the real shape.
TINY_CARRIER_COLORS = {
    "CCGT": "#b20101",
    "solar": "#f9d002",
    "onwind": "#235ebc",
    "hydro": "#08ad97",
    "battery": "#b8ea04",
    "PHS": "#08ad97",
    "AC": "#70af1d",
}

EXPORT_MARGINAL_COST = -178.0


def _hours(n_hours: int) -> np.ndarray:
    return np.arange(n_hours, dtype=np.float64)


def generator_profile(name: str, n_hours: int, year: int) -> np.ndarray:
    """Deterministic capacity factor for a generator column."""
    t = _hours(n_hours)
    phase = 0.1 * (year - 2020)
    if name == "z1 solar":
        return np.clip(np.sin(2 * np.pi * (t / 24.0) - np.pi / 2 + phase), 0.0, 1.0)
    if name == "z2 onwind":
        return 0.5 + 0.3 * np.cos(2 * np.pi * (t / 17.0) + phase)
    if name == "z2 hydro":
        return 0.25 + 0.05 * np.cos(2 * np.pi * (t / 31.0) + phase)
    return np.ones(n_hours)


def load_profile(name: str, n_hours: int, year: int) -> np.ndarray:
    """Deterministic demand for a load column, in MW."""
    t = _hours(n_hours)
    phase = 0.1 * (year - 2020)
    if name == "z1 AC":
        return 100.0 + 50.0 * np.sin(2 * np.pi * (t / 24.0) + phase)
    return 80.0 + 30.0 * np.cos(2 * np.pi * (t / 24.0) + phase)


def _timeseries_frame(
    columns: Sequence[str],
    values,
    years: Sequence[int],
    n_hours: int,
) -> pd.DataFrame:
    frames = []
    for year in years:
        stamps = pd.date_range("2040-01-01", periods=n_hours, freq="h")
        data = {c: values(c, n_hours, year) for c in columns}
        df = pd.DataFrame(data, index=stamps)
        df.index.name = "timestep"
        df["weather_year"] = year
        frames.append(df.set_index("weather_year", append=True).reorder_levels([1, 0]))
    out = pd.concat(frames)
    out.index.names = ["weather_year", "timestep"]
    return out[list(columns)]


def write_tiny_dataset(
    root: Path, *, n_hours: int = 48, years: Sequence[int] = (2020, 2021)
) -> Path:
    """Write a synthetic dataset with the same shape as a pypsa-usa export."""
    root = Path(root)
    static = root / "static"
    ts = root / "timeseries"
    meta = root / "meta"
    for d in (static, ts, meta):
        d.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "name": TINY_BUSES,
            "v_nom": 230.0,
            "carrier": ["AC", "AC", "imports", "exports"],
            # The real exports carry coordinates; `persist.write_system_static`
            # copies them so a later map figure has somewhere to draw.
            "x": [x for x, _ in TINY_BUS_COORDS],
            "y": [y for _, y in TINY_BUS_COORDS],
        }
    ).set_index("name").to_csv(static / "buses.csv")

    gens = pd.DataFrame(
        [
            {
                "name": name,
                "bus": bus,
                "carrier": carrier,
                "p_nom": p_nom,
                "p_nom_extendable": False,
                "p_min_pu": 0.0,
                "p_max_pu": p_max_pu,
                "e_sum_min": -np.inf,
                "e_sum_max": np.inf,
                "marginal_cost": mc,
                "capital_cost": cc,
                "efficiency": eff,
                "committable": False,
                "sign": 1.0,
                "active": True,
                "build_year": build_year,
                "lifetime": lifetime,
            }
            for (
                name,
                bus,
                carrier,
                p_nom,
                mc,
                eff,
                cc,
                p_max_pu,
                _,
                build_year,
                lifetime,
            ) in TINY_GENERATORS
        ]
    ).set_index("name")
    gens.to_csv(static / "generators.csv")

    pd.DataFrame(
        [
            {"name": name, "bus": bus, "carrier": "AC", "p_set": 0.0, "sign": -1.0}
            for name, bus in TINY_LOADS
        ]
    ).set_index("name").to_csv(static / "loads.csv")

    pd.DataFrame(
        [
            {
                "name": name,
                "bus0": bus0,
                "bus1": bus1,
                "carrier": carrier,
                "efficiency": eff,
                "p_nom": p_nom,
                "p_nom_extendable": False,
                "p_min_pu": 0.0,
                "p_max_pu": 1.0,
                "marginal_cost": mc,
                "capital_cost": 0.0,
            }
            for name, bus0, bus1, carrier, p_nom, eff, mc, _ in TINY_LINKS
        ]
    ).set_index("name").to_csv(static / "links.csv")

    pd.DataFrame(
        [
            {
                "name": name,
                "bus": bus,
                "carrier": carrier,
                "p_nom": p_nom,
                "p_nom_extendable": False,
                "p_min_pu": -1.0,
                "p_max_pu": 1.0,
                "max_hours": max_hours,
                "efficiency_store": eff_s,
                "efficiency_dispatch": eff_d,
                "state_of_charge_initial": 0.0,
                "cyclic_state_of_charge": True,
                "marginal_cost": 0.0,
                "capital_cost": cc,
                "standing_loss": 0.0,
                "inflow": 0.0,
                "active": True,
                "build_year": build_year,
                "lifetime": lifetime,
            }
            for (
                name,
                bus,
                carrier,
                p_nom,
                max_hours,
                eff_s,
                eff_d,
                cc,
                build_year,
                lifetime,
            ) in TINY_STORAGE
        ]
    ).set_index("name").to_csv(static / "storage_units.csv")

    pd.DataFrame(
        [
            {
                "name": "z1_exports",
                "bus": "z1_exports",
                "carrier": "exports",
                "e_nom": 0.0,
                "e_nom_extendable": True,
                "marginal_cost": 0.0,
                "capital_cost": 0.0,
                "standing_loss": 0.0,
            }
        ]
    ).set_index("name").to_csv(static / "stores.csv")

    pd.DataFrame(
        [
            {
                "name": k,
                "co2_emissions": v,
                "color": TINY_CARRIER_COLORS.get(k, ""),
                "nice_name": k.replace("_", " ").title(),
            }
            for k, v in TINY_CARRIERS.items()
        ]
    ).set_index("name").to_csv(static / "carriers.csv")

    gen_cols = [g[0] for g in TINY_GENERATORS if g[8]]
    _timeseries_frame(gen_cols, generator_profile, years, n_hours).to_parquet(
        ts / "generators_t_p_max_pu.parquet"
    )

    load_cols = [name for name, _ in TINY_LOADS]
    _timeseries_frame(load_cols, load_profile, years, n_hours).to_parquet(
        ts / "loads_t_p_set.parquet"
    )

    link_cols = [link[0] for link in TINY_LINKS if link[7]]
    _timeseries_frame(
        link_cols, lambda c, n, y: np.full(n, EXPORT_MARGINAL_COST), years, n_hours
    ).to_parquet(ts / "links_t_marginal_cost.parquet")

    for year in years:
        (meta / f"wy{year}.json").write_text(
            json.dumps({"weather_year": int(year), "n_snapshots": int(n_hours)}, indent=2)
        )

    return root


def write_tiny_ucap_csv(
    root: Path,
    *,
    generator_factors: Optional[dict] = None,
    storage_factors: Optional[dict] = None,
) -> Path:
    """Hand-written ``ucap.csv`` with WP2's schema, for the WP1 join tests."""
    generator_factors = generator_factors or {}
    storage_factors = storage_factors or {}

    rows = []
    for name, bus, carrier, p_nom, *_ in TINY_GENERATORS:
        if name not in generator_factors:
            continue
        factor = float(generator_factors[name])
        rows.append(
            {
                "component": "Generator",
                "row": name,
                "carrier": carrier,
                "bus": bus,
                "p_nom_mw": p_nom,
                "unit_size_mw": 50.0,
                "n_units_row": 3,
                "forced_outage_rate": 1.0 - factor,
                "ucap_analytic": factor,
                "ucap_empirical": factor,
                "n_samples": 100,
                "ucap_carrier_bus_empirical": factor,
            }
        )
    for name, bus, carrier, p_nom, *_ in TINY_STORAGE:
        if name not in storage_factors:
            continue
        factor = float(storage_factors[name])
        rows.append(
            {
                "component": "StorageUnit",
                "row": name,
                "carrier": carrier,
                "bus": bus,
                "p_nom_mw": p_nom,
                "unit_size_mw": 50.0,
                "n_units_row": 3,
                "forced_outage_rate": 1.0 - factor,
                "ucap_analytic": factor,
                "ucap_empirical": factor,
                "n_samples": 100,
                "ucap_carrier_bus_empirical": factor,
            }
        )

    path = Path(root) / "ucap.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def data_root() -> Path:
    """The brain repo's ``data/`` directory."""
    try:  # pragma: no cover - WP4 may not have landed
        from zap.experiments.ra.paths import data_root as _data_root

        return Path(_data_root())
    except Exception:
        import os

        env = os.environ.get("CH3_DATA_DIR")
        if env:
            return Path(env)
        import zap

        return Path(zap.__file__).parent.parent.parent / "data"


def real_z4_dir() -> Optional[Path]:
    """``data/ca2040_z4`` if it exists *and* has a weather store, else ``None``."""
    candidate = data_root() / "ca2040_z4"
    if (candidate / "weather.zarr").exists():
        return candidate
    return None
