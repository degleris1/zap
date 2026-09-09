"""A fixture on which the *old* and *new* planning paths build identical systems.

WP5 spec section 7.2 (``memory/plans/2026-09-08-phase1-spec-wp5.md``) needs one
network expressed twice: as a pypsa ``.nc`` for
``experiments/multi_year/runner.py`` (which goes through
``zap.importers.pypsa.load_pypsa_network``) and as a weather-year dataset
directory for ``zap.importers.wy_store.load_system``.  Both are written here
from the *same* in-memory arrays, so any difference between the two device
lists is a difference between the two importers, not between two hand-written
inputs.

The network is deliberately restricted to the subset on which the two importers
agree numerically:

* no export stores and no export links -- ``parse_stores`` is broken (spec D4)
  and a negative link marginal cost makes ``Transporter.operation_cost``
  non-DCP (spec D5);
* all link marginal costs are zero and every link is lossless
  (``efficiency = 1.0``), matched by ``LoadOptions.link_losses = False``;
* every link runs *out of an import-only bus*, so the one-way ``DirectedLine``
  of the new path and the symmetric ``DCLine`` of the old path have the same
  feasible set (see the note on ``p_min_pu`` below);
* storage round-trip efficiencies are 1.0, because the old importer maps
  ``efficiency_dispatch`` onto ``charge_efficiency`` and has no discharge
  efficiency at all;
* time series take only values that are exact in float32, because the weather
  store is float32 while pypsa keeps float64 -- without this the two paths
  differ in the 8th significant digit;
* ``power_unit = cost_unit = 1.0`` and ``demand_scaling = "none"`` on both
  sides.

**Deviation from the spec, reported deliberately.**  Spec section 7.2 asks for
``p_min_pu = -p_max_pu`` on every link so that the two line models share a
feasible set.  That is not buildable: ``DirectedLine.__post_init__`` raises on
``min_power < 0`` (one-way flow is the whole point of the class), so the new
path cannot represent a symmetric link at all.  The equivalent restriction that
*is* buildable is used instead: ``p_min_pu = 0`` and every link sourced at a
bus that carries a generator and nothing else.  Power can only leave such a
bus, so the extra reverse direction the old ``DCLine`` allows is infeasible and
the two feasible sets coincide.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# The network, as plain tables.  Row order here is the row order in both paths.
# --------------------------------------------------------------------------

#: (name, carrier)
EQ_BUSES = [
    ("z1", "AC"),
    ("z2", "AC"),
    ("z1_imports", "imports"),
    ("z2_imports", "imports"),
]

#: name, bus, carrier, p_nom, marginal_cost, efficiency, capital_cost,
#: p_max_pu (static fallback), extendable, p_nom_min, p_nom_max, profile key
#:
#: Every extendable row carries ``p_nom_min == p_nom``.  ``apply_expansion``
#: floors existing capacity at ``p_nom`` (no retirements, WP5-A verification
#: defect 2) while ``experiments/multi_year/runner.py`` -- frozen, and the
#: reference for these tests -- takes ``p_nom_min`` as written; a fixture with
#: ``p_nom_min < p_nom`` would make the two paths solve different LPs for a
#: reason that has nothing to do with the port.  Expansion is still live: every
#: extendable row has head-room up to ``p_nom_max``.
EQ_GENERATORS = [
    ("z1 CCGT", "z1", "CCGT", 100.0, 30.0, 0.5, 29000.0, 1.0, True, 100.0, 300.0, None),
    ("z1 solar", "z1", "solar", 80.0, 0.0, 1.0, 21000.0, 1.0, True, 80.0, 400.0, "solar"),
    ("z1 nuclear", "z1", "nuclear", 30.0, 5.0, 1.0, 0.0, 1.0, False, 0.0, 0.0, None),
    ("z2 CCGT", "z2", "CCGT", 120.0, 35.0, 0.45, 29000.0, 1.0, True, 120.0, 300.0, None),
    ("z2 onwind", "z2", "onwind", 60.0, 0.0, 1.0, 31000.0, 1.0, True, 60.0, 400.0, "wind"),
    (
        "z1_imports unspecified_imports",
        "z1_imports",
        "unspecified_imports",
        50.0,
        60.0,
        1.0,
        0.0,
        0.75,
        False,
        0.0,
        0.0,
        None,
    ),
    (
        "z2_imports unspecified_imports",
        "z2_imports",
        "unspecified_imports",
        40.0,
        65.0,
        1.0,
        0.0,
        0.75,
        False,
        0.0,
        0.0,
        None,
    ),
]

#: name, bus, profile key
EQ_LOADS = [("z1 AC", "z1", "load_z1"), ("z2 AC", "z2", "load_z2")]

#: name, bus0, bus1, carrier, p_nom
EQ_LINKS = [
    ("z1_imports_link", "z1_imports", "z1", "imports", 50.0),
    ("z2_imports_link", "z2_imports", "z2", "imports", 40.0),
]

#: name, bus, carrier, p_nom, max_hours, capital_cost, extendable, p_nom_max
EQ_STORAGE = [("z1 battery", "z1", "battery", 30.0, 4.0, 27491.0, True, 200.0)]

#: The capital costs above are *annual* ($/MW-year), the units PyPSA-USA writes.
#: The fixture is 48 hours long and ``AbstractDevice.sample_time`` pro-rates
#: capital cost only by ``block_hours / total_hours`` -- i.e. not at all when the
#: blocks cover the loaded window -- so an unscaled fixture charges a full year
#: of capex against two days of operations and every extendable row collapses to
#: its lower bound: ``capex_raw`` is 0 and the equivalence test compares two LPs
#: that decide nothing. Pro-rating the costs to the horizon (identically on both
#: sides, since both frames are built from these tables) leaves a live expansion
#: problem while keeping the two paths' inputs bit-identical.
CAPITAL_COST_HOURS = 48.0
CAPITAL_COST_SCALE = CAPITAL_COST_HOURS / 8760.0

EQ_CARRIERS = {
    "CCGT": 0.181,
    "solar": 0.0,
    "onwind": 0.0,
    "nuclear": 0.0,
    "unspecified_imports": 0.0,
    "AC": 0.0,
    "imports": 0.0,
    "battery": 0.0,
}

#: The value ``zap.importers.pypsa.LoadDefaults.marginal_value`` uses, which the
#: new path takes from ``LoadOptions.voll``.  They must agree.
EQ_VOLL = 10_000.0

# --------------------------------------------------------------------------
# Profiles.  Every entry is a dyadic rational or a small integer, i.e. exact in
# float32 -- the weather store is float32 and pypsa is float64.
# --------------------------------------------------------------------------

_SOLAR_DAY = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.125, 0.25, 0.5, 0.75, 0.875,
    1.0, 1.0, 0.875, 0.75, 0.5, 0.25,
    0.125, 0.0, 0.0, 0.0, 0.0, 0.0,
]  # fmt: skip

_WIND_DAY = [
    0.5, 0.625, 0.75, 0.75, 0.625, 0.5,
    0.375, 0.25, 0.25, 0.375, 0.5, 0.625,
    0.75, 0.875, 0.875, 0.75, 0.625, 0.5,
    0.375, 0.375, 0.5, 0.625, 0.75, 0.625,
]  # fmt: skip

_LOAD_Z1_DAY = [
    120.0, 116.0, 112.0, 108.0, 108.0, 112.0,
    124.0, 140.0, 152.0, 156.0, 152.0, 148.0,
    144.0, 144.0, 148.0, 156.0, 168.0, 180.0,
    188.0, 184.0, 172.0, 156.0, 140.0, 128.0,
]  # fmt: skip

_LOAD_Z2_DAY = [
    80.0, 76.0, 72.0, 72.0, 72.0, 76.0,
    84.0, 96.0, 104.0, 108.0, 104.0, 100.0,
    96.0, 96.0, 100.0, 108.0, 116.0, 124.0,
    128.0, 124.0, 116.0, 104.0, 96.0, 88.0,
]  # fmt: skip

_PROFILES = {
    "solar": _SOLAR_DAY,
    "wind": _WIND_DAY,
    "load_z1": _LOAD_Z1_DAY,
    "load_z2": _LOAD_Z2_DAY,
}


def profile(key: str, n_hours: int) -> np.ndarray:
    """The named daily pattern tiled to ``n_hours``, as float64."""
    day = np.asarray(_PROFILES[key], dtype=np.float64)
    reps = int(np.ceil(n_hours / day.size))
    return np.tile(day, reps)[:n_hours]


# --------------------------------------------------------------------------
# Static tables
# --------------------------------------------------------------------------


def _buses_frame() -> pd.DataFrame:
    return pd.DataFrame([{"name": n, "v_nom": 230.0, "carrier": c} for n, c in EQ_BUSES]).set_index(
        "name"
    )


def _generators_frame() -> pd.DataFrame:
    rows = []
    for name, bus, carrier, p_nom, mc, eff, cc, pmax, ext, nmin, nmax, _ in EQ_GENERATORS:
        rows.append(
            {
                "name": name,
                "bus": bus,
                "carrier": carrier,
                "p_nom": p_nom,
                "p_nom_extendable": bool(ext),
                "p_nom_min": nmin if ext else p_nom,
                "p_nom_max": nmax if ext else p_nom,
                "p_min_pu": 0.0,
                "p_max_pu": pmax,
                "marginal_cost": mc,
                "capital_cost": cc * CAPITAL_COST_SCALE,
                "efficiency": eff,
                "committable": False,
                "sign": 1.0,
            }
        )
    return pd.DataFrame(rows).set_index("name")


def _loads_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": name, "bus": bus, "carrier": "AC", "p_set": 0.0, "sign": -1.0}
            for name, bus, _ in EQ_LOADS
        ]
    ).set_index("name")


def _links_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": name,
                "bus0": bus0,
                "bus1": bus1,
                "carrier": carrier,
                "efficiency": 1.0,
                "p_nom": p_nom,
                "p_nom_extendable": False,
                "p_nom_min": p_nom,
                "p_nom_max": p_nom,
                # See the module docstring: one-way, sourced at an import-only
                # bus, so the symmetric DCLine of the old path is equivalent.
                "p_min_pu": 0.0,
                "p_max_pu": 1.0,
                "marginal_cost": 0.0,
                "capital_cost": 0.0,
            }
            for name, bus0, bus1, carrier, p_nom in EQ_LINKS
        ]
    ).set_index("name")


def _storage_frame() -> pd.DataFrame:
    rows = []
    for name, bus, carrier, p_nom, max_hours, cc, ext, nmax in EQ_STORAGE:
        rows.append(
            {
                "name": name,
                "bus": bus,
                "carrier": carrier,
                "p_nom": p_nom,
                "p_nom_extendable": bool(ext),
                # p_nom_min == p_nom: see the note on EQ_GENERATORS.
                "p_nom_min": p_nom,
                "p_nom_max": nmax if ext else p_nom,
                "p_min_pu": -1.0,
                "p_max_pu": 1.0,
                "max_hours": max_hours,
                # Round-trip efficiency 1.0: the old importer has no discharge
                # efficiency and reuses efficiency_dispatch for charging.
                "efficiency_store": 1.0,
                "efficiency_dispatch": 1.0,
                # 50 % initial state of charge, expressed the way the old
                # importer reads it (absolute MWh) and the new one assumes.
                "state_of_charge_initial": 0.5 * p_nom * max_hours,
                "cyclic_state_of_charge": True,
                "marginal_cost": 0.0,
                "capital_cost": cc * CAPITAL_COST_SCALE,
                "standing_loss": 0.0,
                "inflow": 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("name")


def _empty_stores_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "name",
            "bus",
            "carrier",
            "e_nom",
            "e_nom_extendable",
            "marginal_cost",
            "capital_cost",
            "standing_loss",
        ]
    ).set_index("name")


def _carriers_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"name": k, "co2_emissions": v} for k, v in EQ_CARRIERS.items()]
    ).set_index("name")


def _timeseries_frame(
    columns: Sequence[str], keys: Sequence[str], years: Sequence[int], n_hours: int
) -> pd.DataFrame:
    frames = []
    for year in years:
        stamps = pd.date_range("2040-01-01", periods=n_hours, freq="h")
        df = pd.DataFrame({c: profile(k, n_hours) for c, k in zip(columns, keys)}, index=stamps)
        df.index.name = "timestep"
        df["weather_year"] = year
        frames.append(df.set_index("weather_year", append=True).reorder_levels([1, 0]))
    out = pd.concat(frames)
    out.index.names = ["weather_year", "timestep"]
    return out[list(columns)]


# --------------------------------------------------------------------------
# The two writers
# --------------------------------------------------------------------------


def build_pypsa_network(*, n_hours: int = 48):
    """The same system as a ``pypsa.Network`` with ``n_hours`` hourly snapshots."""
    import pypsa

    net = pypsa.Network()
    net.set_snapshots(pd.date_range("2040-01-01", periods=n_hours, freq="h"))

    carriers = _carriers_frame()
    net.madd("Carrier", carriers.index, co2_emissions=carriers["co2_emissions"].values)

    buses = _buses_frame()
    net.madd("Bus", buses.index, v_nom=buses["v_nom"].values, carrier=buses["carrier"].values)

    gens = _generators_frame()
    gen_p_max_pu = pd.DataFrame(index=net.snapshots)
    for name, _, _, _, _, _, _, _, _, _, _, key in EQ_GENERATORS:
        if key is not None:
            gen_p_max_pu[name] = profile(key, n_hours)
    net.madd(
        "Generator",
        gens.index,
        bus=gens["bus"].values,
        carrier=gens["carrier"].values,
        p_nom=gens["p_nom"].values,
        p_nom_extendable=gens["p_nom_extendable"].values,
        p_nom_min=gens["p_nom_min"].values,
        p_nom_max=gens["p_nom_max"].values,
        p_min_pu=gens["p_min_pu"].values,
        p_max_pu=gens["p_max_pu"].values,
        marginal_cost=gens["marginal_cost"].values,
        capital_cost=gens["capital_cost"].values,
        efficiency=gens["efficiency"].values,
    )
    if not gen_p_max_pu.empty:
        net.generators_t.p_max_pu = gen_p_max_pu

    loads = _loads_frame()
    load_p_set = pd.DataFrame(
        {name: profile(key, n_hours) for name, _, key in EQ_LOADS}, index=net.snapshots
    )
    net.madd("Load", loads.index, bus=loads["bus"].values, carrier=loads["carrier"].values)
    net.loads_t.p_set = load_p_set

    links = _links_frame()
    net.madd(
        "Link",
        links.index,
        bus0=links["bus0"].values,
        bus1=links["bus1"].values,
        carrier=links["carrier"].values,
        efficiency=links["efficiency"].values,
        p_nom=links["p_nom"].values,
        p_nom_extendable=links["p_nom_extendable"].values,
        p_nom_min=links["p_nom_min"].values,
        p_nom_max=links["p_nom_max"].values,
        p_min_pu=links["p_min_pu"].values,
        p_max_pu=links["p_max_pu"].values,
        marginal_cost=links["marginal_cost"].values,
        capital_cost=links["capital_cost"].values,
    )

    storage = _storage_frame()
    net.madd(
        "StorageUnit",
        storage.index,
        bus=storage["bus"].values,
        carrier=storage["carrier"].values,
        p_nom=storage["p_nom"].values,
        p_nom_extendable=storage["p_nom_extendable"].values,
        p_nom_min=storage["p_nom_min"].values,
        p_nom_max=storage["p_nom_max"].values,
        max_hours=storage["max_hours"].values,
        efficiency_store=storage["efficiency_store"].values,
        efficiency_dispatch=storage["efficiency_dispatch"].values,
        state_of_charge_initial=storage["state_of_charge_initial"].values,
        cyclic_state_of_charge=storage["cyclic_state_of_charge"].values,
        marginal_cost=storage["marginal_cost"].values,
        capital_cost=storage["capital_cost"].values,
        standing_loss=storage["standing_loss"].values,
    )

    return net


def write_dataset(root: Path, *, n_hours: int = 48, years: Sequence[int] = (2020,)) -> Path:
    """Write the static CSVs, the parquet time series and ``weather.zarr``."""
    from zap.importers.wy_store import convert_dataset

    root = Path(root)
    static = root / "static"
    ts = root / "timeseries"
    meta = root / "meta"
    for d in (static, ts, meta):
        d.mkdir(parents=True, exist_ok=True)

    _buses_frame().to_csv(static / "buses.csv")
    _generators_frame().to_csv(static / "generators.csv")
    _loads_frame().to_csv(static / "loads.csv")
    _links_frame().to_csv(static / "links.csv")
    _storage_frame().to_csv(static / "storage_units.csv")
    _empty_stores_frame().to_csv(static / "stores.csv")
    _carriers_frame().to_csv(static / "carriers.csv")

    gen_cols = [g[0] for g in EQ_GENERATORS if g[11] is not None]
    gen_keys = [g[11] for g in EQ_GENERATORS if g[11] is not None]
    _timeseries_frame(gen_cols, gen_keys, years, n_hours).to_parquet(
        ts / "generators_t_p_max_pu.parquet"
    )

    load_cols = [ld[0] for ld in EQ_LOADS]
    load_keys = [ld[2] for ld in EQ_LOADS]
    _timeseries_frame(load_cols, load_keys, years, n_hours).to_parquet(ts / "loads_t_p_set.parquet")

    for year in years:
        (meta / f"wy{year}.json").write_text(
            json.dumps({"weather_year": int(year), "n_snapshots": int(n_hours)}, indent=2)
        )

    convert_dataset(root)
    return root


def write_equivalence_fixture(root: Path, *, n_hours: int = 48):
    """Write the dataset directory *and* build the matching ``pypsa.Network``.

    Returns ``(dataset_dir, pypsa_network)``.  The caller writes the network to
    a ``.nc`` if the old path needs one.
    """
    dataset = write_dataset(root, n_hours=n_hours, years=(2020,))
    return dataset, build_pypsa_network(n_hours=n_hours)


def load_options(*, n_hours: int = 48):
    """``LoadOptions`` that reproduce the old importer's defaults exactly."""
    from zap.importers.wy_store import HourWindow, LoadOptions

    return LoadOptions(
        years=(2020,),
        window=HourWindow(start=0, stop=n_hours),
        voll=EQ_VOLL,
        demand_scaling="none",
        ucap_derate=False,
        outage_draw=None,
        link_losses=False,
        export_mode="drop",
        carbon_tax=0.0,
        power_unit=1.0,
        cost_unit=1.0,
        storage_init_soc=0.5,
        storage_final_soc=0.5,
    )


def pypsa_args() -> dict:
    """``pypsa_args`` for ``runner.py`` that match :func:`load_options`."""
    return {"power_unit": 1.0, "cost_unit": 1.0}


# --------------------------------------------------------------------------
# Device comparison
# --------------------------------------------------------------------------

#: old class name -> new class name.  WP3 replaced the symmetric ``DCLine``
#: with the one-way ``DirectedLine``; everything else keeps its class.
CLASS_ALIASES = {"DCLine": "DirectedLine"}

#: Per (new) class: the attributes the two importers must agree on.
COMPARED_ATTRS = {
    "Generator": (
        "terminal",
        "nominal_capacity",
        "dynamic_capacity",
        "linear_cost",
        "capital_cost",
        "emission_rates",
    ),
    "Load": ("terminal", "load", "linear_cost"),
    "DirectedLine": (
        "source_terminal",
        "sink_terminal",
        "nominal_capacity",
        "linear_cost",
        "capital_cost",
    ),
    "StorageUnit": (
        "terminal",
        "power_capacity",
        "duration",
        "charge_efficiency",
        "linear_cost",
        "capital_cost",
        "initial_soc",
        "final_soc",
    ),
}

#: Expansion bounds are *not* set by the importer on the new path: WP1's
#: ``load_system`` pins ``min == max == p_nom`` (operations only) and
#: ``planning.expansion.apply_expansion`` is what reads ``p_nom_extendable`` /
#: ``p_nom_min`` / ``p_nom_max`` back off the static CSVs.  Compare them only
#: after expansion has been applied.
BOUND_ATTRS = {
    "Generator": ("min_nominal_capacity", "max_nominal_capacity"),
    "Load": (),
    "DirectedLine": ("min_nominal_capacity", "max_nominal_capacity"),
    "StorageUnit": ("min_power_capacity", "max_power_capacity"),
}

#: Attributes whose *name* differs between the two line classes but whose value
#: must still match: the old ``DCLine.capacity`` is the new
#: ``DirectedLine.max_power`` (both are the per-unit flow limit).
RENAMED_ATTRS = {"DirectedLine": {"capacity": "max_power"}}


def _as_array(value) -> np.ndarray:
    return np.atleast_2d(np.asarray(value, dtype=np.float64))


def _compare(old_value, new_value, label: str, rtol: float) -> None:
    old_arr = _as_array(old_value)
    new_arr = _as_array(new_value)
    try:
        old_b, new_b = np.broadcast_arrays(old_arr, new_arr)
    except ValueError as exc:  # pragma: no cover - a shape bug, not a value bug
        raise AssertionError(
            f"{label}: incompatible shapes {old_arr.shape} (old) vs {new_arr.shape} (new)"
        ) from exc
    np.testing.assert_allclose(new_b, old_b, rtol=rtol, atol=0.0, err_msg=label)


def devices_by_class(devices: list) -> dict:
    return {type(d).__name__: d for d in devices}


def assert_devices_match(
    old_devices: list,
    new_devices: list,
    *,
    rtol: float = 1e-12,
    compare_bounds: bool = True,
) -> None:
    """Assert the two importers built the same system.

    ``old_devices`` come from ``load_pypsa_network`` (via ``MultiYearBlockSampler``),
    ``new_devices`` from ``wy_store.load_system``.  Raises ``AssertionError`` with
    the offending device and attribute named.

    ``compare_bounds=False`` skips the expansion bounds (``min_/max_nominal_capacity``
    and the storage equivalents), which the new path only fills in once
    ``planning.expansion.apply_expansion`` has run -- see :data:`BOUND_ATTRS`.
    """
    old = devices_by_class(old_devices)
    new = devices_by_class(new_devices)

    old_mapped = {CLASS_ALIASES.get(k, k): v for k, v in old.items()}
    if set(old_mapped) != set(new):
        raise AssertionError(
            f"device sets differ: old={sorted(old_mapped)} (raw {sorted(old)}), new={sorted(new)}"
        )

    for cls_name in sorted(new):
        old_dev = old_mapped[cls_name]
        new_dev = new[cls_name]
        if old_dev.num_devices != new_dev.num_devices:
            raise AssertionError(
                f"{cls_name}: {old_dev.num_devices} devices (old) vs {new_dev.num_devices} (new)"
            )
        old_names = list(np.asarray(old_dev.name).reshape(-1))
        new_names = list(np.asarray(new_dev.name).reshape(-1))
        if old_names != new_names:
            raise AssertionError(f"{cls_name}: row names differ\nold={old_names}\nnew={new_names}")

        attrs = tuple(COMPARED_ATTRS[cls_name])
        if compare_bounds:
            attrs += tuple(BOUND_ATTRS[cls_name])
        for attr in attrs:
            old_value = getattr(old_dev, attr, None)
            new_value = getattr(new_dev, attr, None)
            if old_value is None and new_value is None:
                continue
            if old_value is None or new_value is None:
                raise AssertionError(
                    f"{cls_name}.{attr}: present on only one side "
                    f"(old={old_value is not None}, new={new_value is not None})"
                )
            _compare(old_value, new_value, f"{cls_name}.{attr}", rtol)

        for old_attr, new_attr in RENAMED_ATTRS.get(cls_name, {}).items():
            _compare(
                getattr(old_dev, old_attr),
                getattr(new_dev, new_attr),
                f"{cls_name}.{old_attr} (old) vs .{new_attr} (new)",
                rtol,
            )
