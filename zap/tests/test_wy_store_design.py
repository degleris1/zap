"""Design-aware system loading: ``LoadOptions.design_capacity`` (WP-E1).

The defect these tests pin down: ``load_system`` used to compute outage
availability from the static tables' *as-built* ``p_nom``, and
``row_availability`` returns 1.0 for a zero-capacity row.  So every megawatt a
planner built on a retired or greenfield row was perfectly reliable, and
capacity added to a live row was derated by its as-built units only.  The fix
imposes the design on the static tables *before* the outage lookup, so the
loaded system is the designed system.

Everything here runs on the hermetic tiny fixture with a hand-written outage
store; the one integration check against ``data/ca2040_z4`` skips when the
dataset is not present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import pytest
import zarr

from zap.importers.wy_store import (
    DESIGN_CAPACITY_TABLES,
    HourWindow,
    LoadOptions,
    apply_design_capacity,
    convert_dataset,
    design_capacity_digest,
    load_system,
    read_static,
)
from zap.tests.fixtures.tiny_dataset import real_z4_dir, write_tiny_dataset, write_tiny_ucap_csv

N_HOURS = 168
YEARS = (2020,)
DRAWS = (0, 1)

#: Pooled rows of the hand-written store: row -> (component, carrier, bus,
#: unit size MW, slots).  ``z2 old CCGT`` is retired by the fixture's lifetime
#: rule, so its as-built capacity is 0 -- the row the defect exempted entirely.
POOL_ROWS = {
    "z1 CCGT": ("Generator", "CCGT", "z1", 250.0, 6),
    "z2 old CCGT": ("Generator", "CCGT", "z2", 250.0, 6),
    "z1 battery": ("StorageUnit", "battery", "z1", 50.0, 8),
}

#: Forced-outage rate the synthetic uptimes are drawn at.
FOR = 0.10

#: As-built ``p_nom`` overrides written onto the fixture's static tables, so the
#: expansion cases are round numbers of virtual units.
STATIC_P_NOM = {"z1 CCGT": 250.0, "z2 old CCGT": 500.0}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_outage_store(root: Path, *, seed: int = 20260909) -> Path:
    """A hand-written ``outages.zarr`` in WP2's schema with iid uptimes.

    Unit uptimes are iid Bernoulli(1 - FOR) per hour, so a row's mean
    availability is 1 - FOR up to sampling noise and the *same* unit
    realisations are shared by every design (common random numbers).
    """
    unit_id, unit_row, unit_carrier, unit_bus, unit_component = [], [], [], [], []
    unit_slot, unit_offset, unit_units, unit_size = [], [], [], []
    offset = 0
    for row, (component, carrier, bus, size, slots) in POOL_ROWS.items():
        for slot in range(slots):
            unit_id.append(f"{row}#{slot}")
            unit_row.append(row)
            unit_carrier.append(carrier)
            unit_bus.append(bus)
            unit_component.append(component)
            unit_slot.append(slot)
            unit_offset.append(offset)
            unit_units.append(slots)
            unit_size.append(size)
        offset += slots

    n_units = len(unit_id)
    rng = np.random.default_rng(seed)
    available = (rng.random((len(YEARS), len(DRAWS), N_HOURS, n_units)) >= FOR).astype(np.uint8)

    path = Path(root) / "outages.zarr"
    group = zarr.open_group(str(path), mode="w")
    arr = group.create_dataset(
        "available",
        shape=available.shape,
        chunks=(1, 1, N_HOURS, n_units),
        dtype="uint8",
        fill_value=255,
        compressor=numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE),
    )
    arr[:] = available

    done = group.create_dataset(
        "done", shape=(len(YEARS), len(DRAWS)), chunks=(1, 1), dtype="uint8"
    )
    done[:] = 1

    def _int(name, values, dtype="int32"):
        a = group.create_dataset(name, shape=(len(values),), chunks=(len(values),), dtype=dtype)
        a[:] = np.asarray(values, dtype=dtype)

    def _str(name, values):
        a = group.create_dataset(
            name,
            shape=(len(values),),
            chunks=(len(values),),
            dtype=object,
            object_codec=numcodecs.VLenUTF8(),
        )
        a[:] = np.array(values, dtype=object)

    _int("weather_year", list(YEARS))
    _int("draw", list(DRAWS))
    _int("hour", np.arange(N_HOURS))
    _int("unit_slot", unit_slot)
    _int("unit_row_offset", unit_offset)
    _int("unit_row_units", unit_units)
    _int("unit_size_mw", unit_size, dtype="float32")
    _str("unit_id", unit_id)
    _str("unit_row", unit_row)
    _str("unit_carrier", unit_carrier)
    _str("unit_bus", unit_bus)
    _str("unit_component", unit_component)

    group.attrs.update(
        {
            "generator_version": 1,
            "dataset": Path(root).name,
            "base_seed": seed,
            "weather_years": list(YEARS),
            "n_draws": len(DRAWS),
            "hours_per_year": N_HOURS,
        }
    )
    return path


@pytest.fixture
def dataset(tmp_path) -> Path:
    root = write_tiny_dataset(tmp_path / "tiny", n_hours=N_HOURS, years=YEARS)

    gens = pd.read_csv(root / "static" / "generators.csv", index_col=0)
    for name, value in STATIC_P_NOM.items():
        gens.loc[name, "p_nom"] = value
    gens.to_csv(root / "static" / "generators.csv")

    convert_dataset(root)
    _write_outage_store(root)
    return root


def _load(dataset: Path, **kwargs):
    return load_system(
        dataset,
        LoadOptions(
            years=kwargs.pop("years", YEARS),
            window=kwargs.pop("window", HourWindow(0, N_HOURS)),
            **kwargs,
        ),
    )


def _gen(system):
    return system.index.get(system.devices, "Generator")


def _row(system, cls_name: str, name: str) -> int:
    return list(system.index.names[cls_name]).index(name)


def _as_built(system) -> dict[str, np.ndarray]:
    """The loaded system's capacities, as a design mapping (the identity design).

    ``np.array`` copies: a view would let a test's edits reach into the loaded
    system's device arrays.
    """
    storage = system.index.get(system.devices, "StorageUnit")
    line = system.index.get(system.devices, "DirectedLine")
    return {
        "Generator": np.array(_gen(system).nominal_capacity, dtype=float).reshape(-1),
        "StorageUnit": np.array(storage.power_capacity, dtype=float).reshape(-1),
        "DirectedLine": np.array(line.nominal_capacity, dtype=float).reshape(-1),
    }


def _design(system, **overrides) -> dict[str, np.ndarray]:
    """The identity design with named rows moved to a new capacity."""
    capacities = _as_built(system)
    for name, value in overrides.items():
        row_name = name.replace("_", " ")
        for cls_name in ("Generator", "StorageUnit"):
            names = list(system.index.names[cls_name])
            if row_name in names:
                capacities[cls_name][names.index(row_name)] = float(value)
                break
        else:  # pragma: no cover - a typo in a test
            raise KeyError(row_name)
    return capacities


def _pool_and_up(dataset: Path, draw: int):
    """The store's ``UnitPool`` and its ``(n_units, n_hours)`` uptime matrix."""
    from zap.importers.wy_store import _open_outage_store, _unit_pool_from_store

    _, root = _open_outage_store(dataset)
    pool = _unit_pool_from_store(root)
    up = np.asarray(root["available"][0, list(DRAWS).index(draw), :N_HOURS, :], dtype=np.uint8).T
    return pool, up


# ---------------------------------------------------------------------------
# (a) capacity built on a zero-as-built row draws from the pool
# ---------------------------------------------------------------------------


def test_zero_as_built_row_built_by_a_design_is_derated(dataset):
    """The E1 regression: 1,000 MW on a retired row is *not* outage-free."""
    base = _load(dataset, outage_draw=0)
    row = _row(base, "Generator", "z2 old CCGT")
    assert base.index.get(base.devices, "Generator").nominal_capacity.ravel()[row] == 0.0

    # As-built, the row has no capacity, so its availability is 1.0 everywhere:
    # that is the behaviour the design must not inherit.
    np.testing.assert_array_equal(_gen(base).dynamic_capacity[row, :], np.ones(N_HOURS))

    designed = _load(
        dataset,
        outage_draw=0,
        design_capacity=_design(base, z2_old_CCGT=1000.0),
        design_id="expand_retired",
    )
    availability = _gen(designed).dynamic_capacity[row, :]  # p_max_pu is 1.0 for this row

    assert _gen(designed).nominal_capacity.ravel()[row] == pytest.approx(1000.0)
    assert availability.min() < 1.0, "built capacity on a retired row must draw from the pool"
    assert availability.max() <= 1.0
    # 1000 MW / 250 MW units = 4 weight-1 units of iid Bernoulli(1 - FOR).
    n_units = 4
    sigma = math.sqrt(FOR * (1.0 - FOR) / (N_HOURS * n_units))
    assert availability.mean() == pytest.approx(1.0 - FOR, abs=5.0 * sigma)
    assert set(np.unique(availability)) <= {k / n_units for k in range(n_units + 1)}


def test_storage_built_by_a_design_is_derated_and_energy_follows(dataset):
    base = _load(dataset, outage_draw=0)
    row = _row(base, "StorageUnit", "z1 battery")
    designed = _load(dataset, outage_draw=0, design_capacity=_design(base, z1_battery=300.0))
    storage = designed.index.get(designed.devices, "StorageUnit")

    assert storage.power_capacity.ravel()[row] == pytest.approx(300.0)
    # Energy follows p_nom * max_hours, so the 4-hour battery keeps its duration.
    assert storage.duration.ravel()[row] == pytest.approx(4.0)
    availability = storage.power_availability[row, :]
    assert availability.min() < 1.0
    # 300 MW / 50 MW units = 6 weight-1 units.
    assert set(np.unique(availability)) <= {k / 6 for k in range(7)}


# ---------------------------------------------------------------------------
# (b) an expanded live row uses the designed number of units
# ---------------------------------------------------------------------------


def test_expanded_row_uses_the_designed_units(dataset):
    from zap.reliability.outages import row_availability

    base = _load(dataset, outage_draw=0)
    row_name = "z1 CCGT"
    row = _row(base, "Generator", row_name)
    assert _gen(base).nominal_capacity.ravel()[row] == pytest.approx(250.0)

    designed = _load(
        dataset, outage_draw=0, design_capacity=_design(base, z1_CCGT=750.0), design_id="expand"
    )

    pool, up = _pool_and_up(dataset, draw=0)
    expected = row_availability(up, pool, pd.Series({row_name: 750.0}), [row_name])[:, 0]
    np.testing.assert_array_equal(_gen(designed).dynamic_capacity[row, :], expected)

    # ... and it is genuinely three units, not the as-built one.
    as_built = row_availability(up, pool, pd.Series({row_name: 250.0}), [row_name])[:, 0]
    np.testing.assert_array_equal(_gen(base).dynamic_capacity[row, :], as_built)
    assert not np.array_equal(expected, as_built)
    assert set(np.unique(expected)) <= {0.0, 1 / 3, 2 / 3, 1.0}


def test_partial_last_unit_is_weighted(dataset):
    """A capacity between unit sizes weights the last unit by the remainder."""
    from zap.reliability.outages import row_availability

    base = _load(dataset, outage_draw=0)
    row_name = "z1 CCGT"
    row = _row(base, "Generator", row_name)
    designed = _load(dataset, outage_draw=0, design_capacity=_design(base, z1_CCGT=600.0))

    pool, up = _pool_and_up(dataset, draw=0)
    expected = row_availability(up, pool, pd.Series({row_name: 600.0}), [row_name])[:, 0]
    np.testing.assert_array_equal(_gen(designed).dynamic_capacity[row, :], expected)
    # weights [1, 1, 0.4] -> the denominator is 2.4, not 3.
    assert expected.min() < 1.0


# ---------------------------------------------------------------------------
# (c) rows outside the pool stay outage-free whatever the design
# ---------------------------------------------------------------------------


def test_vre_and_import_rows_stay_available(dataset):
    base = _load(dataset, outage_draw=0)
    designed = _load(
        dataset,
        outage_draw=0,
        design_capacity=_design(base, z1_solar=5000.0),
        design_id="solar",
    )
    for name in ("z1 solar", "z1_imports unspecified_imports", "z2 onwind"):
        row = _row(base, "Generator", name)
        np.testing.assert_array_equal(
            _gen(designed).dynamic_capacity[row, :], _gen(base).dynamic_capacity[row, :]
        )
    solar = _row(base, "Generator", "z1 solar")
    assert _gen(designed).nominal_capacity.ravel()[solar] == pytest.approx(5000.0)


# ---------------------------------------------------------------------------
# (d) common random numbers across designs
# ---------------------------------------------------------------------------


def test_common_random_numbers_across_designs(dataset):
    base = _load(dataset, outage_draw=0)
    a = _load(
        dataset,
        outage_draw=0,
        design_capacity=_design(base, z1_CCGT=750.0, z2_old_CCGT=1000.0),
        design_id="A",
    )
    b = _load(
        dataset,
        outage_draw=0,
        design_capacity=_design(base, z1_CCGT=500.0, z2_old_CCGT=1000.0),
        design_id="B",
    )

    shared = _row(base, "Generator", "z2 old CCGT")
    differing = _row(base, "Generator", "z1 CCGT")
    # The row both designs build to 1,000 MW is bit-identical.
    np.testing.assert_array_equal(
        _gen(a).dynamic_capacity[shared, :], _gen(b).dynamic_capacity[shared, :]
    )
    assert not np.array_equal(
        _gen(a).dynamic_capacity[differing, :], _gen(b).dynamic_capacity[differing, :]
    )

    # The unit realisations are shared, not just the row means: design B's two
    # units are design A's first two.
    pool, up = _pool_and_up(dataset, draw=0)
    offset = pool.row_offset["z1 CCGT"]
    np.testing.assert_array_equal(
        _gen(b).dynamic_capacity[differing, :], up[offset : offset + 2, :].mean(axis=0)
    )
    np.testing.assert_array_equal(
        _gen(a).dynamic_capacity[differing, :], up[offset : offset + 3, :].mean(axis=0)
    )


def test_different_draws_do_not_share_availability(dataset):
    base = _load(dataset, outage_draw=0)
    capacities = _design(base, z2_old_CCGT=1000.0)
    d0 = _load(dataset, outage_draw=0, design_capacity=capacities)
    d1 = _load(dataset, outage_draw=1, design_capacity=capacities)
    row = _row(base, "Generator", "z2 old CCGT")
    assert not np.array_equal(_gen(d0).dynamic_capacity[row, :], _gen(d1).dynamic_capacity[row, :])


# ---------------------------------------------------------------------------
# (e) a design that outgrows the pool fails loudly
# ---------------------------------------------------------------------------


def test_design_beyond_the_pool_raises(dataset):
    base = _load(dataset, outage_draw=0)
    # 6 slots x 250 MW = 1,500 MW of pool for this row.
    _load(dataset, outage_draw=0, design_capacity=_design(base, z1_CCGT=1500.0))
    with pytest.raises(ValueError, match="outage pool"):
        _load(
            dataset,
            outage_draw=0,
            design_capacity=_design(base, z1_CCGT=1500.1),
            design_id="too_big",
        )


def test_design_beyond_the_storage_pool_raises(dataset):
    base = _load(dataset, outage_draw=0)
    with pytest.raises(ValueError, match="outage pool"):
        _load(
            dataset,
            outage_draw=0,
            design_capacity=_design(base, z1_battery=10_000.0),
            design_id="too_big",
        )


# ---------------------------------------------------------------------------
# (f) no design, and the identity design, are the old behaviour exactly
# ---------------------------------------------------------------------------


def test_no_design_is_unchanged(dataset):
    """Regression: the as-built numbers with no design are what they always were."""
    gens = read_static(dataset)["generators"]
    system = _load(dataset, outage_draw=0)

    expected = gens["p_nom"].to_numpy(dtype=float).copy()
    expected[list(gens.index).index("z2 old CCGT")] = 0.0  # retired by 2040
    np.testing.assert_allclose(_gen(system).nominal_capacity.ravel(), expected, rtol=1e-12)

    assert system.meta["design_capacity_applied"] is False
    assert system.meta["design_capacity_digest"] is None
    assert system.meta["design_id"] is None
    assert system.meta["design_capacity_summary"] == {}

    # The as-built availability of the two pooled generator rows still comes from
    # their as-built unit counts (1 unit for z1 CCGT, none for the retired row).
    from zap.reliability.outages import row_availability

    pool, up = _pool_and_up(dataset, draw=0)
    for name, capacity in (("z1 CCGT", 250.0), ("z2 old CCGT", 0.0)):
        row = _row(system, "Generator", name)
        expected_row = row_availability(up, pool, pd.Series({name: capacity}), [name])[:, 0]
        np.testing.assert_array_equal(_gen(system).dynamic_capacity[row, :], expected_row)


def test_identity_design_is_element_wise_identical(dataset):
    """Spec check 2: a design equal to as-built perturbs nothing, draw included."""
    base = _load(dataset, outage_draw=0)
    same = _load(dataset, outage_draw=0, design_capacity=_as_built(base), design_id="asbuilt")

    for a, b in zip(base.devices, same.devices, strict=True):
        assert type(a) is type(b)
        for attr in (
            "nominal_capacity",
            "power_capacity",
            "dynamic_capacity",
            "power_availability",
            "min_nominal_capacity",
            "max_nominal_capacity",
            "min_power_capacity",
            "max_power_capacity",
            "linear_cost",
            "capital_cost",
            "duration",
            "load",
        ):
            left, right = getattr(a, attr, None), getattr(b, attr, None)
            if left is None:
                assert right is None
                continue
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right), err_msg=attr)

    assert same.meta["design_capacity_applied"] is True
    assert same.meta["design_id"] == "asbuilt"
    assert same.meta["design_capacity_digest"] == design_capacity_digest(_as_built(base))
    assert same.meta["design_capacity_summary"]["Generator"]["capacity_added_mw"] == 0.0


def test_demand_scaling_does_not_move_with_the_design(dataset):
    """The peak-available denominator is a scenario property, not a design one."""
    base = _load(dataset, demand_scaling="peak_fraction")
    big = _load(
        dataset,
        demand_scaling="peak_fraction",
        design_capacity=_design(base, z2_old_CCGT=1000.0),
    )
    assert big.meta["peaks_at"] == "as_built"
    assert big.meta["peak_available_mw"] == pytest.approx(base.meta["peak_available_mw"])
    assert big.meta["applied_scale"] == pytest.approx(base.meta["applied_scale"])
    np.testing.assert_array_equal(
        base.index.get(base.devices, "Load").load,
        big.index.get(big.devices, "Load").load,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_device_class_raises(dataset):
    with pytest.raises(ValueError, match="unknown device classes"):
        _load(dataset, design_capacity={"Widget": [1.0]})
    with pytest.raises(ValueError, match="unknown device classes"):
        _load(dataset, design_capacity={"ExportSink": [1.0]})


def test_wrong_row_count_raises(dataset):
    base = _load(dataset)
    capacities = _as_built(base)
    capacities["Generator"] = capacities["Generator"][:-1]
    with pytest.raises(ValueError, match="values but"):
        _load(dataset, design_capacity=capacities)


def test_negative_and_non_finite_capacity_raise(dataset):
    base = _load(dataset)
    with pytest.raises(ValueError, match="negative capacity"):
        _load(dataset, design_capacity=_design(base, z1_CCGT=-1.0))
    with pytest.raises(ValueError, match="non-finite"):
        _load(dataset, design_capacity=_design(base, z1_CCGT=float("nan")))


def test_apply_design_capacity_keeps_row_order_and_other_tables(dataset):
    static = read_static(dataset)
    designed, summary = apply_design_capacity(
        static, {"Generator": np.full(len(static["generators"]), 7.0)}
    )
    assert list(designed["generators"].index) == list(static["generators"].index)
    np.testing.assert_array_equal(designed["generators"]["p_nom"].to_numpy(), 7.0)
    # The input table is untouched (copies, not in-place writes) ...
    assert static["generators"]["p_nom"].to_numpy()[0] != 7.0
    # ... and untouched tables are shared, not copied.
    assert designed["storage_units"] is static["storage_units"]
    assert summary["classes"]["Generator"]["n_rows"] == len(static["generators"])
    assert summary["digest"] is not None


def test_design_capacity_digest_is_order_independent_and_value_sensitive():
    a = {"Generator": [1.0, 2.0], "StorageUnit": [3.0]}
    b = {"StorageUnit": [3.0], "Generator": [1.0, 2.0]}
    assert design_capacity_digest(a) == design_capacity_digest(b)
    assert design_capacity_digest({"Generator": [1.0, 2.5]}) != design_capacity_digest(a)
    assert design_capacity_digest(None) is None
    assert design_capacity_digest({}) is None


def test_ucap_with_a_design_is_allowed(dataset):
    """UCAP is a per-row mean read from ``ucap.csv``: capacity-independent."""
    write_tiny_ucap_csv(dataset, generator_factors={"z1 CCGT": 0.9, "z2 old CCGT": 0.9})
    base = _load(dataset, ucap_derate=True)
    designed = _load(dataset, ucap_derate=True, design_capacity=_design(base, z2_old_CCGT=1000.0))
    row = _row(base, "Generator", "z2 old CCGT")
    assert designed.index.get(designed.devices, "Generator").nominal_capacity.ravel()[
        row
    ] == pytest.approx(1000.0)
    np.testing.assert_allclose(_gen(designed).dynamic_capacity[row, :], 0.9, rtol=1e-12)


def test_ucap_and_outage_draw_still_exclusive_with_a_design(dataset):
    base = _load(dataset)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _load(
            dataset,
            ucap_derate=True,
            outage_draw=0,
            design_capacity=_as_built(base),
        )


def test_line_capacity_can_be_designed(dataset):
    base = _load(dataset)
    line = base.index.get(base.devices, "DirectedLine")
    capacities = _as_built(base)
    capacities["DirectedLine"] = capacities["DirectedLine"] * 2.0
    designed = _load(dataset, design_capacity=capacities)
    np.testing.assert_allclose(
        designed.index.get(designed.devices, "DirectedLine").nominal_capacity.ravel(),
        np.asarray(line.nominal_capacity).reshape(-1) * 2.0,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# The ``experiments/ra`` seam: design.json -> design_capacity -> build_system
# ---------------------------------------------------------------------------


def _cfg(dataset: Path) -> dict:
    return {
        "dataset": {
            "dir": str(dataset),
            "years": list(YEARS),
            "window": {"start": 0, "stop": N_HOURS},
        },
        "system": {
            "voll": 10_000.0,
            "demand_scaling": "none",
            "scale_load": 1.0,
            "peak_capacity_fraction": 0.85,
            "clip_scale_to_one": True,
            "link_losses": True,
            "export_mode": "drop",
            "carbon_tax": 0.0,
            "storage_init_soc": 0.5,
            "storage_final_soc": 0.5,
            "storage_soc_mode": "cyclic_free",
            "power_unit": 1.0,
            "cost_unit": 1.0,
        },
        "heuristics": {"ucap_derate": False},
    }


def _write_design_json(path: Path, system, **overrides) -> Path:
    capacities = _design(system, **overrides)
    record = {
        "schema_version": 1,
        "design_id": path.stem,
        "capacities": {
            "Generator": {
                "nominal_capacity": capacities["Generator"].tolist(),
                "names": [str(n) for n in system.index.names["Generator"]],
            },
            "StorageUnit": {
                "power_capacity": capacities["StorageUnit"].tolist(),
                "names": [str(n) for n in system.index.names["StorageUnit"]],
            },
        },
    }
    path.write_text(json.dumps(record))
    return path


def test_read_design_capacity_round_trip(dataset, tmp_path):
    from experiments.ra import system as system_mod

    base = _load(dataset, outage_draw=0)
    path = _write_design_json(tmp_path / "expand.json", base, z2_old_CCGT=1000.0)

    capacities = system_mod.read_design_capacity(path, dataset)
    assert set(capacities) == {"Generator", "StorageUnit"}
    row = _row(base, "Generator", "z2 old CCGT")
    assert capacities["Generator"][row] == pytest.approx(1000.0)

    designed = _load(dataset, outage_draw=0, design_capacity=capacities, design_id="expand")
    assert _gen(designed).dynamic_capacity[row, :].min() < 1.0


def test_read_design_capacity_rejects_a_foreign_dataset(dataset, tmp_path):
    from experiments.ra import system as system_mod

    base = _load(dataset)
    path = tmp_path / "foreign.json"
    record = json.loads(_write_design_json(path, base).read_text())
    record["capacities"]["Generator"]["names"][0] = "somewhere else"
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="different dataset"):
        system_mod.read_design_capacity(path, dataset)


def test_build_system_is_design_aware(dataset):
    from experiments.ra import system as system_mod

    system_mod.clear_system_cache()
    cfg = _cfg(dataset)
    base = system_mod.build_system(cfg, draw=0)
    row = _row(base, "Generator", "z2 old CCGT")

    design = system_mod.Design(design_id="expand", capacities=_design(base, z2_old_CCGT=1000.0))
    designed = system_mod.build_system(cfg, draw=0, design=design)

    assert designed is not base, "the design must not be served from the no-design cache entry"
    assert designed.meta["design_id"] == "expand"
    assert _gen(designed).nominal_capacity.ravel()[row] == pytest.approx(1000.0)
    assert _gen(designed).dynamic_capacity[row, :].min() < 1.0

    # The cache key separates designs, and repeats are served from it.
    again = system_mod.build_system(cfg, draw=0, design=design)
    assert again is designed
    keys = {
        system_mod.system_key(cfg, 0, None, None),
        system_mod.system_key(cfg, 0, "expand", design_capacity_digest(design.capacities)),
    }
    assert len(keys) == 2
    system_mod.clear_system_cache()


def test_design_apply_still_works_for_the_planning_path(dataset):
    from experiments.ra import system as system_mod

    base = _load(dataset)
    row = _row(base, "Generator", "z1 CCGT")
    design = system_mod.Design(design_id="d", capacities=_design(base, z1_CCGT=750.0))
    devices = design.apply(base)
    assert devices[base.index.device_index["Generator"]].nominal_capacity.ravel()[
        row
    ] == pytest.approx(750.0)
    # ... and it leaves the loaded system alone.
    assert _gen(base).nominal_capacity.ravel()[row] == pytest.approx(250.0)


def test_design_capacity_map_rejects_an_underivable_class(dataset):
    from experiments.ra import system as system_mod

    design = system_mod.Design(design_id="d", capacities={"ExportSink": [1.0]})
    with pytest.raises(ValueError, match="cannot impose"):
        system_mod.design_capacity_map(design)
    assert system_mod.design_capacity_map(system_mod.Design()) is None


# ---------------------------------------------------------------------------
# Integration: the real z4 store (39,225 pool units)
# ---------------------------------------------------------------------------


def test_real_z4_design_draws_from_the_pool():
    root = real_z4_dir()
    if root is None or not (root / "outages.zarr").exists():
        pytest.skip("data/ca2040_z4 with outages.zarr is not present")

    options = {"years": (2020,), "window": HourWindow(0, 24), "export_mode": "drop"}
    base = load_system(root, LoadOptions(outage_draw=0, **options))
    gen = base.index.get(base.devices, "Generator")
    capacity = np.asarray(gen.nominal_capacity, dtype=float).reshape(-1)

    from zap.reliability.outages import build_unit_pool, load_outage_params

    pool = build_unit_pool(root, load_outage_params())
    names = [str(n) for n in base.index.names["Generator"]]
    zero_pooled = [i for i, n in enumerate(names) if capacity[i] == 0.0 and n in pool.row_offset]
    if not zero_pooled:
        pytest.skip("no zero-capacity pooled generator row in ca2040_z4")
    row = zero_pooled[0]
    # As-built the row has no capacity, so the outage lookup leaves it at 1.0 and
    # `dynamic_capacity` is the bare weather profile.
    weather = np.asarray(gen.dynamic_capacity[row, :], dtype=float)

    designed_capacity = capacity.copy()
    designed_capacity[row] = 1000.0
    designed = load_system(
        root,
        LoadOptions(
            outage_draw=0,
            design_capacity={"Generator": designed_capacity},
            design_id="z4_integration",
            **options,
        ),
    )
    designed_gen = designed.index.get(designed.devices, "Generator")
    assert designed_gen.nominal_capacity.ravel()[row] == pytest.approx(1000.0)
    assert designed.meta["design_capacity_summary"]["Generator"]["n_rows_built_from_zero"] == 1

    from zap.reliability.outages import row_availability

    expected = row_availability(
        _real_up(root, draw=0, hours=24), pool, pd.Series({names[row]: 1000.0}), [names[row]]
    )[:, 0]
    np.testing.assert_allclose(
        designed_gen.dynamic_capacity[row, :], weather * expected, rtol=1e-12
    )
    assert expected.max() <= 1.0


def _real_up(root: Path, *, draw: int, hours: int) -> np.ndarray:
    from zap.importers.wy_store import _open_outage_store

    _, store = _open_outage_store(root)
    years = [int(y) for y in np.asarray(store["weather_year"][:])]
    draws = [int(d) for d in np.asarray(store["draw"][:])]
    return np.asarray(
        store["available"][years.index(2020), draws.index(draw), :hours, :], dtype=np.uint8
    ).T


def test_design_capacity_tables_cover_the_designable_classes():
    assert DESIGN_CAPACITY_TABLES == {
        "Generator": "generators",
        "StorageUnit": "storage_units",
        "DirectedLine": "links",
    }
