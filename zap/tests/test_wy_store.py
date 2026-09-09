"""Tests for the weather store converter and reader (WP1 of the Phase-1 spec)."""

from __future__ import annotations

import unittest
from pathlib import Path

import cvxpy as cp
import numcodecs
import numpy as np
import pandas as pd
import pytest
import zarr

from zap.devices.injector import Generator, Injector, Load
from zap.devices.storage_unit import StorageUnit
from zap.devices.transporter import DirectedLine
from zap.importers.wy_store import (
    CONVERTER_VERSION,
    HourWindow,
    LoadOptions,
    WeatherStore,
    available_capacity,
    chunk_hours_for,
    convert_dataset,
    detect_model_year,
    import_bus_mask,
    load_system,
    peak_available_mw,
    retired_mask,
)
from zap.planning.operation_objectives import UnservedEnergyObjective
from zap.tests.fixtures.tiny_dataset import (
    EXPORT_MARGINAL_COST,
    TINY_GENERATORS,
    TINY_LINKS,
    TINY_MODEL_YEAR,
    TINY_RETIRED_GENERATORS,
    real_z4_dir,
    write_tiny_dataset,
    write_tiny_ucap_csv,
)

N_HOURS = 48
YEARS = (2020, 2021)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path) -> Path:
    """A converted tiny dataset: static CSVs, parquet timeseries, weather.zarr."""
    root = write_tiny_dataset(tmp_path / "tiny", n_hours=N_HOURS, years=YEARS)
    convert_dataset(root)
    return root


def _static(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / "static" / f"{name}.csv", index_col=0)


def _parquet(root: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(root / "timeseries" / f"{name}.parquet")


# ---------------------------------------------------------------------------
# 1-3: converter
# ---------------------------------------------------------------------------


def test_convert_writes_expected_schema(dataset):
    root = zarr.open_group(str(dataset / "weather.zarr"), mode="r")
    attrs = dict(root.attrs)

    expected_arrays = ["generators_p_max_pu", "links_marginal_cost", "loads_p_set"]
    assert attrs["arrays"] == expected_arrays
    assert attrs["converter_version"] == CONVERTER_VERSION
    assert attrs["dataset"] == "tiny"
    assert attrs["weather_years"] == list(YEARS)
    assert attrs["hours_per_year"] == N_HOURS
    assert attrs["chunk_hours"] == 168
    assert attrs["skipped_years"] == {}
    for key in ("created_utc", "zap_commit", "source_files"):
        assert key in attrs
    assert set(attrs["source_files"]) == {
        "timeseries/generators_t_p_max_pu.parquet",
        "timeseries/links_t_marginal_cost.parquet",
        "timeseries/loads_t_p_set.parquet",
    }
    for value in attrs["source_files"].values():
        assert len(value) == 64

    for name in expected_arrays:
        source = _parquet(dataset, name.replace("_", "_t_", 1))
        n_cols = source.shape[1]
        arr = root[name]
        assert arr.shape == (len(YEARS), N_HOURS, n_cols)
        assert arr.dtype == np.dtype("float32")
        assert arr.chunks == (1, chunk_hours_for(n_cols, 168, N_HOURS), n_cols)
        assert arr.compressor.cname == "zstd"
        assert arr.compressor.shuffle == numcodecs.Blosc.SHUFFLE
        assert np.isnan(arr.fill_value)
        assert arr.attrs["source_file"].endswith(".parquet")
        assert arr.attrs["columns_array"] == f"{name}__columns"
        columns = [str(c) for c in root[f"{name}__columns"][:]]
        assert columns == list(source.columns)

    assert np.array_equal(root["weather_year"][:], np.array(YEARS, dtype=np.int32))
    assert np.array_equal(root["hour"][:], np.arange(N_HOURS, dtype=np.int32))
    assert len(root["timestep_iso"][:]) == N_HOURS
    assert str(root["timestep_iso"][0]).startswith("2040-01-01T00:00")


def test_convert_values_match_parquet(dataset):
    store = WeatherStore.open(dataset)
    rng = np.random.default_rng(0)

    for name in store.arrays:
        source = _parquet(dataset, name.replace("_", "_t_", 1))
        columns = store.columns(name)
        assert list(columns) == list(source.columns)

        for year in YEARS:
            expected = source.loc[year].to_numpy(dtype=np.float64)
            actual = store.read_year(name, year)
            assert actual.shape == expected.shape

            hours = rng.integers(0, N_HOURS, size=200)
            cols = rng.integers(0, len(columns), size=200)
            np.testing.assert_array_equal(
                actual[hours, cols].astype(np.float32),
                expected[hours, cols].astype(np.float32),
            )


def test_convert_refuses_overwrite(dataset):
    with pytest.raises(FileExistsError):
        convert_dataset(dataset)
    out = convert_dataset(dataset, overwrite=True)
    assert out == dataset / "weather.zarr"
    assert WeatherStore.open(dataset).hours_per_year == N_HOURS


# ---------------------------------------------------------------------------
# 4-7: device mapping
# ---------------------------------------------------------------------------


def _load(dataset: Path, **kwargs):
    options = LoadOptions(
        years=kwargs.pop("years", (2020,)),
        window=kwargs.pop("window", HourWindow(0, 24)),
        **kwargs,
    )
    return load_system(dataset, options)


def test_load_system_device_order_and_shapes(dataset):
    system = _load(dataset)

    assert list(system.index.device_index) == [
        "Generator",
        "Load",
        "DirectedLine",
        "StorageUnit",
        "ExportSink",
    ]
    assert list(system.index.device_index.values()) == [0, 1, 2, 3, 4]
    classes = [Generator, Load, DirectedLine, StorageUnit, Injector]
    for device, cls in zip(system.devices, classes):
        assert isinstance(device, cls)

    generator = system.index.get(system.devices, "Generator")
    load = system.index.get(system.devices, "Load")
    assert generator.dynamic_capacity.shape == (len(TINY_GENERATORS), 24)
    assert load.load.shape == (2, 24)
    for device in system.devices:
        assert device.time_horizon in (0, 24)


def test_generator_fields_match_static(dataset):
    system = _load(dataset)
    generator = system.index.get(system.devices, "Generator")
    gens = _static(dataset, "generators")
    carriers = _static(dataset, "carriers")

    active_p_nom = np.where(retired_mask(gens, TINY_MODEL_YEAR), 0.0, gens["p_nom"].to_numpy())
    np.testing.assert_allclose(generator.nominal_capacity.ravel(), active_p_nom, rtol=1e-12)
    np.testing.assert_allclose(
        generator.linear_cost.ravel(), gens["marginal_cost"].to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(
        generator.capital_cost.ravel(), gens["capital_cost"].to_numpy(), rtol=1e-12
    )
    expected_rates = (
        carriers.loc[gens["carrier"], "co2_emissions"].to_numpy() / gens["efficiency"].to_numpy()
    )
    np.testing.assert_allclose(generator.emission_rates.ravel(), expected_rates, rtol=1e-12)
    np.testing.assert_allclose(generator.min_nominal_capacity.ravel(), active_p_nom, rtol=1e-12)
    np.testing.assert_allclose(generator.max_nominal_capacity.ravel(), active_p_nom, rtol=1e-12)
    assert list(generator.name) == list(gens.index)
    assert list(generator.fuel_type) == list(gens["carrier"])

    # A generator with no column in the parquet uses its static p_max_pu.
    fallback = [g[0] for g in TINY_GENERATORS if not g[8]]
    assert fallback == ["z1_imports unspecified_imports"]
    row = list(gens.index).index(fallback[0])
    np.testing.assert_allclose(
        generator.dynamic_capacity[row, :], gens["p_max_pu"].iloc[row], rtol=1e-12
    )

    # A generator with a column matches the parquet slice.
    source = _parquet(dataset, "generators_t_p_max_pu").loc[2020]
    row = list(gens.index).index("z2 onwind")
    np.testing.assert_allclose(
        generator.dynamic_capacity[row, :],
        source["z2 onwind"].to_numpy()[:24].astype(np.float32),
        rtol=1e-6,
    )


def test_directed_link_fields(dataset):
    system = _load(dataset)
    line = system.index.get(system.devices, "DirectedLine")
    links = _static(dataset, "links")
    eff = links["efficiency"].to_numpy()

    np.testing.assert_allclose(line.efficiency.ravel(), eff, rtol=1e-12)
    np.testing.assert_allclose(line.min_power.ravel(), np.zeros(len(links)), atol=0)
    np.testing.assert_allclose(
        line.max_power.ravel(), links["p_max_pu"].to_numpy() * eff, rtol=1e-12
    )
    np.testing.assert_allclose(line.nominal_capacity.ravel(), links["p_nom"].to_numpy(), rtol=1e-12)

    export_row = [link[0] for link in TINY_LINKS if link[7]][0]
    idx = list(links.index).index(export_row)
    np.testing.assert_allclose(
        line.linear_cost[idx, :],
        np.float32(EXPORT_MARGINAL_COST) / eff[idx],
        rtol=1e-6,
    )
    assert line.linear_cost[idx, 0] < 0

    for name in links.index:
        if name == export_row:
            continue
        j = list(links.index).index(name)
        np.testing.assert_allclose(
            line.linear_cost[j, :], links["marginal_cost"].iloc[j] / eff[j], rtol=1e-12
        )

    # link_losses=False sets every efficiency to one.
    lossless = _load(dataset, link_losses=False)
    lossless_line = lossless.index.get(lossless.devices, "DirectedLine")
    np.testing.assert_allclose(lossless_line.efficiency.ravel(), np.ones(len(links)))
    np.testing.assert_allclose(
        lossless_line.max_power.ravel(), links["p_max_pu"].to_numpy(), rtol=1e-12
    )


def test_export_sink_capacity(dataset):
    system = _load(dataset)
    sink = system.index.get(system.devices, "ExportSink")
    links = _static(dataset, "links")
    stores = _static(dataset, "stores")

    expected = links.groupby("bus1")["p_nom"].sum().reindex(stores["bus"]).to_numpy()
    np.testing.assert_allclose(sink.nominal_capacity.ravel(), expected, rtol=1e-12)
    np.testing.assert_allclose(sink.min_power, -np.ones((1, 1)))
    np.testing.assert_allclose(sink.max_power, np.zeros((1, 1)))
    np.testing.assert_allclose(sink.linear_cost, np.zeros((1, 1)))
    # Static bounds -> a static device, so blocked solves can slice it freely.
    assert sink.time_horizon == 0
    assert sink.sample_time(range(0, 12), 24).time_horizon == 0

    dropped = _load(dataset, export_mode="drop")
    assert "ExportSink" not in dropped.index.device_index


# ---------------------------------------------------------------------------
# 8: VOLL load shedding
# ---------------------------------------------------------------------------


def test_voll_load_shedding(dataset):
    # export_mode="drop" removes the (negative-cost) export revenue term so the
    # objective is exactly generation cost + VOLL * ENS.
    system = _load(dataset, window=HourWindow(0, 2), export_mode="drop", voll=9_000.0)
    generator = system.index.get(system.devices, "Generator")
    generator.nominal_capacity = generator.nominal_capacity * 0.01

    outcome = system.network.dispatch(system.devices, solver=cp.HIGHS)
    assert outcome.problem.status == cp.OPTIMAL

    ens = float(UnservedEnergyObjective(system.devices)(outcome))
    assert ens > 0.0

    generation_cost = float(
        generator.operation_cost(
            outcome.power[0], outcome.angle[0], outcome.local_variables[0], la=np
        )
    )
    np.testing.assert_allclose(outcome.problem.value, generation_cost + 9_000.0 * ens, rtol=1e-6)


# ---------------------------------------------------------------------------
# 9: demand scaling
# ---------------------------------------------------------------------------


def test_demand_scaling_clips(dataset):
    baseline = _load(dataset)
    unscaled = baseline.index.get(baseline.devices, "Load").load.copy()

    generous = _load(dataset, demand_scaling="peak_fraction", peak_capacity_fraction=0.85)
    assert generous.meta["implied_scale"] > 1.0
    assert generous.meta["applied_scale"] == 1.0
    np.testing.assert_allclose(
        generous.index.get(generous.devices, "Load").load, unscaled, rtol=1e-12
    )

    tight = _load(dataset, demand_scaling="peak_fraction", peak_capacity_fraction=0.2)
    expected = 0.2 * tight.meta["peak_available_mw"] / tight.meta["peak_load_mw"]
    assert tight.meta["applied_scale"] == pytest.approx(expected)
    np.testing.assert_allclose(
        tight.index.get(tight.devices, "Load").load, unscaled * expected, rtol=1e-12
    )

    # Without the clip the implied factor is used verbatim.
    unclipped = _load(
        dataset,
        demand_scaling="peak_fraction",
        peak_capacity_fraction=0.85,
        clip_scale_to_one=False,
    )
    assert unclipped.meta["applied_scale"] == pytest.approx(unclipped.meta["implied_scale"])

    fixed = _load(dataset, demand_scaling="fixed", scale_load=0.5)
    np.testing.assert_allclose(
        fixed.index.get(fixed.devices, "Load").load, unscaled * 0.5, rtol=1e-12
    )


def test_peak_available_excludes_storage_and_imports(dataset):
    system = _load(dataset)
    meta = system.meta
    gens = _static(dataset, "generators")
    storage = _static(dataset, "storage_units")

    assert meta["peak_available_mw"] < meta["peak_available_incl_storage_mw"]
    assert meta["peak_available_mw"] < meta["peak_available_incl_imports_mw"]
    assert meta["peak_available_incl_storage_mw"] == pytest.approx(
        meta["peak_available_mw"] + storage["p_nom"].sum()
    )

    mask = import_bus_mask(gens["bus"].to_numpy())
    np.testing.assert_array_equal(system.index.import_mask, mask)
    assert mask.sum() == 1

    # available_capacity over the loaded window agrees with a direct computation.
    generator = system.index.get(system.devices, "Generator")
    hourly = available_capacity(system)
    direct = (
        np.asarray(generator.dynamic_capacity)[~mask, :]
        * np.asarray(generator.nominal_capacity)[~mask, :]
    ).sum(axis=0)
    np.testing.assert_allclose(hourly, direct, rtol=1e-12)
    assert peak_available_mw(system) == pytest.approx(direct.max())

    per_bus = available_capacity(system, per_bus=True)
    np.testing.assert_allclose(per_bus.sum(axis=1).to_numpy(), hourly, rtol=1e-12)
    assert "z1_imports" not in per_bus.columns
    assert "z1_imports" in available_capacity(system, per_bus=True, include_imports=True)


# ---------------------------------------------------------------------------
# 10-11: UCAP
# ---------------------------------------------------------------------------


def test_ucap_derate_applies(dataset):
    write_tiny_ucap_csv(
        dataset,
        generator_factors={"z1 CCGT": 0.9, "z2 hydro": 0.8},
        storage_factors={"z1 battery": 0.75},
    )
    base = _load(dataset)
    derated = _load(dataset, ucap_derate=True)

    gens = _static(dataset, "generators")
    factors = {"z1 CCGT": 0.9, "z2 hydro": 0.8}
    base_gen = base.index.get(base.devices, "Generator")
    derated_gen = derated.index.get(derated.devices, "Generator")
    for i, name in enumerate(gens.index):
        factor = factors.get(name, 1.0)
        np.testing.assert_allclose(
            derated_gen.dynamic_capacity[i, :],
            base_gen.dynamic_capacity[i, :] * factor,
            rtol=1e-12,
        )

    storage = derated.index.get(derated.devices, "StorageUnit")
    units = _static(dataset, "storage_units")
    expected = np.array([0.75 if n == "z1 battery" else 1.0 for n in units.index])
    np.testing.assert_allclose(storage.power_availability.ravel(), expected, rtol=1e-12)

    assert derated.meta["ucap_derate"] is True
    assert (
        derated.meta["ucap_csv_sha256"] is not None and len(derated.meta["ucap_csv_sha256"]) == 64
    )


def test_ucap_missing_csv_raises(dataset):
    with pytest.raises(FileNotFoundError):
        _load(dataset, ucap_derate=True)


def test_ucap_and_outage_draw_are_exclusive(dataset):
    with pytest.raises(ValueError):
        _load(dataset, ucap_derate=True, outage_draw=0)


def test_ignore_min_power_false_raises(dataset):
    with pytest.raises(NotImplementedError):
        _load(dataset, ignore_min_power=False)


def test_missing_outage_store_raises(dataset):
    with pytest.raises(FileNotFoundError):
        _load(dataset, outage_draw=0)


# ---------------------------------------------------------------------------
# Outage draws (join with WP2)
# ---------------------------------------------------------------------------


def _write_tiny_outage_store(root: Path) -> Path:
    """Hand-written ``outages.zarr`` in WP2's schema (spec 3.4).

    Two pooled rows -- one generator and one storage unit -- with three slots
    each, so the availability weighting has both an exact and a remainder case.
    """
    unit_size = 50.0
    rows = [("Generator", "z1 CCGT", "CCGT", "z1"), ("StorageUnit", "z1 battery", "battery", "z1")]
    n_slots = 3

    unit_id, unit_row, unit_carrier, unit_bus, unit_component = [], [], [], [], []
    unit_slot, unit_offset, unit_units = [], [], []
    for i, (component, row, carrier, bus) in enumerate(rows):
        for slot in range(n_slots):
            unit_id.append(f"{row}#{slot}")
            unit_row.append(row)
            unit_carrier.append(carrier)
            unit_bus.append(bus)
            unit_component.append(component)
            unit_slot.append(slot)
            unit_offset.append(i * n_slots)
            unit_units.append(n_slots)

    n_units = len(unit_id)
    available = np.ones((1, 1, N_HOURS, n_units), dtype=np.uint8)
    # z1 CCGT (p_nom 100, unit size 50 -> 2 units of weight 1): drop slot 0.
    available[0, 0, 5:11, 0] = 0
    # z1 battery (p_nom 30, unit size 50 -> 1 unit of weight 0.6): drop slot 0.
    available[0, 0, 20:23, 3] = 0

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

    _int("weather_year", [2020])
    _int("draw", [0])
    _int("hour", np.arange(N_HOURS))
    _int("unit_slot", unit_slot)
    _int("unit_row_offset", unit_offset)
    _int("unit_row_units", unit_units)
    _int("unit_size_mw", np.full(n_units, unit_size), dtype="float32")
    _str("unit_id", unit_id)
    _str("unit_row", unit_row)
    _str("unit_carrier", unit_carrier)
    _str("unit_bus", unit_bus)
    _str("unit_component", unit_component)

    group.attrs.update(
        {
            "generator_version": 1,
            "dataset": Path(root).name,
            "base_seed": 20260908,
            "weather_years": [2020],
            "n_draws": 1,
            "hours_per_year": N_HOURS,
        }
    )
    return path


def test_outage_draw_applies(dataset):
    pytest.importorskip("zap.reliability.outages")
    _write_tiny_outage_store(dataset)

    base = _load(dataset, window=HourWindow(0, N_HOURS))
    drawn = _load(dataset, window=HourWindow(0, N_HOURS), outage_draw=0)

    gens = _static(dataset, "generators")
    row = list(gens.index).index("z1 CCGT")
    base_gen = base.index.get(base.devices, "Generator")
    drawn_gen = drawn.index.get(drawn.devices, "Generator")

    # p_nom 100 with 50 MW units -> two weight-1 units; one of them is out.
    expected = np.ones(N_HOURS)
    expected[5:11] = 0.5
    np.testing.assert_allclose(
        drawn_gen.dynamic_capacity[row, :],
        base_gen.dynamic_capacity[row, :] * expected,
        rtol=1e-12,
    )

    # Rows outside the pool are untouched.
    for i, name in enumerate(gens.index):
        if name == "z1 CCGT":
            continue
        np.testing.assert_allclose(
            drawn_gen.dynamic_capacity[i, :], base_gen.dynamic_capacity[i, :], rtol=1e-12
        )

    units = _static(dataset, "storage_units")
    storage = drawn.index.get(drawn.devices, "StorageUnit")
    battery = list(units.index).index("z1 battery")
    expected_storage = np.ones(N_HOURS)
    expected_storage[20:23] = 0.0
    assert storage.power_availability.shape == (len(units), N_HOURS)
    np.testing.assert_allclose(storage.power_availability[battery, :], expected_storage, rtol=1e-12)
    np.testing.assert_allclose(
        storage.power_availability[1 - battery, :], np.ones(N_HOURS), rtol=1e-12
    )

    assert drawn.meta["outage_draw"] == 0
    assert drawn.meta["outage_store_attrs"]["base_seed"] == 20260908


# ---------------------------------------------------------------------------
# 12-13: dispatch and multi-year
# ---------------------------------------------------------------------------


def test_dispatch_48h_highs(dataset):
    system = _load(dataset, window=HourWindow(0, N_HOURS))
    outcome = system.network.dispatch(system.devices, solver=cp.HIGHS)

    assert outcome.problem.status == cp.OPTIMAL
    assert np.isfinite(outcome.problem.value)

    # Nodal power balance: the ground device is appended by dispatch, so the
    # residual is over the returned power lists.
    balance = np.zeros((system.network.num_nodes, N_HOURS))
    devices = system.devices + [outcome.ground]
    for device, power in zip(devices, outcome.power):
        terminals = np.asarray(device.terminals)
        if terminals.ndim == 1:
            terminals = terminals.reshape(-1, 1)
        for k, p in enumerate(power):
            np.add.at(balance, terminals[:, k], np.asarray(p))
    assert np.abs(balance).max() < 1e-6

    generation = outcome.power[0][0]
    assert generation.min() > -1e-8
    assert float(UnservedEnergyObjective(system.devices)(outcome)) < 1e-6


def test_two_years_concatenate(dataset):
    system = _load(dataset, years=YEARS, window=HourWindow(0, 24))
    generator = system.index.get(system.devices, "Generator")
    load = system.index.get(system.devices, "Load")

    assert system.meta["n_hours"] == 48
    assert generator.time_horizon == 48
    assert load.load.shape == (2, 48)

    source = _parquet(dataset, "loads_t_p_set")
    second = source.loc[YEARS[1]].to_numpy(dtype=np.float32)[:24].T
    np.testing.assert_allclose(load.load[:, 24:], second, rtol=1e-6)

    first = source.loc[YEARS[0]].to_numpy(dtype=np.float32)[:24].T
    np.testing.assert_allclose(load.load[:, :24], first, rtol=1e-6)


def test_reader_rejects_unknown_year_and_array(dataset):
    store = WeatherStore.open(dataset)
    with pytest.raises(KeyError):
        store.read_year("loads_p_set", 1999)
    with pytest.raises(KeyError):
        store.read_year("nope", 2020)
    with pytest.raises(ValueError):
        store.read("loads_p_set", 2020, slice(0, 10, 2))


def test_load_system_without_store_raises(tmp_path):
    root = write_tiny_dataset(tmp_path / "bare", n_hours=N_HOURS, years=YEARS)
    with pytest.raises(FileNotFoundError):
        load_system(root, LoadOptions(years=(2020,), window=HourWindow(0, 24)))


# ---------------------------------------------------------------------------
# Asset lifetimes (retirements)
# ---------------------------------------------------------------------------


def test_model_year_is_detected_from_the_store(dataset):
    year, source = detect_model_year(dataset)
    assert year == TINY_MODEL_YEAR
    assert source in ("weather_store_attrs", "weather_store:timestep_iso")

    # ... and without a converted store, from the parquet timestep index.
    bare = write_tiny_dataset(dataset.parent / "bare", n_hours=N_HOURS, years=YEARS)
    year, source = detect_model_year(bare)
    assert year == TINY_MODEL_YEAR
    assert source.startswith("timeseries/")


def test_retired_mask_rule():
    gens = pd.DataFrame(
        {
            "build_year": [1990, 2010, 2040, 1985, 2020, 1942],
            "lifetime": [30.0, 40.0, 30.0, np.inf, np.nan, 0.0],
        }
    )
    np.testing.assert_array_equal(
        retired_mask(gens, 2040), [True, False, False, False, False, False]
    )
    # 2010 + 40 == 2050 > 2040, but by 2050 it is retired (<= is inclusive).
    np.testing.assert_array_equal(retired_mask(gens, 2050)[:2], [True, True])
    # lifetime == 0 is a missing-data marker and means infinite: never retires.
    assert not retired_mask(gens, 3000)[5]
    # No model year, or no lifetime columns at all: nothing retires.
    assert not retired_mask(gens, None).any()
    assert not retired_mask(gens.drop(columns=["lifetime"]), 2040).any()


def test_lifetimes_zero_retired_capacity_and_keep_row_order(dataset):
    system = _load(dataset)
    gens = _static(dataset, "generators")
    generator = system.index.get(system.devices, "Generator")

    # Row order, names and count are untouched: outage-pool offsets, SystemIndex
    # and design.json all index by position.
    assert list(generator.name) == list(gens.index)
    assert generator.num_devices == len(TINY_GENERATORS)
    assert list(system.index.names["Generator"]) == list(gens.index)

    retired = TINY_RETIRED_GENERATORS[0]
    row = list(gens.index).index(retired)
    assert gens["p_nom"].iloc[row] > 0.0  # the CSV still carries the as-built MW
    assert generator.nominal_capacity.ravel()[row] == 0.0
    assert generator.min_nominal_capacity.ravel()[row] == 0.0
    assert generator.max_nominal_capacity.ravel()[row] == 0.0

    # The infinite-lifetime row is untouched, however old it is.
    inf_row = list(gens.index).index("z1 legacy CCGT")
    assert not np.isfinite(gens["lifetime"].iloc[inf_row])
    assert generator.nominal_capacity.ravel()[inf_row] == pytest.approx(gens["p_nom"].iloc[inf_row])

    # Every other row keeps its as-built capacity.
    expected = np.where(retired_mask(gens, TINY_MODEL_YEAR), 0.0, gens["p_nom"].to_numpy())
    np.testing.assert_allclose(generator.nominal_capacity.ravel(), expected, rtol=1e-12)

    # Storage carries lifetimes too, and none of it retires by 2040.
    units = _static(dataset, "storage_units")
    storage = system.index.get(system.devices, "StorageUnit")
    np.testing.assert_allclose(
        storage.power_capacity.ravel(), units["p_nom"].to_numpy(), rtol=1e-12
    )


def test_apply_lifetimes_false_restores_the_old_behaviour(dataset):
    system = _load(dataset, apply_lifetimes=False)
    gens = _static(dataset, "generators")
    generator = system.index.get(system.devices, "Generator")
    np.testing.assert_allclose(
        generator.nominal_capacity.ravel(), gens["p_nom"].to_numpy(), rtol=1e-12
    )
    assert system.meta["model_year"] is None
    assert system.meta["apply_lifetimes"] is False
    assert system.meta["retired_rows"] == {"Generator": 0, "StorageUnit": 0}

    # ... and an explicit model year before any retirement does the same.
    early = _load(dataset, model_year=1999)
    early_gen = early.index.get(early.devices, "Generator")
    np.testing.assert_allclose(
        early_gen.nominal_capacity.ravel(), gens["p_nom"].to_numpy(), rtol=1e-12
    )
    assert early.meta["model_year_source"] == "LoadOptions.model_year"


def test_retirement_meta_and_peak_capacity(dataset):
    system = _load(dataset)
    meta = system.meta
    gens = _static(dataset, "generators")
    retired = TINY_RETIRED_GENERATORS[0]
    retired_mw = float(gens.loc[retired, "p_nom"])

    assert meta["model_year"] == TINY_MODEL_YEAR
    assert meta["apply_lifetimes"] is True
    assert meta["retired_rows"] == {"Generator": 1, "StorageUnit": 0}
    assert meta["retired_capacity_mw"]["Generator"] == pytest.approx(retired_mw)
    assert meta["retired_capacity_mw"]["StorageUnit"] == 0.0
    assert meta["retired_capacity_mw_by_carrier"]["Generator"] == {
        gens.loc[retired, "carrier"]: pytest.approx(retired_mw)
    }
    assert meta["retired_capacity_mw_by_carrier"]["StorageUnit"] == {}
    assert meta["retired_names"]["Generator"] == TINY_RETIRED_GENERATORS

    # available_capacity() takes capacity from the system, so it follows.
    kept = _load(dataset, apply_lifetimes=False)
    assert meta["peak_available_mw"] == pytest.approx(kept.meta["peak_available_mw"] - retired_mw)
    assert peak_available_mw(system) == pytest.approx(peak_available_mw(kept) - retired_mw)


def test_retired_row_gets_the_minimum_outage_pool(dataset):
    """The pool keeps the retired row's slot, sized as if it were a 0 MW candidate."""
    ox = pytest.importorskip("zap.reliability.outages")
    params = ox.load_outage_params()

    pool = ox.build_unit_pool(dataset, params)
    retired = TINY_RETIRED_GENERATORS[0]
    assert retired in pool.row_offset
    assert pool.row_units[retired] == params.min_units_per_row

    # Without the lifetime rule the same row is sized on its 70 MW as-built.
    kept = ox.build_unit_pool(dataset, params, model_year=1999)
    assert kept.row_units[retired] > params.min_units_per_row

    # Offsets stay a running cumulative sum over the same rows in the same order.
    assert list(pool.row_offset) == list(kept.row_offset)
    offset = 0
    for row, n in pool.row_units.items():
        assert pool.row_offset[row] == offset
        offset += n
    assert offset == pool.n_units


# ---------------------------------------------------------------------------
# 14: the real dataset (opt-in)
# ---------------------------------------------------------------------------


@unittest.skipIf(real_z4_dir() is None, "data/ca2040_z4/weather.zarr is not present")
def test_real_z4_structural():
    root = real_z4_dir()
    system = load_system(
        root, LoadOptions(years=(2020,), window=HourWindow(0, 48), demand_scaling="none")
    )

    gens = pd.read_csv(root / "static" / "generators.csv", index_col=0)
    generator = system.index.get(system.devices, "Generator")
    assert list(generator.name) == list(gens.index)

    by_carrier = (
        pd.Series(generator.nominal_capacity.ravel(), index=gens["carrier"].to_numpy())
        .groupby(level=0)
        .sum()
    )
    active = gens.assign(p_nom=np.where(retired_mask(gens, 2040), 0.0, gens["p_nom"].to_numpy()))
    expected = active.groupby("carrier")["p_nom"].sum()
    pd.testing.assert_series_equal(
        by_carrier.sort_index(), expected.sort_index(), check_names=False, rtol=1e-12
    )

    load = system.index.get(system.devices, "Load")
    source = pd.read_parquet(root / "timeseries" / "loads_t_p_set.parquet").loc[2020]
    np.testing.assert_allclose(
        load.load.sum(), source.iloc[:48].to_numpy(dtype=np.float32).sum(), rtol=1e-6
    )

    # peak_available_mw recomputed from CSV + parquet over the full year.
    p_max_pu = pd.read_parquet(root / "timeseries" / "generators_t_p_max_pu.parquet").loc[2020]
    weather = np.empty((len(gens), 8760))
    for i, name in enumerate(gens.index):
        if name in p_max_pu.columns:
            weather[i, :] = p_max_pu[name].to_numpy(dtype=np.float32)
        else:
            weather[i, :] = gens["p_max_pu"].iloc[i]
    mask = import_bus_mask(gens["bus"].to_numpy())
    hourly = (weather[~mask, :] * active["p_nom"].to_numpy()[~mask, None]).sum(axis=0)
    assert system.meta["peak_available_mw"] == pytest.approx(hourly.max(), abs=1.0)

    storage = pd.read_csv(root / "static" / "storage_units.csv", index_col=0)
    assert not retired_mask(storage, 2040).any()
    storage_p_nom = storage["p_nom"].sum()
    assert system.meta["peak_available_incl_storage_mw"] == pytest.approx(
        hourly.max() + storage_p_nom, abs=1.0
    )


@unittest.skipIf(real_z4_dir() is None, "data/ca2040_z4/weather.zarr is not present")
def test_real_z4_lifetime_retirements():
    """The CA 2040 z4 fleet loses 11 rows / 5,623 MW to the lifetime rule.

    Hoover hydro (4 rows, 592 MW) carries lifetime == 0, read as infinite."""
    root = real_z4_dir()
    system = load_system(root, LoadOptions(years=(2020,), window=HourWindow(0, 24)))
    meta = system.meta

    assert meta["model_year"] == 2040
    assert meta["retired_rows"] == {"Generator": 11, "StorageUnit": 0}
    assert meta["retired_capacity_mw"]["Generator"] == pytest.approx(5623.0, abs=1.0)
    assert meta["retired_capacity_mw"]["StorageUnit"] == 0.0

    by_carrier = meta["retired_capacity_mw_by_carrier"]["Generator"]
    assert set(by_carrier) == {"onwind", "biomass", "OCGT", "solar", "oil"}
    assert sum(by_carrier.values()) == pytest.approx(5623.0, abs=1.0)

    gens = pd.read_csv(root / "static" / "generators.csv", index_col=0)
    generator = system.index.get(system.devices, "Generator")
    assert list(generator.name) == list(gens.index)
    mask = retired_mask(gens, 2040)
    np.testing.assert_allclose(generator.nominal_capacity.ravel()[mask], 0.0)
    np.testing.assert_allclose(
        generator.nominal_capacity.ravel()[~mask], gens["p_nom"].to_numpy()[~mask], rtol=1e-12
    )


def test_outage_draw_rejects_unwritten_chunk(dataset):
    """A chunk that was never generated (fill value 255) must not load as availability."""
    pytest.importorskip("zap.reliability.outages")
    import zarr

    path = _write_tiny_outage_store(dataset)
    group = zarr.open_group(str(path), mode="a")
    group["available"][0, 0, 3:6, :] = 255
    with pytest.raises(ValueError, match="fill values|never generated"):
        _load(dataset, window=HourWindow(0, N_HOURS), outage_draw=0)
