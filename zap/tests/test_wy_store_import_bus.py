"""The import-bus rule: import buses keep only the named carriers (issue #35).

``LoadOptions.import_bus_keep_carriers`` holds every generator / storage row on
an import bus (bus name ending in ``_imports``) whose carrier is not in the
tuple at zero capacity.  Rows are kept in place, exactly like a lifetime
retirement, and a design cannot put capacity back on them.  ``None`` (the
default) is the historical system.

Spec: ``memory/plans/2026-09-23-import-bus-resources-spec.md`` section 5.1 in
the brain repo.  Tests 1-7 run on the hermetic tiny fixture with
``extra_import_rows=True`` (one extendable CCGT and one battery on
``z1_imports``); test 8 runs on ``data/ca2040_z4`` and is skipped without it.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from zap.importers.wy_store import (
    IMPORT_BUS_RULE_TABLES,
    HourWindow,
    LoadOptions,
    apply_import_bus_rule,
    convert_dataset,
    import_bus_mask,
    import_bus_removed_mask,
    load_system,
    read_static,
)
from zap.tests.fixtures.tiny_dataset import (
    TINY_EXTRA_IMPORT_GENERATORS,
    TINY_EXTRA_IMPORT_STORAGE,
    real_z4_dir,
    tiny_params_path,
    write_tiny_dataset,
)

KEEP = ("unspecified_imports",)
IMPORT_CCGT = TINY_EXTRA_IMPORT_GENERATORS[0][0]
IMPORT_CCGT_MW = TINY_EXTRA_IMPORT_GENERATORS[0][3]
IMPORT_BATTERY = TINY_EXTRA_IMPORT_STORAGE[0][0]
IMPORT_BATTERY_MW = TINY_EXTRA_IMPORT_STORAGE[0][3]
GENERIC = "z1_imports unspecified_imports"
N_HOURS = 48


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    root = write_tiny_dataset(
        tmp_path_factory.mktemp("tiny_import_bus") / "tiny",
        n_hours=N_HOURS,
        years=(2020,),
        extra_import_rows=True,
    )
    convert_dataset(root)
    return root


def _load(dataset: Path, **kwargs):
    options = LoadOptions(
        years=(2020,),
        window=kwargs.pop("window", HourWindow(0, 24)),
        demand_scaling="none",
        outage_params_path=str(tiny_params_path()),
        **kwargs,
    )
    return load_system(dataset, options)


def _capacities(system, cls_name: str) -> np.ndarray:
    device = system.index.get(system.devices, cls_name)
    attr = "power_capacity" if cls_name == "StorageUnit" else "nominal_capacity"
    return np.asarray(getattr(device, attr), dtype=float).reshape(-1)


def _device_arrays(system) -> dict:
    """Every numpy array on every device, keyed ``(class, attribute)``."""
    out = {}
    for cls_name, i in system.index.device_index.items():
        for attr, value in vars(system.devices[i]).items():
            if isinstance(value, np.ndarray):
                out[(cls_name, attr)] = value
    return out


# ---------------------------------------------------------------------------
# 1. None is the historical system
# ---------------------------------------------------------------------------


def test_none_is_bit_identical(dataset):
    default = _load(dataset)
    explicit = _load(dataset, import_bus_keep_carriers=None)
    a, b = _device_arrays(default), _device_arrays(explicit)
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=str(key))

    info = explicit.meta["import_bus_resources"]
    assert info["keep_carriers"] is None
    assert info["removed_rows"] == {"Generator": 0, "StorageUnit": 0}
    assert info["removed_capacity_mw"] == {"Generator": 0.0, "StorageUnit": 0.0}
    assert info["removed_names"] == {"Generator": [], "StorageUnit": []}
    assert info["design_mw_removed"] == {"Generator": 0.0, "StorageUnit": 0.0}
    # The extra rows really are live under None: the test above means something.
    names = list(default.index.names["Generator"])
    assert _capacities(default, "Generator")[names.index(IMPORT_CCGT)] == IMPORT_CCGT_MW


# ---------------------------------------------------------------------------
# 2. Rows kept, in order, at zero
# ---------------------------------------------------------------------------


def test_rows_kept_in_order_and_zeroed(dataset):
    before = _load(dataset)
    after = _load(dataset, import_bus_keep_carriers=KEEP)

    for cls_name, removed in (("Generator", IMPORT_CCGT), ("StorageUnit", IMPORT_BATTERY)):
        names = list(after.index.names[cls_name])
        assert names == list(before.index.names[cls_name])
        cap_before = _capacities(before, cls_name)
        cap_after = _capacities(after, cls_name)
        i = names.index(removed)
        assert cap_before[i] > 0.0
        assert cap_after[i] == 0.0
        others = np.arange(len(names)) != i
        np.testing.assert_array_equal(cap_after[others], cap_before[others])

    gen_names = list(after.index.names["Generator"])
    assert _capacities(after, "Generator")[gen_names.index(GENERIC)] == 50.0

    info = after.meta["import_bus_resources"]
    assert info["keep_carriers"] == list(KEEP)
    assert info["removed_rows"] == {"Generator": 1, "StorageUnit": 1}
    assert info["removed_names"] == {"Generator": [IMPORT_CCGT], "StorageUnit": [IMPORT_BATTERY]}
    assert info["removed_capacity_mw"] == {
        "Generator": IMPORT_CCGT_MW,
        "StorageUnit": IMPORT_BATTERY_MW,
    }
    assert info["removed_capacity_mw_by_carrier"] == {
        "Generator": {"CCGT": IMPORT_CCGT_MW},
        "StorageUnit": {"battery": IMPORT_BATTERY_MW},
    }
    # Links are never touched: the import link keeps its capacity.
    assert after.meta["import_link_capacity_mw"] == before.meta["import_link_capacity_mw"]


# ---------------------------------------------------------------------------
# 3. A design cannot put capacity back
# ---------------------------------------------------------------------------


def test_design_capacity_on_removed_row_is_zeroed_and_recorded(dataset):
    static = read_static(dataset)
    gens = static["generators"]
    designed = gens["p_nom"].to_numpy(dtype=float).copy()
    designed[list(gens.index).index(IMPORT_CCGT)] = 100.0
    design = {"Generator": designed.tolist()}

    ruled = _load(dataset, import_bus_keep_carriers=KEEP, design_capacity=design)
    names = list(ruled.index.names["Generator"])
    assert _capacities(ruled, "Generator")[names.index(IMPORT_CCGT)] == 0.0
    assert ruled.meta["import_bus_resources"]["design_mw_removed"]["Generator"] == 100.0
    assert ruled.meta["import_bus_resources"]["design_mw_removed"]["StorageUnit"] == 0.0
    # The design itself was applied (the rule only overrides the removed rows).
    assert ruled.meta["design_capacity_applied"] is True

    # Under None the same design is honoured.
    free = _load(dataset, design_capacity=design)
    assert _capacities(free, "Generator")[names.index(IMPORT_CCGT)] == 100.0
    assert free.meta["import_bus_resources"]["design_mw_removed"]["Generator"] == 0.0


# ---------------------------------------------------------------------------
# 4. In-state draws do not move
# ---------------------------------------------------------------------------


def test_in_state_draws_are_unchanged(dataset):
    from zap.reliability.keys import row_specs
    from zap.reliability.outages import load_outage_params, slot_counts

    before = _load(dataset, outage_draw=3)
    after = _load(dataset, outage_draw=3, import_bus_keep_carriers=KEEP)

    gen_before = np.asarray(before.index.get(before.devices, "Generator").dynamic_capacity)
    gen_after = np.asarray(after.index.get(after.devices, "Generator").dynamic_capacity)
    in_state = ~import_bus_mask(before.index.bus["Generator"])
    np.testing.assert_array_equal(gen_after[in_state], gen_before[in_state])

    st_before = np.asarray(before.index.get(before.devices, "StorageUnit").power_availability)
    st_after = np.asarray(after.index.get(after.devices, "StorageUnit").power_availability)
    st_in_state = ~import_bus_mask(before.index.bus["StorageUnit"])
    st_before = np.broadcast_to(st_before, (len(st_in_state), st_before.shape[-1]))
    st_after = np.broadcast_to(st_after, (len(st_in_state), st_after.shape[-1]))
    np.testing.assert_array_equal(st_after[st_in_state], st_before[st_in_state])

    # The removed rows' slots are exactly the difference.
    params = load_outage_params(tiny_params_path())
    specs = row_specs(read_static(dataset), params)
    removed = {IMPORT_CCGT: IMPORT_CCGT_MW, IMPORT_BATTERY: IMPORT_BATTERY_MW}
    removed_slots = sum(slot_counts(specs, removed, params).get(n, 0) for n in removed)
    assert removed_slots > 0
    assert before.meta["outages"]["n_units"] - after.meta["outages"]["n_units"] == removed_slots


# ---------------------------------------------------------------------------
# 5. Peak metrics
# ---------------------------------------------------------------------------


def test_peak_metrics(dataset):
    before = _load(dataset)
    after = _load(dataset, import_bus_keep_carriers=KEEP)
    assert after.meta["peak_available_mw"] == before.meta["peak_available_mw"]
    assert after.meta["peak_available_incl_storage_mw"] == pytest.approx(
        before.meta["peak_available_incl_storage_mw"] - IMPORT_BATTERY_MW, abs=1e-9
    )
    assert (
        after.meta["peak_available_incl_imports_mw"]
        < before.meta["peak_available_incl_imports_mw"]
    )


# ---------------------------------------------------------------------------
# 6. Bad keep tuples
# ---------------------------------------------------------------------------


def test_bad_keep_carriers_raise(dataset):
    static = read_static(dataset)
    with pytest.raises(ValueError, match="empty"):
        apply_import_bus_rule(static, ())
    with pytest.raises(ValueError, match="no import"):
        apply_import_bus_rule(static, ("unspecified_import",))  # the typo
    with pytest.raises(ValueError, match="no import"):
        _load(dataset, import_bus_keep_carriers=("solar",))  # in-state only
    with pytest.raises(ValueError, match="empty"):
        import_bus_removed_mask(static["generators"], ())
    # The mask alone: all-False for None, the two extra rows for KEEP.
    for key in IMPORT_BUS_RULE_TABLES.values():
        assert not import_bus_removed_mask(static[key], None).any()
    gens = static["generators"]
    assert list(gens.index[import_bus_removed_mask(gens, KEEP)]) == [IMPORT_CCGT]


# ---------------------------------------------------------------------------
# 7. A zero-capacity import battery through the LP and ADMM
# ---------------------------------------------------------------------------


def test_zero_capacity_import_battery_lp_and_admm(dataset):
    import torch

    from zap.admm import ADMMSolver

    system = _load(
        dataset, import_bus_keep_carriers=KEEP, window=HourWindow(0, 24), export_mode="drop"
    )
    k = system.index.device_index["StorageUnit"]
    row = list(system.index.names["StorageUnit"]).index(IMPORT_BATTERY)
    assert _capacities(system, "StorageUnit")[row] == 0.0

    outcome = system.network.dispatch(
        system.devices, time_horizon=24, solver=cp.HIGHS, add_ground=True
    )
    assert outcome.problem.status == "optimal"
    assert np.isfinite(outcome.problem.value)
    power = np.asarray(outcome.power[k][0], dtype=float)
    energy = np.asarray(outcome.local_variables[k][0], dtype=float)
    assert np.all(np.isfinite(power)) and np.all(np.isfinite(energy))
    np.testing.assert_allclose(power[row], 0.0, atol=1e-7)
    np.testing.assert_allclose(energy[row], 0.0, atol=1e-7)

    torch_devices = [d.torchify(machine="cpu", dtype=torch.float64) for d in system.devices]
    solver = ADMMSolver(
        machine="cpu",
        dtype=torch.float64,
        num_iterations=5000,
        rho_power=0.1,
        atol=1e-4,
        rtol=1e-4,
        verbose=0,
    )
    state, _history = solver.solve(system.network, torch_devices, 24)
    admm = state.as_outcome()
    admm_power = np.asarray(admm.power[k][0].detach().cpu().numpy(), dtype=float)
    assert np.all(np.isfinite(admm_power))
    np.testing.assert_allclose(admm_power[row], 0.0, atol=1e-6)
    admm_energy = admm.local_variables[k]
    if admm_energy is not None:
        energy_rows = np.asarray(admm_energy[0].detach().cpu().numpy(), dtype=float)
        assert np.all(np.isfinite(energy_rows))
        np.testing.assert_allclose(energy_rows[row], 0.0, atol=1e-6)
    for p in admm.power:
        for terminal in p:
            assert np.all(np.isfinite(terminal.detach().cpu().numpy()))


# ---------------------------------------------------------------------------
# 8. The real dataset
# ---------------------------------------------------------------------------

#: ``data/ca2040_z4/static`` facts (spec section 3.4, reproduced 2026-09-23).
Z4_REMOVED_ROWS = {"Generator": 49, "StorageUnit": 16}
Z4_ZEROED_MW = {"Generator": 6937.62, "StorageUnit": 1714.5}
Z4_GENERIC_ROWS = 6
Z4_GENERIC_MW = 19034.68
Z4_IMPORT_LINK_MW = 19034.68


@unittest.skipIf(real_z4_dir() is None, "data/ca2040_z4/weather.zarr is not present")
def test_real_z4_import_bus_rule():
    root = real_z4_dir()
    options = {"years": (2013,), "window": HourWindow(7, 31), "demand_scaling": "none"}
    before = load_system(root, LoadOptions(**options))
    after = load_system(root, LoadOptions(**options, import_bus_keep_carriers=KEEP))
    info = after.meta["import_bus_resources"]

    assert info["removed_rows"] == Z4_REMOVED_ROWS
    for cls_name, mw in Z4_ZEROED_MW.items():
        assert info["removed_capacity_mw"][cls_name] == pytest.approx(mw, abs=0.01)

    static = read_static(root)
    gens = static["generators"]
    generic = (gens["carrier"] == "unspecified_imports").to_numpy()
    cap = _capacities(after, "Generator")
    assert int(generic.sum()) == Z4_GENERIC_ROWS
    assert cap[generic].sum() == pytest.approx(Z4_GENERIC_MW, abs=0.01)
    np.testing.assert_array_equal(cap[generic], _capacities(before, "Generator")[generic])

    assert after.meta["import_link_capacity_mw"] == pytest.approx(Z4_IMPORT_LINK_MW, abs=0.01)
    assert after.meta["import_link_capacity_mw"] == before.meta["import_link_capacity_mw"]

    # Every import bus keeps exactly one row above zero: its generic imports.
    live = pd.Series(cap > 0.0, index=gens["bus"].to_numpy())
    storage = static["storage_units"]
    live_storage = pd.Series(
        _capacities(after, "StorageUnit") > 0.0, index=storage["bus"].to_numpy()
    )
    import_buses = [b for b in static["buses"].index if str(b).endswith("_imports")]
    assert len(import_buses) == Z4_GENERIC_ROWS
    for bus in import_buses:
        n_live = int(live[live.index == bus].sum()) + int(
            live_storage[live_storage.index == bus].sum()
        )
        assert n_live == 1, bus
