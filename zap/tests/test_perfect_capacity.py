"""Perfect capacity as a hub generator: devices and loader (WP-H0).

Spec: ``memory/plans/2026-09-23-perfect-capacity-hub-spec.md`` sections 3.1,
4.1, 4.2 and 10.1 in the brain repo.  Perfect capacity P is one
``PerfectGenerator`` at a hub bus appended as the last node, joined to every
load bus by a ``PerfectLink`` rated ``PERFECT_LINK_HEADROOM * P``.  P = 0
builds no node and no device.

Tests T-H0.1 ... T-H0.8 of spec section 10.1, on the hermetic tiny fixture
(48 h) and two hand-made toys: a congested two-bus 6 h LP (T-H0.4) and the
ground-free three-bus toy of ``test_accreditation_vjp.py`` (T-H0.8).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from zap.devices import (
    PERFECT_CARRIER,
    PERFECT_HUB_BUS,
    PERFECT_LINK_HEADROOM,
    Generator,
    Load,
    PerfectGenerator,
    PerfectLink,
    StorageUnit,
    perfect_capacity_devices,
    perfect_load_nodes,
)
from zap.devices.transporter import DirectedLine
from zap.importers.wy_store import (
    HourWindow,
    LoadOptions,
    available_capacity,
    bus_peak_shares,
    convert_dataset,
    load_system,
)
from zap.network import PowerNetwork
from zap.tests.fixtures.tiny_dataset import (
    TINY_BUSES,
    TINY_LINKS,
    TINY_LOADS,
    tiny_params_path,
    write_tiny_dataset,
)

N_HOURS = 48
P_MW = 500.0
DATASET_KEYS = ["Generator", "Load", "DirectedLine", "StorageUnit", "ExportSink"]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    root = write_tiny_dataset(
        tmp_path_factory.mktemp("tiny_perfect") / "tiny", n_hours=N_HOURS, years=(2020,)
    )
    convert_dataset(root)
    return root


def _load(dataset: Path, **kwargs):
    options = LoadOptions(
        years=(2020,),
        window=kwargs.pop("window", HourWindow(0, N_HOURS)),
        demand_scaling="none",
        outage_params_path=str(tiny_params_path()),
        **kwargs,
    )
    return load_system(dataset, options)


def _device_state(device) -> dict:
    """Every attribute of a device that can be compared: arrays, scalars, strings."""
    out = {}
    for attr, value in vars(device).items():
        if isinstance(value, np.ndarray):
            out[attr] = value
        elif isinstance(value, pd.Index):
            out[attr] = np.asarray(value)
        elif isinstance(value, (int, float, str, bool, type(None))):
            out[attr] = value
    return out


def _assert_same_devices(a_devices, b_devices):
    assert len(a_devices) == len(b_devices)
    for a, b in zip(a_devices, b_devices):
        assert type(a) is type(b)
        sa, sb = _device_state(a), _device_state(b)
        assert set(sa) == set(sb)
        for key in sa:
            if isinstance(sa[key], np.ndarray):
                np.testing.assert_array_equal(sa[key], sb[key], err_msg=f"{type(a)} {key}")
            else:
                assert sa[key] == sb[key], (type(a), key)


def _assert_same_index(a, b):
    assert a.device_index == b.device_index
    for field in ("names", "carrier", "bus"):
        da, db = getattr(a, field), getattr(b, field)
        assert list(da) == list(db)
        for key in da:
            np.testing.assert_array_equal(np.asarray(da[key]), np.asarray(db[key]))
    for field in ("emission_rates", "vre_mask", "thermal_mask", "import_mask"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


# ---------------------------------------------------------------------------
# T-H0.1 P = 0 bit-identity
# ---------------------------------------------------------------------------


def test_h0_1_zero_perfect_capacity_is_bit_identical(dataset):
    default = _load(dataset)
    explicit = _load(dataset, perfect_capacity_mw=0.0)

    _assert_same_devices(default.devices, explicit.devices)
    _assert_same_index(default.index, explicit.index)
    assert default.network.num_nodes == explicit.network.num_nodes == len(TINY_BUSES)
    for device in explicit.devices:
        assert device.num_nodes == len(TINY_BUSES)
        assert not isinstance(device, (PerfectGenerator, PerfectLink))

    # No node, no device, no index entry: exactly the historical key set.
    assert list(explicit.index.device_index) == DATASET_KEYS
    assert explicit.meta["perfect_capacity_mw"] == 0.0
    assert isinstance(explicit.meta["perfect_capacity_mw"], float)
    assert explicit.meta["perfect_capacity"] is None

    # The meta dicts differ in nothing (the weather-store attrs included).
    assert set(default.meta) == set(explicit.meta)
    for key in default.meta:
        assert default.meta[key] == explicit.meta[key], key


# ---------------------------------------------------------------------------
# T-H0.2 Layout
# ---------------------------------------------------------------------------


def test_h0_2_layout(dataset):
    base = _load(dataset)
    hub = _load(dataset, perfect_capacity_mw=P_MW)
    n = len(TINY_BUSES)

    # The hub is the last node; the network and every device see n + 1 nodes.
    assert hub.network.num_nodes == n + 1
    assert hub.meta["perfect_capacity"]["hub_node"] == n
    assert all(device.num_nodes == n + 1 for device in hub.devices)

    # Existing keys keep their positions; the two new keys are last.
    keys = list(hub.index.device_index)
    assert keys == DATASET_KEYS + ["PerfectGenerator", "PerfectLink"]
    for key in DATASET_KEYS:
        assert hub.index.device_index[key] == base.index.device_index[key]
    assert hub.index.device_index["PerfectGenerator"] == len(base.devices)
    assert hub.index.device_index["PerfectLink"] == len(base.devices) + 1

    # Every dataset device's terminals are unchanged.
    for key in DATASET_KEYS:
        a = base.index.get(base.devices, key)
        b = hub.index.get(hub.devices, key)
        np.testing.assert_array_equal(np.asarray(a.terminals), np.asarray(b.terminals))

    gen = hub.index.get(hub.devices, "PerfectGenerator")
    link = hub.index.get(hub.devices, "PerfectLink")
    assert type(gen) is PerfectGenerator and type(link) is PerfectLink
    np.testing.assert_array_equal(np.asarray(gen.terminal).ravel(), [n])

    # One link per load bus, hub -> bus.
    load_buses = sorted({bus for _, bus in TINY_LOADS})
    expected_nodes = [TINY_BUSES.index(b) for b in load_buses]
    assert link.num_devices == len(load_buses)
    np.testing.assert_array_equal(link.source_terminal, [n] * len(load_buses))
    np.testing.assert_array_equal(link.sink_terminal, expected_nodes)
    assert list(link.name) == [f"{PERFECT_HUB_BUS}->{b}" for b in load_buses]

    # SystemIndex labels.
    assert list(hub.index.names["PerfectGenerator"]) == [PERFECT_HUB_BUS]
    assert list(hub.index.carrier["PerfectGenerator"]) == [PERFECT_CARRIER]
    assert list(hub.index.bus["PerfectGenerator"]) == [PERFECT_HUB_BUS]
    assert list(hub.index.carrier["PerfectLink"]) == [PERFECT_CARRIER] * len(load_buses)
    assert list(hub.index.bus["PerfectLink"]) == load_buses


def test_h0_2_bus_name_table_names_the_hub(dataset):
    persist = pytest.importorskip("ch3.ra.persist")
    hub = _load(dataset, perfect_capacity_mw=P_MW)
    names = persist.bus_name_table(hub, hub.devices)
    assert names.size == len(TINY_BUSES) + 1
    assert names[len(TINY_BUSES)] == PERFECT_HUB_BUS
    for bus in ("z1", "z2"):
        assert names[TINY_BUSES.index(bus)] == bus


# ---------------------------------------------------------------------------
# T-H0.3 Isolation
# ---------------------------------------------------------------------------


def test_h0_3_available_capacity_and_import_mask_unchanged(dataset):
    base = _load(dataset)
    hub = _load(dataset, perfect_capacity_mw=P_MW)
    np.testing.assert_array_equal(available_capacity(base), available_capacity(hub))
    np.testing.assert_array_equal(
        available_capacity(base, include_storage=True, include_imports=True),
        available_capacity(hub, include_storage=True, include_imports=True),
    )
    np.testing.assert_array_equal(base.index.import_mask, hub.index.import_mask)
    for key in ("peak_load_mw", "peak_available_mw", "peak_available_incl_imports_mw"):
        assert base.meta[key] == hub.meta[key]


def test_h0_3_group_matrix_excludes_the_hub_links(dataset):
    base = _load(dataset, import_limit_mw=30.0)
    hub = _load(dataset, import_limit_mw=30.0, perfect_capacity_mw=P_MW)
    line_base = base.index.get(base.devices, "DirectedLine")
    line_hub = hub.index.get(hub.devices, "DirectedLine")
    np.testing.assert_array_equal(line_base.group_matrix, line_hub.group_matrix)
    np.testing.assert_array_equal(line_base.group_limit, line_hub.group_limit)
    assert np.asarray(line_hub.group_matrix).shape == (1, len(TINY_LINKS))

    link = hub.index.get(hub.devices, "PerfectLink")
    assert link.group is None and link.group_limit is None and link.group_matrix is None


def test_h0_3_outage_draw_is_bit_identical_with_and_without_the_hub(dataset):
    pytest.importorskip("zap.reliability.outages")
    for draw in (0, 3):
        base = _load(dataset, outage_draw=draw)
        hub = _load(dataset, outage_draw=draw, perfect_capacity_mw=P_MW)
        np.testing.assert_array_equal(
            base.index.get(base.devices, "Generator").dynamic_capacity,
            hub.index.get(hub.devices, "Generator").dynamic_capacity,
        )
        np.testing.assert_array_equal(
            base.index.get(base.devices, "StorageUnit").power_availability,
            hub.index.get(hub.devices, "StorageUnit").power_availability,
        )
        # Everything but the process-local draw-cache counters.
        strip = {k: v for k, v in base.meta["outages"].items() if k != "cache"}
        assert strip == {k: v for k, v in hub.meta["outages"].items() if k != "cache"}
        assert base.meta["outages"]["cache"]["key"] == hub.meta["outages"]["cache"]["key"]
        # The hub itself is never derated.
        gen = hub.index.get(hub.devices, "PerfectGenerator")
        np.testing.assert_array_equal(gen.dynamic_capacity, np.ones((1, 1)))


def test_h0_3_design_capacity_ignores_the_hub(dataset):
    base = _load(dataset)
    names = list(base.index.names["Generator"])
    capacity = np.asarray(
        base.index.get(base.devices, "Generator").nominal_capacity, dtype=float
    ).ravel()
    capacity[names.index("z1 CCGT")] = 250.0
    design = {"Generator": capacity.tolist()}

    designed = _load(dataset, design_capacity=design)
    designed_hub = _load(dataset, design_capacity=design, perfect_capacity_mw=P_MW)
    for key in DATASET_KEYS:
        a = designed.index.get(designed.devices, key)
        b = designed_hub.index.get(designed_hub.devices, key)
        for attr in ("nominal_capacity", "power_capacity"):
            if hasattr(a, attr):
                np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
    for key in ("design_capacity_applied", "design_capacity_digest", "design_capacity_summary"):
        assert designed.meta[key] == designed_hub.meta[key]
    gen = designed_hub.index.get(designed_hub.devices, "PerfectGenerator")
    np.testing.assert_array_equal(np.asarray(gen.nominal_capacity).ravel(), [P_MW])


def test_h0_3_meta_lists_the_load_buses(dataset):
    hub = _load(dataset, perfect_capacity_mw=P_MW, firm_load_mw=5.0)
    info = hub.meta["perfect_capacity"]
    assert hub.meta["perfect_capacity_mw"] == P_MW
    assert info == {
        "hub_node": len(TINY_BUSES),
        "hub_bus": PERFECT_HUB_BUS,
        "load_buses": ["z1", "z2"],
        "load_nodes": [TINY_BUSES.index("z1"), TINY_BUSES.index("z2")],
        "link_headroom": PERFECT_LINK_HEADROOM,
        "link_mw": PERFECT_LINK_HEADROOM * P_MW,
    }
    # The same support as the firm-load rule's f_n.
    load = hub.index.get(hub.devices, "Load")
    shares = bus_peak_shares(load.load, load.terminal, hub.network.num_nodes)
    assert info["load_nodes"] == np.flatnonzero(shares > 0).tolist()


# ---------------------------------------------------------------------------
# T-H0.4 Dispatch: the hub price is the maximum load-bus price
# ---------------------------------------------------------------------------

T_TOY = 6
VOLL = 10_000.0
TOY_P = 30.0
#: b1 demand; the 80 MW hours are short (20 local + 10 line + 30 hub = 60).
TOY_LOAD_B1 = np.array([80.0, 80.0, 30.0, 80.0, 25.0, 80.0])
TOY_SHORT = TOY_LOAD_B1 > 60.0


def _congested_toy(perfect_mw: float = TOY_P):
    """b0: 100 MW @ 20 $/MWh, 50 MW load; b1: 20 MW @ 30 $/MWh, TOY_LOAD_B1 load.

    The b0 -> b1 line is rated 10 MW, so b1 sheds in its 80 MW hours whatever
    b0 can generate.  The hub (node 2) reaches both buses.
    """
    n = 3
    generators = Generator(
        num_nodes=n,
        name=np.array(["g0", "g1"], dtype=object),
        terminal=np.array([0, 1]),
        nominal_capacity=np.array([100.0, 20.0]),
        dynamic_capacity=np.ones((2, T_TOY)),
        linear_cost=np.array([[20.0], [30.0]]),
    )
    loads = Load(
        num_nodes=n,
        name=np.array(["l0", "l1"], dtype=object),
        terminal=np.array([0, 1]),
        load=np.vstack([np.full(T_TOY, 50.0), TOY_LOAD_B1]),
        linear_cost=np.full((2, 1), VOLL),
    )
    line = DirectedLine(
        num_nodes=n,
        name=np.array(["b0->b1"], dtype=object),
        source_terminal=np.array([0]),
        sink_terminal=np.array([1]),
        min_power=np.zeros((1, 1)),
        max_power=np.ones((1, 1)),
        linear_cost=np.zeros((1, 1)),
        nominal_capacity=np.array([10.0]),
    )
    gen, link = perfect_capacity_devices(
        num_nodes=n,
        hub_node=2,
        load_nodes=[0, 1],
        capacity_mw=perfect_mw,
        load_bus_names=["b0", "b1"],
    )
    return PowerNetwork(n), [generators, loads, line, gen, link]


def test_h0_4_hub_delivers_p_to_the_short_bus_at_the_max_price():
    network, devices = _congested_toy()
    outcome = network.dispatch(devices, T_TOY, solver=cp.HIGHS, add_ground=False)
    assert outcome.problem.status == cp.OPTIMAL

    hub_gen = np.asarray(outcome.power[3][0]).ravel()
    flows = np.asarray(outcome.power[4][1])  # delivered MW at each load bus
    prices = np.asarray(outcome.prices)
    assert prices.shape == (3, T_TOY)
    max_load_price = prices[:2].max(axis=0)

    # Every hour here has a positive load-bus price, so the hub runs at P.
    assert (max_load_price > 0).all()
    np.testing.assert_allclose(hub_gen, TOY_P, atol=1e-7)
    np.testing.assert_allclose(flows.sum(axis=0), hub_gen, atol=1e-7)

    # No link exceeds P, so none is anywhere near its 2P rating.
    assert flows.max() <= TOY_P + 1e-7
    assert flows.min() >= -1e-9
    assert flows.max() < PERFECT_LINK_HEADROOM * TOY_P

    # In the short hours all of P goes to b1, b1 sheds, and its price is VOLL.
    np.testing.assert_allclose(flows[1, TOY_SHORT], TOY_P, atol=1e-7)
    np.testing.assert_allclose(flows[0, TOY_SHORT], 0.0, atol=1e-7)
    np.testing.assert_allclose(prices[1, TOY_SHORT], VOLL, rtol=1e-9)
    np.testing.assert_allclose(prices[0, TOY_SHORT], 20.0, rtol=1e-9)

    # The identity of spec 3.1 in every hour with max_n pi_n > 0.
    positive = max_load_price > 0
    np.testing.assert_allclose(prices[2, positive], max_load_price[positive], atol=1e-9, rtol=0)


def test_h0_4_the_hub_relieves_exactly_p_of_shortfall():
    network, with_hub = _congested_toy()
    _, without = _congested_toy()
    without = without[:3]
    a = network.dispatch(with_hub, T_TOY, solver=cp.HIGHS, add_ground=False)
    b = network.dispatch(without, T_TOY, solver=cp.HIGHS, add_ground=False)

    def shed(outcome):
        load = np.asarray(outcome.power[1][0])
        return (np.vstack([np.full(T_TOY, 50.0), TOY_LOAD_B1]) + load).sum(axis=0)

    np.testing.assert_allclose(shed(b)[TOY_SHORT] - shed(a)[TOY_SHORT], TOY_P, atol=1e-7)


# ---------------------------------------------------------------------------
# T-H0.5 Static horizon
# ---------------------------------------------------------------------------


def test_h0_5_devices_are_static(dataset):
    hub = _load(dataset, perfect_capacity_mw=P_MW)
    for key in ("PerfectGenerator", "PerfectLink"):
        device = hub.index.get(hub.devices, key)
        assert device.time_horizon == 0
        sampled = device.sample_time(np.arange(3, 9), N_HOURS)
        assert type(sampled) is type(device)
        sa, sb = _device_state(device), _device_state(sampled)
        for attr in sa:
            if attr == "capital_cost":  # rescaled by the window ratio; zero stays zero
                np.testing.assert_array_equal(sb[attr], 0.0)
            elif isinstance(sa[attr], np.ndarray):
                np.testing.assert_array_equal(sa[attr], sb[attr], err_msg=f"{key}.{attr}")

    # The whole system still dispatches on a sampled block.
    block = [d.sample_time(np.arange(0, 24), N_HOURS) for d in hub.devices]
    outcome = hub.network.dispatch(block, 24, solver=cp.HIGHS)
    assert outcome.problem.status == cp.OPTIMAL


# ---------------------------------------------------------------------------
# T-H0.6 Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [0.0, -1.0, math.nan, math.inf])
def test_h0_6_builder_refuses_nonpositive_or_nonfinite_capacity(capacity):
    with pytest.raises(ValueError, match="perfect capacity"):
        perfect_capacity_devices(
            num_nodes=3,
            hub_node=2,
            load_nodes=[0, 1],
            capacity_mw=capacity,
            load_bus_names=["a", "b"],
        )


def test_h0_6_builder_refuses_bad_layouts():
    kwargs = {"num_nodes": 3, "hub_node": 2, "capacity_mw": 10.0}
    with pytest.raises(ValueError):
        perfect_capacity_devices(load_nodes=[], load_bus_names=[], **kwargs)
    with pytest.raises(ValueError):
        perfect_capacity_devices(load_nodes=[0, 2], load_bus_names=["a", "hub"], **kwargs)
    with pytest.raises(ValueError):
        perfect_capacity_devices(load_nodes=[0, 1], load_bus_names=["a"], **kwargs)
    with pytest.raises(ValueError):
        perfect_capacity_devices(
            load_nodes=[0, 1], load_bus_names=["a", "b"], link_headroom=1.0, **kwargs
        )


@pytest.mark.parametrize("capacity", [-1.0, math.nan, math.inf])
def test_h0_6_loader_refuses_negative_or_nonfinite_capacity(dataset, capacity):
    with pytest.raises(ValueError, match="perfect_capacity_mw"):
        _load(dataset, perfect_capacity_mw=capacity)


def test_h0_6_loader_refuses_a_dataset_bus_named_perfect_hub(tmp_path):
    root = write_tiny_dataset(tmp_path / "clash", n_hours=N_HOURS, years=(2020,))
    buses_csv = root / "static" / "buses.csv"
    buses = pd.read_csv(buses_csv, index_col=0)
    extra = buses.iloc[[0]].copy()
    extra.index = pd.Index([PERFECT_HUB_BUS], name=buses.index.name)
    pd.concat([buses, extra]).to_csv(buses_csv)
    convert_dataset(root)

    # P = 0 builds the dataset as it is; P > 0 refuses the name clash.
    assert _load(root).network.num_nodes == len(TINY_BUSES) + 1
    with pytest.raises(ValueError, match=PERFECT_HUB_BUS):
        _load(root, perfect_capacity_mw=P_MW)


# ---------------------------------------------------------------------------
# T-H0.7 Unit invariance
# ---------------------------------------------------------------------------


def test_h0_7_unit_invariance(dataset):
    """objective * power_unit * cost_unit is constant; zero costs stay zero."""
    p_mw = 20.0  # partial: the hub displaces some thermal, so the objective is live
    values = {}
    for power_unit, cost_unit in ((1.0, 1.0), (10.0, 1.0), (10.0, 1000.0)):
        system = _load(
            dataset,
            window=HourWindow(0, 24),
            perfect_capacity_mw=p_mw,
            power_unit=power_unit,
            cost_unit=cost_unit,
        )
        gen = system.index.get(system.devices, "PerfectGenerator")
        link = system.index.get(system.devices, "PerfectLink")
        np.testing.assert_allclose(np.asarray(gen.nominal_capacity).ravel(), p_mw / power_unit)
        np.testing.assert_allclose(np.asarray(gen.max_nominal_capacity).ravel(), p_mw / power_unit)
        np.testing.assert_allclose(
            np.asarray(link.nominal_capacity).ravel(), PERFECT_LINK_HEADROOM * p_mw / power_unit
        )
        for device in (gen, link):
            np.testing.assert_array_equal(device.linear_cost, 0.0)
            np.testing.assert_array_equal(device.capital_cost, 0.0)
        np.testing.assert_array_equal(gen.emission_rates, 0.0)
        # meta stays MW-denominated
        assert system.meta["perfect_capacity"]["link_mw"] == PERFECT_LINK_HEADROOM * p_mw

        outcome = system.network.dispatch(system.devices, solver=cp.HIGHS)
        assert outcome.problem.status == cp.OPTIMAL
        hub_mwh = float(
            np.asarray(outcome.power[system.index.device_index["PerfectGenerator"]][0]).sum()
        )
        values[(power_unit, cost_unit)] = (
            outcome.problem.value * power_unit * cost_unit,
            hub_mwh * power_unit,
        )

    reference, hub_mwh = values[(1.0, 1.0)]
    assert abs(reference) > 1.0
    assert hub_mwh > 0.0
    for key, (value, _) in values.items():
        assert value == pytest.approx(reference, rel=1e-7), key


# ---------------------------------------------------------------------------
# T-H0.8 KKT smoke
# ---------------------------------------------------------------------------


@pytest.fixture
def float64_default():
    torch = pytest.importorskip("torch")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield torch
    finally:
        torch.set_default_dtype(previous)


def test_h0_8_kkt_vjp_smoke_with_the_hub(float64_default):
    """SMOKE TEST ONLY (asserts nothing numerical beyond finiteness): the subclasses torchify and differentiate like their parents.

    The hub is refused in planning (spec 4.5); this proves that a KKT/VJP pass
    over a system that carries it runs at float64 without a dtype error
    (LESSONS 2026-09-15).  Ground-free toy of ``test_accreditation_vjp.py``
    (bus 0 sheds in hour 3) with the hub at node 3.
    """
    torch = float64_default
    from zap.layer import DispatchLayer
    from zap.planning import InvestmentObjective
    from zap.planning.operation_objectives import UnservedEnergyObjective
    from zap.planning.problem_cvx import PlanningProblemCVX

    n, t = 4, 6
    generators = Generator(
        num_nodes=n,
        name=np.array(["gen_a", "gen_b", "gen_c"], dtype=object),
        terminal=np.array([0, 1, 2]),
        nominal_capacity=np.array([[50.0], [40.0], [30.0]]),
        dynamic_capacity=np.ones((3, t)),
        linear_cost=np.array([[20.0], [10.0], [30.0]]) * np.ones((1, t)),
        emission_rates=np.zeros((3, 1)),
        capital_cost=np.ones((3, 1)),
    )
    loads = Load(
        num_nodes=n,
        name=np.array(["load_b0", "load_b2"], dtype=object),
        terminal=np.array([0, 2]),
        load=np.vstack([[80.0, 80.0, 80.0, 200.0, 80.0, 80.0], np.full(t, 10.0)]),
        linear_cost=VOLL * np.ones((2, 1)),
    )
    storage = StorageUnit(
        num_nodes=n,
        name=np.array(["batt_b0"], dtype=object),
        terminal=np.array([0]),
        power_capacity=np.array([[20.0]]),
        duration=np.array([[2.0]]),
        charge_efficiency=np.array([[0.9]]),
        discharge_efficiency=np.array([[0.9]]),
        initial_soc=np.array([[0.0]]),
        final_soc=np.array([[0.0]]),
        linear_cost=np.array([[0.0]]),
        soc_mode="fixed",
        capital_cost=np.ones((1, 1)),
    )
    lines = DirectedLine(
        num_nodes=n,
        name=np.array(["ln_b1", "ln_b2"], dtype=object),
        source_terminal=np.array([1, 2]),
        sink_terminal=np.array([0, 0]),
        min_power=np.zeros((2, 1)),
        max_power=np.array([[100.0], [100.0]]),
        linear_cost=np.zeros((2, 1)),
        efficiency=np.array([[0.9], [0.95]]),
    )
    load_nodes = perfect_load_nodes(loads.load, loads.terminal, n)
    np.testing.assert_array_equal(load_nodes, [0, 2])
    gen, link = perfect_capacity_devices(
        num_nodes=n,
        hub_node=3,
        load_nodes=load_nodes,
        capacity_mw=30.0,
        load_bus_names=["b0", "b2"],
    )
    devices = [generators, loads, storage, lines, gen, link]
    network = PowerNetwork(n)

    parameter_names = {
        "generator_capacity": (0, "nominal_capacity"),
        "storage_power": (2, "power_capacity"),
    }
    theta = {
        "generator_capacity": np.array([[50.0], [40.0], [30.0]]),
        "storage_power": np.array([[20.0]]),
    }
    layer = DispatchLayer(
        network,
        devices,
        parameter_names,
        time_horizon=t,
        solver=cp.HIGHS,
        solver_kwargs={},
        add_ground=False,
    )
    problem = PlanningProblemCVX(
        UnservedEnergyObjective(devices),
        InvestmentObjective(devices, layer),
        layer,
        {k: np.zeros_like(v) for k, v in theta.items()},
        {k: np.full_like(v, 1e4) for k, v in theta.items()},
    )
    problem.forward(requires_grad=True, **copy.deepcopy(theta))
    problem.backward()
    dtheta, adjoint, eue = problem.backward_objective(
        UnservedEnergyObjective(devices), return_value=True
    )

    assert eue > 0.0  # the toy still sheds with 30 MW of perfect capacity
    for key, grad in dtheta.items():
        grad = grad.detach().cpu().numpy() if isinstance(grad, torch.Tensor) else np.asarray(grad)
        assert np.all(np.isfinite(grad)), key
    prices = np.asarray(adjoint.prices, dtype=float)
    assert prices.shape == (n, t)
    assert np.all(np.isfinite(prices))
