"""Verifier findings for WP-E1 (2026-09-09), now regression tests.

Finding 1 (confirmed, fixed): the harness's evaluation path never reached the
``LoadOptions.design_capacity`` seam.  ``dispatch.solve_block`` called
``build_system(cfg, draw=task.draw)`` with no design and ``slice_devices``
patched capacities on with ``Design.apply``, so a design evaluated through the
harness had its built capacity outage-free -- the E1 defect the diff was meant
to close.  ``solve_block`` now builds *with* the design and ``slice_devices``
asserts instead of applying (spec WP-E1, 3.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zap.importers.wy_store import convert_dataset
from zap.tests import test_wy_store_design as T
from zap.tests.fixtures.tiny_dataset import write_tiny_dataset


@pytest.fixture
def dataset(tmp_path):
    root = write_tiny_dataset(tmp_path / "tiny", n_hours=T.N_HOURS, years=T.YEARS)
    gens = pd.read_csv(root / "static" / "generators.csv", index_col=0)
    for name, value in T.STATIC_P_NOM.items():
        gens.loc[name, "p_nom"] = value
    gens.to_csv(root / "static" / "generators.csv")
    convert_dataset(root)
    T._write_outage_store(root)
    return root


def _eval_cfg(dataset) -> dict:
    cfg = T._cfg(dataset)
    cfg["selection"] = {"blocks": [24], "reference": "none"}
    cfg["heuristics"]["outage_draws"] = [0]
    return cfg


def _design(system, **overrides):
    from experiments.ra import system as system_mod

    return system_mod.Design(design_id="expand", capacities=T._design(system, **overrides))


def test_harness_block_devices_carry_design_outages(dataset):
    """The finding, as a regression: the block's devices are the designed ones."""
    from experiments.ra import dispatch
    from experiments.ra import system as system_mod
    from experiments.ra.blocks import Block

    system_mod.clear_system_cache()
    cfg = _eval_cfg(dataset)

    base = system_mod.build_system(cfg, draw=0)
    row = T._row(base, "Generator", "z2 old CCGT")
    design = _design(base, z2_old_CCGT=1000.0)

    # What solve_block does now: the design goes into the build.
    loaded = system_mod.build_system(cfg, draw=0, design=design)
    block = Block(index=0, year=T.YEARS[0], start=0, stop=24)
    devices = dispatch.slice_devices(loaded, cfg, block, design=design)
    gen = devices[loaded.index.device_index["Generator"]]
    assert gen.nominal_capacity.ravel()[row] == pytest.approx(1000.0)
    # Built capacity on the retired row must be derated by the pool (E1).
    assert gen.dynamic_capacity[row, :].min() < 1.0
    system_mod.clear_system_cache()


def test_solve_block_scores_the_designed_system(dataset):
    """End to end through ``solve_block``: the design reaches the outage lookup."""
    from experiments.ra import dispatch
    from experiments.ra import system as system_mod
    from experiments.ra.blocks import Block
    from experiments.ra.tasks import Task

    system_mod.clear_system_cache()
    cfg = _eval_cfg(dataset)
    cfg["methods"] = {"lp": {"solver": "HIGHS", "solver_kwargs": {}, "required": True}}

    base = system_mod.build_system(cfg, draw=0)
    row = T._row(base, "Generator", "z2 old CCGT")
    design = _design(base, z2_old_CCGT=1000.0)

    block = Block(index=0, year=T.YEARS[0], start=0, stop=24)
    task = Task(
        task_id="expand-lp-24-y2020-b00000",
        method="lp",
        block_size=24,
        block=block,
        draw=0,
        design_id="expand",
    )

    system_mod.clear_system_cache()
    payload = dispatch.solve_block(task, cfg, design=design)
    assert payload["solver_status"] == "optimal"

    # `solve_block` cached the system it solved: it is the designed one, and its
    # availability for the built row came from the pool.
    built = system_mod.build_system(cfg, draw=0, design=design)
    gen = built.index.get(built.devices, "Generator")
    assert built.meta["design_id"] == "expand"
    assert gen.nominal_capacity.ravel()[row] == pytest.approx(1000.0)
    assert gen.dynamic_capacity[row, :].min() < 1.0

    system_mod.clear_system_cache()


def test_solve_block_costs_the_outages_of_built_capacity(dataset):
    """End to end and *quantitative*: the built row's outages reach the LP.

    The design puts all the system's supply on one pooled row that is retired
    as-built (250 MW = one virtual unit), so the block's cost differs between a
    draw and no draw only because that built capacity is exposed to the pool --
    which is precisely what the E1 defect made impossible.
    """
    from experiments.ra import dispatch
    from experiments.ra import system as system_mod
    from experiments.ra.blocks import Block
    from experiments.ra.tasks import Task

    system_mod.clear_system_cache()
    cfg = _eval_cfg(dataset)
    cfg["methods"] = {"lp": {"solver": "HIGHS", "solver_kwargs": {}, "required": True}}

    base = system_mod.build_system(cfg, draw=0)
    row = T._row(base, "Generator", "z2 old CCGT")
    capacities = {k: np.zeros_like(v) for k, v in T._as_built(base).items()}
    capacities["DirectedLine"] = T._as_built(base)["DirectedLine"]
    capacities["Generator"][row] = 250.0
    design = system_mod.Design(design_id="one_unit", capacities=capacities)

    block = Block(index=0, year=T.YEARS[0], start=0, stop=24)

    def _solve(draw):
        system_mod.clear_system_cache()
        task = Task(
            task_id=f"one_unit-lp-24-y2020-b00000-d{draw}",
            method="lp",
            block_size=24,
            block=block,
            draw=draw,
            design_id="one_unit",
        )
        return dispatch.solve_block(task, cfg, design=design)

    with_draw = _solve(0)
    without_draw = _solve(None)
    assert with_draw["solver_status"] == without_draw["solver_status"] == "optimal"

    # The unit is out in at least one of the block's hours, so the drawn case
    # sheds strictly more energy and costs strictly more.
    assert with_draw["metrics"]["unserved_energy_mwh"] > (
        without_draw["metrics"]["unserved_energy_mwh"] + 1e-6
    )
    assert with_draw["metrics"]["operational_cost"] > without_draw["metrics"]["operational_cost"]
    system_mod.clear_system_cache()


def test_solve_block_derates_built_capacity_relative_to_a_post_hoc_patch(dataset):
    """The quantitative form of the defect: post-hoc patching is outage-free."""
    from experiments.ra import dispatch
    from experiments.ra import system as system_mod
    from experiments.ra.blocks import Block

    system_mod.clear_system_cache()
    cfg = _eval_cfg(dataset)
    base = system_mod.build_system(cfg, draw=0)
    row = T._row(base, "Generator", "z2 old CCGT")
    design = _design(base, z2_old_CCGT=1000.0)
    block = Block(index=0, year=T.YEARS[0], start=0, stop=24)

    # The old path: as-built build, design patched on afterwards.
    patched = design.apply(base)
    patched_gen = patched[base.index.device_index["Generator"]]
    np.testing.assert_array_equal(patched_gen.dynamic_capacity[row, :], np.ones(T.N_HOURS))

    # The new path: same capacity, but the pool derates it.
    built = system_mod.build_system(cfg, draw=0, design=design)
    devices = dispatch.slice_devices(built, cfg, block, design=design)
    gen = devices[built.index.device_index["Generator"]]
    assert gen.nominal_capacity.ravel()[row] == pytest.approx(1000.0)
    assert gen.dynamic_capacity[row, :].mean() < 1.0
    system_mod.clear_system_cache()


def test_slice_devices_refuses_an_undesigned_system(dataset):
    """A build that ignored the design must not pass silently (spec 3.1)."""
    from experiments.ra import dispatch
    from experiments.ra import system as system_mod
    from experiments.ra.blocks import Block

    system_mod.clear_system_cache()
    cfg = _eval_cfg(dataset)
    base = system_mod.build_system(cfg, draw=0)
    design = _design(base, z2_old_CCGT=1000.0)
    block = Block(index=0, year=T.YEARS[0], start=0, stop=24)

    with pytest.raises(ValueError, match="does not carry design"):
        dispatch.slice_devices(base, cfg, block, design=design)
    system_mod.clear_system_cache()


def test_slice_devices_accepts_the_designed_system_and_the_no_design_case(dataset):
    from experiments.ra import dispatch
    from experiments.ra import system as system_mod
    from experiments.ra.blocks import Block

    system_mod.clear_system_cache()
    cfg = _eval_cfg(dataset)
    base = system_mod.build_system(cfg, draw=0)
    block = Block(index=0, year=T.YEARS[0], start=0, stop=24)

    # No design: unchanged behaviour.
    dispatch.slice_devices(base, cfg, block)
    # Identity design: the as-built system already carries it.
    identity = system_mod.Design(design_id="asbuilt", capacities=T._as_built(base))
    dispatch.slice_devices(base, cfg, block, design=identity)

    # Just inside / outside the 1e-6 MW tolerance.
    nudged = T._as_built(base)
    nudged["Generator"][0] += 5e-7
    dispatch.slice_devices(
        base, cfg, block, design=system_mod.Design(design_id="nudged", capacities=nudged)
    )
    nudged["Generator"][0] += 1e-3
    with pytest.raises(ValueError, match="does not carry design"):
        dispatch.slice_devices(
            base, cfg, block, design=system_mod.Design(design_id="nudged", capacities=nudged)
        )
    system_mod.clear_system_cache()


def test_check_design_applied_compares_in_mw(dataset):
    """Device capacities are MW / power_unit; the check scales back up."""
    from experiments.ra import dispatch
    from experiments.ra import system as system_mod
    from zap.importers.wy_store import HourWindow, LoadOptions, load_system

    scaled = load_system(
        dataset,
        LoadOptions(years=T.YEARS, window=HourWindow(0, 24), power_unit=1000.0),
    )
    gen = scaled.index.get(scaled.devices, "Generator")
    mw = np.asarray(gen.nominal_capacity, dtype=float).reshape(-1) * 1000.0
    design = system_mod.Design(design_id="mw", capacities={"Generator": mw})
    dispatch.check_design_applied(scaled, design)

    off = system_mod.Design(design_id="scaled", capacities={"Generator": mw / 1000.0})
    with pytest.raises(ValueError, match="does not carry design"):
        dispatch.check_design_applied(scaled, off)


def test_load_options_refuses_a_design_on_a_scaled_system(dataset):
    from experiments.ra import system as system_mod

    cfg = _eval_cfg(dataset)
    cfg["system"]["power_unit"] = 1000.0
    base = system_mod.build_system(T._cfg(dataset), draw=0)
    design = _design(base, z2_old_CCGT=1000.0)
    with pytest.raises(ValueError, match="power_unit"):
        system_mod.load_options(cfg, draw=0, design=design)
    # ... and without a design the scaled config still loads.
    system_mod.load_options(cfg, draw=0)
    system_mod.clear_system_cache()
