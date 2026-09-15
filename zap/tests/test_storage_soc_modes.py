"""Rolling-horizon state of charge (`StorageUnit.soc_mode = "rolling"`), WP-R0b.

`rolling` is the third boundary condition beside `fixed` and `cyclic_free`. It
pins the *opening* level only -- `energy[:, 0] == initial_soc * E`, the seed the
previous window handed over -- and bounds the closing level below by the level at
the commit boundary, `energy[:, T] >= energy[:, soc_anchor_index]` (the terminal
rule of the rolling-horizon evaluator spec, section 3). The rule is appended as
inequality block **6**, never inserted, because blocks 1 / 3 / 5 are the
SoC-upper, charge-upper and discharge-upper multipliers that ch3's accreditation
reads by index.

The mode is dispatch-only: the implicit-differentiation matrices and the ADMM
prox refuse it, and it is incompatible with the windowed-SoC ADMM layout.

These tests cover section 9 test 5 of the spec and guard the two things the rest
of the codebase depends on: the block *count* the storage residual gate sums
over, and the bit-identity of the `fixed` and `cyclic_free` outputs.
"""

import unittest

import cvxpy as cp
import numpy as np
import torch

import zap
from zap.devices import StorageUnit

torch.set_default_dtype(torch.float64)

T = 6
#: Index of the storage device inside the fixture's device list.
STORAGE = 3
#: `initial_soc * energy_capacity` per row on the fixtures below, in MWh.
OPENING_MWH = np.array([20.0, 10.0])
#: Energy capacity per row, in MWh.
EMAX_MWH = np.array([40.0, 20.0])


def rolling_fixture(kind, soc_mode="rolling", soc_anchor_index=0, storage_cls=StorageUnit):
    """A 2-bus, 6-hour toy with two storage rows, in one of two temporal shapes.

    ``kind == "drain"`` -- cheap generation early (hours 0-2, 0.5 $/MWh), only an
    expensive peaker late (200 $/MWh). A window with a free end fills the battery
    from the cheap hours and empties it into the expensive ones, so the level at
    the commit boundary (hour 3) is far *above* the closing level and the terminal
    rule binds.

    ``kind == "refill"`` -- the mirror image: an expensive deficit early and a
    price-taking surplus late (a PTC-style generator at -50 $/MWh in hours 4-5, so
    charging at the end is strictly profitable). The closing level is above the
    commit boundary's on its own and the terminal rule is slack.

    ``storage_cls`` exists so the baseline test can build the same fixture from a
    pre-change copy of :class:`StorageUnit`.
    """
    net = zap.PowerNetwork(2)

    if kind == "drain":
        load_profile = np.array([10.0, 10.0, 10.0, 60.0, 60.0, 60.0])
        cheap_avail = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        ptc_avail = np.zeros(T)
    elif kind == "refill":
        load_profile = np.array([60.0, 60.0, 10.0, 10.0, 10.0, 10.0])
        cheap_avail = np.zeros(T)
        ptc_avail = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    else:  # pragma: no cover - guards a typo in a test
        raise ValueError(kind)

    generators = zap.Generator(
        num_nodes=2,
        name=np.array(["cheap", "peaker", "ptc"]),
        terminal=np.array([0, 0, 0]),
        nominal_capacity=np.array([100.0, 100.0, 100.0]),
        dynamic_capacity=np.vstack([cheap_avail, np.ones(T), ptc_avail]),
        linear_cost=np.array([0.5, 200.0, -50.0]),
        emission_rates=np.array([0.4, 0.6, 0.0]),
    )
    load = zap.Load(
        num_nodes=2,
        name=np.array(["l0"]),
        terminal=np.array([1]),
        load=load_profile.reshape(1, T),
        linear_cost=np.array([5000.0]),
    )
    line = zap.DCLine(
        num_nodes=2,
        name=np.array(["ln0"]),
        source_terminal=np.array([0]),
        sink_terminal=np.array([1]),
        capacity=np.array([500.0]),
        linear_cost=np.array([0.0]),
    )
    battery = storage_cls(
        num_nodes=2,
        name=np.array(["b0", "b1"]),
        terminal=np.array([1, 1]),
        power_capacity=np.array([20.0, 5.0]),
        duration=np.array([2.0, 4.0]),
        charge_efficiency=np.array([1.0, 1.0]),
        discharge_efficiency=np.array([1.0, 1.0]),
        # A small discharge cost, so a window never dumps energy it has no reason
        # to dump: without it the commit-boundary level is degenerate.
        linear_cost=np.array([1.0, 1.0]),
        initial_soc=np.array([0.5, 0.5]),
        final_soc=np.array([0.5, 0.5]),
        soc_mode=soc_mode,
        **({} if storage_cls is not StorageUnit else {"soc_anchor_index": soc_anchor_index}),
    )
    return net, [generators, load, line, battery], T


def solve(kind, soc_mode="rolling", soc_anchor_index=0):
    net, devices, horizon = rolling_fixture(kind, soc_mode, soc_anchor_index)
    outcome = net.dispatch(devices, time_horizon=horizon, solver=cp.HIGHS, add_ground=False)
    assert outcome.problem.status == cp.OPTIMAL, outcome.problem.status
    energy = np.asarray(outcome.local_variables[STORAGE][0])
    return float(outcome.problem.value), energy


def toy_storage(soc_mode="rolling", soc_anchor_index=0, num_rows=2):
    """A bare `StorageUnit`, no network, for the constraint-structure tests."""
    return StorageUnit(
        num_nodes=1,
        name=np.array([f"b{i}" for i in range(num_rows)]),
        terminal=np.zeros(num_rows, dtype=int),
        power_capacity=np.full(num_rows, 10.0),
        duration=np.full(num_rows, 3.0),
        charge_efficiency=np.full(num_rows, 0.9),
        discharge_efficiency=np.full(num_rows, 0.8),
        linear_cost=np.full(num_rows, 1.0),
        initial_soc=np.full(num_rows, 0.4),
        final_soc=np.full(num_rows, 0.6),
        soc_mode=soc_mode,
        soc_anchor_index=soc_anchor_index,
    )


def numeric_state(device, horizon, seed=0):
    """A numeric ``(power, state)`` pair for evaluating the constraint residuals."""
    rng = np.random.default_rng(seed)
    n = device.num_devices
    energy = rng.uniform(0.0, 30.0, (n, horizon + 1))
    charge = rng.uniform(0.0, 10.0, (n, horizon))
    discharge = rng.uniform(0.0, 10.0, (n, horizon))
    power = [rng.uniform(-10.0, 10.0, (n, horizon))]
    from zap.devices.storage_unit import StorageUnitVariable

    return power, StorageUnitVariable(energy, charge, discharge)


# =====
# Construction and the copy paths
# =====


class TestRollingConstruction(unittest.TestCase):
    def test_rolling_is_an_accepted_mode(self):
        device = toy_storage("rolling", soc_anchor_index=3)
        self.assertEqual(device.soc_mode, "rolling")
        self.assertEqual(device.soc_anchor_index, 3)

    def test_anchor_index_defaults_to_zero(self):
        self.assertEqual(toy_storage("fixed").soc_anchor_index, 0)
        self.assertEqual(StorageUnit.soc_anchor_index, 0)

    def test_unknown_mode_is_still_refused_and_names_rolling(self):
        with self.assertRaises(ValueError) as ctx:
            toy_storage("cyclic")
        self.assertIn("rolling", str(ctx.exception))

    def test_anchor_index_must_be_a_non_negative_integer(self):
        with self.assertRaises(ValueError):
            toy_storage("rolling", soc_anchor_index=-1)
        with self.assertRaises(ValueError):
            toy_storage("rolling", soc_anchor_index=2.5)
        with self.assertRaises(ValueError):
            toy_storage("rolling", soc_anchor_index=True)

    def test_anchor_index_survives_the_copy_paths(self):
        device = toy_storage("rolling", soc_anchor_index=3)

        sampled = device.sample_time(np.arange(4), 8)
        self.assertEqual(sampled.soc_anchor_index, 3)
        self.assertEqual(sampled.soc_mode, "rolling")

        torched = device.torchify(machine="cpu", dtype=torch.float64)
        self.assertEqual(torched.soc_anchor_index, 3)
        self.assertEqual(torched.soc_mode, "rolling")
        # `torchify` walks __dict__ and converts arrays; the anchor must stay an
        # int, not become a 0-d tensor.
        self.assertIsInstance(torched.soc_anchor_index, int)

        scaled = device.sample_time(np.arange(T), T)
        scaled.scale_power(10.0)
        scaled.scale_costs(2.0)
        self.assertEqual(scaled.soc_anchor_index, 3)

        from copy import deepcopy

        self.assertEqual(deepcopy(device).soc_anchor_index, 3)

    def test_anchor_index_survives_the_network_dispatch_copy(self):
        """The device the harness hands to `dispatch` is the one that is sliced."""
        _net, devices, horizon = rolling_fixture("drain", "rolling", soc_anchor_index=3)
        battery = devices[STORAGE]
        sliced = battery.sample_time(np.arange(horizon), horizon)
        self.assertEqual(sliced.soc_anchor_index, 3)
        self.assertEqual(sliced.torchify(machine="cpu", dtype=torch.float64).soc_anchor_index, 3)


# =====
# Constraint structure
# =====


class TestRollingConstraintStructure(unittest.TestCase):
    def test_equality_constraints_have_three_blocks_and_pin_only_the_opening(self):
        device = toy_storage("rolling", soc_anchor_index=3)
        power, state = numeric_state(device, T)
        residuals = device.equality_constraints(power, None, state, la=np)

        self.assertEqual(len(residuals), 3)
        # Same count as cyclic_free, so ch3's "skip residuals[0], sum the rest"
        # storage-residual gate needs no mode switch.
        free = toy_storage("cyclic_free")
        self.assertEqual(len(free.equality_constraints(power, None, state, la=np)), 3)
        self.assertEqual(
            len(toy_storage("fixed").equality_constraints(power, None, state, la=np)), 4
        )

        emax = np.asarray(device.power_capacity) * np.asarray(device.duration)
        expected = state.energy[:, 0:1] - np.asarray(device.initial_soc) * emax
        np.testing.assert_allclose(np.asarray(residuals[2]), expected, atol=0, rtol=0)

        # The first two blocks are the shared ones and are untouched by the mode.
        for i in (0, 1):
            np.testing.assert_array_equal(
                np.asarray(residuals[i]),
                np.asarray(free.equality_constraints(power, None, state, la=np)[i]),
            )

    def test_terminal_rule_is_appended_as_inequality_block_six(self):
        device = toy_storage("rolling", soc_anchor_index=3)
        power, state = numeric_state(device, T)

        rolling = device.inequality_constraints(power, None, state, la=np)
        self.assertEqual(len(rolling), 7)

        # Blocks 0-5 keep their meaning and their position: identical, entry by
        # entry, to the same fixture in `fixed` mode.
        baseline = toy_storage("fixed").inequality_constraints(power, None, state, la=np)
        self.assertEqual(len(baseline), 6)
        for i in range(6):
            np.testing.assert_array_equal(np.asarray(rolling[i]), np.asarray(baseline[i]))

        # ... and the new block is last: energy[:, k] - energy[:, T] <= 0.
        expected = state.energy[:, 3:4] - state.energy[:, T : (T + 1)]
        np.testing.assert_array_equal(np.asarray(rolling[6]), expected)
        self.assertEqual(np.asarray(rolling[6]).shape, (device.num_devices, 1))

    def test_anchor_zero_gives_the_conservative_rule(self):
        device = toy_storage("rolling", soc_anchor_index=0)
        power, state = numeric_state(device, T)
        blocks = device.inequality_constraints(power, None, state, la=np)
        np.testing.assert_array_equal(
            np.asarray(blocks[6]), state.energy[:, 0:1] - state.energy[:, T : (T + 1)]
        )

    def test_anchor_at_the_horizon_is_the_trivial_rule(self):
        device = toy_storage("rolling", soc_anchor_index=T)
        power, state = numeric_state(device, T)
        blocks = device.inequality_constraints(power, None, state, la=np)
        np.testing.assert_allclose(np.asarray(blocks[6]), 0.0, atol=0, rtol=0)

    def test_anchor_past_the_horizon_is_refused(self):
        device = toy_storage("rolling", soc_anchor_index=T + 1)
        power, state = numeric_state(device, T)
        with self.assertRaises(ValueError) as ctx:
            device.inequality_constraints(power, None, state, la=np)
        self.assertIn("soc_anchor_index", str(ctx.exception))

    def test_other_modes_still_have_six_inequality_blocks(self):
        power, state = numeric_state(toy_storage("fixed"), T)
        for mode in ("fixed", "cyclic_free"):
            device = toy_storage(mode)
            self.assertEqual(len(device.inequality_constraints(power, None, state, la=np)), 6)


# =====
# The gates
# =====


class TestRollingGates(unittest.TestCase):
    def test_equality_matrices_refuse_rolling(self):
        device = toy_storage("rolling")
        with self.assertRaises(NotImplementedError) as ctx:
            device._equality_matrices([])
        self.assertIn("dispatch-only", str(ctx.exception))

    def test_inequality_matrices_refuse_rolling(self):
        device = toy_storage("rolling")
        with self.assertRaises(NotImplementedError) as ctx:
            device._inequality_matrices([])
        self.assertIn("dispatch-only", str(ctx.exception))

    def test_differentiation_matrices_still_work_for_the_other_modes(self):
        """The gate must not fire for `fixed` / `cyclic_free`: an empty list of
        constraint objects gets past the mode check and fails later instead."""
        for mode in ("fixed", "cyclic_free"):
            device = toy_storage(mode)
            with self.assertRaises(Exception) as ctx:
                device._equality_matrices([])
            self.assertNotIsInstance(ctx.exception, NotImplementedError)

    def test_windowed_soc_layout_refuses_rolling(self):
        device = toy_storage("rolling", soc_anchor_index=3)
        power, state = numeric_state(device, T)
        # The windowed ADMM layout: S trajectories of W + 1 slots, T + S columns.
        num_windows = 3
        windowed = type(state)(
            np.zeros((device.num_devices, T + num_windows)), state.charge, state.discharge
        )
        self.assertEqual(StorageUnit.num_soc_windows(T + num_windows, T), num_windows)

        with self.assertRaises(NotImplementedError) as ctx:
            device.equality_constraints(power, None, windowed, la=np)
        self.assertIn("windowed", str(ctx.exception))

        with self.assertRaises(NotImplementedError):
            device.inequality_constraints(power, None, windowed, la=np)

    def test_admm_prox_refuses_rolling(self):
        device = toy_storage("rolling").torchify(machine="cpu", dtype=torch.float64)
        z = torch.zeros((device.num_devices, T), dtype=torch.float64)
        with self.assertRaises(NotImplementedError) as ctx:
            device.admm_prox_update(1.0, 1.0, [z], None, inner_iterations=2)
        self.assertIn("WP-R2", str(ctx.exception))


# =====
# The LP: what the terminal rule does
# =====


class TestRollingDispatch(unittest.TestCase):
    def test_the_opening_pin_binds(self):
        for kind in ("drain", "refill"):
            for anchor in (0, 3):
                with self.subTest(kind=kind, anchor=anchor):
                    _obj, energy = solve(kind, "rolling", anchor)
                    np.testing.assert_allclose(energy[:, 0], OPENING_MWH, atol=1e-9)

        # ... and it is a real restriction: `cyclic_free`, free to pick its own
        # level on the same fixture, picks a different one.
        _obj, free = solve("drain", "cyclic_free")
        self.assertGreater(float(np.max(np.abs(free[:, 0] - OPENING_MWH))), 1.0)

    def test_terminal_rule_binds_when_a_free_end_would_drain(self):
        free_obj, free_energy = solve("drain", "rolling", soc_anchor_index=T)
        anchored_obj, anchored = solve("drain", "rolling", soc_anchor_index=3)

        # A free end (the trivial anchor k = T) drains past the commit boundary:
        # the window closes empty, having carried as much energy to hour 3 as it
        # can discharge over hours 3-5 (row 0 fills to its 40 MWh cap; row 1 can
        # only move 5 MW/h, so it carries 15 MWh and no more).
        np.testing.assert_allclose(free_energy[:, T], 0.0, atol=1e-9)
        np.testing.assert_allclose(free_energy[:, 3], np.array([40.0, 15.0]), atol=1e-9)

        # The terminal rule is active at the anchored optimum, at a strictly
        # positive level, and it costs money.
        np.testing.assert_allclose(anchored[:, T] - anchored[:, 3], 0.0, atol=1e-9)
        self.assertTrue(np.all(anchored[:, 3] > 1.0))
        self.assertGreater(anchored_obj, free_obj + 1.0)

        # The conservative k = 0 rule sits strictly between the two: it only
        # forbids closing below the opening level.
        zero_obj, zero_energy = solve("drain", "rolling", soc_anchor_index=0)
        np.testing.assert_allclose(zero_energy[:, T], OPENING_MWH, atol=1e-9)
        np.testing.assert_allclose(zero_energy[:, 3], EMAX_MWH, atol=1e-9)
        self.assertGreater(zero_obj, free_obj + 1.0)
        self.assertGreater(anchored_obj, zero_obj + 1.0)

    def test_terminal_rule_is_slack_when_the_window_refills(self):
        free_obj, _ = solve("refill", "rolling", soc_anchor_index=T)
        anchored_obj, anchored = solve("refill", "rolling", soc_anchor_index=3)

        slack = anchored[:, T] - anchored[:, 3]
        self.assertTrue(np.all(slack > 1e-6), slack)
        # Slack means costless: the anchored window is the free-end window.
        self.assertAlmostEqual(anchored_obj, free_obj, places=6)

    def test_rolling_is_a_restriction_of_a_free_opening(self):
        """`cyclic_free` is not comparable, but the free-end rolling window is a
        relaxation of both anchored ones."""
        free_obj, _ = solve("drain", "rolling", soc_anchor_index=T)
        for anchor in (0, 3):
            obj, _ = solve("drain", "rolling", anchor)
            self.assertGreaterEqual(obj, free_obj - 1e-6)


# =====
# `fixed` and `cyclic_free` are untouched
# =====


class TestExistingModesUnchanged(unittest.TestCase):
    """Values captured from the pre-WP-R0b `storage_unit.py` (git HEAD at the time
    of the change), solved with HiGHS through `PowerNetwork.dispatch`."""

    #: (kind, soc_mode) -> (objective, energy trajectory per row)
    BASELINE = {
        ("drain", "fixed"): (
            30060.0,
            [
                [20.0, 20.0, 20.0, 40.0, 20.0, 20.0, 20.0],
                [10.0, 10.0, 15.0, 20.0, 20.0, 15.0, 10.0],
            ],
        ),
        ("drain", "cyclic_free"): (
            25097.5,
            [
                [0.0, 0.0, 20.0, 40.0, 20.0, 0.0, 0.0],
                [0.0, 5.0, 10.0, 15.0, 10.0, 5.0, 0.0],
            ],
        ),
        ("refill", "fixed"): (
            19530.0,
            [
                [20.0, 20.0, 0.0, 0.0, 0.0, 20.0, 20.0],
                [10.0, 5.0, 0.0, 0.0, 0.0, 5.0, 10.0],
            ],
        ),
        ("refill", "cyclic_free"): (
            14550.0,
            [
                [40.0, 20.0, 10.0, 0.0, 0.0, 20.0, 40.0],
                [10.0, 5.0, 0.0, 0.0, 0.0, 5.0, 10.0],
            ],
        ),
    }

    def test_objectives_and_trajectories_match_the_pre_change_values(self):
        for (kind, mode), (obj, energy) in self.BASELINE.items():
            with self.subTest(kind=kind, mode=mode):
                got_obj, got_energy = solve(kind, mode)
                self.assertAlmostEqual(got_obj, obj, places=6)
                np.testing.assert_allclose(got_energy, np.asarray(energy), atol=1e-8)

    def test_the_boundary_conditions_themselves_are_unchanged(self):
        _, fixed_energy = solve("drain", "fixed")
        np.testing.assert_allclose(fixed_energy[:, 0], OPENING_MWH, atol=1e-9)
        np.testing.assert_allclose(fixed_energy[:, T], OPENING_MWH, atol=1e-9)

        _, free_energy = solve("drain", "cyclic_free")
        np.testing.assert_allclose(free_energy[:, 0], free_energy[:, T], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
