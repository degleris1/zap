"""WP-R0 of ``memory/plans/2026-09-15-rolling-horizon-evaluator-spec.md``.

Tests 1-4 of section 9: a dispatch problem built once with ``cp.Parameter``s and
re-solved on new data must equal a rebuilt problem, must be DPP, must default to
``warm_start=False``, and must not disturb ``PowerNetwork.dispatch``.

Everything runs on hand-built, ground-free networks solved with HiGHS; nothing
reads ``data/``.
"""

import inspect
import time
import unittest
from unittest import mock

import cvxpy as cp
import numpy as np

from zap.devices import Generator, Injector, Load, StorageUnit
from zap.devices.transporter import ACLine, DirectedLine
from zap.importers.multi_year import PARAMETRIZABLE_ATTRS as REEXPORTED_ATTRS
from zap.importers.multi_year import TIME_VARYING_ATTRS
from zap.network import (
    PARAMETRIZABLE_ATTRS,
    DispatchOutcome,
    DispatchProblem,
    PowerNetwork,
    parametrize_devices,
)

SOLVER = cp.HIGHS
VOLL = 10_000.0

#: Device indices of the small fixture, and what the rolling horizon varies.
GENERATORS, LOADS, STORAGE, LINES = 0, 1, 2, 3
ROLLING_PARAMETRIZE = {
    GENERATORS: ["dynamic_capacity"],
    LOADS: ["load"],
    STORAGE: ["power_availability", "initial_soc"],
}


# ---------------------------------------------------------------------------
# Fixture: a 4-bus, ground-free system with a unique optimum
# ---------------------------------------------------------------------------


def small_system(*, hours=6, load=None, gen_availability=None, storage_availability=None,
                 initial_soc=None, soc_mode="fixed", soc_terminal_value=None):
    """A 4-bus system: two loads, three generators, two batteries, six directed lines.

    Bus 0 carries the big load and the cheap generator; bus 1 the expensive one
    behind a 30 MW line; bus 2 only a battery; bus 3 a mid-priced generator and a
    small load.  Costs are all distinct and every line carries a small, distinct
    linear cost, so the LP optimum is a unique vertex -- which is what lets the
    parametric and rebuilt problems be compared on the *primal*, not only on the
    objective.

    Every argument defaults to the baseline value; the tests vary them to make
    three different parameter settings.
    """
    T = hours
    ones = np.ones((1, T))

    if load is None:
        load = np.vstack(
            [
                60.0 + 8.0 * np.arange(T),
                10.0 + 2.0 * np.arange(T),
            ]
        )
    if gen_availability is None:
        gen_availability = np.ones((3, T))
    if storage_availability is None:
        storage_availability = np.ones((2, T))
    if initial_soc is None:
        initial_soc = np.array([[0.5], [0.4]])

    network = PowerNetwork(4)

    generators = Generator(
        num_nodes=4,
        name=np.array(["g_cheap", "g_expensive", "g_mid"], dtype=object),
        terminal=np.array([0, 1, 3]),
        nominal_capacity=np.array([[60.0], [100.0], [40.0]]),
        dynamic_capacity=np.asarray(gen_availability, dtype=float),
        linear_cost=np.array([[20.0], [80.0], [40.0]]) * ones,
        emission_rates=np.zeros((3, 1)),
    )
    loads = Load(
        num_nodes=4,
        name=np.array(["l_b0", "l_b3"], dtype=object),
        terminal=np.array([0, 3]),
        load=np.asarray(load, dtype=float),
        linear_cost=VOLL * np.ones((2, 1)),
    )
    storage = StorageUnit(
        num_nodes=4,
        name=np.array(["batt_b2", "batt_b0"], dtype=object),
        terminal=np.array([2, 0]),
        power_capacity=np.array([[20.0], [10.0]]),
        duration=np.array([[4.0], [2.0]]),
        charge_efficiency=np.array([[0.95], [0.90]]),
        discharge_efficiency=np.array([[0.95], [0.90]]),
        initial_soc=np.asarray(initial_soc, dtype=float),
        final_soc=np.array([[0.5], [0.4]]),
        linear_cost=np.array([[0.05], [0.06]]),
        power_availability=np.asarray(storage_availability, dtype=float),
        # `fixed` so that `initial_soc` actually enters the problem: under
        # `cyclic_free` it is ignored and the parameter would be inert.
        # `rolling` (WP-R1c) also pins the opening level, and is what the
        # `soc_terminal_value` parameter needs.
        soc_mode=soc_mode,
        soc_terminal_value=soc_terminal_value,
    )
    lines = DirectedLine(
        num_nodes=4,
        name=np.array(["b1_b0", "b0_b1", "b2_b0", "b0_b2", "b0_b3", "b3_b0"], dtype=object),
        source_terminal=np.array([1, 0, 2, 0, 0, 3]),
        sink_terminal=np.array([0, 1, 0, 2, 3, 0]),
        min_power=np.zeros((6, 1)),
        max_power=np.array([[30.0], [30.0], [50.0], [50.0], [40.0], [40.0]]),
        linear_cost=np.array([[0.01], [0.02], [0.03], [0.04], [0.05], [0.06]]),
    )

    return network, [generators, loads, storage, lines]


def rolling_settings(hours=6):
    """Three different (load, availability, SoC seed) settings, as a rolling loop makes."""
    T = hours
    rng = np.random.default_rng(20260915)
    settings = []
    for k in range(3):
        load = np.vstack(
            [
                55.0 + 10.0 * np.arange(T) + 5.0 * k,
                8.0 + 3.0 * np.arange(T) - 1.5 * k,
            ]
        )
        gen_availability = np.clip(rng.uniform(0.3, 1.0, size=(3, T)), 0.0, 1.0)
        storage_availability = np.clip(rng.uniform(0.5, 1.0, size=(2, T)), 0.0, 1.0)
        # A scarce setting in the middle, so at least one solve sheds at VOLL.
        if k == 1:
            gen_availability[:, :] = 0.35
        soc = np.array([[0.2 + 0.3 * k], [0.7 - 0.2 * k]])
        settings.append(
            {
                "load": load,
                "gen_availability": gen_availability,
                "storage_availability": storage_availability,
                "initial_soc": soc,
            }
        )
    return settings


def parameter_values(setting):
    return {
        (GENERATORS, "dynamic_capacity"): setting["gen_availability"],
        (LOADS, "load"): setting["load"],
        (STORAGE, "power_availability"): setting["storage_availability"],
        (STORAGE, "initial_soc"): setting["initial_soc"],
    }


def _flatten(value):
    """Every numpy leaf of a (possibly nested, possibly ``None``) outcome field."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [value.ravel()]
    leaves = []
    for item in value:
        leaves += _flatten(item)
    return leaves


# ---------------------------------------------------------------------------
# Test 1 -- parametric == rebuilt
# ---------------------------------------------------------------------------


class ParametricEqualsRebuiltTests(unittest.TestCase):
    """Section 9 test 1: three settings in a row through the *same* problem object."""

    TOL = 1e-9

    def test_three_settings_in_a_row(self):
        hours = 6
        network, devices = small_system(hours=hours)
        problem = network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        )

        sheds = 0
        for k, setting in enumerate(rolling_settings(hours)):
            with self.subTest(setting=k):
                problem.set_parameters(parameter_values(setting))
                got = problem.solve(solver=SOLVER)

                ref_network, ref_devices = small_system(hours=hours, **setting)
                want = ref_network.dispatch(
                    ref_devices, time_horizon=hours, solver=SOLVER, add_ground=False
                )

                scale = max(1.0, abs(want.problem.value))
                self.assertAlmostEqual(
                    got.problem.value, want.problem.value, delta=self.TOL * scale
                )
                for field in ("power", "prices", "local_variables", "angle"):
                    got_leaves = _flatten(getattr(got, field))
                    want_leaves = _flatten(getattr(want, field))
                    self.assertEqual(len(got_leaves), len(want_leaves), field)
                    for a, b in zip(got_leaves, want_leaves):
                        np.testing.assert_allclose(a, b, atol=self.TOL, rtol=0.0, err_msg=field)

                if want.problem.value > 1e5:  # VOLL x ENS dominates
                    sheds += 1

        self.assertGreaterEqual(sheds, 1, "no setting exercised the shedding regime")

    def test_unset_parameters_reproduce_the_baseline(self):
        """A parametrised problem that is never given new values is the same problem."""
        hours = 6
        network, devices = small_system(hours=hours)
        parametric = network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        ).solve(solver=SOLVER)
        plain = network.dispatch(devices, time_horizon=hours, solver=SOLVER, add_ground=False)
        self.assertAlmostEqual(parametric.problem.value, plain.problem.value, places=9)

    def test_devices_are_copies_and_the_originals_are_untouched(self):
        hours = 6
        network, devices = small_system(hours=hours)
        before = devices[LOADS].load.copy()
        problem = network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        )
        problem.set_parameters({(LOADS, "load"): np.full((2, hours), 1.0)})

        np.testing.assert_array_equal(devices[LOADS].load, before)
        self.assertIsNot(problem.devices[LOADS], devices[LOADS])
        self.assertIs(problem.devices[LINES], devices[LINES])  # untouched device, not copied
        self.assertIsInstance(problem.devices[LOADS].load, cp.Parameter)

    def test_set_parameters_rejects_unknown_keys_and_bad_shapes(self):
        hours = 6
        network, devices = small_system(hours=hours)
        problem = network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        )
        with self.assertRaises(KeyError):
            problem.set_parameters({(LINES, "max_power"): np.zeros((6, 1))})
        with self.assertRaises(ValueError):
            problem.set_parameters({(LOADS, "load"): np.zeros((3, hours))})

        # A (N,) seed reshapes onto the (N, 1) parameter, as `make_dynamic` does.
        problem.set_parameters({(STORAGE, "initial_soc"): np.array([0.3, 0.3])})
        np.testing.assert_allclose(
            problem.parameters[(STORAGE, "initial_soc")].value, np.array([[0.3], [0.3]])
        )

    def test_set_parameters_never_broadcasts_over_time(self):
        """A (N,) row into an (N, T) parameter is a typo, not a per-row constant."""
        hours = 3  # == the generator row count, so a (3,) array would broadcast
        network, devices = small_system(hours=hours)
        problem = network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        )
        self.assertEqual(tuple(problem.parameters[(GENERATORS, "dynamic_capacity")].shape), (3, 3))

        for bad in (np.ones(3), np.ones((3, 1)), np.ones((1, 3))):
            with self.assertRaises(ValueError) as ctx:
                problem.set_parameters({(GENERATORS, "dynamic_capacity"): bad})
            self.assertIn("never broadcast", str(ctx.exception))

    def test_parameters_do_not_alias_the_callers_arrays(self):
        """Neither direction: the device's array, nor the array handed to set_parameters."""
        hours = 6
        network, devices = small_system(hours=hours)
        problem = network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        )
        param = problem.parameters[(LOADS, "load")]
        self.assertFalse(np.shares_memory(param.value, devices[LOADS].load))

        # Editing the original device in place must not reach the retained problem.
        before = problem.solve(solver=SOLVER).objective
        devices[LOADS].load[:] = devices[LOADS].load * 3.0
        after = problem.solve(solver=SOLVER).objective
        self.assertAlmostEqual(before, after, places=9)

        # And the array handed to set_parameters is copied, not captured.
        seed = np.array([[0.30], [0.30]])
        problem.set_parameters({(STORAGE, "initial_soc"): seed})
        seed[:] = 0.9
        np.testing.assert_allclose(
            problem.parameters[(STORAGE, "initial_soc")].value, np.array([[0.3], [0.3]])
        )


# ---------------------------------------------------------------------------
# The per-solve snapshot: one retained cp.Problem, many outcomes
# ---------------------------------------------------------------------------


class OutcomeSnapshotTests(unittest.TestCase):
    """`problem` is shared, so `problem.value` is not a property of an outcome."""

    def test_an_earlier_outcome_keeps_its_own_objective(self):
        hours = 6
        network, devices = small_system(hours=hours)
        problem = network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        )
        settings = rolling_settings(hours)

        problem.set_parameters(parameter_values(settings[0]))
        first = problem.solve(solver=SOLVER)
        first_objective = first.objective
        first_stats = first.solver_stats

        problem.set_parameters(parameter_values(settings[1]))
        second = problem.solve(solver=SOLVER)

        # The two settings must actually differ, or the test proves nothing.
        self.assertNotAlmostEqual(first_objective, second.objective, delta=1.0)

        # The live problem has moved on; the snapshot has not.
        self.assertAlmostEqual(second.problem.value, second.objective, places=9)
        self.assertAlmostEqual(first.objective, first_objective, places=12)
        self.assertNotAlmostEqual(first.problem.value, first.objective, delta=1.0)
        self.assertIs(first.solver_stats, first_stats)
        self.assertIsNot(first.solver_stats, second.solver_stats)
        self.assertEqual(first.status, cp.OPTIMAL)
        self.assertEqual(first.n_variables, second.n_variables)
        self.assertGreater(first.n_variables, 0)
        self.assertGreater(first.n_constraints, 0)

    def test_dispatch_populates_the_snapshot_too(self):
        hours = 6
        network, devices = small_system(hours=hours)
        outcome = network.dispatch(devices, time_horizon=hours, solver=SOLVER, add_ground=False)
        self.assertAlmostEqual(outcome.objective, outcome.problem.value, places=12)
        self.assertEqual(outcome.status, outcome.problem.status)
        self.assertIs(outcome.solver_stats, outcome.problem.solver_stats)

    def test_the_new_fields_do_not_disturb_the_sequence_protocol(self):
        """`vectorize` / `shape` / `blocks` / `torchify` iterate the first 8 fields only."""
        hours = 6
        network, devices = small_system(hours=hours)
        outcome = network.dispatch(devices, time_horizon=hours, solver=SOLVER, add_ground=False)
        self.assertEqual(len(outcome), 8)
        self.assertEqual(len(list(outcome)), 8)
        vector = outcome.vectorize()
        self.assertEqual(vector.size, outcome.size)
        self.assertIsNotNone(outcome.shape)
        self.assertIsNotNone(outcome.blocks)
        self.assertIsNotNone(outcome.torchify())
        np.testing.assert_allclose(outcome.package(vector).power[0], outcome.power[0])

    def test_admm_style_construction_leaves_the_snapshot_none(self):
        """`ADMMState.as_outcome` builds by keyword and names none of the new fields."""
        outcome = DispatchOutcome(
            phase_duals=None,
            local_equality_duals=None,
            local_inequality_duals=None,
            local_variables=None,
            power=None,
            angle=None,
            prices=None,
            global_angle=None,
            problem=None,
            ground=None,
        )
        self.assertIsNone(outcome.objective)
        self.assertIsNone(outcome.status)
        self.assertIsNone(outcome.solver_stats)
        self.assertIsNone(outcome.n_variables)
        self.assertIsNone(outcome.n_constraints)


# ---------------------------------------------------------------------------
# Test 2 -- DPP
# ---------------------------------------------------------------------------


class DPPTests(unittest.TestCase):
    """Section 9 test 2: the rolling set is DPP; a parameter product is refused."""

    def test_rolling_set_is_dpp(self):
        hours = 6
        network, devices = small_system(hours=hours)
        problem = network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        )
        self.assertTrue(problem.is_dpp())
        self.assertTrue(problem.problem.is_dpp())
        self.assertEqual(
            sorted(problem.parameters),
            [
                (GENERATORS, "dynamic_capacity"),
                (LOADS, "load"),
                (STORAGE, "initial_soc"),
                (STORAGE, "power_availability"),
            ],
        )

    def test_capacity_beside_availability_raises(self):
        """`multiply(power_capacity, power_availability)` would be parameter x parameter."""
        hours = 6
        network, devices = small_system(hours=hours)
        with self.assertRaises(ValueError) as ctx:
            network.build_dispatch(
                devices,
                time_horizon=hours,
                add_ground=False,
                parametrize={STORAGE: ["power_capacity", "power_availability"]},
            )
        self.assertIn("DPP", str(ctx.exception))

    def test_injector_cost_beside_capacity_raises(self):
        """`Generator.min_power` is `0 * dynamic_capacity`, and cost multiplies it."""
        hours = 6
        network, devices = small_system(hours=hours)
        with self.assertRaises(ValueError):
            network.build_dispatch(
                devices,
                time_horizon=hours,
                add_ground=False,
                parametrize={GENERATORS: ["dynamic_capacity", "linear_cost"]},
            )

    def test_parametrize_rejects_bad_names_and_planning_parameters(self):
        hours = 6
        network, devices = small_system(hours=hours)
        with self.assertRaises(ValueError) as ctx:
            network.build_dispatch(
                devices, time_horizon=hours, add_ground=False,
                parametrize={GENERATORS: ["quadratic_cost"]},  # not in the registry
            )
        self.assertIn("PARAMETRIZABLE_ATTRS", str(ctx.exception))
        with self.assertRaises(IndexError):
            network.build_dispatch(
                devices, time_horizon=hours, add_ground=False, parametrize={99: ["load"]}
            )
        with self.assertRaises(ValueError):
            network.build_dispatch(
                devices,
                time_horizon=hours,
                add_ground=False,
                parameters=[{} for _ in devices],
                parametrize=ROLLING_PARAMETRIZE,
            )

    def test_parametrize_refuses_an_unregistered_device_type(self):
        sink = Injector(
            num_nodes=2,
            name=np.array(["sink"], dtype=object),
            terminal=np.array([0]),
            min_power=np.zeros((1, 1)),
            max_power=np.ones((1, 1)),
            linear_cost=np.zeros((1, 1)),
        )
        with self.assertRaises(ValueError) as ctx:
            parametrize_devices([sink], {0: ["max_power"]})
        self.assertIn("PARAMETRIZABLE_ATTRS", str(ctx.exception))

    def test_parametrize_refuses_a_registered_attribute_that_is_none(self):
        line = DirectedLine(
            num_nodes=2,
            name=np.array(["ln"], dtype=object),
            source_terminal=np.array([0]),
            sink_terminal=np.array([1]),
            min_power=np.zeros((1, 1)),
            max_power=np.ones((1, 1)),
            linear_cost=None,
        )
        with self.assertRaises(ValueError) as ctx:
            parametrize_devices([line], {0: ["linear_cost"]})
        self.assertIn("is None", str(ctx.exception))

    def test_unused_parameters_are_named(self):
        """Under `cyclic_free` the SoC seed pins nothing, so writing it is a no-op."""
        hours = 6
        network, devices = small_system(hours=hours)
        devices[STORAGE].soc_mode = "cyclic_free"
        with self.assertLogs("zap.network", level="WARNING") as logs:
            problem = network.build_dispatch(
                devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
            )
        self.assertEqual(problem.unused_parameters, ((STORAGE, "initial_soc"),))
        self.assertIn("initial_soc", "\n".join(logs.output))

        devices[STORAGE].soc_mode = "fixed"
        quiet = network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        )
        self.assertEqual(quiet.unused_parameters, ())

    def test_registry_covers_the_rolling_set(self):
        self.assertEqual(PARAMETRIZABLE_ATTRS[Load], ["load"])
        self.assertIn("dynamic_capacity", PARAMETRIZABLE_ATTRS[Generator])
        self.assertEqual(
            PARAMETRIZABLE_ATTRS[StorageUnit],
            ["power_availability", "initial_soc", "soc_terminal_value"],
        )
        # One dict, two import paths: `multi_year` re-exports the definition.
        self.assertIs(REEXPORTED_ATTRS, PARAMETRIZABLE_ATTRS)
        # `initial_soc` and `soc_terminal_value` are (N, 1) per window, not
        # time-varying, so they stay out of the weather-year concatenation
        # registry (and out of `sample_time`'s slicing).
        self.assertNotIn("initial_soc", TIME_VARYING_ATTRS[StorageUnit])
        self.assertNotIn("soc_terminal_value", TIME_VARYING_ATTRS[StorageUnit])


# ---------------------------------------------------------------------------
# WP-R1c -- `StorageUnit.soc_terminal_value` as a per-window parameter
# ---------------------------------------------------------------------------


#: The rolling set once the terminal *value* replaces the terminal *rule*.
TERMINAL_VALUE_PARAMETRIZE = {
    GENERATORS: ["dynamic_capacity"],
    LOADS: ["load"],
    STORAGE: ["power_availability", "initial_soc", "soc_terminal_value"],
}


class TerminalValueParameterTests(unittest.TestCase):
    """Section 5.4: the price on stored energy changes per window, so it is a Parameter."""

    TOL = 1e-9
    HOURS = 6
    BASE_PRICE = np.array([[30.0], [20.0]])

    def _system(self, **setting):
        price = setting.pop("soc_terminal_value", self.BASE_PRICE)
        return small_system(
            hours=self.HOURS, soc_mode="rolling", soc_terminal_value=price, **setting
        )

    def test_the_parametrised_problem_is_dpp_and_uses_every_parameter(self):
        network, devices = self._system()
        problem = network.build_dispatch(
            devices,
            time_horizon=self.HOURS,
            add_ground=False,
            parametrize=TERMINAL_VALUE_PARAMETRIZE,
        )
        self.assertTrue(problem.is_dpp())
        self.assertEqual(problem.unused_parameters, ())

    def test_three_settings_in_a_row_equal_a_rebuild(self):
        network, devices = self._system()
        problem = network.build_dispatch(
            devices,
            time_horizon=self.HOURS,
            add_ground=False,
            parametrize=TERMINAL_VALUE_PARAMETRIZE,
        )

        prices = [np.array([[0.0], [0.0]]), np.array([[30.0], [20.0]]), np.array([[900.0], [5.0]])]
        terminal_levels = []
        for k, setting in enumerate(rolling_settings(self.HOURS)):
            with self.subTest(setting=k):
                values = parameter_values(setting)
                values[(STORAGE, "soc_terminal_value")] = prices[k]
                problem.set_parameters(values)
                got = problem.solve(solver=SOLVER)

                ref_network, ref_devices = self._system(soc_terminal_value=prices[k], **setting)
                want = ref_network.dispatch(
                    ref_devices, time_horizon=self.HOURS, solver=SOLVER, add_ground=False
                )
                scale = max(1.0, abs(want.problem.value))
                self.assertAlmostEqual(
                    got.problem.value, want.problem.value, delta=self.TOL * scale
                )
                for field in ("power", "prices", "local_variables"):
                    got_leaves = _flatten(getattr(got, field))
                    want_leaves = _flatten(getattr(want, field))
                    for a, b in zip(got_leaves, want_leaves):
                        np.testing.assert_allclose(a, b, atol=self.TOL, rtol=0.0, err_msg=field)
                energy = np.asarray(got.local_variables[STORAGE][0], dtype=float)
                terminal_levels.append(float(energy[0, self.HOURS]))

        # The parameter is not inert: row 0's 900 $/MWh setting ends fuller than
        # its 0 $/MWh one, through the same problem object.
        self.assertGreater(terminal_levels[2], terminal_levels[0] + 1e-6)


# ---------------------------------------------------------------------------
# Test 3 -- warm_start
# ---------------------------------------------------------------------------


def medium_system(*, hours=24, n_gen=24, n_load=6, n_storage=8, seed=0):
    """A bigger, still tiny, system: big enough that a solve is milliseconds, not microseconds."""
    rng = np.random.default_rng(seed)
    n_nodes = 6
    network = PowerNetwork(n_nodes)

    gen_terminal = rng.integers(0, n_nodes, size=n_gen)
    generators = Generator(
        num_nodes=n_nodes,
        name=np.array([f"g{i}" for i in range(n_gen)], dtype=object),
        terminal=gen_terminal,
        nominal_capacity=rng.uniform(20.0, 80.0, size=(n_gen, 1)),
        dynamic_capacity=rng.uniform(0.4, 1.0, size=(n_gen, hours)),
        linear_cost=np.linspace(10.0, 120.0, n_gen).reshape(-1, 1) * np.ones((1, hours)),
        emission_rates=np.zeros((n_gen, 1)),
    )
    loads = Load(
        num_nodes=n_nodes,
        name=np.array([f"l{i}" for i in range(n_load)], dtype=object),
        terminal=rng.integers(0, n_nodes, size=n_load),
        load=rng.uniform(30.0, 90.0, size=(n_load, hours)),
        linear_cost=VOLL * np.ones((n_load, 1)),
    )
    storage = StorageUnit(
        num_nodes=n_nodes,
        name=np.array([f"b{i}" for i in range(n_storage)], dtype=object),
        terminal=rng.integers(0, n_nodes, size=n_storage),
        power_capacity=rng.uniform(10.0, 40.0, size=(n_storage, 1)),
        duration=rng.uniform(2.0, 8.0, size=(n_storage, 1)),
        charge_efficiency=np.full((n_storage, 1), 0.93),
        discharge_efficiency=np.full((n_storage, 1), 0.93),
        initial_soc=np.full((n_storage, 1), 0.5),
        final_soc=np.full((n_storage, 1), 0.5),
        linear_cost=rng.uniform(0.01, 0.2, size=(n_storage, 1)),
        power_availability=np.ones((n_storage, hours)),
        soc_mode="fixed",
    )

    pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
    source = np.array([p[0] for p in pairs])
    sink = np.array([p[1] for p in pairs])
    lines = DirectedLine(
        num_nodes=n_nodes,
        name=np.array([f"ln{i}" for i in range(len(pairs))], dtype=object),
        source_terminal=source,
        sink_terminal=sink,
        min_power=np.zeros((len(pairs), 1)),
        max_power=np.full((len(pairs), 1), 60.0),
        linear_cost=np.linspace(0.01, 0.2, len(pairs)).reshape(-1, 1),
    )
    return network, [generators, loads, storage, lines]


class WarmStartTests(unittest.TestCase):
    """Section 9 test 3: `warm_start` is off by default on a retained problem."""

    def test_default_is_false(self):
        default = inspect.signature(DispatchProblem.solve).parameters["warm_start"].default
        self.assertIs(default, False)

    def test_false_reaches_cvxpy(self):
        hours = 6
        network, devices = small_system(hours=hours)
        problem = network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        )

        seen = []
        real_solve = cp.Problem.solve

        def spy(self, *args, **kwargs):
            seen.append(kwargs)
            return real_solve(self, *args, **kwargs)

        with mock.patch.object(cp.Problem, "solve", spy):
            problem.solve(solver=SOLVER)
            problem.solve(solver=SOLVER, warm_start=True)
            problem.solve(solver=SOLVER, solver_kwargs={"warm_start": True})

        self.assertIs(seen[0]["warm_start"], False)
        self.assertIs(seen[1]["warm_start"], True)
        self.assertIs(seen[2]["warm_start"], True)
        self.assertNotIn("warm_start", seen[2].get("solver_kwargs", {}))

    def test_resolve_is_not_slower_than_the_first_solve(self):
        """Coarse guard on the 30-40x trap of section 2, finding 1.

        Measured on *solver* time, so that the first solve's canonicalisation is
        not counted against the later ones.  Note the trap is scale-dependent:
        on a fixture this small HiGHS's ``setSolution`` crash start is actually
        mildly *faster*, and the pessimisation only appears on a real 48 h
        window (see ``ch3/ra/scripts/measure_parametric_dispatch.py``).  The
        binding guarantee is therefore :meth:`test_default_is_false` and
        :meth:`test_false_reaches_cvxpy`; this test catches a gross regression.
        """
        hours = 24
        network, devices = medium_system(hours=hours)
        problem = network.build_dispatch(
            devices, time_horizon=hours, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        )

        def solve_time():
            problem.solve(solver=SOLVER)
            reported = problem.problem.solver_stats.solve_time
            return float(reported) if reported else None

        start = time.perf_counter()
        first = solve_time()
        first_wall = time.perf_counter() - start
        if first is None:  # the solver did not report a time; fall back to the wall clock
            first = first_wall

        later = []
        for _ in range(4):
            start = time.perf_counter()
            reported = solve_time()
            later.append(reported if reported is not None else time.perf_counter() - start)

        self.assertLessEqual(
            float(np.median(later)),
            3.0 * first,
            f"re-solving a retained problem got slower (first {first:.4f} s, "
            f"later {sorted(later)}); is `warm_start` leaking back to True?",
        )


# ---------------------------------------------------------------------------
# Test 4 -- `dispatch` is unchanged
# ---------------------------------------------------------------------------


class DispatchUnchangedTests(unittest.TestCase):
    """Section 9 test 4, in-module: `dispatch` == `build_dispatch(...).solve(...)`."""

    def test_dispatch_matches_the_split(self):
        hours = 6
        network, devices = small_system(hours=hours)
        direct = network.dispatch(devices, time_horizon=hours, solver=SOLVER, add_ground=False)
        built = network.build_dispatch(devices, time_horizon=hours, add_ground=False)
        split = built.solve(solver=SOLVER)
        np.testing.assert_allclose(direct.vectorize(), split.vectorize(), atol=0.0, rtol=0.0)
        self.assertIsNone(split.ground)
        self.assertEqual(built.parameters, {})

    def test_ground_is_added_and_reported(self):
        hours = 6
        network, devices = small_system(hours=hours)
        built = network.build_dispatch(devices, time_horizon=hours, add_ground=True)
        self.assertIsNotNone(built.ground)
        # The ground is not part of `devices`; it rides along in its own field,
        # exactly as `DispatchOutcome` carries it.
        self.assertEqual(len(built.devices), len(devices))
        outcome = built.solve(solver=SOLVER)
        self.assertIs(outcome.ground, built.ground)

    def test_time_horizon_is_inferred_through_parameters(self):
        hours = 6
        network, devices = small_system(hours=hours)
        built = network.build_dispatch(
            devices, add_ground=False, parametrize=ROLLING_PARAMETRIZE
        )
        self.assertEqual(built.time_horizon, hours)

    def test_contingency_path_still_builds_and_solves(self):
        """The contingency branch lives in `build_dispatch`; keep it exercised."""
        hours = 3
        network = PowerNetwork(3)
        generators = Generator(
            num_nodes=3,
            name=np.array(["g0", "g1"], dtype=object),
            terminal=np.array([0, 1]),
            nominal_capacity=np.array([[100.0], [100.0]]),
            dynamic_capacity=np.ones((2, hours)),
            linear_cost=np.array([[10.0], [50.0]]) * np.ones((1, hours)),
            emission_rates=np.zeros((2, 1)),
        )
        loads = Load(
            num_nodes=3,
            name=np.array(["l2"], dtype=object),
            terminal=np.array([2]),
            load=np.array([[50.0, 60.0, 70.0]]),
            linear_cost=VOLL * np.ones((1, 1)),
        )
        lines = ACLine(
            num_nodes=3,
            name=np.array(["a", "b", "c"], dtype=object),
            source_terminal=np.array([0, 1, 0]),
            sink_terminal=np.array([2, 2, 1]),
            capacity=np.array([[40.0], [40.0], [40.0]]),
            susceptance=np.ones((3, 1)),
            nominal_capacity=np.ones((3, 1)),
            linear_cost=np.array([[0.01], [0.02], [0.03]]),
        )
        devices = [generators, loads, lines]
        mask = np.zeros((1, 3))
        mask[0, 0] = 1.0

        built = network.build_dispatch(
            devices,
            time_horizon=hours,
            num_contingencies=1,
            contingency_device=2,
            contingency_mask=mask,
        )
        outcome = built.solve(solver=SOLVER)
        self.assertIn(outcome.problem.status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        # Contingency layout: the contingency device's power is [base] + [per contingency].
        self.assertEqual(len(outcome.power[2]), 2)


if __name__ == "__main__":
    unittest.main()
