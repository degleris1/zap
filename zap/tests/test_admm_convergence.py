"""The ADMM convergence test must be in consistent units.

`rtol_dual_use_objective=True` compared a power-valued dual residual against
`rtol * objective` (dollars), so on a system with a large objective the dual
tolerance was orders of magnitude looser than the primal one and the solver
declared convergence far from the optimum (benchmark review, section 2.1).
"""

import math
import unittest
import warnings
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import torch

import zap
from zap.admm import ADMMSolver

torch.set_default_dtype(torch.float64)


def two_bus_system(T=12):
    """2 buses, a cheap capped generator, an expensive one, a line and a load.

    Costs are at power-system scale ($/MWh on GW-scale load), so the objective is
    ~1e7 -- the regime where the objective-scaled dual tolerance goes wrong.
    """
    net = zap.PowerNetwork(2)
    load_profile = np.array(
        [
            2000.0,
            2100.0,
            2300.0,
            2800.0,
            3400.0,
            3900.0,
            4000.0,
            3700.0,
            3200.0,
            2800.0,
            2400.0,
            2100.0,
        ]
    )[:T]

    generators = zap.Generator(
        num_nodes=2,
        name=np.array(["cheap", "peaker"]),
        terminal=np.array([0, 1]),
        nominal_capacity=np.array([3000.0, 4000.0]),
        dynamic_capacity=np.ones((2, T)),
        linear_cost=np.array([20.0, 150.0]),
        emission_rates=np.array([0.4, 0.6]),
    )
    load = zap.Load(
        num_nodes=2,
        name=np.array(["l0"]),
        terminal=np.array([1]),
        load=load_profile.reshape(1, T),
        linear_cost=np.array([10000.0]),
    )
    line = zap.DCLine(
        num_nodes=2,
        name=np.array(["ln0"]),
        source_terminal=np.array([0]),
        sink_terminal=np.array([1]),
        capacity=np.array([5000.0]),
        linear_cost=np.array([0.0]),
    )
    return net, [generators, load, line], T


def storage_system(T=12):
    """3 buses, 2 generators, 1 load, 2 *heterogeneous* StorageUnits.

    Deliberately tiny (T = 12) and deliberately heterogeneous: the inner prox is
    batched per unit, so a fleet whose units share power capacity / duration /
    efficiency cannot expose a per-unit bug (benchmark review 3.1).
    """
    net = zap.PowerNetwork(3)
    load_profile = np.array(
        [200.0, 210.0, 230.0, 280.0, 340.0, 390.0, 400.0, 370.0, 320.0, 280.0, 240.0, 210.0]
    )[:T]

    generators = zap.Generator(
        num_nodes=3,
        name=np.array(["cheap", "peaker"]),
        terminal=np.array([0, 1]),
        nominal_capacity=np.array([300.0, 400.0]),
        dynamic_capacity=np.ones((2, T)),
        linear_cost=np.array([20.0, 150.0]),
        emission_rates=np.array([0.4, 0.6]),
    )
    load = zap.Load(
        num_nodes=3,
        name=np.array(["l0"]),
        terminal=np.array([1]),
        load=load_profile.reshape(1, T),
        linear_cost=np.array([10000.0]),
    )
    lines = zap.DCLine(
        num_nodes=3,
        name=np.array(["ln0", "ln1"]),
        source_terminal=np.array([0, 1]),
        sink_terminal=np.array([1, 2]),
        capacity=np.array([500.0, 500.0]),
        linear_cost=np.array([0.0, 0.0]),
    )
    storage = zap.StorageUnit(
        num_nodes=3,
        name=np.array(["short", "long"]),
        terminal=np.array([1, 2]),
        power_capacity=np.array([40.0, 90.0]),
        duration=np.array([2.0, 8.0]),
        linear_cost=np.array([0.5, 1.5]),
        charge_efficiency=np.array([0.95, 0.87]),
        discharge_efficiency=np.array([0.93, 0.85]),
    )
    return net, [generators, load, lines, storage], T


def solve_admm(net, devices, T, **kwargs):
    torch_devices = [d.torchify(machine="cpu", dtype=torch.float64) for d in devices]
    settings = {
        "machine": "cpu",
        "dtype": torch.float64,
        "num_iterations": 20000,
        "rho_power": 1.0,
        "atol": 1e-4,
        "rtol": 1e-4,
        "verbose": 0,
    }
    settings.update(kwargs)
    solver = ADMMSolver(**settings)
    state, history = solver.solve(net, torch_devices, T)
    return solver, state, history


class TestADMMConvergence(unittest.TestCase):
    def test_declared_convergence_is_near_optimal(self):
        net, devices, T = two_bus_system()
        lp = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)

        solver, state, _ = solve_admm(net, devices, T)

        self.assertTrue(solver.converged, "solver did not converge; the test asserts nothing")
        gap = abs(float(state.objective) - lp.problem.value) / abs(lp.problem.value)
        self.assertLess(gap, 1e-3)

    def test_objective_scaled_dual_tolerance_is_loose_and_deprecated(self):
        """The old flag still works, but warns and stops much further from the optimum."""
        net, devices, T = two_bus_system()
        lp = net.dispatch(devices, time_horizon=T, solver=cp.HIGHS, add_ground=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver, state, _ = solve_admm(net, devices, T, rtol_dual_use_objective=True)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

        self.assertTrue(solver.converged)
        old_gap = abs(float(state.objective) - lp.problem.value) / abs(lp.problem.value)

        _, state_new, _ = solve_admm(net, devices, T)
        new_gap = abs(float(state_new.objective) - lp.problem.value) / abs(lp.problem.value)

        self.assertGreater(old_gap, new_gap)

    def test_history_records_tolerances(self):
        net, devices, T = two_bus_system()
        _, _, history = solve_admm(net, devices, T)

        self.assertEqual(len(history.primal_tol), len(history.power))
        self.assertEqual(len(history.dual_tol), len(history.power))
        self.assertTrue(np.all(np.isfinite(history.dual_tol)))

    def test_minimum_iterations_is_respected(self):
        net, devices, T = two_bus_system()
        solver, _, history = solve_admm(net, devices, T, minimum_iterations=250)
        self.assertGreaterEqual(len(history.power), 250)

    def test_rho_is_clamped(self):
        """The adaptive rule must not walk rho out of [rho_min, rho_max]."""
        net, devices, T = two_bus_system()
        solver, _state, _hist = solve_admm(
            net,
            devices,
            T,
            adaptive_rho=True,
            rho_min=0.5,
            rho_max=2.0,
            tau=2.0,
            adaptation_frequency=1,
            num_iterations=200,
        )
        self.assertLessEqual(solver.rho_power, 2.0 + 1e-12)
        self.assertGreaterEqual(solver.rho_power, 0.5 - 1e-12)


class TestInnerProxWarning(unittest.TestCase):
    """The device records the inner-prox residual; the *solver* judges it (A1).

    `StorageUnit.admm_prox_update` used to warn whenever
    ||x - y||_inf > inner_atol * max(ymax); on ca2040_z4 that threshold is a
    fraction of the largest energy capacity, so the warning fired on healthy
    runs. Whether the residual matters depends on the outer nodal imbalance,
    which only the solver can see.
    """

    LOGGER = "zap.admm.basic_solver"

    def _solve(self, **kwargs):
        net, devices, T = storage_system()
        settings = {
            "rho_power": 0.1,
            "num_iterations": 400,
            "minimum_iterations": 10,
            "adaptive_rho": False,
            "atol": 1e-6,
            "rtol": 1e-6,
        }
        settings.update(kwargs)
        return solve_admm(net, devices, T, **settings)

    def test_inner_prox_warning_is_silent_when_the_prox_is_converged(self):
        with self.assertNoLogs(self.LOGGER, level="WARNING"):
            solver, _, history = self._solve(battery_inner_iterations=2000)
        self.assertIsNone(solver.warn_inner_prox(history))

    def test_inner_prox_warning_fires_when_the_prox_is_the_binding_error(self):
        with self.assertLogs(self.LOGGER, level="WARNING") as caught:
            solver, _, _ = self._solve(battery_inner_iterations=2)
        records = [r for r in caught.records if "inner prox" in r.getMessage()]
        self.assertEqual(len(records), 1, caught.output)
        self.assertIn("battery_inner_iterations", records[0].getMessage())
        self.assertGreater(solver.max_inner_prox_residual, 0.0)

    def test_max_inner_prox_residual_is_reset_per_solve(self):
        net, devices, T = storage_system()
        torch_devices = [d.torchify(machine="cpu", dtype=torch.float64) for d in devices]
        solver = ADMMSolver(
            machine="cpu",
            dtype=torch.float64,
            num_iterations=200,
            minimum_iterations=10,
            rho_power=0.1,
            adaptive_rho=False,
            atol=1e-6,
            rtol=1e-6,
            verbose=0,
            battery_inner_iterations=2000,
        )
        solver.solve(net, torch_devices, T)
        loose = solver.max_inner_prox_residual

        solver.battery_inner_iterations = 2
        solver.solve(net, torch_devices, T)
        tight = solver.max_inner_prox_residual

        # The second solve's (much larger) residual must not be contaminated by
        # the first, and vice versa: the field is per-solve state.
        self.assertGreater(tight, loose)
        self.assertGreater(tight, 1e-3)
        # The per-iteration companion is reset too.
        self.assertLessEqual(solver.last_inner_prox_residual, tight)

    def test_warn_inner_prox_judges_the_final_iteration_not_the_worst(self):
        """The max over the solve is dominated by the first few outer iterations.

        On ca2040_z4 block 200 at `battery_inner_iterations: 200` the worst inner
        residual over the solve is 2.8 MWh while the final iteration's is 0.01 MWh
        and the returned iterate's SoC residual passes the harness gate -- so
        judging the max would warn on a healthy run, which is the noise A1 removes.
        """
        solver = ADMMSolver(machine="cpu", dtype=torch.float64, verbose=0)
        solver.num_node_hours = 100
        history = SimpleNamespace(power=[1e-2 * math.sqrt(100)])  # rms 1e-2 MW
        solver.max_inner_prox_residual = 2.8
        solver.last_inner_prox_residual = 0.01
        self.assertIsNone(solver.warn_inner_prox(history))
        # ... and the final iteration being bad is what does warn.
        solver.last_inner_prox_residual = 2.8
        message = solver.warn_inner_prox(history)
        self.assertIsNotNone(message)
        self.assertIn("final outer iteration", message)

    def test_warn_inner_prox_scales_with_the_outer_imbalance(self):
        """Same inner residual, different imbalance -> different verdict."""
        solver = ADMMSolver(machine="cpu", dtype=torch.float64, verbose=0)
        solver.num_node_hours = 100
        solver.max_inner_prox_residual = 1.0
        solver.last_inner_prox_residual = 1.0
        history = SimpleNamespace(power=[10.0 * math.sqrt(100)])  # rms 10 MW
        self.assertIsNone(solver.warn_inner_prox(history))
        history = SimpleNamespace(power=[1e-3 * math.sqrt(100)])  # rms 1e-3 MW
        self.assertIsNotNone(solver.warn_inner_prox(history))

    def test_rms_imbalance_is_per_node_hour_not_per_terminal(self):
        """`history.power` is the L2 norm of the (num_nodes, T) nodal-imbalance
        array, so the RMS divisor is sqrt(num_nodes * T) -- not the device-terminal
        count, which on ca2040_z4 is ~7x larger and would silence the warning."""
        net, _devices, T = storage_system()
        solver, _, history = self._solve(battery_inner_iterations=50)
        self.assertEqual(solver.num_node_hours, net.num_nodes * T)
        self.assertNotEqual(solver.num_node_hours, solver.total_terminals)
        expected = float(history.power[-1]) / math.sqrt(net.num_nodes * T)

        solver.max_inner_prox_residual = 1e9
        solver.last_inner_prox_residual = 1e9
        message = solver.warn_inner_prox(history)
        self.assertIn(f"{expected:.3g} MW", message)
        self.assertIn("node-hours", message)

    def test_warn_inner_prox_is_silent_below_the_absolute_floor(self):
        solver = ADMMSolver(machine="cpu", dtype=torch.float64, verbose=0)
        solver.num_node_hours = 10
        solver.max_inner_prox_residual = 1e-9
        solver.last_inner_prox_residual = 1e-9
        self.assertIsNone(solver.warn_inner_prox(SimpleNamespace(power=[0.0])))

    def test_warn_inner_prox_warns_on_a_non_finite_residual(self):
        solver = ADMMSolver(machine="cpu", dtype=torch.float64, verbose=0)
        solver.num_node_hours = 10
        solver.max_inner_prox_residual = float("inf")
        solver.last_inner_prox_residual = float("inf")
        self.assertIsNotNone(solver.warn_inner_prox(SimpleNamespace(power=[1e6])))


if __name__ == "__main__":
    unittest.main()
