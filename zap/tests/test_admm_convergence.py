"""The ADMM convergence test must be in consistent units.

`rtol_dual_use_objective=True` compared a power-valued dual residual against
`rtol * objective` (dollars), so on a system with a large objective the dual
tolerance was orders of magnitude looser than the primal one and the solver
declared convergence far from the optimum (benchmark review, section 2.1).
"""

import unittest
import warnings

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


if __name__ == "__main__":
    unittest.main()
