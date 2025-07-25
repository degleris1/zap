import unittest
import cvxpy as cp
import numpy as np
import numpy.testing as npt
import torch
from zap.admm import ADMMSolver
from zap.resource_opt.nu_opt_bridge import NUOptBridge
from experiments.resource_opt_solve.benchmarks.nu_opt_benchmark import NUOptBenchmarkSet

TOL = 1e-2


class TestNUOptBridge(unittest.TestCase):
    def test_simple_log_num_admm(self):
        benchmark = NUOptBenchmarkSet(
            num_problems=1, m=20, n=10, avg_route_length=3, capacity_range=(0.1, 1), base_seed=42
        )
        for i, prob in enumerate(benchmark):
            problem = prob
        problem.solve(solver=cp.CLARABEL)
        ref_obj = -problem.value

        R, capacities, w, _ = benchmark.get_data(0)
        m, n = R.shape

        nu_opt_params = {
            "R": R,
            "capacities": capacities,
            "w": w,
        }

        nu_opt_bridge = NUOptBridge(nu_opt_params)
        machine = "cpu"
        dtype = torch.float32
        admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in nu_opt_bridge.devices]
        admm = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
            tau=2,
            alpha=1.6,
        )
        solution_admm, history_admm = admm.solve(
            nu_opt_bridge.net, admm_devices, nu_opt_bridge.time_horizon
        )
        print(history_admm.power[-1])
        print(history_admm.dual_power[-1])

        self.assertAlmostEqual(
            solution_admm.objective,
            ref_obj,
            delta=TOL,
            msg=f"ADMM objective {solution_admm.objective} differs from reference {ref_obj}",
        )

    def test_simple_mixed_lin_log_num_admm(self):
        benchmark = NUOptBenchmarkSet(
            num_problems=1,
            m=20,
            n=10,
            avg_route_length=3,
            capacity_range=(0.1, 1),
            lin_util_frac=0.1,
            base_seed=42,
        )
        for i, prob in enumerate(benchmark):
            problem = prob
        problem.solve(solver=cp.CLARABEL)
        ref_obj = -problem.value

        R, capacities, w, linear_flow_idxs = benchmark.get_data(0)
        m, n = R.shape

        nu_opt_params = {
            "R": R,
            "capacities": capacities,
            "w": w,
            "lin_device_idxs": linear_flow_idxs,
        }

        nu_opt_bridge = NUOptBridge(nu_opt_params)
        machine = "cpu"
        dtype = torch.float32
        admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in nu_opt_bridge.devices]
        admm = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
            tau=2,
            alpha=1.6,
        )
        solution_admm, history_admm = admm.solve(
            nu_opt_bridge.net, admm_devices, nu_opt_bridge.time_horizon
        )
        print(history_admm.power[-1])
        print(history_admm.dual_power[-1])

        self.assertAlmostEqual(
            solution_admm.objective,
            ref_obj,
            delta=TOL,
            msg=f"ADMM objective {solution_admm.objective} differs from reference {ref_obj}",
        )

    def test_simple_log_num_cvxpy(self):
        benchmark = NUOptBenchmarkSet(
            num_problems=1, m=20, n=10, avg_route_length=3, capacity_range=(0.1, 1), base_seed=42
        )
        for i, prob in enumerate(benchmark):
            problem = prob
        problem.solve(solver=cp.CLARABEL)
        ref_obj = -problem.value

        R, capacities, w = benchmark.get_data(0)
        m, n = R.shape

        nu_opt_params = {
            "R": R,
            "capacities": capacities,
            "w": w,
        }

        nu_opt_bridge = NUOptBridge(nu_opt_params)
        outcome = nu_opt_bridge.solve()

        self.assertAlmostEqual(
            outcome.problem.value,
            ref_obj,
            delta=TOL,
            msg=f"ADMM objective {outcome.problem.value} differs from reference {ref_obj}",
        )
