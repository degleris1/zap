import unittest
import cvxpy as cp
import numpy as np
import numpy.testing as npt
import torch
import scs
from zap.admm import ADMMSolver
from zap.conic.cone_bridge import ConeBridge
from experiments.conic_solve.benchmarks.max_flow_benchmark import MaxFlowBenchmarkSet
from experiments.conic_solve.benchmarks.netlib_benchmark import NetlibBenchmarkSet
from experiments.conic_solve.benchmarks.sparse_cone_benchmark import SparseConeBenchmarkSet


from zap.conic.cone_utils import get_standard_conic_problem, get_conic_solution
from zap.tests.conic.examples import (
    create_simple_problem_zero_nonneg_cones,
    create_simple_problem_soc,
    create_simple_multi_block_problem_soc,
)
from zap.importers.toy import load_test_network


REL_TOL_PCT = 0.1
TOL = 1e-2


class TestConeBridge(unittest.TestCase):
    def test_zero_nonneg_admm(self):
        problem, cone_params = create_simple_problem_zero_nonneg_cones()
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params)
        conic_ruiz_sigma = cone_bridge.sigma
        machine = "cpu"
        dtype = torch.float32
        admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
        admm = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
        )
        solution_admm, _ = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
        self.assertAlmostEqual(
            solution_admm.objective / (conic_ruiz_sigma),
            ref_obj,
            delta=TOL,
            msg=f"ADMM objective {solution_admm.objective} differs from reference {ref_obj}",
        )

    def test_zero_nonneg_cvxpy(self):
        problem, cone_params = create_simple_problem_zero_nonneg_cones()
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value
        cone_bridge = ConeBridge(cone_params)
        conic_ruiz_sigma = cone_bridge.sigma
        outcome = cone_bridge.solve()

        self.assertAlmostEqual(
            outcome.problem.value / (conic_ruiz_sigma),
            ref_obj,
            delta=TOL,
            msg=f"CVXPY objective {outcome.problem.value} differs from reference {ref_obj}",
        )

    def test_non_pypsa_net_admm(self):
        net, devices = load_test_network()
        time_horizon = 4
        machine = "cpu"
        dtype = torch.float32

        ## Solve the conic form of this problem using CVXPY
        outcome = net.dispatch(devices, time_horizon, solver=cp.CLARABEL, add_ground=False)
        problem = outcome.problem
        cone_params, data, cones = get_standard_conic_problem(problem, cp.CLARABEL)
        soln = scs.solve(data, cones, verbose=False)
        ref_obj = soln["info"]["pobj"]

        # Build ConeBridge and ADMM solver
        cone_bridge = ConeBridge(cone_params)
        conic_ruiz_sigma = cone_bridge.sigma
        cone_admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
        cone_admm = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
            track_objective=True,
            rtol_dual_use_objective=True,
        )
        solution_admm, _ = cone_admm.solve(
            net=cone_bridge.net, devices=cone_admm_devices, time_horizon=cone_bridge.time_horizon
        )

        pct_diff = abs((solution_admm.objective / (conic_ruiz_sigma) - ref_obj) / ref_obj)
        self.assertLess(
            pct_diff,
            REL_TOL_PCT,
            msg=f"ADMM objective {solution_admm.objective / (conic_ruiz_sigma)} differs from reference objective {ref_obj} by more than {REL_TOL_PCT * 100:.2f}%",
        )

    def test_soc_admm(self):
        problem, cone_params = create_simple_problem_soc()
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params)
        conic_ruiz_sigma = cone_bridge.sigma
        machine = "cpu"
        dtype = torch.float32
        admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
        admm = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
        )
        solution_admm, _ = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
        self.assertAlmostEqual(
            solution_admm.objective / (conic_ruiz_sigma),
            ref_obj,
            delta=TOL,
            msg=f"ADMM objective {solution_admm.objective} differs from reference {ref_obj}",
        )

    def test_soc_cvxpy(self):
        problem, cone_params = create_simple_problem_soc()
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params)
        conic_ruiz_sigma = cone_bridge.sigma
        outcome = cone_bridge.solve()

        self.assertAlmostEqual(
            outcome.problem.value / (conic_ruiz_sigma),
            ref_obj,
            delta=TOL,
            msg=f"CVXPY objective {outcome.problem.value} differs from reference {ref_obj}",
        )

    def test_multi_block_soc_admm(self):
        problem, cone_params = create_simple_multi_block_problem_soc()
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params)
        conic_ruiz_sigma = cone_bridge.sigma
        machine = "cpu"
        dtype = torch.float32
        admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
        admm = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
        )
        solution_admm, _ = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
        self.assertAlmostEqual(
            solution_admm.objective / (conic_ruiz_sigma),
            ref_obj,
            delta=TOL,
            msg=f"ADMM objective {solution_admm.objective} differs from reference {ref_obj}",
        )

    def test_multi_block_soc_cvxpy(self):
        problem, cone_params = create_simple_multi_block_problem_soc()
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params)
        conic_ruiz_sigma = cone_bridge.sigma
        outcome = cone_bridge.solve()

        self.assertAlmostEqual(
            outcome.problem.value / (conic_ruiz_sigma),
            ref_obj,
            delta=TOL,
            msg=f"CVXPY objective {outcome.problem.value} differs from reference {ref_obj}",
        )

    def test_ruiz_equilibration(self):
        problem, cone_params = create_simple_multi_block_problem_soc()
        A_orig = cone_params["A"]
        b_orig = cone_params["b"]
        c_orig = cone_params["c"]
        cone_bridge = ConeBridge(cone_params, ruiz_iters=5)
        D_vec = cone_bridge.D_vec
        E_vec = cone_bridge.E_vec
        c_hat = cone_bridge.c
        b_hat = cone_bridge.b
        A_hat = cone_bridge.A.toarray()
        sigma = cone_bridge.sigma

        A_hat_recon = np.diag(D_vec) @ A_orig @ np.diag(E_vec)
        b_hat_recon = np.diag(D_vec) @ b_orig
        c_hat_recon = sigma * np.diag(E_vec) @ c_orig

        npt.assert_allclose(A_hat, A_hat_recon)
        npt.assert_allclose(b_hat, b_hat_recon)
        npt.assert_allclose(c_hat, c_hat_recon)

    def test_max_flow(self):
        benchmark = MaxFlowBenchmarkSet(num_problems=1, n=100, base_seed=42)
        for i, prob in enumerate(benchmark):
            if i == 0:
                problem = prob
                cone_params, _, _ = get_standard_conic_problem(problem, solver=cp.CLARABEL)
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params, ruiz_iters=5)
        conic_ruiz_sigma = cone_bridge.sigma
        machine = "cpu"
        dtype = torch.float32
        admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
        admm = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
        )
        solution_admm, _ = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
        self.assertAlmostEqual(
            solution_admm.objective / (conic_ruiz_sigma),
            ref_obj,
            delta=TOL,
            msg=f"CVXPY objective {solution_admm.objective} differs from reference {ref_obj}",
        )

    def test_netlib(self):
        benchmark = NetlibBenchmarkSet(data_dir="data/conic_benchmarks/netlib")
        for i, prob in enumerate(benchmark):
            if i == 0:
                problem = prob
                cone_params, _, _ = get_standard_conic_problem(problem, solver=cp.CLARABEL)
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params, ruiz_iters=5)
        conic_ruiz_sigma = cone_bridge.sigma
        machine = "cpu"
        dtype = torch.float32
        admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
        admm = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
        )
        solution_admm, _ = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
        pct_diff = abs((solution_admm.objective / (conic_ruiz_sigma) - ref_obj) / ref_obj)
        self.assertLess(
            pct_diff,
            REL_TOL_PCT,
            msg=f"ADMM objective {solution_admm.objective / (conic_ruiz_sigma)} differs from reference objective {ref_obj} by more than {REL_TOL_PCT * 100:.2f}%",
        )

    def test_sparse_cone_lp(self):
        benchmark = SparseConeBenchmarkSet(num_problems=3, n=100, p_f=0.5, p_l=0.5)
        for i, prob in enumerate(benchmark):
            if i == 2:
                problem = prob
                cone_params, _, _ = get_standard_conic_problem(problem, solver=cp.CLARABEL)
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params, ruiz_iters=5)
        conic_ruiz_sigma = cone_bridge.sigma
        machine = "cpu"
        dtype = torch.float32
        admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
        admm = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
        )
        solution_admm, _ = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
        self.assertAlmostEqual(
            solution_admm.objective / (conic_ruiz_sigma),
            ref_obj,
            delta=TOL,
            msg=f"CVXPY objective {solution_admm.objective} differs from reference {ref_obj}",
        )

    def test_conic_qp_admm(self):
        problem, cone_params = create_simple_problem_zero_nonneg_cones(quad_obj=True)
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params, ruiz_iters=5)
        conic_ruiz_sigma = cone_bridge.sigma
        machine = "cpu"
        dtype = torch.float32
        admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
        admm = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
        )
        solution_admm, _ = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
        self.assertAlmostEqual(
            solution_admm.objective / (conic_ruiz_sigma),
            ref_obj,
            delta=TOL,
            msg=f"ADMM objective {solution_admm.objective} differs from reference {ref_obj}",
        )

    def test_conic_qp_cvxpy(self):
        problem, cone_params = create_simple_problem_zero_nonneg_cones(quad_obj=True)
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params, ruiz_iters=5)
        conic_ruiz_sigma = cone_bridge.sigma
        outcome = cone_bridge.solve()

        self.assertAlmostEqual(
            outcome.problem.value / (conic_ruiz_sigma),
            ref_obj,
            delta=TOL,
            msg=f"CVXPY objective {outcome.problem.value} differs from reference {ref_obj}",
        )
