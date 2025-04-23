import unittest
import cvxpy as cp
import torch
import scs
from zap.admm import ADMMSolver
from zap.conic.cone_bridge import ConeBridge
from zap.conic.cone_utils import get_standard_conic_problem
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
            solution_admm.objective,
            ref_obj,
            delta=TOL,
            msg=f"ADMM objective {solution_admm.objective} differs from reference {ref_obj}",
        )

    def test_zero_nonneg_cvxpy(self):
        problem, cone_params = create_simple_problem_zero_nonneg_cones()
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value
        cone_bridge = ConeBridge(cone_params)
        outcome = cone_bridge.solve()

        self.assertAlmostEqual(
            outcome.problem.value,
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

        pct_diff = abs((solution_admm.objective - ref_obj) / ref_obj)
        self.assertLess(
            pct_diff,
            REL_TOL_PCT,
            msg=f"ADMM objective {solution_admm.objective} differs from reference objective {ref_obj} by more than {REL_TOL_PCT * 100:.2f}%",
        )

    def test_soc_admm(self):
        problem, cone_params = create_simple_problem_soc()
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params)
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
            solution_admm.objective,
            ref_obj,
            delta=TOL,
            msg=f"ADMM objective {solution_admm.objective} differs from reference {ref_obj}",
        )

    def test_soc_cvxpy(self):
        problem, cone_params = create_simple_problem_soc()
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params)
        outcome = cone_bridge.solve()

        self.assertAlmostEqual(
            outcome.problem.value,
            ref_obj,
            delta=TOL,
            msg=f"CVXPY objective {outcome.problem.value} differs from reference {ref_obj}",
        )

    def test_multi_block_soc_admm(self):
        problem, cone_params = create_simple_multi_block_problem_soc()
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params)
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
            solution_admm.objective,
            ref_obj,
            delta=TOL,
            msg=f"ADMM objective {solution_admm.objective} differs from reference {ref_obj}",
        )

    def test_multi_block_soc_cvxpy(self):
        problem, cone_params = create_simple_multi_block_problem_soc()
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params)
        outcome = cone_bridge.solve()

        self.assertAlmostEqual(
            outcome.problem.value,
            ref_obj,
            delta=TOL,
            msg=f"CVXPY objective {outcome.problem.value} differs from reference {ref_obj}",
        )
