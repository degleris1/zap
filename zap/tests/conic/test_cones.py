import unittest
import cvxpy as cp
import numpy as np
import numpy.testing as npt
import torch
import scs
from zap.admm import ADMMSolver, ADMMLayer
from zap.conic.cone_bridge import ConeBridge
from experiments.conic_solve.benchmarks.max_flow_benchmark import MaxFlowBenchmarkSet
from experiments.conic_solve.benchmarks.netlib_benchmark import NetlibBenchmarkSet
from experiments.conic_solve.benchmarks.sparse_cone_benchmark import SparseConeBenchmarkSet


from zap.conic.cone_utils import get_standard_conic_problem, get_conic_solution, stack_cvxpy_colwise
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
        benchmark = MaxFlowBenchmarkSet(num_problems=1, n=1000, base_seed=42)
        for i, prob in enumerate(benchmark):
            if i == 0:
                problem = prob
                cone_params, _, _ = get_standard_conic_problem(problem, solver=cp.CLARABEL)
        problem.solve(solver=cp.CLARABEL)
        ref_obj = problem.value

        cone_bridge = ConeBridge(cone_params, ruiz_iters=0)
        conic_ruiz_sigma = cone_bridge.sigma
        machine = "cuda"
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

    def test_batching(self):
        benchmark = MaxFlowBenchmarkSet(num_problems=1, n=100, base_seed=42)
        instance_list = []
        for i, prob in enumerate(benchmark):
            if i == 0:
                problem = prob
                instance_list.append(prob)
        T = 2
        for i in range(T-1):
            instance_list.append(problem)
        cone_params_batched = stack_cvxpy_colwise(instance_list)
        cone_params, _, _ = get_standard_conic_problem(problem, solver=cp.CLARABEL)
        cone_bridge = ConeBridge(cone_params_batched, ruiz_iters=0)
        test_outcome = cone_bridge.solve(solver=cp.CLARABEL)
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

    def test_layers(self):

        def compute_objective(st, devices, parameters):
            costs = []
            ## Hack-y we know it's just the first device
            cost = devices[0].operation_cost(st.power[0], st.phase[0], st.local_variables[0], la=torch, **parameters[0])
            return cost

        def assemble_cap(full_like_true, cap_real_vec):
            out = full_like_true.clone() # just for shape
            out[mask_real] = cap_real_vec
            out[-1] = cap_artificial
            return out

        ## Now cook this up using Conic Zap
        n_nodes = 100
        base_seed = 42
        max_flow_benchmark = MaxFlowBenchmarkSet(num_problems=1, n=n_nodes, base_seed=base_seed)
        for prob in max_flow_benchmark:
            problem = prob

        cone_params, data, cones = get_standard_conic_problem(problem, solver=cp.CLARABEL)
        cone_bridge = ConeBridge(cone_params, ruiz_iters = 0)

        ## Set up layer
        machine = 'cuda'
        dtype = torch.float32
        cone_admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
        cone_admm_solver = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
            # alpha=1.6,
            # tau=2,
            adaptive_rho=False,
            num_iterations=10000,
        )

        parameter_names = {"cap_var": (2, "b_d")} # capacities are stored as b_d of Non. Neg Slack Devices
        time_horizon = 1 
        x_star = ADMMLayer(
                cone_bridge.net,
                cone_admm_devices,
                parameter_names=parameter_names,
                time_horizon=time_horizon,
                solver=cone_admm_solver,
                adapt_rho=False,
                warm_start=False,
                verbose=True
        )

        ## Actual Loop
        n_edges = len(cone_bridge.devices[2].b_d)
        mask_real = torch.ones(n_edges, dtype=torch.bool)
        mask_real[-1] = False
        cap_artificial  = 10000.0  
        actual_flow_val = torch.tensor(196.7)
        lmbda = 1e-3

        init = torch.tensor(cone_admm_devices[2].b_d, device="cuda").squeeze(-1)
        cap_var = torch.nn.Parameter(10.0* torch.ones_like(init[mask_real])) # This doesnt include artificial edge
        device = cap_var.device
        optimizer = torch.optim.Adam([cap_var], lr=0.05)
        admm_state = None

        num_iters = 30
        for iter in range(num_iters):
            optimizer.zero_grad()

            
            cap_full = assemble_cap(init, cap_var)
            eta = {"cap_var": cap_full}
            params = x_star.setup_parameters(**eta)


            admm_state = x_star(**eta, initial_state=None)
            flow_val = -compute_objective(admm_state, x_star.devices, params)
            print(flow_val)


            loss = 0.5*(flow_val - actual_flow_val)**2 + lmbda*cap_var.sum()
            loss.backward()

            print("||grad|| on real caps:", cap_var.grad.norm().item())
            print("some grads:", cap_var.grad[:5])

            optimizer.step()

            # Box projection
            cap_var.data.clamp_(1.0, 10.0)
            if iter % 1 == 0:
                print(f"{iter:3d} | flow={flow_val:.4f} | loss={loss.item():.2e}")

        print("pleeeeaaase")
