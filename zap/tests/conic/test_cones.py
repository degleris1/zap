import unittest
import cvxpy as cp
import numpy as np
import numpy.testing as npt
import torch
import scs
import scipy.sparse as sp
from cvxpylayers.torch import CvxpyLayer
from zap.admm import ADMMSolver, ADMMLayer
from zap.conic.conic_layer import ConicADMMLayer
from zap.conic.cone_bridge import ConeBridge
from experiments.conic_solve.benchmarks.max_flow_benchmark import MaxFlowBenchmarkSet
from experiments.conic_solve.benchmarks.netlib_benchmark import NetlibBenchmarkSet
from experiments.conic_solve.benchmarks.sparse_cone_benchmark import SparseConeBenchmarkSet


from zap.conic.cone_utils import get_standard_conic_problem, get_conic_solution, stack_cvxpy_colwise, compute_batch_objectives
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
        b_orig = cone_params["b"].reshape(-1, 1)
        c_orig = cone_params["c"].reshape(-1, 1)

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
        #TODO: Fix the fact that this is not deterministic!
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

        cone_bridge = ConeBridge(cone_params, ruiz_iters=0)
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

        cone_bridge = ConeBridge(cone_params, ruiz_iters=0)
        conic_ruiz_sigma = cone_bridge.sigma
        outcome = cone_bridge.solve()

        self.assertAlmostEqual(
            outcome.problem.value / (conic_ruiz_sigma),
            ref_obj,
            delta=TOL,
            msg=f"CVXPY objective {outcome.problem.value} differs from reference {ref_obj}",
        )

    # def test_layers(self):

    #     def compute_objective(st, devices, parameters):
    #         costs = []
    #         ## Hack-y we know it's just the first device
    #         cost = devices[0].operation_cost(st.power[0], st.phase[0], st.local_variables[0], la=torch, **parameters[0])
    #         return cost

    #     def assemble_cap(full_like_true, cap_real_vec):
    #         out = full_like_true.clone() # just for shape
    #         out[mask_real] = cap_real_vec
    #         out[-1] = cap_artificial
    #         return out

    #     ## Now cook this up using Conic Zap
    #     n_nodes = 100
    #     base_seed = 42
    #     max_flow_benchmark = MaxFlowBenchmarkSet(num_problems=1, n=n_nodes, base_seed=base_seed)
    #     for prob in max_flow_benchmark:
    #         problem = prob

    #     cone_params, data, cones = get_standard_conic_problem(problem, solver=cp.CLARABEL)
    #     cone_bridge = ConeBridge(cone_params, ruiz_iters = 0)

    #     ## Set up layer
    #     machine = 'cuda'
    #     dtype = torch.float32
    #     cone_admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
    #     cone_admm_solver = ADMMSolver(
    #         machine=machine,
    #         dtype=dtype,
    #         atol=1e-6,
    #         rtol=1e-6,
    #         # alpha=1.6,
    #         # tau=2,
    #         adaptive_rho=False,
    #         num_iterations=10000,
    #     )

    #     parameter_names = {"cap_var": (2, "b_d")} # capacities are stored as b_d of Non. Neg Slack Devices
    #     time_horizon = 1 
    #     x_star = ADMMLayer(
    #             cone_bridge.net,
    #             cone_admm_devices,
    #             parameter_names=parameter_names,
    #             time_horizon=time_horizon,
    #             solver=cone_admm_solver,
    #             adapt_rho=False,
    #             warm_start=False,
    #             verbose=True
    #     )

    #     ## Actual Loop
    #     n_edges = len(cone_bridge.devices[2].b_d)
    #     mask_real = torch.ones(n_edges, dtype=torch.bool)
    #     mask_real[-1] = False
    #     cap_artificial  = 10000.0  
    #     actual_flow_val = torch.tensor(196.7)
    #     lmbda = 1e-3

    #     init = torch.tensor(cone_admm_devices[2].b_d, device="cuda").squeeze(-1)
    #     cap_var = torch.nn.Parameter(10.0* torch.ones_like(init[mask_real])) # This doesnt include artificial edge
    #     device = cap_var.device
    #     optimizer = torch.optim.Adam([cap_var], lr=0.05)
    #     admm_state = None

    #     num_iters = 30
    #     for iter in range(num_iters):
    #         optimizer.zero_grad()

            
    #         cap_full = assemble_cap(init, cap_var)
    #         eta = {"cap_var": cap_full}
    #         params = x_star.setup_parameters(**eta)


    #         admm_state = x_star(**eta, initial_state=None)
    #         flow_val = -compute_objective(admm_state, x_star.devices, params)
    #         print(flow_val)


    #         loss = 0.5*(flow_val - actual_flow_val)**2 + lmbda*cap_var.sum()
    #         loss.backward()

    #         print("||grad|| on real caps:", cap_var.grad.norm().item())
    #         print("some grads:", cap_var.grad[:5])

    #         optimizer.step()

    #         # Box projection
    #         cap_var.data.clamp_(1.0, 10.0)
    #         if iter % 1 == 0:
    #             print(f"{iter:3d} | flow={flow_val:.4f} | loss={loss.item():.2e}")

    #     print("pleeeeaaase")

    # def test_conic_layer(self):

    #     # Linear objective, affects only b param (no SOC)
    #     def create_test_case_1():
    #             n_nodes = 10
    #             np.random.seed(42)
                
    #             # Create a random directed graph
    #             edges = []
    #             for i in range(n_nodes):
    #                 for j in range(n_nodes):
    #                     if i != j and np.random.rand() < 0.3:
    #                         edges.append((i, j))
                
    #             n_edges = len(edges)
                
    #             # Create incidence matrix
    #             inc = np.zeros((n_nodes, n_edges))
    #             for e_idx, (i, j) in enumerate(edges):
    #                 inc[i, e_idx] = 1   # Outgoing
    #                 inc[j, e_idx] = -1  # Incoming
                
    #             # Source and sink
    #             source = 0
    #             sink = n_nodes - 1
                
    #             # Flow balance
    #             b = np.zeros(n_nodes)
    #             b[source] = 1   # Source produces 1 unit
    #             b[sink] = -1    # Sink consumes 1 unit
                
    #             # Flow variables and capacity parameters
    #             f = cp.Variable(n_edges)
    #             capacity = cp.Parameter(n_edges, nonneg=True)
                
    #             # Objective: maximize flow
    #             c = np.zeros(n_edges)
    #             for e_idx, (i, j) in enumerate(edges):
    #                 if j == sink:
    #                     c[e_idx] = 1  # Maximize flow into sink
                
    #             # Constraints
    #             constraints = [
    #                 f >= 0,               # Non-negative flow
    #                 f <= capacity,        # Capacity constraints
    #                 inc @ f == b          # Flow conservation
    #             ]
                
    #             objective = cp.Maximize(c @ f)
    #             problem = cp.Problem(objective, constraints)
                
    #             # Initialize capacity parameters
    #             capacity.value = np.random.uniform(0.5, 2.0, n_edges)
                
    #             return problem, [capacity], "Network flow with parameterized capacities"

    #     # Quadratic objective, affects only c param
    #     def create_test_case_2():
    #         n_assets = 10
    #         np.random.seed(43)
            
    #         # Covariance matrix (fixed)
    #         Sigma = np.random.randn(n_assets, n_assets)
    #         Sigma = Sigma.T @ Sigma  # Make positive semidefinite
            
    #         # Expected returns (parameterized)
    #         mu = cp.Parameter(n_assets)
            
    #         # Portfolio weights
    #         w = cp.Variable(n_assets)
            
    #         # Objective: maximize return - lambda * risk
    #         risk_aversion = 0.1
    #         objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))
            
    #         # Constraints
    #         constraints = [
    #             cp.sum(w) == 1,   # Fully invested
    #             w >= 0            # Long only
    #         ]
            
    #         problem = cp.Problem(objective, constraints)
            
    #         # Initialize expected returns
    #         mu.value = np.random.uniform(0.05, 0.15, n_assets)
            
    #         return problem, [mu], "Portfolio optimization with parameterized returns"

    #     # Affects b param (with SOC)
    #     def create_test_case_3():
    #         # Create a simple problem with an SOC constraint
    #         n = 5  # Problem dimension
            
    #         # Define variables and parameters
    #         x = cp.Variable(n)
    #         t = cp.Variable(1)
            
    #         # Parameter that will affect the SOC constraint
    #         soc_param = cp.Parameter(n-1, name="soc_param")
            
    #         # Create an SOC constraint: ||A*x + b||_2 <= t
    #         # Where b is our parameter
    #         A = np.random.randn(n-1, n)
            
    #         # Objective and constraints
    #         objective = cp.Minimize(t)
    #         constraints = [
    #             cp.SOC(t, A @ x + soc_param)  # SOC constraint with parameter
    #         ]
            
    #         problem = cp.Problem(objective, constraints)
            
    #         # Initialize parameter with some value
    #         soc_param.value = np.random.randn(n-1)
            
    #         return problem, [soc_param], "Simple SOC Problem"


    #     problem, parameters, description = create_test_case_3()

    #     ## Set up solver
    #     machine = 'cuda'
    #     dtype = torch.float32
    #     cone_admm_solver = ADMMSolver(
    #         machine=machine,
    #         dtype=dtype,
    #         atol=1e-6,
    #         rtol=1e-6,
    #         # alpha=1.6,
    #         # tau=2,
    #         adaptive_rho=False,
    #         num_iterations=10000,
    #     )

    #     ## Set up layer
    #     x_star = ConicADMMLayer(problem, parameters, cone_admm_solver)


class TestConeBatching(unittest.TestCase):
    def test_lp_batching(self):
        problem_m, problem_n = 5, 10
        problem1, cone_params1 = create_simple_problem_zero_nonneg_cones(
            m=problem_m, n=problem_n, density=0.3, seed=42, quad_obj=False
        )

        np.random.seed(42)
        A2 = sp.random(problem_m, problem_n, density=0.3, format="csc", data_rvs=np.random.randn)
        np.random.seed(43)
        A2.data = np.random.randn(A2.nnz)
        b2 = np.random.randn(problem_m)
        c2 = np.random.randn(problem_n)
        x2 = cp.Variable(problem_n)
        s2 = cp.Variable(problem_m)
        constraints2 = [A2 @ x2 + s2 == b2, s2 >= 0, x2 >= -5, x2 <= 5]
        objective2 = cp.Minimize(c2.T @ x2)
        problem2 = cp.Problem(objective2, constraints2)
        
        cone_params2, _, _ = get_standard_conic_problem(problem2, solver=cp.CLARABEL)
        
        # Verify the sparsity patterns are identical
        assert np.array_equal(cone_params1["A"].indptr, cone_params2["A"].indptr)
        assert np.array_equal(cone_params1["A"].indices, cone_params2["A"].indices)
        assert not np.array_equal(cone_params1["A"].data, cone_params2["A"].data)
        
        problem1.solve(solver=cp.CLARABEL)
        problem2.solve(solver=cp.CLARABEL)
        ref_obj1 = problem1.value
        ref_obj2 = problem2.value

        machine = "cpu"
        dtype = torch.float32
        instance_list = [problem1, problem2]
        T = len(instance_list)
        cone_params_batched = stack_cvxpy_colwise(instance_list)
        
        # Solve batched problem
        cone_bridge = ConeBridge(cone_params_batched, ruiz_iters=0)
        admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
        admm = ADMMSolver(
            machine=machine,
            dtype=dtype,
            atol=1e-6,
            rtol=1e-6,
        )
        solution_admm, history_admm = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
        batch_objectives = compute_batch_objectives(solution_admm, admm_devices)
                
        self.assertAlmostEqual(
            batch_objectives[0].item()/cone_bridge.sigma,
            ref_obj1, 
            delta=TOL,
            msg=f"Batched solution 1 {batch_objectives[0].item()} differs from reference {ref_obj1}"
        )
        self.assertAlmostEqual(
            batch_objectives[1].item()/cone_bridge.sigma,
            ref_obj2,
            delta=TOL,
            msg=f"Batched solution 2 {batch_objectives[1].item()} differs from reference {ref_obj2}"
        )

    def test_qp_batching(self):
        """
        Test batching quadratic programs with different b, c, and A values (same sparsity).
        P does not get batched. 
        """
        problem_m, problem_n = 5, 10
        
        np.random.seed(42)
        L = np.random.randn(problem_n, problem_n)
        P = L.T @ L  # Fixed P for both problems
        
        np.random.seed(42)
        A1 = sp.random(problem_m, problem_n, density=0.3, format="csc", data_rvs=np.random.randn)
        b1 = np.random.randn(problem_m)
        c1 = np.random.randn(problem_n)
        
        x1 = cp.Variable(problem_n)
        s1 = cp.Variable(problem_m)
        constraints1 = [A1 @ x1 + s1 == b1, s1 >= 0, x1 >= -5, x1 <= 5]
        objective1 = cp.Minimize(0.5 * cp.quad_form(x1, P) + c1.T @ x1)
        problem1 = cp.Problem(objective1, constraints1)
        
        np.random.seed(43)
        A2 = A1.copy()
        A2.data = np.random.randn(A1.nnz)
        b2 = np.random.randn(problem_m)
        c2 = np.random.randn(problem_n)
        
        x2 = cp.Variable(problem_n)
        s2 = cp.Variable(problem_m)
        constraints2 = [A2 @ x2 + s2 == b2, s2 >= 0, x2 >= -5, x2 <= 5]
        objective2 = cp.Minimize(0.5 * cp.quad_form(x2, P) + c2.T @ x2)
        problem2 = cp.Problem(objective2, constraints2)
        
        cone_params1, _, _ = get_standard_conic_problem(problem1, solver=cp.CLARABEL)
        cone_params2, _, _ = get_standard_conic_problem(problem2, solver=cp.CLARABEL)
        
        problem1.solve(solver=cp.CLARABEL)
        problem2.solve(solver=cp.CLARABEL)
        ref_obj1 = problem1.value
        ref_obj2 = problem2.value

        # Same sparsity pattern in A
        assert np.array_equal(cone_params1["A"].indptr, cone_params2["A"].indptr)
        assert np.array_equal(cone_params1["A"].indices, cone_params2["A"].indices)
    
        # Exactly the same P
        assert np.array_equal(cone_params1["P"].indptr, cone_params2["P"].indptr)
        assert np.array_equal(cone_params1["P"].indices, cone_params2["P"].indices)
        assert np.array_equal(cone_params1["P"].data, cone_params2["P"].data)
        
        # Stack problems
        instance_list = [problem1, problem2]
        cone_params_batched = stack_cvxpy_colwise(instance_list)
        
        # Solve batched problem
        cone_bridge = ConeBridge(cone_params_batched, ruiz_iters=0)
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
        batch_objectives = compute_batch_objectives(solution_admm, admm_devices)
                
        self.assertAlmostEqual(
            batch_objectives[0].item()/cone_bridge.sigma,
            ref_obj1,
            delta=TOL,
            msg=f"QP Batched solution 1 {batch_objectives[0].item()} differs from reference {ref_obj1}"
        )
        self.assertAlmostEqual(
            batch_objectives[1].item()/cone_bridge.sigma,
            ref_obj2,
            delta=TOL,
            msg=f"QP Batched solution 2 {batch_objectives[1].item()} differs from reference {ref_obj2}"
        )

    def test_socp_batching(self):    
        n, m = 3, 8
        density = 0.3
        np.random.seed(42)
        A1 = sp.random(m, n, density=density, format="csc", data_rvs=np.random.randn)
        b1 = np.random.randn(m)
        c1 = np.random.randn(n)

        x1 = cp.Variable(n)
        s1 = cp.Variable(m)
        constraints1 = [
            A1 @ x1 + s1 == b1,
            x1 >= -5,
            x1 <= 5,
            cp.norm(s1[1:2]) <= s1[0],
            cp.norm(s1[3:5]) <= s1[2],
            cp.norm(s1[6:8]) <= s1[5],
        ]
        objective1 = cp.Minimize(c1.T @ x1)
        problem1 = cp.Problem(objective1, constraints1)
        
        A2 = A1.copy()
        np.random.seed(0)
        A2.data = np.random.randn(A1.nnz)
        b2 = np.random.randn(m)
        c2 = np.random.randn(n)

        x2 = cp.Variable(n)
        s2 = cp.Variable(m)
        constraints2 = [
            A2 @ x2 + s2 == b2,
            x2 >= -5,
            x2 <= 5,
            cp.norm(s2[1:2]) <= s2[0],
            cp.norm(s2[3:5]) <= s2[2],
            cp.norm(s2[6:8]) <= s2[5],
        ]
        objective2 = cp.Minimize(c2.T @ x2)
        problem2 = cp.Problem(objective2, constraints2)
        
        cone_params1, _, _ = get_standard_conic_problem(problem1, solver=cp.CLARABEL)
        cone_params2, _, _ = get_standard_conic_problem(problem2, solver=cp.CLARABEL)
        
        assert np.array_equal(cone_params1["A"].indptr, cone_params2["A"].indptr)
        assert np.array_equal(cone_params1["A"].indices, cone_params2["A"].indices)
        assert not np.array_equal(cone_params1["A"].data, cone_params2["A"].data)
        
        # Solve individual problems
        problem1.solve(solver=cp.CLARABEL)
        problem2.solve(solver=cp.CLARABEL)
        ref_obj1 = problem1.value
        ref_obj2 = problem2.value

        # Solve batched problem
        instance_list = [problem1, problem2]
        cone_params_batched = stack_cvxpy_colwise(instance_list)
        
        cone_bridge = ConeBridge(cone_params_batched, ruiz_iters=5)
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
        
        batch_objectives = compute_batch_objectives(solution_admm, admm_devices)        
        self.assertAlmostEqual(
            batch_objectives[0].item()/cone_bridge.sigma[0],
            ref_obj1,
            delta=TOL,
            msg=f"SOCP Batched solution 1 {batch_objectives[0].item()} differs from reference {ref_obj1}"
        )
        self.assertAlmostEqual(
            batch_objectives[1].item()/cone_bridge.sigma[1],
            ref_obj2,
            delta=TOL,
            msg=f"SOCP Batched solution 2 {batch_objectives[1].item()} differs from reference {ref_obj2}"
        )

class TestConeLayers(unittest.TestCase):
    def test_cvxpy_layer(self):
        n_nodes = 10000
        torch.manual_seed(42)
        def build_max_flow_cvxpy_layer(n, base_seed):
            bench = MaxFlowBenchmarkSet(num_problems=1, n=n_nodes, quad_obj=False, base_seed=42)
            data = bench.get_data(0)
            inc, adj, b, c, cap_true, n_edges = data
            source_sink_edge_idx = int(np.argmax(c))

            f = cp.Variable(n_edges)
            cap = cp.Parameter(n_edges, nonneg=True)
            constraints = []
            constraints.append(f <= cap)
            constraints.append(inc @ f == b)
            obj = c @ f
            problem = cp.Problem(cp.Minimize(-obj), constraints)
            layer = CvxpyLayer(problem, parameters=[cap], variables=[f])

            return layer, torch.tensor(cap_true, dtype=torch.float32), torch.tensor(c, dtype=torch.float32), inc, n_edges

        cvxpy_max_flow_layer, cap_true_tensor, c_tensor, inc, n_edges = build_max_flow_cvxpy_layer(n=n_nodes, base_seed=42)

        # Learn capacities in a training loop
        mask_real = torch.ones(n_edges, dtype=torch.bool)
        mask_real[-1] = False
        cap_artificial  = 10000.0  

        def assemble_cap(full_like_true, cap_real_vec):
            out = full_like_true.clone() # just for shape
            out[mask_real] = cap_real_vec
            out[-1] = cap_artificial
            return out


        cap_var = torch.nn.Parameter(10.0* torch.ones_like(cap_true_tensor[mask_real])) # This doesnt include artificial edge
        optimizer = torch.optim.Adam([cap_var], lr=0.005)
        lmbda = 1e-3 # Sparse regularization 
        # actual_flow_val = torch.dot(cvxpy_max_flow_layer(cap_true_tensor)[0], c_tensor)
        # actual_flow_val = torch.tensor(203.78) # 100 node
        # actual_flow_val = torch.tensor(213.11) # 1000 node
        actual_flow_val = torch.tensor(367.07)
        residual_norms = []
        flow_vals_cvxpy = []
        loss_vals_cvxpy = []

        num_iters = 1
        for iter in range(num_iters):
            optimizer.zero_grad()

            cap_full = assemble_cap(cap_true_tensor, cap_var)
            f_pred  = cvxpy_max_flow_layer(cap_full, solver_args={"mode": "lsqr", "use_indirect": True, "gpu": False})[0]
            flow_val = torch.dot(f_pred, c_tensor)
            flow_vals_cvxpy.append(flow_val.item())
            print(f"completed forward pass: {flow_val.item()}")

            loss = 0.5 * (flow_val - actual_flow_val) ** 2 + lmbda*cap_var.sum()
            loss_vals_cvxpy.append(loss.item())
            print("starting backward pass")
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                cap_var.clamp_(1.0, 10.0)

            if iter % 10 == 0:
                print(f"{iter:3d} | flow={flow_val.item():.4f} | loss={loss.item():.2e}")


