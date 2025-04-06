from zap.conic.cone_bridge import ConeBridge
from zap.conic.cone_utils import get_standard_conic_problem, get_conic_solution
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import torch
from zap.admm import ADMMSolver

np.set_printoptions(formatter={"float": "{:6.3f}".format})


def main():
    n = 3
    m = 8

    np.random.seed(42)
    density = 0.3

    # Create a random sparse matrix A of shape (m, n)
    A = sp.random(m, n, density=density, format="csc", data_rvs=np.random.randn)
    b = np.random.randn(m)

    c = np.random.randn(n)

    x = cp.Variable(n)
    s = cp.Variable(m)

    constraints = []
    constraints.append(A @ x + s == b)
    constraints.append(x >= -5)
    constraints.append(x <= 5)
    constraints.append(cp.norm(s[1:2]) <= s[0])
    constraints.append(cp.norm(s[3:5]) <= s[2])
    constraints.append(cp.norm(s[6:8]) <= s[5])
    objective = cp.Minimize(c.T @ x)

    problem = cp.Problem(objective, constraints)

    cone_params, data, cones = get_standard_conic_problem(problem, solver=cp.CLARABEL)
    cone_bridge = ConeBridge(cone_params)

    ### Test ADMM
    machine = "cpu"
    dtype = torch.float32
    admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
    admm = ADMMSolver(
        machine=machine,
        dtype=dtype,
        atol=1e-6,
        rtol=1e-6,
        # track_objective=False,
        # rtol_dual_use_objective=False,
    )
    solution_admm, history_admm = admm.solve(
        cone_bridge.net, admm_devices, cone_bridge.time_horizon
    )
    ## End Test ADMM
    # x, s = get_conic_solution(solution_admm, cone_bridge)

    outcome = cone_bridge.solve()

    print("helllooooo")


if __name__ == "__main__":
    main()
