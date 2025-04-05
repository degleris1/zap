from zap.conic.cone_bridge import ConeBridge
from zap.conic.cone_utils import get_standard_conic_problem
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import torch
from zap.admm import ADMMSolver

np.set_printoptions(formatter={"float": "{:6.3f}".format})


def main():
    # Make a problem with two SOC constraints to test on
    x = cp.Variable(3)
    y = cp.Variable(4)
    objective = cp.Minimize(cp.sum_squares(x) + cp.sum_squares(y))
    constraints = [
        x[0] >= 0,
        cp.norm(x[1:]) <= x[0],
        x[0] + x[1] >= 1,
        y[0] >= 0,
        cp.norm(y[1:]) <= y[0],
        y[0] - y[1] <= 2,
    ]
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

    outcome = cone_bridge.solve()

    print("helllooooo")


if __name__ == "__main__":
    main()
