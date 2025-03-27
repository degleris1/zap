import torch
import cvxpy as cp
import scs
from zap.importers.toy import load_test_network
from zap.admm import ADMMSolver
from zap.conic.cone_utils import get_standard_conic_problem

from zap.conic.cone_bridge import ConeBridge


def main():
    net, devices = load_test_network()
    time_horizon = 4
    machine = "cpu"
    dtype = torch.float32

    ## Solve the conic form of this problem using CVXPY
    outcome = net.dispatch(devices, time_horizon, solver=cp.CLARABEL, add_ground=False)
    problem = outcome.problem
    cone_params, data, cones = get_standard_conic_problem(problem, cp.CLARABEL)
    soln = scs.solve(data, cones, verbose=False)

    ## Create cone bridge for this same problem
    cone_bridge = ConeBridge(cone_params)
    cone_admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
    cone_admm = ADMMSolver(
        machine=machine,
        dtype=dtype,
        atol=1e-3,
        rtol=1e-3,
        track_objective=True,
        rtol_dual_use_objective=True,
        num_iterations=3000,
    )

    cone_solution_admm, _ = cone_admm.solve(
        net=cone_bridge.net, devices=cone_admm_devices, time_horizon=cone_bridge.time_horizon
    )
    print("SCS Cone Objective: ", soln["info"]["pobj"])
    print("ADMM Cone Objective: ", cone_solution_admm.objective)
    print("pause")


if __name__ == "__main__":
    main()
