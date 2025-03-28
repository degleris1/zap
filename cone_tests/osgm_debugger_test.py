import cvxpy as cp
import numpy as np
import time
import clarabel
import torch
from zap.admm import ADMMSolver
from zap.conic.cone_bridge import ConeBridge
import scipy.sparse as sp
from zap.conic.cone_utils import (
    generate_max_flow_problem,
    is_valid_network,
    get_standard_conic_problem,
)


def main():
    ## Create a large problem that is valid
    n = 100
    seed = 42
    valid_source_sink_path = False

    while not valid_source_sink_path:
        problem, adj, inc = generate_max_flow_problem(n, seed)
        valid_source_sink_path = is_valid_network(adj)
        if not valid_source_sink_path:
            seed += 1

    # Get conic problem form so we can (i) solve standard conic form and (ii) solve using ZAP
    cone_params, data, cones = get_standard_conic_problem(problem, solver=cp.CLARABEL)

    # Solve the conic form using ZAP
    machine = "cpu"
    dtype = torch.float32
    cone_bridge = ConeBridge(cone_params)
    cone_admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
    cone_admm = ADMMSolver(
        machine=machine,
        dtype=dtype,
        atol=1e-6,
        rtol=1e-6,
        num_iterations=10000,
        eta=1e-3,
        use_osgm=True,
    )
    start_time = time.time()
    cone_solution_admm, cone_history_admm = cone_admm.solve(
        net=cone_bridge.net, devices=cone_admm_devices, time_horizon=cone_bridge.time_horizon
    )
    end_time = time.time()
    solve_time = end_time - start_time
    obj_val = cone_solution_admm.objective
    print(f"Objective value: {obj_val}")
    print(f"Time taken: {solve_time:.4f} seconds")


if __name__ == "__main__":
    main()
