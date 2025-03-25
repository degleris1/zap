import marimo as mo
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import torch
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from zap.admm import ADMMSolver

np.set_printoptions(formatter={"float": "{:6.3f}".format})
import sys
import os

from zap.conic.cone_bridge import ConeBridge


def main():
    m, n = 2, 5
    np.random.seed(42)

    # Create a random sparse matrix A
    density = 0.3
    A = sp.random(m, n, density=density, format="csc")
    c = np.random.randn(n)
    b = np.random.randn(m)
    x = cp.Variable(n)
    s = cp.Variable(m)
    constraints = [
        A @ x + s == b,
        s >= 0,
        x >= -5,
        x <= 5,
    ]

    obj = cp.Minimize(c.T @ x)
    problem = cp.Problem(obj, constraints)
    problem.solve()

    probdata, chain, inverse_data = problem.get_problem_data(cp.SCS)
    cone_dims = probdata["dims"]
    cones = {
        "z": cone_dims.zero,
        "l": cone_dims.nonneg,
        "q": cone_dims.soc,
        "ep": cone_dims.exp,
        "s": cone_dims.psd,
    }

    cone_params = {
        "A": probdata["A"],
        "b": probdata["b"],
        "c": probdata["c"],
        "K": cones,
    }

    cone_bridge = ConeBridge(cone_params)
    ### Test CVXPY
    # outcome = cone_bridge.net.dispatch(
    #     cone_bridge.devices, cone_bridge.time_horizon, solver=cp.CLARABEL, add_ground=False
    # )
    ### End Test CVXPY

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

    print("helllooooo")


if __name__ == "__main__":
    main()
