import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import marimo as mo
    import cvxpy as cp
    import numpy as np
    import scipy.sparse as sp
    from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
    np.set_printoptions(formatter={'float': '{:6.3f}'.format})
    import sys
    import os
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(PROJECT_ROOT)
    from zap.devices.cone_bridge import ConeBridge
    return ConeBridge, Dcp2Cone, PROJECT_ROOT, cp, mo, np, os, sp, sys


@app.cell
def _(cp, np, sp):
    m, n = 2, 5 
    np.random.seed(42)

    # Create a random sparse matrix A
    density = 0.3 
    A = sp.random(m, n, density=density, format='csc')
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
    return A, b, c, constraints, density, m, n, obj, problem, s, x


@app.cell
def _(cp, problem):
    probdata, chain, inverse_data = problem.get_problem_data(cp.SCS)
    cone_dims = probdata['dims']
    cones = {
        "z": cone_dims.zero,
        "l": cone_dims.nonneg,
        "q": cone_dims.soc,
        "ep": cone_dims.exp,
        "s": cone_dims.psd,
    }

    cone_params = {
      'A': probdata['A'],
      'b': probdata['b'],
      'c': probdata['c'],
      'K': cones,
    }

    print(f"A.shape: {cone_params['A'].shape}, c.shape: {cone_params['c'].shape}, b.shape: {cone_params['b'].shape}")
    return chain, cone_dims, cone_params, cones, inverse_data, probdata


@app.cell
def _(cones):
    cones
    return


@app.cell
def _(cone_params):
    print(type(cone_params['A']))
    return


@app.cell
def _(cone_params):
    print(cone_params['A'].toarray())
    return


@app.cell
def _(ConeBridge, cone_params):
    cone_bridge = ConeBridge(cone_params)
    return (cone_bridge,)


@app.cell
def _(cone_bridge):
    cone_bridge.variable_device_groups.keys()
    return


@app.cell
def _(cone_bridge):
    cone_bridge.variable_device_groups[3] # Block device group with 3 terminal variable devices
    return


@app.cell
def _(cone_bridge):
    cone_bridge.variable_device_groups[3][0][2].toarray()
    return


@app.cell
def _(cone_bridge):
    cone_bridge.devices[0].cost_vector # There are 3 three terminal devices in the first block
    return


@app.cell
def _(cone_bridge):
    cone_bridge.devices[1].cost_vector # There are 4 two terminal devices in the second block
    return


@app.cell
def _(cone_bridge):
    cone_bridge.devices[1].model_local_variables(1)[0]
    return


@app.cell
def _(cone_bridge):
    A_bv = cone_bridge.devices[0].A_bv
    return (A_bv,)


@app.cell
def _(A_bv):
    A_bv
    return


@app.cell
def _(cone_bridge, cp):
    outcome = cone_bridge.net.dispatch(cone_bridge.devices, cone_bridge.time_horizon, solver=cp.CLARABEL, add_ground=False)
    return (outcome,)


@app.cell
def _():
    from zap.admm import ADMMSolver
    import torch
    return ADMMSolver, torch


@app.cell
def _(cone_bridge, torch):
    machine = "cpu"
    dtype = torch.float32
    admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
    return admm_devices, dtype, machine


@app.cell
def _(ADMMSolver, admm_devices, cone_bridge, dtype, machine):
    admm = ADMMSolver(
        machine=machine,
        dtype=dtype,
        atol=1e-6,
        rtol=1e-6,
    )

    solution_admm, history_admm = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
    return admm, history_admm, solution_admm


@app.cell
def _(A):
    type(A)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
