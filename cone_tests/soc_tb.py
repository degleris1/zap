import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import cvxpy as cp
    import numpy as np
    import scipy.sparse as sp
    import time
    import torch
    from zap.admm import ADMMSolver
    from zap.conic.cone_bridge import ConeBridge
    import scipy.sparse as sp
    import scs
    from zap.conic.cone_utils import get_standard_conic_problem, get_conic_solution
    from zap.conic.variable_device import VariableDevice
    from zap.conic.slack_device import SlackDevice
    return (
        ADMMSolver,
        ConeBridge,
        SlackDevice,
        VariableDevice,
        cp,
        get_conic_solution,
        get_standard_conic_problem,
        np,
        scs,
        sp,
        time,
        torch,
    )


@app.cell
def _(cp, np, sp):
    n = 3 
    m = 8

    np.random.seed(42)
    density = 0.3

    # Create a random sparse matrix A of shape (m, n)
    A = sp.random(m, n, density=density, format='csc', data_rvs=np.random.randn)
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

    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.CLARABEL)

    print("Optimal value:", prob.value)
    print("Optimal x:", x.value)
    print("Optimal s:", s.value)
    return A, b, c, constraints, density, m, n, objective, prob, result, s, x


@app.cell
def _(cp, get_standard_conic_problem, prob):
    cone_params, data, cones = get_standard_conic_problem(prob, solver=cp.CLARABEL)
    return cone_params, cones, data


@app.cell
def _(ConeBridge, cone_params):
    cone_bridge = ConeBridge(cone_params)
    return (cone_bridge,)


@app.cell
def _(cones, data, scs):
    ## Solve conic form using SCS
    soln = scs.solve(data, cones, verbose=False)
    return (soln,)


@app.cell
def _(cone_params):
    cone_params
    return


@app.cell
def _(cone_bridge):
    ## Test cvxpy

    outcome = cone_bridge.solve()
    outcome.problem.value
    return (outcome,)


@app.cell
def _(ADMMSolver, cone_bridge, torch):
    ### Test ADMM
    machine = "cpu"
    dtype = torch.float32
    admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
    admm = ADMMSolver(
        machine=machine,
        dtype=dtype,
        atol=1e-9,
        rtol=1e-9,
        num_iterations=10000
        # track_objective=False,
        # rtol_dual_use_objective=False,
    )
    solution_admm, history_admm = admm.solve(
        cone_bridge.net, admm_devices, cone_bridge.time_horizon)
    return admm, admm_devices, dtype, history_admm, machine, solution_admm


@app.cell
def _(cone_bridge, get_conic_solution, solution_admm):
    x_admm, s_admm = get_conic_solution(solution_admm, cone_bridge)
    return s_admm, x_admm


@app.cell
def _(solution_admm):
    solution_admm.objective
    return


@app.cell
def _(soln):
    soln['info']["pobj"]
    return


@app.cell
def _(cone_params):
    cone_params['A'].toarray().shape
    return


if __name__ == "__main__":
    app.run()
