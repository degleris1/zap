import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cvxpy as cp
    import numpy as np
    import scipy.sparse as sp
    import torch
    import scs

    from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
    from zap.conic.cone_bridge import ConeBridge
    from zap.conic.variable_device import VariableDevice
    from zap.conic.slack_device import SlackDevice

    from zap.admm import ADMMSolver

    np.set_printoptions(formatter={'float': '{:6.3f}'.format})
    return (
        ADMMSolver,
        ConeBridge,
        Dcp2Cone,
        SlackDevice,
        VariableDevice,
        cp,
        mo,
        np,
        scs,
        sp,
        torch,
    )


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
def _(A):
    A.todense()
    return


@app.cell
def _(b):
    b
    return


@app.cell
def _(cone_bridge):
    cone_bridge.devices[2]
    return


@app.cell
def _(s):
    s.value
    return


@app.cell(hide_code=True)
def _(cp, problem, scs):
    probdata, chain, inverse_data = problem.get_problem_data(cp.SCS)

    data = {
      'A': probdata['A'],
      'b': probdata['b'],
      'c': probdata['c']
    }
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

    soln = scs.solve(data, cones, verbose=False)
    return (
        chain,
        cone_dims,
        cone_params,
        cones,
        data,
        inverse_data,
        probdata,
        soln,
    )


@app.cell
def _(ConeBridge, cone_params):
    cone_bridge = ConeBridge(cone_params)
    return (cone_bridge,)


@app.cell
def _(cone_bridge, cp):
    ystar = cone_bridge.solve(cp.CLARABEL)
    return (ystar,)


@app.cell
def _(ystar):
    ystar.problem.value
    return


@app.cell
def _(ystar):
    ystar.problem.status
    return


@app.cell
def _(ystar):
    ystar.local_variables
    return


@app.cell
def _(s, x):
    x.value, s.value
    return


@app.cell
def _(cone_params):
    cone_params["A"].todense()
    return


@app.cell
def _(cone_params):
    cone_params["b"]
    return


@app.cell
def _(cone_bridge):
    cone_bridge.terminal_groups
    return


@app.cell
def _(cone_bridge):
    cone_bridge.device_group_map_list
    return


@app.cell
def _(cone_bridge):
    cone_bridge.devices[2]
    return


@app.cell
def _(cone_params):
    cone_params["b"]
    return


@app.cell
def _(cone_bridge, torch):
    machine = "cpu"
    dtype = torch.float32
    admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
    return admm_devices, dtype, machine


@app.cell
def _(torch):
    print(torch.cuda.is_available())
    return


@app.cell
def _(ADMMSolver, admm_devices, cone_bridge, dtype, machine):
    admm = ADMMSolver(
        machine=machine,
        dtype=dtype,
        atol=1e-3,
        rtol=1e-3,
        track_objective=True,
        rtol_dual_use_objective=True,
        num_iterations = 10000,
    )

    solution_admm, history_admm = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
    return admm, history_admm, solution_admm


@app.cell
def _(solution_admm):
    solution_admm.objective
    return


@app.cell
def _():
    # def get_solution(solution_admm, cone_bridge):
    #     """
    #     Parses out solution from the "flow" view into actual primal and slacks to compare against SCS
    #     """
    #     x = []
    #     s = []
    #     for idx, device in enumerate(cone_bridge.devices):
    #         if type(device) is VariableDevice:
    #             tensor_list = [t.squeeze() for t in solution_admm.power[idx]]
    #             p_tensor = torch.stack(tensor_list, dim=0).flatten() 

    #             A_v = torch.tensor(device.A_v, dtype=torch.float32)
    #             A_expanded = torch.cat([torch.diag(A_v[i]) for i in range(A_v.shape[0])], dim=0)

    #             x_recovered = torch.linalg.lstsq(A_expanded, p_tensor).solution

    #             x.extend(x_recovered.squeeze())


    #         else:
    #             cone_slacks = solution_admm.power[idx][0].flatten()
    #             s.extend(cone_slacks + device.b_d.flatten())


    #     return x, s
    return


@app.cell
def _():
    # x_admm, s_admm = get_solution(solution_admm, cone_bridge)
    return


@app.cell
def _(soln):
    ## SCS Solution
    soln
    return


@app.cell
def _():
    # num_terminals_per_device_list = np.diff(A_sparse.indptr)
    # terminal_groups = np.sort(np.unique(num_terminals_per_device_list))
    # device_group_map_dict = {g: np.argwhere(num_terminals_per_device_list == g).flatten() for g in terminal_groups}
    # device_group_map_list = [np.argwhere(num_terminals_per_device_list == g).flatten() for g in terminal_groups]

    # for group_idx, num_terminals_per_device in enumerate(terminal_groups):
    #     device_idxs = device_group_map_list[group_idx]
    #     num_devices = len(device_idxs)

    #     A_devices = self.A[:, device_idxs]
    #     # (i) A submatrix: (num_terminals_per_device, num_devices)
    #     A_v = A_devices.data.reshape(num_devices,num_terminals_per_device).T
    #     ## (ii) terminal_device_array: (num_devices, num_terminals_per_device)
    #     terminal_device_array = A_devices.indices.reshape(num_devices, num_terminals_per_device)
    #     ## (iii) cost vector (subvector of c taking the corresponding device elements)
    #     cost_vector = self.c[device_idxs]
    return


@app.cell
def _(cone_params):
    print(cone_params['A'].toarray())
    return


if __name__ == "__main__":
    app.run()
