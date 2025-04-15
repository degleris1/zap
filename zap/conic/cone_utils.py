import numpy as np
import cvxpy as cp
import torch
import time
from zap.conic.variable_device import VariableDevice
from zap.conic.slack_device import SecondOrderConeSlackDevice
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from zap.conic.cone_bridge import ConeBridge
from zap.admm import ADMMSolver


def get_standard_conic_problem(problem, solver=cp.SCS):
    reducer = Dcp2Cone(problem=problem, quad_obj=False)
    conic_problem, _ = reducer.apply(problem)
    probdata, _, _ = conic_problem.get_problem_data(solver)
    data = {
        "A": probdata["A"],
        "b": probdata["b"],
        "c": probdata["c"],
    }
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

    return cone_params, data, cones


def get_conic_solution(solution, cone_bridge):
    """
    Given an admm solution, return the primal and slack variables.
    """
    x = []
    s = []
    for idx, device in enumerate(cone_bridge.devices):
        # Parse Variable Devices
        if type(device) is VariableDevice:
            tensor_list = [t.squeeze() for t in solution.power[idx]]
            p_tensor = torch.stack(tensor_list, dim=0).flatten()

            A_v = torch.tensor(device.A_v, dtype=torch.float32)
            A_expanded = torch.cat([torch.diag(A_v[i]) for i in range(A_v.shape[0])], dim=0)

            x_recovered = torch.linalg.lstsq(A_expanded, p_tensor).solution
            x.extend(x_recovered.view(-1).tolist())

        # Parse SOC Slacks
        elif type(device) is SecondOrderConeSlackDevice:
            soc_slacks = np.concatenate([t.flatten().numpy() for t in solution.power[idx]])

            s.extend(soc_slacks + device.b_d.flatten())
        # Parse Zero Cone and Nonnegative Slacks
        else:
            cone_slacks = solution.power[idx][0].flatten()
            s.extend(cone_slacks + device.b_d.flatten())

    return x, s


def get_problem_structure(problem):
    """
    Get the (conic) problem structure from a CVXPY problem.
    Returns as a dict the following information:
    - number of variables
    - number of constraints
    - cone dimensions
    - Sparsity of A
    - number of different variable devices
    - number of slack devices
    """
    structure = {}
    cone_params, data, cones = get_standard_conic_problem(problem)
    A = cone_params["A"]
    z = cones["z"]
    l = cones["l"]
    q = cones["q"]

    structure["m"] = A.shape[0]
    structure["n"] = A.shape[1]
    structure["density"] = A.nnz / (A.shape[0] * A.shape[1])
    structure["z"] = z
    structure["l"] = l
    structure["q"] = q  # this is a list of the sizes of all the SOC cones
    structure["var_devices"] = A.getnnz(axis=0)
    structure["unique_var_device_groups"] = np.unique(structure["var_devices"])
    structure["num_soc_devices"] = len(np.unique(np.array(q)))
    structure["num_var_devices"] = len(structure["unique_var_device_groups"])

    return structure


### Calling Custom Solvers (i.e. anything not via CVXPY, basically GPU accelerated solvers) ###
#######


# Zap
def solve_admm(problem, solver_args):
    """
    Call conic zap on a CVXPY problem.
    """
    cone_params, _, _ = get_standard_conic_problem(problem, solver=cp.SCS)
    cone_bridge = ConeBridge(cone_params)
    machine = solver_args.get("machine", "cpu")
    dtype = torch.float32
    admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
    admm = ADMMSolver(**solver_args)
    start_time = time.time()
    solution_admm, _ = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
    end_time = time.time()
    pobj = solution_admm.objective
    solve_time = end_time - start_time

    return pobj, solve_time


# CuClarabel
def solve_cuclarabel(problem, solver_args):
    """
    Call CuClarabel on a CVXPY problem.
    """
    raise NotImplementedError


# CuOSQP
def solve_cuosqp(problem, solver_args):
    """
    Call CuOSQP on a CVXPY problem.
    """
    raise NotImplementedError


# CuPDLP
def solve_cupdlp(problem, solver_args):
    """
    Call CuPDLP on a CVXPY problem.
    """
    raise NotImplementedError
