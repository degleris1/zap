import numpy as np
import cvxpy as cp
import scipy.sparse as sp
import torch
import json
from zap.conic.variable_device import VariableDevice
from zap.conic.slack_device import (
    SecondOrderConeSlackDevice,
    ZeroConeSlackDevice,
    NonNegativeConeSlackDevice,
)
from zap.conic.quadratic_device import QuadraticDevice
from scipy.sparse.linalg import svds


def get_standard_conic_problem(problem, solver=cp.CLARABEL):
    # reducer = Dcp2Cone(problem=problem, quad_obj=False)
    # conic_problem, _ = reducer.apply(problem)
    probdata, _, _ = problem.get_problem_data(solver)
    data = {
        "P": probdata.get("P", None),
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
        "P": probdata.get("P", None),
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
    y = []
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
        elif type(device) in [ZeroConeSlackDevice, NonNegativeConeSlackDevice]:
            cone_slacks = solution.power[idx][0].flatten()
            s.extend(cone_slacks + device.b_d.flatten())

        elif type(device) is QuadraticDevice:
            # Parse Quadratic Device
            tesnor_list = [t.squeeze() for t in solution.power[idx]]
            p_tensor = torch.stack(tesnor_list, dim=0).flatten()
            y.extend(p_tensor.tolist())

    return x, s, y


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
    structure["cond_number"] = estimate_condition_number_sparse(A)
    structure["z"] = z
    structure["l"] = l
    structure["q"] = q  # this is a list of the sizes of all the SOC cones

    var_devices = A.getnnz(axis=0)
    unique, counts = np.unique(var_devices, return_counts=True)
    var_devices_dict = {int(k): int(v) for k, v in zip(unique, counts)}

    structure["var_devices"] = var_devices
    structure["var_devices_dict"] = json.dumps(var_devices_dict)
    structure["num_soc_devices"] = len(np.unique(np.array(q)))
    structure["num_var_devices"] = len(unique)

    return structure


def estimate_condition_number_sparse(A, fallback_tol=1e-12):
    try:
        _, s_max, _ = svds(A, k=1, which="LM", tol=1e-3)
        sigma_max = s_max[0]

        _, s_min, _ = svds(A, k=1, which="SM", tol=1e-2, maxiter=5000)
        sigma_min = s_min[0]

        if sigma_min < fallback_tol:
            raise ValueError("σ_min too small, possibly unstable")

        return sigma_max / sigma_min

    except Exception as e:
        print(f"Falling back on rough estimate for cond(A): {e}")

        fro_norm = fro_norm = np.sqrt((A.data**2).sum())
        m, n = A.shape
        approx_sigma_min = fro_norm / np.sqrt(min(m, n))

        if approx_sigma_min < fallback_tol:
            return np.inf

        return sigma_max / approx_sigma_min


### Utilities to Perform Ruiz Equilibration ###
def build_symmetric_M(A_csc: sp.csc_matrix, P: sp.csc_matrix | None = None) -> sp.csc_matrix:
    """
    Build the symmetric matrix M = [[P, A.T], [A, 0]]
    """
    m, n = A_csc.shape
    if P is None:
        P = sp.csc_matrix((n, n))
    zero_block = sp.csc_matrix((m, m))
    M = sp.bmat([[P, A_csc.T], [A_csc, zero_block]], format="csc")
    return M


def scale_cols_csc(A_csc: sp.csc_matrix, scale: np.ndarray):
    """
    This is an efficient way to do A@E where E is a diagonal matrix
    (i.e. E = diag(scale)).
    """
    A_csc.data *= np.repeat(scale, np.diff(A_csc.indptr))

    return A_csc


def scale_rows_csr(A_csr: sp.csr_matrix, scale: np.ndarray):
    """
    This is an efficient way to do D@A where D is a diagonal matrix.
    (i.e. D = diag(scale)).
    """
    A_csr.data *= np.repeat(scale, np.diff(A_csr.indptr))

    return A_csr
