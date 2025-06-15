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
import sparse


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


def stack_cvxpy_colwise(instance_list):
    """
    Note in this function we use pydata.sparse to form multi-dimensional sparse arrays (along batch dim).
    """
    # Grab one problem as template
    base = get_standard_conic_problem(instance_list[0])[0]
    base_A = base["A"]
    m, n = base_A.shape
    T = len(instance_list)

    # Initialize batched b and c
    b_mat = np.zeros((m, T))
    c_mat = np.zeros((n, T))

    coos = []
    base_A_coo = sparse.COO.from_scipy_sparse(base_A)
    coos.append(base_A_coo)
    b_mat[:, 0] = base["b"]
    c_mat[:, 0] = base["c"]
    for j, prob in enumerate(instance_list[1:], start=1):
        cone, *_ = get_standard_conic_problem(prob)
        cur_A = cone["A"].tocsc()

        # Ensure sparsity pattern is identical
        assert np.array_equal(cur_A.indptr, base_A.indptr) and np.array_equal(cur_A.indices, base_A.indices)

        b_mat[:, j] = cone["b"]
        c_mat[:, j] = cone["c"]
        cur_coo = sparse.COO.from_scipy_sparse(cone["A"])
        coos.append(cur_coo)

    A_coo = sparse.stack(coos, axis=2)
    A_mat = A_coo.asformat("gcxs", compressed_axes=[1])

    cone_params = dict(P=base["P"], A=A_mat, b=b_mat, c=c_mat, K=base["K"])
    return cone_params

def compute_batch_objectives(state, devices):
    """
    Compute the objective values for each batch in the batched solution.
    
    Args:
        state: ADMMState containing the solution
        devices: List of devices
        
    Returns:
        torch.Tensor of shape (time_horizon,) containing objective values for each batch
    """
    # Get the batch dimension (time_horizon)
    time_horizon = state.power[0][0].shape[-1]
    device = state.power[0][0].device
    
    # Create tensor to hold batch objectives
    batch_objectives = torch.zeros(time_horizon, device=device)
    
    # For each batch
    for t in range(time_horizon):
        batch_costs = []
        
        # For each device
        for i, d in enumerate(devices):
            # Extract this batch's power for all terminals
            batch_power = [terminal[:, t] for terminal in state.power[i]]
            
            # Extract this batch's local variables (if any)
            lv0 = state.local_variables[i]
            lv_slice = None if lv0 is None else  [lv[:, t] for lv in lv0]
            
            # For VariableDevice, we need to use just this batch's cost_vector
            if isinstance(d, VariableDevice):
                c_t = d.cost_vector[:, t]
                x_t = lv_slice[0]
                cost = torch.dot(c_t, x_t)                
            else:
                # For other devices, just compute the cost
                cost = d.operation_cost(batch_power, None, lv_slice, la=torch)
            
            batch_costs.append(cost)
        
        # Sum all device costs for this batch
        batch_objectives[t] = sum(batch_costs)
    
    return batch_objectives


def map_cvxpy_parameters_to_cone_program(problem, parameters, solver=cp.CLARABEL):
    """
    TODO: Re-write this function to not do the difference of applications eventually.
    """    
    probdata, chain, inverse_data = problem.get_problem_data(solver=solver)
    compiler = probdata[cp.settings.PARAM_PROB]
    
    has_quadratic_obj = compiler.reduced_P.matrix_data is not None
    
    # Create parameter mappings using the difference method
    param_mappings = {}
    for i, param in enumerate(parameters):
        param_name = param.name()
        param_id = param.id
        
        original_value = param.value
        
        # Test values
        if param.shape == ():  # Scalar parameter
            test_value1 = 1.0
            test_value2 = 0.0
        else:
            test_value1 = np.ones(param.shape)
            test_value2 = np.zeros(param.shape)
        
        param.value = test_value1
        if has_quadratic_obj:
            P1, c1, d1, A1, b1 = compiler.apply_parameters(quad_obj=True)
        else:
            c1, d1, A1, b1 = compiler.apply_parameters(quad_obj=False)
            P1 = None
        
        param.value = test_value2
        if has_quadratic_obj:
            P2, c2, d2, A2, b2 = compiler.apply_parameters(quad_obj=True)
        else:
            c2, d2, A2, b2 = compiler.apply_parameters(quad_obj=False)
            P2 = None
        
        param.value = original_value
        
        c_diff = np.where(c1 != c2)[0].tolist()
        b_diff = np.where(b1 != b2)[0].tolist()
        d_affected = d1 != d2

        if sp.issparse(A1):
            A_diff = (A1 != A2).nonzero()
            A_affected = list(zip(A_diff[0].tolist(), A_diff[1].tolist()))
        else:
            A_diff = np.where(A1 != A2)
            A_affected = list(zip(A_diff[0].tolist(), A_diff[1].tolist()))
        
        P_affected = []
        if P1 is not None and P2 is not None:
            if sp.issparse(P1):
                P_diff = (P1 != P2).nonzero()
                P_affected = list(zip(P_diff[0].tolist(), P_diff[1].tolist()))
            else:
                P_diff = np.where(P1 != P2)
                P_affected = list(zip(P_diff[0].tolist(), P_diff[1].tolist()))
        
        param_mappings[param_name] = {
            'id': param_id,
            'size': param.size,
            'param_col': compiler.param_id_to_col.get(param_id, None),
            'c': c_diff,
            'b': b_diff,
            'A': A_affected,
            'P': P_affected,
            'd': d_affected

        }
    
    return param_mappings


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


def scale_cols(A, scale):
    """
    Column-scale A in place.
    Supports both SciPy CSC and GCXS matrices.
    """
    if not isinstance(A, sparse.GCXS):
        return scale_cols_csc(A, scale)
    
    # GCXS case
    if hasattr(A, 'compressed_axes') and A.compressed_axes == (1,):
        indptr = A.indptr
        data = A.data
        for j in range(len(indptr) - 1):
            data[indptr[j]:indptr[j + 1]] *= scale[j]
        return A

def scale_rows(A, scale):
    """
    Row-scale A in place.
    Supports both SciPy CSC and GCXS matrices.
    """
    if not isinstance(A, sparse.GCXS):
        A_csr = A.tocsr()
        A_csr = scale_rows_csr(A_csr, scale)
        return A_csr.tocsc()
    else: # GCXS case
        indices = A.indices
        data = A.data
        rows = indices[0]
        data *= scale[rows]
        return A

