import numpy as np
import cvxpy as cp
import scipy.sparse as sp
from zap.conic.cone_utils import get_standard_conic_problem
from experiments.conic_solve.benchmarks.sparse_cone_benchmark import SparseConeBenchmarkSet


def create_simple_problem_zero_nonneg_cones(m=2, n=5, density=0.3, seed=42, quad_obj=False):
    """
    This problem has only zero and nonnegative cone constraints.
    """
    np.random.seed(seed)
    A = sp.random(m, n, density=density, format="csc", data_rvs=np.random.randn)
    c = np.random.randn(n)
    b = np.random.randn(m)

    x = cp.Variable(n)
    s = cp.Variable(m)
    constraints = [A @ x + s == b, s >= 0, x >= -5, x <= 5]
    if quad_obj:
        L = np.random.randn(n, n)
        P = L.T @ L
        objective = cp.Minimize(0.5 * cp.quad_form(x, P) + c.T @ x)
    else:
        objective = cp.Minimize(c.T @ x)
    problem = cp.Problem(objective, constraints)

    # Convert to conic form
    cone_params, _, _ = get_standard_conic_problem(problem, solver=cp.CLARABEL)

    return problem, cone_params


def create_simple_problem_soc(n=3, m=8, density=0.3, seed=42):
    """
    This problem has second order cone constraints.
    """
    np.random.seed(seed)
    A = sp.random(m, n, density=density, format="csc", data_rvs=np.random.randn)
    b = np.random.randn(m)
    c = np.random.randn(n)

    x = cp.Variable(n)
    s = cp.Variable(m)
    constraints = [
        A @ x + s == b,
        x >= -5,
        x <= 5,
        cp.norm(s[1:2]) <= s[0],
        cp.norm(s[3:5]) <= s[2],
        cp.norm(s[6:8]) <= s[5],
    ]
    objective = cp.Minimize(c.T @ x)
    problem = cp.Problem(objective, constraints)

    # Convert to conic form
    cone_params, _, _ = get_standard_conic_problem(problem, solver=cp.CLARABEL)
    return problem, cone_params


def create_simple_multi_block_problem_soc():
    sparse_cone_socp_benchmark = SparseConeBenchmarkSet(num_problems=1, n=15)
    for i, prob in enumerate(sparse_cone_socp_benchmark):
        problem = prob
        cone_params, _, _ = get_standard_conic_problem(problem, solver=cp.CLARABEL)

    return problem, cone_params
