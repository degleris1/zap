import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import scipy.io
    import sys
    import os
    import zap
    import pandas as pd
    from scipy.sparse import csc_matrix
    from experiments.conic_solve.benchmarks.sparse_cone_benchmark import SparseConeBenchmarkSet
    from experiments.conic_solve.benchmarks.lasso_benchmark import LassoBenchmarkSet
    from experiments.conic_solve.benchmarks.maros_benchmark import MarosBenchmarkSet
    from experiments.conic_solve.benchmarks.netlib_benchmark import NetlibBenchmarkSet
    from experiments.conic_solve.benchmarks.max_flow_benchmark import MaxFlowBenchmarkSet
    from zap.conic.cone_utils import get_problem_structure
    return (
        LassoBenchmarkSet,
        MarosBenchmarkSet,
        MaxFlowBenchmarkSet,
        NetlibBenchmarkSet,
        SparseConeBenchmarkSet,
        cp,
        csc_matrix,
        get_problem_structure,
        mo,
        np,
        os,
        pd,
        scipy,
        sys,
        zap,
    )


@app.cell
def _(
    LassoBenchmarkSet,
    MarosBenchmarkSet,
    MaxFlowBenchmarkSet,
    NetlibBenchmarkSet,
    SparseConeBenchmarkSet,
):
    ## Initialize test problem benchmarks (and also that Max flow problem)
    maros_benchmark = MarosBenchmarkSet(data_dir="data/conic_benchmarks/maros")
    netlib_benchmark = NetlibBenchmarkSet(data_dir="data/conic_benchmarks/netlib")
    lasso_small_benchmark = LassoBenchmarkSet(num_problems=3, n=10000, m=2000, density=0.01, base_seed=0)
    lasso_small_sparser_benchmark = LassoBenchmarkSet(num_problems=3, n=30000, m=6000, density=0.01, base_seed=0)
    lasso_small_denser_benchmark = LassoBenchmarkSet(num_problems=3, n=30000, m=6000, density=0.1, base_seed=0)
    sparse_cone_lp_benchmark = SparseConeBenchmarkSet(num_problems=3, n=10000, p_f=0.5, p_l=0.5)
    sparse_cone_socp_benchmark = SparseConeBenchmarkSet(num_problems=3, n=10000)
    max_flow_benchmark = MaxFlowBenchmarkSet(num_problems=3, n=10000, base_seed=42)

    # all_benchmarks = [maros_benchmark, netlib_benchmark, lasso_small_benchmark, lasso_small_sparser_benchmark, lasso_small_denser_benchmark, sparse_cone_lp_benchmark, sparse_cone_socp_benchmark]
    # all_benchmark_names = ["maros", "netlib", "lasso_small", "lasso_small_sparser", "lasso_small_denser", "sparse_cone_lp", "sparse_cone_socp"]

    all_benchmarks = [netlib_benchmark, lasso_small_benchmark, lasso_small_sparser_benchmark, lasso_small_denser_benchmark, sparse_cone_lp_benchmark, sparse_cone_socp_benchmark, max_flow_benchmark]
    all_benchmark_names = ["netlib", "lasso_small", "lasso_small_sparser", "lasso_small_denser", "sparse_cone_lp", "sparse_cone_socp", "max_flow"]
    return (
        all_benchmark_names,
        all_benchmarks,
        lasso_small_benchmark,
        lasso_small_denser_benchmark,
        lasso_small_sparser_benchmark,
        maros_benchmark,
        max_flow_benchmark,
        netlib_benchmark,
        sparse_cone_lp_benchmark,
        sparse_cone_socp_benchmark,
    )


@app.cell
def _(all_benchmark_names, all_benchmarks, get_problem_structure, pd):
    rows = []
    for idx, benchmark in enumerate(all_benchmarks):
        benchmark_name = all_benchmark_names[idx]
        for i, prob in enumerate(benchmark):
            structure = get_problem_structure(prob)
            row = {
                "benchmark_name": benchmark_name,
                "problem_index": i
            }
            row.update(structure)
            rows.append(row)

    benchmark_probs_structure = pd.DataFrame(rows)
    return (
        benchmark,
        benchmark_name,
        benchmark_probs_structure,
        i,
        idx,
        prob,
        row,
        rows,
        structure,
    )


@app.cell
def _(all_benchmarks):
    all_benchmarks
    return


@app.cell
def _(benchmark_probs_structure):
    benchmark_probs_structure.head(20)
    return


@app.cell
def _(scipy):
    # Load a maros problem
    maros_mat_filepath = '/Users/akshaysreekumar/Documents/Stanford/S3L/zap/data/conic_benchmarks/maros/AUG2D.mat'
    netlib_mat_filepath = '/Users/akshaysreekumar/Documents/Stanford/S3L/zap/data/conic_benchmarks/netlib/25fv47.mat'
    data = scipy.io.loadmat(maros_mat_filepath)
    return data, maros_mat_filepath, netlib_mat_filepath


@app.cell
def _(data):
    data
    return


@app.cell
def _():
    # A = csc_matrix(data["A"].astype(float))
    # c = data["c"].flatten().astype(float)
    # b = data["b"].flatten().astype(float)
    # z0 = data["z0"].flatten().astype(float)[0]
    # lo = data["lo"].flatten().astype(float)
    # hi = data["hi"].flatten().astype(float)
    # n = A.shape[1]

    # hi = np.where(hi >= 1e19, np.inf, hi)
    # lo = np.where(lo <= -1e19, -np.inf, lo)


    # x = cp.Variable(n)
    # objective = cp.Minimize(c@x + z0)
    # constraints = [A@x==b, lo <= x, x <= hi]
    # problem = cp.Problem(objective, constraints)
    # problem.solve(solver='CLARABEL')
    return


@app.cell
def _(cp, csc_matrix, data, np):
    P = csc_matrix(data["P"].astype(float))
    q = data["q"].flatten().astype(float)
    A = csc_matrix(data["A"].astype(float))
    l = data["l"].flatten().astype(float)
    u = data["u"].flatten().astype(float)
    r = data["r"].flatten().astype(float)[0]
    m = data["m"].flatten().astype(int)[0]
    n = data["n"].flatten().astype(int)[0]

    l[l > +9e19] = +np.inf
    u[u > +9e19] = +np.inf
    l[l < -9e19] = -np.inf
    u[u < -9e19] = -np.inf

    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x + r)

    # Create the constraints using vectorized operations.
    # This represents l <= A x <= u.
    constraints = [A @ x >= l, A @ x <= u]

    # Build and return the CVXPY problem.
    problem = cp.Problem(objective, constraints)
    return A, P, constraints, l, m, n, objective, problem, q, r, u, x


@app.cell
def _(problem):
    problem.solver_stats.solve_time
    return


@app.cell
def _():
    # problem.solve(solver=cp.SCS, verbose=True)
    return


@app.cell
def _(maros_data):
    maros_data["A"].shape
    return


@app.cell
def _(l):
    l.shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
