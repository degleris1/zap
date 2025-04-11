import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import scipy.io
    from scipy.sparse import csc_matrix
    return cp, csc_matrix, mo, np, scipy


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
def _(cp, problem):
    problem.solve(solver=cp.SCS, verbose=True)
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
