import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import cvxpy as cp
    import numpy as np
    import time
    import torch
    from zap.admm import ADMMSolver
    from zap.conic.cone_bridge import ConeBridge
    import scipy.sparse as sp
    import scs
    from zap.conic.cone_utils import get_standard_conic_problem
    return (
        ADMMSolver,
        ConeBridge,
        cp,
        get_standard_conic_problem,
        np,
        scs,
        sp,
        time,
        torch,
    )


@app.cell
def _(cp):
    # x = cp.Variable(3)
    # y = cp.Variable(4)
    # objective = cp.Minimize(cp.sum_squares(x) + cp.sum_squares(y))
    # constraints = [
    #     x[0] >= 0,
    #     cp.norm(x[1:]) <= x[0],
    #     x[0] + x[1] >= 1,

    #     y[0] >= 0,
    #     cp.norm(y[1:]) <= y[0],
    #     y[0] - y[1] <= 2
    # ]
    # prob = cp.Problem(objective, constraints)
    # result = prob.solve()

    # print("Optimal value:", prob.value)
    # print("Optimal x:", x.value)
    # print("Optimal y:", y.value)
    x = cp.Variable(2)
    y = cp.Variable(2)
    z = cp.Variable(2)

    # The objective minimizes the sum of squares of the first entries.
    objective = cp.Minimize(cp.square(x[0]) + cp.square(y[0]) + cp.square(z[0]))

    # Each SOC constraint enforces that the first entry is at least as large as
    # the absolute value of the second entry. These form three separate cone blocks.
    # The linear constraints couple the three variables:
    #  - x[0] + y[0] + z[0] == 12  forces the sum of the first entries to be 12,
    #  - x[0] - y[0] == 1          forces x[0] to be 1 greater than y[0],
    #  - y[0] - z[0] == 1          forces y[0] to be 1 greater than z[0].
    # With these constraints, we have:
    #   x[0] = y[0] + 1, and z[0] = y[0] - 1, so:
    #   (y[0]+1) + y[0] + (y[0]-1) = 3*y[0] = 12  →  y[0] = 4.
    # Then, x[0] = 5 and z[0] = 3.
    # Thus the objective evaluates to 5^2 + 4^2 + 3^2 = 25 + 16 + 9 = 50.
    constraints = [
        x[0] >= cp.norm(x[1:]),
        y[0] >= cp.norm(y[1:]),
        z[0] >= cp.norm(z[1:]),
        x[0] + y[0] + z[0] == 12,
        x[0] - y[0] == 1,
        y[0] - z[0] == 1
    ]

    # Define and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    print("Optimal value:", prob.value)
    print("Optimal x:", x.value)
    print("Optimal y:", y.value)
    print("Optimal z:", z.value)
    return constraints, objective, prob, x, y, z


@app.cell
def _(cp, get_standard_conic_problem, prob):
    cone_params, data, cones = get_standard_conic_problem(prob, solver=cp.CLARABEL)
    return cone_params, cones, data


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
def _(soln):
    soln['info']["pobj"]
    return


@app.cell
def _(soln):
    soln
    return


app._unparsable_cell(
    r"""
    scaled_x = soln[\"x\"][:-1]
    tau = soln[\"x\"][-1]
    x_og = scaled_x/tau
    obj_og = 
    """,
    name="_"
)


@app.cell
def _(cones):
    cones
    return


@app.cell
def _(cone_params):
    cone_params
    return


@app.cell
def _(cone_params):
    cone_params['A'].toarray()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
