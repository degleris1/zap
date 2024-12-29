import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pyomo
    import pyomo.environ as pyo
    import numpy as np
    import zap
    return mo, np, pyo, pyomo, zap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Basic Model""")
    return


@app.cell
def _(np, pyo):
    def example_pyomo(verbose=False):
        m = 20
        n = 100

        A = np.random.rand(m, n)
        b = np.random.rand(m)
        c = np.random.randn(n)

        power_index = pyo.RangeSet(0, n - 1)
        constr_index = pyo.RangeSet(0, m - 1)

        model = pyo.ConcreteModel()
        model.x = pyo.Var(power_index)

        model.ineq_constr = pyo.Constraint(power_index, expr=model.x >= np.zeros(n))
        model.eq_constr = pyo.Constraint(constr_index, expr=A @ model.x <= b)
        model.objective = pyo.Objective(expr=c.T @ model.x, sense=pyo.minimize)

        solution = pyo.SolverFactory("mosek").solve(model, tee=verbose, options={})

        return model


    _model = example_pyomo()
    # _model.display()
    _x = np.array([_model.x[i].value for i in _model.x])
    return (example_pyomo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dispatch Model""")
    return


@app.cell
def _(np, zap):
    # Initialize network
    net, devices = zap.importers.load_garver_network(line_slack=5.0)
    devices += [zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))]
    time_horizon = 1
    return devices, net, time_horizon


@app.cell
def _(devices, net, pyo, time_horizon):
    from zap.pyomo.model import setup_pyomo_model, parse_output

    _devs = [devices[d] for d in [0, 1, 2, 3]]
    model = setup_pyomo_model(net, _devs, time_horizon)

    solver = pyo.SolverFactory("mosek")
    solver.solve(model, tee=False, options={})

    power, angle = parse_output(_devs, model)
    power
    return angle, model, parse_output, power, setup_pyomo_model, solver


@app.cell
def _(devices, net, time_horizon, zap):
    result = net.dispatch(devices, time_horizon, solver=zap.network.cp.CLARABEL)
    result.power
    return (result,)


if __name__ == "__main__":
    app.run()
