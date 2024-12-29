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
def _(np, zap):
    def split_incidence(A):
        return np.split(A.indices, A.indptr[1:-1])

    def get_node_groups(device: zap.devices.AbstractDevice):
        return [split_incidence(A.tocsr()) for A in device.incidence_matrix]
    return get_node_groups, split_incidence


@app.cell
def _(devices):
    [type(d) for d in devices]
    return


@app.cell
def _(devices, net, pyo, time_horizon):
    from zap.pyomo.devices import convert_to_pyo
    from zap.pyomo.model import setup_pyomo_model, parse_output

    _devs = [devices[d] for d in [0, 1, 3]]
    model = setup_pyomo_model(net, _devs, time_horizon)

    solver = pyo.SolverFactory("mosek")
    solver.solve(model, tee=False, options={})

    power, angle = parse_output(_devs, model)
    power
    return (
        angle,
        convert_to_pyo,
        model,
        parse_output,
        power,
        setup_pyomo_model,
        solver,
    )


@app.cell
def _():
    return


@app.cell
def _():
    # # Global constraints
    # def power_balance_rule(model, n, t):
    #     dev_ind = 0
    #     dev = devices[dev_ind]
    #     return 0.0 == sum(
    #             sum(sum(model.power[dev_ind][tau][k, t] for k in group_map[dev_ind][n]))
    #             for tau in range(dev.num_terminals_per_device)
    #         )


    #     # return 0.0 == sum(
    #     #     sum(
    #     #         sum(sum(model.power[dev_ind][tau][k, t] for k in group_map[dev_ind][n]))
    #     #         for tau in range(dev.num_terminals_per_device)
    #     #     )
    #     #     for dev_ind, dev in enumerate(devices)
    #     # )


    # model.power_balance = pyo.Constraint(node_index, time_index, rule=power_balance_rule)

    # model.power_balance = pyo.Constraint(
    #     node_index,
    #     expr=sum([get_net_power(d, p) for d, p in zip(devices, model.power)]),
    # )

    # Local constraints

    # Objective

    # Debug
    # ind_dev = 0
    # ind_ter = 0
    # dev = devices[0]

    # A = devices[ind_dev].incidence_matrix[ind_ter]  # @ model.power[ind_dev][ind_ter]
    # var = model.power[ind_dev][ind_ter]
    # power_dev = pyo.Expression(
    #     node_index,
    #     time_index,
    #     rule=lambda model, n, t: sum(
    #         A[n, k] * model.power[ind_dev][ind_ter][k, t] for k in range(dev.num_devices)
    #     ),
    # )


    # # TODO Use index groups (reverse sparse index)
    # power_dev.index_set()

    # power_balance
    return


if __name__ == "__main__":
    app.run()
