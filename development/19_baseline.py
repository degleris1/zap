import marimo

__generated_with = "0.10.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pyomo
    import pyomo.environ as pyo
    import numpy as np
    import pandas as pd
    import datetime as dt
    import zap
    import pypsa
    return dt, mo, np, pd, pyo, pyomo, pypsa, zap


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
def _():
    from zap.pyomo.dispatch import setup_pyomo_model, parse_output
    from zap.admm.util import nested_subtract
    return nested_subtract, parse_output, setup_pyomo_model


@app.cell
def _(nested_subtract, np, parse_output, pyo, setup_pyomo_model, zap):
    def norm_check(x, ord=2):
        return np.linalg.norm([np.linalg.norm(np.concatenate(ps), ord=1) for ps in x if ps is not None], ord=1)

    def check_pyomo_dispatch(net, devices, time_horizon):
        model = setup_pyomo_model(net, devices, time_horizon)

        solver = pyo.SolverFactory("mosek")
        solver.solve(model, tee=False, options={})

        power, angle = parse_output(devices, model)

        result = net.dispatch(devices, time_horizon, solver=zap.network.cp.MOSEK, add_ground=False)

        print("Power Error: ", norm_check(nested_subtract(power, result.power)))
        print("Angle Error: ", norm_check(nested_subtract(angle, result.angle)))
        print("Objective Error: ", result.problem.value - pyo.value(model.objective))

        return model, result
    return check_pyomo_dispatch, norm_check


@app.cell
def _(check_pyomo_dispatch, np, zap):
    # Test 6-bus garver network
    def test_small():
        net, devices = zap.importers.load_garver_network(line_slack=5.0)
        devices += [zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))]
        time_horizon = 1

        model, result = check_pyomo_dispatch(net, devices, time_horizon)
        return model

    _model = test_small()
    return (test_small,)


@app.cell
def _(pypsa):
    pn = pypsa.Network()
    pn.import_from_csv_folder("data/pypsa/western/load_medium/elec_s_42")
    return (pn,)


@app.cell
def _(dt, np, pd, pn, zap):
    def get_pypsa_net(drop_battery=True, time_horizon=12):    
        start_date = dt.datetime(2019, 8, 9, 7)
        dates = pd.date_range(
            start_date,
            start_date + dt.timedelta(hours=time_horizon),
            freq="1h",
            inclusive="left",
        )
        net, devices = zap.importers.load_pypsa_network(
            pn,
            dates,
            scale_load=0.7,
            power_unit=1000.0,
            cost_unit=100.0,
            load_cost_perturbation=10.0,
            generator_cost_perturbation=1.0,
            ac_transmission_cost=1.0,
        )
        devices += [zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))]

        if drop_battery:
            devices = [d for d in devices if not isinstance(d, zap.Battery)]

        # Tweak AC line costs
        devices[3].linear_cost += 0.01 * np.random.rand(*devices[3].linear_cost.shape)

        return net, devices, time_horizon
    return (get_pypsa_net,)


@app.cell
def _(check_pyomo_dispatch, get_pypsa_net):
    def test_medium(drop_battery=True):
        net, devices, time_horizon = get_pypsa_net(drop_battery=drop_battery)
        model, result = check_pyomo_dispatch(net, devices, time_horizon)
        return model
    return (test_medium,)


@app.cell
def _():
    # _ = test_medium()
    return


@app.cell
def _(test_medium):
    _ = test_medium(drop_battery=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Bilevel Model""")
    return


@app.cell
def _():
    import pao
    return (pao,)


@app.cell
def _(get_pypsa_net):
    net, devices, time_horizon = get_pypsa_net(drop_battery=False, time_horizon=4)
    return devices, net, time_horizon


@app.cell
def _(devices):
    devices
    return


@app.cell
def _(devices, net, pao, pyo, setup_pyomo_model, time_horizon, zap):
    # Settings
    param_device_types = [zap.Generator, zap.DCLine, zap.Battery]
    param_devices = [i for i in range(len(devices)) if type(devices[i]) in param_device_types]
    print(param_devices)

    # Build model
    pyo_devices = [zap.pyomo.devices.convert_to_pyo(d) for d in devices]

    M = pyo.ConcreteModel()
    M.time_horizon = time_horizon
    M.time_index = pyo.RangeSet(0, time_horizon - 1)
    M.node_index = pyo.RangeSet(0, net.num_nodes - 1)

    # Create parameters
    params = []
    M.param_blocks = pyo.Block(param_devices)
    for p in param_devices:
        par = pyo_devices[p].make_parameteric(M.param_blocks[p])
        pyo_devices[p].add_investment_cost(M.param_blocks[p])

        params += [par]

    # Build dispatch problem
    M.dispatch = pao.pyomo.SubModel(fixed=params)
    setup_pyomo_model(net, devices, time_horizon, model=M.dispatch, pyo_devices=pyo_devices)

    # Create top level objective
    M.objective = pyo.Objective(
        expr=(M.dispatch.objective + sum(M.param_blocks[p].investment_cost for p in param_devices)),
        sense=pyo.minimize,
    )

    # Solve bilevel problem
    mip = pao.Solver("gurobi")
    solver = pao.Solver("pao.pyomo.FA", mip_solver=mip)  #, linearize_bigm=100.0)

    result = solver.solve(M, tee=True)
    return (
        M,
        mip,
        p,
        par,
        param_device_types,
        param_devices,
        params,
        pyo_devices,
        result,
        solver,
    )


@app.cell
def _():
    # pao_model = pao.pyomo.convert.convert_pyomo2MultilevelProblem(M)[0]
    # _opt = pao.Solver("gurobi")
    # _opt.solve(pao_model)

    # FA, MIBS, PCCG, REG
    # help(pao.Solver("pao.pyomo.REG").solve)
    return


@app.cell
def _(M, devices, np, parse_output):
    sum(np.round(parse_output(devices, M.dispatch)[0][0][0], decimals=2))
    return


@app.cell
def _(M, devices):
    sum([M.param_blocks[0].param[k].value for k in range(devices[0].num_devices)])
    return


if __name__ == "__main__":
    app.run()
