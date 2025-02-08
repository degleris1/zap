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
        return np.linalg.norm(
            [np.linalg.norm(np.concatenate(ps), ord=1) for ps in x if ps is not None], ord=1
        )


    def check_pyomo_dispatch(net, devices, time_horizon):
        model = setup_pyomo_model(net, devices, time_horizon)

        solver = pyo.SolverFactory("mosek")
        solver.solve(model, tee=False, options={})

        power, angle = parse_output(devices, model)

        result = net.dispatch(
            devices, time_horizon, solver=zap.network.cp.MOSEK, add_ground=False
        )

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
    pn.import_from_csv_folder("data/pypsa/western/load_medium/elec_s_42_ec")
    return (pn,)


@app.cell
def _(dt, np, pd, pn, zap):
    def get_pypsa_net(drop_battery=True, time_horizon=12, ac_cost=0.0, ac_perturb=0.1):
        start_date = dt.datetime(2019, 8, 14, 7)
        dates = pd.date_range(
            start_date,
            start_date + dt.timedelta(hours=time_horizon),
            freq="1h",
            inclusive="left",
        )
        net, devices = zap.importers.load_pypsa_network(
            pn,
            dates,
            scale_load=1.0,
            power_unit=1000.0,
            cost_unit=100.0 * time_horizon,
            marginal_load_value=500.0,
            load_cost_perturbation=10.0,
            generator_cost_perturbation=1.0,
            scale_generator_capacity_factor=0.7,
            scale_line_capacity_factor=0.7,
            drop_empty_generators=False,
            expand_empty_generators=0.5,
            ac_transmission_cost=ac_cost,
        )
        devices += [zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))]

        if drop_battery or time_horizon == 1:
            devices = [d for d in devices if not isinstance(d, zap.Battery)]

        # Tweak AC line costs
        devices[3].linear_cost += ac_perturb * np.random.rand(*devices[3].linear_cost.shape)

        return net, devices, time_horizon
    return (get_pypsa_net,)


@app.cell
def _(check_pyomo_dispatch, get_pypsa_net):
    def test_medium(drop_battery=True):
        net, devices, time_horizon = get_pypsa_net(
            drop_battery=drop_battery, ac_cost=1.0, ac_perturb=0.1
        )
        model, result = check_pyomo_dispatch(net, devices, time_horizon)
        return model, result
    return (test_medium,)


@app.cell
def _(test_medium):
    toy_pyo, toy_zap = test_medium(drop_battery=False)
    return toy_pyo, toy_zap


@app.cell
def _(pyo, toy_pyo):
    print(pyo.value(toy_pyo.device[0].emissions))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Bilevel Model""")
    return


@app.cell
def _():
    import pao
    from zap.pyomo.bilevel import solve_bilevel_model
    from zap.planning import DispatchCostObjective, EmissionsObjective, MultiObjective
    return (
        DispatchCostObjective,
        EmissionsObjective,
        MultiObjective,
        pao,
        solve_bilevel_model,
    )


@app.cell
def _(get_pypsa_net):
    net, devices, time_horizon = get_pypsa_net(drop_battery=False, time_horizon=1)
    devices
    return devices, net, time_horizon


@app.cell
def _(
    DispatchCostObjective,
    EmissionsObjective,
    MultiObjective,
    devices,
    net,
):
    planner_objective = MultiObjective(
        objectives=[DispatchCostObjective(net, devices), EmissionsObjective(devices)],
        weights=[1.0, 100.0],
    )
    return (planner_objective,)


@app.cell
def _(
    devices,
    net,
    planner_objective,
    solve_bilevel_model,
    time_horizon,
    zap,
):
    bilevel_model, _ = solve_bilevel_model(
        net,
        devices,
        time_horizon,
        planner_objective,
        param_device_types=[zap.Generator, zap.DCLine, zap.Battery],
        mip_solver="gurobi",
        mip_solver_options={"MIPGap": 0.01, "Threads": 64, "MIPFocus": 0},
        pao_solver="pao.pyomo.FA",
        verbose=True,
        integer_investments=False,
        integer_quantity=0.5,
    )
    return (bilevel_model,)


@app.cell(hide_code=True)
def _(bilevel_model, devices, np):
    gen_invest = np.round(
        np.array(
            [
                bilevel_model.param_blocks[0].param[k].value
                for k in range(devices[0].num_devices)
            ]
        )
        - devices[0].min_nominal_capacity.reshape(-1),
        decimals=3,
    )

    line_invest = np.round(
        np.array(
            [
                bilevel_model.param_blocks[2].param[k].value
                for k in range(devices[2].num_devices)
            ]
        )
        - devices[2].nominal_capacity.reshape(-1),
        decimals=3,
    )
    return gen_invest, line_invest


@app.cell
def _(gen_invest):
    {i: gen_invest[i] for i in range(len(gen_invest)) if gen_invest[i] != 0.0}
    return


@app.cell
def _(line_invest):
    line_invest
    return


@app.cell
def _(bilevel_model, devices, np, parse_output):
    sum(np.round(parse_output(devices, bilevel_model.dispatch)[0][0][0], decimals=2))
    return


@app.cell
def _(bilevel_model, pyo):
    pyo.value(bilevel_model.param_blocks[0].investment_cost)
    return


@app.cell
def _(bilevel_model, pyo):
    pyo.value(bilevel_model.dispatch.device[0].emissions)
    return


if __name__ == "__main__":
    app.run()
