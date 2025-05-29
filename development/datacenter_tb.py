import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import os
    import pypsa

    import zap
    from zap.importers.pypsa import load_pypsa_network
    return cp, load_pypsa_network, mo, np, os, plt, pypsa, sns, zap


@app.cell
def _(os, pypsa):
    HOME_PATH = os.environ.get('HOME')
    PYPSA_NETW0RK_PATH = HOME_PATH + '/zap_data/pypsa-networks/western_small/network_2021.nc'
    pn = pypsa.Network(PYPSA_NETW0RK_PATH)
    snapshots = pn.generators_t.p_max_pu.index
    snapshot_data = snapshots[5616:5619] # 8/23/21
    return HOME_PATH, PYPSA_NETW0RK_PATH, pn, snapshot_data, snapshots


@app.cell
def _(snapshot_data):
    snapshot_data
    return


@app.cell
def _():
    return


@app.cell
def _(pypsa_net):
    pypsa_net
    return


@app.cell
def _(load_pypsa_network, np, pn, snapshot_data, zap):
    pypsa_kwargs = {}
    pypsa_net, pypsa_devices = load_pypsa_network(pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0, **pypsa_kwargs)
    dc_profile = np.array([
        [0.7, 0.6, 0.8],
        [0.5, 0.9, 0.4],
        [0.4, 0.4, 0.6],
    ])
    dc_cap0 = np.array([20.0, 20.0, 20.0])
    T=3
    dc_loads = zap.DataCenterLoad(
        num_nodes=101,
        terminal=np.array([0, 1, 2]),
        profiles=dc_profile,
        profile_types=[zap.DataCenterLoad.ProfileType.CUSTOM, zap.DataCenterLoad.ProfileType.CUSTOM, zap.DataCenterLoad.ProfileType.CUSTOM],
        nominal_capacity=dc_cap0,
        linear_cost=np.array([10000, 10000, 10000]),
        settime_horizon=T,
        locations=[zap.devices.injector.Location.URBAN,zap.devices.injector.Location.SUBURBAN,zap.devices.injector.Location.RURAL]
    )
    pypsa_devices.append(dc_loads)
    return (
        T,
        dc_cap0,
        dc_loads,
        dc_profile,
        pypsa_devices,
        pypsa_kwargs,
        pypsa_net,
    )


@app.cell
def _(dc_loads, plt, sns):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    for profile in dc_loads.profile:
        plt.plot(profile)

    ax.set_title("Data Center Load Profiles", fontsize=14, pad=15)
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Load (MW)", fontsize=12)
    ax.legend(title="Data Center ID", title_fontsize=12, fontsize=10)
    plt.tight_layout()
    plt.show()
    return ax, fig, profile


@app.cell
def _(pypsa_devices):
    pypsa_devices
    return


@app.cell
def _(dc_loads):
    print(dc_loads.num_nodes)
    return


@app.cell
def _():
    # baseload = zap.Load(
    #     num_nodes=net.num_nodes,
    #     terminal=np.array([2]),
    #     load=np.array([[10.0, 15.0, 20.0]]),
    #     linear_cost=np.array([500.0]),
    # )

    # gens = zap.Generator(
    #     num_nodes=net.num_nodes,
    #     terminal=np.array([1, 2]),
    #     nominal_capacity=np.array([50.0, 25.0]),
    #     dynamic_capacity=np.array([[0.3, 0.6, 0.3], [1.0, 1.0, 1.0]]),
    #     linear_cost=np.array([30.0, 30.0]),
    #     emission_rates=np.array([0.0, 500.0]),
    # )

    # lines = zap.ACLine(
    #     num_nodes=net.num_nodes,
    #     source_terminal=np.array([0, 1, 2]),
    #     sink_terminal=np.array([1, 2, 0]),
    #     nominal_capacity=np.array([10.0, 20.0, 30.0]),
    #     susceptance=np.array([1 / 10.0, 1 / 20.0, 1 / 30.0]),
    #     capacity=np.ones(3),
    # )

    # ground = zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))
    # devices = [dcloads, baseload, gens, lines, ground]

    # outcome = net.dispatch(
    #     devices, time_horizon=T, solver=cp.CLARABEL, add_ground=False
    # )
    return


@app.cell
def _(T, cp, pypsa_devices, pypsa_net):
    outcome = pypsa_net.dispatch(
        pypsa_devices, time_horizon=T, solver=cp.CLARABEL, add_ground=False
    )
    return (outcome,)


@app.cell
def _(outcome):
    outcome.power
    return


@app.cell
def _():
    # print("Locational prices ($/MWh):\n", outcome.prices, "\n")

    # dc_power = outcome.power[0][0]

    # print("Datacenter withdrawals (MW):\n", dc_power, "\n")

    # dc_profile = dcloads.profile[0]
    # dc_cap0 = dcloads.nominal_capacity[0]
    # expected = -dc_profile * dc_cap0.reshape(-1, 1)
    # print("Should equal:\n", expected)
    return


@app.cell
def _(T, cp, dc_cap0, np, pypsa_devices, pypsa_net, zap):
    ## Try to write a simple exmaple of a planning problem
    xstar = zap.DispatchLayer(
        pypsa_net,
        pypsa_devices,
        parameter_names={"dc_capacity": (5, "nominal_capacity")},
        time_horizon=T,
        solver=cp.CLARABEL,
    )  # Constuct a DispatchLayer

    eta = {"dc_capacity": np.array(dc_cap0)}  # Parameter

    op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices)
    inv_obj = zap.planning.InvestmentObjective(pypsa_devices, xstar)

    P = zap.planning.PlanningProblem(
        operation_objective=op_obj, investment_objective=inv_obj, layer=xstar
    )

    # Add in simplex constraint
    P.extra_projections = {
        "dc_capacity": zap.planning.SimplexBudgetProjection(budget=60, strict=True)
    }

    cost = P(**eta, requires_grad=True)
    grad = P.backward()

    state = P.solve(num_iterations=50)
    return P, cost, eta, grad, inv_obj, op_obj, state, xstar


@app.cell
def _(state):
    state
    return


@app.cell
def _(P):
    print(P.get_op_cost())
    print(P.get_inv_cost())
    return


if __name__ == "__main__":
    app.run()
