import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pypsa
    import seaborn as sns

    import zap
    from zap.importers.pypsa import load_pypsa_network
    return cp, load_pypsa_network, mo, np, os, plt, pypsa, sns, zap


@app.cell
def _(os, pypsa):
    HOME_PATH = os.environ.get("HOME")
    PYPSA_NETW0RK_PATH = (
        HOME_PATH + "/zap_data/pypsa-networks/western_small/network_2021.nc"
    )
    pn = pypsa.Network(PYPSA_NETW0RK_PATH)
    snapshots = pn.generators_t.p_max_pu.index
    snapshot_data = snapshots[5616:5640]  # 8/23/21
    return HOME_PATH, PYPSA_NETW0RK_PATH, pn, snapshot_data, snapshots


@app.cell
def _(snapshot_data):
    snapshot_data
    return


@app.cell
def _(load_pypsa_network, pn, snapshot_data):
    pypsa_kwargs = {}
    pypsa_net, pypsa_devices = load_pypsa_network(
        pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0, **pypsa_kwargs
    )
    return pypsa_devices, pypsa_kwargs, pypsa_net


@app.cell
def _(pypsa_devices):
    pypsa_devices
    return


@app.cell
def _(pypsa_net):
    pypsa_net
    return


@app.cell
def _(np, zap):
    gpu0 = np.array([30, 15, 10])
    cpu0 = np.array([10, 10,  5])
    sto0 = np.array([ 5,  5,  2])

    T = 3
    dc_cap0 = np.array([2.0, 1.50, 1.0])
    net = zap.PowerNetwork(num_nodes=3)
    dcload = zap.DataCenterLoad(
        num_nodes=net.num_nodes,
        time_resolution_hours=1.0,
        terminal=np.array([0, 1, 2]),
        profile_types=[
            zap.DataCenterLoad.ProfileType.DIURNAL,
            zap.DataCenterLoad.ProfileType.DIURNAL,
            zap.DataCenterLoad.ProfileType.DIURNAL,
        ],
        peak_hours=np.array([10, 13, 15]),
        gpu_racks=gpu0,
        cpu_racks=cpu0,
        storage_racks=sto0,
        linear_cost=np.array([600.0, 100, 100]),
        settime_horizon=T
    )
    return T, cpu0, dc_cap0, dcload, gpu0, net, sto0


@app.cell
def _(dcload, plt):
    plt.plot(dcload.profile[0])
    plt.plot(dcload.profile[1])
    plt.plot(dcload.profile[2])
    plt.show()
    return


@app.cell
def _(dcload, plt, sns):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    for profile in dcload.profile:
        plt.plot(profile)

    ax.set_title("Data Center Load Profiles", fontsize=14, pad=15)
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Load (MW)", fontsize=12)
    ax.legend(title="Data Center ID", title_fontsize=12, fontsize=10)
    plt.tight_layout()
    plt.show()
    return ax, fig, profile


@app.cell
def _(T, cp, dcload, net, np, zap):
    baseload = zap.Load(
        num_nodes=net.num_nodes,
        terminal=np.array([2]),
        load=np.array([[10.0, 15.0, 20.0]]),
        linear_cost=np.array([500.0]),
    )

    gens = zap.Generator(
        num_nodes=net.num_nodes,
        terminal=np.array([1, 2]),
        nominal_capacity=np.array([50.0, 25.0]),
        dynamic_capacity=np.array([[0.3, 0.6, 0.3], [1.0, 1.0, 1.0]]),
        linear_cost=np.array([30.0, 30.0]),
        emission_rates=np.array([0.0, 500.0]),
    )

    lines = zap.ACLine(
        num_nodes=net.num_nodes,
        source_terminal=np.array([0, 1, 2]),
        sink_terminal=np.array([1, 2, 0]),
        nominal_capacity=np.array([10.0, 20.0, 30.0]),
        susceptance=np.array([1 / 10.0, 1 / 20.0, 1 / 30.0]),
        capacity=np.ones(3),
    )

    ground = zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))
    devices = [dcload, baseload, gens, lines, ground]

    outcome = net.dispatch(
        devices, time_horizon=T, solver=cp.CLARABEL, add_ground=False
    )
    return baseload, devices, gens, ground, lines, outcome


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


app._unparsable_cell(
    r"""
    ## Try to write a simple exmaple of a planning problem
    xstar = zap.DispatchLayer(
        net,
        devices,
        parameter_names={
            \"gpu_racks\":     (0, \"gpu_racks\"),      
            \"cpu_racks\":     (0, \"cpu_racks\"),
            \"storage_racks\": (0, \"storage_racks\"),
        },,
        time_horizon=T,
        solver=cp.CLARABEL,
    )  # Constuct a DispatchLayer

    eta = {
        \"gpu_racks\":     gpu0.copy(),   # np.array, will be mutated by the optimiser
        \"cpu_racks\":     cpu0.copy(),
        \"storage_racks\": sto0.copy(),
    }

    op_obj = zap.planning.DispatchCostObjective(net, devices)
    inv_obj = zap.planning.InvestmentObjective(devices, xstar)

    P = zap.planning.PlanningProblem(
        operation_objective=op_obj, investment_objective=inv_obj, layer=xstar
    )

    # Add in simplex constraint
    P.extra_projections = {
        \"dc_capacity\": zap.planning.SimplexBudgetProjection(budget=4, strict=True)
    }

    cost = P(**eta, requires_grad=True)
    grad = P.backward()

    state = P.solve(num_iterations=50)
    """,
    name="_"
)


@app.cell
def _(state):
    state
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
