import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cvxpy as cp
    import numpy as np
    import zap
    return cp, mo, np, zap


@app.cell
def _(np, zap):
    net = zap.PowerNetwork(num_nodes=3)
    T = 3
    dc_profile = np.array([
        [0.7, 0.6, 0.8],
        [0.5, 0.9, 0.4],
        [0.4, 0.4, 0.6],
    ])
    dc_cap0 = np.array([2.0, 1.50, 1.0])

    dcload = zap.DataCenterLoad(
        num_nodes=net.num_nodes,
        terminal=np.array([0, 1, 2]),
        profile=dc_profile,
        nominal_capacity=dc_cap0,
        linear_cost=np.array([600.0]),
    )
    return T, dc_cap0, dc_profile, dcload, net


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
        dynamic_capacity=np.array([[0.3, 0.6, 0.3],
                                   [1.0, 1.0, 1.0]]),
        linear_cost=np.array([0.1, 30.0]),
        emission_rates=np.array([0.0, 500.0]),
    )

    lines = zap.ACLine(
        num_nodes=net.num_nodes,
        source_terminal=np.array([0, 1, 2]),
        sink_terminal=np.array([1, 2, 0]),
        nominal_capacity=np.array([10.0, 20.0, 30.0]),
        susceptance=np.array([1/10.0, 1/20.0, 1/30.0]),
        capacity=np.ones(3),
    )

    ground = zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))
    devices = [dcload, baseload, gens, lines, ground]

    outcome = net.dispatch(devices, time_horizon=T, solver=cp.CLARABEL, add_ground=False)
    return baseload, devices, gens, ground, lines, outcome


@app.cell
def _(dc_cap0, dc_profile, outcome):
    print("Locational prices ($/MWh):\n", outcome.prices, "\n")

    dc_power = outcome.power[0][0]
    print("Datacenter withdrawals (MW):\n", dc_power, "\n")

    expected = -dc_profile * dc_cap0.reshape(-1, 1)
    print("Should equal:\n", expected)
    return dc_power, expected


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
