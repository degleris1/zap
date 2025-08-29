import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path

    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pypsa
    import seaborn as sns

    import zap
    from zap.devices import DataCenterLoad
    from zap.importers.pypsa import load_pypsa_network, parse_buses

    return (
        DataCenterLoad,
        Path,
        cp,
        load_pypsa_network,
        mo,
        np,
        os,
        parse_buses,
        pd,
        plt,
        pypsa,
        sns,
        zap,
    )


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
def _(snapshots):
    snapshots_test = snapshots[0:10]
    return (snapshots_test,)


@app.cell
def _(parse_buses, pn):
    # ---- Get core series ----------------------------------------------------
    buses = pn.buses
    load_ts = pn.loads_t.p_set
    avg_load = load_ts.groupby(axis=1, level=0).mean().sum()  # MW per bus
    gen_cap = pn.generators.groupby("bus")["p_nom"].sum()  # MW per bus

    # ---- Normalise ----------------------------------------------------------
    def normalise(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    score = (
        0.6 * normalise(avg_load)
        + 0.4 * normalise(gen_cap).reindex_like(avg_load).fillna(0.0)
        # 0.2 * normalise(lmp).reindex_like(avg_load).fillna(0.0)
    )

    # ---- Select top-10 ------------------------------------------------------
    top10 = score.nlargest(10).index.tolist()
    top10 = [name.replace(" AC", "") for name in top10]

    print("Chosen data center buses:", top10)

    buses, buses_to_index = parse_buses(pn)

    chosen_bus_names = top10
    chosen_node_indices = [buses_to_index[name] for name in chosen_bus_names]

    print(chosen_node_indices)
    return (
        avg_load,
        buses,
        buses_to_index,
        chosen_bus_names,
        chosen_node_indices,
        gen_cap,
        load_ts,
        normalise,
        score,
        top10,
    )


@app.cell
def _(pn):
    pn
    return


@app.cell
def _(pn):
    pn.generators["p_nom"].sum()
    return


@app.cell
def _(load_pypsa_network, pn, snapshot_data):
    pypsa_kwargs = {}
    pypsa_net, pypsa_devices = load_pypsa_network(
        pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0, **pypsa_kwargs
    )
    return pypsa_devices, pypsa_kwargs, pypsa_net


@app.cell
def _(load_pypsa_network, pn, pypsa_kwargs, snapshots_test):
    pypsa_net_test, pypsa_devices_test = load_pypsa_network(
        pn, snapshots_test, power_unit=1.0e3, cost_unit=100.0, **pypsa_kwargs
    )
    return pypsa_devices_test, pypsa_net_test


@app.cell
def _(pypsa_devices_test):
    pypsa_devices_test[3]
    return


@app.cell
def _(cp, pypsa_devices_test, pypsa_net_test):
    outcome_test = pypsa_net_test.dispatch(
        pypsa_devices_test, time_horizon=10, solver=cp.PDLP, add_ground=False
    )
    return (outcome_test,)


@app.cell
def _(outcome_test):
    outcome_test.power[3][0].shape
    return


@app.cell
def _(pypsa_devices):
    pypsa_devices
    return


@app.cell
def _(pypsa_devices):
    ## Look at overall generation and load in this system

    print(
        f"There are {pypsa_devices[0].nominal_capacity.sum()} MW of generation in the system."
    )
    return


@app.cell
def _(plt, pypsa_devices):
    plt.plot(pypsa_devices[1].load.sum(axis=0))
    return


@app.cell
def _(base_outcome, plt):
    plt.plot(base_outcome.power[1][0].sum(axis=0))
    return


@app.cell
def _(cp, pypsa_devices, pypsa_net):
    T = 24
    base_outcome = pypsa_net.dispatch(
        pypsa_devices, time_horizon=T, solver=cp.CLARABEL, add_ground=False
    )
    return T, base_outcome


@app.cell
def _(base_outcome, plt):
    ## Plot prices in base outcome
    print(f"Dispatch Cost is {base_outcome.problem.value}")
    plt.plot(base_outcome.prices.T)
    return


@app.cell
def _(
    DataCenterLoad,
    T,
    chosen_node_indices,
    np,
    pypsa_devices,
    pypsa_net,
    zap,
):
    pypsa_devices_dc = pypsa_devices.copy()
    # terminals = np.arange(pypsa_net.num_nodes)
    # terminals = np.arange(10)
    terminals = np.array(chosen_node_indices)
    n_dc = len(terminals)

    dcloads = DataCenterLoad(
        num_nodes=pypsa_net.num_nodes,
        terminal=terminals,
        profile_types=[zap.DataCenterLoad.ProfileType.DIURNAL] * n_dc,
        nominal_capacity=np.ones((n_dc)),
        linear_cost=np.ones(n_dc) * 1000,
        settime_horizon=T,
        capital_cost=np.array([9.5, 9.5, 9.5, 11.7, 11.7, 11.7, 14.0, 14.0, 14.0, 14.0])
        * 1e6,
    )
    pypsa_devices_dc.append(dcloads)
    return dcloads, n_dc, pypsa_devices_dc, terminals


@app.cell
def _(np):
    np.array([9.5, 9.5, 9.5, 11.7, 11.7, 11.7, 14.0, 14.0, 14.0, 14.0]) * 1e6
    return


@app.cell
def _(T, cp, n_dc, np, pypsa_devices_dc, pypsa_net, zap):
    ## Try to write a simple exmaple of a planning problem
    TOTAL_DC_BUDGET = 1000
    # MW
    xstar = zap.DispatchLayer(
        pypsa_net,
        pypsa_devices_dc,
        parameter_names={"dc_capacity": (5, "nominal_capacity")},
        time_horizon=T,
        solver=cp.CLARABEL,
    )  # Constuct a DispatchLayer

    # lower_bounds = {}
    # upper_bounds = {}
    lower_bounds = {"dc_capacity": np.full(n_dc, 0)}
    upper_bounds = {"dc_capacity": np.full(n_dc, 1000)}

    # eta = {"dc_capacity": np.full(n_dc, TOTAL_DC_BUDGET / n_dc)}
    init_eta = np.zeros(n_dc)
    init_eta[0] = 1000
    # init_eta = np.random.rand(n_dc) * 10
    eta = {"dc_capacity": init_eta}

    op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices_dc)
    inv_obj = zap.planning.InvestmentObjective(pypsa_devices_dc, xstar)

    P = zap.planning.PlanningProblem(
        operation_objective=op_obj,
        investment_objective=inv_obj,
        layer=xstar,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    # Add in simplex constraint
    # P.extra_projections = {}
    P.extra_projections = {
        "dc_capacity": zap.planning.SimplexBudgetProjection(
            budget=TOTAL_DC_BUDGET, strict=True
        )
    }

    cost = P(**eta, requires_grad=True)
    grad = P.backward()

    state = P.solve(num_iterations=100)
    return (
        P,
        TOTAL_DC_BUDGET,
        cost,
        eta,
        grad,
        init_eta,
        inv_obj,
        lower_bounds,
        op_obj,
        state,
        upper_bounds,
        xstar,
    )


@app.cell
def _(grad):
    grad["dc_capacity"]
    return


@app.cell
def _(state):
    state[0]["dc_capacity"].sum()
    return


@app.cell
def _(plt, state):
    plt.plot(state[0]["dc_capacity"])
    return


@app.cell
def _(state):
    state[0]["dc_capacity"]
    return


@app.cell
def _(P):
    P.get_op_cost()
    return


@app.cell
def _(P):
    P.get_inv_cost()
    return


@app.cell
def _(T, cp, pypsa_devices_dc, pypsa_net):
    post_outcome = pypsa_net.dispatch(
        pypsa_devices_dc, time_horizon=T, solver=cp.CLARABEL, add_ground=False
    )
    return (post_outcome,)


@app.cell
def _(post_outcome):
    post_outcome.problem.value
    return


@app.cell
def _(plt, post_outcome):
    plt.plot(post_outcome.power[5][0].T)
    return


@app.cell
def _(pypsa_devices_dc):
    pypsa_devices_dc
    return


@app.cell
def _(np):
    solver_opt_capacities = np.array(
        [
            275.41860954,
            275.39441647,
            275.39291164,
            24.82772319,
            24.82772319,
            24.82772319,
            24.82772319,
            24.82772319,
            24.82772319,
            24.82772319,
        ]
    )

    grid_opt_capacities = np.array(
        [
            109.79731552,
            73.48552114,
            87.88355807,
            109.73627581,
            107.75840246,
            86.61183818,
            86.68299651,
            104.44680737,
            129.91202981,
            103.68525514,
        ]
    )
    return grid_opt_capacities, solver_opt_capacities


@app.cell
def _(
    DataCenterLoad,
    T,
    cp,
    grid_opt_capacities,
    n_dc,
    np,
    pypsa_devices,
    pypsa_net,
    solver_opt_capacities,
    terminals,
    zap,
):
    ### Three scenarios to check

    capital_costs = (
        np.array([9.5, 9.5, 9.5, 11.7, 11.7, 11.7, 14.0, 14.0, 14.0, 14.0]) * 1e6
    )

    # (1) Uniform Distribution. Dispatch cost? Investment cost?

    dcloads_uniform = DataCenterLoad(
        num_nodes=pypsa_net.num_nodes,
        terminal=terminals,
        profile_types=[zap.DataCenterLoad.ProfileType.DIURNAL] * n_dc,
        nominal_capacity=np.ones((n_dc)) * 100,
        linear_cost=np.ones(n_dc) * 1000,
        settime_horizon=T,
        capital_cost=capital_costs,
    )
    pypsa_devices_uniform = pypsa_devices.copy()
    pypsa_devices_uniform.append(dcloads_uniform)
    uniform_outcome = pypsa_net.dispatch(
        pypsa_devices_uniform, time_horizon=T, solver=cp.CLARABEL, add_ground=False
    )
    uniform_dispatch_cost = uniform_outcome.problem.value
    uniform_investment_cost = np.dot(capital_costs, np.ones((n_dc)) * 100)
    print(f"Uniform Dispatch Cost: {uniform_dispatch_cost}")
    print(f"Uniform Investment Cost: {uniform_investment_cost}")

    # (2) Investment Optimized Distribution. Dispatch cost? Investment cost?
    dcloads_investment_opt = DataCenterLoad(
        num_nodes=pypsa_net.num_nodes,
        terminal=terminals,
        profile_types=[zap.DataCenterLoad.ProfileType.DIURNAL] * n_dc,
        nominal_capacity=np.array(
            [
                1000.0 / 3.0,
                1000.0 / 3.0,
                1000.0 / 3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
        linear_cost=np.ones(n_dc) * 1000,
        settime_horizon=T,
        capital_cost=capital_costs,
    )
    pypsa_devices_investment_opt = pypsa_devices.copy()
    pypsa_devices_investment_opt.append(dcloads_investment_opt)
    investment_opt_outcome = pypsa_net.dispatch(
        pypsa_devices_investment_opt,
        time_horizon=T,
        solver=cp.CLARABEL,
        add_ground=False,
    )
    investment_opt_dispatch_cost = investment_opt_outcome.problem.value
    investment_opt_investment_cost = np.dot(
        capital_costs,
        np.array(
            [
                1000.0 / 3.0,
                1000.0 / 3.0,
                1000.0 / 3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )
    print(f"Investment Optimized Dispatch Cost: {investment_opt_dispatch_cost}")
    print(f"Investment Optimized Investment Cost: {investment_opt_investment_cost}")

    # (3) Optimized Distribution. Dispatch cost? Investment cost?
    dcloads_solver_opt = DataCenterLoad(
        num_nodes=pypsa_net.num_nodes,
        terminal=terminals,
        profile_types=[zap.DataCenterLoad.ProfileType.DIURNAL] * n_dc,
        nominal_capacity=solver_opt_capacities,
        linear_cost=np.ones(n_dc) * 1000,
        settime_horizon=T,
        capital_cost=capital_costs,
    )
    pypsa_devices_solver_opt = pypsa_devices.copy()
    pypsa_devices_solver_opt.append(dcloads_solver_opt)
    solver_opt_outcome = pypsa_net.dispatch(
        pypsa_devices_solver_opt, time_horizon=T, solver=cp.CLARABEL, add_ground=False
    )
    solver_opt_dispatch_cost = solver_opt_outcome.problem.value
    solver_opt_investment_cost = np.dot(capital_costs, solver_opt_capacities)
    print(f"Solver Optimized Dispatch Cost: {solver_opt_dispatch_cost}")
    print(f"Solver Optimized Investment Cost: {solver_opt_investment_cost}")

    # (4) Optimized Distribution w/o Investment Costs. Dispatch cost? Investment cost?
    dcloads_grid_opt = DataCenterLoad(
        num_nodes=pypsa_net.num_nodes,
        terminal=terminals,
        profile_types=[zap.DataCenterLoad.ProfileType.DIURNAL] * n_dc,
        nominal_capacity=grid_opt_capacities,
        linear_cost=np.ones(n_dc) * 1000,
        settime_horizon=T,
        capital_cost=capital_costs,
    )
    pypsa_devices_grid_opt = pypsa_devices.copy()
    pypsa_devices_grid_opt.append(dcloads_grid_opt)
    grid_opt_outcome = pypsa_net.dispatch(
        pypsa_devices_grid_opt, time_horizon=T, solver=cp.CLARABEL, add_ground=False
    )
    grid_opt_dispatch_cost = grid_opt_outcome.problem.value
    grid_opt_investment_cost = np.dot(capital_costs, grid_opt_capacities)
    print(f"Grid Optimized Dispatch Cost: {grid_opt_dispatch_cost}")
    print(f"Grid Optimized Investment Cost: {grid_opt_investment_cost}")

    dispatch_costs = [
        uniform_dispatch_cost,
        investment_opt_dispatch_cost,
        solver_opt_dispatch_cost,
        grid_opt_dispatch_cost,
    ]
    investment_costs = [
        uniform_investment_cost,
        investment_opt_investment_cost,
        solver_opt_investment_cost,
        grid_opt_investment_cost,
    ]
    return (
        capital_costs,
        dcloads_grid_opt,
        dcloads_investment_opt,
        dcloads_solver_opt,
        dcloads_uniform,
        dispatch_costs,
        grid_opt_dispatch_cost,
        grid_opt_investment_cost,
        grid_opt_outcome,
        investment_costs,
        investment_opt_dispatch_cost,
        investment_opt_investment_cost,
        investment_opt_outcome,
        pypsa_devices_grid_opt,
        pypsa_devices_investment_opt,
        pypsa_devices_solver_opt,
        pypsa_devices_uniform,
        solver_opt_dispatch_cost,
        solver_opt_investment_cost,
        solver_opt_outcome,
        uniform_dispatch_cost,
        uniform_investment_cost,
        uniform_outcome,
    )


@app.cell
def _(dispatch_costs):
    dispatch_costs
    return


@app.cell
def _(investment_costs):
    investment_costs
    return


@app.cell
def _(dispatch_costs, investment_costs, plt):
    plt.scatter(investment_costs, dispatch_costs)
    return


@app.cell
def _():
    from matplotlib.ticker import ScalarFormatter

    return (ScalarFormatter,)


@app.cell
def _(np, plt):
    def plot_pareto_tradeoff(dispatch_costs, investment_costs, labels):
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(dispatch_costs, investment_costs, s=300)

        # Add labels positioned just left of each point
        for x, y, label in zip(dispatch_costs, investment_costs, labels):
            if label == "Capital Based Distribution":
                continue
            ax.text(x + 0.005 * x, y, label, fontsize=12, ha="left")
        x = dispatch_costs[1]
        y = investment_costs[1]
        label = labels[1]
        ax.text(x - 0.005 * x, y, label, fontsize=12, ha="right")

        # Set axis labels
        ax.set_xlabel("Dispatch Cost (Millions $)", fontsize=14)
        ax.set_ylabel("Investment Cost (Billions $)", fontsize=14)
        ax.set_title(
            "Dispatch vs. Investment Costs under Different Capacity Allocations",
            fontsize=16,
            pad=15,
        )

        # Format ticks as floats with one decimal place + M/B suffixes
        xticks = np.linspace(min(dispatch_costs), max(dispatch_costs), 4)
        yticks = np.linspace(min(investment_costs), max(investment_costs), 4)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xticklabels([f"{x/1e6:.1f}M" for x in xticks])
        ax.set_yticklabels([f"{y/1e9:.1f}B" for y in yticks])

        # Grid styling
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
        ax.tick_params(axis="both", which="both", labelsize=12)

        plt.tight_layout()
        plt.savefig("pareto_optimal_allocations.pdf")
        return fig

    return (plot_pareto_tradeoff,)


@app.cell
def _(os):
    os.getcwd()
    return


@app.cell
def _(dispatch_costs, investment_costs, plot_pareto_tradeoff):
    labels = [
        "Uniform Distribution",
        "Capital Based Distribution",
        "Solver Optimized Distribution",
        "Dispatch Optimized Distribution",
    ]
    fig = plot_pareto_tradeoff(dispatch_costs, investment_costs, labels)
    # plt.show()
    return fig, labels


@app.cell
def _():
    # labels = ['Uniform Distribution', 'Capital Based Distribution', 'Solver Optimized Distribution', 'Dispatch Optimized Distribution']

    # plt.figure(figsize=(8,6))
    # plt.scatter(dispatch_costs, investment_costs)

    # # Add labels at each point
    # for x, y, label in zip(dispatch_costs, investment_costs, labels):
    #     plt.text(x, y, label, fontsize=9, ha='left', va='bottom')

    # plt.xlabel('Dispatch Cost')
    # plt.ylabel('Investment Cost')
    # plt.title('Dispatch vs Investment Costs')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
