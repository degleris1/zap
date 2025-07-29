import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cvxpy as cp
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    import zap
    from zap.devices.injector import ARCHETYPES
    return ARCHETYPES, cp, mo, np, plt, sns, zap


@app.cell
def _(zap):
    # Create a power network
    net = zap.PowerNetwork(num_nodes=6)
    T_hours = 24  # 24 hours of simulation
    time_resolution = 0.25  # 15-minute intervals
    T = int(T_hours / time_resolution)  # 96 time steps

    print(
        f"Network: {net.num_nodes} nodes, {T} time steps ({T_hours} hours at {time_resolution*60:.0f}-min resolution)"
    )
    return T, T_hours, net, time_resolution


@app.cell
def _(T_hours, np, time_resolution, zap):
    dcloads = zap.DataCenterLoad(
        num_nodes=6,
        terminal=np.array([0, 1, 2, 3, 4, 5]),
        nominal_capacity=np.array([150.0, 100.0, 200.0, 75.0, 125.0, 180.0]),  # MW
        profile_types=[
            zap.DataCenterLoad.ProfileType.INTERACTIVE,  # Web services
            zap.DataCenterLoad.ProfileType.AI_TRAIN,  # ML training
            zap.DataCenterLoad.ProfileType.AI_INFER,  # AI inference
            zap.DataCenterLoad.ProfileType.BATCH,  # Background processing
            zap.DataCenterLoad.ProfileType.HPC,  # High-performance computing
            zap.DataCenterLoad.ProfileType.COLOCATION,  # Multi-tenant
        ],
        linear_cost=np.array([100.0, 120.0, 80.0, 90.0, 110.0, 95.0]),  # $/MWh
        settime_horizon=T_hours,
        time_resolution_hours=time_resolution,
        # Advanced rack configuration
        rack_mix={"gpu": 0.25, "cpu": 0.60, "storage": 0.15},  # Modern GPU-heavy mix
        rack_power_kw={"gpu": 200, "cpu": 40, "storage": 25},  # High-performance racks
        pue=1.25,  # Efficient modern facility
        # Planning parameters
        capital_cost=np.array([800.0, 900.0, 700.0, 750.0, 850.0, 775.0]) * 1e6,  # $/MW
    )

    print(f"Created DataCenterLoads with profiles shape: {dcloads.profile.shape}")
    print(f"Archetype types: {[ptype.value for ptype in dcloads.profile_types]}")
    return (dcloads,)


@app.cell
def _(dcloads, np, plt, sns, time_resolution):
    sns.set_theme(style="whitegrid")
    fig_profiles, axes_profiles = plt.subplots(2, 3, figsize=(18, 12))
    axes_profiles = axes_profiles.flatten()

    archetype_names = [
        "Interactive",
        "AI-Train",
        "AI-Infer",
        "Batch",
        "HPC",
        "Colocation",
    ]
    colors_profiles = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (ax, name, color) in enumerate(
        zip(axes_profiles, archetype_names, colors_profiles)
    ):
        hours_profile = np.arange(dcloads.profile.shape[1]) * time_resolution
        profile_mw = dcloads.profile[i] * dcloads.nominal_capacity[i]  # Convert to MW

        ax.plot(hours_profile, profile_mw, linewidth=2.5, color=color, alpha=0.8)
        ax.fill_between(hours_profile, 0, profile_mw, alpha=0.3, color=color)

        # Statistics
        mean_load = profile_mw.mean()
        peak_load = profile_mw.max()
        min_load = profile_mw.min()

        ax.set_title(
            f"{name} Data Center\nCapacity: {dcloads.nominal_capacity[i]:.0f} MW",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Hours", fontsize=12)
        ax.set_ylabel("Power (MW)", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = f"Mean: {mean_load:.1f} MW\nPeak: {peak_load:.1f} MW\nMin: {min_load:.1f} MW"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment="top",
            fontsize=10,
        )

    plt.tight_layout()
    plt.suptitle(
        "Synthetic Data Center Load Profiles by Archetype",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.show()

    # Summary statistics
    total_capacity = dcloads.nominal_capacity.sum()
    total_consumption = (dcloads.profile * dcloads.nominal_capacity.reshape(-1, 1)).sum(
        axis=0
    )
    avg_utilization = total_consumption.mean() / total_capacity

    print(f"\n=== Fleet Summary ===")
    print(f"Total Capacity: {total_capacity:.0f} MW")
    print(f"Average Consumption: {total_consumption.mean():.1f} MW")
    print(f"Fleet Utilization: {avg_utilization:.1%}")
    print(
        f"Peak Consumption: {total_consumption.max():.1f} MW ({total_consumption.max()/total_capacity:.1%})"
    )
    print(
        f"Min Consumption: {total_consumption.min():.1f} MW ({total_consumption.min()/total_capacity:.1%})"
    )
    return (
        archetype_names,
        avg_utilization,
        ax,
        axes_profiles,
        color,
        colors_profiles,
        fig_profiles,
        hours_profile,
        i,
        mean_load,
        min_load,
        name,
        peak_load,
        profile_mw,
        stats_text,
        total_capacity,
        total_consumption,
    )


@app.cell
def _(T, dcloads, net, np, zap):
    generators = zap.Generator(
        num_nodes=net.num_nodes,
        terminal=np.array([0, 1, 2, 3, 4, 5]),
        nominal_capacity=np.array([200.0, 180.0, 250.0, 150.0, 180.0, 220.0]),  # MW
        dynamic_capacity=np.ones((6, T)),  # Full availability
        linear_cost=np.array([45.0, 50.0, 40.0, 55.0, 48.0, 42.0]),  # $/MWh
        emission_rates=np.array([0.5, 0.8, 0.3, 0.9, 0.6, 0.4]) * 1000,  # kg CO2/MWh
        capital_cost=np.array([500.0, 600.0, 400.0, 700.0, 550.0, 480.0]) * 1e6,  # $/MW
    )

    # Transmission lines connecting the nodes
    transmission = zap.ACLine(
        num_nodes=net.num_nodes,
        source_terminal=np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]),
        sink_terminal=np.array([1, 2, 3, 4, 5, 2, 3, 4, 5, 0]),
        nominal_capacity=np.array(
            [300.0, 280.0, 320.0, 290.0, 310.0, 250.0, 270.0, 260.0, 240.0, 280.0]
        ),
        susceptance=np.array(
            [0.1, 0.12, 0.08, 0.11, 0.09, 0.13, 0.10, 0.14, 0.15, 0.09]
        ),
        capacity=np.ones(10),
    )

    # Ground node for voltage reference
    ground = zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))

    # All devices
    devices = [dcloads, generators, transmission, ground]

    print(f"Created infrastructure:")
    print(
        f"- Generators: {len(generators.terminal)} units, {generators.nominal_capacity.sum():.0f} MW total"
    )
    print(f"- Transmission: {len(transmission.nominal_capacity)} lines")
    print(
        f"- Data Centers: {len(dcloads.terminal)} sites, {dcloads.nominal_capacity.sum():.0f} MW total"
    )
    return devices, generators, ground, transmission


@app.cell
def _(T, cp, devices, net):
    print("Running economic dispatch...")

    outcome = net.dispatch(
        devices, time_horizon=T, solver=cp.CLARABEL, add_ground=False
    )

    print(f"Dispatch successful!")
    print(f"Total system cost: ${outcome.problem.value:,.0f}")

    # Extract results
    dc_power = outcome.power[0][
        0
    ]  # Data center power consumption (negative = consumption)
    gen_power = outcome.power[1][0]  # Generator power output (positive = generation)
    line_flows = outcome.power[2]  # Transmission line flows
    prices = outcome.prices  # Locational marginal prices

    print(f"DC consumption: {-dc_power.sum():.1f} MW average")
    print(f"Generation: {gen_power.sum():.1f} MW average")
    return dc_power, gen_power, line_flows, outcome, prices


@app.cell
def _(T, cp, dcloads, devices, net, np, zap):
    print("Setting up capacity planning problem...")

    # Create dispatch layer for planning
    xstar = zap.DispatchLayer(
        net,
        devices,
        parameter_names={
            "dc_capacity": (0, "nominal_capacity")
        },  # Plan data center capacities
        time_horizon=T,
        solver=cp.CLARABEL,
    )

    # Initial capacity allocation
    current_capacity = dcloads.nominal_capacity.copy()
    eta = {"dc_capacity": current_capacity}

    # Define objectives
    op_obj = zap.planning.DispatchCostObjective(net, devices)
    inv_obj = zap.planning.InvestmentObjective(devices, xstar)

    # Create planning problem with budget constraint
    TOTAL_BUDGET = 1000.0  # MW total capacity budget

    P = zap.planning.PlanningProblem(
        operation_objective=op_obj,
        investment_objective=inv_obj,
        layer=xstar,
        lower_bounds={"dc_capacity": np.zeros(len(current_capacity))},
        upper_bounds={"dc_capacity": np.full(len(current_capacity), 300.0)},
    )

    # Add budget constraint
    P.extra_projections = {
        "dc_capacity": zap.planning.SimplexBudgetProjection(
            budget=TOTAL_BUDGET, strict=True
        )
    }

    print(f"Planning problem created with {TOTAL_BUDGET} MW budget")
    print(f"Current allocation: {current_capacity.sum():.0f} MW total")
    return P, TOTAL_BUDGET, current_capacity, eta, inv_obj, op_obj, xstar


@app.cell
def _(P, eta):
    print("Solving capacity planning problem...")

    # Evaluate current solution
    cost = P(**eta, requires_grad=True)
    grad = P.backward()

    print(f"Initial cost: ${float(cost):,.0f}")
    print(f"Initial gradients: {grad['dc_capacity']}")

    # Optimize capacity allocation
    planning_result = P.solve(num_iterations=100)

    optimal_capacity = planning_result[0]["dc_capacity"]

    print(f"Optimization completed!")
    print(f"Optimal allocation: {optimal_capacity}")
    print(f"Budget utilization: {optimal_capacity.sum():.1f} MW")
    return cost, grad, optimal_capacity, planning_result


if __name__ == "__main__":
    app.run()
