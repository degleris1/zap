import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    import pandas as pd
    import pypsa
    import tempfile
    import os
    from copy import deepcopy

    import zap
    from zap import DispatchLayer
    from zap.importers.pypsa import load_pypsa_network, HOURS_PER_YEAR

    return (
        DispatchLayer,
        HOURS_PER_YEAR,
        cp,
        deepcopy,
        load_pypsa_network,
        mo,
        np,
        os,
        pd,
        pypsa,
        tempfile,
        zap,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Budget Constraints for Planning

    This notebook demonstrates the budget constraint functionality in the zap planning module.
    Budget constraints allow you to specify arbitrary linear constraints over investment decision
    variables, such as:

    - **Maximum total capacity**: `solar + wind <= 500 MW`
    - **Minimum renewable share**: `solar + wind >= 0.3 * total`
    - **Regional limits**: `gen_region_A <= 200 MW`

    The key idea is to replace the simple box-projection step in gradient descent with a
    QP-based projection that handles both box constraints and budget constraints.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Create a PyPSA Network
    """
    )
    return


@app.cell
def _(np, pd, pypsa):
    def create_network():
        """Create a simple 3-bus network with extendable generators and lines."""
        n = pypsa.Network()

        # Set snapshots (24 hours)
        snapshots = pd.date_range("2020-01-01", periods=24, freq="h")
        n.set_snapshots(snapshots)

        # Add buses
        n.add("Bus", "bus0", v_nom=380)
        n.add("Bus", "bus1", v_nom=380)
        n.add("Bus", "bus2", v_nom=380)

        # Add carriers
        n.add("Carrier", "gas", co2_emissions=0.2, color="#d35050")
        n.add("Carrier", "solar", co2_emissions=0.0, color="#f9d002")
        n.add("Carrier", "wind", co2_emissions=0.0, color="#235ebc")

        # Add extendable solar generator
        n.add(
            "Generator",
            "gen_solar",
            bus="bus0",
            p_nom=50.0,
            p_nom_extendable=True,
            p_nom_min=10.0,
            capital_cost=50000.0,
            marginal_cost=0.0,
            carrier="solar",
        )

        # Add extendable wind generator
        n.add(
            "Generator",
            "gen_wind",
            bus="bus0",
            p_nom=30.0,
            p_nom_extendable=True,
            p_nom_min=10.0,
            capital_cost=60000.0,
            marginal_cost=0.0,
            carrier="wind",
        )

        # Add extendable gas generator
        n.add(
            "Generator",
            "gen_gas",
            bus="bus1",
            p_nom=100.0,
            p_nom_extendable=True,
            p_nom_min=50.0,
            capital_cost=30000.0,
            marginal_cost=50.0,
            carrier="gas",
        )

        # Add load with daily profile
        load_profile = 80 + 40 * np.sin(np.linspace(0, 2 * np.pi, 24))
        n.add("Load", "load0", bus="bus2", p_set=load_profile)

        # Add extendable AC lines
        n.add(
            "Line",
            "line_0_1",
            bus0="bus0",
            bus1="bus1",
            s_nom=50.0,
            s_nom_extendable=True,
            s_nom_min=20.0,
            capital_cost=10000.0,
            x=0.1,
            r=0.01,
        )

        n.add(
            "Line",
            "line_1_2",
            bus0="bus1",
            bus1="bus2",
            s_nom=50.0,
            s_nom_extendable=True,
            s_nom_min=20.0,
            capital_cost=10000.0,
            x=0.1,
            r=0.01,
        )

        # Add capacity factors for renewables
        solar_cf = 0.5 * (1 + np.sin(np.linspace(-np.pi / 2, np.pi / 2, 24)))
        wind_cf = 0.3 + 0.4 * np.abs(np.sin(np.linspace(0, 4 * np.pi, 24)))

        n.generators_t["p_max_pu"] = pd.DataFrame(
            {"gen_solar": solar_cf, "gen_wind": wind_cf}, index=snapshots
        )

        return n, snapshots

    pypsa_net, snapshots = create_network()
    print(
        f"Created network with {len(pypsa_net.buses)} buses, {len(pypsa_net.generators)} generators, {len(pypsa_net.lines)} lines"
    )
    return pypsa_net, snapshots


@app.cell
def _(mo):
    mo.md(
        """
    ## Import Network into Zap
    """
    )
    return


@app.cell
def _(load_pypsa_network, pypsa_net, snapshots):
    net, devices = load_pypsa_network(pypsa_net, snapshots)
    time_horizon = len(snapshots)

    print(f"Imported {len(devices)} device types:")
    for _i, _dev in enumerate(devices):
        print(f"  [{_i}] {type(_dev).__name__}: {_dev.num_devices} devices - {list(_dev.name)}")
    return devices, net, time_horizon


@app.cell
def _(mo):
    mo.md(
        """
    ## Define Planning Parameters
    """
    )
    return


@app.cell
def _(devices):
    from zap.devices.injector import Generator
    from zap.devices.transporter import ACLine

    # Find device indices
    gen_idx = None
    line_idx = None

    for _idx, _device in enumerate(devices):
        if isinstance(_device, Generator):
            gen_idx = _idx
        elif isinstance(_device, ACLine):
            line_idx = _idx

    # Define which parameters to optimize
    parameter_names = {}
    if gen_idx is not None:
        parameter_names["generator"] = (gen_idx, "nominal_capacity")
    if line_idx is not None:
        parameter_names["line"] = (line_idx, "nominal_capacity")

    print("Parameter names:", parameter_names)
    return gen_idx, parameter_names


@app.cell
def _(mo):
    mo.md(
        """
    ## Create Dispatch Layer
    """
    )
    return


@app.cell
def _(DispatchLayer, cp, devices, net, parameter_names, time_horizon):
    layer = DispatchLayer(
        net,
        devices,
        parameter_names=parameter_names,
        time_horizon=time_horizon,
        solver=cp.MOSEK,
        solver_kwargs={"verbose": False, "accept_unknown": True},
    )
    return (layer,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Define Budget Constraints

    Budget constraints are specified in CSV format with the following columns:
    - `constraint_name`: Groups rows into a single constraint
    - `attribute`: Device attribute (e.g., "nominal_capacity") or "rhs" for the bound
    - `device_name`: Name of the device (must match device.name)
    - `multiplier`: Coefficient for this device
    - `rhs_value`: Right-hand side value (only for "rhs" rows)
    - `sense`: "le" for <= or "ge" for >= (only for "rhs" rows)
    """
    )
    return


@app.cell
def _(mo):
    # Interactive constraint configuration
    constraint_type = mo.ui.dropdown(
        options=["Maximum Total Renewable", "Minimum Renewable Share", "Custom"],
        value="Maximum Total Renewable",
        label="Constraint Type",
    )
    constraint_type
    return (constraint_type,)


@app.cell
def _(constraint_type, mo):
    # Conditional UI based on constraint type
    if constraint_type.value == "Maximum Total Renewable":
        rhs_slider = mo.ui.slider(50, 500, value=150, step=10, label="Max Renewable Capacity (MW)")
        sense_select = mo.ui.dropdown(options=["le", "ge"], value="le", label="Sense")
    elif constraint_type.value == "Minimum Renewable Share":
        rhs_slider = mo.ui.slider(50, 300, value=100, step=10, label="Min Renewable Capacity (MW)")
        sense_select = mo.ui.dropdown(options=["le", "ge"], value="ge", label="Sense")
    else:
        rhs_slider = mo.ui.slider(50, 500, value=200, step=10, label="RHS Value (MW)")
        sense_select = mo.ui.dropdown(options=["le", "ge"], value="le", label="Sense")

    mo.hstack([rhs_slider, sense_select])
    return rhs_slider, sense_select


@app.cell
def _(constraint_type, rhs_slider, sense_select):
    def generate_constraint_csv():
        """Generate constraint CSV based on UI selections."""
        if constraint_type.value == "Maximum Total Renewable":
            return f"""constraint_name,attribute,device_name,multiplier,rhs_value,sense
    max_renewable,nominal_capacity,gen_solar,1,,
    max_renewable,nominal_capacity,gen_wind,1,,
    max_renewable,rhs,,,{rhs_slider.value},{sense_select.value}"""

        elif constraint_type.value == "Minimum Renewable Share":
            return f"""constraint_name,attribute,device_name,multiplier,rhs_value,sense
    min_renewable,nominal_capacity,gen_solar,1,,
    min_renewable,nominal_capacity,gen_wind,1,,
    min_renewable,rhs,,,{rhs_slider.value},{sense_select.value}"""

        else:  # Custom
            return f"""constraint_name,attribute,device_name,multiplier,rhs_value,sense
    custom,nominal_capacity,gen_solar,1,,
    custom,nominal_capacity,gen_wind,1,,
    custom,nominal_capacity,gen_gas,1,,
    custom,rhs,,,{rhs_slider.value},{sense_select.value}"""

    constraint_csv = generate_constraint_csv()
    print("Generated Constraint CSV:")
    print("-" * 60)
    print(constraint_csv)
    return (constraint_csv,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Set Up Planning Problem
    """
    )
    return


@app.cell
def _(deepcopy, devices, layer, net, np, parameter_names, zap):
    # Define objectives
    op_objective = zap.planning.DispatchCostObjective(net, devices)
    inv_objective = zap.planning.InvestmentObjective(devices, layer)

    # Set up bounds
    lower_bounds = {}
    upper_bounds = {}
    initial_params = {}

    for _param_name, (_device_idx, _attr_name) in parameter_names.items():
        _device = devices[_device_idx]
        _current_cap = getattr(_device, _attr_name)

        # Use min/max from device if available
        _min_cap = getattr(_device, f"min_{_attr_name}", np.zeros_like(_current_cap))
        _max_cap = getattr(_device, f"max_{_attr_name}", _current_cap * 5.0)

        if _min_cap is None:
            _min_cap = np.zeros_like(_current_cap)
        if _max_cap is None:
            _max_cap = _current_cap * 5.0

        lower_bounds[_param_name] = _min_cap
        upper_bounds[_param_name] = _max_cap
        initial_params[_param_name] = deepcopy(_current_cap)

    print("Initial Parameters:")
    for _k, _v in initial_params.items():
        print(f"  {_k}: {_v}")

    print("\nLower Bounds:")
    for _k, _v in lower_bounds.items():
        print(f"  {_k}: {_v}")

    print("\nUpper Bounds:")
    for _k, _v in upper_bounds.items():
        print(f"  {_k}: {_v}")
    return (
        initial_params,
        inv_objective,
        lower_bounds,
        op_objective,
        upper_bounds,
    )


@app.cell
def _(
    HOURS_PER_YEAR,
    constraint_csv,
    inv_objective,
    layer,
    lower_bounds,
    op_objective,
    os,
    snapshots,
    tempfile,
    upper_bounds,
    zap,
):
    # Write constraint CSV to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(constraint_csv)
        f.flush()
        constraint_csv_path = f.name

    # Create planning problem with budget constraints
    snapshot_weight = HOURS_PER_YEAR / len(snapshots)

    problem_with_constraints = zap.planning.PlanningProblem(
        operation_objective=op_objective,
        investment_objective=inv_objective,
        layer=layer,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        budget_constraints=constraint_csv_path,
        snapshot_weight=snapshot_weight,
    )

    # Also create one without constraints for comparison
    problem_without_constraints = zap.planning.PlanningProblem(
        operation_objective=op_objective,
        investment_objective=inv_objective,
        layer=layer,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        snapshot_weight=snapshot_weight,
    )

    print("Created planning problems")
    print(f"  With constraints: {len(problem_with_constraints.budget_constraints)} constraint(s)")
    print(
        f"  Without constraints: budget_constraints = {problem_without_constraints.budget_constraints}"
    )

    # Clean up temp file
    os.unlink(constraint_csv_path)
    return problem_with_constraints, problem_without_constraints


@app.cell
def _(mo):
    mo.md(
        """
    ## Optimization Settings
    """
    )
    return


@app.cell
def _(mo):
    num_iterations = mo.ui.slider(5, 100, value=20, step=5, label="Number of Iterations")
    step_size = mo.ui.slider(0.0001, 0.01, value=0.001, step=0.0001, label="Step Size")
    clip_value = mo.ui.slider(100, 10000, value=1000, step=100, label="Gradient Clip")

    mo.hstack([num_iterations, step_size, clip_value])
    return clip_value, num_iterations, step_size


@app.cell
def _(mo):
    mo.md(
        """
    ## Run Optimization
    """
    )
    return


@app.cell
def _(
    clip_value,
    deepcopy,
    initial_params,
    num_iterations,
    problem_with_constraints,
    problem_without_constraints,
    step_size,
    zap,
):
    import zap.planning.trackers as tr

    algorithm = zap.planning.GradientDescent(
        step_size=step_size.value,
        clip=clip_value.value,
    )

    # Solve with constraints
    print("Solving WITH budget constraints...")
    state_constrained, history_constrained = problem_with_constraints.solve(
        num_iterations=num_iterations.value,
        algorithm=algorithm,
        initial_state=deepcopy(initial_params),
        trackers=[tr.LOSS, tr.GRAD_NORM, tr.PROJ_GRAD_NORM],
        verbosity=0,
    )

    # Solve without constraints
    print("Solving WITHOUT budget constraints...")
    state_unconstrained, history_unconstrained = problem_without_constraints.solve(
        num_iterations=num_iterations.value,
        algorithm=algorithm,
        initial_state=deepcopy(initial_params),
        trackers=[tr.LOSS, tr.GRAD_NORM, tr.PROJ_GRAD_NORM],
        verbosity=0,
    )

    print("Done!")
    return (
        history_constrained,
        history_unconstrained,
        state_constrained,
        state_unconstrained,
    )


@app.cell
def _(mo):
    mo.md(
        """
    ## Results
    """
    )
    return


@app.cell
def _(history_constrained, history_unconstrained):
    import matplotlib.pyplot as plt
    import seaborn

    seaborn.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss comparison
    axes[0].plot(history_constrained["loss"], label="With Constraints", linewidth=2)
    axes[0].plot(
        history_unconstrained["loss"], label="Without Constraints", linewidth=2, linestyle="--"
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Total Cost")
    axes[0].set_title("Loss Convergence")
    axes[0].legend()

    # Gradient norm
    axes[1].plot(history_constrained["proj_grad_norm"], label="With Constraints", linewidth=2)
    axes[1].plot(
        history_unconstrained["proj_grad_norm"],
        label="Without Constraints",
        linewidth=2,
        linestyle="--",
    )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Projected Gradient Norm")
    axes[1].set_title("Convergence Check")
    axes[1].legend()
    axes[1].set_yscale("log")

    plt.tight_layout()
    fig
    return (plt,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Final Capacities Comparison
    """
    )
    return


@app.cell
def _(
    devices,
    initial_params,
    parameter_names,
    pd,
    state_constrained,
    state_unconstrained,
):
    # Build comparison dataframe
    comparison_data = []

    for _param_name in parameter_names.keys():
        _device_idx, _attr_name = parameter_names[_param_name]
        _device = devices[_device_idx]
        _names = list(_device.name)

        for _i, _name in enumerate(_names):
            comparison_data.append(
                {
                    "Device": _name,
                    "Parameter": _param_name,
                    "Initial": float(initial_params[_param_name][_i]),
                    "Constrained": float(state_constrained[_param_name][_i]),
                    "Unconstrained": float(state_unconstrained[_param_name][_i]),
                }
            )

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    return (comparison_df,)


@app.cell
def _(comparison_df, np, plt):
    # Bar chart comparison
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    _devices_list = comparison_df["Device"].values
    _x = np.arange(len(_devices_list))
    _width = 0.25

    ax2.bar(_x - _width, comparison_df["Initial"].values, _width, label="Initial", color="#808080")
    ax2.bar(
        _x, comparison_df["Constrained"].values, _width, label="With Constraints", color="#2ecc71"
    )
    ax2.bar(
        _x + _width,
        comparison_df["Unconstrained"].values,
        _width,
        label="Without Constraints",
        color="#3498db",
    )

    ax2.set_xlabel("Device")
    ax2.set_ylabel("Capacity (MW)")
    ax2.set_title("Capacity Comparison: Initial vs Optimized")
    ax2.set_xticks(_x)
    ax2.set_xticklabels(_devices_list, rotation=45, ha="right")
    ax2.legend()

    plt.tight_layout()
    fig2
    return (fig2,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Dispatch Results

    Run dispatch with the optimized capacities to see how the generation mix changes
    with and without budget constraints.
    """
    )
    return


@app.cell
def _(
    cp,
    deepcopy,
    devices,
    net,
    parameter_names,
    state_constrained,
    state_unconstrained,
    time_horizon,
):
    # Update devices with constrained parameters and run dispatch
    _devices_constrained = deepcopy(devices)
    for _param_name, (_device_idx, _attr_name) in parameter_names.items():
        if _param_name in state_constrained:
            setattr(_devices_constrained[_device_idx], _attr_name, state_constrained[_param_name])

    dispatch_constrained = net.dispatch(
        _devices_constrained,
        time_horizon=time_horizon,
        solver=cp.MOSEK,
    )

    # Update devices with unconstrained parameters and run dispatch
    _devices_unconstrained = deepcopy(devices)
    for _param_name, (_device_idx, _attr_name) in parameter_names.items():
        if _param_name in state_unconstrained:
            setattr(
                _devices_unconstrained[_device_idx], _attr_name, state_unconstrained[_param_name]
            )

    dispatch_unconstrained = net.dispatch(
        _devices_unconstrained,
        time_horizon=time_horizon,
        solver=cp.MOSEK,
    )

    print(f"Dispatch (constrained) objective: {dispatch_constrained.problem.value:.2f}")
    print(f"Dispatch (unconstrained) objective: {dispatch_unconstrained.problem.value:.2f}")
    return dispatch_constrained, dispatch_unconstrained


@app.cell
def _(devices, dispatch_constrained, dispatch_unconstrained, np, pd, plt, pypsa_net, time_horizon):
    from zap.devices.injector import Generator as GenDevice

    def get_energy_balance(disp, devs, pypsa_network, t_horizon):
        """Compute energy balance from dispatch results by carrier."""
        carrier_power = {}

        for _dev_idx, _dev in enumerate(devs):
            if not isinstance(_dev, GenDevice):
                continue

            _power_data = disp.power[_dev_idx]
            if isinstance(_power_data, (list, tuple)):
                _power = _power_data[0]
            else:
                _power = _power_data

            _power = np.atleast_2d(_power)
            if _power.shape[0] == t_horizon and _power.shape[1] != t_horizon:
                _power = _power.T

            # Get carrier for each generator
            for _i in range(_dev.num_devices):
                _name = _dev.name[_i] if hasattr(_dev.name, "__getitem__") else f"gen_{_i}"
                if _name in pypsa_network.generators.index:
                    _carrier = pypsa_network.generators.loc[_name, "carrier"]
                else:
                    _carrier = "unknown"

                if _carrier not in carrier_power:
                    carrier_power[_carrier] = np.zeros(t_horizon)
                if _i < _power.shape[0]:
                    carrier_power[_carrier] += _power[_i, :]

        if carrier_power:
            _df = pd.DataFrame(carrier_power)
            _df.index = range(t_horizon)
            return _df
        return pd.DataFrame()

    # Get energy balance for both cases
    energy_constrained = get_energy_balance(dispatch_constrained, devices, pypsa_net, time_horizon)
    energy_unconstrained = get_energy_balance(
        dispatch_unconstrained, devices, pypsa_net, time_horizon
    )

    # Get carrier colors from PyPSA
    carrier_colors = pypsa_net.carriers.color.to_dict() if len(pypsa_net.carriers) > 0 else {}

    # Get load profile
    load_profile = pypsa_net.loads_t.p_set.sum(axis=1).values if len(pypsa_net.loads) > 0 else None

    # Create dispatch comparison plot
    fig_dispatch, axes_dispatch = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _hours = np.arange(time_horizon)

    for _ax, _energy, _title in [
        (axes_dispatch[0], energy_constrained, "Dispatch (With Budget Constraints)"),
        (axes_dispatch[1], energy_unconstrained, "Dispatch (Without Budget Constraints)"),
    ]:
        if not _energy.empty:
            _bottom = np.zeros(len(_hours))
            for _carrier in sorted(_energy.columns):
                _power = _energy[_carrier].values
                _color = carrier_colors.get(_carrier)
                if _color == "":
                    _color = None
                _ax.fill_between(
                    _hours,
                    _bottom,
                    _bottom + _power,
                    label=_carrier,
                    alpha=0.7,
                    color=_color,
                )
                _bottom += _power

        if load_profile is not None:
            _ax.plot(_hours, load_profile, "k-", linewidth=2, label="Load")

        _ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        _ax.set_xlabel("Hour")
        _ax.set_ylabel("Power (MW)")
        _ax.set_title(_title)
        _ax.legend(loc="upper right")
        _ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_dispatch
    return (fig_dispatch,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Verify Budget Constraint Satisfaction
    """
    )
    return


@app.cell
def _(
    constraint_type,
    devices,
    gen_idx,
    rhs_slider,
    sense_select,
    state_constrained,
    state_unconstrained,
):
    # Check if constraints are satisfied
    gen_device = devices[gen_idx]
    gen_names = list(gen_device.name)

    def get_renewable_capacity(state):
        """Sum up solar and wind capacity."""
        total = 0.0
        for i, name in enumerate(gen_names):
            if "solar" in name or "wind" in name:
                total += state["generator"][i]
        return total

    renewable_constrained = get_renewable_capacity(state_constrained)
    renewable_unconstrained = get_renewable_capacity(state_unconstrained)

    print(f"Constraint: {constraint_type.value}")
    print(f"  RHS = {rhs_slider.value}, Sense = {sense_select.value}")
    print()
    print("Renewable Capacity (solar + wind):")
    print(f"  Constrained:   {renewable_constrained:.2f} MW")
    print(f"  Unconstrained: {renewable_unconstrained:.2f} MW")
    print()

    if sense_select.value == "le":
        satisfied = renewable_constrained <= rhs_slider.value + 0.01
        print(
            f"Constraint satisfied? {renewable_constrained:.2f} <= {rhs_slider.value}: {satisfied}"
        )
    else:
        satisfied = renewable_constrained >= rhs_slider.value - 0.01
        print(
            f"Constraint satisfied? {renewable_constrained:.2f} >= {rhs_slider.value}: {satisfied}"
        )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Understanding the Projection

    The budget constraint projection solves a QP at each gradient step:

    ```
    min   (1/2) ||η - η⁺||²
    s.t.  η_min ≤ η ≤ η_max     (box constraints)
          A_le @ η ≤ b_le       (≤ budget constraints)
          A_ge @ η ≥ b_ge       (≥ budget constraints)
    ```

    This finds the closest feasible point to the gradient update `η⁺`, ensuring
    the solution always satisfies both box and budget constraints.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Inspect Constraint Matrices
    """
    )
    return


@app.cell
def _(problem_with_constraints):
    # Show the constraint matrices
    qp = problem_with_constraints._projection_qp

    print("Projection QP Configuration:")
    print(f"  Total dimension: {qp.total_dim}")
    print(f"  Parameter sizes: {qp.param_sizes}")
    print(f"  Parameter offsets: {qp.param_offsets}")
    print()

    if qp.has_le_constraints:
        print("LE Constraints (A @ x <= b):")
        print(f"  A_le shape: {qp.A_le_data.shape}")
        print(f"  A_le:\n{qp.A_le_data.toarray()}")
        print(f"  b_le: {qp.b_le_data}")
        print()

    if qp.has_ge_constraints:
        print("GE Constraints (A @ x >= b):")
        print(f"  A_ge shape: {qp.A_ge_data.shape}")
        print(f"  A_ge:\n{qp.A_ge_data.toarray()}")
        print(f"  b_ge: {qp.b_ge_data}")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Tips for Using Budget Constraints

    1. **CSV Format**: Define constraints in CSV files for easy editing and version control
    2. **Device Names**: Use the exact device names from your network (check `device.name`)
    3. **Sense Selection**: Use "le" for upper bounds, "ge" for lower bounds
    4. **Multipliers**: Use coefficients to weight different devices differently
    5. **Multiple Constraints**: Add multiple constraint groups with different names
    6. **Debugging**: Check `problem._projection_qp.A_le_data` to verify constraint matrices
    """
    )
    return


if __name__ == "__main__":
    app.run()
