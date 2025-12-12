import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # Investment Planning Benchmarking

    This notebook benchmarks Zap's investment planning against PyPSA using the Texas 7-node network.

    Based on `zap/tests/test_pypsa_investment.py::TestTexas7NodeInvestment`
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import cvxpy as cp
    from copy import deepcopy
    import matplotlib.pyplot as plt

    import zap
    from zap.devices.injector import Generator, Load
    from zap.devices.transporter import ACLine
    from zap.devices.storage_unit import StorageUnit
    from zap.importers.pypsa import load_pypsa_network, HOURS_PER_YEAR
    from zap.tests import network_examples as examples

    return (
        ACLine,
        Generator,
        HOURS_PER_YEAR,
        Load,
        StorageUnit,
        cp,
        deepcopy,
        examples,
        load_pypsa_network,
        np,
        pd,
        plt,
        zap,
    )


@app.cell
def _(ACLine, Generator, StorageUnit, np, pd):
    # ============================================================================
    # Plotting Helper Functions (inline)
    # ============================================================================

    def _get_device_carriers(device, pypsa_network, component_name, default_carrier):
        """Get carriers for a device, falling back to PyPSA if not available."""
        if hasattr(device, "carrier") and device.carrier is not None:
            carrier = device.carrier
            if hasattr(carrier, "tolist"):
                return carrier.tolist()
            return list(carrier)

        carriers = []
        pypsa_component = getattr(pypsa_network, component_name)

        for i in range(device.num_devices):
            if hasattr(device, "name") and device.name is not None:
                if hasattr(device.name, "__getitem__"):
                    name = device.name[i]
                else:
                    name = f"Device {i}"
            else:
                name = f"Device {i}"

            if name in pypsa_component.index:
                carriers.append(pypsa_component.loc[name, "carrier"])
            else:
                carriers.append(default_carrier)
        return carriers

    def get_zap_energy_balance(devices, dispatch, pypsa_network, time_horizon):
        """Compute energy balance from Zap dispatch results, matching PyPSA format."""
        carrier_power = {}

        for device_idx, device in enumerate(devices):
            power_data = dispatch.power[device_idx]
            if isinstance(power_data, (list, tuple)):
                power = power_data[0]
            else:
                power = power_data

            power = np.atleast_2d(power)
            if power.shape[0] == time_horizon and power.shape[1] != time_horizon:
                power = power.T

            if isinstance(device, Generator):
                carriers = _get_device_carriers(
                    device, pypsa_network, "generators", "Unknown"
                )
                for i in range(device.num_devices):
                    carrier = carriers[i]
                    if carrier not in carrier_power:
                        carrier_power[carrier] = np.zeros(time_horizon)
                    if i < power.shape[0]:
                        carrier_power[carrier] += power[i, :]

            elif isinstance(device, StorageUnit):
                carriers = _get_device_carriers(
                    device, pypsa_network, "storage_units", "battery"
                )
                for i in range(device.num_devices):
                    carrier = carriers[i]
                    if carrier not in carrier_power:
                        carrier_power[carrier] = np.zeros(time_horizon)
                    if i < power.shape[0]:
                        carrier_power[carrier] += power[i, :]

        if carrier_power:
            df = pd.DataFrame(carrier_power)
            df.index = range(time_horizon)
            return df
        return pd.DataFrame()

    def plot_energy_balance(
        ax, energy_balance, carrier_colors, title="Energy Balance", load_profile=None
    ):
        """Plot stacked area chart for energy balance data."""
        if energy_balance.empty:
            ax.set_title(title)
            return

        hours = energy_balance.index.values
        energy_pos = energy_balance.clip(lower=0)
        energy_neg = energy_balance.clip(upper=0)

        if not energy_pos.empty and energy_pos.sum().sum() > 0:
            bottom = np.zeros(len(hours))
            for carrier in sorted(energy_pos.columns):
                power = energy_pos[carrier].values
                if np.any(power > 0):
                    color = carrier_colors.get(carrier)
                    if color == "":
                        color = None
                    ax.fill_between(
                        hours,
                        bottom,
                        bottom + power,
                        label=carrier,
                        alpha=0.7,
                        color=color,
                    )
                    bottom += power

        if not energy_neg.empty and energy_neg.sum().sum() < 0:
            bottom = np.zeros(len(hours))
            for carrier in sorted(energy_neg.columns):
                power = energy_neg[carrier].values
                if np.any(power < 0):
                    color = carrier_colors.get(carrier)
                    if color == "":
                        color = None
                    ax.fill_between(
                        hours,
                        bottom + power,
                        bottom,
                        label=f"{carrier} (charge)",
                        alpha=0.7,
                        color=color,
                    )
                    bottom += power

        if load_profile is not None:
            ax.plot(hours, load_profile, "k-", linewidth=2, label="Load")

        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        ax.set_ylabel("Power (MW)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def plot_energy_balance_comparison(
        ax_left,
        ax_right,
        pypsa_energy,
        zap_energy,
        carrier_colors,
        load_profile=None,
        title_left="PyPSA Dispatch",
        title_right="Zap Dispatch",
    ):
        """Create side-by-side energy balance plots."""
        ax_left.sharey(ax_right)
        plot_energy_balance(
            ax_left, pypsa_energy, carrier_colors, title_left, load_profile
        )
        plot_energy_balance(
            ax_right, zap_energy, carrier_colors, title_right, load_profile
        )

    def plot_price_comparison(
        ax_left,
        ax_right,
        pypsa_prices,
        zap_prices,
        hours,
        max_buses=10,
        title_left="PyPSA Marginal Prices",
        title_right="Zap Marginal Prices",
    ):
        """Create side-by-side marginal price plots."""
        ax_left.sharey(ax_right)

        for bus_name in pypsa_prices.columns[:max_buses]:
            ax_left.plot(
                hours, pypsa_prices[bus_name].values, label=bus_name[:10], alpha=0.7
            )
        ax_left.set_xlabel("Hour")
        ax_left.set_ylabel("Price ($/MWh)")
        ax_left.set_title(title_left)
        ax_left.grid(True, alpha=0.3)

        for bus in range(min(zap_prices.shape[0], max_buses)):
            ax_right.plot(hours, zap_prices[bus, :], label=f"Bus {bus}", alpha=0.7)
        ax_right.set_xlabel("Hour")
        ax_right.set_ylabel("Price ($/MWh)")
        ax_right.set_title(title_right)
        ax_right.grid(True, alpha=0.3)

    def _extract_scalar_capacity(capacity_value):
        """Extract scalar from potentially array-like capacity value."""
        if hasattr(capacity_value, "numpy"):
            capacity_value = capacity_value.numpy()
        if hasattr(capacity_value, "__len__"):
            arr = np.atleast_1d(capacity_value)
            return float(arr[0])
        return float(capacity_value)

    def aggregate_capacities_by_carrier(pypsa_net, devices, investment_period=None):
        """Aggregate capacities by carrier type for PyPSA and Zap."""
        pypsa_initial = {}
        pypsa_final = {}
        zap_initial = {}
        zap_final = {}

        if investment_period is None and pypsa_net.investment_periods.size > 0:
            investment_period = pypsa_net.investment_periods[0]

        def get_active_assets(component_name):
            if investment_period is not None:
                try:
                    return pypsa_net.get_active_assets(
                        component_name, investment_period
                    )
                except Exception:
                    pass
            attr_map = {
                "Generator": "generators",
                "Line": "lines",
                "StorageUnit": "storage_units",
                "Store": "stores",
                "Link": "links",
            }
            attr_name = attr_map.get(component_name, component_name.lower() + "s")
            return pd.Series(True, index=getattr(pypsa_net, attr_name).index)

        def get_device_by_type(device_type):
            for d in devices:
                if isinstance(d, device_type):
                    return d
            return None

        # Generators
        active_generators = get_active_assets("Generator")
        gen_device = get_device_by_type(Generator)

        for gen_name in pypsa_net.generators.index:
            is_active = (
                active_generators.loc[gen_name]
                if gen_name in active_generators.index
                else False
            )
            p_nom = pypsa_net.generators.loc[gen_name, "p_nom"]
            p_nom_opt = pypsa_net.generators.loc[gen_name, "p_nom_opt"]
            carrier = pypsa_net.generators.loc[gen_name, "carrier"]

            if is_active and p_nom > 0:
                pypsa_initial[carrier] = pypsa_initial.get(carrier, 0) + float(p_nom)
            if p_nom_opt > 0:
                pypsa_final[carrier] = pypsa_final.get(carrier, 0) + float(p_nom_opt)

        if gen_device is not None and hasattr(gen_device, "name"):
            for i, gen_name in enumerate(gen_device.name):
                if gen_name in pypsa_net.generators.index:
                    is_active = (
                        active_generators.loc[gen_name]
                        if gen_name in active_generators.index
                        else False
                    )
                    carrier = pypsa_net.generators.loc[gen_name, "carrier"]
                    p_nom = pypsa_net.generators.loc[gen_name, "p_nom"]

                    if is_active and p_nom > 0:
                        zap_initial[carrier] = zap_initial.get(carrier, 0) + float(
                            p_nom
                        )

                    zap_cap = _extract_scalar_capacity(gen_device.nominal_capacity[i])
                    if zap_cap > 0:
                        zap_final[carrier] = zap_final.get(carrier, 0) + zap_cap

        # Transmission Lines
        active_lines = get_active_assets("Line")
        ac_device = get_device_by_type(ACLine)
        pypsa_initial_line = 0.0
        pypsa_final_line = 0.0
        zap_initial_line = 0.0
        zap_final_line = 0.0

        for line_name in pypsa_net.lines.index:
            is_active = (
                active_lines.loc[line_name]
                if line_name in active_lines.index
                else False
            )
            s_nom = pypsa_net.lines.loc[line_name, "s_nom"]
            s_nom_opt = pypsa_net.lines.loc[line_name, "s_nom_opt"]

            if is_active and s_nom > 0:
                pypsa_initial_line += float(s_nom)
            if s_nom_opt > 0:
                pypsa_final_line += float(s_nom_opt)

        if ac_device is not None and hasattr(ac_device, "name"):
            for i, line_name in enumerate(ac_device.name):
                if line_name in pypsa_net.lines.index:
                    is_active = (
                        active_lines.loc[line_name]
                        if line_name in active_lines.index
                        else False
                    )
                    s_nom = pypsa_net.lines.loc[line_name, "s_nom"]

                    if is_active and s_nom > 0:
                        zap_initial_line += float(s_nom)

                    zap_cap = _extract_scalar_capacity(ac_device.nominal_capacity[i])
                    if zap_cap > 0:
                        zap_final_line += zap_cap

        if pypsa_initial_line > 0:
            pypsa_initial["Transmission"] = pypsa_initial_line
        if pypsa_final_line > 0:
            pypsa_final["Transmission"] = pypsa_final_line
        if zap_initial_line > 0:
            zap_initial["Transmission"] = zap_initial_line
        if zap_final_line > 0:
            zap_final["Transmission"] = zap_final_line

        # Storage Units
        active_storage = get_active_assets("StorageUnit")
        storage_device = get_device_by_type(StorageUnit)
        pypsa_initial_storage = 0.0
        pypsa_final_storage = 0.0
        zap_initial_storage = 0.0
        zap_final_storage = 0.0

        for storage_name in pypsa_net.storage_units.index:
            is_active = (
                active_storage.loc[storage_name]
                if storage_name in active_storage.index
                else False
            )
            p_nom = pypsa_net.storage_units.loc[storage_name, "p_nom"]
            p_nom_opt = pypsa_net.storage_units.loc[storage_name, "p_nom_opt"]

            if is_active and p_nom > 0:
                pypsa_initial_storage += float(p_nom)
            if p_nom_opt > 0:
                pypsa_final_storage += float(p_nom_opt)

        if storage_device is not None and hasattr(storage_device, "name"):
            for i, storage_name in enumerate(storage_device.name):
                if storage_name in pypsa_net.storage_units.index:
                    is_active = (
                        active_storage.loc[storage_name]
                        if storage_name in active_storage.index
                        else False
                    )
                    p_nom = pypsa_net.storage_units.loc[storage_name, "p_nom"]

                    if is_active and p_nom > 0:
                        zap_initial_storage += float(p_nom)

                    zap_cap = _extract_scalar_capacity(storage_device.power_capacity[i])
                    if zap_cap > 0:
                        zap_final_storage += zap_cap

        if pypsa_initial_storage > 0:
            pypsa_initial["Storage"] = pypsa_initial_storage
        if pypsa_final_storage > 0:
            pypsa_final["Storage"] = pypsa_final_storage
        if zap_initial_storage > 0:
            zap_initial["Storage"] = zap_initial_storage
        if zap_final_storage > 0:
            zap_final["Storage"] = zap_final_storage

        return pypsa_initial, pypsa_final, zap_initial, zap_final

    def plot_capacity_comparison(
        ax, pypsa_initial, pypsa_final, zap_initial, zap_final, carrier_colors
    ):
        """Create grouped bar chart comparing initial/final capacities."""
        all_carriers = sorted(
            set(pypsa_final.keys())
            | set(zap_final.keys())
            | set(pypsa_initial.keys())
            | set(zap_initial.keys())
        )

        if not all_carriers:
            ax.set_title("No capacity data available")
            return

        x = np.arange(len(all_carriers))
        width = 0.2

        pypsa_init_vals = [pypsa_initial.get(c, 0) for c in all_carriers]
        zap_init_vals = [zap_initial.get(c, 0) for c in all_carriers]
        pypsa_final_vals = [pypsa_final.get(c, 0) for c in all_carriers]
        zap_final_vals = [zap_final.get(c, 0) for c in all_carriers]

        def get_color(carrier, is_initial, is_pypsa):
            if carrier == "Transmission":
                return (
                    "lightcoral"
                    if is_initial and is_pypsa
                    else "lightblue"
                    if is_initial
                    else "darkred"
                    if is_pypsa
                    else "darkblue"
                )
            elif carrier == "Storage":
                return (
                    "lightgreen"
                    if is_initial and is_pypsa
                    else "lightcyan"
                    if is_initial
                    else "darkgreen"
                    if is_pypsa
                    else "darkcyan"
                )
            return carrier_colors.get(carrier, "gray")

        colors_pypsa_init = [get_color(c, True, True) for c in all_carriers]
        colors_zap_init = [get_color(c, True, False) for c in all_carriers]
        colors_pypsa_final = [get_color(c, False, True) for c in all_carriers]
        colors_zap_final = [get_color(c, False, False) for c in all_carriers]

        ax.bar(
            x - 1.5 * width,
            pypsa_init_vals,
            width,
            label="PyPSA Initial",
            alpha=0.6,
            color=colors_pypsa_init,
            hatch="//",
            edgecolor="white",
        )
        ax.bar(
            x - 0.5 * width,
            zap_init_vals,
            width,
            label="Zap Initial",
            alpha=0.6,
            color=colors_zap_init,
            hatch="//",
            edgecolor="white",
        )
        ax.bar(
            x + 0.5 * width,
            pypsa_final_vals,
            width,
            label="PyPSA Final",
            alpha=0.8,
            color=colors_pypsa_final,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            x + 1.5 * width,
            zap_final_vals,
            width,
            label="Zap Final",
            alpha=0.8,
            color=colors_zap_final,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_ylabel("Total Capacity (MW)")
        ax.set_title("Invested Capacity by Carrier Type")
        ax.set_xticks(x)
        ax.set_xticklabels(all_carriers, rotation=45, ha="right")

        handles, labels = ax.get_legend_handles_labels()
        order = [0, 2, 1, 3]
        ax.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            fontsize=8,
            ncol=2,
            loc="upper left",
        )
        ax.grid(True, alpha=0.3)

    def _extract_param_history(state_history):
        """Extract parameter history from state list."""
        if not state_history:
            return
        first_state = state_history[0]
        param_names = list(first_state.keys())

        for param_name in param_names:
            values = []
            for state in state_history:
                val = state.get(param_name)
                if val is not None:
                    if hasattr(val, "numpy"):
                        val = val.numpy()
                    elif hasattr(val, "detach"):
                        val = val.detach().numpy()
                    values.append(val)
            if values:
                yield param_name, values

    def plot_capacity_evolution(
        ax,
        history,
        devices,
        pypsa_network=None,
        title="Capacity Evolution During Optimization",
    ):
        """Plot the evolution of capacities during optimization iterations."""
        if "param" not in history or not history["param"]:
            ax.text(
                0.5,
                0.5,
                "No param history available.\nAdd PARAM to trackers in solve().",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            return

        state_history = history["param"]
        num_iterations = len(state_history)
        iterations = np.arange(num_iterations)

        carrier_colors = {}
        if pypsa_network is not None and len(pypsa_network.carriers) > 0:
            carrier_colors = pypsa_network.carriers.color.to_dict()

        param_to_device_info = {}
        for device in devices:
            if isinstance(device, Generator):
                param_to_device_info["generator"] = {
                    "device": device,
                    "attr": "nominal_capacity",
                    "label_prefix": "",
                }
            elif isinstance(device, ACLine):
                param_to_device_info["ac_line"] = {
                    "device": device,
                    "attr": "nominal_capacity",
                    "label_prefix": "Line: ",
                }
            elif isinstance(device, StorageUnit):
                param_to_device_info["storage_unit"] = {
                    "device": device,
                    "attr": "power_capacity",
                    "label_prefix": "Storage: ",
                }

        carrier_capacity_history = {}

        for param_name, states in _extract_param_history(state_history):
            if param_name not in param_to_device_info:
                continue

            device_info = param_to_device_info[param_name]
            device = device_info["device"]

            states = np.array(states)
            if states.ndim == 3:
                states = states.squeeze(-1)

            for i in range(states.shape[1]):
                if hasattr(device, "name") and device.name is not None:
                    name = device.name[i]
                else:
                    name = f"{param_name}_{i}"

                if pypsa_network is not None and isinstance(device, Generator):
                    if name in pypsa_network.generators.index:
                        carrier = pypsa_network.generators.loc[name, "carrier"]
                    else:
                        carrier = "other_gen"
                elif isinstance(device, ACLine):
                    carrier = "ac_line"
                elif isinstance(device, StorageUnit):
                    carrier = "storage"
                else:
                    carrier = param_name

                if carrier not in carrier_capacity_history:
                    carrier_capacity_history[carrier] = np.zeros(num_iterations)
                carrier_capacity_history[carrier] += states[:, i]

        for carrier, capacity_values in sorted(carrier_capacity_history.items()):
            color = carrier_colors.get(carrier)
            if color == "" or color is None:
                if carrier == "ac_line":
                    color = "gray"
                elif carrier == "storage":
                    color = "purple"
                else:
                    color = None
            ax.plot(
                iterations,
                capacity_values,
                label=carrier,
                color=color,
                alpha=0.8,
                linewidth=2.0,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Capacity (MW)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best")

    return (
        aggregate_capacities_by_carrier,
        get_zap_energy_balance,
        plot_capacity_comparison,
        plot_capacity_evolution,
        plot_energy_balance_comparison,
        plot_price_comparison,
    )


@app.cell
def _(mo):
    mo.md(
        """
    ## Configuration
    """
    )
    return


@app.cell
def _():
    # Configuration parameters
    NUM_SNAPSHOTS = 48  # Number of hours to simulate
    return (NUM_SNAPSHOTS,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Load Texas 7-Node Network
    """
    )
    return


@app.cell
def _(HOURS_PER_YEAR, NUM_SNAPSHOTS, examples):
    # Load Texas 7-node network
    pypsa_network = examples.load_example_network("texas_7node")

    # Use first NUM_SNAPSHOTS hours for testing
    snapshots = pypsa_network.snapshots[:NUM_SNAPSHOTS]
    pypsa_network.set_snapshots(snapshots)

    # Set snapshot weightings to annualize operational costs
    pypsa_network.snapshot_weightings.loc[:, :] = HOURS_PER_YEAR / len(snapshots)

    print("Network: Texas 7-node")
    print(f"Snapshots: {len(snapshots)} hours")
    print(f"Buses: {len(pypsa_network.buses)}")
    print(f"Generators: {len(pypsa_network.generators)}")
    print(f"Lines: {len(pypsa_network.lines)}")
    print(f"Storage Units: {len(pypsa_network.storage_units)}")
    return pypsa_network, snapshots


@app.cell
def _(mo):
    mo.md(
        """
    ## Import Network to Zap
    """
    )
    return


@app.cell
def _(load_pypsa_network, pypsa_network, snapshots):
    # Import to Zap
    net, devices = load_pypsa_network(pypsa_network, snapshots)

    time_horizon = len(snapshots)
    print(f"Zap network: {net.num_nodes} nodes")
    print(f"Devices: {len(devices)}")
    for i, dev in enumerate(devices):
        print(f"  [{i}] {type(dev).__name__}: {dev.num_devices} units")
    return devices, net, time_horizon


@app.cell
def _(mo):
    mo.md(
        """
    ## Setup Planning Problem
    """
    )
    return


@app.cell
def _(ACLine, Generator, StorageUnit, devices):
    # Define which parameters to optimize
    parameter_names = {}
    for _device_idx, _device in enumerate(devices):
        if isinstance(_device, Generator):
            parameter_names["generator"] = (_device_idx, "nominal_capacity")
        elif isinstance(_device, ACLine):
            parameter_names["ac_line"] = (_device_idx, "nominal_capacity")
        elif isinstance(_device, StorageUnit):
            parameter_names["storage_unit"] = (_device_idx, "power_capacity")

    print("Parameters to optimize:")
    for _param_name, (_idx, _attr) in parameter_names.items():
        print(f"  {_param_name}: device[{_idx}].{_attr}")
    return (parameter_names,)


@app.cell
def _(cp, devices, net, parameter_names, time_horizon, zap):
    # Create dispatch layer for planning
    layer = zap.DispatchLayer(
        net,
        devices,
        parameter_names=parameter_names,
        time_horizon=time_horizon,
        solver=cp.MOSEK,
        solver_kwargs={"verbose": False, "accept_unknown": True},
    )
    return (layer,)


@app.cell
def _(StorageUnit, deepcopy, devices, np, parameter_names):
    # Set up bounds for capacity expansion
    lower_bounds = {}
    upper_bounds = {}

    for _param_name, (_device_idx, _attr_name) in parameter_names.items():
        _device = devices[_device_idx]

        # Handle different device types
        if isinstance(_device, StorageUnit):
            _min_attr = "min_power_capacity"
            _max_attr = "max_power_capacity"
        else:
            _min_attr = "min_nominal_capacity"
            _max_attr = "max_nominal_capacity"

        # Set lower bound
        if hasattr(_device, _min_attr) and getattr(_device, _min_attr) is not None:
            lower_bounds[_param_name] = deepcopy(getattr(_device, _min_attr))
        else:
            lower_bounds[_param_name] = np.zeros_like(getattr(_device, _attr_name))

        # Set upper bound
        if hasattr(_device, _max_attr) and getattr(_device, _max_attr) is not None:
            _max_cap = deepcopy(getattr(_device, _max_attr))
            if np.any(np.isinf(_max_cap)):
                _current_cap = getattr(_device, _attr_name)
                _reasonable_upper = (_current_cap + 1000.0) * 10.0
                _max_cap = np.where(np.isinf(_max_cap), _reasonable_upper, _max_cap)
            upper_bounds[_param_name] = _max_cap
        else:
            _current_cap = getattr(_device, _attr_name)
            upper_bounds[_param_name] = (_current_cap + 1000.0) * 10.0

    # Ensure all lower bounds are non-negative (small positive floor)
    _min_capacity_floor = 0.1  # MW minimum for all devices
    for _param_name in lower_bounds:
        lower_bounds[_param_name] = np.maximum(
            lower_bounds[_param_name], _min_capacity_floor
        )
    print(f"Applied minimum capacity floor: {_min_capacity_floor} MW for all devices")

    # Add higher minimum capacity floor for storage to prevent infeasibility
    # (Storage with very small capacity but non-zero initial/final SOC is infeasible)
    if "storage_unit" in lower_bounds:
        _min_storage_cap = 10.0  # MW minimum
        lower_bounds["storage_unit"] = np.maximum(
            lower_bounds["storage_unit"], _min_storage_cap
        )
        print(f"Applied minimum storage capacity floor: {_min_storage_cap} MW")

    print("Bounds configured:")
    for _param_name in parameter_names.keys():
        print(f"  {_param_name}:")
        print(
            f"    lower: {lower_bounds[_param_name].min():.1f} - {lower_bounds[_param_name].max():.1f}"
        )
        print(
            f"    upper: {upper_bounds[_param_name].min():.1f} - {upper_bounds[_param_name].max():.1f}"
        )
    return lower_bounds, upper_bounds


@app.cell
def _(devices, lower_bounds, np, parameter_names, plt, upper_bounds):
    def _():
        # Plot upper and lower bounds for each parameter
        # Generator gets a wide subplot on top, other params go below
        other_params = {k: v for k, v in parameter_names.items() if k != "generator"}
        num_other = len(other_params)

        # Create figure with generator on top (wide) and others below
        if "generator" in parameter_names and num_other > 0:
            fig_bounds, axes_bounds = plt.subplots(
                2,
                max(num_other, 1),
                figsize=(5 * max(num_other, 1), 10),
                gridspec_kw={"height_ratios": [1.2, 1]},
            )
            # Merge top row for generator
            if num_other > 1:
                gs = axes_bounds[0, 0].get_gridspec()
                for ax in axes_bounds[0, :]:
                    ax.remove()
                ax_gen = fig_bounds.add_subplot(gs[0, :])
            else:
                ax_gen = axes_bounds[0] if num_other == 1 else axes_bounds[0, 0]
                if hasattr(ax_gen, "__iter__"):
                    ax_gen = ax_gen[0]
            axes_other = axes_bounds[1] if num_other > 1 else [axes_bounds[1]]
            if not hasattr(axes_other, "__iter__"):
                axes_other = [axes_other]
        elif "generator" in parameter_names:
            fig_bounds, ax_gen = plt.subplots(1, 1, figsize=(14, 5))
            axes_other = []
        else:
            fig_bounds, axes_bounds = plt.subplots(
                1, num_other, figsize=(5 * num_other, 5)
            )
            ax_gen = None
            axes_other = [axes_bounds] if num_other == 1 else list(axes_bounds)

        # Plot generator bounds (wide subplot on top)
        if "generator" in parameter_names:
            _device_idx, _attr_name = parameter_names["generator"]
            _lower = lower_bounds["generator"].flatten()
            _upper = upper_bounds["generator"].flatten()
            _current = getattr(devices[_device_idx], _attr_name).flatten()

            _x = np.arange(len(_lower))
            _width = 0.25

            ax_gen.bar(
                _x - _width,
                _lower,
                width=_width,
                label="Lower Bound",
                color="tab:blue",
                alpha=0.7,
            )
            ax_gen.bar(
                _x,
                _current,
                width=_width,
                label="Current",
                color="tab:green",
                alpha=0.7,
            )
            ax_gen.bar(
                _x + _width,
                _upper,
                width=_width,
                label="Upper Bound",
                color="tab:red",
                alpha=0.7,
            )

            ax_gen.set_xlabel("Generator Index")
            ax_gen.set_ylabel("Capacity (MW)")
            ax_gen.set_title("Generator Bounds")
            ax_gen.legend()
            ax_gen.grid(True, alpha=0.3)

            if (
                _upper.max() > 0
                and _upper.max()
                / max(_lower[_lower > 0].min() if np.any(_lower > 0) else 1, 1)
                > 100
            ):
                ax_gen.set_yscale("log")

        # Plot other parameters in bottom row
        for _ax, (_param_name, (_device_idx, _attr_name)) in zip(
            axes_other, other_params.items()
        ):
            _lower = lower_bounds[_param_name].flatten()
            _upper = upper_bounds[_param_name].flatten()
            _current = getattr(devices[_device_idx], _attr_name).flatten()

            _x = np.arange(len(_lower))
            _width = 0.25

            _ax.bar(
                _x - _width,
                _lower,
                width=_width,
                label="Lower Bound",
                color="tab:blue",
                alpha=0.7,
            )
            _ax.bar(
                _x,
                _current,
                width=_width,
                label="Current",
                color="tab:green",
                alpha=0.7,
            )
            _ax.bar(
                _x + _width,
                _upper,
                width=_width,
                label="Upper Bound",
                color="tab:red",
                alpha=0.7,
            )

            _ax.set_xlabel("Device Index")
            _ax.set_ylabel("Capacity (MW)")
            _ax.set_title(f"{_param_name.replace('_', ' ').title()} Bounds")
            _ax.legend()
            _ax.grid(True, alpha=0.3)

            if (
                _upper.max() > 0
                and _upper.max()
                / max(_lower[_lower > 0].min() if np.any(_lower > 0) else 1, 1)
                > 100
            ):
                _ax.set_yscale("log")

        plt.tight_layout()
        return fig_bounds

    _()
    return


@app.cell
def _(
    HOURS_PER_YEAR,
    devices,
    layer,
    lower_bounds,
    net,
    snapshots,
    upper_bounds,
    zap,
):
    # Define objectives
    op_objective = zap.planning.DispatchCostObjective(net, devices)
    inv_objective = zap.planning.InvestmentObjective(devices, layer)

    # Snapshot weight to annualize operational costs
    snapshot_weight = HOURS_PER_YEAR / len(snapshots)

    # Create planning problem
    problem = zap.planning.PlanningProblem(
        operation_objective=op_objective,
        investment_objective=inv_objective,
        layer=layer,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        snapshot_weight=snapshot_weight,
    )

    print(
        f"Snapshot weight: {snapshot_weight:.2f} (annualizing {len(snapshots)} hours)"
    )
    return (problem,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Initialize Parameters
    """
    )
    return


@app.cell
def _(deepcopy, devices, parameter_names):
    # Initialize parameters at current values
    initial_params = {}
    for _param_name, (_device_idx, _attr_name) in parameter_names.items():
        initial_params[_param_name] = deepcopy(
            getattr(devices[_device_idx], _attr_name)
        )

    print("Initial parameters:")
    for _param_name, _values in initial_params.items():
        print(f"  {_param_name}: {_values.shape}, sum={_values.sum():.1f}")
    return (initial_params,)


@app.cell
def _(Load, devices, initial_params, np, parameter_names):
    # Warm-start: distribute capacity to match peak load

    # Calculate peak load across all time steps
    peak_load = 0.0
    for _device in devices:
        if isinstance(_device, Load):
            peak_load += np.max(np.sum(_device.load, axis=0))

    # Calculate current total generator capacity
    if "generator" in initial_params:
        current_gen_capacity = np.sum(initial_params["generator"]) / 20.0
    else:
        current_gen_capacity = 0.0

    # Calculate capacity gap
    capacity_gap = peak_load - current_gen_capacity

    print(f"Peak load: {peak_load:.1f} MW")
    print(f"Current generator capacity: {current_gen_capacity:.1f} MW")
    print(f"Capacity gap: {capacity_gap:.1f} MW")

    # Distribute gap evenly across all generators
    if "generator" in initial_params and capacity_gap > 0:
        _device_idx, _attr_name = parameter_names["generator"]
        num_generators = initial_params["generator"].size
        capacity_addition_per_gen = capacity_gap / num_generators

        initial_params["generator"] = (
            initial_params["generator"] + capacity_addition_per_gen
        )

        print(
            f"Added {capacity_addition_per_gen:.1f} MW to each of {num_generators} generators"
        )
        print(
            f"New total generator capacity: {np.sum(initial_params['generator']):.1f} MW"
        )
    elif capacity_gap <= 0:
        print("No capacity gap - generators already exceed peak load")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Solve Planning Problem
    """
    )
    return


@app.cell
def _(initial_params, problem, zap):
    from zap.planning.trackers import (
        LOSS,
        GRAD_NORM,
        PROJ_GRAD_NORM,
        TIME,
        SUBOPTIMALITY,
        PARAM,
    )

    NUM_ITERATIONS = 250  # Gradient descent iterations
    STEP_SIZE = 1e-1  # Learning rate
    CLIP = 1e4  # Gradient clipping

    # Solve planning problem
    optimized_params, history = problem.solve(
        num_iterations=NUM_ITERATIONS,
        algorithm=zap.planning.GradientDescent(step_size=STEP_SIZE, clip=CLIP),
        initial_state=initial_params,
        trackers=[LOSS, GRAD_NORM, PROJ_GRAD_NORM, TIME, SUBOPTIMALITY, PARAM],
    )

    print(f"Optimization completed in {NUM_ITERATIONS} iterations")
    print(f"Final loss: {history['loss'][-1]:.2f}")
    print(f"Final grad norm: {history['grad_norm'][-1]:.6f}")
    return history, optimized_params


@app.cell
def _(devices, np, optimized_params, parameter_names):
    # Update devices with optimized parameters
    for _param_name, (_device_idx, _attr_name) in parameter_names.items():
        if _param_name in optimized_params:
            _new_capacity = optimized_params[_param_name]
            if hasattr(_new_capacity, "numpy"):
                _new_capacity = _new_capacity.numpy()
            setattr(devices[_device_idx], _attr_name, _new_capacity)
            print(f"Updated {_param_name}: sum={np.sum(_new_capacity):.1f}")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Run Dispatch with Optimized Parameters
    """
    )
    return


@app.cell
def _(cp, devices, net, time_horizon):
    # Run dispatch with optimized parameters
    dispatch = net.dispatch(
        devices,
        time_horizon=time_horizon,
        solver=cp.MOSEK,
    )

    print(f"Dispatch objective: {dispatch.problem.value:.2f}")
    return (dispatch,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Compare with PyPSA
    """
    )
    return


@app.cell
def _(HOURS_PER_YEAR, pypsa_network, snapshots):
    # Run PyPSA optimization
    pypsa_opt = pypsa_network.copy()
    pypsa_opt.set_snapshots(snapshots)
    pypsa_opt.snapshot_weightings.loc[:, :] = HOURS_PER_YEAR / len(snapshots)

    has_investment_periods = pypsa_opt.investment_periods.size > 0
    pypsa_opt.optimize(
        solver_name="highs", multi_investment_periods=has_investment_periods
    )

    print(f"PyPSA objective: {pypsa_opt.objective:.2f}")
    return (pypsa_opt,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Visualization
    """
    )
    return


@app.cell
def _(history, np, plt):
    # Plot optimization convergence
    fig_conv, axes_conv = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    axes_conv[0, 0].plot(history["loss"])
    axes_conv[0, 0].set_xlabel("Iteration")
    axes_conv[0, 0].set_ylabel("Loss")
    axes_conv[0, 0].set_title("Loss Convergence")
    axes_conv[0, 0].grid(True, alpha=0.3)

    # Gradient norm
    axes_conv[0, 1].plot(history["grad_norm"])
    axes_conv[0, 1].set_xlabel("Iteration")
    axes_conv[0, 1].set_ylabel("Gradient Norm")
    axes_conv[0, 1].set_title("Gradient Norm")
    axes_conv[0, 1].set_yscale("log")
    axes_conv[0, 1].grid(True, alpha=0.3)

    # Projected gradient norm
    axes_conv[1, 0].plot(history["proj_grad_norm"])
    axes_conv[1, 0].set_xlabel("Iteration")
    axes_conv[1, 0].set_ylabel("Projected Gradient Norm")
    axes_conv[1, 0].set_title("Projected Gradient Norm")
    axes_conv[1, 0].set_yscale("log")
    axes_conv[1, 0].grid(True, alpha=0.3)

    # Time per iteration
    axes_conv[1, 1].plot(np.diff(history["time"]))
    axes_conv[1, 1].set_xlabel("Iteration")
    axes_conv[1, 1].set_ylabel("Time (s)")
    axes_conv[1, 1].set_title("Time per Iteration")
    axes_conv[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_conv
    return


@app.cell
def _(
    aggregate_capacities_by_carrier,
    devices,
    dispatch,
    get_zap_energy_balance,
    history,
    np,
    plot_capacity_comparison,
    plot_capacity_evolution,
    plot_energy_balance_comparison,
    plot_price_comparison,
    plt,
    pypsa_opt,
    time_horizon,
):
    # Create comparison plots
    fig_comp, axes_comp = plt.subplots(4, 2, figsize=(16, 20))

    hours = np.arange(time_horizon)

    # Get energy balance data
    pypsa_energy_raw = (
        pypsa_opt.statistics.energy_balance(
            comps=["Generator", "StorageUnit"],
            aggregate_time=False,
            nice_names=False,
        )
        .droplevel(0)
        .T
    )
    pypsa_energy = pypsa_energy_raw.T.groupby(level="carrier").sum().T
    pypsa_energy.index = hours

    zap_energy = get_zap_energy_balance(devices, dispatch, pypsa_opt, time_horizon)

    # Get carrier colors and load profile
    carrier_colors = (
        pypsa_opt.carriers.color.to_dict() if len(pypsa_opt.carriers) > 0 else {}
    )
    load_profile = (
        pypsa_opt.loads_t.p_set.sum(axis=1).values if len(pypsa_opt.loads) > 0 else None
    )

    # Row 1: Energy balance comparison
    plot_energy_balance_comparison(
        axes_comp[0, 0],
        axes_comp[0, 1],
        pypsa_energy,
        zap_energy,
        carrier_colors,
        load_profile,
        title_left="PyPSA Dispatch by Carrier",
        title_right="Zap Dispatch by Carrier",
    )

    # Row 2: Price comparison
    plot_price_comparison(
        axes_comp[1, 0],
        axes_comp[1, 1],
        pypsa_opt.buses_t.marginal_price,
        dispatch.prices,
        hours,
    )

    # Row 3: Capacity comparison
    (
        pypsa_initial,
        pypsa_final,
        zap_initial,
        zap_final,
    ) = aggregate_capacities_by_carrier(pypsa_opt, devices)
    plot_capacity_comparison(
        axes_comp[2, 0],
        pypsa_initial,
        pypsa_final,
        zap_initial,
        zap_final,
        carrier_colors,
    )

    # Row 3 right: Summary text
    axes_comp[2, 1].axis("off")
    summary_text = f"""
    Optimization Summary
    ====================

    PyPSA Objective: ${pypsa_opt.objective:,.0f}

    Initial Capacities (MW):
      PyPSA: {sum(pypsa_initial.values()):,.0f}
      Zap: {sum(zap_initial.values()):,.0f}

    Final Capacities (MW):
      PyPSA: {sum(pypsa_final.values()):,.0f}
      Zap: {sum(zap_final.values()):,.0f}

    Capacity Expansion:
      PyPSA: {sum(pypsa_final.values()) - sum(pypsa_initial.values()):,.0f} MW
      Zap: {sum(zap_final.values()) - sum(zap_initial.values()):,.0f} MW
    """
    axes_comp[2, 1].text(
        0.1,
        0.9,
        summary_text,
        transform=axes_comp[2, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    # Row 4: Capacity evolution
    plot_capacity_evolution(
        axes_comp[3, 0],
        history,
        devices,
        pypsa_network=pypsa_opt,
        title="Capacity Evolution During Optimization",
    )

    # Row 4 right: Loss evolution zoomed
    axes_comp[3, 1].plot(history["loss"])
    axes_comp[3, 1].axhline(
        y=pypsa_opt.objective, color="r", linestyle="--", label="PyPSA optimal"
    )
    axes_comp[3, 1].set_xlabel("Iteration")
    axes_comp[3, 1].set_ylabel("Loss")
    axes_comp[3, 1].set_title("Loss vs PyPSA Optimal")
    axes_comp[3, 1].set_yscale("log")
    axes_comp[3, 1].legend()
    axes_comp[3, 1].grid(True, alpha=0.3)

    # Add legends
    axes_comp[0, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    axes_comp[1, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.suptitle(
        "Texas 7-Node Investment Planning: PyPSA vs Zap", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    fig_comp
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Detailed Capacity Comparison
    """
    )
    return


@app.cell
def _(Generator, devices, pd, pypsa_opt):
    # Compare generator capacities
    gen_comparison = []

    for _device in devices:
        if isinstance(_device, Generator):
            for _idx, _gen_name in enumerate(_device.name):
                if _gen_name in pypsa_opt.generators.index:
                    _pypsa_cap = pypsa_opt.generators.loc[_gen_name, "p_nom_opt"]
                    _zap_cap = float(_device.nominal_capacity[_idx].flatten()[0])
                    _carrier = pypsa_opt.generators.loc[_gen_name, "carrier"]

                    gen_comparison.append(
                        {
                            "name": _gen_name,
                            "carrier": _carrier,
                            "pypsa_cap": _pypsa_cap,
                            "zap_cap": _zap_cap,
                            "diff": _zap_cap - _pypsa_cap,
                            "diff_pct": (_zap_cap - _pypsa_cap) / _pypsa_cap * 100
                            if _pypsa_cap > 0
                            else 0,
                        }
                    )

    gen_df = pd.DataFrame(gen_comparison)
    gen_df = gen_df.sort_values("carrier")
    gen_df
    return (gen_df,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary Statistics
    """
    )
    return


@app.cell
def _(gen_df, history, np, pypsa_opt):
    # Print summary statistics
    print("=" * 60)
    print("INVESTMENT PLANNING BENCHMARK SUMMARY")
    print("=" * 60)
    print()
    print(f"PyPSA Objective: ${pypsa_opt.objective:,.2f}")
    print(f"Zap Final Loss:  ${history['loss'][-1]:,.2f}")
    print(
        f"Difference:      ${history['loss'][-1] - pypsa_opt.objective:,.2f} ({(history['loss'][-1] / pypsa_opt.objective - 1) * 100:.2f}%)"
    )
    print()
    print("Generator Capacity Comparison:")
    print(f"  Total PyPSA: {gen_df['pypsa_cap'].sum():,.1f} MW")
    print(f"  Total Zap:   {gen_df['zap_cap'].sum():,.1f} MW")
    print(f"  Mean Abs Diff: {np.abs(gen_df['diff']).mean():,.1f} MW")
    print(f"  Max Abs Diff:  {np.abs(gen_df['diff']).max():,.1f} MW")
    print()
    print("By Carrier:")
    for _carrier in gen_df["carrier"].unique():
        _carrier_data = gen_df[gen_df["carrier"] == _carrier]
        print(f"  {_carrier}:")
        print(f"    PyPSA: {_carrier_data['pypsa_cap'].sum():,.1f} MW")
        print(f"    Zap:   {_carrier_data['zap_cap'].sum():,.1f} MW")
    return


if __name__ == "__main__":
    app.run()
