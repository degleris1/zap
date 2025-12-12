"""
Plotting helper functions for PyPSA-Zap comparison tests.

This module provides reusable utilities for creating side-by-side comparison
plots between PyPSA and Zap optimization results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import importlib.util

# Import device classes from Zap if not already imported
try:
    Generator
except NameError:
    from zap.devices.injector import Generator
try:
    Load
except NameError:
    from zap.devices.injector import Load
try:
    ACLine
except NameError:
    from zap.devices.transporter import ACLine
try:
    DCLine
except NameError:
    from zap.devices.transporter import DCLine
try:
    StorageUnit
except NameError:
    from zap.devices.storage_unit import StorageUnit


# Try to import matplotlib
try:
    importlib.util.find_spec("matplotlib")

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def get_zap_energy_balance(
    devices: list,
    dispatch,
    pypsa_network,
    time_horizon: int,
) -> pd.DataFrame:
    """
    Compute energy balance from Zap dispatch results, matching PyPSA statistics format.

    Aggregates power by carrier for generators and storage units.
    Positive values indicate generation/discharge, negative indicates charging.

    Parameters
    ----------
    devices : list
        List of Zap device objects
    dispatch : DispatchOutcome
        Zap dispatch results containing power arrays
    pypsa_network : pypsa.Network
        PyPSA network (used to look up carrier information)
    time_horizon : int
        Number of timesteps

    Returns
    -------
    pd.DataFrame
        Energy balance with carriers as columns and timesteps as index.
        Format matches PyPSA's statistics.energy_balance() output.
    """
    carrier_power = {}

    for device_idx, device in enumerate(devices):
        # Get power from dispatch - handle different array structures
        power_data = dispatch.power[device_idx]
        if isinstance(power_data, (list, tuple)):
            power = power_data[0]
        else:
            power = power_data

        # Ensure power is 2D (num_devices x time_horizon)
        power = np.atleast_2d(power)
        if power.shape[0] == time_horizon and power.shape[1] != time_horizon:
            power = power.T  # Transpose if needed

        if isinstance(device, Generator):
            # Get carrier for each generator
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
            # Get carrier for each storage unit
            carriers = _get_device_carriers(
                device, pypsa_network, "storage_units", "battery"
            )
            for i in range(device.num_devices):
                carrier = carriers[i]
                if carrier not in carrier_power:
                    carrier_power[carrier] = np.zeros(time_horizon)
                if i < power.shape[0]:
                    carrier_power[carrier] += power[i, :]

    # Create DataFrame matching PyPSA format
    if carrier_power:
        df = pd.DataFrame(carrier_power)
        df.index = range(time_horizon)
        return df
    return pd.DataFrame()


def _get_device_carriers(
    device, pypsa_network, component_name: str, default_carrier: str
) -> list:
    """
    Get carriers for a device, falling back to PyPSA if not available on device.

    Parameters
    ----------
    device : Device
        Zap device object
    pypsa_network : pypsa.Network
        PyPSA network for carrier lookup
    component_name : str
        PyPSA component name (e.g., 'generators', 'storage_units')
    default_carrier : str
        Default carrier if not found

    Returns
    -------
    list
        List of carrier strings for each device
    """
    if hasattr(device, "carrier") and device.carrier is not None:
        carrier = device.carrier
        if hasattr(carrier, "tolist"):
            return carrier.tolist()
        return list(carrier)

    # Fall back to PyPSA lookup
    carriers = []
    pypsa_component = getattr(pypsa_network, component_name)

    for i in range(device.num_devices):
        # Handle different name formats (list, array, pandas Index)
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


def plot_energy_balance(
    ax,
    energy_balance: pd.DataFrame,
    carrier_colors: dict,
    title: str = "Energy Balance",
    load_profile: Optional[np.ndarray] = None,
):
    """
    Plot stacked area chart for energy balance data.

    Handles positive values (generation/discharge) stacked upward and
    negative values (charging) stacked downward from zero.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    energy_balance : pd.DataFrame
        Energy balance with carriers as columns, timesteps as index
    carrier_colors : dict
        Mapping of carrier names to colors
    title : str
        Plot title
    load_profile : np.ndarray, optional
        Total load to overlay as line
    """
    if energy_balance.empty:
        ax.set_title(title)
        return

    hours = energy_balance.index.values

    # Separate positive and negative values
    energy_pos = energy_balance.clip(lower=0)
    energy_neg = energy_balance.clip(upper=0)

    # Plot positive values (stacked from bottom up)
    if not energy_pos.empty and energy_pos.sum().sum() > 0:
        bottom = np.zeros(len(hours))
        for carrier in sorted(energy_pos.columns):
            power = energy_pos[carrier].values
            if np.any(power > 0):  # Changed: plot if ANY positive values exist
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

    # Plot negative values (stacked from top down)
    if not energy_neg.empty and energy_neg.sum().sum() < 0:
        bottom = np.zeros(len(hours))
        for carrier in sorted(energy_neg.columns):
            power = energy_neg[carrier].values
            if np.any(power < 0):  # Changed: plot if ANY negative values exist
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

    # Overlay load profile if provided
    if load_profile is not None:
        ax.plot(hours, load_profile, "k-", linewidth=2, label="Load")

    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_energy_balance_comparison(
    ax_left,
    ax_right,
    pypsa_energy: pd.DataFrame,
    zap_energy: pd.DataFrame,
    carrier_colors: dict,
    load_profile: Optional[np.ndarray] = None,
    title_left: str = "PyPSA Dispatch",
    title_right: str = "Zap Dispatch",
):
    """
    Create side-by-side energy balance plots for PyPSA and Zap.

    Parameters
    ----------
    ax_left, ax_right : matplotlib.axes.Axes
        Axes for PyPSA (left) and Zap (right) plots
    pypsa_energy, zap_energy : pd.DataFrame
        Energy balance DataFrames
    carrier_colors : dict
        Mapping of carrier names to colors
    load_profile : np.ndarray, optional
        Total load to overlay
    title_left, title_right : str
        Titles for each subplot
    """
    # Share y-axis between plots
    ax_left.sharey(ax_right)

    plot_energy_balance(ax_left, pypsa_energy, carrier_colors, title_left, load_profile)
    plot_energy_balance(ax_right, zap_energy, carrier_colors, title_right, load_profile)


def plot_price_comparison(
    ax_left,
    ax_right,
    pypsa_prices: pd.DataFrame,
    zap_prices: np.ndarray,
    hours: np.ndarray,
    max_buses: int = 10,
    title_left: str = "PyPSA Marginal Prices",
    title_right: str = "Zap Marginal Prices",
):
    """
    Create side-by-side marginal price plots.

    Parameters
    ----------
    ax_left, ax_right : matplotlib.axes.Axes
        Axes for PyPSA (left) and Zap (right) plots
    pypsa_prices : pd.DataFrame
        PyPSA buses_t.marginal_price DataFrame
    zap_prices : np.ndarray
        Zap dispatch.prices array (num_buses x time_horizon)
    hours : np.ndarray
        Time axis values
    max_buses : int
        Maximum number of buses to plot
    title_left, title_right : str
        Titles for each subplot
    """
    ax_left.sharey(ax_right)

    # PyPSA prices
    for bus_name in pypsa_prices.columns[:max_buses]:
        ax_left.plot(
            hours, pypsa_prices[bus_name].values, label=bus_name[:10], alpha=0.7
        )

    ax_left.set_xlabel("Hour")
    ax_left.set_ylabel("Price ($/MWh)")
    ax_left.set_title(title_left)
    ax_left.grid(True, alpha=0.3)

    # Zap prices
    for bus in range(min(zap_prices.shape[0], max_buses)):
        ax_right.plot(hours, zap_prices[bus, :], label=f"Bus {bus}", alpha=0.7)

    ax_right.set_xlabel("Hour")
    ax_right.set_ylabel("Price ($/MWh)")
    ax_right.set_title(title_right)
    ax_right.grid(True, alpha=0.3)


def plot_line_flows(
    ax_left,
    ax_right,
    pypsa_net,
    dispatch,
    devices: list,
    hours: np.ndarray,
    max_ac_lines: int = 10,
    max_dc_lines: int = 5,
):
    """
    Create side-by-side line flow plots for AC and DC lines.

    Parameters
    ----------
    ax_left, ax_right : matplotlib.axes.Axes
        Axes for PyPSA (left) and Zap (right) plots
    pypsa_net : pypsa.Network
        PyPSA network with optimized results
    dispatch : DispatchOutcome
        Zap dispatch results
    devices : list
        List of Zap devices
    hours : np.ndarray
        Time axis values
    max_ac_lines, max_dc_lines : int
        Maximum number of lines to plot
    """
    # PyPSA line flows
    if (
        len(pypsa_net.lines) > 0
        and hasattr(pypsa_net, "lines_t")
        and hasattr(pypsa_net.lines_t, "p0")
    ):
        line_flows = pypsa_net.lines_t.p0
        for line_name in line_flows.columns[:max_ac_lines]:
            ax_left.plot(
                hours,
                line_flows[line_name].values,
                label=f"AC: {line_name}",
                alpha=0.6,
                linewidth=1.5,
            )

    # PyPSA DC link flows
    if (
        len(pypsa_net.links) > 0
        and hasattr(pypsa_net, "links_t")
        and hasattr(pypsa_net.links_t, "p0")
    ):
        dc_links = (
            pypsa_net.links[pypsa_net.links.carrier == "DC"]
            if "carrier" in pypsa_net.links.columns
            else pypsa_net.links
        )
        if len(dc_links) > 0:
            link_flows = pypsa_net.links_t.p0[dc_links.index]
            for link_name in link_flows.columns[:max_dc_lines]:
                ax_left.plot(
                    hours,
                    link_flows[link_name].values,
                    label=f"DC: {link_name}",
                    alpha=0.6,
                    linewidth=1.5,
                    linestyle="--",
                )

    ax_left.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    ax_left.set_xlabel("Hour")
    ax_left.set_ylabel("Power Flow (MW)")
    ax_left.set_title("PyPSA Line Flows")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=6, ncol=2)

    # Zap line flows
    for device_idx, device in enumerate(devices):
        if isinstance(device, ACLine):
            power = dispatch.power[device_idx][0]
            for i in range(min(max_ac_lines, device.num_devices)):
                name = device.name[i] if hasattr(device, "name") else f"AC Line {i}"
                ax_right.plot(
                    hours,
                    power[i, :],
                    label=f"AC: {name}",
                    alpha=0.6,
                    linewidth=1.5,
                )

        elif isinstance(device, DCLine):
            power = dispatch.power[device_idx][0]
            for i in range(min(max_dc_lines, device.num_devices)):
                name = device.name[i] if hasattr(device, "name") else f"DC Line {i}"
                ax_right.plot(
                    hours,
                    power[i, :],
                    label=f"DC: {name}",
                    alpha=0.6,
                    linewidth=1.5,
                    linestyle="--",
                )

    ax_right.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
    ax_right.set_xlabel("Hour")
    ax_right.set_ylabel("Power Flow (MW)")
    ax_right.set_title("Zap Line Flows")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=6, ncol=2)


def extract_scalar_capacity(capacity_value) -> float:
    """
    Extract scalar from potentially array-like capacity value.

    Parameters
    ----------
    capacity_value : scalar or array-like
        Capacity value that may be scalar, 1-element array, or tensor

    Returns
    -------
    float
        Scalar capacity value
    """
    if hasattr(capacity_value, "numpy"):
        capacity_value = capacity_value.numpy()
    if hasattr(capacity_value, "__len__"):
        arr = np.atleast_1d(capacity_value)
        return float(arr[0])
    return float(capacity_value)


def aggregate_capacities_by_carrier(
    pypsa_net,
    devices: list,
    investment_period=None,
) -> tuple[dict, dict, dict, dict]:
    """
    Aggregate capacities by carrier type for PyPSA and Zap.

    Returns initial and final capacities for both frameworks.

    Parameters
    ----------
    pypsa_net : pypsa.Network
        PyPSA network with optimized results
    devices : list
        List of Zap devices
    investment_period : optional
        Investment period for active assets lookup

    Returns
    -------
    tuple of dicts
        (pypsa_initial, pypsa_final, zap_initial, zap_final)
        Each dict maps carrier/type name to total capacity
    """
    pypsa_initial = {}
    pypsa_final = {}
    zap_initial = {}
    zap_final = {}

    # Determine investment period
    if investment_period is None and pypsa_net.investment_periods.size > 0:
        investment_period = pypsa_net.investment_periods[0]

    # Helper to get active assets
    def get_active_assets(component_name):
        if investment_period is not None:
            try:
                return pypsa_net.get_active_assets(component_name, investment_period)
            except Exception:
                pass
        # Map component names to PyPSA attribute names
        attr_map = {
            "Generator": "generators",
            "Line": "lines",
            "StorageUnit": "storage_units",
            "Store": "stores",
            "Link": "links",
        }
        attr_name = attr_map.get(component_name, component_name.lower() + "s")
        return pd.Series(True, index=getattr(pypsa_net, attr_name).index)

    # Get device by type
    def get_device_by_type(device_type):
        for d in devices:
            if isinstance(d, device_type):
                return d
        return None

    # --- Generators ---
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

        # PyPSA initial (only active)
        if is_active and p_nom > 0:
            pypsa_initial[carrier] = pypsa_initial.get(carrier, 0) + float(p_nom)

        # PyPSA final
        if p_nom_opt > 0:
            pypsa_final[carrier] = pypsa_final.get(carrier, 0) + float(p_nom_opt)

    # Zap generator capacities
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

                # Zap initial (only active)
                if is_active and p_nom > 0:
                    zap_initial[carrier] = zap_initial.get(carrier, 0) + float(p_nom)

                # Zap final
                zap_cap = extract_scalar_capacity(gen_device.nominal_capacity[i])
                if zap_cap > 0:
                    zap_final[carrier] = zap_final.get(carrier, 0) + zap_cap

    # --- Transmission Lines ---
    active_lines = get_active_assets("Line")
    ac_device = get_device_by_type(ACLine)
    pypsa_initial_line = 0.0
    pypsa_final_line = 0.0
    zap_initial_line = 0.0
    zap_final_line = 0.0

    for line_name in pypsa_net.lines.index:
        is_active = (
            active_lines.loc[line_name] if line_name in active_lines.index else False
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

                zap_cap = extract_scalar_capacity(ac_device.nominal_capacity[i])
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

    # --- Storage Units ---
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

                zap_cap = extract_scalar_capacity(storage_device.power_capacity[i])
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
    ax,
    pypsa_initial: dict,
    pypsa_final: dict,
    zap_initial: dict,
    zap_final: dict,
    carrier_colors: dict,
):
    """
    Create grouped bar chart comparing initial/final capacities for PyPSA and Zap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    pypsa_initial, pypsa_final : dict
        PyPSA capacity dicts mapping carrier to MW
    zap_initial, zap_final : dict
        Zap capacity dicts mapping carrier to MW
    carrier_colors : dict
        Mapping of carrier names to colors
    """
    # Get all carriers
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

    # Color helper
    def get_color(carrier, is_initial, is_pypsa):
        if carrier == "Transmission":
            if is_initial:
                return "lightcoral" if is_pypsa else "lightblue"
            return "darkred" if is_pypsa else "darkblue"
        elif carrier == "Storage":
            if is_initial:
                return "lightgreen" if is_pypsa else "lightcyan"
            return "darkgreen" if is_pypsa else "darkcyan"
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
    order = [0, 2, 1, 3]  # PyPSA Initial, PyPSA Final, Zap Initial, Zap Final
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        fontsize=8,
        ncol=2,
        loc="upper left",
    )
    ax.grid(True, alpha=0.3)


def export_comparison_csvs(
    output_dir: Path,
    filename_prefix: str,
    pypsa_initial: dict,
    pypsa_final: dict,
    zap_initial: dict,
    zap_final: dict,
    pypsa_prices: pd.DataFrame,
    zap_prices: np.ndarray,
    hours: np.ndarray,
):
    """
    Export comparison data to CSV files.

    Creates three CSV files:
    1. Capacity comparison by carrier
    2. Price comparison by bus and hour
    3. Summary statistics

    Parameters
    ----------
    output_dir : Path
        Directory to save CSV files
    filename_prefix : str
        Prefix for filenames
    pypsa_initial, pypsa_final : dict
        PyPSA capacity dicts
    zap_initial, zap_final : dict
        Zap capacity dicts
    pypsa_prices : pd.DataFrame
        PyPSA marginal prices
    zap_prices : np.ndarray
        Zap marginal prices
    hours : np.ndarray
        Time axis values
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Capacity comparison
    all_carriers = sorted(set(pypsa_final.keys()) | set(zap_final.keys()))
    if all_carriers:
        pypsa_vals = [pypsa_final.get(c, 0) for c in all_carriers]
        zap_vals = [zap_final.get(c, 0) for c in all_carriers]
        capacity_df = pd.DataFrame(
            {
                "Carrier": all_carriers,
                "PyPSA_Capacity_MW": pypsa_vals,
                "Zap_Capacity_MW": zap_vals,
            }
        )
        capacity_df["Difference_MW"] = (
            capacity_df["Zap_Capacity_MW"] - capacity_df["PyPSA_Capacity_MW"]
        )
        capacity_df["Percent_Difference"] = (
            capacity_df["Difference_MW"] / capacity_df["PyPSA_Capacity_MW"] * 100
        ).round(2)
        capacity_df.to_csv(
            output_dir / f"{filename_prefix}_capacity_comparison.csv", index=False
        )

    # 2. Price comparison
    price_data = []
    for hour_idx, hour in enumerate(hours):
        for bus_idx, bus_name in enumerate(pypsa_prices.columns):
            pypsa_price = pypsa_prices.iloc[hour_idx, bus_idx]
            zap_price = (
                zap_prices[bus_idx, hour_idx]
                if bus_idx < zap_prices.shape[0]
                else np.nan
            )
            price_data.append(
                {
                    "Hour": hour,
                    "Bus": bus_name,
                    "PyPSA_Price": pypsa_price,
                    "Zap_Price": zap_price,
                }
            )
    if price_data:
        price_df = pd.DataFrame(price_data)
        price_df.to_csv(
            output_dir / f"{filename_prefix}_price_comparison.csv", index=False
        )

    # 3. Summary statistics
    summary_data = {
        "Metric": [
            "Total PyPSA Generation Capacity (MW)",
            "Total Zap Generation Capacity (MW)",
            "Total PyPSA Transmission Capacity (MW)",
            "Total Zap Transmission Capacity (MW)",
            "Mean PyPSA Price ($/MWh)",
            "Mean Zap Price ($/MWh)",
            "Max PyPSA Price ($/MWh)",
            "Max Zap Price ($/MWh)",
        ],
        "Value": [
            sum(
                v
                for k, v in pypsa_final.items()
                if k not in ["Transmission", "Storage"]
            ),
            sum(
                v for k, v in zap_final.items() if k not in ["Transmission", "Storage"]
            ),
            pypsa_final.get("Transmission", 0),
            zap_final.get("Transmission", 0),
            float(pypsa_prices.values.mean()),
            float(zap_prices.mean()),
            float(pypsa_prices.values.max()),
            float(zap_prices.max()),
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / f"{filename_prefix}_summary.csv", index=False)

    print(f"CSVs saved to: {output_dir}")


def get_total_load(devices: list) -> Optional[np.ndarray]:
    """
    Get total load profile from Zap devices.

    Parameters
    ----------
    devices : list
        List of Zap devices

    Returns
    -------
    np.ndarray or None
        Total load profile summed across all loads
    """
    for device in devices:
        if isinstance(device, Load):
            return device.load.sum(axis=0)
    return None


def plot_capacity_evolution(
    ax,
    history: dict,
    devices: list,
    pypsa_network=None,
    title: str = "Capacity Evolution During Optimization",
):
    """
    Plot the evolution of capacities during optimization iterations.

    Creates a line plot where x-axis is iteration number and y-axis is capacity,
    with each resource colored differently.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    history : dict
        Optimization history containing 'param' key with parameter trajectories.
        To enable this, pass `trackers=[..., PARAM]` to problem.solve().
    devices : list
        List of Zap devices (used to get resource names)
    pypsa_network : pypsa.Network, optional
        PyPSA network for carrier information and colors
    title : str
        Plot title
    """
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

    # Get carrier colors if available
    carrier_colors = {}
    if pypsa_network is not None and len(pypsa_network.carriers) > 0:
        carrier_colors = pypsa_network.carriers.color.to_dict()

    # Track device names to parameter mapping
    param_to_device_info = {}

    # Map parameter names to device info
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

    # Aggregate capacities by carrier
    carrier_capacity_history = {}  # carrier -> list of capacity sums per iteration

    for param_name, states in _extract_param_history(state_history):
        if param_name not in param_to_device_info:
            continue

        device_info = param_to_device_info[param_name]
        device = device_info["device"]

        # states shape: (num_iterations, num_devices) or (num_iterations, num_devices, 1)
        states = np.array(states)
        if states.ndim == 3:
            states = states.squeeze(-1)  # Remove trailing dimension if present

        # Aggregate by carrier
        for i in range(states.shape[1]):
            # Get device name
            if hasattr(device, "name") and device.name is not None:
                name = device.name[i]
            else:
                name = f"{param_name}_{i}"

            # Determine carrier/category
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

            # Initialize carrier history if needed
            if carrier not in carrier_capacity_history:
                carrier_capacity_history[carrier] = np.zeros(num_iterations)

            # Add this device's capacity to carrier total
            carrier_capacity_history[carrier] += states[:, i]

    # Plot aggregated carrier capacities
    lines_plotted = []
    labels_plotted = []

    for carrier, capacity_values in sorted(carrier_capacity_history.items()):
        # Get color for carrier
        color = carrier_colors.get(carrier)
        if color == "" or color is None:
            if carrier == "ac_line":
                color = "gray"
            elif carrier == "storage":
                color = "purple"
            else:
                color = None

        (line,) = ax.plot(
            iterations,
            capacity_values,
            label=carrier,
            color=color,
            alpha=0.8,
            linewidth=2.0,
        )
        lines_plotted.append(line)
        labels_plotted.append(carrier)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Capacity (MW)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add legend (limit to first 15 entries if too many)
    if len(labels_plotted) <= 15:
        ax.legend(fontsize=7, ncol=2, loc="best")
    else:
        # Show only first 15 with note about others
        ax.legend(
            lines_plotted[:15], labels_plotted[:15], fontsize=6, ncol=2, loc="best"
        )


def _extract_param_history(state_history: list) -> list:
    """
    Extract parameter history from state list.

    Parameters
    ----------
    state_history : list
        List of state dicts from optimization history

    Yields
    ------
    tuple
        (param_name, list of values across iterations)
    """
    if not state_history:
        return

    # Get parameter names from first state
    first_state = state_history[0]
    param_names = list(first_state.keys())

    for param_name in param_names:
        values = []
        for state in state_history:
            val = state.get(param_name)
            if val is not None:
                # Convert torch tensor to numpy if needed
                if hasattr(val, "numpy"):
                    val = val.numpy()
                elif hasattr(val, "detach"):
                    val = val.detach().numpy()
                values.append(val)
        if values:
            yield param_name, values
