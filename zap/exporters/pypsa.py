import pypsa
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

import logging
from zap.network import PowerNetwork, DispatchOutcome
from zap.devices.injector import Generator, Load
from zap.devices.transporter import DCLine, ACLine
from zap.devices.storage_unit import StorageUnit
from zap.devices.store import Store
from zap.importers.pypsa import HOURS_PER_YEAR

logger = logging.getLogger(__name__)


def create_pypsa_buses(network: PowerNetwork) -> pd.DataFrame:
    """Create PyPSA bus dataframe from zap network."""
    n_buses = network.num_nodes
    buses = pd.DataFrame(
        index=[f"bus{i}" for i in range(n_buses)],
        columns=["v_nom", "x", "y", "carrier"],
    )
    buses["v_nom"] = 1.0  # Default voltage
    buses["x"] = 0.0
    buses["y"] = 0.0
    buses["carrier"] = "AC"
    return buses


def export_generators(device: Generator, snapshots: pd.DatetimeIndex) -> Dict[str, Any]:
    """Convert zap Generator to PyPSA generator component."""
    n_snapshots = len(snapshots)

    # Create static generator dataframe
    generators = pd.DataFrame(
        index=device.name,
        columns=[
            "bus",
            "p_nom",
            "p_nom_extendable",
            "p_nom_min",
            "p_nom_max",
            "capital_cost",
            "efficiency",
            "carrier",
        ],
    )

    # Map to bus names
    generators["bus"] = [f"bus{t}" for t in device.terminal]
    generators["p_nom"] = device.nominal_capacity
    generators["p_nom_extendable"] = np.logical_or(
        device.min_nominal_capacity != device.nominal_capacity,
        device.max_nominal_capacity != device.nominal_capacity,
    )
    generators["p_nom_min"] = device.min_nominal_capacity
    generators["p_nom_max"] = device.max_nominal_capacity
    generators["capital_cost"] = device.capital_cost * (HOURS_PER_YEAR / n_snapshots)

    # Set fuel type/carrier if available
    if hasattr(device, "fuel_type"):
        generators["carrier"] = device.fuel_type
    else:
        generators["carrier"] = "unknown"

    # Create time-varying data
    p_max_pu = pd.DataFrame(
        data=device.dynamic_capacity, index=device.name, columns=snapshots
    )

    marginal_cost = pd.DataFrame(
        data=device.linear_cost, index=device.name, columns=snapshots
    )

    return {
        "generators": generators,
        "generators_t": {"p_max_pu": p_max_pu, "marginal_cost": marginal_cost},
    }


def export_loads(device: Load, snapshots: pd.DatetimeIndex) -> Dict[str, Any]:
    """Convert zap Load to PyPSA load component."""
    # Create static load dataframe
    loads = pd.DataFrame(
        index=device.name,
        columns=["bus", "p_set"],
    )

    # Map to bus names
    loads["bus"] = [f"bus{t}" for t in device.terminal]
    loads["p_set"] = device.load[:, 0]  # Use first time period as default

    # Create time-varying data
    p_set = pd.DataFrame(data=device.load, index=device.name, columns=snapshots)

    return {"loads": loads, "loads_t": {"p_set": p_set}}


def export_dc_lines(device: DCLine, snapshots: pd.DatetimeIndex) -> Dict[str, Any]:
    """Convert zap DCLine to PyPSA link component."""
    n_snapshots = len(snapshots)

    # Create static links dataframe
    links = pd.DataFrame(
        index=device.name,
        columns=[
            "bus0",
            "bus1",
            "p_nom",
            "p_nom_extendable",
            "p_nom_min",
            "p_nom_max",
            "capital_cost",
            "efficiency",
            "carrier",
        ],
    )

    # Map to bus names
    links["bus0"] = [f"bus{t}" for t in device.source_terminal]
    links["bus1"] = [f"bus{t}" for t in device.sink_terminal]
    links["p_nom"] = device.nominal_capacity
    links["p_nom_extendable"] = np.logical_or(
        device.min_nominal_capacity != device.nominal_capacity,
        device.max_nominal_capacity != device.nominal_capacity,
    )
    links["p_nom_min"] = device.min_nominal_capacity
    links["p_nom_max"] = device.max_nominal_capacity
    links["capital_cost"] = device.capital_cost * (HOURS_PER_YEAR / n_snapshots)
    links["efficiency"] = 1.0  # Default efficiency
    links["carrier"] = "DC"

    # Create time-varying data
    p_max_pu = pd.DataFrame(data=device.capacity, index=device.name, columns=snapshots)

    marginal_cost = pd.DataFrame(
        data=device.linear_cost, index=device.name, columns=snapshots
    )

    return {
        "links": links,
        "links_t": {"p_max_pu": p_max_pu, "marginal_cost": marginal_cost},
    }


def export_ac_lines(
    device: ACLine, snapshots: pd.DatetimeIndex, susceptance_unit: float = 1.0
) -> Dict[str, Any]:
    """Convert zap ACLine to PyPSA line component."""
    n_snapshots = len(snapshots)

    # Create static lines dataframe
    lines = pd.DataFrame(
        index=device.name,
        columns=[
            "bus0",
            "bus1",
            "s_nom",
            "s_nom_extendable",
            "s_nom_min",
            "s_nom_max",
            "capital_cost",
            "x",
            "r",
            "s_max_pu",
        ],
    )

    # Map to bus names
    lines["bus0"] = [f"bus{t}" for t in device.source_terminal]
    lines["bus1"] = [f"bus{t}" for t in device.sink_terminal]
    lines["s_nom"] = device.nominal_capacity
    lines["s_nom_extendable"] = np.logical_or(
        device.min_nominal_capacity != device.nominal_capacity,
        device.max_nominal_capacity != device.nominal_capacity,
    )
    lines["s_nom_min"] = device.min_nominal_capacity
    lines["s_nom_max"] = device.max_nominal_capacity
    lines["capital_cost"] = device.capital_cost * (HOURS_PER_YEAR / n_snapshots)

    # Convert susceptance back to reactance
    susceptance_scaled = device.susceptance * susceptance_unit
    lines["x"] = 1 / (susceptance_scaled / 1e3)  # Convert from Kilosiemens/MW
    lines["r"] = 0.0  # Default resistance
    lines["s_max_pu"] = device.capacity

    return {
        "lines": lines,
    }


def export_storage_units(
    device: StorageUnit, snapshots: pd.DatetimeIndex
) -> Dict[str, Any]:
    """Convert zap StorageUnit to PyPSA storage_unit component."""
    n_snapshots = len(snapshots)

    # Create static storage_units dataframe
    storage_units = pd.DataFrame(
        index=device.name,
        columns=[
            "bus",
            "p_nom",
            "p_nom_extendable",
            "p_nom_min",
            "p_nom_max",
            "capital_cost",
            "max_hours",
            "efficiency_dispatch",
            "state_of_charge_initial",
            "cyclic_state_of_charge",
        ],
    )

    # Map to bus names
    storage_units["bus"] = [f"bus{t}" for t in device.terminal]
    storage_units["p_nom"] = device.power_capacity
    storage_units["p_nom_extendable"] = False  # Default
    storage_units["p_nom_min"] = device.power_capacity  # Default
    storage_units["p_nom_max"] = device.power_capacity  # Default
    storage_units["capital_cost"] = device.capital_cost * (HOURS_PER_YEAR / n_snapshots)
    storage_units["max_hours"] = device.duration
    storage_units["efficiency_dispatch"] = device.charge_efficiency

    # Compute SOC values
    storage_units["state_of_charge_initial"] = (
        device.initial_soc * device.power_capacity * device.duration
    )
    # storage_units["cyclic_state_of_charge"] =

    # Set cyclic state of charge
    storage_units["cyclic_state_of_charge"] = True

    result = {"storage_units": storage_units, "storage_units_t": {}}

    # Create time-varying marginal cost data if provided
    if hasattr(device, "linear_cost") and device.linear_cost is not None:
        if device.linear_cost.shape[1] == 1:
            storage_units["marginal_cost"] = device.linear_cost.reshape(-1)
        else:
            marginal_cost = pd.DataFrame(
                data=device.linear_cost, index=device.name, columns=snapshots
            )
            result["storage_units_t"]["marginal_cost"] = marginal_cost

    return result


def export_stores(device: Store, snapshots: pd.DatetimeIndex) -> Dict[str, Any]:
    """Convert zap Store to PyPSA store component."""
    n_snapshots = len(snapshots)

    # Create static stores dataframe
    stores = pd.DataFrame(
        index=device.name,
        columns=[
            "bus",
            "e_nom",
            "e_nom_extendable",
            "e_nom_min",
            "e_nom_max",
            "capital_cost",
            "standing_loss",
            "e_initial",
            "e_cyclic",
            "marginal_cost_quadratic",
        ],
    )

    # Map to bus names
    stores["bus"] = [f"bus{t}" for t in device.terminal]
    stores["e_nom"] = device.nominal_energy_capacity
    stores["e_nom_extendable"] = np.logical_or(
        device.min_nominal_energy_capacity != device.nominal_energy_capacity,
        device.max_nominal_energy_capacity != device.nominal_energy_capacity,
    )
    stores["e_nom_min"] = device.min_nominal_energy_capacity
    stores["e_nom_max"] = device.max_nominal_energy_capacity
    stores["capital_cost"] = device.capital_cost * (HOURS_PER_YEAR / n_snapshots)
    stores["standing_loss"] = device.standing_loss

    # Compute SOC values
    stores["e_initial"] = device.initial_soc * device.nominal_energy_capacity
    stores["e_cyclic"] = (device.final_soc / device.initial_soc).fillna(1.0)

    # Set quadratic costs if available
    if device.quadratic_cost is not None:
        stores["marginal_cost_quadratic"] = device.quadratic_cost
    else:
        stores["marginal_cost_quadratic"] = 0.0

    # Create time-varying data
    e_max_pu = pd.DataFrame(
        data=device.max_energy_capacity_availability,
        index=device.name,
        columns=snapshots,
    )

    e_min_pu = pd.DataFrame(
        data=device.min_energy_capacity_availability,
        index=device.name,
        columns=snapshots,
    )

    marginal_cost = pd.DataFrame(
        data=device.linear_cost, index=device.name, columns=snapshots
    )

    return {
        "stores": stores,
        "stores_t": {
            "e_max_pu": e_max_pu,
            "e_min_pu": e_min_pu,
            "marginal_cost": marginal_cost,
        },
    }


def export_to_pypsa(
    network: PowerNetwork,
    devices: List[Any],
    results: DispatchOutcome,
    snapshots: Optional[pd.DatetimeIndex] = None,
    power_unit: float = 1.0,  # MW
    cost_unit: float = 1.0,  # $
    susceptance_unit: float = 1.0,
) -> pypsa.Network:
    """
    Convert a zap network and devices to a PyPSA network.

    Parameters
    ----------
    network : PowerNetwork
        The zap network to export
    devices : List[Any]
        List of zap devices to export
    snapshots : pd.DatetimeIndex, optional
        Time snapshots to use, by default None which creates a simple index
    power_unit : float, optional
        Power unit scaling factor, by default 1.0 (MW)
    cost_unit : float, optional
        Cost unit scaling factor, by default 1.0 ($)
    susceptance_unit : float, optional
        Susceptance unit scaling factor, by default 1.0

    Returns
    -------
    pypsa.Network
        The exported PyPSA network
    """
    # Create empty PyPSA network
    n = pypsa.Network()

    # Create or use provided snapshots
    if snapshots is None:
        n_snapshots = 1
        snapshots = pd.date_range("2020-01-01", periods=n_snapshots, freq="h")
    n.set_snapshots(snapshots)

    # Create buses
    buses = create_pypsa_buses(network)
    n.import_components_from_dataframe(buses, "Bus")
    n.buses_t["marginal_price"] = pd.DataFrame(
        data=results.prices.T, columns=buses.index, index=snapshots
    )

    # Process each device type
    for device in devices:
        # Position in device list
        device_index = devices.index(device)
        # Undo any unit conversions
        if hasattr(device, "scale_power"):
            device.scale_power(1.0 / power_unit)
        if hasattr(device, "scale_costs"):
            device.scale_costs(1.0 / cost_unit)

        match device:
            case Generator():
                data = export_generators(device, snapshots)
                static_data = data["generators"].to_dict(orient="index")
                for name, params in static_data.items():
                    # Add component with only static parameters
                    n.add("Generator", name, **params)

                # Convert power results to a DataFrame with snapshots as index, generator names as columns
                p_values = results.power[device_index][0]
                p_df = pd.DataFrame(
                    data=p_values.T,  # Transpose to have snapshots as rows
                    index=snapshots,
                    columns=data["generators"].index,
                )
                data["generators_t"]["p"] = p_df.T

                # Set time-dependent data separately
                for attr, df in data["generators_t"].items():
                    for name in df.index:
                        if name in static_data:
                            n.generators_t[attr][name] = df.loc[name]

            case Load():
                data = export_loads(device, snapshots)
                static_data = data["loads"].to_dict(orient="index")
                for name, params in static_data.items():
                    # Add component with only static parameters
                    n.add("Load", name, **params)

                # Convert power results to a DataFrame with snapshots as index, names as columns
                p_values = results.power[device_index][0]
                p_df = pd.DataFrame(
                    data=p_values.T, index=snapshots, columns=data["loads"].index
                )
                data["loads_t"]["p"] = p_df.T

                # Set time-dependent data separately
                for attr, df in data["loads_t"].items():
                    for name in df.index:
                        if name in static_data:
                            n.loads_t[attr][name] = df.loc[name]

            case DCLine():
                data = export_dc_lines(device, snapshots)
                static_data = data["links"].to_dict(orient="index")
                for name, params in static_data.items():
                    # Add component with only static parameters
                    n.add("Link", name, **params)

                # Convert power results to DataFrames with snapshots as index, link names as columns
                p0_values = results.power[device_index][0]
                p1_values = results.power[device_index][1]

                # Create DataFrames with link names as columns, snapshots as index
                p0_df = pd.DataFrame(
                    data=p0_values.T,  # Transpose to have snapshots as rows
                    index=snapshots,
                    columns=data["links"].index,
                )
                p1_df = pd.DataFrame(
                    data=p1_values.T,  # Transpose to have snapshots as rows
                    index=snapshots,
                    columns=data["links"].index,
                )

                data["links_t"]["p0"] = p0_df.T
                data["links_t"]["p1"] = p1_df.T

                # Set time-dependent data separately
                for attr, df in data["links_t"].items():
                    for name in df.index:
                        if name in static_data:
                            n.links_t[attr][name] = df.loc[name]

            case ACLine():
                data = export_ac_lines(device, snapshots, susceptance_unit)
                static_data = data["lines"].to_dict(orient="index")
                for name, params in static_data.items():
                    n.add("Line", name, **params)

                # Convert power results to DataFrames with snapshots as index, line names as columns
                p0_values = results.power[device_index][0]
                p1_values = results.power[device_index][1]

                # Create DataFrames with line names as columns, snapshots as index
                p0_df = pd.DataFrame(
                    data=p0_values.T,  # Transpose to have snapshots as rows
                    index=snapshots,
                    columns=data["lines"].index,
                )
                p1_df = pd.DataFrame(
                    data=p1_values.T,  # Transpose to have snapshots as rows
                    index=snapshots,
                    columns=data["lines"].index,
                )

                data["lines_t"] = {"p0": p0_df.T, "p1": p1_df.T}

                # Set time-dependent data separately
                for attr, df in data["lines_t"].items():
                    for name in df.index:
                        if name in static_data:
                            n.lines_t[attr][name] = df.loc[name]

            case StorageUnit():
                data = export_storage_units(device, snapshots)
                static_data = data["storage_units"].to_dict(orient="index")
                for name, params in static_data.items():
                    # Add component with only static parameters
                    n.add("StorageUnit", name, **params)

                # Convert power results to a DataFrame with snapshots as index, names as columns
                p_values = results.power[device_index][0]
                p_df = pd.DataFrame(
                    data=p_values.T,  # Transpose to have snapshots as rows
                    index=snapshots,
                    columns=data["storage_units"].index,
                )
                data["storage_units_t"]["p"] = p_df

                # Set time-dependent data separately
                for attr, df in data["storage_units_t"].items():
                    for name in df.index:
                        if name in static_data:
                            n.storage_units_t[attr][name] = df.loc[name]

            case Store():
                data = export_stores(device, snapshots)
                static_data = data["stores"].to_dict(orient="index")
                for name, params in static_data.items():
                    # Add component with only static parameters
                    n.add("Store", name, **params)

                # Set time-dependent data separately
                for attr, df in data["stores_t"].items():
                    for name in df.index:
                        if name in static_data:
                            n.stores_t[attr][name] = df.loc[name]

            case _:
                logger.warning(f"Unsupported device type: {type(device)}")

    logger.info(
        f"Exported network with {len(n.buses)} buses and {len(snapshots)} snapshots"
    )
    return n
