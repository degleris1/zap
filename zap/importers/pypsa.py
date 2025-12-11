import pypsa
import numpy as np
import pandas as pd
from copy import deepcopy
import logging
from dataclasses import dataclass, field
from typing import Optional, Union, Literal, List, Tuple, Any

from zap.network import PowerNetwork
from zap.devices.injector import Generator, Load
from zap.devices.transporter import DCLine, ACLine
from zap.devices.storage_unit import StorageUnit
from zap.devices.store import Store

pd.set_option("future.no_silent_downcasting", True)
logger = logging.getLogger(__name__)


def get_annuity(num_years: float, discount_rate: float) -> float:
    return discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** num_years)


# Constants
HOURS_PER_YEAR = 365 * 24


@dataclass
class BatteryDefaults:
    discharge_cost: float = 0.0
    quadratic_discharge_cost: float = 0.0
    init_soc: float = 0.5
    final_soc: float = 0.5

    # Default battery cost parameters based on Danish Energy Agency data
    discount_rate: float = 0.07
    battery_duration: float = 25.0
    inverter_duration: float = 10.0

    @property
    def cost_per_mwh(self) -> float:
        return get_annuity(self.battery_duration, self.discount_rate) * 151_940.0

    @property
    def cost_per_mw(self) -> float:
        return get_annuity(self.inverter_duration, self.discount_rate) * 171_200.0

    # fom_per_mw: float = 52_845.0  # this value wasnt being used


@dataclass
class LoadDefaults:
    marginal_value: float = 10000.0
    quadratic_cost: float = 0.0


@dataclass
class TransmissionDefaults:
    ac_cost: float = 0.0


@dataclass
class DefaultConfig:
    """Default internal configuration values."""

    battery: BatteryDefaults = field(default_factory=BatteryDefaults)
    load: LoadDefaults = field(default_factory=LoadDefaults)
    transmission: TransmissionDefaults = field(default_factory=TransmissionDefaults)


# Create a global instance of default configurations
DEFAULT_CONFIG = DefaultConfig()


def parse_buses(net: pypsa.Network) -> Tuple[pd.Index, dict]:
    buses = net.buses.index
    buses_to_index = {bus: i for i, bus in enumerate(buses)}
    return buses, buses_to_index


def parse_network(net: pypsa.Network) -> PowerNetwork:
    buses, _ = parse_buses(net)
    return PowerNetwork(len(buses))


def get_active_assets(
    net: pypsa.Network, asset_type: str, snapshots: pd.date_range
) -> pd.Index:
    """Get the active assets based on build_year and lifetime.

    Filters out assets that have been retired based on build_year + lifetime.
    An asset is active if: retirement_year > snapshot_year (or lifetime is infinite).
    """
    import numpy as np

    if net.investment_periods.size > 0:
        snapshot_year = snapshots.get_level_values(0).unique()[0]
        # Use PyPSA's method first
        active_from_pypsa = net.get_active_assets(asset_type, snapshot_year)

        # Double-check by manually filtering based on build_year + lifetime
        # (PyPSA's method may not always work correctly depending on version/config)
        df = net.df(asset_type)
        if "build_year" in df.columns and "lifetime" in df.columns:
            retirement_year = df["build_year"] + df["lifetime"]
            # Asset is active if it hasn't retired yet (retirement_year > snapshot_year)
            # or if it has infinite lifetime
            is_active = (retirement_year > snapshot_year) | np.isinf(df["lifetime"])
            active_from_lifetime = df.index[is_active]

            # Return the intersection (most conservative)
            # Convert to Index if needed for intersection operation
            active_pypsa_idx = (
                active_from_pypsa
                if isinstance(active_from_pypsa, pd.Index)
                else active_from_pypsa.index
            )
            return active_pypsa_idx.intersection(active_from_lifetime)
        else:
            # No lifetime data, trust PyPSA's method
            return active_from_pypsa
    else:
        return net.df(asset_type).index


def parse_generators(
    pnet: pypsa.Network,
    snapshots: pd.DatetimeIndex,
    rng: np.random.Generator,
    generator_cost_perturbation: float,
    expand_empty_generators: float,
    scale_generator_capacity_factor: float,
    carbon_tax: float,
) -> Generator:
    buses, buses_to_index = parse_buses(pnet)

    active_generator_mask = get_active_assets(pnet, "Generator", snapshots)
    generators = deepcopy(pnet.generators.loc[active_generator_mask])

    terminals = generators.bus.replace(buses_to_index).values.astype(int)

    # Get dynamic data
    dynamic_capacities = (
        pnet.get_switchable_as_dense("Generator", "p_max_pu", snapshots)
        .T.loc[active_generator_mask]
        .values
    )
    dynamic_capacities_min = (
        pnet.get_switchable_as_dense("Generator", "p_min_pu", snapshots)
        .T.loc[active_generator_mask]
        .values
    )
    if dynamic_capacities_min.sum() > 0:
        logger.warning(
            "Some generators have minimum dynamic capacities. This is not yet supported."
        )
    dynamic_costs = (
        pnet.get_switchable_as_dense("Generator", "marginal_cost", snapshots)
        .T.loc[active_generator_mask]
        .values
    )
    dynamic_costs += generator_cost_perturbation * rng.random(
        dynamic_costs.shape
    )  # Perturb costs

    # Build nominal capacities
    # For active non-extendable generators: set min = max = p_nom (fixed capacity)
    non_ext_generators_mask = pnet.generators.index[
        ~pnet.generators.p_nom_extendable
        & pnet.generators.index.isin(active_generator_mask)
    ]
    if not non_ext_generators_mask.empty:
        generators.loc[non_ext_generators_mask, "p_nom_min"] = generators.loc[
            non_ext_generators_mask, "p_nom"
        ]
        generators.loc[non_ext_generators_mask, "p_nom_max"] = generators.loc[
            non_ext_generators_mask, "p_nom"
        ]

    nominal_capacities = generators.p_nom.values
    min_nominal_capacities = generators.p_nom_min.values
    max_nominal_capacities = generators.p_nom_max.values

    # Add emissions rates
    efficiency = generators.efficiency.values
    fuel_emissions_rate = pnet.carriers.loc[
        generators["carrier"].values
    ].co2_emissions.values
    plant_emissions_rate = fuel_emissions_rate / efficiency
    fuel_type = generators.carrier.values

    nominal_capacities = np.maximum(expand_empty_generators, nominal_capacities)

    # sign = generators.sign  # todo Integrate sign for scaling

    dev = Generator(
        num_nodes=len(buses),
        name=generators.index,
        terminal=terminals,
        nominal_capacity=nominal_capacities,
        dynamic_capacity=dynamic_capacities * scale_generator_capacity_factor,
        linear_cost=dynamic_costs + carbon_tax * plant_emissions_rate.reshape(-1, 1),
        capital_cost=generators.capital_cost.values,
        min_nominal_capacity=min_nominal_capacities,
        max_nominal_capacity=max_nominal_capacities,
        emission_rates=plant_emissions_rate,
    )
    dev.fuel_type = fuel_type

    return dev


def parse_loads(
    net: pypsa.Network,
    snapshots: pd.DatetimeIndex,
    rng: np.random.Generator,
    load_cost_perturbation: float,
    scale_load: float,
    defaults: LoadDefaults,
) -> Load:
    buses, buses_to_index = parse_buses(net)

    # Get active loads
    active_load_mask = get_active_assets(net, "Load", snapshots)
    loads = deepcopy(net.loads.loc[active_load_mask])

    terminals = loads.bus.replace(buses_to_index).values.astype(int)
    load = (
        net.get_switchable_as_dense("Load", "p_set", snapshots)
        .T.loc[active_load_mask]
        .values
    )

    # Build and perturb costs
    load_cost = defaults.marginal_value * np.ones(net.loads.shape[0])
    load_cost += load_cost_perturbation * rng.random(load_cost.shape)

    # Add quadratic term
    if defaults.quadratic_cost == 0.0:
        quadratic_cost = None
    else:
        quadratic_cost = defaults.quadratic_cost * np.ones_like(load_cost)

    return Load(
        num_nodes=len(buses),
        name=loads.index,
        terminal=terminals,
        load=load * scale_load,
        linear_cost=load_cost,
        quadratic_cost=quadratic_cost,
    )


def get_source_sinks(
    df: pd.DataFrame, buses_to_index: dict
) -> Tuple[np.ndarray, np.ndarray]:
    sources = df.bus0.replace(buses_to_index).values.astype(int)
    sinks = df.bus1.replace(buses_to_index).values.astype(int)
    return sources, sinks


def parse_dc_lines(
    net: pypsa.Network, snapshots: pd.DatetimeIndex, scale_line_capacity_factor: float
) -> DCLine:
    buses, buses_to_index = parse_buses(net)

    # Filter DC lines
    active_links_mask = get_active_assets(net, "Link", snapshots)
    links = deepcopy(net.links.loc[active_links_mask])

    # For non-extendable links: set min = max = p_nom (fixed capacity)
    non_ext_links_mask = net.links.index[
        ~net.links.p_nom_extendable & active_links_mask
    ]
    if not non_ext_links_mask.empty:
        links.loc[non_ext_links_mask, "p_nom_min"] = links.loc[
            non_ext_links_mask, "p_nom"
        ]
        links.loc[non_ext_links_mask, "p_nom_max"] = links.loc[
            non_ext_links_mask, "p_nom"
        ]

    # For extendable links: keep existing p_nom as initial capacity
    # Investment cost will be calculated as capital_cost * (capacity - p_nom_min)
    net.links.index[net.links.p_nom_extendable & active_links_mask]
    # Note: We keep p_nom as-is to preserve existing capacity

    # links = net.links[net.links.carrier == "DC"] # remove since we want to model multi-carrier networks
    sources, sinks = get_source_sinks(links, buses_to_index)

    # Get dynamic values
    dynamic_max_capacity = (
        net.get_switchable_as_dense("Link", "p_max_pu", snapshots)
        .T.loc[active_links_mask]
        .values
    )
    dynamic_min_capacity = (
        net.get_switchable_as_dense("Link", "p_min_pu", snapshots)
        .T.loc[active_links_mask]
        .values
    )
    if dynamic_min_capacity.sum() > 0:
        logger.warning(
            "Some DC lines have a minimum dynamic capacity. This is not yet supported."
        )
    dynamic_costs = (
        net.get_switchable_as_dense("Link", "marginal_cost", snapshots)
        .T.loc[active_links_mask]
        .values
    )

    return DCLine(
        num_nodes=len(buses),
        name=links.index,
        source_terminal=sources,
        sink_terminal=sinks,
        capacity=dynamic_max_capacity * scale_line_capacity_factor,
        nominal_capacity=links.p_nom.values,
        capital_cost=links.capital_cost.values,
        linear_cost=dynamic_costs,
        min_nominal_capacity=links.p_nom_min.values,
        max_nominal_capacity=links.p_nom_max.values,
    )


def parse_ac_lines(
    net: pypsa.Network,
    dates: pd.DatetimeIndex,
    susceptance_unit: Union[float, Literal["auto"]],
    scale_line_capacity_factor: float,
    b_factor: float,
    defaults: TransmissionDefaults,
) -> ACLine:
    buses, buses_to_index = parse_buses(net)

    active_lines_mask = get_active_assets(net, "Line", dates)
    lines = deepcopy(net.lines.loc[active_lines_mask])

    # Filter lines with infinite reactance
    lines = lines[~np.isinf(lines.x)]

    # For non-extendable lines: set min = max = s_nom (fixed capacity)
    non_ext_lines_mask = net.lines.index[
        ~net.lines.s_nom_extendable & active_lines_mask
    ]
    if not non_ext_lines_mask.empty:
        lines.loc[non_ext_lines_mask, "s_nom_min"] = lines.loc[
            non_ext_lines_mask, "s_nom"
        ]
        lines.loc[non_ext_lines_mask, "s_nom_max"] = lines.loc[
            non_ext_lines_mask, "s_nom"
        ]

    # For extendable lines: keep existing s_nom as initial capacity
    # Investment cost will be calculated as capital_cost * (capacity - s_nom_min)
    net.lines.index[net.lines.s_nom_extendable & active_lines_mask]
    # Note: We keep s_nom as-is to preserve existing capacity

    sources, sinks = get_source_sinks(lines, buses_to_index)

    # Compute per-MW susceptance
    susceptance = 1 / lines.x.values
    susceptance = np.divide(susceptance, np.maximum(lines.s_nom.values, 1e-6))
    susceptance *= 1e3  # Convert to Kilosiemens / per-MW

    if susceptance_unit == "auto":
        susceptance /= np.median(susceptance)
    else:
        susceptance /= susceptance_unit

    susceptance *= b_factor

    return ACLine(
        num_nodes=len(buses),
        name=lines.index,
        source_terminal=sources,
        sink_terminal=sinks,
        susceptance=susceptance,
        capacity=lines.s_max_pu.values * scale_line_capacity_factor,
        nominal_capacity=lines.s_nom.values,
        linear_cost=defaults.ac_cost * np.ones(sources.size),
        capital_cost=lines.capital_cost.values,
        min_nominal_capacity=lines.s_nom_min.values,
        max_nominal_capacity=lines.s_nom_max.values,
    )


def parse_storage_units(
    net: pypsa.Network, snapshots: pd.DatetimeIndex, defaults: BatteryDefaults
) -> Optional[StorageUnit]:
    buses, buses_to_index = parse_buses(net)

    active_storage_mask = get_active_assets(net, "StorageUnit", snapshots)
    storage_units = deepcopy(net.storage_units.loc[active_storage_mask])

    terminals = storage_units.bus.replace(buses_to_index).values.astype(int)

    if defaults.quadratic_discharge_cost == 0.0:
        quadratic_cost = None
    else:
        quadratic_cost = defaults.quadratic_discharge_cost * np.ones(terminals.size)

    # For non-extendable storage units: set min = max = p_nom (fixed capacity)
    non_ext_storage_mask = net.storage_units.index[
        ~net.storage_units.p_nom_extendable & active_storage_mask
    ]
    if not non_ext_storage_mask.empty:
        storage_units.loc[non_ext_storage_mask, "p_nom_min"] = storage_units.loc[
            non_ext_storage_mask, "p_nom"
        ]
        storage_units.loc[non_ext_storage_mask, "p_nom_max"] = storage_units.loc[
            non_ext_storage_mask, "p_nom"
        ]

    # For extendable storage units: keep existing p_nom as initial capacity
    # Investment cost will be calculated as capital_cost * (capacity - p_nom_min)
    net.storage_units.index[net.storage_units.p_nom_extendable & active_storage_mask]
    # Note: We keep p_nom as-is to preserve existing capacity

    # Set SOC set-points
    initial_soc = (
        storage_units.state_of_charge_initial
        / (storage_units.p_nom * storage_units.max_hours)
    ).fillna(defaults.init_soc)
    final_soc = initial_soc * storage_units.cyclic_state_of_charge

    return StorageUnit(
        num_nodes=len(buses),
        name=storage_units.index,
        terminal=terminals,
        power_capacity=storage_units.p_nom.values,
        duration=storage_units.max_hours.values,
        charge_efficiency=storage_units.efficiency_dispatch.values,
        linear_cost=defaults.discharge_cost * np.ones(terminals.size),
        quadratic_cost=quadratic_cost,
        capital_cost=storage_units.capital_cost.values,
        initial_soc=initial_soc.values,
        final_soc=final_soc.values,
    )


def parse_stores(
    net: pypsa.Network, dates: pd.DatetimeIndex, defaults: BatteryDefaults
) -> Optional[Store]:
    buses, buses_to_index = parse_buses(net)

    active_stores_mask = get_active_assets(net, "Store", dates)
    stores = deepcopy(net.stores.loc[active_stores_mask])

    terminals = stores.bus.replace(buses_to_index).values.astype(int)

    # Dynamic Values
    dynamic_capacity = (
        net.get_switchable_as_dense("Store", "e_max_pu", dates)
        .T.loc[active_stores_mask]
        .values
    )
    dynamic_min_capacity = (
        net.get_switchable_as_dense("Store", "e_min_pu", dates)
        .T.loc[active_stores_mask]
        .values
    )
    dynamic_costs = (
        net.get_switchable_as_dense("Store", "marginal_cost", dates)
        .T.loc[active_stores_mask]
        .values
    )

    # Set min and max capacities for non-extendable stores
    non_ext_stores_mask = net.stores.index[
        net.stores.e_nom_extendable & active_stores_mask
    ]
    if not non_ext_stores_mask.empty:
        stores.loc[non_ext_stores_mask, "e_nom_min"] = stores.loc[
            non_ext_stores_mask, "e_nom"
        ]
        stores.loc[non_ext_stores_mask, "e_nom_max"] = stores.loc[
            non_ext_stores_mask, "e_nom"
        ]

    if "marginal_cost_storage" not in stores.columns:
        # this attr was added in recent pypsa version
        stores["marginal_cost_storage"] = 0.0

    # Set SOC set-points
    initial_soc = (stores.e_initial / stores.e_nom).fillna(defaults.init_soc)
    final_soc = initial_soc * stores.e_cyclic

    return Store(
        num_nodes=len(buses),
        name=stores.index,
        terminal=terminals,
        nominal_energy_capacity=stores.e_nom.values,
        min_energy_capacity_availability=dynamic_min_capacity,
        max_energy_capacity_availability=dynamic_capacity,
        standing_loss=stores.standing_loss.values,
        initial_soc=initial_soc.values,
        final_soc=final_soc.values,
        linear_cost=dynamic_costs,
        quadratic_cost=stores.marginal_cost_quadratic.values,
        capital_cost=stores.values,
        min_nominal_energy_capacity=stores.e_nom_min.values,
        max_nominal_energy_capacity=stores.e_nom_max.values,
    )


def drop_empty_components(
    net: pypsa.Network, drop_empty_devices: Union[bool, List[str]]
):
    logger.warning(
        "Dropping empty devices. This feature will limit round-trip with PyPSA."
    )
    if isinstance(drop_empty_devices, bool):
        device_types = ["Generator", "Link", "Line", "StorageUnit", "Store"]
    else:
        device_types = drop_empty_devices

    for device_type in device_types:
        if device_type in net.passive_branch_components:
            attr = "s_nom"
        elif device_type == "Store":
            attr = "e_nom"
        else:
            attr = "p_nom"

        # Drop devices with zero capacity
        zero_capacity_devices = net.df(device_type)[net.df(device_type)[attr] == 0]
        if zero_capacity_devices.empty:
            continue
        logger.info(
            f"Dropping {len(zero_capacity_devices)} {device_type} with zero capacity."
        )
        net.mremove(device_type, zero_capacity_devices.index)


def load_pypsa_network(
    pnet: pypsa.Network,
    snapshots: Optional[pd.DatetimeIndex] = None,
    seed: int = 0,
    generator_cost_perturbation: float = 0.0,
    load_cost_perturbation: float = 0.0,
    drop_empty_devices: Union[bool, List[str]] = False,
    expand_empty_generators: float = 0.0,
    power_unit: float = 1.0,  # MW
    cost_unit: float = 1.0,  # $
    susceptance_unit: Union[float, Literal["auto"]] = "auto",
    scale_load: float = 1.0,
    scale_generator_capacity_factor: float = 1.0,
    scale_line_capacity_factor: float = 1.0,
    carbon_tax: float = 0.0,
    b_factor: float = 1.0,
    defaults: Optional[DefaultConfig] = None,
) -> Tuple[PowerNetwork, List[Any]]:
    if defaults is None:
        defaults = DEFAULT_CONFIG

    if snapshots is None:
        snapshots = pnet.snapshots

    if snapshots.nlevels > 1:
        assert (
            snapshots.get_level_values(0).unique().size == 1
        ), "Network must be solved for a single investment period."

    pnet = deepcopy(pnet)
    network = PowerNetwork(len(pnet.buses))
    rng = np.random.default_rng(seed)

    if (
        (isinstance(drop_empty_devices, list) and "Generator" in drop_empty_devices)
        or (isinstance(drop_empty_devices, bool) and drop_empty_devices)
    ) and expand_empty_generators > 0:
        raise ValueError("Cannot both drop and expand empty Generator.")

    if drop_empty_devices:
        drop_empty_components(pnet, drop_empty_devices)

    generator = parse_generators(
        pnet,
        snapshots,
        rng,
        generator_cost_perturbation,
        expand_empty_generators,
        scale_generator_capacity_factor,
        carbon_tax,
    )

    load = parse_loads(
        pnet, snapshots, rng, load_cost_perturbation, scale_load, defaults.load
    )

    dc_line = parse_dc_lines(pnet, snapshots, scale_line_capacity_factor)

    ac_line = parse_ac_lines(
        pnet,
        snapshots,
        susceptance_unit,
        scale_line_capacity_factor,
        b_factor,
        defaults.transmission,
    )

    storage_unit = parse_storage_units(pnet, snapshots, defaults.battery)

    store = parse_stores(pnet, snapshots, defaults.battery)

    devices = []
    for device in [generator, load, dc_line, ac_line, storage_unit, store]:
        if device is not None and device.num_devices > 0:
            logger.info(
                f"Importing {device.__class__.__name__} with {device.num_devices} devices."
            )
            devices.append(device)

    for device in devices:
        device.scale_costs(cost_unit)
        device.scale_power(power_unit)

    return network, devices
