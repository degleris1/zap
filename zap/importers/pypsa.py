import pypsa
import numpy as np
import pandas as pd
from copy import deepcopy

from zap.network import PowerNetwork
from zap.devices.injector import Generator, Load
from zap.devices.transporter import DCLine, ACLine
from zap.devices.store import Battery

pd.set_option("future.no_silent_downcasting", True)

HOURS_PER_YEAR = 365 * 24


def parse_buses(net: pypsa.Network):
    buses = net.buses.loc[net.buses.carrier != "battery"].index
    buses_to_index = {bus: i for i, bus in enumerate(buses)}
    return buses, buses_to_index


def parse_network(net: pypsa.Network):
    buses, buses_to_index = parse_buses(net)
    return PowerNetwork(len(buses))


def build_dynamic(device, device_t, key, dates):
    # Load dynamic data
    dyn = device_t[key].loc[dates].copy()
    dyn_index = [device.index.get_loc(c) for c in dyn.columns]

    # Replace static values with their dynamic counterparts
    dynamic_values = np.outer(device[key].values, np.ones(len(dates)))
    dynamic_values[dyn_index, :] = dyn.values.T
    return dynamic_values


def parse_generators(
    net: pypsa.Network,
    dates,
    rng: np.random.Generator,
    *,
    generator_cost_perturbation,
    expand_empty_generators,
    drop_empty_generators,
    scale_generator_capacity_factor,
    carbon_tax,
):
    if drop_empty_generators:
        assert expand_empty_generators == 0.0

    buses, buses_to_index = parse_buses(net)
    terminals = net.generators.bus.replace(buses_to_index).values.astype(int)

    # Build dynamic capacities
    dynamic_capacities = build_dynamic(net.generators, net.generators_t, "p_max_pu", dates)
    dynamic_costs = build_dynamic(net.generators, net.generators_t, "marginal_cost", dates)

    # Perturb costs
    dynamic_costs += generator_cost_perturbation * rng.random(dynamic_costs.shape)

    # Build nominal capacities
    nominal_capacities = net.generators.p_nom.values
    min_nominal_capacities = net.generators.p_nom.values
    max_nominal_capacities = net.generators.p_nom_max.values

    # Build capital costs
    capital_costs = net.generators.capital_cost.values * (len(dates) / HOURS_PER_YEAR)

    # Add emissions rates
    efficiency = net.generators.efficiency.values
    fuel_rates = net.carriers.loc[net.generators["carrier"].values].co2_emissions.values
    emissions = fuel_rates / efficiency

    if drop_empty_generators:
        mask = nominal_capacities > 0
        terminals = terminals[mask]
        dynamic_capacities = dynamic_capacities[mask]
        dynamic_costs = dynamic_costs[mask]
        nominal_capacities = nominal_capacities[mask]
        capital_costs = capital_costs[mask]
        min_nominal_capacities = min_nominal_capacities[mask]
        max_nominal_capacities = max_nominal_capacities[mask]
        emissions = emissions[mask]

    else:
        nominal_capacities = np.maximum(expand_empty_generators, nominal_capacities)

    return Generator(
        num_nodes=len(buses),
        terminal=terminals,
        nominal_capacity=nominal_capacities,
        dynamic_capacity=dynamic_capacities * scale_generator_capacity_factor,
        linear_cost=dynamic_costs + carbon_tax * emissions.reshape(-1, 1),
        capital_cost=capital_costs,
        min_nominal_capacity=min_nominal_capacities,
        max_nominal_capacity=max_nominal_capacities,
        emission_rates=emissions,
    )


def parse_loads(
    net: pypsa.Network,
    dates,
    rng: np.random.Generator,
    *,
    load_cost_perturbation,
    marginal_load_value,
    scale_load,
):
    buses, buses_to_index = parse_buses(net)
    terminals = net.loads.bus.replace(buses_to_index).values.astype(int)
    load = build_dynamic(net.loads, net.loads_t, "p_set", dates)

    # Build and perturb costs
    load_cost = marginal_load_value * np.ones(len(buses))
    load_cost += load_cost_perturbation * rng.random(load_cost.shape)

    return Load(
        num_nodes=len(buses),
        terminal=terminals,
        load=load * scale_load,
        linear_cost=load_cost,
    )


def get_source_sinks(df: pd.DataFrame, buses_to_index):
    sources = df.bus0.replace(buses_to_index).values.astype(int)
    sinks = df.bus1.replace(buses_to_index).values.astype(int)

    return sources, sinks


def parse_dc_lines(net: pypsa.Network, dates, *, scale_line_capacity_factor):
    buses, buses_to_index = parse_buses(net)

    links = net.links[net.links.carrier == "DC"]
    sources, sinks = get_source_sinks(links, buses_to_index)

    return DCLine(
        num_nodes=len(buses),
        source_terminal=sources,
        sink_terminal=sinks,
        capacity=links.p_max_pu.values * scale_line_capacity_factor,
        nominal_capacity=links.p_nom.values,
        capital_cost=links.capital_cost.values * (len(dates) / HOURS_PER_YEAR),
    )


def parse_ac_lines(
    net: pypsa.Network, dates, *, ac_transmission_cost, susceptance_unit, scale_line_capacity_factor
):
    buses, buses_to_index = parse_buses(net)

    sources, sinks = get_source_sinks(net.lines, buses_to_index)

    # Compute per-MW susceptance
    susceptance = 1 / net.lines.x.values
    susceptance = np.divide(susceptance, net.lines.s_nom.values)
    susceptance *= 1e3  # Convert to Kilosiemens / per-MW

    if susceptance_unit == "auto":
        susceptance /= np.median(susceptance)
    else:
        susceptance /= susceptance_unit

    return ACLine(
        num_nodes=len(buses),
        source_terminal=sources,
        sink_terminal=sinks,
        susceptance=susceptance,
        capacity=net.lines.s_max_pu.values * scale_line_capacity_factor,
        nominal_capacity=net.lines.s_nom.values,
        linear_cost=ac_transmission_cost * np.ones(sources.size),
        capital_cost=net.lines.capital_cost.values * (len(dates) / HOURS_PER_YEAR),
    )


def parse_batteries(net: pypsa.Network, dates, *, battery_discharge_cost):
    buses, buses_to_index = parse_buses(net)
    terminals = net.storage_units.bus.replace(buses_to_index).values.astype(int)

    return Battery(
        num_nodes=len(buses),
        terminal=terminals,
        power_capacity=net.storage_units.p_nom.values,
        duration=net.storage_units.max_hours.values,
        charge_efficiency=net.storage_units.efficiency_dispatch.values,
        linear_cost=battery_discharge_cost * np.ones(terminals.size),
        capital_cost=net.storage_units.capital_cost.values * (len(dates) / HOURS_PER_YEAR),
    )


def load_pypsa_network(
    net: pypsa.Network,
    dates,
    seed=0,
    battery_discharge_cost=0.0,
    ac_transmission_cost=0.0,
    generator_cost_perturbation=0.0,
    load_cost_perturbation=0.0,
    marginal_load_value=1000.0,
    drop_empty_generators=True,
    expand_empty_generators=0.0,
    power_unit=1.0,  # MW
    cost_unit=1.0,  # $
    susceptance_unit="auto",
    scale_load=1.0,
    scale_generator_capacity_factor=1.0,
    scale_line_capacity_factor=1.0,
    carbon_tax=0.0,
):
    net = deepcopy(net)
    network = PowerNetwork(len(net.buses))

    rng = np.random.default_rng(seed)

    devices = [
        parse_generators(
            net,
            dates,
            rng,
            generator_cost_perturbation=generator_cost_perturbation,
            expand_empty_generators=expand_empty_generators,
            drop_empty_generators=drop_empty_generators,
            scale_generator_capacity_factor=scale_generator_capacity_factor,
            carbon_tax=carbon_tax,
        ),
        parse_loads(
            net,
            dates,
            rng,
            load_cost_perturbation=load_cost_perturbation,
            marginal_load_value=marginal_load_value,
            scale_load=scale_load,
        ),
        parse_dc_lines(net, dates, scale_line_capacity_factor=scale_line_capacity_factor),
        parse_ac_lines(
            net,
            dates,
            ac_transmission_cost=ac_transmission_cost,
            susceptance_unit=susceptance_unit,
            scale_line_capacity_factor=scale_line_capacity_factor,
        ),
        parse_batteries(net, dates, battery_discharge_cost=battery_discharge_cost),
    ]

    for d in devices:
        d.scale_costs(cost_unit)
        d.scale_power(power_unit)

    return network, devices
