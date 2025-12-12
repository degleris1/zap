import numpy as np
from typing import Tuple

from zap.network import PowerNetwork
from zap.devices import AbstractDevice, Generator, Load, DCLine, ACLine, Battery, Ground

TestCase = Tuple[PowerNetwork, list[AbstractDevice]]


def load_battery_network(
    mode="unique",
    battery_discharge_cost=0.0,
) -> TestCase:
    net = PowerNetwork(1)

    # One load
    loads = Load(
        name="load",
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        load=np.array([[10.0, 13.0, 80.0]]),
        linear_cost=np.array([100.0]),
    )

    # One battery
    batteries = Battery(
        name="battery",
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        power_capacity=np.array([10.0]),
        duration=np.array([4.0]),
        linear_cost=np.array([battery_discharge_cost]),
        capital_cost=np.array([15.0]),
    )

    # Two generators
    generators = Generator(
        name="generator",
        num_nodes=net.num_nodes,
        terminal=np.array([0, 0]),
        dynamic_capacity=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        linear_cost=np.array([[1.0, 1.2, 1.4], [30.0, 30.3, 30.6]]),
        nominal_capacity=np.array([15.0, 100.0]),
        capital_cost=np.array([25.0, 5.0]),
        emission_rates=np.array([0.0, 0.5]),
    )

    # One ground
    ground = Ground(num_nodes=net.num_nodes, terminal=np.array([0]))

    devices = [generators, loads, batteries, ground]

    return net, devices


def load_test_network(num_nodes: int = 7, line_type=ACLine) -> TestCase:
    net = PowerNetwork(num_nodes)

    generators = Generator(
        name="generator",
        num_nodes=num_nodes,
        terminal=np.array([0, 1, 3]),
        dynamic_capacity=np.array(
            [
                np.ones(4),  # Peaker
                [0.2, 1.0, 1.0, 0.3],  # Solar panel
                np.ones(4),  # CC Gas
            ]
        ),
        linear_cost=np.array([100.0, 0.5, 40.0]),
        nominal_capacity=np.array([100.0, 50.0, 15.0]),
        capital_cost=np.array([4.0, 10.0, 10.0]),
        emission_rates=np.array([800.0, 0.0, 500.0]),
    )

    loads = Load(
        name="load",
        num_nodes=num_nodes,
        terminal=np.array([0]),
        load=np.array([[30.0, 40.0, 45.0, 80.0]]),
        linear_cost=np.array([200.0]),
    )

    line_kwargs = {
        "name": "line",
        "num_nodes": num_nodes,
        "source_terminal": np.array([0, 1, 3]),
        "sink_terminal": np.array([1, 3, 0]),
        "capacity": np.ones(3),
        "nominal_capacity": np.array([45.0, 50.0, 11.0]),
        "linear_cost": 0.025 * np.ones(3),
        "capital_cost": np.array([100.0, 25.0, 30.0]),
    }

    if line_type == ACLine:
        lines = ACLine(susceptance=np.array([0.1, 0.05, 1.0]), **line_kwargs)
    else:  # line_type == DCLine
        lines = DCLine(**line_kwargs)

    batteries = Battery(
        name="battery",
        num_nodes=num_nodes,
        terminal=np.array([1]),
        power_capacity=np.array([5.0]),
        duration=np.array([4.0]),
        linear_cost=np.array([0.01]),
        capital_cost=np.array([10.0]),
    )

    devices = [generators, loads, lines, batteries]
    return net, devices


def load_garver_network(
    curtailment_cost=500.0,
    init_solar=600.0,
    line_slack=0.0,
    line_cost_scale=1.0,
) -> TestCase:
    """Classic 6-bus network with 3 generators and 5 loads, based on Garver 1970.

    The network is augmented with emissions data, fuel costs, and generator capital costs.

    See Romero et al 2002, **Test systems and mathematical models for transmission network expansion
    planning**, for more details.

        https://digital-library.theiet.org/content/journals/10.1049/ip-gtd_20020026
    """

    net = PowerNetwork(6)

    loads = Load(
        name="load",
        num_nodes=net.num_nodes,
        terminal=np.array([0, 1, 2, 3, 4]),
        load=np.array([80.0, 240.0, 40.0, 160.0, 40.0]),
        linear_cost=curtailment_cost * np.ones(5),
    )

    # Three generators
    # A (gas) peaker, a solar farm, and a (coal) base load plant
    generators = Generator(
        name="generator",
        num_nodes=net.num_nodes,
        terminal=np.array([0, 2, 5]),
        nominal_capacity=np.array([150.0, 360.0, init_solar]),
        dynamic_capacity=np.ones(3),
        linear_cost=np.array([22.4, 36.8, 0.1]),
        emission_rates=np.array([0.440, 1.03, 0.00]),
        capital_cost=np.array([5.0, 30.0, 15.0]),  # TODO - Pick better numbers
    )

    # Raw data
    wire_reactance = np.array(
        [
            0.4,
            0.38,
            0.6,
            0.2,
            0.68,
            0.2,
            0.4,
            0.31,
            0.3,
            0.59,
            0.2,
            0.48,
            0.63,
            0.3,
            0.61,
        ]
    )
    wire_capacity = np.array(
        [100.0, 100, 80, 100, 70, 100, 100, 100, 100, 82, 100, 100, 75, 100, 78]
    )
    wire_cost = np.array([40.0, 38, 60, 20, 68, 20, 40, 31, 30, 59, 20, 48, 63, 30, 61])
    num_wires = np.array([1.0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0])

    # Convert to per-MW values
    nominal_capacity = wire_capacity * num_wires  # Capacity in MW
    cost = wire_cost / wire_capacity  # Cost per MW
    susceptance = 1 / wire_reactance  # Susceptance in Siemens / wire
    susceptance = susceptance / wire_capacity  # Susceptance in Siemens / MW

    lines = ACLine(
        name="line",
        num_nodes=net.num_nodes,
        # Correct for silly 0-indexing
        source_terminal=np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5]) - 1,
        sink_terminal=np.array([2, 3, 4, 5, 6, 3, 4, 5, 6, 4, 5, 6, 5, 6, 6]) - 1,
        susceptance=susceptance,
        capacity=np.ones(num_wires.size),
        nominal_capacity=nominal_capacity,
        capital_cost=cost * line_cost_scale,
        slack=line_slack,
    )

    # Add some helpful metadata for debugging later
    lines.metadata = {
        "wire_reactance": wire_reactance,
        "wire_capacity": wire_capacity,
        "wire_cost": wire_cost,
    }

    devices = [generators, loads, lines]

    return net, devices
