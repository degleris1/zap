import pyomo.environ as pyo
import numpy as np

from zap import PowerNetwork
from zap.devices import AbstractDevice
from .devices import convert_to_pyo


def setup_pyomo_model(
    net: PowerNetwork,
    devices: list[AbstractDevice],
    time_horizon: int,
    model: pyo.ConcreteModel = None,
):
    if model is None:
        model = pyo.ConcreteModel()

    pyo_devices = [convert_to_pyo(d) for d in devices]

    # Indices
    num_devices = len(devices)
    node_index = pyo.RangeSet(0, net.num_nodes - 1)
    time_index = pyo.RangeSet(0, time_horizon - 1)
    group_map = [get_node_groups(d) for d in devices]

    # Model setup
    model = pyo.ConcreteModel()
    model.time_horizon = time_horizon
    model.device = pyo.Block(range(num_devices))
    model.time_index = time_index
    model.node_index = node_index

    # Global stuff
    model.global_angle = pyo.Var(node_index, time_index)

    for d, (dev, group) in enumerate(zip(devices, group_map)):
        # Device-level setup
        nt, nd = dev.num_terminals_per_device, dev.num_devices
        dev_model = model.device[d]

        dev_model.dev_index = pyo.RangeSet(0, nd - 1)
        dev_model.terminal = pyo.Block(range(nt))

        for tau, tgroup in enumerate(group):
            if nt == 1:
                nodes = dev.terminals
            else:
                nodes = dev.terminals[:, tau]

            initialize_terminal_model(dev, dev_model.terminal[tau], nodes, tgroup)

        # Aggregate net power
        dev_model.net_power = pyo.Expression(
            node_index,
            time_index,
            rule=lambda block, n, t: sum(block.terminal[tau].net_power[n, t] for tau in range(nt)),
        )

        # Add local variables
        pyo_devices[d].add_local_variables(dev_model)

        # Add local constraints
        pyo_devices[d].add_local_constraints(dev_model)

        # Add local objective
        pyo_devices[d].add_objective(dev_model)

    # Power balance constraint
    model.power_balance = pyo.Constraint(
        node_index,
        time_index,
        rule=lambda model, n, t: 0.0
        == sum(model.device[d].net_power[n, t] for d in range(num_devices)),
    )

    # Objective
    model.objective = pyo.Objective(
        expr=sum(model.device[d].operation_cost for d in range(num_devices)),
        sense=pyo.minimize,
    )

    return model


def initialize_terminal_model(device: AbstractDevice, block: pyo.Block, nodes, tgroup):
    """
    Initialize a Pyomo model for a single terminal of a device.
    """
    time_index = block.model().time_index
    node_index = block.model().node_index
    dev_index = block.parent_block().dev_index

    # dt_model = dev_model.terminal[tau]

    block.power = pyo.Var(dev_index, block.model().time_index)
    block.net_power = pyo.Expression(
        node_index,
        time_index,
        rule=lambda block, n, t: sum(block.power[k, t] for k in tgroup[n]),
    )

    if device.is_ac:
        block.angle = pyo.Var(dev_index, time_index)
        block.phase_consistency = pyo.Constraint(
            dev_index,
            time_index,
            rule=lambda block, k, t: block.model().global_angle[nodes[k], t] == block.angle[k, t],
        )
    else:
        block.angle = None
        block.phase_consistency = None


def parse_output(devices: list[AbstractDevice], model: pyo.Model):
    # np.array([[model.device[0].terminal[0].power[k, t].value for t in model.time_index] for k in model.device[0].dev_index])
    power = [
        [
            np.array(
                [
                    [model.device[d].terminal[tau].power[k, t].value for t in model.time_index]
                    for k in model.device[d].dev_index
                ]
            )
            for tau in range(dev.num_terminals_per_device)
        ]
        for d, dev in enumerate(devices)
    ]
    angle = [
        [
            np.array(
                [
                    [model.device[d].terminal[tau].angle[k, t].value for t in model.time_index]
                    for k in model.device[d].dev_index
                ]
            )
            for tau in range(dev.num_terminals_per_device)
        ]
        if dev.is_ac
        else None
        for d, dev in enumerate(devices)
    ]
    return power, angle


def split_incidence(A):
    return np.split(A.indices, A.indptr[1:-1])


def get_node_groups(device: AbstractDevice):
    return [split_incidence(A.tocsr()) for A in device.incidence_matrix]
