import pyomo.environ as pyo
import numpy as np

from zap.devices import AbstractDevice, Injector
from zap.devices import Generator, Load, Ground, ACLine, DCLine, Battery


class PyoInjector(Injector):
    def __init__(self, parent: Injector):
        self.x = parent

    def add_local_variables(self, block: pyo.Block):
        return block

    def add_local_constraints(self, block: pyo.Block):
        nominal_capacity = self.x.nominal_capacity.tolist()
        max_power = self.x.max_power.tolist()
        min_power = self.x.min_power.tolist()

        @block.Constraint(block.dev_index, block.model().time_index)
        def min_power_limit(block, k, t):
            return block.terminal[0].power[k, t] >= min_power[k][t] * nominal_capacity[k][0]

        @block.Constraint(block.dev_index, block.model().time_index)
        def max_power_limit(block, k, t):
            return block.terminal[0].power[k, t] <= max_power[k][t] * nominal_capacity[k][0]

        return block

    def add_objective(self, block: pyo.Block):
        nominal_capacity = self.x.nominal_capacity.tolist()
        min_power = self.x.min_power.tolist()

        linear_cost = np.broadcast_to(
            self.x.linear_cost, shape=(self.x.num_devices, self.x.time_horizon)
        )

        @block.Expression()
        def operation_cost(block):
            return sum(
                linear_cost[k, t]
                * (block.terminal[0].power[k, t] - min_power[k][t] * nominal_capacity[k][0])
                for k in block.dev_index
                for t in block.model().time_index
            )

        return block


class PyoGround(Ground):
    def __init__(self, parent: Ground):
        self.x = parent

    def add_local_variables(self, block: pyo.Block):
        return block

    def add_local_constraints(self, block: pyo.Block):
        voltage = self.x.voltage.tolist()

        @block.Constraint(block.dev_index, block.model().time_index)
        def voltage_constraint(block, n, t):
            return block.terminal[0].angle[n, t] == voltage[n][t]

        @block.Constraint(block.dev_index, block.model().time_index)
        def power_constraint(block, n, t):
            return block.terminal[0].power[n, t] == 0.0

        return block

    def add_objective(self, block: pyo.Block):
        block.operation_cost = pyo.Expression(expr=0.0)
        return block


# def equality_constraints(
#         self, power, angle, _, nominal_capacity=None, la=np, envelope=None, mask=None
#     ):
#         return [power[1] + power[0]]

# def inequality_constraints(
#     self, power, angle, _, nominal_capacity=None, la=np, envelope=None, mask=None
# ):
#     nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)

#     return [
#         la.multiply(self.min_power, nominal_capacity) - power[1] - self.slack,
#         power[1] - la.multiply(self.max_power, nominal_capacity) - self.slack,
#     ]


class PyoDCLine(DCLine):
    def __init__(self, parent: DCLine):
        self.x = parent

    def add_local_variables(self, block: pyo.Block):
        return block

    def add_local_constraints(self, block: pyo.Block):
        _add_dc_line_constraints(block, self)
        return block

    def add_objective(self, block: pyo.Block):
        block.operation_cost = pyo.Expression(expr=0.0)
        return block


class PyoACLine(ACLine):
    def __init__(self, parent: ACLine):
        self.x = parent

    def add_local_variables(self, block: pyo.Block):
        return block

    def add_local_constraints(self, block: pyo.Block):
        _add_dc_line_constraints(block, self)

        susceptance = self.x.susceptance.tolist()
        nominal_capacity = self.x.nominal_capacity.tolist()

        @block.Constraint(block.dev_index, block.model().time_index)
        def power_flow_constraint(block, k, t):
            return (
                nominal_capacity[k][0]
                * susceptance[k][0]
                * (block.terminal[0].angle[k, t] - block.terminal[1].angle[k, t])
                == block.terminal[1].power[k, t]
            )

        return block

    def add_objective(self, block: pyo.Block):
        block.operation_cost = pyo.Expression(expr=0.0)
        return block


class PyoBattery(Battery):
    pass


def _add_dc_line_constraints(block: pyo.Block, line):
    nominal_capacity = line.x.nominal_capacity.tolist()
    capacity = line.x.capacity.tolist()
    slack = line.x.slack

    @block.Constraint(block.dev_index, block.model().time_index)
    def power_balance(block, k, t):
        return block.terminal[0].power[k, t] + block.terminal[1].power[k, t] == 0.0

    @block.Constraint(block.dev_index, block.model().time_index)
    def min_power_limit(block, k, t):
        return block.terminal[1].power[k, t] >= -capacity[k][0] * nominal_capacity[k][0] - slack

    @block.Constraint(block.dev_index, block.model().time_index)
    def max_power_limit(block, k, t):
        return block.terminal[1].power[k, t] <= capacity[k][0] * nominal_capacity[k][0] + slack

    return block


def convert_to_pyo(device: AbstractDevice):
    if isinstance(device, Generator) or isinstance(device, Load):
        return PyoInjector(device)
    elif isinstance(device, Ground):
        return PyoGround(device)
    elif isinstance(device, ACLine):
        return PyoACLine(device)
    elif isinstance(device, DCLine):
        return PyoDCLine(device)
    else:
        raise NotImplementedError(f"Device type {type(device)} not supported in Pyomo conversion")
