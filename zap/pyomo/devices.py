import pyomo.environ as pyo
import numpy as np

from zap.devices import AbstractDevice, Injector, Generator, Load, Ground, ACLine, DCLine, Battery


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


class PyoDCLine(DCLine):
    pass


class PyoACLine(ACLine):
    pass


class PyoBattery(Battery):
    pass


def convert_to_pyo(device: AbstractDevice):
    if isinstance(device, Generator) or isinstance(device, Load):
        return PyoInjector(device)
    elif isinstance(device, Ground):
        return PyoGround(device)
    else:
        raise NotImplementedError(f"Device type {type(device)} not supported in Pyomo conversion")
