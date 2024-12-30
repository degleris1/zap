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
            return block.terminal[0].angle[n, t] == voltage[n][0]

        @block.Constraint(block.dev_index, block.model().time_index)
        def power_constraint(block, n, t):
            return block.terminal[0].power[n, t] == 0.0

        return block

    def add_objective(self, block: pyo.Block):
        block.operation_cost = pyo.Expression(expr=0.0)
        return block


class PyoDCLine(DCLine):
    def __init__(self, parent: DCLine):
        self.x = parent

    def add_local_variables(self, block: pyo.Block):
        return block

    def add_local_constraints(self, block: pyo.Block):
        _add_dc_line_constraints(block, self)
        return block

    def add_objective(self, block: pyo.Block):
        _add_dc_line_cost(block, self)
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
        _add_dc_line_cost(block, self)
        return block


class PyoBattery(Battery):
    def __init__(self, parent: Battery):
        self.x = parent

    def add_local_variables(self, block: pyo.Block):
        block.soc = pyo.Var(block.dev_index, block.model().time_index, domain=pyo.NonNegativeReals)
        block.charge = pyo.Var(
            block.dev_index, block.model().time_index, domain=pyo.NonNegativeReals
        )
        block.discharge = pyo.Var(
            block.dev_index, block.model().time_index, domain=pyo.NonNegativeReals
        )

        return block

    def add_local_constraints(self, block: pyo.Block):
        # Inequalityies
        power_capacity = self.x.power_capacity.tolist()
        energy_capacity = (self.x.power_capacity * self.x.duration).tolist()

        @block.Constraint(block.dev_index, block.model().time_index)
        def soc_limit(block, k, t):
            return block.soc[k, t] <= energy_capacity[k][0]

        @block.Constraint(block.dev_index, block.model().time_index)
        def charge_limit(block, k, t):
            return block.charge[k, t] <= power_capacity[k][0]

        @block.Constraint(block.dev_index, block.model().time_index)
        def discharge_limit(block, k, t):
            return block.discharge[k, t] <= power_capacity[k][0]

        # Equalities
        charge_efficiency = self.x.charge_efficiency.tolist()
        initial_soc = self.x.initial_soc.tolist()
        final_soc = self.x.final_soc.tolist()
        T = block.model().time_horizon

        @block.Constraint(block.dev_index, block.model().time_index)
        def battery_power_balance(block, k, t):
            return block.discharge[k, t] - block.charge[k, t] == block.terminal[0].power[k, t]

        @block.Constraint(block.dev_index, range(T - 1))
        def soc_evolution(block, k, t):
            return (
                block.soc[k, t + 1]
                == block.soc[k, t]
                + block.charge[k, t] * charge_efficiency[k][0]
                - block.discharge[k, t]
            )

        @block.Constraint(block.dev_index)
        def initial_soc_constraint(block, k):
            return block.soc[k, 0] == initial_soc[k][0] * energy_capacity[k][0]

        @block.Constraint(block.dev_index)
        def final_soc_constraint(block, k):
            return (
                block.soc[k, T - 1]
                + block.charge[k, T - 1] * charge_efficiency[k][0]
                - block.discharge[k, T - 1]
                == final_soc[k][0] * energy_capacity[k][0]
            )

        return block

        # T = power[0].shape[1]
        # energy_capacity = la.multiply(power_capacity, self.duration)

        # soc_evolution = (
        #     state.energy[:, :-1]
        #     + la.multiply(state.charge, self.charge_efficiency)
        #     - state.discharge
        # )
        # return [
        #     power[0] - (state.discharge - state.charge),
        #     state.energy[:, 1:] - soc_evolution,
        #     state.energy[:, 0:1] - la.multiply(self.initial_soc, energy_capacity),
        #     state.energy[:, T : (T + 1)] - la.multiply(self.final_soc, energy_capacity),
        # ]

    def add_objective(self, block: pyo.Block):
        # cost = la.sum(la.multiply(self.linear_cost, state.discharge))
        linear_cost = self.x.linear_cost.tolist()

        @block.Expression()
        def operation_cost(block):
            return sum(
                linear_cost[k][0] * block.discharge[k, t]
                for k in block.dev_index
                for t in block.model().time_index
            )

        return block


# ========
# Helper Functions
# ========


def convert_to_pyo(device: AbstractDevice):
    if isinstance(device, Generator) or isinstance(device, Load):
        return PyoInjector(device)
    elif isinstance(device, Ground):
        return PyoGround(device)
    elif isinstance(device, ACLine):
        return PyoACLine(device)
    elif isinstance(device, DCLine):
        return PyoDCLine(device)
    elif isinstance(device, Battery):
        return PyoBattery(device)
    else:
        raise NotImplementedError(f"Device type {type(device)} not supported in Pyomo conversion")


def _add_dc_line_cost(block: pyo.Block, line):
    linear_cost = line.x.linear_cost.tolist()

    block.abs_cost = pyo.Var(block.dev_index, block.model().time_index, domain=pyo.NonNegativeReals)

    @block.Constraint(block.dev_index, block.model().time_index)
    def cost_pos_constraint(block, k, t):
        return block.abs_cost[k, t] >= block.terminal[1].power[k, t] * linear_cost[k][0]

    @block.Constraint(block.dev_index, block.model().time_index)
    def cost_neg_constraint(block, k, t):
        return block.abs_cost[k, t] >= -block.terminal[1].power[k, t] * linear_cost[k][0]

    @block.Expression()
    def operation_cost(block):
        return sum(block.abs_cost[k, t] for k in block.dev_index for t in block.model().time_index)

    return block


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
