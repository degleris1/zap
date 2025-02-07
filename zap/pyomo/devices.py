import pyomo.environ as pyo
import numpy as np

from zap.devices import AbstractDevice, Injector
from zap.devices import Generator, Load, Ground, ACLine, DCLine, Battery


class PyoDevice:
    def __init__(self, parent: AbstractDevice):
        self.x = parent
        self.has_parameter = False
        self.param_name = "nominal_capacity"

    def make_parameteric(self, block: pyo.Block, is_int: bool = False, int_quantity: float = 0.1):
        return _add_simple_parameter(
            block, self, attr_name=self.param_name, is_int=is_int, int_quantity=int_quantity
        )

    def add_investment_cost(self, block: pyo.Block):
        return _add_simple_investment_cost(block, self, attr_name=self.param_name)

    def add_local_variables(self, block: pyo.Block):
        return block

    def add_local_constraints(self, block: pyo.Block):
        raise NotImplementedError

    def add_objective(self, block: pyo.Block):
        raise NotImplementedError

    def model_emissions(self, block: pyo.Block):
        block.emissions = pyo.Expression(expr=0.0)
        return block


class PyoInjector(PyoDevice):
    def __init__(self, parent: Injector):
        super().__init__(parent)

    def get_nominal_capacity(self):
        if self.has_parameter:
            return self.param
        else:
            return self.x.nominal_capacity.reshape(-1).tolist()

    def add_local_constraints(self, block: pyo.Block):
        nominal_capacity = self.get_nominal_capacity()
        max_power = self.x.max_power.tolist()
        min_power = self.x.min_power.tolist()

        @block.Constraint(block.dev_index, block.model().time_index)
        def min_power_limit(block, k, t):
            return block.terminal[0].power[k, t] >= min_power[k][t] * nominal_capacity[k]

        @block.Constraint(block.dev_index, block.model().time_index)
        def max_power_limit(block, k, t):
            return block.terminal[0].power[k, t] <= max_power[k][t] * nominal_capacity[k]

        return block

    def add_objective(self, block: pyo.Block):
        nominal_capacity = self.get_nominal_capacity()
        min_power = self.x.min_power.tolist()

        linear_cost = np.broadcast_to(
            self.x.linear_cost, shape=(self.x.num_devices, self.x.time_horizon)
        )

        @block.Expression()
        def operation_cost(block):
            return sum(
                linear_cost[k, t]
                * (block.terminal[0].power[k, t] - min_power[k][t] * nominal_capacity[k])
                for k in block.dev_index
                for t in block.model().time_index
            )

        return block

    def model_emissions(self, block: pyo.Block):
        if hasattr(self.x, "emission_rates") and self.x.emission_rates is not None:
            emissions = self.x.emission_rates.reshape(-1).tolist()
            block.emissions = pyo.Expression(
                expr=sum(
                    emissions[k] * block.terminal[0].power[k, t]
                    for k in block.dev_index
                    for t in block.model().time_index
                )
            )
        else:
            block.emissions = pyo.Expression(expr=0.0)

        return block


class PyoGround(PyoDevice):
    def __init__(self, parent: Ground):
        super().__init__(parent)

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


class PyoDCLine(PyoDevice):
    def __init__(self, parent: Ground):
        super().__init__(parent)

    def get_nominal_capacity(self):
        if self.has_parameter:
            return self.param
        else:
            return self.x.nominal_capacity.reshape(-1).tolist()

    def add_local_constraints(self, block: pyo.Block):
        _add_dc_line_constraints(block, self)
        return block

    def add_objective(self, block: pyo.Block):
        _add_dc_line_cost(block, self)
        return block


class PyoACLine(PyoDevice):
    def __init__(self, parent: Ground):
        super().__init__(parent)

    def get_nominal_capacity(self):
        if self.has_parameter:
            return self.param
        else:
            return self.x.nominal_capacity.reshape(-1).tolist()

    def add_local_constraints(self, block: pyo.Block):
        _add_dc_line_constraints(block, self)

        susceptance = self.x.susceptance.tolist()
        nominal_capacity = self.get_nominal_capacity()

        @block.Constraint(block.dev_index, block.model().time_index)
        def power_flow_constraint(block, k, t):
            return (
                nominal_capacity[k]
                * susceptance[k][0]
                * (block.terminal[0].angle[k, t] - block.terminal[1].angle[k, t])
                == block.terminal[1].power[k, t]
            )

        return block

    def add_objective(self, block: pyo.Block):
        _add_dc_line_cost(block, self)
        return block


class PyoBattery(PyoDevice):
    def __init__(self, parent: Battery):
        super().__init__(parent)
        self.param_name = "power_capacity"

    def get_power_capacity(self):
        if self.has_parameter:
            return self.param
        else:
            return self.x.power_capacity.reshape(-1).tolist()

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
        power_capacity = self.get_power_capacity()
        duration = self.x.duration.reshape(-1).tolist()

        @block.Constraint(block.dev_index, block.model().time_index)
        def soc_limit(block, k, t):
            return block.soc[k, t] <= duration[k] * power_capacity[k]

        @block.Constraint(block.dev_index, block.model().time_index)
        def charge_limit(block, k, t):
            return block.charge[k, t] <= power_capacity[k]

        @block.Constraint(block.dev_index, block.model().time_index)
        def discharge_limit(block, k, t):
            return block.discharge[k, t] <= power_capacity[k]

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
            return block.soc[k, 0] == initial_soc[k][0] * duration[k] * power_capacity[k]

        @block.Constraint(block.dev_index)
        def final_soc_constraint(block, k):
            return (
                block.soc[k, T - 1]
                + block.charge[k, T - 1] * charge_efficiency[k][0]
                - block.discharge[k, T - 1]
                == final_soc[k][0] * duration[k] * power_capacity[k]
            )

        return block

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


def convert_to_pyo(device: AbstractDevice) -> PyoDevice:
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
    nominal_capacity = line.get_nominal_capacity()
    capacity = line.x.capacity.tolist()
    slack = line.x.slack

    @block.Constraint(block.dev_index, block.model().time_index)
    def power_balance(block, k, t):
        return block.terminal[0].power[k, t] + block.terminal[1].power[k, t] == 0.0

    @block.Constraint(block.dev_index, block.model().time_index)
    def min_power_limit(block, k, t):
        return block.terminal[1].power[k, t] >= -capacity[k][0] * nominal_capacity[k] - slack

    @block.Constraint(block.dev_index, block.model().time_index)
    def max_power_limit(block, k, t):
        return block.terminal[1].power[k, t] <= capacity[k][0] * nominal_capacity[k] + slack

    return block


def _add_simple_parameter(
    block: pyo.Block, device, is_int: bool, int_quantity: float, attr_name="nominal_capacity"
):
    block.param = pyo.Var(range(device.x.num_devices))

    # Add constraints on parameter
    if getattr(device.x, "min_" + attr_name) is None:
        min_param = getattr(device.x, attr_name).reshape(-1).tolist()
    else:
        min_param = getattr(device.x, "min_" + attr_name).reshape(-1).tolist()

    if getattr(device.x, "max_" + attr_name) is None:
        max_param = np.inf * np.ones(device.x.num_devices)
    else:
        max_param = getattr(device.x, "max_" + attr_name).reshape(-1).tolist()

    @block.Constraint(range(device.x.num_devices))
    def lower_bound(block, k):
        return block.param[k] >= min_param[k]

    @block.Constraint(range(device.x.num_devices))
    def upper_bound(block, k):
        return block.param[k] <= max_param[k]

    # Add integer variable and constraints
    if is_int:
        print(f"!!!Adding integer variables for device {type(device)} with size {int_quantity}!!!")
        block.int_param = pyo.Var(range(device.x.num_devices), domain=pyo.NonNegativeIntegers)

        block.int_constraint = pyo.Constraint(
            range(device.x.num_devices),
            rule=lambda block, k: block.param[k]
            == min_param[k] + int_quantity * block.int_param[k],
        )

        device.param = block.int_param

    else:
        print(f"!!!Adding continuous variables for device {type(device)}!!!")

    device.has_parameter = True
    device.param = block.param

    return block.param


def _add_simple_investment_cost(block: pyo.Block, device, attr_name="nominal_capacity"):
    capital_cost = device.x.capital_cost.reshape(-1).tolist()

    if getattr(device.x, "min_" + attr_name) is None:
        min_param = getattr(device.x, attr_name).reshape(-1).tolist()
    else:
        min_param = getattr(device.x, "min_" + attr_name).reshape(-1).tolist()

    @block.Expression()
    def investment_cost(block):
        return sum(
            capital_cost[k] * (block.param[k] - min_param[k]) for k in range(device.x.num_devices)
        )

    return block
