import numpy as np
import scipy.sparse as sp

from collections import namedtuple
from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray

from zap.util import replace_none
from .abstract import AbstractDevice, get_time_horizon, make_dynamic

InjectorData = namedtuple(
    "InjectorData",
    [
        "min_power",
        "max_power",
        "linear_cost",
        "quadratic_cost",
        "nominal_capacity",
        "capital_cost",
        "emission_rates",
    ],
)


@dataclass(kw_only=True)
class Injector(AbstractDevice):
    """A single-node device that may deposit or withdraw power from the network."""

    num_nodes: int
    terminal: NDArray
    min_power: NDArray
    max_power: NDArray
    linear_cost: NDArray
    quadratic_cost: Optional[NDArray] = None
    nominal_capacity: Optional[NDArray] = None
    capital_cost: Optional[NDArray] = None
    emission_rates: Optional[NDArray] = None

    def __post_init__(self):
        if self.nominal_capacity is None:
            self.nominal_capacity = np.ones(self.num_devices)

        # Reshape arrays
        self.min_power = make_dynamic(self.min_power)
        self.max_power = make_dynamic(self.max_power)
        self.linear_cost = make_dynamic(self.linear_cost)
        self.quadratic_cost = make_dynamic(self.quadratic_cost)
        self.nominal_capacity = make_dynamic(self.nominal_capacity)
        self.capital_cost = make_dynamic(self.capital_cost)
        self.emission_rates = make_dynamic(self.emission_rates)

        # TODO - Add dimension checks
        pass

    @property
    def terminals(self):
        return self.terminal

    @property
    def time_horizon(self):
        return get_time_horizon(self.min_power)

    def _device_data(self, nominal_capacity=None):
        return InjectorData(
            self.min_power,
            self.max_power,
            self.linear_cost,
            self.quadratic_cost,
            make_dynamic(replace_none(nominal_capacity, self.nominal_capacity)),
            self.capital_cost,
            self.emission_rates,
        )

    def scale_costs(self, scale):
        self.linear_cost /= scale
        if self.quadratic_cost is not None:
            self.quadratic_cost /= scale
        if self.capital_cost is not None:
            self.capital_cost /= scale

        # Emissions are in units of kg/MWh
        # but we scale them with costs so that prices can be stated in $/MWh
        if self.emission_rates is not None:
            self.emission_rates /= scale

    def scale_power(self, scale):
        self.nominal_capacity /= scale

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def equality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        return []

    def inequality_constraints(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)
        power = power[0]

        return [
            la.multiply(data.min_power, data.nominal_capacity) - power,
            power - la.multiply(data.max_power, data.nominal_capacity),
        ]

    def operation_cost(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)

        power = power[0] - la.multiply(data.min_power, data.nominal_capacity)

        cost = la.sum(la.multiply(data.linear_cost, power))
        if data.quadratic_cost is not None:
            cost += la.sum(la.multiply(data.quadratic_cost, la.square(power)))

        return cost

    # ====
    # PLANNING FUNCTIONS
    # ====

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        # Subsample linear cost
        if dev.linear_cost.shape[1] > 1:
            dev.linear_cost = dev.linear_cost[:, time_periods]

        return dev

    # ====
    # DIFFERENTIATION
    # ====

    def _equality_matrices(self, equalities, nominal_capacity=None, la=np):
        return equalities

    def _inequality_matrices(self, inequalities, nominal_capacity=None, la=np):
        size = inequalities[0].power[0].shape[1]
        inequalities[0].power[0] += -sp.eye(size)
        inequalities[1].power[0] += sp.eye(size)
        return inequalities

    # ====
    # ADMM FUNCTIONS
    # ====

    def admm_initialize_power_variables(self, time_horizon: int):
        return [np.zeros((self.num_devices, time_horizon))]

    def admm_initialize_angle_variables(self, time_horizon: int):
        return None

    def admm_prox_update(
        self,
        rho_power,
        rho_angle,
        power,
        angle,
        nominal_capacity=None,
        la=np,
        power_weights=None,
        angle_weights=None,
    ):
        data = self.device_data(nominal_capacity=nominal_capacity, la=la)
        assert angle is None

        Dp2 = [np.power(p, 2) for p in power_weights]

        # Problem is
        #     min_p    a p^2 + b p + (rho / 2) || Dp (p - power) ||_2^2 + {box constraints}
        # Objective derivative is
        #     2 a p + b +  rho Dp^2 (p - power) = 0
        # Which is solved by
        #     p = (rho Dp^2 power - b) / (2 a + rho Dp^2)
        quadratic_cost = 0.0 if data.quadratic_cost is None else data.quadratic_cost

        num = rho_power * Dp2[0] * power[0] - data.linear_cost
        denom = 2 * quadratic_cost + rho_power * Dp2[0]
        p = np.divide(num, denom)

        # Finally, we project onto the box constraints
        pmax = np.multiply(data.max_power, data.nominal_capacity)
        pmin = np.multiply(data.min_power, data.nominal_capacity)
        p = np.clip(p, pmin, pmax)

        return [p], None

    def get_admm_power_weights(self, power, strategy: str, nominal_capacity=None):
        data = self.device_data(nominal_capacity=nominal_capacity, la=np)

        if strategy == "smart_cost":
            avg_cost = np.mean(data.linear_cost, axis=1).reshape((-1, 1))
            return [np.maximum(np.sqrt(1 / (avg_cost + 0.01)), 1.0)]

        if strategy == "smart_bounds":
            return [np.minimum(np.sqrt(1 / (data.nominal_capacity + 1.0)), 1.0)]

        else:
            return [np.ones_like(pi) for pi in power]


class Generator(Injector):
    """An Injector that can only deposit power."""

    def __init__(
        self,
        *,
        num_nodes,
        terminal,
        dynamic_capacity,
        linear_cost,
        quadratic_cost=None,
        nominal_capacity=None,
        capital_cost=None,
        emission_rates=None,
        min_nominal_capacity=None,
        max_nominal_capacity=None,
    ):
        self.num_nodes = num_nodes
        self.terminal = terminal

        if nominal_capacity is None:
            nominal_capacity = np.ones(self.num_devices)

        self.dynamic_capacity = make_dynamic(dynamic_capacity)
        self.nominal_capacity = make_dynamic(nominal_capacity)
        self.linear_cost = make_dynamic(linear_cost)
        self.quadratic_cost = make_dynamic(quadratic_cost)
        self.capital_cost = make_dynamic(capital_cost)
        self.emission_rates = make_dynamic(emission_rates)
        self.min_nominal_capacity = make_dynamic(min_nominal_capacity)
        self.max_nominal_capacity = make_dynamic(max_nominal_capacity)

        # TODO - Add dimension checks
        pass

    @property
    def min_power(self):
        return np.zeros(self.dynamic_capacity.shape)

    @property
    def max_power(self):
        return self.dynamic_capacity

    def scale_power(self, scale):
        if self.min_nominal_capacity is not None:
            self.min_nominal_capacity /= scale
        if self.max_nominal_capacity is not None:
            self.max_nominal_capacity /= scale
        return super().scale_power(scale)

    # ====
    # PLANNING FUNCTIONS
    # ====

    def get_investment_cost(self, nominal_capacity=None, la=np):
        # Get nominal capacity and capital cost
        data = self.device_data(la=la)

        if self.capital_cost is None or nominal_capacity is None:
            return 0.0

        pnom_min = data.nominal_capacity
        capital_cost = data.capital_cost

        return la.sum(la.multiply(capital_cost, (nominal_capacity - pnom_min)))

    def get_emissions(self, power, nominal_capacity=None, la=np):
        data = self.device_data(la=la)

        if data.emission_rates is None:
            return 0.0
        else:
            return la.sum(la.multiply(data.emission_rates, power[0]))

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        if dev.dynamic_capacity.shape[1] > 1:
            dev.dynamic_capacity = dev.dynamic_capacity[:, time_periods]

        return dev


class Load(Injector):
    """An Injector that can only withdraw power."""

    def __init__(self, *, num_nodes, terminal, load, linear_cost, quadratic_cost=None):
        self.num_nodes = num_nodes
        self.terminal = terminal

        self.nominal_capacity = make_dynamic(np.ones(self.num_devices))
        self.load = make_dynamic(load)
        self.linear_cost = make_dynamic(linear_cost)
        self.quadratic_cost = make_dynamic(quadratic_cost)

    @property
    def min_power(self):
        return -self.load

    @property
    def max_power(self):
        return np.zeros(self.load.shape)

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        if dev.load.shape[1] > 1:
            dev.load = dev.load[:, time_periods]

        return dev
