import numpy as np
import scipy.sparse as sp
import torch
from attrs import define, field, Factory

from typing import Optional
from numpy.typing import NDArray

from .abstract import AbstractDevice, get_time_horizon, make_dynamic


@define(kw_only=True, slots=False)
class AbstractInjector(AbstractDevice):
    """A single-node device that may deposit or withdraw power from the network. Abstract type that
    should not be instantiated but contains shared behavior among all subclasses."""

    num_nodes: int  # why do you re-instatiate this?
    name: str
    terminal: NDArray
    nominal_capacity: NDArray = field(
        default=Factory(lambda self: np.ones(self.num_devices), takes_self=True),
        converter=make_dynamic,
    )

    # These properties should be implemented by subclasses
    min_power: NDArray = field(init=False)
    max_power: NDArray = field(init=False)
    linear_cost: NDArray = field(init=False)
    quadratic_cost: Optional[NDArray] = field(init=False)
    capital_cost: Optional[NDArray] = field(init=False)
    emission_rates: Optional[NDArray] = field(init=False)

    @property
    def terminals(self):
        return self.terminal

    @property
    def time_horizon(self):
        return get_time_horizon(self.min_power)

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

        # Invert scaling because term is quadratic
        if self.quadratic_cost is not None:
            self.quadratic_cost *= scale

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def equality_constraints(
        self,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
        envelope=None,
    ):
        return []

    def inequality_constraints(
        self,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
        envelope=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)
        max_power = self.parameterize(max_power=max_power, la=la)
        min_power = self.parameterize(min_power=min_power, la=la)
        power = power[0]

        return [
            la.multiply(min_power, nominal_capacity) - power,
            power - la.multiply(max_power, nominal_capacity),
        ]

    def operation_cost(
        self,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
        envelope=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)
        min_power = self.parameterize(min_power=min_power, la=la)
        linear_cost = self.parameterize(linear_cost=linear_cost, la=la)
        power = power[0] - la.multiply(min_power, nominal_capacity)

        cost = la.sum(la.multiply(linear_cost, power))
        if self.quadratic_cost is not None:
            cost += la.sum(la.multiply(self.quadratic_cost, la.square(power)))

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

    def _equality_matrices(
        self,
        equalities,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
    ):
        return equalities

    def _inequality_matrices(
        self,
        inequalities,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
    ):
        size = inequalities[0].power[0].shape[1]
        inequalities[0].power[0] += -sp.eye(size)
        inequalities[1].power[0] += sp.eye(size)
        return inequalities

    def _hessian_power(
        self,
        hessians,
        power,
        angle,
        _,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        la=np,
    ):
        if self.quadratic_cost is None:
            return hessians

        hessians[0] += 2 * sp.diags(
            (self.quadratic_cost * np.ones_like(power[0])).ravel()
        )
        return hessians

    # ====
    # ADMM FUNCTIONS
    # ====

    def admm_prox_update(
        self,
        rho_power,
        rho_angle,
        power,
        angle,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
        power_weights=None,
        angle_weights=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity)
        max_power = self.parameterize(max_power=max_power)
        min_power = self.parameterize(min_power=min_power)
        linear_cost = self.parameterize(linear_cost=linear_cost)

        # machine, dtype = power[0].device, power[0].dtype
        assert angle is None

        if self.has_changed:
            quadratic_cost = (
                0.0 * self.linear_cost
                if self.quadratic_cost is None
                else self.quadratic_cost
            )
            pmax = torch.multiply(max_power, nominal_capacity)
            pmin = torch.multiply(min_power, nominal_capacity)
            self.admm_data = (quadratic_cost, pmax, pmin)
            self.has_changed = False

        quadratic_cost, pmax, pmin = self.admm_data

        return _admm_prox_update(
            power, rho_power, self.linear_cost, quadratic_cost, pmin, pmax
        )

    def get_admm_power_weights(
        self,
        power,
        strategy: str,
        nominal_capacity=None,
        max_power=None,
        min_power=None,
        linear_cost=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity)
        linear_cost = self.parameterize(linear_cost=linear_cost)

        if strategy == "smart_cost":
            avg_cost = np.mean(linear_cost, axis=1).reshape((-1, 1))
            return [np.maximum(np.sqrt(1 / (avg_cost + 0.01)), 1.0)]

        if strategy == "smart_bounds":
            return [np.minimum(np.sqrt(1 / (nominal_capacity + 1.0)), 1.0)]

        else:
            return [np.ones_like(pi) for pi in power]


@define(kw_only=True, slots=False)
class Injector(AbstractInjector):
    """A single-node device that may deposit or withdraw power from the network."""

    min_power: NDArray = field(converter=make_dynamic)
    max_power: NDArray = field(converter=make_dynamic)
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    capital_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    emission_rates: Optional[NDArray] = field(default=None, converter=make_dynamic)

    # TODO - Add dimension checks


@define(kw_only=True, slots=False)
class Generator(AbstractInjector):
    """An Injector that can only deposit power."""

    dynamic_capacity: NDArray = field(converter=make_dynamic)
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    capital_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)
    emission_rates: Optional[NDArray] = field(default=None, converter=make_dynamic)
    min_nominal_capacity: Optional[NDArray] = field(
        default=None, converter=make_dynamic
    )
    max_nominal_capacity: Optional[NDArray] = field(
        default=None, converter=make_dynamic
    )

    # TODO - Add dimension checks

    @property
    def min_power(self):
        return 0.0 * self.dynamic_capacity

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
        if self.capital_cost is None or nominal_capacity is None:
            return 0.0

        pnom_min = self.nominal_capacity
        capital_cost = self.capital_cost

        return la.sum(la.multiply(capital_cost, (nominal_capacity - pnom_min)))

    def get_emissions(self, power, nominal_capacity=None, la=np):
        if self.emission_rates is None:
            return 0.0
        else:
            return la.sum(la.multiply(self.emission_rates, power[0]))

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        if dev.dynamic_capacity.shape[1] > 1:
            dev.dynamic_capacity = dev.dynamic_capacity[:, time_periods]

        return dev


@define(kw_only=True, slots=False)
class Load(AbstractInjector):
    """An Injector that can only withdraw power."""

    load: NDArray = field(converter=make_dynamic)
    linear_cost: NDArray = field(converter=make_dynamic)
    quadratic_cost: Optional[NDArray] = field(default=None, converter=make_dynamic)

    @property
    def min_power(self):
        return -self.load

    @property
    def max_power(self):
        return 0.0 * self.load

    @property
    def capital_cost(self):
        return None

    @property
    def emission_rates(self):
        return None

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        if dev.load.shape[1] > 1:
            dev.load = dev.load[:, time_periods]

        return dev


@torch.jit.script
def _admm_prox_update(
    power: list[torch.Tensor], rho: float, lin_cost, quad_cost, pmin, pmax
):
    # Problem is
    #     min_p    a (p - pmin)^2 + b (p - pmin) + (rho / 2) || (p - power) ||_2^2 + {box constraints}
    # Objective derivative is
    #     2 a (p - pmin) + b +  rho (p - power) = 0
    # Which is solved by
    #     p = (rho power + 2 a pmin - b) / (2 a + rho )
    num = rho * power[0] + 2 * quad_cost * pmin - lin_cost
    denom = 2 * quad_cost + rho
    p = torch.divide(num, denom)

    # Finally, we project onto the box constraints
    p = torch.clip(p, pmin, pmax)

    return [p], None, None
