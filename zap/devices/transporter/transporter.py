import numpy as np
import scipy.sparse as sp
import torch

from dataclasses import dataclass
from functools import cached_property
from typing import Optional
from numpy.typing import NDArray

from zap.devices.abstract import AbstractDevice, make_dynamic
from zap.util import replace_none


@dataclass(kw_only=True)
class Transporter(AbstractDevice):
    """A two-node device that carries power between nodes.

    The net power of a transporter is always zero.
    """

    num_nodes: int
    source_terminal: NDArray
    sink_terminal: NDArray
    min_power: NDArray
    max_power: NDArray
    linear_cost: NDArray
    quadratic_cost: Optional[NDArray] = None
    nominal_capacity: Optional[NDArray] = None
    capital_cost: Optional[NDArray] = None
    slack: Optional[NDArray] = None
    min_nominal_capacity: Optional[NDArray] = None
    max_nominal_capacity: Optional[NDArray] = None
    reconductoring_cost: Optional[NDArray] = None
    reconductoring_threshold: Optional[NDArray] = None

    def __post_init__(self):
        # Reshape arrays
        self.min_power = make_dynamic(self.min_power)
        self.max_power = make_dynamic(self.max_power)
        self.linear_cost = make_dynamic(self.linear_cost)
        self.quadratic_cost = make_dynamic(self.quadratic_cost)
        self.nominal_capacity = make_dynamic(
            replace_none(self.nominal_capacity, np.ones(self.num_devices))
        )
        self.capital_cost = make_dynamic(self.capital_cost)
        self.slack = 0.0 if self.slack is None else make_dynamic(self.slack)
        self.min_nominal_capacity = make_dynamic(self.min_nominal_capacity)
        self.max_nominal_capacity = make_dynamic(self.max_nominal_capacity)
        self.reconductoring_cost = make_dynamic(self.reconductoring_cost)
        self.reconductoring_threshold = make_dynamic(self.reconductoring_threshold)

        # TODO - Add dimension checks
        pass

    @cached_property
    def terminals(self):
        return np.column_stack((self.source_terminal, self.sink_terminal))

    @property
    def time_horizon(self):
        return 0  # Static device

    def scale_costs(self, scale):
        self.linear_cost /= scale
        if self.quadratic_cost is not None:
            self.quadratic_cost /= scale
        if self.capital_cost is not None:
            self.capital_cost /= scale
        if self.reconductoring_cost is not None:
            self.reconductoring_cost /= scale

    def scale_power(self, scale):
        self.nominal_capacity /= scale
        if self.min_nominal_capacity is not None:
            self.min_nominal_capacity /= scale
        if self.max_nominal_capacity is not None:
            self.max_nominal_capacity /= scale
        self.slack /= scale

        # Invest scaling because term is quadratic
        if self.quadratic_cost is not None:
            self.quadratic_cost *= scale

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def equality_constraints(
        self, power, angle, _, nominal_capacity=None, la=np, envelope=None, mask=None
    ):
        return [power[1] + power[0]]

    def inequality_constraints(
        self, power, angle, _, nominal_capacity=None, la=np, envelope=None, mask=None
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)

        return [
            la.multiply(self.min_power, nominal_capacity) - power[1] - self.slack,
            power[1] - la.multiply(self.max_power, nominal_capacity) - self.slack,
        ]

    def operation_cost(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)

        cost = la.sum(la.multiply(self.linear_cost, la.abs(power[1])))
        if self.quadratic_cost is not None:
            cost += la.sum(la.multiply(self.quadratic_cost, la.square(power[1])))

        return cost

    # ====
    # DIFFERENTIATION
    # ====

    def _equality_matrices(self, equalities, nominal_capacity=None, la=np):
        size = equalities[0].power[0].shape[1]

        equalities[0].power[0] += sp.eye(size)
        equalities[0].power[1] += sp.eye(size)

        return equalities

    def _inequality_matrices(self, inequalities, nominal_capacity=None, la=np):
        size = inequalities[0].power[0].shape[1]

        inequalities[0].power[1] += -sp.eye(size)
        inequalities[1].power[1] += sp.eye(size)

        return inequalities

    def _hessian_power(self, hessians, power, angle, _, nominal_capacity=None, la=np):
        if self.quadratic_cost is None:
            return hessians

        hessians[1] += 2 * sp.diags((self.quadratic_cost * power[1]).ravel())
        return hessians

    # ====
    # PLANNING
    # ====

    def get_investment_cost(self, nominal_capacity=None, la=np):
        # Get original nominal capacity and capital cost
        # Nominal capacity isn't passed here because we want to use the original value
        if self.capital_cost is None or nominal_capacity is None:
            print(f"No capital cost or nominal capacity for device {type(self)}")
            return 0.0

        # Device nominal capacity = min nominal capacity
        # This is not the same as the input `nominal_capacity`
        pnom_min = self.nominal_capacity
        c = self.capital_cost

        if self.reconductoring_cost is None:
            return la.sum(la.multiply(c, (nominal_capacity - pnom_min)))

        else:
            r = self.reconductoring_cost
            alpha = self.reconductoring_threshold

            z = pnom_min * (r + c * alpha - r * alpha)
            return la.sum(
                la.maximum(
                    la.multiply(r, (nominal_capacity - pnom_min)),
                    la.multiply(c, nominal_capacity) - z,
                )
            )

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
        power_weights=None,
        angle_weights=None,
    ):
        assert angle is None
        # machine, dtype = power[0].device, power[0].dtype
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity)

        quadratic_cost = 0.0 if self.quadratic_cost is None else self.quadratic_cost
        pmax = torch.multiply(self.max_power, nominal_capacity) + self.slack
        pmin = torch.multiply(self.min_power, nominal_capacity) - self.slack

        # if power_weights is None:
        #     power_weights = [torch.tensor(1.0, device=machine, dtype=dtype) for _ in power]

        # Dp2 = [torch.pow(p, 2) for p in power_weights]

        # TODO
        # This shouldn't be too hard, we just solve the two cases (p1 < 0) and (p1 > 0)
        # Then project onto [pmin, 0] and [0, pmax]
        # Then pick the better of the two
        # assert torch.sum(torch.abs(self.linear_cost)) == 0.0

        # Problem is
        #     min_p    a p1^2 + b |p1|
        #              + (rho/2) ||D0 (p0 - power0)||_2^2 + (rho/2) ||D1 (p1 - power1)||_2^2
        #              + {p1 box constraints} + {p0 + p1 = 0}
        #
        # Setting p0 = -p1, we remove the equality constraint and reformulate as
        #     min_p1   a p1^2 + b |p1|
        #              + (rho/2) ||D0 (-p1 - power0) ||_2^2 + (rho/2) ||D1 (p1 - power1) ||_2^2
        #              + {p1 box constraints}
        #
        # The objective derivative is then
        #     2 a p1 + b sign(p1) - rho D0^2 (-p1 - power0) + rho D1^2 (p1 - power1),
        #
        # which is solved by
        #     2 a p1 + b sign(p1) + rho D0^2 p1 + rho D0^2 power0 + rho D1^2 p1 - rho D1^2 power1 = 0
        #     2 a p1 + rho (D0^2 + D1^2) p1 = rho D1^2 power1 - rho D0^2 power0 - b sign(p1)
        #     p1 = (rho D1^2 power1 - rho D0^2 power0 - b sign(p1)) / (2 a + rho D0^2 + rho D1^2)

        # Default is sign(num) = +1.0
        num = rho_power * (power[1] - power[0]) - self.linear_cost

        # This term is always positive, so we can pick it after choosing the sign
        denom = 2 * (quadratic_cost + rho_power)
        p1 = torch.divide(num, denom)

        # Finally, we project onto the box constraints
        p1 = torch.clip(p1, pmin, pmax)
        p0 = -p1

        return [p0, p1], None, None
