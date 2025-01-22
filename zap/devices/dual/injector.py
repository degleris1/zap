import numpy as np

from zap.devices.injector import AbstractInjector
from zap.util import envelope_variable, use_envelope


class DualInjector(AbstractInjector):
    def __init__(self, injector: AbstractInjector, max_price=None, **kwargs):
        self.primal = injector
        self.max_price = max_price

        self.num_nodes = injector.num_nodes
        self.terminal = injector.terminal
        self.min_power = injector.min_power
        self.max_power = injector.max_power
        self.linear_cost = injector.linear_cost
        self.quadratic_cost = injector.quadratic_cost
        self.nominal_capacity = injector.nominal_capacity
        self.capital_cost = injector.capital_cost
        self.emission_rates = injector.emission_rates

    # This isn't helpful because of the way dataclasses define attributes
    # def __getattr__(self, attr):
    #     return getattr(self.primal, attr)

    def equality_constraints(
        self, power, angle, _, nominal_capacity=None, la=np, envelope=None
    ):
        return []

    def inequality_constraints(
        self, power, angle, _, nominal_capacity=None, la=np, envelope=None
    ):
        return []

    def operation_cost(
        self, power, angle, _, nominal_capacity=None, la=np, envelope=None
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)

        assert self.quadratic_cost is None

        z = power[0]
        pnom = nominal_capacity
        pmin = self.min_power
        pmax = self.max_power
        c = self.linear_cost

        if use_envelope(envelope):
            print("Envelope relaxation applied to dual injector.")
            env, lower, upper = envelope
            lb, ub = lower["nominal_capacity"], upper["nominal_capacity"]
            z_pnom = envelope_variable(
                pnom, z, lb, ub, -self.max_price, self.max_price, *env
            )
        else:
            z_pnom = la.multiply(z, pnom)

        f1 = la.multiply(z_pnom, pmin)
        f2 = la.multiply(z_pnom, pmax)
        f2 -= la.multiply(c, la.multiply(pnom, (pmax - pmin)))

        return la.sum(la.maximum(f1, f2))
