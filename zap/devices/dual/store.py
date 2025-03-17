import numpy as np
import cvxpy as cp

from zap.devices.storage_unit import StorageUnit
from zap.util import envelope_variable, use_envelope


class DualBattery(StorageUnit):
    def __init__(self, battery: StorageUnit, max_price=None, **kwargs):
        self.primal = battery
        self.max_price = max_price

        self.num_nodes = battery.num_nodes
        self.terminal = battery.terminal
        self.power_capacity = battery.power_capacity
        self.duration = battery.duration
        self.charge_efficiency = battery.charge_efficiency
        self.initial_soc = battery.initial_soc
        self.final_soc = battery.final_soc
        self.linear_cost = battery.linear_cost
        self.quadratic_cost = battery.quadratic_cost
        self.capital_cost = battery.capital_cost
        self.min_power_capacity = battery.min_power_capacity
        self.max_power_capacity = battery.max_power_capacity

    def model_local_variables(self, time_horizon: int) -> list[cp.Variable]:
        return [cp.Variable((self.num_devices, time_horizon))]

    def equality_constraints(self, power, angle, _, power_capacity=None, la=np, envelope=None):
        return []

    def inequality_constraints(self, power, angle, _, power_capacity=None, la=np, envelope=None):
        return []

    def operation_cost(self, power, angle, state, power_capacity=None, la=np, envelope=None):
        assert self.quadratic_cost is None
        power_capacity = self.parameterize(power_capacity=power_capacity, la=la)

        # Dual variables
        z = power[0]
        lamb = state[0]

        # Parameters
        rho = self.charge_efficiency
        pmax = power_capacity

        if not use_envelope(envelope):
            smax = pmax * self.duration

            # Charge and discharge terms
            c_term = la.multiply(-z - la.multiply(rho, lamb), pmax)
            d_term = la.multiply(lamb + z - self.linear_cost, pmax)

            # Energy terms
            s1_term = la.multiply(-lamb[:, [0]], self.initial_soc * smax)
            sT_term = la.multiply(lamb[:, [-1]], self.final_soc * smax)

            # Note - this should have (T-1) columns
            s_term = la.multiply(lamb[:, :-1] - lamb[:, 1:], smax)

        else:
            print("Envelope relaxation applied to dual battery.")
            env, lower, upper = envelope

            # Create envelope variables
            lb, ub = lower["power_capacity"], upper["power_capacity"]
            max_z = self.max_price

            z_pmax = envelope_variable(pmax, z, lb, ub, -max_z, max_z, *env)
            lamb_pmax = envelope_variable(pmax, lamb, lb, ub, -max_z, max_z, *env)
            lamb_smax = la.multiply(self.duration, lamb_pmax)

            # Charge and discharge terms
            # c = z * pmax - rho * lamb * lmax
            # d = lamb * pmax - z * pmax - c_lin * pmax
            c_term = z_pmax - la.multiply(rho, lamb_pmax)
            d_term = lamb_pmax - z_pmax - la.multiply(self.linear_cost, pmax)

            # Energy terms
            s1_term = la.multiply(-lamb_pmax[:, [0]], self.initial_soc)
            sT_term = la.multiply(lamb_pmax[:, [-1]], self.final_soc)
            s_term = lamb_smax[:, :-1] - lamb_smax[:, 1:]

        op_cost = la.sum(la.maximum(0.0, c_term))
        op_cost += la.sum(la.maximum(0.0, d_term))
        op_cost += la.sum(la.maximum(0.0, s_term))
        op_cost += la.sum(s1_term)
        op_cost += la.sum(sT_term)

        return op_cost
