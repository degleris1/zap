import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from typing import Optional
from numpy.typing import NDArray

from zap.devices.abstract import AbstractDevice, make_dynamic


class Store(AbstractDevice):
    """An Injector that stores power between time steps.

    May have a discharge cost.
    """

    def __init__(
        self,
        *,
        num_nodes,
        name,
        terminal,
        nominal_energy_capacity: NDArray,
        min_energy_capacity_availability: Optional[NDArray] = None,
        max_energy_capacity_availability: Optional[NDArray] = None,
        standing_loss: Optional[NDArray] = None,
        initial_soc: Optional[NDArray] = None,
        final_soc: Optional[NDArray] = None,
        linear_cost: Optional[NDArray] = None,
        quadratic_cost: Optional[NDArray] = None,
        linear_storage_cost: Optional[NDArray] = None,
        capital_cost: Optional[NDArray] = None,
        min_nominal_energy_capacity=None,
        max_nominal_energy_capacity=None,
    ):
        if linear_storage_cost is None:
            linear_storage_cost = np.zeros(nominal_energy_capacity.shape)

        if standing_loss is None:
            standing_loss = np.ones(nominal_energy_capacity.shape)

        if min_energy_capacity_availability is None:
            min_energy_capacity_availability = np.zeros(nominal_energy_capacity.shape)

        if max_energy_capacity_availability is None:
            max_energy_capacity_availability = np.ones(nominal_energy_capacity.shape)

        if initial_soc is None:
            initial_soc = 0.5 * np.ones(nominal_energy_capacity.shape)

        if final_soc is None:
            final_soc = 0.5 * np.ones(nominal_energy_capacity.shape)

        self.num_nodes = num_nodes
        self.name = name
        self.terminal = terminal
        self.nominal_energy_capacity = make_dynamic(nominal_energy_capacity)
        self.min_energy_capacity_availability = make_dynamic(
            min_energy_capacity_availability
        )
        self.max_energy_capacity_availability = make_dynamic(
            max_energy_capacity_availability
        )
        self.standing_loss = make_dynamic(standing_loss)
        self.initial_soc = make_dynamic(initial_soc)
        self.final_soc = make_dynamic(final_soc)
        self.linear_cost = make_dynamic(linear_cost)
        self.quadratic_cost = make_dynamic(quadratic_cost)
        self.linear_storage_cost = make_dynamic(linear_storage_cost)
        self.capital_cost = make_dynamic(capital_cost)
        self.min_nominal_energy_capacity = make_dynamic(min_nominal_energy_capacity)
        self.max_nominal_energy_capacity = make_dynamic(max_nominal_energy_capacity)
        self.has_changed = True
        self.rho = -1.0
        # helper for setting cyclical charging

    @property
    def terminals(self):
        return self.terminal

    @property
    def time_horizon(self):
        """
        Returns the time horizon of the device.
        0 if static, otherwise the maximum time horizon of the device.
        """
        if (self.min_nominal_energy_capacity.shape[1] == 1) and (
            self.max_nominal_energy_capacity.shape[1] == 1
        ):
            return 0
        else:
            return max(
                self.min_nominal_energy_capacity.shape[1],
                self.max_nominal_energy_capacity.shape[1],
            )

    def scale_costs(self, scale):
        self.linear_cost /= scale
        self.linear_storage_cost /= scale
        if self.quadratic_cost is not None:
            self.quadratic_cost /= scale
        if self.capital_cost is not None:
            self.capital_cost /= scale

    def scale_power(self, scale):
        self.nominal_energy_capacity /= scale
        if self.min_nominal_energy_capacity is not None:
            self.min_nominal_energy_capacity /= scale
        if self.max_nominal_energy_capacity is not None:
            self.max_nominal_energy_capacity /= scale

        # Invert scaling because term is quadratic
        if self.quadratic_cost is not None:
            self.quadratic_cost *= scale

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def model_local_variables(self, time_horizon: int) -> list[cp.Variable]:
        return [cp.Variable((self.num_devices, time_horizon + 1))]

    def equality_constraints(
        self,
        power,
        angle,
        SOC,
        nominal_energy_capacity=None,
        initial_soc=None,
        final_soc=None,
        la=np,
        envelope=None,
    ):
        SOC = SOC[0]
        nominal_energy_capacity = self.parameterize(
            nominal_energy_capacity=nominal_energy_capacity, la=la
        )
        initial_soc = self.parameterize(initial_soc=initial_soc, la=la)
        final_soc = self.parameterize(final_soc=final_soc, la=la)

        T = power[0].shape[1]

        return [  # eq 0
            # soc[t+1] = soc[t] - power[t] - standing_loss[t],   (power[t] is negative because it is discharge)
            SOC[:, 1:] - SOC[:, :-1] + power[0] + self.standing_loss[:, 1:],
            # initial/final soc constraints
            SOC[:, 0:1] - la.multiply(initial_soc, nominal_energy_capacity),
            SOC[:, T : (T + 1)] - la.multiply(final_soc, nominal_energy_capacity),
        ]

    def inequality_constraints(
        self,
        _1,
        _2,
        SOC,
        nominal_energy_capacity=None,
        initial_soc=None,
        final_soc=None,
        la=np,
        envelope=None,
    ):
        SOC = SOC[0]
        nominal_energy_capacity = self.parameterize(
            nominal_energy_capacity=nominal_energy_capacity, la=la
        )

        max_energy = la.multiply(
            nominal_energy_capacity, self.max_energy_capacity_availability
        )
        min_energy = la.multiply(
            nominal_energy_capacity, self.min_energy_capacity_availability
        )

        return [  # leq 0
            SOC - max_energy,
            min_energy - SOC,
        ]

    def operation_cost(
        self,
        power,
        _,
        state,
        nominal_energy_capacity=None,
        initial_soc=None,
        final_soc=None,
        la=np,
        envelope=None,
    ):
        if state is None:
            return 0.0

        cost = la.sum(la.multiply(self.linear_cost, power[0]))
        cost += la.sum(la.multiply(self.linear_storage_cost, state.discharge))

        if self.quadratic_cost is not None:
            cost += la.sum(la.multiply(self.quadratic_cost, la.square(power[0])))

        return cost

    # ====
    # DIFFERENTIATION
    # ====

    def _soc_boundary_matrix(self, num_devices, time_horizon, index=0):
        soc_first = np.zeros((num_devices, time_horizon + 1))
        soc_first[:, index] = 1.0

        cols = sp.diags(soc_first.ravel(), format="coo").col
        rows = np.arange(num_devices)
        values = np.ones(len(rows))
        shape = (num_devices, num_devices * (time_horizon + 1))

        return sp.coo_matrix((values, (rows, cols)), shape=shape)

    def _soc_difference_matrix(self, num_devices, time_horizon):
        empty = np.zeros((num_devices, time_horizon + 1))

        last_soc = empty.copy()
        last_soc[:, :-1] = -1.0

        next_soc = empty.copy()
        next_soc[:, 1:] = 1.0

        c1 = sp.diags(last_soc.ravel(), format="coo")
        c2 = sp.diags(next_soc.ravel(), format="coo")
        r = np.arange(num_devices * time_horizon)

        cols = np.concatenate([c1.col, c2.col])
        rows = np.concatenate([r, r])
        values = np.concatenate([c1.data, c2.data])
        shape = (num_devices * time_horizon, num_devices * (time_horizon + 1))

        return sp.coo_matrix((values, (rows, cols)), shape=shape)

    def _equality_matrices(
        self,
        equalities,
        nominal_energy_capacity=None,
        initial_soc=None,
        final_soc=None,
        la=np,
    ):
        # Dimensions
        size = equalities[0].power[0].shape[1]
        time_horizon = int(size / self.num_devices)
        # shaped_zeros = np.zeros((self.num_devices, time_horizon))

        # SOC evolution
        # alpha = shaped_zeros + self.charge_efficiency
        soc_diff = self._soc_difference_matrix(self.num_devices, time_horizon)

        equalities[0].local_variables[0] += soc_diff  # Energy
        equalities[0].power[0] = sp.eye(size)  # Charge / discharge

        # Initial / Final SOC
        equalities[1].local_variables[0] += self._soc_boundary_matrix(
            self.num_devices, time_horizon, index=0
        )
        equalities[2].local_variables[0] += self._soc_boundary_matrix(
            self.num_devices, time_horizon, index=-1
        )

        return equalities

    def _inequality_matrices(
        self,
        inequalities,
        nominal_energy_capacity=None,
        initial_soc=None,
        final_soc=None,
        la=np,
    ):
        e_size = inequalities[0].local_variables[0].shape[0]

        inequalities[0].local_variables[0] += sp.eye(e_size)
        inequalities[1].local_variables[0] += -sp.eye(e_size)

        return inequalities

    def _hessian_power(
        self,
        hessians,
        power,
        angle,
        local_vars,
        nominal_energy_capacity=None,
        initial_soc=None,
        final_soc=None,
        la=np,
    ):
        if self.quadratic_cost is None:
            return hessians

        hessians[0] += 2 * sp.diags((self.quadratic_cost * power[0]).ravel())

        return hessians

    # ====
    # PLANNING FUNCTIONS
    # ====

    def get_investment_cost(
        self, nominal_energy_capacity=None, initial_soc=None, final_soc=None, la=np
    ):
        nominal_energy_capacity = self.parameterize(
            nominal_energy_capacity=nominal_energy_capacity, la=la
        )

        if self.capital_cost is None or nominal_energy_capacity is None:
            return 0.0

        # Get original nominal capacity and capital cost
        # Nominal capacity isn't passed here because we want to use the original value
        pnom_min = self.nominal_energy_capacity
        capital_cost = self.capital_cost

        return la.sum(la.multiply(capital_cost, (nominal_energy_capacity - pnom_min)))
