import logging

import torch
import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from typing import Literal, Optional
from collections import namedtuple
from numpy.typing import NDArray

from zap.devices.abstract import AbstractDevice, make_dynamic

logger = logging.getLogger(__name__)

StorageUnitVariable = namedtuple(
    "StorageUnitVariable",
    [
        "energy",
        "charge",
        "discharge",
    ],
)


class StorageUnit(AbstractDevice):
    """An Injector that stores power between time steps.

    May have a discharge cost.

    Parameters
    ----------
    power_availability : Optional[NDArray]
        Dimensionless derate in [0, 1] on the charge and discharge power limits,
        of shape ``(N,)``, ``(N, 1)``, or ``(N, T)``. Defaults to all ones.
        Per decision D8 of the phase-1 spec this derates *power only*; the energy
        (state-of-charge) cap remains ``power_capacity * duration``, because
        deriving a time-varying energy cap can render a block infeasible when the
        state of charge at the moment of an outage exceeds the derated cap.

        Note: ``power_availability`` is expected to be set at construction only.
        The ADMM prox caches ``ymin``/``ymax`` behind ``self.has_changed``; any
        later mutation of ``power_availability`` MUST set ``has_changed = True``
        or ADMM will silently return a feasible-but-wrong dispatch.
    soc_mode : {"fixed", "cyclic_free"}
        Boundary condition on the state of charge of a block.

        * ``"fixed"`` (default, the historical behaviour) pins both endpoints:
          ``energy[:, 0] == initial_soc * E`` and ``energy[:, T] == final_soc * E``.
        * ``"cyclic_free"`` drops both pins and imposes ``energy[:, 0] ==
          energy[:, T]`` instead, so each block chooses its own level.  In this
          mode ``initial_soc`` and ``final_soc`` are **ignored** (they are still
          stored, and still used to initialize the ADMM prox iterate, but they
          constrain nothing).  The level stays inside ``[0, E]`` through the
          existing inequality constraints.

        A plain attribute, not a per-device array: the whole fleet shares one
        mode.  It is set at construction and survives ``sample_time``,
        ``torchify`` and ``scale_power``.  The ADMM prox rebuilds its cached
        data when it changes, but the cvx path reads it on every call.
    """

    #: Class-level default so that subclasses which build their state by hand
    #: (``zap.devices.dual.store.DualBattery``) still answer ``soc_mode``.
    soc_mode: str = "fixed"

    def __init__(
        self,
        *,
        num_nodes,
        name,
        terminal,
        power_capacity: NDArray,
        duration: NDArray,
        charge_efficiency: Optional[NDArray] = None,
        discharge_efficiency: Optional[NDArray] = None,
        initial_soc: Optional[NDArray] = None,
        final_soc: Optional[NDArray] = None,
        linear_cost: Optional[NDArray] = None,
        quadratic_cost: Optional[NDArray] = None,
        capital_cost: Optional[NDArray] = None,
        min_power_capacity=None,
        max_power_capacity=None,
        power_availability: Optional[NDArray] = None,
        soc_mode: Literal["fixed", "cyclic_free"] = "fixed",
    ):
        if linear_cost is None:
            linear_cost = np.zeros(power_capacity.shape)

        if charge_efficiency is None:
            charge_efficiency = np.ones(power_capacity.shape)

        if discharge_efficiency is None:
            discharge_efficiency = np.ones(power_capacity.shape)

        if initial_soc is None:
            initial_soc = 0.5 * np.ones(power_capacity.shape)

        if final_soc is None:
            final_soc = 0.5 * np.ones(power_capacity.shape)

        if power_availability is None:
            power_availability = np.ones(power_capacity.shape)

        if soc_mode not in ("fixed", "cyclic_free"):
            raise ValueError(f"soc_mode must be 'fixed' or 'cyclic_free', got {soc_mode!r}")

        self.num_nodes = num_nodes
        self.name = name
        self.terminal = terminal
        self.power_capacity = make_dynamic(power_capacity)
        self.duration = make_dynamic(duration)
        self.charge_efficiency = make_dynamic(charge_efficiency)
        self.discharge_efficiency = make_dynamic(discharge_efficiency)
        self.initial_soc = make_dynamic(initial_soc)
        self.final_soc = make_dynamic(final_soc)
        self.linear_cost = make_dynamic(linear_cost)
        self.quadratic_cost = make_dynamic(quadratic_cost)
        self.capital_cost = make_dynamic(capital_cost)
        self.min_power_capacity = make_dynamic(min_power_capacity)
        self.max_power_capacity = make_dynamic(max_power_capacity)
        self.power_availability = make_dynamic(power_availability)
        self.soc_mode = str(soc_mode)

        self.has_changed = True
        self.rho = -1.0

    @property
    def terminals(self):
        return self.terminal

    @property
    def time_horizon(self):
        # Static device unless the power availability derate is time-varying
        if self.power_availability.shape[1] == 1:
            return 0
        return self.power_availability.shape[1]

    def _effective_power(self, power_capacity, la=np):
        """Charge/discharge power limit after the availability derate."""
        return la.multiply(power_capacity, self.power_availability)

    def scale_costs(self, scale):
        self.linear_cost /= scale
        if self.quadratic_cost is not None:
            self.quadratic_cost /= scale
        if self.capital_cost is not None:
            self.capital_cost /= scale

    def scale_power(self, scale):
        # NOTE: power_availability is dimensionless and is deliberately not scaled.
        self.power_capacity /= scale
        if self.min_power_capacity is not None:
            self.min_power_capacity /= scale
        if self.max_power_capacity is not None:
            self.max_power_capacity /= scale

        # Invert scaling because term is quadratic
        if self.quadratic_cost is not None:
            self.quadratic_cost *= scale

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def model_local_variables(self, time_horizon: int) -> list[cp.Variable]:
        return StorageUnitVariable(
            cp.Variable((self.num_devices, time_horizon + 1)),
            cp.Variable((self.num_devices, time_horizon)),
            cp.Variable((self.num_devices, time_horizon)),
        )

    def equality_constraints(
        self,
        power,
        angle,
        state,
        power_capacity=None,
        initial_soc=None,
        final_soc=None,
        la=np,
        envelope=None,
    ):
        power_capacity = self.parameterize(power_capacity=power_capacity, la=la)
        initial_soc = self.parameterize(initial_soc=initial_soc, la=la)
        final_soc = self.parameterize(final_soc=final_soc, la=la)

        if not isinstance(state, StorageUnitVariable):
            state = StorageUnitVariable(*state)

        T = power[0].shape[1]
        energy_capacity = la.multiply(power_capacity, self.duration)

        # The windowed ADMM prox returns one SoC trajectory per window, so its
        # `energy` is wider than the LP layout and the recursion has to be
        # evaluated per window. The cvxpy path always hits `num_windows == 1`.
        num_windows = self.num_soc_windows(state.energy.shape[1], T)
        if num_windows > 1:
            return self._windowed_equality_constraints(
                power, state, energy_capacity, initial_soc, final_soc, T, num_windows, la
            )

        soc_evolution = (
            state.energy[:, :-1]
            + la.multiply(state.charge, self.charge_efficiency)
            - la.multiply(state.discharge, 1 / self.discharge_efficiency)
        )
        constraints = [
            power[0] - (state.discharge - state.charge),
            state.energy[:, 1:] - soc_evolution,
        ]

        if self.soc_mode == "cyclic_free":
            # Free cyclic boundary: the block starts wherever it ends. Both pins
            # are dropped, so `initial_soc` / `final_soc` are ignored here.
            constraints.append(state.energy[:, 0:1] - state.energy[:, T : (T + 1)])
        else:
            constraints.append(state.energy[:, 0:1] - la.multiply(initial_soc, energy_capacity))
            constraints.append(
                state.energy[:, T : (T + 1)] - la.multiply(final_soc, energy_capacity)
            )

        return constraints

    @staticmethod
    def num_soc_windows(energy_width: int, time_horizon: int) -> int:
        """How many SoC windows an ``energy`` array of this width represents.

        The LP model carries one trajectory of ``T + 1`` slots.  The ADMM prox run
        with ``battery_window = W`` returns ``S = T / W`` trajectories of ``W + 1``
        slots concatenated -- ``T + S`` columns in all, because every window carries
        its own extra boundary slot -- so ``S = energy_width - T`` inverts the layout.

        Raises
        ------
        ValueError
            If the width is neither layout, which would otherwise surface as an
            opaque broadcasting error inside the SoC recursion.
        """
        if energy_width == time_horizon + 1:
            return 1
        num_windows = energy_width - time_horizon
        if num_windows < 1 or time_horizon % num_windows != 0:
            raise ValueError(
                f"storage `energy` has {energy_width} columns, which is neither the LP "
                f"layout (T + 1 = {time_horizon + 1}) nor a windowed ADMM layout "
                f"(T + S columns for S equal windows dividing T = {time_horizon})"
            )
        return num_windows

    def _windowed_equality_constraints(
        self, power, state, energy_capacity, initial_soc, final_soc, T, num_windows, la
    ):
        """`equality_constraints` for the windowed ADMM prox layout.

        Each window is an independent SoC problem in the prox -- ``get_ymin_ymax``
        applies the same boundary rows to every window -- so the recursion, the
        `fixed` pins and the `cyclic_free` equality are all evaluated *per window*.
        Residuals come back flattened over windows, so a caller that only takes
        ``abs(...).max()`` (the ADMM feasibility gate) does not have to know the
        layout.
        """
        N = power[0].shape[0]
        S = num_windows
        W = T // S

        def per_window(x, width):
            return la.reshape(x, (N, S, width))

        def broadcast_param(param):
            # `make_dynamic` gives (N, 1) for a static parameter and (N, T) for a
            # time-varying one; the first broadcasts over windows, the second is
            # itself windowed.
            if param.shape[1] == 1:
                return la.reshape(param, (N, 1, 1))
            return per_window(param, W)

        charge = per_window(state.charge, W)
        discharge = per_window(state.discharge, W)
        energy = per_window(state.energy, W + 1)

        soc_evolution = (
            energy[:, :, :-1]
            + la.multiply(charge, broadcast_param(self.charge_efficiency))
            - la.multiply(discharge, 1 / broadcast_param(self.discharge_efficiency))
        )

        constraints = [
            power[0] - (state.discharge - state.charge),
            la.reshape(energy[:, :, 1:] - soc_evolution, (N, T)),
        ]

        if self.soc_mode == "cyclic_free":
            constraints.append(energy[:, :, 0] - energy[:, :, W])
        else:
            constraints.append(energy[:, :, 0] - la.multiply(initial_soc, energy_capacity))
            constraints.append(energy[:, :, W] - la.multiply(final_soc, energy_capacity))

        return constraints

    def inequality_constraints(
        self,
        power,
        angle,
        state,
        power_capacity=None,
        initial_soc=None,
        final_soc=None,
        la=np,
        envelope=None,
    ):
        power_capacity = self.parameterize(power_capacity=power_capacity, la=la)

        if not isinstance(state, StorageUnitVariable):
            state = StorageUnitVariable(*state)

        # Energy capacity is NOT derated by power_availability (see D8).
        energy_capacity = la.multiply(power_capacity, self.duration)
        p_eff = self._effective_power(power_capacity, la=la)

        return [
            -state.energy,
            state.energy - energy_capacity,
            -state.charge,
            state.charge - p_eff,
            -state.discharge,
            state.discharge - p_eff,
        ]

    def operation_cost(
        self,
        power,
        angle,
        state,
        power_capacity=None,
        initial_soc=None,
        final_soc=None,
        la=np,
        envelope=None,
    ):
        if state is None:
            return 0.0

        if not isinstance(state, StorageUnitVariable):
            state = StorageUnitVariable(*state)

        cost = la.sum(la.multiply(self.linear_cost, state.discharge))
        if self.quadratic_cost is not None:
            cost += la.sum(la.multiply(self.quadratic_cost, la.square(state.discharge)))

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
        self, equalities, power_capacity=None, initial_soc=None, final_soc=None, la=np
    ):
        # Dimensions
        size = equalities[0].power[0].shape[1]
        time_horizon = int(size / self.num_devices)
        shaped_zeros = np.zeros((self.num_devices, time_horizon))

        # Power balance
        equalities[0].power[0] += sp.eye(size)
        equalities[0].local_variables[1] += sp.eye(size)
        equalities[0].local_variables[2] += -sp.eye(size)

        # SOC evolution
        alpha = shaped_zeros + self.charge_efficiency
        beta = shaped_zeros + self.discharge_efficiency
        soc_diff = self._soc_difference_matrix(self.num_devices, time_horizon)

        equalities[1].local_variables[0] += soc_diff  # Energy
        equalities[1].local_variables[1] += -sp.diags(alpha.ravel())  # Charging
        equalities[1].local_variables[2] += sp.diags(1.0 / beta.ravel())  # Discharging

        # Initial / Final SOC
        first = self._soc_boundary_matrix(self.num_devices, time_horizon, index=0)
        last = self._soc_boundary_matrix(self.num_devices, time_horizon, index=-1)

        if self.soc_mode == "cyclic_free":
            # One row: energy[:, 0] - energy[:, T] == 0.
            equalities[2].local_variables[0] += first - last
        else:
            equalities[2].local_variables[0] += first
            equalities[3].local_variables[0] += last

        return equalities

    def _inequality_matrices(
        self, inequalities, power_capacity=None, initial_soc=None, final_soc=None, la=np
    ):
        size = inequalities[0].power[0].shape[1]
        e_size = inequalities[0].local_variables[0].shape[0]

        inequalities[0].local_variables[0] += -sp.eye(e_size)
        inequalities[1].local_variables[0] += sp.eye(e_size)
        inequalities[2].local_variables[1] += -sp.eye(size)
        inequalities[3].local_variables[1] += sp.eye(size)
        inequalities[4].local_variables[2] += -sp.eye(size)
        inequalities[5].local_variables[2] += sp.eye(size)

        return inequalities

    def _hessian_local_variables(
        self,
        hessians,
        power,
        angle,
        state,
        power_capacity=None,
        initial_soc=None,
        final_soc=None,
        la=np,
    ):
        if self.quadratic_cost is None:
            return hessians

        hessians[2] += 2 * sp.diags((self.quadratic_cost * state.discharge).ravel())
        return hessians

    # ====
    # PLANNING FUNCTIONS
    # ====

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        if dev.power_availability.shape[1] > 1:
            dev.power_availability = dev.power_availability[:, time_periods]
        if dev.linear_cost.shape[1] > 1:
            dev.linear_cost = dev.linear_cost[:, time_periods]

        # Sampled device has fresh prox data
        dev.has_changed = True

        return dev

    def get_investment_cost(self, power_capacity=None, initial_soc=None, final_soc=None, la=np):
        power_capacity = self.parameterize(power_capacity=power_capacity, la=la)

        if self.capital_cost is None or power_capacity is None:
            return 0.0

        # Get original nominal capacity and capital cost
        # Nominal capacity isn't passed here because we want to use the original value
        pnom_min = self.power_capacity
        capital_cost = self.capital_cost

        return la.sum(la.multiply(capital_cost, (power_capacity - pnom_min)))

    # ====
    # ADMM FUNCTIONS
    # ====
    def admm_prox_update(
        self,
        rho_power,
        rho_angle,
        power,
        angle,
        power_capacity=None,
        initial_soc=None,
        final_soc=None,
        power_weights=None,
        angle_weights=None,
        window=None,
        inner_weight=1.0,
        inner_over_relaxation=1.0,
        inner_iterations=25,
        inner_atol=1e-6,
    ):
        inner_weight = rho_power * inner_weight

        power_capacity = self.parameterize(power_capacity=power_capacity)
        initial_soc = self.parameterize(initial_soc=initial_soc)
        final_soc = self.parameterize(final_soc=final_soc)

        N, full_time_horizon = power[0].shape
        T = full_time_horizon if window is None else window  # Window size
        num_scenarios = full_time_horizon // T

        machine, dtype = power[0].device, power[0].dtype

        assert angle is None
        assert full_time_horizon % T == 0

        # Fixed data - constant between solves
        # Update: not constant if rho or inner_weight changes
        # So we update this once per solve
        # if not hasattr(self, "admm_data"):
        #     self.admm_data = battery_prox_data(self, T, rho_power, power[0], inner_weight)

        # Variable data that changes between solves
        # `soc_mode` changes the cached box bounds *and* the cached Schur
        # complement (the prox constraint matrix gains a row), so a mode change
        # must invalidate both, exactly like `has_changed`.
        mode = getattr(self, "soc_mode", "fixed")
        rebuild = self.has_changed or getattr(self, "_prox_soc_mode", None) != mode

        if rebuild:
            # print("Changing battery data.")

            smax = torch.multiply(power_capacity, self.duration)
            gamma1 = torch.multiply(initial_soc, smax)
            gammaT = torch.multiply(final_soc, smax)

            # Charge/discharge bound, possibly time-varying (N, num_scenarios * T)
            pmax_t = torch.multiply(power_capacity, self.power_availability)
            if pmax_t.shape[1] == 1:
                pmax_t = pmax_t.expand(-1, full_time_horizon)
            else:
                assert pmax_t.shape[1] == full_time_horizon, (
                    "Time-varying power_availability must span the full time horizon: "
                    f"got {pmax_t.shape[1]}, expected {full_time_horizon}"
                )

            ymin, ymax = get_ymin_ymax(
                T, num_scenarios, pmax_t, smax, gamma1, gammaT, machine, dtype, soc_mode=mode
            )
            A = A_matrix(T, machine, dtype=dtype)
            b = b_vector(self, T, num_scenarios, machine, dtype=dtype)

            _zT = power[0].reshape(-1, num_scenarios, T, 1)
            _rhs = K_rhs_fixed(rho_power, A, b, _zT)
            num_eq_rows = T + 1 if mode == "cyclic_free" else T
            zero_nu = torch.zeros(
                (_rhs.shape[0], _rhs.shape[1], num_eq_rows, _rhs.shape[3]),
                device=machine,
                dtype=dtype,
            )

            self.temp_data = (smax, gamma1, gammaT, ymin, ymax, A, b, zero_nu)

        if rebuild or self.rho != rho_power:
            # print("Updating battery Schur matrix.")
            self.rho = rho_power
            self.schur = schur_matrix(self, T, rho_power, inner_weight, machine)
            # _K = K_matrix(self, T, rho_power, inner_weight, machine)
            # self.K_inv = torch.linalg.inv(_K)

        self._prox_soc_mode = mode
        self.has_changed = False

        # schur = self.schur
        # K_inv = self.K_inv
        schur = self.schur
        smax, gamma1, gammaT, ymin, ymax, A, b, zero_nu = self.temp_data

        # Changes every proximal evaluation
        zT = power[0].reshape(-1, num_scenarios, T, 1)
        rhs = K_rhs_fixed(rho_power, A, b, zT)

        # Initialize
        x = torch.zeros((N, num_scenarios, 3 * T + 1, 1), device=machine, dtype=dtype)
        y = torch.zeros(x.shape, device=machine, dtype=dtype)
        u = torch.zeros(x.shape, device=machine, dtype=dtype)

        y[:, :, :T, :] = torch.relu(-zT)
        y[:, :, T : (2 * T), :] = torch.relu(zT)
        y[:, :, (2 * T) :, :] = gamma1.reshape(-1, 1, 1, 1)
        y = torch.clip(y, min=ymin, max=ymax)

        # Solve ADMM
        for iter in range(inner_iterations):
            x, y, u = battery_prox_inner(
                x, y, u, rhs, schur, ymin, ymax, inner_weight, inner_over_relaxation
            )

        # Inner stopping test. `x` satisfies the SoC recursion and `y` the box, so
        # ||x - y||_inf is the inner solve's own primal residual, in MW / MWh: the
        # two iterates agree only at convergence. An unconverged prox is silent
        # otherwise -- the outer loop can declare convergence on a dispatch whose SoC
        # recursion is violated by tens of MWh (measured at inner_iterations <= 25) --
        # so *record* it for the caller.  The device deliberately does not judge it:
        # whether the residual matters depends on the outer nodal imbalance, which
        # only the solver can see, so `ADMMSolver.warn_inner_prox` does the warning.
        # `inner_atol` is kept as the inner stopping test's tolerance (A5).
        inner_residual = float(torch.max(torch.abs(x - y)).item())
        self.last_admm_inner_residual = inner_residual

        # Extract results from the *projected* iterate `y`, not from `x`: `y` is
        # the copy that lives in the box, so the returned charge / discharge always
        # satisfy the device's own power bounds (and the SoC its energy bounds and
        # boundary conditions) at any inner iteration count. `x` satisfies the SoC
        # recursion exactly instead; the two agree as the inner solver converges.
        c = y[:, :, :T, 0].reshape(N, -1)
        d = y[:, :, T : (2 * T), 0].reshape(N, -1)
        s = y[:, :, (2 * T) :, 0].reshape(N, -1)

        # (power, angle, local_variables) -- the ADMM solver unpacks three values.
        # NOTE: with a battery window (num_scenarios > 1) the energy variable has
        # num_scenarios * (T + 1) columns, one SoC trajectory per window, and is
        # therefore not the (N, time_horizon + 1) shape the LP model uses.
        local_variables = StorageUnitVariable(s, c, d)
        return [d - c], None, local_variables


# ====
# ADMM UTILITY FUNCTIONS
# ====


def difference_matrix(T, machine=None, dtype=None):
    # Should return a (T, T+1) matrix where
    # D = [-1 1 0 0]
    #     [0 -1 1 0]
    #     [0 0 -1 1]
    # for T = 3

    D1 = torch.eye(T + 1, device=machine, dtype=dtype)[0:T, :]
    D2 = torch.eye(T + 1, device=machine, dtype=dtype)[1:, :]

    return D2 - D1


def b_vector(device: StorageUnit, T, num_scenarios=1, machine=None, dtype=None):
    """Linear cost of the prox variable ``x = [charge; discharge; energy]``.

    ``StorageUnit.operation_cost`` charges ``linear_cost * discharge``, so the cost
    sits on the ``x[T:2T]`` block -- one coefficient *per unit* (and per hour when
    ``linear_cost`` is time-varying), not row 0's coefficient for the whole fleet.

    Returns
    -------
    torch.Tensor of shape ``(N, S, 3T + 1, 1)`` with ``S == 1`` for a static cost
    and ``S == num_scenarios`` for a time-varying one.
    """
    if dtype is None:
        dtype = device.power_capacity.dtype

    cost = device.linear_cost
    N = cost.shape[0]

    if cost.shape[1] == 1:
        alpha = cost.reshape(N, 1, 1).expand(N, 1, T)
        S = 1
    else:
        assert cost.shape[1] == num_scenarios * T, (
            "Time-varying linear_cost must span the full time horizon: "
            f"got {cost.shape[1]}, expected {num_scenarios * T}"
        )
        alpha = cost.reshape(N, num_scenarios, T)
        S = num_scenarios

    b = torch.zeros((N, S, 3 * T + 1, 1), device=machine, dtype=dtype)
    b[:, :, T : (2 * T), 0] = alpha.to(dtype=dtype)
    return b


def C_matrix(device: StorageUnit, T, machine=None, dtype=None):
    """Per-unit state-of-charge dynamics ``C_i x_i = 0``, batched over units.

    Mirrors ``StorageUnit.equality_constraints``:
    ``s[t+1] - s[t] - beta_i * c[t] + d[t] / eta_i == 0``, i.e.
    ``C_i = [-beta_i * I, (1 / eta_i) * I, D]`` with ``D`` the (T, T+1) forward
    difference. Both efficiencies are per unit.

    In ``soc_mode == "cyclic_free"`` one further row ``s_0 - s_T == 0`` is
    appended, which is how the free cyclic boundary condition enters the prox:
    the endpoints are no longer pinned in ``ymin`` / ``ymax``, only tied to each
    other here.

    Returns
    -------
    torch.Tensor of shape ``(N, T, 3T + 1)``, or ``(N, T + 1, 3T + 1)`` in
    ``soc_mode == "cyclic_free"``.
    """
    if dtype is None:
        dtype = device.power_capacity.dtype

    beta = device.charge_efficiency
    eta = device.discharge_efficiency
    assert beta.shape[1] == 1 and eta.shape[1] == 1, (
        "Time-varying charge/discharge efficiencies are not supported by the ADMM prox"
    )
    beta = beta[:, 0].to(dtype=dtype).reshape(-1, 1, 1)
    inv_eta = (1.0 / eta[:, 0]).to(dtype=dtype).reshape(-1, 1, 1)

    D = difference_matrix(T, machine, dtype)
    Id = torch.eye(T, device=machine, dtype=dtype)

    N = beta.shape[0]
    C = torch.zeros((N, T, 3 * T + 1), device=machine, dtype=dtype)
    C[:, :, :T] = -beta * Id
    C[:, :, T : (2 * T)] = inv_eta * Id
    C[:, :, (2 * T) :] = D

    if getattr(device, "soc_mode", "fixed") == "cyclic_free":
        cyclic = torch.zeros((N, 1, 3 * T + 1), device=machine, dtype=dtype)
        cyclic[:, 0, 2 * T] = 1.0  # s_0
        cyclic[:, 0, 3 * T] = -1.0  # s_T
        C = torch.cat([C, cyclic], dim=1)

    return C


def A_matrix(T, machine=None, dtype=None):
    Id = torch.eye(T, device=machine, dtype=dtype)
    return torch.hstack([-Id, Id, torch.zeros(T, T + 1, device=machine, dtype=dtype)])


def _hessian(device: StorageUnit, T, rho, w, machine=None, dtype=None):
    """``H = rho A'A + w I`` plus the quadratic discharge cost, batched if needed."""
    if dtype is None:
        dtype = device.power_capacity.dtype

    A = A_matrix(T, machine, dtype)
    Id = torch.eye(3 * T + 1, device=machine, dtype=dtype)
    H = rho * (A.T @ A) + w * Id

    if device.quadratic_cost is not None:
        # operation_cost charges quadratic_cost * discharge^2, whose Hessian is
        # 2 * quadratic_cost on the discharge block x[T:2T] (per unit).
        q = device.quadratic_cost
        q = q[:, 0] if q.dim() == 2 else q.reshape(-1)
        N = q.shape[0]
        H = H.expand(N, 3 * T + 1, 3 * T + 1).clone()
        idx = torch.arange(T, device=machine) + T
        H[:, idx, idx] += 2.0 * q.to(dtype=dtype).reshape(N, 1)

    return H


def K_matrix(device: StorageUnit, T, rho, w, machine=None):
    """KKT matrix of the prox subproblem, batched over units.

    ``(N, 4T+1, 4T+1)``, or ``(N, 4T+2, 4T+2)`` in ``soc_mode == "cyclic_free"``.
    """
    dtype = device.power_capacity.dtype

    C = C_matrix(device, T, machine, dtype)
    H = _hessian(device, T, rho, w, machine, dtype)
    N = C.shape[0]

    H = H.expand(N, 3 * T + 1, 3 * T + 1) if H.dim() == 2 else H
    CT = C.transpose(-1, -2)

    m = C.shape[1]  # number of equality rows (T, or T + 1 when cyclic_free)
    row1 = torch.cat([H, CT], dim=-1)
    row2 = torch.cat([C, torch.zeros((N, m, m), device=machine, dtype=dtype)], dim=-1)

    return torch.cat([row1, row2], dim=-2)


def schur_matrix(device, T, rho, w, machine=None):
    """Per-unit Schur complement mapping the prox right-hand side to ``x``.

    ``x_i = S_i @ rhs_i`` solves ``[[H, C_i'], [C_i, 0]] [x; nu] = [rhs; 0]``, i.e.
    ``S_i = H^-1 - (C_i H^-1)' (C_i H^-1 C_i')^-1 (C_i H^-1)``.

    Returns
    -------
    torch.Tensor of shape ``(N, 1, 3T + 1, 3T + 1)``, broadcasting over the
    scenario axis of the batched inner solve.
    """
    dtype = device.power_capacity.dtype

    C = C_matrix(device, T, machine, dtype)
    H = _hessian(device, T, rho, w, machine, dtype)

    H_inv = torch.linalg.inv(H)  # (3T+1, 3T+1) or (N, 3T+1, 3T+1)

    # Reduced terms
    Q = C @ H_inv  # (N, T, 3T+1)
    M = Q @ C.transpose(-1, -2)  # (N, T, T)

    # Schur complement, batched over units
    S = H_inv - Q.transpose(-1, -2) @ torch.linalg.inv(M) @ Q
    return S.unsqueeze(1)


def K_rhs_fixed(rho, A, b, z):
    # rho * A.T @ z - b
    return rho * torch.matmul(A.T, z) - b


@torch.jit.script
def battery_prox_inner(x, y, u, rhs, schur, ymin, ymax, w: float, alpha: float = 1.0):
    # x update
    rhs_var = rhs + w * (y - u)

    # full_rhs = torch.cat([rhs_var, zero_nu], dim=2)
    # x = (K_inv @ full_rhs)[:, :, : x.shape[2], :]
    x = schur @ rhs_var

    # over relaxation step
    xp = alpha * x + (1 - alpha) * y

    # y update
    y = torch.clip(xp + u, min=ymin, max=ymax)

    # u update
    u += xp - y

    return x, y, u


def battery_prox_data(device: StorageUnit, T: int, rho, z, weight=1.0):
    machine = z.device

    T_full = z.shape[1]
    num_scenarios = T_full // T
    assert T * num_scenarios == T_full

    # Fixed data that does not change between solves
    dtype = device.power_capacity.dtype
    K = K_matrix(device, T, rho, weight, machine)
    K_inv = torch.linalg.inv(K)

    # zT and rhs are just created to get the dimensions for zero_nu
    zT = z.reshape(-1, num_scenarios, T, 1)
    rhs = K_rhs_fixed(
        rho,
        A_matrix(T, machine, dtype),
        b_vector(device, T, num_scenarios, machine, dtype),
        zT,
    )
    zero_nu = torch.zeros((rhs.shape[0], rhs.shape[1], T, rhs.shape[3]), device=machine)

    return K_inv, zero_nu


def get_ymin_ymax(
    T, num_scenarios, pmax_t, smax, gamma1, gammaT, machine=None, dtype=None, soc_mode="fixed"
):
    """Build the box bounds of the battery prox variable y = (charge, discharge, energy).

    Parameters
    ----------
    T : int
        Window (scenario) length.
    num_scenarios : int
        Number of windows the full horizon is split into.
    pmax_t : torch.Tensor
        Charge/discharge power bound of shape ``(N, num_scenarios * T)``. Callers
        broadcast a static ``(N, 1)`` bound before calling.
    soc_mode : {"fixed", "cyclic_free"}
        ``"fixed"`` pins the first and last energy slot to ``gamma1`` / ``gammaT``.
        ``"cyclic_free"`` leaves both free in ``[0, smax]``; the cyclic condition
        ``s_0 == s_T`` is imposed by the extra row of :func:`C_matrix` instead.

    Returns
    -------
    (ymin, ymax) each of shape ``(N, num_scenarios, 3 * T + 1, 1)``.
    """
    N = pmax_t.shape[0]
    assert pmax_t.shape[1] == num_scenarios * T

    pc = pmax_t.reshape(N, num_scenarios, T)  # Charge / discharge bounds
    pe = smax.reshape(N, 1, 1).expand(N, num_scenarios, T + 1)  # Energy bounds

    ymax = torch.cat([pc, pc.clone(), pe], dim=2)  # (N, num_scenarios, 3T + 1)
    ymin = torch.zeros_like(ymax)

    if soc_mode != "cyclic_free":
        ymin[:, :, 2 * T] = gamma1[:, 0:1]
        ymax[:, :, 2 * T] = gamma1[:, 0:1]
        ymin[:, :, -1] = gammaT[:, 0:1]
        ymax[:, :, -1] = gammaT[:, 0:1]

    ymin = ymin.unsqueeze(-1)
    ymax = ymax.unsqueeze(-1)

    return ymin, ymax
