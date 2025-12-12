import dataclasses
import functools
import warnings
from typing import Optional

import torch
import numpy as np

from zap.network import DispatchOutcome
from zap.devices import Battery
from zap.devices.abstract import AbstractDevice
from zap.util import infer_machine
from zap.admm.util import (
    nested_subtract,
    nested_norm,
    nested_bpax,
    nested_a1bpa2x,
    nested_ax,
    get_num_terminals,
    dc_average,
    ac_average,
    get_terminal_residual,
    apply_incidence_transpose,
    unsqueeze_terminals_times_x,
)


@dataclasses.dataclass
class ADMMState:
    num_terminals: object
    num_ac_terminals: object
    power: object
    phase: object
    dual_power: object
    dual_phase: object
    avg_power: object = None
    avg_phase: object = None
    resid_power: object = None
    resid_phase: object = None
    objective: object = None
    clone_power: object = None
    clone_phase: object = None
    rho_power: object = None
    rho_angle: object = None
    local_variables: object = None

    def update(self, **kwargs):
        """Return a new state with fields updated."""
        return dataclasses.replace(self, **kwargs)

    @functools.cached_property
    def power_weights(self):
        return [None for _ in self.power]

    @functools.cached_property
    def angle_weights(self):
        return [None for _ in self.phase]

    def copy(self):
        return ADMMState(
            num_terminals=self.num_terminals,
            num_ac_terminals=self.num_ac_terminals,
            power=[[pi.clone().detach() for pi in p] for p in self.power],
            phase=[None if v is None else [vi.clone().detach() for vi in v] for v in self.phase],
            dual_power=self.dual_power.clone().detach(),
            dual_phase=[
                None if v is None else [vi.clone().detach() for vi in v] for v in self.dual_phase
            ],
            avg_power=self.avg_power.clone().detach(),
            avg_phase=self.avg_phase.clone().detach(),
            resid_power=[[pi.clone().detach() for pi in p] for p in self.resid_power],
            resid_phase=[
                None if v is None else [vi.clone().detach() for vi in v] for v in self.resid_phase
            ],
            objective=self.objective,
            clone_power=[[pi.clone().detach() for pi in p] for p in self.clone_power],
            clone_phase=self.clone_phase.clone().detach(),
            rho_power=self.rho_power,
            rho_angle=self.rho_angle,
            local_variables=self.local_variables,
        )

    def as_outcome(self) -> DispatchOutcome:
        return DispatchOutcome(
            phase_duals=nested_ax(self.dual_phase, self.rho_angle),
            local_equality_duals=None,
            local_inequality_duals=None,
            local_variables=self.local_variables,
            power=self.power,
            angle=self.phase,
            prices=-self.rho_power * self.dual_power,
            global_angle=None,
            problem=None,
            ground=None,
        )


@dataclasses.dataclass()
class ADMMSolver:
    """Stores ADMM solver parameters and exposes a solve function."""

    machine: str = None
    dtype: object = torch.float32
    num_iterations: int = 10000
    rho_power: float = 1.0
    rho_angle: Optional[float] = 1.0
    alpha: float = 1.0
    atol: float = 1e-5
    rtol: float = 1e-5
    rtol_primal: Optional[float] = None
    rtol_dual: Optional[float] = None
    dual_bias: float = 1.0
    rtol_dual_use_objective: bool = True
    resid_norm: int = 2
    safe_mode: bool = False
    track_objective: bool = True
    battery_window: Optional[int] = None
    battery_inner_weight: float = 1.0
    battery_inner_over_relaxation: float = 1.8
    battery_inner_iterations: int = 10
    minimum_iterations: int = 10
    relative_rho_angle: bool = False
    adaptive_rho: bool = True
    tau: float = 1.1
    adaptation_tolerance: float = 2.0
    adaptation_frequency: int = 50
    verbose: int = 1
    scale_dual_residuals: bool = None  # Deprecated

    def __post_init__(self):
        if self.machine is None:
            # Infer machine
            self.machine = infer_machine()

        if self.battery_window == 0:
            self.battery_window = None

        self.cumulative_iteration = 0

        if self.scale_dual_residuals is not None:
            warnings.warn(
                "scale_dual_residuals is deprecated and will be removed in a future release."
            )
        if isinstance(self.verbose, bool):
            self.verbose = 3 if self.verbose else 0
            warnings.warn(
                "The verbose parameter should be an integer. Setting to 3 (max verbosity) if True, 0 if False."
            )

    def get_rho(self):
        rho_power = self.rho_power
        rho_angle = self.rho_angle

        if self.relative_rho_angle and self.rho_angle is not None:
            rho_angle = rho_angle * rho_power

        elif rho_angle is None:
            rho_angle = rho_power

        return rho_power, rho_angle

    def solve(
        self,
        net,
        devices: list[AbstractDevice],
        time_horizon,
        *,
        parameters=None,
        nu_star=None,
        initial_state=None,
        num_contingencies=0,
        contingency_device: Optional[int] = None,
        contingency_mask=None,
        **kwargs,
    ):
        if num_contingencies > 0:
            assert contingency_device is not None
            assert contingency_mask is not None
            assert contingency_mask.shape == (
                num_contingencies + 1,
                devices[contingency_device].num_devices,
            )

        # Override algorithm settings
        original_settings = {}
        for k, v in kwargs.items():
            original_settings[k] = getattr(self, k)
            setattr(self, k, v)

        if parameters is None:
            parameters = [{} for _ in devices]

        # Initialize
        self.num_dc_terminals = time_horizon * sum(
            d.num_devices * d.num_terminals_per_device for d in devices
        )
        self.num_ac_terminals = time_horizon * sum(
            d.num_devices * d.num_terminals_per_device for d in devices if d.is_ac
        )
        self.total_terminals = self.num_dc_terminals + self.num_ac_terminals
        history = self.initialize_history()

        if initial_state is None:
            st = self.initialize_solver(
                net, devices, time_horizon, num_contingencies, contingency_device
            )
        else:
            st = initial_state

        for d in devices:
            d.has_changed = True

        if self.verbose >= 2:
            print("Initial value of rho_power:", self.rho_power)
            print("Initial value of rho_angle:", self.rho_angle)

        for iteration in range(self.num_iterations):
            self.iteration = iteration + 1
            self.cumulative_iteration += 1

            # (1) Device proximal updates
            st = self.device_updates(
                st, devices, parameters, num_contingencies, contingency_device, contingency_mask
            )

            # (2) Update averages and residuals
            last_avg_phase = st.avg_phase
            last_resid_power = st.resid_power
            st = self.update_averages_and_residuals(
                st, net, devices, time_horizon, num_contingencies
            )

            # (3) Update scaled prices
            st = self.price_updates(st, net, devices, time_horizon)

            # (4) Hisory, convergence checks, numerical checks
            if self.track_objective:
                st = st.update(objective=self.compute_objective(st, devices, parameters))

            self.update_history(history, st, last_avg_phase, last_resid_power, nu_star)

            self.converged = self.has_converged(st, history, num_contingencies)
            if iteration + 1 >= self.minimum_iterations and self.converged:
                print(f"ADMM converged in {len(history.power)} iterations.")
                break  # Quit early

            if self.adaptive_rho and self.iteration % self.adaptation_frequency == 0:
                if self.relative_rho_angle:
                    st = self.adjust_rho(st, history)
                else:
                    st = self.adjust_rho_power_and_angle(st, history)

            if self.safe_mode:
                self.dimension_checks(st, net, devices, time_horizon)
                self.numerical_checks(st, net, devices, time_horizon)

        if not self.converged and self.verbose >= 1:
            print(f"Did not converge. Ran for {self.iteration} iterations.")

        if self.verbose >= 2:
            print("Final value of rho_power:", self.rho_power)
            print("Final value of rho_angle:", self.rho_angle)

        # Restore original settings
        for k, v in original_settings.items():
            setattr(self, k, v)

        return st, history

    # ====
    # Update Rules
    # ====

    def device_updates(
        self,
        st: ADMMState,
        devices,
        parameters,
        num_contingencies,
        contingency_device,
        contingency_mask,
    ):
        for i, dev in enumerate(devices):
            rho_power, rho_angle = self.get_rho()

            set_p = self.set_power(dev, i, st, num_contingencies)
            set_v = self.set_phase(dev, i, st, num_contingencies, contingency_device)

            w_p = st.power_weights[i]
            w_v = st.angle_weights[i]

            if isinstance(dev, Battery):
                kwargs = {
                    "window": self.battery_window,
                    "inner_weight": self.battery_inner_weight,
                    "inner_over_relaxation": self.battery_inner_over_relaxation,
                    "inner_iterations": self.battery_inner_iterations,
                }
            else:
                kwargs = {}

            if num_contingencies > 0 and i == contingency_device:
                # TODO Figure out this scaling
                kwargs["contingency_mask"] = contingency_mask
                # rho_power = rho_power / (num_contingencies + 1)
                # rho_angle = rho_angle / (num_contingencies + 1)

            if num_contingencies > 0 and i != contingency_device:
                rho_power = rho_power * (num_contingencies + 1)
                rho_angle = rho_angle * (num_contingencies + 1)

            p, v, lv = dev.admm_prox_update(
                rho_power,
                rho_angle,
                set_p,
                set_v,
                power_weights=w_p,
                angle_weights=w_v,
                **parameters[i],
                **kwargs,
            )
            st.power[i] = p
            st.phase[i] = v
            st.local_variables[i] = lv

        return st

    def set_power(self, dev: AbstractDevice, dev_index: int, st: ADMMState, nc: int):
        AT_nu = apply_incidence_transpose(dev, st.dual_power)

        # This is a non-contingency device in a contingency-constrained problem
        # We need to aggregrate the contingeny terms
        if nc > 0 and st.power[dev_index][0].dim() == 2:
            AT_nu = [torch.mean(A, dim=-1) for A in AT_nu]
            clones = [torch.mean(v, dim=-1) for v in st.clone_power[dev_index]]

            return [zi - AT_nu_i for zi, AT_nu_i in zip(clones, AT_nu)]

        # Normal case (or contingency device in a contingency-constrained problem)
        else:
            return [zi - AT_nu_i for zi, AT_nu_i in zip(st.clone_power[dev_index], AT_nu)]

        # return [
        #     p - Ai.T @ (st.avg_power + st.dual_power)
        #     for p, Ai in zip(st.power[dev_index], dev.incidence_matrix)
        # ]

    def set_phase(self, dev: AbstractDevice, dev_index: int, st: ADMMState, nc: int, cont_dev: int):
        if st.dual_phase[dev_index] is None:
            return None
        else:
            AT_xi = apply_incidence_transpose(dev, st.clone_phase)

            # This is a non-contingency device in a contingency-constrained problem
            # We need to aggregrate the contingeny terms
            if nc > 0 and dev_index != cont_dev:
                AT_xi = [torch.mean(A, dim=-1) for A in AT_xi]
                duals = [torch.mean(v, dim=-1) for v in st.dual_phase[dev_index]]

                return [AT_xi_i - v for v, AT_xi_i in zip(duals, AT_xi)]

            # Normal case (or contingency device in a contingency-constrained problem)
            else:
                return [AT_xi_i - v for v, AT_xi_i in zip(st.dual_phase[dev_index], AT_xi)]

            # return [
            #     Ai.T @ st.avg_phase - v
            #     for v, Ai in zip(st.dual_phase[dev_index], dev.incidence_matrix)
            # ]

    def update_averages_and_residuals(
        self, st: ADMMState, net, devices, time_horizon, num_contingencies
    ):
        nc = num_contingencies
        machine, dtype = self.machine, self.dtype

        # Note: it's important to do this in two steps so that the correct averages
        # are used to calculate residuals.
        st = st.update(
            avg_power=dc_average(
                st.power, net, devices, time_horizon, st.num_terminals, machine, dtype, nc
            ),
            avg_phase=ac_average(
                st.phase, net, devices, time_horizon, st.num_ac_terminals, machine, dtype, nc
            ),
        )
        st = st.update(
            resid_power=get_terminal_residual(st.power, st.avg_power, devices),
            resid_phase=get_terminal_residual(st.phase, st.avg_phase, devices),
        )

        # Update z and xi
        st = st.update(
            clone_power=nested_a1bpa2x(st.resid_power, st.clone_power, self.alpha, 1 - self.alpha),
            clone_phase=self.alpha * st.avg_phase + (1 - self.alpha) * st.clone_phase,
        )

        return st

    def price_updates(self, st: ADMMState, net, devices, time_horizon):
        return st.update(
            dual_power=st.dual_power + self.alpha * st.avg_power,
            dual_phase=nested_bpax(st.dual_phase, st.resid_phase, self.alpha),
        )

    # ====
    # History, numerical checks, etc
    # ====

    def dimension_checks(self, st: ADMMState, net, devices, time_horizon):
        num_devices = len(devices)
        num_nodes = net.num_nodes

        assert len(st.power) == num_devices
        assert len(st.phase) == num_devices
        assert len(st.dual_phase) == num_devices
        assert st.dual_power.shape == (num_nodes, time_horizon)

        return True

    def numerical_checks(self, st: ADMMState, net, devices, time_horizon):
        # Dual phases should average to zero
        avg_dual_phase = ac_average(
            st.dual_phase, net, devices, time_horizon, st.num_ac_terminals, self.machine, self.dtype
        )
        torch.testing.assert_allclose(nested_norm(avg_dual_phase), 0.0, rtol=1e-8, atol=1e-8)
        return True

    def compute_objective(
        self,
        st: ADMMState,
        devices: list[AbstractDevice],
        parameters,
        as_item=True,
    ):
        costs = []
        for i, d in enumerate(devices):
            if st.power[i][0].dim() == 3:
                # This is a contingency-constrained device
                # Use base case cost
                pi = [p[:, :, 0] for p in st.power[i]]
                vi = [v[:, :, 0] for v in st.phase[i]] if st.phase[i] is not None else None
                costs += [d.operation_cost(pi, vi, None, la=torch, **parameters[i])]
            else:
                costs += [
                    d.operation_cost(
                        st.power[i], st.phase[i], st.local_variables[i], la=torch, **parameters[i]
                    )
                ]

        if as_item:
            return sum(costs).item()
        else:
            return sum(costs)

    def has_converged(self, st: ADMMState, history: ADMMState, num_cont: int):
        p = 2 if self.resid_norm is None else self.resid_norm
        rho_power, rho_angle = self.get_rho()

        # Absolute component
        primal_tol = self.atol * np.power(self.total_terminals * (num_cont + 1), 1 / p)
        dual_tol = primal_tol

        # Relative component
        # We add this check so we don't waste time computing norms if we don't need to
        if self.rtol > 0.0 or self.rtol_primal is not None or self.rtol_dual is not None:
            # assert self.track_objective
            # print("Adding relative component to convergence check.")

            # Compute norm of primal variables
            rtolp = self.rtol if self.rtol_primal is None else self.rtol_primal
            rtol_primal = rtolp * (
                nested_norm(st.power, p).item() + nested_norm(st.phase, p).item()
            )

            # Compute norm of dual variables
            # Option 1 - Just use objective value
            rtold = self.rtol if self.rtol_dual is None else self.rtol_dual
            if self.rtol_dual_use_objective:
                rtol_dual = rtold * history.objective[-1]

            # Option 2 - Use norm of dual variables (this is the proper way to do this)
            else:
                dual_power_scaled = unsqueeze_terminals_times_x(st.num_terminals, st.dual_power)

                rtol_dual = rtold * (
                    rho_power * torch.linalg.vector_norm(dual_power_scaled.ravel(), p).item()
                    + rho_angle * nested_norm(st.dual_phase, p).item()
                )

            primal_tol += rtol_primal
            dual_tol += rtol_dual

        # Track tolerances
        self.primal_tol_power = primal_tol
        self.primal_tol_angle = primal_tol
        self.dual_tol_power = dual_tol
        self.dual_tol_angle = dual_tol
        self.primal_tol = primal_tol
        self.dual_tol = dual_tol

        primal_resid = np.sqrt(history.power[-1] ** 2 + history.phase[-1] ** 2)
        dual_resid = np.sqrt(history.dual_power[-1] ** 2 + history.dual_phase[-1] ** 2)
        converged = (primal_resid < self.primal_tol) and (dual_resid < self.dual_tol)

        if converged:
            return True
        else:
            return False

    def adjust_rho(self, st: ADMMState, history: ADMMState):
        primal_resid = np.sqrt(history.power[-1] ** 2 + history.phase[-1] ** 2)
        dual_resid = np.sqrt(history.dual_power[-1] ** 2 + history.dual_phase[-1] ** 2)

        # Scale by tolerances
        primal_resid /= self.primal_tol
        dual_resid /= self.dual_tol
        dual_resid *= self.dual_bias

        old_rho = self.rho_power
        self.rho_power = self.tweak_rho(
            old_rho, primal_resid, dual_resid, name="both power and angle"
        )

        # Update rho angle
        if (self.rho_angle is not None) and (not self.relative_rho_angle):
            self.rho_angle *= self.rho_power / old_rho

        # Update prices
        rho_power, rho_angle = self.get_rho()  # Use this to set rho in case rho_angle is None
        if self.rho_power != old_rho:
            st = st.update(
                dual_power=st.dual_power * (old_rho / self.rho_power),
                dual_phase=nested_ax(st.dual_phase, old_rho / self.rho_power),
                rho_power=rho_power,
                rho_angle=rho_angle,
            )

        return st

    def adjust_rho_power_and_angle(self, st: ADMMState, history: ADMMState):
        assert not self.relative_rho_angle
        # First adjust power
        primal_resid, dual_resid = history.power[-1], history.dual_power[-1]
        old_rho = self.rho_power

        # Scale by tolerances
        primal_resid /= self.primal_tol
        dual_resid /= self.dual_tol
        dual_resid *= self.dual_bias

        self.rho_power = self.tweak_rho(old_rho, primal_resid, dual_resid, name="power")
        if self.rho_power != old_rho:
            st = st.update(
                dual_power=st.dual_power * (old_rho / self.rho_power),
                rho_power=self.rho_power,
            )

        # Now adjust angle
        primal_resid, dual_resid = history.phase[-1], history.dual_phase[-1]
        old_rho = self.rho_angle

        # Scale by tolerances
        primal_resid /= self.primal_tol
        dual_resid /= self.dual_tol
        dual_resid *= self.dual_bias

        self.rho_angle = self.tweak_rho(old_rho, primal_resid, dual_resid, name="angle")
        if self.rho_angle != old_rho:
            st = st.update(
                dual_phase=nested_ax(st.dual_phase, old_rho / self.rho_angle),
                rho_angle=self.rho_angle,
            )

        return st

    def tweak_rho(self, old_rho, r_primal, r_dual, name="power"):
        if r_primal > self.adaptation_tolerance * r_dual:
            if self.verbose >= 3:
                print(f"Increasing rho to {old_rho * self.tau} for {name}.")
            return old_rho * self.tau
        elif r_dual > self.adaptation_tolerance * r_primal:
            if self.verbose >= 3:
                print(f"Decreasing rho to {old_rho / self.tau} for {name}.")
            return old_rho / self.tau
        else:
            return old_rho

    def update_history(
        self, history: ADMMState, st: ADMMState, last_avg_phase, last_resid_power, nu_star
    ):
        history.objective += [st.objective]
        p = self.resid_norm

        # Primal/dual residuals
        history = self.update_primal_residuals(history, st)
        history = self.update_dual_residuals(history, st, last_resid_power, last_avg_phase)

        if nu_star is not None:
            history.price_error += [
                torch.linalg.vector_norm(
                    (st.dual_power * self.rho_power - nu_star).ravel(), p
                ).item()
            ]

        return history

    def update_primal_residuals(self, history: ADMMState, st: ADMMState):
        p = self.resid_norm

        # Need to scale this because bar p should actually a vector of size (num_terminals, ...),
        # not (num_nodes, ...). However, we store the compressed form because the values for
        # different terminals at the same node are the same.
        power_scaled = unsqueeze_terminals_times_x(st.num_terminals, st.avg_power)

        history.power += [torch.linalg.vector_norm(power_scaled.ravel(), p).item()]
        history.phase += [nested_norm(st.resid_phase, p).item()]
        return history

    def update_dual_residuals(
        self, history: ADMMState, st: ADMMState, last_resid_power, last_avg_phase
    ):
        p = self.resid_norm

        if self.scale_dual_residuals:
            rp, ra = self.get_rho()
        else:
            rp, ra = 1.0, 1.0

        dual_resid_power = nested_subtract(st.resid_power, last_resid_power)
        history.dual_power += [rp * nested_norm(dual_resid_power, p).item()]

        # This should be scaled by the number of terminals
        phase_scaled = unsqueeze_terminals_times_x(
            st.num_ac_terminals, st.avg_phase - last_avg_phase
        )

        history.dual_phase += [ra * torch.linalg.vector_norm(phase_scaled.ravel(), p).item()]

        return history

    # ====
    # Initialization
    # ====

    def initialize_history(self) -> ADMMState:
        history = ADMMState(
            num_terminals=None,
            num_ac_terminals=None,
            power=[],
            phase=[],
            dual_power=[],
            dual_phase=[],
            objective=[],
        )
        history.price_error = []
        return history

    def initialize_solver(
        self,
        net,
        devices: list[AbstractDevice],
        time_horizon: int,
        num_contingencies: int,
        contingency_device: Optional[int],
    ) -> ADMMState:
        machine, dtype = self.machine, self.dtype

        # Setup weights
        self.power_weights = [None for _ in devices]
        self.angle_weights = [None for _ in devices]

        # Setup state
        num_terminals = get_num_terminals(net, devices, machine=machine, dtype=dtype)
        num_ac_terminals = get_num_terminals(
            net, devices, only_ac=True, machine=machine, dtype=dtype
        )

        nc = num_contingencies
        cd = contingency_device

        # Primals
        power_var = [
            d.admm_initialize_power_variables(time_horizon, machine, dtype) for d in devices
        ]
        phase_var = [
            d.admm_initialize_angle_variables(time_horizon, machine, dtype) for d in devices
        ]

        if cd is not None:
            power_var[cd] = devices[cd].admm_initialize_power_variables(
                time_horizon, machine, dtype, num_contingencies=nc
            )
            phase_var[cd] = devices[cd].admm_initialize_angle_variables(
                time_horizon, machine, dtype, num_contingencies=nc
            )

        power_bar = dc_average(
            power_var, net, devices, time_horizon, num_terminals, machine, dtype, nc
        )
        theta_bar = ac_average(
            phase_var, net, devices, time_horizon, num_ac_terminals, machine, dtype, nc
        )

        # Duals
        power_dual = dc_average(
            power_var, net, devices, time_horizon, num_terminals, machine, dtype, nc
        )
        phase_dual = get_terminal_residual(phase_var, theta_bar, devices)

        power_tilde = get_terminal_residual(power_var, power_bar, devices)
        theta_tilde = get_terminal_residual(phase_var, theta_bar, devices)

        # Clones
        # Clones should be initialized with the same shape as the residuals
        # In particular, this means one value per contingency for ALL devices
        if cd is None:
            clone_power = [
                d.admm_initialize_power_variables(time_horizon, machine, dtype) for d in devices
            ]
        else:
            clone_power = [
                d.admm_initialize_power_variables(
                    time_horizon, machine, dtype, num_contingencies=nc
                )
                for d in devices
            ]
        clone_phase = theta_bar.clone().detach()

        rho_power, rho_angle = self.get_rho()

        local_variables = [None for _ in devices]

        return ADMMState(
            num_terminals=num_terminals,
            num_ac_terminals=num_ac_terminals,
            power=power_var,
            phase=phase_var,
            dual_power=power_dual,
            dual_phase=phase_dual,
            avg_power=power_bar,
            avg_phase=theta_bar,
            resid_power=power_tilde,
            resid_phase=theta_tilde,
            clone_power=clone_power,
            clone_phase=clone_phase,
            rho_power=rho_power,
            rho_angle=rho_angle,
            local_variables=local_variables,
        )
