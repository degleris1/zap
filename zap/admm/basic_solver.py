import dataclasses
import functools
import math
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


def _clone_detach(value):
    """Deep-copy tensors, recurse through lists/tuples, pass everything else through."""
    if torch.is_tensor(value):
        return value.clone().detach()
    if isinstance(value, list):
        return [_clone_detach(v) for v in value]
    if isinstance(value, tuple):
        cloned = [_clone_detach(v) for v in value]
        # Preserve namedtuples (e.g. StorageUnitVariable) instead of downcasting.
        return type(value)(*cloned) if hasattr(value, "_fields") else tuple(cloned)
    return value


def _isclose(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return bool(math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0))


@dataclasses.dataclass(frozen=True)
class ADMMLayout:
    """Everything about a problem that an :class:`ADMMState` is only valid for.

    Two states are interchangeable iff their layouts compare equal.  This is the
    fingerprint that makes warm starting safe: a state carried from a different
    horizon, network, device set, machine or dtype is refused (and the solve
    falls back to a cold start) rather than being fed into the iteration, where
    it would either crash deep inside ``dc_average`` or silently mix shapes.

    Deliberately *not* fingerprinted:

    * ``rho_power`` / ``rho_angle`` -- a rho change is an exact change of
      variables on the scaled duals, handled by
      :meth:`ADMMSolver._accept_initial_state`, not an incompatibility.
    * the *values* of the device parameters -- a capacity change is the entire
      reason the warm start exists.
    """

    time_horizon: int
    num_nodes: int
    num_contingencies: int
    contingency_device: Optional[int]
    #: one entry per device: (class name, num_devices, num_terminals_per_device, is_ac)
    device_shapes: tuple
    machine: str
    dtype: str
    #: one entry per device carrying a `soc_mode` (StorageUnit): (index, mode).
    #: The storage boundary condition changes the prox's constraint matrix and
    #: its box bounds, so a state from the other mode is not a valid warm start.
    #: Declared last (with a default) so the older positional constructor still
    #: works; :attr:`_FIELD_ORDER` still reports it before `machine`.
    storage_soc_modes: tuple = ()

    #: Compared in this order by :meth:`explain_mismatch`.
    _FIELD_ORDER = (
        "time_horizon",
        "num_nodes",
        "num_contingencies",
        "contingency_device",
        "device_shapes",
        "storage_soc_modes",
        "machine",
        "dtype",
    )

    @classmethod
    def of(
        cls,
        net,
        devices,
        time_horizon,
        machine,
        dtype,
        num_contingencies: int = 0,
        contingency_device: Optional[int] = None,
    ) -> "ADMMLayout":
        return cls(
            time_horizon=int(time_horizon),
            num_nodes=int(net.num_nodes),
            num_contingencies=int(num_contingencies),
            contingency_device=(None if contingency_device is None else int(contingency_device)),
            device_shapes=tuple(
                (
                    type(d).__name__,
                    int(d.num_devices),
                    int(d.num_terminals_per_device),
                    bool(d.is_ac),
                )
                for d in devices
            ),
            storage_soc_modes=tuple(
                (i, str(d.soc_mode))
                for i, d in enumerate(devices)
                if getattr(d, "soc_mode", None) is not None
            ),
            # `str` normalises so that "cpu" and torch.device("cpu"), or
            # torch.float64 and "torch.float64", compare equal.
            machine=str(machine),
            dtype=str(dtype),
        )

    def explain_mismatch(self, other: "ADMMLayout") -> Optional[str]:
        """``None`` if equal, else a one-line reason naming the first differing field."""
        if not isinstance(other, ADMMLayout):
            return f"layout is a {type(other).__name__}, not an ADMMLayout"
        for name in self._FIELD_ORDER:
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine != theirs:
                return f"{name} changed: state has {theirs!r}, this solve needs {mine!r}"
        return None


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
    #: ``ADMMLayout | None`` -- the problem this state is valid for. ``None`` on a
    #: state produced before warm-start hardening, which is refused as a warm start.
    layout: object = None
    #: Outer ADMM iterations that have gone into producing this state, across all
    #: solves that carried it.
    cumulative_iteration: int = 0

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
        """A detached deep copy, generically over the dataclass fields.

        Generic on purpose: the previous hand-written version enumerated fields
        and hard-coded ``ADMMState(...)``, so it silently dropped any new field
        and *downcast* an :class:`~zap.admm.weighted_solver.ExtendedADMMState`,
        losing its weights.  ``type(self)`` keeps the subclass;
        ``dataclasses.fields`` keeps every field.

        The copy is detached, so a state carried into the next forward pass is an
        autograd leaf and nothing from the previous pass's tape is retained.
        """
        return type(self)(
            **{f.name: _clone_detach(getattr(self, f.name)) for f in dataclasses.fields(self)}
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
    #: Deprecated. When True the dual tolerance is `rtol * objective` (dollars)
    #: while the dual residual is a power (MW) -- a unit mismatch whose severity
    #: scales with the block (124x looser than the primal tolerance on a 24 h
    #: ca2040_z4 block). The dual tolerance is now `rtol * rho * ||nu||`.
    rtol_dual_use_objective: bool = False
    resid_norm: int = 2
    safe_mode: bool = False
    track_objective: bool = True
    battery_window: Optional[int] = None
    battery_inner_weight: float = 1.0
    battery_inner_over_relaxation: float = 1.8
    battery_inner_iterations: int = 200
    minimum_iterations: int = 10
    relative_rho_angle: bool = False
    adaptive_rho: bool = True
    tau: float = 1.1
    adaptation_tolerance: float = 2.0
    adaptation_frequency: int = 50
    #: Clamps on the adaptive-rho rule. Without them the rule can walk rho down
    #: (or up) without bound and the primal residual explodes.
    rho_min: float = 1e-2
    rho_max: float = 1e4
    verbose: int = 1
    scale_dual_residuals: bool = None  # Deprecated

    def __post_init__(self):
        # Warm-start bookkeeping exists from construction, not only after solve().
        self.warm_started = False
        self.warm_start_reason = None
        self.warm_start_rho_rescaled = False
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
        if self.rtol_dual_use_objective:
            warnings.warn(
                "rtol_dual_use_objective compares a power-valued dual residual against "
                "rtol * objective (a cost) and is deprecated; it will be removed in a "
                "future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        # A warm-started solve restarts with `resid_power` carried from the previous
        # parameters, so the first dual residual -- which is a *difference* between
        # consecutive `resid_power` iterates -- is spuriously tiny and the solver can
        # declare convergence long before the primal residual has caught up with the
        # new parameters. `minimum_iterations` is the only defence against this, so it
        # must be meaningful (>= 100 in every ADMM planning config).
        if self.minimum_iterations < 10:
            warnings.warn(
                "minimum_iterations < 10 makes a warm-started solve liable to "
                "declare convergence on a stale dual residual"
            )

        if not (self.rho_min <= self.rho_max):
            raise ValueError(f"rho_min ({self.rho_min}) must not exceed rho_max ({self.rho_max})")
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
        layout = ADMMLayout.of(
            net,
            devices,
            time_horizon,
            self.machine,
            self.dtype,
            num_contingencies,
            contingency_device,
        )

        # Defined before the loop so that `num_iterations=0` returns the initial
        # state instead of raising AttributeError on `self.converged`.
        self.iteration = 0
        self.converged = False
        self.warm_started = False
        self.warm_start_reason = None
        self.warm_start_rho_rescaled = False

        if initial_state is None:
            st = self.initialize_solver(
                net, devices, time_horizon, num_contingencies, contingency_device
            )
        else:
            st, self.warm_start_reason = self._accept_initial_state(initial_state, layout)
            if st is None:
                # A refused warm start is always recoverable: fall back to a cold
                # start with a logged reason rather than killing a multi-hour run.
                if self.verbose >= 1:
                    warnings.warn(
                        "ADMM warm start refused, falling back to a cold start: "
                        f"{self.warm_start_reason}"
                    )
                st = self.initialize_solver(
                    net, devices, time_horizon, num_contingencies, contingency_device
                )
            else:
                self.warm_started = True

        st = st.update(layout=layout)

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

        # Stamp the state with the rho its scaled duals are relative to. `adjust_rho`
        # only writes these fields on the iterations where rho actually moves, so
        # read the current value here rather than trusting the state.
        rho_power_now, rho_angle_now = self.get_rho()
        st = st.update(
            cumulative_iteration=st.cumulative_iteration + self.iteration,
            rho_power=rho_power_now,
            rho_angle=rho_angle_now,
        )

        # Restore original settings
        for k, v in original_settings.items():
            setattr(self, k, v)

        return st, history

    def _accept_initial_state(self, state, layout: "ADMMLayout"):
        """Return ``(state, None)`` if reusable as a warm start, else ``(None, reason)``.

        Rescales the carried *scaled* duals when only rho differs: ``u = nu / rho``,
        so preserving the unscaled price ``nu`` across a rho change is an exact
        change of variables (the same transformation :meth:`adjust_rho` applies
        mid-solve), not an incompatibility.
        """
        if not isinstance(state, ADMMState):
            return None, f"initial_state is a {type(state).__name__}, not an ADMMState"
        if state.layout is None:
            return None, "initial_state carries no layout fingerprint"

        reason = layout.explain_mismatch(state.layout)
        if reason is not None:
            return None, reason

        rho_power, rho_angle = self.get_rho()
        angle_changed = (
            state.rho_angle is not None
            and rho_angle is not None
            and not _isclose(state.rho_angle, rho_angle)
        )
        if state.rho_power is not None and (
            not _isclose(state.rho_power, rho_power) or angle_changed
        ):
            old_angle = state.rho_angle if state.rho_angle is not None else state.rho_power
            new_angle = rho_angle if rho_angle is not None else rho_power
            state = state.update(
                dual_power=state.dual_power * (state.rho_power / rho_power),
                dual_phase=nested_ax(state.dual_phase, old_angle / new_angle),
                rho_power=rho_power,
                rho_angle=rho_angle,
            )
            self.warm_start_rho_rescaled = True

        return state, None

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

        # Record the achieved tolerances so a caller can put them on a run card.
        history.primal_tol += [primal_tol]
        history.dual_tol += [dual_tol]

        primal_resid = np.sqrt(history.power[-1] ** 2 + history.phase[-1] ** 2)
        dual_resid = np.sqrt(history.dual_power[-1] ** 2 + history.dual_phase[-1] ** 2)
        converged = (primal_resid < self.primal_tol) and (dual_resid < self.dual_tol)

        return bool(converged and self.iteration >= self.minimum_iterations)

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
            new_rho = min(old_rho * self.tau, self.rho_max)
            if self.verbose >= 3 and new_rho != old_rho:
                print(f"Increasing rho to {new_rho} for {name}.")
            return new_rho
        elif r_dual > self.adaptation_tolerance * r_primal:
            new_rho = max(old_rho / self.tau, self.rho_min)
            if self.verbose >= 3 and new_rho != old_rho:
                print(f"Decreasing rho to {new_rho} for {name}.")
            return new_rho
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

        # The dual residual of scaled ADMM is rho * A'(z_k - z_{k-1}); scaling it by
        # rho puts it in the same units as the dual tolerance rtol * rho * ||nu||.
        rp, ra = self.get_rho()

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
        history.primal_tol = []
        history.dual_tol = []
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
