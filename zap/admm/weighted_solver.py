import dataclasses
import numpy as np

from zap.devices.abstract import AbstractDevice
from zap.admm.basic_solver import ADMMSolver, ADMMState
from zap.admm.util import (
    nested_map,
    nested_add,
    nested_subtract,
    dc_average,
    ac_average,
)


@dataclasses.dataclass
class ExtendedADMMState(ADMMState):
    copy_power: object = None
    copy_phase: object = None
    full_dual_power: object = None
    _power_weights: object = None
    _angle_weights: object = None
    inv_sq_power_weights: object = None
    avg_inv_sq_power_weights: object = None

    @property
    def power_weights(self):
        return self._power_weights

    @property
    def angle_weights(self):
        return self._angle_weights


@dataclasses.dataclass
class WeightedADMMSolver(ADMMSolver):
    """Stores weighted ADMM solver parameters and exposes a solve function."""

    weighting_strategy: str = "uniform"
    weighting_seed: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert self.weighting_strategy in ["uniform", "random", "smart_cost", "smart_bounds"]

    def initialize_solver(
        self,
        net,
        devices,
        time_horizon,
        num_contingencies: int = 0,
        contingency_device=None,
    ) -> ExtendedADMMState:
        # W1-5: the base class calls this with five arguments; the old
        # three-argument signature raised TypeError the moment it was reached.
        if num_contingencies > 0:
            raise NotImplementedError("WeightedADMMSolver does not support contingencies")

        st = super().initialize_solver(
            net, devices, time_horizon, num_contingencies, contingency_device
        )

        # Set weights
        rng = np.random.default_rng(self.weighting_seed)

        if self.weighting_strategy == "random":
            _power_weights = nested_map(lambda x: 0.5 + rng.random(x.shape), st.power)

        elif self.weighting_strategy in ["smart_cost", "smart_bounds"]:
            # TODO - Put parameters in here
            _power_weights = [
                d.get_admm_power_weights(pi, self.weighting_strategy)
                for (pi, d) in zip(st.power, devices)
            ]

        else:  # uniform
            _power_weights = nested_map(lambda x: np.ones_like(x), st.power)

        _angle_weights = nested_map(lambda x: np.ones_like(x), st.phase)

        # Set weight-related quantities
        inv_sq_power_weights = nested_map(lambda x: np.power(x, -2), _power_weights)
        avg_inv_sq_power_weights = dc_average(
            inv_sq_power_weights, net, devices, time_horizon, st.num_terminals
        )

        return ExtendedADMMState(
            copy_power=st.power.copy(),
            copy_phase=st.phase.copy(),
            full_dual_power=st.power.copy(),
            _power_weights=_power_weights,
            _angle_weights=_angle_weights,
            inv_sq_power_weights=inv_sq_power_weights,
            avg_inv_sq_power_weights=avg_inv_sq_power_weights,
            **{f.name: getattr(st, f.name) for f in dataclasses.fields(st)},
        )

    def set_power(self, dev: AbstractDevice, dev_index: int, st: ExtendedADMMState, nc: int = 0):
        return [
            z - omega for z, omega in zip(st.copy_power[dev_index], st.full_dual_power[dev_index])
        ]

    def set_phase(
        self, dev: AbstractDevice, dev_index: int, st: ExtendedADMMState, nc: int = 0, cont_dev=None
    ):
        if st.dual_phase[dev_index] is None:
            return None
        else:
            return [ksi - nu for ksi, nu in zip(st.copy_phase[dev_index], st.dual_phase[dev_index])]

    def price_updates(self, st: ExtendedADMMState, net, devices, time_horizon):
        # Update duals
        st = st.update(
            full_dual_power=nested_add(
                st.full_dual_power, nested_subtract(st.power, st.copy_power)
            ),
            dual_phase=nested_add(st.dual_phase, nested_subtract(st.phase, st.copy_phase)),
        )
        # Update average price dual, used for tracking LMPs
        st = st.update(
            dual_power=dc_average(st.full_dual_power, net, devices, time_horizon, st.num_terminals)
        )
        return st

    def update_averages_and_residuals(
        self, st: ExtendedADMMState, net, devices, time_horizon, num_contingencies: int = 0
    ):
        st = super().update_averages_and_residuals(
            st, net, devices, time_horizon, num_contingencies
        )

        # ====
        # (1) Update power
        # ====

        # avg_dual_power = dc_average(
        #     st.full_dual_power, net, devices, time_horizon, st.num_terminals
        # )
        # resid_dual_power = get_terminal_residual(st.full_dual_power, avg_dual_power, devices)
        # st = st.update(
        #     copy_power=nested_add(st.resid_power, resid_dual_power),
        # )

        # Get p + omega and avg(p + omega)
        power_dual_plus_primal = nested_add(st.full_dual_power, st.power)
        avg_pdpp = dc_average(power_dual_plus_primal, net, devices, time_horizon, st.num_terminals)

        # Get weighted term
        weight_scaling = avg_pdpp / st.avg_inv_sq_power_weights
        scaled_weights = [
            [-(Ai.T @ weight_scaling) * D_dev_i for Ai, D_dev_i in zip(dev.incidence_matrix, D_dev)]
            for dev, D_dev in zip(devices, st.inv_sq_power_weights)
        ]

        st = st.update(copy_power=nested_add(power_dual_plus_primal, scaled_weights))

        # ====
        # (2) Update phase
        # ====

        avg_dual_phase = ac_average(st.dual_phase, net, devices, time_horizon, st.num_ac_terminals)

        st = st.update(
            # copy_power=nested_add(st.resid_power, resid_dual_power),
            copy_phase=[
                [Ai.T @ (st.avg_phase + avg_dual_phase) for Ai in dev.incidence_matrix]
                for dev in devices
            ],
        )

        # Resid dual power should be zero, let's check
        if self.safe_mode:
            # np.testing.assert_allclose(nested_norm(resid_dual_power), 0.0, atol=1e-6)
            np.testing.assert_allclose(avg_dual_phase, 0.0, atol=1e-8)

        return st

    def dimension_checks(self, st: ExtendedADMMState, net, devices, time_horizon):
        assert len(st.copy_power) == len(st.power)
        assert len(st.full_dual_power) == len(st.power)
        assert len(st.copy_phase) == len(st.phase)

        return super().dimension_checks(st, net, devices, time_horizon)
