import numpy as np
import scipy.sparse as sp
import cvxpy
import torch
from attrs import define, field

from numpy.typing import NDArray

from .abstract import AbstractDevice, get_time_horizon, make_dynamic


@define(kw_only=True, slots=False)
class PowerTarget(AbstractDevice):
    """A single-node device that tries to match its power output to a target value."""

    num_nodes: int
    terminal: NDArray
    target_power: NDArray = field(converter=make_dynamic)
    norm_order: int = field(default=2)

    def __post_init__(self):
        assert self.norm_order in [1, 2]

    @property
    def terminals(self):
        return self.terminal

    @property
    def time_horizon(self):
        return get_time_horizon(self.min_power)

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def equality_constraints(self, power, angle, _, target_power=None, la=np):
        return []

    def inequality_constraints(self, power, angle, _, target_power=None, la=np):
        return []

    def operation_cost(self, power, angle, _, target_power=None, la=np):
        target_power = self.parameterize(target_power=target_power, la=la)

        err = power[0] - target_power
        if self.norm_order == 1:
            return la.sum(la.abs(err))

        else:  # L2
            return (0.5) * la.sum(la.square(err))

    # ====
    # DIFFERENTIATION
    # ====

    def _equality_matrices(self, equalities, target_power=None, la=np):
        return equalities

    def _inequality_matrices(self, inequalities, target_power=None, la=np):
        return inequalities

    def _hessian_power(self, hessians, power, angle, _, target_power=None, la=np):
        target_power = self.parameterize(target_power=target_power, la=la)

        if self.norm_order == 2:
            hessians[0] += sp.diags(np.ones_like(power[0]).ravel())
        else:  # L1
            pass

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
        target_power=None,
        power_weights=None,
        angle_weights=None,
    ):
        target_power = self.parameterize(target_power=target_power)
        assert angle is None

        return _admm_prox_update(power, rho_power, target_power)


@torch.jit.script
def _admm_prox_update(power: list[torch.Tensor], rho: float, target_power: torch.Tensor):
    # Problem is
    #     min_p    (1/2) * (p - p_target)^2 + (rho / 2) || (p - power) ||_2^2
    # Objective derivative is
    #    (p - p_target) +  rho (p - power) = 0
    # Which is solved by
    #     p = (p_target + rho * power) / (1 + rho)

    p = (target_power + rho * power[0]) / (1 + rho)

    return [p], None
