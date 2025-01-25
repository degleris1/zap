import numpy as np
import scipy.sparse as sp
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

        if self.norm_order == 1:
            return _admm_prox_update_l1(power, rho_power, target_power)
        else:  # L2
            return _admm_prox_update_l2(power, rho_power, target_power)


@torch.jit.script
def _admm_prox_update_l2(power: list[torch.Tensor], rho: float, target_power: torch.Tensor):
    # Problem is
    #     min_p    (1/2) * || p - p_target ||_2^2 + (rho / 2) || (p - power) ||_2^2
    # Objective derivative is
    #    (p - p_target) +  rho (p - power) = 0
    # Which is solved by
    #     p = (p_target + rho * power) / (1 + rho)

    p = (target_power + rho * power[0]) / (1 + rho)

    return [p], None


@torch.jit.script
def _admm_prox_update_l1(power: list[torch.Tensor], rho: float, target_power: torch.Tensor):
    # Problem is
    #     min_p    ||p - p_target ||_1 + (rho / 2) || (p - power) ||_2^2
    # Objective subdifferential is
    #    sign(p - p_target) +  rho (p - power) = 0
    #    sign(p - p_target) + rho p = rho power
    # Which is solved by
    #     p = power - 1/rho    if power[0] > p_target + (1.0/rho)
    #     p = power + 1/rho    if power[0] < p_target - (1.0/rho)
    #     p = p_target         otherwise
    # This is essentially the soft threshholding operator.
    # See https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf Section 6.5.2

    if power[0] > target_power + (1.0 / rho):
        p = power[0] - (1.0 / rho)
    elif power[0] < target_power - (1.0 / rho):
        p = power[0] + (1.0 / rho)
    else:  # power[0] in [target_power - (1.0/rho), target_power + (1.0/rho)]
        p = target_power

    return [p], None
