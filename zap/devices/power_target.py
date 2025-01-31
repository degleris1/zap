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
    weights: NDArray = field(default=None, converter=make_dynamic)
    norm_order: int = field(default=2)

    def __attrs_post_init__(self):
        assert self.norm_order in [1, 2]

        if self.weights is None:
            self.weights = np.ones((self.num_devices, 1))

    @property
    def terminals(self):
        return self.terminal

    @property
    def time_horizon(self):
        return get_time_horizon(self.target_power)

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def equality_constraints(self, power, angle, _, target_power=None, weights=None, la=np):
        return []

    def inequality_constraints(self, power, angle, _, target_power=None, weights=None, la=np):
        return []

    def operation_cost(self, power, angle, _, target_power=None, weights=None, la=np):
        target_power = self.parameterize(target_power=target_power, la=la)
        weights = self.parameterize(weights=weights, la=la)

        err = power[0] - target_power
        if self.norm_order == 1:
            return la.sum(la.abs(err) * weights)

        else:  # L2
            return (0.5) * la.sum(la.square(err) * weights)

    # ====
    # DIFFERENTIATION
    # ====

    def _equality_matrices(self, equalities, target_power=None, weights=None, la=np):
        return equalities

    def _inequality_matrices(self, inequalities, target_power=None, weights=None, la=np):
        return inequalities

    def _hessian_power(self, hessians, power, angle, _, target_power=None, weights=None, la=np):
        target_power = self.parameterize(target_power=target_power, la=la)
        weights = self.parameterize(weights=weights, la=la)

        if self.norm_order == 2:
            hessians[0] += sp.diags((np.ones_like(power[0]) * weights).ravel())
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
        weights=None,
        power_weights=None,
        angle_weights=None,
    ):
        target_power = self.parameterize(target_power=target_power)
        weights = self.parameterize(weights=weights)
        assert angle is None

        if self.norm_order == 1:
            return _admm_prox_update_l1(power, rho_power, target_power, weights)
        else:  # L2
            return _admm_prox_update_l2(power, rho_power, target_power, weights)


@torch.jit.script
def _admm_prox_update_l2(
    power: list[torch.Tensor], rho: float, target_power: torch.Tensor, weights: torch.Tensor
):
    # Problem is
    #     min_p    (1/2) * w * || (p - p_target) ||_2^2 + (rho / 2) || (p - power) ||_2^2
    # Objective derivative is
    #    w * (p - p_target) +  rho (p - power) = 0
    # Which is solved by
    #     p = (w * p_target + rho * power) / (w + rho)

    p = (weights * target_power + rho * power[0]) / (weights + rho)

    return [p], None


@torch.jit.script
def _admm_prox_update_l1(
    power: list[torch.Tensor], rho: float, target_power: torch.Tensor, weights: torch.Tensor
):
    # Problem is
    #     min_p    || w (p - p_target) ||_1 + (rho / 2) || (p - power) ||_2^2
    # Objective subdifferential is
    #    w * sign(p - p_target) + rho (p - power) = 0
    #    w * sign(p - p_target) + rho p = rho power
    #    sign(p - p_target) + (rho/w) p = (rho/w) power
    #
    # Which is solved by
    #     p = power - w/rho    if power[0] > p_target + (w/rho)
    #     p = power + w/rho    if power[0] < p_target - (w/rho)
    #     p = p_target         otherwise
    # This is essentially the soft threshholding operator.
    # See https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf Section 6.5.2

    # theta1 = torch.where(bool_mask, angle[1], 0.5 * angle[0] + 0.5 * angle[1] - mu * p1)
    # theta0 = torch.where(bool_mask, angle[0], theta1 + p1 / b)

    # Use torch.where
    big = torch.greater_equal(power[0], target_power + (weights / rho))
    small = torch.less_equal(power[0], target_power - (weights / rho))

    p_big = torch.where(big, power[0] - (weights / rho), 0.0)
    p_small = torch.where(small, power[0] + (weights / rho), 0.0)
    p_mid = torch.where(torch.logical_not(big) * torch.logical_not(small), target_power, 0.0)

    p = p_big + p_small + p_mid

    return [p], None
