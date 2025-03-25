import torch
import numpy as np
import cvxpy as cp
from attrs import define
from typing import List
from numpy.typing import NDArray
from attrs import field
from ..devices.abstract import AbstractDevice, make_dynamic


@define(kw_only=True, slots=False)
class SlackDevice(AbstractDevice):
    """
    Abstract base class for slack devices (specifics depend on the cone)
    """

    num_nodes: int
    terminals: NDArray
    b_d: NDArray = field(converter=make_dynamic)

    @property
    def time_horizon(self) -> int:
        return 1

    def model_local_variables(self, time_horizon: int) -> List[cp.Variable]:
        return None

    def operation_cost(self, power, angle, local_variables, la=np, **kwargs):
        return 0.0

    def equality_constraints(self, power, angle, local_variables, la=np):
        raise NotImplementedError

    def inequality_constraints(self, power, angle, local_variables, la=np):
        raise NotImplementedError

    def admm_prox_update(self, power, rho):
        raise NotImplementedError


# ====
# Zero Cone Slack Device
# ====


@define(kw_only=True, slots=False)
class ZeroConeSlackDevice(SlackDevice):
    """
    Slack device that enforces p_d + b_d = 0 (zero cone)
    """

    def equality_constraints(self, power, _angle, _local_variables, **kwargs):
        return [power[0] + self.b_d]  # == 0

    def inequality_constraints(self, _power, _angle, _local_variables, **kwargs):
        return []

    def admm_prox_update(self, _rho_power, _rho_angle, power, _angle, **kwargs):
        """
        ADMM projection for zero cone:
            p_d^* = -b_d
        """
        return _admm_prox_update_zero(power, self.b_d)


@torch.jit.script
def _admm_prox_update_zero(power: list[torch.Tensor], b_d: torch.Tensor):
    """
    ADMM projection for zero cone:
        p_d^* = -b_d
    """
    return [-b_d], None, None


# ====
# Non-Negative Cone Slack Device
# ====


@define(kw_only=True, slots=False)
class NonNegativeConeSlackDevice(SlackDevice):
    """
    Slack device that enforces p_d + b_d >= 0 (non-negative cone).
    """

    def equality_constraints(self, _power, _angle, _local_variables, **kwargs):
        return []

    def inequality_constraints(self, power, _angle, _local_variables, **kwargs):
        """
        Enforces p_d + b_d >= 0.
        """
        return [-power[0] - self.b_d]  # <= 0

    def admm_prox_update(self, _rho_power, _rho_angle, power, _angle, **kwargs):
        """
        ADMM projection for non-negative cone:
        p_d^* = max(z_d, -b_d)
        """
        return _admm_prox_update_nonneg(power, self.b_d)


@torch.jit.script
def _admm_prox_update_nonneg(power: list[torch.Tensor], b_d: torch.Tensor):
    """
    ADMM projection for non-negative cone:
    p_d^* = max(z_d, -b_d)
    """
    p = torch.maximum(power[0], -b_d)
    return [p], None, None
