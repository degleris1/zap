import torch
import numpy as np
import cvxpy as cp
from attrs import define
from typing import List
from numpy.typing import NDArray
from ..devices.abstract import AbstractDevice


@define(kw_only=True, slots=False)
class QuadraticDevice(AbstractDevice):
    """
    Represents a block of single terminal quadratic devices.
    """

    num_nodes: int
    terminals: NDArray

    @property
    def time_horizon(self) -> int:
        return 1

    def model_local_variables(self, time_horizon: int) -> List[cp.Variable]:
        return None

    def operation_cost(self, power, _angle, local_variables, la=np, **kwargs):
        return 0.5 * la.sum(power[0] ** 2)

    def equality_constraints(self, _power, _angle, _local_variables, **kwargs):
        return []

    def inequality_constraints(self, power, _angle, _local_variables, **kwargs):
        return []

    def admm_prox_update(self, rho_power, _rho_angle, power, _angle, **kwargs):
        return _admm_prox_update(power, rho_power)


@torch.jit.script
def _admm_prox_update(power: list[torch.Tensor], rho: float):
    scaling = rho / (1 + rho)
    y_star = power[0] * scaling
    return [y_star], None, None
