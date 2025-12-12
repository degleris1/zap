import torch
import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from attrs import define
from typing import List
from functools import cached_property
from numpy.typing import NDArray
from zap.util import infer_machine

from ..devices.abstract import AbstractDevice
from ..conic.variable_device import VariableDevice


@define(kw_only=True, slots=False)
class LogUtilityDevice(VariableDevice):
    """
    Represents a block of variable devices that share the same number of terminals.
    """

    num_nodes: int
    terminals: NDArray  # (num_devices, num_terminals_per_device (k))
    A_v: NDArray
    cost_vector: NDArray  # add make dynamic here?

    def __attrs_post_init__(self):
        assert self.A_v.shape == (self.num_terminals_per_device, self.num_devices)

    @property
    def time_horizon(self) -> int:
        return 1

    def model_local_variables(self, time_horizon: int) -> List[cp.Variable]:
        return [cp.Variable((self.num_devices, time_horizon))]

    def operation_cost(self, _power, _angle, local_variables, la=np, **kwargs):
        """
        Cost function:

        f_d(p_d) = min_{x_d} -w*log(x_d) + I{ p_d = A_v x_d }
        """
        x_d = local_variables[0]
        cost = -la.sum(la.multiply(self.cost_vector.reshape(-1, 1), la.log(x_d)))

        return cost

    def equality_constraints(self, power, _angle, local_variables, la=np, **kwargs):
        x_d = local_variables[0]

        return [power[i] - la.multiply(self.A_v[i : i + 1, :].T, x_d) for i in range(len(power))]

    def inequality_constraints(self, _power, _angle, _local_variables, **kwargs):
        return []

    # ====
    # ADMM Functions
    # ====

    def admm_prox_update(self, rho_power, _rho_angle, power, _angle, **kwargs):
        return _admm_prox_update(self.A_v, self.cost_vector, power, rho_power)


@torch.jit.script
def _admm_prox_update(R_v, w_bv, power: list[torch.Tensor], rho: float):
    """
    See Overleaf on Network Utility Maximization with Log Utility 
    """
    # (num_terminals_per_device, num_devices), now it's like A_v
    Z = torch.stack(power, dim=0).squeeze(-1)

    # Because link-route is all 1's, this is the same as Euclidean norm squared of the columns
    # A_d = ||a_d||^2
    A = R_v.sum(dim=0) 
    # A = (R_v * R_v).sum(dim=0) # Needed if we equilibrate 


    # b_d = a_d.T @ z_d
    b = (R_v * Z).sum(dim=0)


    # Get the positive root from the quadratic formula
    disc = torch.sqrt(b.square() + 4.0 * w_bv * A/rho)
    x_star = (b + disc)/(2.0 * A)


    p_tensor = R_v * x_star.unsqueeze(0)

    # go back to list of tensors (list of length number of terminals,
    # each element is num_devices, time_horizon)
    p_list = [p_tensor[i].unsqueeze(-1) for i in range(p_tensor.shape[0])]

    return p_list, None, [x_star.unsqueeze(-1).expand(-1,1)]
