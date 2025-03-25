import torch
import numpy as np
import cvxpy as cp
from attrs import define
from typing import List
from numpy.typing import NDArray

from ..devices.abstract import AbstractDevice


@define(kw_only=True, slots=False)
class VariableDevice(AbstractDevice):
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

        f_d(p_d) = min_{x_d} c_d^T x_d + I{ p_d = A_v x_d }
        """
        x_d = local_variables[0]
        cost = la.sum(la.multiply(self.cost_vector.reshape(-1, 1), x_d))

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


# @torch.jit.script
def _admm_prox_update(A_v, c_bv, power: list[torch.Tensor], rho: float):
    """
    See Overleaf on Conic Translation Sec. 4.1.1 for full details (will update the comments here eventually)
    """
    # (num_terminals_per_device, num_devices), now it's like A_v
    Z = torch.stack(power, dim=0).squeeze(-1)

    # Compute the proximal update efficiently (again see 4.1.1)
    diag_AT_Z = torch.sum(A_v * Z, dim=0)
    c_bv_scaled = (1 / rho) * c_bv
    A_norms_sq = torch.linalg.norm(A_v, dim=0, ord=2) ** 2

    x_star = (diag_AT_Z - c_bv_scaled) / A_norms_sq
    p_tensor = torch.multiply(A_v, x_star)

    # go back to list of tensors (list of length number of terminals,
    # each element is num_devices, time_horizon)
    p_list = [p_tensor[i].unsqueeze(-1) for i in range(p_tensor.shape[0])]

    return p_list, None, [x_star.unsqueeze(-1).expand(-1, 1)]
