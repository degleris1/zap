import torch
import numpy as np
import cvxpy as cp
from attrs import define, field
from typing import Optional, List
from numpy.typing import NDArray

from .abstract import AbstractDevice

@define(kw_only=True, slots=False)
class BlockVariableDevice(AbstractDevice):
    """
    Represents a block of variable devices that share the same number of terminals.
    The name block is really maybe unnecessary because general k terminal devices do this anyway
    """

    num_nodes: int
    terminals: NDArray # (num_devices, num_terminals_per_device (k))
    A_bv: NDArray 
    cost_vector: NDArray

    @property
    def time_horizon(self) -> int:
        return 1

    @property
    def is_ac(self) -> bool:
        return False

    @property
    def is_convex(self) -> bool:
        return True

    def model_local_variables(self, time_horizon: int) -> List[cp.Variable]:
        return [cp.Variable((self.num_devices, time_horizon))]

    def operation_cost(self, power, angle, local_variables, **kwargs):
        """
        Cost function:

        f_d(p_d) = min_{x_d} c_d^T x_d + I{ p_d = A_bv x_d }
        """
        la = kwargs.get("la", np)
        x_d = local_variables[0]
        cost = la.sum(la.multiply(self.cost_vector, x_d))

        return cost

    def equality_constraints(self, power, angle, local_variables, *args, **kwargs):
        la = kwargs.get("la", np)
        return [power[i] - la.multiply(self.A_bv[i], local_variables[0]) for i in range(len(power))]

    def inequality_constraints(self, power, angle, local_variables, **kwargs):
        return []

    # ====
    # ADMM Functions
    # ====

    def admm_prox_update(self, rho_power, rho_angle, power, angle, **kwargs):
        A_bv_tensor = torch.tensor(self.A_bv, dtype=torch.float32)
        c_bv_tensor = torch.tensor(self.cost_vector, dtype=torch.float32)
        print(A_bv_tensor.shape)
        print(c_bv_tensor.shape)
        print(len(power))
        print(power[0].shape)
        return _admm_prox_update(A_bv_tensor, c_bv_tensor, power, rho_power)

    
@torch.jit.script
def _admm_prox_update(A_bv, c_bv, power: list[torch.Tensor], rho: float):
    '''
    See Overleaf on Conic Translation Sec. 4.1.1 for full details (will update the comments here eventually)
    '''
    Z = torch.stack(power, dim=0).squeeze(-1)  # (num_terminals_per_device, num_devices), now it's like A_bv

    # Compute the proximal update efficiently (again see 4.1.1)
    diag_AT_Z = torch.sum(A_bv * Z, dim=0)
    c_bv_scaled = (1 / rho) * c_bv
    A_norms_sq = torch.sum(A_bv * A_bv, dim=0)

    x_star = (diag_AT_Z - c_bv_scaled) / A_norms_sq
    p_tensor = torch.multiply(A_bv, x_star)
    p_list = [p_tensor[i].unsqueeze(-1) for i in range(p_tensor.shape[0])] # go back to list of tensors (list of length number of terminals, each element is num_devices,time_horizon)

    return p_list, None
