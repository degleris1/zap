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

    @cached_property
    def incidence_matrix(self):
        dimensions = (self.num_nodes, self.num_devices)

        matrices = []
        for terminal_index in range(self.num_terminals_per_device):
            if len(self.terminals.shape) == 1:
                rows = self.terminals
            else:
                rows = self.terminals[:, terminal_index]

            # We must ignore the -1 padded rows (these are not real terminals)
            mask = rows >= 0
            rows = rows[mask]
            cols = np.flatnonzero(mask)
            vals = np.ones(len(rows))

            matrices.append(sp.csc_matrix((vals, (rows, cols)), shape=dimensions))

        return matrices

    def torch_terminals(self, time_horizon, machine=None) -> list[torch.Tensor]:
        machine = infer_machine() if machine is None else machine

        # Effectively caching manually
        if (
            hasattr(self, "_torch_terminals")
            and self._torch_terminal_time_horizon == time_horizon
            and self._torch_terminal_machine == machine
        ):
            return self._torch_terminals

        tt = self.terminals
        torch_terminals = []
        if isinstance(tt, np.ndarray):
            tt = torch.tensor(tt, device=machine)

        if len(self.terminals.shape) == 1:
            torch_terminals = [tt.reshape(-1, 1).expand(-1, time_horizon)]
        else:
            for i in range(self.num_terminals_per_device):
                rows = tt[:, i]
                mask = rows >= 0
                terminal_vector = (
                    rows.clone().masked_fill(~mask, 0).reshape(-1, 1)
                )  # Reshape tensor and replace -1 pads with 0
                torch_terminals.append(
                    terminal_vector.expand(-1, time_horizon)
                )  # This won't do anything when time_horizion is just 1

        self._torch_terminals = torch_terminals
        self._torch_terminal_time_horizon = time_horizon
        self._torch_terminal_machine = machine

        return torch_terminals

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


@torch.jit.script
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
