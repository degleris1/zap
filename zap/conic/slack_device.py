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


# ====
# Second Order Cone Slack Device
# ====


@define(kw_only=True, slots=False)
class SecondOrderConeSlackDevice(SlackDevice):
    """
    Slack device that enforces p_d + b_d in the second order cone.
    """

    def equality_constraints(self, _power, _angle, _local_variables, **kwargs):
        return []

    def inequality_constraints(self, power, _angle, _local_variables, **kwargs):
        """
        Enforces SOC constraint.
        """
        z = cp.vstack([p.T for p in power])  # (num_terminals, num_devices)
        s = z + self.b_d
        return [cp.norm(s[1:, i], 2) - s[0, i] for i in range(s.shape[1])]

    def admm_prox_update(self, _rho_power, _rho_angle, power, _angle, **kwargs):
        """
        ADMM projection for second order cone:
        """
        return _admm_prox_update_soc(power, self.b_d)


@torch.jit.script
def _admm_prox_update_soc(power: list[torch.Tensor], b_d: torch.Tensor):
    """
    ADMM projection for second order cone:
    See overleaf for details. Variable notation follows the Overleaf.
    """

    z = torch.stack([p.squeeze(-1) for p in power], dim=0)  # (num_terminals, num_devices)
    s = z + b_d
    k = s[0, :]
    u = s[1:, :]
    r = torch.norm(u, 2, dim=0)

    proj = torch.zeros_like(s)

    # These are column selection masks
    no_projection_mask = r <= k
    zero_projection = k < -r
    # Else we project to the boundary
    boundary_projection_mask = ~(no_projection_mask | zero_projection)

    # Case 1: Already in SOC so we do nothing (we subtract b_d to get just z because we want to return p_d)
    proj[:, no_projection_mask] = z[:, no_projection_mask]

    # Case 2: Project onto the boundary
    r_boundary = r[boundary_projection_mask]
    k_boundary = k[boundary_projection_mask]
    u_boundary = u[:, boundary_projection_mask]

    scale_factor = (r_boundary + k_boundary) / (2 * r_boundary)
    x_star = scale_factor.unsqueeze(0) * u_boundary
    t_star = (r_boundary + k_boundary) / 2
    proj_boundary = torch.cat([t_star.unsqueeze(0), x_star], dim=0)

    proj[:, boundary_projection_mask] = proj_boundary - b_d[:, boundary_projection_mask]

    # Note Case 3 is covered implicitly by our zero initialization of proj

    p_list = [proj[i].unsqueeze(-1) for i in range(proj.shape[0])]
    return p_list, None, None
