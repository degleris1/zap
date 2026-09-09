import numpy as np
import scipy.sparse as sp
import torch

from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray

from zap.devices.abstract import make_dynamic
from zap.util import replace_none
from .transporter import Transporter


@dataclass(kw_only=True)
class DirectedLine(Transporter):
    """A one-way transporter with a conversion efficiency and a signed linear cost.

    ``power[1] = -efficiency * power[0]``; bounds and cost apply to ``power[1]``.
    Unlike ``Transporter`` / ``PowerLine`` the cost is linear in ``power[1]``
    (no ``abs``), so a negative ``linear_cost`` (e.g. export revenue) stays DCP.

    This is the zap analogue of a PyPSA ``Link``: PyPSA links are directional
    (``p_min_pu`` defaults to zero) and carry an efficiency, neither of which the
    symmetric ``DCLine`` can express.
    """

    name: Optional[NDArray] = None
    efficiency: NDArray = None  # (N, 1); defaults to ones in __post_init__

    def __post_init__(self):
        super().__post_init__()
        self.efficiency = make_dynamic(replace_none(self.efficiency, np.ones(self.num_devices)))
        # A lossy line with reverse flow allowed (min_power < 0) would create
        # energy: p0 = -p1/eta withdraws less than |p1| when p1 < 0. Flow must
        # be one-way, which is the point of this device.
        if np.any(np.asarray(self.min_power) < 0):
            raise ValueError(
                "DirectedLine requires min_power >= 0 (one-way flow); "
                "model reverse capacity as a second DirectedLine."
            )

    @property
    def is_ac(self):
        return False

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def equality_constraints(
        self, power, angle, _, nominal_capacity=None, la=np, envelope=None, mask=None
    ):
        return [power[1] + la.multiply(self.efficiency, power[0])]

    # inequality_constraints inherited: bounds on power[1]

    def operation_cost(self, power, angle, _, nominal_capacity=None, la=np, envelope=None):
        cost = la.sum(la.multiply(self.linear_cost, power[1]))
        if self.quadratic_cost is not None:
            cost += la.sum(la.multiply(self.quadratic_cost, la.square(power[1])))

        return cost

    # ====
    # DIFFERENTIATION
    # ====

    def _equality_matrices(self, equalities, nominal_capacity=None, la=np):
        size = equalities[0].power[0].shape[1]
        T = size // self.num_devices

        eff = np.broadcast_to(np.asarray(self.efficiency), (self.num_devices, T)).ravel()

        equalities[0].power[0] += sp.diags(eff)
        equalities[0].power[1] += sp.eye(size)

        return equalities

    # ====
    # ADMM FUNCTIONS
    # ====

    def admm_prox_update(
        self,
        rho_power,
        rho_angle,
        power,
        angle,
        nominal_capacity=None,
        power_weights=None,
        angle_weights=None,
    ):
        assert angle is None
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity)

        quadratic_cost = 0.0 if self.quadratic_cost is None else self.quadratic_cost
        pmax = torch.multiply(self.max_power, nominal_capacity) + self.slack
        pmin = torch.multiply(self.min_power, nominal_capacity) - self.slack
        eta = self.efficiency

        # Problem is
        #     min_p    a p1^2 + b p1
        #              + (rho/2) ||p0 - z0||^2 + (rho/2) ||p1 - z1||^2
        #              + {p1 box constraints} + {p1 + eta p0 = 0}
        #
        # Substituting p0 = -p1 / eta removes the equality constraint. The
        # derivative in p1 is
        #     2 a p1 + b + (rho / eta) (p1 / eta + z0) + rho (p1 - z1),
        # which is solved by
        #     p1 = (rho (z1 - z0 / eta) - b) / (2 a + rho (1 + 1 / eta^2)).
        num = rho_power * (power[1] - torch.divide(power[0], eta)) - self.linear_cost
        denom = 2 * quadratic_cost + rho_power * (1.0 + torch.divide(1.0, eta**2))

        p1 = torch.divide(num, denom)
        p1 = torch.clip(p1, pmin, pmax)
        p0 = -torch.divide(p1, eta)

        return [p0, p1], None, None
