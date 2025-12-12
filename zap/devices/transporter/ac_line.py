import numpy as np
import scipy.sparse as sp
import torch

from zap.devices.abstract import make_dynamic
from zap.util import envelope_variable, use_envelope
from .dc_line import PowerLine


class ACLine(PowerLine):
    """A symmetric transporter with phase angle constraints."""

    def __init__(
        self,
        *,
        num_nodes,
        name,
        source_terminal,
        sink_terminal,
        capacity,  # todo: name this dynamic_capacity to match generator naming?
        susceptance,
        linear_cost=None,
        quadratic_cost=None,
        nominal_capacity=None,
        capital_cost=None,
        slack=None,
        min_nominal_capacity=None,
        max_nominal_capacity=None,
        reconductoring_cost=None,
        reconductoring_threshold=None,
    ):
        self.susceptance = make_dynamic(susceptance)
        self.has_changed = True

        super().__init__(
            num_nodes=num_nodes,
            name=name,
            source_terminal=source_terminal,
            sink_terminal=sink_terminal,
            capacity=capacity,
            linear_cost=linear_cost,
            quadratic_cost=quadratic_cost,
            nominal_capacity=nominal_capacity,
            capital_cost=capital_cost,
            slack=slack,
            min_nominal_capacity=min_nominal_capacity,
            max_nominal_capacity=max_nominal_capacity,
            reconductoring_cost=reconductoring_cost,
            reconductoring_threshold=reconductoring_threshold,
        )

    @property
    def is_ac(self):
        return True

    # ====
    # CORE MODELING FUNCTIONS
    # ====

    def equality_constraints(
        self,
        power,
        angle,
        u,
        nominal_capacity=None,
        susceptance=None,
        la=np,
        envelope=None,
        mask=None,
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)
        susceptance = self.parameterize(susceptance=susceptance, la=la)

        # Regular transporter constraints
        eq_constraints = super().equality_constraints(
            power, angle, u, nominal_capacity=nominal_capacity, la=la
        )

        # Linearized power flow constraints
        angle_diff = angle[0] - angle[1]

        if mask is None:
            b = susceptance
        else:
            b = susceptance - la.multiply(susceptance, mask)

        if use_envelope(envelope):  # When line is plannable
            print("Envelope relaxation applied to AC line.")
            pnom_dtheta = self.get_envelope_variable(
                *envelope, angle_diff, nominal_capacity
            )
        else:
            pnom_dtheta = la.multiply(angle_diff, nominal_capacity)

        eq_constraints += [power[1] - la.multiply(b, pnom_dtheta)]

        return eq_constraints

    def get_envelope_variable(self, env, lb, ub, angle_diff, nominal_capacity):
        # Get lower and upper bounds for the angle difference
        # dtheta_max = fmax / b = pnom * pmax / b
        ub_angle_diff = ub["nominal_capacity"] * self.max_power / self.susceptance
        lb_angle_diff = -ub_angle_diff

        return envelope_variable(
            nominal_capacity,
            angle_diff,
            lb["nominal_capacity"],
            ub["nominal_capacity"],
            lb_angle_diff,
            ub_angle_diff,
            *env,
        )

    # ====
    # PLANNING FUNCTIONS
    # ====

    def sample_time(self, time_periods, original_time_horizon):
        dev = super().sample_time(time_periods, original_time_horizon)

        if dev.susceptance.shape[1] > 1:
            dev.susceptance = dev.susceptance[:, time_periods]

        if dev.nominal_capacity.shape[1] > 1:
            dev.nominal_capacity = dev.nominal_capacity[:, time_periods]

        return dev

    # ====
    # DIFFERENTIATION
    # ====

    def _equality_matrices(
        self, equalities, nominal_capacity=None, susceptance=None, la=np
    ):
        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity, la=la)
        susceptance = self.parameterize(susceptance=susceptance, la=la)

        size = equalities[0].power[0].shape[1]

        time_horizon = int(size / self.num_devices)
        shaped_zeros = np.zeros((self.num_devices, time_horizon))

        b_pnom = la.multiply(susceptance, nominal_capacity)
        b_mat = shaped_zeros + b_pnom

        equalities[1].power[1] += sp.eye(size)
        equalities[1].angle[0] += -sp.diags(b_mat.ravel())
        equalities[1].angle[1] += sp.diags(b_mat.ravel())

        return super()._equality_matrices(
            equalities, nominal_capacity=nominal_capacity, la=la
        )

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
        susceptance=None,
        power_weights=None,
        angle_weights=None,
        cvx_mode=False,
        contingency_mask=None,
    ):
        mask = contingency_mask
        if mask is None:
            nc = 0
        else:
            nc = mask.shape[0]
            mask = mask.T.unsqueeze(1)

        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity)
        susceptance = self.parameterize(susceptance=susceptance)

        if cvx_mode:
            return self.cvx_admm_prox_update(
                rho_power, rho_angle, power, angle, nominal_capacity=None
            )

        quad_cost = 0.0 if self.quadratic_cost is None else self.quadratic_cost

        # Cache once per solve
        if self.has_changed:
            pmax = torch.multiply(self.max_power, nominal_capacity) + self.slack
            pmin = torch.multiply(self.min_power, nominal_capacity) - self.slack
            b_pnom = torch.multiply(susceptance, nominal_capacity)
            mu = torch.divide(1.0, 2.0 * b_pnom)
            bool_mask = None

            if nc > 0:
                # Just expand pmin / pmax
                pmax = pmax.unsqueeze(2)
                pmin = pmin.unsqueeze(2)

                # Multiply susceptance by 1 - mask
                b_pnom = b_pnom.unsqueeze(2)
                b_pnom = b_pnom * (1.0 - mask)

                # Define mu normally
                # When b = 0 (line outage), then mu = infty
                # This will cause numerical issues, so we set mu = 0 for those values and pass the
                # mask to the update function to handle the division by mu
                mu = mu.unsqueeze(2)
                mu = mu * (1.0 - mask)

                # Create a boolean mask for the update function
                bool_mask = mask == 1.0

            self.admm_data = (pmax, pmin, b_pnom, mu, bool_mask)

        pmax, pmin, b_pnom, mu, bool_mask = self.admm_data

        if nc > 0:
            return _admm_prox_update_masked(
                power,
                angle,
                rho_power,
                rho_angle,
                b_pnom,
                mu,
                quad_cost,
                pmin,
                pmax,
                mask,
                bool_mask,
            )
        else:
            return _admm_prox_update(
                power, angle, rho_power, rho_angle, b_pnom, mu, quad_cost, pmin, pmax
            )

    def cvx_admm_prox_update(
        self, rho_power, rho_angle, power, angle, nominal_capacity=None
    ):
        print("Solving AC line prox update via CVX...")
        import cvxpy as cp

        nominal_capacity = self.parameterize(nominal_capacity=nominal_capacity)

        pmax = np.multiply(self.max_power, nominal_capacity) + self.slack
        pmin = np.multiply(self.min_power, nominal_capacity) - self.slack
        susceptance = np.multiply(self.susceptance, nominal_capacity)

        p0 = cp.Variable(power[0].shape)
        p1 = cp.Variable(power[0].shape)
        theta0 = cp.Variable(angle[0].shape)
        theta1 = cp.Variable(angle[0].shape)

        objective = rho_power * (
            cp.sum_squares(p0 - power[0]) + cp.sum_squares(p1 - power[1])
        )
        objective += rho_angle * (
            cp.sum_squares(theta0 - angle[0]) + cp.sum_squares(theta1 - angle[1])
        )

        constraints = [
            p1 + p0 == 0,
            p1 == cp.multiply(susceptance, (theta0 - theta1)),
            p1 <= pmax,
            p1 >= pmin,
        ]

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.MOSEK, verbose=True)

        return [p0.value, p1.value], [theta0.value, theta1.value]


@torch.jit.script
def _admm_prox_update(
    power: list[torch.Tensor],
    angle: list[torch.Tensor],
    rho_power: float,
    rho_angle: float,
    b: torch.Tensor,
    mu: torch.Tensor,
    quad_cost: float,
    pmin: torch.Tensor,
    pmax: torch.Tensor,
):
    # See transporter for details on derivation
    # Here, we also have angle variables
    # However, we can write p0 and theta0 in terms of p1 and theta1
    #   p0 = -p1
    #   theta0 = theta1 + p1 / susceptance
    #
    # The solution for theta1 is:
    #   theta1 = (1/2) (angle1 + angle0) - mu * p1
    # where:
    #   mu = 1 / (2 * susceptance)

    # Now we can solve a minimization problem over just p1
    # with solution:
    #   (2 a + 2 rho + 2 rho mu^2) p + b sign(p)
    #   =
    #   rho (power1 - power0 + mu angle[1] - mu angle[0])
    num = rho_power * (power[1] - power[0]) + rho_angle * mu * (angle[0] - angle[1])
    denom = 2 * (quad_cost + rho_power + rho_angle * torch.pow(mu, 2))

    p1 = torch.divide(num, denom)
    p1 = torch.clip(p1, pmin, pmax)

    # Solve for other variables
    p0 = -p1
    theta1 = 0.5 * angle[0] + 0.5 * angle[1] - mu * p1
    theta0 = theta1 + p1 / b

    return [p0, p1], [theta0, theta1], None


@torch.jit.script
def _admm_prox_update_masked(
    power: list[torch.Tensor],
    angle: list[torch.Tensor],
    rho_power: float,
    rho_angle: float,
    b: torch.Tensor,
    mu: torch.Tensor,
    quad_cost: float,
    pmin: torch.Tensor,
    pmax: torch.Tensor,
    mask: torch.Tensor,
    bool_mask: torch.Tensor,
):
    num = rho_power * (power[1] - power[0]) + rho_angle * mu * (angle[0] - angle[1])
    denom = 2 * (quad_cost + rho_power + rho_angle * torch.pow(mu, 2))

    p1 = torch.divide(num, denom)
    p1 = torch.clip(p1, pmin, pmax)

    # Apply mask, zeroing out values where mask is 1
    p1 = p1 * (1.0 - mask)

    # Solve for other variables
    p0 = -p1
    theta1 = torch.where(bool_mask, angle[1], 0.5 * angle[0] + 0.5 * angle[1] - mu * p1)
    theta0 = torch.where(bool_mask, angle[0], theta1 + p1 / b)

    return [p0, p1], [theta0, theta1], None
