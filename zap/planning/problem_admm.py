import torch

from zap.util import grad_or_zero
from zap.planning.operation_objectives import AbstractOperationObjective
from zap.planning.investment_objectives import AbstractInvestmentObjective
from zap.admm import ADMMLayer

from .problem_abstract import AbstractPlanningProblem


class PlanningProblemADMM(AbstractPlanningProblem):
    """Models long-term multi-value expansion planning."""

    def __init__(
        self,
        operation_objective: AbstractOperationObjective,
        investment_objective: AbstractInvestmentObjective,
        layer: ADMMLayer,
        lower_bounds: dict = None,
        upper_bounds: dict = None,
        snapshot_weight: float = 1.0,
    ):
        # Call super initializer
        self.la = torch
        self.snapshot_weight = snapshot_weight
        self.rho_power_history = []
        self.rho_angle_history = []
        super().__init__(
            operation_objective, investment_objective, layer, lower_bounds, upper_bounds
        )

    def forward(
        self, requires_grad: bool = False, batch=None, initial_state=None, **kwargs
    ):
        # Enable gradient tracking if needed
        if requires_grad:
            print("Tracking gradient during forward pass.")
            for p, v in kwargs.items():
                kwargs[p] = v.detach().clone()  # Copy first
                kwargs[p].requires_grad = True

        params = self.layer.setup_parameters(**kwargs)

        # Forward pass through dispatch layer
        # Store this for efficient backward pass
        # with torch.amp.autocast(device_type=self.layer.solver.machine):
        self.rho_power_history += [self.layer.solver.rho_power]
        self.rho_angle_history += [self.layer.solver.rho_angle]
        admm_state = self.layer.forward(initial_state=initial_state, **kwargs)
        state = admm_state.as_outcome()

        op_cost = self.operation_objective(state, parameters=params, la=torch)
        inv_cost = self.investment_objective(**kwargs, la=torch)

        # Store unweighted costs for tracking
        self.op_cost = op_cost.detach()
        self.inv_cost = inv_cost.detach()

        # Apply snapshot weight to operational cost (to annualize from snapshot)
        self.cost = (self.snapshot_weight * op_cost) + inv_cost
        self.kwargs = kwargs
        self.params = params
        self.state = state
        self.admm_state = admm_state

        return self.cost

    def backward(self):
        # Backward pass through everything!
        self.cost.backward(retain_graph=False)  # Torch backward
        print("Backwards pass evaluated.")

        # Combine gradients
        dtheta = {k: grad_or_zero(v) for k, v in self.kwargs.items()}

        return dtheta
