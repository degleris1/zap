import torch
import numpy as np

import zap.util as util
from zap.network import DispatchOutcome
from zap.layer import DispatchLayer
from zap.planning.operation_objectives import AbstractOperationObjective
from zap.planning.investment_objectives import AbstractInvestmentObjective

from .problem_abstract import AbstractPlanningProblem


class PlanningProblemCVX(AbstractPlanningProblem):
    """Models long-term multi-value expansion planning."""

    def __init__(
        self,
        operation_objective: AbstractOperationObjective,
        investment_objective: AbstractInvestmentObjective,
        layer: DispatchLayer,
        lower_bounds: dict = None,
        upper_bounds: dict = None,
        regularize=1e-6,
        snapshot_weight: float = 1.0,
    ):
        # Call super initializer
        self.la = np
        self.regularize = regularize
        self.snapshot_weight = snapshot_weight
        super().__init__(
            operation_objective, investment_objective, layer, lower_bounds, upper_bounds
        )

    def forward(self, requires_grad: bool = False, batch=None, **kwargs):
        torch_kwargs = {}

        if requires_grad:
            la = torch
        else:
            la = np

        for p, v in kwargs.items():
            if requires_grad:
                torch_kwargs[p] = util.torchify(v, requires_grad=True)
            else:
                torch_kwargs[p] = v

        params = self.layer.setup_parameters(**torch_kwargs)

        # Forward pass through dispatch layer
        # Store this for efficient backward pass
        self.state = self.layer.forward(**kwargs)

        if requires_grad:
            self.torch_state = self.state.torchify(requires_grad=True)
        else:
            self.torch_state = self.state

        op_cost = self.operation_objective(self.torch_state, parameters=params, la=la)
        inv_cost = self.investment_objective(**torch_kwargs, la=la)

        # Store unweighted costs for tracking
        self.op_cost = op_cost
        self.inv_cost = inv_cost

        # Apply snapshot weight to operational cost (to annualize from snapshot)
        self.cost = (self.snapshot_weight * op_cost) + inv_cost

        self.kwargs = kwargs
        self.torch_kwargs = torch_kwargs
        self.params = params

        return self.cost

    def backward(self):
        # Backward pass through operation / investment objective
        self.cost.backward()  # Torch backward

        # Direct component of gradients
        dtheta_direct = {k: util.grad_or_zero(v) for k, v in self.torch_kwargs.items()}

        # Indirect, implicitly differentiated component
        dy = DispatchOutcome(*[util.grad_or_zero(x) for x in self.torch_state])
        dy.ground = self.state.ground

        # Backward pass through layer
        dtheta_op = self.layer.backward(
            self.state, dy, regularize=self.regularize, **self.kwargs
        )

        # Combine gradients
        dtheta = {k: v + dtheta_op[k] for k, v in dtheta_direct.items()}

        return dtheta
