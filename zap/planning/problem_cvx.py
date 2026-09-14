import torch
import numpy as np

import zap.util as util
from typing import Union

from zap.network import DispatchOutcome
from zap.layer import DispatchLayer
from zap.planning.operation_objectives import AbstractOperationObjective
from zap.planning.investment_objectives import AbstractInvestmentObjective
from zap.planning.constraints import BudgetConstraintSet

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
        budget_constraints: Union[str, BudgetConstraintSet, None] = None,
    ):
        # Call super initializer
        # Use property setter for deepcopy compatibility
        self.la = np
        self.regularize = regularize
        self.snapshot_weight = snapshot_weight
        super().__init__(
            operation_objective,
            investment_objective,
            layer,
            lower_bounds,
            upper_bounds,
            budget_constraints,
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
        # Backward pass through operation / investment objective.
        #
        # `retain_graph=True` keeps the autograd graph of the *objective* (not
        # of the dispatch solve, which is not a torch graph at all) alive after
        # this call, so a second seed can be differentiated through the same
        # forward pass -- `backward_objective` below.  The graph is one scalar
        # expression over the dispatch outcome; retaining it costs the same
        # memory the forward pass already holds and nothing more.
        self.cost.backward(retain_graph=True)  # Torch backward

        # Direct component of gradients
        dtheta_direct = {k: util.grad_or_zero(v) for k, v in self.torch_kwargs.items()}

        # Indirect, implicitly differentiated component
        dy = DispatchOutcome(*[util.grad_or_zero(x) for x in self.torch_state])
        dy.ground = self.state.ground

        # Backward pass through layer
        dtheta_op = self.layer.backward(self.state, dy, regularize=self.regularize, **self.kwargs)

        # Combine gradients
        dtheta = {k: v + dtheta_op[k] for k, v in dtheta_direct.items()}

        return dtheta

    def backward_objective(self, objective, *, return_adjoint: bool = True):
        """A second VJP on the forward solve already in ``self.state``.

        ``objective`` is any :class:`AbstractOperationObjective`; it is
        evaluated on the retained ``self.torch_state`` and differentiated with
        :func:`torch.autograd.grad` (never ``.backward()``, which would
        accumulate into the ``.grad`` fields the planning gradient lives in).
        The resulting seed is pushed through :meth:`DispatchLayer.backward`, so
        this costs **one extra KKT triangular solve and no dispatch solve**.

        Returns ``(dtheta, adjoint)`` -- ``dtheta[param]`` is
        ``d objective / d param`` including the direct dependence of the
        objective on the parameters, and ``adjoint`` is the layer's adjoint
        state (``adjoint.prices`` = d objective / d injection).  With
        ``return_adjoint=False`` only ``dtheta`` is returned.

        Requires a ``forward(requires_grad=True)`` and a :meth:`backward` (or
        at least a forward) since the last parameter change: the graph it
        differentiates is the one the forward pass built.
        """
        if getattr(self, "torch_state", None) is None:
            raise RuntimeError(
                "backward_objective needs a forward(requires_grad=True) pass first"
            )

        value = objective(self.torch_state, parameters=self.params, la=torch)

        # Differentiate w.r.t. every leaf of the dispatch state and w.r.t. the
        # parameters themselves (the "direct" term, e.g. an objective that reads
        # a capacity as well as a power flow).
        state_leaves, layout = _tensor_leaves(self.torch_state)
        kwarg_names = [
            k
            for k, v in self.torch_kwargs.items()
            if isinstance(v, torch.Tensor) and v.requires_grad
        ]
        candidates = [x for x in state_leaves if x.requires_grad]
        candidates += [self.torch_kwargs[k] for k in kwarg_names]
        # `torch.autograd.grad` must not see the same tensor twice.
        inputs: list[torch.Tensor] = []
        seen: set[int] = set()
        for tensor in candidates:
            if id(tensor) not in seen:
                seen.add(id(tensor))
                inputs.append(tensor)

        grads: list = [None] * len(inputs)
        if inputs and isinstance(value, torch.Tensor) and value.requires_grad:
            grads = list(
                torch.autograd.grad(
                    value,
                    inputs,
                    retain_graph=True,
                    allow_unused=True,
                )
            )

        # Map the flat gradient list back onto the state structure, with
        # explicit zeros for the entries the objective does not touch (most of
        # them: `UnservedEnergyObjective` reads only the load rows of `power`).
        grad_by_id = {}
        for tensor, grad in zip(inputs, grads):
            grad_by_id[id(tensor)] = grad
        dy = DispatchOutcome(*_rebuild_like(layout, grad_by_id))
        dy.ground = self.state.ground

        dtheta_direct = {}
        for k, v in self.torch_kwargs.items():
            grad = grad_by_id.get(id(v)) if k in kwarg_names else None
            if grad is None:
                dtheta_direct[k] = (
                    torch.zeros_like(v)
                    if isinstance(v, torch.Tensor)
                    else np.zeros_like(np.asarray(v, dtype=float))
                )
            else:
                dtheta_direct[k] = grad

        dtheta_op, adjoint = self.layer.backward(
            self.state,
            dy,
            regularize=self.regularize,
            return_adjoint=True,
            **self.kwargs,
        )
        dtheta = {k: v + dtheta_op[k] for k, v in dtheta_direct.items()}

        if return_adjoint:
            return dtheta, adjoint
        return dtheta


def _tensor_leaves(state: DispatchOutcome):
    """``(leaves, layout)`` of a torchified :class:`DispatchOutcome`.

    ``layout`` mirrors the eight entries of the outcome with the tensors left in
    place, so :func:`_rebuild_like` can walk it and swap each tensor for its
    gradient (or a zero of the same shape).  ``DispatchOutcome`` entries are
    tensors, arbitrarily nested lists of tensors, or ``None``.
    """
    leaves: list[torch.Tensor] = []

    def walk(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return [walk(xi) for xi in x]
        leaves.append(x)
        return x

    layout = [walk(state[i]) for i in range(len(state))]
    return leaves, layout


def _rebuild_like(layout, grad_by_id: dict):
    """The ``layout`` structure with every tensor replaced by its gradient."""

    def walk(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return [walk(xi) for xi in x]
        grad = grad_by_id.get(id(x))
        return torch.zeros_like(x) if grad is None else grad

    return [walk(entry) for entry in layout]
