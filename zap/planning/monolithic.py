import cvxpy as cp
import numpy as np
from copy import deepcopy

from zap.network import DispatchOutcome
from zap.planning.operation_objectives import EmissionsObjective
from zap.planning.problem_abstract import StochasticPlanningProblem, weighted_subproblems


class MonolithicPlanningProblem:
    """Joint single-level capacity-expansion LP.

    Builds and solves the joint problem
        min_{theta, y}  c_inv(theta) + c_op(y; theta)
        s.t.            theta in [lower, upper], y in F(theta)
    using only the primal dispatch problem -- no dual devices, no McCormick
    envelope, no strong-duality coupling.

    For a :class:`StochasticPlanningProblem` the objective is the weighted
    mixture its forward pass evaluates,
    ``sum_i w_i * (snapshot_weight_i * op_i(y_i) + inv_i(theta))``, where
    ``inv_i`` is subproblem ``i``'s investment objective (capital cost already
    pro-rated to the block by ``sample_time``).  Summed over all blocks the
    investment term is therefore ``coverage * CAPEX``, and the LP optimum equals
    ``StochasticPlanningProblem.forward()`` at the LP's optimal parameters.

    Valid when the planning operation_objective matches the dispatch cost.
    Suitable for transport networks (no AC lines) because the primal dispatch
    is a pure LP; AC networks have a bilinear AC-line equality that CVXPY
    would reject without an envelope -- use RelaxedPlanningProblem in that
    case.
    """

    def __init__(
        self,
        problem,
        inf_value=100.0,
        solver=None,
        solver_kwargs=None,
        emissions_limit=None,
    ):
        self.problem = deepcopy(problem)
        self.inf_value = inf_value
        self.solver = solver if solver is not None else self.problem.layer.solver
        self.solver_kwargs = (
            solver_kwargs if solver_kwargs is not None else self.problem.layer.solver_kwargs
        )
        # Hard cap on total emissions summed across subproblems. When None, no
        # emissions constraint is added; the LP keeps its current behavior.
        self.emissions_limit = emissions_limit

    def setup_parameters(self, **kwargs):
        return self.problem.layer.setup_parameters(**kwargs)

    def model_outer_problem(self):
        """Define outer variables, bounds, and investment cost."""
        network_parameters = {
            p: cp.Variable(lower.shape) for p, lower in self.problem.lower_bounds.items()
        }

        lower_bounds = {}
        upper_bounds = {}
        for p in network_parameters.keys():
            lower = self.problem.lower_bounds[p]
            upper = self.problem.upper_bounds[p]
            inf_param = self.inf_value * np.max(lower)
            upper = np.where(upper == np.inf, inf_param, upper)
            lower_bounds[p] = lower
            upper_bounds[p] = upper

        # Investment term: sum_i w_i * inv_i(theta), matching the forward pass of
        # StochasticPlanningProblem exactly.  Every subproblem's devices came out
        # of `sample_time(block, total_hours)`, which pro-rates capital cost by
        # `block_hours / total_hours`, so a single subproblem's investment
        # objective is only that block's share of the capex.  Using
        # `subproblems[0]` alone (as this did) charged the LP
        # `(block_0_hours / total_hours) * CAPEX` instead of `coverage * CAPEX`,
        # i.e. it under-weighted capital cost by the number of blocks.
        subs, weights = weighted_subproblems(self.problem)
        investment_objective = sum(
            w * sub.investment_objective(la=cp, **network_parameters)
            for w, sub in zip(weights, subs)
        )

        return network_parameters, lower_bounds, upper_bounds, investment_objective

    def budget_constraints(self, net_params):
        """The problem's budget constraints, as cvxpy constraints on ``net_params``.

        The gradient path enforces a :class:`BudgetConstraintSet` by projection;
        a single-level LP has to state it, or a budget CSV is silently a no-op.
        """
        budget = getattr(self.problem, "budget_constraints", None)
        if budget is None or len(budget) == 0:
            return []
        return budget.cvxpy_constraints(net_params)

    def setup_inner_problem(self, sub_problem, net_params):
        """Build the primal dispatch problem for one subproblem."""
        net, devices = sub_problem.layer.network, sub_problem.layer.devices
        parameters = sub_problem.layer.setup_parameters(**net_params)

        _, primal_constraints, primal_data = net.model_dispatch_problem(
            devices,
            sub_problem.time_horizon,
            dual=False,
            parameters=parameters,
            envelope=None,
            lower_param=None,
            upper_param=None,
        )

        y = DispatchOutcome(
            power=primal_data["power"],
            angle=primal_data["angle"],
            global_angle=primal_data["global_angle"],
            local_variables=primal_data["local_variables"],
            prices=None,
            phase_duals=None,
            local_equality_duals=None,
            local_inequality_duals=None,
        )

        operation_objective = sub_problem.operation_objective(y, parameters=parameters, la=cp)

        # Build a fresh EmissionsObjective per subproblem -- the device list is
        # sample_time-sliced and differs across subs.  This is built whether or
        # not a cap is imposed: it is a cvxpy *expression*, not a constraint, so
        # it costs nothing to carry, and reading `.value` off the solved problem
        # is how the caller reports the design's emissions without paying for a
        # second full forward pass.
        emissions_term = EmissionsObjective(devices)(y, parameters=parameters, la=cp)

        return operation_objective, primal_constraints, emissions_term, y

    def solve(self):
        """Solve the joint single-level LP."""
        net_params, lower, upper, investment_objective = self.model_outer_problem()
        box_constraints = [lower[p] <= net_params[p] for p in sorted(net_params.keys())]
        box_constraints += [net_params[p] <= upper[p] for p in sorted(net_params.keys())]
        budget_constraints = self.budget_constraints(net_params)

        operation_objectives = []
        primal_constraints = []
        emissions_terms = []
        dispatch_outcomes = []

        subs, weights = weighted_subproblems(self.problem)
        if isinstance(self.problem, StochasticPlanningProblem):
            print(f"Solving monolithic single-level problem with {len(subs)} scenarios.")

        for w, sub in zip(weights, subs):
            op, pc, em, y = self.setup_inner_problem(sub, net_params)
            # Same weighting as StochasticPlanningProblem.forward:
            # w_i * snapshot_weight_i * op_i.  With the uniform weights and unit
            # snapshot weights of phase 1 this is the identity.
            scale = w * float(getattr(sub, "snapshot_weight", 1.0))
            operation_objectives.append(scale * op)
            primal_constraints += list(pc)
            emissions_terms.append(scale * em)
            # The primal dispatch of this subproblem, so a caller can report
            # quantities the objective does not decompose into (e.g. the
            # negative-price part of the operation cost).  These are cvxpy
            # expressions: read `.value` after the solve.
            dispatch_outcomes.append(y)

        constraints = box_constraints + budget_constraints + list(primal_constraints)
        emissions_constraint = None
        if self.emissions_limit is not None:
            emissions_constraint = cp.sum(emissions_terms) <= self.emissions_limit
            constraints.append(emissions_constraint)

        problem = cp.Problem(
            cp.Minimize(investment_objective + cp.sum(operation_objectives)),
            constraints,
        )
        problem.solve(solver=self.solver, **self.solver_kwargs)

        data = {
            "network_parameters": net_params,
            "lower_bounds": lower,
            "upper_bounds": upper,
            "box_constraints": box_constraints,
            "budget_constraints": budget_constraints,
            "investment_objective": investment_objective,
            "operation_objective": operation_objectives,
            "primal_constraints": primal_constraints,
            "emissions_terms": emissions_terms,
            "dispatch_outcomes": dispatch_outcomes,
            "subproblems": list(subs),
            "subproblem_scales": [
                float(w) * float(getattr(sub, "snapshot_weight", 1.0))
                for w, sub in zip(weights, subs)
            ],
            "emissions_limit": self.emissions_limit,
            "emissions_constraint": emissions_constraint,
            "problem": problem,
        }

        optimal_parameters = {p: net_params[p].value for p in net_params.keys()}
        return optimal_parameters, data
