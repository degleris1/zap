import cvxpy as cp
import numpy as np
from copy import deepcopy

from zap.network import DispatchOutcome
from zap.planning.operation_objectives import EmissionsObjective
from zap.planning.problem_abstract import StochasticPlanningProblem


class MonolithicPlanningProblem:
    """Joint single-level capacity-expansion LP.

    Builds and solves the joint problem
        min_{theta, y}  c_inv(theta) + c_op(y; theta)
        s.t.            theta in [lower, upper], y in F(theta)
    using only the primal dispatch problem -- no dual devices, no McCormick
    envelope, no strong-duality coupling.

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

        if isinstance(self.problem, StochasticPlanningProblem):
            inv_func = self.problem.subproblems[0].investment_objective
        else:
            inv_func = self.problem.investment_objective

        investment_objective = inv_func(la=cp, **network_parameters)

        return network_parameters, lower_bounds, upper_bounds, investment_objective

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

        emissions_term = None
        if self.emissions_limit is not None:
            # Build a fresh EmissionsObjective per subproblem -- the device
            # list is sample_time-sliced and differs across subs.
            emissions_term = EmissionsObjective(devices)(
                y, parameters=parameters, la=cp
            )

        return operation_objective, primal_constraints, emissions_term

    def solve(self):
        """Solve the joint single-level LP."""
        net_params, lower, upper, investment_objective = self.model_outer_problem()
        box_constraints = [lower[p] <= net_params[p] for p in sorted(net_params.keys())]
        box_constraints += [net_params[p] <= upper[p] for p in sorted(net_params.keys())]

        operation_objectives = []
        primal_constraints = []
        emissions_terms = []

        if isinstance(self.problem, StochasticPlanningProblem):
            print(
                f"Solving monolithic single-level problem with "
                f"{len(self.problem.subproblems)} scenarios."
            )
            subs = self.problem.subproblems
        else:
            subs = [self.problem]

        for sub in subs:
            op, pc, em = self.setup_inner_problem(sub, net_params)
            operation_objectives.append(op)
            primal_constraints += list(pc)
            if em is not None:
                emissions_terms.append(em)

        constraints = box_constraints + list(primal_constraints)
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
            "investment_objective": investment_objective,
            "operation_objective": operation_objectives,
            "primal_constraints": primal_constraints,
            "emissions_terms": emissions_terms,
            "emissions_limit": self.emissions_limit,
            "emissions_constraint": emissions_constraint,
            "problem": problem,
        }

        optimal_parameters = {p: net_params[p].value for p in net_params.keys()}
        return optimal_parameters, data
