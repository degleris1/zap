"""The single-level LP planning method (``monolithic`` / ``stochastic`` / ``relaxed``).

WP5 spec section 5.5 and decision D-W2: ``monolithic`` and ``stochastic`` are
one implementation at two block sizes, and ``relaxed`` is the same build with a
strong-duality solve instead of a primal one.  All three are presets over this
class; what separates them is ``selection`` and
``planning.single_level.kind``, not the method.
"""

from __future__ import annotations

import time

from zap.planning import MonolithicPlanningProblem, RelaxedPlanningProblem

from ...config import ConfigError
from .. import base, constraints, objectives

__all__ = ["SingleLevelMethod", "optimality_gap", "problem_size"]


def optimality_gap(objective: float, lower_bound: float | None) -> float | None:
    if lower_bound in (None, 0.0):
        return None
    return (float(objective) - float(lower_bound)) / abs(float(lower_bound))


def problem_size(problem) -> dict:
    """Scalar variable / constraint counts of a solved cvxpy problem."""
    metrics = problem.size_metrics
    return {
        "n_variables": int(metrics.num_scalar_variables),
        "n_constraints": int(metrics.num_scalar_eq_constr) + int(metrics.num_scalar_leq_constr),
    }


def _value(expr) -> float | None:
    if expr is None:
        return None
    value = getattr(expr, "value", None)
    return None if value is None else float(value)


def _sum_values(exprs) -> float | None:
    if not exprs:
        return None
    values = [_value(e) for e in exprs]
    if any(v is None for v in values):
        return None
    return float(sum(values))


@base.register_method
class SingleLevelMethod(base.PlanningMethod):
    """One LP over every sampled block; no decomposition, no gradients."""

    name = "single_level"

    def solve(self, ctx: base.PlanningContext) -> base.PlanningResult:
        opts = self.options
        sl = opts["single_level"]
        kind = sl["kind"]
        solver_name = base.require_solver(sl["solver"])
        solver_kwargs = dict(sl["solver_kwargs"] or {})
        limit = constraints.emissions_limit(self.cfg, ctx.annualization_factor)

        t0 = time.perf_counter()
        if kind == "primal":
            single_level = MonolithicPlanningProblem(
                ctx.problem,
                inf_value=float(sl["inf_value"]),
                solver=base.solver_object(solver_name),
                solver_kwargs=solver_kwargs,
                emissions_limit=limit,
            )
        elif kind == "strong_duality":
            if limit is not None:  # pragma: no cover - validate_emissions gets here first
                raise ConfigError(
                    "planning.emissions.mode 'cap' requires "
                    "planning.single_level.kind 'primal' (D-W5)."
                )
            single_level = RelaxedPlanningProblem(
                ctx.problem,
                max_price=float(sl["price_bound"]),
                solver=base.solver_object(solver_name),
                solver_kwargs=solver_kwargs,
            )
        else:
            raise ConfigError(
                f"unknown planning.single_level.kind {kind!r}; "
                "expected 'primal' or 'strong_duality'"
            )

        parameters, data = single_level.solve()
        solve_seconds = time.perf_counter() - t0

        problem = data["problem"]
        if problem.value is None:
            raise RuntimeError(
                f"single-level solve failed: status={problem.status} "
                f"(solver={solver_name}, kind={kind})"
            )

        capex = _value(data.get("investment_objective"))
        opex = _sum_values(data.get("operation_objective"))

        # ``MonolithicPlanningProblem`` builds the (weighted) emissions terms
        # unconditionally, so the emissions of the design are read off the
        # solved LP instead of costing a second full conic forward pass.
        terms = data.get("emissions_terms") or []
        if terms and all(getattr(t, "value", None) is not None for t in terms):
            emissions = float(sum(float(t.value) for t in terms))
        else:  # RelaxedPlanningProblem builds no emissions terms
            emissions = objectives.evaluate_emissions(
                ctx.problem, objectives.emissions_objectives(ctx.problem), parameters
            )

        # `price` mode adds `price * emissions` to every subproblem's operation
        # objective, so the LP optimum is system cost *plus* a carbon payment.
        # The reported objective is system cost (PROJECT.md 2.4): the payment is
        # netted out of both `opex_raw` and `raw`, and recorded on its own.
        carbon_payment = float(objectives.emissions_price(self.cfg)) * emissions
        objective_raw = float(problem.value) - carbon_payment
        if opex is not None:
            opex = opex - carbon_payment

        return base.PlanningResult.from_context(
            ctx,
            method=self.name,
            preset=self.preset,
            kind=kind,
            parameters=parameters,
            objective={
                "raw": objective_raw,
                "capex_raw": capex,
                "opex_raw": opex,
                "carbon_payment_raw": carbon_payment,
                # An LP is its own lower bound (stated, like the objective, net
                # of the carbon payment).
                "lower_bound_raw": objective_raw,
                "optimality_gap": 0.0,
                "emissions_tonnes_raw": emissions,
            },
            solver={
                "name": solver_name,
                "status": str(problem.status),
                "kwargs": solver_kwargs,
                **problem_size(problem),
                "n_subproblems": len(ctx.problem.subproblems),
            },
            timing={"build_s": ctx.build_seconds, "solve_s": solve_seconds},
            compute={"num_workers": int(opts["num_workers"])},
        )
