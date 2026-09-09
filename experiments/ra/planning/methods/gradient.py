"""The gradient-descent decomposition planning method (WP5 spec section 5.6).

Ported from ``experiments/multi_year/runner.py`` steps 6-9 with the changes the
spec mandates:

* the ``peak_net_load_fill=`` keyword is dropped from the dual-ascent inner
  solve -- it was never bound, so that branch raised ``NameError`` on its first
  outer iteration and has never run (spec sections 9.1 / 11);
* the emissions target is annualized like the cap, so the same YAML means the
  same physical target at any block coverage;
* wandb is gone.  The quantities its trackers computed per iteration are
  computed once, after the solve, and land in the result.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass

import zap.planning.trackers as tr
from zap.planning import GradientDescent, MonolithicPlanningProblem

from .. import base, constraints, objectives
from .single_level import optimality_gap

__all__ = ["GradientMethod", "WarmStart"]


@dataclass
class WarmStart:
    """The result of the optional single-level warm-start solve.

    ``objective_raw`` is the LP optimum **in the units of its own block set**:
    every subproblem's capital cost is pro-rated by ``block_hours /
    total_hours``, so an LP over a subset of the blocks is a smaller number than
    the gradient loss over all of them, and is *not* a lower bound on it.  It is
    a valid lower bound only when the warm start ran on exactly the block set
    the gradient loop optimizes (``blocks_match``); otherwise the number is kept
    for the record and no bound (and no optimality gap) is reported.
    """

    parameters: dict | None = None
    objective_raw: float | None = None
    sampled_hours: int | None = None
    blocks_match: bool = False
    seconds: float = 0.0

    @property
    def lower_bound_raw(self) -> float | None:
        return self.objective_raw if self.blocks_match else None

    @property
    def solve_bound(self) -> float | None:
        """What to hand ``problem.solve(lower_bound=...)``.

        ``trackers.suboptimality`` divides by ``lower_bound`` and substitutes
        ``1.0`` for ``None``, which would silently report ``J - 1`` as a gap;
        NaN says "not measured" instead.
        """
        return self.objective_raw if self.blocks_match else math.nan


@base.register_method
class GradientMethod(base.PlanningMethod):
    """Bi-level planning by implicit-differentiation gradient descent."""

    name = "gradient"

    # -- helpers ------------------------------------------------------------

    def _rebuild_problem(self, ctx, blocks):
        """A second ``StochasticPlanningProblem`` over ``blocks`` (warm start only)."""
        return ctx.sampler.create_stochastic_problem(
            blocks=blocks,
            parameter_names=ctx.parameter_names,
            operation_objective_fn=objectives.operation_objective_factory(
                ctx.sampler.base_network, self.cfg
            ),
            investment_objective_fn=objectives.investment_objective_factory(),
            lower_bounds=ctx.lower_bounds,
            upper_bounds=ctx.upper_bounds,
            budget_constraints=constraints.load_budget_constraints(
                self.cfg, ctx.parameter_names, ctx.sampler.base_devices
            ),
            weights=None,
            device_hook=self.prepare_devices,
            **self.layer_kwargs(),
        )

    def warm_start(self, ctx) -> WarmStart:
        """Solve the single-level LP the gradient loop starts from.

        Runs *before* ``initialize_workers``: ``MonolithicPlanningProblem``
        deep-copies the problem and a worker pool is unpicklable.  That ordering
        is load-bearing (spec section 1.1, step 7).
        """
        ws = self.options["warm_start"]
        if not ws["enabled"]:
            return WarmStart()

        problem = ctx.problem
        blocks = list(ctx.blocks)
        num_blocks = ws["num_blocks"]
        if num_blocks is not None and int(num_blocks) < len(ctx.blocks):
            block_size = ctx.blocks[0][1] - ctx.blocks[0][0]
            blocks = ctx.sampler.sample_blocks(
                block_size=block_size, num_blocks=int(num_blocks), strategy="uniform"
            )
            problem = self._rebuild_problem(ctx, blocks)

        solver_name = base.require_solver(ws["solver"])
        t0 = time.perf_counter()
        single_level = MonolithicPlanningProblem(
            problem,
            inf_value=float(self.options["single_level"]["inf_value"]),
            solver=base.solver_object(solver_name),
            solver_kwargs=dict(ws["solver_kwargs"] or {}),
        )
        params, data = single_level.solve()
        seconds = time.perf_counter() - t0

        if data["problem"].value is None:
            raise RuntimeError(
                f"warm-start solve failed: status={data['problem'].status}. "
                "Set planning.warm_start.enabled=false to run without it."
            )

        # Report the LP optimum net of the carbon payment, like every other
        # objective this method records, so the bound and the loss are the same
        # quantity when they are comparable at all.
        objective = float(data["problem"].value)
        terms = data.get("emissions_terms") or []
        if terms and all(getattr(t, "value", None) is not None for t in terms):
            price = float(objectives.emissions_price(self.cfg))
            objective -= price * float(sum(float(t.value) for t in terms))

        return WarmStart(
            parameters=deepcopy(params),
            objective_raw=objective,
            sampled_hours=int(sum(stop - start for start, stop in blocks)),
            blocks_match=list(blocks) == list(ctx.blocks),
            seconds=seconds,
        )

    def _algorithm(self):
        opt = self.options["optimizer"]
        algorithm = GradientDescent(step_size=float(opt["step_size"]), clip=float(opt["clip"]))
        trackers = list(tr.DEFAULT_TRACKERS)
        if opt["save_param_history"]:
            trackers.append(tr.PARAM)
        # `DEFAULT_TRACKERS` carries the 1-norm of the gradient, which is not the
        # quantity `GradientDescent.step` compares to `clip`; the batch is a local
        # of `problem.solve` unless it is tracked.  Both are needed by the
        # iteration tables (spec section 3.4).
        trackers += [tr.BATCH, tr.GRAD_NORM_L2]
        return algorithm, trackers

    def _solve_kwargs(self) -> dict:
        opt = self.options["optimizer"]
        return {
            "num_iterations": int(opt["num_iterations"]),
            "batch_size": int(opt["batch_size"] or 0),
            "batch_strategy": opt["batch_strategy"],
            "verbosity": 0,
            "init_full_loss": bool(opt["init_full_loss"]),
            "peak_net_load_k": opt["peak_net_load_k"],
            "peak_net_load_rerank_every": int(opt["peak_net_load_rerank_every"]),
        }

    # -- solve --------------------------------------------------------------

    def solve(self, ctx: base.PlanningContext) -> base.PlanningResult:
        opts = self.options
        mode = opts["emissions"]["mode"]

        warm = self.warm_start(ctx)

        num_workers = int(opts["num_workers"])
        if num_workers > 1:
            ctx.problem.initialize_workers(num_workers)

        algorithm, trackers = self._algorithm()
        dual = None
        # The carbon price in force at the parameters this method returns.
        price = float(objectives.emissions_price(self.cfg))
        t0 = time.perf_counter()
        try:
            if mode == "dual_ascent":
                params, history, dual = self._dual_ascent(ctx, algorithm, trackers, warm)
                price = float(dual["applied_lambda"])
            else:
                params, history = ctx.problem.solve(
                    algorithm=algorithm,
                    trackers=trackers,
                    initial_state=warm.parameters,
                    lower_bound=warm.solve_bound,
                    **self._solve_kwargs(),
                )
            solve_seconds = time.perf_counter() - t0

            # One full forward pass at the final parameters.  `history["loss"]`
            # is the loss of the last *minibatch*, which is not the objective of
            # the design whenever `batch_size` is set, so capex, opex, emissions
            # and the objective all come from here instead.
            emissions = objectives.evaluate_emissions(
                ctx.problem, objectives.emissions_objectives(ctx.problem), params
            )
            capex = float(ctx.problem.get_inv_cost())
            opex = float(ctx.problem.get_op_cost())
        finally:
            if num_workers > 1:
                ctx.problem.shutdown_workers()

        # `price` / `dual_ascent` add `price * emissions` to every subproblem's
        # operation objective; the reported objective is system cost, so the
        # carbon payment is netted out and recorded separately (PROJECT.md 2.4).
        carbon_payment = price * float(emissions)
        opex -= carbon_payment
        objective_raw = capex + opex

        emissions_block = dict(ctx.meta.get("emissions", {}))
        if dual is not None:
            emissions_block["dual_ascent"] = dual

        return base.PlanningResult.from_context(
            ctx,
            method=self.name,
            preset=self.preset,
            kind=None,
            parameters=params,
            objective={
                "raw": objective_raw,
                "capex_raw": capex,
                "opex_raw": opex,
                "carbon_payment_raw": carbon_payment,
                "lower_bound_raw": warm.lower_bound_raw,
                "optimality_gap": optimality_gap(objective_raw, warm.lower_bound_raw),
                "emissions_tonnes_raw": emissions,
                # The warm-start LP optimum, in the units of *its* block set.
                "warm_start_objective_raw": warm.objective_raw,
                "warm_start_sampled_hours": warm.sampled_hours,
            },
            emissions=emissions_block,
            solver={
                "name": opts["dispatch_solver"],
                "status": "converged",
                "kwargs": dict(opts["dispatch_solver_kwargs"] or {}),
                "n_subproblems": len(ctx.problem.subproblems),
                "num_iterations": int(opts["optimizer"]["num_iterations"]),
            },
            timing={
                "build_s": ctx.build_seconds,
                "warm_start_s": warm.seconds,
                "solve_s": solve_seconds,
            },
            compute={"num_workers": num_workers},
            history=history,
        )

    # -- the dual-ascent outer loop -----------------------------------------

    def _dual_ascent(self, ctx, algorithm, trackers, warm: WarmStart):
        """Subgradient ascent on the carbon price until emissions hit the target.

        Ported from ``runner.py:1144-1268``.  ``validate_emissions`` has already
        guaranteed a target is set for this mode.
        """
        da = self.options["emissions"]["dual_ascent"]
        target = constraints.emissions_target(self.cfg, ctx.annualization_factor)
        dual_step_size = float(da["dual_step_size"])
        max_weight = float(da["max_weight"])
        num_outer = int(da["num_outer_iterations"])
        tolerance = float(da["tolerance"])
        current_lambda = float(da["initial_weight"])
        applied_lambda = float(current_lambda)

        emissions_objs = objectives.emissions_objectives(ctx.problem)

        params = warm.parameters
        history: dict = {}
        lambda_history: list[float] = []
        emissions_history: list[float] = []
        outer_histories: list[dict] = []
        converged = False

        from ..design import serialize_history

        for outer in range(num_outer):
            objectives.update_operation_objectives(
                ctx.problem, ctx.sampler.base_network, current_lambda
            )
            # The LP bound is only valid at the initial multiplier.
            bound = warm.solve_bound if outer == 0 else math.nan
            applied_lambda = float(current_lambda)

            params, history = ctx.problem.solve(
                algorithm=algorithm,
                trackers=trackers,
                initial_state=params,
                lower_bound=bound,
                **self._solve_kwargs(),
            )
            outer_histories.append(serialize_history(history))

            total = objectives.evaluate_emissions(ctx.problem, emissions_objs, params)
            lambda_history.append(float(current_lambda))
            emissions_history.append(float(total))

            gap = (total - target) / abs(target) if target else float("inf")
            if abs(gap) < tolerance:
                converged = True
                break

            current_lambda = max(0.0, current_lambda + dual_step_size * (total - target))
            current_lambda = min(current_lambda, max_weight)

        dual = {
            **dict(da),
            "target_horizon": target,
            "cap_basis": self.options["emissions"]["cap_basis"],
            "final_emissions": emissions_history[-1] if emissions_history else None,
            "final_lambda": float(current_lambda),
            # The multiplier the returned design was actually solved at, which
            # is the last one *applied*, not the last one computed.
            "applied_lambda": float(applied_lambda),
            "lambda_history": lambda_history,
            "emissions_history": emissions_history,
            "num_outer_iterations_completed": len(emissions_history),
            "converged": bool(converged),
            "outer_histories": outer_histories,
        }
        return params, history, dual
