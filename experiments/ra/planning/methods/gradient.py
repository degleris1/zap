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

import logging
import math
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

import zap.planning.trackers as tr
from zap.planning import (
    AdagradDescent,
    AdamDescent,
    CapexScaledDescent,
    GradientDescent,
    MonolithicPlanningProblem,
    TrustRegionDescent,
)
from zap.planning.problem_abstract import _plateau

from .. import base, constraints, objectives
from .single_level import optimality_gap

logger = logging.getLogger(__name__)

__all__ = ["GradientMethod", "WarmStart", "select_iterate"]

#: ``design_selection`` rules that read an iterate back out of the history.
HISTORY_RULES = {"best_sampled": "loss", "best_rolling": "rolling_loss"}


def capex_scale(problem, parameter_names: dict) -> dict[str, np.ndarray]:
    """``gamma``: the coefficient of each parameter row in the investment objective.

    ``InvestmentObjective`` is ``sum_j capital_cost_j * (eta_j - p_nom_j)`` and
    ``AbstractDevice.sample_time`` pro-rates ``capital_cost`` by
    ``block_hours / total_hours``, so the coefficient the *problem* charges is
    ``sum_i w_i * capital_cost_i``, summed over subproblems with their weights.
    That is the same scaling the gradient carries (``_get_batch_weights``
    rescales a minibatch back to the whole block set), so ``|dF/deta_j| /
    gamma_j`` is dimensionless and reads as "this row's marginal value as a
    fraction of its own annualised capex".

    Rows with no capital cost get ``0.0`` and are excluded from the
    stationarity test by :func:`zap.planning.problem_abstract.stationarity_max`.
    """
    from zap.planning.problem_abstract import weighted_subproblems

    subproblems, weights = weighted_subproblems(problem)
    scale: dict[str, np.ndarray] = {}
    for param, (device_index, _attr) in parameter_names.items():
        total = None
        for sub, weight in zip(subproblems, weights):
            device = sub.layer.devices[device_index]
            capital = getattr(device, "capital_cost", None)
            if capital is None:
                continue
            values = float(weight) * np.asarray(capital, dtype=float).reshape(-1)
            total = values if total is None else total + values
        if total is not None:
            scale[param] = total
    return scale


def _last_finite(values) -> float | None:
    """The last finite entry of a history series, or ``None``."""
    for value in reversed(list(values or [])):
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def _finite_argmin(values) -> int | None:
    """Index of the smallest finite entry, or ``None`` if there is none."""
    arr = np.asarray([float("nan") if v is None else float(v) for v in values], dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    return int(np.nanargmin(np.where(np.isfinite(arr), arr, np.inf)))


def select_iterate(history: dict, rule: str) -> tuple[dict | None, int | None, str]:
    """``(parameters, iteration index, rule actually applied)`` for a solve history.

    ``None`` parameters mean "keep what ``solve`` returned" (the final iterate).
    ``history["param"][i]`` and ``history["loss"][i]`` are the *same* state --
    the loop steps, evaluates, then records once -- so the argmin index needs no
    offset.  A rule that cannot be applied (no parameter history, no finite
    loss) degrades to ``final`` and says so in the third element, rather than
    guessing.
    """
    if rule == "final":
        return None, None, "final"
    if rule == "best_checkpointed":
        # Checkpoints are evaluated during the solve, not read back from the
        # history; `GradientMethod` handles this rule itself.
        return None, None, "best_checkpointed"
    key = HISTORY_RULES.get(rule)
    if key is None:
        raise ValueError(f"unknown design_selection rule {rule!r}")

    params = list((history or {}).get("param") or [])
    if not params:
        logger.warning(
            "design_selection %r needs the parameter history "
            "(planning.optimizer.save_param_history); falling back to the final iterate.",
            rule,
        )
        return None, None, "final"

    index = _finite_argmin(list((history or {}).get(key) or []))
    if index is None or index >= len(params):
        logger.warning(
            "design_selection %r found no usable %s history; "
            "falling back to the final iterate.",
            rule,
            key,
        )
        return None, None, "final"
    return deepcopy(params[index]), index, rule


class _Checkpointer:
    """Full-horizon forward passes at selected iterates, taken *during* a solve.

    With a minibatch, ``history["loss"]`` is a 4-block estimate and its argmin
    picks the luckiest batch, not the best design.  This hook evaluates the
    objective over the **whole** block set every ``every`` iterations (plus the
    first and the last iterate, which are always candidates: the first is the
    warm start, and a run that never improves on it must be able to say so), and
    keeps the parameters of each such iterate.

    One checkpoint costs one full forward pass -- a deterministic iteration
    without its backward pass -- so ``every = 20`` over 200 iterations adds
    roughly 10 iterations' worth of forward work to a minibatch run.

    ``outer`` distinguishes the outer iterations of dual ascent; only the last
    outer loop's checkpoints are candidates, because only its parameters are
    returned.
    """

    def __init__(
        self,
        problem,
        emissions_objs,
        *,
        every: int,
        price: float,
        tol_rel_objective: float | None = None,
        tol_window: int = 20,
    ):
        self.problem = problem
        self.emissions_objs = emissions_objs
        self.every = int(every or 0)
        self.price = float(price)
        self.outer = 0
        self.records: list[dict] = []
        self.parameters: dict[tuple[int, int], dict] = {}
        self.seconds = 0.0
        self._seen: set[tuple[int, int]] = set()
        # The objective-plateau test of a *stochastic* cell runs here rather
        # than in `solve`: the per-iteration loss is a 4-block estimate with
        # sigma ~ 1 B$ against a true decrease of ~0.3 M$, so only the
        # checkpoints are a usable series (spec section 3).
        self.tol_rel_objective = (
            None if tol_rel_objective is None else float(tol_rel_objective)
        )
        self.tol_window = int(tol_window or 0)

    def __call__(self, index: int, state: dict, history: dict, final: bool) -> bool:
        """Take a checkpoint if one is due; return True to stop the descent loop."""
        key = (self.outer, int(index))
        if key in self._seen:
            return False
        if not (final or index == 0 or (self.every > 0 and int(index) % self.every == 0)):
            return False
        self._seen.add(key)

        t0 = time.perf_counter()
        emissions = objectives.evaluate_emissions(self.problem, self.emissions_objs, state)
        capex = float(self.problem.get_inv_cost())
        opex = float(self.problem.get_op_cost())
        seconds = time.perf_counter() - t0
        self.seconds += seconds

        carbon_payment = self.price * float(emissions)
        opex -= carbon_payment
        self.records.append(
            {
                "outer_iteration": int(self.outer),
                "iteration": int(index),
                "objective_raw": capex + opex,
                "capex_raw": capex,
                "opex_raw": opex,
                "emissions_tonnes_raw": float(emissions),
                "final": bool(final),
                "seconds": seconds,
            }
        )
        self.parameters[key] = deepcopy(state)

        if final or self.tol_rel_objective is None:
            return False
        series = [
            r["objective_raw"] for r in self.records if r["outer_iteration"] == self.outer
        ]
        return bool(_plateau(series, self.tol_window, self.tol_rel_objective))

    def best(self) -> tuple[dict | None, int | None, float | None]:
        """``(parameters, iteration, objective)`` of the best checkpoint."""
        records = [r for r in self.records if r["outer_iteration"] == self.outer]
        if not records:
            return None, None, None
        index = _finite_argmin([r["objective_raw"] for r in records])
        if index is None:
            return None, None, None
        record = records[index]
        key = (record["outer_iteration"], record["iteration"])
        return (
            deepcopy(self.parameters[key]),
            int(record["iteration"]),
            float(record["objective_raw"]),
        )


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

    def _algorithm(self, ctx=None):
        """``(step rule, trackers)`` for one descent loop.

        ``rule: gradient`` constructs exactly the object the pre-2026-09-10 code
        constructed, with exactly the same two numbers, so the trajectory is
        reproduced bit-for-bit; every other rule is opt-in through
        ``planning.optimizer.rule``.
        """
        opt = self.options["optimizer"]
        rule = str(opt["rule"])
        adam = dict(opt["adam"])
        decay = str(opt["lr_decay"])
        decay_final = float(opt["lr_decay_final_frac"])
        max_step = adam["max_step_mw"]
        max_step = None if max_step is None else float(max_step)

        if rule == "gradient":
            algorithm = GradientDescent(
                step_size=float(opt["step_size"]), clip=float(opt["clip"])
            )
        elif rule == "adam":
            algorithm = AdamDescent(
                step_size=float(opt["step_size"]),
                beta1=float(adam["beta1"]),
                beta2=float(adam["beta2"]),
                eps=float(adam["eps"]),
                max_step_mw=max_step,
                decay=decay,
                decay_final_frac=decay_final,
            )
        elif rule == "adagrad":
            algorithm = AdagradDescent(
                step_size=float(opt["step_size"]),
                eps=float(adam["eps"]),
                max_step_mw=max_step,
                decay=decay,
                decay_final_frac=decay_final,
            )
        elif rule == "capex_scaled":
            algorithm = CapexScaledDescent(
                step_size=float(opt["step_size"]),
                capex=self._capex_scale(ctx),
                floor=float(opt["capex_scaled"]["floor"]),
                max_step_mw=max_step,
                decay=decay,
                decay_final_frac=decay_final,
            )
        elif rule == "trust_region":
            region = dict(opt["trust_region"])
            algorithm = TrustRegionDescent(
                initial_radius_mw=float(region["initial_radius_mw"]),
                max_radius_mw=float(region["max_radius_mw"]),
                min_radius_mw=float(region["min_radius_mw"]),
                eta_low=float(region["eta_low"]),
                eta_high=float(region["eta_high"]),
                expand=float(region["expand"]),
                shrink=float(region["shrink"]),
            )
        else:  # pragma: no cover - `config.validate` rejects this first
            raise base.ConfigError(f"unknown planning.optimizer.rule {rule!r}")

        trackers = list(tr.DEFAULT_TRACKERS)
        if opt["save_param_history"]:
            trackers.append(tr.PARAM)
        # `DEFAULT_TRACKERS` carries the 1-norm of the gradient, which is not the
        # quantity `GradientDescent.step` compares to `clip`; the batch is a local
        # of `problem.solve` unless it is tracked.  Both are needed by the
        # iteration tables (spec section 3.4).  The active-set and realised-step
        # trackers are what make the step columns honest: `step_norm_mw` used to
        # be the pre-projection norm and measured nothing.
        trackers += [
            tr.BATCH,
            tr.GRAD_NORM_L2,
            tr.FREE_GRAD_NORM_L2,
            tr.N_FREE,
            tr.N_AT_LOWER,
            tr.N_AT_UPPER,
            tr.STEP_NORM_MW_ACTUAL,
            tr.STEP_NORM_FREE_MW,
            tr.OPT_DIAG,
        ]
        if int(opt["grad_history_every"] or 0) > 0:
            trackers += [tr.GRAD_SAMPLED, tr.OPT_MOMENTS]
        return algorithm, trackers

    def _capex_scale(self, ctx) -> dict:
        """``gamma`` per parameter row, or ``{}`` when there is no problem yet."""
        if ctx is None:
            return {}
        return capex_scale(ctx.problem, ctx.parameter_names)

    def _solve_kwargs(self, iteration_hook=None, *, stationarity_scale=None) -> dict:
        """The keyword arguments of one ``problem.solve`` call.

        ``max_seconds`` is the soft cap on **one** call, so under
        ``emissions.mode: dual_ascent`` it is a per-outer-iteration budget, not
        a budget for the whole ascent.
        """
        opt = self.options["optimizer"]
        stopping = dict(opt["stopping"])
        return {
            "num_iterations": int(opt["num_iterations"]),
            "batch_size": int(opt["batch_size"] or 0),
            "batch_strategy": opt["batch_strategy"],
            "verbosity": 0,
            "init_full_loss": bool(opt["init_full_loss"]),
            "peak_net_load_k": opt["peak_net_load_k"],
            "peak_net_load_rerank_every": int(opt["peak_net_load_rerank_every"]),
            "time_limit_s": opt["max_seconds"],
            # The minibatch RNG was a hardcoded 42; seeding it from the run's
            # own seed is what makes replicates distinguishable.
            "batch_seed": int(self.selection["seed"]),
            "iteration_hook": iteration_hook,
            # The objective-plateau test belongs to whichever series is the
            # full-horizon one: `history["loss"]` for a full-batch cell, the
            # checkpoints (through `iteration_hook`) for a minibatch cell.
            "tol_rel_objective": (
                stopping["tol_rel_objective"] if int(opt["batch_size"] or 0) <= 0 else None
            ),
            "tol_window": int(stopping["tol_window"]),
            "tol_stationarity": stopping["tol_stationarity"],
            "stationarity_scale": stationarity_scale,
            "grad_history_every": int(opt["grad_history_every"] or 0),
        }

    # -- solve --------------------------------------------------------------

    def solve(self, ctx: base.PlanningContext) -> base.PlanningResult:
        opts = self.options
        mode = opts["emissions"]["mode"]

        warm = self.warm_start(ctx)

        num_workers = int(opts["num_workers"])
        if num_workers > 1:
            ctx.problem.initialize_workers(num_workers)

        algorithm, trackers = self._algorithm(ctx)
        # gamma: the annualised capital cost of every parameter row, which is
        # what the stationarity test measures the (smoothed) gradient against.
        stationarity_scale = capex_scale(ctx.problem, ctx.parameter_names)
        dual = None
        # The carbon price in force at the parameters this method returns.
        price = float(objectives.emissions_price(self.cfg))
        opt = opts["optimizer"]
        rule = str(opt["design_selection"])
        emissions_objs = objectives.emissions_objectives(ctx.problem)
        stopping = dict(opt["stopping"])
        checkpointer = None
        if int(opt["checkpoint_every"] or 0) > 0 or rule == "best_checkpointed":
            checkpointer = _Checkpointer(
                ctx.problem,
                emissions_objs,
                every=int(opt["checkpoint_every"] or 0),
                price=price,
                # A minibatch cell's plateau test runs on the checkpoints; a
                # full-batch cell's runs inside `solve` on the sampled loss,
                # which is the same quantity, so it is not doubled up here.
                tol_rel_objective=(
                    stopping["tol_rel_objective"] if int(opt["batch_size"] or 0) > 0 else None
                ),
                # In CHECKPOINTS, not iterations: `checkpoint_every` iterations
                # separate two entries of this series.
                tol_window=int(stopping["checkpoint_window"]),
            )
        t0 = time.perf_counter()
        try:
            if mode == "dual_ascent":
                params, history, dual = self._dual_ascent(
                    ctx, algorithm, trackers, warm, checkpointer, stationarity_scale
                )
                price = float(dual["applied_lambda"])
            else:
                params, history = ctx.problem.solve(
                    algorithm=algorithm,
                    trackers=trackers,
                    initial_state=warm.parameters,
                    lower_bound=warm.solve_bound,
                    **self._solve_kwargs(
                        iteration_hook=checkpointer,
                        stationarity_scale=stationarity_scale,
                    ),
                )
            solve_seconds = time.perf_counter() - t0
            stop_reason = str(getattr(ctx.problem, "stop_reason", "num_iterations"))

            # Which iterate becomes the design.  The full forward pass below then
            # reports capex / opex / emissions / objective *at the design*, so
            # selecting an earlier iterate costs no extra dispatch solve.
            selected, design_iteration, applied_rule = select_iterate(history, rule)
            if applied_rule == "best_checkpointed":
                selected, design_iteration, _ = (
                    checkpointer.best() if checkpointer is not None else (None, None, None)
                )
                if selected is None:
                    logger.warning(
                        "design_selection 'best_checkpointed' found no usable checkpoint; "
                        "falling back to the final iterate."
                    )
                    applied_rule = "final"
            if selected is not None:
                params = selected

            # One full forward pass at the returned parameters.  `history["loss"]`
            # is the loss of the last *minibatch*, which is not the objective of
            # the design whenever `batch_size` is set, so capex, opex, emissions
            # and the objective all come from here instead.
            emissions = objectives.evaluate_emissions(ctx.problem, emissions_objs, params)
            capex = float(ctx.problem.get_inv_cost())
            opex = float(ctx.problem.get_op_cost())
            # `evaluate_emissions` just ran a full forward pass at `params`, so
            # every subproblem's `state` is the dispatch of the design: split the
            # opex into gross cost and negative-price credit off those states,
            # with no extra solve.
            try:
                credit = objectives.problem_cost_credit(
                    ctx.problem, classes=objectives.CREDIT_CLASSES
                )
                export_revenue = objectives.problem_cost_credit(
                    ctx.problem, classes=objectives.EXPORT_CLASSES
                )
            except Exception as exc:  # reporting must not fail a solve
                logger.warning("could not split the opex by cost sign: %s", exc, exc_info=True)
                credit = None
                export_revenue = None
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

        # All three candidate objectives, always, so the selection is auditable
        # from the design file alone.  `loss` / `rolling_loss` are in the units
        # of the whole block set (`_get_batch_weights` rescales a minibatch), so
        # they are comparable with `raw` -- but they are estimates of it whenever
        # the batch is a strict subset.
        losses = [float(x) for x in (history or {}).get("loss") or []]
        rolling = [float(x) for x in (history or {}).get("rolling_loss") or []]
        best_sampled = _finite_argmin(losses)
        best_rolling = _finite_argmin(rolling)
        af = float(ctx.annualization_factor)
        checkpoints = [
            {**record, "objective_annual": record["objective_raw"] * af}
            for record in (checkpointer.records if checkpointer is not None else [])
        ]
        num_completed = max(0, len(losses) - 1)

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
                **objectives.opex_split(opex, credit, export_revenue),
                "carbon_payment_raw": carbon_payment,
                "lower_bound_raw": warm.lower_bound_raw,
                "optimality_gap": optimality_gap(objective_raw, warm.lower_bound_raw),
                "emissions_tonnes_raw": emissions,
                # The warm-start LP optimum, in the units of *its* block set.
                "warm_start_objective_raw": warm.objective_raw,
                "warm_start_sampled_hours": warm.sampled_hours,
                "design_selection": applied_rule,
                "design_selection_requested": rule,
                "design_iteration": design_iteration,
                "final_sampled_objective_raw": losses[-1] if losses else None,
                "best_sampled_objective_raw": (
                    losses[best_sampled] if best_sampled is not None else None
                ),
                "best_sampled_iteration": best_sampled,
                "best_rolling_objective_raw": (
                    rolling[best_rolling] if best_rolling is not None else None
                ),
                "best_rolling_iteration": best_rolling,
                "checkpoints": checkpoints,
            },
            emissions=emissions_block,
            solver={
                "name": opts["dispatch_solver"],
                "status": "converged",
                "kwargs": dict(opts["dispatch_solver_kwargs"] or {}),
                "n_subproblems": len(ctx.problem.subproblems),
                "num_iterations": int(opt["num_iterations"]),
                "num_iterations_completed": num_completed,
                # `stop_reason` is zap's word for it; `stopped_by` is the
                # campaign vocabulary.  `objective_tolerance` / `stationarity` /
                # `checkpoint_tolerance` are the convergence stops of the
                # 2026-09-10 step-rule spec; the two backstops keep their old
                # names so existing tables and plots are unaffected.
                "stop_reason": stop_reason,
                "stopped_by": {
                    "wall_clock": "max_seconds",
                    "num_iterations": "iterations",
                }.get(stop_reason, stop_reason),
                "max_seconds": opt["max_seconds"],
                "checkpoint_every": int(opt["checkpoint_every"] or 0),
                "batch_seed": int(self.selection["seed"]),
                "rule": str(opt["rule"]),
                "step_size": float(opt["step_size"]),
                "stopping": stopping,
                "stationarity_max_final": _last_finite(
                    (history or {}).get("stationarity_max")
                ),
            },
            timing={
                "build_s": ctx.build_seconds,
                "warm_start_s": warm.seconds,
                "solve_s": solve_seconds,
                "checkpoint_s": (checkpointer.seconds if checkpointer is not None else 0.0),
            },
            compute={"num_workers": num_workers},
            history=history,
        )

    # -- the dual-ascent outer loop -----------------------------------------

    def _dual_ascent(
        self,
        ctx,
        algorithm,
        trackers,
        warm: WarmStart,
        checkpointer=None,
        stationarity_scale=None,
    ):
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
            if checkpointer is not None:
                # Only the last outer loop's checkpoints are candidates, and the
                # carbon price they net out is the one in force now.
                checkpointer.outer = outer
                checkpointer.price = float(current_lambda)

            params, history = ctx.problem.solve(
                algorithm=algorithm,
                trackers=trackers,
                initial_state=params,
                lower_bound=bound,
                **self._solve_kwargs(
                    iteration_hook=checkpointer, stationarity_scale=stationarity_scale
                ),
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
