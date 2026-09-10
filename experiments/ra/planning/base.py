"""Planning-method interface, config defaults and the shared ``build()`` body.

WP5 Task A.  This module owns:

* the ``planning`` / ``selection`` config defaults (spec section 8.1), so the core is
  usable before the harness (Task C) grows the keys;
* :class:`PlanningContext` and :class:`PlanningResult` (spec sections 3.2 and 3.4);
* :class:`PlanningMethod`, whose ``build()`` is shared by every method
  (spec section 3.3) and whose ``solve()`` the concrete methods (Task B) implement;
* the method registry, ``require_solver`` (D-W7) and ``plan()``.
"""

from __future__ import annotations

import abc
import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..config import ConfigError

# ===========================================================================
# Config defaults (spec section 8.1)
# ===========================================================================

#: Defaults for the ``planning:`` block.  Task C copies these into
#: ``configs/base.yaml``; keeping them here means the core validates and runs
#: against a partial config (and against a config written before Task C lands).
PLANNING_DEFAULTS: dict = {
    "method": "monolithic",
    "dispatch_solver": "CLARABEL",
    "dispatch_solver_kwargs": {"verbose": False},
    "regularize": 1.0e-6,
    "num_workers": 1,
    "timeout_s": 21600,
    "required": True,
    "bounds": {"min_capacity_mw": 0.1, "min_storage_mw": 10.0},
    "expansion": {"mode": "pypsa", "max_capacity_multiple": 10.0, "max_capacity_mw": None},
    "single_level": {
        "kind": "primal",
        "solver": "HIGHS",
        "solver_kwargs": {},
        "inf_value": 100.0,
        "price_bound": 100.0,
    },
    "warm_start": {
        "enabled": False,
        "num_blocks": None,
        "solver": "HIGHS",
        "solver_kwargs": {},
    },
    # ``GradientDescent.step`` clips the *total* gradient norm to ``clip`` and
    # then moves ``step_size * clip``, so ``step_size * clip`` is a hard cap on
    # the L2 capacity movement of one iteration, in MW.  The measured gradient
    # norm is ~1.3e4 on the 48-hour test fixture and ~1.8e7 on z4-2020, i.e.
    # always far above ``clip``, so the cap binds every iteration: 0.2 * 5e3 =
    # 1,000 MW per step, ~0.5 % of z4-2020's ~209 GW fleet and more than enough
    # to move the fixture's 700 MW one.  The old (1e-3, 1e3) pair moved at most
    # 1 MW per iteration and made the method a no-op (WP5 verification).
    "optimizer": {
        # A BACKSTOP, not a target: a run is expected to end on
        # `stopping.tol_rel_objective` / `tol_stationarity` or on `max_seconds`.
        # 500 because the two Adam campaign cells stopped on a tolerance at
        # iteration 29 (c4') and 140 (c5'), so 500 leaves headroom of several
        # times the observed need without being a number a run can hit by
        # accident.  It also sets the cosine decay horizon: `lr_decay: cosine`
        # decays over `num_iterations`, so raising this flattens the schedule.
        "num_iterations": 500,
        # Which step rule turns the gradient into a capacity move.  **Adam is
        # the default** (Kamran, 2026-09-10): the historical `gradient` rule
        # normalises by the gradient norm over *all* parameter rows, of which
        # 127 of 166 are structurally frozen on ca2040_z4 and carry 98.7 % of
        # that norm, so it reported a 1,000 MW step while moving the design
        # 13 MW -- and from a short system it moves 1e5-1e6 MW in one iteration,
        # because the step is proportional to the gradient in units nobody set.
        #   adam          -- eta <- eta - step_size * mhat / (sqrt(vhat) + eps)
        #   gradient      -- eta <- eta - step_size * clip * g / ||g||  (archive)
        #   adagrad       -- Adam with no first moment (the fallback)
        #   capex_scaled  -- eta <- eta - step_size * g / gamma (ablation A4)
        #   trust_region  -- steepest descent inside an adaptive radius (A3)
        "rule": "adam",
        # MW per coordinate per iteration for `adam` / `adagrad` /
        # `capex_scaled` (`trust_region` uses its own radius instead).  200 MW
        # clears the largest c4-to-LP gap (3.7 GW) in ~19 iterations.
        #
        # For `rule: gradient` ONLY, `step_size` is a MW^2/$ multiplier and
        # `clip` caps the gradient norm, so `step_size * clip` is the MW cap of
        # one iteration: that rule needs `step_size: 0.2` set explicitly, or it
        # inherits 200 here and asks for 1e6 MW per step (`config.validate`
        # refuses that).
        "step_size": 200.0,
        "clip": 5.0e3,
        "adam": {
            # Averages ~10 gradients, which is the window the measured minibatch
            # SNR (0.20-0.42 per row) needs to recover the deterministic sign.
            "beta1": 0.9,
            "beta2": 0.999,
            # $/MW-yr, NOT the ML default 1e-8: `eps` is in the units of the
            # gradient and sets the magnitude below which a row's gradient is
            # treated as numerical dust.  At 1e-8 such a row still takes a full
            # `step_size` step.
            "eps": 1.0,
            # A hard per-row cap in MW (null -> 3 * step_size, which is what
            # 600 states explicitly): this is what makes the first iteration
            # un-spikeable from any starting point, with no invented capacity
            # bound.
            "max_step_mw": 600.0,
        },
        # Learning-rate schedule for the per-coordinate rules (MW).  Cosine to
        # 5 % of `step_size` over `num_iterations`: the tail is what converges a
        # run whose backstop is wall clock rather than iterations.
        "lr_decay": "cosine",  # none | cosine | inverse_sqrt
        "lr_decay_final_frac": 0.05,
        "capex_scaled": {"floor": 1.0},
        "trust_region": {
            "initial_radius_mw": 50.0,
            "max_radius_mw": 5.0e3,
            "min_radius_mw": 1.0e-3,
            "eta_low": 0.1,
            "eta_high": 0.9,
            "expand": 2.0,
            "shrink": 0.5,
        },
        # Convergence tests.  `num_iterations` is a backstop, not a target:
        # a run is expected to end on a tolerance or on `max_seconds`.  Set a
        # tolerance to null to switch that test off.
        "stopping": {
            # Relative decrease of the objective over a `tol_window` window,
            # below which the loop stops.  Deterministic cells measure it on
            # the sampled loss (which IS the full loss); a minibatch cell
            # measures it on the checkpoint series instead, so it needs
            # `checkpoint_every > 0`.  1e-4 against c4's measured 20-iteration
            # decrease of ~9.4e-4, i.e. it does not fire on a run still moving.
            "tol_rel_objective": 1.0e-4,
            # Window of the plateau test, in ITERATIONS, for a full-batch cell
            # whose per-iteration loss is the full-horizon objective.  The test
            # does not fire before the window is full.
            "tol_window": 20,
            # Window of the same test in CHECKPOINTS for a minibatch cell, whose
            # only full-horizon series is the checkpoints.  5 checkpoints at
            # `checkpoint_every: 20` is 100 iterations; re-using `tol_window`
            # here would be 20 checkpoints = 400 iterations.
            "checkpoint_window": 5,
            # max_j |mhat_j| / gamma_j over interior rows, gamma = annualised
            # capex: "every interior row's marginal value is within 2 % of its
            # own break-even".  Economically interpretable and noise-robust (it
            # reads the EMA, not the raw minibatch gradient).
            "tol_stationarity": 0.02,
        },
        # 0 = off, and off is the default: this is an output-volume knob, not
        # part of the step rule, and a 6,000-iteration run would put ~100 MB of
        # per-row floats in `designs/<id>.history.json`.  The campaign cells set
        # 10.
        "grad_history_every": 0,
        "batch_size": 0,
        "batch_strategy": "sequential",
        "init_full_loss": True,
        "save_param_history": True,
        # Soft wall-clock cap on the *descent loop*, in seconds (null = none).
        # The loop finishes the iteration it is in and breaks, so the design and
        # the history survive.  It does not cover the build, the warm-start LP or
        # the final forward pass, so a task's wall clock exceeds it;
        # `planning.timeout_s` is the *hard* backstop over the whole task, which
        # kills the subprocess and keeps nothing, and must be larger.
        "max_seconds": None,
        # Which iterate becomes the design:
        #   final             -- the last one (the pre-2026-09-09 behaviour);
        #   best_sampled      -- argmin of the sampled loss.  Exact only when
        #                        the batch is the whole block set;
        #   best_rolling      -- argmin of the running mean of the last
        #                        `num_subproblems` losses;
        #   best_checkpointed -- argmin over full-horizon forward passes taken
        #                        every `checkpoint_every` iterations (plus the
        #                        first and last), which is the unbiased rule for
        #                        a minibatch run.  Costs one full forward pass
        #                        per checkpoint.
        # `best_checkpointed` is the default because it is the only rule that is
        # correct for a minibatch as well as for a full batch, so deterministic
        # and stochastic cells are not confounded by their selection rule.
        "design_selection": "best_checkpointed",
        # 0 = no checkpoints.  Required (> 0) by `best_checkpointed`.  At 20 a
        # 200-iteration run pays ~10 extra forward passes, ~5 % of its budget.
        "checkpoint_every": 20,
        # DEPRECATED and ignored: ``objective.raw`` is always the full forward
        # pass (inv + op), never the last minibatch loss.  The key is kept so
        # the config key space is unchanged; setting it true is a ConfigError.
        "eval_final_full_loss": False,
        "peak_net_load_k": None,
        "peak_net_load_rerank_every": 1,
    },
    "admm": {
        "machine": "cpu",
        "dtype": "float32",
        "adapt_rho": False,
        "adapt_rho_rate": 0.1,
        # Carry each block's ADMM state between planner forward passes
        # (``ADMMLayer.warm_start``).  Not to be confused with the top-level
        # ``planning.warm_start``, which is the single-level LP warm start.
        "warm_start": True,
        "warm_start_reset_every": None,
        "solver_kwargs": {"num_iterations": 1000, "rho_power": 0.1, "minimum_iterations": 100},
    },
    "emissions": {
        "mode": "none",
        "price": 0.0,
        "cap": None,
        "cap_basis": "annual",
        "dual_ascent": {
            "target": None,
            "initial_weight": 0.0,
            "dual_step_size": 1.0,
            "max_weight": 1000.0,
            "num_outer_iterations": 10,
            "tolerance": 0.05,
        },
    },
    "budget_constraints": None,
}

#: Defaults for the ``selection:`` keys that ``mode: plan`` reads (R-W9: the
#: dispatch-mode keys of the same block are ignored here).
SELECTION_DEFAULTS: dict = {
    "strategy": "all",
    "block_size": 168,
    "num_blocks": None,
    "avoid_year_boundaries": False,
    "seed": 42,
    # `random` / `stratified` only: draw block starts from the `block_size` grid
    # (0, B, 2B, ...) instead of from every hour, so a subset selection is a
    # subset of `strategy: all` and "12 of the 52 weeks" is literally true.
    "align_blocks": False,
}

#: ``planning.method`` preset -> the class that implements it (D-W2).
METHOD_PRESETS: dict[str, str] = {
    "monolithic": "single_level",
    "stochastic": "single_level",
    "relaxed": "single_level",
    "gradient": "gradient",
    "admm": "admm",
}


def _merge_defaults(defaults: dict, override: Any) -> dict:
    """Deep-merge ``override`` onto ``defaults`` without mutating either."""
    out = copy.deepcopy(defaults)
    if not override:
        return out
    if not isinstance(override, dict):
        raise ConfigError(f"expected a mapping, got {type(override).__name__}")
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_defaults(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def planning_options(cfg: dict) -> dict:
    """The ``planning`` block, with every default filled in."""
    return _merge_defaults(PLANNING_DEFAULTS, (cfg or {}).get("planning"))


def selection_options(cfg: dict) -> dict:
    """The ``mode: plan`` half of the ``selection`` block, defaults filled in."""
    sel = (cfg or {}).get("selection") or {}
    sel = {k: v for k, v in sel.items() if k in SELECTION_DEFAULTS}
    return _merge_defaults(SELECTION_DEFAULTS, sel)


def require_solver(name: str) -> str:
    """Validate a cvxpy solver name (D-W7).

    Both ``DispatchLayer.__init__`` and ``PowerNetwork.dispatch`` default to
    ``cp.ECOS``, which is not installed here, so a forgotten ``solver=`` fails
    deep inside cvxpy with an unrelated message.  Every construction site in
    WP5 goes through this function.
    """
    import cvxpy as cp

    if name is None:
        raise ConfigError(
            "no solver given; cvxpy reports installed_solvers() == "
            f"{sorted(cp.installed_solvers())}"
        )
    key = str(name).upper()
    installed = cp.installed_solvers()
    if key not in installed:
        raise ConfigError(
            f"unknown or uninstalled cvxpy solver {name!r}; "
            f"cvxpy reports installed_solvers() == {sorted(installed)}"
        )
    return key


def solver_object(name: str):
    """The cvxpy solver constant for a validated solver name."""
    import cvxpy as cp

    return getattr(cp, require_solver(name))


# ===========================================================================
# Context and result
# ===========================================================================


@dataclass
class PlanningContext:
    """Everything ``build()`` produced; ``solve()`` consumes it and nothing else."""

    system: Any  # LoadedSystem
    sampler: Any  # SystemBlockSampler
    blocks: list[tuple[int, int]]  # loaded-window index space (D-W9)
    problem: Any  # StochasticPlanningProblem
    parameter_names: dict[str, tuple[int, str]]
    lower_bounds: dict[str, np.ndarray]
    upper_bounds: dict[str, np.ndarray]
    initial_parameters: dict[str, np.ndarray]
    carrier_labels: dict[str, list[str]]
    total_hours: int
    sampled_hours: int
    annualization_factor: float
    build_seconds: float
    meta: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float(self.sampled_hours) / float(self.total_hours)


def annualization_factor(total_hours: int, sampled_hours: int) -> float:
    """``total_hours / sampled_hours`` (spec section 6, decision W-A)."""
    if sampled_hours <= 0:
        raise ValueError("sampled_hours must be positive")
    return float(total_hours) / float(sampled_hours)


def _to_list(value) -> Any:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr.reshape(-1).tolist()


@dataclass
class PlanningResult:
    """The serialisable design object; schema version 1 (spec section 3.4)."""

    SCHEMA_VERSION: ClassVar[int] = 1

    # Every field has a default so the dataclass is constructible purely by
    # keyword, from a partially-filled record (this is how the concrete methods
    # in ``methods/`` build it).
    design_id: str = "design"
    method: str = "unknown"
    preset: str = "unknown"
    parameters: dict[str, np.ndarray] = field(default_factory=dict)
    parameter_names: dict[str, tuple[int, str]] = field(default_factory=dict)
    lower_bounds: dict[str, np.ndarray] = field(default_factory=dict)
    upper_bounds: dict[str, np.ndarray] = field(default_factory=dict)
    capacities: dict[str, dict] = field(default_factory=dict)
    kind: str | None = None
    run_id: str | None = None
    dataset: str | None = None
    years: list = field(default_factory=list)
    window: dict = field(default_factory=dict)
    heuristics: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)
    annualization: dict = field(default_factory=dict)
    objective: dict = field(default_factory=dict)
    emissions: dict = field(default_factory=dict)
    solver: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    compute: dict = field(default_factory=dict)
    history: dict | None = None
    history_path: str | None = None
    meta: dict = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def objective_raw(self) -> float | None:
        v = self.objective.get("raw")
        return None if v is None else float(v)

    @property
    def objective_annual(self) -> float | None:
        v = self.objective.get("annual")
        return None if v is None else float(v)

    def to_record(self) -> dict:
        from .design import result_to_record

        return result_to_record(self)

    def to_design(self):
        from .design import result_to_design

        return result_to_design(self)

    def write(self, run_dir):
        from .design import write_result

        return write_result(self, run_dir)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_context(
        cls,
        ctx: PlanningContext,
        *,
        method: str,
        preset: str,
        parameters: dict,
        design_id: str = "design",
        kind: str | None = None,
        objective: dict | None = None,
        solver: dict | None = None,
        timing: dict | None = None,
        compute: dict | None = None,
        emissions: dict | None = None,
        history: dict | None = None,
        run_id: str | None = None,
    ) -> PlanningResult:
        """Fill the context-derived blocks of a result; ``solve()`` supplies the rest.

        Provided so that every concrete method records the same annualization,
        selection and capacity blocks (spec sections 3.4 and 6).
        """
        obj = dict(objective or {})
        af = float(ctx.annualization_factor)
        annual_keys = {
            "raw": "annual",
            "capex_raw": "capex_annual",
            "opex_raw": "opex_annual",
            # The two halves of `opex_raw`: gross dispatch cost and the credit
            # earned by negative-marginal-cost rows (<= 0).  They annualize the
            # same way, so the identity survives the scaling.
            "opex_gross_raw": "opex_gross_annual",
            "opex_credit_raw": "opex_credit_annual",
            # A component *of* `opex_gross_raw`, not a third term of the
            # identity: the negative-price part contributed by export links.
            "opex_export_revenue_raw": "opex_export_revenue_annual",
            "emissions_tonnes_raw": "emissions_tonnes_annual",
            "carbon_payment_raw": "carbon_payment_annual",
        }
        for key, annual_key in annual_keys.items():
            if obj.get(key) is not None and obj.get(annual_key) is None:
                obj[annual_key] = float(obj[key]) * af

        meta = dict(ctx.meta)

        # Say, on the design itself, what the objective is an objective *of*.
        # A sampled block set makes `annual` an in-sample extrapolation of the
        # sampled blocks, not the objective of the design over the full window
        # (a 12-random-week LP on z4-2020 reported 1,072 M$ of annual opex where
        # the full-window evaluation of the same design was 767 M$).  Nothing
        # downstream can tell the two apart from the number alone, so the design
        # carries the flag.
        selection_meta = dict(meta.get("selection", {}) or {})
        block_size = selection_meta.get("block_size")
        obj["basis"] = "full_window" if ctx.coverage >= 1.0 else "sampled"
        obj["in_sample"] = bool(ctx.coverage < 1.0)
        obj["operational_model"] = {
            "block_size": int(block_size) if block_size is not None else int(ctx.total_hours),
            "n_blocks": len(ctx.blocks),
            "storage_soc_mode": meta.get("storage_soc_mode"),
        }
        return cls(
            design_id=design_id,
            method=method,
            preset=preset,
            kind=kind,
            parameters={k: np.asarray(v) for k, v in parameters.items()},
            parameter_names=dict(ctx.parameter_names),
            lower_bounds=dict(ctx.lower_bounds),
            upper_bounds=dict(ctx.upper_bounds),
            capacities=device_capacities(ctx, parameters),
            run_id=run_id,
            dataset=meta.get("dataset"),
            years=list(meta.get("years", [])),
            window={
                "start": meta.get("window_start"),
                "stop": meta.get("window_stop"),
            },
            heuristics=dict(meta.get("heuristics", {})),
            selection=dict(meta.get("selection", {})),
            annualization={
                "total_hours": int(ctx.total_hours),
                "sampled_hours": int(ctx.sampled_hours),
                "coverage": ctx.coverage,
                "annualization_factor": af,
                "year_factor": meta.get("year_factor"),
                "capital_cost_prorated": True,
                "snapshot_weight": 1.0,
            },
            objective=obj,
            emissions=dict(emissions or meta.get("emissions", {})),
            solver=dict(solver or {}),
            timing=dict(timing or {"build_s": ctx.build_seconds}),
            compute=dict(compute or {}),
            history=history,
            meta=meta,
        )


def device_capacities(ctx: PlanningContext, parameters: dict) -> dict[str, dict]:
    """``{device class: {"names": [...], attr: [...]}}`` for ``design.json``.

    Keyed by device *class* name so it feeds ``experiments.ra.system.Design``
    unchanged (spec section 3.4).
    """
    devices = ctx.sampler.base_devices
    index = getattr(ctx.system, "index", None)
    out: dict[str, dict] = {}
    for param, (device_idx, attr) in ctx.parameter_names.items():
        if param not in parameters:
            continue
        cls_name = type(devices[device_idx]).__name__
        names = None
        if index is not None and cls_name in getattr(index, "names", {}):
            names = [str(n) for n in index.names[cls_name]]
        elif getattr(devices[device_idx], "name", None) is not None:
            names = [str(n) for n in devices[device_idx].name]
        entry = out.setdefault(cls_name, {})
        if names is not None:
            entry["names"] = names
        entry[attr] = _to_list(parameters[param])
    return out


# ===========================================================================
# The method interface
# ===========================================================================

#: Concrete methods register themselves here (Task B).
METHODS: dict[str, type] = {}


def register_method(cls):
    """Class decorator: add a concrete method to :data:`METHODS`."""
    METHODS[cls.name] = cls
    return cls


def _load_methods() -> str | None:
    """Import the concrete methods so they register. Returns the failure, if any."""
    try:
        from . import methods  # noqa: F401
    except ImportError as exc:
        return repr(exc)
    return None


class PlanningMethod(abc.ABC):
    """Base class: ``build()`` is shared, ``solve()`` is per method."""

    name: ClassVar[str] = "abstract"

    def __init__(self, cfg: dict) -> None:
        from . import constraints

        self.cfg = cfg or {}
        self.options = planning_options(self.cfg)
        self.selection = selection_options(self.cfg)
        self.preset = self.options["method"]
        constraints.validate_emissions(self.cfg)

    # -- hooks ------------------------------------------------------------
    def layer_kwargs(self) -> dict:
        """Extra kwargs for ``create_stochastic_problem`` (solver / layer factory)."""
        return {
            "solver": solver_object(self.options["dispatch_solver"]),
            "solver_kwargs": dict(self.options["dispatch_solver_kwargs"] or {}),
        }

    def prepare_devices(self, devices: list) -> list:
        """Identity by default; ``AdmmGradientMethod`` torchifies here."""
        return devices

    # -- shared build -----------------------------------------------------
    def build(self, system) -> PlanningContext:
        from . import constraints, expansion, objectives, parameters
        from . import selection as selection_mod
        from .sampler import SystemBlockSampler

        t0 = time.perf_counter()
        system = expansion.apply_expansion(system, self.cfg)
        sampler = SystemBlockSampler(system)
        selector = selection_mod.make_selector(self.cfg, total_hours=sampler.total_hours)
        blocks = selector.select(sampler)
        if not blocks:
            raise ValueError("period selection returned no blocks")

        weights = selector.weights(blocks)
        if weights is not None:
            raise ConfigError(
                "non-uniform subproblem weights are not supported in phase 1 "
                "(D-W8; StochasticPlanningProblem drops its zero-weight filter, "
                "see the WP5 spec section 9.3)"
            )

        parameter_names = parameters.setup_parameter_names(sampler.base_devices)
        bounds_cfg = self.options["bounds"]
        lower, upper = parameters.setup_bounds(
            sampler.base_devices,
            parameter_names,
            min_capacity_mw=float(bounds_cfg["min_capacity_mw"]),
            min_storage_mw=float(bounds_cfg["min_storage_mw"]),
        )
        initial = parameters.initial_parameters(sampler.base_devices, parameter_names)
        labels = parameters.carrier_labels(sampler.base_devices, parameter_names)

        op_fn = objectives.operation_objective_factory(sampler.base_network, self.cfg)
        inv_fn = objectives.investment_objective_factory()
        budget = constraints.load_budget_constraints(
            self.cfg, parameter_names, sampler.base_devices
        )

        problem = sampler.create_stochastic_problem(
            blocks=blocks,
            parameter_names=parameter_names,
            operation_objective_fn=op_fn,
            investment_objective_fn=inv_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            budget_constraints=budget,
            weights=None,
            device_hook=self.prepare_devices,
            **self.layer_kwargs(),
        )

        sampled_hours = int(sum(stop - start for start, stop in blocks))
        af = annualization_factor(sampler.total_hours, sampled_hours)

        meta = dict(getattr(system, "meta", {}) or {})
        meta["heuristics"] = dict(self.cfg.get("heuristics") or {})
        meta["selection"] = {
            **self.selection,
            "blocks": [[int(a), int(b)] for a, b in blocks],
            "weights": None,
        }
        meta["emissions"] = objectives.emissions_record(self.cfg, af)
        # `system.meta` already carries the mode the system was *built* with;
        # prefer the config when it states one, so `PlanningResult.from_context`
        # can report the operational model without re-reading the config.
        soc_mode = (self.cfg.get("system") or {}).get("storage_soc_mode")
        if soc_mode is not None:
            meta["storage_soc_mode"] = str(soc_mode)
        meta["bounds"] = {
            "min_capacity_mw": float(bounds_cfg["min_capacity_mw"]),
            "min_storage_mw": float(bounds_cfg["min_storage_mw"]),
            **parameters.floor_report(
                sampler.base_devices,
                parameter_names,
                min_capacity_mw=float(bounds_cfg["min_capacity_mw"]),
                min_storage_mw=float(bounds_cfg["min_storage_mw"]),
            ),
        }
        n_years = max(1, len(meta.get("years", [])) or 1)
        meta["year_factor"] = 8760.0 * n_years / float(sampler.total_hours)
        meta["carrier_labels"] = labels
        # The optimizer settings the iteration tables report (step_size / clip),
        # so `planning/history.py` never has to re-read the config.
        meta["optimizer"] = dict(self.options["optimizer"])
        # The as-built capacities, per parameter, in MW.  ``metrics.py`` needs
        # them to report ``capacity_added_mw``, and they only reach it through
        # ``PlanningResult.meta`` (which ``from_context`` copies out of here).
        meta["initial_parameters"] = {
            k: np.asarray(v, dtype=float).reshape(-1).tolist() for k, v in initial.items()
        }
        meta["sampler"] = sampler.summary()

        return PlanningContext(
            system=system,
            sampler=sampler,
            blocks=blocks,
            problem=problem,
            parameter_names=parameter_names,
            lower_bounds=lower,
            upper_bounds=upper,
            initial_parameters=initial,
            carrier_labels=labels,
            total_hours=int(sampler.total_hours),
            sampled_hours=sampled_hours,
            annualization_factor=af,
            build_seconds=time.perf_counter() - t0,
            meta=meta,
        )

    @abc.abstractmethod
    def solve(self, ctx: PlanningContext) -> PlanningResult:  # pragma: no cover - abstract
        ...


def make_method(cfg: dict) -> PlanningMethod:
    """Registry lookup on ``cfg["planning"]["method"]`` (spec section 3.2)."""
    preset = planning_options(cfg)["method"]
    if preset not in METHOD_PRESETS:
        raise ConfigError(
            f"unknown planning method {preset!r}; known methods: {sorted(METHOD_PRESETS)}"
        )
    cls_name = METHOD_PRESETS[preset]
    import_error = _load_methods()
    if cls_name not in METHODS:
        raise ConfigError(
            f"planning method {preset!r} maps to the {cls_name!r} class, which is not "
            f"registered; registered classes: {sorted(METHODS)}."
            + (
                f" Importing experiments.ra.planning.methods failed: {import_error}"
                if import_error
                else ""
            )
        )
    return METHODS[cls_name](cfg)


def plan(system, cfg: dict) -> PlanningResult:
    """``build()`` + ``solve()``, timed (spec section 3.2)."""
    t0 = time.perf_counter()
    method = make_method(cfg)
    ctx = method.build(system)
    result = method.solve(ctx)
    total = time.perf_counter() - t0
    result.timing.setdefault("build_s", ctx.build_seconds)
    result.timing["total_s"] = total
    return result


ObjectiveFactory = Callable[[list], Any]
