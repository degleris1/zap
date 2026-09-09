"""Config loading: include expansion, deep merge, ``--set`` overrides, validation.

A config file is a YAML dict.  It may carry ``includes: [path, ...]``; each
path is resolved relative to the including file's directory first and to the
config root second.  Resolution is depth-first: the includes are merged in
order, then the file's own keys, then (at the top level) the CLI overrides.
The resolved config never contains an ``includes`` key.

Every resolved config is merged on top of ``configs/base.yaml``, which defines
the complete key space; any key absent from ``base.yaml`` is an error, so a
typo cannot silently become a no-op.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from .paths import config_root

#: Sub-trees whose keys are passed straight through to a solver and therefore
#: cannot be validated against ``base.yaml``.
OPAQUE_PATHS = frozenset(
    {
        "methods.lp.solver_kwargs",
        "methods.admm.solver_kwargs",
        "planning.dispatch_solver_kwargs",
        "planning.single_level.solver_kwargs",
        "planning.warm_start.solver_kwargs",
        "planning.admm.solver_kwargs",
    }
)

VALID_DEMAND_SCALING = ("none", "fixed", "peak_fraction")
VALID_EXPORT_MODE = ("sink", "drop")
VALID_REFERENCE = ("window", "full_year", "none")

METHOD_NAMES = ("lp", "admm")

#: Top-level run modes (D-W1).  ``dispatch`` is WP4's block dispatch of a design;
#: ``plan`` is a capacity-expansion solve enumerated as a single task.
MODE_DISPATCH = "dispatch"
MODE_PLAN = "plan"
VALID_MODES = (MODE_DISPATCH, MODE_PLAN)

#: ``heuristics.name`` labels (D-W4).  ``elcc_prm`` is phase-2 work and is
#: rejected here rather than at solve time.
VALID_HEURISTIC_NAMES = ("none", "ucap_derate", "elcc_prm")
UNIMPLEMENTED_HEURISTIC_NAMES = ("elcc_prm",)

#: ``planning.method`` presets (D-W2) and the keys they map onto.
VALID_PLANNING_METHODS = ("monolithic", "stochastic", "relaxed", "gradient", "admm")
VALID_SINGLE_LEVEL_KINDS = ("primal", "strong_duality")
#: ``single_level.kind`` values that name a formulation this codebase cannot
#: build yet.  Rejected at config time, like ``heuristics.name: elcc_prm``,
#: rather than raising deep inside cvxpy.
UNIMPLEMENTED_SINGLE_LEVEL_KINDS = ("strong_duality",)
#: ``planning`` keys that are accepted for key-space stability but no longer do
#: anything.  Setting one to a non-default value is a ConfigError so a stale
#: config cannot silently mean something it no longer means.
DEPRECATED_PLANNING_KEYS = {
    "optimizer.eval_final_full_loss": (
        False,
        (
            "the planning objective is always the full forward pass (inv + op) at the "
            "final parameters, never the last minibatch loss"
        ),
    ),
}
VALID_EXPANSION_MODES = ("none", "pypsa")
VALID_EMISSIONS_MODES = ("none", "price", "cap", "dual_ascent")
VALID_CAP_BASIS = ("annual", "horizon")
VALID_BATCH_STRATEGIES = ("sequential", "fixed", "random")
VALID_MACHINES = ("cpu", "cuda", "mps")

#: ``selection`` keys each mode actually reads (R-W9); logged by ``cli run``.
DISPATCH_SELECTION_KEYS = ("blocks", "reference", "reference_window")
PLAN_SELECTION_KEYS = (
    "strategy",
    "block_size",
    "num_blocks",
    "avoid_year_boundaries",
    "seed",
)


def is_plan_mode(cfg: dict) -> bool:
    return str(cfg.get("mode", MODE_DISPATCH)) == MODE_PLAN


class ConfigError(ValueError):
    """Raised for a malformed, unknown-key, or unresolvable config."""


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level of a config must be a mapping")
    return data


def base_config() -> dict:
    """The default config: the full key space with default values."""
    return load_yaml(config_root() / "base.yaml")


def deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge ``b`` into ``a`` (``b`` wins); neither input is mutated."""
    out = copy.deepcopy(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_include(include: str, parent: Path) -> Path:
    candidates = [parent.parent / include, config_root() / include]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise ConfigError(
        f"{parent}: cannot resolve include {include!r}; tried "
        + ", ".join(str(c) for c in candidates)
    )


def expand_includes(path: Path, _stack: tuple[Path, ...] = ()) -> dict:
    """Load ``path`` and merge in its includes, depth first."""
    path = Path(path).resolve()
    if path in _stack:
        cycle = " -> ".join(str(p) for p in (*_stack, path))
        raise ConfigError(f"circular include: {cycle}")
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    raw = load_yaml(path)
    includes = raw.pop("includes", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise ConfigError(f"{path}: `includes` must be a list of paths")

    merged: dict = {}
    for inc in includes:
        merged = deep_merge(merged, expand_includes(_resolve_include(inc, path), (*_stack, path)))
    return deep_merge(merged, raw)


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse ``a.b.c=value``; the value is parsed as a YAML scalar."""
    if "=" not in item:
        raise ConfigError(f"--set expects key=value, got {item!r}")
    key, _, value = item.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"--set expects a non-empty key, got {item!r}")
    return key.split("."), yaml.safe_load(value)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    out = copy.deepcopy(cfg)
    for item in overrides:
        keys, value = parse_override(item)
        node = out
        for k in keys[:-1]:
            nxt = node.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                node[k] = nxt
            node = nxt
        node[keys[-1]] = value
    return out


def _validate_keys(cfg: dict, schema: dict, prefix: str = "") -> None:
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if key not in schema:
            raise ConfigError(
                f"unknown config key {path!r}; the key space is defined by configs/base.yaml"
            )
        if path in OPAQUE_PATHS:
            continue
        if isinstance(value, dict) and isinstance(schema[key], dict):
            _validate_keys(value, schema[key], prefix=f"{path}.")


def _as_int_list(value, path: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{path} must be a list, got {value!r}")
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{path} must be a list of integers, got {value!r}") from exc


def normalize(cfg: dict) -> dict:
    """Coerce scalar types so that two configs meaning the same thing hash alike."""
    cfg = copy.deepcopy(cfg)

    cfg["name"] = str(cfg["name"])
    cfg["mode"] = str(cfg["mode"])
    cfg["dataset"]["dir"] = str(cfg["dataset"]["dir"])
    cfg["dataset"]["years"] = _as_int_list(cfg["dataset"]["years"], "dataset.years")
    win = cfg["dataset"]["window"]
    win["start"], win["stop"] = int(win["start"]), int(win["stop"])

    sysc = cfg["system"]
    for key in (
        "voll",
        "peak_capacity_fraction",
        "scale_load",
        "carbon_tax",
        "storage_init_soc",
        "storage_final_soc",
    ):
        sysc[key] = float(sysc[key])
    for key in ("clip_scale_to_one", "link_losses"):
        sysc[key] = bool(sysc[key])

    cfg["heuristics"]["name"] = str(cfg["heuristics"]["name"])
    cfg["heuristics"]["ucap_derate"] = bool(cfg["heuristics"]["ucap_derate"])
    cfg["heuristics"]["outage_draws"] = _as_int_list(
        cfg["heuristics"]["outage_draws"], "heuristics.outage_draws"
    )

    sel = cfg["selection"]
    sel["blocks"] = _as_int_list(sel["blocks"], "selection.blocks")
    sel["reference"] = str(sel["reference"])
    ref = sel["reference_window"]
    ref["start"], ref["hours"] = int(ref["start"]), int(ref["hours"])
    sel["strategy"] = str(sel["strategy"])
    sel["block_size"] = None if sel["block_size"] is None else int(sel["block_size"])
    sel["num_blocks"] = None if sel["num_blocks"] is None else int(sel["num_blocks"])
    sel["avoid_year_boundaries"] = bool(sel["avoid_year_boundaries"])
    sel["seed"] = int(sel["seed"])

    _normalize_planning(cfg["planning"])

    for name in METHOD_NAMES:
        method = cfg["methods"][name]
        method["enabled"] = bool(method["enabled"])
        method["required"] = bool(method["required"])
        method["solver"] = str(method["solver"])
        method["timeout_s"] = float(method["timeout_s"])

    return cfg


def _normalize_planning(plan: dict) -> None:
    """Coerce the ``planning`` block in place (``mode: plan`` only)."""
    plan["method"] = str(plan["method"])
    plan["dispatch_solver"] = str(plan["dispatch_solver"])
    plan["regularize"] = float(plan["regularize"])
    plan["num_workers"] = int(plan["num_workers"])
    plan["timeout_s"] = float(plan["timeout_s"])
    plan["required"] = bool(plan["required"])

    bounds = plan["bounds"]
    bounds["min_capacity_mw"] = float(bounds["min_capacity_mw"])
    bounds["min_storage_mw"] = float(bounds["min_storage_mw"])

    exp = plan["expansion"]
    exp["mode"] = str(exp["mode"])
    exp["max_capacity_multiple"] = float(exp["max_capacity_multiple"])
    exp["max_capacity_mw"] = (
        None if exp["max_capacity_mw"] is None else float(exp["max_capacity_mw"])
    )

    sl = plan["single_level"]
    sl["kind"] = str(sl["kind"])
    sl["solver"] = str(sl["solver"])
    sl["inf_value"] = float(sl["inf_value"])
    sl["price_bound"] = float(sl["price_bound"])

    ws = plan["warm_start"]
    ws["enabled"] = bool(ws["enabled"])
    ws["num_blocks"] = None if ws["num_blocks"] is None else int(ws["num_blocks"])
    ws["solver"] = str(ws["solver"])

    opt = plan["optimizer"]
    opt["num_iterations"] = int(opt["num_iterations"])
    opt["step_size"] = float(opt["step_size"])
    opt["clip"] = float(opt["clip"])
    opt["batch_size"] = int(opt["batch_size"])
    opt["batch_strategy"] = str(opt["batch_strategy"])
    for key in ("init_full_loss", "save_param_history", "eval_final_full_loss"):
        opt[key] = bool(opt[key])
    opt["peak_net_load_k"] = None if opt["peak_net_load_k"] is None else int(opt["peak_net_load_k"])
    opt["peak_net_load_rerank_every"] = int(opt["peak_net_load_rerank_every"])

    admm = plan["admm"]
    admm["machine"] = str(admm["machine"])
    admm["dtype"] = str(admm["dtype"])
    admm["adapt_rho"] = bool(admm["adapt_rho"])
    admm["adapt_rho_rate"] = float(admm["adapt_rho_rate"])

    emis = plan["emissions"]
    emis["mode"] = str(emis["mode"])
    emis["price"] = float(emis["price"])
    emis["cap"] = None if emis["cap"] is None else float(emis["cap"])
    emis["cap_basis"] = str(emis["cap_basis"])
    da = emis["dual_ascent"]
    da["target"] = None if da["target"] is None else float(da["target"])
    da["initial_weight"] = float(da["initial_weight"])
    da["dual_step_size"] = float(da["dual_step_size"])
    da["max_weight"] = float(da["max_weight"])
    da["num_outer_iterations"] = int(da["num_outer_iterations"])
    da["tolerance"] = float(da["tolerance"])

    if plan["budget_constraints"] is not None:
        plan["budget_constraints"] = str(plan["budget_constraints"])


def _validate_planning(cfg: dict) -> None:
    """Validate the ``planning`` / ``selection`` keys ``mode: plan`` reads."""
    plan = cfg["planning"]
    if plan["method"] not in VALID_PLANNING_METHODS:
        raise ConfigError(
            f"planning.method must be one of {VALID_PLANNING_METHODS}, got {plan['method']!r}"
        )
    if plan["single_level"]["kind"] not in VALID_SINGLE_LEVEL_KINDS:
        raise ConfigError(
            f"planning.single_level.kind must be one of {VALID_SINGLE_LEVEL_KINDS}, "
            f"got {plan['single_level']['kind']!r}"
        )
    if plan["single_level"]["kind"] in UNIMPLEMENTED_SINGLE_LEVEL_KINDS:
        raise ConfigError(
            "planning.single_level.kind 'strong_duality' has no implementation for this "
            "device set: RelaxedPlanningProblem dualizes the devices through "
            "zap.dual.dualize, whose DUAL_CLASS table has no entry for DirectedLine "
            "(it knows DCLine / ACLine only), so every system this harness builds "
            "raises KeyError: DirectedLine. A dual DirectedLine is phase-2 zap work; "
            "see configs/methods/plan_relaxed.yaml."
        )
    for path, (default, why) in DEPRECATED_PLANNING_KEYS.items():
        node: Any = plan
        for key in path.split("."):
            node = node[key]
        if node != default:
            raise ConfigError(
                f"planning.{path} is deprecated and ignored ({why}); "
                f"remove it from the config or set it back to {default!r}"
            )
    if plan["expansion"]["mode"] not in VALID_EXPANSION_MODES:
        raise ConfigError(
            f"planning.expansion.mode must be one of {VALID_EXPANSION_MODES}, "
            f"got {plan['expansion']['mode']!r}"
        )
    if plan["optimizer"]["batch_strategy"] not in VALID_BATCH_STRATEGIES:
        raise ConfigError(
            f"planning.optimizer.batch_strategy must be one of {VALID_BATCH_STRATEGIES}, "
            f"got {plan['optimizer']['batch_strategy']!r}"
        )
    if plan["emissions"]["mode"] not in VALID_EMISSIONS_MODES:
        raise ConfigError(
            f"planning.emissions.mode must be one of {VALID_EMISSIONS_MODES}, "
            f"got {plan['emissions']['mode']!r}"
        )
    if plan["emissions"]["cap_basis"] not in VALID_CAP_BASIS:
        raise ConfigError(
            f"planning.emissions.cap_basis must be one of {VALID_CAP_BASIS}, "
            f"got {plan['emissions']['cap_basis']!r}"
        )
    if plan["admm"]["machine"] not in VALID_MACHINES:
        raise ConfigError(
            f"planning.admm.machine must be one of {VALID_MACHINES}, "
            f"got {plan['admm']['machine']!r}"
        )
    if plan["admm"]["machine"] == "mps" and plan["admm"]["dtype"] == "float64":
        raise ConfigError(
            "planning.admm.machine 'mps' cannot be combined with dtype 'float64': "
            "torch MPS has no float64 (R-B / R-W3)"
        )
    if plan["num_workers"] < 1:
        raise ConfigError(f"planning.num_workers must be >= 1, got {plan['num_workers']}")

    # A carbon tax is folded into the generators' marginal cost by the importer,
    # while `planning.emissions.price` adds an explicit price * emissions term to
    # every subproblem's operation objective.  Both at once double-counts carbon,
    # and only the second is netted out of the reported system cost.
    priced = plan["emissions"]["mode"] in ("price", "dual_ascent")
    if priced and float(cfg["system"]["carbon_tax"]) != 0.0:
        raise ConfigError(
            f"system.carbon_tax ({cfg['system']['carbon_tax']}) is non-zero and "
            f"planning.emissions.mode is {plan['emissions']['mode']!r}: "
            "the tax is folded into marginal cost by the importer and the price is added "
            "to the planning operation objective, so carbon would be charged twice. "
            "Set exactly one of them."
        )

    sel = cfg["selection"]
    if sel["block_size"] is not None and sel["block_size"] <= 0:
        raise ConfigError(f"selection.block_size must be positive or null, got {sel['block_size']}")
    if sel["num_blocks"] is not None and sel["num_blocks"] <= 0:
        raise ConfigError(f"selection.num_blocks must be positive or null, got {sel['num_blocks']}")

    # The selection registry and the D-W5 emissions matrix live with the
    # planning core; import them lazily so `config` stays importable without it.
    try:
        from .planning.constraints import validate_emissions
        from .planning.selection import SELECTORS
    except ImportError:  # pragma: no cover - the planning core is a hard dependency
        return
    if sel["strategy"] not in SELECTORS:
        raise ConfigError(
            f"unknown selection.strategy {sel['strategy']!r}; known strategies: {sorted(SELECTORS)}"
        )
    validate_emissions(cfg)


def validate(cfg: dict) -> dict:
    """Validate the resolved config; returns it normalized."""
    if "includes" in cfg:
        raise ConfigError("resolved config still carries an `includes` key")
    _validate_keys(cfg, base_config())
    cfg = normalize(cfg)

    if cfg["mode"] not in VALID_MODES:
        raise ConfigError(f"mode must be one of {VALID_MODES}, got {cfg['mode']!r}")
    plan_mode = is_plan_mode(cfg)

    win = cfg["dataset"]["window"]
    if not (0 <= win["start"] < win["stop"]):
        raise ConfigError(f"dataset.window must satisfy 0 <= start < stop, got {win}")
    if not cfg["dataset"]["years"]:
        raise ConfigError("dataset.years must not be empty")

    sysc = cfg["system"]
    if sysc["demand_scaling"] not in VALID_DEMAND_SCALING:
        raise ConfigError(
            f"system.demand_scaling must be one of {VALID_DEMAND_SCALING}, "
            f"got {sysc['demand_scaling']!r}"
        )
    if sysc["export_mode"] not in VALID_EXPORT_MODE:
        raise ConfigError(f"system.export_mode must be one of {VALID_EXPORT_MODE}")

    if plan_mode:
        _validate_planning(cfg)

    sel = cfg["selection"]
    if sel["reference"] not in VALID_REFERENCE:
        raise ConfigError(f"selection.reference must be one of {VALID_REFERENCE}")
    if any(b <= 0 for b in sel["blocks"]):
        raise ConfigError("selection.blocks must be positive hour counts")
    if sel["reference"] == "window" and not plan_mode:
        ref = sel["reference_window"]
        if ref["hours"] <= 0:
            raise ConfigError("selection.reference_window.hours must be positive")
        if ref["start"] < win["start"] or ref["start"] + ref["hours"] > win["stop"]:
            raise ConfigError(
                f"selection.reference_window must lie inside dataset.window ({ref} vs {win})"
            )
        # The blocking-error comparison sums whole blocks inside the reference
        # window, so the window must be an exact number of blocks of every size.
        # Offsets are measured from the window start, where blocking begins.
        for size in sel["blocks"]:
            if (ref["start"] - win["start"]) % size or ref["hours"] % size:
                raise ConfigError(
                    f"selection.reference_window {ref} is not aligned to block size {size}: "
                    f"(start - dataset.window.start) = {ref['start'] - win['start']} and "
                    f"hours = {ref['hours']} must both be multiples of {size}, or blocked "
                    "sums cannot cover the reference window exactly"
                )

    name = cfg["heuristics"]["name"]
    if name not in VALID_HEURISTIC_NAMES:
        raise ConfigError(f"heuristics.name must be one of {VALID_HEURISTIC_NAMES}, got {name!r}")
    if name in UNIMPLEMENTED_HEURISTIC_NAMES:
        raise ConfigError(
            f"heuristics.name {name!r} is phase-2 work and has no implementation "
            "(D-W4); see memory/PROJECT.md 2.2 for the ELCC + PRM plan."
        )
    if name == "ucap_derate" and not cfg["heuristics"]["ucap_derate"]:
        raise ConfigError(
            "heuristics.name is 'ucap_derate' but heuristics.ucap_derate is false: "
            "the label and the mechanism must agree (D-W4)"
        )
    if cfg["heuristics"]["ucap_derate"] and cfg["heuristics"]["outage_draws"]:
        raise ConfigError("heuristics.ucap_derate and heuristics.outage_draws are exclusive")

    if not plan_mode and not any(cfg["methods"][m]["enabled"] for m in METHOD_NAMES):
        raise ConfigError("no method is enabled")

    shard = cfg["execution"]["shard"]
    if shard is not None:
        parse_shard(shard)  # raises on a malformed value

    return cfg


def parse_shard(shard: str) -> tuple[int, int]:
    """Parse ``"k/n"`` into ``(k, n)`` with ``1 <= k <= n``."""
    try:
        k_str, n_str = str(shard).split("/")
        k, n = int(k_str), int(n_str)
    except ValueError as exc:
        raise ConfigError(f"execution.shard must look like 'k/n', got {shard!r}") from exc
    if not (1 <= k <= n):
        raise ConfigError(f"execution.shard must satisfy 1 <= k <= n, got {shard!r}")
    return k, n


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> dict:
    """Load, expand, merge onto the defaults, apply overrides, validate."""
    resolved = expand_includes(Path(path))
    merged = deep_merge(base_config(), resolved)
    merged = apply_overrides(merged, overrides)
    return validate(merged)


def dump_config(cfg: dict, path: Path) -> Path:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
    return path
