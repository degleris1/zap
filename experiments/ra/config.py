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
OPAQUE_PATHS = frozenset({"methods.lp.solver_kwargs", "methods.admm.solver_kwargs"})

VALID_DEMAND_SCALING = ("none", "fixed", "peak_fraction")
VALID_EXPORT_MODE = ("sink", "drop")
VALID_REFERENCE = ("window", "full_year", "none")

METHOD_NAMES = ("lp", "admm")


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

    cfg["heuristics"]["ucap_derate"] = bool(cfg["heuristics"]["ucap_derate"])
    cfg["heuristics"]["outage_draws"] = _as_int_list(
        cfg["heuristics"]["outage_draws"], "heuristics.outage_draws"
    )

    sel = cfg["selection"]
    sel["blocks"] = _as_int_list(sel["blocks"], "selection.blocks")
    sel["reference"] = str(sel["reference"])
    ref = sel["reference_window"]
    ref["start"], ref["hours"] = int(ref["start"]), int(ref["hours"])

    for name in METHOD_NAMES:
        method = cfg["methods"][name]
        method["enabled"] = bool(method["enabled"])
        method["required"] = bool(method["required"])
        method["solver"] = str(method["solver"])
        method["timeout_s"] = float(method["timeout_s"])

    return cfg


def validate(cfg: dict) -> dict:
    """Validate the resolved config; returns it normalized."""
    if "includes" in cfg:
        raise ConfigError("resolved config still carries an `includes` key")
    _validate_keys(cfg, base_config())
    cfg = normalize(cfg)

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

    sel = cfg["selection"]
    if sel["reference"] not in VALID_REFERENCE:
        raise ConfigError(f"selection.reference must be one of {VALID_REFERENCE}")
    if any(b <= 0 for b in sel["blocks"]):
        raise ConfigError("selection.blocks must be positive hour counts")
    if sel["reference"] == "window":
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

    if cfg["heuristics"]["ucap_derate"] and cfg["heuristics"]["outage_draws"]:
        raise ConfigError("heuristics.ucap_derate and heuristics.outage_draws are exclusive")

    if not any(cfg["methods"][m]["enabled"] for m in METHOD_NAMES):
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
