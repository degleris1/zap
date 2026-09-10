"""The evaluation-pipeline seam (thin in phase 1).

Evaluating a set of candidate designs is *the same task machinery* with a
``Design`` attached: one task per (design, block, method, draw), the same ledger,
the same aggregation.  Phase 1 evaluates only the as-built design, which is why
the operational benchmark is written as an evaluation of ``asbuilt``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as metrics_mod
from . import tasks as tasks_mod
from .config import ConfigError, is_plan_mode
from .system import Design

logger = logging.getLogger(__name__)

#: The evaluation config the spec fixes (2.1).  Deviating from any of these is a
#: warning recorded on the card, never a refusal: a chapter figure may
#: legitimately score a design on 24 h blocks to show what blocking does to it.
EXPECTED_EVALUATION_CONFIG = {
    "selection.blocks": [168],
    "selection.reference": "none",
    "system.storage_soc_mode": "cyclic_free",
    "system.demand_scaling": "none",
    "system.export_mode": "drop",
    "dataset.window": {"start": 7, "stop": 8743},
}


def designs_from_run(run_dir: Path, system=None) -> list[Design]:
    """Read every ``designs/*.json`` a planning run wrote (the WP5 seam, 3.4).

    ``system`` is optional; when given, each design's recorded row names are
    asserted equal to the system's, so a design built on a different dataset
    fails loudly instead of silently mis-mapping.
    """
    from .planning.design import design_paths, read_design

    designs = [read_design(path, system=system) for path in design_paths(run_dir)]
    logger.info("read %d design(s) from %s", len(designs), Path(run_dir) / "designs")
    return designs


def evaluate_designs(
    designs: Sequence[Design],
    cfg: dict,
    run_dir: Path,
    *,
    force: bool = False,
    shard: str | None = None,
) -> pd.DataFrame:
    """Score every design on the same blocks and return the aggregated frame.

    ``cfg`` is a **dispatch-mode** config: evaluation is block dispatch of a
    design, whatever produced the design.
    """
    by_id = {d.design_id: d for d in designs}
    if len(by_id) != len(designs):
        raise ValueError("design_id must be unique across designs")

    all_tasks = tasks_mod.enumerate_tasks(cfg, design_ids=tuple(by_id))
    selected = tasks_mod.select_shard(all_tasks, shard)

    tasks_mod.run_tasks(selected, cfg, run_dir, force=force, designs=by_id)

    frame = metrics_mod.aggregate(run_dir, cfg)
    try:
        write_eval_tables(run_dir)
    except Exception as exc:  # a summary table must never fail a run
        logger.warning("could not write the evaluation tables: %s", exc, exc_info=True)
    return frame


# ---------------------------------------------------------------------------
# Design sources: what `ra evaluate` was asked to score (WP-E2, spec 2.1/2.3)
# ---------------------------------------------------------------------------

SOURCES_NAME = "SOURCES.json"
SOURCES_SCHEMA_VERSION = 1
PREFLIGHT_NAME = "preflight.json"
PREFLIGHT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DesignSource:
    """One ``design.json`` and where it came from."""

    design_id: str
    design: Design
    path: Path
    source_run_id: str | None
    sha256: str
    zap_commit: str | None
    record: dict

    def entry(self) -> dict:
        """The ``designs/SOURCES.json`` row for this design."""
        return {
            "design_id": self.design_id,
            "source_run_id": self.source_run_id,
            "source_path": str(self.path),
            "design_json_sha256": self.sha256,
            "zap_commit": self.zap_commit,
            "added_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _design_source(path: Path) -> DesignSource:
    from .planning.design import read_design_record, record_to_design

    path = Path(path)
    record = read_design_record(path)
    design = record_to_design(record)
    return DesignSource(
        design_id=str(record.get("design_id") or path.stem),
        design=design,
        path=path,
        source_run_id=record.get("run_id"),
        sha256=sha256_file(path),
        zap_commit=record.get("zap_commit"),
        record=record,
    )


def check_design_against_config(source: DesignSource, cfg: dict) -> list[str]:
    """Compare a design's provenance with the evaluation config (spec 2.1).

    A **dataset** mismatch raises: the design's capacity vector is positional, so
    scoring it against another dataset's rows would silently mis-map every row.
    A window / years / heuristics mismatch is expected and returns a warning --
    a design planned on 12 weeks is *supposed* to be evaluated on 52.
    """
    warnings: list[str] = []
    dataset = str(cfg["dataset"]["dir"])
    recorded = source.record.get("dataset")
    # Compare by directory *name* too: a design may record an absolute path.
    if (
        recorded is not None
        and str(recorded) != dataset
        and Path(str(recorded)).name != Path(dataset).name
    ):
        raise ConfigError(
            f"design {source.design_id!r} was built on dataset {recorded!r} but this "
            f"evaluation config uses {dataset!r}; the capacity vector is positional, so "
            "scoring it here would mis-map every row"
        )

    window = source.record.get("window") or {}
    cfg_window = cfg["dataset"]["window"]
    if window and (
        int(window.get("start", -1)) != int(cfg_window["start"])
        or int(window.get("stop", -1)) != int(cfg_window["stop"])
    ):
        warnings.append(
            f"design {source.design_id!r} was planned on window {dict(window)} and is being "
            f"evaluated on {dict(cfg_window)}"
        )
    years = [int(y) for y in (source.record.get("years") or [])]
    cfg_years = [int(y) for y in cfg["dataset"]["years"]]
    if years and years != cfg_years:
        warnings.append(
            f"design {source.design_id!r} was planned on years {years} and is being evaluated "
            f"on {cfg_years}"
        )
    heuristics = source.record.get("heuristics") or {}
    if heuristics and str(heuristics.get("name", "none")) != str(cfg["heuristics"]["name"]):
        warnings.append(
            f"design {source.design_id!r} was planned with heuristic "
            f"{heuristics.get('name')!r} and is being evaluated with "
            f"{cfg['heuristics']['name']!r} (evaluation never derates)"
        )
    return warnings


def load_designs(
    design_runs: Sequence[str] = (),
    design_files: Sequence[str] = (),
    cfg: dict | None = None,
    *,
    runs_root: str | Path | None = None,
) -> list[DesignSource]:
    """Every design named by ``--design-run`` / ``--design-file`` (spec 2.1).

    ``--design-run`` takes a *planning run id* and picks up every
    ``designs/*.json`` in it; ``--design-file`` takes one ``design.json`` for
    ad-hoc scoring.  Design ids must be unique across the whole set.
    """
    from . import paths
    from .planning.design import design_paths

    sources: list[DesignSource] = []
    for run in design_runs or ():
        run_dir = Path(run)
        if not run_dir.is_dir():
            run_dir = paths.run_dir(str(run), runs_root)
        found = design_paths(run_dir)
        if not found:
            raise ConfigError(f"--design-run {run!r}: no designs/*.json in {run_dir}")
        sources.extend(_design_source(p) for p in found)
    for path in design_files or ():
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"--design-file {path} does not exist")
        sources.append(_design_source(path))

    if not sources:
        raise ConfigError("ra evaluate needs at least one --design-run or --design-file")

    seen: dict[str, DesignSource] = {}
    for source in sources:
        other = seen.get(source.design_id)
        if other is not None:
            raise ConfigError(
                f"design id {source.design_id!r} appears twice in this invocation "
                f"({other.path} and {source.path}); design ids must be unique"
            )
        seen[source.design_id] = source

    if cfg is not None:
        for source in sources:
            for warning in check_design_against_config(source, cfg):
                logger.warning("%s", warning)
    logger.info(
        "scoring %d design(s): %s", len(sources), ", ".join(s.design_id for s in sources)
    )
    return sources


def read_sources(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "designs" / SOURCES_NAME
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return list(payload.get("designs") or [])


def record_sources(run_dir: Path, sources: Sequence[DesignSource]) -> Path:
    """Append to ``designs/SOURCES.json`` and copy each ``design.json`` in (D4).

    The design set is **not** in the run-id hash, so a run directory accumulates
    designs: adding a sixth design re-runs only its blocks and ``eval.parquet``
    then covers all six.  Two guards make that safe -- task ids begin with the
    design id, so no id can collide, and re-adding an id whose file has a
    *different* sha256 is a hard error rather than a silent overwrite of a
    scored design.
    """
    run_dir = Path(run_dir)
    designs_dir = run_dir / "designs"
    designs_dir.mkdir(parents=True, exist_ok=True)

    existing = {e["design_id"]: e for e in read_sources(run_dir)}
    entries = list(read_sources(run_dir))
    for source in sources:
        previous = existing.get(source.design_id)
        if previous is not None:
            if previous.get("design_json_sha256") != source.sha256:
                raise ConfigError(
                    f"design id {source.design_id!r} has already been scored in {run_dir} "
                    f"from a file with sha256 {previous.get('design_json_sha256')}, but "
                    f"{source.path} hashes to {source.sha256}. Rename the design or use a "
                    "new run directory -- otherwise eval.parquet would pool two different "
                    "systems under one name."
                )
            continue  # same file, nothing to do
        entries.append(source.entry())
        (designs_dir / f"{source.design_id}.json").write_text(
            json.dumps(source.record, indent=2)
        )

    path = designs_dir / SOURCES_NAME
    path.write_text(
        json.dumps(
            {"schema_version": SOURCES_SCHEMA_VERSION, "designs": entries},
            indent=2,
            sort_keys=True,
        )
    )
    return path


# ---------------------------------------------------------------------------
# Preflight (WP-E2, spec 2.1 / 3.2 / 5)
# ---------------------------------------------------------------------------


def _check(name: str, ok: bool, detail: str, **extra) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail, **extra}


def _get(cfg: dict, path: str):
    node = cfg
    for key in path.split("."):
        node = node[key]
    return node


def check_config_gates(cfg: dict) -> list[dict]:
    """The gates of spec 2.1: two refusals, the rest warnings on the card."""
    checks = []
    checks.append(
        _check(
            "mode_is_dispatch",
            not is_plan_mode(cfg),
            f"mode = {cfg['mode']!r} (evaluation is blocked dispatch of a design)",
        )
    )
    checks.append(
        _check(
            "ucap_off",
            not bool(cfg["heuristics"]["ucap_derate"]),
            "heuristics.ucap_derate must be false: evaluation never derates, the criterion "
            "is measured against realised outage draws (a UCAP-derated design is scored "
            "without UCAP)",
        )
    )
    draws = [int(d) for d in cfg["heuristics"]["outage_draws"]]
    allow_no_draws = bool((cfg.get("evaluation") or {}).get("allow_no_draws", False))
    checks.append(
        _check(
            "draws_present",
            bool(draws) or allow_no_draws,
            f"heuristics.outage_draws = {draws} "
            f"(set evaluation.allow_no_draws: true for a weather-only evaluation)",
            n_draws=len(draws),
            allow_no_draws=allow_no_draws,
        )
    )

    deviations = []
    for path, expected in EXPECTED_EVALUATION_CONFIG.items():
        actual = _get(cfg, path)
        if actual != expected:
            deviations.append(f"{path} = {actual!r} (expected {expected!r})")
    for text in deviations:
        logger.warning("evaluation config deviates from the spec default: %s", text)
    checks.append(
        _check(
            "config_defaults",
            True,  # a warning, never a refusal
            "matches the spec's evaluation defaults"
            if not deviations
            else "DEVIATIONS (allowed, recorded on the card): " + "; ".join(deviations),
            deviations=deviations,
        )
    )
    return checks


def check_outage_store(cfg: dict) -> dict:
    """Every requested (year, draw) exists in ``outages.zarr`` and is ``done`` (E9)."""
    draws = [int(d) for d in cfg["heuristics"]["outage_draws"]]
    years = [int(y) for y in cfg["dataset"]["years"]]
    if not draws:
        return _check("outage_store", True, "no outage draws requested", missing=[])

    from zap.importers.wy_store import _open_outage_store

    from .system import dataset_path

    path, root = _open_outage_store(dataset_path(cfg))
    store_years = [int(y) for y in np.asarray(root["weather_year"][:])]
    store_draws = [int(d) for d in np.asarray(root["draw"][:])]
    has_done = "done" in root

    missing = []
    for year in years:
        for draw in draws:
            if year not in store_years:
                missing.append({"year": year, "draw": draw, "why": "year not in store"})
                continue
            if draw not in store_draws:
                missing.append({"year": year, "draw": draw, "why": "draw not in store"})
                continue
            if has_done and not bool(
                np.asarray(root["done"][store_years.index(year), store_draws.index(draw)])
            ):
                missing.append({"year": year, "draw": draw, "why": "chunk not generated"})
    return _check(
        "outage_store",
        not missing,
        f"{len(years) * len(draws) - len(missing)} of {len(years) * len(draws)} "
        f"(year, draw) chunks present and done in {path}"
        + (f"; missing: {missing[:10]}" if missing else ""),
        store=str(path),
        n_missing=len(missing),
        missing=missing[:50],
    )


def _capacity_rows(record: dict):
    """``(class, row name, capacity MW)`` triples of a ``design.json`` record."""
    for cls_name, entry in (record.get("capacities") or {}).items():
        names = entry.get("names")
        attr = next((k for k in entry if k != "names"), None)
        if names is None or attr is None:
            continue
        for name, value in zip(names, entry[attr]):
            yield cls_name, str(name), float(value)


def check_pool_capacity(sources: Sequence[DesignSource], cfg: dict) -> dict:
    """``ceil(C / unit size) <= row_units`` for every pooled row of every design.

    A design that outgrows its row's slice of the pool raises from
    ``_row_weights`` *mid-run*; this is the same arithmetic, done once, before
    any task starts (spec 3.1 consequence (b)).
    """
    if not [int(d) for d in cfg["heuristics"]["outage_draws"]]:
        return _check("pool_capacity", True, "no outage draws requested", worst_ratio=None)

    from zap.importers.wy_store import _open_outage_store, _unit_pool_from_store

    from .system import dataset_path

    _, root = _open_outage_store(dataset_path(cfg))
    pool = _unit_pool_from_store(root)

    worst_ratio = 0.0
    worst_row = None
    failures = []
    n_rows = 0
    for source in sources:
        for _cls, row, capacity in _capacity_rows(source.record):
            if row not in pool.row_offset or capacity <= 0.0:
                continue
            n_rows += 1
            size = float(pool.row_size[row])
            units = int(pool.row_units[row])
            # Exactly `zap.reliability.outages._row_weights`: the epsilon keeps
            # an exact multiple from rounding up to one unit too many.
            needed = max(1, math.ceil(capacity / size - 1e-9))
            ratio = needed / units if units else float("inf")
            if ratio > worst_ratio:
                worst_ratio, worst_row = ratio, (source.design_id, row, needed, units, capacity)
            if needed > units:
                failures.append(
                    {
                        "design_id": source.design_id,
                        "row": row,
                        "capacity_mw": capacity,
                        "unit_size_mw": size,
                        "units_needed": needed,
                        "row_units": units,
                    }
                )
    detail = f"{n_rows} pooled row(s) checked; worst units_needed / row_units = {worst_ratio:.3f}"
    if worst_row is not None:
        detail += (
            f" ({worst_row[0]}: {worst_row[1]!r} needs {worst_row[2]} of {worst_row[3]} units "
            f"for {worst_row[4]:,.1f} MW)"
        )
    if failures:
        detail += f"; {len(failures)} row(s) exceed the pool: {failures[:5]}"
    return _check(
        "pool_capacity",
        not failures,
        detail,
        worst_ratio=worst_ratio,
        n_rows_checked=n_rows,
        failures=failures[:50],
    )


def _initial_by_class(record: dict) -> dict[str, np.ndarray]:
    """``design.json``'s ``initial_parameters`` re-keyed by device class.

    Planning parameters are named ``<class lowercased>_capacity`` /
    ``<class lowercased>_power`` (``planning.parameters.setup_parameter_names``),
    so the class is recoverable from the parameter name.
    """
    classes = {str(c).lower(): str(c) for c in (record.get("capacities") or {})}
    out: dict[str, np.ndarray] = {}
    for param, values in (record.get("initial_parameters") or {}).items():
        base = str(param).rsplit("_", 1)[0].lower()
        cls_name = classes.get(base)
        if cls_name is None:
            continue
        out[cls_name] = np.asarray(values, dtype=float).reshape(-1)
    return out


#: Tolerance of the as-built comparison, in MW.
AS_BUILT_TOL_MW = 1e-6


def check_as_built(sources: Sequence[DesignSource], cfg: dict) -> dict:
    """Each design's ``initial_parameters`` equals a fresh as-built load (D2, §5.4).

    This is what proves the evaluation applied the **same 2040 retirements** as
    planning: if it did not, every design's capex base ``eta^0`` would differ
    from the one its objective was computed against.
    """
    from . import system as system_mod
    from .dispatch import DESIGN_CAPACITY_ATTRS

    loaded = system_mod.build_system(cfg, draw=None, design=None)
    power_unit = float((getattr(loaded, "meta", {}) or {}).get("power_unit", 1.0) or 1.0)
    as_built: dict[str, np.ndarray] = {}
    for cls_name, i in loaded.index.device_index.items():
        device = loaded.devices[i]
        for attr in DESIGN_CAPACITY_ATTRS:
            value = getattr(device, attr, None)
            if value is not None:
                as_built[cls_name] = np.asarray(value, dtype=float).reshape(-1) * power_unit
                break

    mismatches = []
    compared = 0
    for source in sources:
        initial = _initial_by_class(source.record)
        if not initial:
            mismatches.append(
                {
                    "design_id": source.design_id,
                    "why": "design.json records no initial_parameters",
                }
            )
            continue
        for cls_name, values in initial.items():
            actual = as_built.get(cls_name)
            if actual is None:
                mismatches.append(
                    {"design_id": source.design_id, "class": cls_name, "why": "class not built"}
                )
                continue
            if actual.size != values.size:
                mismatches.append(
                    {
                        "design_id": source.design_id,
                        "class": cls_name,
                        "why": f"{values.size} rows recorded vs {actual.size} built",
                    }
                )
                continue
            compared += 1
            diff = np.abs(actual - values)
            if diff.size and float(diff.max()) > AS_BUILT_TOL_MW:
                worst = int(np.argmax(diff))
                mismatches.append(
                    {
                        "design_id": source.design_id,
                        "class": cls_name,
                        "worst_row": worst,
                        "as_built_mw": float(actual[worst]),
                        "recorded_mw": float(values[worst]),
                        "n_rows_differing": int((diff > AS_BUILT_TOL_MW).sum()),
                    }
                )
    return _check(
        "as_built_match",
        not mismatches,
        f"{compared} (design, class) vector(s) match the freshly loaded as-built system "
        f"within {AS_BUILT_TOL_MW} MW"
        + (f"; mismatches: {mismatches[:5]}" if mismatches else ""),
        n_compared=compared,
        mismatches=mismatches[:20],
    )


def preflight(
    sources: Sequence[DesignSource], cfg: dict, *, run_dir: Path | None = None
) -> dict:
    """Everything that can be known to be wrong *before* a task runs (WP-E2).

    Returns the report, also written to ``preflight.json``.  ``ok`` is false if
    any check failed; ``cmd_evaluate`` refuses to run on that.  A ``mode: plan``
    config raises immediately: none of the checks below even means anything for
    a planning config, and its ledger rows are planning objectives, not scores.
    """
    if is_plan_mode(cfg):
        raise ConfigError(
            "ra evaluate needs a `mode: dispatch` config (evaluation is blocked dispatch of "
            f"a design); {cfg.get('name')!r} is `mode: plan`"
        )
    checks = list(check_config_gates(cfg))
    for name, fn in (
        ("outage_store", lambda: check_outage_store(cfg)),
        ("pool_capacity", lambda: check_pool_capacity(sources, cfg)),
        ("as_built_match", lambda: check_as_built(sources, cfg)),
    ):
        try:
            checks.append(fn())
        except Exception as exc:  # a failed check is a failed preflight, never a crash
            logger.warning("preflight check %s could not run: %s", name, exc, exc_info=True)
            checks.append(_check(name, False, f"{type(exc).__name__}: {exc}"))

    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "checked_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "ok": all(c["ok"] for c in checks),
        "designs": [
            {
                "design_id": s.design_id,
                "source_run_id": s.source_run_id,
                "design_json_sha256": s.sha256,
            }
            for s in sources
        ],
        "years": [int(y) for y in cfg["dataset"]["years"]],
        "draws": [int(d) for d in cfg["heuristics"]["outage_draws"]],
        "checks": checks,
    }
    if run_dir is not None:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / PREFLIGHT_NAME).write_text(json.dumps(report, indent=2, default=str))
    for check in checks:
        logger.info(
            "preflight %-16s %s  %s", check["name"], "PASS" if check["ok"] else "FAIL",
            check["detail"],
        )
    return report


# ---------------------------------------------------------------------------
# Train / held-out split provenance (spec E10)
# ---------------------------------------------------------------------------


def splits_info(cfg: dict | None = None, splits_path=None) -> dict:
    """The split lists plus the file's sha256, for ``env.json`` and the card."""
    import yaml

    from .paths import brain_root

    if splits_path is None and cfg is not None:
        splits_path = (cfg.get("evaluation") or {}).get("splits_path")
    path = Path(splits_path) if splits_path else brain_root() / "data" / "splits.yaml"
    info = {
        "splits_path": str(path),
        "splits_sha256": None,
        "heldout_eval": [],
        "train": [],
        "dataset_years": [],
    }
    if not path.exists():
        return info
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):  # pragma: no cover - a malformed file is not fatal
        return info
    info["splits_sha256"] = sha256_file(path)
    for key in ("heldout_eval", "train", "dataset_years"):
        info[key] = [int(y) for y in (payload.get(key) or [])]
    return info


def read_splits(run_dir: Path, cfg: dict | None = None) -> dict:
    """The splits this run was labelled with: ``env.json`` first, the file second.

    Reading ``env.json`` first is the point of E10: once a run is scored, editing
    ``data/splits.yaml`` must not silently reclassify its rows.
    """
    env_path = Path(run_dir) / "env.json"
    if env_path.exists():
        try:
            env = json.loads(env_path.read_text())
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            env = {}
        splits = env.get("splits")
        if isinstance(splits, dict) and splits.get("heldout_eval"):
            return splits
    return splits_info(cfg)


def record_splits(run_dir: Path, cfg: dict) -> dict:
    """Write the split provenance into ``env.json`` (idempotent)."""
    path = Path(run_dir) / "env.json"
    env = {}
    if path.exists():
        try:
            env = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            env = {}
    info = splits_info(cfg)
    env["splits"] = info
    path.write_text(json.dumps(env, indent=2, sort_keys=True, default=str))
    return info


# ---------------------------------------------------------------------------
# Evaluation tables (spec section 3.7)
# ---------------------------------------------------------------------------

EVAL_COLUMNS = (
    "design_id",
    "source_run_id",
    "formulation",
    "heuristic",
    "selection_strategy",
    "emissions_mode",
    "method",
    "block_size",
    "year",
    "draw",
    "n_blocks",
    "hours",
    # `(stop - start)` of the config window for the case's weather year, and the
    # fraction of it the scored blocks actually cover (spec 4.1 / E7).
    "window_hours",
    "coverage",
    "is_holdout",
    "operational_cost",
    "generation_cost",
    "voll_cost",
    "unserved_energy_mwh",
    "lost_load_hours",
    "co2_tonnes",
    "curtailment_mwh",
    "imports_mwh",
    "exports_mwh",
    "demand_mwh",
    "capex_annual_usd",
    "total_cost_usd",
    "eue_mwh",
    "neue",
    "lolh_hours",
    "lolh_frac",
    "lol_any",
    "storage_cycles",
    "min_available_mw",
    "p5_available_mw",
    "mean_price_usd_per_mwh",
    "max_price_usd_per_mwh",
    "build_wall_clock_s",
    "solve_wall_clock_s",
    "wall_clock_s",
)

#: Keyed like ``eval.parquet`` minus (year, draw), which are averaged over:
#: a 24 h-block score and a 168 h-block score of the same design are different
#: dispatches and must not be summed into one hour.
EVAL_ENS_PROFILE_COLUMNS = (
    "design_id",
    "method",
    "block_size",
    "day_of_year",
    "hour_of_day",
    "mean_ens_mwh",
    "p95_ens_mwh",
    "n_draws",
    "n_year_draws_with_ens",
)

#: The columns identifying one scored realisation of one design.
CASE_KEYS = ("design_id", "method", "block_size", "year", "draw")

#: ``design_id`` joins the three original columns (WP-E4): the run card's
#: "Excluded cases" section says *which design* lost cases, and a design id may
#: itself contain hyphens, so it cannot be recovered from ``task_id``.
EVAL_EXCLUDED_COLUMNS = ("task_id", "design_id", "method", "block_size", "year", "draw",
                        "status", "error")


def _holdout_years(splits_path: Path | None) -> set[int]:
    import yaml

    from .paths import brain_root

    path = Path(splits_path) if splits_path is not None else brain_root() / "data" / "splits.yaml"
    if not Path(path).exists():
        return set()
    try:
        payload = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):  # a malformed split file is not fatal
        return set()
    return {int(y) for y in (payload.get("heldout_eval") or [])}


def _design_attributes(run_dir: Path) -> dict[str, dict]:
    from .planning.design import design_paths, read_design_record

    out: dict[str, dict] = {}
    for path in design_paths(run_dir):
        try:
            record = read_design_record(path)
        except (OSError, ValueError) as exc:  # a bad design file is skipped, not fatal
            logger.warning("skipping unreadable design %s: %s", path, exc)
            continue
        out[str(record.get("design_id") or path.stem)] = {
            "source_run_id": record.get("run_id"),
            "formulation": record.get("preset"),
            "heuristic": (record.get("heuristics") or {}).get("name"),
            "selection_strategy": (record.get("selection") or {}).get("strategy"),
            "emissions_mode": (record.get("emissions") or {}).get("mode"),
            "capex_annual_usd": (record.get("objective") or {}).get("capex_annual"),
        }
    return out


def is_evaluation_run(frame: pd.DataFrame, cfg: dict | None = None) -> bool:
    """True only for a *dispatch* run that scored at least one named design.

    Three things are not evaluation runs and must not get an ``eval.parquet``:

    * a ``mode: plan`` run -- its ledger rows carry a ``design_id`` too (the
      design the planner *produced*, named after the task), so a design-id test
      alone fires on every planning run and writes a table whose "operational
      cost" is a planning objective.  P6 prefers ``eval.parquet`` over the
      design record, so that table would silently corrupt the cost figure;
    * a benchmark run, which dispatches ``asbuilt`` and has nothing to compare;
    * an empty ledger.

    ``cfg`` is authoritative when it is available (``cmd_run`` / ``cmd_aggregate``
    both have it).  Without it the ledger's own ``method`` column is used:
    ``tasks.PLAN_METHOD`` marks a planning task.
    """
    if frame is None or frame.empty or "design_id" not in frame.columns:
        return False
    if cfg is not None:
        from .config import is_plan_mode

        if is_plan_mode(cfg):
            return False
    if "method" in frame.columns:
        methods = {str(m) for m in frame["method"].dropna().unique()}
        if tasks_mod.PLAN_METHOD in methods:
            return False
    ids = {str(v) for v in frame["design_id"].dropna().unique()}
    return bool(ids - {"asbuilt", "nan", ""})


def write_eval_tables(run_dir: Path, *, splits_path=None) -> tuple[Path, Path]:
    """``eval.parquet`` + ``eval_ens_profile.parquet`` for a set of evaluated designs.

    One row per (design_id, method, block_size, year, draw): the additive
    metrics summed over blocks, the design's annualised capex, and the
    reliability aggregates.  Rows whose task status is not ``ok`` never enter a
    headline -- they are dropped here and counted in ``eval_excluded.csv``
    (never average an ADMM row that failed its imbalance gate).
    """
    from . import persist

    run_dir = Path(run_dir)
    csv = run_dir / "metrics.csv"
    frame = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    # An explicit path wins; otherwise the run's *recorded* split (env.json)
    # does, falling back to the brain's `data/splits.yaml` for a run that never
    # recorded one (E10: a later edit must not reclassify a finished run).
    holdout = (
        _holdout_years(splits_path)
        if splits_path is not None
        else set(read_splits(run_dir).get("heldout_eval") or [])
    )
    attrs = _design_attributes(run_dir)
    window_hours = _window_hours(run_dir)

    excluded_cols = list(EVAL_EXCLUDED_COLUMNS)
    ok = pd.DataFrame()
    if frame.empty:
        excluded = pd.DataFrame(columns=excluded_cols)
        rows = pd.DataFrame(columns=list(EVAL_COLUMNS))
    else:
        status = frame.get("status", pd.Series(["ok"] * len(frame)))
        bad = frame[status.astype(str) != "ok"]
        excluded = pd.DataFrame(
            {c: bad.get(c, pd.Series([None] * len(bad))) for c in excluded_cols}
        )
        ok = frame[status.astype(str) == "ok"].copy()
        rows = _eval_rows(ok, attrs, holdout, window_hours=window_hours)

    excluded.to_csv(run_dir / "eval_excluded.csv", index=False)
    eval_path = run_dir / "eval.parquet"
    rows.to_parquet(eval_path, index=False)

    profile = _eval_ens_profile(run_dir, persist, ok)
    profile_path = run_dir / "eval_ens_profile.parquet"
    profile.to_parquet(profile_path, index=False)
    logger.info("wrote %s (%d rows) and %s", eval_path, len(rows), profile_path)

    try:
        write_eval_summary(run_dir, rows)
    except Exception as exc:  # a ranking table must never fail a run
        logger.warning("could not write eval_summary: %s", exc, exc_info=True)
    return eval_path, profile_path


def _window_hours(run_dir: Path) -> float:
    """Hours of one weather year's window, from ``config.resolved.yaml``.

    A **case** is one (design, year, draw), so the window it is expected to
    cover is one year's ``[start, stop)`` -- not the multi-year total.  Spec 4.1
    writes ``(stop - start) x n_years``, which is the same number for the
    single-year runs it was written against and would make ``coverage`` equal
    ``1 / n_years`` (and therefore ``total_cost_usd`` NaN) on a 23-year
    evaluation, where every case still covers its own year exactly.
    """
    resolved = Path(run_dir) / "config.resolved.yaml"
    if not resolved.exists():
        return float("nan")
    try:
        import yaml

        payload = yaml.safe_load(resolved.read_text()) or {}
        window = payload["dataset"]["window"]
        return float(int(window["stop"]) - int(window["start"]))
    except Exception:  # noqa: BLE001 - a missing window only costs the coverage guard
        return float("nan")


def _weighted_mean(values, weights) -> float:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def _eval_rows(
    ok: pd.DataFrame, attrs: dict, holdout: set[int], window_hours: float = float("nan")
) -> pd.DataFrame:
    additive = [m for m in metrics_mod.ADDITIVE_METRICS if m in ok.columns]
    keys = ["design_id", "method", "block_size", "year", "draw"]
    for key in keys:
        if key not in ok.columns:
            ok[key] = None
    ok["draw"] = ok["draw"].where(ok["draw"].notna(), -1)

    out = []
    for (design_id, method, block_size, year, draw), group in ok.groupby(keys, dropna=False):
        record = attrs.get(str(design_id), {})
        sums = {m: float(pd.to_numeric(group[m], errors="coerce").sum()) for m in additive}
        hours = float(pd.to_numeric(group.get("hours"), errors="coerce").sum())
        # A design whose record carries no `objective.capex_annual` has an
        # *unknown* capex, not a zero one. Substituting 0 would score it as
        # opex alone and float it to the top of the ranking, ahead of every
        # design that paid for its capacity (verifier F3). A design that
        # genuinely has no investment -- the as-built comparator -- opts in by
        # recording `objective.capex_annual: 0.0` explicitly; see
        # `planning/design.py`.
        capex = record.get("capex_annual_usd")
        capex = float(capex) if capex is not None else float("nan")
        if not np.isfinite(capex):
            logger.warning(
                "design %r has no objective.capex_annual: total_cost_usd is NaN for its "
                "cases and it cannot be ranked. Record an explicit capex (0.0 for an "
                "as-built comparator).",
                design_id,
            )
        operational = sums.get("operational_cost", float("nan"))
        lolh = sums.get("lost_load_hours", float("nan"))
        demand = sums.get("demand_mwh", float("nan"))
        eue = sums.get("unserved_energy_mwh", float("nan"))
        coverage = hours / window_hours if window_hours and np.isfinite(window_hours) else float(
            "nan"
        )
        # E7: annual capex is only a *cost* alongside a whole year of operations.
        # A partial case (a shard still running, a failed block dropped by the
        # `ok` filter) would otherwise report capex + a fraction of the opex as
        # if it were a system cost, which is the wrong way round -- it looks
        # cheap. NaN says "not a total"; `hours` and `coverage` say why.
        full_coverage = np.isfinite(coverage) and abs(coverage - 1.0) <= 1e-9
        row = {
            "design_id": str(design_id),
            "source_run_id": record.get("source_run_id"),
            "formulation": record.get("formulation"),
            "heuristic": record.get("heuristic"),
            "selection_strategy": record.get("selection_strategy"),
            "emissions_mode": record.get("emissions_mode"),
            "method": str(method),
            "block_size": str(block_size),
            "year": int(year) if pd.notna(year) else None,
            "draw": None if draw in (-1, None) else int(draw),
            "n_blocks": len(group),
            "hours": hours,
            "window_hours": float(window_hours),
            "coverage": coverage,
            "is_holdout": bool(pd.notna(year) and int(year) in holdout),
            **sums,
            "capex_annual_usd": capex,
            "total_cost_usd": (
                capex + operational
                if (full_coverage and np.isfinite(capex))
                else float("nan")
            ),
            "eue_mwh": eue,
            "neue": (eue / demand) if (np.isfinite(demand) and demand > 0) else float("nan"),
            "lolh_hours": lolh,
            # Fraction of scored hours with any shortfall (system LOLH / hours);
            # the proposal's LOLP -- P(case has any shortfall) -- is `lol_any`
            # averaged over cases.
            "lolh_frac": (lolh / hours) if hours else float("nan"),
            "lol_any": bool(lolh > 0) if np.isfinite(lolh) else None,
            "storage_cycles": float(
                pd.to_numeric(group.get("storage_cycles"), errors="coerce").sum()
            )
            if "storage_cycles" in group.columns
            else float("nan"),
            "min_available_mw": float(
                pd.to_numeric(group.get("available_mw_min"), errors="coerce").min()
            )
            if "available_mw_min" in group.columns
            else float("nan"),
            # WP-E5 (`output.save_availability`) is not implemented (D6): a
            # percentile cannot be composed from per-block minima, so this stays
            # NaN -- "not measured" -- rather than being approximated.
            "p5_available_mw": float("nan"),
            # Demand-weighted over blocks: each block's `mean_price` is already a
            # demand-weighted mean *within* the block, so an unweighted mean of
            # the blocks would weight a light week like a heavy one (E6).
            "mean_price_usd_per_mwh": _weighted_mean(
                group.get("mean_price", pd.Series(dtype=float)),
                group.get("demand_mwh", pd.Series(dtype=float)),
            )
            if "mean_price" in group.columns
            else float("nan"),
            "max_price_usd_per_mwh": float(
                pd.to_numeric(group.get("max_price"), errors="coerce").max()
            )
            if "max_price" in group.columns
            else float("nan"),
            "build_wall_clock_s": float(
                pd.to_numeric(group.get("build_wall_clock_s"), errors="coerce").sum()
            )
            if "build_wall_clock_s" in group.columns
            else float("nan"),
            "solve_wall_clock_s": float(
                pd.to_numeric(group.get("solve_wall_clock_s"), errors="coerce").sum()
            )
            if "solve_wall_clock_s" in group.columns
            else float("nan"),
            "wall_clock_s": float(
                pd.to_numeric(group.get("wall_clock_s"), errors="coerce").sum()
            )
            if "wall_clock_s" in group.columns
            else float("nan"),
        }
        out.append(row)
    return pd.DataFrame(out, columns=list(EVAL_COLUMNS))


# ---------------------------------------------------------------------------
# eval_summary: the ranking table (WP-E4, spec 4.3)
# ---------------------------------------------------------------------------

EVAL_SUMMARY_COLUMNS = (
    "design_id",
    "source_run_id",
    "formulation",
    "heuristic",
    "selection_strategy",
    "method",
    "block_size",
    "split",
    "n_cases",
    "n_cases_scored",
    "n_years",
    "n_draws",
    "capex_annual_usd",
    "opex_mean_usd",
    "voll_cost_mean_usd",
    "generation_cost_mean_usd",
    "score_usd",
    "score_se_usd",
    "score_p50",
    "score_p90",
    "score_p95",
    "score_max",
    "eue_mwh_mean",
    "eue_p50",
    "eue_p90",
    "eue_p95",
    "eue_max",
    "neue_mean",
    "lolh_mean",
    "lolh_p95",
    "lolp",
    "co2_tonnes_mean",
    "curtailment_mwh_mean",
    "min_available_mw_min",
    "p5_available_mw_mean",
    "rank_score",
)

#: The three splits every design is reported on (spec 4.3 / D3): the headline is
#: `all`, with `holdout` and the train - holdout delta beside it.
EVAL_SPLITS = ("all", "train", "holdout")


def _q(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def _mean(series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def scored_cases(group: pd.DataFrame) -> pd.DataFrame:
    """The cases of ``group`` that survived the coverage / capex guard.

    A case whose ``total_cost_usd`` is NaN is one the run cannot score: it lost
    a block (``coverage < 1``) or its design has no recorded capex.  Such a case
    is **dropped whole** from every statistic, not just from the score --
    averaging its *truncated* ``eue_mwh`` / ``lolh_hours`` / ``lol_any`` into
    the reliability numbers reports the missing block as a block that shed
    nothing, which makes an incomplete run look more reliable than a complete
    one (verifier F2, 2026-09-09).  ``n_cases`` still counts every case, so
    ``n_cases_scored < n_cases`` remains the visible signal.
    """
    if group is None or group.empty or "total_cost_usd" not in group.columns:
        return group if group is not None else pd.DataFrame()
    totals = pd.to_numeric(group["total_cost_usd"], errors="coerce")
    return group[np.isfinite(totals.to_numpy(dtype=float))]


def _summary_row(group: pd.DataFrame, split: str) -> dict:
    """One ranking row: the Score of 3.4 with its Monte-Carlo SE and quantiles.

    ``group`` is every case of one (design, method, block size, split);
    everything except ``n_cases`` is computed over :func:`scored_cases` of it,
    so the score, its uncertainty and every reliability statistic describe
    exactly the same set of realisations.
    """
    full = group
    n_total = len(full)
    group = scored_cases(full)
    capex = _mean(group.get("capex_annual_usd", pd.Series(dtype=float)))
    totals = pd.to_numeric(
        group.get("total_cost_usd", pd.Series(dtype=float)), errors="coerce"
    ).to_numpy(dtype=float)
    totals = totals[np.isfinite(totals)]
    n = totals.size
    eue = pd.to_numeric(group.get("eue_mwh", pd.Series(dtype=float)), errors="coerce").to_numpy(
        dtype=float
    )
    eue = eue[np.isfinite(eue)]
    lolh = pd.to_numeric(
        group.get("lolh_hours", pd.Series(dtype=float)), errors="coerce"
    ).to_numpy(dtype=float)
    lolh = lolh[np.isfinite(lolh)]
    lol_any = group.get("lol_any", pd.Series(dtype=float)).dropna()
    # Identity comes from the *full* group: a design with nothing scored yet
    # still has to appear in the table under its own name.
    first = full.iloc[0] if n_total else {}

    return {
        "design_id": str(first.get("design_id")),
        "source_run_id": first.get("source_run_id"),
        "formulation": first.get("formulation"),
        "heuristic": first.get("heuristic"),
        "selection_strategy": first.get("selection_strategy"),
        "method": str(first.get("method")),
        "block_size": str(first.get("block_size")),
        "split": split,
        # Every case in the split, scored or not.
        "n_cases": n_total,
        # Cases whose `total_cost_usd` survived the coverage / capex guard.
        # *Every* other number in this row -- score, SE, quantiles, EUE, LOLH,
        # LOLP, NEUE, CO2, curtailment -- is over exactly these cases, so
        # `n_cases_scored < n_cases` is the one signal that the headline is
        # incomplete (verifier F2).
        "n_cases_scored": int(n),
        "n_years": int(pd.to_numeric(group.get("year"), errors="coerce").nunique()),
        "n_draws": int(pd.to_numeric(group.get("draw"), errors="coerce").nunique()),
        "capex_annual_usd": capex,
        "opex_mean_usd": _mean(group.get("operational_cost", pd.Series(dtype=float))),
        "voll_cost_mean_usd": _mean(group.get("voll_cost", pd.Series(dtype=float))),
        "generation_cost_mean_usd": _mean(group.get("generation_cost", pd.Series(dtype=float))),
        # Score(eta) = capex + E_case[O(eta)]; O already contains VOLL * ENS
        # (FORMULATIONS 3.4) -- never add voll_cost again.
        "score_usd": float(totals.mean()) if n else float("nan"),
        # The honest uncertainty of a Monte-Carlo criterion, reported next to
        # every score. std over cases / sqrt(n), sample std (ddof=1).
        "score_se_usd": float(totals.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "score_p50": _q(totals, 0.50),
        "score_p90": _q(totals, 0.90),
        "score_p95": _q(totals, 0.95),
        "score_max": float(totals.max()) if n else float("nan"),
        "eue_mwh_mean": float(eue.mean()) if eue.size else float("nan"),
        "eue_p50": _q(eue, 0.50),
        "eue_p90": _q(eue, 0.90),
        "eue_p95": _q(eue, 0.95),
        "eue_max": float(eue.max()) if eue.size else float("nan"),
        "neue_mean": _mean(group.get("neue", pd.Series(dtype=float))),
        "lolh_mean": float(lolh.mean()) if lolh.size else float("nan"),
        "lolh_p95": _q(lolh, 0.95),
        # LOLP is P(a case sheds at all) = mean(lol_any) -- NOT `lolh_frac`,
        # which is the fraction of hours and was reported under this name before
        # 2026-09-09 (FORMULATIONS 6.7).
        "lolp": _mean(lol_any.astype(float)) if len(lol_any) else float("nan"),
        "co2_tonnes_mean": _mean(group.get("co2_tonnes", pd.Series(dtype=float))),
        "curtailment_mwh_mean": _mean(group.get("curtailment_mwh", pd.Series(dtype=float))),
        "min_available_mw_min": float(
            pd.to_numeric(group.get("min_available_mw", pd.Series(dtype=float)), errors="coerce")
            .min()
        )
        if "min_available_mw" in group.columns
        else float("nan"),
        "p5_available_mw_mean": _mean(group.get("p5_available_mw", pd.Series(dtype=float))),
        "rank_score": float("nan"),
    }


def eval_summary(rows: pd.DataFrame, splits: dict | None = None) -> pd.DataFrame:
    """The ranking table of spec 4.3: one row per (design, method, size, split)."""
    if rows is None or rows.empty:
        return pd.DataFrame(columns=list(EVAL_SUMMARY_COLUMNS))

    splits = splits or {}
    train_years = {int(y) for y in (splits.get("train") or [])}
    holdout_years = {int(y) for y in (splits.get("heldout_eval") or [])}
    years = pd.to_numeric(rows.get("year"), errors="coerce")
    is_holdout = (
        rows["is_holdout"].astype(bool)
        if "is_holdout" in rows.columns
        else years.isin(list(holdout_years))
    )
    masks = {
        "all": pd.Series(True, index=rows.index),
        # A year that is in neither list (a dataset year the split file does not
        # mention) belongs to neither split, and is silently in `all` only.
        "train": years.isin(list(train_years)) if train_years else ~is_holdout,
        "holdout": is_holdout,
    }

    out = []
    for _, group in rows.groupby(["design_id", "method", "block_size"], sort=True, dropna=False):
        for split in EVAL_SPLITS:
            subset = group[masks[split].reindex(group.index, fill_value=False)]
            if subset.empty:
                continue
            out.append(_summary_row(subset, split))
    if not out:
        return pd.DataFrame(columns=list(EVAL_SUMMARY_COLUMNS))

    frame = pd.DataFrame(out, columns=list(EVAL_SUMMARY_COLUMNS))
    # Rank over the `all` split (1 = cheapest) and carry that rank onto the
    # design's other splits, so a design has one rank in the table.
    headline = frame[frame["split"] == "all"].copy()
    headline["rank_score"] = (
        headline.groupby(["method", "block_size"])["score_usd"].rank(method="min")
    )
    ranks = {
        (r["design_id"], r["method"], r["block_size"]): r["rank_score"]
        for _, r in headline.iterrows()
    }
    frame["rank_score"] = [
        ranks.get((r.design_id, r.method, r.block_size), float("nan"))
        for r in frame.itertuples(index=False)
    ]
    return frame.sort_values(["method", "block_size", "split", "score_usd"]).reset_index(
        drop=True
    )


def write_eval_summary(run_dir: Path, rows: pd.DataFrame | None = None) -> tuple[Path, Path]:
    """``eval_summary.parquet`` + ``.csv`` beside ``eval.parquet`` (spec 4.3)."""
    run_dir = Path(run_dir)
    if rows is None:
        eval_path = run_dir / "eval.parquet"
        rows = pd.read_parquet(eval_path) if eval_path.exists() else pd.DataFrame()
    summary = eval_summary(rows, read_splits(run_dir))
    parquet = run_dir / "eval_summary.parquet"
    csv = run_dir / "eval_summary.csv"
    summary.to_parquet(parquet, index=False)
    summary.to_csv(csv, index=False)
    logger.info("wrote %s (%d rows) and %s", parquet, len(summary), csv)
    return parquet, csv


def _normalize_case_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce the five case columns so both sides of a join agree on dtypes."""
    frame = frame.copy()
    for col in ("design_id", "method", "block_size"):
        frame[col] = frame[col].astype(str)
    for col in ("year", "draw"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(-1).astype("int64")
    return frame


def case_universe(ok: pd.DataFrame) -> pd.DataFrame:
    """The (design_id, method, block_size, year, draw) cases actually scored.

    One "case" is one weather-year / outage-draw realisation of one design
    under one solver and one block length.  Every reliability average is taken
    over *all* of them -- a case that shed nothing contributes a zero, not a
    missing value -- and ``method`` / ``block_size`` stay in the key because a
    24 h-block dispatch and a 168 h-block dispatch of the same design are
    different answers (LESSONS 2026-09-09: blocking error shows up as ENS).
    """
    if ok.empty:
        return pd.DataFrame(columns=list(CASE_KEYS))
    missing = [c for c in CASE_KEYS if c not in ok.columns]
    if missing:
        return pd.DataFrame(columns=list(CASE_KEYS))
    return _normalize_case_keys(ok[list(CASE_KEYS)]).drop_duplicates().reset_index(drop=True)


def _eval_ens_profile(run_dir: Path, persist, ok: pd.DataFrame) -> pd.DataFrame:
    """R5's input: mean and p95 ENS per (design, day of year, hour of day).

    The average is over the **whole case universe** taken from the ``ok`` rows
    of ``metrics.csv`` -- every (design_id, year, draw) that was scored -- with
    a zero for each case that had no shortfall in that hour.  Averaging only
    over the cases that appear in ``ens_profile/`` would report a conditional
    mean ("how bad when it happens") under the name of an expected value, and
    would make a more reliable design look worse.

    ``method`` and ``block_size`` stay in the key: summing a 24 h-block score
    and a 168 h-block score of the same design into one hour would report ENS
    the system never had.

    ``day_of_year`` / ``hour_of_day`` are derived from the absolute in-year
    hour index (``hour // 24 + 1`` and ``hour % 24``), which is a **UTC** hour
    for the CA2040 exports.  The shipped benchmark window starts at hour 7 so
    that blocks begin at Pacific midnight, so ``hour_of_day`` is 7 hours ahead
    of local time; do not read it as a local clock hour.
    """
    empty = pd.DataFrame(columns=list(EVAL_ENS_PROFILE_COLUMNS))
    if not persist.has_artefact(run_dir, "ens_profile"):
        return empty
    frame = persist.read_combined(run_dir, "ens_profile")
    if frame.empty:
        return empty

    import numpy as np

    frame = _normalize_case_keys(frame)

    cases = case_universe(ok)
    series_keys = ["design_id", "method", "block_size"]
    n_cases = (
        cases.groupby(series_keys).size().to_dict() if not cases.empty else {}
    )

    per_case = (
        frame.groupby(
            [*series_keys, "day_of_year", "hour_of_day", "year", "draw"], dropna=False
        )["ens_mwh"]
        .sum()
        .reset_index()
    )

    rows = []
    for (design_id, method, block_size, day, hour), group in per_case.groupby(
        [*series_keys, "day_of_year", "hour_of_day"], dropna=False
    ):
        values = group["ens_mwh"].to_numpy(dtype=float)
        total = int(n_cases.get((design_id, method, block_size), len(values)))
        total = max(total, len(values))
        padded = np.concatenate([values, np.zeros(total - len(values))])
        rows.append(
            {
                "design_id": str(design_id),
                "method": str(method),
                "block_size": str(block_size),
                "day_of_year": int(day),
                "hour_of_day": int(hour),
                "mean_ens_mwh": float(padded.mean()),
                "p95_ens_mwh": float(np.quantile(padded, 0.95)),
                "n_draws": total,
                "n_year_draws_with_ens": int((values > 0).sum()),
            }
        )
    if not rows:
        return empty
    return pd.DataFrame(rows, columns=list(EVAL_ENS_PROFILE_COLUMNS))
