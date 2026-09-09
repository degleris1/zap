"""The evaluation-pipeline seam (thin in phase 1).

Evaluating a set of candidate designs is *the same task machinery* with a
``Design`` attached: one task per (design, block, method, draw), the same ledger,
the same aggregation.  Phase 1 evaluates only the as-built design, which is why
the operational benchmark is written as an evaluation of ``asbuilt``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import metrics as metrics_mod
from . import tasks as tasks_mod
from .system import Design

logger = logging.getLogger(__name__)


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
    "capex_annual_usd",
    "total_cost_usd",
    "eue_mwh",
    "lolh_hours",
    "lolh_frac",
    "lol_any",
    "min_available_mw",
    "p5_available_mw",
    "mean_price_usd_per_mwh",
    "max_price_usd_per_mwh",
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

EVAL_EXCLUDED_COLUMNS = ("task_id", "status", "error")


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
    holdout = _holdout_years(splits_path)
    attrs = _design_attributes(run_dir)

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
        rows = _eval_rows(ok, attrs, holdout)

    excluded.to_csv(run_dir / "eval_excluded.csv", index=False)
    eval_path = run_dir / "eval.parquet"
    rows.to_parquet(eval_path, index=False)

    profile = _eval_ens_profile(run_dir, persist, ok)
    profile_path = run_dir / "eval_ens_profile.parquet"
    profile.to_parquet(profile_path, index=False)
    logger.info("wrote %s (%d rows) and %s", eval_path, len(rows), profile_path)
    return eval_path, profile_path


def _eval_rows(ok: pd.DataFrame, attrs: dict, holdout: set[int]) -> pd.DataFrame:
    import numpy as np

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
        capex = record.get("capex_annual_usd")
        capex = float(capex) if capex is not None else float("nan")
        operational = sums.get("operational_cost", float("nan"))
        lolh = sums.get("lost_load_hours", float("nan"))
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
            "is_holdout": bool(pd.notna(year) and int(year) in holdout),
            **sums,
            "capex_annual_usd": capex,
            "total_cost_usd": (capex if np.isfinite(capex) else 0.0) + operational,
            "eue_mwh": sums.get("unserved_energy_mwh", float("nan")),
            "lolh_hours": lolh,
            # Fraction of scored hours with any shortfall (system LOLH / hours);
            # the proposal's LOLP -- P(case has any shortfall) -- is `lol_any`
            # averaged over cases.
            "lolh_frac": (lolh / hours) if hours else float("nan"),
            "lol_any": bool(lolh > 0) if np.isfinite(lolh) else None,
            "min_available_mw": float(
                pd.to_numeric(group.get("available_mw_min"), errors="coerce").min()
            )
            if "available_mw_min" in group.columns
            else float("nan"),
            # Only computable from hourly data; NaN says "not measured".
            "p5_available_mw": float("nan"),
            "mean_price_usd_per_mwh": float(
                pd.to_numeric(group.get("mean_price"), errors="coerce").mean()
            )
            if "mean_price" in group.columns
            else float("nan"),
            "max_price_usd_per_mwh": float(
                pd.to_numeric(group.get("max_price"), errors="coerce").max()
            )
            if "max_price" in group.columns
            else float("nan"),
        }
        out.append(row)
    return pd.DataFrame(out, columns=list(EVAL_COLUMNS))


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
