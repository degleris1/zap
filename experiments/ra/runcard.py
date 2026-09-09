"""The human-readable run card, written from ``metrics.csv`` and the config."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .blocks import reference_bounds
from .config import is_plan_mode
from .identity import run_id, zap_commit, zap_dirty
from .metrics import ADDITIVE_METRICS, PLANNING_METRICS, deviation_vs_reference

#: Modelling choices that phase 1 deliberately does not exercise, stated on the
#: card so the chapter cannot accidentally claim them.
NOT_EXERCISED = (
    "annual energy-budget pro-rating (no `e_sum` limits exist in ca2040_z4; the hook is a no-op)",
    "unit commitment / minimum stable levels (`p_min_pu` is ignored)",
    "thermal outage draws (weather-driven availability only unless `heuristics.outage_draws` is set)",
    "capacity expansion (operations only: `p_nom` is fixed)",
)

SUMMARY_METRICS = (
    ADDITIVE_METRICS
    + (
        "storage_cycles",
        "storage_cycles_per_day",
        "mean_price",
        "max_price",
        "admm_max_imbalance_mw",
    )
    + PLANNING_METRICS
)

#: What a ``mode: plan`` run does *not* exercise, stated on the card.
NOT_EXERCISED_PLAN = (
    (
        "non-uniform subproblem weights (D-W8: every block is weighted 1.0 and "
        "`snapshot_weight` is left at 1.0)"
    ),
    "k-medoids / gradient-stress period selection (registered, phase-2, raises on use)",
    "ELCC + PRM accreditation (`heuristics.name: elcc_prm` is rejected at config time)",
    "PyPSA export of the design (D-W6: the artefact is `design.json`)",
)


def cpu_count() -> tuple[int, str]:
    """Cores to charge CPU-seconds against, and where that number came from."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return int(slurm), "SLURM_CPUS_PER_TASK"
        except ValueError:  # pragma: no cover - defensive
            pass
    return (os.cpu_count() or 1), "os.cpu_count()"


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if pd.isna(value):
            return "-"
        return f"{value:,.4g}"
    return str(value)


def _table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_(no rows)_\n"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(_fmt(v) for v in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows]) + "\n"


def summarize(df: pd.DataFrame, bounds: tuple[int, int] | None) -> pd.DataFrame:
    """One row per (method, block size): metrics over the window and the reference window."""
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []
    for (method, block_size), group in df.groupby(["method", "block_size"], sort=True):
        ok = group[group["status"] == "ok"]
        row = {
            "method": method,
            "block_size": block_size,
            "n_tasks": len(group),
            "n_ok": len(ok),
            "n_failed": int((group["status"] == "failed").sum()),
            "n_timeout": int((group["status"] == "timeout").sum()),
            "n_infeasible": int((group["status"] == "infeasible").sum()),
            "hours_covered": int(pd.to_numeric(ok.get("hours"), errors="coerce").sum())
            if "hours" in ok.columns
            else 0,
            "wall_clock_s": float(pd.to_numeric(group["wall_clock_s"], errors="coerce").sum()),
            "solve_wall_clock_s": float(
                pd.to_numeric(group.get("solve_wall_clock_s"), errors="coerce").sum()
            )
            if "solve_wall_clock_s" in group.columns
            else float("nan"),
        }
        for metric in SUMMARY_METRICS:
            if metric not in ok.columns:
                continue
            values = pd.to_numeric(ok[metric], errors="coerce")
            row[metric] = (
                float(values.sum()) if metric in ADDITIVE_METRICS else float(values.mean())
            )
        if bounds is not None and not ok.empty and "start" in ok.columns:
            inside = ok[(ok["start"] >= bounds[0]) & (ok["stop"] <= bounds[1])]
            for metric in ADDITIVE_METRICS:
                if metric in inside.columns:
                    row[f"ref_window_{metric}"] = float(
                        pd.to_numeric(inside[metric], errors="coerce").sum()
                    )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Planning-run sections (``mode: plan``)
# ---------------------------------------------------------------------------


def read_design_records(run_dir: Path) -> list[dict]:
    """Every ``designs/*.json`` of a run, oldest id first (best effort)."""
    try:
        from .planning.design import design_paths, read_design_record
    except Exception:  # noqa: BLE001 - the card must never fail on an import
        return []
    records = []
    for path in design_paths(run_dir):
        try:
            records.append(read_design_record(path))
        except (OSError, ValueError):  # pragma: no cover - defensive
            continue
    return records


def _carrier_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Designed capacity by carrier, one column per planning task."""
    if df is None or df.empty or "capacity_mw_by_carrier" not in df.columns:
        return pd.DataFrame()
    columns: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        blob = row.get("capacity_mw_by_carrier")
        if not isinstance(blob, str) or not blob:
            continue
        try:
            columns[str(row.get("design_id", row.get("task_id")))] = json.loads(blob)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            continue
    if not columns:
        return pd.DataFrame()
    carriers = sorted({c for col in columns.values() for c in col})
    rows = [
        {"carrier": carrier, **{name: col.get(carrier, 0.0) for name, col in columns.items()}}
        for carrier in carriers
    ]
    rows.append(
        {"carrier": "**total**", **{name: sum(col.values()) for name, col in columns.items()}}
    )
    return pd.DataFrame(rows)


def planning_sections(run_dir: Path, cfg: dict, df: pd.DataFrame | None) -> list[str]:
    """The ``mode: plan`` half of the card (spec sections 3.4, 4, 6 and 9.5)."""
    plan = cfg["planning"]
    sel = cfg["selection"]
    records = read_design_records(run_dir)
    lines: list[str] = []

    lines.append("## Planning method\n")
    lines.append(f"- **Preset:** `{plan['method']}` (`planning.method`)")
    lines.append(f"- **Single-level kind:** `{plan['single_level']['kind']}`")
    lines.append(
        f"- **Solvers:** single-level `{plan['single_level']['solver']}`, "
        f"dispatch `{plan['dispatch_solver']}`"
    )
    lines.append(
        f"- **Workers:** {plan['num_workers']} — **timeout:** {plan['timeout_s']:,.0f} s "
        "(enforced: the solve runs in a worker process and is killed at the cap)"
    )
    lines.append(f"- **Expansion bounds:** `{plan['expansion']['mode']}`")
    lines.append(f"- **Budget constraints:** `{plan['budget_constraints']}`")
    lines.append(
        f"- **Heuristic:** `{cfg['heuristics']['name']}` "
        f"(ucap_derate={cfg['heuristics']['ucap_derate']}, "
        f"outage_draws={cfg['heuristics']['outage_draws']})"
    )
    lines.append("")

    lines.append("## Period selection\n")
    lines.append(f"- **Strategy:** `{sel['strategy']}` — **seed:** {sel['seed']}")
    lines.append(
        f"- **Block size:** {sel['block_size']} h "
        f"({'the whole loaded horizon' if sel['block_size'] is None else 'blocked'})"
    )
    lines.append(f"- **num_blocks:** {sel['num_blocks']}")
    lines.append(f"- **avoid_year_boundaries:** {sel['avoid_year_boundaries']}")
    lines.append("")

    lines.append("## Annualization\n")
    lines.append(
        "Each block's devices are pro-rated by `sample_time`, so "
        "`objective_raw = coverage * CAPEX + OPEX(sampled hours)`. **Every monetary "
        "number below is annual** — `raw * annualization_factor` (spec section 6, W-A). "
        "`year_factor` scales a partial-year window to a full year and is **not** "
        "applied anywhere: it is printed so a 48-hour objective is not mistaken for "
        "an annual system cost."
    )
    lines.append("")
    if records:
        rows = []
        for record in records:
            ann = record.get("annualization", {})
            rows.append(
                {
                    "design_id": record.get("design_id"),
                    "total_hours": ann.get("total_hours"),
                    "sampled_hours": ann.get("sampled_hours"),
                    "coverage": ann.get("coverage"),
                    "annualization_factor": ann.get("annualization_factor"),
                    "year_factor": ann.get("year_factor"),
                    "n_blocks": len(record.get("selection", {}).get("blocks") or []),
                    "weights": "uniform",
                    "snapshot_weight": ann.get("snapshot_weight", 1.0),
                    "capital_cost_prorated": ann.get("capital_cost_prorated", True),
                }
            )
        lines.append(_table(pd.DataFrame(rows)))
    else:
        lines.append("_(no design.json written)_\n")
    lines.append("")

    lines.append("## Objective\n")
    lines.append(
        "The objective is **system cost**: capex + opex, net of any carbon payment "
        "(PROJECT.md 2.4). Under `emissions.mode: price` or `dual_ascent` the solver "
        "minimizes `system cost + price * emissions`; that payment is a transfer, not a "
        "resource cost, so it is subtracted from `opex` and from the objective and "
        "reported on its own as `carbon_payment_annual`. `lower_bound_raw` is the "
        "warm-start LP optimum, and is only reported when the warm start ran on the "
        "*same* block set as the method it bounds (otherwise its capital cost is "
        "pro-rated to a different coverage and it is not a bound at all)."
    )
    lines.append("")
    if records:
        rows = []
        for record in records:
            obj = record.get("objective", {})
            rows.append(
                {
                    "design_id": record.get("design_id"),
                    "objective_annual": obj.get("annual"),
                    "capex_annual": obj.get("capex_annual"),
                    "opex_annual": obj.get("opex_annual"),
                    "carbon_payment_annual": obj.get("carbon_payment_annual"),
                    "emissions_t_annual": obj.get("emissions_tonnes_annual"),
                    "objective_raw": obj.get("raw"),
                    "lower_bound_raw": obj.get("lower_bound_raw"),
                    "optimality_gap": obj.get("optimality_gap"),
                    "status": record.get("solver", {}).get("status"),
                }
            )
        lines.append(_table(pd.DataFrame(rows)))
    else:
        lines.append("_(no design.json written)_\n")
    lines.append("")

    lines.append("## Emissions\n")
    if records:
        emissions = records[0].get("emissions", {})
        for key in ("mode", "price", "cap", "cap_basis", "cap_applied"):
            lines.append(f"- **{key}:** {_fmt(emissions.get(key))}")
        payment = records[0].get("objective", {}).get("carbon_payment_annual")
        lines.append(f"- **carbon_payment_annual:** {_fmt(payment)} (netted out of the objective)")
    else:
        lines.append(f"- **mode:** {plan['emissions']['mode']}")
    lines.append("")

    lines.append("## Capacity bounds and floors\n")
    lines.append(
        "The capacity floors are a modelling choice, not a numerical guard "
        "(spec section 9.5): they forbid retirement below the floor on every row they "
        "raise. `rows_raised_by_floor` counts the rows whose lower bound the floor "
        "actually moved."
    )
    lines.append("")
    bounds = plan["bounds"]
    lines.append(f"- **min_capacity_mw:** {bounds['min_capacity_mw']}")
    lines.append(f"- **min_storage_mw:** {bounds['min_storage_mw']}")
    floors = None
    if df is not None and not df.empty and "rows_raised_by_floor" in df.columns:
        values = df["rows_raised_by_floor"].dropna()
        if not values.empty:
            floors = values.iloc[0]
    lines.append(f"- **rows_raised_by_floor:** {floors if floors is not None else '-'}")
    lines.append("")

    lines.append("## Designed capacity by carrier (MW)\n")
    carriers = _carrier_frame(df)
    lines.append(_table(carriers) if not carriers.empty else "_(no capacities recorded)_\n")
    lines.append("")

    lines.append("## Designs\n")
    if records:
        for record in records:
            lines.append(
                f"- `designs/{record.get('design_id')}.json` — schema "
                f"v{record.get('schema_version')}, method `{record.get('method')}` "
                f"(preset `{record.get('preset')}`)"
            )
    else:
        lines.append("_(none)_")
    lines.append("")
    return lines


def admm_gap_vs_lp(df: pd.DataFrame) -> pd.DataFrame:
    """Per-block-size ADMM cost gap against the LP solved on the *same* blocks.

    This is the quantity the operational benchmark actually wants, and the run
    already contains both solves, so it costs nothing (benchmark review 4.3).
    Only blocks solved successfully by both methods are compared.
    """
    empty = pd.DataFrame(
        columns=[
            "block_size",
            "n_blocks",
            "lp_cost",
            "admm_cost",
            "gap_rel",
            "worst_block_gap_rel",
            "max_imbalance_mw",
            "admm_solve_s_per_block",
            "lp_solve_s_per_block",
        ]
    )
    if df is None or df.empty or "method" not in df.columns:
        return empty
    ok = df[df.get("status", "ok") == "ok"]
    if "operational_cost" not in ok.columns:
        return empty

    # Blocks are identified by (year, index, draw) -- the same block seen by both
    # methods -- not by task_id, which carries the method name.
    key_cols = [c for c in ("year", "block_index", "draw", "design_id") if c in ok.columns]
    if not key_cols:
        return empty
    ok = ok.copy()
    ok["_block_key"] = [
        "|".join(str(v) for v in row)
        for row in ok[key_cols].astype(str).itertuples(index=False, name=None)
    ]

    rows = []
    for block_size, group in ok.groupby("block_size", sort=True):
        lp = group[group["method"] == "lp"].set_index("_block_key")
        admm = group[group["method"] == "admm"].set_index("_block_key")
        shared = lp.index.intersection(admm.index)
        if len(shared) == 0:
            continue
        lp_cost = pd.to_numeric(lp.loc[shared, "operational_cost"], errors="coerce")
        admm_cost = pd.to_numeric(admm.loc[shared, "operational_cost"], errors="coerce")
        per_block = (admm_cost - lp_cost) / lp_cost
        rows.append(
            {
                "block_size": block_size,
                "n_blocks": len(shared),
                "lp_cost": float(lp_cost.sum()),
                "admm_cost": float(admm_cost.sum()),
                "gap_rel": float(admm_cost.sum() / lp_cost.sum() - 1.0),
                "worst_block_gap_rel": float(per_block.abs().max()),
                "max_imbalance_mw": float(
                    pd.to_numeric(
                        admm.loc[shared].get("admm_max_imbalance_mw"), errors="coerce"
                    ).max()
                )
                if "admm_max_imbalance_mw" in admm.columns
                else float("nan"),
                "admm_solve_s_per_block": float(
                    pd.to_numeric(admm.loc[shared, "solve_wall_clock_s"], errors="coerce").mean()
                ),
                "lp_solve_s_per_block": float(
                    pd.to_numeric(lp.loc[shared, "solve_wall_clock_s"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows) if rows else empty


def write_card(
    run_dir: Path,
    cfg: dict,
    df: pd.DataFrame | None = None,
    *,
    system_meta: dict | None = None,
) -> Path:
    """Write ``CARD.md``; returns its path."""
    run_dir = Path(run_dir)
    if df is None:
        metrics_csv = run_dir / "metrics.csv"
        df = pd.read_csv(metrics_csv) if metrics_csv.exists() else pd.DataFrame()

    if system_meta is None:
        meta_path = run_dir / "system_meta.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                system_meta = json.load(f)
    system_meta = system_meta or {}

    plan_mode = is_plan_mode(cfg)
    bounds = None if plan_mode else reference_bounds(cfg)
    summary = summarize(df, bounds)
    deviations = None
    deviation_error = None
    if not plan_mode:
        try:
            deviations = deviation_vs_reference(df, bounds)
        except ValueError as exc:  # a misaligned or incomplete comparison; say so
            deviation_error = str(exc)

    lines: list[str] = []
    lines.append(f"# Run card — {run_id(cfg)}\n")
    lines.append(f"- **Run id:** `{run_id(cfg)}`")
    lines.append(f"- **Name:** {cfg['name']}")
    lines.append(f"- **zap commit:** `{zap_commit()}`{' (dirty)' if zap_dirty() else ''}")
    lines.append(f"- **Resolved config:** `{(run_dir / 'config.resolved.yaml')}`")
    lines.append(f"- **Written:** {datetime.now(timezone.utc).isoformat()}")
    cores, cores_source = cpu_count()
    lines.append(f"- **Cores charged:** {cores} (from `{cores_source}`)")
    lines.append("")

    lines.append("## Dataset\n")
    lines.append(f"- **Dataset:** `{cfg['dataset']['dir']}`")
    lines.append(f"- **Weather years:** {cfg['dataset']['years']}")
    lines.append(f"- **Window (hours):** {cfg['dataset']['window']}")
    lines.append(f"- **Mode:** `{cfg['mode']}`")
    if plan_mode:
        lines.append(
            "- **Reference solve:** _not applicable in `mode: plan`_ "
            f"(`selection.reference: {cfg['selection']['reference']}` is a dispatch-mode key)"
        )
    else:
        lines.append(f"- **Reference solve:** {cfg['selection']['reference']} -> {bounds}")
    store_attrs = system_meta.get("weather_store_attrs")
    if store_attrs:
        lines.append("- **Weather store provenance:**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(store_attrs, indent=2, sort_keys=True, default=str))
        lines.append("```")
    else:
        lines.append("- **Weather store provenance:** _not recorded in this process_")
    lines.append("")

    lines.append("## Demand scaling\n")
    if system_meta:
        for key in (
            "peak_load_mw",
            "peak_available_mw",
            "implied_scale",
            "applied_scale",
            "voll",
        ):
            lines.append(f"- **{key}:** {_fmt(system_meta.get(key))}")
        applied = system_meta.get("applied_scale")
        if applied is not None and abs(float(applied) - 1.0) < 1e-12:
            lines.append(
                "- Demand scaling was **clipped to 1.0**: peak load is already below "
                "`peak_capacity_fraction * peak_available_mw`, so no scaling was applied."
            )
    else:
        lines.append(
            f"- Requested: `{cfg['system']['demand_scaling']}` with "
            f"`peak_capacity_fraction = {cfg['system']['peak_capacity_fraction']}`, "
            f"`clip_scale_to_one = {cfg['system']['clip_scale_to_one']}`."
        )
        lines.append("- _Realized scaling not recorded: no system was built in this process._")
    lines.append("")

    lines.append("## Tasks\n")
    if df is None or df.empty:
        lines.append("_(no task results)_\n")
    else:
        counts = df["status"].value_counts().to_dict()
        lines.append(
            f"- **Total:** {len(df)} — " + ", ".join(f"{k}: {v}" for k, v in counts.items())
        )
        wall = float(pd.to_numeric(df["wall_clock_s"], errors="coerce").sum())
        solve_wall = (
            float(pd.to_numeric(df["solve_wall_clock_s"], errors="coerce").sum())
            if "solve_wall_clock_s" in df.columns
            else float("nan")
        )
        lines.append(f"- **Wall clock, whole tasks (incl. building the system):** {wall:,.1f} s")
        lines.append(f"- **Wall clock, solver only:** {solve_wall:,.1f} s")
        lines.append(
            f"- **CPU-seconds (solver wall x {cores} cores from `{cores_source}`):** "
            f"{solve_wall * cores:,.1f}"
        )
        failed = df[df["status"] != "ok"]
        if not failed.empty:
            lines.append("")
            lines.append("### Tasks that did not succeed\n")
            cols = [
                c for c in ("task_id", "method", "block_size", "status", "error") if c in failed
            ]
            lines.append(_table(failed[cols]))
    lines.append("")

    if plan_mode:
        lines.extend(planning_sections(run_dir, cfg, df))

    lines.append("## Metrics by (method, block size)\n")
    lines.append(_table(summary))
    lines.append("")

    if plan_mode:
        lines.append("## Not exercised\n")
        for item in NOT_EXERCISED_PLAN:
            lines.append(f"- {item}")
        lines.append("")
        path = run_dir / "CARD.md"
        path.write_text("\n".join(lines))
        return path

    admm_gap = admm_gap_vs_lp(df)
    if not admm_gap.empty:
        lines.append("## ADMM vs LP, block by block\n")
        lines.append(
            "The ADMM number is the cost of a *dispatch heuristic* evaluated at the "
            "solver's iterate: it neither upper- nor lower-bounds the LP. Read it "
            "next to `max_imbalance_mw` -- an iterate that misses nodal power "
            "balance is not a dispatch at all, and rows above "
            "`methods.admm.max_imbalance_mw` are recorded `infeasible` and excluded."
        )
        lines.append("")
        lines.append(_table(admm_gap))
        lines.append("")

    lines.append("## Blocking error vs the reference solve\n")
    lines.append(
        "Sums are restricted to blocks lying entirely inside the reference window; "
        "blocks straddling the boundary are excluded and counted. "
        "`ref_window_hours_covered` must equal `ref_window_hours` for the comparison "
        "to be like-for-like."
    )
    lines.append("")
    if deviation_error is not None:
        lines.append(f"**Not computed:** {deviation_error}")
    else:
        lines.append(_table(deviations))
    lines.append("")

    lines.append("## Not exercised\n")
    for item in NOT_EXERCISED:
        lines.append(f"- {item}")
    lines.append("")

    path = run_dir / "CARD.md"
    path.write_text("\n".join(lines))
    return path
