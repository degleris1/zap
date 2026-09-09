"""The human-readable run card, written from ``metrics.csv`` and the config."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .blocks import reference_bounds
from .identity import run_id, zap_commit, zap_dirty
from .metrics import ADDITIVE_METRICS, deviation_vs_reference

#: Modelling choices that phase 1 deliberately does not exercise, stated on the
#: card so the chapter cannot accidentally claim them.
NOT_EXERCISED = (
    "annual energy-budget pro-rating (no `e_sum` limits exist in ca2040_z4; the hook is a no-op)",
    "unit commitment / minimum stable levels (`p_min_pu` is ignored)",
    "thermal outage draws (weather-driven availability only unless `heuristics.outage_draws` is set)",
    "capacity expansion (operations only: `p_nom` is fixed)",
)

SUMMARY_METRICS = ADDITIVE_METRICS + ("storage_cycles", "mean_price", "max_price")


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

    bounds = reference_bounds(cfg)
    summary = summarize(df, bounds)
    deviation_error = None
    try:
        deviations = deviation_vs_reference(df, bounds)
    except ValueError as exc:  # a misaligned or incomplete comparison; say so
        deviations = None
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

    lines.append("## Metrics by (method, block size)\n")
    lines.append(_table(summary))
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
