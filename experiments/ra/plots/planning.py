"""Planning plots P1-P9.

Phase A implements P1 (capacity by carrier), P2 (the design matrix) and P6 (the
cost decomposition).  P3, P4, P5, P7, P8 and P9 are registered with their
schema and raise ``NotImplementedError`` (D8).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import phase_b, register, style

logger = logging.getLogger(__name__)

#: Power-capacity parameters, in MW; storage also gets an energy row in MWh.
_POWER_ATTRS = ("nominal_capacity", "power_capacity")


def _columns(plot_id: str) -> tuple[str, ...]:
    from . import PLOTS

    return PLOTS[plot_id].columns


def _param_for_class(record: dict, cls_name: str) -> str | None:
    """``Generator -> generator_capacity`` (the runner's naming convention)."""
    for param in record.get("parameter_names") or {}:
        if str(param).rsplit("_", 1)[0] == str(cls_name).lower():
            return param
    return None


def design_capacity_table(runs) -> pd.DataFrame:
    """One row per (run, design, device class, carrier, unit).

    ``design.json`` records row *names* but not carriers, so the carrier of each
    row comes from the dataset's static tables via
    ``RunHandle.row_carriers()``.  When the dataset is not reachable the carrier
    degrades to ``"unknown"`` rather than dropping the row.
    """
    rows = []
    for run in runs:
        carriers = run.row_carriers()
        durations = run.row_durations()
        for record in run.designs():
            initial = record.get("initial_parameters") or {}
            for cls_name, entry in (record.get("capacities") or {}).items():
                attr = next((k for k in entry if k in _POWER_ATTRS), None)
                if attr is None:
                    continue
                names = [str(n) for n in (entry.get("names") or [])]
                designed = np.asarray(entry.get(attr) or [], dtype=float)
                param = _param_for_class(record, cls_name)
                as_built = np.asarray(initial.get(param) or [], dtype=float)
                if as_built.size != designed.size:
                    as_built = np.full(designed.shape, np.nan)
                if len(names) != designed.size:
                    names = [f"{cls_name}_{i}" for i in range(designed.size)]
                by_class = carriers.get(cls_name, {})
                for i, name in enumerate(names):
                    carrier = str(by_class.get(name, "unknown"))
                    rows.append(
                        {
                            "run_id": run.run_id,
                            "label": run.label,
                            "design_id": str(record.get("design_id")),
                            "formulation": record.get("preset"),
                            "heuristic": (record.get("heuristics") or {}).get("name"),
                            "selection_strategy": (record.get("selection") or {}).get("strategy"),
                            "emissions_mode": (record.get("emissions") or {}).get("mode"),
                            "device_class": str(cls_name),
                            "carrier": carrier,
                            "unit": style.unit_label("power"),
                            "as_built": style.convert(float(as_built[i]), "power"),
                            "designed": style.convert(float(designed[i]), "power"),
                        }
                    )
                    duration = durations.get(name)
                    if cls_name == "StorageUnit" and duration:
                        rows.append(
                            {
                                "run_id": run.run_id,
                                "label": run.label,
                                "design_id": str(record.get("design_id")),
                                "formulation": record.get("preset"),
                                "heuristic": (record.get("heuristics") or {}).get("name"),
                                "selection_strategy": (record.get("selection") or {}).get(
                                    "strategy"
                                ),
                                "emissions_mode": (record.get("emissions") or {}).get("mode"),
                                "device_class": str(cls_name),
                                "carrier": carrier,
                                "unit": style.unit_label("energy"),
                                "as_built": style.convert(
                                    float(as_built[i]) * float(duration), "energy"
                                ),
                                "designed": style.convert(
                                    float(designed[i]) * float(duration), "energy"
                                ),
                            }
                        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no design capacities: designs/*.json carry no capacity block")
    frame["delta"] = frame["designed"] - frame["as_built"]
    return frame


# ---------------------------------------------------------------------------
# P1 -- capacity by carrier
# ---------------------------------------------------------------------------


@register(
    "P1",
    title="Designed capacity by carrier",
    tier="report",
    needs=("designs",),
    columns=(
        "run_id",
        "label",
        "design_id",
        "device_class",
        "carrier",
        "unit",
        "as_built",
        "designed",
        "delta",
    ),
)
def p1_capacity_by_carrier(runs, **_):
    """As-built vs designed capacity per carrier, in GW and GWh.

    ``delta`` is ``designed - as_built``: negative means the planner retired
    capacity, which it can only do when the row's lower bound is below its
    as-built value.
    """
    detail = design_capacity_table(runs)
    keys = ["run_id", "label", "design_id", "device_class", "carrier", "unit"]
    table = detail.groupby(keys, as_index=False)[["as_built", "designed", "delta"]].sum()
    table = table.sort_values(keys).reset_index(drop=True)

    units = [u for u in (style.unit_label("power"), style.unit_label("energy"))
             if u in set(table["unit"])]
    fig, axes = plt.subplots(len(units), 1, figsize=(9.5, 3.4 * len(units)), squeeze=False)
    for ax, unit in zip(axes[:, 0], units):
        # Bars are drawn per *display* carrier (batteries merged into BESS) and
        # ordered by the global stacking order; the returned table stays per
        # carrier, so `raw_data` keeps every battery type separately.
        sub = table[table["unit"] == unit]
        sub = sub.assign(carrier=style.display_carriers(sub["carrier"]))
        order = style.stack_order(sub["carrier"].unique())
        pivot_built = sub.pivot_table(index="carrier", columns="label", values="as_built",
                                      aggfunc="sum").fillna(0.0).reindex(order)
        pivot_new = sub.pivot_table(index="carrier", columns="label", values="delta",
                                    aggfunc="sum").fillna(0.0).reindex(order)
        positions = np.arange(len(pivot_built.index))
        width = 0.8 / max(1, len(pivot_built.columns))
        for j, label in enumerate(pivot_built.columns):
            offset = positions + j * width - 0.4 + width / 2
            colors = [style.carrier_color(c) for c in pivot_built.index]
            ax.bar(offset, pivot_built[label], width=width, color=colors, label=f"{label} as-built")
            ax.bar(offset, pivot_new[label].clip(lower=0), width=width,
                   bottom=pivot_built[label], color=colors, hatch="//", edgecolor="white",
                   linewidth=0.3, label=f"{label} new build")
            retired = pivot_new[label].clip(upper=0)
            ax.bar(offset, retired, width=width, color="0.6", label=f"{label} retired")
        ax.set_xticks(positions)
        ax.set_xticklabels(pivot_built.index, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(unit)
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        ax.legend(seen.values(), seen.keys(), fontsize=6, ncol=3)
    fig.suptitle("P1 - designed capacity by carrier")
    fig.tight_layout()
    return fig, table[list(_columns("P1"))]


# ---------------------------------------------------------------------------
# P2 -- the design matrix
# ---------------------------------------------------------------------------


@register(
    "P2",
    title="Design matrix: formulation x carrier",
    tier="report",
    needs=("designs",),
    columns=(
        "run_id",
        "label",
        "design_id",
        "formulation",
        "heuristic",
        "selection_strategy",
        "emissions_mode",
        "carrier",
        "unit",
        "designed",
        "delta",
    ),
)
def p2_design_matrix(runs, **_):
    """Designed capacity per carrier for every (formulation, heuristic, selection).

    One row of the heat map per run; the columns are carriers.  This is the
    figure the chapter's comparison matrix (PROJECT.md 2.2) is read off.
    """
    detail = design_capacity_table(runs)
    keys = [
        "run_id",
        "label",
        "design_id",
        "formulation",
        "heuristic",
        "selection_strategy",
        "emissions_mode",
        "carrier",
        "unit",
    ]
    table = detail.groupby(keys, as_index=False, dropna=False)[["designed", "delta"]].sum()
    table = table.sort_values(keys).reset_index(drop=True)

    power = table[table["unit"] == style.unit_label("power")]
    power = power.assign(carrier=style.display_carriers(power["carrier"]))
    pivot = power.pivot_table(index="label", columns="carrier", values="designed", aggfunc="sum")
    pivot = pivot.fillna(0.0)
    pivot = pivot[style.stack_order(pivot.columns)]
    fig, ax = plt.subplots(figsize=(1.1 * max(4, len(pivot.columns)) + 2,
                                    0.6 * max(2, len(pivot.index)) + 2))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label=f"designed [{style.unit_label('power')}]")
    ax.set_title("P2 - design matrix")
    fig.tight_layout()
    return fig, table[list(_columns("P2"))]


# ---------------------------------------------------------------------------
# P6 -- cost decomposition
# ---------------------------------------------------------------------------

P6_COMPONENTS = ("capex", "opex", "voll_ens", "carbon_payment", "total")


@register(
    "P6",
    title="System cost decomposition",
    tier="report",
    needs=("designs",),
    columns=("run_id", "label", "design_id", "source", "component", "value_bn_usd"),
)
def p6_cost_decomposition(runs, **_):
    """Capex, opex, VOLL.ENS and the carbon payment, per design.

    Prefers ``eval.parquet`` (``source="eval"``) and falls back to the design's
    own objective (``source="design"``); the two are never mixed in one bar,
    because the design's opex is its *sampled* operating cost annualised while
    the evaluation's is the cost of the same design on the evaluation blocks.
    A design's own objective cannot separate VOLL.ENS from the rest of opex
    (``Load.linear_cost`` folds it in), so that component is NaN for
    ``source="design"``.
    """
    rows = []
    for run in runs:
        eval_table = run.eval_table() if run.has("eval") else None
        for record in run.designs():
            design_id = str(record.get("design_id"))
            objective = record.get("objective") or {}
            if eval_table is not None and (eval_table["design_id"] == design_id).any():
                sub = eval_table[eval_table["design_id"] == design_id]
                capex = float(pd.to_numeric(sub["capex_annual_usd"], errors="coerce").mean())
                voll = float(pd.to_numeric(sub["voll_cost"], errors="coerce").sum())
                operational = float(
                    pd.to_numeric(sub["operational_cost"], errors="coerce").sum()
                )
                values = {
                    "capex": capex,
                    "opex": operational - voll,
                    "voll_ens": voll,
                    "carbon_payment": float("nan"),
                    "total": capex + operational,
                }
                source = "eval"
            else:
                capex = _num(objective.get("capex_annual"))
                opex = _num(objective.get("opex_annual"))
                values = {
                    "capex": capex,
                    "opex": opex,
                    # Not separable from opex on the design record alone.
                    "voll_ens": float("nan"),
                    "carbon_payment": _num(objective.get("carbon_payment_annual")),
                    "total": _num(objective.get("annual"), capex + opex),
                }
                source = "design"
            for component in P6_COMPONENTS:
                rows.append(
                    {
                        "run_id": run.run_id,
                        "label": run.label,
                        "design_id": design_id,
                        "source": source,
                        "component": component,
                        "value_bn_usd": style.convert(values[component], "cost"),
                    }
                )
    table = pd.DataFrame(rows, columns=list(_columns("P6")))
    if table.empty:
        raise ValueError("no designs to decompose")

    stacked = table[table["component"] != "total"]
    labels = list(dict.fromkeys(stacked["label"]))
    fig, ax = plt.subplots(figsize=(1.6 * max(3, len(labels)) + 2, 4))
    bottom = np.zeros(len(labels))
    colors = {"capex": "#1f5f8b", "opex": "#4c7a4c", "voll_ens": "#e3120b",
              "carbon_payment": "#7b3f8c"}
    for component in ("capex", "opex", "voll_ens", "carbon_payment"):
        values = np.array(
            [
                np.nansum(
                    stacked[
                        (stacked["label"] == label) & (stacked["component"] == component)
                    ]["value_bn_usd"].to_numpy()
                )
                for label in labels
            ]
        )
        values = np.nan_to_num(values, nan=0.0)
        ax.bar(labels, values, bottom=bottom, color=colors[component], label=component)
        bottom = bottom + values
    sources = ", ".join(sorted(set(table["source"])))
    ax.set_ylabel(f"annual cost [{style.unit_label('cost')}]")
    ax.set_title(f"P6 - system cost decomposition (source: {sources})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, table


def _num(value, default=float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


# ---------------------------------------------------------------------------
# Phase B (schema now, body later)
# ---------------------------------------------------------------------------

phase_b(
    "P3",
    title="Capacity trajectory over planning iterations",
    tier="debug",
    needs=("iteration_capacity",),
    columns=(
        "run_id",
        "label",
        "design_id",
        "outer_iteration",
        "iteration",
        "carrier",
        "capacity_gw",
    ),
)

phase_b(
    "P4",
    title="Objective trajectory: sampled vs annualised",
    tier="report",
    needs=("iterations",),
    columns=(
        "run_id",
        "label",
        "design_id",
        "outer_iteration",
        "iteration",
        "sampled_objective_annual_bn_usd",
        "rolling_objective_annual_bn_usd",
        "estimated_full_objective_annual_bn_usd",
        "suboptimality",
    ),
)

phase_b(
    "P5",
    title="Gradient norm, clip fraction and step size",
    tier="debug",
    needs=("iterations",),
    columns=(
        "run_id",
        "label",
        "design_id",
        "outer_iteration",
        "iteration",
        "grad_norm_l1",
        "grad_norm_l2",
        "proj_grad_norm_l1",
        "clip_fraction",
        "step_norm_mw",
    ),
)

phase_b(
    "P7",
    title="Emissions and the carbon multiplier over dual ascent",
    tier="report",
    needs=("iterations",),
    columns=(
        "run_id",
        "label",
        "design_id",
        "outer_iteration",
        "lambda_carbon",
        "emissions_mt",
        "target_mt",
        "gap_frac",
    ),
)

phase_b(
    "P8",
    title="Time to a design: wall clock vs CPU-seconds",
    tier="report",
    needs=("iterations", "metrics"),
    columns=(
        "run_id",
        "label",
        "design_id",
        "method",
        "iteration",
        "time_s",
        "cores",
        "cpu_seconds",
        "objective_annual_bn_usd",
    ),
)

phase_b(
    "P9",
    title="Which periods the stochastic planner sampled",
    tier="report",
    needs=("iteration_blocks",),
    columns=(
        "run_id",
        "label",
        "design_id",
        "subproblem_id",
        "block_start",
        "block_stop",
        "year",
        "day_of_year",
        "hours",
        "n_times_sampled",
        "share_of_iterations",
        "weight",
    ),
)
