"""Planning plots P1-P9.

Implemented: P1 (capacity by carrier), P2 (the design matrix), P3 (capacity
trajectory), P4 (objective trajectory), P5 (gradient diagnostics), P6 (the cost
decomposition) and P9 (sampled periods).  P7 (emissions vs the carbon price) and
P8 (the cost-vs-reliability frontier) stay registered with their schema and raise
``NotImplementedError`` (D8): both need artefacts no run has produced yet -- P7
an emissions-constrained sweep (every phase-2 cell ran ``carbon_tax: 0``,
``emissions.mode: none``) and P8 ``eval.parquet`` from ``ra evaluate``.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import phase_b, register, style
from .loader import MissingDataError

logger = logging.getLogger(__name__)

#: Power-capacity parameters, in MW; storage also gets an energy row in MWh.
_POWER_ATTRS = ("nominal_capacity", "power_capacity")

#: A carrier below this (GW or GWh) is dropped from P1: the expansion floors
#: leave 0.1-0.4 MW on rows nothing was built on.
MIN_VISIBLE_GW = 0.005


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
    fig, axes = plt.subplots(len(units), 1, figsize=(11.5, 3.8 * len(units)), squeeze=False)
    for ax, unit in zip(axes[:, 0], units):
        # Bars are drawn per *display* carrier (batteries merged into BESS) and
        # ordered by the global stacking order; the returned table stays per
        # carrier, so `raw_data` keeps every battery type separately.  Hue is the
        # **run**, because comparing runs is what this figure is for; a carrier
        # that is zero in every run (and never built) is dropped -- it is an
        # empty slot on the x axis and says nothing.
        sub = table[table["unit"] == unit]
        sub = sub.assign(carrier=style.display_carriers(sub["carrier"]))
        pivot_built = sub.pivot_table(index="carrier", columns="label", values="as_built",
                                      aggfunc="sum").fillna(0.0)
        pivot_new = sub.pivot_table(index="carrier", columns="label", values="delta",
                                    aggfunc="sum").fillna(0.0)
        # "Any capacity" means a *visible* amount: the expansion floors put
        # 0.1-0.4 MW on rows nothing was ever built on, and an invisible bar
        # under a tick label is a slot the reader has to rule out by squinting.
        floor = MIN_VISIBLE_GW
        present = [
            c for c in style.stack_order(sub["carrier"].unique())
            if float(pivot_built.loc[c].abs().max()) > floor
            or float(pivot_new.loc[c].abs().max()) > floor
        ]
        pivot_built, pivot_new = pivot_built.reindex(present), pivot_new.reindex(present)
        positions = np.arange(len(present), dtype=float)
        labels = list(pivot_built.columns)
        width = 0.8 / max(1, len(labels))
        for j, label in enumerate(labels):
            offset = positions + (j - (len(labels) - 1) / 2) * width
            colour = style.run_style(j)["color"]
            built = pivot_built[label].to_numpy()
            new_build = pivot_new[label].clip(lower=0).to_numpy()
            ax.bar(offset, built, width=width, color=colour, label=label)
            ax.bar(offset, new_build, width=width, bottom=built, color=colour,
                   hatch="///", edgecolor="white", linewidth=0.3,
                   label="new build (hatched)" if j == 0 else None)
            retired = pivot_new[label].clip(upper=0).to_numpy()
            if np.any(retired):
                ax.bar(offset, retired, width=width, color="0.6",
                       label="retired" if j == 0 else None)
        ax.set_xticks(positions)
        ax.set_xticklabels(present, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(unit)
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        # Explicit head room: a bar flush against the frame reads as clipped.
        top = float(np.nanmax((pivot_built + pivot_new.clip(lower=0)).to_numpy()))
        bottom = float(np.nanmin(pivot_new.clip(upper=0).to_numpy()))
        ax.set_ylim(min(0.0, bottom) * 1.1 - 1e-9, top * 1.10 + 1e-9)
        handles, labels_ = ax.get_legend_handles_labels()
        seen = dict(zip(labels_, handles))
        style.outside_legend(ax, handles=list(seen.values()), labels=list(seen))
    style.finish(fig, "P1 - designed capacity by carrier (as-built solid, new build hatched)")
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

    # The cell is what each formulation *built* -- designed minus as-built --
    # because the as-built fleet is identical in every row and would otherwise
    # set the colour scale for every carrier nobody touched.  A carrier no cell
    # built is dropped; the totals panel is on the same rows.
    power = table[table["unit"] == style.unit_label("power")]
    power = power.assign(carrier=style.display_carriers(power["carrier"]))
    pivot = power.pivot_table(index="label", columns="carrier", values="delta", aggfunc="sum")
    pivot = pivot.fillna(0.0)
    pivot = pivot[style.stack_order(pivot.columns)]
    built = pivot.loc[:, pivot.abs().max(axis=0) > MIN_VISIBLE_GW]
    totals = pivot.sum(axis=1)

    fig, axes = plt.subplots(
        1, 2, figsize=(1.6 * max(4, built.shape[1]) + 7, 0.55 * max(3, len(pivot.index)) + 2.8),
        gridspec_kw={"width_ratios": [max(2.0, built.shape[1] * 0.9), 2.2]},
    )
    ax = axes[0]
    if built.empty or built.shape[1] == 0:
        # Every design equals its as-built fleet: an empty heat map is a
        # degenerate axes, so say what happened instead of drawing nothing.
        ax.axis("off")
        ax.set_title("P2 - what each formulation built")
        ax.text(0.5, 0.5, "no design built anything above as-built",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.4")
        axes[1].barh(np.arange(len(totals)), totals.to_numpy(),
                     color=[style.run_style(i)["color"] for i in range(len(totals))])
        axes[1].set_yticks(np.arange(len(totals)))
        axes[1].set_yticklabels(list(totals.index), fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel(f"total built [{style.unit_label('power')}]")
        axes[1].set_title("total")
        style.finish(fig)
        return fig, table[list(_columns("P2"))]
    limit = float(np.nanmax(np.abs(built.to_numpy())))
    image = ax.imshow(built.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=limit)
    ax.set_xticks(np.arange(built.shape[1]))
    ax.set_xticklabels(built.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(built.index)))
    ax.set_yticklabels(built.index, fontsize=8)
    for i in range(built.shape[0]):
        for j in range(built.shape[1]):
            value = built.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.6 * limit else "#111111")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label=f"built above as-built [{style.unit_label('power')}]")
    ax.set_title("P2 - what each formulation built")

    ax2 = axes[1]
    positions = np.arange(len(totals))
    ax2.barh(positions, totals.to_numpy(),
             color=[style.run_style(i)["color"] for i in range(len(totals))])
    for i, value in enumerate(totals.to_numpy()):
        ax2.annotate(f"{value:.2f}", xy=(value, i), xytext=(3, 0),
                     textcoords="offset points", va="center", fontsize=8)
    ax2.set_yticks(positions)
    ax2.set_yticklabels(list(totals.index), fontsize=8)
    # `imshow` already draws row 0 at the top; `barh` does not, so only the bars
    # are flipped -- inverting both put the two panels in opposite orders.
    ax2.invert_yaxis()
    ax2.set_xlabel(f"total built [{style.unit_label('power')}]")
    ax2.set_title("total")
    ax2.margins(x=0.18)
    style.finish(fig)
    return fig, table[list(_columns("P2"))]


# ---------------------------------------------------------------------------
# P3 / P4 / P5 -- what the gradient planner did, iteration by iteration
# ---------------------------------------------------------------------------


def _iteration_runs(runs, table_name: str):
    """The runs that carry ``table_name``, with the others named in the error."""
    have = [run for run in runs if run.has(table_name)]
    if not have:
        raise MissingDataError(
            runs[0].run_id,
            table_name,
            f"none of {', '.join(r.label for r in runs)} wrote {table_name}; only the "
            "gradient methods iterate (a single-level LP solves once)",
        )
    return have


def _run_line_style(position: int, run) -> dict:
    """Colour by run; solid for a full-batch run, dashed for a minibatch one."""
    st = style.run_style(position)
    return {"color": st["color"], "linestyle": st["linestyle"]}


@register(
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
def p3_capacity_trajectory(runs, *, moving_only=True, **_):
    """One line per built carrier against iteration, every run overlaid.

    Carriers are the *display* carriers (batteries merged into ``BESS``).  With
    ``moving_only`` (the default) a carrier whose capacity never changes over the
    run -- every thermal row, hydro, the lines -- is dropped from the figure: it
    is a flat line at its as-built value and says nothing.  The table keeps every
    carrier either way.
    """
    frames = []
    for run in _iteration_runs(runs, "iteration_capacity"):
        frame = run.iteration_capacity()
        frame = frame.assign(carrier=style.display_carriers(frame["carrier"]))
        grouped = frame.groupby(
            ["outer_iteration", "iteration", "carrier"], as_index=False
        )["capacity_mw"].sum()
        grouped["run_id"] = run.run_id
        grouped["label"] = run.label
        grouped["design_id"] = str((run.designs() or [{}])[0].get("design_id"))
        grouped["capacity_gw"] = style.convert(grouped["capacity_mw"], "power")
        frames.append(grouped.drop(columns=["capacity_mw"]))
    table = pd.concat(frames, ignore_index=True)[list(_columns("P3"))]
    table = table.sort_values(["run_id", "iteration", "carrier"]).reset_index(drop=True)

    moving = table.groupby("carrier")["capacity_gw"].agg(lambda s: float(s.max() - s.min()))
    order = style.stack_order(table["carrier"].unique())
    drawn = [c for c in order if not moving_only or moving.get(c, 0.0) > 1e-6]
    if not drawn:  # nothing moved at all: draw everything rather than an empty axes
        drawn = order
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for i, (_run_id, group) in enumerate(table.groupby("run_id", sort=True)):
        st = style.run_style(i)
        label = group["label"].iloc[0]
        for carrier in drawn:
            sub = group[group["carrier"] == carrier].sort_values("iteration")
            if sub.empty:
                continue
            ax.plot(sub["iteration"], sub["capacity_gw"], color=style.carrier_color(carrier),
                    linestyle=st["linestyle"], linewidth=1.4,
                    label=f"{carrier} - {label}")
    ax.set_xlabel("planning iteration")
    ax.set_ylabel(f"capacity [{style.unit_label('power')}]")
    ax.set_title(
        "P3 - capacity trajectory"
        + (" (carriers that move; the rest are flat)" if moving_only else "")
    )
    style.outside_legend(ax)
    style.finish(fig)
    return fig, table


@register(
    "P4",
    title="Objective trajectory: sampled vs annualised",
    tier="report",
    needs=("designs",),
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
def p4_objective_trajectory(runs, **_):
    """Sampled, rolling and checkpointed objectives per iteration, in B$/yr.

    Every run that iterates contributes three series; every run that does *not*
    (a single-level LP solves once) contributes a horizontal reference line at
    its own annualised objective, which is what makes "how far above the LP
    optimum did the gradient method stop" readable off the figure.  A minibatch
    run's sampled series is a noisy estimator of the annual objective -- on the
    phase-2 campaign a four-week estimate has sigma ~ 1 B$ -- so the checkpoints
    (``estimated_full_objective_annual``, full-coverage forward passes) are the
    only series comparable to the reference lines.
    """
    rows = []
    iterating = [run for run in runs if run.has("iterations")]
    for run in iterating:
        frame = run.iterations()
        rows.append(
            pd.DataFrame(
                {
                    "run_id": run.run_id,
                    "label": run.label,
                    "design_id": frame.get("design_id", pd.Series(dtype=str)).astype(str),
                    "outer_iteration": pd.to_numeric(frame["outer_iteration"], errors="coerce"),
                    "iteration": pd.to_numeric(frame["iteration"], errors="coerce"),
                    "sampled_objective_annual_bn_usd": style.convert(
                        pd.to_numeric(frame["sampled_objective_annual"], errors="coerce"), "cost"
                    ),
                    "rolling_objective_annual_bn_usd": style.convert(
                        pd.to_numeric(frame["rolling_objective_raw"], errors="coerce")
                        * pd.to_numeric(frame.get("annualization_factor", 1.0), errors="coerce"),
                        "cost",
                    ),
                    "estimated_full_objective_annual_bn_usd": style.convert(
                        pd.to_numeric(frame["estimated_full_objective_annual"], errors="coerce"),
                        "cost",
                    ),
                    "suboptimality": pd.to_numeric(
                        frame.get("suboptimality", np.nan), errors="coerce"
                    ),
                }
            )
        )
    references = {}
    for run in runs:
        if run.has("iterations"):
            continue
        for record in run.designs():
            value = _num((record.get("objective") or {}).get("annual"))
            if np.isfinite(value):
                references[run.label] = style.convert(value, "cost")
    if not rows and not references:
        raise MissingDataError(runs[0].run_id, "iterations", "no iteration history and no design objective")
    table = (
        pd.concat(rows, ignore_index=True)[list(_columns("P4"))]
        if rows
        else pd.DataFrame(columns=list(_columns("P4")))
    )

    # Two panels on one x: the *estimator* and the *estimate*.  A minibatch's
    # sampled series swings several B$ around a design worth ~6, so drawing it on
    # the same axes as the trajectory flattens every curve that matters into one
    # line -- and clipping the axis would hide how noisy the estimator is.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [3.0, 2.0]})
    ax, ax_raw = axes
    groups = list(table.groupby("run_id", sort=True)) if not table.empty else []
    # Scale the top panel to the *estimates* first: the rolling mean of a
    # minibatch is wild for the first few iterations and the sampled series is
    # wilder still, so both are drawn against limits they did not set.
    settled = []
    if not table.empty:
        settled += list(
            table.loc[table["iteration"] >= 5, "rolling_objective_annual_bn_usd"].dropna()
        )
        settled += list(table["estimated_full_objective_annual_bn_usd"].dropna())
    settled += list(references.values())
    limits = None
    if settled:
        low, high = float(np.nanmin(settled)), float(np.nanmax(settled))
        pad = max(0.02 * max(abs(low), abs(high)), 0.02 * (high - low), 1e-6)
        limits = (low - pad, high + pad)

    for i, (_run_id, group) in enumerate(groups):
        st = style.run_style(i)
        label = group["label"].iloc[0]
        group = group.sort_values("iteration")
        ax.plot(group["iteration"], group["rolling_objective_annual_bn_usd"],
                color=st["color"], linewidth=1.4, linestyle="--", label=f"{label} rolling")
        checkpoints = group[group["estimated_full_objective_annual_bn_usd"].notna()]
        if not checkpoints.empty:
            ax.plot(checkpoints["iteration"],
                    checkpoints["estimated_full_objective_annual_bn_usd"],
                    color=st["color"], marker=st["marker"], markersize=4, linewidth=1.6,
                    label=f"{label} full-coverage estimate")
        # The sampled series joins the top panel only when it *fits* there: for a
        # full-batch run it is the trajectory itself, for a minibatch run it
        # would be a picket fence of clipped spikes across the estimates.
        sampled = group["sampled_objective_annual_bn_usd"]
        on_scale = (
            limits is not None
            and float(sampled.between(*limits).mean()) > 0.9
        )
        if on_scale:
            ax.plot(group["iteration"], sampled, color=st["color"], linewidth=0.9,
                    alpha=0.75, label=f"{label} sampled (= full coverage)")
        ax_raw.plot(group["iteration"], group["sampled_objective_annual_bn_usd"],
                    color=st["color"], linewidth=0.8, alpha=0.7, label=f"{label} sampled")
    for j, (label, value) in enumerate(sorted(references.items())):
        for target in (ax, ax_raw):
            target.axhline(value, color="0.35", linewidth=1.0,
                           linestyle=style.RUN_STYLES[j % len(style.RUN_STYLES)],
                           label=f"{label} (solved once)" if target is ax else None)
    if limits is not None:
        ax.set_ylim(*limits)
    ax.set_ylabel(f"annualised objective\n[{style.unit_label('cost')}/yr]", fontsize=9)
    ax.set_title("P4 - objective trajectory (rolling and full-coverage estimates)")
    style.outside_legend(ax)
    ax_raw.set_xlabel("planning iteration")
    ax_raw.set_ylabel(f"sampled estimate\n[{style.unit_label('cost')}/yr]", fontsize=9)
    ax_raw.set_title("the estimator itself: one point per iteration's batch", fontsize=9)
    style.outside_legend(ax_raw)
    style.finish(fig)
    return fig, table


@register(
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
        "free_grad_norm_l2",
        "step_norm_free_mw",
        "stationarity_max",
    ),
)
def p5_gradient_diagnostics(runs, **_):
    """Gradient norm, step norm and clip fraction -- is the run stalled or clipped?

    Three stacked panels sharing the iteration axis (never a twin axis): the
    gradient L2 norm in $/MW on a log axis, the realised step in MW, and the
    fraction of parameters at the clip.  A step norm pinned at the clip bound for
    every iteration says the step rule, not convergence, is what stopped the run.

    Runs made after the step-rule change (zap ``ba5f6e6``) also carry the
    *honest* columns: ``free_grad_norm_l2`` (the gradient over rows strictly
    inside their bounds -- on ``ca2040_z4`` 160 of 166 rows are frozen or on a
    floor and carry ~99 % of ``grad_norm_l2``), ``step_norm_free_mw`` (the
    realised post-projection step on those free rows; ``step_norm_mw`` was the
    pre-projection step under the clipped rule) and ``stationarity_max``
    (``max_j |m_hat_j| / gamma_j`` over interior rows, the quantity the
    ``tol_stationarity`` test reads).  They are drawn on the same panels as
    dashed/thin lines and are NaN for older runs.
    """
    frames = []
    for run in _iteration_runs(runs, "iterations"):
        frame = run.iterations()
        frames.append(
            pd.DataFrame(
                {
                    "run_id": run.run_id,
                    "label": run.label,
                    "design_id": frame.get("design_id", pd.Series(dtype=str)).astype(str),
                    "outer_iteration": pd.to_numeric(frame["outer_iteration"], errors="coerce"),
                    "iteration": pd.to_numeric(frame["iteration"], errors="coerce"),
                    "grad_norm_l1": pd.to_numeric(frame.get("grad_norm_l1"), errors="coerce"),
                    "grad_norm_l2": pd.to_numeric(frame.get("grad_norm_l2"), errors="coerce"),
                    "proj_grad_norm_l1": pd.to_numeric(
                        frame.get("proj_grad_norm_l1"), errors="coerce"
                    ),
                    "clip_fraction": pd.to_numeric(frame.get("clip_fraction"), errors="coerce"),
                    "step_norm_mw": pd.to_numeric(frame.get("step_norm_mw"), errors="coerce"),
                    "free_grad_norm_l2": pd.to_numeric(
                        frame.get("free_grad_norm_l2"), errors="coerce"
                    ),
                    "step_norm_free_mw": pd.to_numeric(
                        frame.get("step_norm_free_mw"), errors="coerce"
                    ),
                    "stationarity_max": pd.to_numeric(
                        frame.get("stationarity_max"), errors="coerce"
                    ),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)[list(_columns("P5"))]
    table = table.sort_values(["run_id", "iteration"]).reset_index(drop=True)

    # Each panel draws one primary column per run and, where a run recorded
    # it, one "honest" companion column as a thinner dotted line: the free-row
    # gradient next to the whole-vector norm, the realised free-row step next
    # to the step the rule proposed, and the stationarity measure (dimensionless,
    # its own scale) next to the clip fraction.
    panels = (
        ("grad_norm_l2", "free_grad_norm_l2", "gradient L2 norm [$/MW]\n(dotted: free rows only)", True),
        ("step_norm_mw", "step_norm_free_mw", "step norm [MW]\n(dotted: free rows, post-projection)", False),
        ("clip_fraction", "stationarity_max", "clip fraction\n(dotted: max |m_hat|/gamma, interior rows)", False),
    )
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.5, 2.6 * len(panels)), sharex=True)
    for ax, (column, companion, ylabel, log) in zip(axes, panels):
        for i, (_run_id, group) in enumerate(table.groupby("run_id", sort=True)):
            st = style.run_style(i)
            group = group.sort_values("iteration")
            label = group["label"].iloc[0]
            ax.plot(group["iteration"], group[column], color=st["color"],
                    linestyle=st["linestyle"], linewidth=1.3, label=label)
            extra = group[companion]
            if extra.notna().any() and not np.allclose(
                extra.fillna(0.0), group[column].fillna(0.0)
            ):
                ax.plot(group["iteration"], extra, color=st["color"], linestyle=":",
                        linewidth=1.0, alpha=0.9, label=f"{label} ({companion})")
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("planning iteration")
    style.finish(fig, "P5 - gradient diagnostics")
    return fig, table


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
def p6_cost_decomposition(runs, *, opex_split=None, **_):
    """Capex, opex, VOLL.ENS and the carbon payment, per design.

    ``opex_split`` (a JSON file, or a mapping keyed by run id or label) adds a
    second panel and the rows ``opex_gross`` / ``wind_credit`` /
    ``opex_net_redispatch`` with ``source="redispatch"``: net operating cost is a
    small difference of two large numbers on this system -- the extendable 2040
    ``onwind`` rows carry a negative marginal cost (a production tax credit) -- so
    a bar of net opex alone hides which designs actually dispatch more fuel.  Each
    entry needs ``blocked_gross_opex``, ``blocked_wind_credit`` and
    ``blocked_net_opex`` in dollars, measured on one common operational model.

    Prefers ``eval.parquet`` (``source="eval"``) and falls back to the design's
    own objective (``source="design"``); the two are never mixed in one bar,
    because the design's opex is its *sampled* operating cost annualised while
    the evaluation's is the cost of the same design on the evaluation blocks.
    A design's own objective cannot separate VOLL.ENS from the rest of opex
    (``Load.linear_cost`` folds it in), so that component is NaN for
    ``source="design"``.
    """
    split = _load_opex_split(opex_split)
    coverage: dict[str, float] = {}
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
                coverage[run.label] = _num(
                    (record.get("annualization") or {}).get("coverage"), 1.0
                )
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
            entry = split.get(run.run_id) or split.get(run.label) or {}
            for component, key in (
                ("opex_gross", "blocked_gross_opex"),
                ("wind_credit", "blocked_wind_credit"),
                ("opex_net_redispatch", "blocked_net_opex"),
                ("voll_ens_redispatch", "blocked_ens_mwh"),
            ):
                if key not in entry:
                    continue
                value = _num(entry[key])
                if component == "voll_ens_redispatch":
                    value = value * float(run.voll)
                rows.append(
                    {
                        "run_id": run.run_id,
                        "label": run.label,
                        "design_id": design_id,
                        "source": "redispatch",
                        "component": component,
                        "value_bn_usd": style.convert(value, "cost"),
                    }
                )
    table = pd.DataFrame(rows, columns=list(_columns("P6")))
    if table.empty:
        raise ValueError("no designs to decompose")

    stacked = table[(table["source"] != "redispatch") & (table["component"] != "total")]
    redispatch = table[table["source"] == "redispatch"]
    labels = list(dict.fromkeys(table["label"]))
    width = 1.6 * max(3, len(labels)) + 2
    if redispatch.empty:
        fig, ax = plt.subplots(figsize=(width, 4))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(2 * width, 4.4))
        ax = axes[0]
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
    sources = ", ".join(sorted(set(stacked["source"])))
    ax.set_ylabel(f"annual cost [{style.unit_label('cost')}]")
    ax.set_title(f"P6 - system cost decomposition (source: {sources})")
    ax.set_xticks(np.arange(len(labels)))
    # A design solved on a *sample* reports an extrapolation, not a year: say so
    # on the bar rather than in a caption nobody reads next to the number.
    ax.set_xticklabels(
        [
            label + ("\n(in-sample, x%.3g)" % (1.0 / coverage[label])
                     if coverage.get(label, 1.0) < 0.999 else "")
            for label in labels
        ],
        rotation=20, ha="right", fontsize=8,
    )
    for position, total in enumerate(bottom):
        ax.annotate(f"{total:.3f}", xy=(position, total), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylim(0.0, float(np.nanmax(bottom)) * 1.12)
    ax.legend(fontsize=7, loc="lower left")

    if not redispatch.empty:
        ax2 = axes[1]
        positions = np.arange(len(labels))
        def series(component):
            return np.array([
                float(
                    redispatch[
                        (redispatch["label"] == label)
                        & (redispatch["component"] == component)
                    ]["value_bn_usd"].sum()
                )
                for label in labels
            ])
        gross, credit = series("opex_gross"), series("wind_credit")
        net = series("opex_net_redispatch")
        ax2.bar(positions, gross, width=0.6, color="#8b1f1f", label="gross dispatch cost")
        ax2.bar(positions, credit, width=0.6, color="#4c7a4c", label="wind credit (negative)")
        ax2.plot(positions, net, color="#111111", marker="o", markersize=5, linewidth=0,
                 label="net opex (gross + credit)")
        voll = series("voll_ens_redispatch")
        ax2.plot(positions, voll, color=style.carrier_color("unserved"), marker="x",
                 markersize=6, linewidth=0,
                 label=f"VOLL x ENS ({'zero everywhere' if not np.any(voll) else 'non-zero'})")
        ax2.axhline(0.0, color="0.3", linewidth=0.8)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels, rotation=20, ha="right")
        ax2.set_ylabel(f"annual operating cost [{style.unit_label('cost')}]")
        ax2.set_title("net opex is a difference of two large numbers (re-dispatched)")
        ax2.legend(fontsize=7)
    style.finish(fig)
    return fig, table


def _load_opex_split(value) -> dict:
    """``opex_split`` as a mapping: a path to JSON, a mapping, or nothing."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    import json
    from pathlib import Path

    return json.loads(Path(value).expanduser().read_text())


# ---------------------------------------------------------------------------
# P9 -- which periods the planner sampled
# ---------------------------------------------------------------------------

#: Hours in a week; the campaign's blocks are aligned weeks.
WEEK_HOURS = 168


def _sampled_blocks(run) -> pd.DataFrame:
    """Every (block, iteration) the run's planner saw, from whichever source has it.

    A gradient run logs one row per (iteration, block) in
    ``iterations/*.iteration_blocks.parquet``.  A single-level LP solves once and
    logs nothing, but its ``design.json`` records the blocks the sample was drawn
    from (``selection.blocks``) -- that *is* its one and only iteration.
    """
    if run.has("iteration_blocks"):
        frame = run.iteration_blocks()
        return frame.assign(
            block_start=pd.to_numeric(frame["block_start"], errors="coerce").astype(int),
            block_stop=pd.to_numeric(frame["block_stop"], errors="coerce").astype(int),
        )
    rows = []
    for record in run.designs():
        selection = record.get("selection") or {}
        blocks = selection.get("blocks") or []
        year = int((record.get("years") or [0])[0]) if record.get("years") else 0
        for index, pair in enumerate(blocks):
            start, stop = int(pair[0]), int(pair[1])
            rows.append(
                {
                    "iteration": 0,
                    "outer_iteration": 0,
                    "subproblem_id": index,
                    "block_start": start,
                    "block_stop": stop,
                    "year": year,
                    "day_of_year": start // 24,
                    "hours": stop - start,
                    "weight": 1.0,
                }
            )
    if not rows:
        raise MissingDataError(
            run.run_id,
            "iteration_blocks",
            f"run {run.run_id} logged no sampled blocks and its design records no "
            "selection.blocks (a full-coverage method samples nothing)",
        )
    return pd.DataFrame(rows)


def _weekly_peak_net_load(path, window_start: int) -> pd.DataFrame:
    """``week_of_year -> peak net load`` from an O3/O2 table written by ``ra plot``.

    Passed in rather than recomputed: net load is a property of the weather year
    and the *as-built* fleet, and phase 1 already published it for this dataset
    (``figures/phase1/raw_data/O3_lp-unscaled.csv.gz``, unscaled demand, the same
    window).  A planning run writes no hourly data of its own.
    """
    frame = pd.read_csv(path)
    column = "net_load_gw" if "net_load_gw" in frame.columns else "net_load_mw"
    values = pd.to_numeric(frame[column], errors="coerce")
    if column.endswith("_mw"):
        values = style.convert(values, "power")
    hours = pd.to_numeric(frame["hour"], errors="coerce")
    weeks = ((hours - int(window_start)) // WEEK_HOURS).astype(int)
    out = pd.DataFrame({"week_of_year": weeks, "net_load_gw": values})
    return out.groupby("week_of_year", as_index=False)["net_load_gw"].max().rename(
        columns={"net_load_gw": "peak_net_load_gw"}
    )


@register(
    "P9",
    title="Which periods the stochastic planner sampled",
    tier="report",
    needs=("designs",),
    columns=(
        "run_id",
        "label",
        "design_id",
        "subproblem_id",
        "block_start",
        "block_stop",
        "year",
        "day_of_year",
        "week_of_year",
        "hours",
        "n_times_sampled",
        "share_of_iterations",
        "weight",
        "peak_net_load_gw",
    ),
)
def p9_sampled_periods(runs, *, peak_net_load=None, window_start=None, **_):
    """How often each week was sampled, against how hard that week is.

    Top panel: sampled count per week, one bar group per run.  Bottom panel: the
    weekly **peak net load** of the same year (``peak_net_load``, a CSV with
    ``hour`` and ``net_load_gw``), so a week the planner never drew can be read
    against how much stress it carried -- the point of the period-selection
    study.  One raster panel per run that iterates, iteration x week, showing
    *when* each week was drawn rather than only how often.  Never a twin axis:
    the count and the GW live in their own panels on a shared x.
    """
    rows = []
    rasters = {}
    for run in runs:
        blocks = _sampled_blocks(run)
        start = int(window_start if window_start is not None else run.window[0])
        blocks = blocks.assign(
            week_of_year=((blocks["block_start"] - start) // WEEK_HOURS).astype(int)
        )
        iterations = int(blocks["iteration"].nunique())
        counts = blocks.groupby(
            ["subproblem_id", "block_start", "block_stop", "year", "day_of_year",
             "week_of_year", "hours"], as_index=False
        ).agg(n_times_sampled=("iteration", "nunique"), weight=("weight", "mean"))
        counts["run_id"] = run.run_id
        counts["label"] = run.label
        counts["design_id"] = str((run.designs() or [{}])[0].get("design_id"))
        counts["share_of_iterations"] = counts["n_times_sampled"] / max(1, iterations)
        rows.append(counts)
        if run.has("iteration_blocks"):
            rasters[run.label] = blocks[["iteration", "week_of_year"]]
    table = pd.concat(rows, ignore_index=True)

    overlay = (
        _weekly_peak_net_load(peak_net_load, table["block_start"].min()
                              if window_start is None else window_start)
        if peak_net_load is not None
        else pd.DataFrame(columns=["week_of_year", "peak_net_load_gw"])
    )
    table = table.merge(overlay, on="week_of_year", how="left")
    if "peak_net_load_gw" not in table.columns:
        table["peak_net_load_gw"] = np.nan
    table = table[list(_columns("P9"))].sort_values(
        ["run_id", "week_of_year"]
    ).reset_index(drop=True)

    n_panels = 2 + len(rasters)
    heights = [2.6, 2.0] + [2.2] * len(rasters)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(11, sum(heights)), sharex=True,
        gridspec_kw={"height_ratios": heights},
    )
    axes = np.atleast_1d(axes)
    weeks = np.arange(int(table["week_of_year"].min()), int(table["week_of_year"].max()) + 1)

    ax = axes[0]
    labels = list(dict.fromkeys(table["label"]))
    width = 0.8 / max(1, len(labels))
    for i, label in enumerate(labels):
        # Sum over blocks that share a week: a run with sub-weekly blocks puts
        # several of them in one week, and the bar is "times this week was seen".
        sub = (
            table[table["label"] == label]
            .groupby("week_of_year")["n_times_sampled"].sum()
        )
        counts = [float(sub.get(w, 0.0)) for w in weeks]
        ax.bar(weeks + (i - (len(labels) - 1) / 2) * width, counts, width=width,
               color=style.run_style(i)["color"], label=label)
    ax.set_ylabel("times sampled")
    ax.set_title("P9 - which weeks the planner sampled")
    ax.legend(fontsize=7)

    ax = axes[1]
    if not overlay.empty:
        curve = overlay.set_index("week_of_year").reindex(weeks)
        ax.bar(weeks, curve["peak_net_load_gw"], width=0.8, color="0.75")
        worst = curve["peak_net_load_gw"].nlargest(2)
        for week, value in worst.items():
            ax.bar([week], [value], width=0.8, color=style.carrier_color("unserved"))
            ax.annotate(f"w{int(week)}", xy=(week, value), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7)
            for other in axes:
                other.axvline(week, color=style.carrier_color("unserved"),
                              linewidth=0.8, alpha=0.35, zorder=0)
        ax.set_ylim(bottom=float(np.nanmin(curve["peak_net_load_gw"])) * 0.9)
    else:
        ax.text(0.5, 0.5, "no net-load overlay supplied", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="0.4")
    ax.set_ylabel(f"weekly peak\nnet load [{style.unit_label('power')}]", fontsize=8)

    for ax, (label, raster) in zip(axes[2:], sorted(rasters.items())):
        grid = np.zeros((int(raster["iteration"].max()) + 1, len(weeks)))
        lookup = {w: i for i, w in enumerate(weeks)}
        for row in raster.itertuples():
            column = lookup.get(int(row.week_of_year))
            if column is not None:
                grid[int(row.iteration), column] = 1.0
        ax.pcolormesh(
            np.append(weeks - 0.5, weeks[-1] + 0.5),
            np.arange(grid.shape[0] + 1),
            grid, cmap="Greys", vmin=0.0, vmax=1.0, shading="flat",
        )
        ax.set_ylabel(f"{label}\niteration", fontsize=8)
    axes[-1].set_xlabel("week of the window (0 = the first aligned week)")
    style.finish(fig)
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


