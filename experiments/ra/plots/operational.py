"""Operational plots O1-O11 (phase A).

Every function takes ``runs: list[RunHandle]`` and returns ``(fig, table)``;
the table's columns are pinned by the registration decorator.

Common options
--------------
``window=(start, stop)``
    absolute in-year hours; default the run's whole ``dataset.window``.
``year``, ``method``, ``block_size``
    restrict the series; default everything present.

Caveat repeated in several docstrings: ``available_capacity_mw`` is generators
times weather times the outage/UCAP derate, plus storage *power* times its
availability.  Storage availability here is **not** SoC-limited: a battery
counts at full power for every hour of the year.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import register, style

logger = logging.getLogger(__name__)

QUANTILE_GRID = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0)

#: Per-block metrics O9 draws.
O9_METRICS = (
    "operational_cost",
    "unserved_energy_mwh",
    "co2_tonnes",
    "curtailment_mwh",
    "solve_wall_clock_s",
)

#: A flow at or above this share of a line's capacity counts as "at capacity".
AT_CAPACITY = 0.99


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def collect_hourly(
    runs,
    quantities,
    *,
    window=None,
    year=None,
    method=None,
    block_size=None,
) -> pd.DataFrame:
    """The hourly rows of every run, tagged with ``run_id`` / ``label``."""
    frames = []
    for run in runs:
        frame = run.hourly(
            quantities=list(quantities),
            hours=tuple(window) if window is not None else None,
            years=[int(year)] if year is not None else None,
            methods=[str(method)] if method is not None else None,
            block_sizes=[str(block_size)] if block_size is not None else None,
        )
        frame = frame.assign(run_id=run.run_id, label=run.label)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        raise ValueError(
            f"no hourly rows for quantities={list(quantities)} "
            f"(window={window}, year={year}, method={method}, block_size={block_size})"
        )
    return out


def pick_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one (method, block_size) per run: the one covering the most hours.

    Load and available capacity do not depend on the solve, so every solved
    block size carries the same values; keeping them all would double-count.
    """
    keep = []
    for run_id, group in frame.groupby("run_id", sort=True):
        counts = (
            group.groupby(["method", "block_size"])["hour"]
            .nunique()
            .reset_index()
            .sort_values(["hour", "method", "block_size"], ascending=[False, True, True])
        )
        method, block_size = counts.iloc[0]["method"], counts.iloc[0]["block_size"]
        keep.append(group[(group["method"] == method) & (group["block_size"] == block_size)])
    return pd.concat(keep, ignore_index=True)


def _pivot_hours(frame: pd.DataFrame, quantity: str, index=("run_id", "label", "year", "hour")):
    sub = frame[frame["quantity"] == quantity]
    if sub.empty:
        return pd.DataFrame(columns=[*index, "value"])
    return sub.groupby(list(index), as_index=False)["value"].sum()


def _series_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """``load``, ``vre_available`` and ``available`` totals per (run, year, hour)."""
    index = ["run_id", "label", "year", "hour"]
    load = _pivot_hours(frame, "load_mw").rename(columns={"value": "load_mw"})
    available = _pivot_hours(frame, "available_capacity_mw").rename(
        columns={"value": "available_mw"}
    )
    vre = frame[
        (frame["quantity"] == "available_capacity_mw")
        & (frame["carrier"].isin(style.VRE_CARRIERS))
    ]
    vre = (
        vre.groupby(index, as_index=False)["value"].sum().rename(columns={"value": "vre_mw"})
        if not vre.empty
        else pd.DataFrame(columns=[*index, "vre_mw"])
    )
    out = load.merge(available, on=index, how="outer").merge(vre, on=index, how="left")
    for col in ("load_mw", "available_mw", "vre_mw"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)
    out["net_load_mw"] = out["load_mw"] - out["vre_mw"]
    return out.sort_values(index).reset_index(drop=True)


def _facet_axes(n: int, *, width=9.5, height=3.4, sharex=True):
    fig, axes = plt.subplots(n, 1, figsize=(width, height * n), sharex=sharex, squeeze=False)
    return fig, list(axes[:, 0])


def _stack(ax, pivot: pd.DataFrame, title: str, ylabel: str) -> None:
    if pivot.empty:
        ax.set_title(f"{title} (no data)")
        return
    carriers = list(pivot.columns)
    ax.stackplot(
        pivot.index.to_numpy(),
        [pivot[c].to_numpy() for c in carriers],
        labels=carriers,
        colors=[style.carrier_color(c) for c in carriers],
        linewidth=0.0,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=4, fontsize=7, loc="upper left")


# ---------------------------------------------------------------------------
# O1 -- available capacity by carrier
# ---------------------------------------------------------------------------


@register(
    "O1",
    title="Available capacity by carrier",
    tier="debug",
    needs=("hourly",),
    columns=(
        "run_id",
        "label",
        "method",
        "block_size",
        "year",
        "hour",
        "carrier",
        "available_gw",
    ),
)
def o1_available_capacity(runs, *, window=None, year=None, method=None, block_size=None, **_):
    """Stacked hourly available capacity per carrier, with the peak annotated.

    ``available_capacity_mw`` is generators x weather x outage/UCAP derate plus
    storage power x availability.  The storage term is **not** SoC-limited.
    """
    frame = collect_hourly(
        runs,
        ["available_capacity_mw"],
        window=window,
        year=year,
        method=method,
        block_size=block_size,
    )
    table = (
        frame.groupby(
            ["run_id", "label", "method", "block_size", "year", "hour", "carrier"],
            as_index=False,
        )["value"]
        .sum()
        .rename(columns={"value": "available_gw"})
    )
    table["available_gw"] = style.convert(table["available_gw"], "power")
    table = table.sort_values(["run_id", "method", "block_size", "year", "hour", "carrier"])
    table = table.reset_index(drop=True)

    groups = list(table.groupby(["run_id", "label", "method", "block_size"], sort=True))
    fig, axes = _facet_axes(len(groups))
    for ax, ((_run_id, label, meth, size), group) in zip(axes, groups):
        pivot = group.pivot_table(
            index="hour", columns="carrier", values="available_gw", aggfunc="sum"
        ).fillna(0.0)
        _stack(ax, pivot, f"{label} - {meth} / {size} h blocks", f"available [{style.unit_label('power')}]")
        total = pivot.sum(axis=1)
        if not total.empty:
            peak_hour = int(total.idxmax())
            ax.annotate(
                f"peak {total.max():.2f} GW @ h{peak_hour}",
                xy=(peak_hour, total.max()),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=7,
            )
    axes[-1].set_xlabel("hour of the weather year")
    fig.suptitle("O1 - available capacity by carrier")
    fig.tight_layout()
    return fig, table[list(_columns("O1"))]


# ---------------------------------------------------------------------------
# O2 -- load, VRE and net load
# ---------------------------------------------------------------------------


@register(
    "O2",
    title="Load, VRE availability and net load",
    tier="debug",
    needs=("hourly",),
    columns=(
        "run_id",
        "label",
        "year",
        "hour",
        "load_gw",
        "vre_available_gw",
        "net_load_gw",
        "net_load_rank",
        "net_load_duration_frac",
    ),
)
def o2_net_load(runs, *, window=None, year=None, method=None, block_size=None, **_):
    """Net-load time series and its duration curve.

    Net load is gross demand minus *available* VRE (before curtailment), so the
    curve is a property of the system and the weather, not of the dispatch.
    """
    frame = pick_series(
        collect_hourly(
            runs,
            ["load_mw", "available_capacity_mw"],
            window=window,
            year=year,
            method=method,
            block_size=block_size,
        )
    )
    totals = _series_totals(frame)
    table = totals[["run_id", "label", "year", "hour"]].copy()
    table["load_gw"] = style.convert(totals["load_mw"], "power")
    table["vre_available_gw"] = style.convert(totals["vre_mw"], "power")
    table["net_load_gw"] = style.convert(totals["net_load_mw"], "power")
    table["net_load_rank"] = (
        table.groupby("run_id")["net_load_gw"].rank(ascending=False, method="first").astype(int)
    )
    counts = table.groupby("run_id")["net_load_gw"].transform("size")
    table["net_load_duration_frac"] = table["net_load_rank"] / counts
    table = table.sort_values(["run_id", "year", "hour"]).reset_index(drop=True)

    fig, (ax_ts, ax_dc) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    for i, (_run_id, group) in enumerate(table.groupby("run_id", sort=True)):
        st = style.run_style(i)
        label = group["label"].iloc[0]
        ax_ts.plot(group["hour"], group["load_gw"], color=st["color"], linestyle="-",
                   linewidth=1.0, label=f"{label} load")
        ax_ts.plot(group["hour"], group["vre_available_gw"], color=st["color"],
                   linestyle=":", linewidth=1.0, label=f"{label} VRE available")
        ax_ts.plot(group["hour"], group["net_load_gw"], color=st["color"],
                   linestyle="--", linewidth=1.2, label=f"{label} net load")
        ordered = group.sort_values("net_load_rank")
        ax_dc.plot(ordered["net_load_duration_frac"], ordered["net_load_gw"],
                   color=st["color"], linestyle=st["linestyle"], linewidth=1.2, label=label)
    ax_ts.set_xlabel("hour of the weather year")
    ax_ts.set_ylabel(f"power [{style.unit_label('power')}]")
    ax_ts.legend(fontsize=7, ncol=2)
    ax_dc.set_xlabel("fraction of hours at or above")
    ax_dc.set_ylabel(f"net load [{style.unit_label('power')}]")
    ax_dc.set_title("net-load duration curve")
    ax_dc.legend(fontsize=7)
    fig.suptitle("O2 - load, VRE availability and net load")
    fig.tight_layout()
    return fig, table[list(_columns("O2"))]


# ---------------------------------------------------------------------------
# O3 -- headroom
# ---------------------------------------------------------------------------


@register(
    "O3",
    title="Capacity headroom over net load",
    tier="debug",
    needs=("hourly",),
    columns=(
        "run_id",
        "label",
        "year",
        "hour",
        "available_gw",
        "net_load_gw",
        "headroom_gw",
        "headroom_frac",
        "ens_gw",
    ),
)
def o3_headroom(
    runs, *, window=None, year=None, method=None, block_size=None, threshold=0.05, **_
):
    """Headroom = available capacity - net load, shaded below ``threshold``.

    ``available_gw`` is the same total O1 stacks, so the two figures agree.
    Hours with unserved energy are marked.
    """
    frame = collect_hourly(
        runs,
        ["load_mw", "available_capacity_mw", "unserved_mw"],
        window=window,
        year=year,
        method=method,
        block_size=block_size,
    )
    picked = pick_series(frame)
    totals = _series_totals(picked)
    ens = _pivot_hours(frame, "unserved_mw").rename(columns={"value": "ens_mw"})
    totals = totals.merge(ens, on=["run_id", "label", "year", "hour"], how="left")
    totals["ens_mw"] = totals["ens_mw"].fillna(0.0)

    table = totals[["run_id", "label", "year", "hour"]].copy()
    table["available_gw"] = style.convert(totals["available_mw"], "power")
    table["net_load_gw"] = style.convert(totals["net_load_mw"], "power")
    table["headroom_gw"] = table["available_gw"] - table["net_load_gw"]
    table["headroom_frac"] = np.where(
        table["available_gw"] > 0, table["headroom_gw"] / table["available_gw"], np.nan
    )
    table["ens_gw"] = style.convert(totals["ens_mw"], "power")
    table = table.sort_values(["run_id", "year", "hour"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (_run_id, group) in enumerate(table.groupby("run_id", sort=True)):
        st = style.run_style(i)
        ax.plot(group["hour"], group["headroom_gw"], color=st["color"],
                linestyle=st["linestyle"], linewidth=1.2, label=group["label"].iloc[0])
        tight = group["headroom_frac"] < float(threshold)
        ax.fill_between(group["hour"], 0, group["headroom_gw"], where=tight,
                        color=st["color"], alpha=0.25, linewidth=0.0)
        shed = group[group["ens_gw"] > 0]
        if not shed.empty:
            ax.scatter(shed["hour"], shed["headroom_gw"], s=12,
                       color=style.carrier_color("unserved"), zorder=5, label=None)
        if not group.empty:
            worst = group.loc[group["headroom_gw"].idxmin()]
            ax.annotate(
                f"min {worst['headroom_gw']:.2f} GW @ h{int(worst['hour'])}",
                xy=(worst["hour"], worst["headroom_gw"]),
                xytext=(4, 8), textcoords="offset points", fontsize=7,
            )
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xlabel("hour of the weather year")
    ax.set_ylabel(f"headroom [{style.unit_label('power')}]")
    ax.set_title(f"O3 - headroom (shaded below {float(threshold):.0%} of available)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, table[list(_columns("O3"))]


# ---------------------------------------------------------------------------
# O4 -- dispatch stack
# ---------------------------------------------------------------------------

_O4_SERIES = (
    ("dispatch_mw", "generation"),
    ("storage_discharge_mw", "storage_discharge"),
    ("storage_charge_mw", "storage_charge"),
    ("load_mw", "load"),
    ("unserved_mw", "unserved"),
)


def _o4_table(runs, *, window, year, method, block_size) -> pd.DataFrame:
    frame = collect_hourly(
        runs,
        [q for q, _ in _O4_SERIES],
        window=window,
        year=year,
        method=method,
        block_size=block_size,
    )
    series = {q: s for q, s in _O4_SERIES}
    frame = frame.assign(series=frame["quantity"].map(series))
    frame.loc[frame["series"] == "load", "carrier"] = "load"
    frame.loc[frame["series"] == "unserved", "carrier"] = "unserved"
    table = (
        frame.groupby(
            ["run_id", "label", "method", "block_size", "year", "hour", "series", "carrier"],
            as_index=False,
        )["value"]
        .sum()
        .rename(columns={"value": "value_gw"})
    )
    table["value_gw"] = style.convert(table["value_gw"], "power")
    order = ["run_id", "method", "block_size", "year", "hour", "series", "carrier"]
    return table.sort_values(order).reset_index(drop=True)


@register(
    "O4",
    title="Dispatch by carrier",
    tier="debug",
    needs=("hourly",),
    columns=(
        "run_id",
        "label",
        "method",
        "block_size",
        "year",
        "hour",
        "series",
        "carrier",
        "value_gw",
    ),
)
def o4_dispatch(runs, *, window=None, year=None, method=None, block_size=None, **_):
    """Stacked generation plus storage, with load overlaid and charging negative.

    ``storage_charge`` is written **positive** in the CSV; only the figure
    negates it.
    """
    table = _o4_table(runs, window=window, year=year, method=method, block_size=block_size)
    groups = list(table.groupby(["run_id", "label", "method", "block_size"], sort=True))
    fig, axes = _facet_axes(len(groups))
    for ax, ((_run_id, label, meth, size), group) in zip(axes, groups):
        up = group[group["series"].isin(("generation", "storage_discharge"))]
        pivot = up.pivot_table(
            index="hour", columns="carrier", values="value_gw", aggfunc="sum"
        ).fillna(0.0)
        _stack(ax, pivot, f"{label} - {meth} / {size} h blocks",
               f"power [{style.unit_label('power')}]")
        charge = group[group["series"] == "storage_charge"].groupby("hour")["value_gw"].sum()
        if not charge.empty:
            ax.bar(charge.index, -charge.to_numpy(), width=1.0,
                   color=style.carrier_color("storage_charge"), label="storage charge")
        load = group[group["series"] == "load"].groupby("hour")["value_gw"].sum()
        if not load.empty:
            ax.plot(load.index, load.to_numpy(), color=style.carrier_color("load"),
                    linewidth=1.2, label="load")
        shed = group[group["series"] == "unserved"].groupby("hour")["value_gw"].sum()
        shed = shed[shed > 0]
        if not shed.empty:
            ax.scatter(shed.index, shed.to_numpy(), s=12,
                       color=style.carrier_color("unserved"), label="unserved", zorder=5)
        ax.legend(ncol=4, fontsize=7, loc="upper left")
    axes[-1].set_xlabel("hour of the weather year")
    fig.suptitle("O4 - dispatch by carrier")
    fig.tight_layout()
    return fig, table[list(_columns("O4"))]


# ---------------------------------------------------------------------------
# O5 -- A minus B
# ---------------------------------------------------------------------------


@register(
    "O5",
    title="Dispatch difference between two runs",
    tier="report",
    needs=("hourly",),
    columns=(
        "label_a",
        "label_b",
        "year",
        "hour",
        "series",
        "carrier",
        "value_a_gw",
        "value_b_gw",
        "diff_gw",
    ),
    min_runs=2,
    max_runs=2,
)
def o5_dispatch_difference(runs, *, window=None, year=None, method=None, block_size=None, **_):
    """A - B by carrier, as a diverging stack.  Exactly two runs.

    The two runs must share (year, window): a mismatched hour set means the two
    dispatches are not comparable hour by hour, and raises.
    """
    a, b = runs
    key = ["year", "hour", "series", "carrier"]
    # One (method, block_size) per run: a run that solved several block sizes
    # carries the same hours several times, and summing them would double the
    # difference (`pick_series` keeps the one covering the most hours).
    table_a = pick_series(
        _o4_table([a], window=window, year=year, method=method, block_size=block_size)
    )
    table_b = pick_series(
        _o4_table([b], window=window, year=year, method=method, block_size=block_size)
    )
    hours_a = set(zip(table_a["year"], table_a["hour"]))
    hours_b = set(zip(table_b["year"], table_b["hour"]))
    if hours_a != hours_b:
        raise ValueError(
            f"O5 needs the same (year, hour) set in both runs: {a.label} has "
            f"{len(hours_a)} and {b.label} has {len(hours_b)}; "
            f"{len(hours_a ^ hours_b)} differ"
        )

    left = table_a.groupby(key, as_index=False)["value_gw"].sum().rename(
        columns={"value_gw": "value_a_gw"}
    )
    right = table_b.groupby(key, as_index=False)["value_gw"].sum().rename(
        columns={"value_gw": "value_b_gw"}
    )
    merged = left.merge(right, on=key, how="outer").fillna({"value_a_gw": 0.0, "value_b_gw": 0.0})
    merged["diff_gw"] = merged["value_a_gw"] - merged["value_b_gw"]
    merged.insert(0, "label_b", b.label)
    merged.insert(0, "label_a", a.label)
    table = merged.sort_values(key).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot = table[table["series"] == "generation"].pivot_table(
        index="hour", columns="carrier", values="diff_gw", aggfunc="sum"
    ).fillna(0.0)
    for carrier in pivot.columns:
        ax.plot(pivot.index, pivot[carrier], linewidth=1.0,
                color=style.carrier_color(carrier), label=carrier)
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xlabel("hour of the weather year")
    ax.set_ylabel(f"{a.label} - {b.label} [{style.unit_label('power')}]")
    ax.set_title(f"O5 - dispatch difference: {a.label} minus {b.label}")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    return fig, table[list(_columns("O5"))]


# ---------------------------------------------------------------------------
# O6 -- state of charge
# ---------------------------------------------------------------------------


@register(
    "O6",
    title="Storage state of charge",
    tier="debug",
    needs=("hourly",),
    columns=(
        "run_id",
        "label",
        "method",
        "block_size",
        "year",
        "hour",
        "carrier",
        "soc_gwh",
        "soc_frac",
        "block_boundary",
    ),
)
def o6_state_of_charge(runs, *, window=None, year=None, method=None, block_size=None, **_):
    """SoC per storage carrier, with the block boundaries drawn.

    ``soc_frac`` is normalised by the **maximum SoC observed in the series**,
    not by installed energy capacity, which the hourly store does not carry.
    The vertical lines are the cyclic-SoC resets: an independent block cannot
    carry energy across one.
    """
    frame = collect_hourly(
        runs,
        ["storage_soc_mwh"],
        window=window,
        year=year,
        method=method,
        block_size=block_size,
    )
    table = (
        frame.groupby(
            ["run_id", "label", "method", "block_size", "year", "hour", "carrier"],
            as_index=False,
        )["value"]
        .sum()
        .rename(columns={"value": "soc_gwh"})
    )
    table["soc_gwh"] = style.convert(table["soc_gwh"], "energy")
    peak = table.groupby(["run_id", "method", "block_size", "carrier"])["soc_gwh"].transform("max")
    table["soc_frac"] = np.where(peak > 0, table["soc_gwh"] / peak, np.nan)

    starts = {run.run_id: run.window[0] for run in runs}
    offset = table["run_id"].map(starts).fillna(0).astype(int)
    sizes = pd.to_numeric(table["block_size"], errors="coerce")
    table["block_boundary"] = np.where(
        sizes.notna() & (sizes > 0), ((table["hour"] - offset) % sizes.fillna(1)) == 0, False
    )
    order = ["run_id", "method", "block_size", "year", "hour", "carrier"]
    table = table.sort_values(order).reset_index(drop=True)

    groups = list(table.groupby(["run_id", "label", "method", "block_size"], sort=True))
    fig, axes = _facet_axes(len(groups), height=3.0)
    for ax, ((_run_id, label, meth, size), group) in zip(axes, groups):
        for carrier, sub in group.groupby("carrier", sort=True):
            ax.plot(sub["hour"], sub["soc_gwh"], linewidth=1.1,
                    color=style.carrier_color(carrier), label=carrier)
        for hour in group.loc[group["block_boundary"], "hour"].unique():
            ax.axvline(hour, color="0.75", linewidth=0.6, zorder=0)
        ax.set_title(f"{label} - {meth} / {size} h blocks")
        ax.set_ylabel(f"SoC [{style.unit_label('energy')}]")
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("hour of the weather year")
    fig.suptitle("O6 - storage state of charge")
    fig.tight_layout()
    return fig, table[list(_columns("O6"))]


# ---------------------------------------------------------------------------
# O7 -- line loading
# ---------------------------------------------------------------------------


@register(
    "O7",
    title="Transmission line loading",
    tier="report",
    needs=("hourly", "system_static"),
    columns=(
        "run_id",
        "label",
        "method",
        "block_size",
        "line",
        "bus0",
        "bus1",
        "carrier",
        "capacity_gw",
        "mean_flow_gw",
        "peak_flow_gw",
        "mean_loading",
        "peak_loading",
        "hours_at_capacity",
    ),
)
def o7_line_loading(
    runs, *, window=None, year=None, method=None, block_size=None, as_map=False, **_
):
    """Mean and peak loading per directed line, as a horizontal bar chart.

    The county **map** (bus ``x`` / ``y`` are in ``system_static.json``) is out
    of scope: z4 has four nodes and a map of it says nothing.
    """
    if as_map:
        raise NotImplementedError(
            "O7 as_map=True (the geographic line-loading map) is out of scope for phase A; "
            "system_static.json already carries the bus coordinates it will need "
            "(see memory/plans/2026-09-09-plots-spec.md section 7)"
        )
    frame = collect_hourly(
        runs,
        ["line_flow_mw"],
        window=window,
        year=year,
        method=method,
        block_size=block_size,
    )
    static = {}
    for run in runs:
        for line in run.system_static().get("lines") or []:
            static[(run.run_id, str(line["name"]))] = line

    grouped = frame.groupby(
        ["run_id", "label", "method", "block_size", "name"], as_index=False
    ).agg(mean_flow_mw=("value", "mean"), peak_flow_mw=("value", "max"))

    rows = []
    for _, row in grouped.iterrows():
        info = static.get((row["run_id"], row["name"]), {})
        capacity = info.get("capacity_mw")
        capacity = float(capacity) if capacity else float("nan")
        sub = frame[
            (frame["run_id"] == row["run_id"])
            & (frame["method"] == row["method"])
            & (frame["block_size"] == row["block_size"])
            & (frame["name"] == row["name"])
        ]
        loading = sub["value"] / capacity if capacity and np.isfinite(capacity) else None
        rows.append(
            {
                "run_id": row["run_id"],
                "label": row["label"],
                "method": row["method"],
                "block_size": row["block_size"],
                "line": row["name"],
                "bus0": info.get("bus0"),
                "bus1": info.get("bus1"),
                "carrier": info.get("carrier"),
                "capacity_gw": style.convert(capacity, "power"),
                "mean_flow_gw": style.convert(row["mean_flow_mw"], "power"),
                "peak_flow_gw": style.convert(row["peak_flow_mw"], "power"),
                "mean_loading": float(loading.mean()) if loading is not None else float("nan"),
                "peak_loading": float(loading.max()) if loading is not None else float("nan"),
                "hours_at_capacity": int((loading >= AT_CAPACITY).sum())
                if loading is not None
                else 0,
            }
        )
    table = pd.DataFrame(rows, columns=list(_columns("O7")))
    table = table.sort_values(["run_id", "method", "block_size", "line"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.32 * len(table) + 1.2)))
    ticks = [f"{r.label}/{r.line}" for r in table.itertuples()]
    positions = np.arange(len(table))
    ax.barh(positions, table["mean_loading"], height=0.55, color="#4c7a4c", label="mean loading")
    ax.scatter(table["peak_loading"], positions, color="#8b1f1f", s=16, label="peak loading",
               zorder=5)
    ax.set_yticks(positions)
    ax.set_yticklabels(ticks, fontsize=7)
    ax.axvline(1.0, color="0.4", linewidth=0.8)
    ax.set_xlabel("flow / capacity")
    ax.set_title("O7 - line loading")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, table


# ---------------------------------------------------------------------------
# O8 -- prices
# ---------------------------------------------------------------------------


@register(
    "O8",
    title="Load-bus price distribution",
    tier="report",
    needs=("hourly",),
    columns=(
        "run_id",
        "label",
        "method",
        "block_size",
        "bus",
        "quantile",
        "price_usd_per_mwh",
        "n_hours",
        "share_hours_at_voll",
    ),
)
def o8_prices(runs, *, window=None, year=None, method=None, block_size=None, **_):
    """Price quantiles, pooled and per bus.  **Load buses only** (D5).

    ``ca2040_z4`` prices the import bus at its marginal unit and the export
    buses at a degenerate negative number; neither is a system price, so the
    hourly writer never records them and this plot has no option to add them.
    """
    frame = collect_hourly(
        runs,
        ["price_usd_per_mwh"],
        window=window,
        year=year,
        method=method,
        block_size=block_size,
    )
    voll = {run.run_id: run.voll for run in runs}

    rows = []
    keys = ["run_id", "label", "method", "block_size"]
    for (run_id, label, meth, size), group in frame.groupby(keys, sort=True):
        for bus, sub in [("__all__", group), *sorted(group.groupby("bus"))]:
            prices = sub["value"].to_numpy(dtype=float)
            limit = voll.get(run_id, 0.0)
            share = float(np.mean(prices >= limit)) if limit and prices.size else 0.0
            for q in QUANTILE_GRID:
                rows.append(
                    {
                        "run_id": run_id,
                        "label": label,
                        "method": meth,
                        "block_size": size,
                        "bus": bus,
                        "quantile": float(q),
                        "price_usd_per_mwh": float(np.quantile(prices, q)) if prices.size
                        else float("nan"),
                        "n_hours": int(prices.size),
                        "share_hours_at_voll": share,
                    }
                )
    table = pd.DataFrame(rows, columns=list(_columns("O8")))

    fig, (ax_cdf, ax_box) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    pooled = table[table["bus"] == "__all__"]
    for i, (_run_id, group) in enumerate(pooled.groupby("run_id", sort=True)):
        st = style.run_style(i)
        for (meth, size), sub in group.groupby(["method", "block_size"], sort=True):
            ax_cdf.plot(
                np.maximum(sub["price_usd_per_mwh"], 1e-3),
                sub["quantile"],
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                markersize=3,
                linewidth=1.1,
                label=f"{sub['label'].iloc[0]} {meth}/{size}",
            )
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel("price [$/MWh] (log)")
    ax_cdf.set_ylabel("cumulative share of hours")
    for run in runs:
        if run.voll:
            ax_cdf.axvline(run.voll, color="#e3120b", linewidth=0.8)
            ax_cdf.annotate(f"VOLL {run.voll:,.0f}", xy=(run.voll, 0.05), fontsize=7,
                            rotation=90, ha="right")
    ax_cdf.legend(fontsize=7)

    per_bus = table[(table["bus"] != "__all__") & (table["quantile"] == 0.5)]
    if not per_bus.empty:
        labels = [f"{r.label}/{r.bus}" for r in per_bus.itertuples()]
        ax_box.bar(np.arange(len(per_bus)), per_bus["price_usd_per_mwh"], color="#1f5f8b")
        ax_box.set_xticks(np.arange(len(per_bus)))
        ax_box.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax_box.set_ylabel("median price [$/MWh]")
    ax_box.set_title("per load bus")
    fig.suptitle("O8 - load-bus prices")
    fig.tight_layout()
    return fig, table


# ---------------------------------------------------------------------------
# O9 -- per-block metrics vs the reference
# ---------------------------------------------------------------------------


@register(
    "O9",
    title="Per-block metrics and blocking error",
    tier="debug",
    needs=("metrics",),
    columns=(
        "run_id",
        "label",
        "method",
        "block_size",
        "year",
        "block_index",
        "start",
        "stop",
        "hours",
        "metric",
        "value",
        "cumulative_value",
        "reference_value",
        "cumulative_dev_rel",
    ),
)
def o9_block_metrics(runs, *, metrics=O9_METRICS, **_):
    """Per-block value and the running total against the reference solve.

    ``reference_value`` is the reference solve's total for the whole window
    (``metrics.deviation_vs_reference``); ``cumulative_dev_rel`` is therefore
    only meaningful at the last block, where the blocks tile the window.  A run
    without a reference row leaves both columns NaN and the panel is omitted.
    """
    rows = []
    for run in runs:
        frame = run.metrics()
        frame = frame[frame.get("status", "ok").astype(str) == "ok"]
        deviation = run.deviation()
        for metric in metrics:
            if metric not in frame.columns:
                continue
            ref_lookup = {}
            if not deviation.empty:
                sub = deviation[deviation["metric"] == metric]
                ref_lookup = {
                    (str(r["method"]), str(r["block_size"])): float(r["reference"])
                    for _, r in sub.iterrows()
                }
            blocked = frame[frame["block_size"].astype(str) != "reference"]
            for (meth, size), group in blocked.groupby(["method", "block_size"], sort=True):
                group = group.sort_values("block_index")
                values = pd.to_numeric(group[metric], errors="coerce")
                cumulative = values.cumsum()
                reference = ref_lookup.get((str(meth), str(size)), float("nan"))
                for (_, row), value, cum in zip(group.iterrows(), values, cumulative):
                    rows.append(
                        {
                            "run_id": run.run_id,
                            "label": run.label,
                            "method": str(meth),
                            "block_size": str(size),
                            "year": int(row["year"]),
                            "block_index": int(row["block_index"]),
                            "start": int(row["start"]),
                            "stop": int(row["stop"]),
                            "hours": int(row["hours"]),
                            "metric": metric,
                            "value": float(value),
                            "cumulative_value": float(cum),
                            "reference_value": reference,
                            "cumulative_dev_rel": (float(cum) / reference - 1.0)
                            if reference and np.isfinite(reference)
                            else float("nan"),
                        }
                    )
    table = pd.DataFrame(rows, columns=list(_columns("O9")))
    if table.empty:
        raise ValueError("no per-block metric rows; metrics.csv has no successful blocked tasks")

    present = [m for m in metrics if m in set(table["metric"])]
    fig, axes = plt.subplots(len(present), 1, figsize=(9.5, 2.4 * len(present)), squeeze=False)
    for ax, metric in zip(axes[:, 0], present):
        sub = table[table["metric"] == metric]
        for i, ((_run_id, meth, size), group) in enumerate(
            sub.groupby(["run_id", "method", "block_size"], sort=True)
        ):
            st = style.run_style(i)
            ax.plot(group["start"], group["value"], color=st["color"],
                    linestyle=st["linestyle"], linewidth=1.0,
                    label=f"{group['label'].iloc[0]} {meth}/{size}")
        ax.set_ylabel(metric, fontsize=7)
        ax.legend(fontsize=6, ncol=3)
    axes[-1, 0].set_xlabel("block start hour")
    fig.suptitle("O9 - per-block metrics")
    fig.tight_layout()
    return fig, table


# ---------------------------------------------------------------------------
# O10 -- ADMM convergence
# ---------------------------------------------------------------------------


@register(
    "O10",
    title="ADMM convergence trace",
    tier="debug",
    needs=("admm_trace",),
    columns=(
        "run_id",
        "label",
        "block_size",
        "year",
        "block_index",
        "iteration",
        "objective",
        "primal_power",
        "primal_phase",
        "dual_power",
        "dual_phase",
        "primal_tol",
        "dual_tol",
    ),
)
def o10_admm_trace(runs, *, block_index=None, **_):
    """Primal / dual residuals against their tolerances, objective on a twin axis.

    The objective is in **solver units** (the system is scaled by ``power_unit``
    / ``cost_unit``; both are columns of ``admm_trace.parquet``).  There is no
    per-iteration ``max_imbalance_mw``: only the final iterate is evaluated, and
    that number lives in ``metrics.csv``.
    """
    frames = []
    for run in runs:
        frame = run.admm_trace(block_index=block_index)
        frames.append(frame.assign(run_id=run.run_id, label=run.label))
    table = pd.concat(frames, ignore_index=True)[list(_columns("O10"))]
    table = table.sort_values(["run_id", "block_size", "block_index", "iteration"])
    table = table.reset_index(drop=True)

    groups = list(table.groupby(["run_id", "label", "block_size", "block_index"], sort=True))
    fig, axes = _facet_axes(len(groups), height=2.8, sharex=False)
    for ax, ((_run_id, label, size, index), group) in zip(axes, groups):
        ax.semilogy(group["iteration"], group["primal_power"].abs(), color="#1f5f8b",
                    label="primal power")
        ax.semilogy(group["iteration"], group["dual_power"].abs(), color="#d2691e",
                    label="dual power")
        ax.semilogy(group["iteration"], group["primal_tol"].abs(), color="#1f5f8b",
                    linestyle=":", label="primal tol")
        ax.semilogy(group["iteration"], group["dual_tol"].abs(), color="#d2691e",
                    linestyle=":", label="dual tol")
        twin = ax.twinx()
        twin.plot(group["iteration"], group["objective"], color="0.35", linewidth=0.9)
        twin.set_ylabel("objective [solver units]", fontsize=7)
        twin.grid(False)
        ax.set_title(f"{label} - block {index} ({size} h)")
        ax.set_ylabel("residual")
        ax.legend(fontsize=6, ncol=2)
    axes[-1].set_xlabel("ADMM iteration")
    fig.suptitle("O10 - ADMM convergence")
    fig.tight_layout()
    return fig, table


# ---------------------------------------------------------------------------
# O11 -- solve time
# ---------------------------------------------------------------------------


@register(
    "O11",
    title="Solve time by block size",
    tier="report",
    needs=("metrics",),
    columns=(
        "run_id",
        "label",
        "method",
        "block_size",
        "hours_per_block",
        "n_blocks",
        "total_hours",
        "solve_s_total",
        "solve_s_mean",
        "solve_s_p95",
        "wall_s_total",
        "cores",
        "cpu_seconds",
        "n_variables_mean",
        "n_constraints_mean",
    ),
)
def o11_solve_time(runs, **_):
    """Solve time against block length, and the CPU-seconds it costs.

    ``cores`` comes from the run's ``env.json`` (SLURM's ``cpus_per_task`` when
    the run was on Sherlock, else ``os.cpu_count()`` at run time).  When it is
    absent both ``cores`` and ``cpu_seconds`` are NaN -- they are never guessed.
    """
    rows = []
    for run in runs:
        frame = run.metrics()
        frame = frame[frame.get("status", "ok").astype(str) == "ok"]
        cores = run.cores
        for (meth, size), group in frame.groupby(["method", "block_size"], sort=True):
            solve = pd.to_numeric(group.get("solve_wall_clock_s"), errors="coerce")
            wall = pd.to_numeric(group.get("wall_clock_s"), errors="coerce")
            hours = pd.to_numeric(group.get("hours"), errors="coerce")
            wall_total = float(wall.sum())
            rows.append(
                {
                    "run_id": run.run_id,
                    "label": run.label,
                    "method": str(meth),
                    "block_size": str(size),
                    "hours_per_block": float(hours.mean()),
                    "n_blocks": len(group),
                    "total_hours": float(hours.sum()),
                    "solve_s_total": float(solve.sum()),
                    "solve_s_mean": float(solve.mean()),
                    "solve_s_p95": float(solve.quantile(0.95)) if len(solve) else float("nan"),
                    "wall_s_total": wall_total,
                    "cores": cores,
                    "cpu_seconds": wall_total * cores if np.isfinite(cores) else float("nan"),
                    "n_variables_mean": float(
                        pd.to_numeric(group.get("n_variables"), errors="coerce").mean()
                    ),
                    "n_constraints_mean": float(
                        pd.to_numeric(group.get("n_constraints"), errors="coerce").mean()
                    ),
                }
            )
    table = pd.DataFrame(rows, columns=list(_columns("O11")))
    if table.empty:
        raise ValueError("no successful tasks in metrics.csv")

    fig, (ax_time, ax_cpu) = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, ((_run_id, meth), group) in enumerate(table.groupby(["run_id", "method"], sort=True)):
        st = style.run_style(i)
        group = group.sort_values("hours_per_block")
        label = f"{group['label'].iloc[0]} {meth}"
        ax_time.plot(group["hours_per_block"], group["solve_s_mean"], color=st["color"],
                     linestyle=st["linestyle"], marker=st["marker"], label=label)
        ax_cpu.plot(group["hours_per_block"], group["cpu_seconds"], color=st["color"],
                    linestyle=st["linestyle"], marker=st["marker"], label=label)
    ax_time.set_yscale("log")
    ax_time.set_xlabel("hours per block")
    ax_time.set_ylabel("mean solve time [s] (log)")
    ax_time.legend(fontsize=7)
    ax_cpu.set_xlabel("hours per block")
    ax_cpu.set_ylabel("CPU-seconds")
    ax_cpu.legend(fontsize=7)
    fig.suptitle("O11 - solve time")
    fig.tight_layout()
    return fig, table


def _columns(plot_id: str) -> tuple[str, ...]:
    from . import PLOTS

    return PLOTS[plot_id].columns
