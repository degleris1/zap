"""Operational plots O1-O13 (phase A).

Every function takes ``runs: list[RunHandle]`` and returns ``(fig, table)``;
the table's columns are pinned by the registration decorator.

Common options
--------------
``window=(start, stop)``
    absolute in-year hours; default the run's whole ``dataset.window``.
``year``, ``method``, ``block_size``
    restrict the series; default everything present.

Caveat repeated in several docstrings: ``available_capacity_mw`` is *in-state*
generators (import rows excluded) times weather times the outage/UCAP derate,
plus storage *power* times its availability.  Storage availability here is
**not** SoC-limited: a battery counts at full power for every hour of the year.
This is the same membership as ``available_mw_min`` in ``metrics.csv``.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, NullFormatter

from . import register, style
from .loader import MissingDataError

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

#: The longest span O2 draws hour by hour; above it the time panel is a weekly
#: mean.  Four weeks of hourly data is ~670 points across a 5-inch panel, which
#: is the most that still resolves individual hours.
HOURLY_PANEL_HOURS = 4 * 168

#: The same threshold for a *stacked* plot (O1, O4, any future stacked-by-carrier
#: figure): above it the hourly stack is replaced by a 3x4 grid of months, each
#: panel the mean 24 h profile of that month (Kamran, 2026-09-09).  A year of
#: hourly stacked bands is unreadable at any figure width.
STACKED_HOURLY_HOURS = HOURLY_PANEL_HOURS

#: Hours to subtract from an absolute (UTC) hour of the weather year to get local
#: Pacific time.  This is the harness's own fixed convention -- the shipped
#: window starts at hour 7 so that blocks begin at local midnight -- and it does
#: not model DST.
PACIFIC_UTC_OFFSET_HOURS = 7

#: The weather stores are 8,760 hours for every weather year, leap years
#: included (``data/*/MANIFEST.md``), so hour -> calendar date is resolved
#: against a fixed **non-leap** year; using the real leap year would slide every
#: month after February by a day.
CALENDAR_REFERENCE_YEAR = 2001

#: Series drawn **below** the x axis, in the order they leave it.  Each is
#: stacked *by carrier*, in the same colours and the same order as above the
#: axis, so the band nearest the axis below is the same carrier as the band
#: nearest it above (Kamran, 2026-09-09).  ``exports`` is here as the declared
#: seam: no hourly quantity writes it today (`persist.HOURLY_QUANTITIES` has no
#: export series and the benchmark runs use ``export_mode: drop``), and it is
#: drawn the moment a table carries it.
BELOW_AXIS_SERIES = ("storage_charge", "exports")

#: The hatch that marks a below-axis band as the *charging* half of its carrier.
#: The hue stays the carrier's, so one legend entry covers both signs.
BELOW_AXIS_HATCH = "///"

MONTH_LABELS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


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


#: A weekly bin holding fewer than this many hours is dropped rather than drawn
#: as a mean over a handful of them.
WEEKLY_BIN_MIN_HOURS = 84


def _weekly_mean(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """``frame`` averaged into 168 h bins, indexed by each bin's first hour.

    The bins are aligned to the **first hour of the window**, not to hour 0 of
    the year: the shipped window starts at hour 7 (local midnight), and binning
    on the absolute hour left a 7-hour tail bin whose mean was seven night hours
    and dropped off the end of the figure.  A trailing partial bin shorter than
    ``WEEKLY_BIN_MIN_HOURS`` is dropped for the same reason.
    """
    start = int(frame["hour"].min())
    binned = frame.assign(_bin=((frame["hour"] - start) // 168).astype(int))
    sizes = binned.groupby("_bin")["hour"].size()
    keep = set(sizes[sizes >= WEEKLY_BIN_MIN_HOURS].index)
    if keep:  # a frame shorter than one bin keeps its single partial bin
        binned = binned[binned["_bin"].isin(keep)]
    out = binned.groupby("_bin", as_index=False)[list(columns)].mean()
    out["hour"] = binned.groupby("_bin", as_index=False)["hour"].min()["hour"]
    return out.sort_values("hour").reset_index(drop=True)


def _span_hours(table: pd.DataFrame) -> int:
    """The number of hours the plotted window covers (inclusive of both ends)."""
    if table.empty:
        return 0
    return int(table["hour"].max() - table["hour"].min()) + 1


def is_monthly_view(table: pd.DataFrame) -> bool:
    """Whether a stacked plot of ``table`` becomes a monthly-profile grid."""
    return _span_hours(table) > STACKED_HOURLY_HOURS


def local_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``hour_of_day``, ``month``, ``day`` and ``day_index`` (local Pacific).

    ``day_index`` counts local days from the start of the weather year and is the
    key a "one day per month" plot groups on; ``day`` is the day of the month.
    Hours before the first local midnight (0-6 of a window that starts at hour 0)
    are clipped into day 0 so the calendar columns stay consistent with it.
    """
    local = frame["hour"].astype(int) - PACIFIC_UTC_OFFSET_HOURS
    clipped = local.clip(lower=0)
    stamps = pd.Timestamp(f"{CALENDAR_REFERENCE_YEAR}-01-01") + pd.to_timedelta(
        clipped, unit="h"
    )
    return frame.assign(
        hour_of_day=(local % 24).astype(int),
        month=stamps.dt.month.astype(int),
        day=stamps.dt.day.astype(int),
        day_index=(clipped // 24).astype(int),
    )


def _monthly_figure(groups, *, title_of, ylabel, draw, width=13.0, height=6.8):
    """One 3x4 grid of month panels per group -- mean 24 h profile per month.

    ``draw(ax, month_rows, handles)`` renders one month; it registers its artists
    in ``handles`` (``label -> artist``) so the grid can carry a single legend in
    the stacking order rather than twelve identical ones.
    """
    groups = list(groups)
    fig = plt.figure(figsize=(width, height * len(groups)))
    # Always a subfigure per group, even for one group: the group's title is the
    # subfigure's, so it cannot be overwritten by a figure-level ``suptitle``,
    # and ``subplots_adjust`` (which reserves the legend strip) survives -- the
    # monthly grid deliberately does **not** run ``tight_layout``.
    subfigs = list(fig.subfigures(len(groups), 1, squeeze=False)[:, 0])
    for subfig, (keys, group) in zip(subfigs, groups):
        axes = subfig.subplots(3, 4, sharex=True, sharey=True)
        subfig.subplots_adjust(right=0.84, top=0.88, hspace=0.45, wspace=0.15)
        local = local_calendar(group)
        handles: dict[str, object] = {}
        for index in range(12):
            ax = axes[index // 4][index % 4]
            ax.set_title(MONTH_LABELS[index], fontsize=8)
            ax.set_xlim(0, 23)
            ax.set_xticks([0, 6, 12, 18])
            ax.tick_params(labelsize=7)
            month_rows = local[local["month"] == index + 1]
            if month_rows.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="0.5")
                continue
            draw(ax, month_rows, handles)
        for row in range(3):
            axes[row][0].set_ylabel(ylabel, fontsize=8)
        for col in range(4):
            axes[2][col].set_xlabel("hour of day (Pacific)", fontsize=8)
        subfig.suptitle(title_of(keys, group), fontsize=10)
        if handles:
            subfig.legend(list(handles.values()), list(handles), loc="center right", fontsize=7)
    return fig


def _below_axis_stack(ax, group, *, index: str, aggfunc: str, labelled=(), handles=None):
    """Stack ``BELOW_AXIS_SERIES`` under the axis, by carrier, mirrored.

    Same colours and same stacking sequence as above the axis, hatched to mark
    the sign; a carrier already drawn above is *not* labelled again, so the
    legend carries one entry per carrier covering both halves and never a
    separate "storage charge" entry.
    """
    rows = group[group["series"].isin(BELOW_AXIS_SERIES)]
    if rows.empty:
        return []
    pivot = _stack_pivot(rows, "value_gw", index, aggfunc)
    if pivot.empty or not float(np.abs(pivot.to_numpy()).max()):
        return []
    labelled = set(labelled)
    polygons = ax.stackplot(
        pivot.index.to_numpy(),
        [-pivot[c].to_numpy() for c in pivot.columns],
        colors=[style.carrier_color(c) for c in pivot.columns],
        labels=["_nolegend_" if c in labelled else str(c) for c in pivot.columns],
        linewidth=0.0,
    )
    for carrier, polygon in zip(pivot.columns, polygons):
        polygon.set_hatch(BELOW_AXIS_HATCH)
        polygon.set_edgecolor("white")
        polygon.set_linewidth(0.0)
        if handles is not None:
            handles.setdefault(carrier, polygon)
    return list(pivot.columns)


def _stack_pivot(frame: pd.DataFrame, value: str, index: str, aggfunc: str) -> pd.DataFrame:
    """A carrier pivot: display groups merged, columns in the stacking order.

    The battery carriers are summed into one ``BESS`` column here -- *after* the
    plot's table is built, so the CSV beside the figure stays per-carrier.
    """
    frame = frame.assign(carrier=style.display_carriers(frame["carrier"]))
    pivot = frame.pivot_table(
        index=index, columns="carrier", values=value, aggfunc=aggfunc
    ).fillna(0.0)
    if pivot.empty:
        return pivot
    return pivot[style.stack_order(pivot.columns)]


def _facet_axes(n: int, *, width=9.5, height=3.4, sharex=True):
    fig, axes = plt.subplots(n, 1, figsize=(width, height * n), sharex=sharex, squeeze=False)
    return fig, list(axes[:, 0])


def _block_label(block_size) -> str:
    """``24`` -> ``"24 h blocks"``; ``"reference"`` -> ``"reference"``.

    ``block_size`` is the string ``"reference"`` on a reference solve, and
    appending the suffix unconditionally produced facet titles reading
    "reference h blocks".
    """
    text = str(block_size)
    try:
        hours = float(text)
    except (TypeError, ValueError):
        return text
    if not np.isfinite(hours):
        return text
    return f"{text} h blocks"


def _facet_title(label: str, method, block_size) -> str:
    return f"{label} - {method} / {_block_label(block_size)}"


def _outside_legend(ax, **kwargs) -> None:
    """A legend in the right margin, so it cannot cover the data or a callout.

    ``save`` writes with ``bbox_inches="tight"``, so the margin costs figure
    width, never plot area.
    """
    options = {"fontsize": 7, "loc": "upper left", "bbox_to_anchor": (1.005, 1.0),
               "borderaxespad": 0.0, "ncol": 1}
    options.update(kwargs)
    ax.legend(**options)


def _annotate_point(ax, x, y, text: str, *, xs=None, ys=None) -> None:
    """Label a point, offset *away* from the nearer edge of the axes.

    An annotation pinned to the left of an early peak used to land inside the
    legend; the legend now lives outside the axes, and the offset flips near the
    right edge so the text cannot run off it -- or, for a minimum sitting on the
    bottom of the frame, above the point instead of below it.
    """

    def _fraction(value, limits, given):
        values = np.asarray(limits if given is None else [np.min(given), np.max(given)],
                            dtype=float)
        lo, hi = float(values.min()), float(values.max())
        return (float(value) - lo) / ((hi - lo) or 1.0)

    right_edge = _fraction(x, ax.get_xlim(), xs) > 0.75
    bottom_edge = _fraction(y, ax.get_ylim(), ys) < 0.2
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(-4 if right_edge else 4, 10 if bottom_edge else -12),
        textcoords="offset points",
        fontsize=7,
        ha="right" if right_edge else "left",
        va="bottom" if bottom_edge else "top",
    )


def _stack(ax, pivot: pd.DataFrame, title: str, ylabel: str) -> None:
    if pivot.empty:
        ax.set_title(f"{title} (no data)")
        return
    carriers = style.stack_order(pivot.columns)
    pivot = pivot[carriers]
    ax.stackplot(
        pivot.index.to_numpy(),
        [pivot[c].to_numpy() for c in carriers],
        labels=carriers,
        colors=[style.carrier_color(c) for c in carriers],
        linewidth=0.0,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _outside_legend(ax)


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

    ``available_capacity_mw`` is in-state generators (imports excluded) x
    weather x outage/UCAP derate plus storage power x availability.  The
    storage term is **not** SoC-limited.

    Available capacity does not depend on the solve, so -- exactly as O2 and O3
    do -- one (method, block_size) per run is kept (``pick_series``): faceting
    by it drew the same panel once per solved block size.
    """
    frame = pick_series(
        collect_hourly(
            runs,
            ["available_capacity_mw"],
            window=window,
            year=year,
            method=method,
            block_size=block_size,
        )
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

    fig = _o1_figure(table)
    return fig, table[list(_columns("O1"))]


def _o1_peak(group: pd.DataFrame) -> tuple[int, float]:
    total = group.groupby("hour")["available_gw"].sum()
    return int(total.idxmax()), float(total.max())


def _o1_figure(table: pd.DataFrame):
    """O1's figure: hourly stacks for a short window, a monthly grid for a long one."""
    # One facet per *run*: `pick_series` left a single (method, block_size) per
    # run, so this groupby no longer multiplies the panels.
    groups = list(table.groupby(["run_id", "label", "method", "block_size"], sort=True))
    ylabel = f"available [{style.unit_label('power')}]"

    if is_monthly_view(table):
        def draw(ax, month_rows, handles):
            pivot = _stack_pivot(month_rows, "available_gw", "hour_of_day", "mean")
            polygons = ax.stackplot(
                pivot.index.to_numpy(),
                [pivot[c].to_numpy() for c in pivot.columns],
                colors=[style.carrier_color(c) for c in pivot.columns],
                labels=list(pivot.columns),
                linewidth=0.0,
            )
            for carrier, polygon in zip(pivot.columns, polygons):
                handles.setdefault(carrier, polygon)

        def title_of(keys, group):
            _run_id, label, meth, size = keys
            hour, peak = _o1_peak(group)
            return (
                "O1 - available capacity by carrier, mean 24 h profile by month\n"
                f"{_facet_title(label, meth, size)} - hourly peak {peak:.2f} GW @ h{hour}"
            )

        return _monthly_figure(groups, title_of=title_of, ylabel=ylabel, draw=draw)

    fig, axes = _facet_axes(len(groups))
    for ax, ((_run_id, label, meth, size), group) in zip(axes, groups):
        pivot = _stack_pivot(group, "available_gw", "hour", "sum")
        _stack(ax, pivot, _facet_title(label, meth, size), ylabel)
        total = pivot.sum(axis=1)
        if not total.empty:
            peak_hour = int(total.idxmax())
            _annotate_point(
                ax,
                peak_hour,
                total.max(),
                f"peak {total.max():.2f} GW @ h{peak_hour}",
                xs=pivot.index.to_numpy(),
                ys=total.to_numpy(),
            )
    axes[-1].set_xlabel("hour of the weather year")
    style.finish(fig, "O1 - available capacity by carrier")
    return fig


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

    The left panel is **hourly only for spans of at most four weeks**
    (``HOURLY_PANEL_HOURS``); over a longer span -- a whole year is 8,736 points
    -- it is a weekly mean, which is what is legible at that width, and the
    hourly record stays in the CSV and in the duration curve on the right.  Pass
    ``--window`` for the hourly view of a week.  Colour is the *quantity* (fixed
    order: load, VRE available, net load) and the linestyle is the run, so no two
    series are told apart by a dash pattern alone.
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

    span = int(table["hour"].max() - table["hour"].min()) + 1
    hourly_view = span <= HOURLY_PANEL_HOURS
    quantities = (
        ("load_gw", "load", "load"),
        ("vre_available_gw", "vre_available", "VRE available"),
        ("net_load_gw", "net_load", "net load"),
    )

    fig, (ax_ts, ax_dc) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    many = len(set(table["run_id"])) > 1
    for i, (_run_id, group) in enumerate(table.groupby("run_id", sort=True)):
        st = style.run_style(i)
        label = group["label"].iloc[0]
        series = group if hourly_view else _weekly_mean(group, [q for q, _, _ in quantities])
        for column, key, name in quantities:
            ax_ts.plot(
                series["hour"],
                series[column],
                color=style.SERIES_COLORS[key],
                linestyle=st["linestyle"],
                linewidth=1.1,
                label=f"{label} {name}" if many else name,
            )
        ordered = group.sort_values("net_load_rank")
        ax_dc.plot(ordered["net_load_duration_frac"], ordered["net_load_gw"],
                   color=st["color"], linestyle=st["linestyle"], linewidth=1.2, label=label)
    ax_ts.set_xlabel("hour of the weather year")
    ax_ts.set_ylabel(f"power [{style.unit_label('power')}]")
    ax_ts.set_title(
        f"hourly ({span} h)" if hourly_view else f"weekly mean ({span} h; hourly in the CSV)"
    )
    # Below the panel: three full-length series leave no corner free, and a
    # legend that covers the load peak is exactly the collision this pass removed.
    ax_ts.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax_dc.set_xlabel("fraction of hours at or above")
    ax_dc.set_ylabel(f"net load [{style.unit_label('power')}]")
    ax_dc.set_title("net-load duration curve")
    ax_dc.legend(fontsize=7)
    style.finish(fig, "O2 - load, VRE availability and net load")
    return fig, table[list(_columns("O2"))]


# ---------------------------------------------------------------------------
# O3 -- headroom
# ---------------------------------------------------------------------------


@register(
    "O3",
    title="Capacity headroom over gross load",
    tier="debug",
    needs=("hourly",),
    columns=(
        "run_id",
        "label",
        "year",
        "hour",
        "available_gw",
        "load_gw",
        "dispatchable_gw",
        "net_load_gw",
        "headroom_gw",
        "headroom_frac",
        "ens_gw",
    ),
)
def o3_headroom(
    runs, *, window=None, year=None, method=None, block_size=None, threshold=0.05, **_
):
    """Headroom = available capacity - gross load, shaded below ``threshold``.

    ``available_gw`` is the same total O1 stacks -- in-state generators
    (imports excluded) plus storage power, and it already contains VRE
    availability -- so headroom is taken against gross ``load_gw``; the
    equivalent dispatchable-vs-net-load split is written alongside as
    ``dispatchable_gw`` and ``net_load_gw`` (``dispatchable - net_load`` is the
    same number).  Subtracting net load from the *total* would count VRE
    twice.  Hours with unserved energy are marked.
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
    # From `picked`, not `frame`: a benchmark run carries the same hours once per
    # solved (method, block size), and summing them multiplied the marked ENS.
    ens = _pivot_hours(picked, "unserved_mw").rename(columns={"value": "ens_mw"})
    totals = totals.merge(ens, on=["run_id", "label", "year", "hour"], how="left")
    totals["ens_mw"] = totals["ens_mw"].fillna(0.0)

    table = totals[["run_id", "label", "year", "hour"]].copy()
    table["available_gw"] = style.convert(totals["available_mw"], "power")
    table["load_gw"] = style.convert(totals["load_mw"], "power")
    table["dispatchable_gw"] = style.convert(totals["available_mw"] - totals["vre_mw"], "power")
    table["net_load_gw"] = style.convert(totals["net_load_mw"], "power")
    table["headroom_gw"] = table["available_gw"] - table["load_gw"]
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
            _annotate_point(
                ax,
                worst["hour"],
                worst["headroom_gw"],
                f"min {worst['headroom_gw']:.2f} GW @ h{int(worst['hour'])}",
                xs=group["hour"].to_numpy(),
                ys=group["headroom_gw"].to_numpy(),
            )
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xlabel("hour of the weather year")
    ax.set_ylabel(f"headroom [{style.unit_label('power')}]")
    ax.set_title(f"O3 - headroom (shaded below {float(threshold):.0%} of available)")
    ax.legend(fontsize=7)
    style.finish(fig)
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
    fig = _o4_figure(table)
    return fig, table[list(_columns("O4"))]


def _o4_figure(table: pd.DataFrame):
    """O4's figure: hourly stacks for a short window, a monthly grid for a long one."""
    groups = list(table.groupby(["run_id", "label", "method", "block_size"], sort=True))
    ylabel = f"power [{style.unit_label('power')}]"
    up_series = ("generation", "storage_discharge")

    if is_monthly_view(table):
        def draw(ax, month_rows, handles):
            index, aggfunc = "hour_of_day", "mean"
            up = month_rows[month_rows["series"].isin(up_series)]
            # Sum the carriers within an hour first, then average over the month:
            # a plain mean over rows would divide by the number of *rows*.
            up = up.groupby(["hour_of_day", "carrier", "hour"], as_index=False)["value_gw"].sum()
            pivot = _stack_pivot(up, "value_gw", index, aggfunc)
            if not pivot.empty:
                polygons = ax.stackplot(
                    pivot.index.to_numpy(),
                    [pivot[c].to_numpy() for c in pivot.columns],
                    colors=[style.carrier_color(c) for c in pivot.columns],
                    labels=list(pivot.columns),
                    linewidth=0.0,
                )
                for carrier, polygon in zip(pivot.columns, polygons):
                    handles.setdefault(carrier, polygon)
            # Charging (and exports) below the axis: one band per carrier, in the
            # carrier's own colour, mirroring the order above it.
            _below_axis_stack(
                ax, month_rows, index=index, aggfunc=aggfunc,
                labelled=set(pivot.columns), handles=handles,
            )
            for name, label in (("load", "load"), ("unserved", "unserved")):
                hourly = (
                    month_rows[month_rows["series"] == name]
                    .groupby(["hour", "hour_of_day"], as_index=False)["value_gw"]
                    .sum()
                    .groupby("hour_of_day")["value_gw"]
                    .mean()
                )
                if hourly.empty or not float(hourly.abs().max()):
                    continue
                if name == "load":
                    artist = ax.plot(hourly.index, hourly.to_numpy(),
                                     color=style.carrier_color(name), linewidth=1.2,
                                     label=label)[0]
                else:
                    artist = ax.scatter(hourly.index, hourly.to_numpy(), s=10,
                                        color=style.carrier_color(name), label=label, zorder=5)
                handles.setdefault(label, artist)

        def title_of(keys, group):
            _run_id, label, meth, size = keys
            return (
                "O4 - dispatch by carrier, mean 24 h profile by month\n"
                f"{_facet_title(label, meth, size)}"
            )

        return _monthly_figure(groups, title_of=title_of, ylabel=ylabel, draw=draw)

    fig, axes = _facet_axes(len(groups))
    for ax, ((_run_id, label, meth, size), group) in zip(axes, groups):
        up = group[group["series"].isin(up_series)]
        pivot = _stack_pivot(up, "value_gw", "hour", "sum")
        _stack(ax, pivot, _facet_title(label, meth, size), ylabel)
        # Charging (and exports) below the axis: one band per carrier, in the
        # carrier's own colour, mirroring the order above it.
        _below_axis_stack(ax, group, index="hour", aggfunc="sum", labelled=set(pivot.columns))
        load = group[group["series"] == "load"].groupby("hour")["value_gw"].sum()
        if not load.empty:
            ax.plot(load.index, load.to_numpy(), color=style.carrier_color("load"),
                    linewidth=1.2, label="load")
        shed = group[group["series"] == "unserved"].groupby("hour")["value_gw"].sum()
        shed = shed[shed > 0]
        if not shed.empty:
            ax.scatter(shed.index, shed.to_numpy(), s=12,
                       color=style.carrier_color("unserved"), label="unserved", zorder=5)
        _outside_legend(ax)
    axes[-1].set_xlabel("hour of the weather year")
    style.finish(fig, "O4 - dispatch by carrier")
    return fig


# ---------------------------------------------------------------------------
# O4b -- dispatch on each month's peak-net-load day
# ---------------------------------------------------------------------------


def _peak_net_load_days(runs, *, window, year, method, block_size) -> pd.DataFrame:
    """Per (run, month): the local day whose peak net load is the month's highest.

    Net load is ``load - available VRE``, the same definition O2 plots, so the
    choice of day is a property of the weather and the demand and **not** of the
    solve: every facet of a run shows the same day, which is what makes the
    facets comparable.  Ties go to the earliest day.
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
    totals = local_calendar(_series_totals(frame))
    totals["net_load_gw"] = style.convert(totals["net_load_mw"], "power")
    daily = (
        totals.sort_values(["run_id", "day_index", "net_load_gw", "hour"],
                           ascending=[True, True, False, True])
        .groupby(["run_id", "month", "day_index"], as_index=False)
        .first()
    )
    best = (
        daily.sort_values(["run_id", "month", "net_load_gw", "day_index"],
                          ascending=[True, True, False, True])
        .groupby(["run_id", "month"], as_index=False)
        .first()
    )
    return best.rename(
        columns={"net_load_gw": "peak_net_load_gw", "hour_of_day": "peak_net_load_hour"}
    )[["run_id", "month", "day_index", "day", "peak_net_load_gw", "peak_net_load_hour"]]


def _net_load_rows(runs, *, window, year, method, block_size) -> pd.DataFrame:
    """The hourly net-load series of each run, as ``series="net_load"`` rows."""
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
    rows = totals[["run_id", "label", "year", "hour"]].copy()
    rows["series"] = "net_load"
    rows["carrier"] = "net_load"
    rows["value_gw"] = style.convert(totals["net_load_mw"], "power")
    return rows


@register(
    "O4b",
    title="Dispatch on each month's peak-net-load day",
    tier="debug",
    needs=("hourly",),
    columns=(
        "run_id",
        "label",
        "method",
        "block_size",
        "year",
        "month",
        "day",
        "day_index",
        "hour",
        "hour_of_day",
        "series",
        "carrier",
        "value_gw",
        "peak_net_load_gw",
        "peak_net_load_hour",
    ),
)
def o4b_peak_day_dispatch(runs, *, window=None, year=None, method=None, block_size=None, **_):
    """The **actual hourly** dispatch of the worst day of each month, in a 3x4 grid.

    O4's monthly grid averages every day of a month, which is the right picture
    of a typical day and the wrong one of a hard day.  This draws one real day
    per month: the local-Pacific day whose peak net load (load minus available
    solar and wind, as in O2) is the month's highest.  Same stacking order, same
    per-carrier charging below the axis, with load *and* net load overlaid.
    """
    selection = _peak_net_load_days(
        runs, window=window, year=year, method=method, block_size=block_size
    )
    dispatch = local_calendar(
        _o4_table(runs, window=window, year=year, method=method, block_size=block_size)
    )
    net_load = local_calendar(
        _net_load_rows(runs, window=window, year=year, method=method, block_size=block_size)
    )
    # The net-load line is a property of the run, so it is repeated once per
    # (method, block_size) facet of that run.
    facets = dispatch[["run_id", "label", "method", "block_size"]].drop_duplicates()
    net_load = net_load.drop(columns=["label"]).merge(facets, on="run_id", how="inner")
    combined = pd.concat([dispatch, net_load], ignore_index=True)

    table = combined.merge(selection, on=["run_id", "month", "day_index", "day"], how="inner")
    if table.empty:
        raise MissingDataError(
            runs[0].run_id,
            "hourly",
            "O4b found no hour on a peak-net-load day; the window carries no full local day",
        )
    order = ["run_id", "method", "block_size", "month", "hour", "series", "carrier"]
    table = table.sort_values(order).reset_index(drop=True)
    table = table[list(_columns("O4b"))]

    groups = list(table.groupby(["run_id", "label", "method", "block_size"], sort=True))
    up_series = ("generation", "storage_discharge")

    def draw(ax, month_rows, handles):
        up = month_rows[month_rows["series"].isin(up_series)]
        pivot = _stack_pivot(up, "value_gw", "hour_of_day", "sum")
        if not pivot.empty:
            polygons = ax.stackplot(
                pivot.index.to_numpy(),
                [pivot[c].to_numpy() for c in pivot.columns],
                colors=[style.carrier_color(c) for c in pivot.columns],
                labels=list(pivot.columns),
                linewidth=0.0,
            )
            for carrier, polygon in zip(pivot.columns, polygons):
                handles.setdefault(carrier, polygon)
        _below_axis_stack(
            ax, month_rows, index="hour_of_day", aggfunc="sum",
            labelled=set(pivot.columns), handles=handles,
        )
        for name, label in (("load", "load"), ("net_load", "net load"), ("unserved", "unserved")):
            series = (
                month_rows[month_rows["series"] == name]
                .groupby("hour_of_day")["value_gw"]
                .sum()
            )
            if series.empty or not float(series.abs().max()):
                continue
            if name == "unserved":
                artist = ax.scatter(series.index, series.to_numpy(), s=10,
                                    color=style.carrier_color(name), label=label, zorder=5)
            else:
                artist = ax.plot(
                    series.index, series.to_numpy(), color=style.carrier_color(name),
                    linewidth=1.3, linestyle="-" if name == "load" else "--", label=label,
                )[0]
            handles.setdefault(label, artist)
        row = month_rows.iloc[0]
        # Two lines: the one-line form of this title is wider than a panel and
        # ran into its neighbour's.
        ax.set_title(
            f"{MONTH_LABELS[int(row['month']) - 1]} - day {int(row['day']):02d}\n"
            f"peak net load {float(row['peak_net_load_gw']):.1f} GW "
            f"at {int(row['peak_net_load_hour']):02d}:00",
            fontsize=7.5,
        )

    def title_of(keys, group):
        _run_id, label, meth, size = keys
        return (
            "O4b - dispatch on each month's peak-net-load day\n"
            f"{_facet_title(label, meth, size)}"
        )

    fig = _monthly_figure(
        groups, title_of=title_of, ylabel=f"power [{style.unit_label('power')}]", draw=draw
    )
    return fig, table


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
    pivot = _stack_pivot(
        table[table["series"] == "generation"], "diff_gw", "hour", "sum"
    )
    for carrier in pivot.columns:
        ax.plot(pivot.index, pivot[carrier], linewidth=1.0,
                color=style.carrier_color(carrier), label=carrier)
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_xlabel("hour of the weather year")
    ax.set_ylabel(f"{a.label} - {b.label} [{style.unit_label('power')}]")
    ax.set_title(f"O5 - dispatch difference: {a.label} minus {b.label}")
    _outside_legend(ax)
    style.finish(fig)
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
        # One line per *display* carrier: the battery rows sum into BESS.
        shown = group.assign(carrier=style.display_carriers(group["carrier"]))
        shown = shown.groupby(["carrier", "hour"], as_index=False)["soc_gwh"].sum()
        for carrier in style.stack_order(shown["carrier"].unique()):
            sub = shown[shown["carrier"] == carrier].sort_values("hour")
            ax.plot(sub["hour"], sub["soc_gwh"], linewidth=1.1,
                    color=style.carrier_color(carrier), label=carrier)
        for hour in group.loc[group["block_boundary"], "hour"].unique():
            ax.axvline(hour, color="0.75", linewidth=0.6, zorder=0)
        ax.set_title(_facet_title(label, meth, size))
        ax.set_ylabel(f"SoC [{style.unit_label('energy')}]")
        _outside_legend(ax)
    axes[-1].set_xlabel("hour of the weather year")
    style.finish(fig, "O6 - storage state of charge")
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

    One (method, block_size) per run (``pick_series``, as O2 / O3): a run that
    solved three block sizes used to draw each line three times under tick labels
    that did not say which solve was which.  The kept series is named in the axis
    title.

    The county **map** (bus ``x`` / ``y`` are in ``system_static.json``) is out
    of scope: z4 has four nodes and a map of it says nothing.
    """
    if as_map:
        raise NotImplementedError(
            "O7 as_map=True (the geographic line-loading map) is out of scope for phase A; "
            "system_static.json already carries the bus coordinates it will need "
            "(see memory/plans/2026-09-09-plots-spec.md section 7)"
        )
    frame = pick_series(
        collect_hourly(
            runs,
            ["line_flow_mw"],
            window=window,
            year=year,
            method=method,
            block_size=block_size,
        )
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
    solves = ", ".join(
        f"{r.label}: {r.method} / {_block_label(r.block_size)}"
        for r in table.drop_duplicates(["run_id"]).itertuples()
    )
    ax.set_title(f"O7 - line loading ({solves})", fontsize=8)
    ax.legend(fontsize=7)
    style.finish(fig)
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

    The left panel is the pooled CDF of every (method, block_size) -- comparing
    them *is* the plot.  The right panel is one bar per (bus, run), hue = run,
    from one representative solve per run (``pick_series``): drawing a bar per
    (bus, method, block_size) repeated every bus under identical tick labels.
    Every method's per-bus quantile stays in the CSV.
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
    # The representative (method, block_size) of each run, for the per-bus panel.
    representative = {
        row.run_id: (row.method, row.block_size)
        for row in pick_series(frame)[["run_id", "method", "block_size"]]
        .drop_duplicates()
        .itertuples()
    }

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
        # Colour is the run; linestyle and marker separate that run's solves, which
        # otherwise shared one style and could not be told apart in the legend.
        for j, ((meth, size), sub) in enumerate(
            group.groupby(["method", "block_size"], sort=True)
        ):
            ax_cdf.plot(
                np.maximum(sub["price_usd_per_mwh"], 1e-3),
                sub["quantile"],
                color=st["color"],
                linestyle=style.RUN_STYLES[j % len(style.RUN_STYLES)],
                marker=style.RUN_MARKERS[j % len(style.RUN_MARKERS)],
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
    per_bus = per_bus[
        [
            representative.get(r.run_id) == (r.method, r.block_size)
            for r in per_bus.itertuples()
        ]
    ]
    if not per_bus.empty:
        buses = sorted(set(per_bus["bus"]))
        run_ids = sorted(set(per_bus["run_id"]))
        positions = np.arange(len(buses), dtype=float)
        width = 0.8 / max(1, len(run_ids))
        for i, run_id in enumerate(run_ids):
            sub = per_bus[per_bus["run_id"] == run_id].set_index("bus")
            meth, size = representative[run_id]
            heights = [float(sub["price_usd_per_mwh"].get(b, np.nan)) for b in buses]
            ax_box.bar(
                positions + (i - (len(run_ids) - 1) / 2) * width,
                heights,
                width=width,
                color=style.run_style(i)["color"],
                label=f"{sub['label'].iloc[0]} {meth}/{size}",
            )
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(buses, rotation=0, fontsize=7)
        if len(run_ids) > 1:
            ax_box.legend(fontsize=7)
    ax_box.set_xlabel("load bus")
    ax_box.set_ylabel("median price [$/MWh]")
    ax_box.set_title("per load bus (one representative solve per run)")
    style.finish(fig, "O8 - load-bus prices")
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
    style.finish(fig, "O9 - per-block metrics")
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
    """Primal / dual residuals against their tolerances, objective below them.

    Each block gets **two stacked panels sharing one x axis**: the residuals (log
    y, $MW$ / $MWh$) above and the objective (linear y, solver units) below.  The
    objective used to be drawn on ``ax.twinx()``, i.e. a second y axis on the same
    panel -- two scales in one frame, which the project's style rules out because
    the crossing point of two such curves is an artefact of the scaling.

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
    # Two rows per block -- residuals (log y) then objective (linear y) -- sharing
    # the block's iteration axis. No twin axes anywhere.
    fig, axes = plt.subplots(
        2 * len(groups),
        1,
        figsize=(9.5, 3.6 * len(groups)),
        squeeze=False,
        gridspec_kw={"height_ratios": [2.0, 1.0] * len(groups)},
    )
    axes = list(axes[:, 0])
    for position, ((_run_id, label, size, index), group) in enumerate(groups):
        ax = axes[2 * position]
        ax_obj = axes[2 * position + 1]
        ax_obj.sharex(ax)
        ax.semilogy(group["iteration"], group["primal_power"].abs(), color="#1f5f8b",
                    label="primal power")
        ax.semilogy(group["iteration"], group["dual_power"].abs(), color="#d2691e",
                    label="dual power")
        ax.semilogy(group["iteration"], group["primal_tol"].abs(), color="#1f5f8b",
                    linestyle=":", label="primal tol")
        ax.semilogy(group["iteration"], group["dual_tol"].abs(), color="#d2691e",
                    linestyle=":", label="dual tol")
        ax.set_title(f"{label} - block {index} ({_block_label(size)})")
        ax.set_ylabel("residual")
        ax.tick_params(labelbottom=False)
        _outside_legend(ax)
        ax_obj.plot(group["iteration"], group["objective"], color="0.35", linewidth=0.9,
                    label="objective")
        ax_obj.set_ylabel("objective\n[solver units]", fontsize=7)
        ax_obj.set_xlabel("ADMM iteration")
    style.finish(fig, "O10 - ADMM convergence")
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
    # Log x: 24 h, 168 h and a full year (8,736 h) are three decades apart, and on
    # a linear axis the two block sizes collapse onto each other whenever a
    # full-year point shares the axis.
    for ax in (ax_time, ax_cpu):
        ax.set_xscale("log")
        ax.set_xlabel("hours per block (log)")
        ax.set_xticks(sorted(set(table["hours_per_block"].dropna())))
        ax.get_xaxis().set_major_formatter(FormatStrFormatter("%g"))
        ax.get_xaxis().set_minor_formatter(NullFormatter())
        ax.legend(fontsize=7)
    ax_time.set_yscale("log")
    ax_time.set_ylabel("mean solve time [s] (log)")
    ax_cpu.set_ylabel("CPU-seconds")
    style.finish(fig, "O11 - solve time")
    return fig, table


# ---------------------------------------------------------------------------
# O12 / O13 -- ADMM dual error vs the reference LP, per (bus, hour)
# ---------------------------------------------------------------------------


def _price_error(runs, *, block_size=None, load_buses_only=True) -> pd.DataFrame:
    """`price_error.parquet` of every run, concatenated and filtered."""
    frames = []
    for run in runs:
        frame = run.price_error()
        frames.append(frame.assign(run_id=run.run_id, label=run.label))
    table = pd.concat(frames, ignore_index=True)
    if load_buses_only and "is_load_bus" in table.columns:
        table = table[table["is_load_bus"].astype(bool)]
    if block_size is not None:
        table = table[table["block_size"].astype(str) == str(block_size)]
    return table


@register(
    "O12",
    title="ADMM dual error heat map (bus x hour)",
    tier="debug",
    needs=("price_error",),
    columns=(
        "run_id",
        "label",
        "block_size",
        "block_start_hour",
        "bus",
        "hour",
        "delta_price",
        "delta_price_vs_block_lp",
    ),
)
def o12_price_error_heatmap(runs, *, block_size=None, **_):
    """ADMM minus the **same-block** LP price, per (bus, hour). Load buses only.

    One panel per block, every block size unless `block_size=168` (or 24) narrows
    it. The 168 h blocks are the interesting ones: on `ca2040_z4` at `rho 0.1` the
    load-bus dual error plateaus at ~3 $/MWh while the power residuals keep
    falling, so the question is *where* it sits -- which buses, and which hours of
    the block.

    The colour is `delta_price_vs_block_lp` (the pure solver error), not
    `delta_price` (the distance to the reference solve): the blocked *LP* is just
    as far from the reference as ADMM is, because that distance is a blocking
    error. Both columns are in the CSV. Non-load buses are excluded because the
    import and export buses price degenerately (a flat 56-61 $/MWh at every
    iteration) and would set the colour scale.
    """
    table = _price_error(runs, block_size=block_size)
    table = table[list(_columns("O12"))].sort_values(
        ["run_id", "block_start_hour", "bus", "hour"]
    )
    table = table.reset_index(drop=True)
    quantity = (
        "delta_price_vs_block_lp"
        if table["delta_price_vs_block_lp"].notna().any()
        else "delta_price"
    )

    groups = list(table.groupby(["run_id", "label", "block_start_hour"], sort=True))
    if not groups:
        raise MissingDataError(
            runs[0].run_id,
            "price_error",
            f"no load-bus price-error rows for block_size {block_size!r}",
        )
    limit = float(np.nanmax(np.abs(table[quantity].to_numpy(dtype=float)))) or 1.0
    fig, axes = _facet_axes(len(groups), height=2.6, sharex=False)
    for ax, ((_run_id, label, start), group) in zip(axes, groups):
        pivot = group.pivot_table(
            index="bus", columns="hour", values=quantity, aggfunc="mean", observed=True
        )
        image = ax.pcolormesh(
            pivot.columns.to_numpy(dtype=float),
            np.arange(pivot.shape[0], dtype=float),
            pivot.to_numpy(),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="nearest",
        )
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(list(pivot.index), fontsize=7)
        ax.set_title(
            f"{label} - block at hour {start} "
            f"({_block_label(group['block_size'].iloc[0])})"
        )
        fig.colorbar(image, ax=ax, label="$/MWh")
    axes[-1].set_xlabel("absolute hour of the weather year")
    reference = "same-block LP" if quantity == "delta_price_vs_block_lp" else "reference LP"
    style.finish(fig, f"O12 - ADMM minus {reference} price, load buses")
    return fig, table


@register(
    "O13",
    title="ADMM dual error per hour, by block size",
    tier="report",
    needs=("price_error",),
    columns=(
        "run_id",
        "label",
        "block_size",
        "hour",
        "hour_of_day",
        "hours_into_block",
        "max_abs_delta_price",
        "mean_abs_delta_price",
        "max_abs_delta_price_vs_block_lp",
        "n_buses",
    ),
)
def o13_price_error_per_hour(runs, **_):
    """Max |ADMM - same-block LP| over load buses, per hour, one series per block size.

    Two panels: against the absolute hour (does the error sit at particular hours
    of the record?) and against the hour *into the block* (is it a boundary effect
    of the block's own cyclic storage condition?). 24 h against 168 h on the same
    hours is the comparison that separates the solver from the block length.

    Plotted against the **same-block** LP. `max_abs_delta_price` (the distance to
    the reference solve) is kept in the CSV, but it is a blocking error: a blocked
    LP shows the same number.
    """
    table = _price_error(runs)
    rows = []
    for keys, group in table.groupby(["run_id", "label", "block_size", "hour"], sort=True):
        run_id, label, size, hour = keys
        delta = pd.to_numeric(group["delta_price"], errors="coerce").abs()
        block_delta = pd.to_numeric(
            group.get("delta_price_vs_block_lp", pd.Series(dtype=float)), errors="coerce"
        ).abs()
        start = int(group["block_start_hour"].iloc[0])
        rows.append(
            {
                "run_id": run_id,
                "label": label,
                "block_size": size,
                "hour": int(hour),
                # The exports are UTC hours; the shipped window starts at hour 7 so
                # blocks begin at Pacific midnight (see persist.ENS_PROFILE_COLUMNS).
                "hour_of_day": int(hour) % 24,
                "hours_into_block": int(hour) - start,
                "max_abs_delta_price": float(delta.max()),
                "mean_abs_delta_price": float(delta.mean()),
                "max_abs_delta_price_vs_block_lp": float(block_delta.max())
                if block_delta.notna().any()
                else float("nan"),
                "n_buses": len(group),
            }
        )
    out = pd.DataFrame(rows, columns=list(_columns("O13")))
    if out.empty:
        raise MissingDataError(runs[0].run_id, "price_error", "no load-bus price-error rows")

    quantity = (
        "max_abs_delta_price_vs_block_lp"
        if out["max_abs_delta_price_vs_block_lp"].notna().any()
        else "max_abs_delta_price"
    )
    fig, (ax_abs, ax_into) = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for i, ((_run_id, label, size), group) in enumerate(
        out.groupby(["run_id", "label", "block_size"], sort=True)
    ):
        st = style.run_style(i)
        ax_abs.plot(
            group["hour"],
            group[quantity],
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=1.0,
            label=f"{label} {_block_label(size)}",
        )
        into = group.groupby("hours_into_block")[quantity].max()
        ax_into.plot(
            into.index,
            into.to_numpy(),
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=1.0,
            label=f"{label} {_block_label(size)}",
        )
    reference = "same-block LP" if quantity.endswith("block_lp") else "reference LP"
    ax_abs.set_xlabel("absolute hour of the weather year")
    ax_abs.set_ylabel(f"max |ADMM - {reference}| [$/MWh]")
    ax_abs.legend(fontsize=7)
    ax_into.set_xlabel("hours into the block")
    ax_into.set_ylabel(f"max |ADMM - {reference}| [$/MWh]")
    ax_into.legend(fontsize=7)
    style.finish(fig, f"O13 - ADMM dual error per hour vs the {reference}, load buses")
    return fig, out


def _columns(plot_id: str) -> tuple[str, ...]:
    from . import PLOTS

    return PLOTS[plot_id].columns
