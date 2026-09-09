"""Weather-store diagnostics: demand and available capacity by weather year.

Self-contained on purpose -- it imports only from :mod:`zap.importers.wy_store`
so that the rest of the ``experiments.ra`` package can evolve independently.

For every dataset and every weather year in its ``weather.zarr`` it computes,
from the weather store alone (no outage draw, no UCAP derate):

* total annual demand (TWh),
* peak demand (GW),
* peak available capacity (GW) in three flavours -- the headline in-state
  generators-only number, plus the incl-storage and incl-imports denominators.

The metric computation takes an *availability array*, never the store, so a
later ``--outage-draw N`` / ``--ucap`` flag can produce the same table and
figures for a derated system by passing a different array (or by letting
``load_system`` compose it and reading ``Generator.dynamic_capacity``).

Usage::

    python -m experiments.ra.diagnostics --datasets ca2040_z4 ca2040_county \
        --out /path/to/figures/phase1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from zap.importers.wy_store import (  # noqa: E402
    HourWindow,
    LoadOptions,
    LoadedSystem,
    WeatherStore,
    available_capacity,
    load_system,
)

#: One colour per dataset, used in every figure.
DATASET_COLORS = {"ca2040_z4": "#1f5f8b", "ca2040_county": "#d2691e"}
FALLBACK_COLORS = ["#4c7a4c", "#7b3f8c", "#8b1f1f"]

GRID_KWARGS = dict(color="0.85", linewidth=0.6)

#: Fraction of one weather-year slot covered by a group of bars.
GROUP_WIDTH = 0.8
#: Background behind direct labels, so they stay readable over bars and lines.
LABEL_BBOX = {"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.8}
#: On-screen gap between the bars of one year, in device pixels.
BAR_GAP_PX = 2.0

#: Annual demand and peak load are the same numbers in both resolutions
#: (county is a spatial disaggregation of the same load); both bars are kept so
#: the layout matches the other figures, and the note says so on the figure.
IDENTICAL_NOTE = (
    "Series are identical by construction: ca2040_county is a spatial "
    "disaggregation of the same load."
)

FIGURES = [
    ("demand_twh", "demand_twh", "Annual demand (TWh)", IDENTICAL_NOTE),
    ("peak_load_gw", "peak_load_gw", "Peak demand (GW)", IDENTICAL_NOTE),
    ("peak_available_gw", "peak_avail_gw", "Peak available capacity (GW)", None),
]

RATIO_FIGURE = ("peak_load_over_avail", "peak_load_over_avail", "Peak load / peak available")

#: Dashed horizontal references drawn on the ratio figure, as (value, label).
RATIO_REFERENCES = ((0.85, "0.85"), (1.0, "1.0"))

CSV_COLUMNS = [
    "dataset",
    "weather_year",
    "demand_twh",
    "peak_load_gw",
    "peak_avail_gw",
    "peak_avail_incl_storage_gw",
    "peak_avail_incl_imports_gw",
    "peak_load_over_avail",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def default_data_root() -> Path:
    env = os.environ.get("CH3_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data"


def year_metrics(
    system: LoadedSystem, *, availability: Optional[np.ndarray] = None
) -> dict[str, float]:
    """Demand and available-capacity metrics for one loaded weather year.

    ``availability`` is an optional ``(n_hours, n_gen)`` override; leaving it
    ``None`` uses the system's own ``Generator.dynamic_capacity``, which already
    carries any outage draw or UCAP derate that ``load_system`` composed.
    """
    load = system.index.get(system.devices, "Load")
    demand = np.asarray(load.load, dtype=np.float64)

    common = dict(availability=availability)
    hourly = available_capacity(system, **common)
    with_storage = available_capacity(system, include_storage=True, **common)
    with_imports = available_capacity(system, include_imports=True, **common)

    peak_load_gw = float(demand.sum(axis=0).max()) / 1e3
    peak_avail_gw = float(hourly.max()) / 1e3

    return {
        "demand_twh": float(demand.sum()) / 1e6,
        "peak_load_gw": peak_load_gw,
        "peak_avail_gw": peak_avail_gw,
        "peak_avail_incl_storage_gw": float(with_storage.max()) / 1e3,
        "peak_avail_incl_imports_gw": float(with_imports.max()) / 1e3,
        "peak_load_over_avail": peak_load_gw / peak_avail_gw if peak_avail_gw > 0 else np.nan,
    }


def dataset_table(
    dataset_dir: Path,
    *,
    years: Optional[Sequence[int]] = None,
    ucap_derate: bool = False,
    outage_draw: Optional[int] = None,
) -> pd.DataFrame:
    """One row per weather year of ``dataset_dir``."""
    store = WeatherStore.open(dataset_dir)
    selected = [int(y) for y in (years if years is not None else store.years)]

    rows = []
    for year in selected:
        options = LoadOptions(
            years=(year,),
            window=HourWindow(0, store.hours_per_year),
            demand_scaling="none",
            ucap_derate=ucap_derate,
            outage_draw=outage_draw,
        )
        system = load_system(dataset_dir, options)
        row = {"dataset": dataset_dir.name, "weather_year": year}
        row.update(year_metrics(system))
        rows.append(row)
        print(
            f"  {dataset_dir.name} {year}: demand {row['demand_twh']:.1f} TWh, "
            f"peak load {row['peak_load_gw']:.2f} GW, "
            f"peak available {row['peak_avail_gw']:.2f} GW"
        )

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _color(dataset: str, order: int) -> str:
    return DATASET_COLORS.get(dataset, FALLBACK_COLORS[order % len(FALLBACK_COLORS)])


def _resize_bars(fig, ax, series) -> None:
    """Give every year's bars a fixed on-screen gap of ``BAR_GAP_PX`` pixels."""
    n_series = len(series)
    if n_series == 0:
        return
    fig.canvas.draw()
    x_lo, x_hi = ax.get_xlim()
    width_px = ax.get_window_extent().width
    if width_px <= 0 or x_hi <= x_lo:
        return
    gap = BAR_GAP_PX * (x_hi - x_lo) / width_px
    bar_width = (GROUP_WIDTH - (n_series - 1) * gap) / n_series
    if bar_width <= 0:
        return
    for order, item in enumerate(series):
        offset = -GROUP_WIDTH / 2 + order * (bar_width + gap)
        for year, bar in zip(item["x"], item["bars"]):
            bar.set_x(float(year) + offset)
            bar.set_width(bar_width)


def _annotate_extremes(ax, item, order: int) -> None:
    """Label only the tallest and shortest bar of one series.

    ``order`` lifts each series' labels onto its own row so that two series
    whose extremes fall on neighbouring years do not overprint each other.
    """
    x, y, bars = item["x"], item["y"], item["bars"]
    if len(y) == 0:
        return
    for index in {int(np.argmax(y)), int(np.argmin(y))}:
        bar = bars[index]
        ax.annotate(
            f"{int(x[index])}: {y[index]:,.4g}",
            xy=(bar.get_x() + bar.get_width() / 2, y[index]),
            xytext=(0, 3 + 11 * order),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=item["color"],
            bbox=LABEL_BBOX,
        )


def _figure(
    table: pd.DataFrame,
    column: str,
    ylabel: str,
    path: Path,
    references: Sequence[tuple[float, str]] = (),
    note: str | None = None,
) -> Path:
    """Grouped bar chart of ``column``: one bar per dataset per weather year."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", **GRID_KWARGS)

    series = []
    for order, (dataset, group) in enumerate(table.groupby("dataset", sort=True)):
        group = group.sort_values("weather_year")
        x = group["weather_year"].to_numpy()
        y = group[column].to_numpy(dtype=float)
        color = _color(dataset, order)
        bars = ax.bar(x.astype(float), y, width=GROUP_WIDTH, color=color, label=dataset)
        series.append({"x": x, "y": y, "color": color, "bars": list(bars)})

    years = table["weather_year"].to_numpy(dtype=float)
    ax.set_xlim(years.min() - 0.6, years.max() + 0.6)
    ax.set_xticks(sorted(set(years.tolist())))
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)

    top = max(float(np.nanmax(item["y"])) for item in series)
    top = max([top, *(value for value, _ in references)])
    ax.set_ylim(0.0, top * 1.22)

    for value, label in references:
        ax.axhline(value, linestyle="--", linewidth=1.0, color="0.35")
        ax.annotate(
            label,
            xy=(1.0, value),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 1),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
            bbox=LABEL_BBOX,
        )

    ax.set_xlabel("weather year")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="upper left", ncols=len(series))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    bottom = 0.06 if note else 0.0
    fig.tight_layout(rect=(0.0, bottom, 1.0, 1.0))
    if note:
        fig.text(0.5, 0.015, note, ha="center", fontsize=8, color="0.35")

    _resize_bars(fig, ax, series)
    for order, item in enumerate(series):
        _annotate_extremes(ax, item, order)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_figures(table: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for stem, column, ylabel, note in FIGURES:
        paths.append(_figure(table, column, ylabel, out_dir / f"{stem}.png", note=note))
    stem, column, ylabel = RATIO_FIGURE
    paths.append(
        _figure(table, column, ylabel, out_dir / f"{stem}.png", references=RATIO_REFERENCES)
    )
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.ra.diagnostics")
    parser.add_argument("--datasets", nargs="+", default=["ca2040_z4", "ca2040_county"])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--years", type=int, nargs="*", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--ucap",
        action="store_true",
        help="Derate availability with ucap.csv (requires WP2's ucap.csv).",
    )
    parser.add_argument(
        "--outage-draw",
        type=int,
        default=None,
        help="Derate availability with one draw of outages.zarr.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.data_root if args.data_root is not None else default_data_root()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    for name in args.datasets:
        dataset_dir = Path(name)
        if not dataset_dir.is_absolute() and not (dataset_dir / "weather.zarr").exists():
            dataset_dir = root / name
        print(f"{dataset_dir}")
        tables.append(
            dataset_table(
                dataset_dir,
                years=args.years,
                ucap_derate=args.ucap,
                outage_draw=args.outage_draw,
            )
        )

    table = pd.concat(tables, ignore_index=True)
    csv_path = out_dir / "availability_by_year.csv"
    table.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    for path in write_figures(table, out_dir):
        print(f"Wrote {path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
