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

#: Line style per dataset. Annual demand is identical in the two resolutions
#: (county is a spatial disaggregation of the same load), so the series overlap
#: exactly in some figures; distinct dashes keep both visible.
DATASET_STYLES = {"ca2040_z4": ("-", "o"), "ca2040_county": ("--", "s")}
FALLBACK_STYLE = ("-.", "^")

GRID_KWARGS = dict(color="0.85", linewidth=0.6)

FIGURES = [
    ("demand_twh", "demand_twh", "Annual demand (TWh)"),
    ("peak_load_gw", "peak_load_gw", "Peak demand (GW)"),
    ("peak_available_gw", "peak_avail_gw", "Peak available capacity (GW)"),
]

RATIO_FIGURE = ("peak_load_over_avail", "peak_load_over_avail", "Peak load / peak available")

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


def _annotate_extremes(ax, x, y, color):
    if len(x) == 0:
        return
    for index, va, dy in ((int(np.argmax(y)), "bottom", 6), (int(np.argmin(y)), "top", -6)):
        ax.annotate(
            f"{int(x[index])}: {y[index]:,.4g}",
            xy=(x[index], y[index]),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            color=color,
        )


def _figure(table: pd.DataFrame, column: str, ylabel: str, path: Path, reference=None) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.set_axisbelow(True)
    ax.grid(True, **GRID_KWARGS)

    for order, (dataset, group) in enumerate(table.groupby("dataset", sort=True)):
        group = group.sort_values("weather_year")
        x = group["weather_year"].to_numpy()
        y = group[column].to_numpy(dtype=float)
        color = _color(dataset, order)
        linestyle, marker = DATASET_STYLES.get(dataset, FALLBACK_STYLE)
        ax.plot(
            x,
            y,
            marker=marker,
            markersize=4.5,
            markerfacecolor="none" if linestyle != "-" else color,
            linestyle=linestyle,
            linewidth=1.6,
            color=color,
            label=dataset,
        )
        _annotate_extremes(ax, x, y, color)

    if reference is not None:
        ax.axhline(reference, linestyle="--", linewidth=1.0, color="0.35")
        ax.annotate(
            f"{reference:g}",
            xy=(table["weather_year"].max(), reference),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=8,
            color="0.35",
        )

    ax.margins(x=0.04, y=0.14)  # headroom for the min/max labels
    ax.set_xlabel("weather year")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_figures(table: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for stem, column, ylabel in FIGURES:
        paths.append(_figure(table, column, ylabel, out_dir / f"{stem}.png"))
    stem, column, ylabel = RATIO_FIGURE
    paths.append(_figure(table, column, ylabel, out_dir / f"{stem}.png", reference=0.85))
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
