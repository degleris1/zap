"""Colours, units and figure IO for the ``ra plot`` catalogue.

The colour map is **checked in**, seeded from the ``color`` column of
``data/ca2040_z4/static/carriers.csv`` (PyPSA-USA's own palette), so figures
match the upstream convention *and* a dataset edit cannot silently restyle a
chapter figure.  Nothing here reads ``data/`` at plot time.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

#: One colour per carrier.  The first block is the ``carriers.csv`` vocabulary of
#: the CA2040 datasets; the second is the pseudo-carriers the plots invent for
#: series that are not generation.
CARRIER_COLORS: dict[str, str] = {
    # --- thermal / firm ---
    "CCGT": "#b20101",
    "CCGT-95CCS": "#800000",
    "OCGT": "#d35050",
    "hydrogen_ct": "#ea048a",
    "coal": "#707070",
    "oil": "#262626",
    "nuclear": "#ff9000",
    "biomass": "#0c6013",
    "waste": "#68896b",
    "geothermal": "#ba91b1",
    "hydro": "#08ad97",
    # --- variable renewables ---
    "offwind_floating": "#11a1c1",
    "onwind": "#235ebc",
    "solar": "#f9d002",
    # --- storage ---
    # The three battery rows are a *duration ramp*, not three shades of one
    # green: upstream ships #b8ea04 / #a4d600 / #90c200, which differ by an
    # OKLab dE of 6 and are indistinguishable where they matter most (O6 draws
    # the 4 h battery and the generic battery as its two dominant series).
    # #b8ea04 (upstream, generic battery) is kept as the light end and the two
    # duration rows are re-stepped down in lightness; every pair of the three
    # now clears the data-viz gates (normal-vision OKLab dE >= 15, protan /
    # deutan dE >= 8) -- measured 24.3 / 23.3 (battery-4hr), 49.5 / 48.8
    # (battery-8hr), 25.4 / 25.3 (4hr-8hr) with the bundled `dataviz`
    # validate_palette.py (tested in test_ra_plots.py).  What no re-step
    # fixes here is the red/green
    # collapse the *rest* of this map already has under simulated protanopia
    # (battery vs solar dE 0.6, the CCGT reds vs any green ~3-6): that is a
    # property of the upstream PyPSA-USA palette, unchanged by this entry, and
    # a CVD-safe rebuild of all 30 carriers is a separate decision.
    # PHS is *not* upstream's teal any more: `hydro` is #08ad97 too, and the two
    # were one colour above and below the axis.  #7c70fc clears the data-viz
    # gates against every carrier it can sit next to -- hydro 26.1 normal /
    # 20.6 CVD, BESS 47.5 / 45.3, solar 44.7 / 42.3, onwind 15.4 / 12.0,
    # offwind_floating 18.5 / 13.2 (OKLab dE x100, min of protan and deutan),
    # measured with the bundled `dataviz` validate_palette.py.
    "PHS": "#7c70fc",
    #: The one battery colour: `battery`, `4hr_battery_storage`,
    #: `8hr_battery_storage` and anything else matching *battery* are merged into
    #: the display group `BESS` (:func:`display_carrier`), which keeps upstream's
    #: generic-battery hue.  The per-carrier entries below stay for readers that
    #: colour a raw carrier directly; no plot does.
    "BESS": "#b8ea04",
    "battery": "#b8ea04",
    "4hr_battery_storage": "#5aa02c",
    "8hr_battery_storage": "#185430",
    "demand_response": "#dd2e23",
    # --- trade and network (no colour upstream; assigned here) ---
    "imports": "#9a7fbd",
    "unspecified_imports": "#7e63a3",
    "exports": "#5a4a78",
    "AC": "#70af1d",
    "DC": "#8a1caf",
    "AC_exp": "#4f7d14",
    # --- device-class rows ---
    # `iterations/*.iteration_capacity.parquet` reports a *device class* rather
    # than a carrier for the parameters that are not per-generator (every
    # storage row is one `StorageUnit` series, every line one `DirectedLine`),
    # so P3 sees these two names where it sees carriers elsewhere.  They are
    # display categories with their own colours, not carriers.
    "StorageUnit": "#7fb3d5",
    "DirectedLine": "#70af1d",
    # --- pseudo-carriers ---
    "load": "#111111",
    "net_load": "#d2691e",
    "unserved": "#e3120b",
    "curtailment": "#c9a227",
    "storage_charge": "#7fb3d5",
    "storage_discharge": "#b8ea04",
}

#: The ``carriers.csv`` vocabulary of the CA2040 datasets plus the plots'
#: pseudo-carriers; every entry must have a colour (tested).
DATASET_CARRIERS: tuple[str, ...] = (
    "CCGT",
    "CCGT-95CCS",
    "OCGT",
    "hydrogen_ct",
    "coal",
    "oil",
    "nuclear",
    "biomass",
    "waste",
    "geothermal",
    "hydro",
    "offwind_floating",
    "onwind",
    "solar",
    "PHS",
    "BESS",
    "battery",
    "4hr_battery_storage",
    "8hr_battery_storage",
    "demand_response",
    "imports",
    "unspecified_imports",
    "exports",
    "AC",
    "DC",
    "AC_exp",
    "StorageUnit",
    "DirectedLine",
    "load",
    "net_load",
    "unserved",
    "curtailment",
    "storage_charge",
    "storage_discharge",
)

#: **The** stacking order for every stacked-by-carrier plot, bottom to top above
#: the x axis: baseload, then variable renewables, then batteries / storage
#: discharge, then thermal, with trade last.  One list, used by every stacked
#: plot *and* its legend, so the order cannot drift between figures (Kamran,
#: 2026-09-09).  Storage *charging* and exports are drawn **below** the axis --
#: also by carrier, in the carrier's own colour and in this same sequence, so
#: the band nearest the axis below is the same carrier as the band nearest it
#: above (see ``operational.BELOW_AXIS_SERIES``).  A hatch marks the charging
#: half; the hue does not change, so one legend entry covers both signs and
#: there is no separate "storage charge" entry.  A carrier not listed here is stacked on top, in
#: alphabetical order, and warns -- see :func:`stack_order`.
#:
#: ``demand_response`` is the one addition to the order as dictated: it is a
#: storage-class carrier in :data:`CARRIER_COLORS`, and leaving it out would put
#: it above the thermal block with a warning on every dispatch figure that has it.
CARRIER_STACK_ORDER: tuple[str, ...] = (
    # --- baseload ---
    "nuclear",
    "coal",
    "geothermal",
    "biomass",
    "waste",
    "hydro",
    "PHS",
    # --- variable renewables ---
    "solar",
    "onwind",
    "offwind_floating",
    # --- batteries / storage discharge ---
    # One slot for the whole battery fleet: plots group by *display* carrier
    # (:func:`display_carrier`), so the three battery rows arrive here as `BESS`.
    "BESS",
    "demand_response",
    # --- thermal ---
    "CCGT",
    "CCGT-95CCS",
    "OCGT",
    "hydrogen_ct",
    "oil",
    # --- trade ---
    "imports",
    "unspecified_imports",
    "exports",
    # --- network (P1/P2 carry line rows; no dispatch stack ever does) ---
    "AC",
    "AC_exp",
    "DC",
    # --- device-class rows (see CARRIER_COLORS) ---
    "StorageUnit",
    "DirectedLine",
)

#: Carriers whose available capacity is weather-driven (O2's VRE series).  Kept
#: in sync with ``zap.importers.wy_store.VRE_CARRIERS``.
VRE_CARRIERS = frozenset({"solar", "onwind", "offwind_floating"})

#: Colour + linestyle + marker per run position, so overlaid runs stay
#: distinguishable in print and where they overlap exactly.
RUN_COLORS = ("#1f5f8b", "#d2691e", "#4c7a4c", "#7b3f8c", "#8b1f1f", "#555555")
RUN_STYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)))
RUN_MARKERS = ("o", "s", "^", "D", "v", "P")

GRID_KWARGS = {"color": "0.85", "linewidth": 0.6}

#: Fixed hue order for the plots whose series are *quantities*, not runs (O2's
#: load / VRE / net load).  These three are never interchangeable, so they get
#: names rather than a palette position, and each one keeps its colour while the
#: run stays the linestyle -- one encoding per dimension, and no series that is
#: told apart only by a dash pattern at 8,736 points.
SERIES_COLORS = {
    "load": "#111111",
    "vre_available": "#1f5f8b",
    "net_load": "#d2691e",
}

#: Inches of head room reserved above the axes for a figure-level title.
SUPTITLE_INCHES = 0.45

#: ``kind -> (label, factor from the MW / $ / tonne base)``.  Every value column
#: of every CSV carries its unit as a suffix and holds the value **as plotted**.
UNITS = {
    "power": ("GW", 1e-3),
    "energy": ("GWh", 1e-3),
    "cost": ("B$", 1e-9),
    "price": ("$/MWh", 1.0),
    "emissions": ("Mt", 1e-6),
}

_WARNED: set[str] = set()


def carrier_color(name: str) -> str:
    """The checked-in colour of a carrier, or a deterministic grey-blue."""
    key = str(name)
    if key in CARRIER_COLORS:
        return CARRIER_COLORS[key]
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(
            "carrier %r has no entry in plots.style.CARRIER_COLORS; using a hashed fallback",
            key,
        )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=3).digest()
    # A muted grey-blue family, so an unknown carrier is visibly not a named one.
    r = 90 + digest[0] % 70
    g = 105 + digest[1] % 70
    b = 130 + digest[2] % 70
    return f"#{r:02x}{g:02x}{b:02x}"


#: Carriers merged into one **display** category before any plot colours,
#: stacks or labels them.  The merge happens at the plot layer only: every
#: persisted table (``raw_data/*.csv``, ``hourly.parquet``, ``design.json``)
#: keeps the per-carrier rows, so a number can still be read per battery type.
CARRIER_DISPLAY_GROUPS: dict[str, str] = {
    "battery": "BESS",
    "4hr_battery_storage": "BESS",
    "8hr_battery_storage": "BESS",
}

#: Any carrier whose name contains this (case-insensitively) and is not already
#: in :data:`CARRIER_DISPLAY_GROUPS` also merges into ``BESS`` -- the datasets
#: name duration variants freely (``2hr_battery_storage`` and friends) and a new
#: one must not silently become its own band.
BESS_MATCH = "battery"
BESS_GROUP = "BESS"


def display_carrier(name) -> str:
    """The display category of a raw carrier: batteries collapse to ``BESS``."""
    key = str(name)
    if key in CARRIER_DISPLAY_GROUPS:
        return CARRIER_DISPLAY_GROUPS[key]
    if BESS_MATCH in key.lower():
        return BESS_GROUP
    return key


def display_carriers(values):
    """:func:`display_carrier` over a pandas Series (or any iterable)."""
    if isinstance(values, pd.Series):
        return values.map(display_carrier)
    return [display_carrier(v) for v in values]


def stack_order(carriers) -> list[str]:
    """``carriers`` in :data:`CARRIER_STACK_ORDER`, unknown ones on top, sorted.

    Every stacked plot orders both its bands and its legend with this, so two
    figures of the same system always stack in the same order.  An unknown
    carrier is not dropped -- it goes on top and warns once, so a new carrier
    shows up in the figure *and* in the log rather than silently reordering it.
    """
    present = {str(c) for c in carriers}
    known = [c for c in CARRIER_STACK_ORDER if c in present]
    unknown = sorted(present.difference(CARRIER_STACK_ORDER))
    for name in unknown:
        key = f"stack:{name}"
        if key not in _WARNED:
            _WARNED.add(key)
            logger.warning(
                "carrier %r is not in plots.style.CARRIER_STACK_ORDER; stacking it on "
                "top (add it to the list to pin its position)",
                name,
            )
    return known + unknown


def run_style(position: int) -> dict:
    return {
        "color": RUN_COLORS[position % len(RUN_COLORS)],
        "linestyle": RUN_STYLES[position % len(RUN_STYLES)],
        "marker": RUN_MARKERS[position % len(RUN_MARKERS)],
    }


def convert(values, kind: str):
    return values * UNITS[kind][1]


def unit_label(kind: str) -> str:
    return UNITS[kind][0]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "plot"


def apply_rc() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_KWARGS["color"],
            "grid.linewidth": GRID_KWARGS["linewidth"],
            "legend.frameon": False,
        }
    )


def outside_legend(ax, **kwargs) -> None:
    """A legend in the right margin, so it cannot cover the data or a callout.

    :func:`save` writes with ``bbox_inches="tight"``, so the margin costs figure
    width, never plot area.
    """
    options = {"fontsize": 7, "loc": "upper left", "bbox_to_anchor": (1.005, 1.0),
               "borderaxespad": 0.0, "ncol": 1}
    options.update(kwargs)
    ax.legend(**options)


def finish(fig, suptitle: str | None = None, *, reserve_inches: float = SUPTITLE_INCHES):
    """``tight_layout``, reserving head room for ``suptitle`` on tall figures.

    ``tight_layout`` knows nothing about ``suptitle``, and a faceted figure's
    height grows with the facet count, so the default ``y=0.98`` lands *inside*
    the first facet's title as soon as there are six or more facets.  Reserving
    a fixed number of *inches* -- a figure fraction that shrinks as the figure
    grows -- keeps the gap the same on a 1-facet and a 9-facet figure.
    """
    if suptitle:
        height = float(fig.get_figheight()) or 1.0
        reserve = min(0.30, float(reserve_inches) / height)
        fig.suptitle(suptitle, y=1.0 - 0.30 * reserve)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - reserve))
    else:
        fig.tight_layout()
    return fig


def save(
    fig, table: pd.DataFrame, plot_id: str, out_dir, stem: str, data_dir=None
) -> tuple[Path, Path]:
    """Write ``<plot_id>_<stem>.png`` into ``out_dir`` and the ``.csv`` beside it.

    ``data_dir`` sends the table somewhere else (``figures/raw_data/<study>/``),
    for a figures directory that holds images only.  The two names always match.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir if data_dir is None else Path(data_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    name = f"{plot_id}_{slugify(stem)}"
    png = out_dir / f"{name}.png"
    csv = table_dir / f"{name}.csv"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    table.to_csv(csv, index=False)
    plt.close(fig)
    return png, csv
