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
    "PHS": "#08ad97",
    "battery": "#b8ea04",
    "4hr_battery_storage": "#a4d600",
    "8hr_battery_storage": "#90c200",
    "demand_response": "#dd2e23",
    # --- trade and network (no colour upstream; assigned here) ---
    "imports": "#9a7fbd",
    "unspecified_imports": "#7e63a3",
    "exports": "#5a4a78",
    "AC": "#70af1d",
    "DC": "#8a1caf",
    "AC_exp": "#4f7d14",
    # --- pseudo-carriers ---
    "load": "#111111",
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
    "load",
    "unserved",
    "curtailment",
    "storage_charge",
    "storage_discharge",
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


def save(fig, table: pd.DataFrame, plot_id: str, out_dir, stem: str) -> tuple[Path, Path]:
    """Write ``<plot_id>_<stem>.png`` and ``.csv`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{plot_id}_{slugify(stem)}"
    png = out_dir / f"{name}.png"
    csv = out_dir / f"{name}.csv"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    table.to_csv(csv, index=False)
    plt.close(fig)
    return png, csv
