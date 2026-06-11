"""Shared paper-figure style so every figure in the deliverable-GW set is coherent.

Single panel, NO title (captions go in the paper), clean axes, deliverable GW on a
shared vocabulary. Import and call ``setup()`` at the top of a figure script, make
ONE axes with ``fig, ax = new()``, and ``save(fig, name)`` to write a matched
PNG+PDF into development/paper/figures/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# A small, consistent, colorblind-safe palette (Okabe-Ito).
C = {
    "strong": "#0072B2",  # grid-strength (the good policy)
    "cheap": "#D55E00",  # cheap-land (the naive policy that erodes)
    "random": "#999999",
    "uniform": "#CC79A7",
    "ceiling": "#009E73",  # free-allocation ceiling
    "accent": "#E69F00",
    "ink": "#222222",
}


def setup():
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "standard",  # constrained_layout already frames the panel
            "font.size": 10,
            "font.family": "sans-serif",
            "axes.titlesize": 10,  # titles are intentionally unused
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def new(w=3.6, h=2.7):
    """One single-column panel (no title).

    Uses constrained_layout so a tall rotated y-axis label is never clipped (it
    accounts for the label's real extent, which tight_layout/tight-bbox underestimate).
    """
    return plt.subplots(figsize=(w, h), layout="constrained")


def save(fig, name):
    # Let savefig(bbox="tight") do the cropping; a prior tight_layout() squeezes the
    # panel first and can leave a tall rotated y-label overshooting the crop box.
    png = os.path.join(FIGDIR, name + ".png")
    pdf = os.path.join(FIGDIR, name + ".pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png
