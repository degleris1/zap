"""24h seed-pinned feasibility wall figure.

Single claim: under the frozen 25-bad manifest, distributing a fixed DC fleet over
more sites improves must-serve feasible fraction. The plotted interval labels are
explicit: thin whiskers are the 8-seed range; thick whiskers are bootstrap CIs over
seed x pool samples.

Run:
  uv run python development/paper/fig_phaseb_wall.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PHASEB = os.path.join(
    HERE, "..", "results", "placement_robustness", "phaseB_finish_partA_25bad_24h.json"
)


def _stat(cell, kind):
    s = cell["wall"][kind]
    mean = float(s["mean"])
    rlo, rhi = s.get("seed_range", [s.get("min"), s.get("max")])
    ci = s.get("bootstrap_ci_seed_pool_mean", [mean, np.nan, np.nan])
    return mean, float(rlo), float(rhi), float(ci[1]), float(ci[2])


def _asym_yerr(mean, lo, hi):
    return [[max(0.0, mean - lo)], [max(0.0, hi - mean)]]


def fig_phaseb_wall(data):
    figstyle.setup()
    fig, axes = figstyle.plt.subplots(1, 2, figsize=(6.8, 2.9), layout="constrained",
                                      sharey=True)
    colors = {"conc": figstyle.C["cheap"], "dist": figstyle.C["strong"]}
    labels = {"conc": "concentrated", "dist": "distributed"}
    fleets = [str(int(x)) for x in data["fleets"]]
    x = np.arange(len(fleets), dtype=float)
    width = 0.34

    for ax, workload in zip(axes, ["inference", "training"]):
        for offset, kind in [(-width / 2, "conc"), (width / 2, "dist")]:
            means, rlo, rhi, clo, chi = [], [], [], [], []
            for B in fleets:
                m, a, b, c, d = _stat(data["partA"][B][workload], kind)
                means.append(m)
                rlo.append(a)
                rhi.append(b)
                clo.append(c)
                chi.append(d)
            xpos = x + offset
            ax.bar(xpos, means, width=width, color=colors[kind], alpha=0.82,
                   label=labels[kind], zorder=3)
            for xx, m, a, b, c, d in zip(xpos, means, rlo, rhi, clo, chi):
                # Thin full seed range.
                ax.errorbar([xx], [m], yerr=_asym_yerr(m, a, b), fmt="none",
                            ecolor=figstyle.C["ink"], elinewidth=0.8, capsize=2.5,
                            alpha=0.75, zorder=4)
                # Thicker bootstrap CI over seed x pool samples.
                if np.isfinite(c) and np.isfinite(d):
                    ax.errorbar([xx], [m], yerr=_asym_yerr(m, c, d), fmt="none",
                                ecolor=figstyle.C["ink"], elinewidth=2.0, capsize=0,
                                alpha=0.95, zorder=5)
        ax.set_xlabel("Fleet size (GW)")
        ax.set_xticks(x)
        ax.set_xticklabels(fleets)
        ax.text(0.02, 0.96, workload, transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color=figstyle.C["ink"])
        ax.grid(axis="y", alpha=0.22, zorder=1)

    axes[0].set_ylabel("Must-serve feasible fraction")
    axes[0].set_ylim(0, 1.02)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[0].text(0.0, -0.32, "thin: 8-seed range; thick: bootstrap CI",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=7.5,
                 color=figstyle.C["ink"])
    return figstyle.save(fig, "fig_phaseb_wall")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phaseb", default=DEFAULT_PHASEB)
    args = ap.parse_args()
    with open(args.phaseb) as f:
        data = json.load(f)
    path = fig_phaseb_wall(data)
    print("fig_phaseb_wall ->", path)
    for workload in ("inference", "training"):
        for B in [str(int(x)) for x in data["fleets"]]:
            wall = data["partA"][B][workload]["wall"]
            print(f"{workload:9s} B={B:>2s}: conc {wall['conc']['mean']:.3f} "
                  f"dist {wall['dist']['mean']:.3f} sep={wall['separated']}")


if __name__ == "__main__":
    main()
