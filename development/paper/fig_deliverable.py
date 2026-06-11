"""Deliverable firm-DC frontier figures (single claim each, no titles).

Two panels read from the DC deliverable-frontier study JSON (dc_deliverable_frontier.py):
  fig_deliverable        : total DELIVERABLE firm-DC GW vs max site size kappa, four
                           curves (LP-optimal, N-1-aware, grid-strength, cheap-land).
                           The claim: the optimizer beats the heuristics beats the naive
                           cheap-land default, and raising the per-site cap raises total
                           deliverable GW (a strong node absorbs several DCs' worth).
  fig_deliverable_sites  : number of SITES used vs kappa (same four policies); fewer,
                           larger sites as the cap loosens.

This REPLACES the flawed forced-uniform spread metric (which collapsed to ~0 the moment
one near-dead bus entered the top-k). Dead buses are excluded up front by the study.

Usage:
  .venv/bin/python development/paper/fig_deliverable.py --workload inference
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402

# (json-key, legend label, color-key, line style). Grouped: grid-aware optimizers
# (top, cool/green) vs developer/market objectives (bottom, warm). PTDF-sensitivity and
# cheap-power are the two added optimizers.
SERIES = [
    ("lp_optimal", "LP-optimal", "ceiling", "-o"),
    ("ptdf_sensitivity", "PTDF-sensitivity", "ink", "-.v"),
    ("grid_strength", "grid-strength", "accent", "--^"),
    ("lp_n1", "N-1-aware", "strong", "-s"),
    ("cheap_power", "cheap-power (LMP)", "uniform", "--x"),
    ("cheap_land", "cheap-land", "cheap", ":d"),
]


def _xy(rows, key, field):
    """Return (kappa, value) pairs where the value is finite (skips nan policies)."""
    xs, ys = [], []
    for r in rows:
        v = r[key].get(field)
        if v is not None and v == v:  # finite (NaN != NaN)
            xs.append(r["kappa"])
            ys.append(v)
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="development/results/deliverable_frontier/deliverable_frontier_full.json")
    ap.add_argument("--workload", default="inference")
    args = ap.parse_args()

    d = json.load(open(args.data))
    wd = d["workloads"][args.workload]
    rows = sorted(wd["rows"], key=lambda r: r["kappa"])

    figstyle.setup()
    C = figstyle.C

    # --- (1) deliverable GW vs kappa ---
    fig, ax = figstyle.new(w=3.8, h=2.8)
    for key, label, ckey, ls in SERIES:
        xs, ys = _xy(rows, key, "gw")
        if xs:
            ax.plot(xs, ys, ls, color=C[ckey], label=label)
    ax.set_xlabel("Max site size κ (GW)")
    ax.set_ylabel("Deliverable firm-DC (GW)")
    ax.legend(loc="upper left", handlelength=1.8, fontsize=8)
    p1 = figstyle.save(fig, "fig_deliverable")

    # --- (2) number of sites vs kappa ---
    fig, ax = figstyle.new(w=3.8, h=2.8)
    for key, label, ckey, ls in SERIES:
        xs, ys = _xy(rows, key, "sites")
        if xs:
            ax.plot(xs, ys, ls, color=C[ckey], label=label)
    ax.set_xlabel("Max site size κ (GW)")
    ax.set_ylabel("Number of sites used")
    ax.legend(loc="upper right", handlelength=1.8, fontsize=8)
    p2 = figstyle.save(fig, "fig_deliverable_sites")

    # --- (3) optimizer-detail: % of LP-optimal deliverable at the plateau (max kappa) ---
    big = rows[-1]
    lp = big["lp_optimal"]["gw"]
    bars = [("ptdf_sensitivity", "PTDF-sens", "ink"),
            ("grid_strength", "grid-strength", "accent"),
            ("cheap_land", "cheap-land", "cheap"),
            ("cheap_power", "cheap-power", "uniform"),
            ("lp_n1", "N-1-aware", "strong")]
    fig, ax = figstyle.new(w=4.0, h=2.8)
    xs, labels, cols, pcts = [], [], [], []
    for i, (key, lab, ck) in enumerate(bars):
        v = big.get(key, {}).get("gw")
        if v is None or v != v:
            continue
        xs.append(i)
        labels.append(lab)
        cols.append(C[ck])
        pcts.append(100.0 * v / lp)
    ax.axhline(100, color=C["ceiling"], lw=1.2, ls="--", zorder=1, label="LP-optimal")
    ax.bar(xs, pcts, color=cols, width=0.66, zorder=3)
    for x, p in zip(xs, pcts):
        ax.text(x, p + 1.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("% of LP-optimal deliverable")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower left", fontsize=8)
    p3 = figstyle.save(fig, "fig_deliverable_gap")

    print("fig_deliverable       ->", p1)
    print("fig_deliverable_sites ->", p2)
    print("fig_deliverable_gap   ->", p3)
    print(f"WORKLOAD={args.workload}  "
          f"alive={len(wd.get('candidates_alive', []))} "
          f"dead-excluded={len(wd.get('candidates_dead', []))}")
    for r in rows:
        print(f"  kappa={r['kappa']:5.2f}  LP={r['lp_optimal']['gw']}  "
              f"PTDF={r.get('ptdf_sensitivity', {}).get('gw')}  "
              f"GS={r['grid_strength']['gw']}  N1={r['lp_n1']['gw']}  "
              f"CP={r.get('cheap_power', {}).get('gw')}  CL={r['cheap_land']['gw']}")


if __name__ == "__main__":
    main()
