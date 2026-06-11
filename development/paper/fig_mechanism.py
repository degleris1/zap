"""Mechanism figures: the price + deliverability signature of concentrating a fleet.

A FIXED 6 GW firm-DC fleet on the cleaned 490-node WECC is pushed toward its
deliverable limit by CONCENTRATING it onto fewer, larger sites (the candidate pool
and total GW are held fixed; only the per-site lump grows). This is a monotone
steepening as the fleet concentrates -- not a sharp spike. Split into two single-claim
panels:

  fig_mechanism_price -- concentration raises the congestion price
  fig_mechanism_feas  -- concentration collapses deliverability

Data: development/results/placement_congestion/congestion_results.json (H1 sweep,
inference workload, load x1.2). Bootstrap CIs are the price band. Run:
  .venv/bin/python development/paper/fig_mechanism.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import figstyle  # noqa: E402

RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "results",
    "placement_congestion",
    "congestion_results.json",
)
WORKLOAD = "inference"  # representative; diurnal load, the cleaner monotone signal
XLABEL = "Per-site DC size (GW), 6 GW fleet"


def _rows(d):
    rows = d["workloads"][WORKLOAD]["h1_sweep"]
    x = [r["site_gw"] for r in rows]
    disp = [r["d_disp"][0] for r in rows]
    disp_lo = [r["d_disp"][1] for r in rows]
    disp_hi = [r["d_disp"][2] for r in rows]
    feas = [r["feas"] for r in rows]
    return x, disp, disp_lo, disp_hi, feas


def fig_price(d):
    """Claim: concentration raises the congestion price.

    Incremental LMP dispersion ($/MWh) vs per-site DC size, with bootstrap CI band.
    """
    x, disp, disp_lo, disp_hi, _ = _rows(d)

    figstyle.setup()
    fig, ax = figstyle.new()
    ax.fill_between(x, disp_lo, disp_hi, color=figstyle.C["cheap"], alpha=0.18, lw=0)
    ax.plot(x, disp, "-o", color=figstyle.C["cheap"], zorder=3)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Added LMP dispersion (\\$/MWh)")
    ax.set_ylim(0, max(disp_hi) * 1.08)
    ax.set_xlim(min(x) - 0.05, max(x) + 0.05)
    return figstyle.save(fig, "fig_mechanism_price")


def fig_feas(d):
    """Claim: concentration collapses deliverability.

    Must-serve feasible fraction of hours vs per-site DC size.
    """
    x, _, _, _, feas = _rows(d)

    figstyle.setup()
    fig, ax = figstyle.new()
    ax.plot(x, feas, "-s", color=figstyle.C["strong"], zorder=3)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Must-serve feasible fraction")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(min(x) - 0.05, max(x) + 0.05)
    return figstyle.save(fig, "fig_mechanism_feas")


def main():
    d = json.load(open(RESULTS))
    wl = d["workloads"][WORKLOAD]
    p1 = fig_price(d)
    p2 = fig_feas(d)
    print("fig_mechanism_price ->", p1)
    print("fig_mechanism_feas  ->", p2)

    print(f"\nworkload={WORKLOAD}  load x{d['load_scale']}  fleet {d['fleet']} GW")
    print(f"base (no-DC) LMP dispersion: ${wl['base_disp']:.0f}/MWh")
    print("site_gw  d_disp[med,lo,hi]  feas")
    for r in wl["h1_sweep"]:
        dd = r["d_disp"]
        print(f"  {r['site_gw']:.2f}   {dd[0]:5.1f}[{dd[1]:.1f},{dd[2]:.1f}]   {r['feas']:.3f}")


if __name__ == "__main__":
    main()
