"""Plots for the cleaned-grid congestion study: H2 marginal-impact landscape + H1
concentration sweep (Δ LMP-dispersion & feasibility vs # sites)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "development/results/placement_congestion"
d = json.load(open(os.path.join(OUT, "congestion_results.json")))
W = d["workloads"]; wl = list(W.keys()); COL = {"inference": "#1f77b4", "training": "#d62728"}
B = d["fleet"]

# ===================== H2: marginal congestion-impact landscape ===================== #
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for w in wl:
    L = [x for x in W[w]["landscape"] if x["feas"] > 0.99 and np.isfinite(x["d_disp"])]
    dd = np.sort([x["d_disp"] for x in L])
    x = np.linspace(0, 100, len(dd))
    ax[0].plot(x, dd, lw=2, color=COL[w], label=f"{w} (n={len(dd)})")
ax[0].axhline(0, ls=":", c="gray")
ax[0].set_xlabel("clean locations, ranked relieving → worsening (percentile)")
ax[0].set_ylabel("Δ LMP-dispersion from a 0.25 GW block ($/MWh)")
ax[0].set_title("H2: where you place DC changes grid congestion")
ax[0].legend(fontsize=9)
ax[0].annotate("relieving\n(soak up surplus)", (5, ax[0].get_ylim()[0]*0.6), fontsize=8, color="green")
ax[0].annotate("worsening\n(behind binding lines)", (60, ax[0].get_ylim()[1]*0.6), fontsize=8, color="firebrick")
# (b) Δcongestion vs DC price (worsening nodes cost more)
w0 = wl[0]
L = [x for x in W[w0]["landscape"] if x["feas"] > 0.99 and np.isfinite(x["d_disp"]) and np.isfinite(x["dc_lmp"])]
ax[1].scatter([x["dc_lmp"] for x in L], [x["d_disp"] for x in L], s=18, color=COL[w0], alpha=0.6)
ax[1].axhline(0, ls=":", c="gray")
ax[1].set_xlabel("price the DC pays at that node ($/MWh)")
ax[1].set_ylabel("Δ LMP-dispersion (congestion added, $/MWh)")
ax[1].set_title(f"H2: congestion-worsening nodes are the pricey ones ({w0})")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "H2_congestion.png"), dpi=130)
print("saved H2_congestion.png")

# ===================== H1: concentration sweep ===================== #
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for w in wl:
    rows = sorted(W[w]["h1_sweep"], key=lambda r: r["m"])
    m = np.array([r["m"] for r in rows])
    dm = np.array([r["d_disp"][0] for r in rows]); lo = np.array([r["d_disp"][1] for r in rows]); hi = np.array([r["d_disp"][2] for r in rows])
    ax[0].plot(m, dm, "-o", color=COL[w], lw=2, label=w)
    ax[0].fill_between(m, lo, hi, color=COL[w], alpha=0.18)
    ax[1].plot(m, [r["feas"] for r in rows], "-o", color=COL[w], lw=2, label=w)
ax[0].axhline(0, ls=":", c="gray")
ax[0].set_xlabel(f"# sites the {B} GW fleet is spread over  (← concentrate · distribute →)")
ax[0].set_ylabel("Δ LMP-dispersion = added congestion ($/MWh)")
ax[0].set_title(f"H1: concentrating adds more congestion than distributing ({B} GW)")
ax[0].legend(fontsize=9)
ax[1].set_xlabel(f"# sites the {B} GW fleet is spread over  (← concentrate · distribute →)")
ax[1].set_ylabel("fraction of hours the fleet is feasible (must-serve)")
ax[1].set_ylim(0, 1.05)
ax[1].set_title("H1: concentrating is infeasible more often; distributing scales")
ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "H1_congestion.png"), dpi=130)
print("saved H1_congestion.png")
