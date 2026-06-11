"""Clear plots for the integrated DC-placement study (H1, H2, H3).
Reads integrated_results.json, writes one figure per hypothesis."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "development/results/placement_integrated"
d = json.load(open(os.path.join(OUT, "integrated_results.json")))
W = d["workloads"]
wl = list(W.keys())
COL = {"inference": "#1f77b4", "training": "#d62728"}
CLIP = 1000.0  # display cap for raw prices ($/MWh)


# ======================= H2: LOCATIONAL PRICE ============================== #
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
# (a) sorted DC price across all usable locations -> the locational spread
for w in wl:
    rk = [r for r in W[w]["ranking"] if r["feas_frac"] > 0.99 and np.isfinite(r["dc_lmp"])]
    prices = np.clip(sorted(r["dc_lmp"] for r in rk), -100, CLIP)
    x = np.linspace(0, 100, len(prices))
    ax[0].plot(x, prices, label=f"{w}  (n={len(prices)} sites)", color=COL.get(w), lw=2)
ax[0].axhline(0, ls=":", c="gray", lw=1)
ax[0].set_xlabel("locations, ranked cheapest → priciest (percentile)")
ax[0].set_ylabel("electricity price a 0.5 GW data center pays ($/MWh)")
ax[0].set_title("H2: where you place a data center sets its power price")
ax[0].legend(fontsize=9)
ax[0].annotate("sink nodes:\nDC is paid to\nabsorb surplus", (3, -60), fontsize=8, color="green")
ax[0].annotate("deficit nodes:\nprices spike to VOLL", (55, 750), fontsize=8, color="firebrick")
# (b) siting curves: price vs size at a good / median / bad node (inference)
w0 = wl[0]
for nd, c in W[w0]["curves"].items():
    pts = [p for p in c["pts"] if p["feas"] > 0.5]
    gw = [p["gw"] for p in pts]
    pr = [min(p["dc_lmp"], CLIP) for p in pts]
    ax[1].plot(gw, pr, marker="o", lw=2, label=f"{c['label']} (node {nd})")
ax[1].axhline(0, ls=":", c="gray", lw=1)
ax[1].set_xlabel("data-center size at that one location (GW)")
ax[1].set_ylabel("price the data center pays ($/MWh)")
ax[1].set_title(f"H2: siting curves ({w0} workload)")
ax[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "H2_locational_price.png"), dpi=130)
print("saved H2_locational_price.png")


# ============== H1: QUALITY vs SPREAD (the disentangling 2x2) ============== #
QUAL = ["good (sink)", "random (cheap land)", "bad (deficit)"]
SPREAD = ["few big", "many small"]
SCOL = {"few big": "#ff7f0e", "many small": "#2ca02c"}


def cell(w, q, s, metric):
    r = W[w]["h1_2x2"].get(f"{q} | {s}")
    if r is None or r["feas_frac"] < 0.5:
        return None  # infeasible
    return r[metric]["median"]


fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
w0 = wl[0]
for ai, (metric, lab) in enumerate([("d_shed", "extra load shed (% of demand)"),
                                     ("dc_lmp", "price the data centers pay ($/MWh)")]):
    xs = np.arange(len(QUAL)); width = 0.38
    for j, s in enumerate(SPREAD):
        vals = [cell(w0, q, s, metric) for q in QUAL]
        heights = [v if v is not None else 0 for v in vals]
        ax[ai].bar(xs + (j - 0.5) * width, heights, width, label=s, color=SCOL[s])
        for xi, v in zip(xs + (j - 0.5) * width, vals):
            if v is None:
                ax[ai].text(xi, 5, "concentrating\nhere is\nINFEASIBLE", rotation=0, ha="center",
                            va="bottom", fontsize=7.5, color="firebrick", fontweight="bold")
    ax[ai].set_xticks(xs); ax[ai].set_xticklabels([q.split(" (")[0] + "\n" + q.split("(")[1][:-1] for q in QUAL])
    ax[ai].set_ylabel(lab)
    ax[ai].set_title(f"H1: {lab.split('(')[0].strip()}  ({w0}, {d['fleet']} GW fleet)")
    ax[ai].legend(title="site granularity", fontsize=9)
    if metric == "dc_lmp":
        ax[ai].axhline(0, ls=":", c="gray", lw=1)
ax[1].annotate("at EVERY node quality, many small sites\nare cheaper than few big lumps\n(distributing helps)",
               (0.5, 0.97), xycoords="axes fraction", fontsize=8.5, ha="center", va="top", color="navy")
ax[0].annotate("good vs bad NODE (quality)\nis the dominant axis",
               (0.5, 0.95), xycoords="axes fraction", fontsize=8.5, ha="center", va="top", color="navy")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "H1_quality_vs_spread.png"), dpi=130)
print("saved H1_quality_vs_spread.png")


# ===================== H3: WORKLOAD SENSITIVITY =========================== #
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
# (a) ROBUST price range by workload (p5..median..p95 of usable-node DC price)
for i, w in enumerate(wl):
    p = np.array([r["dc_lmp"] for r in W[w]["ranking"] if r["feas_frac"] > 0.99 and np.isfinite(r["dc_lmp"])])
    p5, p50, p95 = np.percentile(p, 5), np.median(p), np.percentile(p, 95)
    ax[0].errorbar([i], [p50], yerr=[[p50 - p5], [p95 - p50]], fmt="o", ms=10, capsize=8,
                   color=COL.get(w), lw=2)
    ax[0].text(i + 0.08, p95, f"p95 ${p95:.0f}", fontsize=9, va="center")
    ax[0].text(i + 0.08, p5, f"p5 ${p5:.0f}", fontsize=9, va="center")
    ncliff = int((p > 1000).sum())
    ax[0].text(i, ax[0].get_ylim()[0], f"{ncliff} nodes\ncliff to VOLL", ha="center", fontsize=8, color="firebrick")
ax[0].set_xticks(range(len(wl))); ax[0].set_xticklabels(wl); ax[0].set_xlim(-0.5, len(wl) - 0.2)
ax[0].set_ylabel("DC price across locations: p5 – median – p95 ($/MWh)")
ax[0].set_title("H3: placement price-spread is wider for flat (training) load")
# (b) extra shed for the same placements, per workload
keys = ["good (sink) | many small", "random (cheap land) | many small", "bad (deficit) | many small"]
short = ["good\nnodes", "random\nnodes", "bad\nnodes"]
xs = np.arange(len(keys)); width = 0.38
for j, w in enumerate(wl):
    vals = [W[w]["h1_2x2"].get(k) for k in keys]
    h = [(v["d_shed"]["median"] if v and v["feas_frac"] > 0.5 else 0) for v in vals]
    ax[1].bar(xs + (j - 0.5) * width, h, width, label=w, color=COL.get(w))
    for xi, v in zip(xs + (j - 0.5) * width, vals):
        if not v or v["feas_frac"] < 0.5:
            ax[1].text(xi, 0, "infeas", rotation=90, ha="center", va="bottom", fontsize=8, color="firebrick")
ax[1].set_xticks(xs); ax[1].set_xticklabels(short)
ax[1].set_ylabel("extra load shed (% of demand)")
ax[1].set_title("H3: same placement, different workload (many-small fleet)")
ax[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "H3_workload.png"), dpi=130)
print("saved H3_workload.png")
print("done ->", OUT)
