"""Plots for the penetration sweep + bootstrap CIs (H1 gap vs fleet, H2 prices w/ CI)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "development/results/placement_integrated"
d = json.load(open(os.path.join(OUT, "sweep_results.json")))
W = d["workloads"]; wl = list(W.keys())
QCOL = {"good (sink)": "#2ca02c", "random": "#1f77b4", "bad (deficit)": "#d62728"}
CLIP = 1200


def series(w, quality, field):
    """Return budgets, median, lo, hi for a quality's `field` ([m,lo,hi])."""
    rows = sorted([r for r in W[w]["sweep"] if r["quality"] == quality], key=lambda r: r["B"])
    B = [r["B"] for r in rows]
    m = [min(r[field][0], CLIP) if np.isfinite(r[field][0]) else np.nan for r in rows]
    lo = [min(r[field][1], CLIP) if np.isfinite(r[field][1]) else np.nan for r in rows]
    hi = [min(r[field][2], CLIP) if np.isfinite(r[field][2]) else np.nan for r in rows]
    feas = [r["big_feas"] if "big" in field else r.get("small_feas", 1) for r in rows]
    return np.array(B), np.array(m), np.array(lo), np.array(hi), np.array(feas)


# ================= FIG A: penetration (good/sink pool = clean) ============== #
GCLIP = 2600
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
w0 = wl[0]
Q = "good (sink)"
# (a) same best nodes, scaled up: few-big lumps vs many-small sites, DC price vs fleet
B, mb, lob, hib, _ = series(w0, Q, "big_price")
_, ms, los, his, _ = series(w0, Q, "small_price")
ax[0].plot(B, ms, "-s", color="#2ca02c", lw=2.5, label="many small (0.2 GW) sites")
ax[0].fill_between(B, los, his, color="#2ca02c", alpha=0.2)
big_ok = np.isfinite(mb)
ax[0].plot(B[big_ok], mb[big_ok], "-o", color="#ff7f0e", lw=2.5, label="few big (1 GW) lumps")
ax[0].fill_between(B[big_ok], lob[big_ok], hib[big_ok], color="#ff7f0e", alpha=0.2)
rows = sorted([r for r in W[w0]["sweep"] if r["quality"] == Q], key=lambda r: r["B"])
saturate = [r["B"] for r in rows if r["big_feas"] < 0.99]
if saturate:
    ax[0].axvline(min(saturate), ls="--", c="firebrick", alpha=0.6)
    ax[0].text(min(saturate), ax[0].get_ylim()[1]*0.85, "  big lumps start\n  going infeasible",
               color="firebrick", fontsize=9)
ax[0].set_xlabel("total data-center fleet (GW)"); ax[0].set_ylabel("DC price at the best nodes ($/MWh)")
ax[0].set_title(f"Concentrate vs distribute on the SAME best nodes ({w0})")
ax[0].legend(fontsize=9)
# (b) distribute benefit (price gap) vs fleet, inference vs training, w/ CI
for w, c in zip(wl, ["#1f77b4", "#d62728"]):
    rows = sorted([r for r in W[w]["sweep"] if r["quality"] == Q], key=lambda r: r["B"])
    Bg = np.array([r["B"] for r in rows])
    mg = np.array([min(r["gap_price"][0], GCLIP) if np.isfinite(r["gap_price"][0]) else np.nan for r in rows])
    lo = np.array([min(r["gap_price"][1], GCLIP) if np.isfinite(r["gap_price"][1]) else np.nan for r in rows])
    hi = np.array([min(r["gap_price"][2], GCLIP) if np.isfinite(r["gap_price"][2]) else np.nan for r in rows])
    ok = np.isfinite(mg)
    ax[1].plot(Bg[ok], mg[ok], "-o", color=c, lw=2.5, label=w)
    ax[1].fill_between(Bg[ok], lo[ok], hi[ok], color=c, alpha=0.18)
    sat = [r["B"] for r in rows if r["big_feas"] < 0.5]
    if sat:
        ax[1].plot(min(sat), GCLIP, "x", color=c, ms=12, mew=3)
        ax[1].text(min(sat), GCLIP*0.92, f"  {w}: concentrating\n  infeasible beyond here", color=c, fontsize=8)
ax[1].axhline(0, ls=":", c="gray")
ax[1].set_xlabel("total data-center fleet (GW)")
ax[1].set_ylabel("distribute benefit:  price(few big) − price(many small)  ($/MWh)")
ax[1].set_title("Distribute-beats-concentrate gap widens with build-out")
ax[1].legend(fontsize=9, title="workload")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "penetration.png"), dpi=130)
print("saved penetration.png")


# ================= FIG B: bootstrap CIs (H1 @ 4GW, H2 nodes) ================= #
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
# (a) H1 at fleet=4 GW: price per (quality x spread) with bootstrap CI
B0 = 4.0
quals = list(QCOL.keys()); xs = np.arange(len(quals)); width = 0.38
for j, (field, slab, col) in enumerate([("big_price", "few big", "#ff7f0e"),
                                        ("small_price", "many small", "#2ca02c")]):
    ms, los, his = [], [], []
    for q in quals:
        r = [x for x in W[w0]["sweep"] if x["quality"] == q and x["B"] == B0][0]
        v = r[field]
        ms.append(v[0] if np.isfinite(v[0]) else 0)
        los.append((v[0] - v[1]) if np.isfinite(v[0]) else 0)
        his.append((v[2] - v[0]) if np.isfinite(v[0]) else 0)
    ax[0].bar(xs + (j - 0.5) * width, ms, width, yerr=[los, his], capsize=4, label=slab, color=col)
    for xi, q in zip(xs + (j - 0.5) * width, quals):
        r = [x for x in W[w0]["sweep"] if x["quality"] == q and x["B"] == B0][0]
        if not np.isfinite(r[field][0]):
            ax[0].text(xi, 5, "infeasible", rotation=90, ha="center", fontsize=8, color="firebrick")
ax[0].set_xticks(xs); ax[0].set_xticklabels([q.split(" (")[0] for q in quals])
ax[0].set_ylabel("DC price ($/MWh)"); ax[0].axhline(0, ls=":", c="gray")
ax[0].set_title(f"H1 with 95% CIs ({w0}, {B0:.0f} GW fleet)"); ax[0].legend(fontsize=9)
# (b) H2 representative node prices with CI, both workloads -- symlog (spans -$600..$90k)
labs = ["best-sink", "median", "worst-deficit"]; xs = np.arange(len(labs))
for j, w in enumerate(wl):
    pr = np.array([W[w]["h2_ci"][l]["price"] for l in labs])
    lo = np.array([W[w]["h2_ci"][l]["price"] - W[w]["h2_ci"][l]["lo"] for l in labs])
    hi = np.array([W[w]["h2_ci"][l]["hi"] - W[w]["h2_ci"][l]["price"] for l in labs])
    ax[1].errorbar(xs + (j - 0.5) * width, pr, yerr=[np.abs(lo), np.abs(hi)], fmt="o", ms=8,
                   capsize=6, lw=2, label=w, color=["#1f77b4", "#d62728"][j])
ax[1].set_yscale("symlog", linthresh=100)
ax[1].axhline(0, ls=":", c="gray")
ax[1].set_xticks(xs); ax[1].set_xticklabels(["best sink\n(surplus)", "median\nnode", "worst deficit\n(chokepoint)"])
ax[1].set_xlim(-0.6, len(labs) - 0.4)
ax[1].set_ylabel("DC price at node ($/MWh, symlog)")
ax[1].set_title("H2 locational price with 95% CIs (sink → deficit)"); ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "bootstrap_CIs.png"), dpi=130)
print("saved bootstrap_CIs.png")
