"""
Plots for the DC spread-frontier study (development/dc_spread_frontier.py).

Reads spread_frontier_<tag>.json and produces, per workload:
  (a) g(k), f(k), gap(k) vs footprint k  (cheap-land ordering, 95%-reliability)
  (b) k_min(D) vs target D with bootstrap CI band (monte-carlo fleets)
  (c) g(k) erosion: cheap-land vs grid-strength ordering
  (d) g(k) drop per added node vs that node's standalone headroom h(n)

Usage:
  .venv/bin/python development/plot_spread_frontier.py \
      --in development/results/spread_frontier/spread_frontier_pilot.json
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _ks(curve):
    return sorted(int(k) for k in curve.keys())


def plot_workload(w, r, outdir, tag):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"DC spread frontier -- {w} (95%-reliability nameplate GW)")

    # (a) g, f, gap on cheap-land
    cl = {int(k): v for k, v in r["cheap_land"].items()}
    ks = _ks(cl)
    g = [cl[k]["g"]["q05"] for k in ks]
    f = [cl[k]["f"]["q05"] for k in ks]
    gap = [f[i] - g[i] for i in range(len(ks))]
    a = ax[0, 0]
    a.plot(ks, f, "o-", label="f(k) free ceiling", color="tab:green")
    a.plot(ks, g, "s-", label="g(k) forced-uniform", color="tab:blue")
    a.plot(ks, gap, "^--", label="gap = cost of uniformity", color="tab:red")
    er = r["erosion"]
    a.axvline(er["k_star"], color="gray", ls=":", label=f"k*={er['k_star']}")
    a.set_xscale("log")
    a.set_xlabel("footprint k (sites)")
    a.set_ylabel("deliverable nameplate (GW)")
    a.set_title("(a) deliverable vs spread (cheap-land)")
    a.legend(fontsize=8)

    # (b) k_min(D) vs D with bootstrap CI band over fleets
    a = ax[0, 1]
    D_grid = r["D_grid"]
    pf = r["per_fleet_kmin"]
    # per-D distribution of k_min across fleets (None -> drop)
    med, lo, hi = [], [], []
    Dplot = []
    for j, D in enumerate(D_grid):
        vals = [fk[str(D)] if str(D) in fk else fk.get(D) for fk in pf]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        Dplot.append(D)
        med.append(np.median(vals))
        lo.append(np.percentile(vals, 10))
        hi.append(np.percentile(vals, 90))
    if Dplot:
        a.plot(Dplot, med, "o-", color="tab:purple", label="median k_min")
        a.fill_between(Dplot, lo, hi, alpha=0.25, color="tab:purple", label="[p10,p90] fleets")
    st = r["slope_test"]
    a.set_xlabel("target deliverable D (GW)")
    a.set_ylabel("k_min(D) (sites)")
    a.set_title(f"(b) k_min(D): slope={st['slope']:.2f} "
                f"CI[{st['lo']:.2f},{st['hi']:.2f}] P(>0)={st['p_gt0']:.2f}")
    a.legend(fontsize=8)

    # (c) erosion: cheap-land vs grid-strength g(k)
    a = ax[1, 0]
    gs = {int(k): v for k, v in r["grid_strength"].items()}
    gg = [gs[k]["g"]["q05"] for k in ks]
    a.plot(ks, g, "s-", label="cheap-land", color="tab:blue")
    a.plot(ks, gg, "d-", label="grid-strength", color="tab:orange")
    a.set_xscale("log")
    a.set_xlabel("footprint k (sites)")
    a.set_ylabel("g(k) forced-uniform (GW)")
    a.set_title("(c) erosion is a siting artifact?")
    a.legend(fontsize=8)

    # (d) marginal deliverable per added spread (cheap-land order)
    a = ax[1, 1]
    steps_k, steps_dg = [], []
    for i in range(1, len(ks)):
        steps_k.append(ks[i])
        steps_dg.append(g[i] - g[i - 1])
    a.axhline(0, color="gray", ls=":")
    a.plot(steps_k, steps_dg, "o-", color="tab:red")
    a.set_xscale("log")
    a.set_xlabel("footprint k (sites)")
    a.set_ylabel("Δ g(k) when widening")
    a.set_title("(d) marginal deliverable per added spread")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(outdir, f"spread_frontier_{tag}_{w}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    data = json.load(open(args.infile))
    tag = os.path.basename(args.infile).replace("spread_frontier_", "").replace(".json", "")
    outdir = args.outdir or os.path.dirname(args.infile)
    os.makedirs(outdir, exist_ok=True)
    for w, r in data["workloads"].items():
        p = plot_workload(w, r, outdir, tag)
        print(f"saved -> {p}")


if __name__ == "__main__":
    main()
