"""
LOAD-STRESS SWEEP: do the placement effects scale with how tight the grid is?

For load x{1.1, 1.2, 1.3} (looser -> tighter grid), recompute the two headline
effects and show they grow with tightness:
  * H2  locational price spread  (p5..p95 of the price a 0.5 GW site pays, across
        ~100 locations) -- should widen as the grid tightens.
  * H1  distribute-beats-concentrate gap at a fixed 4 GW fleet on the best (sink)
        nodes: price(few big 1 GW lumps) - price(many small 0.2 GW sites).
Also tracks base shedding and #usable locations as tightness rises.

Reuses the panel / metric machinery; ranks a stride-sampled node set per (load,
workload) for tractability. Two workloads (inference, training).

Usage:
  .venv/bin/python development/dc_placement_loadsweep.py
"""
import argparse, os, json, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_placement_study import (load_dc_profile, sample_panel_indices, with_dc, metrics,
                                 load_land_weights, draw_fleet)
from dc_placement_integrated import build_raw_hour, panel_for, panel_eval, pack, WORKLOADS


def node_price(panel, base_ms, nd, gw):
    """Median DC price (over feasible hours) for a `gw` site at node nd, and feas frac."""
    ms = panel_eval(panel, [nd], [gw])
    pr = [m["dc_lmp"] for m, b in zip(ms, base_ms) if m["feasible"] and b["feasible"]]
    return (float(np.median(pr)) if pr else np.nan), float(np.mean([m["feasible"] for m in ms]))


def fleet_price(panel, base_ms, terms, caps):
    ms = panel_eval(panel, terms, caps)
    pr = [m["dc_lmp"] for m, b in zip(ms, base_ms) if m["feasible"] and b["feasible"]]
    return (float(np.median(pr)) if pr else np.nan), float(np.mean([m["feasible"] for m in ms]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=12)
    ap.add_argument("--loads", default="1.1,1.2,1.3")
    ap.add_argument("--node-stride", type=int, default=5, help="sample every Nth node for ranking")
    ap.add_argument("--fleet", type=float, default=4.0)
    ap.add_argument("--big", type=float, default=1.0)
    ap.add_argument("--small", type=float, default=0.2)
    ap.add_argument("--pool", type=int, default=30)
    ap.add_argument("--land-cost", default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--outdir", default="development/results/placement_integrated")
    args = ap.parse_args()
    loads = [float(x) for x in args.loads.split(",")]

    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    idx = sample_panel_indices(len(snaps), args.n_snaps, None, None, 0)
    rng = np.random.default_rng(0)

    print(f"# LOAD-STRESS SWEEP  loads={loads}  fleet={args.fleet}GW  node-stride={args.node_stride}")
    out = {"loads": loads, "fleet": args.fleet, "workloads": {w: {"rows": []} for w in WORKLOADS}}

    for ls in loads:
        raw_panel = [build_raw_hour(pn, snaps, h, ls, 1.0, 1.0) for h in idx]
        n_nodes = raw_panel[0][0].num_nodes
        sample = list(range(0, n_nodes, args.node_stride))
        land, weights = load_land_weights(args.land_cost, n_nodes)
        for w, wpath in WORKLOADS.items():
            lf = load_dc_profile(wpath)
            panel = panel_for(raw_panel, lf)
            base_ms = [metrics(net, devs, devs, 1) for (net, devs, _, _) in panel]
            base_shed = float(np.median([m["shed_pct"] for m in base_ms if m["feasible"]]))
            # rank sampled nodes by price of a 0.5 GW site
            ranked = []
            for nd in sample:
                p, f = node_price(panel, base_ms, nd, 0.5)
                if f > 0.99 and np.isfinite(p):
                    ranked.append((nd, p))
            ranked.sort(key=lambda x: x[1])
            prices = np.array([p for _, p in ranked])
            sink = [nd for nd, _ in ranked[:args.pool]]
            deficit = [nd for nd, _ in ranked[-args.pool:]][::-1]
            usable_nodes = [nd for nd, _ in ranked]
            # H1 gap on sink (good) nodes at fixed fleet
            B = args.fleet
            big_t, big_c = pack(sink, B, args.big)
            sm_t, sm_c = pack(sink, B, args.small)
            bigp, bigf = fleet_price(panel, base_ms, big_t, big_c)
            smp, smf = fleet_price(panel, base_ms, sm_t, sm_c)
            # H1 gap on random (realistic) nodes
            rfleet = draw_fleet(rng, usable_nodes, {n: weights[n] for n in usable_nodes}, B, args.small)[0]
            rb_t, rb_c = pack(rfleet, B, args.big)
            rs_t, rs_c = pack(rfleet, B, args.small)
            rbp, rbf = fleet_price(panel, base_ms, rb_t, rb_c)
            rsp, rsf = fleet_price(panel, base_ms, rs_t, rs_c)
            row = {"load": ls, "base_shed": base_shed, "n_usable": len(ranked),
                   "p5": float(np.percentile(prices, 5)), "median": float(np.median(prices)),
                   "p95": float(np.percentile(prices, 95)),
                   "n_sink": int((prices < 0).sum()), "n_cliff": int((prices > 1000).sum()),
                   "good_big": bigp, "good_big_feas": bigf, "good_small": smp,
                   "good_gap": (bigp - smp) if (np.isfinite(bigp) and np.isfinite(smp)) else np.nan,
                   "rand_big": rbp, "rand_big_feas": rbf, "rand_small": rsp,
                   "rand_gap": (rbp - rsp) if (np.isfinite(rbp) and np.isfinite(rsp)) else np.nan}
            out["workloads"][w]["rows"].append(row)
            print(f"  load x{ls} {w:9s}: base_shed {base_shed:.2f}% | usable {len(ranked)}/{len(sample)} "
                  f"| price p5..p95 ${row['p5']:.0f}..${row['p95']:.0f} (spread ${row['p95']-row['p5']:.0f}) "
                  f"| good-gap ${row['good_gap']:.0f}" + ("" if bigf > 0.99 else f" (big feas {bigf:.2f})"))

    json.dump(out, open(os.path.join(args.outdir, "loadsweep_results.json"), "w"), indent=2, default=float)
    print(f"\nsaved -> {os.path.join(args.outdir, 'loadsweep_results.json')}")
    _plot(args.outdir, out)


def _plot(outdir, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    W = out["workloads"]; wl = list(W.keys()); COL = {"inference": "#1f77b4", "training": "#d62728"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # (a) H2 price spread vs load (p5..p95 band per workload)
    for w in wl:
        r = W[w]["rows"]; L = [x["load"] for x in r]
        p5 = [x["p5"] for x in r]; p95 = [min(x["p95"], 1000) for x in r]; med = [x["median"] for x in r]
        ax[0].fill_between(L, p5, p95, color=COL[w], alpha=0.18)
        ax[0].plot(L, med, "-o", color=COL[w], lw=2, label=f"{w} (median)")
        ax[0].plot(L, p95, "--", color=COL[w], lw=1)
        ax[0].plot(L, p5, "--", color=COL[w], lw=1)
    ax[0].axhline(0, ls=":", c="gray")
    ax[0].set_xlabel("load-stress multiplier (tighter →)"); ax[0].set_ylabel("DC price across locations: p5–median–p95 ($/MWh)")
    ax[0].set_title("H2: locational price spread widens as the grid tightens"); ax[0].legend(fontsize=9)
    # (b) H1 distribute gap vs load
    for w in wl:
        r = W[w]["rows"]; L = [x["load"] for x in r]
        gg = [x["good_gap"] for x in r]; rg = [x["rand_gap"] for x in r]
        ax[1].plot(L, gg, "-o", color=COL[w], lw=2, label=f"{w}: best nodes")
        ax[1].plot(L, rg, "--s", color=COL[w], lw=1.5, alpha=0.7, label=f"{w}: random nodes")
    ax[1].axhline(0, ls=":", c="gray")
    ax[1].set_xlabel("load-stress multiplier (tighter →)")
    ax[1].set_ylabel("distribute benefit: price(few big) − price(many small) ($/MWh)")
    ax[1].set_title("H1: distribute-beats-concentrate gap grows with tightness"); ax[1].legend(fontsize=8)
    # (c) grid tightness indicators
    for w in [wl[0]]:
        r = W[w]["rows"]; L = [x["load"] for x in r]
        ax2 = ax[2]
        ax2.plot(L, [x["base_shed"] for x in r], "-o", color="darkorange", lw=2, label="base load shed (%)")
        ax2.set_ylabel("base load shed (% of demand)", color="darkorange")
        ax3 = ax2.twinx()
        ax3.plot(L, [x["n_cliff"] for x in r], "-^", color="purple", lw=2, label="# VOLL-cliff nodes")
        ax3.set_ylabel("# nodes that cliff to VOLL", color="purple")
    ax[2].set_xlabel("load-stress multiplier (tighter →)")
    ax[2].set_title("How tight the grid gets (inference)")
    fig.tight_layout(); p = os.path.join(outdir, "load_stress_sweep.png"); fig.savefig(p, dpi=130)
    print("saved", p)


if __name__ == "__main__":
    main()
