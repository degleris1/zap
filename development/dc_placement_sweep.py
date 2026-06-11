"""
Penetration sweep + bootstrap CIs for the integrated DC-placement study.

Reuses the (expensive) node ranking already computed in integrated_results.json to
build good/random/bad pools, then:
  * BOOTSTRAP across the 12 panel hours -> 95% CIs on the H1 2x2 cells and on the
    locational siting prices (H2), so differences come with error bars.
  * PENETRATION SWEEP: fleet 2->10 GW, tracking whether the distribute-beats-
    concentrate gap (few big 1 GW lumps vs many small 0.2 GW sites) widens with the
    build-out, with CI bands.

Usage:
  .venv/bin/python development/dc_placement_sweep.py
"""
import argparse, os, json, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_placement_study import load_dc_profile, sample_panel_indices, with_dc, metrics, load_land_weights, draw_fleet
from dc_placement_integrated import build_raw_hour, panel_for, panel_eval, pack, WORKLOADS

NBOOT = 2000


def per_hour(panel, base_ms, terms, caps):
    """Per-hour incremental effects (NaN where infeasible), aligned by hour index."""
    ms = panel_eval(panel, terms, caps)
    H = len(ms)
    shed = np.full(H, np.nan); price = np.full(H, np.nan)
    for i, (m, b) in enumerate(zip(ms, base_ms)):
        if m["feasible"] and b["feasible"]:
            shed[i] = max(m["shed_pct"] - b["shed_pct"], 0.0)
            price[i] = m["dc_lmp"]
    return shed, price


def boot_ci(vals, rng, stat=np.median):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    boots = np.array([stat(rng.choice(v, v.size, replace=True)) for _ in range(NBOOT)])
    return float(stat(v)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def boot_paired_gap(a, b, rng):
    """CI on median(a-b) over hours where BOTH feasible (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan, int(mask.sum())
    diff = a[mask] - b[mask]
    boots = np.array([np.median(rng.choice(diff, diff.size, replace=True)) for _ in range(NBOOT)])
    return float(np.median(diff)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--integrated", default="development/results/placement_integrated/integrated_results.json")
    ap.add_argument("--n-snaps", type=int, default=12)
    ap.add_argument("--load-scale", type=float, default=1.2)
    ap.add_argument("--gen-scale", type=float, default=1.0)
    ap.add_argument("--budgets", default="2,3,4,6,8,10")
    ap.add_argument("--big", type=float, default=1.0)
    ap.add_argument("--small", type=float, default=0.2)
    ap.add_argument("--pool", type=int, default=60, help="good/bad pool size (>= max fleet / small)")
    ap.add_argument("--land-cost", default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--outdir", default="development/results/placement_integrated")
    args = ap.parse_args()

    budgets = [float(x) for x in args.budgets.split(",")]
    prev = json.load(open(args.integrated))
    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    idx = sample_panel_indices(len(snaps), args.n_snaps, None, None, 0)
    raw_panel = [build_raw_hour(pn, snaps, h, args.load_scale, args.gen_scale, 1.0) for h in idx]
    n_nodes = raw_panel[0][0].num_nodes
    land, weights = load_land_weights(args.land_cost, n_nodes)
    rng = np.random.default_rng(0)

    print(f"# PENETRATION SWEEP + BOOTSTRAP  load x{args.load_scale}  budgets={budgets}GW  nboot={NBOOT}")
    out = {"budgets": budgets, "load_scale": args.load_scale, "workloads": {}}

    for w, wpath in WORKLOADS.items():
        hourly_lf = load_dc_profile(wpath)
        panel = panel_for(raw_panel, hourly_lf)
        base_ms = [metrics(net, devs, devs, 1) for (net, devs, _, _) in panel]
        # pools from the saved ranking (sorted cheap->pricey)
        rk = [r for r in prev["workloads"][w]["ranking"] if r["feas_frac"] > 0.99]
        sink = [int(r["node"]) for r in rk[:args.pool]]
        deficit = [int(r["node"]) for r in rk[-args.pool:]][::-1]
        usable = [int(r["node"]) for r in rk]
        print(f"\n## WORKLOAD = {w}  (sink pool {len(sink)}, deficit pool {len(deficit)}, usable {len(usable)})")

        # ---- H2 with CIs: representative siting nodes (sink / median / deficit) ----
        h2 = {}
        for lab, nd in [("best-sink", sink[0]), ("median", usable[len(usable)//2]), ("worst-deficit", deficit[0])]:
            _, price = per_hour(panel, base_ms, [nd], [0.5])
            m, lo, hi = boot_ci(price, rng)
            h2[lab] = {"node": nd, "price": m, "lo": lo, "hi": hi}
            print(f"   H2 {lab:13s} node {nd:3d}: ${m:.0f}  [{lo:.0f}, {hi:.0f}]")

        # ---- penetration sweep with bootstrap on the gap ----
        rows = []
        print(f"  {'fleet':>5} | {'quality':>18} | {'fewbig$':>16} {'small$':>16} {'gap(big-small)$':>22}")
        for B in budgets:
            rfleet = draw_fleet(rng, usable, {n: weights[n] for n in usable}, B, args.small)[0]
            pools = {"good (sink)": sink, "random": rfleet, "bad (deficit)": deficit}
            for qn, pool in pools.items():
                tb, cb = pack(pool, B, args.big);  sb_shed, sb_price = per_hour(panel, base_ms, tb, cb)
                ts, cs = pack(pool, B, args.small); ss_shed, ss_price = per_hour(panel, base_ms, ts, cs)
                bigp = boot_ci(sb_price, rng); smp = boot_ci(ss_price, rng)
                gap = boot_paired_gap(sb_price, ss_price, rng)   # +ve => big more expensive => distribute helps
                gshed = boot_paired_gap(sb_shed, ss_shed, rng)
                rows.append({"B": B, "quality": qn,
                             "big_price": bigp, "small_price": smp, "gap_price": gap, "gap_shed": gshed,
                             "big_feas": float(np.mean(np.isfinite(sb_price))),
                             "small_feas": float(np.mean(np.isfinite(ss_price)))})
                if qn == "random":
                    fb = f"${bigp[0]:.0f}[{bigp[1]:.0f},{bigp[2]:.0f}]" if np.isfinite(bigp[0]) else "INFEAS"
                    sm = f"${smp[0]:.0f}[{smp[1]:.0f},{smp[2]:.0f}]"
                    gp = f"${gap[0]:.0f}[{gap[1]:.0f},{gap[2]:.0f}]" if np.isfinite(gap[0]) else "n/a"
                    print(f"  {B:5.1f} | {qn:>18} | {fb:>16} {sm:>16} {gp:>22}")
        out["workloads"][w] = {"h2_ci": h2, "sweep": rows, "sink": sink, "deficit": deficit}

    json.dump(out, open(os.path.join(args.outdir, "sweep_results.json"), "w"), indent=2, default=float)
    print(f"\nsaved -> {os.path.join(args.outdir, 'sweep_results.json')}")


if __name__ == "__main__":
    main()
