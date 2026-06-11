"""
Penetration + load-stress sweeps for the cleaned-grid congestion study.

Reuses the cleaned-panel + concentration machinery from dc_placement_congestion.
For each setting, compares CONCENTRATE (fleet on the fewest 1 GW sites) vs
DISTRIBUTE (fleet over many small sites) on added congestion (Δ LMP-dispersion,
robust) and must-serve feasibility, with bootstrap CIs. Bad buses fixed (detected
at load x1.0). Both workloads.

  Penetration: load x1.2, fleet in {3,6,10} GW -> does the distribute benefit grow?
  Load-stress: fleet 6 GW, load in {1.1,1.2,1.3} -> do effects scale with tightness?

Usage:
  .venv/bin/python development/dc_placement_congestion_sweeps.py
"""
import argparse, os, json, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_placement_study import (load_dc_profile, sample_panel_indices, metrics,
                                 find_bad_buses, load_land_weights, draw_fleet)
from dc_placement_integrated import build_raw_hour, WORKLOADS
from dc_placement_congestion import cleaned_panel, incr, boot_ci, pack


def conc_vs_dist(panel, base_ms, clean_nodes, nodes, weights, rng, B, n_pools, pool_size):
    """Concentrate (fewest 1 GW sites) vs distribute (pool_size small sites). The added
    congestion is compared PAIRED over hours where BOTH are feasible (same pool+hour),
    so the comparison isn't biased by concentrate only being measurable in its easy
    hours. Feasibility is reported separately (the hard must-serve limit)."""
    m_conc, m_dist = int(np.ceil(B)), pool_size
    gap, cd, dd, cf, df = [], [], [], [], []
    for _ in range(n_pools):
        pool = draw_fleet(rng, nodes, {n: weights[n] for n in nodes}, B, B / pool_size)[0]
        tc, cc = pack(pool, B, m_conc); rc = incr(panel, base_ms, clean_nodes, tc, cc)
        td, cdp = pack(pool, B, m_dist); rd = incr(panel, base_ms, clean_nodes, td, cdp)
        both = np.isfinite(rc["d_disp"]) & np.isfinite(rd["d_disp"])
        gap.extend((rc["d_disp"][both] - rd["d_disp"][both]).tolist())   # +ve = concentrate worse
        cd.extend(rc["d_disp"][np.isfinite(rc["d_disp"])].tolist())
        dd.extend(rd["d_disp"][np.isfinite(rd["d_disp"])].tolist())
        cf.append(float(rc["feas"].mean())); df.append(float(rd["feas"].mean()))
    return {"gap": boot_ci(gap, rng), "conc_disp": boot_ci(cd, rng), "dist_disp": boot_ci(dd, rng),
            "conc_feas": float(np.mean(cf)), "dist_feas": float(np.mean(df)),
            "m_conc": m_conc, "m_dist": m_dist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=12)
    ap.add_argument("--fleets", default="3,6,10")
    ap.add_argument("--loads", default="1.1,1.2,1.3")
    ap.add_argument("--base-load", type=float, default=1.2, help="load for the penetration sweep")
    ap.add_argument("--base-fleet", type=float, default=6.0, help="fleet for the load-stress sweep")
    ap.add_argument("--n-pools", type=int, default=12)
    ap.add_argument("--pool-size", type=int, default=40)
    ap.add_argument("--land-cost", default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--outdir", default="development/results/placement_congestion")
    args = ap.parse_args()
    fleets = [float(x) for x in args.fleets.split(",")]
    loads = [float(x) for x in args.loads.split(",")]

    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    idx = sample_panel_indices(len(snaps), args.n_snaps, None, None, 0)
    n_nodes = build_raw_hour(pn, snaps, idx[0], 1.0, 1.0, 1.0)[0].num_nodes
    land, weights = load_land_weights(args.land_cost, n_nodes)

    # bad buses fixed at x1.0
    raw0 = [build_raw_hour(pn, snaps, h, 1.0, 1.0, 1.0) for h in idx]
    bad = find_bad_buses([(net, devs, dt, 0.0) for (net, devs, dt, hod) in raw0])
    clean_nodes = [i for i in range(n_nodes) if i not in set(bad)]
    print(f"# CONGESTION SWEEPS  bad buses {len(bad)} (fixed at x1.0), clean nodes {len(clean_nodes)}")

    def panels_for(load):
        raw = [build_raw_hour(pn, snaps, h, load, 1.0, 1.0) for h in idx]
        return raw

    out = {"bad": bad, "penetration": {"load": args.base_load, "fleets": fleets, "workloads": {}},
           "loadstress": {"fleet": args.base_fleet, "loads": loads, "workloads": {}}}

    # ---------- penetration (vary fleet at base load) ----------
    print(f"\n## PENETRATION  load x{args.base_load}, fleets {fleets} GW")
    raw_b = panels_for(args.base_load)
    for w, wpath in WORKLOADS.items():
        lf = load_dc_profile(wpath)
        panel = cleaned_panel(raw_b, lf, bad)
        base_ms = [metrics(net, devs, devs, 1, price_nodes=clean_nodes) for (net, devs, _, _) in panel]
        rng = np.random.default_rng(0)
        rows = []
        for B in fleets:
            r = conc_vs_dist(panel, base_ms, clean_nodes, clean_nodes, weights, rng, B, args.n_pools, args.pool_size)
            rows.append({"B": B, **r})
            print(f"   {w:9s} B={B:4.0f}: paired gap (conc−dist) Δdisp ${r['gap'][0]:.0f}"
                  f"[{r['gap'][1]:.0f},{r['gap'][2]:.0f}]  | feas conc {r['conc_feas']:.2f}"
                  f" vs dist {r['dist_feas']:.2f}")
        out["penetration"]["workloads"][w] = rows

    # ---------- load-stress (vary load at base fleet) ----------
    print(f"\n## LOAD-STRESS  fleet {args.base_fleet} GW, loads {loads}")
    for w, wpath in WORKLOADS.items():
        lf = load_dc_profile(wpath)
        rows = []
        for ls in loads:
            panel = cleaned_panel(panels_for(ls), lf, bad)
            base_ms = [metrics(net, devs, devs, 1, price_nodes=clean_nodes) for (net, devs, _, _) in panel]
            base_disp = float(np.median([m["lmp_disp"] for m in base_ms if m["feasible"]]))
            rng = np.random.default_rng(0)
            r = conc_vs_dist(panel, base_ms, clean_nodes, clean_nodes, weights, rng,
                             args.base_fleet, args.n_pools, args.pool_size)
            rows.append({"load": ls, "base_disp": base_disp, **r})
            print(f"   {w:9s} load x{ls}: base-disp ${base_disp:.0f} | paired gap (conc−dist) "
                  f"${r['gap'][0]:.0f}[{r['gap'][1]:.0f},{r['gap'][2]:.0f}] | feas conc {r['conc_feas']:.2f}"
                  f" vs dist {r['dist_feas']:.2f}")
        out["loadstress"]["workloads"][w] = rows

    json.dump(out, open(os.path.join(args.outdir, "congestion_sweeps.json"), "w"), indent=2, default=float)
    print(f"\nsaved -> {os.path.join(args.outdir, 'congestion_sweeps.json')}")
    _plot(args.outdir, out)


def _plot(outdir, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    COL = {"inference": "#1f77b4", "training": "#d62728"}
    # ---- penetration ----
    P = out["penetration"]; wl = list(P["workloads"].keys())
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for w in wl:
        r = P["workloads"][w]; B = [x["B"] for x in r]
        g = [x["gap"][0] for x in r]
        ax[0].plot(B, g, "-o", color=COL[w], lw=2, label=w)
        ax[0].fill_between(B, [x["gap"][1] for x in r], [x["gap"][2] for x in r], color=COL[w], alpha=0.15)
        ax[1].plot(B, [x["conc_feas"] for x in r], "-o", color=COL[w], lw=2, label=f"{w}: concentrate")
        ax[1].plot(B, [x["dist_feas"] for x in r], "--s", color=COL[w], lw=2, alpha=0.7, label=f"{w}: distribute")
    ax[0].axhline(0, ls=":", c="gray")
    ax[0].set_xlabel("data-center fleet (GW)")
    ax[0].set_ylabel("extra congestion from concentrating  (conc − dist, $/MWh)")
    ax[0].set_title(f"Penetration: distribute benefit grows with fleet (load x{P['load']})"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("data-center fleet (GW)"); ax[1].set_ylabel("feasible fraction of hours")
    ax[1].set_ylim(0, 1.05); ax[1].set_title("Penetration: feasibility (must-serve)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "penetration_congestion.png"), dpi=130)
    print("saved penetration_congestion.png")
    # ---- load-stress ----
    Ls = out["loadstress"]; fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for w in wl:
        r = Ls["workloads"][w]; L = [x["load"] for x in r]
        ax[0].plot(L, [x["base_disp"] for x in r], ":^", color=COL[w], lw=1.5, alpha=0.6, label=f"{w}: base (no DC)")
        ax[0].plot(L, [x["gap"][0] for x in r], "-o", color=COL[w], lw=2, label=f"{w}: distribute benefit (conc−dist)")
        ax[0].fill_between(L, [x["gap"][1] for x in r], [x["gap"][2] for x in r], color=COL[w], alpha=0.15)
        ax[1].plot(L, [x["conc_feas"] for x in r], "-o", color=COL[w], lw=2, label=f"{w}: concentrate")
        ax[1].plot(L, [x["dist_feas"] for x in r], "--s", color=COL[w], lw=2, alpha=0.7, label=f"{w}: distribute")
    ax[0].axhline(0, ls=":", c="gray")
    ax[0].set_xlabel("load-stress multiplier (tighter →)"); ax[0].set_ylabel("$/MWh")
    ax[0].set_title(f"Load-stress: base congestion + distribute benefit (fleet {Ls['fleet']} GW)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("load-stress multiplier (tighter →)"); ax[1].set_ylabel("feasible fraction of hours")
    ax[1].set_ylim(0, 1.05); ax[1].set_title("Load-stress: feasibility (must-serve)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "loadstress_congestion.png"), dpi=130)
    print("saved loadstress_congestion.png")


if __name__ == "__main__":
    main()
