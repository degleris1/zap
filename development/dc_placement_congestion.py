"""
DC PLACEMENT vs GRID CONGESTION (cleaned 490-node WECC, must-serve DC).

Redesign per the approved plan. Measures grid CONGESTION (not the DC's private bill)
via the dual formulation, on a CLEANED network, with must-serve (guaranteed) DC.

  Cleaning: ~25 structurally-pathological buses (unservable, or +1e4 / -$600 LMP
  pockets) are detected ONCE on the no-DC base at load x1.0 (where pockets are
  distinguishable from real congestion) and held fixed; their phantom loads are
  zeroed and they are excluded from candidate pools and from price statistics. This
  removes the artifacts that made the old "sink" nodes (155/160/166) look best.

  Congestion metric (robust, dual-based): primary = LMP dispersion (p90-p10 of nodal
  LMPs over clean buses) and congestion rent / merchandising surplus ($/h); cross-
  check = line-flow shadow-price rent + #binding lines. All effects reported
  INCREMENTAL to the no-DC base (so residual base shed cancels).

  H2 (where): per-node marginal congestion impact -- the change in congestion from a
      standard 0.25 GW must-serve block. A continuous landscape (relieving < 0,
      worsening > 0), no good/med/bad bins.
  H1 (spread): a CONCENTRATION SWEEP -- fix the fleet, draw a random cheap-land pool,
      and sweep the number of sites the fleet is spread over (concentrate few 1 GW
      lumps -> distribute many 0.2 GW sites). Spread is isolated from "where" because
      the candidate pool is held fixed and random. Bootstrap CIs over pools/hours.
  H3: every result for inference (diurnal) and training (flat) workloads.

Usage:
  .venv/bin/python development/dc_placement_congestion.py --load-scale 1.2
"""
import argparse, os, json, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_placement_study import (load_dc_profile, sample_panel_indices, with_dc, metrics,
                                 find_bad_buses, clean_devices, load_land_weights, draw_fleet)
from dc_placement_integrated import build_raw_hour, WORKLOADS

NBOOT = 1500


def cleaned_panel(raw_panel, hourly_lf, bad):
    """Attach workload load-factor and clean each hour's devices."""
    return [(net, clean_devices(devs, bad), date, float(hourly_lf[hod]))
            for (net, devs, date, hod) in raw_panel]


def incr(panel, base_ms, clean_nodes, terms, caps):
    """Per-hour incremental congestion vs the no-DC base, aligned by hour (NaN where
    infeasible). Returns dict of arrays: d_rent ($/h), d_disp ($/MWh), dc_lmp, feas."""
    H = len(panel)
    d_rent = np.full(H, np.nan); d_disp = np.full(H, np.nan); dclmp = np.full(H, np.nan)
    feas = np.zeros(H, bool)
    for i, ((net, devs, _, lf), b) in enumerate(zip(panel, base_ms)):
        m = metrics(net, with_dc(devs, terms, caps, lf), devs, 1,
                    dc_terminals=terms, price_nodes=clean_nodes)
        feas[i] = m["feasible"]
        if m["feasible"] and b["feasible"]:
            d_rent[i] = m["cong_rent"] - b["cong_rent"]
            d_disp[i] = m["lmp_disp"] - b["lmp_disp"]
            dclmp[i] = m["dc_lmp"]
    return {"d_rent": d_rent, "d_disp": d_disp, "dc_lmp": dclmp, "feas": feas}


def boot_ci(vals, rng, stat=np.median):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    b = np.array([stat(rng.choice(v, v.size, replace=True)) for _ in range(NBOOT)])
    return float(stat(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def pack(pool, total, m):
    """Spread `total` GW over m sites from pool, each <= 1 GW."""
    m = min(m, len(pool))
    site = total / m
    return list(pool[:m]), [site] * m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=12)
    ap.add_argument("--load-scale", type=float, default=1.2)
    ap.add_argument("--fleet", type=float, default=6.0, help="fleet for the H1 concentration sweep (GW)")
    ap.add_argument("--probe", type=float, default=0.25, help="H2 marginal-impact block (GW)")
    ap.add_argument("--node-stride", type=int, default=4, help="sample every Nth clean node for H2")
    ap.add_argument("--n-pools", type=int, default=15, help="random cheap-land pools for H1")
    ap.add_argument("--pool-size", type=int, default=40, help="candidate locations per random pool")
    ap.add_argument("--land-cost", default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--outdir", default="development/results/placement_congestion")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    idx = sample_panel_indices(len(snaps), args.n_snaps, None, None, 0)
    raw = [build_raw_hour(pn, snaps, h, args.load_scale, 1.0, 1.0) for h in idx]
    n_nodes = raw[0][0].num_nodes
    rng = np.random.default_rng(0)

    # --- detect bad buses ONCE at load x1.0 (fixed across the study) ---
    raw0 = [build_raw_hour(pn, snaps, h, 1.0, 1.0, 1.0) for h in idx]
    base0 = [(net, devs, date, 0.0) for (net, devs, date, hod) in raw0]   # lf irrelevant for base
    bad = find_bad_buses(base0)
    bad_set = set(bad)
    clean_nodes = [i for i in range(n_nodes) if i not in bad_set]
    nodes = clean_nodes if not args.max_nodes else clean_nodes[:args.max_nodes]
    land, weights = load_land_weights(args.land_cost, n_nodes)
    print(f"# DC PLACEMENT vs CONGESTION (cleaned grid)  load x{args.load_scale}")
    print(f"# bad buses removed: {len(bad)} (detected at x1.0) | clean nodes: {len(clean_nodes)}")

    out = {"load_scale": args.load_scale, "fleet": args.fleet, "n_bad": len(bad),
           "bad_buses": bad, "workloads": {}}

    for w, wpath in WORKLOADS.items():
        lf = load_dc_profile(wpath)
        panel = cleaned_panel(raw, lf, bad)
        base_ms = [metrics(net, devs, devs, 1, price_nodes=clean_nodes) for (net, devs, _, _) in panel]
        base_disp = np.median([m["lmp_disp"] for m in base_ms if m["feasible"]])
        base_shed = np.median([m["shed_pct"] for m in base_ms if m["feasible"]])
        print(f"\n## WORKLOAD={w}  base(no DC): LMP-disp ${base_disp:.0f}, shed {base_shed:.2f}%")

        # ---- H2: marginal congestion-impact landscape ----
        h2nodes = nodes[::args.node_stride]
        land_rel = []
        for nd in h2nodes:
            r = incr(panel, base_ms, clean_nodes, [nd], [args.probe])
            land_rel.append({"node": nd,
                             "d_disp": float(np.nanmedian(r["d_disp"])) if np.isfinite(r["d_disp"]).any() else np.nan,
                             "d_rent": float(np.nanmedian(r["d_rent"])) if np.isfinite(r["d_rent"]).any() else np.nan,
                             "dc_lmp": float(np.nanmedian(r["dc_lmp"])) if np.isfinite(r["dc_lmp"]).any() else np.nan,
                             "feas": float(r["feas"].mean())})
        dd = np.array([x["d_disp"] for x in land_rel if np.isfinite(x["d_disp"])])
        print(f"   H2 marginal congestion impact of a {args.probe}GW block over {len(h2nodes)} nodes:")
        print(f"      relieving (Δdisp<0): {int((dd<0).sum())} | neutral(~0): {int((np.abs(dd)<2).sum())} "
              f"| worsening(>0): {int((dd>2).sum())} | range Δdisp ${dd.min():.0f}..${dd.max():.0f}")

        # ---- H1: concentration sweep (isolate spread from where) ----
        B = args.fleet
        m_grid = sorted(set([int(np.ceil(B)), int(np.ceil(B))+1, 8, 12, 20, min(args.pool_size, 40)]))
        m_grid = [m for m in m_grid if m >= int(np.ceil(B)) and m <= args.pool_size]
        sweep = {m: {"d_rent": [], "d_disp": [], "feas": []} for m in m_grid}
        for _ in range(args.n_pools):
            pool = draw_fleet(rng, nodes, {n: weights[n] for n in nodes}, B, B / args.pool_size)[0]
            for m in m_grid:
                terms, caps = pack(pool, B, m)
                r = incr(panel, base_ms, clean_nodes, terms, caps)
                sweep[m]["d_rent"].extend(r["d_rent"][np.isfinite(r["d_rent"])].tolist())
                sweep[m]["d_disp"].extend(r["d_disp"][np.isfinite(r["d_disp"])].tolist())
                sweep[m]["feas"].append(float(r["feas"].mean()))
        print(f"   H1 concentration sweep (fleet {B}GW over m sites; lower m = more concentrated):")
        print(f"      {'#sites':>6} {'site(GW)':>9} {'Δrent($/h)':>20} {'Δdisp($)':>16} {'feasible':>9}")
        h1rows = []
        for m in m_grid:
            rent = boot_ci(sweep[m]["d_rent"], rng); disp = boot_ci(sweep[m]["d_disp"], rng)
            fe = float(np.mean(sweep[m]["feas"]))
            h1rows.append({"m": m, "site_gw": B / m, "d_rent": rent, "d_disp": disp, "feas": fe})
            print(f"      {m:6d} {B/m:9.2f} {rent[0]:8.0f}[{rent[1]:.0f},{rent[2]:.0f}]"
                  f" {disp[0]:6.1f}[{disp[1]:.1f},{disp[2]:.1f}] {fe:9.2f}")

        out["workloads"][w] = {"base_disp": float(base_disp), "base_shed": float(base_shed),
                               "landscape": land_rel, "h1_sweep": h1rows}

    json.dump(out, open(os.path.join(args.outdir, "congestion_results.json"), "w"), indent=2, default=float)
    print(f"\nsaved -> {os.path.join(args.outdir, 'congestion_results.json')}")


if __name__ == "__main__":
    main()
