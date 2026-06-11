"""
PHASE-B FINISH: firm up the 3 deferred robustness items for the DC-placement paper.

Reuses the exact machinery from the lever/congestion study; only varies seeds, fleets, the
uniform-line reinforcement range, the chunking definitions, and the siting-weight assumption.

  Part A (items 1+2): for B in {3,6,10}, both workloads, n_seeds x n_pools:
     - WALL: concentrate vs distribute must-serve feasibility (conc_vs_dist) -> mean[min,max] over seeds
     - UNIFORM-LINE curve: feasibility of the CONCENTRATED fleet when ALL lines are scaled by (1+X),
       X in {0,0.1,0.25,0.5,1,2,4} (up to ~5x), seed-averaged, with annualized $ -> crossover X/$ where
       uniform-line feasibility first reaches distribute's feasibility (or "never within 5x").
  Part B (item 3b): wall under different chunking -- spread in {20,40,80 sites} x clump in {0.5,1,2 GW},
     fleets {6,10}, both workloads, seed-averaged. The wall should survive every definition.
  Part C (item 3a, deliverability side): wall under cheap-land siting weights (w=1/land) vs UNIFORM
     weights, fleets {6,10}. The wall should hold whether or not data centers chase cheap land.

Dollar-only cost sensitivity (line/gen/DC-build) is analytic and lives in dc_cost_sensitivity.py.

Usage:
  .venv/bin/python development/dc_phaseB_finish.py --quick
  .venv/bin/python development/dc_phaseB_finish.py --tag full
"""
import argparse, os, json, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_placement_study import (load_dc_profile, sample_panel_indices, metrics,
                                find_bad_buses, load_land_weights, draw_fleet)
from dc_placement_integrated import build_raw_hour, WORKLOADS
from dc_placement_congestion import cleaned_panel, incr, boot_ci, pack
from dc_placement_congestion_sweeps import conc_vs_dist
from dc_placement_levers import expand_uniform, eval_capacities_full, calibrate_usd, dc_build_cost
from dc_cleaning import DEFAULT_BAD_BUSES_JSON, cleaning_metadata, resolve_bad_buses

X_UNIFORM = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]      # up to ~5x all lines


def seed_stat(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return {"mean": np.nan, "min": np.nan, "max": np.nan, "seed_range": [np.nan, np.nan], "n": 0}
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            "seed_range": [float(v.min()), float(v.max())], "n": int(v.size)}


def boot_mean_ci(vals, seed=12345, nboot=2000):
    """Bootstrap CI for a mean over seed x pool samples."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return [np.nan, np.nan, np.nan]
    rng = np.random.default_rng(seed)
    b = np.array([np.mean(rng.choice(v, v.size, replace=True)) for _ in range(nboot)])
    return [float(np.mean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def write_json(path, out, partial, progress=None):
    out["_partial"] = bool(partial)
    if progress is not None:
        out["progress"] = progress
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)


def uniform_curve_one_seed(panel, base_ms, clean_nodes, eta_line, cc_line, nodes, weights,
                           B, m_conc, n_pools, pool_size, usd_line, seed,
                           progress_cb=None):
    """One seed: draw n_pools CONCENTRATE fleets, return per-X mean feasibility + inv ($) over pools."""
    rng = np.random.default_rng(seed)
    wdict = {n: weights[n] for n in nodes}
    xs = [0.0] + X_UNIFORM
    feas = {x: [] for x in xs}
    inv = {x: [] for x in xs}
    for pi in range(n_pools):
        pool = draw_fleet(rng, nodes, wdict, B, B / pool_size)[0]
        ct, ccp = pack(pool, B, m_conc)
        for x in xs:
            cap, iv = expand_uniform(eta_line, cc_line, x, usd_line)
            _, f, _, _ = eval_capacities_full(panel, base_ms, clean_nodes, 3, cap, ct, ccp, rng)
            feas[x].append(f); inv[x].append(iv)
        if progress_cb is not None:
            progress_cb(seed, pi + 1)
    return {x: (float(np.mean(feas[x])), float(np.median(inv[x]))) for x in xs}


def crossover(curve_meanfeas, target):
    """First X (and $) where seed-averaged uniform-line feasibility >= target (distribute feas)."""
    xs = [0.0] + X_UNIFORM
    for x in xs:
        f, iv = curve_meanfeas[x]
        if f >= target:
            return {"x": x, "usd": iv, "feas": f}
    top = curve_meanfeas[xs[-1]]
    return {"x": None, "usd": None, "feas_at_max": top[0], "note": f"never reaches {target:.2f} within +{int(xs[-1]*100)}%"}


def conc_vs_dist_progress(panel, base_ms, clean_nodes, nodes, weights, rng, B, n_pools,
                          pool_size, progress_cb=None):
    """Same calculation as conc_vs_dist, with optional per-pool progress callbacks."""
    m_conc, m_dist = int(np.ceil(B)), pool_size
    gap, cd, dd, cf, df = [], [], [], [], []
    wdict = {n: weights[n] for n in nodes}
    for pi in range(n_pools):
        pool = draw_fleet(rng, nodes, wdict, B, B / pool_size)[0]
        tc, cc = pack(pool, B, m_conc)
        rc = incr(panel, base_ms, clean_nodes, tc, cc)
        td, cdp = pack(pool, B, m_dist)
        rd = incr(panel, base_ms, clean_nodes, td, cdp)
        both = np.isfinite(rc["d_disp"]) & np.isfinite(rd["d_disp"])
        gap.extend((rc["d_disp"][both] - rd["d_disp"][both]).tolist())
        cd.extend(rc["d_disp"][np.isfinite(rc["d_disp"])].tolist())
        dd.extend(rd["d_disp"][np.isfinite(rd["d_disp"])].tolist())
        cf.append(float(rc["feas"].mean()))
        df.append(float(rd["feas"].mean()))
        if progress_cb is not None:
            progress_cb(pi + 1)
    return {"gap": boot_ci(gap, rng), "conc_disp": boot_ci(cd, rng),
            "dist_disp": boot_ci(dd, rng), "conc_feas": float(np.mean(cf)),
            "dist_feas": float(np.mean(df)), "conc_feas_pools": cf,
            "dist_feas_pools": df, "m_conc": m_conc, "m_dist": m_dist}


def wall_seeds(panel, base_ms, clean_nodes, nodes, weights, B, n_seeds, n_pools, pool_size,
               progress_cb=None):
    """conc/dist feasibility + paired gap across n_seeds; returns seed_stat dicts."""
    cf, df, gaps, cf_pools, df_pools = [], [], [], [], []
    mc = md = None
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        def pool_progress(pool_done, seed=s):
            if progress_cb is not None:
                progress_cb(seed, pool_done)

        r = conc_vs_dist_progress(panel, base_ms, clean_nodes, nodes, weights, rng, B,
                                  n_pools, pool_size, pool_progress)
        cf.append(r["conc_feas"]); df.append(r["dist_feas"]); gaps.append(r["gap"][0])
        cf_pools.extend(r["conc_feas_pools"]); df_pools.extend(r["dist_feas_pools"])
        mc, md = r["m_conc"], r["m_dist"]
    conc, dist = seed_stat(cf), seed_stat(df)
    conc["bootstrap_ci_seed_pool_mean"] = boot_mean_ci(cf_pools)
    dist["bootstrap_ci_seed_pool_mean"] = boot_mean_ci(df_pools)
    return {"conc": conc, "dist": dist, "gap_disp": seed_stat(gaps),
            "m_conc": mc, "m_dist": md,
            "separated": bool(np.nanmin(df) > np.nanmax(cf))}    # dist worst > conc best


def build_base(pn, snaps, idx, load_scale, bad_buses_json=None):
    raw = [build_raw_hour(pn, snaps, h, load_scale, 1.0, 1.0) for h in idx]
    n_nodes = raw[0][0].num_nodes
    detected_bad = None
    if not bad_buses_json:
        raw0 = [build_raw_hour(pn, snaps, h, 1.0, 1.0, 1.0) for h in idx]
        detected_bad = find_bad_buses([(net, devs, dt, 0.0) for (net, devs, dt, _) in raw0])
    bad = resolve_bad_buses(bad_buses_json, detected_bad)
    clean = [i for i in range(n_nodes) if i not in set(bad)]
    return raw, n_nodes, bad, clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=12)
    ap.add_argument("--load-scale", type=float, default=1.2)
    ap.add_argument("--fleets", default="3,6,10",
                    help="Part A fleet sizes in GW")
    ap.add_argument("--uniform-fleets", default="",
                    help="comma-separated Part A fleets that need the uniform-line curve; "
                         "default is every Part A fleet")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed cells already present in the output JSON")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--n-pools", type=int, default=24)
    ap.add_argument("--uniform-pools", type=int, default=12,
                    help="pools for the uniform-line curve (smoother metric -> fewer pools needed)")
    ap.add_argument("--b-seeds", type=int, default=4, help="seeds for Part B/C robustness sweeps")
    ap.add_argument("--b-pools", type=int, default=16, help="pools for Part B/C robustness sweeps")
    ap.add_argument("--pool-size", type=int, default=40)
    ap.add_argument("--land-cost", default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--bad-buses-json", default=DEFAULT_BAD_BUSES_JSON,
                    help="frozen pathological-bus manifest for citable runs; pass '' to redetect")
    ap.add_argument("--outdir", default="development/results/placement_robustness")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--parts", default="ABC", help="which parts to run")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_snaps = 6; args.n_seeds = 2; args.n_pools = 6
        args.uniform_pools = 4; args.b_seeds = 2; args.b_pools = 6
        global X_UNIFORM
        X_UNIFORM = [0.25, 1.0, 4.0]
    os.makedirs(args.outdir, exist_ok=True)

    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    idx = sample_panel_indices(len(snaps), args.n_snaps, None, None, 0)
    panel_hours = [str(snaps[i]) for i in idx]
    bad_manifest = args.bad_buses_json or None
    raw, n_nodes, bad, clean = build_base(pn, snaps, idx, args.load_scale, bad_manifest)
    land, weights = load_land_weights(args.land_cost, n_nodes)
    uni_w = {i: 1.0 for i in range(n_nodes)}        # uniform siting weights (Part C)
    usd_line, usd_gen, usd_rep = calibrate_usd(pn, raw[0][1])
    eta_line = np.asarray(raw[0][1][3].nominal_capacity, float)
    cc_line = np.asarray(raw[0][1][3].capital_cost, float)
    fleets = [float(x) for x in args.fleets.split(",")]
    print(f"# PHASE-B FINISH  load x{args.load_scale}  bad {len(bad)}  clean {len(clean)}  "
          f"seeds {args.n_seeds} pools {args.n_pools}  usd_line {usd_line:.3g}")

    out = {"load_scale": args.load_scale, "n_snaps": args.n_snaps,
           "panel_selection": "sample_panel_indices(n_total, n_snaps, None, None, seed=0)",
           "panel_indices": [int(i) for i in idx],
           "panel_hours": panel_hours,
           "pool_size": args.pool_size, "fleets": fleets,
           **cleaning_metadata(bad_manifest, bad),
           "clean_nodes_count": len(clean), "n_seeds": args.n_seeds,
           "n_pools": args.n_pools, "X_uniform": X_UNIFORM, "usd_line": usd_line}
    path = os.path.join(args.outdir, f"phaseB_finish_{args.tag}.json")
    if args.resume and os.path.exists(path):
        prev = json.load(open(path))
        same_cleaning = (prev.get("bad_buses") == out.get("bad_buses")
                         and prev.get("bad_buses_json") == out.get("bad_buses_json"))
        same_panel = (prev.get("n_snaps") == out.get("n_snaps")
                      and prev.get("n_seeds") == out.get("n_seeds")
                      and prev.get("n_pools") == out.get("n_pools")
                      and prev.get("panel_indices") == out.get("panel_indices"))
        if same_cleaning and same_panel:
            prev.update({k: v for k, v in out.items()
                         if k not in ("partA", "partB", "partC", "progress", "_partial")})
            out = prev
        else:
            print("# resume ignored: existing output metadata does not match this run")
    write_json(path, out, partial=True, progress="initialized")

    # precompute per-workload cleaned panel + base metrics (reused across parts)
    panels, basems = {}, {}
    for w in WORKLOADS:
        lf = load_dc_profile(WORKLOADS[w])
        cp = cleaned_panel(raw, lf, bad)
        panels[w] = cp
        basems[w] = [metrics(net, devs, devs, 1, price_nodes=clean) for (net, devs, _, _) in cp]

    # ---------- PART A: seed-pinned wall + uniform-line crossover ----------
    if "A" in args.parts:
        print("\n## PART A -- seed-pinned wall + uniform-line crossover")
        A = out.setdefault("partA", {})
        uniform_fleets = ({int(float(x)) for x in args.uniform_fleets.split(",") if x}
                          if args.uniform_fleets else {int(B) for B in fleets})
        for B in fleets:
            A.setdefault(str(int(B)), {})
            for w in WORKLOADS:
                if w in A[str(int(B))]:
                    print(f"   B={B:>4} {w:9s}: resume existing cell")
                    continue

                def wall_progress(seed_done, pool_done, B=B, w=w):
                    write_json(path, out, partial=True,
                               progress={"part": "A", "stage": "wall", "fleet_gw": float(B),
                                         "workload": w, "seed_done": int(seed_done),
                                         "pool_done": int(pool_done)})

                wl = wall_seeds(panels[w], basems[w], clean, clean, weights, B,
                                args.n_seeds, args.n_pools, args.pool_size, wall_progress)
                # uniform-line curve, seed-averaged
                xs = [0.0] + X_UNIFORM
                if int(B) in uniform_fleets:
                    per_seed = []
                    for s in range(args.n_seeds):
                        def uniform_progress(seed_done, pool_done, B=B, w=w):
                            write_json(path, out, partial=True,
                                       progress={"part": "A", "stage": "uniform",
                                                 "fleet_gw": float(B), "workload": w,
                                                 "seed_done": int(seed_done),
                                                 "pool_done": int(pool_done)})

                        per_seed.append(uniform_curve_one_seed(
                            panels[w], basems[w], clean, eta_line, cc_line, clean, weights, B,
                            wl["m_conc"], args.uniform_pools, args.pool_size, usd_line, s,
                            uniform_progress))
                    curve = {x: (float(np.mean([ps[x][0] for ps in per_seed])),
                                 float(np.median([ps[x][1] for ps in per_seed]))) for x in xs}
                    xover = crossover(curve, wl["dist"]["mean"])
                else:
                    curve = {0.0: (wl["conc"]["mean"], 0.0)}
                    xover = {"x": None, "usd": None, "feas_at_max": wl["conc"]["mean"],
                             "note": "uniform-line curve skipped for this non-crossover fleet"}
                A[str(int(B))][w] = {"wall": wl, "uniform_curve": {str(x): curve[x] for x in curve},
                                     "crossover_to_dist": xover}
                cstr = (f"x={xover['x']} (${xover['usd']/1e9:.0f}B)" if xover["x"] is not None
                        else xover["note"])
                print(f"   B={B:>4} {w:9s}: conc {wl['conc']['mean']:.2f}[{wl['conc']['min']:.2f},"
                      f"{wl['conc']['max']:.2f}] dist {wl['dist']['mean']:.2f}[{wl['dist']['min']:.2f},"
                      f"{wl['dist']['max']:.2f}] sep={wl['separated']} | uniform-line reaches dist at {cstr}")
                write_json(path, out, partial=True,
                           progress={"part": "A", "fleet_gw": float(B), "workload": w})

    # ---------- PART B: chunking sensitivity ----------
    if "B" in args.parts:
        print("\n## PART B -- chunking sensitivity (spread sites x clump GW)")
        B_ = {}
        out["partB"] = B_
        for B in (6.0, 10.0):
            B_[str(int(B))] = {}
            for spread in (20, 40, 80):
                for clump in (0.5, 1.0, 2.0):
                    key = f"spread{spread}_clump{clump}"
                    cell = {}
                    for w in WORKLOADS:
                        # m_dist = #spread sites; m_conc = ceil(B/clump) lumps of `clump` GW
                        wl = wall_chunk(panels[w], basems[w], clean, weights, B, spread, clump,
                                        args.b_seeds, args.b_pools)
                        cell[w] = wl
                    B_[str(int(B))][key] = cell
                    print(f"   B={B:>4} {key:18s}: "
                          + "  ".join(f"{w[:3]} conc {cell[w]['conc']['mean']:.2f} dist "
                                      f"{cell[w]['dist']['mean']:.2f} sep={cell[w]['separated']}"
                                      for w in WORKLOADS))
                    write_json(path, out, partial=True,
                               progress={"part": "B", "fleet_gw": float(B), "cell": key})

    # ---------- PART C: land-siting sensitivity ----------
    if "C" in args.parts:
        print("\n## PART C -- siting weights: cheap-land vs uniform")
        C = {}
        out["partC"] = C
        for B in (6.0, 10.0):
            C[str(int(B))] = {}
            for wname, wt in (("cheap_land", weights), ("uniform", uni_w)):
                cell = {}
                for w in WORKLOADS:
                    wl = wall_seeds(panels[w], basems[w], clean, clean, wt, B,
                                    args.b_seeds, args.b_pools, args.pool_size)
                    cell[w] = wl
                C[str(int(B))][wname] = cell
                print(f"   B={B:>4} {wname:11s}: "
                      + "  ".join(f"{w[:3]} conc {cell[w]['conc']['mean']:.2f} dist "
                                  f"{cell[w]['dist']['mean']:.2f} sep={cell[w]['separated']}"
                                  for w in WORKLOADS))
                write_json(path, out, partial=True,
                           progress={"part": "C", "fleet_gw": float(B), "siting": wname})

    write_json(path, out, partial=False, progress="complete")
    print(f"\nsaved -> {path}")


def wall_chunk(panel, base_ms, clean, weights, B, spread, clump, n_seeds, n_pools):
    """Wall with explicit chunking: distribute over `spread` sites; concentrate into lumps of `clump` GW.
    Reproduces conc_vs_dist's paired logic but with caller-set m_conc/m_dist and pool_size=spread."""
    from dc_placement_congestion import incr, boot_ci
    m_conc = max(1, int(np.ceil(B / clump)))
    m_dist = spread
    pool_size = max(spread, m_conc)
    wdict = {n: weights[n] for n in clean}
    cf, df = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        cfs, dfs = [], []
        for _ in range(n_pools):
            pool = draw_fleet(rng, clean, wdict, B, B / pool_size)[0]
            tc, cc = pack(pool, B, m_conc); rc = incr(panel, base_ms, clean, tc, cc)
            td, dd = pack(pool, B, m_dist); rd = incr(panel, base_ms, clean, td, dd)
            cfs.append(float(rc["feas"].mean())); dfs.append(float(rd["feas"].mean()))
        cf.append(float(np.mean(cfs))); df.append(float(np.mean(dfs)))
    return {"conc": seed_stat(cf), "dist": seed_stat(df), "m_conc": m_conc, "m_dist": m_dist,
            "separated": bool(np.nanmin(df) > np.nanmax(cf))}


if __name__ == "__main__":
    main()
