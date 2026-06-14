"""
ROBUSTNESS of the must-serve feasibility WALL (the Phase-A headline) to the two axes a
reviewer would attack: (1) the network-CLEANING choice (bad-bus detection thresholds) and
(2) SEASON (winter-only vs summer-only sub-panels, vs the year-spanning panel).

The wall = concentrate-vs-distribute must-serve feasibility on the cleaned 490-node WECC.
We reuse the exact conc_vs_dist machinery from the congestion study; here we only vary the
cleaning thresholds and the panel window, and report conc/dist feasibility + the paired
dispersion-gap CI at fleets {6,10} GW for both workloads. If dist_feas >> conc_feas in every
cell, the wall is robust.

Load-stress robustness is already covered by congestion_sweeps.json (wall invariant to load
x1.1/1.2/1.3); the default panel is already year-spanning, so season here is a stricter check.

Usage:
  .venv/bin/python development/dc_wall_robustness.py            # full
  .venv/bin/python development/dc_wall_robustness.py --quick
"""
import argparse, os, json, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_placement_study import (load_dc_profile, sample_panel_indices, metrics,
                                find_bad_buses, load_land_weights)
from dc_placement_integrated import build_raw_hour, WORKLOADS
from dc_placement_congestion import cleaned_panel
from dc_placement_congestion_sweeps import conc_vs_dist

VOLL = 1000.0
# bad-bus detection threshold variants (hi_cap multiple of VOLL, lo_cap $, shed_frac)
THRESHOLDS = {
    "loose":   dict(hi_cap=2.0 * VOLL, lo_cap=-400.0, shed_frac=0.95),
    "default": dict(hi_cap=1.5 * VOLL, lo_cap=-200.0, shed_frac=0.90),
    "strict":  dict(hi_cap=1.2 * VOLL, lo_cap=-100.0, shed_frac=0.80),
}
# season windows (start hour-of-year, length in hours) on the 8760 grid
SEASONS = {
    "winter": (0, 24 * 60),          # Jan-Feb
    "spring": (24 * 90, 24 * 60),    # Apr-May
    "summer": (24 * 181, 24 * 60),   # Jul-Aug
    "fall":   (24 * 273, 24 * 60),   # Oct-Nov
    "year":   (None, None),          # the default year-spanning panel
}


def sample_labels(snaps, idx):
    return [str(snaps[i]) for i in idx]


def wall(panel, clean_nodes, weights, B, n_pools, pool_size, w):
    lf = load_dc_profile(WORKLOADS[w])
    cp = cleaned_panel(panel, lf, BAD)
    base_ms = [metrics(net, devs, devs, 1, price_nodes=clean_nodes) for (net, devs, _, _) in cp]
    rng = np.random.default_rng(0)
    r = conc_vs_dist(cp, base_ms, clean_nodes, clean_nodes, weights, rng, B, n_pools, pool_size)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=12)
    ap.add_argument("--load-scale", type=float, default=1.2)
    ap.add_argument("--fleets", default="6,10")
    ap.add_argument("--n-pools", type=int, default=24)
    ap.add_argument("--pool-size", type=int, default=40)
    ap.add_argument("--land-cost", default=None,
                    help="land $/acre CSV; omit for UNIFORM siting (cross-network default). "
                         "The 490 CSV is 490-specific and would median-fill other nets. "
                         "(This study re-detects bad buses per threshold by design -- that IS "
                         "the robustness axis -- deterministically at seed=0, so it takes no "
                         "frozen --bad-buses-json.)")
    ap.add_argument("--outdir", default="development/results/placement_robustness")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_snaps = 6; args.n_pools = 6; args.fleets = "10"
    os.makedirs(args.outdir, exist_ok=True)
    fleets = [float(x) for x in args.fleets.split(",")]

    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    n_total = len(snaps)
    idx_year = sample_panel_indices(n_total, args.n_snaps, None, None, 0)
    n_nodes = build_raw_hour(pn, snaps, idx_year[0], 1.0, 1.0, 1.0)[0].num_nodes
    land, weights = load_land_weights(args.land_cost, n_nodes)

    out = {"load_scale": args.load_scale, "fleets": fleets, "n_pools": args.n_pools,
           "n_snaps": args.n_snaps, "year_panel_indices": idx_year,
           "year_panel_hours": sample_labels(snaps, idx_year),
           "threshold_axis": {}, "season_axis": {}}

    # ============ AXIS 1: bad-bus cleaning thresholds (year panel, fixed) ============
    print("## AXIS 1 -- CLEANING-THRESHOLD sensitivity (year panel)")
    raw_year = [build_raw_hour(pn, snaps, h, args.load_scale, 1.0, 1.0) for h in idx_year]
    raw0_year = [build_raw_hour(pn, snaps, h, 1.0, 1.0, 1.0) for h in idx_year]
    global BAD
    for name, th in THRESHOLDS.items():
        BAD = find_bad_buses([(net, devs, dt, 0.0) for (net, devs, dt, _) in raw0_year], **th)
        clean_nodes = [i for i in range(n_nodes) if i not in set(BAD)]
        print(f"  threshold={name:8s} n_bad={len(BAD):3d}  clean={len(clean_nodes)}")
        out["threshold_axis"][name] = {"n_bad": len(BAD), "bad_buses": BAD,
                                       "sample_indices": idx_year,
                                       "sample_hours": sample_labels(snaps, idx_year),
                                       "fleets": {}}
        for B in fleets:
            cell = {}
            for w in WORKLOADS:
                r = wall(raw_year, clean_nodes, weights, B, args.n_pools, args.pool_size, w)
                cell[w] = {"conc_feas": r["conc_feas"], "dist_feas": r["dist_feas"], "gap": r["gap"]}
                print(f"      B={B:>4} {w:9s}: conc {r['conc_feas']:.2f}  dist {r['dist_feas']:.2f}"
                      f"  gap(conc-dist)Ddisp ${r['gap'][0]:.0f}[{r['gap'][1]:.0f},{r['gap'][2]:.0f}]")
            out["threshold_axis"][name]["fleets"][str(int(B))] = cell

    # ============ AXIS 2: season (bad buses fixed at DEFAULT, detected on year x1.0) ============
    print("\n## AXIS 2 -- SEASON sensitivity (bad buses fixed = default, year x1.0)")
    BAD = find_bad_buses([(net, devs, dt, 0.0) for (net, devs, dt, _) in raw0_year],
                         **THRESHOLDS["default"])
    clean_nodes = [i for i in range(n_nodes) if i not in set(BAD)]
    for name, (ws, wl) in SEASONS.items():
        idx = sample_panel_indices(n_total, args.n_snaps, ws, wl, 0)
        raw = [build_raw_hour(pn, snaps, h, args.load_scale, 1.0, 1.0) for h in idx]
        print(f"  season={name:7s} hours={[str(snaps[i])[-21:-11] for i in idx][:4]}...")
        out["season_axis"][name] = {
            "window_start": ws,
            "window_length": wl,
            "sample_indices": idx,
            "sample_hours": sample_labels(snaps, idx),
            "n_bad": len(BAD),
            "bad_buses": BAD,
            "fleets": {},
        }
        for B in fleets:
            cell = {}
            for w in WORKLOADS:
                r = wall(raw, clean_nodes, weights, B, args.n_pools, args.pool_size, w)
                cell[w] = {"conc_feas": r["conc_feas"], "dist_feas": r["dist_feas"], "gap": r["gap"]}
                print(f"      B={B:>4} {w:9s}: conc {r['conc_feas']:.2f}  dist {r['dist_feas']:.2f}"
                      f"  gap ${r['gap'][0]:.0f}[{r['gap'][1]:.0f},{r['gap'][2]:.0f}]")
            out["season_axis"][name]["fleets"][str(int(B))] = cell

    path = os.path.join(args.outdir, f"wall_robustness_{args.tag}.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\nsaved -> {path}")


BAD = []
if __name__ == "__main__":
    main()
