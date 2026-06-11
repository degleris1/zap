"""
INTEGRATED DC PLACEMENT STUDY on a STRESSED 490-node WECC grid.

Calibration showed the grid is only interesting when stressed: at load x1.2 the
electricity price a 1 GW data center pays swings ~30x ($23 -> $727/MWh) depending
on WHERE you put it. This script studies all three questions on that stressed grid,
reporting INCREMENTAL effects vs the no-DC base (so the handful of structurally
unservable buses -- a network data artifact -- cancel out).

  H2  LOCATIONAL PRICE: where you place data centers swings local LMP enormously.
      -> rank every usable location (sink nodes with surplus generation vs deficit
         nodes), and draw siting curves (price & extra shedding vs GW injected).
  H1  DISTRIBUTE vs CONCENTRATE, disentangled from placement QUALITY:
      -> a 2x2 of {good (sink) nodes, random cheap-land, bad (deficit) nodes}
         x {few big 1 GW sites, many small 0.2 GW sites} at a fixed fleet.
         Separates "where" (quality) from "how spread" (granularity).
  H3  WORKLOAD x PLACEMENT: does the answer change for inference (diurnal) vs
      training (flat) workloads? -> repeat H2 ranking + the H1 table per workload.

Usage:
  .venv/bin/python development/dc_placement_integrated.py --load-scale 1.2
"""
import argparse, os, json, sys

import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_placement_study import (
    POWER_UNIT, COST_UNIT, VOLL,
    load_dc_profile, sample_panel_indices, with_dc, metrics, summarize, summarize_all,
    feas_frac, load_land_weights, draw_fleet, screen_usable,
)
from zap.importers.pypsa import load_pypsa_network
from copy import deepcopy

WORKLOADS = {
    "inference": "development/load_profiles/example_inference_azure_conv.csv",  # diurnal 0.55-0.75
    "training":  "development/load_profiles/dummy_train.csv",                   # flat ~0.90
}


# --------------------------------------------------------------------------- #
def build_raw_hour(pn, snaps, h, ls, gs, lns):
    """One hour's network/devices (no DC) + its hour-of-day. Workload-independent."""
    import pandas as pd
    net, devices = load_pypsa_network(pn, snaps[h:h + 1], power_unit=POWER_UNIT, cost_unit=COST_UNIT)
    devices = deepcopy(devices)
    devices[1].load *= ls
    devices[0].dynamic_capacity *= gs
    devices[3].nominal_capacity *= lns
    ts = snaps[h]
    hod = int(pd.Timestamp(ts[-1] if isinstance(ts, tuple) else ts).hour)
    return net, devices, str(ts), hod


def panel_for(raw_panel, hourly_lf):
    """Attach a workload's hour-of-day load factor to each raw hour -> a panel the
    dc_placement_study helpers (panel_eval, metrics) understand."""
    return [(net, devs, date, float(hourly_lf[hod])) for (net, devs, date, hod) in raw_panel]


def panel_eval(panel, terminals, caps):
    return [metrics(net, with_dc(devs, terminals, caps, lf), devs, 1, dc_terminals=terminals)
            for (net, devs, _, lf) in panel]


def incr(panel, base_ms, terminals, caps):
    """Evaluate a placement and return INCREMENTAL effects vs the no-DC base, per hour,
    aggregated: extra grid stress, extra % load shed, and the DC-terminal price."""
    ms = panel_eval(panel, terminals, caps)
    d_stress, d_shed, dclmp, feas = [], [], [], []
    for m, b in zip(ms, base_ms):
        feas.append(m["feasible"])
        if m["feasible"] and b["feasible"]:
            d_stress.append(m["sum_u2"] - b["sum_u2"])
            d_shed.append(max(m["shed_pct"] - b["shed_pct"], 0.0))
            dclmp.append(m["dc_lmp"])
    f = lambda v: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90))} if v else {"median": np.inf, "p90": np.inf}
    return {"feas_frac": float(np.mean(feas)), "n_sites": len(terminals),
            "d_stress": f(d_stress), "d_shed": f(d_shed), "dc_lmp": f(dclmp)}


def rank_locations(panel, base_ms, nodes, probe):
    """For every location: the median DC price and extra shedding from a `probe`-GW
    site, across the panel. The locational price signal (H2)."""
    rows = []
    for nd in nodes:
        r = incr(panel, base_ms, [nd], [probe])
        rows.append({"node": int(nd), "dc_lmp": r["dc_lmp"]["median"],
                     "d_shed": r["d_shed"]["median"], "d_stress": r["d_stress"]["median"],
                     "feas_frac": r["feas_frac"]})
    rows.sort(key=lambda r: (-r["feas_frac"], r["dc_lmp"]))   # feasible + cheap first
    return rows


def pack(pool, total, site):
    n = int(np.ceil(total / site - 1e-9))
    n = min(n, len(pool))
    terms = list(pool[:n])
    caps = [site] * n
    if caps:
        caps[-1] = total - site * (n - 1)
    return terms, caps


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-scale", type=float, default=1.2, help="stress the grid (calibrated 1.2)")
    ap.add_argument("--gen-scale", type=float, default=1.0)
    ap.add_argument("--fleet", type=float, default=4.0, help="total DC fleet for the H1 2x2 (GW)")
    ap.add_argument("--big", type=float, default=1.0, help="big-site size (GW)")
    ap.add_argument("--small", type=float, default=0.2, help="small-site size (GW)")
    ap.add_argument("--probe", type=float, default=0.5, help="probe site size for ranking (GW)")
    ap.add_argument("--n-good", type=int, default=20, help="size of the good/bad node pools")
    ap.add_argument("--land-cost", default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--outdir", default="development/results/placement_integrated")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    idx = sample_panel_indices(len(snaps), args.n_snaps, None, None, args.seed)
    raw_panel = [build_raw_hour(pn, snaps, h, args.load_scale, args.gen_scale, 1.0) for h in idx]
    n_nodes = raw_panel[0][0].num_nodes
    nodes = list(range(n_nodes if not args.max_nodes else args.max_nodes))
    land, weights = load_land_weights(args.land_cost, n_nodes)
    rng = np.random.default_rng(args.seed)

    print(f"# INTEGRATED DC PLACEMENT (stressed grid)  network={os.path.basename(args.network)}")
    print(f"# load x{args.load_scale}  gen x{args.gen_scale}  | {len(raw_panel)} hours across the year"
          f"  | fleet {args.fleet} GW, sites <= {args.big} GW")

    out = {"load_scale": args.load_scale, "fleet": args.fleet, "panel_idx": idx, "workloads": {}}

    for wname, wpath in WORKLOADS.items():
        hourly_lf = load_dc_profile(wpath)
        panel = panel_for(raw_panel, hourly_lf)
        base_ms = [metrics(net, devs, devs, 1) for (net, devs, _, _) in panel]
        base = summarize_all(base_ms)
        print(f"\n{'='*78}\n## WORKLOAD = {wname.upper()}  (load-factor {hourly_lf.min():.2f}-{hourly_lf.max():.2f})")
        print(f"# base (no DC): stress {base['sum_u2']['median']:.0f}, p95 ${base['p95_lmp']['median']:.0f}, "
              f"shed {base['shed_pct']['median']:.2f}%  [shedding mostly from ~10 unservable buses; "
              f"we report effects INCREMENTAL to this]")

        # ---- H2: locational price ranking + siting curves ----
        ranked = rank_locations(panel, base_ms, nodes, args.probe)
        usable = [r for r in ranked if r["feas_frac"] > 0.99]
        lmps = np.array([r["dc_lmp"] for r in usable])
        print(f"\n## H2  LOCATIONAL PRICE  (a {args.probe} GW site; {len(usable)}/{len(nodes)} usable locations)")
        print(f"#  DC price by location: cheapest ${lmps.min():.0f}  median ${np.median(lmps):.0f}  "
              f"priciest ${lmps.max():.0f}  -> {lmps.max()-lmps.min():.0f} $/MWh spread from placement alone")
        sink = [r["node"] for r in usable[:args.n_good]]               # cheap/surplus nodes
        deficit = [r["node"] for r in usable[-args.n_good:]][::-1]     # priciest feasible nodes
        print(f"#  top sink (cheap) nodes: {sink[:6]}  ...  top deficit (pricey) nodes: {deficit[:6]}")

        curve_nodes = [usable[0]["node"], usable[len(usable)//2]["node"], usable[-1]["node"]]
        labels = ["best-sink", "median", "worst-deficit"]
        print("#  siting curves (DC price $/MWh @ injected GW):")
        curves = {}
        for nd, lab in zip(curve_nodes, labels):
            pts = []
            for g in [0.1, 0.25, 0.5, 1.0]:
                r = incr(panel, base_ms, [nd], [g])
                pts.append({"gw": g, "dc_lmp": r["dc_lmp"]["median"], "d_shed": r["d_shed"]["median"],
                            "feas": r["feas_frac"]})
            curves[nd] = {"label": lab, "pts": pts}
            s = "  ".join(f"{p['gw']}GW:${p['dc_lmp']:.0f}" + ("" if p["feas"] > 0.99 else f"(f{p['feas']:.1f})") for p in pts)
            print(f"     node {nd:3d} ({lab:13s}): {s}")

        # ---- H1: quality x spread 2x2 ----
        B = args.fleet
        rfleet = draw_fleet(rng, [r["node"] for r in usable], {r["node"]: weights[r["node"]] for r in usable}, B, args.small)[0]
        pools = {"good (sink)": sink, "random (cheap land)": rfleet, "bad (deficit)": deficit}
        print(f"\n## H1  QUALITY x SPREAD  (fleet {B} GW; incremental vs base)")
        print(f"  {'placement':>20} {'spread':>10} {'#sites':>6} {'works':>5} {'+stress':>8} {'+shed%':>7} {'DC$':>6}")
        h1 = {}
        for qname, pool in pools.items():
            for sname, site in [("few big", args.big), ("many small", args.small)]:
                terms, caps = pack(pool, B, site)
                r = incr(panel, base_ms, terms, caps)
                h1[f"{qname} | {sname}"] = r
                print(f"  {qname:>20} {sname:>10} {len(terms):6d} {r['feas_frac']:5.2f} "
                      f"{r['d_stress']['median']:8.1f} {r['d_shed']['median']:7.3f} {r['dc_lmp']['median']:6.0f}")

        out["workloads"][wname] = {
            "base": base, "lmp_spread": [float(lmps.min()), float(np.median(lmps)), float(lmps.max())],
            "ranking": ranked, "sink": sink, "deficit": deficit,
            "curves": {str(k): v for k, v in curves.items()}, "h1_2x2": h1}

    # ---- H3 cross-workload headline ----
    print(f"\n{'='*78}\n## H3  DOES WORKLOAD CHANGE THE ANSWER?")
    wl = list(out["workloads"].keys())
    print(f"  {'metric':>34} | " + " | ".join(f"{w:>12}" for w in wl))
    a, b = out["workloads"][wl[0]], out["workloads"][wl[1]]
    print(f"  {'price spread $/MWh (placement)':>34} | " +
          " | ".join(f"{out['workloads'][w]['lmp_spread'][2]-out['workloads'][w]['lmp_spread'][0]:12.0f}" for w in wl))
    for key in ["good (sink) | many small", "bad (deficit) | few big", "random (cheap land) | many small"]:
        print(f"  {key+' +shed%':>34} | " +
              " | ".join(f"{out['workloads'][w]['h1_2x2'][key]['d_shed']['median']:12.3f}" for w in wl))

    json.dump(out, open(os.path.join(args.outdir, "integrated_results.json"), "w"), indent=2, default=float)
    print(f"\nsaved -> {os.path.join(args.outdir, 'integrated_results.json')}")


if __name__ == "__main__":
    main()
