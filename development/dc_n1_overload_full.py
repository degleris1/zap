"""
PUBLISHABLE N-1 OVERLOAD STUDY: the three siting contrasts under contingencies.

Builds on the differential N-1 overload metric (dc_n1_overload.py) -- total
post-contingency corridor overload O(placement) = sum_t sum_k sum_l
(|f+LODF[:,k]f_k| - Fbar)_+, minimized over the base dispatch (and placement for
the aware case), full N-1 via exact LODF, solved with HiGHS.

Three contrasts that mirror the base-case siting work, now under N-1:

  A. BUDGET SWEEP -- added N-1 overload vs DC fleet B for aware / uniform /
     concentrated. Tests whether the smart-vs-naive gap grows with budget
     (the base-case "edge grows with budget" result).

  B. CONCENTRATE -> DISTRIBUTE -- fixed B, sweep the per-site cap kappa; the
     optimizer is forced to spread as kappa shrinks. Added N-1 overload vs
     spread (the "100x1MW < 1x100MW" thesis, under N-1).

  C. GRID-STRENGTH vs CHEAP-LAND siting -- rank candidate buses by N-1 corridor
     exposure (analytical: coupling of a bus injection to the binding corridors
     via PTDF/LODF) vs by land cost vs random; place the top-k uniformly. Tests
     whether siting onto grid-strong buses dominates cheap-but-exposed land (the
     spread-frontier "siting dominates spreading" result).

Usage:
  .venv/bin/python development/dc_n1_overload_full.py --quick --tag smoke
  .venv/bin/python development/dc_n1_overload_full.py --tag pilot
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("pypsa").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_spread_frontier import build_panels  # noqa: E402
from dc_placement_integrated import WORKLOADS  # noqa: E402
from dc_n1_overload import (  # noqa: E402
    make_problem, build_n1_matrix, solve_overload, realized_shed, FixedPlacementEvaluator,
)


def _sites(d, tol=1e-3):
    return 0 if d is None else int((np.asarray(d) > tol).sum())


def node_n1_marginal(ev, cand, ovl_none, probe=0.25):
    """True per-candidate N-1 standalone headroom (lower = grid-strong).

    For each candidate bus, place a small DC probe there *alone*, solve the
    min-overload dispatch (load-shed held at the no-DC baseline), and record the
    added N-1 overload. This is the exact N-1 analogue of the spread-frontier's
    standalone headroom h(n)=f({n}); it correctly captures both relief and the
    creation of new overloads (which the cheap analytical exposure proxy missed).

    Uses the warm-start ``FixedPlacementEvaluator`` (one compiled LP, only the DC
    injection's equality RHS changes per probe), turning ~60 cold ~5s solves into
    one cold solve + warm re-solves.
    """
    marg = np.zeros(len(cand))
    for i in range(len(cand)):
        d = np.zeros(len(cand))
        d[i] = probe
        ovl, _st = ev.evaluate(d)
        marg[i] = (ovl - ovl_none) if np.isfinite(ovl) else np.inf
    return marg


def run_workload(panel, clean, weights, args):
    cand = [int(x) for x in clean]
    if args.cand_stride > 1:
        cand = cand[:: args.cand_stride]
    hp0 = make_problem(panel, cand)
    hp0.build(site_cap=0.0)
    M, rhs, mon_rows, _k = build_n1_matrix(hp0, args.eps_lodf)
    print(f"   {len(cand)} candidates, {len(panel)}h, {M.shape[0]} N-1 pairs", flush=True)

    args_ = (panel, cand, M, rhs, mon_rows)
    cap = args.site_cap

    # no-DC reference: the grid's own N-1 stress, and the baseline load shed it
    # requires. All strategies then hold shed at this baseline (+tol) so the metric
    # is pure N-1 overload and DC never cannibalizes load.
    _, ovl_none, _, _, _, hp_n = solve_overload(*args_, 0.0, None,
                                                d_fixed=np.zeros(len(cand)), objective="n1")
    shed_cap = realized_shed(hp_n) + args.shed_tol
    print(f"   no-DC N-1 overload={ovl_none:.3f}, baseline shed={realized_shed(hp_n):.2f} GW "
          f"-> shed_cap={shed_cap:.2f}", flush=True)

    # ONE compiled LP for every FIXED-placement evaluation (probe ranking, uniform,
    # concentrated, siting): the DC placement enters only the equality RHS, so the model
    # is built once and re-solved warm. The free-d "aware" optimization still uses
    # solve_overload (d is a Variable there; only ~5 solves/budget).
    ev = FixedPlacementEvaluator(panel, cand, M, rhs, mon_rows, shed_cap=shed_cap)

    # re-evaluate the no-DC overload under the shed cap (consistent reference)
    ovl_none, _st = ev.evaluate(np.zeros(len(cand)))

    # grid-strength ranking via TRUE per-node N-1 marginal (probe-based)
    marg = node_n1_marginal(ev, cand, ovl_none, probe=args.probe)
    strong_order = np.argsort(marg).tolist()                  # lowest N-1 marginal first
    cheap_order = np.argsort([-weights[c] for c in cand]).tolist()  # cheapest land first
    rng = np.random.default_rng(args.seed)
    rand_order = rng.permutation(len(cand)).tolist()
    strong_node = strong_order[0]
    print(f"   N-1 node marginal (added overload / {args.probe}GW probe): "
          f"best={marg[strong_node]:.3f} (node {cand[strong_node]})  worst={np.nanmax(marg):.3f}", flush=True)

    def added(ovl):
        return None if not np.isfinite(ovl) else round(ovl - ovl_none, 3)

    # ---- A. budget sweep: aware / uniform / concentrated ----
    sweep = []
    for B in args.budgets:
        d_a, ovl_a, _, _, ta, _ = solve_overload(*args_, B, cap, objective="n1", shed_cap=shed_cap)
        d_u = np.full(len(cand), B / len(cand))
        ovl_u, _ = ev.evaluate(d_u)
        d_c = np.zeros(len(cand))
        d_c[strong_node] = B
        ovl_c, _ = ev.evaluate(d_c)
        sweep.append({"B": B, "added_aware": added(ovl_a), "added_uniform": added(ovl_u),
                      "added_concentrated": added(ovl_c), "sites_aware": _sites(d_a), "secs": round(ta, 1)})
        print(f"   B={B}: added overload  aware={added(ovl_a)} ({_sites(d_a)} sites)  "
              f"uniform={added(ovl_u)}  concentrated={added(ovl_c)}", flush=True)

    # ---- B. concentrate -> distribute (per-site cap sweep at fixed B) ----
    spread = []
    for kap in args.kappas:
        d, ovl, _, _, _, _ = solve_overload(*args_, args.budget, kap, objective="n1", shed_cap=shed_cap)
        spread.append({"kappa": kap, "added": added(ovl), "sites": _sites(d)})
    print(f"   spread sweep (B={args.budget}): "
          f"{[(s['kappa'], s['added'], s['sites']) for s in spread]}", flush=True)

    # ---- C. grid-strength vs cheap-land vs random siting ----
    orders = {"grid_strength": strong_order, "cheap_land": cheap_order, "random": rand_order}
    siting = {}
    for k in args.footprints:
        siting[k] = {}
        for name, order in orders.items():
            sel = order[:k]
            d = np.zeros(len(cand))
            d[sel] = args.budget / k
            ovl, _ = ev.evaluate(d)
            siting[k][name] = added(ovl)
        print(f"   siting k={k} (B={args.budget}): "
              f"grid_strength={siting[k]['grid_strength']}  cheap_land={siting[k]['cheap_land']}  "
              f"random={siting[k]['random']}", flush=True)

    return {
        "n_candidates": len(cand), "n_pairs": int(M.shape[0]),
        "overload_no_dc": ovl_none, "site_cap": cap,
        "budget_sweep": sweep, "spread_sweep": spread, "siting_orderings": siting,
        "node_n1_marginal": {int(cand[i]): float(marg[i]) for i in range(len(cand))},
        "candidates": cand,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    # Single representative stressed hour: keeps the LP small enough that HiGHS is
    # reliable (it returns false-infeasible / suboptimal on the ~85k-var 3-hour LPs),
    # and matches how the base-case Q1/Q2 studies pick one congested hour.
    ap.add_argument("--n-snaps", type=int, default=1)
    ap.add_argument("--load-scale", type=float, default=1.2)
    ap.add_argument("--budget", type=float, default=3.0, help="fixed B for spread + siting sweeps")
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0])
    ap.add_argument("--kappas", type=float, nargs="+", default=[3.0, 1.0, 0.5, 0.25, 0.1])
    ap.add_argument("--footprints", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--site-cap", type=float, default=0.5)
    ap.add_argument("--probe", type=float, default=0.25, help="probe GW for per-node N-1 marginal")
    ap.add_argument("--shed-tol", type=float, default=0.5,
                    help="GW of load shed allowed above the no-DC baseline")
    ap.add_argument("--eps-lodf", type=float, default=0.05)
    ap.add_argument("--cand-stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--land-cost", default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--outdir", default="development/results/n1_overload")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_snaps = 1
        args.cand_stride = 24
        args.budgets = [1.0, 3.0, 5.0]
        args.kappas = [3.0, 0.5, 0.1]
        args.footprints = [5, 20]
        args.eps_lodf = 0.06
    os.makedirs(args.outdir, exist_ok=True)

    print(f"## N-1 OVERLOAD (full) tag={args.tag} load_scale={args.load_scale} "
          f"budgets={args.budgets} cap={args.site_cap}GW eps={args.eps_lodf}")
    pn, n_nodes, bad, clean, weights, panels = build_panels(args)
    print(f"   network: {n_nodes} nodes, {len(bad)} pathological, {len(clean)} clean")
    out = {"load_scale": args.load_scale, "n_snaps": args.n_snaps, "site_cap": args.site_cap,
           "budgets": args.budgets, "kappas": args.kappas, "footprints": args.footprints,
           "eps_lodf": args.eps_lodf, "workloads": {}}
    for w in args.workloads:
        print(f"\n-- workload={w}", flush=True)
        t0 = time.time()
        out["workloads"][w] = run_workload(panels[w], clean, weights, args)
        print(f"   ({time.time()-t0:.0f}s)", flush=True)
    path = os.path.join(args.outdir, f"n1_overload_full_{args.tag}.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
