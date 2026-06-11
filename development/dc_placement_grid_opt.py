"""
OPTIMIZE DC PLACEMENT FOR THE GRID (490-node WECC), realistically.

We DO optimize where the data centers go -- to minimize grid stress -- but with
realistic constraints so the optimum is not the degenerate "pile everything on
one node" answer:

  minimize_{site capacities}  grid stress (sum of line-utilization^2)
  subject to   0 <= capacity_i <= 1 GW   (no single site bigger than a real campus)
               sum_i capacity_i = fleet   (total data-center demand to place)
               every grid location is a candidate (no hand-picked shortlist)

The per-site 1 GW cap forces the optimizer to SPREAD the fleet over many realistic
sites instead of clustering -- so the answer is realistic, not degenerate.

Method: zap's differentiable bi-level planner (LineUtilizationObjective + projected
gradient with BoxBudgetProjection). We optimize on the most-congested hour of the
year (plan for the binding condition), then EVALUATE the resulting layout across a
12-hour panel spread over the whole year, with raw (unclipped) prices and a real,
time-varying data-center workload. We compare the grid-optimized layout against:
  * realistic siting  -- data centers drawn at random onto cheap land (what actually
    happens when nobody coordinates with the grid)
  * uniform           -- the fleet spread evenly over every usable location

Usage:
  .venv/bin/python development/dc_placement_grid_opt.py --budgets 3,4,5,6
"""
import argparse, os, json, time

import numpy as np
import cvxpy as cp
import zap

# reuse the panel / metrics / realistic-siting machinery from the descriptive study
from dc_placement_study import (
    POWER_UNIT, COST_UNIT, VOLL, DEFAULT_DC_PROFILE,
    load_dc_profile, build_hour, sample_panel_indices, panel_eval, metrics, with_dc,
    summarize_all, load_land_weights, screen_usable, realistic_fleets, across_fleets,
)


def _panel_cvar(panel, cand, alloc, alpha):
    """CVaR_alpha of grid stress across the panel hours for a candidate allocation:
    the AVERAGE of the worst ceil(alpha*H) hours' grid stress. Returns +inf if the
    allocation is infeasible in ANY hour (so the optimizer only accepts layouts that
    work every hour). This hedges the bad-hour tail without overfitting to the single
    worst hour."""
    terms = [cand[i] for i in range(len(cand)) if alloc[i] > 1e-9]
    caps = [float(alloc[i]) for i in range(len(cand)) if alloc[i] > 1e-9]
    stresses = []
    for (net, devs, _, lf) in panel:
        m = metrics(net, devs, devs, 1) if not terms else \
            metrics(net, with_dc(devs, terms, caps, lf), devs, 1, terms)
        if not m["feasible"]:
            return np.inf               # must work in every hour
        stresses.append(m["sum_u2"])
    stresses.sort(reverse=True)
    k = max(1, int(np.ceil(alpha * len(stresses))))
    return float(np.mean(stresses[:k]))


def greedy_optimize(panel, cand, budgets, cap, alpha, chunk=0.5):
    """Robust grid placement by greedy water-filling on PANEL-WIDE CVaR of grid stress:
    repeatedly add a `chunk` of demand at the candidate location that most reduces the
    worst-hours' grid stress across the year, never exceeding the per-site cap, and only
    accepting placements feasible in every hour. Plain (non-differentiable) dispatches
    -> no solver fragility. Records the allocation at each requested fleet size."""
    n = len(cand)
    alloc = np.zeros(n)
    placed, Bmax = 0.0, max(budgets)
    recorded, targets = {}, sorted(budgets)
    t0 = time.time()
    while placed < Bmax - 1e-9:
        best, best_s = None, np.inf
        for i in range(n):
            if alloc[i] >= cap - 1e-9:
                continue
            trial = alloc.copy()
            trial[i] = min(cap, trial[i] + chunk)
            s = _panel_cvar(panel, cand, trial, alpha)
            if s < best_s:
                best_s, best = s, i
        if best is None or not np.isfinite(best_s):
            break                       # cannot place any more demand feasibly anywhere
        alloc[best] = min(cap, alloc[best] + chunk)
        placed += chunk
        for B in targets:
            if B not in recorded and placed >= B - 1e-9:
                recorded[B] = alloc.copy()
    return recorded, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-scale", type=float, default=1.0)
    ap.add_argument("--gen-scale", type=float, default=1.0)
    ap.add_argument("--line-scale", type=float, default=1.0)
    ap.add_argument("--budgets", default="3,4,5,6", help="fleet sizes to place (GW)")
    ap.add_argument("--per-site-cap", type=float, default=1.0, help="max GW at one location")
    ap.add_argument("--small-site", type=float, default=0.2, help="site size for realistic siting")
    ap.add_argument("--n-fleets", type=int, default=24, help="random realistic fleets per budget")
    ap.add_argument("--max-cand", type=int, default=100,
                    help="candidate locations for the greedy optimizer (spread across the grid)")
    ap.add_argument("--chunk", type=float, default=0.5, help="greedy demand increment (GW)")
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="CVaR tail fraction: optimize the average of the worst alpha share of hours")
    ap.add_argument("--dc-profile", default=DEFAULT_DC_PROFILE)
    ap.add_argument("--land-cost", default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--usable-cache", default="development/results/placement_grid_opt/usable_nodes.json")
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--outdir", default="development/results/placement_grid_opt")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    n_total = len(snaps)
    cap, small = args.per_site_cap, args.small_site
    budgets = [float(x) for x in args.budgets.split(",")]

    print(f"# OPTIMIZE DC PLACEMENT FOR THE GRID  network={os.path.basename(args.network)}")
    print(f"# per-site cap={cap}GW  budgets={budgets}GW  | minimize grid stress, all locations candidates")

    hourly_lf = load_dc_profile(args.dc_profile)
    print(f"# DC workload = {os.path.basename(args.dc_profile)} (real, time-varying; load-factor "
          f"{hourly_lf.min():.2f}-{hourly_lf.max():.2f})")

    idx = sample_panel_indices(n_total, args.n_snaps, None, None, args.seed)
    panel = [build_hour(pn, snaps, h, args.load_scale, args.gen_scale, args.line_scale, hourly_lf)
             for h in idx]
    n_nodes = panel[0][0].num_nodes
    nodes = list(range(n_nodes if not args.max_nodes else args.max_nodes))
    print(f"# panel: {len(panel)} hours across the year")

    base_ms = [metrics(net, devs, devs, 1) for (net, devs, _, _) in panel]
    base = summarize_all(base_ms)
    base_u2 = [m["sum_u2"] if m["feasible"] else -1 for m in base_ms]
    plan_h = int(np.argmax(base_u2))      # most-congested panel hour = planning condition
    print(f"# base (no DC): stress {base['sum_u2']['median']:.0f}, p95 ${base['p95_lmp']['median']:.0f}, "
          f"shed {base['shed_pct']['median']:.2f}%  | planning on hour {plan_h} (most congested)")

    # ---- usable locations (drop dead-ends), cached ----
    usable = None
    if os.path.exists(args.usable_cache):
        c = json.load(open(args.usable_cache))
        if c.get("key") == [args.n_snaps, args.seed, args.load_scale, args.gen_scale, args.line_scale, small, len(nodes)]:
            usable = c["usable"]
            print(f"# usable locations (cached): {len(usable)}/{len(nodes)}")
    if usable is None:
        print(f"# screening {len(nodes)} locations for dead-ends (a {small}GW site must work every hour)...")
        usable = screen_usable(panel, nodes, probe=small)
        json.dump({"key": [args.n_snaps, args.seed, args.load_scale, args.gen_scale,
                           args.line_scale, small, len(nodes)], "usable": usable},
                  open(args.usable_cache, "w"))
        print(f"# usable locations: {len(usable)}/{len(nodes)} (dropped {len(nodes)-len(usable)} dead-ends)")

    land, weights = load_land_weights(args.land_cost, n_nodes)
    rng = np.random.default_rng(args.seed)
    net_p, devs_p, _, lf_p = panel[plan_h]
    # candidate pool for the optimizer: usable locations spread across the grid
    # (subsampled for tractable greedy; not a hand-picked shortlist).
    if len(usable) > args.max_cand:
        cand = [usable[i] for i in np.linspace(0, len(usable) - 1, args.max_cand).astype(int)]
        cand = sorted(set(cand))
    else:
        cand = list(usable)
    print(f"# greedy optimizer over {len(cand)} candidate locations (cap {cap}GW each), "
          f"minimizing CVaR(alpha={args.alpha}) of stress across ALL {len(panel)} hours...")
    recorded, opt_s = greedy_optimize(panel, cand, budgets, cap, args.alpha, chunk=args.chunk)
    print(f"# greedy placement done in {opt_s:.0f}s")

    # ---- for each fleet size: compare grid-optimized vs realistic vs uniform on the panel ----
    print(f"\n## OPTIMIZED-FOR-GRID vs REALISTIC vs UNIFORM  [median across {len(panel)} hours; "
          f"realistic = median over {args.n_fleets} random cheap-land fleets]")
    print(f"  {'fleet':>5} | {'OPT:#sites':>10} {'works':>5} {'stress':>7} {'p95$':>6} {'shed%':>6}"
          f" | {'REAL: stress':>12} {'p95$':>6} {'shed%':>6}"
          f" | {'UNIF: stress':>12} {'shed%':>6} | {'opt vs real':>11}")
    rows = []
    for B in budgets:
        alloc = recorded.get(B)
        if alloc is None:                 # greedy couldn't place this much at the peak hour
            opt_sites, opt_caps = [], []
            m_opt = {"feas_frac": 0.0, **{k: {"median": np.inf, "p10": np.inf, "p90": np.inf} for k in
                     ["median_lmp", "p95_lmp", "dc_lmp", "n_binding", "max_util", "sum_u2", "shed_pct"]}}
        else:
            opt_sites = [cand[i] for i in range(len(cand)) if alloc[i] > 1e-9]
            opt_caps = [float(alloc[i]) for i in range(len(cand)) if alloc[i] > 1e-9]
            m_opt = summarize_all(panel_eval(panel, opt_sites, opt_caps))
        # realistic cheap-land siting (small sites)
        real = realistic_fleets(panel, rng, usable, weights, B, small, args.n_fleets)
        r_st, r_p95, r_sh = across_fleets(real, "sum_u2"), across_fleets(real, "p95_lmp"), across_fleets(real, "shed_pct")
        # uniform across usable
        m_uni = summarize_all(panel_eval(panel, usable, [B / len(usable)] * len(usable)))
        gain = 100 * (1 - m_opt["sum_u2"]["median"] / r_st["median"]) if np.isfinite(r_st["median"]) and r_st["median"] > 0 and np.isfinite(m_opt["sum_u2"]["median"]) else np.nan
        rows.append({"B": B, "opt": m_opt, "n_opt_sites": len(opt_sites),
                     "opt_alloc": {int(s): float(c) for s, c in zip(opt_sites, opt_caps)},
                     "real_stress": r_st, "real_p95": r_p95, "real_shed": r_sh,
                     "uniform": m_uni, "opt_vs_real_pct": gain})
        print(f"  {B:5.1f} | {len(opt_sites):10d} {m_opt['feas_frac']:5.2f} {m_opt['sum_u2']['median']:7.0f} "
              f"{m_opt['p95_lmp']['median']:6.0f} {m_opt['shed_pct']['median']:6.2f}"
              f" | {r_st['median']:12.0f} {r_p95['median']:6.0f} {r_sh['median']:6.2f}"
              f" | {m_uni['sum_u2']['median']:12.0f} {m_uni['shed_pct']['median']:6.2f}"
              f" | {gain:10.1f}%")

    # ---- headline ----
    print(f"\n## HEADLINE  (real DC workload, raw prices, {len(panel)}-hour panel)")
    for r in (rows[0], rows[-1]):
        print(f"  {r['B']:.0f} GW fleet: grid-optimized uses {r['n_opt_sites']} sites, "
              f"stress {r['opt']['sum_u2']['median']:.0f} / shed {r['opt']['shed_pct']['median']:.2f}%  "
              f"vs realistic siting stress {r['real_stress']['median']:.0f} / shed {r['real_shed']['median']:.2f}%  "
              f"-> optimizing cuts stress {r['opt_vs_real_pct']:.0f}%")

    out = {"network": os.path.basename(args.network), "panel_idx": idx, "plan_hour": plan_h,
           "per_site_cap": cap, "small_site": small, "budgets": budgets, "n_usable": len(usable),
           "base": base, "rows": rows}
    json.dump(out, open(os.path.join(args.outdir, "grid_opt_results.json"), "w"), indent=2, default=float)
    print(f"\nsaved -> {os.path.join(args.outdir, 'grid_opt_results.json')}")
    _plot(args.outdir, base, rows, cap)


def _plot(outdir, base, rows, cap):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plots skipped: {e})"); return
    Bs = [r["B"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].axhline(base["sum_u2"]["median"], ls="--", c="gray", label="no DC (base)")
    ax[0].plot(Bs, [r["opt"]["sum_u2"]["median"] for r in rows], marker="o", label="grid-optimized")
    ax[0].plot(Bs, [r["real_stress"]["median"] for r in rows], marker="s", label="realistic (cheap land)")
    ax[0].fill_between(Bs, [r["real_stress"]["p10"] for r in rows],
                       [r["real_stress"]["p90"] for r in rows], alpha=0.15)
    ax[0].plot(Bs, [r["uniform"]["sum_u2"]["median"] for r in rows], marker="^", ls=":", label="uniform")
    ax[0].set_xlabel("data-center fleet placed (GW)"); ax[0].set_ylabel("grid stress (median)")
    ax[0].set_title("Optimizing for the grid vs realistic siting"); ax[0].legend(fontsize=8)
    ax[1].plot(Bs, [r["opt"]["shed_pct"]["median"] for r in rows], marker="o", label="grid-optimized")
    ax[1].plot(Bs, [r["real_shed"]["median"] for r in rows], marker="s", label="realistic (cheap land)")
    ax[1].set_xlabel("data-center fleet placed (GW)"); ax[1].set_ylabel("load cut (% of demand)")
    ax[1].set_title("Load shed: optimized vs realistic"); ax[1].legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(outdir, "grid_opt.png"); fig.savefig(p, dpi=120)
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
