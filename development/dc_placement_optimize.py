"""
GRADIENT-BASED DC PLACEMENT on the 490-node WECC network.

Runs the actual zap bi-level planner:
    minimize_{dc_capacity}  LineUtilizationObjective(sum u^2)  +  InvestmentObjective
    s.t.  0 <= dc_capacity_i <= per_site_cap   (realistic: a site is small)
          sum_i dc_capacity_i == budget        (operator's total fleet)
via projected gradient descent (BoxBudgetProjection).

Realistic fine-grained placement: many candidate sites, small per-site caps
(e.g. 250 MW), so no single node can absorb the whole budget. Compares the
OPTIMIZED allocation against uniform / concentrated baselines on robust metrics.

Usage:
  .venv/bin/python development/dc_placement_optimize.py \
      --budget 2.0 --site-cap 0.25 --iters 60 --step 2e-3
"""
import argparse, os, time, json, contextlib, io
from copy import deepcopy
import cvxpy as cp
import numpy as np
import zap
from zap.importers.pypsa import load_pypsa_network

POWER_UNIT, COST_UNIT, VOLL = 1.0e3, 100.0, 1000.0
DEFAULT_SITES = [5, 20, 40, 60, 70, 80, 100, 110, 120, 140, 160, 180, 200, 210, 220,
                 240, 250, 260, 280, 300, 320, 330, 350, 370, 400, 420, 440, 460, 480, 150, 50]


def load_net(path, s0, slen, ls, gs, lns):
    import pypsa
    pn = pypsa.Network(os.path.expanduser(path))
    snaps = pn.generators_t.p_max_pu.index
    net, devices = load_pypsa_network(pn, snaps[s0:s0 + slen], power_unit=POWER_UNIT, cost_unit=COST_UNIT)
    devices = deepcopy(devices)
    devices[1].load *= ls; devices[0].dynamic_capacity *= gs; devices[3].nominal_capacity *= lns
    return pn, net, devices, str(snaps[s0])


def pick_snapshot(path, w0, wlen, ls, gs, lns):
    import pypsa
    pn = pypsa.Network(os.path.expanduser(path))
    snaps = pn.generators_t.p_max_pu.index
    best, bn = w0, -1
    for s in range(w0, w0 + wlen):
        net, devices = load_pypsa_network(pn, snaps[s:s+1], power_unit=POWER_UNIT, cost_unit=COST_UNIT)
        devices = deepcopy(devices)
        devices[1].load *= ls; devices[0].dynamic_capacity *= gs; devices[3].nominal_capacity *= lns
        try:
            oc = net.dispatch(devices, time_horizon=1, solver=cp.CLARABEL, add_ground=False)
        except AssertionError:
            continue
        mu = np.asarray(oc.local_inequality_duals[3][0]) + np.asarray(oc.local_inequality_duals[3][1])
        n = int(np.sum(mu.max(axis=1) > 1e-4))
        if n > bn:
            bn, best = n, s
    return best, bn


def make_dc(net, sites, caps, T):
    return zap.DataCenterLoad(
        num_nodes=net.num_nodes, terminal=np.array(sites), profiles=[np.ones(T)] * len(sites),
        nominal_capacity=np.array(caps, dtype=float), linear_cost=np.zeros(len(sites)),
        settime_horizon=T, capital_cost=np.zeros(len(sites)))


def metrics(net, devices, dc, T, dc_terminals):
    devs = devices + [dc]
    try:
        oc = net.dispatch(devs, time_horizon=T, solver=cp.CLARABEL, add_ground=False)
    except AssertionError:
        return {"feasible": False, "sum_u2": np.inf, "p95_lmp": np.inf, "dc_lmp": np.inf,
                "median_lmp": np.inf, "n_binding": -1, "shed_pct": np.inf}
    p = np.clip(np.asarray(oc.prices) * COST_UNIT, 0, VOLL)
    A = devices[3]
    flow = np.abs(np.asarray(oc.power[3][1]))
    limit = np.maximum(np.asarray(A.max_power) * np.asarray(A.nominal_capacity), 1e-9)
    u = flow / limit
    mu = np.asarray(oc.local_inequality_duals[3][0]) + np.asarray(oc.local_inequality_duals[3][1])
    L = devices[1]
    req = (np.asarray(L.load) * np.asarray(L.nominal_capacity)).sum()
    served = (-np.asarray(oc.power[1][0])).sum()
    return {"feasible": True, "sum_u2": float((u**2).sum()), "p95_lmp": float(np.percentile(p, 95)),
            "dc_lmp": float(np.mean(p[np.array(dc_terminals)])), "median_lmp": float(np.median(p)),
            "n_binding": int(np.sum(mu.max(axis=1) > 1e-4)),
            "shed_pct": float(100 * max(req - served, 0.0) / req)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--snap-start", type=int, default=5616)
    ap.add_argument("--snap-win", type=int, default=18)
    ap.add_argument("--snap-len", type=int, default=2)
    ap.add_argument("--load-scale", type=float, default=1.0)
    ap.add_argument("--gen-scale", type=float, default=1.24)
    ap.add_argument("--line-scale", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=2.0, help="total DC budget (GW)")
    ap.add_argument("--site-cap", type=float, default=0.25, help="per-site cap (GW); realistic ~0.1-0.5")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--step", type=float, default=2e-3)
    ap.add_argument("--sites", default="")
    ap.add_argument("--outdir", default="development/results/placement_optimize")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sites_all = [int(x) for x in args.sites.split(",")] if args.sites else DEFAULT_SITES
    snap, nb = pick_snapshot(args.network, args.snap_start, args.snap_win,
                             args.load_scale, args.gen_scale, args.line_scale)
    pn, net, devices, date0 = load_net(args.network, snap, args.snap_len,
                                       args.load_scale, args.gen_scale, args.line_scale)
    T = args.snap_len
    sites = [s for s in sites_all if s < net.num_nodes]
    n = len(sites)
    B, cap = args.budget, args.site_cap
    assert n * cap >= B, f"infeasible: {n} sites x {cap} GW < budget {B} GW"
    print(f"# 490-node GRADIENT PLACEMENT  snap={snap}({nb} binding) date={date0}")
    print(f"# {n} candidate sites | budget={B}GW | per-site cap={cap}GW (={cap*1000:.0f}MW) "
          f"| min #sites used >= {int(np.ceil(B/cap))}")

    dc_idx = len(devices)
    layer = zap.DispatchLayer(net, devices + [make_dc(net, sites, np.full(n, B/n), T)],
                              parameter_names={"dc_capacity": (dc_idx, "nominal_capacity")},
                              time_horizon=T, solver=cp.CLARABEL)
    op_obj = zap.planning.LineUtilizationObjective(net, layer.devices, metric="quadratic")
    inv_obj = zap.planning.InvestmentObjective(layer.devices, layer)
    P = zap.planning.PlanningProblem(
        operation_objective=op_obj, investment_objective=inv_obj, layer=layer,
        lower_bounds={"dc_capacity": np.zeros(n)}, upper_bounds={"dc_capacity": np.full(n, cap)})
    P.extra_projections = {"dc_capacity": zap.planning.BoxBudgetProjection(
        budget=B, lower_bounds=np.zeros(n), upper_bounds=np.full(n, cap))}

    # feasible uniform start (cap-respecting), then solve
    init = P.extra_projections["dc_capacity"](np.full(n, B / n))
    from zap.planning.solvers import GradientDescent
    algo = GradientDescent(step_size=args.step, clip=1e3)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        state, hist = P.solve(algorithm=algo, num_iterations=args.iters,
                              initial_state={"dc_capacity": init}, verbosity=0)
    opt = np.asarray(state["dc_capacity"]).ravel()
    opt = P.extra_projections["dc_capacity"](opt)  # clean tiny violations
    print(f"# solved {args.iters} iters in {time.time()-t0:.1f}s  "
          f"loss {hist['loss'][0]:.3f} -> {hist['loss'][-1]:.3f}")

    # ---- baselines ----
    base_devs = layer.devices[:-1]   # devices without DC
    dc_term = sites
    def ev(caps):
        return metrics(net, base_devs, make_dc(net, sites, caps, T), T, dc_term)

    uniform = np.full(n, B / n)
    # concentrate at best/worst single feasible node: probe each at min(cap, B)... but cap<B,
    # so 'concentrate' here means as few sites as possible at full cap.
    k_min = int(np.ceil(B / cap))
    # rank sites by individual marginal stress (Δsum_u2 for a small block)
    base0 = ev(np.zeros(n))["sum_u2"]
    marg = []
    for i in range(n):
        c = np.zeros(n); c[i] = min(cap, 0.1)
        marg.append((ev(c)["sum_u2"] - base0) / min(cap, 0.1))
    order = np.argsort(marg)
    def pack(idx_order):
        caps = np.zeros(n); rem = B
        for i in idx_order:
            take = min(cap, rem); caps[i] = take; rem -= take
            if rem <= 1e-9: break
        return caps
    conc_best = pack(order)            # fill cheapest sites first (still concentrated to k_min sites)
    conc_worst = pack(order[::-1])     # fill most-congested sites first

    results = {"optimized": ev(opt), "uniform": ev(uniform),
               "concentrate@cheap": ev(conc_best), "concentrate@costly": ev(conc_worst)}
    print(f"\n{'strategy':>18} {'#sites':>7} {'maxcap':>7} {'sum_u2':>8} {'p95_lmp':>8} "
          f"{'dc_lmp':>7} {'n_bind':>7} {'shed%':>7}")
    allocs = {"optimized": opt, "uniform": uniform, "concentrate@cheap": conc_best,
              "concentrate@costly": conc_worst}
    for name, m in results.items():
        a = allocs[name]
        ns = int((a > 1e-4).sum())
        if m["feasible"]:
            print(f"{name:>18} {ns:7d} {a.max():7.3f} {m['sum_u2']:8.1f} {m['p95_lmp']:8.1f} "
                  f"{m['dc_lmp']:7.1f} {m['n_binding']:7d} {m['shed_pct']:7.3f}")
        else:
            print(f"{name:>18} {ns:7d} {a.max():7.3f}   INFEASIBLE")

    # headline
    o, u, cc = results["optimized"], results["uniform"], results["concentrate@costly"]
    print("\n## HEADLINE")
    print(f"  optimized uses {int((opt>1e-4).sum())} sites (max {opt.max()*1000:.0f}MW), "
          f"sum_u2={o['sum_u2']:.1f}, p95_LMP=${o['p95_lmp']:.0f}, dc_LMP=${o['dc_lmp']:.0f}")
    print(f"  uniform   : sum_u2={u['sum_u2']:.1f}, p95_LMP=${u['p95_lmp']:.0f}, dc_LMP=${u['dc_lmp']:.0f}")
    if cc['feasible']:
        print(f"  worst-pack: sum_u2={cc['sum_u2']:.1f}, p95_LMP=${cc['p95_lmp']:.0f}")
    if o['sum_u2'] < u['sum_u2']:
        print(f"  -> optimizer beats uniform by {100*(1-o['sum_u2']/u['sum_u2']):.1f}% on sum_u2, "
              f"dc-LMP ${u['dc_lmp']:.0f}->${o['dc_lmp']:.0f}")

    out = {"snapshot": snap, "date": date0, "budget": B, "site_cap": cap, "sites": sites,
           "loss_traj": [float(x) for x in hist["loss"]], "alloc_optimized": opt.tolist(),
           "alloc_uniform": uniform.tolist(), "results": results,
           "marginal_stress": [float(x) for x in marg]}
    with open(os.path.join(args.outdir, "optimize_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    _plot(args.outdir, sites, opt, np.array(marg), hist["loss"], results, B, cap)
    print(f"saved -> {args.outdir}/optimize_results.json")


def _plot(outdir, sites, opt, marg, loss, results, B, cap):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plots skipped: {e})"); return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    order = np.argsort(marg)
    ax[0].bar(range(len(sites)), opt[order] * 1000, color="seagreen")
    ax[0].axhline(cap * 1000, ls="--", c="r", alpha=0.5, label=f"site cap {cap*1000:.0f}MW")
    ax[0].set_xlabel("candidate site (sorted by marginal stress, cheap->costly)")
    ax[0].set_ylabel("optimized DC capacity (MW)")
    ax[0].set_title(f"Optimized fine-grained allocation (B={B}GW)"); ax[0].legend(fontsize=8)
    ax[1].plot(loss, marker=".")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("operation objective (sum u^2)")
    ax[1].set_title("Projected-gradient convergence")
    names = [k for k in results if results[k]["feasible"]]
    ax[2].bar(range(len(names)), [results[k]["sum_u2"] for k in names], color="steelblue")
    ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax[2].set_ylabel("congestion stress sum u^2"); ax[2].set_title("Optimized vs baselines")
    fig.tight_layout(); p = os.path.join(outdir, "placement_optimize.png"); fig.savefig(p, dpi=120)
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
