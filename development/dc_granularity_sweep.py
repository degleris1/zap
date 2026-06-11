"""
GRANULARITY SWEEP: how does the optimizer's congestion saving change as we make
data-center siting finer (smaller per-site caps => more, smaller sites) and as we
sample the grid more densely (more candidate nodes)?

Fixed total budget B; sweep per-site cap in {500,250,100,50} MW over a dense
candidate set. For each, run the gradient planner (LineUtilizationObjective +
BoxBudgetProjection) and compare optimized vs uniform.
"""
import argparse, os, time, json, contextlib, io
from copy import deepcopy
import cvxpy as cp
import numpy as np
import zap
from zap.importers.pypsa import load_pypsa_network
from zap.planning.solvers import GradientDescent

POWER_UNIT, COST_UNIT, VOLL = 1.0e3, 100.0, 1000.0


def load_net(path, s0, slen, ls, gs, lns):
    import pypsa
    pn = pypsa.Network(os.path.expanduser(path))
    snaps = pn.generators_t.p_max_pu.index
    net, devices = load_pypsa_network(pn, snaps[s0:s0+slen], power_unit=POWER_UNIT, cost_unit=COST_UNIT)
    devices = deepcopy(devices)
    devices[1].load *= ls; devices[0].dynamic_capacity *= gs; devices[3].nominal_capacity *= lns
    return net, devices


def pick_snapshot(path, w0, wlen, ls, gs, lns):
    import pypsa
    pn = pypsa.Network(os.path.expanduser(path))
    snaps = pn.generators_t.p_max_pu.index
    best, bn = w0, -1
    for s in range(w0, w0+wlen):
        net, devices = load_pypsa_network(pn, snaps[s:s+1], power_unit=POWER_UNIT, cost_unit=COST_UNIT)
        devices = deepcopy(devices)
        devices[1].load *= ls; devices[0].dynamic_capacity *= gs; devices[3].nominal_capacity *= lns
        try:
            oc = net.dispatch(devices, time_horizon=1, solver=cp.CLARABEL, add_ground=False)
        except AssertionError:
            continue
        mu = np.asarray(oc.local_inequality_duals[3][0]) + np.asarray(oc.local_inequality_duals[3][1])
        n = int(np.sum(mu.max(axis=1) > 1e-4))
        if n > bn: bn, best = n, s
    return best, bn


def make_dc(net, sites, caps, T):
    return zap.DataCenterLoad(num_nodes=net.num_nodes, terminal=np.array(sites),
        profiles=[np.ones(T)]*len(sites), nominal_capacity=np.array(caps, float),
        linear_cost=np.zeros(len(sites)), settime_horizon=T, capital_cost=np.zeros(len(sites)))


def stress(net, devices, dc, T):
    oc = net.dispatch(devices + [dc], time_horizon=T, solver=cp.CLARABEL, add_ground=False)
    A = devices[3]
    flow = np.abs(np.asarray(oc.power[3][1]))
    limit = np.maximum(np.asarray(A.max_power) * np.asarray(A.nominal_capacity), 1e-9)
    u = flow / limit
    p = np.clip(np.asarray(oc.prices) * COST_UNIT, 0, VOLL)
    return float((u**2).sum()), float(np.percentile(p, 95))


def optimize(net, devices, sites, B, cap, T, iters, step):
    n = len(sites)
    dc_idx = len(devices)
    layer = zap.DispatchLayer(net, devices + [make_dc(net, sites, np.full(n, B/n), T)],
        parameter_names={"dc_capacity": (dc_idx, "nominal_capacity")}, time_horizon=T, solver=cp.CLARABEL)
    P = zap.planning.PlanningProblem(
        operation_objective=zap.planning.LineUtilizationObjective(net, layer.devices, metric="quadratic"),
        investment_objective=zap.planning.InvestmentObjective(layer.devices, layer), layer=layer,
        lower_bounds={"dc_capacity": np.zeros(n)}, upper_bounds={"dc_capacity": np.full(n, cap)})
    P.extra_projections = {"dc_capacity": zap.planning.BoxBudgetProjection(
        budget=B, lower_bounds=np.zeros(n), upper_bounds=np.full(n, cap))}
    init = P.extra_projections["dc_capacity"](np.full(n, B/n))
    with contextlib.redirect_stdout(io.StringIO()):
        state, _ = P.solve(algorithm=GradientDescent(step_size=step, clip=1e3),
                           num_iterations=iters, initial_state={"dc_capacity": init}, verbosity=0)
    opt = P.extra_projections["dc_capacity"](np.asarray(state["dc_capacity"]).ravel())
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--snap-start", type=int, default=5616)
    ap.add_argument("--snap-win", type=int, default=18)
    ap.add_argument("--snap-len", type=int, default=2)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--caps", default="0.5,0.25,0.1,0.05")
    ap.add_argument("--stride", type=int, default=6, help="candidate every Nth node (grid density)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--step", type=float, default=2e-3)
    ap.add_argument("--outdir", default="development/results/granularity_sweep")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    snap, nb = pick_snapshot(args.network, args.snap_start, args.snap_win, 1.0, 1.24, 1.0)
    net, devices = load_net(args.network, snap, args.snap_len, 1.0, 1.24, 1.0)
    T = args.snap_len
    caps = [float(x) for x in args.caps.split(",")]
    B = args.budget
    # Candidate set: every Nth node, but drop transmission-locked pockets that are
    # individually INFEASIBLE at the largest cap (they'd crash the inner dispatch).
    raw = [s for s in range(0, net.num_nodes, args.stride)]
    cmax = max(caps)
    sites = []
    for s in raw:
        try:
            net.dispatch(devices + [make_dc(net, [s], [cmax], T)], time_horizon=T,
                         solver=cp.CLARABEL, add_ground=False)
            sites.append(s)
        except AssertionError:
            pass
    if len(sites) * min(caps) < B:
        print(f"# warning: only {len(sites)} feasible sites; tighten stride or caps");
    print(f"# screened {len(raw)} -> {len(sites)} feasible candidate sites (dropped chokepoint pockets)")
    print(f"# GRANULARITY SWEEP  490-node snap={snap}({nb} binding)  budget={B}GW  "
          f"candidates={len(sites)} (stride {args.stride})")
    base_u2, base_p95 = stress(net, devices, make_dc(net, sites, np.full(len(sites), 1e-6), T), T)
    print(f"# base (no DC): sum_u2={base_u2:.1f} p95_lmp=${base_p95:.0f}")
    print(f"\n{'cap(MW)':>8} {'min#sites':>9} {'opt_u2':>8} {'uni_u2':>8} {'opt_gain%':>9} "
          f"{'opt_p95':>8} {'uni_p95':>8} {'opt#sites':>9} {'time':>6}")
    rows = []
    for cap in caps:
        if len(sites) * cap < B:
            print(f"{cap*1000:8.0f}  -- infeasible: {len(sites)} sites x {cap}GW < {B}GW"); continue
        t0 = time.time()
        opt = optimize(net, devices, sites, B, cap, T, args.iters, args.step)
        uni = np.full(len(sites), B/len(sites))
        # uniform must respect cap; if B/n > cap it's infeasible uniform -> clip+renorm not needed since small
        ou2, op95 = stress(net, devices, make_dc(net, sites, opt, T), T)
        uu2, up95 = stress(net, devices, make_dc(net, sites, uni, T), T)
        dt = time.time() - t0
        nsites = int((opt > 1e-4).sum())
        gain = 100*(1 - ou2/uu2)
        rows.append({"cap_gw": cap, "opt_u2": ou2, "uni_u2": uu2, "gain_pct": gain,
                     "opt_p95": op95, "uni_p95": up95, "opt_sites": nsites,
                     "min_sites": int(np.ceil(B/cap))})
        print(f"{cap*1000:8.0f} {int(np.ceil(B/cap)):9d} {ou2:8.1f} {uu2:8.1f} {gain:9.1f} "
              f"{op95:8.1f} {up95:8.1f} {nsites:9d} {dt:6.1f}s")

    out = {"snapshot": snap, "budget": B, "candidates": len(sites), "stride": args.stride,
           "base_u2": base_u2, "rows": rows}
    json.dump(out, open(os.path.join(args.outdir, "granularity.json"), "w"), indent=2, default=float)
    _plot(args.outdir, rows, base_u2, B, len(sites))
    print(f"saved -> {args.outdir}/granularity.json")


def _plot(outdir, rows, base_u2, B, ncand):
    if not rows: return
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception: return
    caps = [r["cap_gw"]*1000 for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(caps, [r["opt_u2"] for r in rows], "o-", label="optimized")
    ax[0].plot(caps, [r["uni_u2"] for r in rows], "s--", label="uniform")
    ax[0].axhline(base_u2, ls=":", c="gray", label="no DC")
    ax[0].set_xlabel("per-site cap (MW)  [finer ->]"); ax[0].invert_xaxis()
    ax[0].set_ylabel("congestion stress sum u^2"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Finer siting -> lower stress (B={B}GW, {ncand} sites)")
    ax[1].plot(caps, [r["gain_pct"] for r in rows], "o-", color="seagreen")
    ax[1].set_xlabel("per-site cap (MW)  [finer ->]"); ax[1].invert_xaxis()
    ax[1].set_ylabel("optimizer gain vs uniform (%)")
    ax[1].set_title("Smart placement value vs granularity")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "granularity.png"), dpi=120)
    print(f"saved -> {outdir}/granularity.png")


if __name__ == "__main__":
    main()
