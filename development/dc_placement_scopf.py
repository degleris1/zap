"""
N-1 / SCOPF-aware gradient DC placement.

Optimizes per-site DC capacities so that the grid stays uncongested NOT just in the
base case but under single-line (N-1) contingencies. Uses the differentiable ADMM
SCOPF layer (contingency-aware) + LineUtilizationObjective(scenario_aggregation) +
the fixed BoxBudgetProjection.

Evaluation is exact brute-force N-1 via CVX (remove each critical line, re-dispatch,
measure congestion), comparing:
   - SCOPF-optimized placement (contingency-aware)
   - base-case-optimized placement (ignores contingencies)
   - uniform placement

Run a fast smoke test first:
  .venv/bin/python development/dc_placement_scopf.py --smoke \
     --network ~/zap_data/pypsa-networks/western_small/network_2023.nc \
     --snap-len 2 --contingencies 5 --admm-iters 300 --outer-iters 4
"""
import argparse, os, time, contextlib, io, json
from copy import deepcopy
import cvxpy as cp
import numpy as np
import torch
import scipy.sparse as sp
import zap
from zap.importers.pypsa import load_pypsa_network
from zap.admm import ADMMSolver, ADMMLayer
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


def make_dc(net, sites, caps, T):
    return zap.DataCenterLoad(num_nodes=net.num_nodes, terminal=np.array(sites),
        profiles=[np.ones(T)]*len(sites), nominal_capacity=np.array(caps, float),
        linear_cost=np.zeros(len(sites)), settime_horizon=T, capital_cost=np.zeros(len(sites)))


def base_dispatch(net, devices, T):
    return net.dispatch(devices, time_horizon=T, solver=cp.CLARABEL, add_ground=False)


def critical_lines(net, devices, T, k):
    oc = base_dispatch(net, devices, T)
    A = devices[3]
    flow = np.abs(np.asarray(oc.power[3][1]))
    limit = np.maximum(np.asarray(A.max_power) * np.asarray(A.nominal_capacity), 1e-9)
    u = (flow / limit).max(axis=1)            # per-line max utilization
    return np.argsort(u)[-k:][::-1].tolist()  # top-k most loaded


def build_mask(num_lines, crit, device="cpu"):
    nc = len(crit)
    m = sp.lil_matrix((nc, num_lines))
    for i, c in enumerate(crit):
        m[i, c] = 1.0
    tm = torch.tensor(m.todense(), dtype=torch.float32, device=device)
    return torch.vstack([torch.zeros(num_lines, dtype=torch.float32, device=device), tm])  # (nc+1, L)


def n1_eval(net, devices_np, sites, caps, T, crit):
    """Exact brute-force N-1: base + each critical line outaged; report sum_u2 stats."""
    A_idx = 3
    def one(devs):
        try:
            oc = net.dispatch(devs, time_horizon=T, solver=cp.CLARABEL, add_ground=False)
        except AssertionError:
            return None
        A = devs[A_idx]
        flow = np.abs(np.asarray(oc.power[A_idx][1]))
        limit = np.maximum(np.asarray(A.max_power) * np.asarray(A.nominal_capacity), 1e-9)
        p = np.clip(np.asarray(oc.prices) * COST_UNIT, 0, VOLL)
        return float(((flow / limit) ** 2).sum()), float(np.percentile(p, 95))

    devs0 = devices_np + [make_dc(net, sites, caps, T)]
    base = one(devs0)
    scen = []
    for c in crit:
        d2 = deepcopy(devices_np)
        d2[A_idx].nominal_capacity = np.asarray(d2[A_idx].nominal_capacity, float).copy()
        d2[A_idx].nominal_capacity[c] *= 1e-4         # outage: capacity ~ 0
        r = one(d2 + [make_dc(net, sites, caps, T)])
        scen.append(r)
    feas = base is not None and all(s is not None for s in scen)
    # Full N-1 security set = base case + every single-line outage (scenario 0 is base,
    # matching scenario_aggregation="mean" in the differentiable objective).
    all_u2 = ([base[0]] if base else []) + [s[0] for s in scen if s]
    return {"base_u2": base[0] if base else np.inf,
            "mean_n1_u2": float(np.mean(all_u2)) if all_u2 else np.inf,        # mean over base+contingencies
            "worst_n1_u2": float(np.max([s[0] for s in scen if s])) if any(scen) else np.inf,
            "worst_n1_p95": float(np.max([s[1] for s in scen if s])) if any(scen) else np.inf,
            "n_infeasible_scen": int(sum(s is None for s in scen)), "feasible": feas}


def scopf_optimize(net, devices_np, sites, B, cap, T, crit, admm_iters, outer_iters, step, agg):
    n = len(sites)
    devs_np = devices_np + [make_dc(net, sites, np.full(n, B/n), T)]
    devs_t = [d.torchify(machine="cpu", dtype=torch.float32) for d in devs_np]
    line_idx, dc_idx = 3, len(devs_np) - 1
    num_lines = devs_np[line_idx].nominal_capacity.shape[0]
    mask = build_mask(num_lines, crit)
    nc = len(crit)

    # Fixed rho (NOT adaptive) so the unrolled ADMM map is smooth and autograd
    # through it gives a consistent gradient; force a fixed iteration depth
    # (minimum==num) so the unroll length doesn't vary between forward passes.
    solver = ADMMSolver(dtype=torch.float32, num_iterations=admm_iters, minimum_iterations=admm_iters,
                        atol=0.0, adaptive_rho=False, rho_power=1.0, rho_angle=1.0, resid_norm=2)
    layer = ADMMLayer(network=net, devices=devs_t,
                      parameter_names={"dc_capacity": (dc_idx, "nominal_capacity")}, time_horizon=T,
                      solver=solver, num_contingencies=nc, contingency_device=line_idx, contingency_mask=mask)
    op_obj = zap.planning.LineUtilizationObjective(net, devs_t, metric="quadratic", scenario_aggregation=agg)
    inv_obj = zap.planning.InvestmentObjective(devs_t, layer)
    P = zap.planning.PlanningProblem(operation_objective=op_obj, investment_objective=inv_obj, layer=layer,
        lower_bounds={"dc_capacity": np.zeros(n)}, upper_bounds={"dc_capacity": np.full(n, cap)})
    P.extra_projections = {"dc_capacity": zap.planning.BoxBudgetProjection(
        budget=B, lower_bounds=np.zeros(n), upper_bounds=np.full(n, cap))}
    init = torch.tensor(P.extra_projections["dc_capacity"](np.full(n, B/n)), dtype=torch.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        state, hist = P.solve(algorithm=GradientDescent(step_size=step, clip=1e3),
                              num_iterations=outer_iters, initial_state={"dc_capacity": init}, verbosity=0)
    opt = state["dc_capacity"].detach().cpu().numpy().ravel()
    opt = P.extra_projections["dc_capacity"](opt)
    return opt, hist


def mean_n1_u2(net, devices_np, sites, caps, T, crit):
    """Mean sum_u2 over the N-1 security set {base + each critical-line outage}."""
    m = n1_eval(net, devices_np, sites, caps, T, crit)
    return m["mean_n1_u2"] if m["feasible"] else np.inf


def scopf_greedy_exact(net, devices_np, sites, B, cap, T, crit, probe=None):
    """SCOPF-aware placement via EXACT N-1 (no ADMM autograd): rank sites by their
    mean post-contingency marginal stress (over base + each line outage), then
    water-fill the budget to caps. Robust + interpretable; uses real line-outage
    dispatches as the objective (consistent with scenario_aggregation='mean')."""
    n = len(sites)
    probe = probe if probe is not None else min(cap, 0.25)
    base0 = mean_n1_u2(net, devices_np, sites, np.full(n, 1e-6), T, crit)
    marg = np.full(n, np.inf)
    for i in range(n):
        c = np.full(n, 1e-6); c[i] = probe
        w = mean_n1_u2(net, devices_np, sites, c, T, crit)
        marg[i] = (w - base0) / probe if np.isfinite(w) else np.inf
    order = np.argsort(marg)  # lowest worst-case N-1 marginal first
    caps_out = np.zeros(n); rem = B
    for i in order:
        if not np.isfinite(marg[i]):
            continue
        take = min(cap, rem); caps_out[i] = take; rem -= take
        if rem <= 1e-9:
            break
    return caps_out, marg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/zap_data/pypsa-networks/western_small/network_2023.nc")
    ap.add_argument("--snap-start", type=int, default=5448)
    ap.add_argument("--snap-len", type=int, default=2)
    ap.add_argument("--load-scale", type=float, default=1.0)
    ap.add_argument("--gen-scale", type=float, default=1.24)
    ap.add_argument("--line-scale", type=float, default=0.7, help="n-1 proxy derate for the *base* network")
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--site-cap", type=float, default=0.25)
    ap.add_argument("--contingencies", type=int, default=6)
    ap.add_argument("--admm-iters", type=int, default=500)
    ap.add_argument("--outer-iters", type=int, default=8)
    ap.add_argument("--step", type=float, default=2e-3)
    ap.add_argument("--agg", default="mean", choices=["mean", "max", "sum"])
    ap.add_argument("--engine", default="exact", choices=["exact", "grad"],
                    help="exact: greedy on real N-1 dispatches (robust); grad: differentiable ADMM (finicky)")
    ap.add_argument("--sites", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="development/results/placement_scopf")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    net, devices = load_net(args.network, args.snap_start, args.snap_len,
                            args.load_scale, args.gen_scale, args.line_scale)
    T = args.snap_len
    if args.sites:
        sites = [int(x) for x in args.sites.split(",")]
    elif net.num_nodes <= 110:
        sites = [32, 82, 50, 18, 15, 22, 43, 14, 23, 20, 94, 65, 78]
    else:
        sites = list(range(0, net.num_nodes, 16))
    sites = [s for s in sites if s < net.num_nodes]
    n = len(sites)
    B, cap = args.budget, args.site_cap
    assert n * cap >= B, f"infeasible: {n} sites x {cap} < {B}"
    crit = critical_lines(net, devices, T, args.contingencies)
    print(f"# SCOPF PLACEMENT  net={os.path.basename(args.network)} nodes={net.num_nodes} "
          f"lines={devices[3].nominal_capacity.shape[0]}")
    print(f"# {n} sites | B={B}GW cap={cap*1000:.0f}MW | {len(crit)} N-1 contingencies (lines {crit}) "
          f"| agg={args.agg}")

    if args.smoke:
        t0 = time.time()
        devs_np = devices + [make_dc(net, sites, np.full(n, B/n), T)]
        devs_t = [d.torchify(machine="cpu", dtype=torch.float32) for d in devs_np]
        mask = build_mask(devs_np[3].nominal_capacity.shape[0], crit)
        solver = ADMMSolver(dtype=torch.float32, num_iterations=args.admm_iters, minimum_iterations=min(100, args.admm_iters),
                            atol=3e-3, adaptive_rho=True, rho_power=1.0, rho_angle=1.0, resid_norm=2)
        layer = ADMMLayer(network=net, devices=devs_t, parameter_names={"dc_capacity": (len(devs_np)-1, "nominal_capacity")},
                          time_horizon=T, solver=solver, num_contingencies=len(crit), contingency_device=3, contingency_mask=mask)
        op_obj = zap.planning.LineUtilizationObjective(net, devs_t, metric="quadratic", scenario_aggregation=args.agg)
        P = zap.planning.PlanningProblem(operation_objective=op_obj,
            investment_objective=zap.planning.InvestmentObjective(devs_t, layer), layer=layer,
            lower_bounds={"dc_capacity": np.zeros(n)}, upper_bounds={"dc_capacity": np.full(n, cap)})
        P.extra_projections = {"dc_capacity": zap.planning.BoxBudgetProjection(
            budget=B, lower_bounds=np.zeros(n), upper_bounds=np.full(n, cap))}
        eta = {"dc_capacity": torch.tensor(np.full(n, B/n), dtype=torch.float32)}
        with contextlib.redirect_stdout(io.StringIO()):
            J = P(**eta, requires_grad=True); g = P.backward()
        print(f"[smoke] ADMM SCOPF forward+backward OK in {time.time()-t0:.1f}s  "
              f"J={float(J):.4f}  grad nonzero={int((np.abs(g['dc_capacity'].detach().numpy())>1e-9).sum())}/{n}")
        return

    # ---- full: optimize SCOPF-aware, compare to base-case-opt and uniform ----
    t0 = time.time()
    if args.engine == "grad":
        opt_scopf, hist = scopf_optimize(net, devices, sites, B, cap, T, crit,
                                         args.admm_iters, args.outer_iters, args.step, args.agg)
        print(f"# SCOPF(grad) solve {args.outer_iters} outer iters in {time.time()-t0:.1f}s  "
              f"loss {hist['loss'][0]:.3f} -> {hist['loss'][-1]:.3f}")
    else:
        opt_scopf, marg = scopf_greedy_exact(net, devices, sites, B, cap, T, crit)
        hist = {"loss": []}
        print(f"# SCOPF(exact-N1 greedy) in {time.time()-t0:.1f}s  "
              f"used {int((opt_scopf>1e-4).sum())} sites; worst-N1 marginal range "
              f"[{np.nanmin(marg[np.isfinite(marg)]):.1f}, {np.nanmax(marg[np.isfinite(marg)]):.1f}]")

    # base-case (no contingencies) optimizer via cvx layer
    dc_idx = len(devices)
    layer_b = zap.DispatchLayer(net, devices + [make_dc(net, sites, np.full(n, B/n), T)],
        parameter_names={"dc_capacity": (dc_idx, "nominal_capacity")}, time_horizon=T, solver=cp.CLARABEL)
    Pb = zap.planning.PlanningProblem(
        operation_objective=zap.planning.LineUtilizationObjective(net, layer_b.devices, metric="quadratic"),
        investment_objective=zap.planning.InvestmentObjective(layer_b.devices, layer_b), layer=layer_b,
        lower_bounds={"dc_capacity": np.zeros(n)}, upper_bounds={"dc_capacity": np.full(n, cap)})
    Pb.extra_projections = {"dc_capacity": zap.planning.BoxBudgetProjection(
        budget=B, lower_bounds=np.zeros(n), upper_bounds=np.full(n, cap))}
    initb = Pb.extra_projections["dc_capacity"](np.full(n, B/n))
    with contextlib.redirect_stdout(io.StringIO()):
        sb, _ = Pb.solve(algorithm=GradientDescent(step_size=args.step, clip=1e3),
                         num_iterations=max(args.outer_iters, 40), initial_state={"dc_capacity": initb}, verbosity=0)
    opt_base = Pb.extra_projections["dc_capacity"](np.asarray(sb["dc_capacity"]).ravel())
    uniform = np.full(n, B/n)

    print(f"\n# Exact N-1 evaluation ({len(crit)} contingencies):")
    print(f"{'placement':>16} {'base_u2':>8} {'mean_N1_u2':>10} {'worst_N1_u2':>11} {'worst_N1_p95':>12} {'infeas_scen':>11}")
    res = {}
    for name, caps in [("scopf-opt", opt_scopf), ("base-opt", opt_base), ("uniform", uniform)]:
        m = n1_eval(net, devices, sites, caps, T, crit)
        res[name] = m
        print(f"{name:>16} {m['base_u2']:8.1f} {m['mean_n1_u2']:10.1f} {m['worst_n1_u2']:11.1f} "
              f"{m['worst_n1_p95']:12.1f} {m['n_infeasible_scen']:11d}")

    s, b, u = res["scopf-opt"], res["base-opt"], res["uniform"]
    print("\n## HEADLINE  (SCOPF target = MEAN over N-1 security set {base + each outage})")
    print(f"  SCOPF-aware  : mean-N1={s['mean_n1_u2']:.1f}  base={s['base_u2']:.1f}  worst-N1={s['worst_n1_u2']:.1f}")
    print(f"  base-case-opt: mean-N1={b['mean_n1_u2']:.1f}  base={b['base_u2']:.1f}  worst-N1={b['worst_n1_u2']:.1f}")
    print(f"  uniform      : mean-N1={u['mean_n1_u2']:.1f}  base={u['base_u2']:.1f}  worst-N1={u['worst_n1_u2']:.1f}")
    if np.isfinite(s['mean_n1_u2']) and np.isfinite(b['mean_n1_u2']) and b['mean_n1_u2'] > 0:
        print(f"  -> SCOPF-aware placement cuts MEAN post-contingency congestion by "
              f"{100*(1-s['mean_n1_u2']/b['mean_n1_u2']):.1f}% vs base-case-only optimization")
        print(f"     (worst-case N-1 is a harder minimax this greedy does not target; base-opt can win there)")
    out = {"network": os.path.basename(args.network), "sites": sites, "budget": B, "cap": cap,
           "contingencies": crit, "alloc_scopf": opt_scopf.tolist(), "alloc_base": opt_base.tolist(),
           "results": res, "loss_traj": [float(x) for x in hist["loss"]]}
    json.dump(out, open(os.path.join(args.outdir, "scopf_results.json"), "w"), indent=2, default=float)
    print(f"saved -> {args.outdir}/scopf_results.json")


if __name__ == "__main__":
    main()
