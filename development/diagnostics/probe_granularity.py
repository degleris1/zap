"""
Core constructive test: does spreading a FIXED DC budget over more nodes reduce
grid stress vs concentrating it? (the '100x1MW beats 1x100MW' claim)

For budget B GW, split equally over the first k candidate nodes (k=1 is full
concentration). Report stress metrics vs k, at NATURAL line capacities.
"""
import argparse
import os
from copy import deepcopy
import cvxpy as cp
import numpy as np
import zap
from zap.importers.pypsa import load_pypsa_network

POWER_UNIT, COST_UNIT = 1.0e3, 100.0
CANDS = [32, 82, 50, 18, 15, 22, 43, 14, 23, 20, 94, 65, 78]


def load(path, s0, slen, ls, gs, lns, crush):
    import pypsa
    pn = pypsa.Network(os.path.expanduser(path))
    snaps = pn.generators_t.p_max_pu.index
    net, devices = load_pypsa_network(pn, snaps[s0:s0+slen], power_unit=POWER_UNIT, cost_unit=COST_UNIT)
    devices = deepcopy(devices)
    devices[1].load *= ls; devices[0].dynamic_capacity *= gs; devices[3].nominal_capacity *= lns
    if crush:
        devices[3].nominal_capacity[168] = 0.5; devices[3].nominal_capacity[176] = 0.5
        devices[3].nominal_capacity[49] = 0.3
    return net, devices


def make_devs(devices, terminals, caps, T):
    devs = deepcopy(devices)
    dc = zap.DataCenterLoad(
        num_nodes=devs[0].num_nodes, terminal=np.array(terminals),
        profiles=[np.ones(T)] * len(terminals), nominal_capacity=np.array(caps, dtype=float),
        linear_cost=np.zeros(len(terminals)), settime_horizon=T, capital_cost=np.zeros(len(terminals)),
    )
    devs.append(dc); return devs


def metrics(net, devs, devices, T):
    try:
        oc = net.dispatch(devs, time_horizon=T, solver=cp.CLARABEL, add_ground=False)
    except AssertionError:
        return {"cost": float("nan"), "max_lmp": float("inf"), "meanmax_lmp": float("inf"),
                "lmp_cvar95": float("inf"), "max_util": float("inf"), "n_lines_90": -1,
                "sum_u2": float("inf"), "shed_pct": float("inf"), "infeasible": True}
    p = np.asarray(oc.prices) * COST_UNIT
    A = devs[3]
    flow = np.abs(np.asarray(oc.power[3][1]))
    limit = np.maximum(np.asarray(A.max_power) * np.asarray(A.nominal_capacity), 1e-9)
    u = flow / limit
    line_max_u = u.max(axis=1)
    L = devices[1]
    req = (np.asarray(L.load) * np.asarray(L.nominal_capacity)).sum()
    served = (-np.asarray(oc.power[1][0])).sum()
    shed = max(req - served, 0.0)
    return {
        "cost": float(oc.problem.value),
        "max_lmp": float(p.max()),
        "meanmax_lmp": float(np.mean(p.max(axis=1))),
        "lmp_cvar95": float(np.mean(np.sort(p.ravel())[int(0.95*p.size):])),
        "max_util": float(u.max()),
        "n_lines_90": int((line_max_u > 0.9).sum()),
        "sum_u2": float((u**2).sum()),
        "shed_pct": float(100*shed/req),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/zap_data/pypsa-networks/western_small/network_2023.nc")
    ap.add_argument("--snapshot", type=int, default=5448)
    ap.add_argument("--snap-len", type=int, default=24)
    ap.add_argument("--line-scale", type=float, default=1.0)
    ap.add_argument("--load-scale", type=float, default=1.27)
    ap.add_argument("--gen-scale", type=float, default=1.24)
    ap.add_argument("--crush", action="store_true")
    ap.add_argument("--budget", type=float, default=5.0, help="total DC budget GW")
    args = ap.parse_args()

    net, devices = load(args.network, args.snapshot, args.snap_len, args.load_scale, args.gen_scale, args.line_scale, args.crush)
    T = args.snap_len
    B = args.budget
    base = metrics(net, make_devs(devices, [CANDS[0]], [0.0], T), devices, T)
    print(f"network={os.path.basename(args.network)} line*{args.line_scale} crush={args.crush} budget={B}GW")
    print(f"BASE (no DC): max_lmp={base['max_lmp']:.1f} meanmax={base['meanmax_lmp']:.1f} "
          f"max_util={base['max_util']:.3f} n90={base['n_lines_90']} sumU2={base['sum_u2']:.1f}\n")

    print("GRANULARITY: split fixed budget B over first k candidate nodes (k=1 = concentrate)")
    print(f"{'k':>3} {'per-node':>9} {'max_lmp':>8} {'meanmax':>8} {'cvar95':>8} "
          f"{'max_util':>8} {'n90':>4} {'sumU2':>8} {'shed%':>7}")
    for k in [1, 2, 3, 5, 8, 10, 13]:
        if k > len(CANDS):
            continue
        terms = CANDS[:k]
        caps = [B / k] * k
        m = metrics(net, make_devs(devices, terms, caps, T), devices, T)
        print(f"{k:3d} {B/k:9.3f} {m['max_lmp']:8.1f} {m['meanmax_lmp']:8.2f} {m['lmp_cvar95']:8.2f} "
              f"{m['max_util']:8.3f} {m['n_lines_90']:4d} {m['sum_u2']:8.1f} {m['shed_pct']:7.3f}")

    print("\nCONCENTRATION: all B at a single node, for each candidate")
    print(f"{'node':>5} {'max_lmp':>8} {'meanmax':>8} {'max_util':>8} {'n90':>4} {'sumU2':>8} {'shed%':>7}")
    conc = []
    for nd in CANDS:
        m = metrics(net, make_devs(devices, [nd], [B], T), devices, T)
        conc.append((nd, m))
        print(f"{nd:5d} {m['max_lmp']:8.1f} {m['meanmax_lmp']:8.2f} {m['max_util']:8.3f} "
              f"{m['n_lines_90']:4d} {m['sum_u2']:8.1f} {m['shed_pct']:7.3f}")
    worst = max(conc, key=lambda x: x[1]['sum_u2'])
    spread10 = metrics(net, make_devs(devices, CANDS[:10], [B/10]*10, T), devices, T)
    print(f"\nHEADLINE @ B={B}GW: worst-node concentration (node {worst[0]}) sumU2={worst[1]['sum_u2']:.1f}, "
          f"max_lmp={worst[1]['max_lmp']:.0f}, shed={worst[1]['shed_pct']:.2f}%")
    print(f"            10-way spread sumU2={spread10['sum_u2']:.1f}, max_lmp={spread10['max_lmp']:.0f}, "
          f"shed={spread10['shed_pct']:.2f}%")
    print(f"            stress reduction (sumU2): {100*(1-spread10['sum_u2']/worst[1]['sum_u2']):.1f}%")


if __name__ == "__main__":
    main()
