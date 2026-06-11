"""
Map placement heterogeneity: inject a fixed block at each candidate node and
measure the marginal system response. If every node has the same dCost/dGW
(== system energy price) and ~zero dLMP, placement is degenerate. Spread across
nodes == placement matters.
"""
import argparse
import os
from copy import deepcopy

import cvxpy as cp
import numpy as np

import zap
from zap.importers.pypsa import load_pypsa_network

POWER_UNIT, COST_UNIT = 1.0e3, 100.0


def load(path, s0, slen, ls, gs, lns, crush):
    import pypsa
    pn = pypsa.Network(os.path.expanduser(path))
    snaps = pn.generators_t.p_max_pu.index
    net, devices = load_pypsa_network(pn, snaps[s0:s0+slen], power_unit=POWER_UNIT, cost_unit=COST_UNIT)
    devices = deepcopy(devices)
    devices[1].load *= ls
    devices[0].dynamic_capacity *= gs
    devices[3].nominal_capacity *= lns
    if crush:
        devices[3].nominal_capacity[168] = 0.5
        devices[3].nominal_capacity[176] = 0.5
        devices[3].nominal_capacity[49] = 0.3
    return net, devices


def dispatch(net, devices, T):
    return net.dispatch(devices, time_horizon=T, solver=cp.CLARABEL, add_ground=False)


def add_dc(devices, node, gw, T):
    devs = deepcopy(devices)
    dc = zap.DataCenterLoad(
        num_nodes=devs[0].num_nodes, terminal=np.array([node]),
        profiles=[np.ones(T)], nominal_capacity=np.array([float(gw)]),
        linear_cost=np.zeros(1), settime_horizon=T, capital_cost=np.zeros(1),
    )
    devs.append(dc); return devs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/zap_data/pypsa-networks/western_small/network_2023.nc")
    ap.add_argument("--snapshot", type=int, default=5448)
    ap.add_argument("--snap-len", type=int, default=24)
    ap.add_argument("--line-scale", type=float, default=1.0)
    ap.add_argument("--block", type=float, default=2.0, help="GW block injected per node")
    ap.add_argument("--nodes", default="", help="comma list; default=candidate set")
    ap.add_argument("--crush", action="store_true")
    args = ap.parse_args()

    cands = [32, 82, 50, 18, 15, 22, 43, 14, 23, 20, 94, 65, 78]
    nodes = [int(x) for x in args.nodes.split(",")] if args.nodes else cands

    net, devices = load(args.network, args.snapshot, args.snap_len,
                        1.27, 1.24, args.line_scale, args.crush)
    T = args.snap_len
    base = dispatch(net, devices, T)
    base_cost = base.problem.value
    base_p = np.asarray(base.prices) * COST_UNIT
    print(f"network={os.path.basename(args.network)} line*{args.line_scale} crush={args.crush} "
          f"block={args.block}GW  base_cost={base_cost:.3f}")
    print(f"base LMP: mean={base_p.mean():.2f} max={base_p.max():.2f} $/MWh\n")
    print(f"{'node':>5} {'base_LMP':>9} {'node_LMP+':>10} {'dLMP/GW':>9} "
          f"{'dCost/GW':>9} {'dMaxLMP':>9}")
    rows = []
    for nd in nodes:
        devs = add_dc(devices, nd, args.block, T)
        oc = dispatch(net, devs, T)
        p = np.asarray(oc.prices) * COST_UNIT
        b_lmp = base_p[nd].mean()
        n_lmp = p[nd].mean()
        dcost = (oc.problem.value - base_cost) / args.block * COST_UNIT  # $/MWh-equiv marginal
        dlmp = (n_lmp - b_lmp) / args.block
        dmax = p.max() - base_p.max()
        rows.append((nd, dlmp, dcost))
        print(f"{nd:5d} {b_lmp:9.2f} {n_lmp:10.2f} {dlmp:9.2f} {dcost:9.2f} {dmax:9.2f}")

    dlmps = np.array([r[1] for r in rows])
    dcosts = np.array([r[2] for r in rows])
    print(f"\nSpread across nodes:")
    print(f"  dLMP/GW : min={dlmps.min():.2f} max={dlmps.max():.2f} "
          f"range={dlmps.max()-dlmps.min():.2f} $/MWh per GW")
    print(f"  dCost/GW: min={dcosts.min():.2f} max={dcosts.max():.2f} "
          f"range={dcosts.max()-dcosts.min():.2f}")
    print("  -> Large range => placement matters. ~0 range => degenerate (copperplate).")


if __name__ == "__main__":
    main()
