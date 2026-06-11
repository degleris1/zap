"""
Diagnostic probe for the DC-placement degeneracy.

Loads a PyPSA network into zap, runs a base DCOPF, and reports the quantities
that determine whether DC placement *can* matter:

  - generation headroom vs load (is the system slack?)
  - LMP flatness across nodes/time (uncongested => flat == single system price)
  - line utilization distribution + count of binding lines
  - response of LMPs / congestion to injecting a block of load at one node,
    swept from 0 to MAX GW (where is the "cliff"?)

Usage:
  .venv/bin/python development/diagnostics/probe_congestion.py \
      --network ~/zap_data/pypsa-networks/western_small/network_2023.nc \
      --snapshot 5448 --inject-node 20 --max-inject 6.0

All zap power units are GW (power_unit=1e3), costs are $/MWh after x100 (cost_unit=100).
"""

import argparse
import os
import sys
from copy import deepcopy

import cvxpy as cp
import numpy as np

import zap
from zap.importers.pypsa import load_pypsa_network, parse_buses

POWER_UNIT = 1.0e3   # MW per zap unit  => zap power is in GW
COST_UNIT = 100.0    # divide $/MWh by this; multiply LMP by this to recover $/MWh


def load(network_path, snap_start, snap_len, load_scale, gen_scale, line_scale, crush):
    import pypsa

    pn = pypsa.Network(os.path.expanduser(network_path))
    snaps = pn.generators_t.p_max_pu.index
    snapshot_data = snaps[snap_start : snap_start + snap_len]

    net, devices = load_pypsa_network(
        pn, snapshot_data, power_unit=POWER_UNIT, cost_unit=COST_UNIT
    )
    devices = deepcopy(devices)
    # device order: [Generator, Load, DCLine, ACLine, Battery]
    devices[1].load *= load_scale
    devices[0].dynamic_capacity *= gen_scale
    devices[3].nominal_capacity *= line_scale
    if crush:
        # the manual line crush used in pushing_capacity.py
        devices[3].nominal_capacity[168] = 0.5
        devices[3].nominal_capacity[176] = 0.5
        devices[3].nominal_capacity[49] = 0.3
    return pn, net, devices, snapshot_data


def line_util(outcome, devices, line_idx=3):
    d = devices[line_idx]
    flow = np.asarray(outcome.power[line_idx][1])  # (L, T)
    limit = np.asarray(d.max_power) * np.asarray(d.nominal_capacity)  # (L,1) per-unit*GW
    limit = np.maximum(limit, 1e-9)
    u = np.abs(flow) / limit  # (L,T) broadcast
    return u


def binding_lines(outcome, line_idx=3, eps=1e-4):
    mu_lo = np.asarray(outcome.local_inequality_duals[line_idx][0])
    mu_hi = np.asarray(outcome.local_inequality_duals[line_idx][1])
    mu = mu_lo + mu_hi  # (L,T)
    line_max = mu.max(axis=1)
    return mu, int(np.sum(line_max > eps))


def add_dc(devices, node, gw, profile_path, T):
    devs = deepcopy(devices)
    # constant block load of length T (per-unit profile == 1 at every step)
    dc = zap.DataCenterLoad(
        num_nodes=devs[0].num_nodes,
        terminal=np.array([node]),
        profiles=[np.ones(T)],
        nominal_capacity=np.array([float(gw)]),
        linear_cost=np.zeros(1),
        settime_horizon=T,
        capital_cost=np.zeros(1),
    )
    devs.append(dc)
    return devs


def lmp_dollars(outcome):
    """Return LMP array in $/MWh (nodes x time)."""
    return np.asarray(outcome.prices) * COST_UNIT


def summarize_lmp(outcome, tag):
    p = lmp_dollars(outcome)
    flat = p.std() / (abs(p.mean()) + 1e-9)
    print(f"  [{tag}] LMP $/MWh: min={p.min():.2f} median={np.median(p):.2f} "
          f"mean={p.mean():.2f} max={p.max():.2f} std={p.std():.3f} CoV={flat:.4f} "
          f"n_unique={len(np.unique(np.round(p,3)))}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/zap_data/pypsa-networks/western_small/network_2023.nc")
    ap.add_argument("--snapshot", type=int, default=5448)
    ap.add_argument("--snap-len", type=int, default=24)
    ap.add_argument("--load-scale", type=float, default=1.27)
    ap.add_argument("--gen-scale", type=float, default=1.24)
    ap.add_argument("--line-scale", type=float, default=0.7)
    ap.add_argument("--crush", action="store_true")
    ap.add_argument("--inject-node", type=int, default=20)
    ap.add_argument("--max-inject", type=float, default=6.0)
    ap.add_argument("--inject-step", type=float, default=0.5)
    ap.add_argument("--profile", default=os.path.expanduser(
        "~/zap/development/load_profiles/example_inference_azure_conv.csv"))
    args = ap.parse_args()

    print(f"=== NETWORK: {args.network}  snapshot={args.snapshot}:{args.snapshot+args.snap_len} "
          f"load*{args.load_scale} gen*{args.gen_scale} line*{args.line_scale} crush={args.crush} ===")

    pn, net, devices, snaps = load(
        args.network, args.snapshot, args.snap_len,
        args.load_scale, args.gen_scale, args.line_scale, args.crush
    )
    T = args.snap_len
    G, L, A = devices[0], devices[1], devices[3]
    n_nodes = net.num_nodes
    n_lines = A.nominal_capacity.shape[0]

    # NB: Load.load stays in MW; the GW conversion lives in Load.nominal_capacity (=1/power_unit).
    gen_cap_gw = float(np.sum(G.nominal_capacity))   # GW (sum of p_nom/1000)
    eff_load = (np.asarray(L.load) * np.asarray(L.nominal_capacity)).sum(axis=0)  # GW per t
    peak_load_gw = float(np.max(eff_load))
    avail_gen_gw = float(np.max((G.nominal_capacity * G.dynamic_capacity).sum(axis=0)))
    print(f"nodes={n_nodes} lines={n_lines} "
          f"gen_nameplate={gen_cap_gw:.1f}GW avail_gen_peak={avail_gen_gw:.1f}GW "
          f"peak_load={peak_load_gw:.1f}GW  reserve_margin={(avail_gen_gw/peak_load_gw-1)*100:.0f}%")

    # --- base dispatch ---
    base = net.dispatch(devices, time_horizon=T, solver=cp.CLARABEL, add_ground=False)
    print(f"base dispatch status={base.problem.status} cost={base.problem.value:.4f}")
    summarize_lmp(base, "base")
    u = line_util(base, devices)
    line_max_u = u.max(axis=1)
    mu, nb = binding_lines(base, eps=1e-4)
    print(f"  line util (max over time): >0.5:{int((line_max_u>0.5).sum())} "
          f">0.8:{int((line_max_u>0.8).sum())} >0.95:{int((line_max_u>0.95).sum())} "
          f">0.99:{int((line_max_u>0.99).sum())} | binding(dual>1e-4):{nb} / {n_lines}")
    print(f"  max util over all lines/time = {u.max():.3f}")

    base_p = lmp_dollars(base)
    base_node_lmp = base_p[args.inject_node].mean()
    print(f"  LMP at inject-node {args.inject_node}: mean={base_node_lmp:.2f} $/MWh")

    # --- injection sweep at one node ---
    print(f"\n--- inject at node {args.inject_node}, sweep 0..{args.max_inject} GW ---")
    print(f"{'GW':>6} {'cost':>12} {'maxLMP':>9} {'meanLMP':>9} "
          f"{'node_LMP':>9} {'n_bind':>7} {'max_util':>8}")
    grid = np.arange(0.0, args.max_inject + 1e-9, args.inject_step)
    for gw in grid:
        if gw == 0.0:
            oc, devs = base, devices
        else:
            devs = add_dc(devices, args.inject_node, gw, args.profile, T)
            oc = net.dispatch(devs, time_horizon=T, solver=cp.CLARABEL, add_ground=False)
        p = lmp_dollars(oc)
        _, nbind = binding_lines(oc, eps=1e-4)
        uu = line_util(oc, devs)
        node_lmp = p[args.inject_node].mean()
        print(f"{gw:6.2f} {oc.problem.value:12.4f} {p.max():9.2f} {p.mean():9.3f} "
              f"{node_lmp:9.2f} {nbind:7d} {uu.max():8.3f}")

    print("\n=== interpretation ===")
    print("If max/mean/node LMP and n_bind stay constant across the sweep, the grid is")
    print("uncongested at this operating point and placement is degenerate by construction.")


if __name__ == "__main__":
    sys.exit(main())
