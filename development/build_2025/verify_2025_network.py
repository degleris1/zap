"""
Verify a freshly-built PyPSA-USA network is (a) the 2025 weather year, (b) higher
resolution, and (c) loads + dispatches in zap so the study pipeline will run on it.

Usage:
  python development/build_2025/verify_2025_network.py /path/to/elec_s1000_c1000.nc
"""
import sys, os
import numpy as np
import pandas as pd
import pypsa


def main(path):
    pn = pypsa.Network(os.path.expanduser(path))
    idx = pn.generators_t.p_max_pu.index
    ts = pd.DatetimeIndex(idx.get_level_values(-1) if getattr(idx, "nlevels", 1) > 1 else idx)
    years = sorted(set(int(y) for y in ts.year))
    print(f"file        : {path}")
    print(f"buses       : {len(pn.buses)}   lines: {len(pn.lines)}   generators: {len(pn.generators)}")
    print(f"snapshots   : {len(pn.snapshots)}  ({ts[0]} -> {ts[-1]})")
    print(f"years present: {years}")
    ok_year = years == [2025]
    ok_res = len(pn.buses) > 490
    print(f"[{'OK' if ok_year else 'XX'}] weather year is 2025")
    print(f"[{'OK' if ok_res else 'XX'}] higher resolution than the 490-node baseline")

    # zap round-trip + a base dispatch at a summer hour
    try:
        import cvxpy as cp
        import zap
        from zap.importers.pypsa import load_pypsa_network
        summer = int(np.argmax((ts.month == 8) & (ts.day == 1) & (ts.hour == 12)))
        sd = idx[summer:summer + 2]
        net, devices = load_pypsa_network(pn, sd, power_unit=1.0e3, cost_unit=100.0)
        oc = net.dispatch(devices, time_horizon=2, solver=cp.CLARABEL, add_ground=False)
        mu = (np.asarray(oc.local_inequality_duals[3][0]) + np.asarray(oc.local_inequality_duals[3][1]))
        nb = int(np.sum(mu.max(axis=1) > 1e-4))
        p = np.asarray(oc.prices) * 100.0
        print(f"[OK] zap load + dispatch works: {net.num_nodes} nodes, "
              f"{devices[3].nominal_capacity.shape[0]} lines, {nb} binding lines, "
              f"median LMP ${np.median(np.clip(p,0,1000)):.1f}/MWh at {sd[0]}")
        print("\nReady. Run the study:")
        print(f"  python development/dc_placement_study.py --network {path} --year 2025 --budget 3.0")
    except Exception as e:
        print(f"[XX] zap dispatch failed: {e}")
        return 1
    return 0 if (ok_year and ok_res) else 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: verify_2025_network.py /path/to/network.nc"); sys.exit(1)
    sys.exit(main(sys.argv[1]))
