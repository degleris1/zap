"""Pilot audit for a NEW PyPSA network before any citable DC-placement run.

Go/no-go checks, in the order they can kill a study:
  1. unit conventions (gen marginal $/MWh range, total load in GW) -- the
     probe_units.py lesson: any convention mismatch invalidates everything;
  2. genuine congestion at natural caps (binding lines, LMP spread) -- the
     101-node copperplate lesson: an over-aggregated network is degenerate;
  3. dispatch tractability (timed solve per hour) -- sets the budget for a
     full 8-seed x 24-pool wall replication;
  4. penetration math -- the fleet GW that matches WECC-490's B=10 at this
     network's load (penetration-matched, never absolute GW).

Uses the same loader/metrics as the citable studies (load_pypsa_network with
POWER_UNIT/COST_UNIT via dc_placement_study/dc_placement_integrated) so the
pilot sees exactly what the studies would see.

Run:
  uv run python development/diagnostics/pilot_network.py \
      --network ~/Downloads/texas_elec_s500_c500.nc
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pypsa  # noqa: E402
from dc_placement_study import COST_UNIT, metrics, dev_index  # noqa: E402
from dc_placement_integrated import build_raw_hour  # noqa: E402

# WECC-490 reference for penetration matching (scaled load at the wall study's
# load x1.2, computed by running this pilot on western_elec_s490_c490.nc).
WECC490_HEADLINE_B = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--n-hours", type=int, default=3,
                    help="stressed hours to dispatch (peak + median + p90 of load)")
    ap.add_argument("--load-scales", default="1.0,1.2")
    ap.add_argument("--wecc-load-gw", type=float, default=None,
                    help="WECC-490 total load (GW, at x1.0) for penetration matching")
    args = ap.parse_args()

    t0 = time.time()
    pn = pypsa.Network(os.path.expanduser(args.network))
    print(f"[load] {time.time() - t0:.1f}s | buses {len(pn.buses)} | AC lines {len(pn.lines)} "
          f"| links {len(pn.links)} | gens {len(pn.generators)} | loads {len(pn.loads)} "
          f"| storage {len(pn.storage_units)} | snapshots {len(pn.snapshots)} "
          f"[{pn.snapshots[0]} .. {pn.snapshots[-1]}]")

    snaps = pn.generators_t.p_max_pu.index
    tot = pn.loads_t.p_set.reindex(snaps).sum(axis=1)
    order = np.argsort(-tot.values)
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[(len(order) * 9) // 10])]
    hours = sorted(set(picks))[: args.n_hours]
    print(f"[hours] peak/median/p90-load snapshots: {[str(snaps[h]) for h in hours]}")
    print(f"[load]  raw total load: peak {tot.values[order[0]] / 1e3:.1f} GW, "
          f"mean {tot.values.mean() / 1e3:.1f} GW")

    # --- unit audit on the peak hour, natural caps ------------------------- #
    t0 = time.time()
    net, devs, date, hod = build_raw_hour(pn, snaps, hours[0], 1.0, 1.0, 1.0)
    print(f"[build] one-hour zap network in {time.time() - t0:.1f}s | "
          f"devices: {[type(d).__name__ for d in devs]}")
    G, L, A = devs[0], devs[1], devs[dev_index(devs, "ACLine")]
    gen_cost = np.asarray(G.linear_cost).ravel() * COST_UNIT
    load_gw = float((np.asarray(L.load) * np.asarray(L.nominal_capacity)).sum())
    avail_gw = float((np.asarray(G.dynamic_capacity).ravel()
                      * np.asarray(G.nominal_capacity).ravel()).sum())
    print(f"[units] gen marginal cost: {gen_cost.min():.1f} .. {gen_cost.max():.1f} $/MWh "
          f"(median {np.median(gen_cost):.1f}) -- expect ~0-100, NOT 0-10000")
    print(f"[units] total load {load_gw:.1f} GW | available gen {avail_gw:.1f} GW "
          f"| reserve margin {100 * (avail_gw - load_gw) / max(load_gw, 1e-9):.0f}% (peak hour)")

    # --- timed dispatches -------------------------------------------------- #
    rows = []
    for ls in [float(x) for x in args.load_scales.split(",")]:
        for h in hours:
            net, devs, date, hod = build_raw_hour(pn, snaps, h, ls, 1.0, 1.0)
            t0 = time.time()
            m = metrics(net, devs, devs, 1)
            dt = time.time() - t0
            rows.append((date, ls, dt, m))
            print(f"[dispatch] {date} x{ls:.1f} | {dt:6.1f}s | feas {m['feasible']} | "
                  f"LMP med {m['median_lmp']:8.1f} p95 {m['p95_lmp']:8.1f} "
                  f"disp {m['lmp_disp']:8.1f} $/MWh | binding {m['n_binding']:4d} | "
                  f"max_u {m['max_util']:.2f} | shed {m['shed_pct']:.2f}%")

    # --- verdicts ----------------------------------------------------------- #
    ok = [m for (_, _, _, m) in rows if m["feasible"]]
    if ok:
        med_binding = int(np.median([m["n_binding"] for m in ok]))
        med_disp = float(np.median([m["lmp_disp"] for m in ok]))
        avg_t = float(np.mean([t for (_, _, t, _) in rows]))
        print(f"\n[verdict] median binding lines {med_binding} | median LMP dispersion "
              f"{med_disp:.1f} $/MWh | mean dispatch {avg_t:.1f}s")
        print("[verdict] copperplate risk:",
              "HIGH (few binding lines, flat prices -- 101-node deja vu)"
              if med_binding < 5 and med_disp < 5 else "low (genuine congestion)")
        full_run_s = avg_t * 8 * 24 * 24 * 3   # seeds x pools x snaps x fleets, wall only
        print(f"[verdict] full wall replication (8 seeds x 24 pools x 24 snaps x 3 fleets) "
              f"~ {full_run_s / 3600:.0f}h at this dispatch speed "
              f"(reduced 4x12 pin ~ {full_run_s / 4 / 3600:.0f}h)")
    if args.wecc_load_gw:
        b_eq = WECC490_HEADLINE_B * load_gw / args.wecc_load_gw
        print(f"[penetration] B equivalent to WECC-490's 10 GW: "
              f"{b_eq:.1f} GW (this net {load_gw:.0f} GW vs WECC {args.wecc_load_gw:.0f} GW peak)")


if __name__ == "__main__":
    main()
