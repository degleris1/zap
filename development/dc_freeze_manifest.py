"""
Freeze a per-network cleaning manifest ONCE, reproducibly.

The 25-bad list (`bad_buses_490_load1p0_25.json`) is WECC-490-specific. For a
citable run on any other network we need its own manifest, detected the SAME way
the studies detect (so a frozen file is interchangeable with `--bad-buses-json ''`
re-detection): `find_bad_buses` at load x1.0, DEFAULT thresholds, on the seed-0
panel of `--n-snaps` hours. This mirrors dc_spread_frontier.py:345-352 exactly.

Detect once, freeze, and point every study at the file -- so cleaning never varies
with a study's panel/seed. Expect single-digit manifests on the rebuilt nets (the
1-2 stray buses), not 25.

Usage:
  python development/dc_freeze_manifest.py \
      --network /scratch/users/gfw/pypsa-usa/resources/Default/eastern/elec_s3000_c3000.nc \
      --out development/results/cleaning/bad_buses_eastern_load1p0.json
"""
import argparse
import os
import json

import pypsa

from dc_placement_study import sample_panel_indices, find_bad_buses
from dc_placement_integrated import build_raw_hour


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--n-snaps", type=int, default=12,
                    help="panel size for detection (match the studies' default = 12)")
    ap.add_argument("--out", required=True, help="output manifest JSON path")
    args = ap.parse_args()

    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    idx = sample_panel_indices(len(snaps), args.n_snaps, None, None, 0)
    raw0 = [build_raw_hour(pn, snaps, h, 1.0, 1.0, 1.0) for h in idx]
    bad = find_bad_buses([(net, devs, dt, 0.0) for (net, devs, dt, _) in raw0])

    meta = {
        "network": os.path.basename(args.network),
        "n_snaps": args.n_snaps,
        "load_scale": 1.0,
        "threshold": "default",
        "panel_seed": 0,
        "panel_idx": idx,
        "n_bad": len(bad),
        "bad_buses": [int(b) for b in bad],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"# froze {len(bad)} bad buses -> {args.out}")
    print(f"# bad_buses = {bad}")


if __name__ == "__main__":
    main()
