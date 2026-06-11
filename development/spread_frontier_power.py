"""
Pilot-then-size power analysis for the DC spread-frontier slope test.

Reads a pilot spread_frontier_<tag>.json and, per workload, estimates how many
Monte-Carlo fleets the FULL run needs for 80% power to reject slope(k_min vs D) = 0
at alpha = 0.05.

Method: the bootstrap CI in the pilot gives the sampling SE of the slope at the pilot
fleet count n0 (SE0 ~ (hi-lo)/(2*1.96)). SE scales as 1/sqrt(n_fleets), so to make the
two-sided test reach 80% power we need slope / SE_full >= z_{0.975}+z_{0.80} ~ 2.802,
i.e.  n_full = n0 * (2.802 * SE0 / slope)^2.

Usage:
  .venv/bin/python development/spread_frontier_power.py \
      --in development/results/spread_frontier/spread_frontier_pilot.json
"""
import argparse
import json
import math

Z = 1.959963985  # z_{0.975}
ZP = 0.841621234  # z_{0.80}
NEED = Z + ZP     # ~2.8016


def size_one(st, n0):
    slope = st.get("slope")
    lo, hi = st.get("lo"), st.get("hi")
    if slope is None or lo is None or hi is None or not math.isfinite(slope) or slope == 0:
        return None
    se0 = (hi - lo) / (2.0 * Z)
    if se0 <= 0:
        return {"slope": slope, "se0": se0, "n0": n0, "n_full": n0,
                "note": "pilot CI already degenerate; n0 sufficient"}
    n_full = n0 * (NEED * se0 / slope) ** 2
    return {"slope": slope, "se0": se0, "n0": n0,
            "p_gt0_pilot": st.get("p_gt0"),
            "ci": [lo, hi],
            "n_full": math.ceil(n_full),
            "powered_at_pilot": bool(lo > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    args = ap.parse_args()
    d = json.load(open(args.infile))
    n0 = d["n_fleets"]
    print(f"pilot: n_fleets={n0}, n_snaps={d['n_snaps']}, n_clean={d['n_clean']}")
    rec = []
    for w, r in d["workloads"].items():
        s = size_one(r["slope_test"], n0)
        print(f"\n[{w}]")
        if s is None:
            print("  slope undefined -- cannot size (increase pilot fleets/targets)")
            continue
        print(f"  slope={s['slope']:.3f}  SE0={s['se0']:.3f}  CI={[round(x,3) for x in s['ci']]}"
              f"  P(>0)={s['p_gt0_pilot']:.3f}")
        print(f"  powered at pilot (CI>0)? {s['powered_at_pilot']}")
        print(f"  --> fleets for 80% power at full run: n_full ~ {s['n_full']}")
        rec.append(s["n_full"])
    if rec:
        print(f"\nRECOMMENDATION: run full study with n_fleets >= {max(rec)} "
              f"(max across workloads), n_snaps=12, node-stride=1.")


if __name__ == "__main__":
    main()
