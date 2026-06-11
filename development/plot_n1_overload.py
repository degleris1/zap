"""Plot the N-1 overload study (development/dc_n1_overload_full.py output).

Three panels per workload mirroring the base-case siting figures, now on the
differential N-1 corridor-overload metric:
  (A) added N-1 overload vs DC fleet B (aware / uniform / concentrated)
  (B) added N-1 overload vs spread (per-site cap kappa -> #sites)
  (C) added N-1 overload vs footprint k (grid-strength vs cheap-land vs random)

Usage: .venv/bin/python development/plot_n1_overload.py [results/n1_overload/n1_overload_full_pilot.json]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _xy(xs, ys):
    """Drop points whose y is None/non-finite (e.g. infeasible configs)."""
    out = [(x, y) for x, y in zip(xs, ys) if y is not None]
    return [p[0] for p in out], [p[1] for p in out]


def plot(path):
    data = json.load(open(path))
    wls = list(data["workloads"])
    fig, axes = plt.subplots(len(wls), 3, figsize=(15, 4.2 * len(wls)), squeeze=False)
    for r, w in enumerate(wls):
        d = data["workloads"][w]

        # A: budget sweep
        ax = axes[r][0]
        sw = d["budget_sweep"]
        B = [s["B"] for s in sw]
        for key, lab, mk in [("added_aware", "N-1-aware", "o-"),
                             ("added_uniform", "uniform", "s--"),
                             ("added_concentrated", "concentrated", "^:")]:
            ax.plot(*_xy(B, [s[key] for s in sw]), mk, label=lab)
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_xlabel("DC fleet B (GW)")
        ax.set_ylabel("added N-1 overload (GW-sum)")
        ax.set_title(f"[{w}] A. added contingency overload vs budget")
        ax.legend()
        ax.grid(alpha=0.3)

        # B: concentrate -> distribute
        ax = axes[r][1]
        sp = d["spread_sweep"]
        sx, sy = _xy([s["sites"] for s in sp], [s["added"] for s in sp])
        ax.plot(sx, sy, "o-")
        for s in sp:
            if s["added"] is not None:
                ax.annotate(f"cap {s['kappa']}", (s["sites"], s["added"]), fontsize=7,
                            textcoords="offset points", xytext=(4, 4))
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_xlabel("# sites used (← concentrate · distribute →)")
        ax.set_ylabel("added N-1 overload (GW-sum)")
        ax.set_title(f"[{w}] B. concentrate → distribute (B={data['budgets'][len(data['budgets'])//2]}…)")
        ax.grid(alpha=0.3)

        # C: siting orderings
        ax = axes[r][2]
        sit = d["siting_orderings"]
        ks = sorted(int(k) for k in sit)
        for name, lab, mk in [("grid_strength", "grid-strength", "o-"),
                              ("cheap_land", "cheap-land", "s--"),
                              ("random", "random", "^:")]:
            ys = [sit[str(k)][name] if str(k) in sit else sit[k][name] for k in ks]
            ax.plot(*_xy(ks, ys), mk, label=lab)
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_xlabel("footprint k (top-k of ordering)")
        ax.set_ylabel("added N-1 overload (GW-sum)")
        ax.set_title(f"[{w}] C. grid-strength vs cheap-land siting")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = path.replace(".json", ".png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "development/results/n1_overload/n1_overload_full_pilot.json"
    if not os.path.exists(p):
        raise SystemExit(f"not found: {p}")
    plot(p)
