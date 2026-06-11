"""C5 ROBUSTNESS figure: the deliverable-GW siting thesis survives under full N-1.

Single claim, no title: under full single-line-outage (N-1) contingencies, grid-strength
siting RELIEVES post-contingency corridor overload while naive (cheap-land / random / uniform)
siting AGGRAVATES it -- the same ordering as the base-case deliverability story, now under a
stricter operating standard. (On this corridor-limited grid N-1 is a robustness check, not a
separate result: it confirms, it does not reorder.)

Metric: added N-1 overload = O(placement) - O(no-DC), where O = sum over outages x lines x
hours of (|f + LODF[:,k] f_k| - Fbar)_+  (GW). Negative = the DC placement RELIEVES contingency
overload; positive = it aggravates. Exact LODF, all non-radial outages, single representative
stressed hour, reliable ~20-candidate config (HiGHS is reliable at this size).

Data: development/results/n1_overload/n1_overload_full_c5.json
Run:  .venv/bin/python development/paper/fig_n1.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                    "n1_overload", "n1_overload_full_c5.json")
WORKLOAD = "inference"


def main():
    d = json.load(open(DATA))
    wd = d["workloads"][WORKLOAD]
    sw = sorted(wd["budget_sweep"], key=lambda s: s["B"])
    B = [s["B"] for s in sw]

    figstyle.setup()
    fig, ax = figstyle.new(w=3.8, h=2.8)
    C = figstyle.C
    ax.axhline(0, color=C["ink"], lw=0.8, alpha=0.6, zorder=1)
    ax.plot(B, [s["added_uniform"] for s in sw], "--s", color=C["cheap"], zorder=3,
            label="uniform (naive)")
    ax.plot(B, [s["added_aware"] for s in sw], "-o", color=C["strong"], zorder=4,
            label="grid-strength (aware)")
    ax.plot(B, [s["added_concentrated"] for s in sw], ":^", color=C["random"], zorder=2,
            label="concentrated (on strong bus)")

    ax.set_xlabel("DC fleet (GW)")
    ax.set_ylabel("Added N-1 overload (GW)")
    ax.text(B[-1], 0.04, "aggravates ↑", fontsize=7.5, color=C["ink"], alpha=0.7,
            ha="right", va="bottom")
    ax.text(B[-1], -0.04, "relieves ↓", fontsize=7.5, color=C["ink"], alpha=0.7,
            ha="right", va="top")
    ax.legend(loc="upper left", handlelength=1.8, fontsize=8)
    path = figstyle.save(fig, "fig_n1_robustness")
    print("fig_n1_robustness ->", path)
    print(f"WORKLOAD={WORKLOAD}  no-DC N-1 overload={wd['overload_no_dc']:.2f} GW")
    for s in sw:
        print(f"  B={s['B']}: aware={s['added_aware']}  uniform={s['added_uniform']}  "
              f"concentrated={s['added_concentrated']}")


if __name__ == "__main__":
    main()
