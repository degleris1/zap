"""COST-LEVER figures: how cheaply each policy clears the data-center deliverability wall.

  fig_lever          -- cost frontier: deliverable firm-DC GW vs annualized $/yr per lever
  fig_lever_bar      -- one-glance: $/yr to fully clear the wall (or "doesn't")
  fig_colocated_opex -- co-located gas as the DC OPERATOR's PRIVATE cost (annualized CAPEX +
                        recurring FUEL OpEx, stacked), on a payer axis DISTINCT from the
                        grid/ratepayer transmission capex. Single claim: the recurring OpEx
                        DOMINATES the capex and is borne privately; clearing the wall with
                        on-site gas shifts a large ongoing private cost onto the operator.
                        The in-place-gen lever is DROPPED (existing generators are sunk).
                        Data: development/dc_colocated_opex.py (colocated_opex.json).

Deliverable firm-DC GW = must-serve feasible fraction x fleet GW. Headline fleet = 10 GW.

Two data sources, each used for what it measures cleanly (stated in the caption):
  * The conc-vs-distribute WALL is a Monte-Carlo-averaged property -> 24-pool full run
    (levers_full.json): concentrated 1.88 GW (feas 0.19) vs distributed 7.81 GW (0.78).
  * The GRID levers (transmission, in-place gen, JOINT gen+tx, CO-LOCATED gen) are traced
    in the same canonical full lever JSON so the wall and lever curves share the frozen
    25-bad cleaning protocol and 24-pool basis.

Corrected generation story (the point of this rebuild): scaling EXISTING/upstream
generation can't move the transmission-bound wall (flat at $1.36B/yr), and joint gen+tx
doesn't beat transmission alone -- but CO-LOCATED generation (new capacity AT the DC
buses, behind the bottleneck) is tested as a private, behind-the-meter wall-clearing
lever. Distribution remains the zero-grid-capital comparison.

Run: .venv/bin/python development/paper/fig_lever.py
"""
import json
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "placement_levers")
FULL = os.path.join(RES, "levers_full.json")   # canonical 24-pool wall + lever curves
# co-located gas OpEx reframe (DC-operator private cost split); produced by
# development/dc_colocated_opex.py.
OPEX = os.path.join(RES, "colocated_opex.json")
WORKLOAD = "inference"
B = 10.0  # fleet GW
FREE = 3e-3  # x-position for "$0/free" levers on a log axis


def curve(points):
    """(inv_usd $B/yr, deliverable GW) for a lever's list of sweep points."""
    x = np.array([p["inv_usd"] / 1e9 for p in points], float)
    y = np.array([p["feas"] * B for p in points], float)
    return np.maximum(x, FREE), y


def load(levers_path=FULL):
    fleet = json.load(open(levers_path))["workloads"][WORKLOAD]["fleets"]["10"]
    return fleet, fleet["match_distribution"]


def fig_lever(fleet, full):
    figstyle.setup()
    fig, ax = figstyle.new(w=4.2, h=3.0)
    C = figstyle.C

    colo_x, colo_y = curve(fleet["colocated_generation"])
    gen_x, gen_y = curve(fleet["generation"])
    jt_x, jt_y = curve(fleet["joint_gen_tx"])
    txu_x, txu_y = curve(fleet["transmission_uniform"])

    # CO-LOCATED generation -- the hero: climbs to the full fleet (wall cleared) cheaply
    ax.plot(colo_x, colo_y, "-s", color=C["ceiling"], ms=5, zorder=5,
            label="co-located generation")
    # transmission, uniform (reinforce every line): grid/ratepayer comparison
    ax.plot(txu_x, txu_y, "-o", color=C["cheap"], ms=4.5, zorder=3,
            label="transmission (uniform)")
    # in-place / upstream generation: flat -- can't move the transmission-bound wall
    ax.plot(gen_x, gen_y, "-^", color=C["ink"], ms=4.5, alpha=0.85, zorder=3,
            label="generation (in-place)")
    # joint gen+tx: tracks tx-alone (the gen half is wasted)
    ax.plot(jt_x, jt_y, "-D", color=C["uniform"], ms=4, alpha=0.85, zorder=3,
            label="gen + transmission")
    # distribution: free (24-pool validated), concentrated -> distributed
    conc24, dist24 = full["conc_feas"] * B, full["dist_feas"] * B
    ax.scatter([FREE], [dist24], marker="*", s=300, color=C["strong"], edgecolor="k",
               linewidth=0.7, zorder=6, label="distribution (24-pool frontier, $0)")
    ax.text(FREE * 1.25, dist24, "24-pool\nfrontier", ha="left", va="center",
            fontsize=7.2, color=C["strong"], fontweight="bold")

    ax.axhline(B, color=C["ink"], lw=0.8, ls=":", alpha=0.5, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Annualized grid investment ($B/yr)   [leftmost = $0]")
    ax.set_ylabel("Deliverable firm-DC (GW)")
    ax.set_ylim(0, B * 1.05)
    ax.legend(loc="center left", handlelength=1.6, borderpad=0.3, fontsize=7.5)
    path = figstyle.save(fig, "fig_lever")
    return path, (conc24, dist24, colo_x, colo_y, gen_x, gen_y, jt_x, jt_y, txu_x, txu_y)


def load_opex(opex_path=OPEX):
    """Load the canonical co-located-gas OpEx reframe (dc_colocated_opex.py)."""
    return json.load(open(opex_path)), opex_path


def fig_colocated_opex(opex):
    """SINGLE CLAIM: co-located gas is the DC OPERATOR's PRIVATE cost (annualized CAPEX +
    recurring FUEL OpEx, stacked), and the RECURRING OpEx dominates -- shown on a payer
    axis DISTINCT from the grid/ratepayer transmission capex (which buys only partial
    relief and is borne by ratepayers, never summed with the private cost). The in-place
    gen lever is dropped: existing generators are sunk."""
    figstyle.setup()
    fig, ax = figstyle.new(w=4.2, h=3.0)
    C = figstyle.C

    capex = opex["capex_usd_yr"] / 1e9
    opex_central = opex["opex_usd_yr_central"] / 1e9
    sens = opex["opex_usd_yr_sensitivity"]              # {"$30":.., "$35":.., "$50":..}
    opex_lo = sens["$30"] / 1e9
    opex_hi = sens["$50"] / 1e9
    tx_usd = (opex.get("transmission_grid_max_usd_yr") or 0.0) / 1e9
    tx_feas = opex.get("transmission_grid_max_feas")
    gas_gw = opex["gas_capacity_gw"]
    gen_twh = opex["annual_generation_twh"]

    # Two payer categories on the x-axis, kept VISUALLY SEPARATE (a gap + a divider),
    # never stacked into one total.
    x_priv, x_grid = 0.0, 1.0
    w = 0.5

    # --- DC-operator PRIVATE: stacked capex (base) + recurring fuel OpEx (top) ---
    ax.bar(x_priv, capex, width=w, color=C["accent"], zorder=3,
           label="on-site gas CAPEX (private)")
    ax.bar(x_priv, opex_central, width=w, bottom=capex, color=C["cheap"], zorder=3,
           label="on-site gas fuel OpEx (private, recurring)")
    # OpEx sensitivity band ($30-$50/MWh) drawn as an error whisker on the OpEx top.
    ax.errorbar(x_priv, capex + opex_central,
                yerr=[[opex_central - opex_lo], [opex_hi - opex_central]],
                fmt="none", ecolor=C["ink"], elinewidth=1.4, capsize=4, zorder=5)
    ax.text(x_priv + w / 2 + 0.04, capex / 2, f"capex ${capex:.2f}B", ha="left",
            va="center", fontsize=7.5, color=C["accent"], fontweight="bold")
    ax.text(x_priv, capex + opex_central / 2, f"fuel OpEx\n${opex_central:.2f}B",
            ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(x_priv, capex + opex_hi + 0.25,
            f"OpEx {opex_central / capex:.1f}x capex\n({gas_gw:.0f} GW gas, {gen_twh:.0f} TWh/yr)",
            ha="center", va="bottom", fontsize=7.5, color=C["cheap"], fontweight="bold")

    # --- GRID / RATEPAYER: transmission capex (separate payer) ---
    ax.bar(x_grid, tx_usd, width=w, color=C["strong"], zorder=3,
           label="transmission CAPEX (grid / ratepayer)")
    feas_note = f"partial relief\n(feas {tx_feas:.2f}, wall open)" if tx_feas else "partial relief"
    ax.text(x_grid, tx_usd + 0.25, f"${tx_usd:.1f}B\n{feas_note}", ha="center", va="bottom",
            fontsize=7.5, color=C["strong"], fontweight="bold")

    # payer divider so the two are never read as a single stacked total
    ax.axvline(0.5, color=C["ink"], lw=0.7, ls=":", alpha=0.5, zorder=1)
    ax.set_xticks([x_priv, x_grid])
    ax.set_xticklabels(["DC OPERATOR\n(private $)", "GRID\n(ratepayer $)"])
    ax.set_ylabel("Annualized cost ($B/yr)")
    ax.set_ylim(0, max(capex + opex_hi, tx_usd) * 1.42)
    ax.set_xlim(-0.7, 1.7)
    ax.legend(loc="upper left", handlelength=1.3, borderpad=0.3, fontsize=6.8,
              bbox_to_anchor=(-0.02, 1.0))
    path = figstyle.save(fig, "fig_colocated_opex")
    return path, dict(capex=capex, opex=opex_central, opex_lo=opex_lo, opex_hi=opex_hi,
                      tx=tx_usd, ratio=opex_central / capex if capex else float("nan"))


def fig_lever_bar(fleet):
    """$/yr to FULLY clear the wall (feas -> ~1.0) per lever; 'never' if it doesn't."""
    figstyle.setup()
    fig, ax = figstyle.new(w=4.2, h=3.0)
    C = figstyle.C

    def clear_point(points, target=0.99):
        hits = [p for p in points if p["feas"] >= target]
        if not hits:
            return None, None
        p = min(hits, key=lambda row: row["inv_usd"])
        return p["inv_usd"] / 1e9, p.get("g_co_gw")

    colo, colo_gw = clear_point(fleet["colocated_generation"])
    txu, _ = clear_point(fleet["transmission_uniform"])
    gen, _ = clear_point(fleet["generation"])
    jt, _ = clear_point(fleet["joint_gen_tx"])

    labels = ["distribution", "co-located\ngen", "transmission\n(uniform)",
              "gen\n(in-place)", "gen +\ntransmission"]
    vals = [0.0, colo, txu, gen, jt]
    colors = [C["strong"], C["ceiling"], C["cheap"], C["ink"], C["uniform"]]
    x = np.arange(len(labels))
    top = max([v for v in vals if v] + [colo or 1.0]) * 1.3

    for i, v in enumerate(vals):
        if v is None:
            ax.text(x[i], top * 0.04, "does not\nclear", ha="center", va="bottom",
                    fontsize=8, color=colors[i])
        else:
            ax.bar(x[i], v, color=colors[i], width=0.62, zorder=3)
            lbl = "$0" if v < 1e-6 else f"${v:.2f}B"
            ax.text(x[i], v + top * 0.02, lbl, ha="center", va="bottom",
                    fontsize=9, color=colors[i], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Annualized $/yr to clear the wall")
    ax.set_ylim(0, top)
    path = figstyle.save(fig, "fig_lever_bar")
    return path, dict(colo=colo, colo_gw=colo_gw, txu=txu, gen=gen, jt=jt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levers", default=FULL)
    ap.add_argument("--opex", default=OPEX)
    args = ap.parse_args()

    fleet, full = load(args.levers)
    p1, v = fig_lever(fleet, full)
    p2, cc = fig_lever_bar(fleet)
    print("fig_lever     ->", p1)
    print("fig_lever_bar ->", p2)
    try:
        opex, opex_path = load_opex(args.opex)
        p3, oc = fig_colocated_opex(opex)
        print("fig_colocated_opex ->", p3, f"(data: {opex_path})")
        print(f"  DC-operator PRIVATE: capex ${oc['capex']:.2f}B + recurring fuel OpEx "
              f"${oc['opex']:.2f}B/yr (band ${oc['opex_lo']:.2f}-${oc['opex_hi']:.2f}B @ "
              f"$30-$50/MWh); OpEx/capex {oc['ratio']:.1f}x (recurring fuel dominates, "
              f"borne privately).  GRID/ratepayer transmission ${oc['tx']:.1f}B/yr (separate "
              f"payer, partial relief). In-place gen DROPPED (existing gens sunk).")
    except Exception as e:
        print(f"(fig_colocated_opex skipped: {e})")
    conc24, dist24 = v[0], v[1]
    print(f"\nWORKLOAD={WORKLOAD}  fleet={B:.0f} GW")
    print(f"  WALL (24-pool): concentrated {conc24:.2f} GW (feas {full['conc_feas']:.2f}) "
          f"-> distributed {dist24:.2f} GW (feas {full['dist_feas']:.2f}) for $0 (distribution)")
    print(f"  CO-LOCATED gen: fully clears wall at ${cc['colo']:.2f}B/yr "
          f"(co-located {cc['colo_gw']:.0f} GW on-site)")
    print(f"  transmission(uniform): ${cc['txu']}  in-place gen: ${cc['gen']}  "
          f"joint gen+tx: ${cc['jt']}  (None = never clears in swept range)")


if __name__ == "__main__":
    main()
