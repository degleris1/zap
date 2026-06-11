"""
COST-ASSUMPTION SENSITIVITY (analytic; NO re-dispatch).

The deliverability result (what share of the day a fleet can be powered) does NOT depend on any cost
assumption -- costs only set the dollar figures. The grid expansion choices are made by congestion
(shadow prices / uniform), not by cost, so changing a cost is a pure linear rescale of the dollar
columns. This script recomputes the paper's dollar figures under a range of cost assumptions:

  - line build cost  x {0.5, 1.0, 1.5}   -> rescales "$ to match distribution" from the
                                            lever-frontier statistic (primary) and the
                                            Phase-B uniform-line crossover (secondary)
  - generator cost   x {0.5, 1.0, 1.5}   -> rescales generation-expansion $ (a minor lever)
  - DC build cost    in {9, 12, 15} M/MW -> rescales data-center investment (capex term; land ~ fixed)
  - land surface     x {0.5, 1.0, 2.0}   -> rescales the (small) DC land term; does NOT change
                                            deliverability (site selection uses RELATIVE land, so a
                                            uniform land scaling leaves the cheap-land ranking, hence
                                            the fleet, unchanged). Land's DELIVERABILITY effect is
                                            tested separately in dc_phaseB_finish.py Part C.

Reads: levers_full.json (frontier-match $, DC build $, generation $) +
phaseB_finish_<tag>.json (secondary uniform crossover $).
Writes: results/placement_robustness/cost_sensitivity_<tag>.json + a console table.

Usage:
  .venv/bin/python development/dc_cost_sensitivity.py --tag full
"""
import argparse, os, json

CRF = 0.07 * (1.07 ** 20) / (1.07 ** 20 - 1)       # ~0.09439 (annualize 20yr @ 7%)
BASE_DC_CAPEX = 12.0e6                               # $/MW baseline used in the runs


def annualize(x):
    return x * CRF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levers", default="development/results/placement_levers/levers_full.json")
    ap.add_argument("--phaseb", default="development/results/placement_robustness/phaseB_finish_full.json")
    ap.add_argument("--outdir", default="development/results/placement_robustness")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    lev = json.load(open(args.levers))
    pb = json.load(open(args.phaseb)) if os.path.exists(args.phaseb) else None

    out = {"line_cost_mult": [0.5, 1.0, 1.5], "gen_cost_mult": [0.5, 1.0, 1.5],
           "dc_capex_per_mw": [9e6, 12e6, 15e6], "land_mult": [0.5, 1.0, 2.0], "results": {}}

    for w in lev["workloads"]:
        out["results"][w] = {}
        for B, f in lev["workloads"][w]["fleets"].items():
            cell = {}
            Bgw = float(B)
            # ---- Primary "$ to match distribution" via lever-frontier uniform-line curve ----
            md = f.get("match_distribution", {})
            primary_match = md.get("txu_usd_to_match_feas")
            cell["frontier_match_usd_primary"] = {
                str(m): (primary_match * m if primary_match is not None else None)
                for m in out["line_cost_mult"]
            }
            cell["frontier_match_note"] = (
                None if primary_match is not None
                else "lever frontier never reaches distributed feasibility within swept range"
            )
            # ---- Secondary Phase-B seed-pinned uniform-line crossover ----
            xo = None
            if pb and "partA" in pb and B in pb["partA"] and w in pb["partA"][B]:
                xo = pb["partA"][B][w]["crossover_to_dist"]
            crossover = (xo["usd"] if (xo and xo.get("usd")) else None)
            cell["phaseb_crossover_usd_secondary"] = {
                str(m): (crossover * m if crossover is not None else None)
                for m in out["line_cost_mult"]
            }
            cell["phaseb_crossover_note"] = (
                None if crossover is not None
                else (xo.get("note") if xo else "no phaseB crossover available")
            )
            # ---- generation-expansion $ (max eps) rescaled by gen cost ----
            gmax = f["generation"][-1]["inv_usd"] if f.get("generation") else None
            cell["gen_expansion_usd"] = {str(m): (gmax * m if gmax is not None else None)
                                         for m in out["gen_cost_mult"]}
            # ---- DC investment: capex term scales with $/MW; land term ~ fixed (small) ----
            dc_build = f["dc_distribute"]["dc_build_usd_distribute"]   # annualized capex+land at $12M/MW
            land_term = dc_build - annualize(Bgw * 1e3 * BASE_DC_CAPEX)  # back out annualized land $
            cell["dc_invest_usd"] = {}
            for capex in out["dc_capex_per_mw"]:
                for lm in out["land_mult"]:
                    cell["dc_invest_usd"][f"capex{int(capex/1e6)}M_land{lm}"] = (
                        annualize(Bgw * 1e3 * capex) + land_term * lm)
            out["results"][w][B] = cell

    path = os.path.join(args.outdir, f"cost_sensitivity_{args.tag}.json")
    json.dump(out, open(path, "w"), indent=2, default=float)

    # console table (B=10 headline)
    print("# COST SENSITIVITY (analytic; deliverability unaffected by any cost)")
    for w in out["results"]:
        c = out["results"][w].get("10")
        if not c:
            continue
        fm = c["frontier_match_usd_primary"]
        xo = c["phaseb_crossover_usd_secondary"]
        print(f"\n## {w}  B=10")
        if fm["1.0"] is not None:
            print(f"  PRIMARY frontier-match $ via uniform line-build (line cost x0.5/1/1.5): "
                  f"${fm['0.5']/1e9:.1f}B / ${fm['1.0']/1e9:.1f}B / ${fm['1.5']/1e9:.1f}B")
        else:
            print(f"  PRIMARY frontier-match $: {c['frontier_match_note']}")
        if xo["1.0"] is not None:
            print(f"  SECONDARY Phase-B crossover $ (same multipliers): "
                  f"${xo['0.5']/1e9:.1f}B / ${xo['1.0']/1e9:.1f}B / ${xo['1.5']/1e9:.1f}B")
        else:
            print(f"  SECONDARY Phase-B crossover: {c['phaseb_crossover_note']}")
        dci = c["dc_invest_usd"]
        print(f"  DC investment (capex 9/12/15 M/MW, land x1): "
              f"${dci['capex9M_land1.0']/1e6:.0f}M / ${dci['capex12M_land1.0']/1e6:.0f}M / "
              f"${dci['capex15M_land1.0']/1e6:.0f}M per yr")
        gx = c["gen_expansion_usd"]
        if gx["1.0"] is not None:
            print(f"  generation-expansion $ (x0.5/1/1.5): ${gx['0.5']/1e6:.0f}M / "
                  f"${gx['1.0']/1e6:.0f}M / ${gx['1.5']/1e6:.0f}M")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
