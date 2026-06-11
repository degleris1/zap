"""
CO-LOCATED GAS as a DC-OPERATOR PRIVATE COST (capex + recurring fuel OpEx) -- a
focused reframe of the co-located-generation lever on the cleaned 490-node panel.

WHY THIS EXISTS (correcting the old lever framing):
  1. WHO PAYS. On-site gas at the DC buses is the DATA-CENTER OPERATOR's cost of
     ownership -- PRIVATE capital + fuel -- NOT grid/ratepayer transmission capex.
     The two must live on SEPARATE payer axes and are NEVER summed.
  2. OPEX, NOT JUST CAPEX. Existing grid generators are sunk ("there will always be
     generators"), so the in-place-gen CAPEX lever is moot and is DROPPED here. What
     matters for on-site gas is the RECURRING FUEL/OpEx burned to fill the DC's
     corridor import-deficit (the firm DC load the grid cannot deliver).
  3. THE DYNAMIC. On a corridor-limited grid the deficit is large and persistent, so
     the operator pays a large recurring fuel cost (>> the annualized capex). On-site
     gas "clears the wall" but shifts a big ongoing PRIVATE cost onto the operator.

WHAT THIS SCRIPT DOES (CONCENTRATED 10 GW placement = the wall case):
  - Builds the cleaned 490-node inference panel (the same cleaning as the lever study).
  - Places the CONCENTRATED 10 GW must-serve DC fleet (the ~1.88 GW grid-deliverable
    wall), then adds on-site gas at the DC buses sized to CLEAR the wall (make the
    must-serve DC feasible in every panel hour). The gas Generator has a cheap linear
    cost so it dispatches ahead of grid imports to offset the un-deliverable draw.
  - Reads the on-site gas's ACTUAL DISPATCHED generation per panel hour (the device's
    power output, GW), sums it, and ANNUALIZES by the panel's representativeness:
    mean power x 8760 h/yr -> annual generation (TWh/yr, MWh/yr).
  - Reports the two DC-OPERATOR private cost components:
        CAPEX = on-site MW x ONSITE_GEN_PER_MW_YR ($85k/MW-yr, annualized $900/kW @ CRF)
        OPEX  = annual gas generation (MWh) x GAS_MARGINAL_USD_PER_MWH
    plus an OpEx sensitivity at $30 and $50/MWh.
  - Contrasts against the grid/ratepayer cost of the (partial) transmission alternative
    from the lever data (transmission_uniform ~ $10B/yr for partial relief) -- but keeps
    them as SEPARATE payer categories (DC-operator PRIVATE vs grid RATEPAYER).

COST ASSUMPTIONS (stated):
  - On-site gas CAPEX: $900/kW overnight x CRF(20yr, 7%) = ~$85,000/MW-yr (same constant
    the lever study uses, ONSITE_GEN_PER_MW_YR).
  - On-site gas OpEx (marginal): heat rate ~7.5 MMBtu/MWh x gas ~$4/MMBtu (= $30/MWh fuel)
    + ~$5/MWh VOM = ~$35/MWh -> GAS_MARGINAL_USD_PER_MWH. Sensitivity at $30 / $50/MWh.
  - Transmission (grid/ratepayer): from the lever study's transmission_uniform curve,
    its largest spend (~$10B/yr) buys only PARTIAL feasibility relief (does NOT clear the
    wall). Reported as the grid-side comparison, separate payer.

SMOKE ONLY -- runs a tiny panel (n_snaps<=2, single concentrated placement) in <~2 min.
  .venv/bin/python development/dc_colocated_opex.py --smoke
Full (still single placement, fuller panel):
  .venv/bin/python development/dc_colocated_opex.py --n-snaps 12
"""
import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_placement_study import (  # noqa: E402
    load_dc_profile, sample_panel_indices, with_dc, metrics, find_bad_buses,
    load_land_weights, draw_fleet,
)
from dc_placement_integrated import build_raw_hour, WORKLOADS  # noqa: E402
from dc_placement_congestion import cleaned_panel, pack  # noqa: E402
from dc_placement_levers import (  # noqa: E402
    colocated_gen_device, ONSITE_GEN_PER_MW_YR, CRF,
)
from dc_cleaning import DEFAULT_BAD_BUSES_JSON, cleaning_metadata, resolve_bad_buses  # noqa: E402

# --------------------------------------------------------------------------- #
# DC-OPERATOR on-site gas OpEx (marginal $/MWh of dispatched generation):
#   heat rate ~7.5 MMBtu/MWh x gas ~$4/MMBtu  (= $30/MWh fuel)  +  ~$5/MWh VOM  ~= $35/MWh.
GAS_MARGINAL_USD_PER_MWH = 35.0
GAS_OPEX_SENSITIVITY = (30.0, 35.0, 50.0)         # $/MWh low / central / high
GAS_CARBON_TCO2_PER_MWH = 0.37                    # central CCGT emissions factor
CARBON_PRICE_USD_PER_TCO2 = 50.0
CARBON_PRICE_SENSITIVITY = (25.0, 50.0, 100.0)

# Device index of the on-site gas Generator in the final dispatched device list.
# Final list = [Gen, Load, DCLine, ACLine, Battery] + [on-site gas] + [DataCenterLoad],
# so on-site gas is at index 5 and the DC is the last device.
GAS_DEV_IDX = 5


# --------------------------------------------------------------------------- #
def dispatched_gas_gw(net, devices_no_gas, terms, caps, lf, g_co):
    """Place the must-serve DC fleet (terms/caps) AND g_co GW of on-site gas at those
    buses, dispatch ONE hour, and return (feasible, dispatched on-site gas GW summed
    over the DC buses). The gas Generator (cheap linear cost) dispatches ahead of grid
    imports to offset the DC's un-deliverable draw, so its output is the import-deficit
    it backfills. Returns (feasible, gas_gw, dc_draw_gw)."""
    d2 = list(deepcopy(devices_no_gas)) + [colocated_gen_device(devices_no_gas, terms, g_co)]
    devs = with_dc(d2, terms, caps, lf)
    import cvxpy as cp
    try:
        oc = net.dispatch(devs, time_horizon=1, solver=cp.CLARABEL, add_ground=False)
    except AssertionError:
        return False, np.nan, float(np.sum(caps)) * lf
    if oc is None:
        return False, np.nan, float(np.sum(caps)) * lf
    # on-site gas is a single-terminal Generator -> oc.power[GAS_DEV_IDX] == [array(nb, T)]
    gas_gw = float(np.sum(oc.power[GAS_DEV_IDX][0]))      # GW (POWER_UNIT internal)
    dc_draw_gw = float(np.sum(caps)) * lf                 # firm DC draw this hour (GW)
    return True, gas_gw, dc_draw_gw


def feasible_with_gas(panel, terms, caps, g_co):
    """Is the must-serve DC feasible in EVERY panel hour with g_co GW of on-site gas?"""
    for (net, devs, _, lf) in panel:
        ok, _, _ = dispatched_gas_gw(net, devs, terms, caps, lf, g_co)
        if not ok:
            return False
    return True


def smallest_gas_to_clear(panel, terms, caps, fleet_gw, grid=0.5):
    """Smallest on-site gas GW (on a `grid`-GW lattice) that makes the must-serve DC
    feasible in every panel hour. The deficit cannot exceed the fleet size, so the
    search tops out at fleet_gw. Returns the clearing GW (or fleet_gw if even that
    can't, which would flag a non-deficit infeasibility)."""
    g = 0.0
    while g < fleet_gw - 1e-9:
        if feasible_with_gas(panel, terms, caps, g):
            return g
        g = min(fleet_gw, g + grid)
    return fleet_gw


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=12)
    ap.add_argument("--n-candidates", type=int, default=24,
                    help="candidate hours sampled before keeping the most-congested n_snaps "
                         "(dispatch is ~0.3 s/hour, so a wide candidate net is cheap and ensures "
                         "the panel actually exercises the deliverability wall, not easy hours)")
    ap.add_argument("--load-scale", type=float, default=1.2)
    ap.add_argument("--fleet", type=float, default=10.0, help="concentrated DC fleet GW")
    ap.add_argument("--pool-seeds", type=int, default=4,
                    help="number of random concentrated-pool seeds to scan; the DEEPEST-wall "
                         "pool (most persistent import-deficit) is used as the canonical wall case")
    ap.add_argument("--gas-grid", type=float, default=0.5,
                    help="GW lattice for the smallest-gas-to-clear search")
    ap.add_argument("--land-cost",
                    default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--bad-buses-json", default=DEFAULT_BAD_BUSES_JSON,
                    help="frozen pathological-bus manifest for citable runs; pass '' to redetect")
    ap.add_argument("--levers-json", default="development/results/placement_levers/levers_full.json")
    ap.add_argument("--out", default="development/results/placement_levers/colocated_opex.json")
    ap.add_argument("--carbon-tco2-per-mwh", type=float, default=GAS_CARBON_TCO2_PER_MWH)
    ap.add_argument("--carbon-price", type=float, default=CARBON_PRICE_USD_PER_TCO2)
    ap.add_argument("--carbon-price-sensitivity", default="25,50,100",
                    help="comma-separated $/tCO2 sensitivity for the gas carbon adder")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny panel (n_snaps=2) + coarse gas lattice for a <2 min sanity run")
    args = ap.parse_args()

    if args.smoke:
        args.n_snaps = 2
        args.n_candidates = 12
        args.gas_grid = 1.0
    carbon_prices = [float(x) for x in args.carbon_price_sensitivity.split(",")]

    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    # Sample a wide candidate net, then keep the most-congested n_snaps (below) so even a
    # tiny smoke panel exercises the deliverability wall rather than easy hours.
    n_cand = max(args.n_candidates, args.n_snaps)
    cand_idx = sample_panel_indices(len(snaps), n_cand, None, None, 0)

    # ---- shared base build/clean/panel (the same cleaning the lever study uses) ----
    raw = [build_raw_hour(pn, snaps, h, args.load_scale, 1.0, 1.0) for h in cand_idx]
    n_nodes = raw[0][0].num_nodes
    bad_manifest = args.bad_buses_json or None
    detected_bad = None
    if bad_manifest is None:
        raw0 = [build_raw_hour(pn, snaps, h, 1.0, 1.0, 1.0) for h in cand_idx]
        detected_bad = find_bad_buses([(net, devs, dt, 0.0) for (net, devs, dt, hod) in raw0])
    bad = resolve_bad_buses(bad_manifest, detected_bad)
    clean_nodes = [i for i in range(n_nodes) if i not in set(bad)]
    land, weights = load_land_weights(args.land_cost, n_nodes)

    print(f"# CO-LOCATED GAS OPEX (DC-operator private cost)  "
          f"network={os.path.basename(args.network)}")
    print(f"# load x{args.load_scale}  bad buses {len(bad)}  clean nodes {len(clean_nodes)}  "
          f"n_snaps {args.n_snaps}{'  [SMOKE]' if args.smoke else ''}")
    print(f"# COST ASSUMPTIONS: on-site gas CAPEX ${ONSITE_GEN_PER_MW_YR:,.0f}/MW-yr "
          f"($900/kW overnight x CRF {CRF:.4f}); OpEx GAS_MARGINAL_USD_PER_MWH "
          f"= ${GAS_MARGINAL_USD_PER_MWH:.0f}/MWh (HR ~7.5 MMBtu/MWh x $4/MMBtu + $5 VOM); "
          f"OpEx sensitivity {GAS_OPEX_SENSITIVITY} $/MWh")

    # ---- CONCENTRATED 10 GW placement (the WALL case): fewest 1 GW sites on cheap land ----
    # Random cheap-land concentration produces a wide range of wall DEPTHS (some pools land
    # on a near-deliverable corridor, others deep behind it); the lever study's ~6 GW
    # headline is the Monte-Carlo average dominated by deep-wall pools. To represent that
    # canonical wall (not a lucky shallow pool), pick the concentrated pool, among
    # --pool-seeds candidates, with the DEEPEST persistent deficit (largest fraction of
    # candidate hours where the bare must-serve DC is infeasible). Stated, deterministic.
    B = args.fleet
    m_conc = int(np.ceil(B))                       # concentrate = fewest 1 GW sites
    inf_lf = load_dc_profile(WORKLOADS["inference"])
    cand_panel = cleaned_panel(raw, inf_lf, bad)

    def pool_wall_depth(seed):
        rng = np.random.default_rng(seed)
        pl = draw_fleet(rng, clean_nodes, {n: weights[n] for n in clean_nodes},
                        B, B / 40.0)[0]
        tm, cp_ = pack(pl, B, m_conc)
        infeas = 0
        for (net, devs, _, lf) in cand_panel:
            m = metrics(net, with_dc(devs, tm, cp_, lf), devs, 1,
                        dc_terminals=tm, price_nodes=clean_nodes)
            infeas += int(not m["feasible"])
        return infeas, tm, cp_

    seeds = list(range(args.pool_seeds))
    depths = [(pool_wall_depth(s), s) for s in seeds]
    (best_infeas, terms, caps), best_seed = max(depths, key=lambda t: t[0][0])
    print(f"# pool wall-depth scan (infeasible candidate hours / {len(cand_panel)}): "
          f"{ {s: d[0] for (d, s) in depths} } -> picked deepest-wall seed {best_seed} "
          f"({best_infeas}/{len(cand_panel)} infeasible)")

    # ---- keep the most-congested n_snaps for this concentrated fleet (wall hours first) ----
    # Rank candidate hours by bare-fleet must-serve INFEASIBILITY (the wall), then by
    # n_binding lines (most congested) so even a 2-hour smoke panel hits the deficit.
    cand_feas, cand_score = [], []
    for (net, devs, _, lf) in cand_panel:
        m = metrics(net, with_dc(devs, terms, caps, lf), devs, 1,
                    dc_terminals=terms, price_nodes=clean_nodes)
        cand_feas.append(bool(m["feasible"]))
        # infeasible -> most congested (rank first); else rank by binding lines
        cand_score.append((-(0 if m["feasible"] else 1), -int(m["n_binding"])))
    order = sorted(range(len(cand_panel)), key=lambda i: cand_score[i])
    keep = sorted(order[:args.n_snaps])
    panel = [cand_panel[i] for i in keep]
    bare_feas = [cand_feas[i] for i in keep]
    print(f"# CONCENTRATED fleet B={B:.0f} GW at {len(terms)} buses {list(map(int, terms))}")
    print(f"# kept {len(panel)}/{len(cand_panel)} most-congested candidate hours "
          f"(idx {[int(cand_idx[i]) for i in keep]}); bare-fleet must-serve feasible in "
          f"{int(np.sum(bare_feas))}/{len(panel)} kept hours "
          f"({int(np.sum(cand_feas))}/{len(cand_panel)} across all candidates)")

    # ---- size on-site gas to CLEAR the wall (feasible in every panel hour) ----
    g_clear = smallest_gas_to_clear(panel, terms, caps, B, grid=args.gas_grid)
    print(f"# on-site gas to clear the wall: {g_clear:.1f} GW (search lattice {args.gas_grid} GW)")

    # ---- read DISPATCHED on-site gas per hour at the clearing capacity ----
    gas_gw_hours, dc_draw_hours, feas_hours = [], [], []
    for (net, devs, _, lf) in panel:
        ok, gas_gw, dc_draw = dispatched_gas_gw(net, devs, terms, caps, lf, g_clear)
        feas_hours.append(ok)
        gas_gw_hours.append(gas_gw if ok else np.nan)
        dc_draw_hours.append(dc_draw)
    gas_gw_hours = np.asarray(gas_gw_hours, float)
    dc_draw_hours = np.asarray(dc_draw_hours, float)
    all_feas = bool(np.all(feas_hours))

    mean_gas_gw = float(np.nanmean(gas_gw_hours))           # GW (mean power across panel)
    mean_dc_gw = float(np.mean(dc_draw_hours))
    # annualize by the panel's representativeness: mean power x 8760 h/yr.
    annual_gen_mwh = mean_gas_gw * 1.0e3 * 8760.0           # GW->MW (x1000) x 8760 h
    annual_gen_twh = annual_gen_mwh / 1.0e6                 # MWh -> TWh
    cap_factor = (mean_gas_gw / g_clear) if g_clear > 0 else float("nan")

    # ---- DC-operator PRIVATE cost components (NEVER summed with grid $) ----
    onsite_mw = g_clear * 1.0e3
    capex_usd_yr = onsite_mw * ONSITE_GEN_PER_MW_YR
    opex_usd_yr = {f"${c:.0f}": annual_gen_mwh * c for c in GAS_OPEX_SENSITIVITY}
    opex_central = annual_gen_mwh * GAS_MARGINAL_USD_PER_MWH
    carbon_tco2_yr = annual_gen_mwh * args.carbon_tco2_per_mwh
    carbon_usd_yr = carbon_tco2_yr * args.carbon_price
    carbon_usd_yr_sensitivity = {f"${c:.0f}": carbon_tco2_yr * c for c in carbon_prices}
    fuel_plus_carbon_opex_usd_yr = opex_central + carbon_usd_yr

    # ---- grid/ratepayer transmission comparison (SEPARATE payer, never summed) ----
    txu_max_usd, txu_max_feas = None, None
    try:
        lev = json.load(open(args.levers_json))["workloads"]["inference"]["fleets"][str(int(B))]
        txu = lev["transmission_uniform"]
        last = max(txu, key=lambda r: r["inv_usd"])
        txu_max_usd, txu_max_feas = float(last["inv_usd"]), float(last["feas"])
    except Exception as e:
        print(f"# (transmission comparison unavailable: {e})")

    # ---- report ----
    print("\n=== DC-OPERATOR PRIVATE COST (on-site gas) ===")
    print(f"  on-site gas CAPACITY      : {g_clear:.2f} GW ({onsite_mw:,.0f} MW)  "
          f"[~ the un-deliverable DC; fleet {B:.0f} GW, mean DC draw {mean_dc_gw:.2f} GW]")
    print(f"  on-site gas mean dispatch : {mean_gas_gw:.2f} GW  (capacity factor "
          f"{cap_factor:.2f}  -> runs near-continuously to backfill the deficit)")
    print(f"  annual generation         : {annual_gen_twh:.2f} TWh/yr  "
          f"({annual_gen_mwh:,.0f} MWh/yr = mean power x 8760)")
    print(f"  CAPEX (private)           : ${capex_usd_yr/1e9:.3f} B/yr  "
          f"(= {onsite_mw:,.0f} MW x ${ONSITE_GEN_PER_MW_YR:,.0f}/MW-yr)")
    print(f"  OPEX  (private, recurring): ${opex_central/1e9:.3f} B/yr @ "
          f"${GAS_MARGINAL_USD_PER_MWH:.0f}/MWh central")
    for k, v in opex_usd_yr.items():
        print(f"          OpEx @ {k}/MWh    : ${v/1e9:.3f} B/yr")
    print(f"  CARBON adder              : ${carbon_usd_yr/1e9:.3f} B/yr @ "
          f"{args.carbon_tco2_per_mwh:.2f} tCO2/MWh and ${args.carbon_price:.0f}/tCO2")
    for k, v in carbon_usd_yr_sensitivity.items():
        print(f"          carbon @ {k}/tCO2 : ${v/1e9:.3f} B/yr")
    print(f"  fuel + carbon OpEx        : ${fuel_plus_carbon_opex_usd_yr/1e9:.3f} B/yr")
    ratio = opex_central / capex_usd_yr if capex_usd_yr else float("nan")
    print(f"  OPEX / CAPEX              : {ratio:.2f}x  "
          f"({'OPEX DOMINATES (recurring fuel >> annualized capex)' if ratio > 1 else 'capex dominates'})")

    print("\n=== GRID / RATEPAYER comparison (SEPARATE payer -- never summed) ===")
    if txu_max_usd is not None:
        print(f"  transmission (uniform), grid/ratepayer capex: ${txu_max_usd/1e9:.2f} B/yr "
              f"for only PARTIAL relief (feas -> {txu_max_feas:.2f}, wall NOT cleared)")
        print("  -> on-site gas is the DC OPERATOR's PRIVATE cost; transmission is the GRID's. "
              "They are different payers and are reported on different axes.")

    print("\n=== SMOKE SANITY ===")
    s_deficit = abs(g_clear - (B - mean_dc_gw)) < 0.5 * B or g_clear >= 0.5 * B
    s_opex_dom = ratio > 1.0
    s_feas = all_feas
    print(f"  all panel hours feasible with gas : {s_feas}")
    print(f"  gas capacity ~ the deficit         : {s_deficit}  "
          f"(gas {g_clear:.1f} GW vs fleet {B:.0f} GW, mean draw {mean_dc_gw:.2f} GW)")
    print(f"  OpEx dominates CAPEX               : {s_opex_dom}  (ratio {ratio:.2f}x)")
    print(f"  SANITY HELD                        : {bool(s_feas and s_deficit and s_opex_dom)}")

    out = {
        "network": os.path.basename(args.network), "load_scale": args.load_scale,
        **cleaning_metadata(bad_manifest, bad),
        "smoke": bool(args.smoke), "n_snaps": args.n_snaps, "fleet_gw": B,
        "concentrated_buses": list(map(int, terms)),
        "bare_feasible_hours": int(np.sum(bare_feas)), "panel_hours": len(panel),
        "gas_capacity_gw": g_clear, "gas_capacity_mw": onsite_mw,
        "gas_mean_dispatch_gw": mean_gas_gw, "gas_capacity_factor": cap_factor,
        "mean_dc_draw_gw": mean_dc_gw,
        "annual_generation_twh": annual_gen_twh, "annual_generation_mwh": annual_gen_mwh,
        "onsite_gen_per_mw_yr": ONSITE_GEN_PER_MW_YR, "crf": CRF,
        "gas_marginal_usd_per_mwh": GAS_MARGINAL_USD_PER_MWH,
        "gas_opex_sensitivity": list(GAS_OPEX_SENSITIVITY),
        "capex_usd_yr": capex_usd_yr, "opex_usd_yr_central": opex_central,
        "opex_usd_yr_sensitivity": opex_usd_yr, "opex_over_capex": ratio,
        "carbon_tco2_per_mwh": args.carbon_tco2_per_mwh,
        "carbon_price_usd_per_tco2": args.carbon_price,
        "carbon_tco2_yr": carbon_tco2_yr,
        "carbon_usd_yr": carbon_usd_yr,
        "carbon_price_sensitivity": carbon_prices,
        "carbon_usd_yr_sensitivity": carbon_usd_yr_sensitivity,
        "fuel_plus_carbon_opex_usd_yr": fuel_plus_carbon_opex_usd_yr,
        "transmission_grid_max_usd_yr": txu_max_usd, "transmission_grid_max_feas": txu_max_feas,
        "sanity_all_feasible": s_feas, "sanity_gas_is_deficit": bool(s_deficit),
        "sanity_opex_dominates": s_opex_dom,
        "sanity_held": bool(s_feas and s_deficit and s_opex_dom),
    }
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
