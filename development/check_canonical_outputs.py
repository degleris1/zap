"""Acceptance checks for citable DC-placement rerun artifacts."""

import argparse
import json
import math
import os

from dc_cleaning import DEFAULT_BAD_BUSES_JSON, load_bad_buses


def load_json(path):
    with open(path) as f:
        return json.load(f)


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def same_path(left, right):
    return os.path.abspath(left) == os.path.abspath(right)


def check_cleaning(name, data, manifest, manifest_path):
    require(data.get("n_bad") == len(manifest), f"{name}: n_bad != {len(manifest)}")
    require(data.get("bad_buses") == manifest, f"{name}: bad_buses != manifest")
    require(data.get("bad_buses_json"), f"{name}: missing bad_buses_json")
    require(same_path(data["bad_buses_json"], manifest_path),
            f"{name}: bad_buses_json != {manifest_path}")


def check_levers(path, manifest, manifest_path, expected_n_snaps=12, require_flex=False,
                 flex_tol=1e-6):
    data = load_json(path)
    require(data.get("_partial") is False, f"{path}: lever rerun is partial")
    check_cleaning(path, data, manifest, manifest_path)
    require(data.get("n_snaps") == expected_n_snaps, f"{path}: n_snaps != {expected_n_snaps}")
    require(data.get("n_pools") == 24, f"{path}: n_pools != 24")
    require("crf" in data, f"{path}: missing CRF")
    require("onsite_gen_per_mw_yr" in data, f"{path}: missing onsite gen capex assumption")
    for workload in ("inference", "training"):
        fleet = data["workloads"][workload]["fleets"]["10"]
        dist = fleet["dc_distribute"]
        conc_feas = float(dist["feas_concentrate"])
        dist_feas = float(dist["feas_distribute"])
        require(math.isfinite(conc_feas), f"{path}: B10 {workload} concentrate feasibility non-finite")
        require(math.isfinite(dist_feas), f"{path}: B10 {workload} distribute feasibility non-finite")
        colo = fleet.get("colocated_generation", [])
        require(colo, f"{path}: missing colocated_generation for B10 {workload}")
        require(fleet.get("transmission_uniform"), f"{path}: missing uniform tx curve")
        require(fleet.get("transmission"), f"{path}: missing targeted tx curve")
        if require_flex:
            flex = fleet.get("flexible_dc")
            require(flex, f"{path}: missing flexible_dc for B10 {workload}")
            require("dc_flex_voll_internal" in data, f"{path}: missing flex VOLL")
            zero = [row for row in flex if abs(float(row["flex_frac"])) <= 1e-12]
            require(len(zero) == 1, f"{path}: missing exactly one flex_frac=0 row for {workload}")
            delta = abs(float(zero[0]["firm_feas"]) - float(dist["feas_concentrate"]))
            require(delta <= flex_tol,
                    f"{path}: flex 0% feasibility differs from concentrate by {delta}")
            for row in flex:
                for key in ("firm_feas", "served_flex_frac", "served_flex_gw",
                            "requested_flex_gw", "base_shed_delta_pct_mean"):
                    val = row[key][0] if isinstance(row[key], list) else row[key]
                    require(math.isfinite(float(val)), f"{path}: non-finite flexible_dc {key}")
    return data


def check_phaseb(path, manifest, manifest_path, expected_n_snaps=12,
                 expected_fleets=("6", "10"), require_panel_metadata=False,
                 require_intervals=False):
    data = load_json(path)
    require(data.get("_partial") is False, f"{path}: Phase-B rerun is partial")
    check_cleaning(path, data, manifest, manifest_path)
    require(data.get("n_snaps") == expected_n_snaps, f"{path}: n_snaps != {expected_n_snaps}")
    require(data.get("n_seeds") == 8, f"{path}: n_seeds != 8")
    require(data.get("n_pools") == 24, f"{path}: n_pools != 24")
    if require_panel_metadata:
        require(data.get("panel_selection"), f"{path}: missing panel_selection")
        require(len(data.get("panel_indices", [])) == expected_n_snaps,
                f"{path}: panel_indices length != {expected_n_snaps}")
        require(len(data.get("panel_hours", [])) == expected_n_snaps,
                f"{path}: panel_hours length != {expected_n_snaps}")
    part_a = data.get("partA", {})
    for B in expected_fleets:
        require(B in part_a, f"{path}: missing Part A fleet B={B}")
        for workload in ("inference", "training"):
            wall = part_a[B][workload]["wall"]
            for kind in ("conc", "dist"):
                stat = wall[kind]
                require(math.isfinite(float(stat["mean"])),
                        f"{path}: B{B} {workload} {kind} mean non-finite")
                if require_intervals:
                    require("seed_range" in stat,
                            f"{path}: B{B} {workload} {kind} missing seed_range")
                    require("bootstrap_ci_seed_pool_mean" in stat,
                            f"{path}: B{B} {workload} {kind} missing bootstrap CI")
            require("separated" in wall, f"{path}: B{B} {workload} missing separated field")
    for workload in ("inference", "training"):
        crossover = part_a["10"][workload].get("crossover_to_dist")
        require(crossover is not None, f"{path}: B10 {workload} missing tx crossover")
    return data


def check_opex(path, manifest, manifest_path, expected_n_snaps=12, require_carbon=False):
    data = load_json(path)
    check_cleaning(path, data, manifest, manifest_path)
    require(data.get("n_snaps") == expected_n_snaps, f"{path}: n_snaps != {expected_n_snaps}")
    for key in ("crf", "capex_usd_yr", "opex_usd_yr_central", "opex_over_capex"):
        require(key in data, f"{path}: missing {key}")
    require(math.isfinite(float(data["opex_over_capex"])), f"{path}: non-finite opex_over_capex")
    if require_carbon:
        for key in ("carbon_tco2_per_mwh", "carbon_price_usd_per_tco2",
                    "carbon_tco2_yr", "carbon_usd_yr", "fuel_plus_carbon_opex_usd_yr"):
            require(key in data, f"{path}: missing {key}")
        expected = (float(data["annual_generation_mwh"])
                    * float(data["carbon_tco2_per_mwh"])
                    * float(data["carbon_price_usd_per_tco2"]))
        require(abs(float(data["carbon_usd_yr"]) - expected) <= max(1e-6, abs(expected) * 1e-12),
                f"{path}: carbon_usd_yr does not match generation * rate * price")
    return data


def check_deliverable(path, manifest, manifest_path):
    data = load_json(path)
    check_cleaning(path, data, manifest, manifest_path)
    require(data.get("n_snaps") == 12, f"{path}: n_snaps != 12")
    require(data.get("n1_citable") is False, f"{path}: N-1 rows not marked non-citable")
    for workload in data["workloads"]:
        rows = data["workloads"][workload]["rows"]
        require(rows, f"{path}: no deliverable rows for {workload}")
        for row in rows:
            status = row["lp_n1"]["status"]
            require(status == "skipped_non_citable" or row["lp_n1"]["gw"] != row["lp_n1"]["gw"],
                    f"{path}: N-1 row for {workload} is not quarantined")
    return data


def check_cost_sensitivity(path):
    data = load_json(path)
    for workload, fleets in data.get("results", {}).items():
        for B, cell in fleets.items():
            require("frontier_match_usd_primary" in cell,
                    f"{path}: {workload} B{B} missing primary frontier-match cost")
            require("phaseb_crossover_usd_secondary" in cell,
                    f"{path}: {workload} B{B} missing secondary crossover cost")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_BAD_BUSES_JSON)
    ap.add_argument("--levers", default="development/results/placement_levers/levers_full.json")
    ap.add_argument("--phaseb",
                    default="development/results/placement_robustness/phaseB_finish_partA_25bad.json")
    ap.add_argument("--opex", default="development/results/placement_levers/colocated_opex.json")
    ap.add_argument("--deliverable",
                    default="development/results/deliverable_frontier/deliverable_frontier_full.json")
    ap.add_argument("--check-24h", action="store_true")
    ap.add_argument("--levers24", default="development/results/placement_levers/levers_24h.json")
    ap.add_argument("--phaseb24",
                    default="development/results/placement_robustness/phaseB_finish_partA_25bad_24h.json")
    ap.add_argument("--opex24", default="development/results/placement_levers/colocated_opex_24h.json")
    ap.add_argument("--cost24",
                    default="development/results/placement_robustness/cost_sensitivity_24h.json")
    args = ap.parse_args()

    manifest = load_bad_buses(args.manifest)
    require(len(manifest) == 25, "manifest must contain 25 bad buses")
    for path in (args.levers, args.phaseb, args.opex, args.deliverable):
        require(os.path.exists(path), f"missing artifact: {path}")

    check_levers(args.levers, manifest, args.manifest)
    check_phaseb(args.phaseb, manifest, args.manifest)
    check_opex(args.opex, manifest, args.manifest)
    check_deliverable(args.deliverable, manifest, args.manifest)
    if args.check_24h:
        for path in (args.levers24, args.phaseb24, args.opex24, args.cost24):
            require(os.path.exists(path), f"missing 24h artifact: {path}")
        check_phaseb(args.phaseb24, manifest, args.manifest, expected_n_snaps=24,
                     expected_fleets=("3", "6", "10"), require_panel_metadata=True,
                     require_intervals=True)
        check_levers(args.levers24, manifest, args.manifest, expected_n_snaps=24,
                     require_flex=True)
        check_opex(args.opex24, manifest, args.manifest, expected_n_snaps=24,
                   require_carbon=True)
        check_cost_sensitivity(args.cost24)
    print("canonical acceptance checks passed")


if __name__ == "__main__":
    main()
