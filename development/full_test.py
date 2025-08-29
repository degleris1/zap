import os
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import pypsa

import zap
from zap.geo_land import (
    build_bus_to_county,
    build_county_land_cost_from_points,
    compute_capital_cost_per_mw_from_county,
    load_county_land_cost,
)
from zap.importers.pypsa import load_pypsa_network, parse_buses

HOME_PATH = os.environ.get("HOME")
PYPSA_NETW0RK_PATH = (
    HOME_PATH + "/zap_data/pypsa-networks/western_small/network_2021.nc"
)
# CONFIG: set these to your local data locations
PYPSA_CSV_DIR = Path(os.environ.get("ZAP_PYPSA_CSV_DIR", PYPSA_NETW0RK_PATH)).resolve()
BUS_TO_COUNTY_CSV = Path(
    os.environ.get("ZAP_BUS_TO_COUNTY", "/abs/path/buses_locs_info.csv")
).resolve()
COUNTY_LAND_CSV = Path(
    os.environ.get("ZAP_COUNTY_LAND_COST", "/abs/path/county_land_cost.csv")
).resolve()
# Optional: a point-level land file to derive county medians if COUNTY_LAND_CSV missing
POINT_LAND_SRC = Path(
    os.environ.get("ZAP_POINT_LAND_SRC", "/abs/path/land_lookup_details.csv")
).resolve()

# DC cost params
ACRES_PER_MW = float(os.environ.get("ZAP_ACRES_PER_MW", "0.20"))
GAMMA_PER_MW = float(os.environ.get("ZAP_GAMMA_PER_MW", "7500000"))

# Time settings
TIME_HORIZON_HOURS = 24
TIME_RES_HOURS = 1.0


def select_top10_wecc_buses(pn: pypsa.Network) -> list[str]:
    # Score based on average load and generator nameplate at the bus (mirrors marimo logic)
    load_ts = pn.loads_t.p_set
    avg_load = load_ts.groupby(axis=1, level=0).mean().sum()  # MW per bus
    gen_cap = pn.generators.groupby("bus")["p_nom"].sum()  # MW per bus

    def normalise(s):
        s = s.copy()
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s

    score = 0.6 * normalise(avg_load) + 0.4 * normalise(gen_cap).reindex_like(
        avg_load
    ).fillna(0.0)
    top10 = score.nlargest(10).index.tolist()
    # Clean names like "<bus> AC" suffix if present
    top10 = [name.replace(" AC", "") for name in top10]
    return top10


def main():
    # 1) Load PyPSA network from CSVs
    assert PYPSA_CSV_DIR.exists(), f"Missing PYPSA csv folder: {PYPSA_CSV_DIR}"
    pn = pypsa.Network()
    pn.import_from_csv_folder(PYPSA_CSV_DIR)

    # 2) Pick top-10 WECC buses by score
    chosen_bus_names = select_top10_wecc_buses(pn)
    print("Chosen buses:", chosen_bus_names)

    # 3) Build persistent bus->county mapping if not present
    if not BUS_TO_COUNTY_CSV.exists():
        print(f"Building bus->county map at {BUS_TO_COUNTY_CSV} ...")
        build_bus_to_county(pn, BUS_TO_COUNTY_CSV)

    # 4) Load or derive county land cost
    try:
        land_df = load_county_land_cost([COUNTY_LAND_CSV])
        print(f"Loaded county land cost from {COUNTY_LAND_CSV}")
    except FileNotFoundError:
        # Try to derive from a point-level file
        assert (
            POINT_LAND_SRC.exists()
        ), f"Missing county land file {COUNTY_LAND_CSV} and no point-level source at {POINT_LAND_SRC}."
        print(f"Deriving county land cost from {POINT_LAND_SRC} -> {COUNTY_LAND_CSV}")
        build_county_land_cost_from_points(POINT_LAND_SRC, COUNTY_LAND_CSV)
        land_df = load_county_land_cost([COUNTY_LAND_CSV])

    # 5) Compute per-bus capital cost via county FIPS
    capital_per_mw = compute_capital_cost_per_mw_from_county(
        bus_ids=chosen_bus_names,
        bus_to_county_csv=BUS_TO_COUNTY_CSV,
        county_land_cost_csv=COUNTY_LAND_CSV,
        acres_per_mw=ACRES_PER_MW,
        gamma_per_mw=GAMMA_PER_MW,
    )
    print("Capital cost per MW ($):", capital_per_mw)

    # 6) Convert PyPSA network to zap network and devices
    dates = pd.date_range(
        "2019-01-01",
        periods=int(TIME_HORIZON_HOURS / TIME_RES_HOURS),
        freq=f"{int(TIME_RES_HOURS)}H",
    )
    net, devices = load_pypsa_network(
        pn,
        dates,
        power_unit=1.0,  # MW
        cost_unit=1.0,  # $
        scale_load=1.0,
        drop_empty_generators=True,
        expand_empty_generators=0.0,
        scale_generator_capacity_factor=1.0,
        scale_line_capacity_factor=1.0,
        marginal_load_value=1000.0,
    )

    # 7) Create DataCenterLoad for the chosen buses (AI inference)
    # Map chosen bus names to zap node indices
    buses, buses_to_index = parse_buses(pn)
    terminals = np.array([buses_to_index[b] for b in chosen_bus_names], dtype=int)

    n_dc = len(chosen_bus_names)
    # Start with zero capacity; planning will allocate a budget across nodes
    nominal_capacity = np.zeros(n_dc)  # MW
    # Simple per-node linear cost for operation (e.g., electricity price weight)
    linear_cost = 100.0 * np.ones(n_dc)

    dcloads = zap.DataCenterLoad(
        num_nodes=net.num_nodes,
        terminal=terminals,
        nominal_capacity=nominal_capacity,
        profile_types=[zap.DataCenterLoad.ProfileType.AI_INFER] * n_dc,
        time_resolution_hours=TIME_RES_HOURS,
        settime_horizon=TIME_HORIZON_HOURS,
        linear_cost=linear_cost,
        pue=1.25,
    )
    # Assign capital_cost computed from location (column vector)
    dcloads.capital_cost = np.array(capital_per_mw).reshape((n_dc, 1))

    devices = [dcloads] + devices  # Data centers at index 0 for planning reference

    # 8) Build and solve a small planning problem with a total budget
    TOTAL_DC_BUDGET = 1000.0  # MW
    xstar = zap.DispatchLayer(
        net,
        devices,
        parameter_names={"dc_capacity": (0, "nominal_capacity")},  # device 0 is dcloads
        time_horizon=int(TIME_HORIZON_HOURS / TIME_RES_HOURS),
        solver=cp.CLARABEL,
    )

    lower_bounds = {"dc_capacity": np.zeros(n_dc)}
    upper_bounds = {"dc_capacity": np.full(n_dc, 300.0)}
    eta = {"dc_capacity": nominal_capacity.copy()}

    op_obj = zap.planning.DispatchCostObjective(net, devices)
    inv_obj = zap.planning.InvestmentObjective(devices, xstar)
    P = zap.planning.PlanningProblem(
        operation_objective=op_obj,
        investment_objective=inv_obj,
        layer=xstar,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    P.extra_projections = {
        "dc_capacity": zap.planning.SimplexBudgetProjection(
            budget=TOTAL_DC_BUDGET, strict=True
        )
    }

    print("Solving planning problem...")
    state = P.solve(num_iterations=50)
    print("Done.")

    # 9) Report results
    final_cap = state.parameters["dc_capacity"]
    results = pd.DataFrame(
        {
            "bus_id": chosen_bus_names,
            "cap_mw": final_cap,
            "capex_per_mw": np.array(capital_per_mw, dtype=float),
            "total_capex": final_cap * np.array(capital_per_mw, dtype=float),
        }
    )
    print(results.sort_values("cap_mw", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
