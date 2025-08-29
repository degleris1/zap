from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

import zap
from zap.importers.pypsa import parse_buses

from .bus_catalog import build_bus_catalog
from .land_costs import LandCostService


def make_datacenter_device(
    pn: pypsa.Network,
    chosen_bus_names: list[str],
    land_df: pd.DataFrame,
    *,
    acres_per_mw: float,
    gamma_per_mw: float,
    cost_unit: float,
    time_horizon_hours: int,
    time_res_hours: float,
    profile_type: zap.devices.injector.DataCenterLoad.ProfileType = zap.devices.injector.DataCenterLoad.ProfileType.AI_INFER,
    linear_cost_per_mw: float = 1000.0,
    pue: float = 1.20,
) -> zap.devices.injector.DataCenterLoad:
    # Build / reuse bus catalog
    bus_catalog = build_bus_catalog(pn)

    # Map bus names to node indices used by zap
    buses, buses_to_index = parse_buses(pn)
    terminals = np.array([buses_to_index[b] for b in chosen_bus_names], dtype=int)
    n = len(terminals)

    # Land costs with fallbacks
    land_srv = LandCostService(land_df)
    sub_catalog = bus_catalog.loc[chosen_bus_names]
    try:
        land_per_acre = land_srv.land_cost_per_acre_for(sub_catalog)
    except ValueError:
        by_state = land_srv.by_state
        land_per_acre = sub_catalog["state_fips"].map(by_state)
        if land_per_acre.isna().any():
            land_per_acre = land_per_acre.fillna(land_df["land_usd_per_acre"].mean())

    # Compute capital cost per MW aligned to terminals
    idx_by_node = sub_catalog.reset_index().set_index("node_index")
    capital_costs = []
    for node in terminals:
        row = idx_by_node.loc[node]
        bus_name = row["bus_name"]
        per_mw = float(gamma_per_mw) + float(acres_per_mw) * float(
            land_per_acre.loc[bus_name]
        )
        capital_costs.append((1.0 / cost_unit) * per_mw)
    capital_costs = np.array(capital_costs, dtype=float)

    # Device parameters
    T = int(time_horizon_hours / time_res_hours)
    profile_types = [profile_type] * n
    linear_cost = np.ones(n) * float(linear_cost_per_mw)
    nominal_capacity = np.zeros(n)  # planning can set these later

    # Guard for profile generation; zeros remain for investment cost
    nominal_for_profile = np.maximum(nominal_capacity, 1e-6)

    dcl = zap.DataCenterLoad(
        num_nodes=len(buses),
        terminal=terminals,
        profile_types=profile_types,
        nominal_capacity=nominal_for_profile,
        linear_cost=linear_cost,
        settime_horizon=T,
        time_resolution_hours=float(time_res_hours),
        capital_cost=capital_costs,
        pue=pue,
    )
    return dcl
