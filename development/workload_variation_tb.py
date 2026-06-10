import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import cvxpy as cp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    import zap
    from zap.importers.pypsa import load_pypsa_network, parse_buses
    import os
    from pathlib import Path
    import pypsa
    from zap.devices import ACLine
    import pandas as pd
    import geopandas as gpd
    return (
        ACLine,
        Path,
        cp,
        gpd,
        load_pypsa_network,
        mo,
        np,
        os,
        parse_buses,
        pd,
        plt,
        pypsa,
        sns,
        zap,
    )


@app.cell
def upsample_zap_devices():
    def upsample_zap_devices(devices, factor=4, original_timesteps=24):
        """Upsample time-varying attributes of zap devices by repeating each timestep."""
        upsampled_zap_devices = []
        for dev in devices:
            upsampled_dev = dev.sample_time(original_timesteps*factor, original_timesteps)
            upsampled_zap_devices.append(upsampled_dev)

        return upsampled_zap_devices
            # for attr in ['dynamic_capacity', 'load', 'linear_cost']:
            #     if hasattr(dev, attr):
            #         val = getattr(dev, attr)
            #         if val is not None and val.ndim == 2 and val.shape[1] == original_timesteps:
            #             setattr(dev, attr, np.repeat(val, factor, axis=1))
    return (upsample_zap_devices,)


@app.cell
def _(os, pypsa):
    HOME_PATH = os.environ.get("HOME")
    PYPSA_NETW0RK_PATH = (
        HOME_PATH + "/zap_data/pypsa-networks/western_small/network_2023.nc"
    )
    pn = pypsa.Network(PYPSA_NETW0RK_PATH)
    snapshots = pn.generators_t.p_max_pu.index
    snapshot_data = snapshots[5616:5640]  # 8/23/21 # hourly
    return HOME_PATH, PYPSA_NETW0RK_PATH, pn, snapshot_data, snapshots


@app.cell
def _(gpd, pn):
    b = pn.buses.copy()
    gdf = gpd.GeoDataFrame(
        b, geometry=gpd.points_from_xy(b["x"], b["y"]), crs="EPSG:4326"
    )

    county_url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
    counties = gpd.read_file(county_url)[["STATEFP","COUNTYFP","GEOID","NAME","STATE_NAME","geometry"]]

    j = gpd.sjoin(gdf, counties.to_crs("EPSG:4326"), how="left", predicate="within")

    pn.buses["county_fips"] = j["GEOID"]  # 5-digit FIPS
    pn.buses["county_name"] = j["NAME"]
    pn.buses["state_fips"]  = j["STATEFP"]
    pn.buses["state_name"]  = j["STATE_NAME"]
    return b, counties, county_url, gdf, j


@app.cell
def _(load_pypsa_network, pn, snapshot_data, upsample_zap_devices):
    pypsa_kwargs = {}
    pypsa_net, pypsa_devices = load_pypsa_network(
        pn, snapshot_data, power_unit=1.0e3, cost_unit=100.0, **pypsa_kwargs
    )

    pypsa_devices = upsample_zap_devices(pypsa_devices, factor=4, original_timesteps=24)
    return pypsa_devices, pypsa_kwargs, pypsa_net


@app.cell
def _(np, pypsa_devices):
    props = vars(pypsa_devices[2])

    for key,val in props.items():
        if isinstance(val, np.ndarray):
            print(f"{key} has dimension: {val.shape}" )
    return key, props, val


@app.cell
def _(ACLine, np, zap):
    def calculate_ptdf(
            network: zap.PowerNetwork,
            devices: list[zap.devices.AbstractDevice],
            slack_bus: int = 0,
        ) -> np.ndarray:
            """Calculate the power transfer distribution factors for the given network and devices.

            :param network: Zap network.
            :param devices: Zap devices.

            return: PTDF matrix shape (num_lines, num_buses)
            """
            ac_lines = [d for d in devices if isinstance(d, ACLine)][0]

            Y_bus = compute_Y_bus(ac_lines)
            B_inv = np.linalg.pinv(Y_bus)

            A = ac_lines.incidence_matrix[1] - ac_lines.incidence_matrix[0]
            sus = np.diag(ac_lines.susceptance[:, 0] * ac_lines.nominal_capacity[:, 0])

            ptdf = (sus @ A.T) @ B_inv

            return ptdf

    def compute_Y_bus(ac_lines: ACLine) -> np.ndarray:
        """Compute the Y-bus matrix for the given AC lines.

        :param ac_lines: Zap AC line objects
        """
        sus = ac_lines.susceptance[:, 0] * ac_lines.nominal_capacity[:, 0]
        A = ac_lines.incidence_matrix[1] - ac_lines.incidence_matrix[0]

        Y_bus = A @ np.diag(sus) @ A.T
        return Y_bus
    return calculate_ptdf, compute_Y_bus


@app.cell
def _(calculate_ptdf, pypsa_devices, pypsa_net):
    ptdf = calculate_ptdf(pypsa_net, pypsa_devices)
    return (ptdf,)


@app.cell
def _(cp, pypsa_devices, pypsa_net):
    outcome_test = pypsa_net.dispatch(
        pypsa_devices, time_horizon=96, solver=cp.CLARABEL, add_ground=False
    )
    return (outcome_test,)


@app.cell
def _(ACLine, np, outcome_test, pypsa_devices):
    ac_lines = [d for d in pypsa_devices if isinstance(d, ACLine)][0]
    nom = np.asarray(ac_lines.nominal_capacity)[:, 0] # [L,]
    cap = np.asarray(ac_lines.capacity) # All 1's anyway
    Fmax = nom[:, None] * cap # [L, 1]
    f = np.asarray(outcome_test.power[3][1]) # [L, T] of line flows
    return Fmax, ac_lines, cap, f, nom


@app.cell
def _(Fmax, f, np, parse_buses, pn, ptdf):
    eps = 1e-9
    PTDF = np.asarray(ptdf) # (L, N)
    PTDF_abs = np.abs(PTDF) + eps # (L, N)
    Fmax_T = np.repeat(Fmax, repeats=f.shape[1], axis=1)  # (L, T)
    # line headroom margin: (L, T)
    margin = np.clip(Fmax_T - np.abs(f), 0.0, None)
    ratio = margin.T[:, :, None] / PTDF_abs[None, :, :] # (T, L, N)
    h_tj = ratio.min(axis=1) # (T, N)

    # Conservative aggregation across time
    h_j = np.quantile(h_tj, 0.05, axis=0)                  # (N,)

    # top10_bus_idx = np.argsort(-h_j)[:10]
    top10_bus_idx = [i for i in range(101)]
    print("Top-10 bus indices by PTDF headroom:", top10_bus_idx)
    print("Top-10 scores:", h_j[top10_bus_idx])

    buses, buses_to_index = parse_buses(pn) # buses_to_index is dict of "pyspa_bus_name": "zap_terminal"
    index_to_bus = {idx: name for name, idx in buses_to_index.items()}
    pypsa_bus_names = [index_to_bus[i] for i in top10_bus_idx]
    print(pypsa_bus_names)
    return (
        Fmax_T,
        PTDF,
        PTDF_abs,
        buses,
        buses_to_index,
        eps,
        h_j,
        h_tj,
        index_to_bus,
        margin,
        pypsa_bus_names,
        ratio,
        top10_bus_idx,
    )


@app.cell
def _(index_to_bus):
    index_to_bus
    return


@app.cell
def _(selected_node_fips):
    selected_node_fips
    return


@app.cell
def _(pd, pn, pypsa_bus_names):
    selected_node_fips = pn.buses.loc[pypsa_bus_names, "county_fips"]
    county_land_lut_df = pd.read_csv("development/county_land_lut.csv")
    return county_land_lut_df, selected_node_fips


@app.cell
def _(county_land_lut_df, index_to_bus, selected_node_fips):
    # selected_node_fips is currently a Series: index=bus_name, values=county_fips
    # Turn it into a DataFrame so we can add columns + merge
    sel = selected_node_fips.rename("county_fips").to_frame()

    # 1) add terminal column (bus -> terminal)
    # NOTE: index_to_bus in your code is {terminal_idx -> bus_name}
    # so invert it to get {bus_name -> terminal_idx}
    bus_to_terminal = {bus: term for term, bus in index_to_bus.items()}
    sel["terminal"] = sel.index.map(bus_to_terminal)

    # 2) join with county_land_lut on FIPS
    # make sure both are same dtype (often leading zeros matter, so use strings)
    sel["county_fips"] = sel["county_fips"].astype(str).str.zfill(5)
    county_land_lut_df["county_fips"] = county_land_lut_df["county_fips"].astype(str).str.zfill(5)

    sel = sel.merge(
        county_land_lut_df,
        left_on="county_fips",
        right_on="county_fips",
        how="left",
    )

    # 3) "cost for each terminal"
    # Replace "cost" with the actual column name in county_land_lut_df (e.g., 'land_cost', 'usd_per_mw', etc.)
    terminal_cost = (
        sel.groupby("terminal")["land_usd2017_per_acre"]
          .first()   # or .mean(), depending on what you want
          .sort_index()
    )

    sel, terminal_cost
    return bus_to_terminal, sel, terminal_cost


@app.cell
def _(sel):
    sel.to_csv("terminal_land_cost_mapping.csv")
    return


@app.cell
def _(np, sel):
    np.array(sel.terminal)
    return


@app.cell
def _():
    # net = zap.PowerNetwork(num_nodes=3)

    # # DataCenterLoad using CSV profile
    # dcload = zap.DataCenterLoad(
    #     num_nodes=pypsa_net.num_nodes,
    #     terminal=np.array([0]),  # Connected to node 0
    #     nominal_capacity=np.array([100.0]),  # 100 MW capacity
    #     profiles=["development/load_profiles/example_inference_azure_conv.csv"],
    #     linear_cost=np.array([5000.0]),  # $/MWh curtailment cost
    #     time_resolution_hours=0.25,  # 15-minute intervals (matches CSV)
    #     capital_cost = []
    #     settime_horizon=24.0  # 24-hour horizon
    # )
    return


@app.cell
def _(np, pypsa_devices, pypsa_net, sel, zap):
    pypsa_devices_dc = pypsa_devices.copy()
    dc_terminals = np.array(sel.terminal)
    acres_per_mw = 0.56 # current guess at acres/MW, sweep later
    fixed_capex_per_mw = 12e6 # capital gamma $/MW
    inflation = 1.32
    dc_captial_costs = np.array(sel.land_usd2017_per_acre)*inflation*acres_per_mw*0.0 #+ fixed_capex_per_mw
    n_dc = len(dc_terminals)

    dcloads = zap.DataCenterLoad(
        num_nodes=pypsa_net.num_nodes,
        terminal=dc_terminals,
        profiles=n_dc*["development/load_profiles/example_inference_azure_conv.csv"],
        nominal_capacity=1e-3 * np.ones((n_dc)),
        linear_cost=np.ones(n_dc) * 500000.0,
        settime_horizon=96,
        capital_cost=dc_captial_costs,
    )
    pypsa_devices_dc.append(dcloads)
    return (
        acres_per_mw,
        dc_captial_costs,
        dc_terminals,
        dcloads,
        fixed_capex_per_mw,
        inflation,
        n_dc,
        pypsa_devices_dc,
    )


@app.cell
def _(pypsa_devices):
    type(pypsa_devices[0].dynamic_capacity)
    return


@app.cell
def _(cp, pypsa_devices_dc, pypsa_net):
    outcome_test2 = pypsa_net.dispatch(
        pypsa_devices_dc, time_horizon=96, solver=cp.CLARABEL, add_ground=False
    )
    return (outcome_test2,)


@app.cell
def _(pypsa_devices_dc):
    for d_x in pypsa_devices_dc:
        print(d_x.time_horizon)
    return (d_x,)


@app.cell
def _(cp, n_dc, np, pypsa_devices_dc, pypsa_net, zap):
    ## Try to write a simple exmaple of a planning problem
    # pypsa_devices_dc = pypsa_devices
    TOTAL_DC_BUDGET = 1
    # MW
    xstar = zap.DispatchLayer(
        pypsa_net,
        pypsa_devices_dc,
        parameter_names={"dc_capacity": (5, "nominal_capacity")},
        time_horizon=96,
        solver=cp.CLARABEL,
    )  # Constuct a DispatchLayer

    # lower_bounds = {}
    # upper_bounds = {}
    lower_bounds = {"dc_capacity": np.full(n_dc, 0)}
    upper_bounds = {"dc_capacity": np.full(n_dc, 0.250)}

    eta = {"dc_capacity": np.full(n_dc, TOTAL_DC_BUDGET / n_dc)}
    init_eta = np.zeros(n_dc)
    # init_eta[0] = 1000
    init_eta = np.random.rand(n_dc)
    eta = {"dc_capacity": init_eta}

    op_obj = zap.planning.DispatchCostObjective(pypsa_net, pypsa_devices_dc)
    inv_obj = zap.planning.InvestmentObjective(pypsa_devices_dc, xstar)

    P = zap.planning.PlanningProblem(
        operation_objective=op_obj,
        investment_objective=inv_obj,
        layer=xstar,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    # Enforce per-site caps AND the budget jointly. NOTE: SimplexBudgetProjection
    # only enforces x>=0 and sum(x)==budget -- it SILENTLY IGNORES the 0.25 GW
    # per-site upper_bounds (a single node could grab the whole budget). Use
    # BoxBudgetProjection to enforce 0 <= x <= cap together with sum(x)==budget.
    P.extra_projections = {
        "dc_capacity": zap.planning.BoxBudgetProjection(
            budget=TOTAL_DC_BUDGET,
            lower_bounds=lower_bounds["dc_capacity"],
            upper_bounds=upper_bounds["dc_capacity"],
        )
    }

    cost = P(**eta, requires_grad=True)
    grad = P.backward()

    state = P.solve(num_iterations=5)
    return (
        P,
        TOTAL_DC_BUDGET,
        cost,
        eta,
        grad,
        init_eta,
        inv_obj,
        lower_bounds,
        op_obj,
        state,
        upper_bounds,
        xstar,
    )


@app.cell
def _(state):
    state
    return


if __name__ == "__main__":
    app.run()
