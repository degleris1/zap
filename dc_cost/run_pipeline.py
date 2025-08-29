import argparse
import json
import time
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

import zap
from zap.importers.pypsa import load_pypsa_network, parse_buses

from .bus_catalog import build_bus_catalog, save_bus_catalog
from .land_costs import LandCostService


def select_top10_wecc_buses(pn: pypsa.Network) -> list[str]:
    # Score based on average load and generator nameplate at the bus
    load_ts = pn.loads_t.p_set
    # Avoid deprecated axis=1 grouping; transpose and group by top-level columns (bus)
    avg_load = load_ts.T.groupby(level=0).mean().sum(axis=1)  # MW per bus
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


def plot_pareto_tradeoff(dispatch_costs, investment_costs, labels, out_pdf: Path):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(dispatch_costs, investment_costs, s=200)

    for x, y, label in zip(dispatch_costs, investment_costs, labels):
        ax.text(x, y, label, fontsize=9, ha="left", va="bottom")

    ax.set_xlabel("Dispatch Cost ($)", fontsize=12)
    ax.set_ylabel("Investment Cost ($)", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    plt.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf)
    return fig


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pypsa_any(path_like: Path) -> pypsa.Network:
    path_like = Path(path_like)
    if path_like.is_file() and path_like.suffix.lower() in {".nc", ".netcdf", ".h5"}:
        return pypsa.Network(str(path_like))
    if path_like.is_dir():
        # Prefer a NetCDF inside if present
        nc = sorted(list(path_like.glob("*.nc")))
        if nc:
            return pypsa.Network(str(nc[0]))
        # Fallback to CSV folder
        pn = pypsa.Network()
        pn.import_from_csv_folder(path_like)
        return pn
    raise FileNotFoundError(f"Path not found or unsupported: {path_like}")


def run(args: argparse.Namespace):
    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load PyPSA network from .nc or CSVs
    src = Path(args.pypsa_csv).resolve()
    pn = _load_pypsa_any(src)
    if pn.buses.empty:
        raise SystemExit(f"No buses found in network loaded from {src}.")

    # 2) Build/refresh bus catalog and persist
    bus_catalog = build_bus_catalog(pn)

    # Map node_index using zap mapping (non-battery buses)
    buses, buses_to_index = parse_buses(pn)
    bus_catalog["node_index"] = bus_catalog.index.map(buses_to_index)
    save_bus_catalog(bus_catalog, artifacts_dir)

    # 3) Select candidate sites (top-10 by score)
    chosen_bus_names = select_top10_wecc_buses(pn)
    terminals = np.array([buses_to_index[b] for b in chosen_bus_names], dtype=int)
    n_dc = len(terminals)

    pd.DataFrame(
        {
            "bus_name": chosen_bus_names,
            "bus_id": bus_catalog.loc[chosen_bus_names, "bus_id"].values,
            "node_index": terminals,
        }
    ).to_csv(artifacts_dir / "selected_buses.csv", index=False)

    # 4) Normalize land costs and compute capital costs aligned to terminals
    land_df = pd.read_csv(args.land_details)
    land_srv = LandCostService(land_df)

    # Restrict to selected buses for robustness
    sub_catalog = bus_catalog.loc[chosen_bus_names]
    try:
        land_per_acre_sub = land_srv.land_cost_per_acre_for(sub_catalog)
    except ValueError:
        # Fallback: state mean, then global mean
        by_state = land_srv.by_state
        land_per_acre_sub = sub_catalog["state_fips"].map(by_state)
        if land_per_acre_sub.isna().any():
            land_per_acre_sub = land_per_acre_sub.fillna(
                land_df["land_usd_per_acre"].mean()
            )

    # Compute capital cost vector aligned with terminals
    idx_by_node_sub = sub_catalog.reset_index().set_index("node_index")
    capital_costs = []
    for node in terminals:
        row = idx_by_node_sub.loc[node]
        bus_name = row["bus_name"]
        per_mw = float(args.gamma_per_mw) + float(args.acres_per_mw) * float(
            land_per_acre_sub.loc[bus_name]
        )
        capital_costs.append((1.0 / args.cost_unit) * per_mw)
    capital_costs = np.array(capital_costs, dtype=float)

    # Persist detailed capital costs for the chosen terminals
    cap_rows = []
    idx_by_node = idx_by_node_sub
    for node in terminals:
        row = idx_by_node.loc[node]
        bus_name = row["bus_name"]
        cap_rows.append(
            {
                "bus_name": bus_name,
                "bus_id": row["bus_id"],
                "node_index": int(node),
                "acres_per_mw": float(args.acres_per_mw),
                "gamma_per_mw": float(args.gamma_per_mw),
                "land_usd_per_acre": float(land_per_acre_sub.loc[bus_name]),
                "capital_cost_per_mw": float(
                    capital_costs[list(terminals).index(node)]
                ),
            }
        )
    pd.DataFrame(cap_rows).to_csv(
        artifacts_dir / "dc_capital_costs_demo.csv", index=False
    )

    # 5) Convert PyPSA network to zap network and devices
    requested_T = int(args.time_horizon_hours / args.time_res_hours)
    # Use network's own snapshots to avoid date mismatches
    dates = pn.snapshots[:requested_T]
    time_horizon = len(dates)

    net, devices = load_pypsa_network(
        pn,
        dates,
        power_unit=args.power_unit,
        cost_unit=args.cost_unit,
        scale_load=1.0,
        drop_empty_generators=True,
        expand_empty_generators=0.0,
        scale_generator_capacity_factor=1.0,
        scale_line_capacity_factor=1.0,
        marginal_load_value=1000.0,
    )

    # 6) DataCenterLoad devices for each scenario (simple)
    profile_types = [zap.DataCenterLoad.ProfileType.AI_INFER] * n_dc
    linear_cost = np.ones(n_dc) * 1000
    T = time_horizon

    # Scenario capacities (replicate marimo demo if n_dc>=10, else adapt)
    def _clip_or_pad(arr, n):
        arr = np.array(arr, dtype=float)
        if len(arr) >= n:
            return arr[:n]
        out = np.zeros(n)
        out[: len(arr)] = arr
        return out

    uniform_capacities = np.ones(n_dc) * 100.0
    investment_opt_capacities = _clip_or_pad(
        [1000.0 / 3.0, 1000.0 / 3.0, 1000.0 / 3.0], n_dc
    )
    solver_opt_capacities = _clip_or_pad(
        [
            275.41860954,
            275.39441647,
            275.39291164,
            24.82772319,
            24.82772319,
            24.82772319,
            24.82772319,
            24.82772319,
            24.82772319,
            24.82772319,
        ],
        n_dc,
    )
    grid_opt_capacities = _clip_or_pad(
        [
            109.79731552,
            73.48552114,
            87.88355807,
            109.73627581,
            107.75840246,
            86.61183818,
            86.68299651,
            104.44680737,
            129.91202981,
            103.68525514,
        ],
        n_dc,
    )

    scenarios = [
        ("Uniform Distribution", uniform_capacities),
        ("Capital Based Distribution", investment_opt_capacities),
        ("Solver Optimized Distribution", solver_opt_capacities),
        ("Dispatch Optimized Distribution", grid_opt_capacities),
    ]

    dispatch_costs = []
    investment_costs = []

    for label, caps in scenarios:
        # Avoid zero-site divide-by-zero in synthetic profile generation
        caps_for_profile = np.maximum(caps, 1e-6)
        dcl = zap.DataCenterLoad(
            num_nodes=net.num_nodes,
            terminal=terminals,
            profile_types=profile_types,
            nominal_capacity=caps_for_profile,
            linear_cost=linear_cost,
            settime_horizon=T,
            time_resolution_hours=float(args.time_res_hours),
            capital_cost=capital_costs,
        )
        devs = devices + [dcl]
        outcome = net.dispatch(
            devs, time_horizon=T, solver=cp.CLARABEL, add_ground=False
        )
        dispatch_costs.append(outcome.problem.value)
        investment_costs.append(float(np.dot(capital_costs, caps)))

    # 7) Plot Pareto and save CSV
    labels = [s[0] for s in scenarios]
    plot_pareto_tradeoff(
        dispatch_costs,
        investment_costs,
        labels,
        artifacts_dir / "pareto_optimal_allocations.pdf",
    )

    pd.DataFrame(
        {
            "label": labels,
            "dispatch_cost": dispatch_costs,
            "investment_cost": investment_costs,
        }
    ).to_csv(artifacts_dir / f"pareto_{int(time.time())}.csv", index=False)

    # 8) Scenario JSON and validation report
    scenario = {
        "gamma_per_mw": float(args.gamma_per_mw),
        "acres_per_mw": float(args.acres_per_mw),
        "selected_buses": chosen_bus_names,
        "terminals": terminals.tolist(),
        "power_unit": float(args.power_unit),
        "cost_unit": float(args.cost_unit),
        "land_details": str(Path(args.land_details).resolve()),
        "land_details_sha256": sha256_file(Path(args.land_details)),
        "pypsa_source": str(src),
        "time_horizon_hours": int(args.time_horizon_hours),
        "time_res_hours": float(args.time_res_hours),
    }
    with open(artifacts_dir / "scenario.json", "w") as f:
        json.dump(scenario, f, indent=2)

    # Validation (very light)
    with open(artifacts_dir / "VALIDATION_REPORT.txt", "w") as f:
        f.write(
            "Alignment OK\n"
            if len(capital_costs) == len(terminals)
            else "Alignment FAIL\n"
        )
        f.write("Units scaled by 1/cost_unit\n")
        f.write(
            "No NaNs in capital costs\n"
            if np.isfinite(capital_costs).all()
            else "NaNs present\n"
        )

    print("Artifacts written to:", artifacts_dir)


def main():
    ap = argparse.ArgumentParser(description="Run end-to-end DC cost and Pareto demo.")
    ap.add_argument(
        "--pypsa-csv", required=True, help="Path to PyPSA CSV folder or .nc file."
    )
    ap.add_argument(
        "--land-details",
        default="dc_cost/land_lookup_details.csv",
        help="Path to land_lookup_details.csv (bus_id, county_fips, state_fips, land_usd_per_acre).",
    )
    ap.add_argument("--acres-per-mw", type=float, default=0.20)
    ap.add_argument("--gamma-per-mw", type=float, default=12_000_000.0)
    ap.add_argument("--time-horizon-hours", type=int, default=24)
    ap.add_argument("--time-res-hours", type=float, default=1.0)
    ap.add_argument("--power-unit", type=float, default=1.0e3)
    ap.add_argument("--cost-unit", type=float, default=100.0)
    ap.add_argument("--artifacts", type=str, default="artifacts")
    args = ap.parse_args()

    run(args)


if __name__ == "__main__":
    main()
