import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_number(x: str) -> float:
    s = str(x).strip().replace("_", "")
    mul = 1.0
    if s[-1:].lower() == "k":
        mul = 1e3
        s = s[:-1]
    elif s[-1:].lower() == "m":
        mul = 1e6
        s = s[:-1]
    return float(s) * mul


def parse_list(arg: str, item_parser=float):
    parts = [p for p in str(arg).split(",") if p.strip() != ""]
    out = []
    for p in parts:
        out.append(item_parser(p.strip()))
    return out


def load_land(land_path: Path) -> pd.DataFrame:
    df = pd.read_csv(land_path)
    # Normalize columns
    if "bus_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "bus_id"})
        df["bus_id"] = df["bus_id"].apply(lambda i: f"bus_{i}")
    # locate land value column
    cand = [
        c
        for c in df.columns
        if c.lower()
        in {
            "land_usd_per_acre",
            "land_cost_$per_acre",
            "land_cost_per_acre",
            "land_per_acre",
            "land_cost",
        }
    ]
    if not cand:
        raise ValueError(
            "Could not find a land value column. Expected one of: land_usd_per_acre, land_cost_$per_acre, land_cost_per_acre, land_per_acre, land_cost"
        )
    col = cand[0]
    df = df[["bus_id", col]].rename(columns={col: "land_usd_per_acre"}).copy()
    df["land_usd_per_acre"] = pd.to_numeric(df["land_usd_per_acre"], errors="coerce")
    if df["land_usd_per_acre"].isna().any():
        raise ValueError(
            "Found NaNs in land_usd_per_acre. Please check your input file."
        )
    return df


def compute_capital_cost_per_mw_for_buses(
    land_csv: Path,
    bus_ids: list[str],
    acres_per_mw: float,
    gamma_per_mw: float,
) -> list[float]:
    """
    Compute per-MW capital cost for each bus_id as:
        capital_cost_per_mw = gamma_per_mw + acres_per_mw * land_usd_per_acre

    Returns a list aligned with bus_ids.
    """
    land = load_land(land_csv)
    lookup = dict(zip(land["bus_id"], land["land_usd_per_acre"]))
    costs = []
    missing = []
    for b in bus_ids:
        if b not in lookup:
            missing.append(b)
            costs.append(math.nan)
        else:
            costs.append(gamma_per_mw + acres_per_mw * float(lookup[b]))
    if missing:
        raise ValueError(
            f"The following bus_ids are missing from {land_csv}: {', '.join(missing)}"
        )
    return costs


from typing import Optional


def compute_dc_cost(
    land: pd.DataFrame,
    cap_mw_list: list[float],
    acres_per_mw_list: list[float],
    gamma_list: list[float],
    bus_ids: Optional[list[str]] = None,
    top_n: int = 0,
) -> pd.DataFrame:
    """
    Library entry point mirroring the CLI behavior. Returns a DataFrame with
    the same columns as the CLI output.
    """
    df_land = land.copy()
    if bus_ids:
        df_land = df_land[df_land["bus_id"].isin(bus_ids)].copy()
        if df_land.empty:
            raise ValueError("No matching bus_id after filter.")

    scenarios = []
    for C in cap_mw_list:
        for A in acres_per_mw_list:
            for G in gamma_list:
                scenarios.append((C, A, G))

    frames = []
    for C, A, G in scenarios:
        tmp = df_land.copy()
        tmp["cap_mw"] = C
        tmp["acres_per_mw"] = A
        tmp["gamma_per_mw"] = G
        tmp["land_per_mw"] = A * tmp["land_usd_per_acre"]
        tmp["total_per_mw"] = tmp["gamma_per_mw"] + tmp["land_per_mw"]
        tmp["total_cost"] = C * tmp["total_per_mw"]
        frames.append(tmp)

    out = pd.concat(frames, ignore_index=True)

    if top_n and top_n > 0:
        out = (
            out.sort_values(["cap_mw", "acres_per_mw", "gamma_per_mw", "total_cost"])
            .groupby(
                ["cap_mw", "acres_per_mw", "gamma_per_mw"],
                as_index=False,
                group_keys=False,
            )
            .head(top_n)
        )

    out = out[
        [
            "bus_id",
            "cap_mw",
            "gamma_per_mw",
            "acres_per_mw",
            "land_usd_per_acre",
            "land_per_mw",
            "total_per_mw",
            "total_cost",
        ]
    ]
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Compute full data center cost scenarios from land_lookup.csv."
    )
    ap.add_argument(
        "--land",
        type=Path,
        required=True,
        help="Path to land_lookup.csv (must include bus_id, land_usd_per_acre).",
    )
    ap.add_argument(
        "--cap-mw",
        type=str,
        required=True,
        help="Capacity MW; comma-separated for multiple (e.g., 10,50,100).",
    )
    ap.add_argument(
        "--acres-per-mw",
        type=str,
        default="0.25,0.15",
        help="Acres per MW; comma-separated (default: 0.25,0.15).",
    )
    ap.add_argument(
        "--gamma",
        type=str,
        required=True,
        help="Building cost $/MW; comma-separated (e.g., 7.5M,12.5M).",
    )
    ap.add_argument(
        "--bus-ids",
        type=str,
        default="",
        help="Optional comma-separated list of bus_ids to include.",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="If >0, return the N cheapest buses per (cap, A, gamma) scenario.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("dc_cost_results.csv"),
        help="Output CSV path (default: dc_cost_results.csv).",
    )
    args = ap.parse_args()

    land = load_land(args.land)

    # Filter buses if requested
    if args.bus_ids.strip():
        ids = [s.strip() for s in args.bus_ids.split(",") if s.strip() != ""]
        land = land[land["bus_id"].isin(ids)].copy()
        if land.empty:
            raise SystemExit("No matching bus_id after filter.")

    caps = parse_list(args.cap_mw, float)
    acres = parse_list(args.acres_per_mw, float)
    gammas = parse_list(args.gamma, parse_number)  # supports 7.5M

    # Cartesian product over scenarios
    scenarios = []
    for C in caps:
        for A in acres:
            for G in gammas:
                scenarios.append((C, A, G))

    # Build results
    frames = []
    for C, A, G in scenarios:
        tmp = land.copy()
        tmp["cap_mw"] = C
        tmp["acres_per_mw"] = A
        tmp["gamma_per_mw"] = G
        tmp["land_per_mw"] = A * tmp["land_usd_per_acre"]
        tmp["total_per_mw"] = tmp["gamma_per_mw"] + tmp["land_per_mw"]
        tmp["total_cost"] = C * tmp["total_per_mw"]
        frames.append(tmp)

    out = pd.concat(frames, ignore_index=True)

    # Optionally keep only top-N per scenario
    if args.top_n and args.top_n > 0:
        out = (
            out.sort_values(["cap_mw", "acres_per_mw", "gamma_per_mw", "total_cost"])
            .groupby(
                ["cap_mw", "acres_per_mw", "gamma_per_mw"],
                as_index=False,
                group_keys=False,
            )
            .head(args.top_n)
        )

    # Reorder columns for clarity
    out = out[
        [
            "bus_id",
            "cap_mw",
            "gamma_per_mw",
            "acres_per_mw",
            "land_usd_per_acre",
            "land_per_mw",
            "total_per_mw",
            "total_cost",
        ]
    ]

    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
