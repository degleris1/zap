import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

NCHS_MULT = {
    "Large Central Metro": 1.50,
    "Large Fringe Metro": 1.35,
    "Medium Metro": 1.25,
    "Small Metro": 1.15,
    "Micropolitan": 1.05,
    "Noncore (Rural)": 1.00,
    "Unclassified": 1.00,
}

CODE_TO_LABEL = {
    "1": "Large Central Metro",
    "2": "Large Fringe Metro",
    "3": "Medium Metro",
    "4": "Small Metro",
    "5": "Micropolitan",
    "6": "Noncore (Rural)",
}


def read_csv_safely(path: Path, **kwargs) -> pd.DataFrame:
    """Try multiple encodings for robustness."""
    encodings = ["utf-8", "latin-1", "windows-1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Could not read {path} with common encodings. Last error: {last_err}"
    )


def zfill_str(x, n):
    if pd.isna(x):
        return None
    s = str(x).strip()
    # Handle accidental floats like '53.0'
    if "." in s:
        s = s.split(".", 1)[0]
    return s.zfill(n)


def load_buses(path: Path) -> pd.DataFrame:
    df = read_csv_safely(path, dtype=str)
    # Normalize county_fips or construct from state_fips + county code if present
    if "county_fips" not in df.columns:
        if {"state_fips", "county_fips_code"}.issubset(set(df.columns)):
            df["county_fips"] = df["state_fips"].apply(lambda s: zfill_str(s, 2)) + df[
                "county_fips_code"
            ].apply(lambda s: zfill_str(s, 3))
        else:
            raise ValueError(
                "buses CSV must include `county_fips` or (`state_fips`,`county_fips_code`)."
            )
    df["county_fips"] = df["county_fips"].apply(lambda s: zfill_str(s, 5))
    # Coerce coords if present
    for col in ("x", "y", "lon", "lat", "longitude", "latitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_nchs(path: Path) -> pd.DataFrame:
    nchs = read_csv_safely(path, dtype=str)
    # Build county_fips: prefer STFIPS + CTYFIPS; else FIPS
    if {"STFIPS", "CTYFIPS"}.issubset(set(nchs.columns)):
        nchs["county_fips"] = nchs["STFIPS"].apply(lambda s: zfill_str(s, 2)) + nchs[
            "CTYFIPS"
        ].apply(lambda s: zfill_str(s, 3))
    elif "FIPS" in nchs.columns:
        nchs["county_fips"] = nchs["FIPS"].apply(lambda s: zfill_str(s, 5))
    else:
        # Try generic match
        fips_col = next((c for c in nchs.columns if "fips" in c.lower()), None)
        if not fips_col:
            raise ValueError("NCHS CSV must include STFIPS+CTYFIPS or FIPS.")
        nchs["county_fips"] = nchs[fips_col].apply(lambda s: zfill_str(s, 5))
    # Map code to label (prefer CODE2023)
    if "CODE2023" in nchs.columns:
        nchs["nchs_code"] = nchs["CODE2023"].str.strip()
    else:
        # Fall back to any column containing 'code'
        code_cands = [c for c in nchs.columns if "code" in c.lower()]
        nchs["nchs_code"] = nchs[code_cands[0]].str.strip() if code_cands else None
    nchs["nchs_class"] = nchs["nchs_code"].map(CODE_TO_LABEL).fillna("Unclassified")
    nchs["u_mult"] = nchs["nchs_class"].map(NCHS_MULT).fillna(1.00)
    return nchs[["county_fips", "nchs_class", "u_mult"]]


def load_state_baseline(path: Path) -> pd.DataFrame:
    """Optional per-state baseline file with columns:
    - state_fips (2-digit) OR state_name
    - base_per_acre (float)"""
    sb = read_csv_safely(path, dtype=str)
    cols = {c.lower(): c for c in sb.columns}
    if "base_per_acre" not in {c.lower() for c in sb.columns}:
        raise ValueError("state_baseline CSV must include `base_per_acre`.")
    # Normalize base column name
    base_col = cols.get("base_per_acre")
    sb = sb.rename(columns={base_col: "base_per_acre"})
    sb["base_per_acre"] = pd.to_numeric(sb["base_per_acre"], errors="coerce")
    # Normalize keys
    if "state_fips" in cols:
        sb = sb.rename(columns={cols["state_fips"]: "state_fips"})
        sb["state_fips"] = sb["state_fips"].apply(lambda s: zfill_str(s, 2))
    elif "state_name" in cols:
        sb = sb.rename(columns={cols["state_name"]: "state_name"})
    else:
        raise ValueError(
            "state_baseline CSV must include `state_fips` or `state_name`."
        )
    return sb


def synthesize_bus_id(row, idx: int) -> str:
    base = f"{row.get('county_fips','NA')}|{row.get('x','')}|{row.get('y','')}|{idx}"
    return "bus_" + hashlib.md5(base.encode()).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser(
        description="Create land_lookup.csv and land_lookup_details.csv from buses and NCHS urban–rural classifications."
    )
    ap.add_argument(
        "--buses",
        required=True,
        type=Path,
        help="Path to buses CSV (must include county_fips or state_fips+county_fips_code).",
    )
    ap.add_argument(
        "--nchs",
        required=True,
        type=Path,
        help="Path to NCHS county urban–rural CSV (2023).",
    )
    ap.add_argument(
        "--baseline",
        type=float,
        default=100000.0,
        help="Global baseline in $/acre (default: 100000).",
    )
    ap.add_argument(
        "--state-baseline",
        type=Path,
        default=None,
        help="Optional CSV with per-state baseline (state_fips OR state_name, base_per_acre).",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Output directory for the two CSVs (default: current dir).",
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    buses = load_buses(args.buses)
    nchs = load_nchs(args.nchs)

    # Merge NCHS multipliers
    df = buses.merge(nchs, on="county_fips", how="left")

    # Choose baseline: per-state overrides if provided, else global
    df["base_per_acre"] = float(args.baseline)
    if args.state_baseline is not None:
        sb = load_state_baseline(args.state_baseline)
        if "state_fips" in sb.columns and "state_fips" in df.columns:
            df = df.merge(
                sb[["state_fips", "base_per_acre"]].rename(
                    columns={"base_per_acre": "_state_base"}
                ),
                on="state_fips",
                how="left",
            )
        elif "state_name" in sb.columns and "state_name" in df.columns:
            df = df.merge(
                sb[["state_name", "base_per_acre"]].rename(
                    columns={"base_per_acre": "_state_base"}
                ),
                on="state_name",
                how="left",
            )
        else:
            print(
                "Warning: state_baseline key did not match (need state_fips or state_name in both). Using global baseline."
            )
        df["base_per_acre"] = df["_state_base"].combine_first(df["base_per_acre"])
        if "_state_base" in df.columns:
            df = df.drop(columns=["_state_base"])

    # Fill missing NCHS multiplier and label
    df["u_mult"] = df.get("u_mult", np.nan).fillna(1.00)
    df["nchs_class"] = df.get("nchs_class", "Unclassified").fillna("Unclassified")

    # Compute land_usd_per_acre
    df["land_usd_per_acre"] = df["base_per_acre"] * df["u_mult"]

    # Ensure bus_id
    if "bus_id" not in df.columns:
        df["bus_id"] = [synthesize_bus_id(row, i) for i, row in df.iterrows()]

    # Output CSVs
    lookup = df[["bus_id", "land_usd_per_acre"]].copy()
    details_cols = ["bus_id", "county_fips"]
    for c in (
        "state_fips",
        "state_name",
        "nchs_class",
        "u_mult",
        "base_per_acre",
        "land_usd_per_acre",
    ):
        if c in df.columns:
            details_cols.append(c)
    details = df[details_cols].copy()

    lookup_path = args.outdir / "land_lookup.csv"
    details_path = args.outdir / "land_lookup_details.csv"
    lookup.to_csv(lookup_path, index=False)
    details.to_csv(details_path, index=False)

    # Brief report
    counts = (
        details["nchs_class"]
        .value_counts(dropna=False)
        .rename_axis("nchs_class")
        .reset_index(name="num_buses")
    )
    print("Wrote:", lookup_path, "and", details_path)
    print("\nNCHS class distribution:")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
