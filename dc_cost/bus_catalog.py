# bus_catalog.py
import hashlib
from pathlib import Path

import geopandas as gpd
import pandas as pd


def _slugify_bus(name: str) -> str:
    # stable 8-hex suffix from name; readable prefix
    h = hashlib.blake2s(name.encode(), digest_size=4).hexdigest()
    return f"bus_{h}"


def build_bus_catalog(pn) -> pd.DataFrame:
    b = pn.buses.copy()
    b["bus_name"] = b.index
    b["bus_id"] = b["bus_name"].map(_slugify_bus)
    # geometry
    gdf = gpd.GeoDataFrame(
        b, geometry=gpd.points_from_xy(b["x"], b["y"]), crs="EPSG:4326"
    )

    # counties
    county_url = (
        "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
    )
    counties = gpd.read_file(county_url)[
        ["STATEFP", "COUNTYFP", "GEOID", "NAME", "STATE_NAME", "geometry"]
    ]
    j = gpd.sjoin(gdf, counties.to_crs("EPSG:4326"), how="left", predicate="within")

    # zero-pad FIPS
    j["state_fips"] = j["STATEFP"].astype(str).str.zfill(2)
    j["county_fips"] = j["GEOID"].astype(str).str.zfill(5)

    out = pd.DataFrame(
        {
            "bus_name": j["bus_name"],
            "bus_id": j["bus_id"],
            "node_index": pd.RangeIndex(
                len(j)
            ),  # update via your parse_buses if needed
            "x": j["x"],
            "y": j["y"],
            "county_fips": j["county_fips"],
            "county_name": j["NAME"],
            "state_fips": j["state_fips"],
            "state_name": j["STATE_NAME"],
        }
    ).set_index("bus_name")

    return out


def save_bus_catalog(
    catalog: pd.DataFrame, artifacts_dir: Path | str = "artifacts"
) -> None:
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out_dir / "bus_catalog.parquet")
    catalog.to_csv(out_dir / "bus_catalog.csv")
