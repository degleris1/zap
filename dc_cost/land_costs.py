import pandas as pd


class LandCostService:
    def __init__(self, land_df: pd.DataFrame):
        # Expect columns: bus_id, county_fips, state_fips, land_usd_per_acre
        self.land = land_df.copy()
        required = {"bus_id", "county_fips", "state_fips", "land_usd_per_acre"}
        missing = required - set(self.land.columns)
        if missing:
            raise ValueError(
                f"land_df missing required columns: {sorted(missing)}; got {sorted(self.land.columns)}"
            )

        self.land["county_fips"] = self.land["county_fips"].astype(str).str.zfill(5)
        self.land["state_fips"] = self.land["state_fips"].astype(str).str.zfill(2)
        self.land["land_usd_per_acre"] = pd.to_numeric(
            self.land["land_usd_per_acre"], errors="coerce"
        )

        if self.land["land_usd_per_acre"].isna().any():
            bad = self.land[self.land["land_usd_per_acre"].isna()]
            raise ValueError(
                "Found NaNs in land_usd_per_acre for rows: "
                + ", ".join(bad["bus_id"].astype(str).tolist())
            )

        self.by_bus = self.land.set_index("bus_id")["land_usd_per_acre"]
        self.by_county = self.land.groupby("county_fips")["land_usd_per_acre"].mean()
        self.by_state = self.land.groupby("state_fips")["land_usd_per_acre"].mean()

    def land_cost_per_acre_for(self, bus_catalog: pd.DataFrame) -> pd.Series:
        s = pd.Series(index=bus_catalog.index, dtype=float)

        # try bus_id
        mask = bus_catalog["bus_id"].isin(self.by_bus.index)
        s.loc[mask] = bus_catalog.loc[mask, "bus_id"].map(self.by_bus)

        # fallback county
        missing = s[s.isna()].index
        if len(missing):
            s.loc[missing] = bus_catalog.loc[missing, "county_fips"].map(self.by_county)

        # fallback state
        missing = s[s.isna()].index
        if len(missing):
            s.loc[missing] = bus_catalog.loc[missing, "state_fips"].map(self.by_state)

        # final check
        if s.isna().any():
            holes = s[s.isna()].index.tolist()
            raise ValueError(f"Missing land_usd_per_acre for: {holes}")

        return s
