# Build kit: higher-resolution, 2025-weather-year PyPSA-USA western network

Run this **on a workstation/cluster** (not the zap sandbox). It produces a clustered
western network (default 1000 nodes, 2025 weather year) that drops straight into the
DC-placement study pipeline.

## Why it can't be a download
PyPSA-USA does not distribute prebuilt clustered networks, and its ERA5 weather cutouts
exist on Zenodo only for **2019–2023** (2024/2025 = HTTP 404). So 2025 must be built
locally with `atlite` from ERA5-2025, which needs a free **Copernicus CDS API key**.

## Steps
```bash
# 1. one-time: get a CDS key, accept the ERA5 licence, write ~/.cdsapirc:
#      url: https://cds.climate.copernicus.eu/api
#      key: <UID>:<APIKEY>
#
# 2. build (clones pypsa-usa, makes the conda env, builds the 2025 cutout + network):
bash build_2025.sh ~/pypsa-usa-2025

# 3. verify:
python verify_2025_network.py ~/pypsa-usa-2025/pypsa-usa/workflow/resources/western/elec_s1000_c1000.nc

# 4. rerun the study (from the zap repo), now 2025 + high-res:
python development/dc_placement_study.py    --network <net>.nc --year 2025 --budget 3.0
python development/dc_placement_optimize.py  --network <net>.nc --budget 2.0 --site-cap 0.25
python development/dc_granularity_sweep.py    --network <net>.nc --budget 3.0
python development/dc_placement_scopf.py      --network <net>.nc --engine exact
```

## Files
- `config.western_2025_highres.yaml` — config override (clusters, 2025 snapshots+weather, atlite cutout). Deep-merges over PyPSA-USA defaults.
- `build_2025.sh` — clone + conda env + config + snakemake build. Edit `CLUSTERS` for 1000/2000.
- `verify_2025_network.py` — checks year==2025, resolution > 490, and a zap load+dispatch.

## Cost / caveats
- ~30 GB disk, several hours (the ERA5-2025 cutout build is the long pole).
- conda/mamba + `cdsapi`; env pins `pypsa==0.30.2`, `atlite==0.3.0`, `snakemake-minimal==7.32.4`.
- `clusters` must be ≤ the base TAMU western node count (~few thousand); 1000/2000 are fine.
- If the CDS download stalls, it's usually the licence not accepted or a stale `~/.cdsapirc`
  (the CDS API moved to `https://cds.climate.copernicus.eu/api` in 2024 — no `/v2`).
- For a **no-CDS** alternative, set the year to 2023 in the config (its cutout *is* on Zenodo);
  you still get the higher-resolution network, just the most-recent prebuilt weather year.
