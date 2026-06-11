# Pulling a higher-resolution (and 2025) PyPSA-USA western network

**Status (checked 2026-06-05): a 2025 network cannot be pulled or readily built right now.**
Verified directly against the PyPSA-USA workflow + its Zenodo data:

- PyPSA-USA does **not** distribute prebuilt clustered networks — `elec_s{simpl}_c{clusters}...nc`
  files are *build outputs* of the snakemake workflow, not downloads.
- The build needs conda/mamba + a **567 MB** data bundle
  (`zenodo.org/records/14219029/files/pypsa_usa_data.zip`) + a **~4.95 GB** weather cutout
  (`zenodo.org/records/14611937/files/usa_era5_{YEAR}.nc`) + a multi-hour snakemake run.
- **Weather cutouts exist only for 2019–2023** (2024 and 2025 both return HTTP 404). So a
  **2025** network requires generating a 2025 cutout from ERA5-2025 via `atlite`, which needs a
  Copernicus CDS API key + a multi-GB ERA5 download + hours of processing.
- The existing `elec_s490_c490.nc` is already the **most recent available weather year (2023)**.

None of that is doable inside this sandbox (no conda, ~5.5 GB downloads, hours, CDS auth). The
recipe below is what to run on a capable machine; the study pipeline here is already wired to
consume the result (`--network <file> --year <YYYY>`).

## Recipe (run on a workstation/cluster with conda + ~30 GB free)

```bash
git clone https://github.com/PyPSA/pypsa-usa.git && cd pypsa-usa
mamba env create -f workflow/envs/environment.yaml && conda activate pypsa-usa
cd workflow
```

Edit `config/config.yaml` (overrides `repo_data/config/config.default.yaml`):

```yaml
scenario:
  interconnect: [western]
  clusters: [1000]        # higher resolution (was 490). try 1000 / 2000
  simpl: [200]            # pre-cluster simplification (>= clusters//2 is typical)
  opts: [REM-3h]
  ll: [v1.0]

# --- weather / demand year ---
renewable_weather_years: [2023]   # 2025 NOT available prebuilt; see note below
snapshots:
  start: "2023-01-01"
  end:   "2024-01-01"
  inclusive: "left"
```

Build just the electricity network (skip the expensive solve):

```bash
snakemake -j4 --use-conda \
  "resources/western/elec_s200_c1000_ec_l v1.0_REM-3h.nc"   # (remove the space; matches your opts/ll)
# or simply build everything up to the network:
snakemake -j4 --use-conda western_elec_networks
```

The resulting `.nc` (≈ `resources/western/elec_s200_c1000_ec_lv1.0_REM-3h.nc`) is what you point the
study at.

### To actually get **2025** (extra steps; the blocker)
There is no prebuilt 2025 cutout. Build one first:
1. Get a Copernicus CDS API key (`~/.cdsapirc`).
2. In config set `enable.build_cutout: true`, `renewable_weather_years: [2025]`,
   `snapshots: {start: "2025-01-01", end: "2026-01-01"}`.
3. `snakemake -j1 --use-conda build_cutout` (downloads ERA5-2025, hours, multi-GB),
   then build the network as above.

## Then rerun the study (already year-/resolution-aware)

```bash
# point at the new network; --year selects a summer-peak window in that calendar year
.venv/bin/python development/dc_placement_study.py    --network /path/elec_s200_c1000_....nc --year 2025 --budget 3.0
.venv/bin/python development/dc_placement_optimize.py  --network /path/elec_s200_c1000_....nc --budget 2.0 --site-cap 0.25
.venv/bin/python development/dc_granularity_sweep.py   --network /path/elec_s200_c1000_....nc --budget 3.0
.venv/bin/python development/dc_placement_scopf.py     --network /path/elec_s200_c1000_....nc --engine exact
```

`resolve_year_window()` in `dc_placement_study.py` picks a summer window in the requested year if
the network covers it, and prints a clear warning + falls back otherwise (verified: `--year 2023`
→ summer 2023; `--year 2025` on the 2023 net → warns and falls back). Everything else (snapshot
auto-selection of the most-congested hour, robust metrics, optimizer, granularity, SCOPF) is
resolution-agnostic and runs unchanged at 1000/2000 nodes — only slower.
```
