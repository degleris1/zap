# 1000-node 2023 build — status (blocked at add_electricity)

Attempted a higher-resolution (1000-cluster) western 2023 PyPSA-USA build locally
(`~/pypsa-usa-2023/`, 26 GB, micromamba env `pypsa-usa`). Got ~90% of the way, then
hit a hard PyPSA-USA↔pypsa version incompatibility.

## What completed (cached, resumable)
- micromamba env `pypsa-usa` (pypsa 0.30.2, atlite 0.3.0, snakemake 7.32.4)
- TAMU base network + bus regions + powerplants (PUDL/duckdb) + demand (`elec_base_network_dem.nc`)
- **era5_2023 cutout fully downloaded (8.5 GB)** at `cutouts/usa_era5_2023.nc`
- renewable profiles: `profile_solar.nc`, `profile_offwind_floating.nc`

## Seven issues worked through (config/env), in order
1. NREL EFS demand host `data.nrel.gov` has **no DNS** here → switched `demand.profile` to `eia`.
2. Snakefile expects a *full* config (it doesn't load `config.default.yaml`) → assembled one.
3. GODEEEP `data/zenodo/` dir missing (downloader lacks `parents=True`) → `mkdir`.
4. GODEEEP on-demand file download returned invalid netCDF → switched `renewable.dataset` to **atlite**.
5. `onwind` profile: `ValueError: Item wrong length 3 vs 18360` in the "average distances" loop.
6. `solar` profile: same bug (atlite 0.3.0 `cutout.grid` indexing) → **patched** `scripts/build_renewable_profiles.py`
   to stub `average_distance`/`centre_of_mass` (cosmetic transmission-distance costs only).
7. **`add_electricity` (BLOCKER):** `ValueError: Setting with non-unique columns is not allowed`
   / "new components for Generator are not unique" when attaching the 4192 solar p_max_pu series
   (duplicate generator IDs). This is a genuine PyPSA-USA↔pypsa-0.30.2 incompatibility, not config.

## Root cause
Current PyPSA-USA HEAD (`f38d756`, 2026-05-11) does not cleanly build with the env's pinned
package versions — multiple script-level bugs. The existing `elec_s490_c490.nc` was built with an
**older** PyPSA-USA (its baked-in config has `renewable.dataset: None` and no `renewable_scenarios`
— i.e. predates the godeeep/dataset feature). Reproducing it on current HEAD fights these changes.

## Recommended fix (for a clean finish)
Build on the PyPSA-USA commit/release that produced the 490 (older), where add_electricity etc.
work, instead of HEAD:
```bash
cd ~/pypsa-usa-2023/pypsa-usa && git log --oneline   # find a tag around when 490 was built
git checkout <older-tag>                              # e.g. a v0.x release from early 2025
# recreate env from that version's workflow/envs/environment.yaml, then snakemake the same target
```
The 8.5 GB cutout and base data can be reused. Alternatively, debug the add_electricity
duplicate-generator issue in `scripts/add_electricity.py` (the solar p_max_pu attach).

## Config used
`~/pypsa-usa-2023/pypsa-usa/workflow/config/config.yaml` (western, tamu, 2023, clusters=1000,
atlite+era5_2023, onwind dropped, demand=eia). Patched script backup at
`scripts/build_renewable_profiles.py.bak`.
