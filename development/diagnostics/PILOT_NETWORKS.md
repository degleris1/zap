# Multi-network pilot verdicts (2026-06-11)

Pilot tool: `development/diagnostics/pilot_network.py` (uses the studies' own
loader/metrics: `load_pypsa_network`, POWER_UNIT=1e3, COST_UNIT=100). Run before
ANY citable study on a new network.

## Verdicts

| network | buses | AC lines | lines/bus | islands | largest island | stranded load buses | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `western_elec_s490_c490.nc` (reference) | 490 | 1250 | 2.55 | 1 | 490 | 0 | **GO** (the canonical net) |
| `texas_elec_s500_c500.nc` | 500 | **89** | 0.18 | **419** | **18** | 392/494 | **NO-GO — broken build** |
| `eastern_elec_s3000_c3000.nc` | 3000 | **1721** | 0.57 | **1546** | 880 | 1450/2994 | **NO-GO — broken build** |

A connected N-bus network needs ≥ N−1 branches; both new builds are far below
that, so the breakage is structural (missing lines), not parametric.

## Symptoms (why results from these files would be garbage)

- **Texas**: zero `links` → zap importer crashes outright
  (`cp.Variable((0,1))` for the empty DCLine device). Even if patched, 419
  islands make dispatch meaningless.
- **Eastern**: dispatches "succeed" but shed **~40% of base load at load×1.0**
  (WECC-490 sheds ~0.5–4% pre-cleaning), LMP dispersion ~$1850/MWh (VOLL
  pockets on stranded buses), gen marginal cost max $3818/MWh. Every island
  balances locally; "deliverability" is undefined. Any wall measured here would
  be an artifact of disconnection, not congestion.
- Eastern is a **2025 weather year** build (WECC is 2023) — fine for
  generalization once fixed, but note the cross-year difference.

## DIAGNOSED (2026-06-11, CONFIRMED in code): upstream TAMU "calibration" deletes intra-zonal lines

**Root cause (supersedes the ReEDS-default hypothesis below, which the user's configs
disproved — they DO set `transmission_network: 'tamu'`):** PyPSA-USA HEAD `f38d756`
(2026-05-11) added `calibrate_tamu_transmission_capacity()` in
`workflow/scripts/cluster_network.py` (~line 589), which runs unconditionally for
every TAMU build. It keys every clustered line by its endpoint zones
(`"p10-p40"`) and **`mremove`s any line whose key is absent from the REEDS
interface table**. REEDS interfaces are inter-zonal only, so ALL intra-zonal
lines (`"p10-p10"`) are deleted — correct only when clusters == zones, grid-
shredding at sub-zonal resolution. Build logs show
`"N lines removed (not in REEDS data)"`. The healthy 490 predates this commit.

**Fix:** patch the removal with `if region0 != region1:` (keep intra-zonal lines),
or pin pypsa-usa before `f38d756`; give each build a distinct `run: name:`;
force-rerun from `cluster_network`; re-verify with `pilot_network.py`
(expect 1 sub-network, lines/bus ≈ 2–3, intra-zone lines ≫ 0). Note: the new
builds carry **2022 weather** relabeled to 2025 horizons (WECC-490 is 2023 weather).

## (superseded) earlier hypothesis: ReEDS zonal topology instead of TAMU nodal

The new `elec_s1493_c1493.nc` WECC rebuild is broken identically (409 lines /
1493 buses, 1120 islands, largest 97, all buses v_nom=230) — so the bug is in
the shared pipeline config, not interconnect-specific. The smoking gun:

- healthy 490: **996 intra-zone + 254 cross-zone** lines (zones = the `p##` in
  bus names like `p10 0`);
- broken 1493: **0 intra-zone, 409 cross-zone** — every surviving line spans
  two different `p`-zones. (Stale-busmap mixing was tested and ruled out: only
  8/409 of the broken net's line pairs exist in the healthy 490 line list.)

Zero intra-zonal transmission is the signature of building on the **ReEDS
zonal/interface (ITL) network** — which only has inter-zonal corridors — while
clustering to many buses per zone. The build kit's own config warns about
exactly this and overrides it (`development/build_2025/config.western_2025_highres.yaml`):

```yaml
# The repo default is the ReEDS NARIS network, whose western zonal min is only 34 nodes,
# so it CANNOT cluster to 1000. The TAMU synthetic nodal network supports an arbitrary
# cluster count ...
model_topology:
  transmission_network: 'tamu'
  topological_boundaries: 'county'
```

**Fix:** the configs used for the texas/eastern/1493 builds are missing (or not
applying) this `model_topology` block, so snakemake fell back to the repo
default ReEDS topology. Add the block to each config (with the right
`interconnect:`), force-rerun from the network-building rule (don't trust
cached intermediates: `snakemake --forceall` or delete
`resources/<run>/elec_base_network*.nc` + downstream), and re-verify with
`pilot_network.py` — expect lines/bus ≈ 2–3 and 1 sub-network.

The healthy Feb-5 490 build used TAMU (`build_2025/BUILD_2023_STATUS.md`:
"TAMU base network + bus regions + ...").

## Useful numbers captured for when fixed builds arrive

- Eastern total load ≈ 540 GW raw peak / ~321 GW at the zap-scaled peak hour →
  **penetration-matched fleet ≈ 41 GW** (vs WECC-490's B=10 at 79 GW).
- ERCOT peak (typical) ~85 GW → expect a B≈10–11 GW match; re-derive from the
  fixed file.
- Eastern dispatch speed at 3000 buses is NOT a blocker: ~0.3 s/hour solve
  (CLARABEL), same order as WECC-490 — a full 8-seed×24-pool×24-snap wall
  replication is compute-feasible (<2h/workload) once the build is fixed.
- Per-network cleaning manifests must be re-derived (the 25-bad list is
  WECC-490-specific), and `dc_placement_levers.py` panel-index recording should
  be patched first.
- If a fixed ERCOT build still has zero DC links, the zap importer needs a
  small fix (skip empty device classes); study scripts also index devices
  positionally (0=gen, 1=load, 3=ACLine) — they must look devices up by type
  before running on any network whose device list differs.

## Housekeeping

`~/Downloads/elec_s490_c490.nc` was renamed to `western_elec_s490_c490.nc`;
a symlink now preserves the old path that every script default points at.
