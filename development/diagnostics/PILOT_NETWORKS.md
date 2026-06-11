# Multi-network pilot verdicts (2026-06-11)

Pilot tool: `development/diagnostics/pilot_network.py` (uses the studies' own
loader/metrics: `load_pypsa_network`, POWER_UNIT=1e3, COST_UNIT=100). Run before
ANY citable study on a new network.

> **STATUS 2026-06-11: FIXED.** All three broken builds were rebuilt and now pass
> the pilot + a `dc_placement_study` datacenter smoke test. See "RESOLVED" below.

## Verdicts

### CURRENT (after the fix)

| network | buses | AC lines | lines/bus | islands | largest island | base shed ×1.0 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `western_elec_s490_c490.nc` (reference) | 490 | 1250 | 2.55 | 1 | 490 | ~0% | **GO** (canonical) |
| `elec_s500_c500.nc` (texas/ERCOT) | 500 | 1065 | 2.13 | 2 | 499 | 0.00% | **GO** (rebuilt) |
| `elec_s1493_c1493.nc` (western hi-res) | 1493 | 3012 | 2.02 | 1 | 1493 | 0.00% | **GO** (rebuilt) |
| `elec_s3000_c3000.nc` (eastern) | 3000 | 7884 | 2.63 | 2 | 2998 | 1.55% | **GO** (rebuilt) |

Rebuilt files: `/scratch/users/gfw/pypsa-usa/resources/Default/{texas,western,eastern}/`.
The 1–2 residual islands are single stray buses (the giant component holds all but
1–2); routine `find_bad_buses`/`clean_devices` candidates, not structural breakage.
All three run pilot + datacenter smokes end-to-end with genuine congestion (copperplate
risk: low). intra-zone lines: texas 976, western 2603, eastern 6163 (were ~0 when broken).

### ORIGINAL (broken builds, pre-fix — kept for the record)

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

## RESOLVED (2026-06-11): fix applied + all three rebuilt and smoke-tested

Two code changes, then a rebuild of only the `cluster_network` step (the cached
simplified `elec_s{N}.nc` intermediates predate the calibration, so no cutout/
renewable rebuild was needed — `snakemake --configfile config/config.{texas,default,
eastern}.yaml resources/Default/{ic}/elec_s{N}_c{N}.nc`):

1. **`pypsa-usa/workflow/scripts/cluster_network.py`** (`calibrate_tamu_transmission_capacity`,
   the `lines_not_in_reeds` branch ~line 745): guard the removal with `if region0 != region1:`
   so intra-zonal lines (no REEDS interface by construction) are kept. Rebuild logs now read
   e.g. eastern "1712 lines updated, **39** removed" (was: nearly all intra-zonal removed).

2. **`zap/zap/importers/pypsa.py`** (`load_pypsa_network`): drop empty device classes
   (`devices = [d for d in devices if d.num_devices > 0]`). ERCOT has **zero DC links**, and
   zap otherwise built `cp.Variable((0,T))` → CVXPY "Invalid dimensions (0,1)". Because this
   makes the device list network-dependent, the consumers now locate devices **by type**:
   added `dev_index(devices, name)` in `dc_placement_study.py` and replaced every fixed
   ACLine index (`devices[3]`, `oc.power[3]`, `local_inequality_duals[3]`, `panel[0][1][3]`)
   in `dc_placement_study.py`, `dc_placement_integrated.py`, and `pilot_network.py`.
   Behavior-preserving for nets that have DC links (ACLine stays at index 3). Generator(0)/
   Load(1) precede DCLine so their indices never move.

**Pilot numbers (×1.0 / ×1.2):** texas LMP-disp ~3–60 $/MWh, binding 8–35, shed ≤0.5%;
western disp ~10–635, binding 8–17, shed ≤0.8%; eastern disp ~30–173, binding 28–41,
shed 0.6–2.9%, load 321 GW (matches the projection below). All "copperplate risk: low".
**Datacenter smoke (`dc_placement_study`, tiny knobs)** completed on all three incl. the
no-DC texas: base grid-stress / p95-price / shed = texas 76 / $46 / 0%, western 112 / $55 /
0%, eastern 410 / $77 / 1.55%; DC-growth, BIG-vs-SMALL, HEADLINE, JSON+PNG all produced.

Still TODO before *citable* runs (not blockers for "the build works"): re-derive each net's
cleaning manifest (the 25-bad list is WECC-490-specific) for the 1–2 stray buses; the
eastern 2022-vs-2023 weather caveat stands.

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
