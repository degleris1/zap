# REVISION PLAN — recasting the DC-placement paper onto the corrected 490-node model

**Status:** tracking doc only. Do NOT edit the `.tex` prose yet — this lists, section by section,
what is contaminated, the corrected number + its source, and a tone note. Edits are deferred to a
later pass. Restraint: keep the paper's structure and voice; change claims and numbers, not the arc.

**Why this exists.** Every quantitative result in the current draft traces to a BROKEN 101-node WECC
setup (`western_small/network_2023.nc`, scaled `load×1.27 / gen×1.24 / lines×0.7`) that produced a
**VOLL=$1000/MWh load-shedding cliff, not congestion** (`cloud_congestion.csv` has system-max-LMP
median = exactly 1000). The corrected science runs on the **490-node network** (`elec_s490_c490.nc`),
cleaned (25 pathological buses zeroed, detected once at load×1.0 and held fixed), must-serve DC,
metrics = **LMP-dispersion (p90−p10) + must-serve feasibility**, incremental to a no-DC base.
Bootstrap CIs are used where computed; seed-pinned wall min/max brackets are seed ranges, not CIs.
See memory `dc-paper-recast-decisions`, `dc-placement-integrated-study`.

## Drift-repair status (2026-06-09)
- `.tex` remains deferred. `development/check_canonical_outputs.py` now passes on
  the citable JSON set; paper prose is still intentionally untouched until the
  next pass.
- Frozen cleaning manifest:
  `development/results/cleaning/bad_buses_490_load1p0_25.json`. It was verified
  against the matching 25-bus lists in `levers_full.json` and
  `placement_congestion/congestion_results.json`; `spread_frontier_full.json`
  records `n_bad=25` but not the actual list, so it is not the manifest source.
  New citable runs must pass this manifest path explicitly with `--bad-buses-json`;
  provenance-by-default is not acceptable for the 24h rerun.
- Wall-statistic hierarchy (locked for the evidence pass): the 24h seed-pinned
  Phase-B Part A artifact is the intended headline once generated; the existing
  12h seed-pinned artifact is a consistency check; the 24-pool lever artifact is
  the lever-frontier statistic and must not be quoted as "the wall."
- `placement_levers/levers_full.json` is now the accepted canonical B=10 lever
  frontier (`n_bad=25`, 12 hours, 24 pools, `_partial: false`) with colocated
  generation, joint gen+tx, and widened uniform transmission. It does not contain
  rerun B=3/B=6 lever cells; use Phase-B Part A for those wall-size claims unless
  lower-fleet lever curves are explicitly rerun.
- `placement_robustness/phaseB_finish_full.json` is legacy `n_bad=17`, not the
  mislabeled 25-bad result. Its Parts B/C remain qualitative/directional only.
  New 25-bad Part A output comes from
  `placement_robustness/phaseB_finish_partA_25bad.json`.
- `placement_levers/colocated_opex.json` is refreshed under the frozen 25-bad
  cleaning set: 7.0 GW onsite gas clears the selected deep-wall 10 GW case,
  with $0.595B/yr annualized capex and $2.064B/yr central fuel OpEx
  (OpEx/capex 3.47x). Treat this as the cost/payer-axis evidence. The accepted
  24-pool lever frontier separately shows colocated generation fully clears the
  B=10 wall at ~8.0 GW onsite for inference ($0.678B/yr capex-only lever axis)
  and ~10.0 GW for training ($0.848B/yr).
- Deliverable-frontier framing decision: free continuous allocation estimates a
  free-allocation hosting level and can be siting-insensitive. It does not
  contradict the spread-frontier result, which studies lumpy, capped, must-serve
  fleets. The paper claim should be: free allocation exposes a high intrinsic
  ceiling; siting/spreading matter because real fleets are lumpy, capped,
  must-serve, and payer-constrained. N-1 deliverable-frontier rows are
  non-citable unless regenerated with a stable N-1 method; guardrail runs use
  `--skip-n1`. The accepted 12-hour guardrail reaches LP-optimal 8.58 GW
  (inference) and 6.72 GW (training) at large kappa, with only 6 alive nodes after
  year-firm screening; policy-order sanity is not a headline because cheap-land
  and grid-strength nearly tie or swap order at high kappa.
- Season robustness is under-documented in the old JSON because window indices were
  not recorded. The script now records them for reruns; until then, do not lean on
  a strong "season-robust" claim.

## Candidate corrected results (accepted JSON gates)
- **HEADLINE wall (24h, generated 2026-06-10, structural checks PASSED):**
  `placement_robustness/phaseB_finish_partA_25bad_24h.json` (8 seeds × 24 pools ×
  24 snaps, panel indices recorded). Brackets are **seed×pool bootstrap CIs**:
  - B=10 inference: concentrate **0.350 [0.288, 0.413]**, distribute **0.748 [0.687, 0.806]** (separated)
  - B=10 training:  concentrate **0.206 [0.151, 0.263]**, distribute **0.648 [0.581, 0.714]** (separated)
  - B=6 separated both workloads; **B=3 inference NOT separated** (0.705 vs 0.896),
    **B=3 training separated** (0.590 vs 0.859) — the wall emerges with fleet size,
    earlier for flat (training) workloads.
  - Seed-range separation also holds at B≥6: distribute's worst seed > concentrate's best.
- Lever frontier: `placement_levers/levers_24h.json` (B=10, 24 pools, 24-hr panel,
  load×1.2, colocated + flexible-DC levers, `_partial: false`); `levers_full.json`
  (12h) is the consistency check — frontier values agree to ~0.5%.
- Onsite OpEx + carbon: `placement_levers/colocated_opex_24h.json` (sanity checks held).
- Wall robustness: `development/results/placement_robustness/wall_robustness_full.json` for cleaning/window sensitivity only after the season-window audit is refreshed or caveated.
- 12h seed-pinned consistency check (`phaseB_finish_partA_25bad.json`; brackets are
  **8-seed ranges**): inference 0.35 [0.18, 0.60] vs 0.75 [0.66, 0.86]; training
  0.21 [0.08, 0.45] vs 0.65 [0.55, 0.73] — matches the 24h headline (panel-robust).
- Independent corroboration: `placement_congestion/congestion_results.json` (H1/H2) + `congestion_sweeps.json` (penetration/load-stress).
- Finishing robustness: old `phaseB_finish_full.json` is legacy `n_bad=17`; Parts
  B/C are qualitative legacy only. `cost_sensitivity_24h.json` reports
  frontier-match transmission cost as primary and Phase-B crossover as secondary
  (generated, both conventions recorded).

---

## The narrative shift (apply throughout)
OLD spine: "distribution lowers LMPs ~42.6% and is ~8× cheaper than transmission."
NEW spine (defensible): **the levers act on different axes.** Distributing a must-serve DC fleet
**clears a deliverability (must-serve feasibility) wall** that concentration triggers — at ~$0 grid
capital, robust to cleaning, pool seed, and panel size. **HEADLINE numbers (24h, bootstrap CIs,
`phaseB_finish_partA_25bad_24h.json`):** at 10 GW the wall is firm and seed-separated
(inference concentrate 0.350 [0.288,0.413] vs distribute 0.748 [0.687,0.806];
training 0.206 [0.151,0.263] vs 0.648 [0.581,0.714]). At 6 GW also separated
(inference 0.532 vs 0.859; training 0.365 vs 0.755). **At 3 GW the wall has not yet
emerged for inference** (0.705 vs 0.896, NOT seed-separated) but **training separates
already at B=3** (0.590 vs 0.859) — fleet size AND workload shape jointly set the
wall's onset. The 12h pin (0.345/0.746; 0.207/0.646) matches the 24h headline almost
exactly — the wall is panel-robust.
Grid-side levers of realistic size do NOT cheaply restore that deliverability:
matching distribution via uniform transmission takes tens of billions per year
and is config/source dependent. In the accepted 24h lever frontier, matching
distribution costs ~$35.7B/yr (inference) and ~$62.8B/yr (training), while even
~$80.0B/yr does not fully clear either workload. The Phase-B Part A discrete
crossover at 24h is **+200% / ~$40.0B/yr for all crossing cells (B=6/B=10, both
workloads)** — the 12h "+100%/$20B" inference cell did not survive the doubled
panel; never quote it for 24h results. **Modest DC demand flexibility does not clear
the wall either** (5/10/20% curtailable: inference 0.186→0.347 max; training
0.083→0.170) — it is dominated by distribution and onsite generation.
*Targeted* cheap transmission ($1–3 M) buys congestion-**dispersion** relief (~23%) but not
deliverability; generation expansion buys little, expensively. Onsite gas fully clears the
wall at low capex (~$0.6–0.85B/yr) but its recurring private cost is $2.0B/yr fuel, rising to
**$3.1B/yr (≈5.2× capex) with a $50/tCO₂ carbon price** — the payer-axis result.
**A second-order nuance:** spreading
is not monotonic — a *moderate* spread (~40 sites) is most deliverable, while over-spreading onto many
weak nodes erodes it (qualitative trend).

---

## Section-by-section

### Abstract (`main.tex`)
- **REMOVE** "reduces peak LMP impacts by over 42.6%" and "deploying 1 GW more … before the same
  constraints bind" and the "minimizes dispatch costs and LMPs" bilevel claim — all 101-node artifacts.
- **REPLACE** with the feasibility-wall result + the different-axes framing. Tone: lead with
  deliverability, state CIs, avoid a single false-precision LMP percentage.

### Intro (`intro.tex`)
- **Figure `fig:cloud-plots`** (`dispatch_cost_percent_increase.pdf`, `max_lmp_increase.pdf`) → ARTIFACT
  (101-node VOLL cliff, source `distributed_investment2.py`/`eenergy-plotting.ipynb`). **Replace** with a
  490-node figure: per-node marginal congestion-impact landscape (H2, from `congestion_results.json`)
  and/or the feasibility-wall-vs-penetration plot (`congestion_sweeps.json`,
  `placement_levers/levers_frontier.png`).
- The "knee points are location-dependent" paragraph survives in spirit (H2 shows a sparse chokepoint
  tail) but must be re-sourced to the 490-node landscape.
- **Contributions list:** drop "minimizes LMPs", "42.6%", "8× cheaper". Reframe around (i) the
  feasibility wall, (ii) different-axes lever comparison with an honest uniform-tx cost, (iii) workload
  (training vs inference) sensitivity, (iv) robustness.

### Problem setting + Methods (`problem-setting.tex`, `methods.tex`) — FORMULATION REALIGNMENT
- **`eq:h_lmp` (smooth-max LMP objective) is NOT what produced any corrected result.** Decision:
  state the optimized objective as the **smooth line-utilization surrogate Σu²**
  (`LineUtilizationObjective`), and present **LMP-dispersion + must-serve feasibility as evaluation
  outcomes** (co-equal). Rationale to include briefly: on this LP/linear-cost grid raw nodal LMP has a
  zero gradient when uncongested and saturates at VOLL when shedding, so it is not a usable
  optimization target (this is a legitimate methods point, not a limitation to hide).
- **Method = DUAL-TRACK (state explicitly which engine produced which result):**
  - Headline congestion/lever/wall results come from **sampling/enumeration over realistic placements
    + plain DC-OPF dispatch + bootstrap CIs** (NOT the bilevel PGD). Grid levers are evaluated by
    capacity expansion (uniform sweep for the deliverability question; dual-ranked iterative expansion
    for targeted dispersion relief), not PGD.
  - The **differentiable bilevel PGD planner** is presented only for the narrower DC-*capacity*
    optimization where it is numerically reliable (`dc_placement_optimize.py`). Note it is unreliable
    at 490-node scale when differentiating w.r.t. 1250 line / 844 gen capacities (non-optimal solver
    status) — so it is not used for the grid-lever or wall results.
- **Caveats to note for `LMPObjective` if h_LMP is kept anywhere:** the code's `meansmoothmax`/
  `sumsmoothmax` smooth over the TIME axis then reduce over nodes — the transpose of written
  `eq:h_lmp` (smooth-max over nodes per time, summed over time); and `LMPObjective.is_convex`
  returns `True` though Λ(η)=−dual is not convex in η. Cleanest path: drop h_LMP from the formulation
  (per above) so these never matter.
- Keep the dispatch DC-OPF formulation, the cost models (DC build $12M/MW + land, CRF 20yr@7% — this
  is implemented and correct), and the projection/budget machinery (`BoxBudgetProjection` verified).

### Experiment (`experiment.tex`) — REWRITE the network description
- **CURRENT:** 101-node, `load×1.27 / gen×1.24 / lines×0.7`. **REPLACE** with: 490-node
  `elec_s490_c490.nc` (1250 AC lines, genuinely congested at natural caps — no line de-rating needed);
  the cleaning protocol (25 bad buses detected once at load×1.0, held fixed; excluded from candidate
  pools and price stats; base shed ≈ 0); load×1.2 stress; a 12-hour panel **stratified across the
  whole year** (each hour dispatched independently at T=1 because storage couples timesteps).
- Drop the manufactured-congestion justification entirely.

### Results (`results.tex`) — REPLACE Table 1 and §4.1/§4.2
- **`tab:synopsis_tradeoffs_singlecol_ops` (Base 196.78 / SN 378.4 / Dist 217.2 / +tx / +gen, 42.6%,
  8×) → DELETE (all 101-node VOLL artifacts).** Replace with a **lever cost-effectiveness table** from
  the accepted B=10 `levers_full.json`:
  - **Feasibility wall** (must-serve feasible fraction; 8-seed-pinned, HEADLINE =
    `phaseB_finish_partA_25bad_24h.json`, bootstrap CIs; 12h
    `phaseB_finish_partA_25bad.json` is the consistency check). Lead with B=10
    (inf 0.350 [0.288,0.413] vs 0.748 [0.687,0.806]; train 0.206 vs 0.648);
    B=6 also separated; **B=3 inference NOT separated (wall not yet emerged),
    B=3 training already separated**. The old `phaseB_finish_full.json` values
    are legacy `n_bad=17` and must not be quoted next to 25-bad headlines.
  - **Uniform transmission** ("$ to match distribution / clear the wall"): the accepted 24h
    lever frontier says matching distribution costs **$35.7B/yr** (inference) and
    **$62.8B/yr** (training); even the widest uniform sweep, **$80.0B/yr**, only
    reaches ~0.87 / 0.84 feasibility and does **not** fully clear. The Phase-B
    Part A discrete crossover (secondary convention) at 24h is **+200% /
    $40.0B/yr for all crossing cells** (B=6 and B=10, both workloads); the 12h
    "+100%/$20B" inference cell did not survive the doubled panel. Report the
    achieved-feasibility-vs-X curve; never a single flat "$X clears it."
  - **Targeted transmission** (~$1–3 M): dispersion relief inf B10 119→92 $/MWh (~23%), train 183→97
    (~47%); ~no feasibility gain. Caveat: the dispersion-vs-$ outcome curves carry
    pool-level noise (7 logged monotonicity violations at 24h) — present dispersion
    as a mechanism signal, not a headline.
  - **Flexible DC (new lever, 24h):** 5/10/20% curtailable on the concentrated fleet
    raises firm feasibility only to 0.347 max (inference) / 0.170 (training), and the
    curtailable slice is served ~31% of hours — **demand flexibility of realistic size
    does not clear the wall**; dominated by distribution and onsite generation.
  - **Co-located generation**: the only swept lever that fully clears the B=10
    wall in both workloads, at ~8.0 GW onsite for inference ($0.678B/yr) and
    ~10.0 GW for training ($0.848B/yr) in capex-only lever accounting. Pair it
    with `colocated_opex_24h.json`: the deep-wall case needs 7.0 GW onsite gas,
    $0.595B/yr capex, $2.007B/yr central fuel OpEx (3.38× capex), and **+$1.06B/yr
    carbon at $50/tCO₂ → $3.07B/yr recurring (≈5.2× capex)** ($25/$50/$100
    sensitivity recorded), borne privately by the operator rather than the
    grid/ratepayer.
  - **Generation**: little congestion relief; do NOT state the "500×" tx/gen ratio (dual-sparsity
    artifact — 21/1250 lines vs ~230/844 gens, not comparable levers).
- **§ Distribution as a load-side lever (`dc_allocation_bar_plot`, `kde_p95_lmp`):** P95-LMP KDE figure
  is a 101-node artifact → replace with the feasibility-wall + LMP-dispersion (incremental) figures
  from the corrected study. The qualitative claim "optimized/distributed beats single-node" survives;
  the mechanism changes from price-relief to deliverability + dispersion.
- **"Dispatch cost nearly constant" paragraph:** was spun as reassuring; it is actually the artifact's
  tell (flat cost + swinging LMP = a few VOLL nodes). Remove or recast.
- **Workload section:** training (flat) is consistently worse than inference (diurnal) at every
  operating point (wall harder, feasibility lower) — this survives and strengthens; cite the per-workload
  feasibility numbers.

### Motivation / Related (`motivation.tex`, `related.tex`)
- Largely survive (qualitative). Ensure the "distribution shifts the knee" language maps to the
  feasibility-wall result, not the deleted price claims.

---

## Robustness round — gated by canonical rerun

The old Phase-B-finish run is useful for direction and code audit, but it used
`n_bad=17`. The citable Part A HEADLINE is now
`phaseB_finish_partA_25bad_24h.json` (24 snaps, bootstrap CIs, panel indices
recorded); `phaseB_finish_partA_25bad.json` (12h) is the consistency check.
Partial files carry `_partial: true` and must not be cited.

**[x] 1. Seed-pinned wall for B=3/6/10** (`phaseB_finish_partA_25bad_24h.json`
partA; 8 seeds × 24 pools × 24 snaps). B=10 and B=6 are seed-separated for both
workloads. **B=3: inference NOT separated (0.705 vs 0.896 — the wall has not
emerged at 3 GW); training separated (0.590 vs 0.859 — flat workloads hit the
wall earlier).** The 12h pin matches B=10 almost exactly (panel-robust).

**[x] 2. Line-build crossover** (`phaseB_finish_partA_25bad_24h.json` partA
`crossover_to_dist`). At 24h ALL crossing cells (B=6 and B=10, both workloads)
cross at **+200% uniform line reinforcement / $40.0B/yr**; the 12h B=10
inference "+100%/$20B" cell did not survive the doubled panel (the crossover is
a discrete-grid statistic — quote it as a grid cell, not a precise dollar).
B=3 is a non-crossover fleet (uniform curve skipped). NEVER write "$20B clears
the wall"; for 24h results the secondary convention is $40B. The lever-frontier
match costs ($35.7B/$62.8B at 24h) remain the primary convention; inference's
$35.7B sits inside the $20–40B grid cell, so the conventions are coherent.

**[x] 3a. Cost sensitivity** (`cost_sensitivity_full.json`; analytic; `cost_sensitivity_24h.json`
for the new pass). Deliverability is **cost-independent — only dollar figures rescale, all linearly.**
The 24h cost table must make the lever-frontier match convention primary and record the Phase-B
crossover as secondary. DC investment $8.5/11.3/14.2 B/yr for build cost $9/12/15 M/MW (10 GW);
the land term is immaterial (<0.01% of DC cost). State the dollar claims as robust across plausible
cost ranges.

**[x] 3b. Chunking + siting robustness** (`phaseB_finish_full.json` partB/partC;
legacy `n_bad=17`, lighter sampling).
The **mean** wall (distribute ≥ concentrate) persists across every chunking definition (spread
20/40/80 × clump 0.5/1/2 GW) and under both cheap-land and uniform siting — state as "direction
preserved across chunking and siting," NOT "statistically significant under chunking" (most cells are
not seed-separated). The **spread sweet-spot** (moderate spread ~40 sites best; over-spreading erodes
deliverability) is a **qualitative trend** (underpowered) — present it as such, not a significant result.

## Remaining (not blocking)
- Final numbers-audit pass on whatever lands in the `.tex` (the JSON-level numbers are critic-verified).
- The actual `.tex` prose edits (deferred to the next pass, guided by this doc).
- Deferred SHOULD robustness not validated in this pass: exact bad-count `{17,25,30}`,
  shed-tolerance sensitivity, solver cross-check, and refreshed spread-frontier provenance.
