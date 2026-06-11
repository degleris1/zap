# DC-Placement Degeneracy: Diagnosis, Fixes, and a Working Pipeline

**Question we were handed:** distributing data centers across WECC with a bi-level
optimizer barely changed LMPs/congestion on the 101-node network — you could
inject up to ~2 GW at many nodes and see *nothing*. Three hypotheses: (1) too few
nodes, (2) units off-scale, (3) the problem is set up wrong. Goal: be able to ask
"I have X GW to place over WECC — what are the effects?" and to show that
*distributing* compute (100×1 MW) stresses the grid less than *concentrating* it
(1×100 MW).

## TL;DR

- **Hypothesis 1 (too few nodes) is the dominant cause — confirmed.** At *natural*
  line capacities the candidate-node LMP spread is **3.3 $/MWh on the 101-node
  network vs 348 $/MWh on the 490-node network** — a ~100× difference from spatial
  resolution alone. The clustered 101-node WECC behaves as a **near-copperplate**
  (transmission corridors are over-aggregated, so a few GW of load is re-served by
  the same marginal generators with no price impact). The 490-node network is
  genuinely congested (22 binding lines at base) and gives a rich, non-degenerate
  placement problem **with no line-crushing required**.
- **Hypothesis 2 (units) is mostly a red herring.** Power is correctly in GW and
  `LMP × 100 = $/MWh` is correct (generator marginal costs 0–63 $/MWh, base LMP
  median ≈ 30 $/MWh — physically sane). The one real bug: the **`× 4` LMP
  multiplier in `pushing_capacity.py` is wrong** (it's the upsample factor; a
  per-timestep price must not be multiplied by it). It inflates reported $/MWh 4×
  but cancels in *relative* comparisons. Also note `Load.load` stays in **MW** — the
  GW conversion lives in `Load.nominal_capacity (= 1/power_unit)`.
- **Hypothesis 3 (setup) is real and compounding.** On the 101-node network the
  team manufactured congestion by scaling line capacities to 0.5–0.7× and crushing
  3 specific lines by hand — arbitrary and non-reproducible. Worse, `load*1.27`
  pushes the system to a **13% reserve margin**, so concentrated injections jump
  straight from "uncongested" to **load-shedding at VOLL ($1000/MWh)** with no
  smooth congestion band in between. That cliff — not units — is what makes the
  LMP-based gradients degenerate.

## What we measured (reproducible)

All scripts in `development/diagnostics/`. Units: `power_unit=1e3` (GW),
`cost_unit=100` (`LMP×100 = $/MWh`), `VOLL=$1000/MWh`.

### 1. The 101-node grid is a near-copperplate
`probe_node_sensitivity.py` injects a 2 GW block at each candidate node:

| Regime | dLMP/GW across candidate nodes |
|---|---|
| Natural caps (`line*1.0`) | **0.26 – 0.83 $/MWh per GW** → 2 GW moves local LMP ≤ $1.7. Degenerate. |
| De-rated (`line*0.7`) | 1.4 – 133 $/MWh per GW → placement matters, but it is *manufactured* by the de-rating. |

Even concentrating **8 GW** at one node only moves the smooth stress proxy
`sum u²` from 79 → 83. Placement genuinely does not matter here.

### 2. Units are fine; only the `×4` LMP factor is a bug
`probe_units.py`: gen marginal costs 0–63 $/MWh, base LMP median 30.2 $/MWh, **0%
load shed at base**. Upsampling 24→96 scales the dispatch *cost* by 4.01× (it's 4
copies of a day) but leaves the per-timestep LMP unchanged (30.17 → 30.33) — so the
`np.mean(...) * 100 * 4` in `pushing_capacity.py` overstates LMP by 4×.

### 3. The 490-node grid is genuinely congested
`probe_congestion.py` / `dc_placement_study.py` on `elec_s490_c490.nc` (1250 AC
lines vs 251), natural caps, Aug-23 snapshot:

- Base: **22 binding lines**, p95 LMP = 6× median (vs 1.1× on 101-node).
- Candidate-node probe-LMP spread: **14.7 – 362.6 $/MWh** (range 348).
- Siting is wildly heterogeneous: node 200 absorbs 3 GW for **+3 $/MWh**; node 20
  is fine to 2 GW then **cliffs to VOLL** at 3 GW; nodes 150/350/50 are **infeasible**
  at 2–3 GW (hard chokepoints).

## The working pipeline — `development/dc_placement_study.py`

Run:
```bash
.venv/bin/python development/dc_placement_study.py \
    --network ~/Downloads/elec_s490_c490.nc --snap-len 4 --budget 3.0
```
Outputs `development/results/placement_study/{study_results.json, placement_study.png}`.
It (a) auto-selects the most-congested hour in a window, (b) ranks nodes by
marginal stress / feasibility, (c) draws **per-node siting curves** (Q1), and
(d) compares **concentrate vs distribute** across a budget sweep (Q2). It uses
robust metrics (median / p95 LMP clipped to VOLL, DC-terminal LMP, `sum u²`,
binding-line count, % load shed) to tolerate the 490-node's weakly-connected
price pockets.

### Headline results (490-node, natural caps, Aug-23 peak hour)

**Q1 — placement effects are large and heterogeneous** (answers "what happens if I
put X GW here?"): some nodes (200, 300) absorb several GW with ~no price impact;
others (20, 480) are fine until a threshold then cliff to VOLL; others (150, 350,
50, 400) cannot take 2–3 GW at all. A DC operator's siting choice swings the
local price from **~$24 to $1000/MWh**.

**Q2 — distributing beats concentrating, and the gap grows with scale:**

| total budget B | concentrate @ best single node | distribute over 5 nodes |
|---:|---|---|
| 3 GW | `sum u²` 295, feasible | `sum u²` 274 |
| 4 GW | `sum u²` 318, feasible | `sum u²` 278 |
| **5 GW** | **INFEASIBLE** (grid can't serve it) | `sum u²` 287, feasible |
| 8 GW | INFEASIBLE | `sum u²` 315, feasible |

Concentrating even at the *best* node hits a feasibility wall at 5 GW; the same
budget spread over 5 nodes stays feasible past 8 GW at lower stress. Concentrating
at a *chokepoint* (node 20) drives p95 LMP to $235 and sheds 0.7% of load at just
3 GW, vs $151 and no incremental shed when distributed. This is the
"100×1 MW < 1×100 MW" result — demonstrated with no hand-tuned line limits.

Contrast: the identical pipeline on the 101-node network shows concentrate ≈
distribute (`sum u²` ~74–83 either way, all nodes feasible to 8 GW) — i.e. the
degeneracy, reproduced.

## Recommended setup going forward

1. **Use the 490-node network** (`elec_s490_c490.nc`) for any placement study. The
   101-node clustering is too aggregated to study transmission-driven siting.
2. **Drop the line-crushing and the `line*0.5–0.7` hacks.** They are unnecessary at
   490 nodes and make results depend on arbitrary choices.
3. **Don't over-scale load into the VOLL cliff.** Use `load*1.0` (≈34% reserve) so
   the binding phenomenon is *congestion* (smooth) rather than involuntary shedding
   (a cliff). Pick a genuinely congested hour (the script auto-selects one).
4. **Optimize on the smooth convex stress proxy, not raw LMP.** `LineUtilizationObjective`
   (Σ u²) already exists for exactly this reason ("create gradients even when no
   line constraints bind"); raw LMP gradients are zero in uncongested regions and
   saturate at VOLL in shed regions. Report LMP percentiles / DC-terminal LMP as
   *outcomes*.
5. **Fix the `×4`** in `pushing_capacity.py` LMP reporting (and prefer not
   upsampling-by-repeat; it adds no information and muddies annualization).
6. **N-1 / SCOPF is the principled way to add more binding constraints** if you want
   the 101-node or want even more congestion: post-contingency limits bind smoothly
   and continuously, no hand-crushing needed. The old `scopf_dc_planning_demo.py`
   notebook/script is deleted as superseded; use `dc_placement_scopf.py` for the
   exact-N-1 greedy path and `dc_n1_overload*.py` for the citable full-LODF
   robustness checks.
7. **Handle 490-node price pockets:** a few weakly-connected buses show LMPs of
   ±10⁴ $/MWh (interior-point conditioning near binding duals / curtailment
   pockets). Clip to `[0, VOLL]` for reporting, or inspect/ground those buses.

## Gradient-based planner on 490-node + optimization-correctness fixes

We wired up the *actual* zap bi-level planner (`development/dc_placement_optimize.py`):
`minimize_{dc_capacity} LineUtilizationObjective(Σu²) + InvestmentObjective` s.t.
`0 ≤ dc_capacity_i ≤ per_site_cap`, `Σ dc_capacity_i = budget`, via projected
gradient descent with `BoxBudgetProjection`. It is **tractable** (forward+backward
≈ 0.2 s each on 490 nodes; a 60-iteration solve ≈ 24 s).

**Realistic fine-grained placement** (per-site cap = 250 MW, 31 candidate sites):
the optimizer spreads the budget over 18–24 sites, respects the cap, and beats the
baselines — and its edge **grows with budget** (more stress → more value from smart siting):

| Budget | optimized Σu² (sites) | uniform | greedy-cheap | worst-pack | optimizer vs uniform |
|---:|---|---|---|---|---|
| 2 GW | 136.8 (18) | 143.8 | 138.3 | 155.5 | **−4.9%** |
| 4 GW | 138.4 (24) | 150.2 | 142.2 | 157.4 | **−7.9%**, DC-LMP $59→$57 |

### Critic-verified gradient correctness
A finite-difference check (`development/diagnostics/critic_gradcheck.py`) confirms the
implicit-diff gradient of `dc_capacity` through `LineUtilizationObjective` is correct
to **3e-11** (machine precision) in smooth regions, 7e-8 with a loaded line. The
degenerate `min_power == max_power` encoding of the DC load differentiates correctly.
Crucially, `LMPObjective` has a **genuinely zero gradient in flat-price (uncongested,
linear-cost) regimes** — correct behavior, and exactly why `LineUtilizationObjective`
(Σu², always-nonzero gradient) is the right optimization target. A quadratic generation
cost would also make LMP objectives smooth/responsive.

### Bugs found and FIXED in the planning library
A projection/solver critic (`development/diagnostics/critic_projections.py`) found two
real bugs — the user's suspicion about the simplex/box constraints was correct:

1. **`SimplexBudgetProjection` pre-clipped negatives** (`np.maximum(x,0)` before the
   Duchi algorithm) → wrong Euclidean projection whenever the input had negative
   coordinates (which every gradient step produces). E.g. `[-1,-2,-0.5,-3,-0.1]`,
   budget 2.5 returned `[0.5×5]` vs the true `[0.367,0,0.867,0,1.267]` (error 0.77).
   **Fixed** in `zap/planning/projection.py` (run Duchi on the raw input; keep the
   `≥0`-clip shortcut only for the `strict=False` ≤-budget case). Now matches an
   independent QP to 3.6e-7, and projected-gradient descent converges (gap −4e-9,
   was stuck at 1e-2).
2. **`lower/upper_bounds` were silently ignored during `solve()`** (the per-coord clip
   in `project()` is commented out). So `SimplexBudgetProjection` + per-site
   `upper_bounds` silently violated the caps — a **live bug in
   `development/workload_variation_tb.py`**, which sets 0.25 GW caps with a simplex
   projection (one node could get 2.5 GW = 10× its cap). **Fixed** by adding a guard in
   `solve()` that raises a clear error in that case, directing you to
   `BoxBudgetProjection` (which the critic verified is correct). ⚠️ This means
   `workload_variation_tb.py` will now error until you switch it to
   `BoxBudgetProjection(budget, lower_bounds, upper_bounds)`.

Also fixed: the gradient-step branch now uses the budget-tangent-projected gradient
(`grad − mean`) for *both* simplex and box budget projections (was box-only), and the
unconditional `print(grad)/print(proj_grad)/print(state)` debug spam in the solve loop
is gated behind `verbosity >= 3`.

## Granularity: how fine should DC siting be?

`development/dc_granularity_sweep.py` sweeps the per-site cap over a dense candidate
set (77 feasible nodes, stride-6 sampling of the 490-node grid, chokepoint pockets
pre-screened out), fixed budget 3 GW, and runs the gradient optimizer for each cap:

| per-site cap | optimized Σu² | uniform | optimizer gain | opt p95-LMP | #sites used |
|---:|---|---|---|---|---|
| 500 MW | 127.0 | 137.1 | **7.3%** | $113 | 22 |
| 250 MW | 129.4 | 137.1 | 5.6% | $131 | 25 |
| 100 MW | 132.8 | 137.1 | 3.1% | $136 | 60 |
| 50 MW | 134.4 | 137.1 | 1.9% | $142 | 62 |

Two honest takeaways: (1) **finer = more realistic/robust but lower optimizer value** —
when sites must be tiny, even naive uniform is *forced* to spread, so smart placement
helps less (7.3%→1.9%); (2) the lowest achievable stress actually *rises* as caps
shrink, because coarse caps let the optimizer pack the prime nodes. Also note: **every
DC placement here sits below the no-DC baseline (Σu²=141.7)** — at this hour, distributed
DC load *relieves* congestion (it absorbs local generation that would otherwise overload
export corridors), and optimized placement relieves it most. "Increase grid granularity"
= denser candidate sampling (we only have the 490-node network locally; a finer base grid
would need a higher-cluster PyPSA-USA build — see `development/PYPSA_USA_HIGHRES_2025.md`).

**On pulling a higher-resolution / 2025 network (checked 2026-06):** not obtainable in this
environment. PyPSA-USA doesn't distribute prebuilt clustered networks, and its weather cutouts
exist only for **2019–2023** (2024/2025 → HTTP 404), so a 2025 network needs a custom ERA5-2025
cutout (atlite + Copernicus CDS key) + a multi-hour conda build (567 MB bundle + 4.95 GB cutout).
The study pipeline is now **year-aware** (`--year`, `resolve_year_window()`): point it at any such
network and it selects a summer-peak window in the requested year (or warns + falls back). Exact
build recipe in `development/PYPSA_USA_HIGHRES_2025.md`; a **runnable build kit** (config +
`build_2025.sh` + `verify_2025_network.py`) is in `development/build_2025/`.

> **UPDATE 2026-06-11 — higher-resolution networks ARE now built (supersedes the
> "not obtainable" note above).** Three higher-res TAMU/2025 PyPSA-USA networks were
> built on Sherlock and fixed (see `diagnostics/PILOT_NETWORKS.md` for the build bug
> and fix): ERCOT `elec_s500_c500` (500 buses / 1065 lines), a hi-res WECC
> `elec_s1493_c1493` (1493 / 3012), and the Eastern Interconnect `elec_s3000_c3000`
> (3000 / 7884), all at `/scratch/users/gfw/pypsa-usa/resources/Default/{texas,western,eastern}/`.

## Multi-network generalization (2026-06-11): the headline reproduces beyond WECC-490

Ran `dc_placement_study.py` (the Q1 siting + Q2 concentrate-vs-distribute pipeline,
natural caps, `load*1.0`) on all three new networks. **The WECC-490 headline is NOT an
artifact of that one network — "distribute beats concentrate, concentrate hits a
feasibility wall first" reproduces, and sharpens on the larger, more-congested grids.**

| network (buses, AC lines) | base Σu² / p95 LMP / shed | distribute-wins frac (small<big) by budget | concentrate (1-GW sites) |
|---|---|---|---|
| `elec_s500_c500` ERCOT | 76 / $46 / 0% | 0.20 → 0.60 → 0.80 (B=2,4,6 GW) | feasible to 6 GW (267% reserve) |
| `elec_s1493_c1493` WECC hi-res | 103 / $55 / 0% | 1.00 / 0.60 / 0.40 (B=3,6,9) | **INFEASIBLE at B=3 and B=9** |
| `elec_s3000_c3000` Eastern | 391 / $77 / 1.5% | **1.00 at every budget** (B=4,8,12) | **INFEASIBLE at B=8 and B=12** |

- **Eastern is the cleanest reproduction:** concentrating 1-GW sites is INFEASIBLE at 8 GW
  while the same budget over 40–60 small sites stays feasible at ~+3% stress; distribute
  wins 100% of matched random draws. This is the "1×100 MW ≫ 100×1 MW feasibility wall"
  result from the 490-node Q2, now on the Eastern Interconnect.
- **ERCOT is the weakest** (small grid, 267% reserve margin) — concentrate stays feasible
  at these budgets, but the distribute-wins fraction still climbs 0.20 → 0.80 with budget.
- **Grid stress grows monotonically and prices stay physical** (gen costs 0–234/3818,
  LMP medians $33–73) on every network — Q1 holds.

**These are DIRECTIONAL validation runs, NOT citable.** Fast knobs were used: short panels
(3–4 hours), few fleets (8–10), and a capped candidate set (`--max-nodes` = first 100–250
node indices, *not* the strided clean-node sampling the 490 study uses). No bus cleaning was
applied, so the known ±10⁴ price pockets / weakly-connected buses are still present — visible
as western's worst-case p95 = $8028 at B=6 and eastern's pinned 1.54% base shed (its 2 stray
buses). Before any citable multi-network claim: re-derive each net's `find_bad_buses` cleaning
manifest (the 25-bad list is WECC-490-specific), use all clean nodes with strided sampling, and
match fleets to each net's penetration (ERCOT B≈10, Eastern B≈41 — see `PILOT_NETWORKS.md`).
Repro: `dc_placement_study.py --network <net> --n-snaps 4 --n-fleets 10 --max-nodes N --budgets "..."`;
outputs under `/scratch/users/gfw/study_{texas,western,eastern}/`.

## N-1 / SCOPF-aware placement — `development/dc_placement_scopf.py`

Two engines:
- `--engine grad`: the differentiable **ADMM SCOPF layer** (contingency-aware) +
  `LineUtilizationObjective(scenario_aggregation)` + `BoxBudgetProjection`. It builds and
  runs (forward+backward ≈1 s on 101-node), **but the gradient is unreliable at tractable
  iteration counts** — differentiating through truncated/early-stopped ADMM gives a noisy
  descent direction and the loss can *increase*. Reliable use needs many ADMM iterations
  (the original demo used 5000) and careful rho handling, which is slow and memory-heavy.
- `--engine exact` (default): a robust **greedy water-filling on exact N-1 dispatches** —
  rank sites by their mean post-contingency marginal Σu² (real line-outage CVX dispatches),
  then fill the budget to caps. Reliably produces a contingency-aware allocation and compares
  it against base-case-only optimization and uniform on exact N-1 metrics.

**Honest result (490-node, 16 capped sites @ 250 MW, B=2 GW, 5 N-1 contingencies):**

| placement | base Σu² | mean-N1 Σu² | worst-N1 Σu² |
|---|---|---|---|
| SCOPF-aware | 142.5 | 310.5 | **491.0** (best worst-case) |
| base-case-opt | 142.4 | 309.6 | 512.5 |
| uniform | 144.9 | **298.0** (best mean) | 527.2 |

The effects are **small (~4%) and the ranking is not robust across metrics**: SCOPF-aware wins
on worst-case N-1 but uniform wins on mean N-1. The reason is important and reinforces the core
thesis: **with realistic per-site caps the budget is already forced to spread, and that spreading
is most of the contingency hedge** — so optimizing the distribution further (vs naive uniform)
only moves N-1 metrics a few percent. SCOPF-aware optimization buys a modest worst-case
robustness improvement; it is not a large effect at this scale. The machinery is in place for
regimes where contingencies dominate (larger budgets / fewer-larger sites / a proper worst-case
minimax objective).

(On the 101-node, even N-1 barely separates the land-cost candidates — consistent with the
copperplate finding; the SCOPF effect, such as it is, shows on the congested 490-node.)

### N-1 via exact LODF + the differential-overload metric (supersedes the above) — `dc_n1_overload*.py`

The `dc_placement_scopf.py` result above was limited by its engine (noisy unrolled-ADMM gradient,
or a greedy on only ~5 hand-picked contingencies). We replaced it with **exact PTDF/LODF
distribution factors** (new core module `zap/contingency/`, validated to 1e-12 against zap's own
`model_contingency_problem` oracle), covering the **full N-1 set (all ~1246 outages)** at
base-case LP cost.

**Why the old result was ~4% and non-robust — the 490-net is N-1 *corridor-limited*.** Base
economic dispatch runs the median line at 9% utilization but ~16 corridors at 100%; **613/1246
single-line outages push a corridor to 130–232% of rating, at *every* load level** (structural,
not stress — emergency ratings barely help). So the grid's *own* N-1 stress dominates and a few
GW of DC only perturbs it. Strict "firm N-1 hosting capacity" is therefore degenerate (preventive
needs ~40 GW shed even with no DC; optimal corrective is ~800k vars/hr) — which is exactly why
the prior work used `line*0.7` as a crude N-1 proxy.

**Metric that *is* well-defined and decision-relevant:** total post-contingency overload
`O = Σ_t Σ_k Σ_l (|f + LODF[:,k] f_k| − Fbar)_+`, minimized over the base dispatch (and placement,
for the aware case), load-shed hard-capped at the no-DC baseline. Convex, full-N-1, HiGHS,
single representative stressed hour. Three contrasts reproduce the base-case siting story **under
N-1** (pilot, B in GW, "added overload" = O(placement) − O(no-DC), negative = relief):

- **Budget sweep:** smart (aware) placement *relieves* (~−0.16) while **uniform adds a growing
  amount (0.28 → 0.93 → 1.76 over B = 1 → 3 → 5)** — the smart-vs-naive gap grows with the fleet
  (the base-case "edge grows with budget", now under N-1).
- **Concentrate → distribute** (per-site-cap sweep): concentrating on grid-strong nodes beats
  forced spreading onto weaker nodes — *siting* matters more than *spreading*.
- **Grid-strength vs cheap-land siting** (the sharpest): siting ranked by true per-node N-1 probe
  marginal **relieves overload (−0.19)** while cheap-land (+0.72) and random (+0.62) *aggravate*
  it. **Where you site DC dominates the N-1 outcome** — mirrors the spread-frontier "siting
  dominates spreading."

**Role in the paper — a ROBUSTNESS CHECK, not a separate pillar (locked).** Because the binding
contingencies are outages *of the corridors already congested in base*, N-1-aware ≈
base-congestion-aware: full N-1 **does not reorder placements**, it reproduces the base-case
siting story (same sign, "siting dominates spreading", gap grows with fleet). So this section's
job is to *foreclose the obvious objection* — "your base-case relief is an artifact; under
contingencies smart siting might be bad" — by showing the siting direction is preserved in the
tested single-hour, B≤5, site-cap-0.5 N-1 regime. It does **not** cover the B=10 headline wall
configuration and should be cited only as directional robustness, not as a stricter-standard
confirmation of the headline. (Making N-1 a distinct contribution would require a priced/standard N-1
deliverability metric AND a network whose contingency-binding corridors are not already
base-congested — out of scope here.) The "~4%" wash-out of the old `dc_placement_scopf.py` was
the corridor-limited grid, not the method.

Files: `development/dc_n1_overload.py` (core LP), `dc_n1_overload_full.py` (the three contrasts),
`plot_n1_overload.py`; core `zap/contingency/{ptdf,lodf}.py` (validated to 1e-12 vs zap's own
contingency oracle, `zap/tests/test_contingency_lodf.py`). `dc_placement_scopf.py` is superseded.

## Workload notebook fix

`development/workload_variation_tb.py` had the live cap-violation bug (SimplexBudget +
0.25 GW per-site `upper_bounds`); patched to `BoxBudgetProjection(budget, lower_bounds,
upper_bounds)` so the per-site caps are actually enforced.

## Extra library fix surfaced by the SCOPF path

`PlanningProblem.project()` now casts the projected state back to the planner's array type.
`BoxBudgetProjection`/`SimplexBudgetProjection` return numpy, but the ADMM/SCOPF planner runs
in torch and then calls `.detach()` — so `BoxBudgetProjection` was previously **unusable**
with the SCOPF planner (AttributeError). Fixed in `zap/planning/problem_abstract.py:project()`.

## Spread frontier: "optimal spread scales with budget" (recast of the under-powered ~40-sites trend)

The Part-B chunking sweep produced a qualitative, under-powered trend ("~40 sites most
deliverable; over-spreading onto 80 weak nodes erodes it"), flagged as not significant. It
was fatally limited: a 3-point grid {20,40,80} (cannot locate an interior optimum at the
middle bin), a binary solver-success metric (high variance), and budget-confounding (the
interior peak showed only at B=10; B=6 was monotone-decreasing). Recast in
`development/dc_spread_frontier.py` as a continuous, properly-powered study.

**Metric correction (important).** The old "feasibility" only checked that the LP *solves*.
Because firm load is soft (VOLL=10 internal) and must-serve DC is hard, a solve can succeed
by DC **cannibalizing base load** — so "feasible" conflated *solvable* with *deliverable*,
inflating deliverable to ~60 GW. Fixed: a placement is deliverable only if base-load shed
stays within 1% of the no-DC baseline (firm demand served). Two continuous curves vs
footprint k: `g(k)` forced-uniform deliverable nameplate (bisection on must-serve DC) and
`f(k)` free-allocation ceiling (DC as a curtailable high-value load, one LP, allocation
free). Everything in nameplate GW; per hour actual = nameplate×lf; reliability = 5th-pct
over the panel (deliverable in ≥95% of hours).

**Result (full run, 490-node, load_scale 1.2, 12 Monte-Carlo sitings, 12h panel, all 465
clean nodes ranked):**
- **Headline test PASSES** (vs the old "trend"): slope of k_min(D) vs target D is
  **+5.40, 95% CI [3.30, 7.56]** (inference) and **+7.03, [4.51, 9.62]** (training),
  P(slope>0)=1.0. k_min(D) = smallest footprint whose g(k) ≥ D; it rises
  1→5→10→**40** sites as D grows. The original "~40 sites" is simply k_min for a larger
  fleet — and it **scales with budget**. The fraction of random sitings that can deliver
  D at all falls from 12/12 to 6/12 as D grows: the wall. Power calc
  (`spread_frontier_power.py`): only ~4 fleets needed for 80% power — massively powered.
- **Mechanism.** Forced-uniform `g(k) ≈ k × (capacity of the WEAKEST node in the top-k
  set)`; it collapses to ~0 once a near-dead clean node enters the cheap-land ordering
  (k≈320), while the free ceiling `f(k)` rises monotonically (puts ~0 on weak nodes).
  `gap=f−g` ("cost of forced uniformity") grows with k — that *is* the erosion. Invariants
  hold: f≥g everywhere, f monotone.
- **Siting dominates spreading (the sharpest result).** Grid-strength ordering (nodes by
  descending standalone headroom h(n)=f({n})) delivers ~5× more than cheap-land at k=20–40
  (19 vs 4 GW) AND defers the collapse: at k=320 grid-strength still delivers 8 GW while
  cheap-land has already collapsed to ~0. So over-spreading does not erode deliverability
  per se — *spreading onto cheap-but-weak land does*. Erosion is a siting artifact.
- Figures: `development/results/spread_frontier/spread_frontier_full_{inference,training}.png`.

Files: `development/dc_spread_frontier.py` (study), `plot_spread_frontier.py` (4-panel
figures), `spread_frontier_power.py` (pilot-then-size). Base-case only; N-1 deferred.
Also removed a stray per-dispatch debug `print()` in `zap/network.py`.

## Drift-repair guardrails (2026-06-09)

- **Frozen cleaning.** Citable reruns must use
  `development/results/cleaning/bad_buses_490_load1p0_25.json` and report the
  manifest path, the 25 bus IDs, `n_bad`, and the relevant cost assumptions.
  The manifest was verified against the existing lever and congestion artifacts;
  spread-frontier's old JSON did not record the actual list.
- **Wall-statistic hierarchy.** The 24h seed-pinned Phase-B Part A artifact
  (`phaseB_finish_partA_25bad_24h.json`, generated 2026-06-10, structural checks
  PASSED) **is the headline**. The 12h seed-pinned artifact is a consistency check;
  its brackets are 8-seed ranges, not confidence intervals (the 24h artifact
  carries seed×pool bootstrap CIs). The 24-pool lever frontier
  (`levers_full.json` / `levers_24h.json`) is the lever-frontier statistic
  and must not be quoted as "the wall."
- **Deliverable frontier vs spread frontier.** `dc_deliverable_frontier.py` answers
  a different question from forced-uniform `g(k)`: it permits free continuous
  allocation and therefore estimates a free-allocation hosting level. Under
  free allocation, siting can look nearly insensitive near the hosting ceiling.
  That does not undercut the spread-frontier mechanism; it clarifies it. Siting
  and spreading matter in the paper because real fleets are lumpy, capped,
  must-serve, and payer-constrained, not because the grid lacks aggregate hosting
  capacity when allocation is free.
  The accepted 12-hour, load×1.2 guardrail is now
  `development/results/deliverable_frontier/deliverable_frontier_full.json`
  (`n_bad=25`, `--skip-n1`): LP-optimal free allocation reaches 8.58 GW
  (inference) and 6.72 GW (training) at large kappa, with only 6 alive nodes after
  year-firm screening. The policy-order sanity check is intentionally not used as
  a headline because cheap-land and grid-strength nearly tie or swap order at high
  kappa under free allocation.
- **N-1 deliverable-frontier rows are non-citable.** The script now has `--skip-n1`
  and records `n1_citable: false`; the accepted guardrail encodes skipped rows as
  `status: skipped_non_citable` with `gw: NaN`. Do not quote old N-1
  deliverable-frontier rows.
- **Onsite generation.** `development/results/placement_levers/colocated_opex.json`
  is refreshed under the frozen 25-bad protocol. In the selected deep-wall 10 GW
  concentrated case, 7.0 GW onsite gas clears all 12 kept hours, with $0.595B/yr
  annualized capex and $2.064B/yr central fuel OpEx (OpEx/capex 3.47x). This is
  the payer-axis result: the lever shifts cost from grid/ratepayer transmission
  capital to private operator capex plus recurring fuel. The accepted B=10
  canonical lever frontier (`placement_levers/levers_full.json`, 24 pools, 12
  hours, `_partial: false`) confirms the onsite story in both workloads:
  inference clears at ~8.0 GW onsite ($0.678B/yr capex-only lever accounting),
  training at ~10.0 GW onsite ($0.848B/yr), while uniform transmission does not
  fully clear either workload even at ~$80.0B/yr.
- **Transmission crossover, accepted numbers.** In the B=10 lever frontier,
  near-uniform transmission must spend about $35.8B/yr (inference) and $62.4B/yr
  (training) to match the distributed fleet's feasibility; targeted transmission
  improves dispersion but not feasibility. The seed-pinned Phase-B Part A uniform
  crossover is less conservative: B=10 inference crosses at +100% / ~$20.0B/yr,
  B=10 training at +200% / ~$40.0B/yr. Frontier-match is the primary cost convention;
  crossover is secondary. Quote the source and axis.
- **Legacy status.** `placement_robustness/phaseB_finish_full.json` is legacy
  `n_bad=17`, not 25-bad. New Part A wall and uniform-tx crossover claims must
  come from `phaseB_finish_partA_25bad_24h.json` after it is generated, with
  `phaseB_finish_partA_25bad.json` retained only as the 12h consistency check;
  old Parts B/C remain qualitative.
  The accepted 25-bad Part A result is seed-separated for B=10 and B=6 in both
  workloads. The canonical lever frontier currently carries the B=10 headline
  fleet; do not cite B=3/B=6 lever curves from it unless those cells are rerun.
  Old season robustness is under-documented because the JSON did not record
  sampled window indices; reruns now record them.
- **Deferred SHOULD robustness.** Exact bad-count `{17,25,30}`, shed-tolerance
  sensitivity, solver cross-check, and refreshed spread-frontier provenance are
  deferred from the full evidence repair pass unless explicit artifacts are added.

## 24h evidence pass — accepted results (2026-06-10)

All four 24h artifacts are generated, `_partial: false`, structural checks pass
(`check_canonical_outputs.py`), frozen 25-bad manifest recorded in every file.
**The 24h artifacts are now the headline sources** per the wall-statistic hierarchy.

- **Wall (headline, `phaseB_finish_partA_25bad_24h.json`; 8 seeds × 24 pools × 24
  snaps, panel indices recorded; brackets = seed×pool bootstrap CIs):**

  | B (GW) | workload | concentrate | distribute | seed-separated |
  |---:|---|---|---|---|
  | 3 | inference | 0.705 [0.640, 0.769] | 0.896 [0.849, 0.938] | **No** |
  | 3 | training | 0.590 [0.520, 0.658] | 0.859 [0.807, 0.906] | Yes |
  | 6 | inference | 0.532 [0.464, 0.600] | 0.859 [0.807, 0.906] | Yes |
  | 6 | training | 0.365 [0.299, 0.430] | 0.755 [0.693, 0.813] | Yes |
  | 10 | inference | 0.350 [0.288, 0.413] | 0.748 [0.687, 0.806] | Yes |
  | 10 | training | 0.206 [0.151, 0.263] | 0.648 [0.581, 0.714] | Yes |

  B=10 matches the 12h pin (0.345/0.746; 0.207/0.646) almost exactly →
  **panel-robust**. The B=3 cells complete the emergence story: at 3 GW
  inference is NOT seed-separated (concentration still mostly deliverable),
  while **training separates already at B=3** — workload shape moves the wall's
  onset, sharpening "training is strictly worse."
- **Crossover moved at 24h (quote carefully).** All crossing cells (B=6 both,
  B=10 both) now cross at **+200% / ~$40.0B/yr**; the 12h B=10-inference
  +100%/$20B cell no longer reaches distribution's bar with the doubled panel.
  The crossover is a discrete-grid statistic and panel-sensitive between grid
  cells; the frontier-match convention (primary) is stable: **$35.7B/yr
  (inference) / $62.8B/yr (training)** in `levers_24h.json` vs $35.8B/$62.4B at
  12h. Inference primary ($35.7B) now sits inside the $20–40B grid cell —
  conventions are coherent. Never quote the old flat "$20B" for 24h results.
- **Lever frontier panel-robust (`levers_24h.json`):** conc/dist feasibility
  0.186/0.783 (inference), 0.083/0.724 (training) vs 0.188/0.781, 0.083/0.722 at
  12h. Targeted tx and grid-side gen still never reach distribution; joint
  gen+tx does not beat tx-alone; uniform tx at ~$80B/yr tops out ≈0.87/0.84.
  Colocated full-clear: ~8 GW / $0.678B/yr (inference), ~10 GW / $0.848B/yr
  (training) — same as 12h.
- **Flexible-DC lever (NEW, `levers_24h.json flexible_dc`; curtailment penalty
  just below base VOLL = physical-headroom semantics, `flex_voll` recorded):**
  making a concentrated B=10 fleet partially curtailable raises firm
  feasibility 0.186 → 0.201 / 0.248 / 0.347 at 5/10/20% flex (inference) and
  0.083 → 0.083 / 0.083 / 0.170 (training). The curtailable slice is itself
  served only ~31% of hours (inference, 20% flex). **Modest demand flexibility
  does NOT clear the wall** — it is dominated by distribution (0.75–0.78 at $0
  grid capital) and onsite generation (full clear). This forecloses the
  "just make the DC flexible" objection with data.
- **Onsite gas + carbon (`colocated_opex_24h.json`, all sanity checks held):**
  7.0 GW gas clears all 24 hours (CF 0.935, 57.3 TWh/yr); capex $0.595B/yr,
  fuel OpEx $2.007B/yr central (3.38×), **carbon at $50/tCO₂ adds $1.061B/yr →
  fuel+carbon $3.068B/yr ≈ 5.2× capex** ($25/$50/$100 sensitivity recorded).
  The payer-axis claim strengthens under any carbon price.
- **Cost sensitivity (`cost_sensitivity_24h.json`):** frontier-match primary
  (training B=10: $31.4/62.8/94.1B for line-cost ×0.5/1/1.5), Phase-B crossover
  secondary ($20/40/60B); DC investment $8.5/11.3/14.2B/yr; all linear.
- **Caveats carried forward:** (i) 7 monotonicity violations in the
  LMP-dispersion *outcome* curves (GEN/TXU; worst: training uniform-tx d_disp
  70.8 at +10% → 195.3 at +25%) — dispersion-vs-$ curves are noisy across pools
  (price pockets); feasibility curves are unaffected. Present dispersion as a
  mechanism signal with this noise acknowledged. (ii) `dc_placement_levers.py`
  does not record panel window indices (only `n_snaps`/`rep_hour`) — patch
  before the next citable lever run. (iii) The levers frontier figure is now
  tagged (`levers_frontier_{tag}.png`) after the 24h run overwrote the untagged
  canonical PNG; `levers_frontier_full.png` (12h) was regenerated from
  `levers_full.json`. (iv) Paper figures `fig_lever/fig_lever_bar/
  fig_colocated_opex` now render from 24h inputs; `*_12h.{png,pdf}` backups kept.
  `fig_phaseb_wall.{png,pdf}` (the headline figure) renders from the 24h artifact.

## Files

- `development/diagnostics/probe_congestion.py` — single-node injection sweep + congestion stats
- `development/diagnostics/probe_node_sensitivity.py` — dLMP/GW spread across nodes
- `development/diagnostics/probe_units.py` — unit audit (×100 ok, ×4 bug, shedding check)
- `development/diagnostics/probe_granularity.py` — granularity / VOLL-regime exploration
- `development/dc_placement_study.py` — the end-to-end study (Q1 siting curves + Q2 concentrate-vs-distribute)
- `development/dc_placement_optimize.py` — the gradient-based planner (LineUtilizationObjective + BoxBudgetProjection), realistic per-site caps
- `development/diagnostics/critic_gradcheck.py` — finite-difference gradient correctness check
- `development/diagnostics/critic_projections.py` — projection-vs-QP correctness check
- `development/diagnostics/time_planner_490.py` — planner tractability timing on 490 nodes
- `development/dc_granularity_sweep.py` — per-site-cap granularity sweep + dense candidate sampling
- `development/dc_placement_scopf.py` — N-1/SCOPF-aware placement (exact-N-1 greedy + differentiable-ADMM engines)
- Library fixes: `zap/planning/projection.py` (SimplexBudgetProjection), `zap/planning/problem_abstract.py` (bounds guard, grad branch, debug prints, project() torch-cast)
- Notebook fix: `development/workload_variation_tb.py` (SimplexBudget → BoxBudgetProjection)
