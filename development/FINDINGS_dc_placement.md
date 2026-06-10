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
contingencies smart siting might be bad" — by showing the deliverable-GW siting thesis **survives
under full, exact N-1**. It confirms the thesis under a stricter operating standard; it is not a
fourth metric. (Making N-1 a distinct contribution would require a priced/standard N-1
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
