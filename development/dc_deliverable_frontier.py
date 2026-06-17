"""
DC DELIVERABLE FRONTIER -- replacing the flawed forced-uniform spread metric.

WHY THIS EXISTS
---------------
The old spread-frontier outcome g(k) = forced-UNIFORM must-serve deliverable is a
weakest-link artifact: it splits the fleet EQUALLY over the top-k candidate nodes, so
a single near-dead bus entering the top-k collapses the whole metric to ~0 (T <= k *
min headroom). That is a property of the uniform-split rule, not of the grid's hosting
capacity.

THE NEW METRIC
--------------
Deliverable firm-DC GW = the maximum TOTAL must-serve DC that the grid can host across
the stressed panel with:
  * each node capped at a MAX SITE SIZE kappa (0 <= d_n <= kappa),
  * MULTI-DC-PER-NODE allowed -- a strong node may absorb up to kappa (which can be
    several small DCs' worth); the node's true capacity is its year-firm headroom, not
    "one DC",
  * non-DC load shed HARD-CAPPED at the no-DC baseline (firm demand stays served; DC
    never cannibalizes load),
  * joint feasibility across EVERY panel hour (one placement, all hours).

We NEVER site on dead buses: nodes whose year-firm headroom h_firm(n) < a small
threshold are dropped from the candidate set up front.

THE STUDY
---------
Sweep the max site size kappa. For each kappa report (total deliverable GW, #sites
used) for FOUR siting policies:
  * LP-optimal (base)    : max sum(d_n) s.t. 0<=d_n<=kappa + base-case joint feasibility
                           (HostingCapacityProblem, enforce_n1=False, shed hard-capped).
  * LP-optimal N-1-aware : same but enforce_n1=True. RELIABILITY: HiGHS-via-scipy is
                           only reliable at <=~24 candidates and a SINGLE representative
                           hour for the N-1 case (it false-infeasibles above that), so
                           this optimizer is run single-hour at <=24 candidates.
  * grid-strength (policy): greedily fill clean nodes in DESCENDING year-firm-headroom
                           order, each up to kappa, taking the max total that stays
                           jointly feasible (bisect a global scale on the ordered fill).
  * cheap-land (policy)  : same greedy fill but in ASCENDING land-cost order (naive).

Expected sanity: deliverable GW is non-decreasing in kappa (looser cap can only host
more), #sites non-increasing; LP-optimal >= grid-strength >= cheap-land; all > 0 (no
dead-bus collapse); multi-DC-per-node happens (some d_n exceeds a single small DC when
kappa is large).

Usage:
  .venv/bin/python development/dc_deliverable_frontier.py --smoke
  .venv/bin/python development/dc_deliverable_frontier.py --tag pilot   # size later
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings

import cvxpy as cp
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("pypsa").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_spread_frontier import build_panels  # noqa: E402
from dc_placement_integrated import WORKLOADS  # noqa: E402
from n1_hosting import HostingCapacityProblem  # noqa: E402
from dc_placement_study import (load_land_weights, dispatch, dev_index, sample_panel_indices,  # noqa: E402
                                POWER_UNIT, COST_UNIT)
from dc_cleaning import DEFAULT_BAD_BUSES_JSON, cleaning_metadata  # noqa: E402

# HiGHS-via-scipy: clean LP duals + fast. Used for the (small) N-1-aware optimizer.
HIGHS = dict(solver=cp.SCIPY, scipy_options={"method": "highs"})
# A node whose year-firm headroom is below this (GW) is a "dead" bus -- excluded.
DEAD_HEADROOM_GW = 0.1
# Feasibility tolerance on the hard shed-cap (small slack above the no-DC baseline).
SHED_SLACK = 1e-4


# --------------------------------------------------------------------------- #
# panel helpers (the panel is a list of (net, devs, date, lf) tuples)
# --------------------------------------------------------------------------- #
def panel_snapshots(panel):
    return [d for (_, d, _, _) in panel]


def panel_lfs(panel):
    return [float(lf) for (_, _, _, lf) in panel]


def make_problem(panel, candidates, single_hour=None):
    """A HostingCapacityProblem over the panel (or one hour) for `candidates`.

    must_serve_load=False so non-DC load is sheddable; we hold that shed at the no-DC
    baseline with a HARD cap (see solve_lp_optimal), so DC fills only genuine spare
    headroom and never cannibalizes firm load.
    """
    net = panel[0][0]
    snaps = panel_snapshots(panel)
    lfs = panel_lfs(panel)
    if single_hour is not None:
        snaps = [snaps[single_hour]]
        lfs = [lfs[single_hour]]
    return HostingCapacityProblem(
        net, snaps, candidate_buses=candidates, load_factors=lfs,
        must_serve_load=False, include_batteries=False, include_dc_lines=True,
    )


def baseline_shed(panel, single_hour=None, enforce_n1=False):
    """No-DC non-DC load shed (GW, summed over hours) -- the floor we hold shed at.

    Solved with a zero-cap HostingCapacityProblem (d pinned to 0), so it is the exact
    no-DC baseline the hosting LP must not exceed. With ``enforce_n1`` this is the
    N-1-secure baseline shed -- which on this corridor-limited grid is FAR larger than
    the base-case baseline (preventive N-1 needs tens of GW of shed even with no DC), so
    the N-1 optimizer must be held at the N-1 baseline, NOT the base-case one."""
    hp = make_problem(panel, candidates=[0], single_hour=single_hour)
    hp.build(site_cap=0.0)
    res = hp.solve(enforce_n1=enforce_n1, return_shadow_prices=False, shed_penalty=1e3)
    return float(res.total_shed) if np.isfinite(res.total_shed) else 0.0


# --------------------------------------------------------------------------- #
# LP-optimal: max sum(d) s.t. 0<=d<=kappa, joint feasibility, shed held at baseline
# --------------------------------------------------------------------------- #
def solve_lp_optimal(panel, candidates, kappa, shed_cap, enforce_n1, single_hour=None,
                     solver=None, lazy=True):
    """Maximize total firm DC over `candidates`, each node capped at kappa, with the
    non-DC shed HARD-CAPPED at `shed_cap` (the no-DC baseline + slack). Multi-DC-per-node
    is automatic: a node's d_n can be anywhere in [0, kappa] regardless of "one DC".

    Returns (total_gw, n_sites, placement, status). n_sites counts nodes with d_n>1e-6.
    Solver failures (raised by an unstable LP backend on the degenerate N-1 problem) are
    caught and reported as a non-optimal status rather than crashing the sweep.
    """
    hp = make_problem(panel, candidates, single_hour=single_hour)
    hp.build(site_cap=kappa)
    # hold non-DC shed at the no-DC baseline as a HARD constraint (clean: DC is then
    # maximized over genuine spare headroom, not traded against firm-load shedding).
    if hp.shed_vars:
        hp.base_constraints = list(hp.base_constraints) + [
            cp.sum([cp.sum(s) for s in hp.shed_vars]) <= shed_cap + SHED_SLACK
        ]
    # shed_penalty=0 so the objective is exactly sum(d) (shed is already pinned).
    try:
        res = hp.solve(enforce_n1=enforce_n1, return_shadow_prices=False,
                       shed_penalty=0.0, solver=solver, lazy=lazy)
    except Exception as e:  # noqa: BLE001  (unstable LP backend can raise on the N-1 LP)
        return float("nan"), 0, None, f"solver_error:{type(e).__name__}"
    if res.status not in ("optimal", "optimal_inaccurate") or not np.isfinite(res.H):
        return float("nan"), 0, None, res.status
    d = np.asarray(res.placement).ravel()
    return float(d.sum()), int((d > 1e-6).sum()), d, res.status


def solve_n1_optimal(panel, candidates, kappa, shed_cap, single_hour):
    """N-1-aware LP-optimal, with a solver fallback for the (documented) fragility of
    the degenerate N-1 LP. Tries the default native HiGHS first; if it returns a
    non-optimal status or raises, retries once on HiGHS-via-scipy (cp.SCIPY). Either
    backend can false-infeasible at some kappa on this corridor-limited grid -- that is
    a known reliability caveat, so this is best-effort and may legitimately return nan.
    """
    gw, ns, d, st = solve_lp_optimal(panel, candidates, kappa, shed_cap, enforce_n1=True,
                                     single_hour=single_hour)
    if np.isfinite(gw):
        return gw, ns, d, st
    gw2, ns2, d2, st2 = solve_lp_optimal(panel, candidates, kappa, shed_cap,
                                         enforce_n1=True, single_hour=single_hour,
                                         solver=cp.SCIPY)
    return (gw2, ns2, d2, st2) if np.isfinite(gw2) else (gw, ns, d, st)


# --------------------------------------------------------------------------- #
# year-firm headroom h_firm(n): largest single-node DC must-serve-feasible EVERY hour
# --------------------------------------------------------------------------- #
def node_firm_headroom(panel, node, shed_cap_per_hour, hi=50.0, tol=0.02, max_iter=18):
    """h_firm(node) = min over panel hours of the node's standalone hostable firm DC.

    For each hour we host the largest single-site DC at `node` whose addition keeps the
    non-DC shed at that hour's no-DC baseline; the year-firm value is the MIN over hours
    (must be deliverable in EVERY hour). Computed as a single per-hour LP-optimal solve
    with one candidate and cap=hi (the LP maximizes that node's d directly, so no
    bisection is needed for the standalone case)."""
    h_min = np.inf
    for t in range(len(panel)):
        gw, _ns, _d, status = solve_lp_optimal(
            panel, [int(node)], kappa=hi, shed_cap=shed_cap_per_hour[t],
            enforce_n1=False, single_hour=t,
        )
        if status not in ("optimal", "optimal_inaccurate") or not np.isfinite(gw):
            h_min = 0.0
            break
        h_min = min(h_min, gw)
    return float(0.0 if not np.isfinite(h_min) else h_min)


def per_hour_baseline_shed(panel):
    """No-DC shed (GW) for each panel hour separately (the per-hour floor)."""
    return [baseline_shed(panel, single_hour=t) for t in range(len(panel))]


# --------------------------------------------------------------------------- #
# greedy ordered-fill policies (grid-strength, cheap-land): bisect a global scale
# --------------------------------------------------------------------------- #
def feasible_placement(panel, candidates, d_vec, shed_cap):
    """Is the FIXED placement d_vec (over `candidates`) jointly feasible across the
    panel with non-DC shed held at shed_cap? Pins d and checks the LP solves."""
    hp = make_problem(panel, candidates)
    hp.build(site_cap=None)
    cons = list(hp.base_constraints) + [hp.d == np.asarray(d_vec, float)]
    if hp.shed_vars:
        cons.append(cp.sum([cp.sum(s) for s in hp.shed_vars]) <= shed_cap + SHED_SLACK)
    prob = cp.Problem(cp.Minimize(0), cons)
    prob.solve(**HIGHS)
    return prob.status in ("optimal", "optimal_inaccurate")


def greedy_ordered_fill(panel, ordered_nodes, kappa, shed_cap, h_firm, tol=0.02,
                        max_iter=16):
    """Greedy ordered fill: fill `ordered_nodes` each up to kappa (capped also by the
    node's own year-firm headroom), then bisect a single global scale s in [0,1] on that
    ordered fill so the WHOLE placement stays jointly feasible.

    The per-node target is min(kappa, h_firm(node)) -- a node never receives more firm DC
    than it can host alone -- and the scale trims the aggregate to joint feasibility.
    Returns (total_gw, n_sites, placement).
    """
    nodes = [int(n) for n in ordered_nodes]
    target = np.array([min(kappa, h_firm.get(n, 0.0)) for n in nodes], float)
    if target.sum() <= tol:
        return 0.0, 0, np.zeros(len(nodes))
    # full fill feasible? then that's the answer.
    if feasible_placement(panel, nodes, target, shed_cap):
        d = target
        return float(d.sum()), int((d > 1e-6).sum()), d
    # bisect the global scale s on s*target.
    lo, hi = 0.0, 1.0
    if not feasible_placement(panel, nodes, tol * target, shed_cap):
        return 0.0, 0, np.zeros(len(nodes))
    lo = tol
    for _ in range(max_iter):
        if hi - lo <= 0.02:
            break
        mid = 0.5 * (lo + hi)
        if feasible_placement(panel, nodes, mid * target, shed_cap):
            lo = mid
        else:
            hi = mid
    d = lo * target
    return float(d.sum()), int((d > 1e-6).sum()), d


def solve_weighted_policy(panel, candidates, kappa, shed_cap, pref_weights):
    """A siting POLICY as an order-respecting LP: deploy DC to maximize the policy-weighted
    value sum(w_n d_n) (w_n > 0, higher = more preferred by the policy) subject to per-site
    cap kappa, base-case joint feasibility, and non-DC shed held at the baseline.

    The LP fills the policy's FAVOURED buses first (highest w per unit), then less-favoured
    ones, until the corridor capacity binds -- so the achieved TOTAL sum(d) genuinely
    depends on the policy: a criterion that favours grid-weak buses (cheap land / cheap
    power) binds the corridors sooner and delivers LESS than one that favours grid-strong /
    low-corridor-impact buses. (This is the fix for the order-independent greedy fill.)
    Returns (total_gw, n_sites, placement).
    """
    hp = make_problem(panel, candidates)
    hp.build(site_cap=kappa)
    cons = list(hp.base_constraints)
    if hp.shed_vars:
        cons.append(cp.sum([cp.sum(s) for s in hp.shed_vars]) <= shed_cap + SHED_SLACK)
    w = np.asarray(pref_weights, float).ravel()
    try:
        prob = cp.Problem(cp.Maximize(w @ hp.d), cons)
        prob.solve(**HIGHS)
    except Exception:  # noqa: BLE001
        return float("nan"), 0, None
    if prob.status not in ("optimal", "optimal_inaccurate") or hp.d.value is None:
        return float("nan"), 0, None
    d = np.asarray(hp.d.value).ravel()
    return float(d.sum()), int((d > 1e-6).sum()), d


# --------------------------------------------------------------------------- #
# extra siting orderings: cheap-power (low base LMP) and PTDF corridor-sensitivity
# --------------------------------------------------------------------------- #
def extra_orderings(panel, alive, hour, util_thresh=0.9):
    """Per-ALIVE-node rank keys for two more siting policies, from a no-DC dispatch at
    `hour`. Both are used ONLY as ascending rank keys, so their absolute scale (and the
    known LMP unit quirks) are irrelevant -- only the ordering matters.

      * cheap-power: base-case LMP at the node (lowest price first -- the developer's
        energy-cost objective). Cheap-LMP buses are often export-congested, so this is
        expected to UNDER-deliver vs the grid-aware policies.
      * ptdf-sensitivity: total |PTDF| coupling of the node's injection to the binding
        (base-congested) corridors (least corridor impact first -- the principled
        grid-aware heuristic; should track LP-optimal closely).
    """
    from zap.contingency import (
        build_signed_incidence, branch_susceptance, build_ptdf, thermal_limits,
    )

    net, devs = panel[hour][0], panel[hour][1]
    n_nodes = int(np.asarray(devs[0].num_nodes))
    oc = dispatch(net, devs, 1)  # no-DC base dispatch
    lmp_all = np.asarray(oc.prices).ravel()

    ai = dev_index(devs, "ACLine")  # NOT a fixed index: ERCOT has no DCLine, so ACLine shifts
    ac = devs[ai]
    A = build_signed_incidence(ac, n_nodes)
    b = branch_susceptance(ac)
    ptdf = build_ptdf(A, b)  # L x N
    f = np.asarray(oc.power[ai][1]).ravel()
    fbar = np.maximum(thermal_limits(ac).ravel(), 1e-9)
    binding = (np.abs(f) / fbar) > util_thresh
    if binding.sum() == 0:  # fall back to the most-loaded decile if nothing hits thresh
        binding = (np.abs(f) / fbar) >= np.percentile(np.abs(f) / fbar, 90)
    sens_all = np.abs(ptdf[binding, :]).sum(axis=0)  # per-node corridor coupling

    lmp = {int(n): float(lmp_all[n]) for n in alive}
    sens = {int(n): float(sens_all[n]) for n in alive}
    return lmp, sens, int(binding.sum())


# --------------------------------------------------------------------------- #
# checkpoint / resume: the kappa-independent year-firm headroom (node_firm_headroom
# over every candidate) is the dominant cost -- eastern spent ~38h there and the run
# saved nothing until the first kappa finished. We cache h_firm to its own file,
# flushed every 25 candidates, and skip kappa already in the result JSON, so a
# timeout/node-death only ever costs <=25 candidates and the kappa frontier advances
# forward across resubmits. Reuse is guarded by a FINGERPRINT: cached physics belongs
# to one (network, panel, candidate set, cleaning); a mismatch recomputes fresh.
# --------------------------------------------------------------------------- #
def _fingerprint(args, workload, panel_idx, bad_buses):
    """Identity of a deliverable-frontier computation. h_firm + completed-kappa rows are
    only reusable across runs whose fingerprint matches exactly."""
    return {
        "network": os.path.basename(os.path.expanduser(args.network)),
        "n_snaps": int(args.n_snaps),
        "load_scale": float(args.load_scale),
        "panel_idx": [int(i) for i in panel_idx],
        "bad_buses": sorted(int(b) for b in bad_buses),
        "cand_stride": int(args.cand_stride),
        "max_candidates": (int(args.max_candidates) if args.max_candidates else None),
        "dead_headroom": float(args.dead_headroom),
        "workload": workload,
    }


def _fp_mismatch(a, b):
    """Name of the first fingerprint field that differs (None if identical / either missing)."""
    if a is None or b is None:
        return "absent"
    for k in b:
        if a.get(k) != b.get(k):
            return k
    return None


def _load_hfirm_cache(path, fp):
    """(h_firm {int:float}, done set) from a fingerprint-matching cache, else ({}, set())."""
    if not path or not os.path.exists(path):
        return {}, set()
    try:
        d = json.load(open(path))
    except Exception:  # noqa: BLE001 (a half-written cache must not crash the run)
        print(f"   h_firm cache unreadable ({path}) -> recomputing", flush=True)
        return {}, set()
    miss = _fp_mismatch(d.get("fingerprint"), fp)
    if miss is not None:
        print(f"   h_firm cache fingerprint mismatch ({miss}) -> ignoring cache", flush=True)
        return {}, set()
    h = {int(k): float(v) for k, v in d.get("h_firm", {}).items()}
    done = {int(x) for x in d.get("done", list(h.keys()))}
    return h, done


def _save_hfirm_cache(path, fp, h_firm, done):
    """Atomic write (tmp + os.replace) so a SIGKILL mid-write never corrupts the cache."""
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"fingerprint": fp, "done": sorted(int(x) for x in done),
                   "h_firm": {int(k): float(v) for k, v in h_firm.items()}}, f, default=float)
    os.replace(tmp, path)


def _seed_hfirm_from_result(resume_block, cand):
    """Fallback when no separate cache exists yet: lift a COMPLETE h_firm (covering every
    current candidate) from a prior result JSON's workload block -- how the pulled-back
    eastern run, whose h_firm is embedded in the result file, resumes without the headroom
    loop. Returns ({}, set()) unless the block's h_firm covers all `cand` (a partial embedded
    h_firm isn't resumable -- the result file records no per-candidate 'done' list). Caller
    must already have fingerprint-gated `resume_block`."""
    if not resume_block:
        return {}, set()
    h = {int(k): float(v) for k, v in (resume_block.get("h_firm") or {}).items()}
    if h and all(int(n) in h for n in cand):
        return h, {int(n) for n in cand}
    return {}, set()


def _result_block_compatible(blk, prev_doc, fp):
    """Is a prior result workload-block safe to resume kappa/h_firm from? Prefer the block's
    own embedded fingerprint (newer runs); fall back to the result file's top-level
    cleaning/panel fields for legacy files that predate the fingerprint."""
    bfp = blk.get("fingerprint")
    if bfp is not None:
        miss = _fp_mismatch(bfp, fp)
        if miss:
            print(f"   prior result fingerprint mismatch ({miss}) -> fresh kappa", flush=True)
        return miss is None
    legacy_ok = (
        float(prev_doc.get("load_scale", -1)) == fp["load_scale"]
        and int(prev_doc.get("n_snaps", -1)) == fp["n_snaps"]
        and sorted(int(b) for b in prev_doc.get("bad_buses", [])) == fp["bad_buses"]
        and float(prev_doc.get("dead_headroom_gw", -1)) == fp["dead_headroom"]
    )
    print("   legacy result (no fingerprint); top-level fields "
          f"{'match -> resuming' if legacy_ok else 'differ -> fresh kappa'}", flush=True)
    return legacy_ok


# --------------------------------------------------------------------------- #
# per-workload study over the kappa grid
# --------------------------------------------------------------------------- #
def run_workload(panel, clean, weights, args, on_row=None, *, workload=None,
                 fingerprint=None, cache_path=None, resume_block=None):
    t0 = time.time()
    # candidate set: clean nodes, subsampled by cand-stride for tractability.
    cand = [int(n) for n in clean]
    if args.cand_stride > 1:
        cand = cand[:: args.cand_stride]
    if args.max_candidates and len(cand) > args.max_candidates:
        cand = cand[: args.max_candidates]

    # no-DC shed floors (panel total + per-hour) -- the deliverability baseline.
    shed_cap_total = baseline_shed(panel)
    shed_cap_hour = per_hour_baseline_shed(panel)
    print(f"   baseline non-DC shed: total={shed_cap_total:.3f} GW over {len(panel)}h",
          flush=True)

    # year-firm headroom per candidate; EXCLUDE dead buses (h_firm < threshold).
    # CHECKPOINT/RESUME: load any fingerprint-matching cache (or seed from a prior result
    # JSON's embedded h_firm), compute only the candidates still missing, and flush every
    # 25 newly-computed so a timeout/node-death costs <=25 candidates, not the whole loop.
    h_firm, done = _load_hfirm_cache(cache_path, fingerprint)
    if not h_firm:
        h_firm, done = _seed_hfirm_from_result(resume_block, cand)
        if h_firm:
            print(f"   h_firm seeded from result JSON ({len(h_firm)} candidates) "
                  f"-> skipping headroom loop", flush=True)
    elif done:
        print(f"   h_firm cache hit: {len(done & set(cand))}/{len(cand)} candidates cached",
              flush=True)
    computed = 0
    for n in cand:
        if n in done:
            continue
        h_firm[n] = node_firm_headroom(panel, n, shed_cap_hour)
        done.add(n)
        computed += 1
        if computed % 25 == 0:
            print(f"   headroom progress: {len(done & set(cand))}/{len(cand)} candidates "
                  f"({computed} this run)", flush=True)
            _save_hfirm_cache(cache_path, fingerprint, h_firm, done)
    if computed:
        _save_hfirm_cache(cache_path, fingerprint, h_firm, done)
        print(f"   headroom complete: {len(cand)} candidates ({computed} computed this run)",
              flush=True)
    else:
        print(f"   headroom: all {len(cand)} candidates from cache/seed", flush=True)
    alive = [n for n in cand if h_firm[n] >= args.dead_headroom]
    dead = [n for n in cand if h_firm[n] < args.dead_headroom]
    print(f"   year-firm headroom computed for {len(cand)} candidates; "
          f"excluded {len(dead)} dead buses (h_firm < {args.dead_headroom} GW), "
          f"{len(alive)} alive", flush=True)
    if not alive:
        return {"error": "no alive candidates", "n_dead": len(dead)}

    # orderings over ALIVE candidates only. `weights` here is the per-node RAW land
    # cost ($/acre) array; cheap-land fills lowest-cost nodes first (ascending).
    # Per-policy PREFERENCE WEIGHTS over alive nodes (higher = more preferred). Each
    # policy's deliverable is then the order-respecting weighted LP (solve_weighted_policy).
    order_hour = int(np.argmax(panel_lfs(panel))) if panel_lfs(panel) else 0
    lmp, ptdf_sens, n_binding = extra_orderings(panel, alive, order_hour)

    def pref_weight(getval, prefer_low):
        """Map a per-node criterion to positive weights in [0.05, 1] (floor keeps the
        least-preferred node still fillable). prefer_low=True -> lower raw value preferred."""
        a = np.array([getval(n) for n in alive], float)
        rng = float(a.max() - a.min())
        u = (a - a.min()) / rng if rng > 1e-12 else np.zeros_like(a)
        pref = (1.0 - u) if prefer_low else u
        return 0.05 + 0.95 * pref

    w_gs = pref_weight(lambda n: h_firm[n], prefer_low=False)     # grid-strength: high h_firm
    w_cl = pref_weight(lambda n: weights[n], prefer_low=True)     # cheap-land: low land cost
    w_cp = pref_weight(lambda n: lmp[n], prefer_low=True)         # cheap-power: low LMP
    w_pt = pref_weight(lambda n: ptdf_sens[n], prefer_low=True)   # PTDF: low corridor coupling
    print(f"   policy weights from hour {order_hour}: {n_binding} binding corridors; "
          f"LMP range [{min(lmp.values()):.3g}, {max(lmp.values()):.3g}]", flush=True)

    lfs = panel_lfs(panel)
    n1_hour = int(np.argmax(lfs)) if lfs else 0
    # N-1 rows are best-effort diagnostics and are not citable for the deliverable
    # guardrail. Canonical runs skip them so the figure cannot accidentally cite them.
    n1_cand = alive[: args.n1_max_candidates]
    if args.skip_n1:
        n1_shed_cap = None
        print("   N-1 deliverable rows skipped (non-citable guardrail run)", flush=True)
    else:
        n1_shed_cap = baseline_shed(panel, single_hour=n1_hour, enforce_n1=True)
        print(f"   N-1 baseline shed (hour {n1_hour}): {n1_shed_cap:.2f} GW "
              f"(base-case floor {shed_cap_hour[n1_hour]:.2f} GW)", flush=True)

    out = {
        "fingerprint": fingerprint,
        "candidates_alive": alive, "candidates_dead": dead,
        "h_firm": {int(n): float(h_firm[n]) for n in cand},
        "shed_cap_total": shed_cap_total, "n1_hour": n1_hour,
        "n1_shed_cap": n1_shed_cap, "kappa_grid": list(args.kappa), "rows": [],
    }

    # RESUME kappa: reuse fingerprint-matched rows from a prior (truncated) run, keyed by
    # kappa value, and only compute the kappa still missing -- forward work as we improve k.
    done_kappa = {float(r["kappa"]): r for r in (resume_block or {}).get("rows", [])}
    n_resumed = 0
    for kappa in args.kappa:
        if float(kappa) in done_kappa:  # already computed in a prior run -- reuse in order
            out["rows"].append(done_kappa[float(kappa)])
            n_resumed += 1
            continue
        # LP-optimal (base): full panel, all alive candidates. Default solver auto-pick
        # (HiGHS/CLARABEL) -- the base LP has no N-1 rows, so it is reliable at scale.
        lp_gw, lp_sites, lp_d, lp_st = solve_lp_optimal(
            panel, alive, kappa, shed_cap_total, enforce_n1=False,
        )
        if args.skip_n1:
            n1_gw, n1_sites, n1_st = float("nan"), 0, "skipped_non_citable"
        else:
            # LP-optimal N-1-aware: single hour, <= n1_max_candidates, shed held at
            # the N-1-secure no-DC baseline. Best-effort only.
            n1_gw, n1_sites, _n1_d, n1_st = solve_n1_optimal(
                panel, n1_cand, kappa, n1_shed_cap, single_hour=n1_hour,
            )
        # order-respecting weighted-LP policies (each fills its favoured buses first).
        gs_gw, gs_sites, _gs_d = solve_weighted_policy(panel, alive, kappa, shed_cap_total, w_gs)
        cl_gw, cl_sites, _cl_d = solve_weighted_policy(panel, alive, kappa, shed_cap_total, w_cl)
        cp_gw, cp_sites, _cp_d = solve_weighted_policy(panel, alive, kappa, shed_cap_total, w_cp)
        pt_gw, pt_sites, _pt_d = solve_weighted_policy(panel, alive, kappa, shed_cap_total, w_pt)
        # multi-DC-per-node evidence on the LP-optimal placement.
        max_dn = float(np.max(lp_d)) if lp_d is not None else float("nan")
        row = {
            "kappa": float(kappa),
            "lp_optimal": {"gw": lp_gw, "sites": lp_sites, "status": lp_st,
                           "max_dn": max_dn},
            "lp_n1": {"gw": n1_gw, "sites": n1_sites, "status": n1_st,
                      "n_candidates": len(n1_cand)},
            "grid_strength": {"gw": gs_gw, "sites": gs_sites},
            "cheap_land": {"gw": cl_gw, "sites": cl_sites},
            "cheap_power": {"gw": cp_gw, "sites": cp_sites},
            "ptdf_sensitivity": {"gw": pt_gw, "sites": pt_sites},
        }
        out["rows"].append(row)
        print(f"   kappa={kappa:5.2f}  LP={lp_gw:6.2f}/{lp_sites:3d}s  "
              f"N1={n1_gw:6.2f}/{n1_sites:3d}s  GS={gs_gw:6.2f}/{gs_sites:3d}s  "
              f"CL={cl_gw:6.2f}/{cl_sites:3d}s  CP={cp_gw:6.2f}/{cp_sites:3d}s  "
              f"PT={pt_gw:6.2f}/{pt_sites:3d}s  max_dn={max_dn:.2f}", flush=True)
        out["seconds"] = round(time.time() - t0, 1)
        if on_row is not None:  # incremental: flush after each kappa to survive timeout/preempt
            on_row(out)

    if n_resumed:
        print(f"   resumed {n_resumed} completed kappa; computed "
              f"{len(args.kappa) - n_resumed} new this run", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- #
def sanity_checks(out):
    """Print + return the smoke-test sanity checks for one workload result."""
    rows = out["rows"]
    checks = {}

    def nondec(key):
        v = [r[key]["gw"] for r in rows if np.isfinite(r[key]["gw"])]
        return all(b >= a - 1e-6 for a, b in zip(v, v[1:]))

    def noninc_sites(key):
        v = [r[key]["sites"] for r in rows]
        return all(b <= a for a, b in zip(v, v[1:]))

    checks["lp_gw_nondecreasing"] = nondec("lp_optimal")
    checks["lp_sites_nonincreasing"] = noninc_sites("lp_optimal")
    checks["ordering_lp>=gs>=cl"] = all(
        (r["lp_optimal"]["gw"] >= r["grid_strength"]["gw"] - 1e-3)
        and (r["grid_strength"]["gw"] >= r["cheap_land"]["gw"] - 1e-3)
        for r in rows if np.isfinite(r["lp_optimal"]["gw"])
    )
    checks["all_positive"] = all(
        r["lp_optimal"]["gw"] > 0 and r["grid_strength"]["gw"] > 0
        and r["cheap_land"]["gw"] > 0 for r in rows
    )
    # multi-DC-per-node: at the largest kappa, some node holds more than a single small
    # DC (use 0.5 GW as a "single small DC" reference).
    big = rows[-1]
    checks["multi_dc_per_node"] = big["lp_optimal"]["max_dn"] > 0.5 + 1e-6 \
        and big["kappa"] > 0.5
    print("\n   SANITY CHECKS:")
    for k, v in checks.items():
        print(f"     [{'PASS' if v else 'FAIL'}] {k}")
    out["sanity_checks"] = checks
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=3)
    ap.add_argument("--load-scale", type=float, default=1.2)
    ap.add_argument("--kappa", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--cand-stride", type=int, default=1)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--n1-max-candidates", type=int, default=24)
    ap.add_argument("--dead-headroom", type=float, default=DEAD_HEADROOM_GW)
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS.keys()))
    ap.add_argument("--land-cost",
                    default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--bad-buses-json", default=DEFAULT_BAD_BUSES_JSON,
                    help="frozen pathological-bus manifest for citable runs; pass '' to redetect")
    ap.add_argument("--skip-n1", action="store_true",
                    help="skip fragile N-1 rows; canonical deliverable-frontier runs use this")
    ap.add_argument("--outdir", default="development/results/deliverable_frontier")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any h_firm cache / prior result and recompute from scratch")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_snaps = 2
        args.cand_stride = 16  # ~30 raw candidates -> ~10 alive after dead-bus exclusion
        args.max_candidates = 30
        args.n1_max_candidates = 12
        args.kappa = [0.5, 2.0]
        args.workloads = ["inference"]
        args.outdir = "/tmp/deliverable_frontier_smoke"
    os.makedirs(args.outdir, exist_ok=True)

    print(f"## DC DELIVERABLE FRONTIER (tag={args.tag}) load_scale={args.load_scale} "
          f"kappa={args.kappa}")
    pn, n_nodes, bad, clean, weights, panels = build_panels(args)
    # panel_idx is a deterministic function of (network snapshots, n_snaps, seed=0) -- compute
    # it from the already-loaded pn (no reload) for the resume fingerprint.
    panel_idx = sample_panel_indices(len(pn.generators_t.p_max_pu.index), args.n_snaps, None, None, 0)
    # land cost per node (weights from build_panels is 1/normalized-land; we want the
    # raw land cost for the cheap-land ASCENDING order, lowest cost first).
    land, _w = load_land_weights(args.land_cost, n_nodes)
    print(f"   network: {n_nodes} nodes, {len(bad)} pathological, {len(clean)} clean, "
          f"{args.n_snaps}h panel", flush=True)

    bad_manifest = args.bad_buses_json or None
    path = os.path.join(args.outdir, f"deliverable_frontier_{args.tag}.json")
    # read any prior (possibly truncated) result ONCE up front -- the per-kappa _flush below
    # overwrites `path`, so capture resume material before the first write clobbers it.
    prev_doc = {}
    if not args.fresh and os.path.exists(path):
        try:
            prev_doc = json.load(open(path))
        except Exception:  # noqa: BLE001
            print(f"   prior result unreadable ({path}) -> fresh run", flush=True)
    prev_workloads = prev_doc.get("workloads", {}) if isinstance(prev_doc, dict) else {}

    out = {"network": os.path.basename(os.path.expanduser(args.network)),
           "load_scale": args.load_scale, "n_snaps": args.n_snaps, "n_clean": len(clean),
           **cleaning_metadata(bad_manifest, bad), "power_unit": POWER_UNIT,
           "cost_unit": COST_UNIT, "kappa_grid": list(args.kappa),
           "dead_headroom_gw": args.dead_headroom, "n1_citable": False,
           "n1_note": "N-1 rows are non-citable diagnostics and are skipped in canonical runs"
                      if args.skip_n1 else "N-1 rows are non-citable diagnostics",
           "workloads": {}}
    for w in args.workloads:
        print(f"\n-- workload={w}", flush=True)

        def _flush(partial, w=w):  # per-kappa incremental save: a timeout keeps finished kappas
            out["workloads"][w] = partial
            json.dump(out, open(path, "w"), indent=2, default=float)

        fp = _fingerprint(args, w, panel_idx, bad)
        cache_path = os.path.join(args.outdir, f"h_firm_{args.tag}_{w}.json")
        blk = prev_workloads.get(w)
        resume_block = blk if (blk and _result_block_compatible(blk, prev_doc, fp)) else None
        r = run_workload(panels[w], clean, land, args, on_row=_flush, workload=w,  # land = cheap-order key
                         fingerprint=fp, cache_path=cache_path, resume_block=resume_block)
        if "error" not in r:
            sanity_checks(r)
        out["workloads"][w] = r
        json.dump(out, open(path, "w"), indent=2, default=float)  # incremental: survive timeout/preempt
        print(f"   saved (incremental, {len(out['workloads'])} workload(s)) -> {path}", flush=True)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
