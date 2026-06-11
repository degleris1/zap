"""
DIFFERENTIAL N-1 OVERLOAD: does DC placement relieve or aggravate post-contingency
corridor stress?

WHY THIS FRAMING
----------------
The cleaned 490-node WECC is N-1 *corridor-limited*: ~16 lines run at 100% in the
base economic dispatch and ~600 single-line outages push one of them to 130-230%
of rating, at every load level (see the N-1 diagnostics). So strict "firm N-1
hosting capacity" is degenerate -- preventive needs ~40 GW shed, full corrective
is ~800k variables. Rather than force an ill-posed feasibility question, we
measure the quantity that IS well-defined, fast, and decision-relevant:

    O(placement) = total post-contingency overload
                 = sum_t sum_k sum_l ( |f^{(k)}_l| - Fbar_l )_+

using the exact, validated LODF map f^{(k)} = f + LODF[:,k] f_k (preventive DC).
This is a convex (piecewise-linear) function of the dispatch, so minimizing it
over the base dispatch (and, for the aware case, over placement) is one LP.

We compare, for a fixed DC fleet (budget B):
  * N-1-aware  : placement that MINIMIZES total N-1 overload.
  * base-blind : placement that minimizes BASE congestion (ignores N-1), then its
                 N-1 overload evaluated with the operator's best dispatch.
  * uniform    : equal split over candidates, N-1 overload evaluated likewise.

Headline = how much N-1-aware siting cuts contingency overload vs blind/uniform,
and which corridors carry it. Solved with HiGHS (cp.SCIPY) in seconds.

Usage:
  .venv/bin/python development/dc_n1_overload.py --quick
  .venv/bin/python development/dc_n1_overload.py --tag pilot --budget 3
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
import scipy.sparse as sp

warnings.filterwarnings("ignore")
logging.getLogger("pypsa").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_spread_frontier import build_panels  # noqa: E402
from dc_placement_integrated import WORKLOADS  # noqa: E402
from n1_hosting import HostingCapacityProblem  # noqa: E402
from zap.contingency.lodf import sparsify_lodf  # noqa: E402

HIGHS = dict(solver=cp.SCIPY, scipy_options={"method": "highs"})


def make_problem(panel, candidates):
    net = panel[0][0]
    snaps = [d for (_, d, _, _) in panel]
    lfs = [float(lf) for (_, _, _, lf) in panel]
    return HostingCapacityProblem(net, snaps, candidate_buses=candidates, load_factors=lfs,
                                  must_serve_load=False, include_batteries=False,
                                  include_dc_lines=True)


def build_n1_matrix(hp, eps_lodf):
    """Sparse selection matrix M (n_pairs x L): row r = f[mon] + LODF[mon,k] f[k].

    Restricted to significant pairs (|LODF| > eps) over non-radial outages.
    rhs[r] = Fbar[mon]. The N-1 overload of a flow vector f is
    sum( (|M f| - rhs)_+ ).
    """
    SL = sparsify_lodf(hp.lodf, eps_lodf).tocoo()
    outaged = set(int(k) for k in hp.outaged)
    rows, cols, vals, rhs = [], [], [], []
    pair = []
    r = 0
    for mon, k, v in zip(SL.row, SL.col, SL.data):
        mon, k = int(mon), int(k)
        if k not in outaged or mon == k:
            continue
        rows += [r, r]
        cols += [mon, k]
        vals += [1.0, float(v)]
        rhs.append(hp.F_bar[mon])
        pair.append((mon, k))
        r += 1
    M = sp.csr_matrix((vals, (rows, cols)), shape=(r, hp.L))
    mon_rows = np.array([p[0] for p in pair], dtype=int)  # monitored line per row
    k_rows = np.array([p[1] for p in pair], dtype=int)  # outaged line per row
    return M, np.array(rhs), mon_rows, k_rows


def solve_overload(panel, candidates, M, rhs, mon_rows, budget, cap, d_fixed=None,
                   objective="n1", shed_cap=None):
    """Minimize N-1 overload (objective='n1') or base congestion ('base') over the
    base dispatch, with DC placement free (budget B) or fixed to d_fixed.

    Returns (placement, total_n1_overload, per_corridor_overload, status, secs).
    """
    t0 = time.time()
    hp = make_problem(panel, candidates)
    # when the placement is pinned (d_fixed), the per-site cap is irrelevant and may
    # conflict with a concentrated placement, so drop it.
    hp.build(site_cap=None if d_fixed is not None else cap)
    cons = list(hp.base_constraints)
    if d_fixed is not None:
        cons.append(hp.d == d_fixed)
    else:
        cons.append(cp.sum(hp.d) == budget)

    # soft N-1 overload slacks (the metric)
    n1_slacks = []
    for t in range(hp.T):
        f = hp.flow_expr[t]
        s = cp.Variable(M.shape[0], nonneg=True)
        n1_slacks.append(s)
        Mf = M @ f
        cons += [Mf <= rhs + s, -Mf <= rhs + s]
    n1_overload = cp.sum([cp.sum(s) for s in n1_slacks])
    shed = cp.sum([cp.sum(x) for x in hp.shed_vars]) if hp.shed_vars else cp.Constant(0.0)

    # Load-shed handling: with shed_cap given, hold shed at the no-DC baseline as a
    # HARD constraint and optimize N-1 overload ALONE (clean metric, no overload/shed
    # trade-off). Without it, fall back to a heavy penalty.
    if shed_cap is not None and hp.shed_vars:
        cons.append(shed <= shed_cap)
        shed_term = 0.0
    else:
        shed_term = 1e2 * shed

    if objective == "n1":
        obj = cp.Minimize(n1_overload + shed_term)
    else:  # base-congestion proxy: keep base flows under 90% of rating
        b_slacks = []
        for t in range(hp.T):
            f = hp.flow_expr[t]
            bs = cp.Variable(hp.L, nonneg=True)
            b_slacks.append(bs)
            cons += [f <= 0.9 * hp.F_bar + bs, -f <= 0.9 * hp.F_bar + bs]
        obj = cp.Minimize(cp.sum([cp.sum(b) for b in b_slacks]) + shed_term)

    prob = cp.Problem(obj, cons)
    prob.solve(**HIGHS)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, float("nan"), None, prob.status, time.time() - t0, hp

    # per-corridor overload at the solution (slack summed over hours, by monitored line)
    per_line = np.zeros(hp.L)
    for s in n1_slacks:
        np.add.at(per_line, mon_rows, np.asarray(s.value).ravel())
    total = float(n1_overload.value)
    return np.asarray(hp.d.value).ravel(), total, per_line, prob.status, time.time() - t0, hp


def realized_shed(hp):
    """Total non-DC load shed (GW) at the current solution of ``hp``."""
    if not hp.shed_vars:
        return 0.0
    return float(sum(float(np.asarray(s.value).sum()) for s in hp.shed_vars))


def panel_snapshots(panel):
    """Per-hour device lists from a panel (matches make_problem's snapshot extraction)."""
    return [d for (_, d, _, _) in panel]


class FixedPlacementEvaluator:
    """Re-solve the N-1 overload LP for many FIXED DC placements off ONE compiled problem.

    The DC placement ``d`` is a ``cvxpy.Parameter`` (DPP-valid: a constant matrix times the
    parameter), so the full N-1 overload problem -- theta/flow/shed Variables, the sparse DC
    flow definition ``flow == b*(A^T theta)``, nodal balance ``A@flow == p``, base thermal
    ``|flow|<=Fbar``, the soft N-1 slacks ``|M@flow| <= rhs + s``, the hard
    ``sum(shed) <= shed_cap`` cap, and the ``Minimize(sum s)`` objective with the DC term
    ``-A_C@(lf*d)`` -- is built and canonicalized ONCE. For a pinned ``d`` the only canonical
    datum that changes is the equality RHS ``b``; ``c``, matrices ``G,A`` and inequality RHS
    ``h`` are invariant. cvxpy caches the canonicalization, so deriving the new ``b`` per
    placement is ~3 ms (vs. rebuilding ``HostingCapacityProblem`` + re-canonicalizing each
    call in ``solve_overload``).

    Solver: each placement is solved on a fresh HiGHS interior-point (IPM) instance built
    from the cached canonical data. Empirically on this LP (490-node net, ~90k rows) the
    rebuild+canonicalization is cheap (~0.5 s) -- the cost is the LP solve itself (~3.5-5 s),
    which is irreducible per distinct placement. The expected win from a warm-started dual
    simplex did NOT materialise: this theta-space N-1 LP is so degenerate that a warm
    re-solve after an RHS change needs thousands of iterations and frequently CYCLES for
    minutes (0/60 single-node probes finished under 1500 simplex iters). IPM is cycling-free,
    solves any placement in ~3.5 s, and -- unlike the simplex-via-SCIPY path in
    ``solve_overload`` -- never returns false-infeasible at this size. The net per-solve
    speedup is modest (~1.4x: ~3.5 s vs ~5 s) but the path is reliable. ``run_crossover on``
    yields a clean kOptimal / kInfeasible status; the objective is the total N-1 overload
    to ~1e-7 of ``solve_overload`` (verified).

    Reproduces ``solve_overload(..., objective="n1", d_fixed=d, shed_cap=...)`` exactly.
    """

    def __init__(self, panel, candidates, M, rhs, mon_rows, shed_cap=None):
        self.M = M
        self.rhs = rhs
        self.mon_rows = mon_rows
        self.shed_cap = shed_cap

        hp = make_problem(panel, candidates)
        self.hp = hp
        self.nC = hp.nC
        self.L = hp.L

        AT = hp.A.T.tocsr()
        refs = [int(comp[0]) for comp in hp.components]

        self.d_param = cp.Parameter(hp.nC)  # the DC placement (set per evaluate())

        cons = []
        shed_vars = []
        n1_slacks = []
        for t, devices in enumerate(panel_snapshots(panel)):
            lf = hp.load_factors[t]
            p_base, _gen_var, extra, icons = hp._build_injection(devices, lf)
            shed_vars += [v for (kind, v) in extra if kind == "shed"]
            cons += icons
            # shared DC withdrawal via the Parameter (constant matrix @ parameter => DPP)
            p = p_base - hp.A_C @ (lf * self.d_param)

            theta = cp.Variable(hp.N)
            flow = cp.Variable(hp.L)
            cons.append(flow == cp.multiply(hp.b, AT @ theta))
            cons.append(hp.A @ flow == p)
            cons.append(theta[refs] == 0)
            cons.append(flow <= hp.F_bar)
            cons.append(-flow <= hp.F_bar)

            s = cp.Variable(M.shape[0], nonneg=True)
            n1_slacks.append(s)
            Mf = M @ flow
            cons += [Mf <= rhs + s, -Mf <= rhs + s]

        if shed_cap is not None and shed_vars:
            cons.append(cp.sum([cp.sum(v) for v in shed_vars]) <= shed_cap)

        n1_overload = cp.sum([cp.sum(s) for s in n1_slacks])
        self.prob = cp.Problem(cp.Minimize(n1_overload), cons)

    def _canonical_data(self, d_vector):
        """Canonical LP data for placement ``d``. The placement enters only the equality
        RHS ``b``; everything else is invariant, and cvxpy's canonicalization is cached, so
        this is ~3 ms after the first call."""
        self.d_param.value = np.asarray(d_vector, dtype=float).ravel()
        data, _chain, _inv = self.prob.get_problem_data(solver=cp.SCIPY)
        return data

    def _make_lp(self, data):
        """Build a HiGHS LP object from canonical cvxpy data:
        minimize ``c @ x`` s.t. ``G x <= h`` then ``A x == b`` (stacked as row bounds)."""
        import highspy

        G = data["G"].tocsc()
        h = np.asarray(data["h"]).ravel()
        Aeq = data["A"].tocsc()
        beq = np.asarray(data["b"]).ravel()
        c = np.asarray(data["c"]).ravel()
        nvar = c.shape[0]

        Acon = sp.vstack([G, Aeq]).tocsc()
        inf = highspy.kHighsInf
        row_lo = np.concatenate([np.full(G.shape[0], -inf), beq])
        row_hi = np.concatenate([h, beq])

        lp = highspy.HighsLp()
        lp.num_col_ = nvar
        lp.num_row_ = Acon.shape[0]
        lp.col_cost_ = c.tolist()
        lp.col_lower_ = [-inf] * nvar
        lp.col_upper_ = [inf] * nvar
        lp.row_lower_ = row_lo.tolist()
        lp.row_upper_ = row_hi.tolist()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = Acon.indptr.tolist()
        lp.a_matrix_.index_ = Acon.indices.tolist()
        lp.a_matrix_.value_ = Acon.data.tolist()
        return lp

    def _solve_lp(self, data):
        """Solve one placement on a FRESH HiGHS interior-point instance.

        Solver choice: interior-point (IPM), NOT warm-started dual simplex. This LP (the
        ~90k-row theta-space N-1 model) is highly degenerate: a warm dual-simplex re-solve
        after an RHS change needs thousands of iterations and frequently CYCLES (crawling
        for minutes) -- the warm basis buys essentially nothing here (measured: 0/60
        single-node probes finished under 1500 simplex iters). IPM is cycling-free and
        solves any placement in ~3-4 s regardless of how far the RHS moved, and (unlike the
        simplex-via-SCIPY path in ``solve_overload``) it does not return false-infeasible
        on this size. ``run_crossover on`` finishes at a vertex so the status is a clean
        kOptimal / kInfeasible (no kUnknown).

        A FRESH instance per call (vs. reusing one and only updating the RHS) is
        deliberate: a reused IPM instance was observed to drift into kUnknown after many
        RHS updates (mislabelling feasible placements as infeasible). Rebuilding the model
        is cheap (~0.05 s) next to the ~3.5 s solve, so freshness is essentially free and
        makes every evaluation independent and correct.
        """
        import highspy

        hs = highspy.Highs()
        hs.setOptionValue("output_flag", False)
        hs.setOptionValue("solver", "ipm")
        hs.setOptionValue("run_crossover", "on")
        hs.setOptionValue("time_limit", float(os.environ.get("FPE_TIME_LIMIT", "120.0")))
        hs.passModel(self._make_lp(data))
        hs.run()
        st = hs.getModelStatus()
        name = st.name if hasattr(st, "name") else str(st)
        if name != "kOptimal":
            return float("nan"), name
        return float(hs.getObjectiveValue()), "optimal"

    def evaluate(self, d_vector):
        """Set the placement, solve, return (total_n1_overload, status).

        The DC placement enters only the cached canonicalization's equality RHS, so the
        per-placement work is the (cached) ~3 ms cvxpy data fetch plus one fresh IPM solve.
        The objective value ``c @ x`` is exactly the total N-1 overload (the ``Minimize(sum
        slacks)`` objective has zero offset). Returns ``nan`` for an infeasible placement
        (matches ``solve_overload`` returning a non-optimal status).
        """
        t0 = time.time()
        data = self._canonical_data(d_vector)
        result, name = self._solve_lp(data)
        if os.environ.get("FPE_VERBOSE"):
            import sys as _sys
            print(f"[FPE] {time.time()-t0:.2f}s status={name} "
                  f"sites={int((np.asarray(d_vector) > 1e-6).sum())}", file=_sys.stderr, flush=True)
        return result, name


def run_workload(panel, clean, args):
    cand = [int(x) for x in clean]
    if args.cand_stride > 1:
        cand = cand[:: args.cand_stride]
    hp0 = make_problem(panel, cand)
    hp0.build(site_cap=0.0)
    M, rhs, mon_rows, _k = build_n1_matrix(hp0, args.eps_lodf)
    print(f"   {len(cand)} candidates, {len(panel)}h, {M.shape[0]} N-1 pairs, "
          f"{len(hp0.outaged)} outages, {len(np.where(hp0.is_radial)[0])} radial", flush=True)

    B, cap = args.budget, args.site_cap
    args_ = (panel, cand, M, rhs, mon_rows, B, cap)

    # aware: minimize N-1 overload over placement
    d_aware, ovl_aware, pl_aware, _, ta, _ = solve_overload(*args_, objective="n1")
    # blind: minimize base congestion, then evaluate its N-1 overload (operator's best dispatch)
    d_blind, _, _, _, _, _ = solve_overload(*args_, objective="base")
    _, ovl_blind, _, _, _, _ = solve_overload(*args_, d_fixed=d_blind, objective="n1")
    # uniform: equal split, N-1 overload evaluated likewise
    d_unif = np.full(len(cand), B / len(cand))
    _, ovl_unif, _, _, _, _ = solve_overload(*args_, d_fixed=d_unif, objective="n1")
    # reference: no-DC overload (the grid's own N-1 stress)
    _, ovl_none, _, _, _, _ = solve_overload(*args_, d_fixed=np.zeros(len(cand)), objective="n1")

    def cut(a, b):
        return None if (not np.isfinite(a) or not np.isfinite(b) or b <= 0) else round(100 * (1 - a / b), 1)

    top_lines = [int(i) for i in np.argsort(pl_aware)[-10:][::-1]] if pl_aware is not None else []
    print("   N-1 overload (GW summed over outages x lines x hours):", flush=True)
    print(f"     no-DC   = {ovl_none:8.2f}   (grid's own N-1 stress)", flush=True)
    print(f"     aware   = {ovl_aware:8.2f}   ({int((d_aware>1e-3).sum())} sites, {ta:.1f}s)", flush=True)
    print(f"     blind   = {ovl_blind:8.2f}", flush=True)
    print(f"     uniform = {ovl_unif:8.2f}", flush=True)
    print(f"   -> N-1-aware siting cuts added contingency overload vs blind by "
          f"{cut(ovl_aware-ovl_none, ovl_blind-ovl_none)}%, vs uniform by "
          f"{cut(ovl_aware-ovl_none, ovl_unif-ovl_none)}%", flush=True)
    return {
        "n_candidates": len(cand), "n_pairs": int(M.shape[0]), "budget": B, "cap": cap,
        "overload": {"none": ovl_none, "aware": ovl_aware, "blind": ovl_blind, "uniform": ovl_unif},
        "cut_aware_vs_blind_pct": cut(ovl_aware - ovl_none, ovl_blind - ovl_none),
        "cut_aware_vs_uniform_pct": cut(ovl_aware - ovl_none, ovl_unif - ovl_none),
        "placement_aware": d_aware.tolist() if d_aware is not None else None,
        "placement_blind": d_blind.tolist() if d_blind is not None else None,
        "top_overloaded_lines": top_lines,
        "candidates": cand,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=4)
    ap.add_argument("--load-scale", type=float, default=1.2)
    ap.add_argument("--budget", type=float, default=3.0, help="DC fleet (GW)")
    ap.add_argument("--site-cap", type=float, default=0.5, help="per-site cap (GW)")
    ap.add_argument("--eps-lodf", type=float, default=1e-2)
    ap.add_argument("--cand-stride", type=int, default=8)
    ap.add_argument("--land-cost", default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--outdir", default="development/results/n1_overload")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_snaps = 2
        args.cand_stride = 24
    os.makedirs(args.outdir, exist_ok=True)

    print(f"## DIFFERENTIAL N-1 OVERLOAD (tag={args.tag}) load_scale={args.load_scale} "
          f"B={args.budget}GW cap={args.site_cap}GW")
    pn, n_nodes, bad, clean, weights, panels = build_panels(args)
    print(f"   network: {n_nodes} nodes, {len(bad)} pathological, {len(clean)} clean")
    out = {"load_scale": args.load_scale, "budget": args.budget, "site_cap": args.site_cap,
           "n_snaps": args.n_snaps, "eps_lodf": args.eps_lodf, "workloads": {}}
    for w in WORKLOADS:
        print(f"\n-- workload={w}")
        out["workloads"][w] = run_workload(panels[w], clean, args)
    path = os.path.join(args.outdir, f"n1_overload_{args.tag}.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
