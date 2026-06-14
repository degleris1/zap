"""
DC SPREAD FRONTIER -- recasting the under-powered "~40 sites is most deliverable"
finding (Part B of dc_phaseB_finish.py) into a rigorous, continuous study.

WHY THIS EXISTS
---------------
The prior claim was: "moderate spread (~40 sites) is most deliverable; over-spreading
onto 80 weak nodes erodes it." It was flagged as a trend, not a result, because:
  (1) it came from a 3-point grid {20,40,80} -- you cannot locate an interior optimum
      whose maximum is the middle bin;
  (2) the metric was BINARY solver-success over a 12-hour panel (high variance);
  (3) it was budget-confounded -- the interior peak appears only at B=10; at B=6 the
      curve is monotone-decreasing. "Spread vs budget" was never isolated.

RECAST
------
Replace binary feasibility with a CONTINUOUS deliverable-GW outcome (a margin). Two
curves, indexed by footprint k = number of candidate sites:

  g(k)  forced-UNIFORM deliverable: largest DC NAMEPLATE T that can be served when the
        fleet is split equally over k sites (each must-serve, draw (T/k)*lf per hour).
        Found by bisection on the hard must-serve DataCenterLoad. Non-monotone -- this
        is where erosion lives.

  f(k)  free-allocation CEILING: largest total DC the grid can host over the same k
        candidate nodes with allocation FREE (a single LP: DC modelled as a high-VOLL
        curtailable Load, maximize served). Monotone non-decreasing in k by the subset
        argument -- it cannot show erosion, so it is the ceiling, not the phenomenon.

  gap(k) = f(k) - g(k)  "cost of forced uniformity"; its growth with k IS the erosion.

Under this margin metric the nominal budget B is redundant (uniform allocation scales
out of the physics), so the two budget regimes collapse to ONE curve. "Optimal spread
scales with budget" is then operationalized honestly as:
        k_min(D) = min{ k : g(k) >= D }   -- smallest footprint that delivers target D,
and the headline test is whether k_min(D) INCREASES with D.

Everything is NAMEPLATE GW. Per hour, actual draw = nameplate * lf(hour); g bisects
nameplate directly, f's actual ceiling is converted to nameplate via /lf. Reliability
across the panel: D is "deliverable" if served in >=95% of hours, i.e. the 5th
percentile of the per-hour curve (also report worst-hour=min and mean).

Orderings of candidate nodes as k grows:
  * cheap-land   (primary, realistic siting; the policy that produces erosion)
  * grid-strength(counterfactual; nodes by descending standalone headroom h(n))
  * monte-carlo  (land-weighted random fleets -> siting error bars; used for the slope test)
Per-node standalone headroom h(n) = f({n}) defines "weak" and yields the grid-strength order.

Scope: cleaned 490-node WECC, load_scale 1.2, single-period base-case DC-OPF, workloads
{inference, training}. N-1 / SCOPF deferred.

Usage:
  .venv/bin/python development/dc_spread_frontier.py --quick          # smoke test
  .venv/bin/python development/dc_spread_frontier.py --tag pilot      # pilot (size later)
"""
import argparse
import json
import logging
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("pypsa").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_placement_study import (  # noqa: E402
    sample_panel_indices, find_bad_buses, load_land_weights, dispatch,
    POWER_UNIT, COST_UNIT,
)
from dc_placement_integrated import build_raw_hour, WORKLOADS, load_dc_profile  # noqa: E402
from dc_placement_congestion import cleaned_panel  # noqa: E402
from dc_cleaning import DEFAULT_BAD_BUSES_JSON, cleaning_metadata, resolve_bad_buses  # noqa: E402

import zap  # noqa: E402

# DC reward (curtailment VOLL) for the free-allocation ceiling LP. Internal units:
# base load VOLL is 10 (=1000/COST_UNIT), generator marginal costs are 0-0.76. We set
# the DC reward BELOW base VOLL so base (firm) load is served first and DC only fills
# genuine spare headroom -- "deliverable" = hosted WITHOUT displacing firm load. (A
# reward above base VOLL would let DC cannibalize base load and inflate the metric.)
DC_VOLL = 5.0
# per-node nameplate cap in the ceiling LP -- large enough never to bind (physics binds).
NODE_CAP = 100.0
# a placement is "deliverable" only if it adds essentially no base-load shedding beyond
# the no-DC baseline (firm demand stays served). Fraction of base load.
SHED_TOL = 0.01


def base_shed_frac(net, devs):
    """Base-load shed fraction for the no-DC dispatch this hour (~0 on the cleaned net,
    but >0 possible at load_scale>1). Used as the incremental-shed baseline."""
    oc = dispatch(net, devs, 1)
    if oc is None:
        return np.inf
    L = devs[1]
    req = float((np.asarray(L.load) * np.asarray(L.nominal_capacity)).sum())
    served = float((-np.asarray(oc.power[1][0])).sum())
    return max(req - served, 0.0) / max(req, 1e-9)


def base_shed_panel(panel):
    return np.array([base_shed_frac(net, devs) for (net, devs, _, _) in panel])


# --------------------------------------------------------------------------- #
# f(k): free-allocation deliverable ceiling (one LP, allocation free)
# --------------------------------------------------------------------------- #
def dc_as_curtailable(devices, nodes, cap_each=NODE_CAP, voll=DC_VOLL):
    """Append the DC at `nodes` as a high-VOLL curtailable Load (served power is a free
    variable in [0, cap_each] per node). Returns (devs, dc_index).

    NOTE: net.dispatch is read-only on devices (it only locally rebinds the list to add
    a ground), so we SHARE the base device objects and only build the small new DC
    device -- avoiding a full deepcopy of the device list on every solve (~3x speedup)."""
    nodes = list(nodes)
    dc = zap.Load(
        num_nodes=devices[0].num_nodes,
        terminal=np.array(nodes),
        load=np.full(len(nodes), float(cap_each)),
        linear_cost=np.full(len(nodes), float(voll)),
    )
    return list(devices) + [dc], len(devices)


def with_dc_fast(devices, terminals, caps, lf):
    """Must-serve DC at `terminals` (draw nameplate*lf), sharing base devices (no
    deepcopy; see dc_as_curtailable note). Mirrors dc_placement_study.with_dc."""
    terminals = list(terminals)
    dc = zap.DataCenterLoad(
        num_nodes=devices[0].num_nodes, terminal=np.array(terminals),
        profiles=[np.array([lf], dtype=float)] * len(terminals),
        nominal_capacity=np.array(caps, dtype=float),
        linear_cost=np.zeros(len(terminals)), settime_horizon=1,
        capital_cost=np.zeros(len(terminals)),
    )
    return list(devices) + [dc]


def f_actual_hour(net, devs, nodes):
    """Max total DC ACTUAL power (GW) the grid can host over `nodes` with allocation
    free, base-case feasibility. Always solvable (DC can curtail to 0)."""
    devs2, idx = dc_as_curtailable(devs, nodes)
    oc = dispatch(net, devs2, 1)
    if oc is None:                       # should not happen (curtailable), guard anyway
        return 0.0
    served = float(-np.asarray(oc.power[idx][0]).sum())
    return max(served, 0.0)


def f_curve_hours(panel, nodes):
    """Per-hour nameplate ceiling f_h = f_actual_h / lf_h over the panel."""
    out = np.full(len(panel), np.nan)
    for i, (net, devs, _, lf) in enumerate(panel):
        if lf <= 1e-9:
            continue
        out[i] = f_actual_hour(net, devs, nodes) / lf
    return out


# --------------------------------------------------------------------------- #
# g(k): forced-uniform must-serve deliverable nameplate, by bisection
# --------------------------------------------------------------------------- #
def _uniform_feasible(net, devs, nodes, lf, T, base_shed):
    """Is nameplate T (split EQUALLY over `nodes` as must-serve DC) DELIVERABLE this
    hour -- i.e. the LP solves AND base-load shedding does not rise more than SHED_TOL
    above the no-DC baseline (firm demand stays served, DC does not cannibalize load)?"""
    k = len(nodes)
    caps = [T / k] * k
    oc = dispatch(net, with_dc_fast(devs, nodes, caps, lf), 1)
    if oc is None:
        return False
    L = devs[1]
    req = float((np.asarray(L.load) * np.asarray(L.nominal_capacity)).sum())
    served = float((-np.asarray(oc.power[1][0])).sum())
    shed = max(req - served, 0.0) / max(req, 1e-9)
    return shed <= base_shed + SHED_TOL


def g_actual_hour(net, devs, nodes, lf, hi, base_shed, tol=0.02, max_iter=16):
    """Largest must-serve NAMEPLATE T (equal split over `nodes`) deliverable this hour
    without extra base shedding. Bisected in [0, hi]; hi is the free ceiling f_h."""
    if hi <= tol or lf <= 1e-9 or not np.isfinite(base_shed):
        return 0.0
    if not _uniform_feasible(net, devs, nodes, lf, tol, base_shed):
        return 0.0
    if _uniform_feasible(net, devs, nodes, lf, hi, base_shed):
        return hi
    lo, h = tol, hi
    stop = max(tol, 0.03 * hi)          # relative tol -> ~5-6 bisection solves
    for _ in range(max_iter):
        if h - lo <= stop:
            break
        mid = 0.5 * (lo + h)
        if _uniform_feasible(net, devs, nodes, lf, mid, base_shed):
            lo = mid
        else:
            h = mid
    return lo


def g_curve_hours(panel, base_shed, nodes, f_hours):
    """Per-hour forced-uniform nameplate g_h over the panel, bisected within [0, f_h]."""
    out = np.full(len(panel), np.nan)
    for i, (net, devs, _, lf) in enumerate(panel):
        hi = f_hours[i]
        if not np.isfinite(hi):
            continue
        out[i] = g_actual_hour(net, devs, nodes, lf, hi, base_shed[i])
    return out


# --------------------------------------------------------------------------- #
# hour reducers: 95%-reliability (primary), worst-hour, mean
# --------------------------------------------------------------------------- #
def reduce_hours(vals, how="q05"):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    if how == "q05":         # deliverable in >=95% of hours
        return float(np.percentile(v, 5))
    if how == "min":         # worst hour
        return float(v.min())
    if how == "mean":
        return float(v.mean())
    raise ValueError(how)


# --------------------------------------------------------------------------- #
# per-node standalone headroom h(n) = f({n}); defines "weak" + grid-strength order
# --------------------------------------------------------------------------- #
def node_headroom(panel, nodes, how="q05"):
    """h(n) for each node in `nodes`: reduced per-hour single-site free ceiling."""
    h = {}
    for nd in nodes:
        fh = f_curve_hours(panel, [int(nd)])
        h[int(nd)] = reduce_hours(fh, how)
    return h


# --------------------------------------------------------------------------- #
# curves over an adaptive k-grid for a fixed node ORDERING (prefix nesting)
# --------------------------------------------------------------------------- #
def adaptive_kgrid(n_clean, coarse=(5, 10, 20, 40, 80, 160, 320)):
    g = sorted({k for k in coarse if 1 <= k <= n_clean} | {1, n_clean})
    return g


def refine_kgrid(kgrid, knee_k, n_clean, span=2):
    """Add denser points (~steps of ceil((b-a)/4)) in the [knee/2, knee*2] window."""
    lo = max(1, knee_k // span)
    hi = min(n_clean, knee_k * span)
    if hi - lo < 4:
        return kgrid
    extra = set(np.linspace(lo, hi, 7).astype(int).tolist())
    return sorted(set(kgrid) | extra | {lo, hi})


def curves_for_ordering(panel, base_shed, ordered_nodes, kgrid, reducers=("q05", "min", "mean")):
    """For each k in kgrid: f and g reduced over hours, using the top-k prefix of
    `ordered_nodes`. Returns {k: {"f": {red:val}, "g": {red:val}}}."""
    res = {}
    for k in kgrid:
        nodes = [int(x) for x in ordered_nodes[:k]]
        fh = f_curve_hours(panel, nodes)
        gh = g_curve_hours(panel, base_shed, nodes, fh)
        res[int(k)] = {
            "f": {r: reduce_hours(fh, r) for r in reducers},
            "g": {r: reduce_hours(gh, r) for r in reducers},
        }
    return res


# --------------------------------------------------------------------------- #
# derived quantities + inference
# --------------------------------------------------------------------------- #
def k_min(curve_g, kgrid, D, reducer="q05"):
    """Smallest k whose g(k) >= D. None if D exceeds max_k g(k)."""
    for k in sorted(kgrid):
        if np.isfinite(curve_g[k]["g"][reducer]) and curve_g[k]["g"][reducer] >= D:
            return int(k)
    return None


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.ptp(x[m]) == 0:
        return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])


def bootstrap_slope_ci(per_fleet_kmin, D_grid, rng, nboot=2000):
    """per_fleet_kmin: list over fleets of dict {D: k_min or None}. Bootstrap the
    fleet-pooled OLS slope of k_min vs D (drop D where k_min is None for that fleet)."""
    fleets = list(range(len(per_fleet_kmin)))
    slopes = []
    for _ in range(nboot):
        samp = rng.choice(fleets, size=len(fleets), replace=True)
        xs, ys = [], []
        for fi in samp:
            km = per_fleet_kmin[fi]
            for D in D_grid:
                if km.get(D) is not None:
                    xs.append(D)
                    ys.append(km[D])
        s = ols_slope(xs, ys)
        if np.isfinite(s):
            slopes.append(s)
    if not slopes:
        return {"slope": np.nan, "lo": np.nan, "hi": np.nan, "p_gt0": np.nan, "n": 0}
    slopes = np.asarray(slopes)
    # point estimate on the full (unresampled) pool
    xs, ys = [], []
    for km in per_fleet_kmin:
        for D in D_grid:
            if km.get(D) is not None:
                xs.append(D)
                ys.append(km[D])
    return {
        "slope": ols_slope(xs, ys),
        "lo": float(np.percentile(slopes, 2.5)),
        "hi": float(np.percentile(slopes, 97.5)),
        "p_gt0": float(np.mean(slopes > 0)),
        "n": int(slopes.size),
    }


# --------------------------------------------------------------------------- #
def weighted_permutation(rng, clean, weights):
    """One land-weighted random ordering of all clean nodes (cheap land earlier)."""
    w = np.array([weights[n] for n in clean], float)
    w = w / w.sum()
    return [int(x) for x in rng.choice(clean, size=len(clean), replace=False, p=w)]


def build_panels(args):
    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    idx = sample_panel_indices(len(snaps), args.n_snaps, None, None, 0)
    raw = [build_raw_hour(pn, snaps, h, args.load_scale, 1.0, 1.0) for h in idx]
    n_nodes = raw[0][0].num_nodes
    bad_manifest = getattr(args, "bad_buses_json", None) or None
    detected_bad = None
    if bad_manifest is None:
        raw0 = [build_raw_hour(pn, snaps, h, 1.0, 1.0, 1.0) for h in idx]
        detected_bad = find_bad_buses([(net, devs, dt, 0.0) for (net, devs, dt, _) in raw0])
    bad = resolve_bad_buses(bad_manifest, detected_bad)
    clean = [i for i in range(n_nodes) if i not in set(bad)]
    land, weights = load_land_weights(args.land_cost, n_nodes)
    panels = {}
    for w in args.workloads:
        hourly_lf = load_dc_profile(WORKLOADS[w])
        panels[w] = cleaned_panel(raw, hourly_lf, bad)
    return pn, n_nodes, bad, clean, weights, panels


def run_workload(panel, clean, weights, args, rng):
    """All curves + inference for one workload."""
    n_clean = len(clean)
    kgrid = adaptive_kgrid(n_clean)

    # no-DC base-load shed per hour (incremental-shed baseline for deliverability)
    base_shed = base_shed_panel(panel)

    # per-node standalone headroom (defines weak + grid-strength order)
    hnodes = [int(x) for x in clean]
    if args.node_stride > 1:
        hnodes = hnodes[:: args.node_stride]
    print(f"   headroom: {len(hnodes)} nodes ...", flush=True)
    head = node_headroom(panel, hnodes, how="q05")

    # orderings
    cheap_order = sorted(clean, key=lambda n: -weights[n])
    strong_order = sorted(head.keys(), key=lambda n: -(head[n] if np.isfinite(head[n]) else -1))
    # extend grid-strength order with un-probed nodes (stride) appended by cheap-land
    strong_set = set(strong_order)
    strong_order = strong_order + [n for n in cheap_order if n not in strong_set]

    out = {"kgrid": kgrid, "headroom": head, "base_shed": [float(x) for x in base_shed]}

    # deterministic orderings (single curve; hour-only variance)
    print(f"   kgrid={kgrid}", flush=True)
    print("   cheap_land curve ...", flush=True)
    out["cheap_land"] = curves_for_ordering(panel, base_shed, cheap_order, kgrid)
    print("   grid_strength curve ...", flush=True)
    out["grid_strength"] = curves_for_ordering(panel, base_shed, strong_order, kgrid)

    # refine the k-grid near the cheap-land knee (peak of g_q05), then recompute primary
    g_q05 = {k: out["cheap_land"][k]["g"]["q05"] for k in kgrid}
    knee = max(g_q05, key=lambda k: (g_q05[k] if np.isfinite(g_q05[k]) else -np.inf))
    rk = refine_kgrid(kgrid, knee, n_clean)
    if rk != kgrid:
        out["kgrid_refined"] = rk
        out["cheap_land_refined"] = curves_for_ordering(panel, base_shed, cheap_order, rk)

    # monte-carlo fleets (siting error bars; the slope test)
    mc = []
    for fi in range(args.n_fleets):
        print(f"   mc fleet {fi + 1}/{args.n_fleets} ...", flush=True)
        order = weighted_permutation(rng, clean, weights)
        mc.append(curves_for_ordering(panel, base_shed, order, kgrid, reducers=("q05",)))
    out["mc_curves"] = mc

    # k_min(D) per fleet + bootstrap slope test
    gmax = [max((c[k]["g"]["q05"] for k in kgrid if np.isfinite(c[k]["g"]["q05"])), default=0.0)
            for c in mc]
    D_hi = float(np.median(gmax)) if gmax else 1.0
    D_grid = [round(d, 3) for d in np.linspace(0.5, max(D_hi, 1.0), args.n_targets)]
    per_fleet_kmin = [{D: k_min(c, kgrid, D, "q05") for D in D_grid} for c in mc]
    out["D_grid"] = D_grid
    out["per_fleet_kmin"] = per_fleet_kmin
    out["slope_test"] = bootstrap_slope_ci(per_fleet_kmin, D_grid, rng, nboot=args.nboot)

    # secondary: erosion contrast g(k*)-g(k_max) on cheap-land
    ks = sorted(kgrid)
    gq = {k: out["cheap_land"][k]["g"]["q05"] for k in ks}
    kstar = max(ks, key=lambda k: (gq[k] if np.isfinite(gq[k]) else -np.inf))
    out["erosion"] = {
        "k_star": int(kstar), "g_star": gq[kstar],
        "k_max": int(ks[-1]), "g_max": gq[ks[-1]],
        "drop": (gq[kstar] - gq[ks[-1]]) if np.isfinite(gq[ks[-1]]) else np.nan,
    }
    # secondary: gap growth f-g on cheap-land
    out["gap"] = {int(k): (out["cheap_land"][k]["f"]["q05"] - out["cheap_land"][k]["g"]["q05"])
                  for k in ks}

    # invariant checks
    viol = [int(k) for k in ks
            if np.isfinite(out["cheap_land"][k]["f"]["q05"])
            and np.isfinite(out["cheap_land"][k]["g"]["q05"])
            and out["cheap_land"][k]["g"]["q05"] > out["cheap_land"][k]["f"]["q05"] + 1e-6]
    out["invariant_g_le_f_violations"] = viol
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=12)
    ap.add_argument("--load-scale", type=float, default=1.2)
    ap.add_argument("--n-fleets", type=int, default=10, help="monte-carlo siting draws")
    ap.add_argument("--n-targets", type=int, default=8, help="D-grid size for k_min(D)")
    ap.add_argument("--node-stride", type=int, default=1, help="subsample nodes for headroom")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--land-cost", default=None,
                    help="land $/acre CSV; omit for UNIFORM siting (cross-network default).")
    ap.add_argument("--bad-buses-json", default=DEFAULT_BAD_BUSES_JSON,
                    help="frozen pathological-bus manifest for citable runs; pass '' to redetect")
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS.keys()),
                    help="which DC workload profiles to run (default all). Use 'inference' for the "
                         "primary arm; run 'training' as a separate job for the robustness arm so "
                         "neither single-workload job risks a walltime timeout.")
    ap.add_argument("--outdir", default="development/results/spread_frontier")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_snaps = 4
        args.n_fleets = 3
        args.n_targets = 5
        args.node_stride = 16
        args.nboot = 300
    os.makedirs(args.outdir, exist_ok=True)

    print(f"## DC SPREAD FRONTIER (tag={args.tag}) load_scale={args.load_scale}")
    pn, n_nodes, bad, clean, weights, panels = build_panels(args)
    print(f"   network: {n_nodes} nodes, {len(bad)} pathological, {len(clean)} clean, "
          f"{args.n_snaps}h panel, {args.n_fleets} MC fleets")

    rng = np.random.default_rng(args.seed)
    bad_manifest = args.bad_buses_json or None
    out = {"load_scale": args.load_scale, "n_snaps": args.n_snaps, "n_fleets": args.n_fleets,
           "n_clean": len(clean), **cleaning_metadata(bad_manifest, bad),
           "power_unit": POWER_UNIT, "cost_unit": COST_UNIT, "workloads": {}}
    path = os.path.join(args.outdir, f"spread_frontier_{args.tag}.json")
    for w in args.workloads:
        print(f"\n-- workload={w}")
        r = run_workload(panels[w], clean, weights, args, rng)
        st = r["slope_test"]
        er = r["erosion"]
        print(f"   slope k_min(D) vs D = {st['slope']:.2f}  95%CI[{st['lo']:.2f},{st['hi']:.2f}]  "
              f"P(slope>0)={st['p_gt0']:.3f}")
        print(f"   erosion: g(k*={er['k_star']})={er['g_star']:.2f} vs "
              f"g(k_max={er['k_max']})={er['g_max']:.2f}  drop={er['drop']:.2f}")
        if r["invariant_g_le_f_violations"]:
            print(f"   !! g>f invariant violated at k={r['invariant_g_le_f_violations']}")
        out["workloads"][w] = r
        json.dump(out, open(path, "w"), indent=2, default=float)  # incremental: survive timeout/preempt
        print(f"   saved (incremental, {len(out['workloads'])} workload(s)) -> {path}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
