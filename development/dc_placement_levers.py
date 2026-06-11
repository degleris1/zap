"""
CONGESTION-MITIGATION LEVERS: a calibrated, FAIR cost-effectiveness frontier
(cleaned 490-node WECC, must-serve DC).

Compares three levers for relieving grid congestion, every one optimized against
the SAME congestion surrogate (sum_u^2 line utilization) so the comparison is
apples-to-apples (the old 101-node table compared a DC tuned for LMP against grid
levers tuned for dispatch cost -- not a fair fight, and on a copperplate to boot):

  (1) DC-DISTRIBUTION (load-side): spread a fixed DC fleet over many cheap-land
      sites vs concentrate it. The DC gets built either way, so distributing is
      ~$0 of GRID capital. Relief comes from the distribute-vs-concentrate gap.
  (2) TRANSMISSION expansion (grid-side): lines may grow up to (1+epsL)*nominal.
  (3) GENERATION expansion (grid-side): gens may grow up to (1+epsG)*nominal.

The deliverable is a FRONTIER: x = grid investment $ (annualized), y = congestion
RELIEF measured two co-equal ways, both INCREMENTAL to the no-DC base:
    * Delta(LMP-dispersion p90-p10)   (more negative = more relief)
    * Delta(must-serve feasibility fraction)  (more positive = more relief)
DC-distribution sits at x ~= 0 grid-$; tx/gen trace rising-cost curves. NO raw
LMP-level claims -- dispersion + feasibility only.

Grid levers are optimized on a representative congested hour with the gradient
planner (LineUtilizationObjective sum_u^2 + InvestmentObjective, box bounds), then
the resulting FIXED capacities are evaluated across the FULL panel via the same
incremental-metrics path used by the congestion study. If the gradient planner is
unreliable (loss up / NaN / non-monotone in eps), we fall back to a robust greedy
marginal-relief fill and log which engine was used.

Usage:
  .venv/bin/python development/dc_placement_levers.py --quick
  .venv/bin/python development/dc_placement_levers.py --tag full
"""
import argparse, os, json, sys
from copy import deepcopy

import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_placement_study import (load_dc_profile, sample_panel_indices, with_dc, metrics,
                                find_bad_buses, load_land_weights, draw_fleet, dispatch,
                                VOLL, COST_UNIT)
from dc_placement_integrated import build_raw_hour, WORKLOADS
from dc_placement_congestion import cleaned_panel, incr, boot_ci, pack
from dc_cleaning import DEFAULT_BAD_BUSES_JSON, cleaning_metadata, resolve_bad_buses

import zap

# DC build cost: Gamma_build = $12M/MW capex + land (0.25 acres/MW * land_$/acre),
# annualized over 20 yr @ 7% (capital recovery factor).
CRF = 0.07 * (1.07 ** 20) / (1.07 ** 20 - 1)        # ~= 0.09439
DC_CAPEX_PER_MW = 12.0e6                              # $/MW
DC_ACRES_PER_MW = 0.25

# CO-LOCATED on-site generation cost (the new wall-clearing lever): on-site gas/CCGT
# overnight capex ~$900/kW (CENTRAL of the $700-1100/kW range), annualized with the
# SAME CRF the study uses everywhere -> ONSITE_GEN_PER_MW_YR $/MW/yr. This is priced
# CONSISTENTLY with the in-place gen lever: that lever charges added MW at the
# generator's own PyPSA annualized $/MW (capacity-weighted ~$185/kW-yr across the
# existing fleet, which is inflated by renewables+storage; on-site gas at ~$85/kW-yr
# is a hair cheaper, as expected for a single dispatchable technology). We report the
# co-located lever's spend BOTH in $/yr and as the equivalent internal capital_cost
# (= $/MW-yr / usd_gen) so the reader can see it lives in the same units as in-place.
ONSITE_GEN_OVERNIGHT_PER_MW = 900.0e3                # $/MW overnight (~$900/kW on-site gas)
ONSITE_GEN_PER_MW_YR = ONSITE_GEN_OVERNIGHT_PER_MW * CRF   # ~= $84,953/MW/yr
ONSITE_GEN_LINEAR_COST = 1.0e-3                      # internal $/MWh: cheap enough to run on-site
DC_FLEX_VOLL = VOLL / COST_UNIT - 1.0e-6             # just below firm-load VOLL, in internal units


# --------------------------------------------------------------------------- #
def annualize(capital_usd):
    return float(capital_usd) * CRF


def dc_build_cost(terms, caps_gw, land):
    """Annualized DC build cost ($/yr): $12M/MW capex + 0.25 acres/MW * land_$/acre,
    summed over sites. caps in GW -> MW via *1000. The capex term is identical for
    concentrate vs distribute (same total MW); only the land term differs slightly by
    site, which is what the cost column reflects."""
    mw = np.asarray(caps_gw, float) * 1.0e3
    capex = (mw * DC_CAPEX_PER_MW).sum()
    land_cost = sum(mw_i * DC_ACRES_PER_MW * float(land[t]) for mw_i, t in zip(mw, terms))
    return annualize(capex + land_cost)


# --------------------------------------------------------------------------- #
def n_binding_lines(net, devs):
    """How many AC lines bind at the no-DC base (used to pick the representative hour)."""
    oc = net.dispatch(devs, time_horizon=1, solver=cp.CLARABEL, add_ground=False)
    A = devs[3]
    mu = (np.asarray(oc.local_inequality_duals[3][0]) +
          np.asarray(oc.local_inequality_duals[3][1]))
    return int(np.sum(mu.max(axis=1) > 1e-4))


def representative_hour(panel):
    """Pick the panel hour with the most binding AC lines at base (most congested)."""
    counts = []
    for (net, devs, _, _) in panel:
        try:
            counts.append(n_binding_lines(net, devs))
        except Exception:
            counts.append(-1)
    return int(np.argmax(counts)), counts


# --------------------------------------------------------------------------- #
# Grid-lever optimization: SHADOW-PRICE-RANKED EXPANSION.
#
# Why not the gradient planner? Differentiating the 490-node DC-OPF w.r.t. 1250 line
# (or 844 gen) capacities is numerically fragile EXACTLY on the congested hours we
# care about -- cvxpy/CLARABEL returns a non-optimal status and network.dispatch
# raises (see zap/network.py:497). A brute-force finite-difference greedy is correct
# but does O(steps x n_assets) full dispatches (~2.8 h per tx solve at 490 nodes).
#
# Instead we use LP duality: the marginal congestion relief of relaxing an asset's
# capacity limit by a unit is exactly its binding shadow price (the dual of that
# limit). So we (i) dispatch the congested rep-hour WITH the DC placed ONCE, (ii)
# expand the assets whose limit is binding (dual > tol) up to (1+eps)*nominal -- these
# are precisely the assets that relieve congestion -- and (iii) price the expansion at
# the device capital cost. One dispatch per eps; deterministic; monotone in eps.
def calibrate_usd(pn, devs):
    """Factor converting model-internal capital cost to real annualized $/yr, derived
    by matching total device capital to PyPSA's. Returns (usd_per_internal_line,
    usd_per_internal_gen) plus a report dict. Falls back to 1.0 (with a loud flag) if
    PyPSA capital costs are absent/zero so totals can't be matched."""
    rep = {}
    def factor(dev_idx, pn_df, kind):
        cc = np.asarray(devs[dev_idx].capital_cost, float).ravel()
        nom = np.asarray(devs[dev_idx].nominal_capacity, float).ravel()
        internal_total = float((cc * nom).sum())
        real_total = np.nan
        try:
            real_total = float((pn_df["capital_cost"].fillna(0.0).values *
                                pn_df["s_nom" if kind == "line" else "p_nom"].fillna(0.0).values).sum())
        except Exception:
            real_total = np.nan
        # Per-element scaling (proration x cost_unit x power_unit) is uniform across a
        # device class, so the total-to-total ratio is exact even if pn has extra empty
        # rows (dropped on import; they carry p_nom/s_nom~0 so add ~0 to real_total).
        ok = np.isfinite(real_total) and real_total > 0 and internal_total > 0
        f = (real_total / internal_total) if ok else 1.0
        rep[kind] = {"internal_total": internal_total, "real_total": real_total,
                     "n_dev": len(cc), "n_pn": len(pn_df), "usd_per_internal": f,
                     "count_mismatch": len(cc) != len(pn_df), "calibrated": bool(ok)}
        return f
    usd_line = factor(3, pn.lines, "line")
    usd_gen = factor(0, pn.generators, "gen")
    return usd_line, usd_gen, rep


def _binding_score(net, base_devs, dev_idx, cap_arr, terms, caps, lf, tol=1e-4,
                   scales=(1.0, 0.5, 0.25)):
    """LP shadow price per asset of device `dev_idx`, judged WITH the DC placed at the current
    capacities `cap_arr`. Lines: sum of the two flow-limit duals; gens: the upper-bound dual.
    If the must-serve DC is infeasible at full size, scale it DOWN to the largest feasible
    fraction so the duals still point at DC-INDUCED congestion (not the no-DC base -- that was
    the M1 bug). Returns (score array, relaxed_flag) or (None, True) if infeasible even small."""
    d = deepcopy(base_devs)
    d[dev_idx].nominal_capacity = np.asarray(cap_arr, float).reshape(
        np.asarray(base_devs[dev_idx].nominal_capacity).shape).copy()
    for s in scales:
        oc = dispatch(net, with_dc(d, terms, [c * s for c in caps], lf), 1)
        if oc is not None:
            ld = oc.local_inequality_duals
            if dev_idx == 3:
                sc = (np.asarray(ld[3][0]) + np.asarray(ld[3][1])).max(axis=1)
            else:
                sc = np.asarray(ld[0][0]).max(axis=1)
            return sc, (s < 1.0)
    return None, True


def expand_iterative(net, base_devs, dev_idx, eps, usd_per_internal, terms, caps, lf,
                     rounds=4, tol=1e-4):
    """ITERATIVE re-rank expansion (fixes the single-shot bias): expand the currently-binding
    assets toward the (1+eps)*nominal ceiling over several rounds, re-dispatching between rounds
    so congestion that MOVES to a neighbouring asset is also relieved (a one-shot expansion
    freezes the binding set and understates the grid lever -- critic finding C2). Duals come from
    a DC-loaded dispatch, relaxed to the feasibility boundary if must-serve is infeasible
    (critic finding M1). Returns (new_cap reshaped, inv_usd_real, n_expanded, relaxed_frac)."""
    eta = np.asarray(base_devs[dev_idx].nominal_capacity, float)
    shape = eta.shape
    eta0 = eta.ravel().copy()
    cap_cost = np.asarray(base_devs[dev_idx].capital_cost, float).ravel()
    if eps <= 0:
        return eta.copy(), 0.0, 0, 0.0
    ceiling = (1.0 + eps) * eta0
    cap = eta0.copy()
    step = eps / rounds                          # fraction of eta0 added per round
    touched = np.zeros_like(cap, bool)
    relaxed = 0
    for _ in range(rounds):
        score, was_relaxed = _binding_score(net, base_devs, dev_idx, cap, terms, caps, lf, tol)
        relaxed += int(was_relaxed)
        if score is None:
            break
        room = cap < ceiling - 1e-12
        binding = (score > tol) & room
        if not binding.any():
            break
        cap[binding] = np.minimum(ceiling[binding], cap[binding] + step * eta0[binding])
        touched |= binding
    inv = float((cap_cost * (cap - eta0)).sum()) * float(usd_per_internal)
    return cap.reshape(shape), inv, int(touched.sum()), relaxed / float(rounds)


# --------------------------------------------------------------------------- #
# Evaluate a FIXED set of capacities (from the rep-hour optimization) across the
# FULL panel: apply to each hour's devices[dev_idx].nominal_capacity (as a scale of
# the per-hour nominal -- capacities differ by hour only via line/gen scale, which
# is 1.0 here, so the absolute optimized array applies directly), inject the DC, and
# compute incremental dispersion + feasibility vs the no-DC base.
def eval_capacities_full(panel, base_ms, clean_nodes, dev_idx, new_cap, terms, caps, rng):
    """Apply optimized capacities to every panel hour, place the DC fleet (terms/caps),
    return (d_disp boot CI, d_feas, raw arrays). base_ms is the no-DC base at NOMINAL
    capacities (the honest baseline -- relief is measured against the un-expanded grid)."""
    H = len(panel)
    d_disp = np.full(H, np.nan); feas = np.zeros(H, bool)
    shape = np.asarray(panel[0][1][dev_idx].nominal_capacity).shape
    for i, ((net, devs, _, lf), b) in enumerate(zip(panel, base_ms)):
        d2 = deepcopy(devs)
        d2[dev_idx].nominal_capacity = new_cap.reshape(shape).copy()
        m = metrics(net, with_dc(d2, terms, caps, lf), d2, 1,
                    dc_terminals=terms, price_nodes=clean_nodes)
        feas[i] = m["feasible"]
        if m["feasible"] and b["feasible"]:
            d_disp[i] = m["lmp_disp"] - b["lmp_disp"]
    return boot_ci(d_disp, rng), float(feas.mean()), d_disp, feas


def base_dc_feas(panel, base_ms, clean_nodes, terms, caps, rng):
    """The DC placed on the UN-expanded grid (the reference the grid levers improve on,
    and what DC-distribution itself delivers). Returns (d_disp CI, d_feas)."""
    r = incr(panel, base_ms, clean_nodes, terms, caps)
    return boot_ci(r["d_disp"], rng), float(r["feas"].mean())


# --------------------------------------------------------------------------- #
# FLEXIBLE DC lever: split the same nameplate fleet into a firm must-serve block
# and a curtailable block. The flexible block's curtailment penalty is deliberately
# just below firm-load VOLL: high enough that physically deliverable flexible demand
# is served, but lower than base-load VOLL so it does not cannibalize firm demand.
def with_flexible_dc(devices, terminals, caps, lf, flex_frac, flex_voll=DC_FLEX_VOLL):
    """Append a firm+curtailable DC split.

    Returns (devices_with_dc, flex_device_index, requested_flex_gw). When flex_frac is
    zero this intentionally falls back to with_dc(...) so the 0% flex row is exactly
    comparable to the concentrated baseline.
    """
    terminals = list(map(int, terminals))
    caps = np.asarray(caps, float)
    flex_frac = float(flex_frac)
    if flex_frac <= 0.0:
        return with_dc(devices, terminals, caps, lf), None, 0.0
    if flex_frac > 1.0:
        raise ValueError(f"flex_frac must be in [0,1], got {flex_frac}")

    devs = deepcopy(devices)
    firm_caps = caps * (1.0 - flex_frac)
    flex_load = caps * flex_frac * float(lf)
    if np.any(firm_caps > 1e-12):
        firm = zap.DataCenterLoad(
            num_nodes=devs[0].num_nodes,
            terminal=np.asarray(terminals, dtype=int),
            profiles=[np.array([lf], dtype=float)] * len(terminals),
            nominal_capacity=firm_caps,
            linear_cost=np.zeros(len(terminals)),
            settime_horizon=1,
            capital_cost=np.zeros(len(terminals)),
        )
        devs.append(firm)
    flex_idx = None
    requested_flex = float(flex_load.sum())
    if requested_flex > 1e-12:
        flex = zap.Load(
            num_nodes=devs[0].num_nodes,
            terminal=np.asarray(terminals, dtype=int),
            load=flex_load,
            linear_cost=np.full(len(terminals), float(flex_voll)),
        )
        flex_idx = len(devs)
        devs.append(flex)
    return devs, flex_idx, requested_flex


def eval_flexible_full(panel, base_ms, clean_nodes, terms, caps, flex_frac, rng,
                       flex_voll=DC_FLEX_VOLL, shed_tol_pct=1e-6):
    """Evaluate a firm+flex DC split over the panel.

    `firm_feas` is the solver feasibility of the firm block with flexible demand allowed
    to curtail. `deliverable_feas` additionally requires no material increase in
    non-DC base-load shedding. Flexible service is reported separately so flex is not
    hidden as load shedding.
    """
    H = len(panel)
    d_disp = np.full(H, np.nan)
    firm_feas = np.zeros(H, bool)
    deliverable_feas = np.zeros(H, bool)
    served_frac = np.ones(H, float)
    served_gw = np.zeros(H, float)
    requested_gw = np.zeros(H, float)
    base_shed_delta = np.full(H, np.nan)

    for i, ((net, devs, _, lf), b) in enumerate(zip(panel, base_ms)):
        d2, flex_idx, requested = with_flexible_dc(devs, terms, caps, lf, flex_frac, flex_voll)
        requested_gw[i] = requested
        oc = dispatch(net, d2, 1)
        if oc is None:
            served_frac[i] = 0.0 if requested > 0 else 1.0
            continue
        m = metrics(net, d2, devs, 1, dc_terminals=terms, price_nodes=clean_nodes, oc=oc)
        firm_feas[i] = bool(m["feasible"])
        if requested > 0 and flex_idx is not None:
            served = float(-np.asarray(oc.power[flex_idx][0]).sum())
            served_gw[i] = max(served, 0.0)
            served_frac[i] = max(0.0, min(1.0, served_gw[i] / max(requested, 1e-12)))
        else:
            served_gw[i] = 0.0
            served_frac[i] = 1.0
        if m["feasible"] and b["feasible"]:
            d_disp[i] = m["lmp_disp"] - b["lmp_disp"]
            base_shed_delta[i] = m["shed_pct"] - b["shed_pct"]
            deliverable_feas[i] = base_shed_delta[i] <= shed_tol_pct

    details = {
        "deliverable_feas": deliverable_feas,
        "served_frac": served_frac,
        "served_gw": served_gw,
        "requested_flex_gw": requested_gw,
        "base_shed_delta_pct": base_shed_delta,
    }
    return boot_ci(d_disp, rng), float(firm_feas.mean()), d_disp, firm_feas, details


# --------------------------------------------------------------------------- #
def expand_uniform(eta0_arr, cap_cost, X, usd_per_internal):
    """Best-case 'dumb' transmission reinforcement: expand EVERY line by the same X (no dual
    targeting), so it cannot be handicapped by mis-ranking (critic finding: targeted expansion
    of only the rep-hour-binding lines understates what transmission can do for deliverability).
    Returns (new_cap, inv_usd_real). Cost = X * (sum capital_cost*nominal) * usd."""
    new = (1.0 + X) * eta0_arr
    inv = float((np.asarray(cap_cost).ravel() * (new - eta0_arr).ravel()).sum()) * float(usd_per_internal)
    return new, inv


# --------------------------------------------------------------------------- #
# CO-LOCATED GENERATION lever (the key new one). The in-place gen lever scales the
# EXISTING upstream generators in place -- behind the binding corridor, so the extra
# MW can't cross and feasibility is flat. Co-located generation puts NEW dispatchable
# capacity AT the DC buses, so the must-serve DC's NET grid draw at those buses falls
# (draw - on_site_gen) and it stops importing across the saturated corridors. At
# on-site capacity = the DC draw, net draw -> 0 and the wall is cleared with NO grid
# import. This is a load-side / behind-the-meter generation build, NOT grid expansion.
def colocated_gen_device(devs, terms, g_total_gw):
    """A NEW dispatchable Generator placed AT the DC buses (`terms`), total g_total_gw
    GW split evenly across the sites. dynamic_capacity=1 (always available), cheap
    linear_cost so it runs ahead of grid imports, capital_cost=0 (its cost is charged
    separately at ONSITE_GEN_PER_MW_YR -- see colocated_cost). Capacities are in GW
    (POWER_UNIT=1000), the same internal units as the DC nominal_capacity it offsets."""
    nb = len(terms)
    g_per_bus = (g_total_gw / nb) if nb else 0.0
    return zap.Generator(
        num_nodes=devs[0].num_nodes, terminal=np.asarray(terms, dtype=int),
        nominal_capacity=np.full((nb, 1), g_per_bus, dtype=float),
        dynamic_capacity=np.ones((nb, 1), dtype=float),
        linear_cost=np.full((nb, 1), ONSITE_GEN_LINEAR_COST, dtype=float),
        capital_cost=np.zeros((nb, 1), dtype=float),
    )


def colocated_cost(g_total_gw):
    """Annualized $/yr of g_total_gw GW of on-site generation = added MW * $/MW-yr."""
    return float(g_total_gw) * 1.0e3 * ONSITE_GEN_PER_MW_YR


def eval_colocated_full(panel, base_ms, clean_nodes, terms, caps, g_co, rng):
    """Place the DC fleet (terms/caps) AND g_co GW of on-site generation at those buses
    across every panel hour; return (d_disp boot CI, d_feas, d_disp vec, feas vec).
    Mirrors eval_capacities_full but injects an extra device instead of expanding one.
    g_co=0 reproduces the bare-fleet anchor exactly (no device added)."""
    H = len(panel)
    d_disp = np.full(H, np.nan); feas = np.zeros(H, bool)
    for i, ((net, devs, _, lf), b) in enumerate(zip(panel, base_ms)):
        d2 = list(deepcopy(devs))
        if g_co > 0:
            d2 = d2 + [colocated_gen_device(devs, terms, g_co)]
        m = metrics(net, with_dc(d2, terms, caps, lf), d2, 1,
                    dc_terminals=terms, price_nodes=clean_nodes)
        feas[i] = m["feasible"]
        if m["feasible"] and b["feasible"]:
            d_disp[i] = m["lmp_disp"] - b["lmp_disp"]
    return boot_ci(d_disp, rng), float(feas.mean()), d_disp, feas


# --------------------------------------------------------------------------- #
# JOINT generation + transmission lever: expand binding AC lines (dev 3) AND binding
# EXISTING generators (dev 0) TOGETHER, alternating rounds so relief that moves from a
# line to a gen (or back) is chased by both. Cost = line$ + gen$ (both via the study's
# usd calibration + CRF). The question: does combining beat tx-alone, or is the
# in-place-gen spend simply wasted because the wall is transmission-bound?
def expand_joint_gen_tx(net, base_devs, eps, usd_line, usd_gen, terms, caps, lf,
                        rounds=4, tol=1e-4):
    """Alternate single-round line/gen expansions for `rounds` rounds (re-dispatching
    between, via _binding_score, so the binding set is refreshed). Returns
    (new_cap_line, new_cap_gen, inv_line_usd, inv_gen_usd, n_line, n_gen, relaxed_frac)."""
    cap_line = np.asarray(base_devs[3].nominal_capacity, float).copy()
    cap_gen = np.asarray(base_devs[0].nominal_capacity, float).copy()
    if eps <= 0:
        return (cap_line, cap_gen, 0.0, 0.0, 0, 0, 0.0)
    eta_line = cap_line.ravel().copy(); eta_gen = cap_gen.ravel().copy()
    cc_line = np.asarray(base_devs[3].capital_cost, float).ravel()
    cc_gen = np.asarray(base_devs[0].capital_cost, float).ravel()
    ceil_line = (1.0 + eps) * eta_line; ceil_gen = (1.0 + eps) * eta_gen
    cl = eta_line.copy(); cg = eta_gen.copy()
    step = eps / rounds
    tl = np.zeros_like(cl, bool); tg = np.zeros_like(cg, bool)
    relaxed = 0
    for _ in range(rounds):
        # ---- transmission half-round (duals judged with current line+gen caps) ----
        d = deepcopy(base_devs)
        d[3].nominal_capacity = cl.reshape(cap_line.shape).copy()
        d[0].nominal_capacity = cg.reshape(cap_gen.shape).copy()
        sl, rl = _binding_score(net, d, 3, cl, terms, caps, lf, tol)
        relaxed += int(rl)
        if sl is not None:
            room = cl < ceil_line - 1e-12
            b = (sl > tol) & room
            cl[b] = np.minimum(ceil_line[b], cl[b] + step * eta_line[b]); tl |= b
        # ---- generation half-round (existing gens, in place) ----
        d2 = deepcopy(base_devs)
        d2[3].nominal_capacity = cl.reshape(cap_line.shape).copy()
        d2[0].nominal_capacity = cg.reshape(cap_gen.shape).copy()
        sg, rg = _binding_score(net, d2, 0, cg, terms, caps, lf, tol)
        relaxed += int(rg)
        if sg is not None:
            room = cg < ceil_gen - 1e-12
            b = (sg > tol) & room
            cg[b] = np.minimum(ceil_gen[b], cg[b] + step * eta_gen[b]); tg |= b
    inv_line = float((cc_line * (cl - eta_line)).sum()) * float(usd_line)
    inv_gen = float((cc_gen * (cg - eta_gen)).sum()) * float(usd_gen)
    return (cl.reshape(cap_line.shape), cg.reshape(cap_gen.shape),
            inv_line, inv_gen, int(tl.sum()), int(tg.sum()), relaxed / float(2 * rounds))


def eval_joint_full(panel, base_ms, clean_nodes, cap_line, cap_gen, terms, caps, rng):
    """Apply BOTH expanded line (dev 3) and gen (dev 0) capacities to every panel hour,
    place the DC, return (d_disp CI, d_feas, d_disp vec, feas vec)."""
    H = len(panel)
    d_disp = np.full(H, np.nan); feas = np.zeros(H, bool)
    sl = np.asarray(panel[0][1][3].nominal_capacity).shape
    sg = np.asarray(panel[0][1][0].nominal_capacity).shape
    for i, ((net, devs, _, lf), b) in enumerate(zip(panel, base_ms)):
        d2 = deepcopy(devs)
        d2[3].nominal_capacity = cap_line.reshape(sl).copy()
        d2[0].nominal_capacity = cap_gen.reshape(sg).copy()
        m = metrics(net, with_dc(d2, terms, caps, lf), d2, 1,
                    dc_terminals=terms, price_nodes=clean_nodes)
        feas[i] = m["feasible"]
        if m["feasible"] and b["feasible"]:
            d_disp[i] = m["lmp_disp"] - b["lmp_disp"]
    return boot_ci(d_disp, rng), float(feas.mean()), d_disp, feas


def _accum():
    return {"inv": [], "feas": [], "disp": [], "nexp": [], "rel": []}


def _flex_accum():
    return {"firm_feas": [], "deliverable_feas": [], "disp": [],
            "served_frac": [], "served_gw": [], "requested_flex_gw": [],
            "base_shed_delta_pct": []}


def _eval_into(a, inv, cap, terms, caps, panel, base_ms, clean_nodes, dev_idx, rng, nexp, rel):
    """Evaluate a fixed expansion `cap` with fleet (terms,caps) across the panel and append the
    per-pool feasibility + the per-hour incremental-dispersion samples to accumulator `a`."""
    _, feas, dvec, _ = eval_capacities_full(panel, base_ms, clean_nodes, dev_idx, cap,
                                            terms, caps, rng)
    a["inv"].append(float(inv)); a["feas"].append(float(feas))
    a["disp"].extend(np.asarray(dvec)[np.isfinite(dvec)].tolist())
    a["nexp"].append(int(nexp)); a["rel"].append(float(rel))


def _append(a, inv, feas, dvec, nexp, rel):
    """Append a precomputed (feas, dvec) evaluation to accumulator `a` (for levers that
    don't expand a single device via eval_capacities_full -- co-located gen, joint)."""
    a["inv"].append(float(inv)); a["feas"].append(float(feas))
    a["disp"].extend(np.asarray(dvec)[np.isfinite(dvec)].tolist())
    a["nexp"].append(int(nexp)); a["rel"].append(float(rel))


def _append_flex(a, firm_feas, dvec, details):
    """Append one pooled firm+flex evaluation."""
    a["firm_feas"].append(float(firm_feas))
    a["disp"].extend(np.asarray(dvec)[np.isfinite(dvec)].tolist())
    for key in ("deliverable_feas", "served_frac", "served_gw",
                "requested_flex_gw", "base_shed_delta_pct"):
        vals = np.asarray(details[key], float)
        a[key].extend(vals[np.isfinite(vals)].tolist())


def write_json(path, out, partial, progress=None):
    out["_partial"] = bool(partial)
    if progress is not None:
        out["progress"] = progress
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)


def conc_vs_dist_progress(panel, base_ms, clean_nodes, nodes, weights, rng, B, n_pools,
                          pool_size, progress_cb=None):
    """Progress-reporting copy of congestion_sweeps.conc_vs_dist for long canonical runs."""
    m_conc, m_dist = int(np.ceil(B)), pool_size
    gap, cd, dd, cf, df = [], [], [], [], []
    wdict = {n: weights[n] for n in nodes}
    for pi in range(n_pools):
        if progress_cb is not None:
            progress_cb(pi, "paired_draw")
        pool = draw_fleet(rng, nodes, wdict, B, B / pool_size)[0]
        tc, cc = pack(pool, B, m_conc)
        if progress_cb is not None:
            progress_cb(pi, "paired_concentrate")
        rc = incr(panel, base_ms, clean_nodes, tc, cc)
        td, cdp = pack(pool, B, m_dist)
        if progress_cb is not None:
            progress_cb(pi, "paired_distribute")
        rd = incr(panel, base_ms, clean_nodes, td, cdp)
        both = np.isfinite(rc["d_disp"]) & np.isfinite(rd["d_disp"])
        gap.extend((rc["d_disp"][both] - rd["d_disp"][both]).tolist())
        cd.extend(rc["d_disp"][np.isfinite(rc["d_disp"])].tolist())
        dd.extend(rd["d_disp"][np.isfinite(rd["d_disp"])].tolist())
        cf.append(float(rc["feas"].mean()))
        df.append(float(rd["feas"].mean()))
        if progress_cb is not None:
            progress_cb(pi + 1, "paired_complete")
    return {"gap": boot_ci(gap, rng), "conc_disp": boot_ci(cd, rng),
            "dist_disp": boot_ci(dd, rng), "conc_feas": float(np.mean(cf)),
            "dist_feas": float(np.mean(df)), "m_conc": m_conc, "m_dist": m_dist}


def interp_cost_for_target(points, target, increasing):
    """Given (cost, metric) points, return the cost at which `metric` first reaches `target`.
    `increasing=True` for metrics that rise toward the target (e.g. feasibility climbing to
    distribution's level); `increasing=False` for metrics that fall (e.g. dispersion dropping).
    Linear-interpolate; return None if the target is never reached within the spend range."""
    pts = sorted([(float(c), float(m)) for c, m in points if np.isfinite(c) and np.isfinite(m)])
    if not pts:
        return None
    reached = (lambda y: y >= target) if increasing else (lambda y: y <= target)
    if reached(pts[0][1]):
        return pts[0][0]                       # already met at the cheapest point
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if reached(y1):
            if y1 == y0:
                return x1
            t = (target - y0) / (y1 - y0)
            return float(x0 + max(0.0, min(1.0, t)) * (x1 - x0))
    return None                                # never reached within eps range


def interp_grid_cost_for_relief(curve, target_disp):
    """Given a lever curve [{eps, inv_usd, d_disp:[med,..]}...] sorted by inv, find the
    grid $ needed to MATCH a target Delta-dispersion (more negative = more relief).
    Linear-interpolate in (inv, d_disp). Returns $ or None if the lever never reaches
    the target within its eps range."""
    pts = [(c["inv_usd"], c["d_disp"][0]) for c in curve if np.isfinite(c["d_disp"][0])]
    pts = sorted(pts)
    if not pts:
        return None
    # d_disp should decrease (more relief) as inv grows; find first crossing
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 - target_disp) * (y1 - target_disp) <= 0 and y1 != y0:
            t = (target_disp - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))
    # never crossed: if the most-relieving point already beats target, ~free at min inv
    if pts[-1][1] <= target_disp:
        return float(pts[0][0])
    return None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=12)
    ap.add_argument("--load-scale", type=float, default=1.2)
    ap.add_argument("--fleets", default="3,6,10")
    ap.add_argument("--eps-line", default="0.02,0.05,0.10,0.20")
    ap.add_argument("--eps-gen", default="0.025,0.05,0.10")
    ap.add_argument("--x-uniform", default="0.1,0.25,0.5,1.0",
                    help="uniform (all-lines) transmission reinforcement factors -- the unbiased "
                         "best-case tx lever for clearing the deliverability wall")
    ap.add_argument("--g-colocated", default="2,4,6,8,10",
                    help="on-site (co-located) generation GW levels swept ON the concentrated "
                         "fleet -- the new wall-clearing load-side generation lever")
    ap.add_argument("--flex-fracs", default="0,0.05,0.10,0.20",
                    help="firm+flex DC sweep fractions; flex is curtailable at --flex-voll")
    ap.add_argument("--flex-voll", type=float, default=DC_FLEX_VOLL,
                    help="internal $/MWh curtailment penalty for flexible DC, just below base VOLL")
    ap.add_argument("--eps-joint", default="0.05,0.10,0.20",
                    help="joint gen+tx expansion fractions (lines AND existing gens together)")
    ap.add_argument("--n-pools", type=int, default=24)
    ap.add_argument("--pool-size", type=int, default=40)
    ap.add_argument("--land-cost", default="development/results/placement_study/node_land_cost.csv")
    ap.add_argument("--bad-buses-json", default=DEFAULT_BAD_BUSES_JSON,
                    help="frozen pathological-bus manifest for citable runs; pass '' to redetect")
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--outdir", default="development/results/placement_levers")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed workload/fleet cells already present in the output JSON")
    ap.add_argument("--quick", action="store_true",
                    help="fewer hours/pools/iters/eps for fast end-to-end validation")
    args = ap.parse_args()

    if args.quick:
        args.n_snaps = 6
        args.n_pools = 6
        args.eps_line = "0.05,0.20"
        args.eps_gen = "0.05"
        args.x_uniform = "0.25,1.0"
        args.g_colocated = "4,10"
        args.eps_joint = "0.10,0.20"

    fleets = [float(x) for x in args.fleets.split(",")]
    eps_line = [float(x) for x in args.eps_line.split(",")]
    eps_gen = [float(x) for x in args.eps_gen.split(",")]
    x_unif = [float(x) for x in args.x_uniform.split(",")]
    g_colo = [float(x) for x in args.g_colocated.split(",")]
    flex_fracs = [float(x) for x in args.flex_fracs.split(",")]
    eps_joint = [float(x) for x in args.eps_joint.split(",")]
    if any(f < 0.0 or f > 1.0 for f in flex_fracs):
        raise ValueError("--flex-fracs entries must be in [0,1]")

    os.makedirs(args.outdir, exist_ok=True)

    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    idx = sample_panel_indices(len(snaps), args.n_snaps, None, None, 0)

    # ---- shared base build/clean/panel (computed ONCE, reused everywhere) ----
    raw = [build_raw_hour(pn, snaps, h, args.load_scale, 1.0, 1.0) for h in idx]
    n_nodes = raw[0][0].num_nodes
    bad_manifest = args.bad_buses_json or None
    detected_bad = None
    if bad_manifest is None:
        raw0 = [build_raw_hour(pn, snaps, h, 1.0, 1.0, 1.0) for h in idx]
        detected_bad = find_bad_buses([(net, devs, dt, 0.0) for (net, devs, dt, hod) in raw0])
    bad = resolve_bad_buses(bad_manifest, detected_bad)
    clean_nodes = [i for i in range(n_nodes) if i not in set(bad)]
    land, weights = load_land_weights(args.land_cost, n_nodes)

    print(f"# DC PLACEMENT LEVERS (fair sum_u^2 frontier)  network={os.path.basename(args.network)}")
    print(f"# load x{args.load_scale}  bad buses {len(bad)} (fixed at x1.0)  clean nodes {len(clean_nodes)}")
    print(f"# fleets {fleets} GW  eps_line {eps_line}  eps_gen {eps_gen}  "
          f"n_pools {args.n_pools}{'  [QUICK]' if args.quick else ''}")

    # ---- representative congested hour (chosen on the no-DC base, workload-agnostic) ----
    inf_lf = load_dc_profile(WORKLOADS["inference"])
    panel_inf = cleaned_panel(raw, inf_lf, bad)
    rep, bind_counts = representative_hour(panel_inf)
    print(f"# representative congested hour = panel[{rep}] "
          f"({bind_counts[rep]} binding lines)  binding counts={bind_counts}")

    # ---- engine: shadow-price-ranked expansion (gradient planner is numerically
    # unreliable on congested hours; brute-greedy is intractable -- see expand_iterative) ----
    engine = "shadow_greedy"
    usd_line, usd_gen, usd_rep = calibrate_usd(pn, raw[0][1])
    print(f"# engine = {engine}  |  $/internal  line={usd_line:.4g} "
          f"(calibrated={usd_rep['line']['calibrated']})  gen={usd_gen:.4g} "
          f"(calibrated={usd_rep['gen']['calibrated']})")
    if not (usd_rep['line']['calibrated'] and usd_rep['gen']['calibrated']):
        print("# WARNING: capital-cost calibration FELL BACK to 1.0 for a lever -- "
              "grid $ axis is in MODEL UNITS, not real $. CRITIC: flag this.")
    # co-located gen cost assumption, stated up front and equated to the in-place units.
    colo_internal_cc = ONSITE_GEN_PER_MW_YR / float(usd_gen) if usd_gen else float("nan")
    print(f"# CO-LOCATED gen cost: ${ONSITE_GEN_OVERNIGHT_PER_MW/1e3:.0f}/kW overnight x CRF "
          f"{CRF:.4f} = ${ONSITE_GEN_PER_MW_YR/1e3:.1f}/kW-yr (${ONSITE_GEN_PER_MW_YR:,.0f}/MW-yr); "
          f"equivalent internal capital_cost {colo_internal_cc:.4g} (in-place-gen units)")

    stamp = args.tag
    path = os.path.join(args.outdir, f"levers_{stamp}.json")

    out = {"network": os.path.basename(args.network), "load_scale": args.load_scale,
           "n_snaps": args.n_snaps, "n_pools": args.n_pools,
           "pool_size": args.pool_size, "fleets": fleets,
           **cleaning_metadata(bad_manifest, bad), "clean_nodes_count": len(clean_nodes),
           "engine_used": engine, "usd_calibration": usd_rep, "rep_hour": rep,
           "bind_counts": bind_counts, "crf": CRF, "dc_capex_per_mw": DC_CAPEX_PER_MW,
           "onsite_gen_overnight_per_mw": ONSITE_GEN_OVERNIGHT_PER_MW,
           "onsite_gen_per_mw_yr": ONSITE_GEN_PER_MW_YR,
           "onsite_gen_internal_capital_cost_equiv": colo_internal_cc,
           "dc_flex_voll_internal": args.flex_voll,
           "dc_flex_voll_note": "curtailment penalty just below base-load VOLL; physical headroom metric",
           "workloads": {}}
    if args.resume and os.path.exists(path):
        prev = json.load(open(path))
        same_cleaning = (prev.get("bad_buses") == out.get("bad_buses")
                         and prev.get("bad_buses_json") == out.get("bad_buses_json"))
        same_panel = (prev.get("n_snaps") == out.get("n_snaps")
                      and prev.get("n_pools") == out.get("n_pools"))
        if same_cleaning and same_panel:
            prev.update({k: v for k, v in out.items()
                         if k not in ("workloads", "progress", "_partial",
                                      "monotonicity_violations")})
            out = prev
        else:
            print("# resume ignored: existing output metadata does not match this run")

    monotonicity_violations = []

    write_json(path, out, partial=True, progress="initialized")

    for w, wpath in WORKLOADS.items():
        lf = load_dc_profile(wpath)
        panel = cleaned_panel(raw, lf, bad)
        base_ms = [metrics(net, devs, devs, 1, price_nodes=clean_nodes)
                   for (net, devs, _, _) in panel]
        base_disp = float(np.median([m["lmp_disp"] for m in base_ms if m["feasible"]]))
        # representative congested hour for THIS workload (same index, workload's load-factor)
        rep_net_w, rep_devs_w, _, lf_rep_w = panel[rep]
        print(f"\n## WORKLOAD={w}  base(no DC) LMP-disp ${base_disp:.0f}")
        wout = out["workloads"].setdefault(w, {"base_disp": base_disp, "fleets": {}})
        wout["base_disp"] = base_disp

        for B in fleets:
            if str(int(B)) in wout.get("fleets", {}):
                print(f"\n   --- fleet B={B:.0f} GW: resume existing cell ---")
                continue
            print(f"\n   --- fleet B={B:.0f} GW ---")
            rng = np.random.default_rng(0)
            fout = {}

            # ===== robust paired H1 statistic (separate from the frontier; reported as-is) =====
            def paired_progress(pool_done, stage):
                write_json(path, out, partial=True,
                           progress={"workload": w, "fleet_gw": float(B),
                                     "pool_done": int(pool_done), "n_pools": args.n_pools,
                                     "stage": stage})

            cd = conc_vs_dist_progress(panel, base_ms, clean_nodes, clean_nodes, weights, rng,
                                       B, args.n_pools, args.pool_size, paired_progress)

            # ---- pooled lever frontier (SHARED pools across all levers; eps=0 computed once) ----
            # For each random cheap-land pool: form a concentrate and a distribute fleet, then
            # trace tx & gen expansion on the CONCENTRATED fleet (how much grid $ buys back
            # distribution's free relief) and tx on the DISTRIBUTED fleet (the complementarity
            # curve). Grid expansion uses the ITERATIVE re-rank engine with feasibility-relaxed
            # duals (critic C2/M1). eps=0 = bare fleet; the DC-distribution lever is the $0
            # conc->dist move on exactly this pooled basis (anchors are mutually consistent).
            m_conc, m_dist = cd["m_conc"], cd["m_dist"]
            wdict = {n: weights[n] for n in clean_nodes}
            eg_line, eg_gen = [0.0] + list(eps_line), [0.0] + list(eps_gen)
            eg_unif = [0.0] + list(x_unif)
            eg_colo = [0.0] + list(g_colo)          # on-site gen GW levels (0 = bare fleet)
            eg_jt = [0.0] + list(eps_joint)
            acc = {"TX": {e: _accum() for e in eg_line},
                   "GEN": {e: _accum() for e in eg_gen},
                   "TXU": {e: _accum() for e in eg_unif},
                   "DIST+TX": {e: _accum() for e in eg_line},
                   "COLO": {g: _accum() for g in eg_colo},
                   "JOINT": {e: _accum() for e in eg_jt},
                   "FLEX": {f: _flex_accum() for f in flex_fracs}}
            eta_line = np.asarray(rep_devs_w[3].nominal_capacity, float)
            eta_gen = np.asarray(rep_devs_w[0].nominal_capacity, float)
            cc_line = np.asarray(rep_devs_w[3].capital_cost, float)
            rng_p = np.random.default_rng(7)

            def pool_progress(pool_done, stage):
                write_json(path, out, partial=True,
                           progress={"workload": w, "fleet_gw": float(B),
                                     "pool_done": int(pool_done), "n_pools": args.n_pools,
                                     "stage": stage})

            for pi in range(args.n_pools):
                pool = draw_fleet(rng_p, clean_nodes, wdict, B, B / args.pool_size)[0]
                ct, ccp = pack(pool, B, m_conc)        # concentrate fleet
                dt, dcp = pack(pool, B, m_dist)        # distribute fleet
                # eps=0 anchors (bare fleets, no expansion)
                pool_progress(pi, "anchors_tx0")
                _eval_into(acc["TX"][0.0], 0.0, eta_line, ct, ccp, panel, base_ms, clean_nodes, 3, rng, 0, 0.0)
                pool_progress(pi, "anchors_gen0")
                _eval_into(acc["GEN"][0.0], 0.0, eta_gen, ct, ccp, panel, base_ms, clean_nodes, 0, rng, 0, 0.0)
                pool_progress(pi, "anchors_txu0")
                _eval_into(acc["TXU"][0.0], 0.0, eta_line, ct, ccp, panel, base_ms, clean_nodes, 3, rng, 0, 0.0)
                pool_progress(pi, "anchors_dist_tx0")
                _eval_into(acc["DIST+TX"][0.0], 0.0, eta_line, dt, dcp, panel, base_ms, clean_nodes, 3, rng, 0, 0.0)
                pool_progress(pi, "anchors")
                for e in eps_line:
                    pool_progress(pi, f"targeted_tx_{e:g}_conc")
                    cap, inv, nx, rl = expand_iterative(rep_net_w, rep_devs_w, 3, e, usd_line, ct, ccp, lf_rep_w)
                    _eval_into(acc["TX"][e], inv, cap, ct, ccp, panel, base_ms, clean_nodes, 3, rng, nx, rl)
                    pool_progress(pi, f"targeted_tx_{e:g}_dist")
                    capd, invd, nxd, rld = expand_iterative(rep_net_w, rep_devs_w, 3, e, usd_line, dt, dcp, lf_rep_w)
                    _eval_into(acc["DIST+TX"][e], invd, capd, dt, dcp, panel, base_ms, clean_nodes, 3, rng, nxd, rld)
                pool_progress(pi, "targeted_tx")
                for e in eps_gen:
                    cap, inv, nx, rl = expand_iterative(rep_net_w, rep_devs_w, 0, e, usd_gen, ct, ccp, lf_rep_w)
                    _eval_into(acc["GEN"][e], inv, cap, ct, ccp, panel, base_ms, clean_nodes, 0, rng, nx, rl)
                pool_progress(pi, "generation")
                for X in x_unif:    # UNBIASED best-case: uniform reinforcement of ALL lines
                    cap, inv = expand_uniform(eta_line, cc_line, X, usd_line)
                    _eval_into(acc["TXU"][X], inv, cap, ct, ccp, panel, base_ms, clean_nodes, 3, rng,
                               int(eta_line.size), 0.0)
                pool_progress(pi, "uniform_tx")
                # CO-LOCATED gen on the CONCENTRATED fleet: add on-site MW at the DC buses.
                _, f0, dv0, _ = eval_colocated_full(panel, base_ms, clean_nodes, ct, ccp, 0.0, rng)
                _append(acc["COLO"][0.0], 0.0, f0, dv0, 0, 0.0)
                for g in g_colo:
                    _, fc, dvc, _ = eval_colocated_full(panel, base_ms, clean_nodes, ct, ccp, g, rng)
                    _append(acc["COLO"][g], colocated_cost(g), fc, dvc, len(ct), 0.0)
                pool_progress(pi, "colocated")
                # FLEXIBLE DC on the CONCENTRATED fleet: a fraction of the same DC draw is
                # curtailable at a high value; firm feasibility and served flex are reported separately.
                for f in flex_fracs:
                    _, ff, dvf, _, det = eval_flexible_full(
                        panel, base_ms, clean_nodes, ct, ccp, f, rng,
                        flex_voll=args.flex_voll)
                    _append_flex(acc["FLEX"][f], ff, dvf, det)
                pool_progress(pi, "flexible_dc")
                # JOINT gen+tx on the CONCENTRATED fleet: expand binding lines AND gens together.
                _, fj0, dvj0, _ = eval_joint_full(panel, base_ms, clean_nodes, eta_line, eta_gen,
                                                  ct, ccp, rng)
                _append(acc["JOINT"][0.0], 0.0, fj0, dvj0, 0, 0.0)
                for e in eps_joint:
                    clJ, cgJ, ilJ, igJ, nlJ, ngJ, rlJ = expand_joint_gen_tx(
                        rep_net_w, rep_devs_w, e, usd_line, usd_gen, ct, ccp, lf_rep_w)
                    _, fj, dvj, _ = eval_joint_full(panel, base_ms, clean_nodes, clJ, cgJ,
                                                    ct, ccp, rng)
                    _append(acc["JOINT"][e], ilJ + igJ, fj, dvj, nlJ + ngJ, rlJ)
                pool_progress(pi + 1, "pool_complete")

            def _finalize(accd, eg, tag):
                curve, prev = [], None
                for e in eg:
                    a = accd[e]
                    disp_ci = boot_ci(a["disp"], rng)
                    feas = float(np.mean(a["feas"]))
                    feas_ci = boot_ci(a["feas"], rng, stat=np.mean)
                    curve.append({"eps": e, "inv_usd": float(np.median(a["inv"])),
                                  "feas": feas, "feas_ci": feas_ci, "d_disp": disp_ci,
                                  "n_pools": len(a["feas"]), "n_expanded": int(np.median(a["nexp"])),
                                  "relaxed_frac": float(np.mean(a["rel"]))})
                    if (prev is not None and np.isfinite(disp_ci[0]) and np.isfinite(prev)
                            and disp_ci[0] > prev * 1.01 + 0.5):     # relative tol vs bootstrap noise
                        monotonicity_violations.append({"workload": w, "B": B, "lever": tag,
                                                        "eps": e, "d_disp": disp_ci[0], "prev": prev})
                    prev = disp_ci[0]
                    print(f"      [{tag}] eps={e:.3f} inv=${curve[-1]['inv_usd']/1e6:8.1f}M "
                          f"exp={curve[-1]['n_expanded']:4d} relax={curve[-1]['relaxed_frac']:.2f} "
                          f"Ddisp ${disp_ci[0]:7.1f} feas {feas:.2f}[{feas_ci[1]:.2f},{feas_ci[2]:.2f}]")
                return curve

            def _finalize_flex(accd, fracs):
                curve = []
                for f in fracs:
                    a = accd[f]
                    disp_ci = boot_ci(a["disp"], rng)
                    firm_feas = float(np.mean(a["firm_feas"])) if a["firm_feas"] else float("nan")
                    firm_feas_ci = boot_ci(a["firm_feas"], rng, stat=np.mean)
                    deliverable_feas = (float(np.mean(a["deliverable_feas"]))
                                        if a["deliverable_feas"] else float("nan"))
                    served_frac_ci = boot_ci(a["served_frac"], rng, stat=np.mean)
                    served_gw_ci = boot_ci(a["served_gw"], rng, stat=np.mean)
                    requested_flex_gw_ci = boot_ci(a["requested_flex_gw"], rng, stat=np.mean)
                    shed_vals = np.asarray(a["base_shed_delta_pct"], float)
                    shed_vals = shed_vals[np.isfinite(shed_vals)]
                    base_shed_mean = float(np.mean(shed_vals)) if shed_vals.size else float("nan")
                    base_shed_max = float(np.max(shed_vals)) if shed_vals.size else float("nan")
                    curve.append({
                        "flex_frac": float(f),
                        "firm_feas": firm_feas,
                        "firm_feas_ci": firm_feas_ci,
                        "deliverable_feas_no_base_shed": deliverable_feas,
                        "served_flex_frac": served_frac_ci,
                        "served_flex_gw": served_gw_ci,
                        "requested_flex_gw": requested_flex_gw_ci,
                        "base_shed_delta_pct_mean": base_shed_mean,
                        "base_shed_delta_pct_max": base_shed_max,
                        "d_disp": disp_ci,
                        "n_pools": len(a["firm_feas"]),
                        "flex_voll_internal": float(args.flex_voll),
                    })
                    print(f"      [FLEX] frac={f:.2f} firm-feas {firm_feas:.2f}"
                          f"[{firm_feas_ci[1]:.2f},{firm_feas_ci[2]:.2f}] "
                          f"served-flex {served_frac_ci[0]:.2f} "
                          f"Ddisp ${disp_ci[0]:7.1f}")
                return curve

            tx_curve = _finalize(acc["TX"], eg_line, "TX")
            gen_curve = _finalize(acc["GEN"], eg_gen, "GEN")
            txu_curve = _finalize(acc["TXU"], eg_unif, "TXU")
            comb_curve = _finalize(acc["DIST+TX"], eg_line, "DIST+TX")
            colo_curve = _finalize(acc["COLO"], eg_colo, "COLO")
            joint_curve = _finalize(acc["JOINT"], eg_jt, "JOINT")
            flex_curve = _finalize_flex(acc["FLEX"], flex_fracs)
            for r, g in zip(colo_curve, eg_colo):        # COLO's "eps" axis is GW of on-site gen
                r["g_co_gw"] = float(g)
            fout["transmission"] = tx_curve              # targeted, cheap -> dispersion relief
            fout["transmission_uniform"] = txu_curve     # all-lines best-case -> wall-clearing $
            fout["generation"] = gen_curve               # in-place (upstream) -- flat
            fout["combined_dist_tx"] = comb_curve
            fout["colocated_generation"] = colo_curve    # NEW: on-site gen at DC buses
            fout["joint_gen_tx"] = joint_curve           # NEW: lines + existing gens together
            fout["flexible_dc"] = flex_curve              # NEW: demand flexibility, not grid capex

            # ===== DC-distribution lever: the $0 move from the concentrate anchor (tx eps=0) to
            # the distribute anchor (comb eps=0) -- same pooled basis as the grid curves. =====
            conc_anchor, dist_anchor = tx_curve[0], comb_curve[0]   # both eps=0 (no expansion)
            conc_feas, dist_feas = conc_anchor["feas"], dist_anchor["feas"]
            conc_disp, dist_disp = conc_anchor["d_disp"], dist_anchor["d_disp"]
            rng_b = np.random.default_rng(1)
            pool0 = draw_fleet(rng_b, clean_nodes, wdict, B, B / args.pool_size)[0]
            dist_build = dc_build_cost(*pack(pool0, B, cd["m_dist"]), land)
            conc_build = dc_build_cost(*pack(pool0, B, cd["m_conc"]), land)
            # CI-gated relief: is distribute's feasibility advantage over concentrate significant?
            feas_ci_sig = dist_anchor["feas_ci"][1] > conc_anchor["feas_ci"][2]   # dist lo > conc hi
            fout["dc_distribute"] = {
                "grid_inv_usd": 0.0,
                "dc_build_usd_distribute": dist_build, "dc_build_usd_concentrate": conc_build,
                "m_conc": cd["m_conc"], "m_dist": cd["m_dist"],
                "d_disp_distribute": dist_disp, "feas_distribute": dist_feas,
                "feas_ci_distribute": dist_anchor["feas_ci"], "feas_ci_concentrate": conc_anchor["feas_ci"],
                "d_disp_concentrate": conc_disp, "feas_concentrate": conc_feas,
                "relief_feas_vs_conc": dist_feas - conc_feas,
                "relief_feas_significant": bool(feas_ci_sig),
                "relief_disp_vs_conc": conc_disp[0] - dist_disp[0],
                # robust paired H1 stat (independent stream -- for cross-reference, not the frontier):
                "paired_gap_disp_conc_minus_dist": cd["gap"],
                "paired_conc_feas": cd["conc_feas"], "paired_dist_feas": cd["dist_feas"],
            }
            flex_zero = next((r for r in flex_curve if abs(r["flex_frac"]) <= 1e-12), None)
            if flex_zero is not None:
                fout["dc_distribute"]["flex_zero_feas_delta_vs_concentrate"] = (
                    flex_zero["firm_feas"] - conc_feas)
            print(f"      [DC-dist] grid-$0  m {cd['m_conc']}->{cd['m_dist']} | "
                  f"dist feas {dist_feas:.2f}[{dist_anchor['feas_ci'][1]:.2f},{dist_anchor['feas_ci'][2]:.2f}] "
                  f"vs conc feas {conc_feas:.2f}[{conc_anchor['feas_ci'][1]:.2f},{conc_anchor['feas_ci'][2]:.2f}] "
                  f"{'SIG' if feas_ci_sig else 'ns'} | relief Ddisp ${conc_disp[0]-dist_disp[0]:.1f} "
                  f"| DC build ${dist_build/1e6:.0f}M/yr")

            # ===== derived: grid $ for the CONCENTRATED fleet to MATCH distribution (free) =====
            # Feasibility is the primary axis (the deliverability wall); dispersion secondary.
            # "$ to clear the wall" = grid $ to raise CONCENTRATE feasibility up to DISTRIBUTE's.
            # Use the UNBIASED uniform-tx lever for the headline (targeted tx is for dispersion).
            txu_feas_match = interp_cost_for_target(
                [(r["inv_usd"], r["feas"]) for r in txu_curve], dist_feas, increasing=True)
            tx_feas_match = interp_cost_for_target(
                [(r["inv_usd"], r["feas"]) for r in tx_curve], dist_feas, increasing=True)
            gen_feas_match = interp_cost_for_target(
                [(r["inv_usd"], r["feas"]) for r in gen_curve], dist_feas, increasing=True)
            # dispersion match: targeted tx is the relevant (cheap) dispersion lever
            tx_disp_match = interp_cost_for_target(
                [(r["inv_usd"], r["d_disp"][0]) for r in tx_curve], dist_disp[0], increasing=False)
            # NEW levers: $ to match distribution AND $ to fully clear the wall (feas -> 1.0).
            colo_feas_match = interp_cost_for_target(
                [(r["inv_usd"], r["feas"]) for r in colo_curve], dist_feas, increasing=True)
            joint_feas_match = interp_cost_for_target(
                [(r["inv_usd"], r["feas"]) for r in joint_curve], dist_feas, increasing=True)
            # "fully clear the wall" = reach feasibility 1.0 (whole fleet deliverable).
            WALL = 0.999
            colo_usd_clear = interp_cost_for_target(
                [(r["inv_usd"], r["feas"]) for r in colo_curve], WALL, increasing=True)
            colo_gw_clear = interp_cost_for_target(
                [(r["g_co_gw"], r["feas"]) for r in colo_curve], WALL, increasing=True)
            txu_usd_clear = interp_cost_for_target(
                [(r["inv_usd"], r["feas"]) for r in txu_curve], WALL, increasing=True)
            colo_full = colo_curve[-1]
            fout["match_distribution"] = {
                "dist_feas": dist_feas, "conc_feas": conc_feas, "dist_disp": dist_disp[0],
                "txu_usd_to_match_feas": txu_feas_match,      # headline: $ to clear the wall
                "tx_usd_to_match_feas": tx_feas_match,        # targeted (usually n/a)
                "gen_usd_to_match_feas": gen_feas_match,
                "colocated_usd_to_match_feas": colo_feas_match,
                "joint_usd_to_match_feas": joint_feas_match,
                "tx_usd_to_match_disp": tx_disp_match,
                # full wall-clearing (feas -> 1.0 = whole fleet deliverable):
                "colocated_usd_to_full_feas": colo_usd_clear,
                "colocated_gw_to_full_feas": colo_gw_clear,
                "txu_usd_to_full_feas": txu_usd_clear,
                "colocated_max_feas": colo_full["feas"],
                "colocated_max_feas_usd": colo_full["inv_usd"],
                "colocated_max_g_co_gw": colo_full["g_co_gw"],
                "joint_max_feas": joint_curve[-1]["feas"],
                "joint_max_feas_usd": joint_curve[-1]["inv_usd"],
                "joint_beats_tx_alone": bool(joint_curve[-1]["feas"] > txu_curve[-1]["feas"] + 1e-9),
                "txu_max_feas": txu_curve[-1]["feas"], "txu_max_feas_usd": txu_curve[-1]["inv_usd"],
            }
            _fmt = lambda x: ("$%.0fM" % (x / 1e6)) if x else "n/a (never reaches)"
            print(f"      [MATCH] distribution feas {dist_feas:.2f} @ grid-$0 (conc {conc_feas:.2f}) | "
                  f"UNIFORM-tx $ to clear wall: {_fmt(txu_feas_match)} | "
                  f"targeted-tx: {_fmt(tx_feas_match)} | gen: {_fmt(gen_feas_match)}")
            print(f"      [WALL]  CO-LOCATED gen full-clear (feas->1.0): "
                  f"{('%.1f GW on-site = %s/yr' % (colo_gw_clear, _fmt(colo_usd_clear))) if colo_gw_clear else 'NOT cleared in sweep'} "
                  f"| max feas {colo_full['feas']:.2f} @ {colo_full['g_co_gw']:.0f}GW "
                  f"({_fmt(colo_full['inv_usd'])}/yr) | UNIFORM-tx full-clear {_fmt(txu_usd_clear)}")
            print(f"      [JOINT] gen+tx max feas {joint_curve[-1]['feas']:.2f} "
                  f"({_fmt(joint_curve[-1]['inv_usd'])}/yr)  vs  tx-alone max feas "
                  f"{txu_curve[-1]['feas']:.2f} ({_fmt(txu_curve[-1]['inv_usd'])}/yr) -> "
                  f"{'BEATS tx-alone' if joint_curve[-1]['feas'] > txu_curve[-1]['feas']+1e-9 else 'does NOT beat tx-alone'}")

            wout["fleets"][str(int(B))] = fout
            write_json(path, out, partial=True,
                       progress={"workload": w, "fleet_gw": float(B)})

    out["monotonicity_violations"] = monotonicity_violations
    if monotonicity_violations:
        print(f"\n# WARNING: {len(monotonicity_violations)} monotonicity violation(s) logged")

    write_json(path, out, partial=False, progress="complete")
    print(f"\nsaved -> {path}")
    _plot(args.outdir, out, stamp)


# --------------------------------------------------------------------------- #
def _plot(outdir, out, stamp):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plot skipped: {e})"); return
    workloads = list(out["workloads"].keys())
    fleets = sorted(out["workloads"][workloads[0]]["fleets"].keys(), key=int)
    fig, axes = plt.subplots(2, len(workloads), figsize=(7 * len(workloads), 9), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for wi, w in enumerate(workloads):
        ax_d, ax_f = axes[0][wi], axes[1][wi]
        for fi, B in enumerate(fleets):
            c = cmap(fi / max(1, len(fleets) - 1))
            f = out["workloads"][w]["fleets"][B]
            tx = f["transmission"]; gen = f["generation"]; dc = f["dc_distribute"]
            # tx/gen curves operate on the CONCENTRATED fleet; DC-dist (star) and the
            # concentrate reference (x) are the load-side endpoints at $0 grid.
            # dispersion frontier
            ax_d.plot([r["inv_usd"] / 1e6 for r in tx], [r["d_disp"][0] for r in tx],
                      "-o", color=c, label=f"{B}GW tx")
            ax_d.plot([r["inv_usd"] / 1e6 for r in gen], [r["d_disp"][0] for r in gen],
                      "--s", color=c, alpha=0.7, label=f"{B}GW gen")
            ax_d.scatter([0], [dc["d_disp_distribute"][0]], marker="*", s=200,
                         color=c, edgecolor="k", zorder=5, label=f"{B}GW DC-dist")
            ax_d.scatter([0], [dc["d_disp_concentrate"][0]], marker="X", s=90,
                         color=c, edgecolor="k", zorder=5)
            # feasibility companion
            ax_f.plot([r["inv_usd"] / 1e6 for r in tx], [r["feas"] for r in tx], "-o", color=c)
            ax_f.plot([r["inv_usd"] / 1e6 for r in gen], [r["feas"] for r in gen], "--s",
                      color=c, alpha=0.7)
            ax_f.scatter([0], [dc["feas_distribute"]], marker="*", s=200, color=c,
                         edgecolor="k", zorder=5)
            ax_f.scatter([0], [dc["feas_concentrate"]], marker="X", s=90, color=c,
                         edgecolor="k", zorder=5)
        ax_d.set_xlabel("grid investment ($M/yr, annualized)  [on concentrated fleet]")
        ax_d.set_ylabel("Δ LMP-dispersion (p90−p10), $/MWh  (lower = more relief)")
        ax_d.set_title(f"{w}: congestion-relief frontier  (★ DC-dist, ✕ concentrate @ $0)")
        ax_d.legend(fontsize=7); ax_d.grid(alpha=0.3)
        ax_f.set_xlabel("grid investment ($M/yr, annualized)")
        ax_f.set_ylabel("must-serve feasible fraction")
        ax_f.set_title(f"{w}: feasibility frontier"); ax_f.grid(alpha=0.3)
    fig.suptitle("Congestion-mitigation levers (fair sum_u² objective): DC-distribution @ $0 grid vs tx/gen",
                 fontsize=13)
    fig.tight_layout()
    p = os.path.join(outdir, f"levers_frontier_{stamp}.png")
    fig.savefig(p, dpi=130)
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
