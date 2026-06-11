"""
DC PLACEMENT STUDY (unstructured, panel-of-hours, no LMP clipping).

Built on the diagnosis that the 101-node clustered WECC is a near-copperplate
whereas the 490-node network is genuinely congested. Answers the two research
questions on a chosen network:
  (Q1) "I have X GW of DC to place over WECC -- what are the effects of placement
        on LMPs / congestion?"  -> per-node siting curves + node ranking
  (Q2) "Does distributing compute geographically reduce grid stress?"
        (100 x 1MW better than 1 x 100MW)  -> strategy table + budget sweep

This version removes three structural simplifications that pre-decided the result:
  * NO LMP CLIPPING. Prices are reported raw; the weakly-connected price pockets
    are part of the unstructured signal, not hidden. (median is outlier-robust;
    p95 / DC-terminal LMP now expose the real tails.)
  * NO CANDIDATE SET. Every one of the ~490 buses is a placement candidate; the
    node ranking is over the whole network, not a hand-picked shortlist.
  * NO SINGLE CHERRY-PICKED PEAK HOUR. Instead of selecting the single most-
    congested hour, the study runs over a PANEL of hours stratified across the
    year, dispatches each INDEPENDENTLY at T=1 (the network has storage, which
    couples time -- so non-consecutive hours must not be concatenated), and
    reports median + [p10, p90] across the panel plus a paired concentrate-vs-
    distribute win-rate.

DC load is still a flat, inelastic block (profiles == 1, linear_cost == 0):
flexibility is deliberately left unrepresentable here and factored in later.

Usage:
  .venv/bin/python development/dc_placement_study.py \
      --network ~/Downloads/elec_s490_c490.nc --n-snaps 12 --budget 3.0
"""
import argparse
import os
import json
from copy import deepcopy

import cvxpy as cp
import numpy as np

import zap
from zap.importers.pypsa import load_pypsa_network

POWER_UNIT, COST_UNIT, VOLL = 1.0e3, 100.0, 1000.0
DEFAULT_DC_PROFILE = "development/load_profiles/example_inference_azure_conv.csv"


# --------------------------------------------------------------------------- #
def load_dc_profile(path):
    """Load a real DC workload trace (per-unit 0-1) and reduce it to a 24-element
    HOURLY load factor (mean of the sub-hour samples in each hour). The DC is NOT
    flat: at each dispatched hour it draws nameplate * load_factor(hour-of-day).
    This is a fixed, inelastic time-varying shape -- flexibility (curtail/shift)
    is deliberately still NOT represented."""
    import pandas as pd
    df = pd.read_csv(os.path.expanduser(path))
    hr = df["timestamp_hr"].astype(float).values
    w = df["watts"].astype(float).values
    hod = np.floor(hr).astype(int) % 24
    lf = np.array([w[hod == h].mean() if np.any(hod == h) else np.nan for h in range(24)])
    if np.isnan(lf).any():                      # fill gaps by interpolation
        good = ~np.isnan(lf)
        lf = np.interp(np.arange(24), np.arange(24)[good], lf[good])
    return lf


def _hour_of_day(ts):
    import pandas as pd
    if isinstance(ts, tuple):
        ts = ts[-1]
    return int(pd.Timestamp(ts).hour)


def build_hour(pn, snaps, h, ls, gs, lns, hourly_lf):
    """Independent single-hour (T=1) context. Each panel hour is its own dispatch
    because the network has storage (96 batteries) that couples timesteps -- you
    cannot concatenate non-consecutive hours into one problem without giving the
    batteries free arbitrage across the discontinuities. The hour also carries the
    DC workload load-factor for its time-of-day."""
    net, devices = load_pypsa_network(pn, snaps[h:h + 1], power_unit=POWER_UNIT, cost_unit=COST_UNIT)
    devices = deepcopy(devices)
    devices[1].load *= ls
    devices[0].dynamic_capacity *= gs
    devices[3].nominal_capacity *= lns
    lf = float(hourly_lf[_hour_of_day(snaps[h])])
    return net, devices, str(snaps[h]), lf


def with_dc(devices, terminals, caps, lf):
    """Append an inelastic DC whose draw is nameplate(cap) * lf for this hour."""
    devs = deepcopy(devices)
    terminals = list(terminals)
    dc = zap.DataCenterLoad(
        num_nodes=devs[0].num_nodes, terminal=np.array(terminals),
        profiles=[np.array([lf], dtype=float)] * len(terminals),
        nominal_capacity=np.array(caps, dtype=float),
        linear_cost=np.zeros(len(terminals)), settime_horizon=1,
        capital_cost=np.zeros(len(terminals)),
    )
    devs.append(dc)
    return devs


def dispatch(net, devs, T):
    try:
        return net.dispatch(devs, time_horizon=T, solver=cp.CLARABEL, add_ground=False)
    except AssertionError:
        return None


def congestion_rent(oc, devices):
    """Transmission congestion rent = merchandising surplus, from the LMP (the
    power-balance dual) only -- robust, formulation-agnostic. Equals
    -Sum_n LMP_n * (net injection at n by all NON-transmission devices) which, by
    nodal balance, equals the rent collected on the lines. =0 iff LMPs are uniform
    (no congestion). Returned in $/h (POWER_UNIT converts GW*$/MWh -> $/h)."""
    p = np.asarray(oc.prices) * COST_UNIT                 # (N, T) $/MWh
    N = p.shape[0]
    inj = np.zeros((N, p.shape[1]))
    for i, d in enumerate(devices):
        pw = oc.power[i]
        if len(pw) != 1:                                   # transporter (2 terminals) -> skip
            continue
        term = np.asarray(d.terminal).ravel()
        np.add.at(inj, term, np.asarray(pw[0]))            # injection (>0) / withdrawal (<0)
    rent = -(p * inj).sum() * POWER_UNIT                   # $/h
    return float(rent)


def metrics(net, devs, devices, T, dc_terminals=None, price_nodes=None, oc=None):
    """Congestion-focused metrics on RAW prices (no clipping). `price_nodes` limits
    LMP percentile/dispersion stats to clean (non-pathological) buses."""
    oc = dispatch(net, devs, T) if oc is None else oc
    if oc is None:
        return {"feasible": False, "median_lmp": np.inf, "p95_lmp": np.inf,
                "dc_lmp": np.inf, "n_binding": -1, "max_util": np.inf,
                "sum_u2": np.inf, "shed_pct": np.inf, "cost": np.inf,
                "cong_rent": np.inf, "lmp_disp": np.inf, "cong_rent_lines": np.inf}
    p = np.asarray(oc.prices) * COST_UNIT          # RAW $/MWh, unclipped
    pn = p if price_nodes is None else p[np.asarray(price_nodes)]
    A = devices[3]
    flow = np.abs(np.asarray(oc.power[3][1]))
    limit = np.maximum(np.asarray(A.max_power) * np.asarray(A.nominal_capacity), 1e-9)
    u = flow / limit
    mu = (np.asarray(oc.local_inequality_duals[3][0]) +
          np.asarray(oc.local_inequality_duals[3][1]))
    n_binding = int(np.sum(mu.max(axis=1) > 1e-4))
    # line-dual congestion rent (cross-check only): sum mu * limit, $/h
    cong_rent_lines = float((mu.max(axis=1) * np.asarray(limit).ravel()).sum() * COST_UNIT * POWER_UNIT)
    L = devices[1]
    req = (np.asarray(L.load) * np.asarray(L.nominal_capacity)).sum()
    served = (-np.asarray(oc.power[1][0])).sum()
    shed_pct = float(100 * max(req - served, 0.0) / req)
    dc_lmp = np.nan
    if dc_terminals is not None and len(dc_terminals):
        dc_lmp = float(np.mean(p[np.array(dc_terminals)]))
    return {"feasible": True, "median_lmp": float(np.median(pn)),
            "p95_lmp": float(np.percentile(pn, 95)), "dc_lmp": dc_lmp,
            "lmp_disp": float(np.percentile(pn, 90) - np.percentile(pn, 10)),
            "cong_rent": congestion_rent(oc, devs),
            "cong_rent_lines": cong_rent_lines,
            "n_binding": n_binding, "max_util": float(u.max()),
            "sum_u2": float((u ** 2).sum()), "shed_pct": shed_pct,
            "cost": float(oc.problem.value)}


# --------------------------------------------------------------------------- #
# Network cleaning: a handful of structurally-unservable / weakly-connected buses
# create fake +-1e4 LMPs and persistent base shedding. Detect and neutralize them
# so base shed ~ 0 and prices are physical.
def find_bad_buses(panel, hi_cap=1.5 * VOLL, lo_cap=-200.0, shed_frac=0.9):
    """Flag ONLY clearly-pathological buses, leaving the legitimately-congested ones
    ($200-1000, the signal) in: extreme LMP (a +1e4 pocket > hi_cap=1.5*VOLL, OR a
    deep-negative pocket < lo_cap=-$200 -- genuine curtailment is -$10..-50, so -$600
    is a weakly-connected artifact), or a bus that sheds >=shed_frac of its OWN load at
    base (structurally unservable). Done on the no-DC base across all panel hours."""
    bad = set()
    for (net, devs, _, _) in panel:
        oc = dispatch(net, devs, 1)
        if oc is None:
            continue
        p = np.asarray(oc.prices)[:, 0] * COST_UNIT
        for n in np.where((p > hi_cap) | (p < lo_cap))[0]:
            bad.add(int(n))
        L = devs[1]
        load = (np.asarray(L.load) * np.asarray(L.nominal_capacity))[:, 0]
        served = (-np.asarray(oc.power[1][0]))[:, 0]
        term = np.asarray(L.terminal).ravel()
        shed = load - served
        for i in np.where((load > 1e-6) & (shed >= shed_frac * load))[0]:
            bad.add(int(term[i]))
    return sorted(bad)


def clean_devices(devices, bad_nodes):
    """Neutralize pathological buses by ZEROING their (unservable / pocket) loads so
    they stop driving fake shedding and LMP pockets. No grounding -- fixing many bus
    angles over-constrains the DC power flow and makes shedding worse."""
    devs = deepcopy(devices)
    if not bad_nodes:
        return devs
    bad = set(int(b) for b in bad_nodes)
    L = devs[1]
    term = np.asarray(L.terminal).ravel()
    mask = np.array([int(t) in bad for t in term])
    if mask.any():
        L.load = np.asarray(L.load).copy()
        L.load[mask, :] = 0.0
    return devs


# --------------------------------------------------------------------------- #
# Panel aggregation: every placement is evaluated across ALL panel hours and
# summarized by median + [p10, p90] over the feasible hours, plus a feasibility
# fraction. This replaces the single-snapshot point estimate.
def panel_eval(panel, terminals, caps):
    """Return the list of per-hour metric dicts for a placement. Each hour uses its
    own workload load-factor (the DC is time-varying, not flat)."""
    return [metrics(net, with_dc(devs, terminals, caps, lf), devs, 1, dc_terminals=terminals)
            for (net, devs, _, lf) in panel]


def summarize(ms, key):
    vals = [m[key] for m in ms if m["feasible"] and np.isfinite(m[key])]
    if not vals:
        return {"median": np.inf, "p10": np.inf, "p90": np.inf, "n": 0}
    return {"median": float(np.median(vals)), "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90)), "n": len(vals)}


def feas_frac(ms):
    return float(np.mean([m["feasible"] for m in ms])) if ms else 0.0


def summarize_all(ms):
    out = {"feas_frac": feas_frac(ms)}
    for k in ["median_lmp", "p95_lmp", "dc_lmp", "lmp_disp", "cong_rent", "cong_rent_lines",
              "n_binding", "max_util", "sum_u2", "shed_pct"]:
        out[k] = summarize(ms, k)
    return out


# --------------------------------------------------------------------------- #
# Realistic siting: data centers are NOT placed to help the grid. They land where
# land is cheap, drawn at random -- so we report the DISTRIBUTION of grid impact
# over many realistic layouts, not one hand-built (degenerate) optimum.
def load_land_weights(path, n_nodes):
    """Per-location siting weight ~ 1/land_cost (cheap land -> more data centers)."""
    import pandas as pd
    df = pd.read_csv(path).set_index("node")["land_usd_per_acre"]
    land = np.array([float(df.get(i, df.median())) for i in range(n_nodes)])
    w = 1.0 / np.maximum(land, np.nanmin(land[land > 0]))
    return land, w


def screen_usable(panel, nodes, probe):
    """Drop dead-end / disconnected locations: a node is usable only if a small
    real site is feasible there in EVERY panel hour."""
    usable = []
    for nd in nodes:
        ms = panel_eval(panel, [nd], [probe])
        if all(m["feasible"] for m in ms):
            usable.append(nd)
    return usable


def draw_fleet(rng, usable, weights, total, site):
    """One realistic fleet: ceil(total/site) sites drawn WITHOUT replacement from
    usable locations with probability ~ cheap-land weight; last site takes the
    remainder so the fleet sums to `total`."""
    n = int(np.ceil(total / site - 1e-9))
    n = min(n, len(usable))
    w = np.array([weights[u] for u in usable], dtype=float)
    w = w / w.sum()
    terms = list(rng.choice(usable, size=n, replace=False, p=w))
    caps = [site] * n
    caps[-1] = total - site * (n - 1)
    return terms, caps


def realistic_fleets(panel, rng, usable, weights, total, site, n_fleets):
    """Draw n_fleets random fleets; return per-fleet panel-median summaries so we
    can report typical vs worst-case realistic layouts."""
    rows = []
    for _ in range(n_fleets):
        terms, caps = draw_fleet(rng, usable, weights, total, site)
        s = summarize_all(panel_eval(panel, terms, caps))
        rows.append({"n_sites": len(terms), "feas_frac": s["feas_frac"],
                     "sum_u2": s["sum_u2"]["median"], "p95_lmp": s["p95_lmp"]["median"],
                     "dc_lmp": s["dc_lmp"]["median"], "shed_pct": s["shed_pct"]["median"],
                     "median_lmp": s["median_lmp"]["median"]})
    return rows


def across_fleets(rows, key):
    v = np.array([r[key] for r in rows if np.isfinite(r[key])], dtype=float)
    if v.size == 0:
        return {"median": np.inf, "p10": np.inf, "p90": np.inf, "max": np.inf, "n": 0}
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max()), "n": int(v.size)}


# --------------------------------------------------------------------------- #
def sample_panel_indices(n_total, n_snaps, win_start=None, win_len=None, seed=0):
    """Stratified panel of hours. Default: spread evenly across the whole year so
    the study sees a distribution of operating conditions rather than one peak.
    If a window is given, spread evenly within it."""
    if win_start is not None and win_len is not None:
        lo, hi = win_start, min(win_start + win_len, n_total) - 1
    else:
        lo, hi = 0, n_total - 1
    return np.unique(np.linspace(lo, hi, n_snaps).astype(int)).tolist()


def rank_nodes_panel(panel, nodes, probe_gw, budget):
    """Rank EVERY node by its median DC-terminal LMP response to a probe block
    across the panel, and record how often it can take the full budget alone.
    No hand-picked candidate set -- this is the whole network."""
    rows = []
    for nd in nodes:
        mp = panel_eval(panel, [nd], [probe_gw])
        mf = panel_eval(panel, [nd], [budget])
        rows.append({
            "node": int(nd),
            "probe_lmp_med": summarize(mp, "dc_lmp")["median"],
            "probe_sum_u2_med": summarize(mp, "sum_u2")["median"],
            "feas_frac_probe": feas_frac(mp),
            "feas_frac_budget": feas_frac(mf),
            "lmp_at_budget_med": summarize(mf, "dc_lmp")["median"],
            "shed_at_budget_med": summarize(mf, "shed_pct")["median"],
        })
    # best = most-feasible at probe, then lowest median probe LMP
    rows.sort(key=lambda r: (-r["feas_frac_probe"], r["probe_lmp_med"]))
    return rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="~/Downloads/elec_s490_c490.nc")
    ap.add_argument("--n-snaps", type=int, default=12,
                    help="number of hours in the panel (stratified across the year)")
    ap.add_argument("--win-start", type=int, default=None,
                    help="optional: restrict the panel to a window starting here")
    ap.add_argument("--win-len", type=int, default=None,
                    help="optional: window length for the panel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-scale", type=float, default=1.0)
    ap.add_argument("--gen-scale", type=float, default=1.0)
    ap.add_argument("--line-scale", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=5.0,
                    help="total data-center fleet to place over the grid (GW)")
    ap.add_argument("--budgets", default="3,4,5,6",
                    help="fleet sizes to sweep (GW)")
    ap.add_argument("--per-site-cap", type=float, default=1.0,
                    help="max GW at ONE grid location (largest realistic single campus)")
    ap.add_argument("--probe-gw", type=float, default=0.5,
                    help="size of one realistic large site, used to rank locations (GW)")
    ap.add_argument("--small-site", type=float, default=0.2,
                    help="site size in the 'many small sites' strategy (GW)")
    ap.add_argument("--n-fleets", type=int, default=24,
                    help="how many random realistic fleets to draw per (fleet size, site size)")
    ap.add_argument("--land-cost", default="development/results/placement_study/node_land_cost.csv",
                    help="per-location land $/acre (cheap land -> more likely to be sited)")
    ap.add_argument("--dc-profile", default=DEFAULT_DC_PROFILE,
                    help="real DC workload trace (per-unit) -- DC draw = nameplate * load_factor(hour)")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="optional cap on #nodes ranked (debug/quick runs); default = all")
    ap.add_argument("--outdir", default="development/results/placement_study")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    import pypsa
    pn = pypsa.Network(os.path.expanduser(args.network))
    snaps = pn.generators_t.p_max_pu.index
    n_total = len(snaps)

    print(f"# DC PLACEMENT STUDY (unstructured)  network={os.path.basename(args.network)}")
    print(f"# scaling load*{args.load_scale} gen*{args.gen_scale} line*{args.line_scale} "
          f"budget={args.budget}GW  | NO clip, ALL nodes, panel of hours")

    hourly_lf = load_dc_profile(args.dc_profile)
    print(f"# DC workload profile = {os.path.basename(args.dc_profile)}  "
          f"hourly load-factor: min={hourly_lf.min():.2f} mean={hourly_lf.mean():.2f} "
          f"max={hourly_lf.max():.2f} (inelastic, time-varying -- NOT flat)")

    idx = sample_panel_indices(n_total, args.n_snaps, args.win_start, args.win_len, args.seed)
    panel = [build_hour(pn, snaps, h, args.load_scale, args.gen_scale, args.line_scale, hourly_lf)
             for h in idx]
    dates = [d for (_, _, d, _) in panel]
    n_nodes = panel[0][0].num_nodes
    n_lines = panel[0][1][3].nominal_capacity.shape[0]
    nodes = list(range(n_nodes))
    if args.max_nodes:
        nodes = nodes[:args.max_nodes]
    print(f"# panel: {len(panel)} hours  idx={idx}")
    print(f"#   dates={dates}")
    print(f"# nodes={n_nodes} lines={n_lines}  | realistic siting (cheap land, random draws)")

    # ---- base (no DC) across the panel ----
    base_ms = [metrics(net, devs, devs, 1) for (net, devs, _, _) in panel]
    base = summarize_all(base_ms)
    print(f"# BASE (no DC, panel medians): median_lmp={base['median_lmp']['median']:.1f} "
          f"p95_lmp={base['p95_lmp']['median']:.1f} n_binding={base['n_binding']['median']:.0f} "
          f"max_util={base['max_util']['median']:.2f} sum_u2={base['sum_u2']['median']:.1f} "
          f"shed={base['shed_pct']['median']:.3f}%  feas={base['feas_frac']:.2f}")

    big, small = args.per_site_cap, args.small_site
    B = args.budget
    budgets = [float(x) for x in args.budgets.split(",")]

    # ---- realistic siting setup: cheap-land weights + drop dead-end locations ----
    land, weights = load_land_weights(args.land_cost, n_nodes)
    print(f"\n# land $/acre: cheapest={land.min():.0f}  median={np.median(land):.0f}  "
          f"priciest={land.max():.0f}  (data centers favor cheap land)")
    usable = screen_usable(panel, nodes, probe=small)
    print(f"# usable locations (a {small} GW site works every panel hour): {len(usable)}/{len(nodes)} "
          f"(dropped {len(nodes)-len(usable)} dead-end/overloaded corners)")
    rng = np.random.default_rng(args.seed)

    # ---- Part 1: AS THE FLEET GROWS (realistic small-site siting) ----
    print(f"\n## GRID STRESS AS REALISTIC DC DEMAND GROWS")
    print(f"#  many {small} GW sites on cheap land, {args.n_fleets} random fleets each; "
          f"median across fleets [worst-case fleet in brackets]")
    print(f"#  base (no DC): grid-stress {base['sum_u2']['median']:.0f}, "
          f"p95 price ${base['p95_lmp']['median']:.0f}, shed {base['shed_pct']['median']:.2f}%")
    print(f"  {'fleet':>6} {'#sites':>6} | {'stress(med)':>11} {'[worst]':>9} | {'p95$ (med)':>10} {'[worst]':>9}"
          f" | {'shed% med':>9} {'[worst]':>8}")
    grow = []
    for Bs in budgets:
        rows = realistic_fleets(panel, rng, usable, weights, Bs, small, args.n_fleets)
        st, pr, sh = across_fleets(rows, "sum_u2"), across_fleets(rows, "p95_lmp"), across_fleets(rows, "shed_pct")
        grow.append({"B": Bs, "site": small, "n_sites": rows[0]["n_sites"], "fleets": rows,
                     "stress": st, "p95": pr, "shed": sh})
        print(f"  {Bs:6.1f} {rows[0]['n_sites']:6d} | {st['median']:11.0f} {st['max']:9.0f} | "
              f"{pr['median']:10.0f} {pr['max']:9.0f} | {sh['median']:9.3f} {sh['max']:8.3f}")

    # ---- Part 2: FEW BIG vs MANY SMALL sites (realistic siting, same cheap-land draw) ----
    print(f"\n## FEW BIG ({big} GW) vs MANY SMALL ({small} GW) SITES  -- realistic cheap-land siting")
    print(f"#  {args.n_fleets} random fleets each; median across fleets [worst-case in brackets]")
    print(f"  {'fleet':>6} | {'BIG: #':>6} {'stress':>8} {'[worst]':>8} {'p95$':>6} {'shed%':>6}"
          f" | {'SMALL: #':>8} {'stress':>8} {'[worst]':>8} {'p95$':>6} {'shed%':>6} | {'small<big':>9}")
    bvs = []
    for Bs in budgets:
        rb = realistic_fleets(panel, rng, usable, weights, Bs, big, args.n_fleets)
        rs = realistic_fleets(panel, rng, usable, weights, Bs, small, args.n_fleets)
        sb, ss = across_fleets(rb, "sum_u2"), across_fleets(rs, "sum_u2")
        # how often is a random small-site fleet less stressful than a random big-site fleet?
        small_better = float(np.mean([s < b for b, s in zip(
            [r["sum_u2"] for r in rb], [r["sum_u2"] for r in rs])]))
        bvs.append({"B": Bs, "big": rb, "small": rs, "small_better": small_better})
        pb, ph = across_fleets(rb, "p95_lmp"), across_fleets(rb, "shed_pct")
        qb, qh = across_fleets(rs, "p95_lmp"), across_fleets(rs, "shed_pct")
        print(f"  {Bs:6.1f} | {rb[0]['n_sites']:6d} {sb['median']:8.0f} {sb['max']:8.0f} "
              f"{pb['median']:6.0f} {ph['median']:6.2f} | {rs[0]['n_sites']:8d} {ss['median']:8.0f} "
              f"{ss['max']:8.0f} {qb['median']:6.0f} {qh['median']:6.2f} | {small_better:9.2f}")

    # ---- headline ----
    print(f"\n## HEADLINE  (realistic cheap-land siting; medians over {args.n_fleets} fleets x {len(panel)} hours)")
    g3, g6 = grow[0], grow[-1]
    print(f"  base grid (no DC): stress {base['sum_u2']['median']:.0f}, "
          f"p95 price ${base['p95_lmp']['median']:.0f}, shed {base['shed_pct']['median']:.2f}%")
    print(f"  {g3['B']:.0f} GW of DC: stress {g3['stress']['median']:.0f} "
          f"(+{100*(g3['stress']['median']/base['sum_u2']['median']-1):.0f}%), "
          f"p95 ${g3['p95']['median']:.0f}, shed {g3['shed']['median']:.2f}% "
          f"(worst-case fleet sheds {g3['shed']['max']:.2f}%)")
    print(f"  {g6['B']:.0f} GW of DC: stress {g6['stress']['median']:.0f} "
          f"(+{100*(g6['stress']['median']/base['sum_u2']['median']-1):.0f}%), "
          f"p95 ${g6['p95']['median']:.0f}, shed {g6['shed']['median']:.2f}% "
          f"(worst-case fleet sheds {g6['shed']['max']:.2f}%)")
    sw = np.mean([x["small_better"] for x in bvs])
    print(f"  many small sites beat few big sites on grid stress in {100*sw:.0f}% of matched random draws")

    out = {"network": os.path.basename(args.network), "panel_idx": idx, "dates": dates,
           "scaling": {"load": args.load_scale, "gen": args.gen_scale, "line": args.line_scale},
           "budgets": budgets, "per_site_cap": big, "small_site": small, "n_fleets": args.n_fleets,
           "n_nodes": n_nodes, "n_usable": len(usable), "usable": usable, "base": base,
           "grow": grow, "big_vs_small": bvs}
    with open(os.path.join(args.outdir, "study_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nsaved -> {os.path.join(args.outdir, 'study_results.json')}")
    _plots(args.outdir, base, grow, bvs, big, small)


def _plots(outdir, base, grow, bvs, big, small):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(plots skipped: {e})")
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    Bs = [g["B"] for g in grow]
    # 1) grid stress as the fleet grows (median band + worst-case)
    ax = axes[0]
    med = [g["stress"]["median"] for g in grow]
    p10 = [g["stress"]["p10"] for g in grow]
    p90 = [g["stress"]["p90"] for g in grow]
    worst = [g["stress"]["max"] for g in grow]
    ax.axhline(base["sum_u2"]["median"], ls="--", c="gray", label="no DC (base)")
    ax.fill_between(Bs, p10, p90, alpha=0.2, color="steelblue")
    ax.plot(Bs, med, marker="o", color="steelblue", label="typical fleet")
    ax.plot(Bs, worst, marker="^", ls=":", color="firebrick", label="worst-case fleet")
    ax.set_xlabel("data-center demand added (GW)"); ax.set_ylabel("grid stress")
    ax.set_title("Grid stress as DC demand grows"); ax.legend(fontsize=8)
    # 2) involuntary load shed as the fleet grows
    ax = axes[1]
    ax.plot(Bs, [g["shed"]["median"] for g in grow], marker="o", color="darkorange", label="typical")
    ax.plot(Bs, [g["shed"]["max"] for g in grow], marker="^", ls=":", color="firebrick", label="worst-case")
    ax.set_xlabel("data-center demand added (GW)"); ax.set_ylabel("load cut (% of demand)")
    ax.set_title("How much load gets cut"); ax.legend(fontsize=8)
    # 3) few big vs many small sites
    ax = axes[2]
    bigm = [np.median([r["sum_u2"] for r in x["big"]]) for x in bvs]
    smallm = [np.median([r["sum_u2"] for r in x["small"]]) for x in bvs]
    ax.plot([x["B"] for x in bvs], bigm, marker="o", label=f"few big ({big} GW) sites")
    ax.plot([x["B"] for x in bvs], smallm, marker="s", label=f"many small ({small} GW) sites")
    ax.set_xlabel("data-center demand added (GW)"); ax.set_ylabel("grid stress (median)")
    ax.set_title("Few big vs many small sites"); ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "placement_study.png")
    fig.savefig(p, dpi=120)
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
