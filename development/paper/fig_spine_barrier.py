"""Single-claim figures for the data-center-siting spine/barrier story.

Spine metric: DELIVERABLE firm-DC GW = g(k) [q05], the forced-uniform deliverable
nameplate GW over the top-k sites of an ordering, deliverable in >=95% of panel hours.

  fig_spine   -- grid-strength siting delivers several-fold more than cheap-land
  fig_barrier -- the smallest footprint needed grows with the target (the barrier)
  fig_wall    -- beyond a few GW most naive sitings fail entirely (the wall)
  fig_slope   -- the barrier scaling is significant and robust (bootstrap of the slope)

Each figure carries exactly one claim, no title (captions live in the paper).

Data: development/results/spread_frontier/spread_frontier_full.json
Run:  .venv/bin/python development/paper/fig_spine_barrier.py
"""

import json
import os
import sys

import numpy as np
from matplotlib.ticker import ScalarFormatter

sys.path.insert(0, "development/paper")
import figstyle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "results", "spread_frontier", "spread_frontier_full.json")


def load():
    with open(DATA) as fp:
        return json.load(fp)


def curve_q05(workload, ordering, which, refined=False):
    """Return (ks, vals) for g or f at the q05 reducer over the (refined) k-grid.

    Falls back to the coarse grid if no refined curve exists for the ordering.
    """
    use_refined = refined and (ordering + "_refined") in workload
    d = workload[ordering + "_refined"] if use_refined else workload[ordering]
    ks = workload["kgrid_refined"] if use_refined else workload["kgrid"]
    vals = [d[str(k)][which]["q05"] for k in ks]
    return np.asarray(ks, float), np.asarray(vals, float)


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.ptp(x[m]) == 0:
        return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])


def bootstrap_slope_samples(per_fleet_kmin, D_grid, rng, nboot=2000):
    """Bootstrap DISTRIBUTION of the fleet-pooled OLS slope of k_min vs D.

    Mirrors bootstrap_slope_ci in development/dc_spread_frontier.py but returns the
    full array of resampled slopes (so the figure can show the distribution) plus the
    point estimate on the unresampled pool. per_fleet_kmin entries key D by str(D).
    """
    n = len(per_fleet_kmin)
    fleets = list(range(n))
    slopes = []
    for _ in range(nboot):
        samp = rng.choice(fleets, size=n, replace=True)
        xs, ys = [], []
        for fi in samp:
            km = per_fleet_kmin[fi]
            for D in D_grid:
                v = km.get(str(D))
                if v is not None:
                    xs.append(D)
                    ys.append(v)
        s = ols_slope(xs, ys)
        if np.isfinite(s):
            slopes.append(s)
    slopes = np.asarray(slopes, float)
    # point estimate on the full (unresampled) pool
    xs, ys = [], []
    for km in per_fleet_kmin:
        for D in D_grid:
            v = km.get(str(D))
            if v is not None:
                xs.append(D)
                ys.append(v)
    return slopes, ols_slope(xs, ys)


# --------------------------------------------------------------------------- #
def fig_spine(data):
    """Claim: grid-strength siting delivers several-fold more than cheap-land.

    Just the two g(k) deliverable-GW curves; no free ceiling, no annotations.
    """
    figstyle.setup()
    fig, ax = figstyle.new()

    wl = data["workloads"]["inference"]  # representative; training tells same story

    ks_gs, g_gs = curve_q05(wl, "grid_strength", "g")
    ax.plot(
        ks_gs,
        g_gs,
        "-o",
        color=figstyle.C["strong"],
        label="Grid-strength siting",
        zorder=3,
    )

    # cheap-land on the refined grid so the rise-then-collapse is fully resolved.
    ks_cl, g_cl = curve_q05(wl, "cheap_land", "g", refined=True)
    ax.plot(
        ks_cl,
        g_cl,
        "--s",
        color=figstyle.C["cheap"],
        label="Cheap-land siting",
        zorder=2,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Footprint (sites)")
    ax.set_ylabel("Deliverable DC (GW)")
    ax.set_xticks([1, 5, 10, 20, 40, 80, 160, 320])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(0.9, 520)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")

    return figstyle.save(fig, "fig_spine")


def _barrier_arrays(wl):
    """Median / p25 / p75 footprint and deliverable-fraction across MC fleets."""
    D_grid = wl["D_grid"]
    pfk = wl["per_fleet_kmin"]
    n_fleets = len(pfk)
    Ds, med, lo, hi, frac = [], [], [], [], []
    for D in D_grid:
        vals = [f[str(D)] for f in pfk if f[str(D)] is not None]
        Ds.append(D)
        frac.append(len(vals) / n_fleets)
        if vals:
            med.append(np.median(vals))
            lo.append(np.percentile(vals, 25))
            hi.append(np.percentile(vals, 75))
        else:
            med.append(np.nan)
            lo.append(np.nan)
            hi.append(np.nan)
    return (
        np.asarray(Ds),
        np.asarray(med),
        np.asarray(lo),
        np.asarray(hi),
        np.asarray(frac),
    )


def fig_barrier(data):
    """Claim: the smallest footprint needed grows with the target (the barrier).

    Median k_min(D) with a p25-p75 band over MC fleets. No slope text, no feasibility
    overlay (those are fig_slope and fig_wall).
    """
    figstyle.setup()
    fig, ax = figstyle.new()

    wl = data["workloads"]["inference"]
    Ds, med, lo, hi, _ = _barrier_arrays(wl)

    ok = ~np.isnan(med)
    ax.fill_between(
        Ds[ok],
        lo[ok],
        hi[ok],
        color=figstyle.C["strong"],
        alpha=0.18,
        lw=0,
        label="p25-p75 across fleets",
        zorder=1,
    )
    ax.plot(
        Ds[ok],
        med[ok],
        "-o",
        color=figstyle.C["ink"],
        label="Median footprint needed",
        zorder=3,
    )

    ax.set_xlabel("Target deliverable DC (GW)")
    ax.set_ylabel("Footprint needed (sites)")
    ax.set_ylim(0, np.nanmax(hi) * 1.12)
    ax.set_xlim(Ds.min() - 0.2, Ds.max() + 0.2)
    ax.legend(loc="upper left")

    return figstyle.save(fig, "fig_barrier")


def fig_wall(data):
    """Claim: beyond a few GW most naive sitings fail entirely (the wall).

    Fraction of MC fleets that can deliver D at all (k_min not null) vs D.
    """
    figstyle.setup()
    fig, ax = figstyle.new(h=2.9)  # extra height so the long y-label is never clipped

    wl = data["workloads"]["inference"]
    Ds, _, _, _, frac = _barrier_arrays(wl)

    ax.plot(Ds, frac, "-o", color=figstyle.C["cheap"], zorder=3)

    ax.set_xlabel("Target deliverable DC (GW)")
    ax.set_ylabel("Fraction of sitings that deliver")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(Ds.min() - 0.2, Ds.max() + 0.2)

    return figstyle.save(fig, "fig_wall")


def fig_slope(data, seed=0, nboot=3000):
    """Claim: the barrier scaling is significant and robust.

    Bootstrap DISTRIBUTION of the fleet-pooled OLS slope of k_min vs D, for both
    workloads side by side. Point estimate + 95% CI marked, a line at slope=0, and
    P(slope>0) annotated. The slope is sites-of-footprint per GW of fleet target.
    """
    figstyle.setup()
    fig, ax = figstyle.new(w=3.9, h=2.9)

    rng = np.random.default_rng(seed)
    order = ["inference", "training"]
    cols = {"inference": figstyle.C["strong"], "training": figstyle.C["cheap"]}

    ax.axhline(0, color=figstyle.C["ink"], lw=1.0, ls=":", zorder=1)

    positions = [1, 2]
    for pos, name in zip(positions, order):
        wl = data["workloads"][name]
        samples, point = bootstrap_slope_samples(
            wl["per_fleet_kmin"], wl["D_grid"], rng, nboot=nboot
        )
        lo, hi = np.percentile(samples, [2.5, 97.5])
        p_gt0 = float(np.mean(samples > 0))

        vp = ax.violinplot(
            [samples],
            positions=[pos],
            widths=0.7,
            showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_facecolor(cols[name])
            body.set_edgecolor(cols[name])
            body.set_alpha(0.35)

        # 95% CI as a vertical bar, point estimate as a marker.
        ax.vlines(pos, lo, hi, color=cols[name], lw=2.4, zorder=3)
        ax.plot(
            pos,
            point,
            "o",
            color=cols[name],
            markersize=7,
            markeredgecolor="k",
            markeredgewidth=0.6,
            zorder=4,
        )

        # annotate point + CI and P(slope>0)
        ax.text(
            pos,
            hi + 0.35,
            f"+{point:.1f}\n[{lo:.1f}, {hi:.1f}]",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=figstyle.C["ink"],
        )
        ax.text(
            pos,
            lo - 0.35,
            f"P(>0)={p_gt0:.2f}",
            ha="center",
            va="top",
            fontsize=8,
            color=cols[name],
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(["inference", "training"])
    ax.set_xlim(0.5, 2.5)
    ax.set_ylabel("Barrier slope (sites per GW)")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(-2.0, ymax + 1.0)

    return figstyle.save(fig, "fig_slope")


def main():
    data = load()
    paths = {
        "fig_spine": fig_spine(data),
        "fig_barrier": fig_barrier(data),
        "fig_wall": fig_wall(data),
        "fig_slope": fig_slope(data),
    }
    for k, v in paths.items():
        print(f"{k:12s} -> {v}")


if __name__ == "__main__":
    main()
