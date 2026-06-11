"""Spatial storytelling figures for the data-center-siting paper, on the REAL bus
coordinates of the 490-node PyPSA WECC network. Every number comes from the .nc
geometry, the spread-frontier JSON, the land-cost CSV, or a recomputed base dispatch.

Four single-claim, untitled, paper-quality panels (captions live in the paper):
  fig_map_headroom -- deliverability is a spatial property of the grid.
  fig_map_policy   -- grid-strength and cheap-land siting pick different buses.
  fig_alloc_bar    -- the policies allocate capacity very differently (NOT a map).
  fig_map_corridor -- the N-1/base limits are a few specific corridors, not diffuse.

NODE -> COORDINATE MAPPING (critical, confirmed):
  zap.importers.pypsa.parse_buses builds node indices 0..N-1 from
  net.buses.loc[net.buses.carrier != "battery"].index, in that order. In
  elec_s490_c490.nc all 490 buses have carrier "AC" (no battery buses), so the zap
  node index is exactly the position in pn.buses.index. We verified: (a) the spatial
  extent of those buses is WECC (lon -124.9..-102.5, lat 31.5..48.9); (b) the ACLine
  device's source/sink terminals reproduce pn.lines bus0/bus1 mapped through that same
  order, element-for-element (so line-device order == pn.lines order). Both pass.

Usage:
  .venv/bin/python development/paper/fig_maps.py            # all four
  .venv/bin/python development/paper/fig_maps.py headroom   # one figure
"""
import argparse
import json
import logging
import os
import sys
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("pypsa").setLevel(logging.ERROR)

PC = ccrs.PlateCarree()  # the bus coordinates are lon/lat (plate carree)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle  # noqa: E402

NETWORK = "~/Downloads/elec_s490_c490.nc"
SPREAD = os.path.join(HERE, "..", "results", "spread_frontier", "spread_frontier_full.json")
LANDCSV = os.path.join(HERE, "..", "results", "placement_study", "node_land_cost.csv")
WORKLOAD = "inference"  # diurnal trace; the representative workload used elsewhere
K = 20  # fixed footprint for the policy comparison


# --------------------------------------------------------------------------- #
# geometry + data loaders
# --------------------------------------------------------------------------- #
def load_geometry():
    """Return (lon, lat, lines_src, lines_snk) keyed by zap node index 0..N-1.

    lon/lat are length-N arrays indexed by node. lines_* are node-index endpoints
    of every (finite-reactance) AC line, in pn.lines order. The node order is
    pn.buses.loc[carrier != 'battery'].index, matching zap.importers.pypsa.parse_buses.
    """
    import pypsa

    pn = pypsa.Network(os.path.expanduser(NETWORK))
    buses = pn.buses.loc[pn.buses.carrier != "battery"].index
    b2i = {b: i for i, b in enumerate(buses)}
    lon = pn.buses.loc[buses, "x"].values.astype(float)
    lat = pn.buses.loc[buses, "y"].values.astype(float)
    lines = pn.lines[~np.isinf(pn.lines.x)]
    src = lines.bus0.map(b2i).values.astype(int)
    snk = lines.bus1.map(b2i).values.astype(int)
    # sanity: WECC extent
    assert -126 < lon.min() and lon.max() < -101, "lon extent not WECC"
    assert 30 < lat.min() and lat.max() < 50, "lat extent not WECC"
    return lon, lat, src, snk


def load_signals():
    """headroom h(n) (GW, clean nodes only) and land cost (USD/acre, all nodes)."""
    d = json.load(open(SPREAD))
    hr = {int(k): float(v) for k, v in d["workloads"][WORKLOAD]["headroom"].items()}
    land = pd.read_csv(LANDCSV).set_index("node")["land_usd_per_acre"]
    return hr, land


def policy_nodes(hr, land, k=K):
    """Top-k grid-strength (descending headroom) and top-k cheap-land (ascending land
    cost), both restricted to the clean nodes that have a headroom value."""
    clean = list(hr.keys())
    strong = sorted(clean, key=lambda n: -hr[n])[:k]
    landmed = float(land.median())
    cheap = sorted(clean, key=lambda n: float(land.get(n, landmed)))[:k]
    return strong, cheap


def base_line_utilization():
    """Recompute the no-DC base dispatch on the cleaned 490-node inference panel and
    return per-line utilization u = |f| / Fbar. Returns (u_max, u_mean) over the
    panel hours, length = n AC lines (== pn.lines order). Uses the same machinery the
    deliverability study uses (dc_spread_frontier.build_panels + dispatch)."""
    sys.path.insert(0, os.path.join(HERE, ".."))
    from dc_spread_frontier import build_panels
    from dc_placement_study import dispatch

    args = argparse.Namespace(
        network=NETWORK,
        n_snaps=12,
        load_scale=1.2,
        land_cost=os.path.relpath(LANDCSV),
    )
    _, _, _, _, _, panels = build_panels(args)
    panel = panels[WORKLOAD]
    A = panel[0][1][3]  # ACLine device (index 3 in the device list)
    limit = np.maximum(
        np.asarray(A.max_power).ravel() * np.asarray(A.nominal_capacity).ravel(), 1e-9
    )
    rows = []
    for net, devs, _, _ in panel:
        oc = dispatch(net, devs, 1)
        if oc is None:
            continue
        f = np.abs(np.asarray(oc.power[3][1]).ravel())
        rows.append(f / limit)
    U = np.array(rows)
    return U.max(axis=0), U.mean(axis=0)


# --------------------------------------------------------------------------- #
# shared map basemap
# --------------------------------------------------------------------------- #
def _new_map(w=4.6, h=5.4):
    """A WECC GeoAxes (PlateCarree) with a real US basemap UNDER the network:
    light land/ocean fill, faint state boundaries + national borders + coastline.
    The power network and markers are drawn on top with transform=PC."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(w, h), subplot_kw={"projection": PC})
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f6f5f1", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e9f0f5", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#e9f0f5",
                   edgecolor="none", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#cccccc",
                   linewidth=0.4, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#a9a9a9",
                   linewidth=0.6, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#a9a9a9",
                   linewidth=0.6, zorder=1)
    ax.axis("off")
    return fig, ax


def _draw_skeleton(ax, lon, lat, src, snk, color="#8f9499", lw=0.4, alpha=0.85):
    """Thin AC-line segments: the grid skeleton, drawn over the US basemap."""
    from matplotlib.collections import LineCollection

    segs = [[(lon[s], lat[s]), (lon[t], lat[t])] for s, t in zip(src, snk)]
    lc = LineCollection(segs, colors=color, linewidths=lw, alpha=alpha,
                        transform=PC, zorder=2)
    ax.add_collection(lc)
    pad_x = 0.03 * (lon.max() - lon.min())
    pad_y = 0.03 * (lat.max() - lat.min())
    ax.set_extent([lon.min() - pad_x, lon.max() + pad_x,
                   lat.min() - pad_y, lat.max() + pad_y], crs=PC)


# --------------------------------------------------------------------------- #
# fig 1: headroom map -- deliverability is geographic
# --------------------------------------------------------------------------- #
def fig_map_headroom():
    lon, lat, src, snk = load_geometry()
    hr, _ = load_signals()
    nodes = np.array(sorted(hr.keys()))
    h = np.array([hr[n] for n in nodes])

    # headroom is right-skewed (median 1.7, max 11.4 GW); cap the color at the 95th
    # pct so the weak/strong contrast is visible across the bulk, with extend='max'
    # flagging the few very-strong outliers.
    vmax = float(np.percentile(h, 95))

    figstyle.setup()
    fig, ax = _new_map()
    _draw_skeleton(ax, lon, lat, src, snk)
    sc = ax.scatter(
        lon[nodes], lat[nodes], c=h, cmap="viridis", s=24,
        edgecolors="white", linewidths=0.25, zorder=3, vmin=0, vmax=vmax,
        transform=PC,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.030, pad=0.02, extend="max")
    cb.set_label("Standalone deliverable headroom (GW)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    path = figstyle.save(fig, "fig_map_headroom")
    print("saved", path)
    print(f"  {len(nodes)} clean buses, headroom {h.min():.2f}..{h.max():.2f} GW "
          f"(mean {h.mean():.2f})")
    return path


# --------------------------------------------------------------------------- #
# fig 2: policy map -- grid-strength vs cheap-land pick different buses
# --------------------------------------------------------------------------- #
def fig_map_policy():
    lon, lat, src, snk = load_geometry()
    hr, land = load_signals()
    strong, cheap = policy_nodes(hr, land, K)

    # marker size ~ headroom (deliverable potential) of the chosen bus, common scale
    def sizes(ns):
        return 45 + 55 * np.array([hr[n] / max(hr.values()) for n in ns])

    figstyle.setup()
    fig, ax = _new_map()
    _draw_skeleton(ax, lon, lat, src, snk)

    sg, sc = np.array(strong), np.array(cheap)
    ax.scatter(lon[sg], lat[sg], s=sizes(strong), marker="o",
               facecolors="none", edgecolors=figstyle.C["strong"], linewidths=2.0,
               zorder=4, label="grid-strength (top $h$)", transform=PC)
    ax.scatter(lon[sc], lat[sc], s=sizes(cheap), marker="^",
               color=figstyle.C["cheap"], alpha=0.9, zorder=3,
               edgecolors="white", linewidths=0.4, label="cheap-land", transform=PC)

    overlap = sorted(set(strong) & set(cheap))
    if overlap:
        ov = np.array(overlap)
        ax.scatter(lon[ov], lat[ov], s=110, marker="o", facecolors="none",
                   edgecolors=figstyle.C["ink"], linewidths=1.6, zorder=5,
                   label="chosen by both", transform=PC)

    leg = ax.legend(loc="lower left", fontsize=8, handletextpad=0.4, borderpad=0.4,
                    labelspacing=0.3, framealpha=0.85, facecolor="white", edgecolor="none")
    leg.set_zorder(10)

    path = figstyle.save(fig, "fig_map_policy")
    print("saved", path)
    print(f"  k={K}: overlap={len(overlap)} buses; "
          f"strong-h {hr[strong[-1]]:.1f}..{hr[strong[0]]:.1f} GW, "
          f"cheap-land-h mean {np.mean([hr[n] for n in cheap]):.1f} GW")
    return path


# --------------------------------------------------------------------------- #
# fig 3: allocation bar -- the policies allocate very differently (NOT a map)
# --------------------------------------------------------------------------- #
def fig_alloc_bar():
    """Cumulative standalone deliverable headroom each policy secures as it adds sites
    in priority order (1..K). Each added site contributes its own standalone headroom
    h(n) = f({n}); the cumulative sum is the deliverable GW the policy concentrates on
    its first j sites. Grid-strength piles capacity onto a few strong buses (rises
    steeply, high ceiling); cheap-land spreads thin onto weak buses (rises slowly, low
    ceiling). Standalone h is an upper bound on what a fleet can co-host, but the gap
    between the two curves is the spatial allocation story."""
    hr, land = load_signals()
    strong, cheap = policy_nodes(hr, land, K)

    # deliverable contribution of each chosen bus = its standalone headroom h(n).
    # cumulative over the priority order is the spine: total deliverable GW vs #sites.
    def cum(ns):
        return np.cumsum([hr[n] for n in ns])

    cs, cc = cum(strong), cum(cheap)
    x = np.arange(1, K + 1)

    figstyle.setup()
    fig, ax = figstyle.new(w=3.9, h=2.9)
    w = 0.42
    ax.bar(x - w / 2, cs, width=w, color=figstyle.C["strong"], label="grid-strength")
    ax.bar(x + w / 2, cc, width=w, color=figstyle.C["cheap"], label="cheap-land")
    ax.set_xlabel("number of sites added (priority order)")
    ax.set_ylabel("cumulative standalone\ndeliverable headroom (GW)")
    ax.set_xlim(0.3, K + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(loc="upper left")

    path = figstyle.save(fig, "fig_alloc_bar")
    print("saved", path)
    print(f"  k={K}: grid-strength secures {cs[-1]:.1f} GW deliverable headroom; "
          f"cheap-land {cc[-1]:.1f} GW ({cs[-1] / cc[-1]:.1f}x)")
    return path


# --------------------------------------------------------------------------- #
# fig 4: corridor map -- the N-1/base limits are a few specific corridors
# --------------------------------------------------------------------------- #
def fig_map_corridor(thresh=0.9):
    """Persistently base-congested corridors: lines whose MEAN base utilization across
    the panel exceeds `thresh` (binding most hours, not a one-off worst hour). This
    isolates the handful of structural corridors that cap deliverability."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    lon, lat, src, snk = load_geometry()
    _, u_mean = base_line_utilization()

    hot = np.where(u_mean > thresh)[0]
    figstyle.setup()
    fig, ax = _new_map()
    _draw_skeleton(ax, lon, lat, src, snk)

    # highlight the congested corridors in a bold color, lw ~ utilization. Many WECC
    # corridors are short electrically-adjacent segments, so mark their endpoint buses
    # too -- the highlighted CLUSTERS are what carry the "few specific corridors" claim.
    segs = [[(lon[src[i]], lat[src[i]]), (lon[snk[i]], lat[snk[i]])] for i in hot]
    lws = 2.0 + 2.6 * (u_mean[hot] - thresh) / max(1.0 - thresh, 1e-6)
    lc = LineCollection(segs, colors=figstyle.C["accent"], linewidths=lws,
                        transform=PC, zorder=4)
    ax.add_collection(lc)
    ends = np.unique(np.concatenate([src[hot], snk[hot]]))
    ax.scatter(lon[ends], lat[ends], s=26, color=figstyle.C["accent"],
               edgecolors="white", linewidths=0.3, zorder=5, transform=PC)

    handles = [
        Line2D([0], [0], color="#cccccc", lw=1.2, label="AC line"),
        Line2D([0], [0], color=figstyle.C["accent"], lw=2.4,
               label=f"persistently congested ($\\bar u>{thresh:g}$)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              handletextpad=0.5, borderpad=0.3)

    path = figstyle.save(fig, "fig_map_corridor")
    print("saved", path)
    print(f"  {len(hot)} of {len(src)} AC lines at mean base u>{thresh} "
          f"(max mean u={u_mean.max():.2f})")
    return path


# --------------------------------------------------------------------------- #
FIGS = {
    "headroom": fig_map_headroom,
    "policy": fig_map_policy,
    "alloc": fig_alloc_bar,
    "corridor": fig_map_corridor,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=list(FIGS),
                    help="subset of: " + ", ".join(FIGS))
    args = ap.parse_args()
    for name in args.which:
        if name not in FIGS:
            raise SystemExit(f"unknown figure {name!r}; choose from {list(FIGS)}")
        FIGS[name]()


if __name__ == "__main__":
    main()
