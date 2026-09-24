"""Perfect capacity as a hub generator (perfect-capacity hub spec, 2026-09-23).

Perfect capacity is the standard adequacy construct "an always-available,
outage-free resource of P MW": the unit a reliability calibration searches in.
Here it is **one generator of nameplate P at a new hub bus**, joined to every
load bus by a one-way, lossless, zero-cost link rated ``PERFECT_LINK_HEADROOM *
P``.  The dispatch sends P wherever the shortfall is, so the allocation across
buses is endogenous and the realised hub-link flows are the "by bus" record.

Why the links are rated ``2 P`` and not ``P``: every link flow is bounded by
the generator's output, ``f_n <= g_H <= P < 2 P``, so no link cap ever binds
and the hub's nodal price is pinned to the **maximum load-bus price** in every
hour in which that maximum is positive (spec section 3.1).  With a rating of
exactly ``P`` the link that carries all of P sits at its cap and the hub price
is only bracketed, not unique.

Both device classes are subclasses **with no overrides**: every model, KKT,
ADMM, scaling and sampling method is the parent's.  They exist so that code
that groups devices by ``type(device).__name__`` -- accreditation rows, the
network credit, generation by carrier, ``available_capacity`` -- excludes the
construct by default.  This module is library code and never imports ``ch3``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from attrs import define

from zap.devices.injector import Generator
from zap.devices.transporter import DirectedLine

#: The hub's bus name.  A dataset bus of this name is refused by the loader.
PERFECT_HUB_BUS = "perfect_hub"
#: The construct's carrier label.  Never in ``carriers.csv``; used for labels only.
PERFECT_CARRIER = "perfect_capacity"
#: Link rating as a multiple of P.  Anything > 1 keeps every link strictly
#: below its cap (section 3.1); 2 is the spec's choice.
PERFECT_LINK_HEADROOM = 2.0


@define(kw_only=True, slots=False)  # attrs, like Generator
class PerfectGenerator(Generator):
    """Perfect capacity: one always-available, zero-cost, outage-free row at the hub.

    A subclass with **no overrides**, so every model, KKT and ADMM method is
    Generator's.  It exists so that code grouping devices by
    ``type(device).__name__`` excludes it by default (accreditation rows, the
    network credit, c_max, generation by carrier, ``available_capacity``).
    """


@dataclass(kw_only=True)  # dataclass, like DirectedLine
class PerfectLink(DirectedLine):
    """One-way, lossless, zero-cost hub -> load-bus link rated headroom * P.  No overrides."""


def perfect_load_nodes(load_profile, load_terminal, num_nodes: int) -> np.ndarray:
    """Sorted node indices with ``bus_peak_shares(...) > 0``.

    The loader's own definition of a load bus -- the same ``f_n`` support the
    firm-load rule splits over -- so the hub links reach exactly the buses a
    firm-load increment would.  ``load_profile`` is ``(rows, T)`` MW,
    ``load_terminal`` the rows' nodes.
    """
    # Imported here, not at module level: `wy_store` imports this module, and a
    # top-level import the other way would be circular.
    from zap.importers.wy_store import bus_peak_shares

    shares = bus_peak_shares(load_profile, load_terminal, int(num_nodes))
    return np.flatnonzero(shares > 0.0).astype(int)


def perfect_capacity_devices(
    *,
    num_nodes: int,
    hub_node: int,
    load_nodes: Sequence[int],
    capacity_mw: float,
    load_bus_names: Sequence[str],
    link_headroom: float = PERFECT_LINK_HEADROOM,
) -> tuple[PerfectGenerator, PerfectLink]:
    """The hub generator and its links, in MW (the loader scales them afterwards).

    Every array is width 1, so both devices are **static**: ``time_horizon`` is
    0 and ``sample_time`` slices nothing.  ``capacity_mw <= 0`` is refused:
    P = 0 means "no construct", and the loader never calls this then.
    """
    capacity = float(capacity_mw)
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError(
            f"perfect capacity must be finite and > 0 MW, got {capacity_mw!r} "
            "(P = 0 means no hub: do not build the devices)"
        )
    headroom = float(link_headroom)
    if not math.isfinite(headroom) or headroom <= 1.0:
        raise ValueError(
            f"link_headroom must be finite and > 1 so no hub link can bind, got {link_headroom!r}"
        )
    num_nodes = int(num_nodes)
    hub_node = int(hub_node)
    if not 0 <= hub_node < num_nodes:
        raise ValueError(f"hub_node {hub_node} is not a node of a {num_nodes}-node network")
    nodes = np.asarray(load_nodes, dtype=int).reshape(-1)
    names = [str(b) for b in load_bus_names]
    if nodes.size == 0:
        raise ValueError("perfect capacity needs at least one load node to deliver to")
    if len(names) != nodes.size:
        raise ValueError(
            f"{nodes.size} load node(s) but {len(names)} load bus name(s); they pair one to one"
        )
    if np.any(nodes == hub_node) or np.any(nodes < 0) or np.any(nodes >= num_nodes):
        raise ValueError(
            f"load nodes {nodes.tolist()} must be nodes of the network other than the hub "
            f"({hub_node})"
        )
    if np.unique(nodes).size != nodes.size:
        raise ValueError(f"load nodes {nodes.tolist()} contain duplicates")

    generator = PerfectGenerator(
        num_nodes=num_nodes,
        name=np.array([PERFECT_HUB_BUS], dtype=object),
        terminal=np.array([hub_node]),
        nominal_capacity=np.array([capacity]),
        dynamic_capacity=np.ones((1, 1)),
        linear_cost=np.zeros((1, 1)),
        capital_cost=np.array([0.0]),
        emission_rates=np.array([0.0]),
        min_nominal_capacity=np.array([capacity]),
        max_nominal_capacity=np.array([capacity]),
    )
    # `_build_generators` sets the carrier the same way (not an attrs field).
    generator.fuel_type = np.array([PERFECT_CARRIER], dtype=object)

    n = nodes.size
    link_mw = headroom * capacity
    link = PerfectLink(
        num_nodes=num_nodes,
        name=np.array([f"{PERFECT_HUB_BUS}->{bus}" for bus in names], dtype=object),
        source_terminal=np.full(n, hub_node, dtype=int),
        sink_terminal=nodes.copy(),
        min_power=np.zeros((n, 1)),
        max_power=np.ones((n, 1)),
        linear_cost=np.zeros((n, 1)),
        efficiency=np.ones(n),
        nominal_capacity=np.full(n, link_mw),
        capital_cost=np.zeros(n),
        min_nominal_capacity=np.full(n, link_mw),
        max_nominal_capacity=np.full(n, link_mw),
        # Never in an aggregate (import) group: the group lives on the
        # dataset's own `DirectedLine` device, and this is a separate device.
        group=None,
        group_limit=None,
    )
    return generator, link
