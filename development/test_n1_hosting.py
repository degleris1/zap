"""Correctness tests for the preventive N-1 hosting-capacity LP.

These exercise the LP's *semantics* against hand-derived expected values on a
small network whose N-1 limit is analytically obvious, plus the structural
properties the method must satisfy (lazy == upfront, aware <= blind,
monotonicity, lazy-generation soundness, shadow-price sanity).
"""

import os
import sys

import numpy as np

from zap.network import PowerNetwork
from zap.devices import Generator, Load, ACLine
from zap.contingency.lodf import post_contingency_flows

sys.path.insert(0, os.path.dirname(__file__))
from n1_hosting import HostingCapacityProblem  # noqa: E402  (study module lives in development/)


def two_path(linecap=30.0, load=5.0):
    """Big generator at bus 0 feeds a DC at bus 3 over two parallel paths
    (0-1-3 and 0-2-3, plus a weak 1-2 tie); every line has capacity ``linecap``.

    Base case splits the DC flow across both paths, but a single outage forces
    it all onto the surviving path, so the N-1 hosting capacity at bus 3 is
    ``linecap - load`` (the bus-1 load shares the surviving corridor).  This
    analytic value is the test oracle.
    """
    net = PowerNetwork(4)
    gen = Generator(num_nodes=4, terminal=np.array([0]), dynamic_capacity=np.array([[1000.0]]),
                    linear_cost=np.array([1.0]), nominal_capacity=np.array([1.0]))
    ld = Load(num_nodes=4, terminal=np.array([1]), load=np.array([[load]]), linear_cost=np.array([1e4]))
    lines = ACLine(num_nodes=4, source_terminal=np.array([0, 1, 0, 2, 1]),
                   sink_terminal=np.array([1, 3, 2, 3, 2]), capacity=np.ones(5),
                   nominal_capacity=np.full(5, linecap), susceptance=np.array([1.0, 1, 1, 1, 0.5]))
    return net, [gen, ld, lines]


def _solve(net, devices, candidates, site_cap=500.0, enforce_n1=True, lazy=True):
    hp = HostingCapacityProblem(net, [devices], candidate_buses=candidates,
                                include_batteries=False, include_dc_lines=False)
    hp.build(site_cap=site_cap)
    res = hp.solve(enforce_n1=enforce_n1, lazy=lazy)
    return hp, res


def test_hosting_value_matches_analytic():
    net, devices = two_path(linecap=30.0, load=5.0)
    _, res = _solve(net, devices, [3])
    assert res.status == "optimal"
    assert abs(res.H - 25.0) < 1e-4  # linecap - load


def test_lazy_equals_upfront():
    net, devices = two_path()
    _, lazy = _solve(net, devices, [3], lazy=True)
    _, upfront = _solve(net, devices, [3], lazy=False)
    assert abs(lazy.H - upfront.H) < 1e-6
    assert lazy.n_rounds > 1  # lazy generation actually discovered constraints


def test_n1_aware_below_blind():
    net, devices = two_path()
    _, aware = _solve(net, devices, [3], enforce_n1=True)
    _, blind = _solve(net, devices, [3], enforce_n1=False)
    # ignoring N-1 over-promises hosting capacity
    assert blind.H > aware.H + 1.0
    assert abs(aware.H - 25.0) < 1e-4


def test_solution_is_n1_secure():
    """Lazy-generation soundness: at the returned solution, a *dense* re-check of
    every non-radial outage shows no thermal overload (5.7)."""
    net, devices = two_path()
    hp, res = _solve(net, devices, [3])
    f = np.asarray(hp.flow_expr[0].value).ravel()
    fpc = post_contingency_flows(f, hp.lodf, hp.outaged)
    max_overload = float((np.abs(fpc) - hp.F_bar[:, None]).max())
    assert max_overload <= 1e-5
    # base case is also within limits
    assert float((np.abs(f) - hp.F_bar).max()) <= 1e-5


def test_binding_set_and_shadow_prices():
    net, devices = two_path()
    hp, res = _solve(net, devices, [3])
    assert len(res.active_contingencies) > 0  # some outage binds
    # at least one line carries a positive marginal value of reinforcement
    assert any(v > 1e-6 for v in res.duals.values())


def test_monotone_in_line_limit():
    # tighter thermal limits cannot increase hosting capacity (5.3)
    H = {}
    for cap in (20.0, 30.0, 40.0):
        net, devices = two_path(linecap=cap)
        _, res = _solve(net, devices, [3])
        H[cap] = res.H
    assert H[20.0] <= H[30.0] + 1e-6 <= H[40.0] + 1e-6
    assert abs(H[20.0] - 15.0) < 1e-4  # cap - load


def test_monotone_in_candidate_set():
    # adding candidate buses cannot decrease hosting capacity (5.4)
    net, devices = two_path()
    _, small = _solve(net, devices, [3])
    _, big = _solve(net, devices, [3, 2])
    assert big.H >= small.H - 1e-6
