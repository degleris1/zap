"""Tractability check: build the gradient-based planner on the 490-node network
and time ONE forward + backward through DispatchLayer (cvxpy implicit diff)."""
import os, time
from copy import deepcopy
import cvxpy as cp
import numpy as np
import zap
from zap.importers.pypsa import load_pypsa_network

POWER_UNIT, COST_UNIT = 1.0e3, 100.0
SITES = [20, 100, 200, 250, 300, 350, 400, 480, 150, 50, 60, 70, 120, 180, 220, 330]

import pypsa
pn = pypsa.Network(os.path.expanduser("~/Downloads/elec_s490_c490.nc"))
snaps = pn.generators_t.p_max_pu.index
for snlen in [1, 2]:
    t0 = time.time()
    net, devices = load_pypsa_network(pn, snaps[5616:5616+snlen], power_unit=POWER_UNIT, cost_unit=COST_UNIT)
    devices = deepcopy(devices)
    devices[0].dynamic_capacity *= 1.24
    sites = [s for s in SITES if s < net.num_nodes]
    n = len(sites)
    dc = zap.DataCenterLoad(
        num_nodes=net.num_nodes, terminal=np.array(sites),
        profiles=[np.ones(snlen)] * n, nominal_capacity=np.full(n, 2.0 / n),
        linear_cost=np.zeros(n), settime_horizon=snlen, capital_cost=np.zeros(n),
    )
    devices.append(dc)
    dc_idx = len(devices) - 1
    print(f"\n[snap_len={snlen}] load={time.time()-t0:.1f}s  sites={n} dc_idx={dc_idx}")

    layer = zap.DispatchLayer(net, devices,
                              parameter_names={"dc_capacity": (dc_idx, "nominal_capacity")},
                              time_horizon=snlen, solver=cp.CLARABEL)
    op_obj = zap.planning.LineUtilizationObjective(net, devices, metric="quadratic")
    inv_obj = zap.planning.InvestmentObjective(devices, layer)
    P = zap.planning.PlanningProblem(operation_objective=op_obj, investment_objective=inv_obj,
                                     layer=layer,
                                     lower_bounds={"dc_capacity": np.zeros(n)},
                                     upper_bounds={"dc_capacity": np.full(n, 0.25)})
    P.extra_projections = {"dc_capacity": zap.planning.BoxBudgetProjection(
        budget=2.0, lower_bounds=np.zeros(n), upper_bounds=np.full(n, 0.25))}

    eta = {"dc_capacity": np.full(n, 2.0 / n)}
    t1 = time.time()
    J = P(**eta, requires_grad=True)
    tf = time.time() - t1
    t2 = time.time()
    g = P.backward()
    tb = time.time() - t2
    print(f"  forward(J={float(J):.4f})={tf:.1f}s  backward={tb:.1f}s")
    print(f"  grad dc_capacity: min={g['dc_capacity'].min():.4f} max={g['dc_capacity'].max():.4f} "
          f"nonzero={int((np.abs(g['dc_capacity'])>1e-9).sum())}/{n}")
