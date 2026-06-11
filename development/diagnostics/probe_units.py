"""Audit units: confirm LMP x100 == $/MWh by comparing to generator marginal
costs, quantify load-shedding (VOLL) regime, and show the upsample x4 inflation."""
import os
from copy import deepcopy
import cvxpy as cp
import numpy as np
import pypsa
import zap
from zap.importers.pypsa import load_pypsa_network

POWER_UNIT, COST_UNIT = 1.0e3, 100.0
pn = pypsa.Network(os.path.expanduser("~/zap_data/pypsa-networks/western_small/network_2023.nc"))
snaps = pn.generators_t.p_max_pu.index
sd = snaps[5448:5472]
net, devices = load_pypsa_network(pn, sd, power_unit=POWER_UNIT, cost_unit=COST_UNIT)
devices = deepcopy(devices)
devices[1].load *= 1.27
devices[0].dynamic_capacity *= 1.24
devices[3].nominal_capacity *= 1.0
G, L = devices[0], devices[1]

# Generator marginal cost in scaled units and real $/MWh
glc = np.asarray(G.linear_cost)  # scaled ($/MWh / cost_unit)
print(f"Generator linear_cost (scaled): min={glc.min():.3f} med={np.median(glc):.3f} max={glc.max():.3f}")
print(f"  -> x{COST_UNIT} = $/MWh: min={glc.min()*COST_UNIT:.1f} med={np.median(glc)*COST_UNIT:.1f} max={glc.max()*COST_UNIT:.1f}")
print(f"Load marginal value (VOLL) scaled: {np.asarray(L.linear_cost).max():.3f} -> ${np.asarray(L.linear_cost).max()*COST_UNIT:.0f}/MWh")

base = net.dispatch(devices, time_horizon=24, solver=cp.CLARABEL, add_ground=False)
p = np.asarray(base.prices) * COST_UNIT
print(f"\nBase LMP $/MWh: median={np.median(p):.2f}  (should ~= median gen cost {np.median(glc)*COST_UNIT:.1f})")

# Load served vs requested -> shedding?
served = -np.asarray(base.power[1][0])          # MW-equiv served (positive)
requested = np.asarray(L.load) * np.asarray(L.nominal_capacity)  # GW per (i,t)
served_gw = served * np.asarray(L.nominal_capacity) if served.shape == requested.shape else served
total_req = requested.sum()
# base.power[1][0] is the load device power in GW already (= -load*nominal). Compare directly:
load_power_gw = -np.asarray(base.power[1][0])   # GW served
print(f"\nLoad requested total (GW-steps): {requested.sum():.2f}")
print(f"Load served total   (GW-steps): {load_power_gw.sum():.2f}")
shed = requested.sum() - load_power_gw.sum()
print(f"Load SHED (GW-steps): {shed:.4f}  ({100*shed/requested.sum():.3f}% of demand)")
print(f"Fraction of node-times at VOLL (LMP>500$): {100*np.mean(p>500):.3f}%")

# Upsample x4 cost inflation check
up = [d.sample_time(96, 24) for d in devices]
base96 = net.dispatch(up, time_horizon=96, solver=cp.CLARABEL, add_ground=False)
print(f"\nDispatch cost 24-step: {base.problem.value:.3f}")
print(f"Dispatch cost 96-step (4x repeat): {base96.problem.value:.3f}  ratio={base96.problem.value/base.problem.value:.3f}")
p96 = np.asarray(base96.prices) * COST_UNIT
print(f"LMP median 24-step: {np.median(p):.2f}  96-step: {np.median(p96):.2f}  (price is per-timestep; x4 in pushing_capacity is a BUG)")
