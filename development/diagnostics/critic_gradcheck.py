"""
Correctness critic: finite-difference check of the implicit-differentiation
gradient of a line-congestion operation objective w.r.t. per-node data-center
capacities, through the zap bi-level planner.

We build a small toy network (5 nodes, AC lines with susceptance, two
generators, a base load, and a DataCenterLoad with 3 candidate terminals),
set up a DispatchLayer + PlanningProblem with:
    parameter_names = {"dc_capacity": (dc_idx, "nominal_capacity")}
    operation_objective = LineUtilizationObjective(metric="quadratic")
    investment_objective = InvestmentObjective (DC capital_cost = 0)
and compare:
    analytic   g = P.backward()["dc_capacity"]
vs
    finite-diff central differences of P.forward(requires_grad=False, ...)

The DataCenterLoad is a FIXED load: min_power == max_power == -profile, so its
nominal_capacity enters BOTH inequality constraints of the injector. The KKT
parameter-VJP must capture that dependence; this script is the decisive test.

Run:
    .venv/bin/python development/diagnostics/critic_gradcheck.py
"""

import numpy as np
import cvxpy as cp

import zap
from zap.planning import (
    LineUtilizationObjective,
    LMPObjective,
    InvestmentObjective,
    PlanningProblem,
)

np.set_printoptions(precision=6, suppress=True, linewidth=140)

# Device index of the DataCenterLoad in the `devices` list we build below.
DC_IDX = 0
LINE_IDX = 3  # ACLine device index
SOLVER = cp.CLARABEL
# CLARABEL is interior-point; ask for high accuracy so the implicit-diff
# linearization point and the FD perturbed solves are consistent.
SOLVER_KWARGS = {
    "tol_gap_abs": 1e-10,
    "tol_gap_rel": 1e-10,
    "tol_feas": 1e-10,
    "tol_infeas_abs": 1e-10,
    "tol_infeas_rel": 1e-10,
    "max_iter": 200,
}


def build_network(line_caps, dc_terminals=(0, 2, 4), dc_profile_scale=1.0, gen_quadratic=0.0):
    """
    5-node ring-ish network.

      - gens at node 1 (cheap) and node 3 (expensive)
      - a base load at node 4
      - a DataCenterLoad with terminals dc_terminals (fixed loads)
      - AC lines forming a connected graph with given nominal capacities
      - ground at node 0

    Single time period (T=1) keeps the dispatch deterministic & smooth.

    gen_quadratic>0 adds a quadratic generation cost.  This matters for the
    LMP check: with purely linear gen cost and no binding line, every LMP
    equals the marginal generator's cost (spatially flat, insensitive to DC
    capacity) so the LMP objective is genuinely CONSTANT and its true gradient
    is 0.  A quadratic cost makes LMPs respond smoothly to load, giving a
    nontrivial, smooth LMP gradient to finite-difference against.
    """
    T = 1
    net = zap.PowerNetwork(num_nodes=5)

    # DataCenterLoad: 3 candidate terminals, per-unit constant profile = 1.0.
    # nominal_capacity (GW) is the decision parameter we differentiate w.r.t.
    n_dc = len(dc_terminals)
    profiles = [np.array([dc_profile_scale]) for _ in range(n_dc)]  # per-unit, T=1
    dc = zap.DataCenterLoad(
        num_nodes=net.num_nodes,
        terminal=np.array(dc_terminals),
        nominal_capacity=np.array([0.5] * n_dc),  # placeholder; overwritten by layer param
        profiles=profiles,
        linear_cost=np.array([1.0] * n_dc),
        capital_cost=np.array([0.0] * n_dc),  # isolate the operation-objective gradient
        settime_horizon=T,
    )

    baseload = zap.Load(
        num_nodes=net.num_nodes,
        terminal=np.array([4]),
        load=np.array([[1.0]]),
        linear_cost=np.array([500.0]),
    )

    gens = zap.Generator(
        num_nodes=net.num_nodes,
        terminal=np.array([1, 3]),
        nominal_capacity=np.array([10.0, 10.0]),
        dynamic_capacity=np.array([[1.0], [1.0]]),
        linear_cost=np.array([10.0, 40.0]),  # node 1 cheap, node 3 expensive
        quadratic_cost=(
            None if gen_quadratic == 0.0 else np.array([gen_quadratic, gen_quadratic])
        ),
        emission_rates=np.array([0.0, 0.0]),
    )

    # AC lines: a connected graph on 5 nodes (spanning tree + a couple extra
    # so there is real flow-splitting / KCL-driven congestion).
    src = np.array([0, 1, 2, 3, 0, 1])
    snk = np.array([1, 2, 3, 4, 2, 4])
    susc = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    lines = zap.ACLine(
        num_nodes=net.num_nodes,
        source_terminal=src,
        sink_terminal=snk,
        nominal_capacity=np.array(line_caps, dtype=float),
        susceptance=susc,
        capacity=np.ones(len(src)),  # max_power = capacity = 1.0 (per-unit)
    )

    ground = zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))
    devices = [dc, baseload, gens, lines, ground]
    return net, devices, T


def make_problem(net, devices, T, op_obj_kind="lineutil"):
    layer = zap.DispatchLayer(
        net,
        devices,
        parameter_names={"dc_capacity": (DC_IDX, "nominal_capacity")},
        time_horizon=T,
        solver=SOLVER,
        solver_kwargs=SOLVER_KWARGS,
        add_ground=False,
    )

    if op_obj_kind == "lineutil":
        op_obj = LineUtilizationObjective(
            net, devices, metric="quadratic", line_device_idx=[LINE_IDX]
        )
    elif op_obj_kind == "lmp":
        op_obj = LMPObjective(net, devices, lmp_metric="l2")
    else:
        raise ValueError(op_obj_kind)

    inv_obj = InvestmentObjective(devices, layer)
    P = PlanningProblem(operation_objective=op_obj, investment_objective=inv_obj, layer=layer)
    return P, layer, op_obj


def line_utilization_report(P, layer, op_obj, eta):
    """Solve at eta and report per-line |flow|/limit and the objective value."""
    y = layer.forward(dc_capacity=eta)
    params = layer.setup_parameters(dc_capacity=eta)
    u = op_obj.utilization(y, parameters=params, la=np)
    obj = float(op_obj.forward(y, parameters=params, la=np))
    return u.ravel(), obj


def analytic_grad(P, eta):
    P(dc_capacity=eta, requires_grad=True)
    g = P.backward()["dc_capacity"]
    return np.asarray(g).ravel().astype(float)


def fd_grad(P, eta, h=5e-4):
    n = len(eta)
    g = np.zeros(n)
    for i in range(n):
        ep = eta.copy()
        em = eta.copy()
        ep[i] += h
        em[i] -= h
        fp = float(P.forward(requires_grad=False, dc_capacity=ep))
        fm = float(P.forward(requires_grad=False, dc_capacity=em))
        g[i] = (fp - fm) / (2 * h)
    return g


def compare(name, g_an, g_fd):
    # Scale-aware relative error: normalize by the gradient *magnitude*, not by
    # each (possibly ~0) component.  A coordinate whose true gradient is ~0 has
    # an undefined elementwise rel err; what matters is err relative to the size
    # of the gradient we are trying to recover.
    scale = max(np.max(np.abs(g_fd)), 1e-12)
    abs_err = np.abs(g_an - g_fd)
    rel = abs_err / scale
    # Sign check only meaningful for components that aren't numerically zero
    # relative to the gradient scale.
    nonzero = np.abs(g_fd) > 1e-6 * scale
    signs_ok = bool(np.all(np.sign(np.round(g_an, 10))[nonzero] == np.sign(g_fd)[nonzero]))
    print(f"\n=== {name} ===")
    print(f"  analytic   : {g_an}")
    print(f"  finite-diff: {g_fd}")
    print(f"  abs err    : {abs_err}")
    print(f"  rel err (/||g||): {rel}")
    print(f"  max rel err: {rel.max():.3e}   max abs err: {abs_err.max():.3e}   "
          f"||g||={scale:.3e}")
    print(f"  signs match: {signs_ok}")
    return rel.max(), signs_ok


def run_case(title, line_caps, eta, op_kind="lineutil", h=5e-4, gen_quadratic=0.0):
    print("\n" + "#" * 78)
    print(f"# {title}")
    print(f"#   line_caps={line_caps}  eta={eta}  op={op_kind}  h={h}  "
          f"gen_quad={gen_quadratic}")
    print("#" * 78)

    net, devices, T = build_network(line_caps, gen_quadratic=gen_quadratic)
    P, layer, op_obj = make_problem(net, devices, T, op_obj_kind=op_kind)

    if op_kind == "lineutil":
        u, obj = line_utilization_report(P, layer, op_obj, eta)
        print(f"  line utilizations |f|/limit: {u}")
        print(f"  operation objective value  : {obj:.6f}")
        binding = np.where(u > 0.999)[0]
        print(f"  near-binding lines (u>0.999): {binding.tolist()}")

    g_an = analytic_grad(P, eta)
    g_fd = fd_grad(P, eta, h=h)
    return compare(title, g_an, g_fd)


def main():
    results = {}

    # ------------------------------------------------------------------
    # Case A: SMOOTH regime, LineUtilizationObjective.
    # Generous line caps so nothing binds -> objective is smooth quadratic.
    # This is the decisive quantitative check.
    # ------------------------------------------------------------------
    line_caps_smooth = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    eta = np.array([0.5, 0.5, 0.5])
    results["A_lineutil_smooth"] = run_case(
        "A) LineUtilization, SMOOTH (no binding lines)",
        line_caps_smooth, eta, op_kind="lineutil",
    )

    # Same case, second step size, to confirm FD convergence (not a fluke).
    run_case(
        "A') LineUtilization, SMOOTH, h=1e-3",
        line_caps_smooth, eta, op_kind="lineutil", h=1e-3,
    )

    # Asymmetric eta (different per-node capacities) still in smooth regime.
    results["A2_lineutil_smooth_asym"] = run_case(
        "A2) LineUtilization, SMOOTH, asymmetric eta",
        line_caps_smooth, np.array([0.3, 0.7, 0.4]), op_kind="lineutil",
    )

    # ------------------------------------------------------------------
    # Case B0: LMPObjective with LINEAR gen cost, no binding line.
    # DEGENERATE: every LMP == marginal gen cost (spatially flat), so the
    # objective is genuinely CONSTANT in DC capacity -> true gradient is 0.
    # Both analytic and FD are ~0; we report it but do NOT use it for the
    # quantitative verdict (rel err is meaningless when ||g|| ~ 0).
    # ------------------------------------------------------------------
    results["B0_lmp_flat_degenerate"] = run_case(
        "B0) LMPObjective (l2), LINEAR gen cost -> FLAT LMPs (true grad ~ 0)",
        line_caps_smooth, eta, op_kind="lmp",
    )

    # ------------------------------------------------------------------
    # Case B: LMPObjective with QUADRATIC gen cost, no binding line.
    # Now LMPs respond smoothly to load => nontrivial, smooth LMP gradient.
    # This is the meaningful LMP gradient check.
    # ------------------------------------------------------------------
    results["B_lmp_smooth"] = run_case(
        "B) LMPObjective (l2), QUADRATIC gen cost -> smooth nonzero LMP gradient",
        line_caps_smooth, eta, op_kind="lmp", gen_quadratic=20.0,
    )

    # ------------------------------------------------------------------
    # Case C: A line is BINDING. Finite-diff expected to be unreliable
    # (objective only piecewise-smooth at the kink). We report it and
    # do NOT count it as a correctness failure unless signs are wrong.
    # ------------------------------------------------------------------
    line_caps_tight = [0.6, 0.6, 5.0, 5.0, 5.0, 5.0]
    results["C_lineutil_binding"] = run_case(
        "C) LineUtilization, a line BINDING (FD unreliable at kink)",
        line_caps_tight, eta, op_kind="lineutil",
    )

    # ------------------------------------------------------------------
    # Case D: Sign / direction sanity check.
    # Put a tight-ish (but not binding) line on the path that feeds a DC
    # terminal, and confirm dObj/d eta_node > 0 there (more DC capacity at a
    # node that loads a congested line increases LineUtilization).
    # ------------------------------------------------------------------
    print("\n" + "#" * 78)
    print("# D) SIGN / DIRECTION sanity check")
    print("#" * 78)
    # Moderately tight lines so loading matters but nothing binds at eta.
    line_caps_mod = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    net, devices, T = build_network(line_caps_mod)
    P, layer, op_obj = make_problem(net, devices, T, op_obj_kind="lineutil")
    eta0 = np.array([0.5, 0.5, 0.5])
    u0, obj0 = line_utilization_report(P, layer, op_obj, eta0)
    g_an = analytic_grad(P, eta0)
    print(f"  utilizations at eta0: {u0}")
    print(f"  analytic grad d(LineUtil)/d eta: {g_an}")
    # Numerically confirm: bump each node up by a finite amount, objective rises.
    for i in range(3):
        ep = eta0.copy()
        ep[i] += 0.1
        _, obji = line_utilization_report(P, layer, op_obj, ep)
        sign = "INCREASES" if obji > obj0 else "decreases"
        print(
            f"  +0.1 GW at node {i}: obj {obj0:.5f} -> {obji:.5f}  "
            f"({sign})  [analytic g_{i}={g_an[i]:+.4f}]"
        )

    # ------------------------------------------------------------------
    # VERDICT
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    SMOOTH_KEYS = [
        "A_lineutil_smooth",
        "A2_lineutil_smooth_asym",
        "B_lmp_smooth",
    ]
    worst = 0.0
    all_signs = True
    for k in SMOOTH_KEYS:
        rel, signs = results[k]
        worst = max(worst, rel)
        all_signs = all_signs and signs
        print(f"  {k:32s}: max rel err = {rel:.3e}, signs ok = {signs}")
    print(f"\n  Smooth-regime worst-case max rel err (||g||-normalized): {worst:.3e}")
    rel0, _ = results["B0_lmp_flat_degenerate"]
    print(f"  Degenerate flat-LMP case (true grad~0, EXCLUDED): max rel err = {rel0:.3e} "
          f"(both ~0; meaningless)")
    relC, signsC = results["C_lineutil_binding"]
    print(f"  Binding-line case (FD unreliable at kink)       : max rel err = {relC:.3e}, "
          f"signs ok = {signsC}")

    tol = 0.05  # 5%
    if worst < tol and all_signs:
        print(
            f"\n  VERDICT: CORRECT. The dc_capacity gradient through "
            f"LineUtilizationObjective\n  (and LMPObjective) is correct in smooth "
            f"regions (max rel err {worst*100:.3f}% < {tol*100:.0f}%)."
        )
    else:
        print(
            f"\n  VERDICT: SUSPECT. Smooth-regime max rel err {worst*100:.3f}% "
            f"exceeds {tol*100:.0f}% or signs disagree -- investigate KKT "
            f"parameter-VJP for the DataCenterLoad inequality constraints."
        )


if __name__ == "__main__":
    main()
