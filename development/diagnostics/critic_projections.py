"""
Numeric correctness checks for the budget/box projections used by the
gradient-based planning solver in zap.

We compare zap's projection implementations against an INDEPENDENT cvxpy QP
that computes the true Euclidean projection onto the relevant feasible set.

Run with:  .venv/bin/python development/diagnostics/critic_projections.py
"""

import numpy as np
import cvxpy as cp

from zap.planning.projection import SimplexBudgetProjection, BoxBudgetProjection


def true_simplex_proj(y, budget, strict=True):
    """Euclidean projection onto {x >= 0, sum(x) == budget} (strict)
    or {x >= 0, sum(x) <= budget} (non-strict)."""
    y = np.asarray(y, dtype=float).ravel()
    x = cp.Variable(len(y))
    cons = [x >= 0]
    if strict:
        cons.append(cp.sum(x) == budget)
    else:
        cons.append(cp.sum(x) <= budget)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - y)), cons)
    prob.solve(solver=cp.CLARABEL)
    return x.value


def true_box_proj(y, budget, lower, upper):
    """Euclidean projection onto {lower <= x <= upper, sum(x) == budget}."""
    y = np.asarray(y, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    x = cp.Variable(len(y))
    cons = [x >= lower, x <= upper, cp.sum(x) == budget]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - y)), cons)
    prob.solve(solver=cp.CLARABEL)
    return x.value


def report(name, y, got, ref, budget=None):
    got = np.asarray(got, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    err = np.max(np.abs(got - ref))
    print(f"\n--- {name} ---")
    print(f"  input y      = {np.array2string(np.asarray(y, float), precision=4)}")
    print(f"  zap result   = {np.array2string(got, precision=4)}")
    print(f"  true proj    = {np.array2string(ref, precision=4)}")
    print(f"  max |diff|   = {err:.3e}")
    if budget is not None:
        print(f"  zap sum      = {got.sum():.6f}   (budget {budget})")
        print(f"  zap min      = {got.min():.6f}")
    status = "OK" if err < 1e-5 else "*** MISMATCH ***"
    print(f"  verdict      = {status}")
    return err


def main():
    np.random.seed(0)
    print("=" * 70)
    print("PART 3 — SimplexBudgetProjection (strict=True): {x>=0, sum==budget}")
    print("=" * 70)

    budget = 2.5
    cases = [
        ("already feasible-ish, positive", np.array([0.5, 0.5, 0.5, 0.5, 0.5])),
        ("sum > budget, all positive", np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
        ("sum < budget, all positive", np.array([0.1, 0.2, 0.1, 0.05, 0.05])),
        ("contains negatives", np.array([-1.0, 0.3, 2.0, -0.5, 1.0])),
        ("all negative", np.array([-1.0, -2.0, -0.5, -3.0, -0.1])),
        ("one big positive, rest negative", np.array([10.0, -1.0, -1.0, -1.0, -1.0])),
    ]
    max_err = 0.0
    for name, y in cases:
        got = SimplexBudgetProjection(budget, strict=True)(y.copy())
        ref = true_simplex_proj(y, budget, strict=True)
        max_err = max(max_err, report("strict | " + name, y, got, ref, budget))

    print("\n" + "=" * 70)
    print("PART 3b — SimplexBudgetProjection (strict=False): {x>=0, sum<=budget}")
    print("=" * 70)
    cases_ns = [
        ("sum < budget (should stay put)", np.array([0.1, 0.2, 0.1, 0.05, 0.05])),
        ("sum > budget, positive", np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
        ("negatives, sum<budget after clip", np.array([-1.0, 0.3, 0.2, -0.5, 0.1])),
        ("negatives, sum>budget after clip", np.array([-1.0, 3.0, 2.0, -0.5, 4.0])),
    ]
    for name, y in cases_ns:
        got = SimplexBudgetProjection(budget, strict=False)(y.copy())
        ref = true_simplex_proj(y, budget, strict=False)
        max_err = max(max_err, report("nonstrict | " + name, y, got, ref, budget))

    print("\n" + "=" * 70)
    print("PART 4 — BoxBudgetProjection: {lower<=x<=upper, sum==budget}")
    print("=" * 70)
    n = 6
    box_budget = 2.5
    lower = np.zeros(n)
    upper = np.full(n, 1.0)  # per-site cap 1.0; sum(upper)=6 >= 2.5 >= sum(lower)=0
    for trial in range(3):
        y = np.random.randn(n) * 2.0
        got = BoxBudgetProjection(box_budget, lower, upper)(y.copy())
        ref = true_box_proj(y, box_budget, lower, upper)
        max_err = max(max_err, report(f"box trial {trial}", y, got, ref, box_budget))

    # nontrivial lower bounds
    lower2 = np.array([0.1, 0.0, 0.2, 0.0, 0.3, 0.0])
    upper2 = np.array([0.6, 0.5, 0.7, 0.4, 0.9, 0.5])
    for trial in range(2):
        y = np.random.randn(n) * 1.5
        got = BoxBudgetProjection(box_budget, lower2, upper2)(y.copy())
        ref = true_box_proj(y, box_budget, lower2, upper2)
        max_err = max(max_err, report(f"box nontrivial bounds {trial}", y, got, ref, box_budget))

    print("\n" + "=" * 70)
    print("PART 4b — BoxBudgetProjection feasibility guard")
    print("=" * 70)
    try:
        BoxBudgetProjection(0.0, np.full(3, 1.0), np.full(3, 2.0))  # sum(lower)=3 > 0
        print("  budget too small: NO error raised  *** BAD ***")
    except ValueError as e:
        print(f"  budget too small: ValueError raised OK -> {e}")
    try:
        BoxBudgetProjection(100.0, np.full(3, 0.0), np.full(3, 1.0))  # sum(upper)=3 < 100
        print("  budget too large: NO error raised  *** BAD ***")
    except ValueError as e:
        print(f"  budget too large: ValueError raised OK -> {e}")

    print("\n" + "=" * 70)
    print(f"OVERALL max projection error vs true QP: {max_err:.3e}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # PART 1/2 demonstration: SimplexBudgetProjection ignores upper bounds
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 1 — Does SimplexBudgetProjection respect per-site upper caps?")
    print("=" * 70)
    cap = 0.25  # like workload_variation_tb.py upper_bounds = 0.250
    n = 10
    budget = 2.5  # == n*cap exactly here, but a single coordinate can still exceed cap
    # craft a gradient-step state that wants everything piled on one node
    y = np.zeros(n)
    y[0] = 5.0
    out = SimplexBudgetProjection(budget, strict=True)(y.copy())
    print(f"  per-site cap (upper_bounds) = {cap}")
    print(f"  simplex projection output   = {np.array2string(out, precision=4)}")
    print(f"  max coordinate              = {out.max():.4f}")
    print(f"  number of coords > cap      = {(out > cap + 1e-9).sum()}")
    if out.max() > cap + 1e-9:
        print("  => SimplexBudgetProjection VIOLATES the per-site cap. Footgun confirmed.")
    box = BoxBudgetProjection(budget, np.zeros(n), np.full(n, cap))(y.copy())
    print(f"  BoxBudgetProjection output  = {np.array2string(box, precision=4)}")
    print(f"  Box max coordinate          = {box.max():.4f} (should be <= {cap})")

    # ------------------------------------------------------------------
    # PART 2 — convergence: does raw-grad PGD with zap's simplex projection
    # reach the true optimum? (It should, but the buggy pre-clip breaks it.)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2 — raw-grad PGD vs mean-removed-grad PGD on a simplex QP")
    print("=" * 70)
    np.random.seed(1)
    m = 5
    A = np.random.randn(m, m)
    A = A.T @ A + np.eye(m)
    bb = np.random.randn(m)
    bud = 2.5

    xv = cp.Variable(m)
    cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, A) - bb @ xv),
               [xv >= 0, cp.sum(xv) == bud]).solve(solver=cp.CLARABEL)
    xstar = xv.value
    fstar = 0.5 * xstar @ A @ xstar - bb @ xstar

    def grad(x):
        return A @ x - bb

    def run(use_meanremove, proj, iters=20000, eta=0.01):
        x = np.full(m, bud / m)
        for _ in range(iters):
            g = grad(x)
            if use_meanremove:
                g = g - g.mean()
            x = proj(x - eta * g)
        return 0.5 * x @ A @ x - bb @ x - fstar

    Pzap = SimplexBudgetProjection(bud, strict=True)

    def proj_qp(y):
        z = cp.Variable(m)
        cp.Problem(cp.Minimize(cp.sum_squares(z - y)),
                   [z >= 0, cp.sum(z) == bud]).solve(solver=cp.CLARABEL)
        return z.value

    print(f"  zap-proj + RAW grad        optimality gap: {run(False, Pzap):.2e}   <- nonzero = BUG")
    print(f"  zap-proj + mean-removed    optimality gap: {run(True,  Pzap):.2e}   <- mean-removal masks it")
    print(f"  TRUE QP-proj + RAW grad    optimality gap: {run(False, proj_qp):.2e}   <- correct PGD converges")
    print("\n  => raw-grad PGD with a CORRECT projection converges. zap's projection bug")
    print("     poisons the raw-grad path; mean-removal accidentally compensates.")


if __name__ == "__main__":
    main()
