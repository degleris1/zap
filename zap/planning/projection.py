import cvxpy as cp
import numpy as np


class Projection:
    def __call__(self, x):
        raise NotImplementedError


class SimplexBudgetProjection(Projection):
    def __init__(self, budget, strict=True):
        self.budget = budget
        self.strict = strict

    def __call__(self, x):
        """
        Euclidean projection onto {x >= 0, sum(x) == budget} (strict) or
        {x >= 0, sum(x) <= budget} (non-strict), via Duchi et al. (2008).
        https://ai.stanford.edu/~jduchi/projects/jd_ss_ys_l1.pdf

        NB: the Duchi algorithm must run on the RAW input, not a >=0-clipped copy.
        Pre-clipping negatives discards the mass they should redistribute onto the
        surviving coordinates and yields an incorrect projection (e.g. inputs with
        negative entries, which routinely arise from a gradient step).
        """
        x = np.asarray(x, dtype=float).ravel()
        if not self.strict:
            # Projection onto {x >= 0, sum <= budget}: if clipping to the positive
            # orthant already satisfies the budget, that IS the projection.
            clipped = np.maximum(x, 0.0)
            if clipped.sum() <= self.budget:
                return clipped
        u = np.sort(x)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - self.budget))[0][-1]
        theta = (cssv[rho] - self.budget) / (rho + 1)
        return np.maximum(x - theta, 0.0)


class BoxBudgetProjection(Projection):
    """Project onto intersection of box constraints and budget equality."""

    def __init__(self, budget, lower_bounds, upper_bounds):
        self.budget = budget
        self.lower_bounds = np.asarray(lower_bounds).ravel()
        self.upper_bounds = np.asarray(upper_bounds).ravel()
        self.n = len(self.lower_bounds)

        # Check feasibility
        if np.sum(self.lower_bounds) > budget + 1e-9:
            raise ValueError(f"Budget {budget} too small for sum of lower bounds {np.sum(self.lower_bounds)}")
        if np.sum(self.upper_bounds) < budget - 1e-9:
            raise ValueError(f"Budget {budget} too large for sum of upper bounds {np.sum(self.upper_bounds)}")

    def __call__(self, y):
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
        y = np.asarray(y, dtype=float).ravel()
        x = cp.Variable(self.n)
        objective = cp.Minimize(cp.sum_squares(x - y))
        constraints = [
            x >= self.lower_bounds,
            x <= self.upper_bounds,
            cp.sum(x) == self.budget
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)
        if problem.status not in ("optimal", "optimal_inaccurate") or x.value is None:
            raise ValueError(f"BoxBudgetProjection failed with status {problem.status}")
        return x.value
