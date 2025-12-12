import numpy as np
import cvxpy as cp
from .abstract_benchmark import AbstractBenchmarkSet


class LassoBenchmarkSet(AbstractBenchmarkSet):
    def __init__(
        self,
        num_problems: int,
        n: int = 8,
        m: int = 50,
        density: float = 0.1,
        noise_scale: float = 0.1,
        base_seed: int = 0,
    ):
        super().__init__(data_dir=None, num_problems=num_problems)
        self.n = n
        self.m = m
        self.density = density
        self.base_seed = base_seed

    def get_data(self, identifier: int):
        """
        Generates data for LASSO problems.
        Follows LASSO problem generation steps SCS paper.
        """
        seed = self.base_seed + identifier
        rng = np.random.default_rng(seed)

        F = rng.normal(loc=0.0, scale=1.0, size=(self.m, self.n))

        z_true = np.zeros(self.n)
        num_nonzero = int(np.round(self.density * self.n))
        nonzero_indices = rng.choice(self.n, size=num_nonzero, replace=False)
        z_true[nonzero_indices] = rng.normal(loc=1.0, scale=0.5, size=num_nonzero)

        w = rng.normal(loc=0.0, scale=0.1, size=(self.m,))
        g = F @ z_true + w

        mu_max = np.linalg.norm(F.T @ g, ord=np.inf)
        mu = 0.1 * mu_max

        return F, g, mu

    def create_problem(self, data):
        """
        Create a LASSO Problem in CVXPY.
        The formulation is:
            minimize 0.5 * ||F z - g||^2_2 + mu * ||z||_1
        """
        F, g, mu = data
        z = cp.Variable(self.n)

        cost = 0.5 * cp.sum_squares(F @ z - g) + mu * cp.norm(z, 1)
        problem = cp.Problem(cp.Minimize(cost))
        return problem
