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
        Simplex projection algorithm from Duchi et al. (2008)
        https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
        """
        x = np.maximum(x, 0.0)
        s = x.sum()
        if (self.strict and abs(s - self.budget) < 1e-12) or (not self.strict and s <= self.budget):
            return x
        u = np.sort(x)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - self.budget))[0][-1]
        theta = (cssv[rho] - self.budget) / (rho + 1)
        return np.maximum(x - theta, 0.0)
