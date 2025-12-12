import numpy as np
import cvxpy as cp
import scipy.io as spio
from scipy.sparse import csc_matrix
from typing import Tuple

from .abstract_benchmark import AbstractBenchmarkSet


class NetlibBenchmarkSet(AbstractBenchmarkSet):
    def get_data(
        self, identifier: any
    ) -> Tuple[int, np.ndarray, float, csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads a Netlib .mat file and returns the data needed to build a problem.
        """
        data = spio.loadmat(identifier)
        A = csc_matrix(data["A"].astype(float))
        c = data["c"].flatten().astype(float)
        b = data["b"].flatten().astype(float)
        z0 = data["z0"].flatten().astype(float)[0]
        lo = data["lo"].flatten().astype(float)
        hi = data["hi"].flatten().astype(float)
        n = A.shape[1]

        hi = np.where(hi >= 1e19, np.inf, hi)
        lo = np.where(lo <= -1e19, -np.inf, lo)

        return n, c, z0, A, b, lo, hi

    def create_problem(
        self, data: Tuple[int, np.ndarray, float, csc_matrix, np.ndarray, np.ndarray, np.ndarray]
    ) -> cp.Problem:
        """
        Creates a CVXPY problem from the Netlib data.
        The formulation is:
            minimize: c^T x + z0
            subject to: Ax == b, lo <= x <= hi
        """
        n, c, z0, A, b, lo, hi = data

        x = cp.Variable(n)
        objective = cp.Minimize(c @ x + z0)
        constraints = [A @ x == b, lo <= x, x <= hi]
        problem = cp.Problem(objective, constraints)

        return problem
