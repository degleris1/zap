import numpy as np
import cvxpy as cp
import scipy.io as spio
from scipy.sparse import csc_matrix
from typing import Tuple

from .abstract_benchmark import AbstractBenchmarkSet


class MarosBenchmarkSet(AbstractBenchmarkSet):
    def get_data(
        self, identifier: any
    ) -> Tuple[int, csc_matrix, np.ndarray, float, csc_matrix, np.ndarray, np.ndarray]:
        """
        Loads a Maros–Meszaros .mat file and returns the data needed to build a problem.
        """
        data = spio.loadmat(identifier)

        P = csc_matrix(data["P"].astype(float))
        q = data["q"].flatten().astype(float)
        A = csc_matrix(data["A"].astype(float))
        l = data["l"].flatten().astype(float)
        u = data["u"].flatten().astype(float)
        r = data["r"].flatten().astype(float)[0]
        m = int(data["m"].flatten().astype(int)[0])
        n = int(data["n"].flatten().astype(int)[0])

        l = np.where(l > 9e19, np.inf, l)
        l = np.where(l < -9e19, -np.inf, l)
        u = np.where(u > 9e19, np.inf, u)
        u = np.where(u < -9e19, -np.inf, u)

        return n, P, q, r, A, l, u

    def create_problem(
        self, data: Tuple[int, csc_matrix, np.ndarray, float, csc_matrix, np.ndarray, np.ndarray]
    ) -> cp.Problem:
        """
        Creates a CVXPY problem from the Maros–Meszaros data.
        The formulation is:
            minimize: 0.5 * x^T P x + q^T x + r
            subject to: l <= A x <= u
        """
        n, P, q, r, A, l, u = data

        x = cp.Variable(n)
        objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x + r)
        constraints = [A @ x >= l, A @ x <= u]
        problem = cp.Problem(objective, constraints)

        return problem
