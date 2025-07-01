import numpy as np
import cvxpy as cp
import scipy.sparse as sp
from experiments.conic_solve.benchmarks.abstract_benchmark import AbstractBenchmarkSet

class NUOptBenchmarkSet(AbstractBenchmarkSet):
    def __init__(
        self,
        num_problems: int,
        m: int,
        n: int,
        avg_route_length: float,
        capacity_range: tuple,
        w_range: tuple = None,  # Optional weight range
        base_seed: int = 0
    ):
        super().__init__(data_dir=None, num_problems=num_problems)
        self.m = m
        self.n = n
        self.avg_route_length = avg_route_length
        self.capacity_range = capacity_range
        self.w_range = w_range
        self.base_seed = base_seed
        
    def _build_sparse_R(self, m, n, avg_route_length, rng):
        """
        Build a sparse link-route matrix R of dimension (m,n) in CSC format with 
        avg_route_length non-zero entries per column on average.
        """
        data_vals = []
        row_indices = []
        col_ptrs = [0]

        for col in range(n):
            # Sample number of links for this route (could vary around the average)
            col_nnz = max(1, int(rng.poisson(avg_route_length)))
            col_nnz = min(col_nnz, m)
            
            # Choose which links this route uses
            rows_for_col = rng.choice(m, size=col_nnz, replace=False)
            vals_for_col = np.ones(col_nnz)

            # Sort row indices for CSC format
            sorted_idx = np.argsort(rows_for_col)
            rows_for_col = rows_for_col[sorted_idx]
            vals_for_col = vals_for_col[sorted_idx]

            data_vals.append(vals_for_col)
            row_indices.append(rows_for_col)
            col_ptrs.append(col_ptrs[-1] + col_nnz)

        data_vals = np.concatenate(data_vals)
        row_indices = np.concatenate(row_indices)

        R = sp.csc_matrix((data_vals, row_indices, col_ptrs), shape=(m, n))
        return R

    def get_data(self, identifier: int):
        rng = np.random.default_rng(self.base_seed + identifier)
        
        # Generate link-route matrix R
        p = self.avg_route_length / self.m
        R = self._build_sparse_R(self.m, self.n, self.avg_route_length, rng)
        
        # Generate capacities uniformly
        c_min, c_max = self.capacity_range
        c = rng.uniform(c_min, c_max, size=self.m)

        # Generate w
        if self.w_range is not None:
            w_min, w_max = self.w_range
            w = rng.uniform(w_min, w_max, size=self.n)
        else:
            w = np.ones(self.n)
        
        return R, c, w
        
    def create_problem(self, data):
        R, c, w = data
        f = cp.Variable(self.n)
        
        constraints = [R @ f <= c, f >= 0]
        
        objective = cp.Maximize(cp.sum(cp.multiply(w, cp.log(f))))
        
        problem = cp.Problem(objective, constraints)
        return problem