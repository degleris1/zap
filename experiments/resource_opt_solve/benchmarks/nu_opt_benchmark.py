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
        w_range: tuple = None,
        link_congest_num_frac: float = None,  # What fraction of links to congest
        link_congest_util_frac: float = None,  # For congested links, what fraction of flows should use them
        congested_capacity_scale: float = None,
        lin_util_frac: float = None,  # Fraction of flows with linear utilities
        base_seed: int = 0,
    ):
        super().__init__(data_dir=None, num_problems=num_problems)
        self.m = m
        self.n = n
        self.avg_route_length = avg_route_length
        self.capacity_range = capacity_range
        self.w_range = w_range
        self.link_congest_num_frac = link_congest_num_frac
        self.link_congest_util_frac = link_congest_util_frac
        self.congested_capacity_scale = congested_capacity_scale
        self.lin_util_frac = lin_util_frac
        self.base_seed = base_seed

    def _build_sparse_R(self, m, n, avg_route_length, rng):
        """
        Build a sparse link-route matrix R of dimension (m,n) in CSC format with
        avg_route_length non-zero entries per column on average.
        """
        data_vals = []
        row_indices = []
        col_ptrs = [0]

        if self.link_congest_num_frac is not None and self.link_congest_util_frac is not None:
            n_congest = max(1, int(round(self.link_congest_num_frac * m)))
            congested_links = rng.choice(m, size=n_congest, replace=False)
        else:
            congested_links = None

        self.congested_links = congested_links

        for col in range(n):
            # Sample number of links for this route (could vary around the average)
            col_nnz = max(1, int(rng.poisson(avg_route_length)))
            col_nnz = min(col_nnz, m)

            selected_rows = set()
            # Try and pick congested links favorably when figuring out routes
            if congested_links is not None:
                for link in congested_links:
                    if rng.random() < self.link_congest_util_frac:
                        selected_rows.add(link)
                        if len(selected_rows) >= col_nnz:
                            break

                # We may have to still add links to the route to hit the col_nnz
                remaining = col_nnz - len(selected_rows)
                if remaining > 0:
                    # Use set logic to avoid the links (rows) we already selected
                    candidate_pool = np.setdiff1d(
                        np.arange(m), np.fromiter(selected_rows, dtype=int), assume_unique=True
                    )
                    new_rows = rng.choice(candidate_pool, size=remaining, replace=False)
                    selected_rows.update(new_rows.tolist())

                # Choose which links this route uses
                rows_for_col = np.array(list(selected_rows), dtype=int)
            else:
                rows_for_col = rng.choice(m, size=col_nnz, replace=False)

            rows_for_col.sort()
            vals_for_col = np.ones(len(rows_for_col))


            data_vals.append(vals_for_col)
            row_indices.append(rows_for_col)
            col_ptrs.append(col_ptrs[-1] + len(rows_for_col))

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

        # Potentially modify capacities for congested links
        if self.congested_links is not None and self.congested_capacity_scale is not None:
            c[self.congested_links] *= self.congested_capacity_scale

        # Generate w
        if self.w_range is not None:
            w_min, w_max = self.w_range
            w = rng.uniform(w_min, w_max, size=self.n)
        else:
            w = np.ones(self.n)

        if self.lin_util_frac is not None:
            n_lin_utils = max(1, int(round(self.lin_util_frac * self.n)))
            linear_flow_idxs = rng.choice(self.n, size=n_lin_utils, replace=False)
        else:
            linear_flow_idxs = None

        return R, c, w, linear_flow_idxs

    def create_problem(self, data):
        R, c, w, linear_flow_idxs = data
        f = cp.Variable(self.n)

        constraints = [R @ f <= c, f >= 0]

        if linear_flow_idxs is not None:
            lin_mask = np.zeros(self.n, dtype=bool)
            lin_mask[linear_flow_idxs] = True
            log_mask = ~lin_mask

            lin_objective = cp.sum(cp.multiply(w[lin_mask], f[lin_mask]))
            log_objective = cp.sum(cp.multiply(w[log_mask], cp.log(f[log_mask])))
            objective = cp.Maximize(lin_objective + log_objective)
        else:
            # Purely logarithmic objective
            objective = cp.Maximize(cp.sum(cp.multiply(w, cp.log(f))))

        problem = cp.Problem(objective, constraints)
        return problem
