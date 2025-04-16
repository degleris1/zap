import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from .abstract_benchmark import AbstractBenchmarkSet


class SparseConeBenchmarkSet(AbstractBenchmarkSet):
    """
    Simplified version of what they do in the SCS paper (only generating primal feasible)
    """

    def __init__(
        self,
        num_problems: int,
        n: int,
        p_f: float = 0.1,  # fraction of zero cone
        p_l: float = 0.3,  # fraction of nonneg cone
        factor: int = 4,  # factor to multiply n by to get m
        base_seed: int = 0,
    ):
        super().__init__(data_dir=None, num_problems=num_problems)
        self.n = n
        self.p_f = p_f
        self.p_l = p_l
        self.factor = factor
        self.base_seed = base_seed

    def _partition_rows_into_cones(self, m: int, rng: np.random.Generator):
        # Basic counts for zero, linear, and leftover (SOC)
        z = int(np.floor(m * self.p_f))
        l = int(np.floor(m * self.p_l))
        leftover = m - z - l

        # Partition leftover into SOC blocks
        # With random block sizes up to max_q
        max_q = int(np.ceil(m / np.log(m))) if m > 1 else m
        soc_sizes = []
        while leftover > 0:
            block_sz = rng.integers(low=1, high=max_q + 1)
            block_sz = min(block_sz, leftover)
            soc_sizes.append(block_sz)
            leftover -= block_sz

        return z, l, soc_sizes

    def _build_sparse_A(self, m: int, n: int, rng: np.random.Generator) -> sp.csc_matrix:
        """
        Build a sparse matrix A of dimension (m,n) in CSC format with ~sqrt(n) non-zero entries
        per column.
        """
        col_nnz = int(np.ceil(np.sqrt(n)))
        data_vals = []
        row_indices = []
        col_ptrs = [0]

        for col in range(n):
            rows_for_col = rng.choice(m, size=col_nnz, replace=False)
            vals_for_col = rng.normal(loc=0.0, scale=1.0, size=col_nnz)

            sorted_idx = np.argsort(rows_for_col)
            rows_for_col = rows_for_col[sorted_idx]
            vals_for_col = vals_for_col[sorted_idx]

            data_vals.append(vals_for_col)
            row_indices.append(rows_for_col)
            col_ptrs.append(col_ptrs[-1] + col_nnz)

        data_vals = np.concatenate(data_vals)
        row_indices = np.concatenate(row_indices)

        A = sp.csc_matrix((data_vals, row_indices, col_ptrs), shape=(m, n))
        return A

    def _make_feasible_x_and_b(
        self, A: sp.csc_matrix, z: int, l: int, soc_sizes: list, rng: np.random.Generator
    ):
        """
        Creates a random feasible x and a corresponding b
        to satisfy the cones in the problem.
        """
        m, n = A.shape
        x_feas = rng.normal(loc=0, scale=1, size=n)
        Ax_full = A @ x_feas

        b = np.zeros(m, dtype=float)

        # Zero cone
        if z > 0:
            b[:z] = Ax_full[:z]

        # Nonneg cone
        if l > 0:
            start_lin = z
            end_lin = z + l
            margin = rng.uniform(low=0.1, high=1.0, size=l)
            b[start_lin:end_lin] = Ax_full[start_lin:end_lin] - margin

        # SOC
        start_soc = z + l
        for block_sz in soc_sizes:
            end_soc = start_soc + block_sz
            if block_sz == 1:
                # Single-dimensional SOC block means >= 0
                margin = rng.uniform(low=0.1, high=1.0)
                b[start_soc] = Ax_full[start_soc] - margin
            else:
                t_val = rng.uniform(low=1.0, high=2.0)
                x_block_val = rng.normal(loc=0.0, scale=0.5, size=block_sz - 1)
                while np.linalg.norm(x_block_val, 2) >= t_val:
                    x_block_val *= 0.5
                full_block = np.concatenate(([t_val], x_block_val))
                b[start_soc:end_soc] = Ax_full[start_soc:end_soc] - full_block

            start_soc = end_soc

        return x_feas, b

    def get_data(self, identifier: int):
        rng = np.random.default_rng(self.base_seed + identifier)

        n = self.n
        m = self.factor * n

        z, l, soc_sizes = self._partition_rows_into_cones(m, rng)
        A = self._build_sparse_A(m, n, rng)
        x_feas, b = self._make_feasible_x_and_b(A, z, l, soc_sizes, rng)
        c = rng.normal(loc=0.0, scale=1.0, size=n)
        cone_partitions = {"z": z, "l": l, "soc_sizes": soc_sizes}
        return (A, b, c, cone_partitions)

    def create_problem(self, data):
        """
        Build the actual CVXPY Problem:

          minimize    c^T x
          subject to  Ax - b in K (product of cones)
        """
        A, b, c, cone_partitions = data
        z = cone_partitions["z"]
        l = cone_partitions["l"]
        soc_sizes = cone_partitions["soc_sizes"]

        m, n = A.shape
        x = cp.Variable(n)

        Ax = A @ x
        constraints = []

        if z > 0:
            constraints.append(Ax[:z] == b[:z])

        if l > 0:
            start_lin = z
            end_lin = z + l
            constraints.append(Ax[start_lin:end_lin] >= b[start_lin:end_lin])

        start_soc = z + l
        for block_sz in soc_sizes:
            end_soc = start_soc + block_sz
            if block_sz == 1:
                constraints.append(Ax[start_soc] >= b[start_soc])
            else:
                t = Ax[start_soc] - b[start_soc]
                x_block = Ax[start_soc + 1 : end_soc] - b[start_soc + 1 : end_soc]
                constraints.append(cp.SOC(t, x_block))
            start_soc = end_soc

        objective = cp.Minimize(c @ x)
        problem = cp.Problem(objective, constraints)
        return problem
