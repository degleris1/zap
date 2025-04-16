import numpy as np
import cvxpy as cp
import math
from collections import deque
from scipy.sparse import coo_matrix
from .abstract_benchmark import AbstractBenchmarkSet


class MaxFlowBenchmarkSet(AbstractBenchmarkSet):
    """
    Generates random max flow problems with n nodes
    """

    def __init__(
        self,
        num_problems: int,
        n: int,
        quad_obj: bool = False,
        gamma: float = 1.0,
        base_seed: int = 0,
    ):
        super().__init__(data_dir=None, num_problems=num_problems)
        self.n = n
        self.quad_obj = quad_obj
        self.gamma = gamma
        self.base_seed = base_seed

    def _generate_random_dir_adj_matrix(self, seed=42, capacity_range=(1, 10)):
        rng = np.random.default_rng(seed)
        INF = 1e3 * (capacity_range[1])

        m_target = int(math.ceil(self.n * math.log(self.n)))
        m_oversample = 4 * m_target  # we're gonna kill the lower triangular half
        linear_indices = rng.choice(self.n * self.n, size=m_oversample, replace=False)
        rows, cols = np.unravel_index(linear_indices, (self.n, self.n))

        # Keep only above the diagonal (upper triangular part)
        mask = rows < cols
        rows_filtered = rows[mask]
        cols_filtered = cols[mask]

        capacities = rng.integers(capacity_range[0], capacity_range[1], size=len(rows_filtered))

        # Add the artifical node from t (last node) to s (first node)
        rows_filtered = np.append(rows_filtered, self.n - 1)
        cols_filtered = np.append(cols_filtered, 0)
        capacities = np.append(capacities, INF)

        adjacency = coo_matrix(
            (capacities, (rows_filtered, cols_filtered)), shape=(self.n, self.n), dtype=np.float32
        )
        return adjacency

    @staticmethod
    def _is_valid_network(adj, source=0, sink=None):
        """
        Network is valid if the adjacency has a directed path from s to t
        """

        adj = adj.tocsr()
        if sink is None:
            sink = adj.shape[0] - 1

        # Use BFS to look for the sink when starting from the source
        visited = set()
        queue = deque([source])

        while queue:
            u = queue.popleft()
            if u == sink:
                return True

            # Basically going to use some CSR sorcery here
            # for a node u, we want to get its neighbors really efficiently
            # This is really just saying lets look at the row, and then within that what are the
            # non zero columns
            # That is just indptr[u] to indptr[u+1], and then use that to index into the indices array
            # We can actually shorthand that to just adj[u]—that's just the row u also in CSR format
            # So then the neighbors of u are just adj[u].indices
            row = adj[u]
            # Neighbors of u
            for v in row.indices:
                # Non-zero capacity means there's an edge
                if row[0, v] > 0 and v not in visited:
                    visited.add(v)
                    queue.append(v)

        return False

    @staticmethod
    def _adjacency_to_incidence(adj):
        """
        Given an adjacency matrix in COO format, return the incidence matrix in CSC format
        """
        n_nodes = adj.shape[0]
        n_edges = adj.nnz

        row = np.concatenate([adj.row, adj.col])
        col = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
        data = np.concatenate([np.ones(n_edges), -1 * np.ones(n_edges)])

        return coo_matrix((data, (row, col)), shape=(n_nodes, n_edges)).tocsc()

    def _generate_max_flow_problem(self, data):
        """
        Generate a random max flow problem with n nodes
        """
        inc, adj, b, c, capacities, n_edges = data
        f = cp.Variable(n_edges)
        constraints = []
        constraints.append(f <= capacities)
        constraints.append(inc @ f == b)
        if self.quad_obj:
            # -f^TQf + c^Tf, where Q = gamma*I
            obj = -self.gamma * cp.sum_squares(f) + c @ f
        else:
            obj = c @ f
        problem = cp.Problem(cp.Minimize(-obj), constraints)

        return problem, adj, inc

    def get_data(self, identifier: int):
        rng = np.random.default_rng(self.base_seed + identifier)

        source = 0
        sink = self.n - 1
        adj = self._generate_random_dir_adj_matrix(seed=rng)
        inc = self._adjacency_to_incidence(adj)

        n_nodes, n_edges = inc.shape
        edge_list = [(int(u), int(v)) for (u, v) in zip(adj.row, adj.col)]

        # Create cost vector—only cost for artificial edge
        c = np.zeros(n_edges, dtype=np.float32)
        for i, (u, v) in enumerate(edge_list):
            if u == sink and v == source:
                c[i] = 1

        # Get capacities for constraints
        capacities = adj.data

        # Create b vector—
        b = np.zeros(n_nodes, dtype=np.float32)

        return (inc, adj, b, c, capacities, n_edges)

    def create_problem(self, data):
        """
        Build the actual CVXPY Problem:

          minimize    -f^TQf + c^Tf, where Q = gamma*I
          subject to  f <= capacities
                        inc @ f == b
        """
        valid_source_sink_path = False
        while not valid_source_sink_path:
            problem, adj, inc = self._generate_max_flow_problem(data)
            valid_source_sink_path = self._is_valid_network(adj)
            if not valid_source_sink_path:
                self.base_seed += 1

        return problem
