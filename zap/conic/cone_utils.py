import numpy as np
import cvxpy as cp
import math
from scipy.sparse import coo_matrix
from collections import deque


def get_standard_conic_problem(problem, solver):
    probdata, _, _ = problem.get_problem_data(solver)
    data = {
        "A": probdata["A"],
        "b": probdata["b"],
        "c": probdata["c"],
    }
    cone_dims = probdata["dims"]
    cones = {
        "z": cone_dims.zero,
        "l": cone_dims.nonneg,
        "q": cone_dims.soc,
        "ep": cone_dims.exp,
        "s": cone_dims.psd,
    }
    cone_params = {
        "A": probdata["A"],
        "b": probdata["b"],
        "c": probdata["c"],
        "K": cones,
    }

    return cone_params, data, cones


### MAX NETWORK FLOW UTILS ###
def generate_random_dir_adj_matrix(n, seed=42, capacity_range=(1, 10)):
    rng = np.random.default_rng(seed)
    INF = 1e3 * (capacity_range[1])

    m_target = int(math.ceil(n * math.log(n)))
    m_oversample = 4 * m_target  # we're gonna kill the lower triangular half
    linear_indices = rng.choice(n * n, size=m_oversample, replace=False)
    rows, cols = np.unravel_index(linear_indices, (n, n))

    # Keep only above the diagonal (upper triangular part)
    mask = rows < cols
    rows_filtered = rows[mask]
    cols_filtered = cols[mask]

    capacities = rng.integers(capacity_range[0], capacity_range[1], size=len(rows_filtered))

    # Add the artifical node from t (last node) to s (first node)
    rows_filtered = np.append(rows_filtered, n - 1)
    cols_filtered = np.append(cols_filtered, 0)
    capacities = np.append(capacities, INF)

    adjacency = coo_matrix(
        (capacities, (rows_filtered, cols_filtered)), shape=(n, n), dtype=np.float32
    )
    return adjacency


def is_valid_network(adj, source=0, sink=None):
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


def adjacency_to_incidence(adj):
    """
    Given an adjacency matrix in COO format, return the incidence matrix in CSC format
    """
    n_nodes = adj.shape[0]
    n_edges = adj.nnz

    row = np.concatenate([adj.row, adj.col])
    col = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    data = np.concatenate([np.ones(n_edges), -1 * np.ones(n_edges)])

    return coo_matrix((data, (row, col)), shape=(n_nodes, n_edges)).tocsc()


def generate_max_flow_problem(n, seed=42):
    """
    Generate a random max flow problem with n nodes
    """

    source = 0
    sink = n - 1
    adj = generate_random_dir_adj_matrix(n, seed=seed)
    inc = adjacency_to_incidence(adj)

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

    # Formulate problem in CVXPY
    f = cp.Variable(n_edges)
    constraints = []
    constraints.append(f <= capacities)
    constraints.append(inc @ f == b)
    obj = c @ f
    problem = cp.Problem(cp.Maximize(obj), constraints)

    return problem, adj, inc
