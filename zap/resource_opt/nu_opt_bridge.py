import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from zap.network import PowerNetwork
from ..conic.variable_device import VariableDevice
from .utility_device import LogUtilityDevice
from ..conic.slack_device import (
    ZeroConeSlackDevice,
    NonNegativeConeSlackDevice,
    SecondOrderConeSlackDevice,
)
from scipy.sparse import csc_matrix, isspmatrix_csc, vstack
from zap.conic.cone_utils import (
    build_symmetric_M,
    scale_cols_csc,
    scale_rows_csr,
)
from copy import deepcopy


class NUOptBridge:
    def __init__(self, nu_opt_params: dict, grouping_params: dict | None = None, ruiz_iters: int = 0):
        self.R = nu_opt_params["R"]
        self.capacities = nu_opt_params["capacities"]
        self.w = nu_opt_params["w"]
        self.G = self.R
        self.net = None
        self.time_horizon = 1
        self.devices = []

        # flow_lengths = np.array(self.R.sum(axis=0)).flatten()  # Path lengths per flow
        # link_flows = np.array(self.R.sum(axis=1)).flatten()     # Flow counts per link
        
        # # 1. Capacity scaling using 25th percentile (avoid min bottleneck)
        # capacity_scale = 1 / np.quantile(self.capacities[self.capacities > 0], 0.25)
        # capacities_scaled = self.capacities * capacity_scale
        
        # # 2. Row scaling: 1/sqrt(link_flows) to balance hub links
        # row_scale = 1 / np.sqrt(np.maximum(link_flows, 1))  # Avoid division by zero
        # R_scaled = scale_rows_csr(self.R.tocsr(), row_scale)
        # capacities_scaled *= row_scale
        
        # # 3. Column scaling: 1/sqrt(flow_lengths) to normalize long paths
        # col_scale = 1 / np.sqrt(np.maximum(flow_lengths, 1))
        # R_scaled = scale_cols_csc(R_scaled.tocsc(), col_scale)
        
        # # 4. Compute ρ initialization heuristic
        # avg_path_length = np.mean(flow_lengths)
        # rho_init = 1 / np.log1p(avg_path_length)  # Empirical heuristic

        # self.R = R_scaled
        # self.capacities = capacities_scaled


        if not isspmatrix_csc(self.R):
            self.R = csc_matrix(self.R)

        grouping_params = grouping_params or {}
        self.variable_grouping_strategy = grouping_params.get(
            "variable_grouping_strategy", "binned_terminal_groups"
        )
        self.variable_grouping_bin_edges = grouping_params.get(
            "variable_grouping_bin_edges", (0, 10, 100, 1000)
            # "variable_grouping_bin_edges", (0, 5, 10, 15, 20, 100, 1000)

        )
        self._transform()

    def _transform(self):
        # self.G, _, _, self.capacities, self.obj_shift = self._equilibrate_ruiz(self.G, self.capacities)
        self._build_network()
        self._group_variable_devices()
        self._create_variable_devices()
        self._group_slack_devices()
        self._create_slack_devices()

    def _equilibrate_ruiz(self, R, c, max_iters=4, tol=1e-3):
        """
        Ruiz row/column equilibration for a *CSC* matrix R (in-place).
        Returns: (R_scaled, row_scale, col_scale)
        """
        if not isspmatrix_csc(R):
            R = csc_matrix(R)          # make a copy in CSC

        m, n = R.shape
        d_r = np.ones(m)              # diag scaling factors we accumulate
        d_c = np.ones(n)

        for _ in range(max_iters):
            # ---- column scaling  ------------------------------------------
            col_norm = np.sqrt(R.power(2).sum(axis=0)).A1
            gamma_c  = np.where(col_norm > 0, 1.0 / np.sqrt(col_norm), 1.0)
            scale_cols_csc(R, gamma_c)         # in-place
            d_c *= gamma_c

            # ---- row scaling  ---------------------------------------------
            row_norm = np.sqrt(R.power(2).sum(axis=1)).A1
            gamma_r  = np.where(row_norm > 0, 1.0 / np.sqrt(row_norm), 1.0)
            scale_rows_csr(R.tocsr(), gamma_r) # need CSR view for rows
            d_r *= gamma_r

            if (np.max(col_norm) < 1 + tol) and (np.max(row_norm) < 1 + tol):
                break

        # scale the capacity vector so the inequality stays equivalent
        c_scaled = d_r * c
        obj_shift = (self.w * np.log(d_c)).sum()
        return R, d_r, d_c, c_scaled, obj_shift

    def _build_network(self):
        self.net = PowerNetwork(self.G.shape[0])

    def _group_variable_devices(self):
        """
        Figure out the appropriate grouping of variable devices based on the number of terminals they have.
        """
        if self.variable_grouping_strategy == "discrete_terminal_groups":
            # Each group consists of devices with exactly the same number of terminals
            self._compute_discrete_terminal_groups()
        elif self.variable_grouping_strategy == "binned_terminal_groups":
            # Each group consists of devices with a number of terminals in the same bin
            self._compute_binned_terminal_groups(self.variable_grouping_bin_edges)

    def _compute_binned_terminal_groups(self, bin_edges):
        """
        This function makes a separate group for each set of devices with a number of terminals in the same bin.
        """
        self.device_group_map_list = []
        self.terminal_groups = []

        num_terminals_per_device_list = np.diff(self.G.indptr)
        positive_mask = num_terminals_per_device_list > 0
        filtered_counts = num_terminals_per_device_list[positive_mask]
        device_idxs = np.nonzero(positive_mask)[0]

        # Account for upper bound bin (last bin takes everything else)
        edges = np.asarray(bin_edges, dtype=np.int64)
        edges = np.concatenate(
            [edges, [np.iinfo(np.int64).max]]
        )  # Do this instead of np.inf to keep it in int

        # Bin the devices using np.digitize
        bin_idx = np.digitize(filtered_counts, edges, right=True) - 1

        for bin in range(edges.size - 1):
            bin_device_idxs = device_idxs[bin_idx == bin]
            if bin_device_idxs.size:  # Only care about non-empty bins
                self.device_group_map_list.append(bin_device_idxs)

    def _compute_discrete_terminal_groups(self):
        """
        This function makes a separate group for each set of devices with exactly the same number of terminals.
        For example, if we have 3 devices with 2 terminals and 4 devices with 3 terminals, we will have two groups:
        - Group 1: 3 devices with 2 terminals
        - Group 2: 4 devices with 3 terminals
        Importantly, assigns self.terminal_grupps and self.device_group_map_list
        """
        num_terminals_per_device_list = np.diff(self.G.indptr)

        # Tells you what are the distinct number of terminals a device could have (ignore devices with 0 terminals)
        filtered_counts = num_terminals_per_device_list[num_terminals_per_device_list > 0]
        self.terminal_groups = np.sort(np.unique(filtered_counts))

        # List of lists—each sublist contains the indices of devices with the same number of terminals
        self.device_group_map_list = [
            np.argwhere(num_terminals_per_device_list == g).flatten() for g in self.terminal_groups
        ]

    def _create_variable_devices(self):
        for group_idx, device_idxs in enumerate(self.device_group_map_list):
            num_devices = len(device_idxs)
            if num_devices == 0:
                continue

            A_sub = self.G[:, device_idxs]  # Still sparse representation
            nnz_per_col = np.diff(A_sub.indptr)
            # We don't pad to the bin edge, but to the max number of terminals in the bin group
            k_max = nnz_per_col.max()

            # A_v is a submatrix of A: (num (max) terminals i.e. k_max, num_devices)
            A_v = np.zeros((k_max, num_devices), dtype=A_sub.data.dtype)
            terms = -np.ones(
                (num_devices, k_max), dtype=np.int64
            )  # We use -1 for padding here (because 0 is an actual terminal)

            # Populate the padded matrix A_v column by column using sparse info from A_sub
            for j, col_idx in enumerate(range(num_devices)):
                start, end = A_sub.indptr[j : j + 2]
                k = end - start  # Number of terminals (non-zero entries) in this column
                if k == 0:
                    continue
                A_v[:k, j] = A_sub.data[start:end]
                terms[j, :k] = A_sub.indices[start:end]

            utility_weights = self.w[device_idxs]

            device = LogUtilityDevice(
                num_nodes=self.net.num_nodes,
                terminals=terms,
                A_v=A_v,
                cost_vector=utility_weights, # In this case, these are the scaling factors for the log utilities
            )
            self.devices.append(device)

    def _group_slack_devices(self):
        """
        In NUM there are only non-negative cone slack devices.
        """

        num_nonneg_cone = self.capacities.shape[0]
        self.slack_indices = np.arange(num_nonneg_cone)

        # Group nonneg cone slacks
        start_nonneg = 0
        end_nonneg = start_nonneg + num_nonneg_cone
        self.nonneg_cone_slacks = list(
            zip(self.slack_indices[start_nonneg:end_nonneg], self.capacities[start_nonneg:end_nonneg])
        )

    def _create_slack_devices(self):
        if self.nonneg_cone_slacks:
            terminals, b_d_values = zip(*self.nonneg_cone_slacks)
            nonneg_cone_device = NonNegativeConeSlackDevice(
                num_nodes=self.net.num_nodes,
                terminals=np.array(terminals),
                b_d=np.array(b_d_values), # In this case, these are the link capacities
            )
            self.devices.append(nonneg_cone_device)

    def solve(self, solver=cp.CLARABEL, **kwargs):
        return self.net.dispatch(
            self.devices, self.time_horizon, add_ground=False, solver=solver, **kwargs
        )
