import numpy as np
import cvxpy as cp
import scipy.sparse as sp
import sksparse.cholmod as cholmod

from zap.network import PowerNetwork
from .variable_device import VariableDevice
from .slack_device import (
    ZeroConeSlackDevice,
    NonNegativeConeSlackDevice,
    SecondOrderConeSlackDevice,
)
from .quadratic_device import QuadraticDevice
from scipy.sparse import csc_matrix, isspmatrix_csc, vstack
from zap.conic.cone_utils import (
    build_symmetric_M,
    scale_cols_csc,
    scale_rows_csr,
)
from copy import deepcopy


class ConeBridge:
    def __init__(self, cone_params: dict, grouping_params: dict | None = None, ruiz_iters: int = 5):
        self.P = cone_params["P"]
        self.A = cone_params["A"]
        self.b = cone_params["b"]
        self.c = cone_params["c"]
        self.K = cone_params["K"]
        self.G = self.A
        self.net = None
        self.time_horizon = 1
        self.devices = []

        self.ruiz_iters = ruiz_iters
        self.D_vec = None
        self.E_vec = None
        self.sigma = 1

        if not isspmatrix_csc(self.A):
            self.A = csc_matrix(self.A)

        if self.P is not None and not isspmatrix_csc(self.P):
            self.P = csc_matrix(self.P)

        grouping_params = grouping_params or {}
        self.variable_grouping_strategy = grouping_params.get(
            "variable_grouping_strategy", "binned_terminal_groups"
        )
        self.slack_grouping_strategy = grouping_params.get(
            "slack_grouping_strategy", "binned_soc_groups"
        )
        self.variable_grouping_bin_edges = grouping_params.get(
            "variable_grouping_bin_edges", (0, 10, 100, 1000)
        )
        self.slack_grouping_bin_edges = grouping_params.get(
            "slack_grouping_bin_edges", (0, 10, 100, 1000)
        )
        self._transform()

    def _transform(self):
        if self.ruiz_iters > 0:
            self._equilibrate_ruiz()
        self._factorize_P()
        self._build_network()
        self._group_variable_devices()
        self._create_variable_devices()
        self._group_slack_devices()
        self._create_slack_devices()
        self._create_quadratic_devices()

    def _factorize_P(self):
        """
        Sparse LDLT factorization of P.
        First permutes P by a permutation matrix Q and then computes LDL.T.
        Also returns permutation of A, c, and forms the stacked network matrix G = [A;B]
        """
        if self.P is None:
            self.Q = None
            self.G = self.A
            return

        # Perform sparse LDLT factorization after adding epsilon to ensure positive definiteness
        n = self.P.shape[0]
        factor = cholmod.cholesky(
            self.P + 1e-12 * sp.eye(n, format="csc"), mode="simplicial", ordering_method="amd"
        )
        self.Q = factor.P()
        L, D = factor.L_D()

        # Permute A and c
        self.A = self.A[:, self.Q]  # A_perm = AQ.T
        self.c = self.c[self.Q]  # c_perm = Qc

        # Construct B
        sqrtD = np.sqrt(D.diagonal())
        L_T_csr = L.T.tocsr()
        B = scale_rows_csr(L_T_csr, sqrtD)
        nz_rows = np.nonzero(sqrtD > 0)[0]
        self.B = (B[nz_rows, :]).tocsc()

        # Construct G
        self.G = vstack([self.A, -1 * self.B], format="csc")

    def _equilibrate_ruiz(self):
        """
        Follows Ruiz Equilibration from Clarabel and OSQP.
        Builds the symmetric matrix M = [[P, A.T], [A, 0]]
        """
        P = deepcopy(self.P)
        A = deepcopy(self.A)
        c = deepcopy(self.c)
        b = deepcopy(self.b)

        m, n = A.shape
        self.D_vec = np.ones(m)
        self.E_vec = np.ones(n)
        self.sigma = 1
        num_zero_cone = self.K["z"]
        num_nonneg_cone = self.K["l"]
        soc_sizes = self.K.get("q", [])
        soc_starts = np.cumsum([0] + soc_sizes[:-1]) + (num_zero_cone + num_nonneg_cone)

        for i in range(self.ruiz_iters):
            # Compute scale factors
            M = build_symmetric_M(A_csc=A, P=P)
            col_inf_norms = np.abs(M).max(axis=0).toarray().ravel()
            delta = np.ones_like(col_inf_norms)
            delta[col_inf_norms > 0] = 1 / np.sqrt(col_inf_norms[col_inf_norms > 0])
            delta_cols = delta[:n]
            delta_rows = delta[n:]

            # Column scaling as part of M equilibration
            c = delta_cols * c
            self.E_vec = self.E_vec * delta_cols
            A = scale_cols_csc(A, delta_cols)

            # Row scaling as part of M equilibration
            b = delta_rows * b
            self.D_vec = self.D_vec * delta_rows
            A_csr = A.tocsr()
            A_csr = scale_rows_csr(A_csr, delta_rows)

            # Preserve cone membership
            for s, sz in zip(soc_starts, soc_sizes):
                block_idxs = slice(s, s + sz)
                g = np.exp(np.mean(np.log(self.D_vec[block_idxs])))
                correction_factors = np.ones(m)
                correction_factors[block_idxs] = g / self.D_vec[block_idxs]

                if not np.allclose(correction_factors, 1):
                    A_csr = scale_rows_csr(A_csr, correction_factors)
                    b = b * correction_factors
                    self.D_vec[block_idxs] = g

            A = A_csr.tocsc()

            # Scaling factor sigma and P updates
            if P is not None:
                P_csr = scale_rows_csr(P.tocsr(), delta_cols)
                P = scale_cols_csc(P_csr.tocsc(), delta_cols)
                P_inf_norm_mean = np.abs(P).max(axis=0).mean()

            else:
                P_inf_norm_mean = -np.inf

            c_inf_norm = np.max(np.abs(c))
            sigma_step = 1 / max(c_inf_norm, P_inf_norm_mean)
            proposed_sigma = self.sigma * sigma_step

            if proposed_sigma <= 1e-2:
                sigma_step = 1.0
            else:
                proposed_sigma = np.clip(proposed_sigma, None, 1e4)
                sigma_step = proposed_sigma / self.sigma

            self.sigma *= sigma_step
            c = c * sigma_step
            if P is not None:
                P = P * sigma_step

        # Update the cone parameters
        self.P = P
        self.A = A
        self.b = b
        self.c = c

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

            cost_vec = self.c[device_idxs]

            device = VariableDevice(
                num_nodes=self.net.num_nodes,
                terminals=terms,
                A_v=A_v,
                cost_vector=cost_vec,
            )
            self.devices.append(device)

    def _group_slack_devices(self):
        """
        Currently assuming all zero cones before non-negative cones in CVXPY
        """

        num_zero_cone = self.K["z"]
        num_nonneg_cone = self.K["l"]
        self.slack_indices = np.arange(self.b.shape[0])

        # Group zero cone slacks
        self.zero_cone_slacks = list(
            zip(self.slack_indices[:num_zero_cone], self.b[:num_zero_cone])
        )

        # Group nonneg cone slacks
        start_nonneg = num_zero_cone
        end_nonneg = start_nonneg + num_nonneg_cone
        self.nonneg_cone_slacks = list(
            zip(self.slack_indices[start_nonneg:end_nonneg], self.b[start_nonneg:end_nonneg])
        )

        # Group SOC cone slacks
        if self.slack_grouping_strategy == "discrete_soc_groups":
            self._compute_discrete_soc_groups(soc_start=end_nonneg)
        elif self.slack_grouping_strategy == "binned_soc_groups":
            self._compute_binned_soc_groups(
                bin_edges=self.slack_grouping_bin_edges, soc_start=end_nonneg
            )

    def _compute_binned_soc_groups(self, bin_edges, soc_start):
        """
        This function makes a separate group for each set of SOC slacks in the same bin.
        """
        self.soc_blocks = self.K["q"]
        start_idx = soc_start
        # Account for upper bound bin (last bin takes everything else)
        edges = np.asarray(bin_edges, dtype=np.int64)
        edges = np.concatenate(
            [edges, [np.iinfo(np.int64).max]]
        )  # Do this instead of np.inf to keep it in int

        soc_bins = [[] for _ in range(len(edges) - 1)]

        # Loop through the blocks of SOC slacks
        # and assign them to the appropriate bin
        # We are interested in saving the start and end indices of the blocks,
        # as well as the size of the block
        # (which is the number of terminals in the block)
        for k in self.soc_blocks:
            start, end = start_idx, start_idx + k

            # Bin assignment
            bin_idx = np.digitize(k, edges, right=True) - 1
            soc_bins[bin_idx].append((start, end, k))
            start_idx = end

        # Remove empty bins
        self.soc_device_group_map_list = [bin_block for bin_block in soc_bins if bin_block]

    def _compute_discrete_soc_groups(self, soc_start):
        """
        This function makes a separate device for each block of SOC slacks.
        We are creating a dict like {block_size: [(start, end), ...]}
        where each entry corresponds to a block of SOC slacks,
        and each tuple in the list is for a block (device) of that size.
        We convert this into the list of lists of tuples for compatability
        with the binning approach.
        """
        self.soc_device_group_map_list = []

        # This is like self.terminal_groups for variable devices
        self.soc_terminal_groups = np.sort(np.unique(self.K["q"]))
        self.soc_blocks = self.K["q"]
        self.soc_block_idxs_dict = {group_size: [] for group_size in self.soc_terminal_groups}
        for block_size in self.soc_blocks:
            start = soc_start
            end = soc_start + block_size
            self.soc_block_idxs_dict[block_size].append((start, end, block_size))
            soc_start += block_size
        self.soc_device_group_map_list = [
            self.soc_block_idxs_dict[block_size] for block_size in sorted(self.soc_block_idxs_dict)
        ]

    def _create_slack_devices(self):
        if self.zero_cone_slacks:
            terminals, b_d_values = zip(*self.zero_cone_slacks)
            zero_cone_device = ZeroConeSlackDevice(
                num_nodes=self.net.num_nodes,
                terminals=np.array(terminals),
                b_d=np.array(b_d_values),
            )
            self.devices.append(zero_cone_device)

        if self.nonneg_cone_slacks:
            terminals, b_d_values = zip(*self.nonneg_cone_slacks)
            nonneg_cone_device = NonNegativeConeSlackDevice(
                num_nodes=self.net.num_nodes,
                terminals=np.array(terminals),
                b_d=np.array(b_d_values),
            )
            self.devices.append(nonneg_cone_device)

        # Create SOC devices
        for bin_blocks in self.soc_device_group_map_list:
            k_max = max(k for _, _, k in bin_blocks)
            num_devices = len(bin_blocks)

            b_d_array = np.zeros((k_max, num_devices), dtype=self.b.dtype)
            terminals = -np.ones((num_devices, k_max), dtype=np.int64)
            for bin_idx, (start, end, k) in enumerate(bin_blocks):
                b_d_array[:k, bin_idx] = self.b[start:end]
                terminals[bin_idx, :k] = self.slack_indices[start:end]

            soc_cone_device = SecondOrderConeSlackDevice(
                num_nodes=self.net.num_nodes,
                terminals=terminals,
                b_d=b_d_array,
                terminals_per_device=np.array([k for _, _, k in bin_blocks]),
            )
            self.devices.append(soc_cone_device)

    def _create_quadratic_devices(self):
        if self.P is None:
            return
        self.quadratic_indices = np.arange(len(self.b), self.G.shape[0])
        quadratic_device = QuadraticDevice(
            num_nodes=self.net.num_nodes,
            terminals=np.array(self.quadratic_indices),
        )

        self.devices.append(quadratic_device)

    def solve(self, solver=cp.CLARABEL, **kwargs):
        return self.net.dispatch(
            self.devices, self.time_horizon, add_ground=False, solver=solver, **kwargs
        )
