import numpy as np
import cvxpy as cp

from zap.network import PowerNetwork
from ..conic.variable_device import VariableDevice
from .utility_device import LogUtilityDevice
from ..conic.slack_device import NonNegativeConeSlackDevice
from scipy.sparse import csc_matrix, isspmatrix_csc


class NUOptBridge:
    def __init__(self, nu_opt_params: dict, grouping_params: dict | None = None):
        self.R = nu_opt_params["R"]
        self.capacities = nu_opt_params["capacities"]
        self.w = nu_opt_params["w"]
        self.lin_device_idxs = nu_opt_params.get("lin_device_idxs", None)
        self.G = self.R
        self.net = None
        self.time_horizon = 1
        self.devices = []

        if not isspmatrix_csc(self.R):
            self.R = csc_matrix(self.R)

        grouping_params = grouping_params or {}
        self.variable_grouping_strategy = grouping_params.get(
            "variable_grouping_strategy", "discrete_terminal_groups"
        )
        self.variable_grouping_bin_edges = grouping_params.get(
            "variable_grouping_bin_edges",
            (0, 10, 100, 1000),
        )
        self._transform()

    def _transform(self):
        self._build_network()
        self._group_variable_devices()
        # Create LinearUtilityDevices
        self._create_variable_devices(
            device_group_map_list=self.lin_device_group_map_list, device_type=VariableDevice
        )
        self._create_variable_devices(
            device_group_map_list=self.log_device_group_map_list, device_type=LogUtilityDevice
        )

        self._group_slack_devices()
        self._create_slack_devices()

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

        # Account for the fact that we might also have LinearUtilityDevices (these are just VariableDevices)
        lin_mask = np.zeros(len(num_terminals_per_device_list), dtype=bool)
        if self.lin_device_idxs is not None:
            lin_mask[self.lin_device_idxs] = True
            log_mask = ~lin_mask
        else:
            log_mask = np.ones(len(num_terminals_per_device_list), dtype=bool)

        def build(mask):
            terminal_groups = np.sort(np.unique(num_terminals_per_device_list[mask]))
            device_group_map_list = [
                np.argwhere(mask & (num_terminals_per_device_list == g)).flatten()
                for g in terminal_groups
            ]

            return terminal_groups, device_group_map_list

        (self.lin_terminal_groups, self.lin_device_group_map_list) = build(lin_mask)
        (self.log_terminal_groups, self.log_device_group_map_list) = build(log_mask)

    def _create_variable_devices(self, device_group_map_list, device_type):
        for group_idx, device_idxs in enumerate(device_group_map_list):
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

            if device_type is VariableDevice:
                utility_weights *= -1

            device = device_type(
                num_nodes=self.net.num_nodes,
                terminals=terms,
                A_v=A_v,
                cost_vector=utility_weights,  # In this case, these are the scaling factors for the log utilities
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
            zip(
                self.slack_indices[start_nonneg:end_nonneg],
                self.capacities[start_nonneg:end_nonneg],
            )
        )

    def _create_slack_devices(self):
        if self.nonneg_cone_slacks:
            terminals, b_d_values = zip(*self.nonneg_cone_slacks)
            nonneg_cone_device = NonNegativeConeSlackDevice(
                num_nodes=self.net.num_nodes,
                terminals=np.array(terminals),
                b_d=np.array(b_d_values),  # In this case, these are the link capacities
            )
            self.devices.append(nonneg_cone_device)

    def solve(self, solver=cp.CLARABEL, **kwargs):
        return self.net.dispatch(
            self.devices, self.time_horizon, add_ground=False, solver=solver, **kwargs
        )
