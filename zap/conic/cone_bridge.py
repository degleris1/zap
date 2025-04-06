import numpy as np
import cvxpy as cp

from zap.network import PowerNetwork
from .variable_device import VariableDevice
from .slack_device import (
    ZeroConeSlackDevice,
    NonNegativeConeSlackDevice,
    SecondOrderConeSlackDevice,
)
from scipy.sparse import csc_matrix, isspmatrix_csc


class ConeBridge:
    def __init__(self, cone_params: dict):
        self.A = cone_params["A"]
        self.b = cone_params["b"]
        self.c = cone_params["c"]
        self.K = cone_params["K"]
        self.net = None
        self.time_horizon = 1
        self.devices = []

        if not isspmatrix_csc(self.A):
            self.A = csc_matrix(self.A)
        self._transform()

    def _transform(self):
        self._build_network()
        self._group_variable_devices()
        self._create_variable_devices()
        self._group_slack_devices()
        self._create_slack_devices()

    def _build_network(self):
        self.net = PowerNetwork(self.A.shape[0])

    def _group_variable_devices(self):
        """
        Figure out the appropriate grouping of variable devices based on the number of terminals they have
        """
        ## TODO: Expand this to other potential grouping strategies

        num_terminals_per_device_list = np.diff(self.A.indptr)

        # Tells you what are the distinct number of terminals a device could have (ignore devices with 0 terminals)
        filtered_counts = num_terminals_per_device_list[num_terminals_per_device_list > 0]
        self.terminal_groups = np.sort(np.unique(filtered_counts))

        # List of lists—each sublist contains the indices of devices with the same number of terminals
        self.device_group_map_list = [
            np.argwhere(num_terminals_per_device_list == g).flatten() for g in self.terminal_groups
        ]

    def _create_variable_devices(self):
        for group_idx, num_terminals_per_device in enumerate(self.terminal_groups):
            # Retrieve relevant columns of A
            device_idxs = self.device_group_map_list[group_idx]
            num_devices = len(device_idxs)

            A_devices = self.A[:, device_idxs]

            # (i) A_v is a submatrix of A: (num_terminals, num_devices)
            A_v = A_devices.data.reshape((num_devices, num_terminals_per_device)).T

            # (ii) terminal_device_array: (num_devices, num_terminals_per_device)
            terminal_device_array = A_devices.indices.reshape(
                (num_devices, num_terminals_per_device)
            )

            # (iii) cost vector (subvector of c taking the corresponding device elements)
            cost_vector = self.c[device_idxs]

            device = VariableDevice(
                num_nodes=self.net.num_nodes,
                terminals=terminal_device_array,
                A_v=A_v,
                cost_vector=cost_vector,
            )
            self.devices.append(device)

    def _group_slack_devices(self):
        """
        Currently assuming all zero cones before non-negative cones in CVXPY
        """

        num_zero_cone = self.K["z"]
        num_nonneg_cone = self.K["l"]
        # This is like self.terminal_groups for variable devices
        self.soc_terminal_groups = np.sort(np.unique(self.K["q"]))
        self.soc_blocks = self.K["q"]
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
        # We are creating a dict like {block_size: [(start, end), ...]}
        # where each entry corresponds to a block of SOC slacks,
        # and each tuple in the list is for a block (device) of that size
        self.soc_block_idxs_dict = {group_size: [] for group_size in self.soc_terminal_groups}
        soc_start = end_nonneg
        for block_size in self.soc_blocks:
            start = soc_start
            end = soc_start + block_size
            self.soc_block_idxs_dict[block_size].append((start, end))
            soc_start += block_size

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
        for group_idx, num_terminals_per_device in enumerate(self.soc_terminal_groups):
            group_slices = self.soc_block_idxs_dict[num_terminals_per_device]
            b_d_array = np.column_stack([self.b[start:end] for (start, end) in group_slices])
            terminal_device_array = np.row_stack(
                [self.slack_indices[start:end] for (start, end) in group_slices]
            )
            soc_cone_device = SecondOrderConeSlackDevice(
                num_nodes=self.net.num_nodes,
                terminals=terminal_device_array,
                b_d=b_d_array,
            )
            self.devices.append(soc_cone_device)

    def solve(self, solver=cp.CLARABEL, **kwargs):
        return self.net.dispatch(
            self.devices, self.time_horizon, add_ground=False, solver=solver, **kwargs
        )
