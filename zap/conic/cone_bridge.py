import numpy as np
import cvxpy as cp

from zap.network import PowerNetwork
from .variable_device import VariableDevice
from .slack_device import ZeroConeSlackDevice, NonNegativeConeSlackDevice
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

        # Tells you what are the distinct number of terminals a device could have
        self.terminal_groups = np.sort(np.unique(num_terminals_per_device_list))

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
        # num_nonneg_cone = self.K["l"]
        slack_indices = np.arange(self.b.shape[0])

        self.zero_cone_slacks = list(zip(slack_indices[:num_zero_cone], self.b[:num_zero_cone]))
        self.nonneg_cone_slacks = list(zip(slack_indices[num_zero_cone:], self.b[num_zero_cone:]))

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

    def solve(self, solver=cp.CLARABEL, **kwargs):
        return self.net.dispatch(
            self.devices, self.time_horizon, add_ground=False, solver=solver, **kwargs
        )
