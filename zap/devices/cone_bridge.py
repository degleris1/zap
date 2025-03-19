import zap.network
from zap.devices.block_variable_device import BlockVariableDevice
from zap.devices.slack_device import ZeroConeSlackDevice, NonNegativeConeSlackDevice
import numpy as np
import scipy.sparse as sp 



class ConeBridge:
    def __init__(self, cone_params: dict):
        self.cone_params = cone_params
        self.net = None
        self.time_horizon = 1
        self.devices = []
        self._transform()

    def _transform(self):
        self._build_network()
        self._group_variable_devices()
        self._create_block_variable_devices()
        self._group_slack_devices()
        self._create_slack_devices()

    def _build_network(self):
        A = self.cone_params["A"]
        self.net = zap.network.PowerNetwork(A.shape[0])

    def _group_variable_devices(self):
        """
        Figure out the appropriate grouping of variable devices based on the number of terminals they have
        """
        A = self.cone_params["A"]
        self.variable_device_groups = {}
        
        # Iterate over columns—each one is a variable device
        for j in range(A.shape[1]): 
            terminals = np.nonzero(A[:, j])[0]  # Nets connected to variable j
            num_terminals = len(terminals)
            
            if num_terminals not in self.variable_device_groups:
                self.variable_device_groups[num_terminals] = []
            
            # This is a dict where keys are num_terminals, and value is a list of tuples (which device/variable j, which terminal indices they connect to, and the actual non-zero subvector)
            self.variable_device_groups[num_terminals].append((j, terminals, A[terminals, j]))

    def _create_block_variable_devices(self):
        c = self.cone_params["c"]
        
        # Outer loop is over how many block devices we have
        for num_terminals, devices in self.variable_device_groups.items():
            A_bv_list = [] # block variable submatrix of A as a list
            cost_vector = []

            terminals_per_device = []

            # Inner loop is over how many variable devices in a block device group
            for j, terminals, A_subvec in devices:
                terminals_per_device.append(np.array(terminals))
                A_bv_list.append(A_subvec.toarray()) # Make dense so torch doesn't deal with sparse matrices
                cost_vector.append(c[j]) 

            A_bv_dense = np.hstack(A_bv_list)
            device = BlockVariableDevice(
                num_nodes=self.net.num_nodes,
                terminals=np.array(terminals_per_device),
                A_bv=A_bv_dense,
                cost_vector=np.array(cost_vector)
            )
            self.devices.append(device)

    def _group_slack_devices(self):
        """
        Currently assuming all zero cones before non-negative cones in CVXPY?? 
        Not actually sure that's right
        """
        self.zero_cone_slacks = []
        self.nonneg_cone_slacks = []

        b = self.cone_params["b"]
        num_zero_cone = self.cone_params["K"]["z"]
        num_nonneg_cone = self.cone_params["K"]["l"]

        for i in range(num_zero_cone):
            self.zero_cone_slacks.append((i, b[i]))

        for i in range(num_zero_cone, num_zero_cone + num_nonneg_cone):
            self.nonneg_cone_slacks.append((i, b[i]))
    
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

