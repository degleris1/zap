import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from typing import List
from zap.admm.layer import ADMMLayer
from zap.admm.basic_solver import ADMMSolver, ADMMState
from zap.conic.cone_bridge import ConeBridge
from zap.conic.cone_utils import get_standard_conic_problem, map_cvxpy_parameters_to_cone_program


class ConicADMMLayer(ADMMLayer):
    """Wrapper layer for conic problems to use ADMM Layer."""
    
    def __init__(
        self,
        problem: cp.Problem,
        parameters: List[cp.Parameter],
        solver: ADMMSolver = ADMMSolver(num_iterations=100, rho_power=1.0),
        ruiz_iters: int = 0,
        warm_start: bool = True,
        adapt_rho: bool = False,
        adapt_rho_rate: float = 0.1,
        verbose: bool = False,
    ):
        cone_params, data, cones = get_standard_conic_problem(problem, solver=cp.CLARABEL)
        cone_bridge = ConeBridge(cone_params, ruiz_iters=ruiz_iters)
        cvxpy_param_mappings = map_cvxpy_parameters_to_cone_program(problem, parameters)
        
        # Convert parameter mappings to device attribute mappings
        parameter_names = self._build_parameter_names(cone_bridge, cvxpy_param_mappings, parameters)
        
        device_machine = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float32
        cone_admm_devices = [d.torchify(machine=device_machine, dtype=dtype) for d in cone_bridge.devices]
        
        super().__init__(
            network=cone_bridge.net,
            devices=cone_admm_devices,
            parameter_names=parameter_names,
            time_horizon=1,
            solver=solver,
            warm_start=warm_start,
            adapt_rho=adapt_rho,
            adapt_rho_rate=adapt_rho_rate,
            verbose=verbose
        )
        
        self.problem = problem
        self.parameters = parameters
        self.param_mappings = param_mappings

    def setup_parameters(self, **kwargs):
        """ 
        For conic parameters, we can modify specific indices of device attributes.
        Example of what parameter_names should look like in conic case:
        parameter_names = {
            "cap_var": (2, "b_d"),                # Full attribute parameterization
            "partial_cap": (2, "b_d", [0, 3, 5])  # Partial (index-based) parameterization
        }
        """
        # Check that arguments match parameters
        assert kwargs.keys() == self.parameter_names.keys()

        # Match parameters to devices
        parameters = [{} for _ in self.devices]
        for param_name, param_info in self.parameter_names.items():
            device_idx, attr_name, indices = param_info if len(param_info) == 3 else (*param_info, None)
        
            if indices is None:
                # Full attribute parameterization
                parameters[device_idx][attr_name] = kwargs[param_name]
            else:
                # Index-based parameterization
                parameters[device_idx][attr_name] = (indices, kwargs[param_name])
        
        return parameters
        
    def _build_parameter_names(self, cone_bridge, param_mappings, parameters):
        """
        Build parameter_names dictionary that maps CVXPY parameter names
        to device attributes.
        """
        parameter_names = {}
        
        for param in parameters:
            param_name = param.name()
            mapping = param_mappings[param_name]

            b_mappings = []
            if mapping['b']: # This parameter affects the conic parameter "b"
                b_mappings = self._map_b_to_devices(cone_bridge, mapping['b'])
            
            c_mappings = []
            if mapping['c']:
                c_mappings = self._map_c_to_devices(cone_bridge, mapping['c'])
            
            device_idx, attr_name = self._map_param_to_device(cone_bridge, mapping)
            
            parameter_names[param_name] = (device_idx, attr_name)
            
        return parameter_names

    def _map_b_to_devices(self, cone_bridge, b_indices):
        """
        Map b vector indices to device attributes.
        """
        mappings = []
        
        # Get cone dimensions
        num_zero_cone = cone_bridge.K["z"]
        num_nonneg_cone = cone_bridge.K["l"]
        
        # Find which device index each of these correspond to
        zero_device_idx = None
        nonneg_device_idx = None
        soc_device_indices = []
        
        for i, device in enumerate(cone_bridge.devices):
            device_type = type(device).__name__
            if device_type == 'ZeroConeSlackDevice':
                zero_device_idx = i
            elif device_type == 'NonNegativeConeSlackDevice':
                nonneg_device_idx = i
            elif device_type == 'SecondOrderConeSlackDevice':
                soc_device_indices.append(i)
        
        # Check zero cone indices
        zero_indices = [i for i in b_indices if i < num_zero_cone]
        if zero_indices and zero_device_idx is not None:
            mappings.append((zero_device_idx, 'b_d', zero_indices))
        
        # Check non-negative cone indices
        nonneg_start = num_zero_cone
        nonneg_end = nonneg_start + num_nonneg_cone
        nonneg_indices = [i - nonneg_start for i in b_indices if nonneg_start <= i < nonneg_end]
        if nonneg_indices and nonneg_device_idx is not None:
            mappings.append((nonneg_device_idx, 'b_d', nonneg_indices))
        
        # Check SOC cone indices
        if soc_device_indices:
            # SOC cones start after zero and nonneg cones
            soc_start = nonneg_end
            
            # Get the SOC cone sizes
            soc_sizes = cone_bridge.K.get("q", [])
            
            # Calculate starting indices for each SOC cone
            soc_cone_starts = np.cumsum([0] + soc_sizes[:-1]) + soc_start
            
            # Group SOC indices by which cone they belong to
            for soc_idx, (cone_start, cone_size) in enumerate(zip(soc_cone_starts, soc_sizes)):
                cone_end = cone_start + cone_size
                cone_indices = [i for i in b_indices if cone_start <= i < cone_end]
                
                if cone_indices:
                    # Find which SOC device contains this cone
                    # This is a simplification - in practice, we would need to examine
                    # each SOC device's terminals to determine the correct mapping
                    device_idx = soc_device_indices[0]  # Use first SOC device for now
                    
                    # Adjust indices relative to start of SOC section
                    local_indices = [i - cone_start for i in cone_indices]
                    
                    mappings.append((device_idx, 'b_d', local_indices))
        
        return mappings

    def _map_c_to_devices(self, cone_bridge, c_indices):
        """
        Map c vector indices to device attributes.
        """
        mappings = []
        var_device_indices = []
        for i, device in enumerate(cone_bridge.devices):
            if type(device).__name__ == 'VariableDevice':
                var_device_indices.append(i)
        
        # Which idx variable device are you, which idx device are you (always the same in current implementation of cone_bridge)
        for var_idx, device_idx in enumerate(var_device_indices):
        # Get the original c indices for this VariableDevice.
        # For example (in this case contiguous but not necessarily)
        # cone_bridge.device_group_map_list
        # [array([0, 1, 2, 3, 4, 5, 6, 7]), array([8, 9])]
            original_indices = cone_bridge.device_group_map_list[var_idx]

            # Create a mapping from global c index to local index within this device
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(original_indices)}


            # Find c indices that would map to this specific device (translated to local index)
            group_c_indices = [global_to_local[i] for i in c_indices if i in global_to_local]

            if group_c_indices:
                mappings.append((device_idx, 'cost_vector', group_c_indices))
                

        return mappings
    


        
    def forward(self, *parameter_values, initial_state=None):
        """
        Forward pass that's more like CvxpyLayer interface.
        """
        # Map the ordered parameter values to the named parameters
        kwargs = {param.name(): val for param, val in zip(self.parameters, parameter_values)}
        
        # Call the parent class's forward method
        state = super().forward(initial_state=initial_state, **kwargs)
        
        # Extract objective value for convenience
        objective = compute_objective(state, self.devices, self.setup_parameters(**kwargs))
        
        return state, objective