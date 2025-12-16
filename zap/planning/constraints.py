"""Budget constraints for the planning module.

This module provides support for arbitrary linear budget constraints over
upper-level decision variables in the gradient-based planning algorithm.

The key idea is to replace the box-projection step with a QP-based
projection that handles both box constraints and budget constraints:

    min   (1/2) ||η - η⁺||²
    s.t.  η_min ≤ η ≤ η_max     (box constraints)
          A_le @ η ≤ b_le       (≤ budget constraints)
          A_ge @ η ≥ b_ge       (≥ budget constraints)
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


@dataclass
class BudgetConstraint:
    """A single budget constraint: sum(coefficients * variables) <= or >= rhs.

    Attributes:
        name: Identifier for this constraint
        coefficients: Maps param_name -> {array_index: multiplier}
        rhs: Right-hand side value of the constraint
        sense: "le" for ≤ constraint, "ge" for ≥ constraint
    """

    name: str
    coefficients: dict[str, dict[int, float]] = field(default_factory=dict)
    rhs: float = 0.0
    sense: str = "le"  # "le" (≤) or "ge" (≥)

    def __post_init__(self):
        if self.sense not in ("le", "ge"):
            raise ValueError(f"sense must be 'le' or 'ge', got '{self.sense}'")


class BudgetConstraintSet:
    """Collection of budget constraints with methods to build A, b matrices.

    This class parses constraint specifications from CSV files and builds
    the sparse constraint matrices needed for the projection QP.

    Attributes:
        constraints: List of BudgetConstraint objects
        device_mapping: Maps (device_name, attribute) -> (param_name, array_index)
    """

    def __init__(
        self,
        constraints: list[BudgetConstraint],
        parameter_names: dict[str, tuple[int, str]],
        devices: list,
    ):
        self.constraints = constraints
        self.parameter_names = parameter_names
        self.devices = devices
        self.device_mapping = _build_device_name_mapping(devices, parameter_names)

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        parameter_names: dict[str, tuple[int, str]],
        devices: list,
    ) -> "BudgetConstraintSet":
        """Parse a CSV file into a BudgetConstraintSet.

        CSV Format:
            constraint_name,attribute,device_name,multiplier,rhs_value,sense
            max_wind,nominal_capacity,node_01_wind,1,,
            max_wind,nominal_capacity,node_02_wind,1,,
            max_wind,rhs,,,100,le
            min_solar,nominal_capacity,solar_01,1,,
            min_solar,rhs,,,50,ge

        - constraint_name: Groups rows into a single constraint
        - attribute: Device attribute name (e.g., "nominal_capacity") or "rhs" for the bound
        - device_name: Name of the device (matches device.name)
        - multiplier: Coefficient for this device in the constraint
        - rhs_value: Only used when attribute=="rhs", defines the RHS bound
        - sense: Only used when attribute=="rhs", either "le" (≤) or "ge" (≥). Defaults to "le".

        Args:
            csv_path: Path to the CSV file
            parameter_names: Maps param_name -> (device_index, attribute_name)
            devices: List of device objects

        Returns:
            BudgetConstraintSet with parsed constraints
        """
        df = pd.read_csv(csv_path)

        # Build device name mapping
        device_mapping = _build_device_name_mapping(devices, parameter_names)

        constraints = []
        for constraint_name, group in df.groupby("constraint_name"):
            # Extract RHS row
            rhs_rows = group[group["attribute"] == "rhs"]
            if len(rhs_rows) == 0:
                raise ValueError(f"Constraint '{constraint_name}' is missing an 'rhs' row")
            if len(rhs_rows) > 1:
                raise ValueError(f"Constraint '{constraint_name}' has multiple 'rhs' rows")

            rhs_row = rhs_rows.iloc[0]
            rhs = float(rhs_row["rhs_value"])

            # Get sense (default to "le")
            if "sense" in rhs_row and pd.notna(rhs_row["sense"]):
                sense = str(rhs_row["sense"]).lower()
            else:
                sense = "le"

            # Extract coefficient rows
            coef_rows = group[group["attribute"] != "rhs"]

            coefficients: dict[str, dict[int, float]] = {}
            for _, row in coef_rows.iterrows():
                device_name = row["device_name"]
                attribute = row["attribute"]
                multiplier = float(row["multiplier"])

                # Look up which parameter and index this device maps to
                key = (device_name, attribute)
                if key not in device_mapping:
                    available = [k for k in device_mapping.keys() if k[1] == attribute]
                    raise ValueError(
                        f"Device '{device_name}' with attribute '{attribute}' not found. "
                        f"Available devices for this attribute: {[k[0] for k in available]}"
                    )

                param_name, array_index = device_mapping[key]

                if param_name not in coefficients:
                    coefficients[param_name] = {}
                coefficients[param_name][array_index] = multiplier

            constraints.append(
                BudgetConstraint(
                    name=str(constraint_name),
                    coefficients=coefficients,
                    rhs=rhs,
                    sense=sense,
                )
            )

        return cls(constraints, parameter_names, devices)

    def build_constraint_matrices(
        self,
        param_sizes: dict[str, int],
        param_offsets: dict[str, int],
        total_dim: int,
    ) -> tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix, np.ndarray]:
        """Build sparse constraint matrices A_le, b_le, A_ge, b_ge.

        Args:
            param_sizes: Maps param_name -> size of parameter array
            param_offsets: Maps param_name -> offset in flattened vector
            total_dim: Total dimension of flattened state vector

        Returns:
            A_le: Sparse matrix for ≤ constraints (shape: num_le_constraints x total_dim)
            b_le: RHS vector for ≤ constraints
            A_ge: Sparse matrix for ≥ constraints
            b_ge: RHS vector for ≥ constraints
        """
        le_constraints = [c for c in self.constraints if c.sense == "le"]
        ge_constraints = [c for c in self.constraints if c.sense == "ge"]

        A_le, b_le = self._build_matrices(le_constraints, param_sizes, param_offsets, total_dim)
        A_ge, b_ge = self._build_matrices(ge_constraints, param_sizes, param_offsets, total_dim)

        return A_le, b_le, A_ge, b_ge

    def _build_matrices(
        self,
        constraints: list[BudgetConstraint],
        _param_sizes: dict[str, int],
        param_offsets: dict[str, int],
        total_dim: int,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Build A matrix and b vector for a list of constraints."""
        if len(constraints) == 0:
            return sp.csr_matrix((0, total_dim)), np.array([])

        rows = []
        cols = []
        data = []
        b = []

        for i, constraint in enumerate(constraints):
            for param_name, idx_to_coef in constraint.coefficients.items():
                offset = param_offsets[param_name]
                for array_idx, coef in idx_to_coef.items():
                    rows.append(i)
                    cols.append(offset + array_idx)
                    data.append(coef)
            b.append(constraint.rhs)

        A = sp.csr_matrix(
            (data, (rows, cols)), shape=(len(constraints), total_dim), dtype=np.float64
        )
        return A, np.array(b, dtype=np.float64)

    def __len__(self):
        return len(self.constraints)


class ProjectionQP:
    """Vectorized CVXPY problem for projecting onto box + budget constraints.

    This class pre-compiles a parameterized QP problem that can be efficiently
    solved at each iteration of the planning algorithm to project the gradient
    descent update onto the feasible region.

    The QP solved is:
        min   (1/2) ||η - η⁺||²
        s.t.  η_min ≤ η ≤ η_max
              A_le @ η ≤ b_le
              A_ge @ η ≥ b_ge

    Attributes:
        parameter_names: Maps param_name -> (device_index, attribute_name)
        lower_bounds: Box constraint lower bounds
        upper_bounds: Box constraint upper bounds
        budget_constraints: Optional BudgetConstraintSet
    """

    def __init__(
        self,
        parameter_names: dict[str, tuple[int, str]],
        lower_bounds: dict[str, np.ndarray],
        upper_bounds: dict[str, np.ndarray],
        budget_constraints: Optional[BudgetConstraintSet] = None,
    ):
        self.parameter_names = parameter_names
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.budget_constraints = budget_constraints

        self._compile_problem()

    def _compile_problem(self):
        """Pre-compile the parameterized CVXPY problem."""
        # Compute flattened dimensions
        self.param_sizes: dict[str, int] = {}
        self.param_offsets: dict[str, int] = {}
        offset = 0

        for param_name in self.parameter_names.keys():
            lb = self.lower_bounds[param_name]
            size = lb.size if hasattr(lb, "size") else np.array(lb).size
            self.param_sizes[param_name] = size
            self.param_offsets[param_name] = offset
            offset += size

        self.total_dim = offset

        if self.total_dim == 0:
            self.problem = None
            return

        # CVXPY variable
        self.eta = cp.Variable(self.total_dim)

        # CVXPY parameters
        self.eta_target = cp.Parameter(self.total_dim)
        self.lb_param = cp.Parameter(self.total_dim)
        self.ub_param = cp.Parameter(self.total_dim)

        # Objective: minimize (1/2) ||η - η⁺||²
        objective = cp.Minimize(0.5 * cp.sum_squares(self.eta - self.eta_target))

        # Box constraints
        constraints = [
            self.eta >= self.lb_param,
            self.eta <= self.ub_param,
        ]

        # Budget constraints
        self.has_le_constraints = False
        self.has_ge_constraints = False

        if self.budget_constraints is not None and len(self.budget_constraints) > 0:
            A_le, b_le, A_ge, b_ge = self.budget_constraints.build_constraint_matrices(
                self.param_sizes, self.param_offsets, self.total_dim
            )

            if A_le.shape[0] > 0:
                self.has_le_constraints = True
                self.A_le_data = A_le
                self.b_le_data = b_le
                # Store as dense for CVXPY parameter (sparse parameter handling is tricky)
                self.A_le = cp.Parameter((A_le.shape[0], self.total_dim))
                self.b_le = cp.Parameter(A_le.shape[0])
                constraints.append(self.A_le @ self.eta <= self.b_le)

            if A_ge.shape[0] > 0:
                self.has_ge_constraints = True
                self.A_ge_data = A_ge
                self.b_ge_data = b_ge
                self.A_ge = cp.Parameter((A_ge.shape[0], self.total_dim))
                self.b_ge = cp.Parameter(A_ge.shape[0])
                constraints.append(self.A_ge @ self.eta >= self.b_ge)

        self.problem = cp.Problem(objective, constraints)

    def _flatten(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten state dict into a single vector."""
        result = np.zeros(self.total_dim)
        for param_name in self.parameter_names.keys():
            offset = self.param_offsets[param_name]
            size = self.param_sizes[param_name]
            result[offset : offset + size] = np.asarray(state[param_name]).ravel()
        return result

    def _flatten_bounds(
        self, bounds: dict[str, np.ndarray], large_value: float = 1e12
    ) -> np.ndarray:
        """Flatten bounds dict into a single vector.

        Args:
            bounds: Dict mapping param_name -> bound array
            large_value: Replace inf/-inf with this value for CVXPY compatibility

        Returns:
            Flattened bounds vector with finite values only
        """
        result = np.zeros(self.total_dim)
        for param_name in self.parameter_names.keys():
            offset = self.param_offsets[param_name]
            size = self.param_sizes[param_name]
            bound_val = bounds[param_name]
            # Handle scalar bounds
            if np.isscalar(bound_val):
                val = bound_val
                if np.isinf(val):
                    val = large_value if val > 0 else -large_value
                result[offset : offset + size] = val
            else:
                arr = np.asarray(bound_val).ravel().copy()
                # Replace infinities with large finite values
                arr[np.isposinf(arr)] = large_value
                arr[np.isneginf(arr)] = -large_value
                result[offset : offset + size] = arr
        return result

    def _unflatten(self, flat: np.ndarray, state_template: dict) -> dict[str, np.ndarray]:
        """Unflatten a vector back into a state dict."""
        result = {}
        for param_name in self.parameter_names.keys():
            offset = self.param_offsets[param_name]
            size = self.param_sizes[param_name]
            # Preserve original shape
            original_shape = np.asarray(state_template[param_name]).shape
            result[param_name] = flat[offset : offset + size].reshape(original_shape)
        return result

    def project(self, state: dict, la=np) -> dict:
        """Project state onto the feasible set.

        Args:
            state: Dict mapping param_name -> array (numpy or torch)
            la: Linear algebra module (np or torch)

        Returns:
            Projected state dict with same types as input
        """
        if self.problem is None:
            return state

        # Check if using torch
        is_torch = la == torch or any(isinstance(v, torch.Tensor) for v in state.values())

        # Convert torch -> numpy for CVXPY
        if is_torch:
            device = None
            dtype = None
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    device = v.device
                    dtype = v.dtype
                    break
            state_np = {
                k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in state.items()
            }
        else:
            state_np = state
            device = None
            dtype = None

        # Flatten state
        eta_target = self._flatten(state_np)

        # Set CVXPY parameter values
        self.eta_target.value = eta_target
        self.lb_param.value = self._flatten_bounds(self.lower_bounds)
        self.ub_param.value = self._flatten_bounds(self.upper_bounds)

        if self.has_le_constraints:
            self.A_le.value = self.A_le_data.toarray()
            self.b_le.value = self.b_le_data

        if self.has_ge_constraints:
            self.A_ge.value = self.A_ge_data.toarray()
            self.b_ge.value = self.b_ge_data

        # Solve QP
        try:
            self.problem.solve(solver=cp.CLARABEL, warm_start=True)
        except cp.error.SolverError:
            # Fallback to ECOS if CLARABEL fails
            self.problem.solve(solver=cp.ECOS, warm_start=True)

        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            warnings.warn(
                f"Projection QP returned status '{self.problem.status}'. "
                "Returning unprojected state."
            )
            return state

        # Unflatten result
        result_np = self._unflatten(self.eta.value, state_np)

        # Convert back to torch if needed
        if is_torch and device is not None:
            return {k: torch.tensor(v, device=device, dtype=dtype) for k, v in result_np.items()}

        return result_np


def _build_device_name_mapping(
    devices: list, parameter_names: dict[str, tuple[int, str]]
) -> dict[tuple[str, str], tuple[str, int]]:
    """Build mapping from (device_name, attribute) -> (param_name, array_index).

    Args:
        devices: List of device objects
        parameter_names: Maps param_name -> (device_index, attribute_name)

    Returns:
        Dict mapping (device_name, attribute) -> (param_name, array_index)
    """
    mapping = {}

    for param_name, (device_idx, attr_name) in parameter_names.items():
        device = devices[device_idx]
        device_names = device.name

        # Handle different name formats: str, pd.Index, np.ndarray, list
        if isinstance(device_names, str):
            # Single device with string name
            mapping[(device_names, attr_name)] = (param_name, 0)
        elif hasattr(device_names, "__iter__"):
            # Multiple devices (pd.Index, np.ndarray, list)
            for i, name in enumerate(device_names):
                mapping[(str(name), attr_name)] = (param_name, i)
        else:
            # Fallback for single device
            mapping[(str(device_names), attr_name)] = (param_name, 0)

    return mapping
