import torch
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from copy import deepcopy
from collections import namedtuple
from functools import cached_property
from typing import Optional
from numpy.typing import NDArray

from zap.util import grad_or_zero, torchify


ConstraintMatrix = namedtuple(
    "ConstraintMatrix",
    [
        "power",
        "angle",
        "local_variables",
    ],
)


def get_time_horizon(array: NDArray) -> int:
    if len(array.shape) < 2:
        return 1
    else:
        return array.shape[1]


def make_dynamic(array: Optional[NDArray]) -> NDArray:
    if array is None or isinstance(array, float):
        return array

    if len(array.shape) == 1:
        if isinstance(array, np.ndarray):
            return np.expand_dims(array, axis=1)
        else:
            return torch.unsqueeze(array, dim=1)
    else:
        return array


class AbstractDevice:
    # Fields

    terminals: NDArray
    num_nodes: int
    time_horizon: int

    # ====
    # Core Functionality
    # ====

    # Optional
    @property
    def is_ac(self):
        return False

    # Optional
    @property
    def is_convex(self):
        return True

    # Optional
    def model_local_variables(self, time_horizon: int) -> list[cp.Variable]:
        return None

    def operation_cost(self, power, angle, local_variables, **kwargs):
        raise NotImplementedError

    def equality_constraints(self, power, angle, local_variables, **kwargs):
        raise NotImplementedError

    def inequality_constraints(self, power, angle, local_variables, **kwargs):
        raise NotImplementedError

    def _device_data(self, **kwargs):
        raise NotImplementedError

    def scale_costs(self, scale):
        raise NotImplementedError

    def scale_power(self, scale):
        raise NotImplementedError

    # ====
    # KKT Jacobian Functionality
    # ====

    def _equality_matrices(self, equalities, **kwargs):
        raise NotImplementedError

    def _inequality_matrices(self, inequalities, **kwargs):
        raise NotImplementedError

    # ====
    # ADMM Functionality
    # ====

    def admm_initialize_power_variables(self, time_horizon: int):
        raise NotImplementedError

    def admm_initialize_angle_variables(self, time_horizon: int):
        raise NotImplementedError

    def get_admm_power_weights(self, power, strategy: str, **kwargs):
        return [np.ones_like(pi) for pi in power]

    # def admm_initialize_local_variables(self, time_horizon: int):
    #     raise NotImplementedError

    def admm_prox_update(
        self, rho_power, rho_angle, power, angle, power_weights=None, angle_weights=None, **kwargs
    ):
        raise NotImplementedError

    # ====
    # Planning Functionality
    # ====

    def get_investment_cost(self, **kwargs):
        return 0.0

    def get_emissions(self, power, **kwargs):
        return 0.0

    def sample_time(self, time_periods, original_time_horizon):
        new_device = deepcopy(self)
        # Rescale capital costs by ratio of time horizons
        if hasattr(new_device, "capital_cost"):
            if new_device.capital_cost is not None:
                new_device.capital_cost /= original_time_horizon
                new_device.capital_cost *= np.size(time_periods)

        return new_device

    # ====
    # Shared Functionality (No need to override)
    # ====

    # Properties

    @property
    def num_terminals_per_device(self) -> int:
        terminals = self.terminals
        assert len(terminals.shape) <= 2

        if len(terminals.shape) == 1:
            return 1
        else:
            return terminals.shape[1]

    def torch_terminals(self, time_horizon, machine="cpu") -> list[torch.Tensor]:
        # Effectively caching manually
        if (
            hasattr(self, "_torch_terminals")
            and self._torch_terminal_time_horizon == time_horizon
            and self._torch_terminal_machine == machine
        ):
            return self._torch_terminals

        if len(self.terminals.shape) == 1:
            torch_terminals = [
                torch.tensor(self.terminals, device=machine).reshape(-1, 1).expand(-1, time_horizon)
            ]
        else:
            torch_terminals = [
                torch.tensor(self.terminals[:, i], device=machine)
                .reshape(-1, 1)
                .expand(-1, time_horizon)
                for i in range(self.num_terminals_per_device)
            ]

        self._torch_terminals = torch_terminals
        self._torch_terminal_time_horizon = time_horizon
        self._torch_terminal_machine = machine

        return torch_terminals

    @property
    def num_devices(self) -> int:
        return self.terminals.shape[0]

    @cached_property
    def incidence_matrix(self):
        dimensions = (self.num_nodes, self.num_devices)

        matrices = []
        for terminal_index in range(self.num_terminals_per_device):
            vals = np.ones(self.num_devices)

            if len(self.terminals.shape) == 1:
                rows = self.terminals
            else:
                rows = self.terminals[:, terminal_index]

            cols = np.arange(self.num_devices)

            matrices.append(sp.csc_matrix((vals, (rows, cols)), shape=dimensions))

        return matrices

    # Modeling Tools

    def device_data(self, la=np, machine=None, **kwargs):
        data = self._device_data(**kwargs)
        if machine is None:
            machine = "cuda" if torch.cuda.is_available() else "cpu"

        if la == torch:
            if machine == "cuda":
                print(f"Warning: moving data to GPU for device {type(self)}")

            data = type(data)(*[torchify(x, machine=machine) for x in data])

        return data

    def initialize_power(self, time_horizon: int) -> list[cp.Variable]:
        return [
            cp.Variable((self.num_devices, time_horizon))
            for _ in range(self.num_terminals_per_device)
        ]

    def initialize_angle(self, time_horizon: int) -> list[cp.Variable]:
        if self.is_ac:
            return [
                cp.Variable((self.num_devices, time_horizon))
                for _ in range(self.num_terminals_per_device)
            ]
        else:
            return None

    # Differentiation Tools

    def equality_matrices(self, equalities, power, angle, local_vars, **kwargs):
        equalities = self.get_empty_constraint_matrix(equalities, power, angle, local_vars)
        return self._equality_matrices(equalities, **kwargs)

    def inequality_matrices(self, inequalities, power, angle, local_vars, **kwargs):
        inequalities = self.get_empty_constraint_matrix(inequalities, power, angle, local_vars)
        return self._inequality_matrices(inequalities, **kwargs)

    def _get_empty_constraint_matrix(self, constr, power, angle, local_vars):
        num_constr = constr.size

        if angle is None:
            angle_mats = [sp.coo_matrix((num_constr, 0))]
        else:
            angle_mats = [sp.coo_matrix((num_constr, a.size)) for a in angle]

        if local_vars is None:
            local_mats = [sp.coo_matrix((num_constr, 0))]
        else:
            local_mats = [sp.coo_matrix((num_constr, u.size)) for u in local_vars]

        return ConstraintMatrix(
            power=[sp.coo_matrix((num_constr, p.size)) for p in power],
            angle=angle_mats,
            local_variables=local_mats,
        )

    def get_empty_constraint_matrix(self, constraints, power, angle, local_vars):
        return [self._get_empty_constraint_matrix(c, power, angle, local_vars) for c in constraints]

    def operation_cost_gradients(self, power, angle, local_variables, **kwargs):
        power = torchify(power, requires_grad=True)
        angle = torchify(angle, requires_grad=True)
        local_variables = torchify(local_variables, requires_grad=True)

        C = self.operation_cost(power, angle, local_variables, **kwargs, la=torch)
        if C.requires_grad:
            C.backward()

        return (
            grad_or_zero(power, to_numpy=True),
            grad_or_zero(angle, to_numpy=True),
            grad_or_zero(local_variables, to_numpy=True),
        )

    def lagrangian(
        self, power, angle, local_vars, equality_duals, inequality_duals, la=np, **kwargs
    ):
        # Cost term
        L = self.operation_cost(power, angle, local_vars, **kwargs, la=la)

        # Constraint terms
        eqs = self.equality_constraints(power, angle, local_vars, **kwargs, la=la)
        eq_terms = [
            la.sum(la.multiply(constraint, dual)) for constraint, dual in zip(eqs, equality_duals)
        ]

        ineqs = self.inequality_constraints(power, angle, local_vars, **kwargs, la=la)
        ineq_terms = [
            la.sum(la.multiply(constraint, dual))
            for constraint, dual in zip(ineqs, inequality_duals)
        ]

        for term in eq_terms + ineq_terms:
            L += term

        return L

    def lagrangian_gradients(
        self,
        power,
        angle,
        local_vars,
        equality_duals,
        inequality_duals,
        la=np,
        create_graph=False,
        **kwargs,
    ):
        power = torchify(power, requires_grad=True)
        angle = torchify(angle, requires_grad=True)
        local_vars = torchify(local_vars, requires_grad=True)

        equality_duals = torchify(equality_duals)
        inequality_duals = torchify(inequality_duals)

        L = self.lagrangian(
            power, angle, local_vars, equality_duals, inequality_duals, la=torch, **kwargs
        )

        if L.requires_grad:
            L.backward(create_graph=create_graph)

        if la == np:
            to_numpy = True
        else:
            to_numpy = False

        return (
            grad_or_zero(power, to_numpy=to_numpy),
            grad_or_zero(angle, to_numpy=to_numpy),
            grad_or_zero(local_vars, to_numpy=to_numpy),
        )
