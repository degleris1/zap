import itertools
import torch
import time  # noqa
import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from typing import Optional
from dataclasses import dataclass
from itertools import repeat
from functools import cached_property
from collections.abc import Sequence

from zap.devices.abstract import AbstractDevice
from zap.devices.ground import Ground
from zap.util import torchify, torch_sparse, grad_or_zero, expand_params


@dataclass
class DispatchOutcome(Sequence):
    phase_duals: object
    local_equality_duals: object
    local_inequality_duals: object
    local_variables: object
    power: object
    angle: object
    prices: object
    global_angle: object
    problem: object = None
    ground: object = None

    def __getitem__(self, i):
        match i:
            case 0:
                return self.phase_duals
            case 1:
                return self.local_equality_duals
            case 2:
                return self.local_inequality_duals
            case 3:
                return self.local_variables
            case 4:
                return self.power
            case 5:
                return self.angle
            case 6:
                return self.prices
            case 7:
                return self.global_angle
            case _:
                raise IndexError

    def __len__(self):
        return 8

    @property
    def time_horizon(self):
        return self.global_angle.shape[1]

    @cached_property
    def size(self):
        return self.vectorize().size

    @cached_property
    def shape(self):
        mu_shape = [np.shape(mu) for mu in self.phase_duals]
        lambda_eq_shape = [[np.shape(lam_k) for lam_k in lam] for lam in self.local_equality_duals]
        lambda_ineq_shape = [
            [np.shape(lam_k) for lam_k in lam] for lam in self.local_inequality_duals
        ]
        u_shape = [[] if u is None else [np.shape(u_k) for u_k in u] for u in self.local_variables]
        p_shape = [np.shape(p) for p in self.power]
        a_shape = [np.shape(a) for a in self.angle]
        prices_shape = np.shape(self.prices)
        global_angle_shape = np.shape(self.global_angle)

        assert mu_shape == a_shape
        assert prices_shape == global_angle_shape

        return DispatchOutcome(
            *[
                mu_shape,
                lambda_eq_shape,
                lambda_ineq_shape,
                u_shape,
                p_shape,
                a_shape,
                prices_shape,
                global_angle_shape,
            ],
        )

    @cached_property
    def blocks(self):
        blocks, _ = self._build_blocks_recursively(self.shape, [], 0)
        return DispatchOutcome(*blocks)

    @cached_property
    def big_blocks(self):
        return DispatchOutcome(
            *[
                self._big_block(prop_name)
                for prop_name in [
                    "phase_duals",
                    "local_equality_duals",
                    "local_inequality_duals",
                    "local_variables",
                    "power",
                    "angle",
                    "prices",
                    "global_angle",
                ]
            ]
        )

    @property
    def big_dims(self):
        return DispatchOutcome(*[b[1] - b[0] for b in self.big_blocks])

    def _big_block(self, prop_name):
        block = getattr(self.blocks, prop_name)
        first_index = self._extremal_index(block, reducer=np.min)
        last_index = self._extremal_index(block, reducer=np.max)
        return first_index, last_index

    def _extremal_index(self, block, reducer):
        if isinstance(block, tuple):
            return reducer(block)
        else:
            return reducer([self._extremal_index(b, reducer=reducer) for b in block])

    def _build_blocks_recursively(self, shape, blocks, offset):
        # Recursive case
        if len(shape) > 0 and isinstance(shape[0], (list, tuple)):
            subblocks = []
            for shape_k in shape:
                new_block, offset = self._build_blocks_recursively(shape_k, subblocks, offset)
                subblocks += [new_block]

            return subblocks, offset

        # Base case
        if len(shape) == 0:
            delta = 0
        else:  # isinstance(shape[0], int)
            delta = np.prod(shape)

        return (offset, offset + delta), offset + delta

    def _safe_cat(self, x):
        if len(x) > 0:
            return np.concatenate(x)
        else:
            return []

    def _total_len(self, variable):
        return sum([0 if x is None else sum([xi.size for xi in x]) for x in variable])

    def torchify(self, requires_grad=False, machine=None):
        return DispatchOutcome(
            *[torchify(x, requires_grad=requires_grad, machine=machine) for x in self]
        )

    def vectorize(self):
        # Duals
        mu = self._safe_cat([np.array(mu).ravel() for mu in self.phase_duals if mu is not None])
        lambda_eq = [
            self._safe_cat([lam.ravel() for lam in lambda_eq])
            for lambda_eq in self.local_equality_duals
        ]
        lambda_ineq = [
            self._safe_cat([lam.ravel() for lam in lambda_ineq])
            for lambda_ineq in self.local_inequality_duals
        ]

        # Primals
        u = [
            np.concatenate([ui.ravel() for ui in u]) for u in self.local_variables if u is not None
        ]
        p = self._safe_cat([np.array(p).ravel() for p in self.power])
        a = self._safe_cat([np.array(a).ravel() for a in self.angle if a is not None])

        return self._safe_cat(
            [
                mu,  # Interface
                self._safe_cat(lambda_eq),  # Local
                self._safe_cat(lambda_ineq),  # Local
                self._safe_cat(u),  # Local
                p,  # Interface
                a,  # Interface
                self.prices.ravel(),  # Global
                self.global_angle.ravel(),  # Global
            ]
        )

    def package(self, vec):
        x = DispatchOutcome(
            *self._package(vec, self.blocks, self.shape),
            problem=self.problem,
            ground=self.ground,
        )

        # Replace Nones with empty arrays for the constraints
        for i, eq in enumerate(x.local_equality_duals):
            if eq is None:
                x.local_equality_duals[i] = []
        for i, ineq in enumerate(x.local_inequality_duals):
            if ineq is None:
                x.local_inequality_duals[i] = []

        return x

    def _package(self, vec, blocks, shapes):
        # Base case
        if isinstance(blocks, tuple):
            x = vec[blocks[0] : blocks[1]]
            if len(x) == 0:
                return None
            elif len(shapes) > 1:
                return x.reshape(shapes)
            else:
                return x

        # Recursive case
        return [self._package(vec, blk, shp) for blk, shp in zip(blocks, shapes)]


@dataclass
class PowerNetwork:
    """Defines the domain (nodes and settlement points) of the electrical system."""

    num_nodes: int

    def operation_cost(self, devices, power, angle, local_variables, parameters=None, la=np):
        parameters = expand_params(parameters, devices)

        costs = [
            d.operation_cost(p, v, u, **param, la=la)
            for d, p, v, u, param in zip(devices, power, angle, local_variables, parameters)
        ]
        return sum(costs)

    def model_variables(self, devices, time_horizon):
        global_angle = cp.Variable((self.num_nodes, time_horizon))
        power = [d.initialize_power(time_horizon) for d in devices]
        angle = [d.initialize_angle(time_horizon) for d in devices]
        local_variables = [d.model_local_variables(time_horizon) for d in devices]

        return global_angle, power, angle, local_variables

    def model_contingency_problem(
        self,
        devices,
        time_horizon,
        *,
        parameters=None,
        contingency_device=None,
        contingency_mask=None,
    ):
        num_contingencies = contingency_mask.shape[0]
        cd = contingency_device

        # Create base case variables
        base_variables = self.model_variables(devices, time_horizon)
        global_angle, power, angle, local_variables = base_variables

        # Create contingency variables
        global_angle_cont = [
            cp.Variable((self.num_nodes, time_horizon)) for _ in range(num_contingencies)
        ]
        power_cont = [devices[cd].initialize_power(time_horizon) for _ in range(num_contingencies)]
        angle_cont = [devices[cd].initialize_angle(time_horizon) for _ in range(num_contingencies)]
        local_cont = [
            devices[cd].model_local_variables(time_horizon) for _ in range(num_contingencies)
        ]

        # Model base case
        costs, constraints, data = self.model_dispatch_problem(
            devices, time_horizon, parameters=parameters, variables=base_variables
        )

        # Model contingencies
        for c in range(num_contingencies):
            mask = contingency_mask[c, :].T

            # Power balance
            pc = [power_cont[c] if i == cd else p for i, p in enumerate(power)]
            pow_bal_cont = cp.sum([get_net_power(d, p, la=cp) for d, p in zip(devices, pc)])
            pow_bal_cont = pow_bal_cont == 0

            # Phase consistency
            ac = [angle_cont[c] if i == cd else a for i, a in enumerate(angle)]
            phase_cons_cont = [
                match_phases(d, v, global_angle_cont[c]) for d, v in zip(devices, ac)
            ]

            # Contingency device constraints
            local_eqs_cont = [
                hi == 0
                for hi in devices[cd].equality_constraints(
                    power_cont[c],
                    angle_cont[c],
                    local_cont[c],
                    **parameters[cd],
                    la=cp,
                    mask=mask,
                )
            ]
            local_ineqs_cont = [
                gi <= 0
                for gi in devices[cd].inequality_constraints(
                    power_cont[c],
                    angle_cont[c],
                    local_cont[c],
                    **parameters[cd],
                    la=cp,
                    mask=mask,
                )
            ]

            constraints += [pow_bal_cont] + local_eqs_cont + local_ineqs_cont
            for pc in phase_cons_cont:
                constraints += pc

        # Combine data
        data["global_angle"] = [global_angle] + global_angle_cont
        data["power"][cd] = [power[cd]] + power_cont
        data["angle"][cd] = [angle[cd]] + angle_cont
        data["local_variables"][cd] = [local_variables[cd]] + local_cont

        # TODO - Add contingency duals
        # "phase_consistency": phase_consistency,
        # "power_balance": power_balance,

        return costs, constraints, data

    def model_dispatch_problem(
        self,
        devices,
        time_horizon,
        *,
        parameters=None,
        dual=False,
        envelope=None,
        lower_param=None,
        upper_param=None,
        variables=None,
    ):
        if envelope is not None:
            assert len(envelope) == 2
            assert lower_param is not None
            assert upper_param is not None

            device_envelopes = [(envelope, lb, ub) for lb, ub in zip(lower_param, upper_param)]
        else:
            lower_param = [None for _ in devices]
            upper_param = [None for _ in devices]

            device_envelopes = [None for _ in devices]

        # Initialize variables
        if variables is None:
            global_angle, power, angle, local_variables = self.model_variables(
                devices, time_horizon
            )
        else:
            global_angle, power, angle, local_variables = variables

        # Model constraints
        if dual:
            # Swap constraints
            net_power = cp.sum([get_net_power(d, p, la=cp) for d, p in zip(devices, angle)])
            phase_consistency = [match_phases(d, v, global_angle) for d, v in zip(devices, power)]

        else:  # Primal
            net_power = cp.sum([get_net_power(d, p, la=cp) for d, p in zip(devices, power)])
            phase_consistency = [match_phases(d, v, global_angle) for d, v in zip(devices, angle)]

        power_balance = net_power == 0

        local_equalities = [
            [hi == 0 for hi in d.equality_constraints(p, v, u, **param, la=cp, envelope=env)]
            for d, p, v, u, param, env in zip(
                devices, power, angle, local_variables, parameters, device_envelopes
            )
        ]

        local_inequalities = [
            [gi <= 0 for gi in d.inequality_constraints(p, v, u, **param, la=cp, envelope=env)]
            for d, p, v, u, param, env in zip(
                devices, power, angle, local_variables, parameters, device_envelopes
            )
        ]

        # Model objective
        costs = [
            d.operation_cost(p, v, u, **param, la=cp, envelope=env)
            for d, p, v, u, param, env in zip(
                devices, power, angle, local_variables, parameters, device_envelopes
            )
        ]

        constraints = list(
            itertools.chain(
                [power_balance],
                *phase_consistency,
                *local_equalities,
                *local_inequalities,
            )
        )

        data = {
            "global_angle": global_angle,
            "power": power,
            "angle": angle,
            "local_variables": local_variables,
            "phase_consistency": phase_consistency,
            "power_balance": power_balance,
            "local_equalities": local_equalities,
            "local_inequalities": local_inequalities,
        }

        return costs, constraints, data

    def dispatch(
        self,
        devices: list[AbstractDevice],
        time_horizon=None,
        *,
        solver=cp.ECOS,
        parameters=None,
        add_ground=True,
        dual=False,
        solver_kwargs={},
        num_contingencies=0,
        contingency_device: Optional[int] = None,
        contingency_mask=None,
    ) -> DispatchOutcome:
        # Compute time horizon automatically
        if time_horizon is None:
            time_horizon = max([d.time_horizon for d in devices])

        parameters = expand_params(parameters, devices)

        # Add ground if necessary
        ground = None
        if add_ground:
            ground = Ground(num_nodes=self.num_nodes, terminal=np.array([0]))
            devices = devices + [ground]
            parameters = parameters + [{}]

        # Type checks
        assert all([d.num_nodes == self.num_nodes for d in devices])
        assert time_horizon > 0
        assert all([d.time_horizon in [0, time_horizon] for d in devices])
        if num_contingencies > 0:
            assert contingency_device is not None
            assert contingency_mask.shape == (
                num_contingencies,
                devices[contingency_device].num_devices,
            )

        if num_contingencies > 0:
            costs, constraints, data = self.model_contingency_problem(
                devices,
                time_horizon,
                parameters=parameters,
                contingency_device=contingency_device,
                contingency_mask=contingency_mask,
            )
        else:
            costs, constraints, data = self.model_dispatch_problem(
                devices, time_horizon, parameters=parameters, dual=dual
            )

        # Unpack data
        power, angle, local_variables = (
            data["power"],
            data["angle"],
            data["local_variables"],
        )
        global_angle = data["global_angle"]
        power_balance, phase_consistency = (
            data["power_balance"],
            data["phase_consistency"],
        )
        local_equalities, local_inequalities = (
            data["local_equalities"],
            data["local_inequalities"],
        )

        # Formulate and solve cvxpy problem
        objective = cp.Minimize(cp.sum(costs))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver, **solver_kwargs)
        assert problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]

        # Evaluate variables
        power = nested_evaluate(power)
        angle = nested_evaluate(angle)
        global_angle = nested_evaluate(global_angle)
        local_variables = nested_evaluate(local_variables)

        return DispatchOutcome(
            global_angle=global_angle,
            power=power,
            angle=angle,
            local_variables=local_variables,
            prices=None if isinstance(power_balance, bool) else -power_balance.dual_value,
            phase_duals=[
                [pci.dual_value for pci in pc] if len(pc) > 0 else None for pc in phase_consistency
            ],
            local_equality_duals=[[lci.dual_value for lci in lc] for lc in local_equalities],
            local_inequality_duals=[[lci.dual_value for lci in lc] for lc in local_inequalities],
            problem=problem,
            ground=ground,
        )

    def kkt(self, devices, result, parameters=None, la=np):
        parameters = expand_params(parameters, devices)

        if result.ground is not None:
            devices = devices + [result.ground]
            parameters = parameters + [{}]

        power = result.power
        angle = result.angle
        local_vars = result.local_variables
        lambda_eq = result.local_equality_duals
        lambda_ineq = result.local_inequality_duals

        # Local constraints - primal feasibility
        kkt_local_equalities = [
            d.equality_constraints(p, a, u, la=la, **param)
            for d, p, a, u, param in zip(devices, power, angle, local_vars, parameters)
        ]

        kkt_local_inequalities = [
            [
                la.multiply(hi, lamb_i)
                for hi, lamb_i in zip(d.inequality_constraints(p, a, u, la=la, **param), lamb)
            ]
            for d, p, a, u, param, lamb in zip(
                devices, power, angle, local_vars, parameters, lambda_ineq
            )
        ]

        # Local variables - dual feasibility
        local_grads = [
            d.lagrangian_gradients(p, a, u, lam_eq, lam_ineq, la=la, **param)
            for d, p, a, u, param, lam_eq, lam_ineq in zip(
                devices, power, angle, local_vars, parameters, lambda_eq, lambda_ineq
            )
        ]

        kkt_power = [grad[0] for grad in local_grads]
        for kp, d in zip(kkt_power, devices):
            nu_local = apply_incidence_transpose(
                d, repeat(result.prices, d.num_terminals_per_device), la=la
            )
            for kpi, nu_local_i in zip(kp, nu_local):
                kpi -= nu_local_i

        kkt_local_angles = [grad[1] for grad in local_grads]
        for kp, d, mu in zip(kkt_local_angles, devices, result.phase_duals):
            if d.is_ac:
                for kpi, mui in zip(kp, mu):
                    kpi -= mui

        kkt_local_variables = [grad[2] for grad in local_grads]

        return DispatchOutcome(
            global_angle=self._kkt_global_angle(devices, result, la=la),
            power=kkt_power,
            angle=kkt_local_angles,
            local_variables=kkt_local_variables,
            prices=self._kkt_power_balance(devices, result, la=la),
            phase_duals=self._kkt_phase_consistency(devices, result, la=la),
            local_equality_duals=kkt_local_equalities,
            local_inequality_duals=kkt_local_inequalities,
            problem="KKT",
            ground=result.ground,
        )

    def _kkt_power_balance(self, devices, dispatch_outcome, la=np):
        net_powers = [get_net_power(d, p, la=la) for d, p in zip(devices, dispatch_outcome.power)]
        return sum(net_powers)  # , axis=0) (No axis because we use built-in sum)

    def _kkt_global_angle(self, devices, dispatch_outcome, la=np):
        angle_duals = [
            sum(apply_incidence(d, mu, la=la))
            for d, mu in zip(devices, dispatch_outcome.phase_duals)
            if d.is_ac
        ]
        return sum(angle_duals)  # , axis=0)

    def _kkt_phase_consistency(self, devices, dispatch_outcome, la=np):
        # Compute observed global angles
        theta_terminals = [
            apply_incidence_transpose(
                d,
                repeat(dispatch_outcome.global_angle, d.num_terminals_per_device),
                la=la,
            )
            for d in devices
        ]

        # Compute phase differences
        phase_diffs = [
            [thetai - ai for thetai, ai in zip(theta, a)] if a is not None else None
            for theta, a in zip(theta_terminals, dispatch_outcome.angle)
        ]

        return phase_diffs

    def kkt_jacobian_variables(self, devices, x: DispatchOutcome, parameters=None, vectorize=True):
        parameters = expand_params(parameters, devices)

        if x.ground is not None:
            devices = devices + [x.ground]
            parameters = parameters + [{}]

        assert len(devices) == len(parameters) == len(x.power)

        # Build incidence matrix for angle-related variables
        # For multiple time periods, we first build the incidence matrix for a single
        # time period, then apply the Kron product with the identity matrix
        angle_incidence = sum([d.incidence_matrix for d in devices if d.is_ac], [])
        angle_incidence = sp.hstack(angle_incidence)
        angle_incidence = sp.kron(angle_incidence, sp.eye(x.time_horizon))

        power_incidence = sum([d.incidence_matrix for d in devices], [])
        power_incidence = sp.hstack(power_incidence)
        power_incidence = sp.kron(power_incidence, sp.eye(x.time_horizon))

        # Build Jacobian in blocks
        # Outer block is the rows, inner block is the columns
        dims = x.big_dims
        blocks = x.big_blocks
        x_vec = x.vectorize()

        jac = DispatchOutcome(
            *[
                DispatchOutcome(*[sp.coo_matrix((dims[row], dims[col])) for col in range(len(x))])
                for row in range(len(x))
            ]
        )

        # Construct block row by block row
        # Part 1 - Phase duals (interface)
        # Jacobian is just the matrices of the equality constraint: A[d][t].T @ theta - phi[d][t]
        jac.phase_duals.angle = -sp.eye(dims.phase_duals)
        jac.phase_duals.global_angle = angle_incidence.T

        # Part 2 - Local equality duals (local)
        eq_mats = [
            d.equality_matrices(eq, p, a, u, **param)
            for d, eq, p, a, u, param in zip(
                devices,
                x.local_equality_duals,
                x.power,
                x.angle,
                x.local_variables,
                parameters,
            )
        ]

        A_p = sp.block_diag([_blockify(eqm, p, "power") for eqm, p in zip(eq_mats, x.power)])
        A_a = sp.block_diag([_blockify(eqm, a, "angle") for eqm, a in zip(eq_mats, x.angle)])
        A_u = sp.block_diag(
            [_blockify(eqm, u, "local_variables") for eqm, u in zip(eq_mats, x.local_variables)]
        )

        jac.local_equality_duals.power = A_p
        jac.local_equality_duals.angle = A_a
        jac.local_equality_duals.local_variables = A_u

        # Part 3 - Local inequcality duals (local)
        inequalities = x._safe_cat(
            [
                x._safe_cat([hi.ravel() for hi in d.inequality_constraints(p, a, u, **param)])
                for d, p, a, u, param in zip(
                    devices, x.power, x.angle, x.local_variables, parameters
                )
            ]
        )

        ineq_mats = [
            d.inequality_matrices(ineq, p, a, u, **param)
            for d, ineq, p, a, u, param in zip(
                devices,
                x.local_inequality_duals,
                x.power,
                x.angle,
                x.local_variables,
                parameters,
            )
        ]

        C_p = sp.block_diag([_blockify(ineqm, p, "power") for ineqm, p in zip(ineq_mats, x.power)])
        C_a = sp.block_diag([_blockify(ineqm, a, "angle") for ineqm, a in zip(ineq_mats, x.angle)])
        C_u = sp.block_diag(
            [
                _blockify(ineqm, u, "local_variables")
                for ineqm, u in zip(ineq_mats, x.local_variables)
            ]
        )

        i_lieq = blocks.local_inequality_duals
        diag_lamb = sp.diags(x_vec[i_lieq[0] : i_lieq[1]])

        # diag(C*x - d)
        jac.local_inequality_duals.local_inequality_duals = sp.diags(inequalities)

        # diag(lamb) * C
        jac.local_inequality_duals.power = diag_lamb * C_p
        jac.local_inequality_duals.angle = diag_lamb * C_a
        jac.local_inequality_duals.local_variables = diag_lamb * C_u

        # Part 4 - Local variables (local)
        local_hessians = [
            sp.block_diag(d.hessian_local_variables(p, a, u, **param))  # Block diagonal by var
            for d, p, a, u, param in zip(devices, x.power, x.angle, x.local_variables, parameters)
        ]
        H_u = sp.block_diag(local_hessians)

        jac.local_variables.local_variables += H_u
        jac.local_variables.local_equality_duals = A_u.T
        jac.local_variables.local_inequality_duals = C_u.T

        # Part 5 - Power (interface)
        power_hessians = [
            sp.block_diag(d.hessian_power(p, a, u, **param))  # Block diagonal by terminal
            for d, p, a, u, param in zip(devices, x.power, x.angle, x.local_variables, parameters)
        ]
        H_p = sp.block_diag(power_hessians)

        jac.power.power += H_p
        jac.power.prices = -power_incidence.T
        jac.power.local_equality_duals = A_p.T
        jac.power.local_inequality_duals = C_p.T

        # Part 6 - Angle (interface)
        angle_hessians = [
            sp.block_diag(d.hessian_angle(p, a, u, **param))  # Block diagonal by terminal
            for d, p, a, u, param in zip(devices, x.power, x.angle, x.local_variables, parameters)
        ]
        H_a = sp.block_diag(angle_hessians)

        jac.angle.angle += H_a
        jac.angle.phase_duals = -sp.eye(dims.angle)
        jac.angle.local_equality_duals = A_a.T
        jac.angle.local_inequality_duals = C_a.T

        # Part 7 - Prices, nu (global)
        # Only participates in the power balance constraint, just a single constraint
        # per node and time period
        # Constraint:      sum(A[d][t] @ p[d][t]) == 0      (dual variable nu)
        # Jacobian:        A[d][t]                          (in row of p[d][t])
        jac.prices.power = power_incidence

        # Part 8 - Global angle, theta (global)
        # Only participates in the phase consistency constraints (d = device, t = terminal)
        # Constraint:       A[d][t].T @ theta == phi[d][t]      (dual variable mu[d][t])
        # Jacobian:         A[d][t]                             (in column of mu[d][t])
        jac.global_angle.phase_duals = angle_incidence

        if vectorize:
            return sp.vstack([sp.hstack([Jij for Jij in Ji]) for Ji in jac], format="csc")
        else:
            return jac

    def kkt_vjp_variables(
        self,
        grad,
        devices,
        x: DispatchOutcome,
        parameters=None,
        vectorize=True,
        regularize=0.0,
        linear_solver="scipy",
    ):
        # Choose linear solver based on whether or no sparse_dot_mkl is available
        if linear_solver == "default":
            try:
                from sparse_dot_mkl import sparse_qr_solve_mkl
            except ImportError:
                linear_solver = "scipy"
            else:
                linear_solver = "mkl"

        if isinstance(grad, DispatchOutcome):
            grad = grad.vectorize()

        # start = time.time()
        jac = self.kkt_jacobian_variables(devices, x, parameters=parameters, vectorize=True)
        # print("Build Jacobian: ", time.time() - start)

        # Transpose and regularize
        jac_t = jac.T.tocsc()
        if regularize > 0.0:
            jac_t += regularize * sp.eye(jac_t.shape[0])

        # start = time.time()

        if linear_solver == "mkl":
            from sparse_dot_mkl import sparse_qr_solve_mkl

            grad_back = sparse_qr_solve_mkl(jac_t.tocsr(), grad)

        elif linear_solver == "scipy":  # Use scipy
            lu_factors = sp.linalg.splu(jac_t)
            grad_back = lu_factors.solve(grad)

        else:
            raise ValueError(f"Unknown linear solver: {linear_solver}")

        # print("Solve linear system:", time.time() - start)

        if vectorize:
            return grad_back
        else:
            return x.package(grad_back)

    def kkt_vjp_parameters(
        self,
        grad,
        devices,
        x: DispatchOutcome,
        parameters=None,
        param_ind=None,
        param_name=None,
    ):
        if x.ground is not None:
            devices = devices + [x.ground]
            parameters = parameters + [{}]

        assert param_ind is not None
        assert param_name is not None

        devices = [d.torchify(machine=None) for d in devices]

        # Pacakge for efficient computation
        if not isinstance(grad, DispatchOutcome):
            grad = x.package(grad)

        # Setup torch
        x_tc = x.torchify(requires_grad=True)
        i = param_ind
        param_i = {k: v for k, v in parameters[i].items()}
        param_i[param_name] = torchify(param_i[param_name], requires_grad=True)
        param_i_grad = torch.zeros_like(param_i[param_name])

        # Gradient of Lagrangian VJP
        # Compute Lagrangian and differentiate wrt primals
        lagrange = devices[i].lagrangian(
            x_tc.power[i],
            x_tc.angle[i],
            x_tc.local_variables[i],
            x_tc.local_equality_duals[i],
            x_tc.local_inequality_duals[i],
            la=torch,
            **param_i,
        )

        # Compute first and second derivatives of gradient of Lagrangian
        def add_dL_contribution(param_i_grad, attr):
            if getattr(x_tc, attr)[i] is None:
                return param_i_grad

            dL_var = torch.autograd.grad(
                lagrange,
                getattr(x_tc, attr)[i],
                create_graph=True,
                allow_unused=True,
                materialize_grads=True,
            )

            dL_var_theta = torch.autograd.grad(
                dL_var,
                param_i[param_name],
                grad_outputs=[torchify(p) for p in getattr(grad, attr)[i]],
                allow_unused=True,
            )
            if dL_var_theta[0] is not None:
                param_i_grad += dL_var_theta[0]

            return param_i_grad

        param_i_grad = add_dL_contribution(param_i_grad, "power")
        param_i_grad = add_dL_contribution(param_i_grad, "angle")
        param_i_grad = add_dL_contribution(param_i_grad, "local_variables")

        # Compute equality VJP
        equalities = devices[i].equality_constraints(
            x_tc.power[i], x_tc.angle[i], x_tc.local_variables[i], **param_i, la=torch
        )
        for d_nu, eq in zip(torchify(grad.local_equality_duals[i]), equalities):
            eq: torch.Tensor
            if eq.requires_grad:
                eq.backward(d_nu, retain_graph=True)

        # Compute inequality VJP
        ineqs = devices[i].inequality_constraints(
            x_tc.power[i], x_tc.angle[i], x_tc.local_variables[i], **param_i, la=torch
        )
        for d_lam, lam, ineq in zip(
            torchify(grad.local_inequality_duals[i]),
            x_tc.local_inequality_duals[i],
            ineqs,
        ):
            if ineq.requires_grad:
                ineq.backward(torch.multiply(lam, d_lam), retain_graph=True)

        param_i_grad += grad_or_zero(param_i[param_name])

        return param_i_grad


def nested_evaluate(variable):
    # Base cases
    if variable is None:
        return None
    if isinstance(variable, cp.Variable):  # This eval has to come before the Sequence eval
        return variable.value

    # Recursive case
    return [nested_evaluate(xi) for xi in variable]
    # return [[xi.value for xi in x] if (x is not None) else None for x in variable]


def apply_incidence(device: AbstractDevice, x, la=np):
    if la == torch:
        incidence = torch_sparse(device.incidence_matrix)
    else:
        incidence = device.incidence_matrix

    return [Ai @ xi for Ai, xi in zip(incidence, x)]


def apply_incidence_transpose(device: AbstractDevice, x, la=np):
    if la == torch:
        incidence = torch_sparse(device.incidence_matrix)
    else:
        incidence = device.incidence_matrix

    return [Ai.T @ xi for Ai, xi in zip(incidence, x)]


def get_net_power(device: AbstractDevice, p: list[cp.Variable], la=np):
    if p is not None:
        return sum(apply_incidence(device, p, la=la))
    else:
        return 0.0


def match_phases(device: AbstractDevice, v, global_v):
    if v is not None:
        return [Ai.T @ global_v == vi for Ai, vi in zip(device.incidence_matrix, v)]
    else:
        return []


def _blockify(eqm, power, prop_name):
    if len(eqm) >= 1:
        return sp.vstack([sp.hstack(getattr(eqmi, prop_name)) for eqmi in eqm])
    else:
        p_size = sum([p.size for p in power]) if power is not None else 0
        return sp.coo_matrix((0, p_size))
