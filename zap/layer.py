import torch
import cvxpy as cp
from typing import Any
from copy import deepcopy

from zap.network import PowerNetwork, DispatchOutcome
from zap.devices.abstract import AbstractDevice


class DispatchLayer:
    """Maps device parameters to dispatch outcomes."""

    def __init__(
        self,
        network: PowerNetwork,
        devices: list[AbstractDevice],
        parameter_names: dict[str, tuple[int, str]],
        time_horizon: int = 1,
        solver=cp.ECOS,
        solver_kwargs={},
        add_ground=False,
        num_contingencies=0,
        contingency_device=None,
        contingency_mask=None,
    ):
        self.network = network
        self.devices = devices
        self.parameter_names = parameter_names
        self.time_horizon = time_horizon
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.add_ground = add_ground
        self.num_contingencies = num_contingencies
        self.contingency_device = contingency_device
        self.contingency_mask = contingency_mask

        # TODO - check that parameters match devices
        # TODO - check that parameters are unique?
        pass

    def __call__(self, **kwargs) -> Any:
        return self.forward(**kwargs)

    def setup_parameters(self, **kwargs):
        # Check that arguments match parameters
        assert kwargs.keys() == self.parameter_names.keys()

        # Match parameters to devices
        parameters = [{} for _ in self.devices]
        for k, (i, name) in self.parameter_names.items():
            parameters[i][name] = kwargs[k]

        return parameters

    def forward(self, **kwargs) -> DispatchOutcome:
        parameters = self.setup_parameters(**kwargs)

        return self.network.dispatch(
            self.devices,
            time_horizon=self.time_horizon,
            parameters=parameters,
            solver=self.solver,
            solver_kwargs=self.solver_kwargs,
            add_ground=self.add_ground,
            num_contingencies=self.num_contingencies,
            contingency_device=self.contingency_device,
            contingency_mask=self.contingency_mask,
        )

    def backward(
        self,
        z: DispatchOutcome,
        dz: DispatchOutcome,
        regularize=1e-8,
        return_adjoint: bool = False,
        **kwargs,
    ):
        """VJP of the dispatch map: ``dtheta = dz^T (dz/dtheta)``.

        With ``return_adjoint=True`` the adjoint state itself is returned
        alongside ``dtheta`` as ``(dtheta, dz_bar)``, where
        ``dz_bar = inv(JK_z.T) @ dz`` is packaged into a
        :class:`~zap.network.DispatchOutcome`.  Its ``prices`` entry is
        ``+d objective / d demand`` at each (node, hour) -- equivalently
        ``-d objective / d injection``, the same convention as
        ``DispatchOutcome.prices`` itself.  Measured against one-sided finite
        differences of the re-solved dispatch on 2026-09-13 (accreditation spec
        implementation note 29); it is the quantity the planner-side MRI
        (``plans/2026-09-11-accreditation-spec.md`` D9) needs and which is
        otherwise thrown away.  ``return_adjoint=False`` returns exactly what
        this method always returned.
        """
        parameters = self.setup_parameters(**kwargs)

        # dtheta = -JK_theta.T @ inv(JK_z.T) @ dz
        # dz_bar = inv(JK_z.T) @ dz
        # start = time.time()
        dz_bar = self.network.kkt_vjp_variables(
            dz, self.devices, z, parameters=parameters, regularize=regularize, vectorize=False
        )
        # print("J_var.T @ x: ", time.time() - start)

        # dtheta = -JK_theta.T @ dz_bar
        # start = time.time()
        dtheta = {}
        for key, (i, name) in self.parameter_names.items():
            dtheta[key] = -self.network.kkt_vjp_parameters(
                dz_bar,
                self.devices,
                z,
                parameters=parameters,
                param_ind=i,
                param_name=name,
            ).numpy()
        # print("J_theta.T @ x: ", time.time() - start)

        if return_adjoint:
            return dtheta, dz_bar

        return dtheta

    def initialize_parameters(self, requires_grad=False):
        initial_parameters = {}
        for name, (index, attr) in self.parameter_names.items():
            x = getattr(self.devices[index], attr)

            if isinstance(x, torch.Tensor):
                x = x.clone().detach()
                if requires_grad:
                    x.requires_grad = True
                initial_parameters[name] = x
            else:
                initial_parameters[name] = deepcopy(x)

        return initial_parameters
