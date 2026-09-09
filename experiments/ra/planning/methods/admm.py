"""The ADMM gradient planning method (WP5 spec section 5.7).

New code: ``runner.py``'s ``admm_gpu`` branch is an unconditional
``NotImplementedError``, so there is nothing to extract.  The reference is
``experiments/plan/runner.py:288-316``.

The method is a :class:`~.gradient.GradientMethod` that changes exactly two
things: the block devices are torchified, and the dispatch layer becomes an
``ADMMLayer``.  ``zap.planning.PlanningProblem`` already returns
``PlanningProblemADMM`` when it sees one, so nothing else branches.
"""

from __future__ import annotations

import torch

from zap.admm import ADMMLayer, ADMMSolver

from ...config import ConfigError
from .. import base
from .gradient import GradientMethod

__all__ = ["TORCH_DTYPES", "VALID_MACHINES", "AdmmGradientMethod", "validate_admm"]

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

VALID_MACHINES = ("cpu", "cuda", "mps")


def validate_admm(cfg: dict) -> tuple[str, torch.dtype]:
    """Validate ``planning.admm`` and return ``(machine, torch dtype)`` (R-W3)."""
    admm = base.planning_options(cfg)["admm"]
    machine = str(admm["machine"])
    dtype_name = str(admm["dtype"])

    if machine not in VALID_MACHINES:
        raise ConfigError(f"planning.admm.machine={machine!r} is not one of {list(VALID_MACHINES)}")
    if dtype_name not in TORCH_DTYPES:
        raise ConfigError(
            f"planning.admm.dtype={dtype_name!r} is not one of {sorted(TORCH_DTYPES)}"
        )
    if machine == "mps" and dtype_name == "float64":
        raise ConfigError(
            "planning.admm: torch MPS has no float64; use dtype: float32 or machine: cpu"
        )
    if machine == "cuda" and not torch.cuda.is_available():
        raise ConfigError("planning.admm.machine='cuda' but torch.cuda.is_available() is False")
    if machine == "mps" and not torch.backends.mps.is_available():
        raise ConfigError(
            "planning.admm.machine='mps' but torch.backends.mps.is_available() is False"
        )
    return machine, TORCH_DTYPES[dtype_name]


@base.register_method
class AdmmGradientMethod(GradientMethod):
    """Gradient planning whose inner dispatch is solved by ADMM."""

    name = "admm"

    #: Set by :meth:`build`; ``ADMMLayer`` needs both and ``layer_kwargs()``
    #: (spec section 3.2) is handed neither.
    network = None
    parameter_names = None

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        self.machine, self.torch_dtype = validate_admm(cfg)

    # -- build hooks --------------------------------------------------------

    def build(self, ctx_system):
        """Stash the network and parameter names, then run the shared ``build``.

        Both are invariant under ``apply_expansion``: it shares
        ``LoadedSystem.network`` and only rewrites capacity *bounds*, which
        ``setup_parameter_names`` does not read.  So the shared body itself is
        untouched, as spec section 3.2 requires.
        """
        from .. import parameters as parameters_mod

        self.network = ctx_system.network
        self.parameter_names = parameters_mod.setup_parameter_names(ctx_system.devices)
        return super().build(ctx_system)

    def prepare_devices(self, devices: list) -> list:
        return [d.torchify(machine=self.machine, dtype=self.torch_dtype) for d in devices]

    def layer_kwargs(self) -> dict:
        admm = self.options["admm"]
        solver_kwargs = dict(admm["solver_kwargs"] or {})
        adapt_rho = bool(admm["adapt_rho"])
        adapt_rho_rate = float(admm["adapt_rho_rate"])
        machine, dtype = self.machine, self.torch_dtype
        network, parameter_names = self.network, self.parameter_names

        def layer_factory(devices, time_horizon):
            return ADMMLayer(
                network,
                devices,
                parameter_names,
                time_horizon=time_horizon,
                solver=ADMMSolver(machine=machine, dtype=dtype, **solver_kwargs),
                adapt_rho=adapt_rho,
                adapt_rho_rate=adapt_rho_rate,
            )

        return {"layer_factory": layer_factory}
