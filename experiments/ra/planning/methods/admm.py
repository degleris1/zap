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

import collections
import warnings

import numpy as np
import torch

from zap.admm import ADMMLayer, ADMMSolver

from ...config import ConfigError
from .. import base
from .gradient import GradientMethod

__all__ = [
    "MIN_WARM_START_MINIMUM_ITERATIONS",
    "TORCH_DTYPES",
    "VALID_MACHINES",
    "AdmmGradientMethod",
    "validate_admm",
    "validate_warm_start",
]

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

VALID_MACHINES = ("cpu", "cuda", "mps")

#: ``ADMMSolver``'s dual residual is a difference between consecutive iterates,
#: so the first one after a warm restart is ~0 and a warm-started solve can
#: declare convergence long before the primal residual has caught up with the
#: new parameters (spec 2026-09-09 section 3.4; observed as "converged in 1
#: iterations").  ``minimum_iterations`` is the only defence, so every ADMM
#: *planning* config is expected to set it at least this high.
MIN_WARM_START_MINIMUM_ITERATIONS = 100

#: Keep the run card bounded on a 6-hour run: at most this many forward-pass
#: records per block layer are aggregated (the most recent ones).
MAX_WARM_START_STATS_PER_LAYER = 10_000


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

    validate_warm_start(admm)
    return machine, TORCH_DTYPES[dtype_name]


def validate_warm_start(admm: dict) -> tuple[bool, int | None]:
    """Validate the two ``planning.admm`` warm-start keys (spec section 6).

    Returns ``(warm_start, warm_start_reset_every)``.  Also warns when
    ``solver_kwargs.minimum_iterations`` is below
    :data:`MIN_WARM_START_MINIMUM_ITERATIONS` while warm starts are on -- a
    warning rather than an error, because it degrades the *quality* of a solve
    and must not fail a config that was written before this key existed.
    """
    warm_start = bool(admm.get("warm_start", True))
    reset_every = admm.get("warm_start_reset_every")
    if reset_every is not None:
        reset_every = int(reset_every)
        if reset_every < 1:
            raise ConfigError(
                f"planning.admm.warm_start_reset_every={reset_every} must be >= 1; "
                "use null to disable periodic resets"
            )

    if warm_start:
        solver_kwargs = dict(admm.get("solver_kwargs") or {})
        minimum_iterations = solver_kwargs.get(
            "minimum_iterations", ADMMSolver.__dataclass_fields__["minimum_iterations"].default
        )
        if int(minimum_iterations) < MIN_WARM_START_MINIMUM_ITERATIONS:
            warnings.warn(
                "planning.admm.solver_kwargs.minimum_iterations="
                f"{int(minimum_iterations)} < {MIN_WARM_START_MINIMUM_ITERATIONS} with "
                "planning.admm.warm_start: true; a warm-started solve can declare "
                "convergence on a stale dual residual (spec section 3.4)",
                stacklevel=2,
            )
    return warm_start, reset_every


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
        # One layer, and one ADMMSolver, per block: the carried ADMM state and
        # rho are therefore per block and never cross-contaminate (spec 2.5).
        warm_start, reset_every = validate_warm_start(admm)

        def layer_factory(devices, time_horizon):
            return ADMMLayer(
                network,
                devices,
                parameter_names,
                time_horizon=time_horizon,
                solver=ADMMSolver(machine=machine, dtype=dtype, **solver_kwargs),
                warm_start=warm_start,
                warm_start_reset_every=reset_every,
                adapt_rho=adapt_rho,
                adapt_rho_rate=adapt_rho_rate,
            )

        return {"layer_factory": layer_factory}

    # -- solve --------------------------------------------------------------

    def solve(self, ctx):
        """The shared gradient solve, plus the warm-start block of the run card."""
        result = super().solve(ctx)
        result.compute = {
            **(result.compute or {}),
            "admm_warm_start": self._warm_start_summary(ctx),
        }
        return result

    def _warm_start_summary(self, ctx) -> dict:
        """Aggregate every block layer's ``warm_start_stats`` (spec section 2.6).

        Empty-but-present when the layer predates the warm-start bookkeeping, so
        a run card always carries the block and never a missing key.
        """
        admm = self.options["admm"]
        warm_start, reset_every = validate_warm_start(admm)

        stats: list[dict] = []
        for sub in getattr(ctx.problem, "subproblems", []) or []:
            layer = getattr(sub, "layer", None)
            layer_stats = list(getattr(layer, "warm_start_stats", []) or [])
            stats.extend(layer_stats[-MAX_WARM_START_STATS_PER_LAYER:])

        warm = [s for s in stats if s.get("warm_started")]
        cold = [s for s in stats if not s.get("warm_started")]
        reasons = collections.Counter(s["reason"] for s in stats if s.get("reason"))

        def mean_iterations(rows: list[dict]) -> float | None:
            values = [s["iterations"] for s in rows if s.get("iterations") is not None]
            return float(np.mean(values)) if values else None

        return {
            "enabled": warm_start,
            "reset_every": reset_every,
            "forward_passes": len(stats),
            "warm_passes": len(warm),
            "cold_passes": len(cold),
            "mean_iterations_warm": mean_iterations(warm),
            "mean_iterations_cold": mean_iterations(cold),
            "rho_rescaled_passes": sum(1 for s in stats if s.get("rho_rescaled")),
            "refusal_reasons": dict(reasons),
        }
