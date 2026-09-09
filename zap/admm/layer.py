from typing import Optional

import numpy as np
import torch

from zap.admm.basic_solver import ADMMSolver, ADMMState
from zap.network import PowerNetwork
from zap.devices.abstract import AbstractDevice
from zap.layer import DispatchLayer

#: Cap on `warm_start_stats` so a multi-hour run cannot grow it without bound.
MAX_WARM_START_STATS = 10_000


class ADMMLayer(DispatchLayer):
    """Maps device parameters to dispatch outcomes.

    Warm starts
    -----------
    The layer keeps the ADMM state produced by its last :meth:`forward` and, when
    ``warm_start`` is set, hands a detached copy of it to the next solve as
    ``initial_state``.  In the planning path there is one layer (and one
    :class:`ADMMSolver`) per block, so the carried state is per-block and blocks
    never contaminate each other.

    A capacity change does **not** invalidate the carried state -- it is the
    reason the warm start exists.  Neither does staleness: under minibatching a
    block that is skipped for several planner iterations simply keeps the state
    from whenever it was last visited.  That only degrades the *quality* of the
    warm start, never its validity, because the layout is unchanged.  What does
    invalidate it -- a different horizon, network, device set, machine or dtype --
    is caught by the :class:`~zap.admm.basic_solver.ADMMLayout` fingerprint
    stored inside the state, and the solve falls back to a cold start with a
    logged reason.

    An explicitly passed ``initial_state=`` always wins over the carried state.

    Args:
        warm_start: carry the state between forward passes.
        warm_start_reset_every: if a positive int ``N``, forward call ``i``
            (0-based) cold-starts iff ``i % N == 0``; call 0 is cold regardless
            (there is no state yet).  ``None`` or ``0`` disables periodic resets.
            It exists for the case where a stale state after a large parameter
            jump is *worse* than cold.
    """

    def __init__(
        self,
        network: PowerNetwork,
        devices: list[AbstractDevice],
        parameter_names: dict[str, tuple[int, str]],
        time_horizon: int = 1,
        solver: Optional[ADMMSolver] = None,
        warm_start: bool = True,
        warm_start_reset_every: Optional[int] = None,
        adapt_rho: bool = False,
        adapt_rho_rate: float = 0.1,
        verbose: bool = False,
        num_contingencies=0,
        contingency_device=None,
        contingency_mask=None,
    ):
        self.network = network
        self.devices = devices
        self.parameter_names = parameter_names
        self.time_horizon = time_horizon
        # A fresh solver per layer: a shared default instance would leak
        # warm-start flags between layers (and across threads).
        self.solver = solver if solver is not None else ADMMSolver(num_iterations=100, rho_power=1.0)
        self.warm_start = warm_start
        self.warm_start_reset_every = warm_start_reset_every
        self.adapt_rho = adapt_rho
        self.adapt_rho_rate = adapt_rho_rate
        self.verbose = verbose
        self.num_contingencies = num_contingencies
        self.contingency_device = contingency_device
        self.contingency_mask = contingency_mask

        #: The state carried between forward passes. `None` replaces the old
        #: `hasattr(self, "state")` idiom, so `reset_warm_start` has something to do.
        self.state = None
        self.history = None
        #: Completed forward calls.
        self.forward_count = 0
        #: One dict per forward call, for the run card. Capped at MAX_WARM_START_STATS.
        self.warm_start_stats: list[dict] = []

    def forward(self, initial_state=None, **kwargs) -> ADMMState:
        parameters = self.setup_parameters(**kwargs)

        if initial_state is None and self.warm_start and self.state is not None:
            reset_every = self.warm_start_reset_every
            periodic_reset = bool(reset_every) and (self.forward_count % reset_every == 0)
            if not periodic_reset:
                initial_state = self.state.copy()

        state, history = self.solver.solve(
            self.network,
            self.devices,
            self.time_horizon,
            parameters=parameters,
            initial_state=initial_state,
            num_contingencies=self.num_contingencies,
            contingency_device=self.contingency_device,
            contingency_mask=self.contingency_mask,
        )

        self.history = history
        self.state = state
        self.forward_count += 1

        self.warm_start_stats.append(
            {
                "forward": self.forward_count - 1,
                "warm_started": bool(getattr(self.solver, "warm_started", False)),
                "reason": getattr(self.solver, "warm_start_reason", None),
                "rho_rescaled": bool(getattr(self.solver, "warm_start_rho_rescaled", False)),
                "iterations": len(history.power),
                "converged": bool(getattr(self.solver, "converged", False)),
            }
        )
        if len(self.warm_start_stats) > MAX_WARM_START_STATS:
            del self.warm_start_stats[:-MAX_WARM_START_STATS]

        if self.adapt_rho:
            Jstar, n = history.objective[-1], self.solver.total_terminals
            self.solver.rho_power = self.adapt_rho_rate * Jstar / np.sqrt(n)
            print(f"Reset rho to {self.solver.rho_power}")

        if self.verbose:
            primal_resid = np.sqrt(history.power[-1] ** 2 + history.phase[-1] ** 2)
            dual_resid = np.sqrt(history.dual_power[-1] ** 2 + history.dual_phase[-1] ** 2)
            print(f"Primal residual: {primal_resid}")
            print(f"Dual residual: {dual_resid}")

        return state

    # ====
    # Warm start management
    # ====

    def reset_warm_start(self) -> None:
        """Drop the carried state; the next forward cold-starts."""
        self.state = None

    def save_warm_start(self, path) -> None:
        """Persist the carried state across a process (e.g. SLURM job) boundary.

        Seam only: nothing in the harness calls this yet.  The layout fingerprint
        travels inside the state, so a state restored onto a differently built
        system is refused by the solver rather than corrupting the run.
        """
        torch.save({"state": self.state}, path)

    def load_warm_start(self, path) -> bool:
        """Restore a state saved by :meth:`save_warm_start`.

        Returns ``False`` and leaves the carried state alone if ``path`` does not
        exist or holds no state.
        """
        import os

        if not os.path.exists(path):
            return False

        payload = torch.load(path, weights_only=False)
        state = payload.get("state") if isinstance(payload, dict) else None
        if state is None:
            return False

        self.state = state
        return True

    def backward(self, z, dz, **kwargs):
        # PlanningProblemADMM differentiates by unrolling the ADMM iterations with
        # plain torch autograd, so this implicit-VJP hook is never reached.
        assert NotImplementedError
