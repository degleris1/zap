"""Block sampling over a :class:`~zap.importers.wy_store.LoadedSystem`.

This is the WP5 replacement for ``zap.importers.multi_year.MultiYearBlockSampler``
(spec section 5.1).  The old class stays in place — ``experiments/multi_year/runner.py``
still imports it and is the reference implementation for the equivalence test —
but its ``__init__`` and ``_concatenate_timeseries`` are pypsa loading and
cross-year concatenation that ``load_system`` already does.

The four ``_sample_*`` bodies are **copied verbatim** from that class: same RNG
call order, same ``rng.choice(list(available))`` idiom, same ``sorted()``.  For
identical ``total_hours``, ``year_boundaries`` and ``seed`` the two samplers
return identical block lists, and ``test_ra_planning_core.py`` pins that
equality without touching any data.

Blocks are ``(start, stop)`` pairs in *loaded-window index space*: indices into
the concatenated horizon ``0 .. total_hours`` (D-W9).  ``to_blocks`` derives the
year-relative :class:`~experiments.ra.blocks.Block` records used for reporting.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from zap.layer import DispatchLayer
from zap.planning import PlanningProblem, StochasticPlanningProblem
from zap.planning.constraints import BudgetConstraintSet

from ..blocks import Block
from ..dispatch import check_block_horizon

logger = logging.getLogger(__name__)


def _torchify_bounds(bounds: dict | None, layer) -> dict | None:
    """Move numpy planning bounds onto ``layer``'s torch device/dtype.

    ``parameters.setup_bounds`` returns numpy arrays, but a torch
    (``PlanningProblemADMM``) subproblem is stacked with ``torch.max`` /
    ``torch.min`` in ``StochasticPlanningProblem.__init__``, which raises on a
    numpy array.  Only the ADMM path (``layer_factory``) goes through here.
    """
    if bounds is None:
        return None

    import torch

    solver = getattr(layer, "solver", None)
    dtype = getattr(solver, "dtype", torch.float64)
    device = getattr(solver, "machine", "cpu")
    return {
        key: value
        if torch.is_tensor(value)
        else torch.tensor(np.asarray(value), dtype=dtype, device=device)
        for key, value in bounds.items()
    }


class SystemBlockSampler:
    """Sample contiguous time blocks from a loaded multi-year horizon."""

    def __init__(self, system) -> None:
        """Derive the horizon geometry from a ``LoadedSystem``."""
        meta = getattr(system, "meta", {}) or {}
        years = list(meta.get("years", []))
        num_years = len(years) or 1
        total_hours = int(meta.get("n_hours", 0))
        if total_hours <= 0:
            raise ValueError(
                "LoadedSystem.meta['n_hours'] is missing or non-positive; the sampler "
                "cannot derive the horizon geometry."
            )
        if total_hours % num_years != 0:
            raise ValueError(
                f"{total_hours} hours do not divide evenly into {num_years} years; "
                "SystemBlockSampler assumes load_system concatenated equal-length years "
                "(spec 5.2). Pass per-year lengths explicitly if this ever changes."
            )

        self.system = system
        self.base_network = system.network
        self.base_devices = system.devices
        self.years = years
        self.num_years = num_years
        self.hours_per_year = [total_hours // num_years] * num_years
        self.total_hours = total_hours
        self.year_boundaries = np.cumsum([0] + self.hours_per_year).tolist()

        bad = [
            f"{type(d).__name__}(time_horizon={d.time_horizon})"
            for d in self.base_devices
            if d.time_horizon not in (0, self.total_hours)
        ]
        if bad:
            raise ValueError(
                f"device(s) do not span the loaded horizon of {self.total_hours} hours: "
                f"{', '.join(bad)}. Their time-varying attributes were not concatenated."
            )

    # -----------------------------------------------------------------
    # Sampling  (bodies ported verbatim from MultiYearBlockSampler)
    # -----------------------------------------------------------------

    def sample_blocks(
        self,
        block_size: int | None = 168,
        num_blocks: int | None = None,
        strategy: str = "all",
        avoid_year_boundaries: bool = False,
        seed: int | None = None,
    ) -> list[tuple[int, int]]:
        """Sample contiguous time blocks from the combined horizon.

        ``block_size=None`` is the monolithic sentinel: one block of
        ``total_hours``.
        """
        if block_size is None:
            return [(0, self.total_hours)]

        rng = np.random.default_rng(seed)

        if strategy == "all":
            return self._sample_all(block_size)
        elif strategy == "uniform":
            n_blocks = num_blocks or (self.total_hours // block_size)
            return self._sample_uniform(block_size, n_blocks)
        elif strategy == "random":
            if num_blocks is None:
                raise ValueError("num_blocks required for 'random' strategy")
            return self._sample_random(block_size, num_blocks, avoid_year_boundaries, rng)
        elif strategy == "stratified":
            if num_blocks is None:
                raise ValueError("num_blocks required for 'stratified' strategy")
            return self._sample_stratified(block_size, num_blocks, rng)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _sample_all(self, block_size: int) -> list[tuple[int, int]]:
        """Return all consecutive non-overlapping blocks."""
        num_complete = self.total_hours // block_size
        blocks = [(i * block_size, (i + 1) * block_size) for i in range(num_complete)]

        remainder = self.total_hours % block_size
        if remainder > 0:
            logger.info(
                f"Using {num_complete} complete blocks of {block_size} hours. "
                f"Remainder of {remainder} hours not included."
            )

        return blocks

    def _sample_uniform(self, block_size: int, num_blocks: int) -> list[tuple[int, int]]:
        """Sample evenly spaced blocks across the horizon."""
        if num_blocks * block_size > self.total_hours:
            raise ValueError(
                f"Cannot fit {num_blocks} blocks of {block_size} hours "
                f"in {self.total_hours} total hours"
            )

        spacing = self.total_hours // num_blocks
        return [(i * spacing, i * spacing + block_size) for i in range(num_blocks)]

    def _sample_random(
        self,
        block_size: int,
        num_blocks: int,
        avoid_boundaries: bool,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """Sample random non-overlapping blocks."""
        if avoid_boundaries:
            # Only allow starts that don't cross year boundaries
            valid_starts = []
            for y in range(self.num_years):
                year_start = self.year_boundaries[y]
                year_end = self.year_boundaries[y + 1]
                max_start = year_end - block_size
                if max_start >= year_start:
                    valid_starts.extend(range(year_start, max_start + 1))
        else:
            valid_starts = list(range(self.total_hours - block_size + 1))

        # Sample non-overlapping blocks
        blocks = []
        available = set(valid_starts)

        while len(blocks) < num_blocks and available:
            start = rng.choice(list(available))
            blocks.append((start, start + block_size))

            # Remove overlapping starts
            for s in range(max(0, start - block_size + 1), start + block_size):
                available.discard(s)

        if len(blocks) < num_blocks:
            logger.warning(
                f"Could only sample {len(blocks)} non-overlapping blocks (requested {num_blocks})"
            )

        return sorted(blocks)

    def _sample_stratified(
        self,
        block_size: int,
        num_blocks: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """Sample blocks stratified by year (proportional to year length)."""
        blocks = []

        # Distribute blocks proportionally to year length
        blocks_per_year = []
        remaining = num_blocks
        for y in range(self.num_years):
            year_hours = self.hours_per_year[y]
            # Proportional allocation
            year_blocks = int(num_blocks * year_hours / self.total_hours)
            blocks_per_year.append(year_blocks)
            remaining -= year_blocks

        # Distribute remaining blocks to largest years
        for i in range(remaining):
            blocks_per_year[i % self.num_years] += 1

        # Sample within each year
        for y in range(self.num_years):
            year_start = self.year_boundaries[y]
            year_end = self.year_boundaries[y + 1]
            year_hours = year_end - year_start

            n_year_blocks = blocks_per_year[y]
            if n_year_blocks == 0:
                continue

            max_start = year_end - block_size
            if max_start < year_start:
                logger.warning(f"Year {y} too short for block_size {block_size}")
                continue

            valid_starts = list(range(year_start, max_start + 1))

            # Sample non-overlapping within this year
            year_blocks = []
            available = set(valid_starts)

            while len(year_blocks) < n_year_blocks and available:
                start = rng.choice(list(available))
                year_blocks.append((start, start + block_size))

                for s in range(max(year_start, start - block_size + 1), start + block_size):
                    available.discard(s)

            blocks.extend(year_blocks)

        return sorted(blocks)

    # -----------------------------------------------------------------
    # Problem construction
    # -----------------------------------------------------------------

    def create_stochastic_problem(
        self,
        blocks: list[tuple[int, int]],
        parameter_names: dict[str, tuple[int, str]],
        operation_objective_fn: Callable,
        investment_objective_fn: Callable,
        lower_bounds: dict | None = None,
        upper_bounds: dict | None = None,
        budget_constraints: str | BudgetConstraintSet | None = None,
        weights: list[float] | None = None,
        device_hook: Callable[[list], list] | None = None,
        layer_factory: Callable | None = None,
        **layer_kwargs,
    ) -> StochasticPlanningProblem:
        """Build a ``StochasticPlanningProblem``, one subproblem per block.

        Ported from ``MultiYearBlockSampler.create_stochastic_problem`` with
        exactly three additions (spec section 5.1): ``device_hook`` (ADMM
        torchification, applied after ``sample_time``), ``layer_factory``
        (``ADMMLayer`` instead of ``DispatchLayer``), and ``weights``
        (uniform in phase 1, D-W8).

        ``dev.sample_time(block_indices, self.total_hours)`` is what pro-rates
        the capital cost onto the block (spec section 6) — ``total_hours`` is the
        *loaded* horizon, never the sampled one.
        """
        # --- ADMM path fixes (WP5-B report) --------------------------------
        # `PlanningProblemADMM.__init__` takes no `budget_constraints`, and its
        # bounds must be torch tensors because `StochasticPlanningProblem`
        # torch-stacks them.  Both are decided by `layer_factory` (only the ADMM
        # method passes one).
        is_admm = layer_factory is not None
        if is_admm and budget_constraints is not None:
            raise NotImplementedError(
                "budget constraints are not supported on the ADMM planning path: "
                "PlanningProblemADMM takes no budget_constraints argument, so a "
                "budget CSV would be silently ignored. Use a cvx dispatch method."
            )

        problem_kwargs: dict = {}
        if budget_constraints is not None:
            problem_kwargs["budget_constraints"] = budget_constraints

        problems = []

        for block_idx, (start, end) in enumerate(blocks):
            block_indices = list(range(start, end))
            block_hours = end - start

            # Use sample_time() which handles capital cost scaling
            block_devices = [
                dev.sample_time(block_indices, self.total_hours) for dev in self.base_devices
            ]
            if device_hook is not None:
                block_devices = device_hook(block_devices)

            check_block_horizon(block_devices, block_hours)

            # Create dispatch layer for this block
            if layer_factory is None:
                layer = DispatchLayer(
                    self.base_network,
                    block_devices,
                    parameter_names,
                    time_horizon=block_hours,
                    **layer_kwargs,
                )
            else:
                layer = layer_factory(block_devices, block_hours)
                if block_idx == 0:
                    lower_bounds = _torchify_bounds(lower_bounds, layer)
                    upper_bounds = _torchify_bounds(upper_bounds, layer)

            # Create objectives
            op_objective = operation_objective_fn(block_devices)
            inv_objective = investment_objective_fn(block_devices, layer)

            # Create planning problem
            prob = PlanningProblem(
                operation_objective=op_objective,
                investment_objective=inv_objective,
                layer=layer,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                **problem_kwargs,
            )
            problems.append(prob)

            if (block_idx + 1) % 100 == 0:
                logger.info(f"Created {block_idx + 1}/{len(blocks)} subproblems")

        logger.info(f"Created StochasticPlanningProblem with {len(problems)} subproblems")

        # Default weights = [1.0, ...] works correctly with scaled capital costs
        return StochasticPlanningProblem(
            problems, weights=weights, budget_constraints=budget_constraints
        )

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------

    def get_block_year(self, block_start: int) -> int:
        """Which year index a block starting at ``block_start`` belongs to."""
        for y in range(self.num_years):
            if block_start < self.year_boundaries[y + 1]:
                return y
        return self.num_years - 1

    def to_blocks(self, pairs: list[tuple[int, int]]) -> list[Block]:
        """Year-relative :class:`Block` records, for reporting only (D-W9)."""
        out = []
        for i, (start, stop) in enumerate(pairs):
            y = self.get_block_year(start)
            offset = self.year_boundaries[y]
            year = self.years[y] if y < len(self.years) else y
            out.append(
                Block(index=i, year=int(year), start=int(start - offset), stop=int(stop - offset))
            )
        return out

    def summary(self) -> dict:
        """A JSON-able summary of the loaded horizon."""
        return {
            "num_years": self.num_years,
            "years": [int(y) for y in self.years],
            "hours_per_year": list(self.hours_per_year),
            "total_hours": int(self.total_hours),
            "year_boundaries": list(self.year_boundaries),
            "num_devices": len(self.base_devices),
            "device_types": [type(d).__name__ for d in self.base_devices],
        }
