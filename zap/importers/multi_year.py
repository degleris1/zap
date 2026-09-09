"""Multi-year stochastic planning with block sampling.

This module provides utilities for loading multiple PyPSA networks (e.g., different
weather years) and creating stochastic planning problems by sampling time blocks.
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable, Union
import logging

from zap.devices.injector import Generator, Load
from zap.devices.storage_unit import StorageUnit
from zap.devices.transporter import DirectedLine
from zap.layer import DispatchLayer
from zap.planning import PlanningProblem, StochasticPlanningProblem
from zap.planning.constraints import BudgetConstraintSet

from zap.importers.pypsa import load_pypsa_network

logger = logging.getLogger(__name__)


# Registry of time-varying attributes for each device type
TIME_VARYING_ATTRS = {
    Generator: ["dynamic_capacity", "linear_cost"],
    Load: ["load"],
    StorageUnit: ["power_availability", "linear_cost"],
    DirectedLine: ["max_power", "min_power", "linear_cost"],
    # Store is deliberately excluded (see phase-1 spec, decision D4).
    # Store: [
    #     "max_energy_capacity_availability",
    #     "min_energy_capacity_availability",
    #     "linear_cost",
    # ],
    # ACLine: ["susceptance", "nominal_capacity"],
    # DCLine: ["nominal_capacity"],
}


def concatenate_time_varying_attrs(
    base_dev,
    year_dev,
    *,
    expected_base_hours: int,
    expected_year_hours: int,
    reorder_idx=None,
) -> list[str]:
    """Concatenate one year's time-varying attributes onto the base device, in place.

    Only the attributes registered in :data:`TIME_VARYING_ATTRS` for the base
    device's type are considered, and only when both arrays are 2D and their
    time dimensions match the expected horizons (this filters out constant
    attributes that ``make_dynamic`` broadcast to shape ``(n, 1)``).

    Returns the list of attribute names that were actually concatenated.
    """
    concatenated = []

    for attr in TIME_VARYING_ATTRS.get(type(base_dev), []):
        base_arr = getattr(base_dev, attr, None)
        year_arr = getattr(year_dev, attr, None)

        if base_arr is None or year_arr is None:
            continue

        # Only concatenate if 2D (num_devices, time) AND actually time-varying
        # An attribute is time-varying if shape[1] matches the year's time horizon
        if base_arr.ndim != 2 or year_arr.ndim != 2:
            continue

        # Apply reordering to align device axis if needed
        if reorder_idx is not None:
            year_arr = year_arr[reorder_idx]

        if base_arr.shape[1] == expected_base_hours and year_arr.shape[1] == expected_year_hours:
            setattr(base_dev, attr, np.concatenate([base_arr, year_arr], axis=1))
            concatenated.append(attr)

    return concatenated


class MultiYearBlockSampler:
    """Load multiple weather years and sample time blocks for stochastic planning.

    This class handles:
    1. Loading topology from the first PyPSA network
    2. Concatenating time-varying data from all networks
    3. Sampling contiguous time blocks
    4. Creating StochasticPlanningProblem with proper capital cost scaling

    Example
    -------
    >>> networks = [pypsa.Network(f"year_{y}.nc") for y in range(2000, 2020)]
    >>> sampler = MultiYearBlockSampler(networks)
    >>> blocks = sampler.sample_blocks(block_size=168, strategy='all')
    >>> problem = sampler.create_stochastic_problem(blocks, parameter_names, ...)
    """

    def __init__(
        self,
        pypsa_networks: list,
        snapshots_per_network: Optional[list[pd.DatetimeIndex]] = None,
        **load_kwargs,
    ):
        """
        Initialize the sampler by loading and concatenating multi-year data.

        Parameters
        ----------
        pypsa_networks : list[pypsa.Network]
            List of PyPSA networks, one per weather year
        snapshots_per_network : list[pd.DatetimeIndex], optional
            Snapshots to use for each network. If None, uses each network's snapshots.
        **load_kwargs
            Additional arguments passed to load_pypsa_network (e.g., config, scaling)
        """
        if len(pypsa_networks) == 0:
            raise ValueError("Must provide at least one PyPSA network")

        # Handle snapshots
        if snapshots_per_network is None:
            snapshots_per_network = [None] * len(pypsa_networks)

        # Load base topology and devices from first network
        first_snapshots = snapshots_per_network[0]
        self.base_network, self.base_devices = load_pypsa_network(
            pypsa_networks[0], snapshots=first_snapshots, **load_kwargs
        )

        # Track time dimensions
        self.num_years = len(pypsa_networks)
        self.hours_per_year = []
        for net, snaps in zip(pypsa_networks, snapshots_per_network):
            if snaps is not None:
                self.hours_per_year.append(len(snaps))
            else:
                self.hours_per_year.append(len(net.snapshots))

        self.total_hours = sum(self.hours_per_year)
        self.year_boundaries = np.cumsum([0] + self.hours_per_year).tolist()

        # Store load kwargs for subsequent network loading
        self._load_kwargs = load_kwargs

        # Concatenate timeseries from all networks
        if len(pypsa_networks) > 1:
            self._concatenate_timeseries(pypsa_networks, snapshots_per_network)

        logger.info(
            f"Loaded {self.num_years} years with {self.total_hours} total hours. "
            f"Year boundaries: {self.year_boundaries}"
        )

    def _concatenate_timeseries(
        self,
        networks: list,
        snapshots_list: list[Optional[pd.DatetimeIndex]],
    ):
        """Concatenate time-varying attributes from all networks into base_devices.

        Parameters
        ----------
        networks : list[pypsa.Network]
            All PyPSA networks (including the first one already loaded)
        snapshots_list : list[pd.DatetimeIndex]
            Snapshots for each network
        """
        # Build type -> device index mapping for base devices
        # Match by device type (Generator, Load, etc.) rather than full name tuple,
        # since name ordering can differ across weather-year networks.
        device_type_to_idx = {}
        for idx, dev in enumerate(self.base_devices):
            dev_type = type(dev).__name__
            device_type_to_idx[dev_type] = idx

        # Process each subsequent network
        for net_idx, net in enumerate(networks[1:], start=1):
            snapshots = snapshots_list[net_idx]

            # Load this network's devices to get timeseries
            _, year_devices = load_pypsa_network(net, snapshots=snapshots, **self._load_kwargs)

            # Match and concatenate timeseries by device type
            for year_dev in year_devices:
                dev_type = type(year_dev).__name__

                if dev_type not in device_type_to_idx:
                    logger.warning(
                        f"Device type '{dev_type}' in year {net_idx} not found in base network, skipping"
                    )
                    continue

                base_idx = device_type_to_idx[dev_type]
                base_dev = self.base_devices[base_idx]

                # Compute reorder index if individual device names differ in order
                reorder_idx = self._get_reorder_index(base_dev, year_dev, net_idx, dev_type)

                # Concatenate each time-varying attribute
                concatenate_time_varying_attrs(
                    base_dev,
                    year_dev,
                    expected_base_hours=self.year_boundaries[net_idx],
                    expected_year_hours=self.hours_per_year[net_idx],
                    reorder_idx=reorder_idx,
                )

            logger.debug(f"Concatenated timeseries from year {net_idx}")

    @staticmethod
    def _get_reorder_index(base_dev, year_dev, net_idx, dev_type):
        """Compute reorder index to align year device names to base device order.

        Returns None if names already match in order, or an integer array of
        indices that reorders year_dev rows to match base_dev ordering.
        """
        base_names = getattr(base_dev, "name", None)
        year_names = getattr(year_dev, "name", None)

        if base_names is None or year_names is None:
            return None

        # Convert to pandas Index for comparison
        if not isinstance(base_names, pd.Index):
            base_names = pd.Index(base_names)
        if not isinstance(year_names, pd.Index):
            year_names = pd.Index(year_names)

        # If names match exactly in order, no reordering needed
        if base_names.equals(year_names):
            return None

        # If same names but different order, compute reorder index
        if len(base_names) == len(year_names) and set(base_names) == set(year_names):
            reorder_idx = year_names.get_indexer(base_names)
            if (reorder_idx == -1).any():
                logger.warning(
                    f"{dev_type} in year {net_idx}: name reindexing failed for some devices"
                )
                return None
            logger.info(
                f"{dev_type} in year {net_idx}: reordering {len(base_names)} devices to match base network"
            )
            return reorder_idx

        # Names are truly different — warn but proceed with direct concatenation
        base_set = set(base_names)
        year_set = set(year_names)
        missing = base_set - year_set
        extra = year_set - base_set
        logger.warning(
            f"{dev_type} in year {net_idx}: device name mismatch. "
            f"{len(missing)} in base but not year, {len(extra)} in year but not base. "
            f"Proceeding with direct concatenation (assuming corresponding order)."
        )
        return None

    def sample_blocks(
        self,
        block_size: int = 168,
        num_blocks: Optional[int] = None,
        strategy: str = "all",
        avoid_year_boundaries: bool = False,
        seed: Optional[int] = None,
    ) -> list[tuple[int, int]]:
        """
        Sample contiguous time blocks from the combined horizon.

        Parameters
        ----------
        block_size : int
            Hours per block (default 168 = 1 week)
        num_blocks : int, optional
            Number of blocks to sample. Required for 'random' and 'stratified'.
            Ignored for 'all' and 'uniform'.
        strategy : str
            Sampling strategy:
            - 'all': All consecutive non-overlapping blocks (uses all data)
            - 'uniform': Evenly spaced blocks across the horizon
            - 'random': Random non-overlapping blocks
            - 'stratified': Equal blocks sampled from each year
        avoid_year_boundaries : bool
            If True, blocks won't span year boundaries (only for 'random')
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        list[tuple[int, int]]
            List of (start_idx, end_idx) tuples defining each block
        """
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

    def create_stochastic_problem(
        self,
        blocks: list[tuple[int, int]],
        parameter_names: dict[str, tuple[int, str]],
        operation_objective_fn: Callable,
        investment_objective_fn: Callable,
        lower_bounds: Optional[dict] = None,
        upper_bounds: Optional[dict] = None,
        budget_constraints: Optional[Union[str, BudgetConstraintSet]] = None,
        **layer_kwargs,
    ) -> StochasticPlanningProblem:
        """
        Create a StochasticPlanningProblem from sampled blocks.

        Parameters
        ----------
        blocks : list[tuple[int, int]]
            List of (start, end) tuples from sample_blocks()
        parameter_names : dict[str, tuple[int, str]]
            Maps parameter name -> (device_index, attribute_name)
        operation_objective_fn : Callable
            Function(devices) -> OperationObjective
        investment_objective_fn : Callable
            Function(devices, layer) -> InvestmentObjective
        lower_bounds : dict, optional
            Lower bounds on parameters {param_name: array}
        upper_bounds : dict, optional
            Upper bounds on parameters {param_name: array}
        budget_constraints : str or BudgetConstraintSet, optional
            Budget constraints for the planning problem
        **layer_kwargs
            Additional arguments for DispatchLayer (solver, solver_kwargs, etc.)

        Returns
        -------
        StochasticPlanningProblem
            Ready for optimization with proper capital cost scaling
        """
        problems = []

        for block_idx, (start, end) in enumerate(blocks):
            block_indices = list(range(start, end))
            block_hours = end - start

            # Use sample_time() which handles capital cost scaling
            block_devices = [
                dev.sample_time(block_indices, self.total_hours) for dev in self.base_devices
            ]

            # Create dispatch layer for this block
            layer = DispatchLayer(
                self.base_network,
                block_devices,
                parameter_names,
                time_horizon=block_hours,
                **layer_kwargs,
            )

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
                budget_constraints=budget_constraints,
            )
            problems.append(prob)

            if (block_idx + 1) % 100 == 0:
                logger.info(f"Created {block_idx + 1}/{len(blocks)} subproblems")

        logger.info(f"Created StochasticPlanningProblem with {len(problems)} subproblems")

        # Default weights = [1.0, ...] works correctly with scaled capital costs
        return StochasticPlanningProblem(problems, budget_constraints=budget_constraints)

    def get_block_year(self, block_start: int) -> int:
        """Return which year a block starting at block_start belongs to."""
        for y in range(self.num_years):
            if block_start < self.year_boundaries[y + 1]:
                return y
        return self.num_years - 1

    def summary(self) -> dict:
        """Return a summary of the loaded data."""
        return {
            "num_years": self.num_years,
            "hours_per_year": self.hours_per_year,
            "total_hours": self.total_hours,
            "year_boundaries": self.year_boundaries,
            "num_devices": len(self.base_devices),
            "device_types": [type(d).__name__ for d in self.base_devices],
        }
