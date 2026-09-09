"""
Multi-year stochastic planning experiment runner.

This module provides functionality for running stochastic planning experiments
with multiple weather years, using block sampling for computational tractability.

Example usage:
    python experiments/multi_year/runner.py config.yaml
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import torch
import pypsa
import yaml
import json
import logging
import time

try:
    import wandb
except ImportError:
    wandb = None

from pathlib import Path
from typing import Union, List
from copy import deepcopy

import zap
import zap.planning.trackers as tr
from zap.importers.multi_year import MultiYearBlockSampler
from zap.planning import (
    InvestmentObjective,
    MonolithicPlanningProblem,
    RelaxedPlanningProblem,
    GradientDescent,
)

# Methods that produce the optimum from a single LP solve (Step 6 only; no gradient descent).
_LP_ONLY_METHODS = {"monolithic", "relaxed", "stochastic"}
_KNOWN_METHODS = _LP_ONLY_METHODS | {"gradient", "admm_gpu"}
from zap.planning.operation_objectives import DispatchCostObjective, EmissionsObjective

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", message="datetime.datetime.utcnow", category=DeprecationWarning)


# ============================================================================
# Path Configuration
# ============================================================================

ZAP_PATH = Path(zap.__file__).parent.parent
DATA_PATH = ZAP_PATH / "data"
EXPERIMENT_PATH = ZAP_PATH / "experiments" / "multi_year"
EXPERIMENT_DATA_PATH = EXPERIMENT_PATH / "data"
OUTPUT_PATH = EXPERIMENT_PATH / "outputs"


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_PYPSA_ARGS = {
    "power_unit": 1.0e3,
    "cost_unit": 10.0,
}

DEFAULT_DISPATCH_SOLVER = "CLARABEL"
DEFAULT_DISPATCH_SOLVER_KWARGS = {"verbose": False}
DEFAULT_RELAXATION_SOLVER = "CLARABEL"
DEFAULT_RELAXATION_SOLVER_KWARGS = {"verbose": False}


# ============================================================================
# Config Loading
# ============================================================================


def load_config(config_path: Union[str, Path]) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`. Override wins on conflict."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_and_merge_configs(paths: List[Union[str, Path]]) -> dict:
    """Load and deep-merge multiple YAML configs in order. Later files override earlier."""
    merged: dict = {}
    for p in paths:
        merged = deep_merge(merged, load_config(p))
    return merged


def datadir(*args) -> Path:
    """Get path relative to output directory."""
    path = OUTPUT_PATH.joinpath(*args)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================================
# Network Loading
# ============================================================================


def list_available_networks(data_path: Path = None) -> List[str]:
    """
    List available PyPSA network files in the data directory.

    Parameters
    ----------
    data_path : Path, optional
        Path to data directory. Defaults to EXPERIMENT_DATA_PATH.

    Returns
    -------
    List[str]
        List of network filenames (sorted by name)
    """
    if data_path is None:
        data_path = EXPERIMENT_DATA_PATH

    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_path}")
        return []

    networks = sorted([f.name for f in data_path.glob("*.nc")])
    return networks


def load_multi_year_networks(
    network_files: List[str] = None,
    data_path: Path = None,
    snapshots: pd.DatetimeIndex = None,
    hours_per_year: int = None,
) -> tuple[List[pypsa.Network], List[pd.DatetimeIndex]]:
    """
    Load multiple PyPSA networks from the data directory.

    Parameters
    ----------
    network_files : List[str], optional
        List of network filenames to load. If None, loads all .nc files.
    data_path : Path, optional
        Path to data directory. Defaults to EXPERIMENT_DATA_PATH.
    snapshots : pd.DatetimeIndex, optional
        Specific snapshots to use. If None, uses all snapshots from each network.
    hours_per_year : int, optional
        Limit number of hours per year. If None, uses all hours.

    Returns
    -------
    tuple[List[pypsa.Network], List[pd.DatetimeIndex]]
        Lists of loaded networks and their snapshots
    """
    if data_path is None:
        data_path = EXPERIMENT_DATA_PATH

    if network_files is None:
        network_files = list_available_networks(data_path)

    if len(network_files) == 0:
        raise ValueError(f"No network files found in {data_path}")

    networks = []
    snapshots_list = []

    for filename in network_files:
        filepath = data_path / filename
        logger.info(f"Loading network: {filepath}")

        net = pypsa.Network(str(filepath))
        networks.append(net)

        # Determine snapshots for this network
        if snapshots is not None:
            net_snapshots = snapshots
        elif hours_per_year is not None:
            net_snapshots = net.snapshots[:hours_per_year]
        else:
            net_snapshots = net.snapshots

        snapshots_list.append(net_snapshots)
        logger.info(f"  Loaded {len(net_snapshots)} snapshots from {filename}")

    logger.info(f"Loaded {len(networks)} networks")
    return networks, snapshots_list


# ============================================================================
# Experiment Setup
# ============================================================================


def setup_parameter_names(devices: list) -> dict[str, tuple[int, str]]:
    """
    Create parameter_names dict for planning from device list.

    Returns mapping from parameter name to (device_index, attribute_name).
    Uses device type and index for simple, consistent naming.
    """
    parameter_names = {}

    for i, dev in enumerate(devices):
        dev_type = type(dev).__name__.lower()

        # Check for expandable capacity
        if hasattr(dev, "nominal_capacity"):
            cap = dev.nominal_capacity
            if cap is not None and hasattr(dev, "capital_cost") and dev.capital_cost is not None:
                parameter_names[f"{dev_type}_capacity"] = (i, "nominal_capacity")

        if hasattr(dev, "power_capacity"):
            cap = dev.power_capacity
            if cap is not None and hasattr(dev, "capital_cost") and dev.capital_cost is not None:
                parameter_names[f"{dev_type}_power"] = (i, "power_capacity")

    return parameter_names


def setup_bounds(
    devices: list,
    parameter_names: dict[str, tuple[int, str]],
    power_unit: float = 1.0,
    min_capacity_floor_mw: float = 0.1,
    min_storage_floor_mw: float = 10.0,
) -> tuple[dict, dict]:
    """
    Build explicit lower/upper bounds for planning parameters.

    Mirrors the bounds logic in development/19_benchmarking.py to prevent
    dispatch infeasibility from zero-capacity devices.

    Parameters
    ----------
    devices : list
        List of ZAP device objects (already scaled by power_unit).
    parameter_names : dict
        Maps parameter name -> (device_index, attribute_name).
    power_unit : float
        The power_unit used when importing the network. Floors specified
        in MW are divided by this to match device units.
    min_capacity_floor_mw : float
        Minimum capacity in MW for all devices.
    min_storage_floor_mw : float
        Higher minimum in MW for storage to avoid SOC infeasibility.

    Returns
    -------
    tuple[dict, dict]
        (lower_bounds, upper_bounds) dicts keyed by parameter name.
    """
    from zap.devices.storage_unit import StorageUnit

    # Convert MW floors to device units
    min_capacity_floor = min_capacity_floor_mw / power_unit
    min_storage_floor = min_storage_floor_mw / power_unit

    lower_bounds = {}
    upper_bounds = {}

    for param_name, (device_idx, attr_name) in parameter_names.items():
        device = devices[device_idx]
        current_cap = getattr(device, attr_name)

        # Lower bound: use device min if available, else zeros
        if isinstance(device, StorageUnit):
            min_attr = "min_power_capacity"
            max_attr = "max_power_capacity"
        else:
            min_attr = "min_nominal_capacity"
            max_attr = "max_nominal_capacity"

        lb = getattr(device, min_attr, None)
        if lb is None:
            lb = np.zeros_like(current_cap)
        else:
            lb = lb.copy()

        # Upper bound: use device max if available, else large multiple
        ub = getattr(device, max_attr, None)
        if ub is None:
            ub = (current_cap + 1000.0 / power_unit) * 10.0
        else:
            ub = ub.copy()
            # Replace infinities with a reasonable cap
            if np.any(np.isinf(ub)):
                reasonable = (current_cap + 1000.0 / power_unit) * 10.0
                ub = np.where(np.isinf(ub), reasonable, ub)

        # Apply floors
        lb = np.maximum(lb, min_capacity_floor)
        if isinstance(device, StorageUnit):
            lb = np.maximum(lb, min_storage_floor)

        lower_bounds[param_name] = lb
        upper_bounds[param_name] = ub

    return lower_bounds, upper_bounds


def export_solved_network(
    pypsa_network_path: Path,
    devices: list,
    parameter_names: dict[str, tuple[int, str]],
    optimal_params: dict[str, np.ndarray],
    pypsa_args: dict = None,
) -> pypsa.Network:
    """
    Apply optimized capacities to a PyPSA network by updating p_nom_opt values.

    This function:
    1. Loads one of the original PyPSA network files
    2. Updates p_nom_opt (or s_nom_opt) with the optimized capacities
    3. Returns the modified network

    Parameters
    ----------
    pypsa_network_path : Path
        Path to the PyPSA network file to load and modify
    devices : list
        List of ZAP device objects (for mapping device names)
    parameter_names : dict[str, tuple[int, str]]
        Maps parameter name -> (device_index, attribute_name)
    optimal_params : dict[str, np.ndarray]
        Optimal parameter values from optimization
    pypsa_args : dict, optional
        PyPSA import arguments (power_unit, cost_unit, etc.)

    Returns
    -------
    pypsa.Network
        PyPSA network with optimized p_nom_opt values applied
    """
    if pypsa_args is None:
        pypsa_args = DEFAULT_PYPSA_ARGS

    power_unit = pypsa_args.get("power_unit", 1.0)

    logger.info("=" * 60)
    logger.info("Exporting optimized capacities to PyPSA network")
    logger.info("=" * 60)

    # Step 1: Load the PyPSA network
    logger.info(f"Loading PyPSA network from {pypsa_network_path}")
    network = pypsa.Network(str(pypsa_network_path))

    # Step 2: Apply optimal parameters
    logger.info("Applying optimized capacities...")

    for param_name, param_value in optimal_params.items():
        if param_name not in parameter_names:
            continue

        device_idx, attr_name = parameter_names[param_name]
        device = devices[device_idx]

        # Get device names (could be pd.Index or np.ndarray)
        device_names = device.name
        if isinstance(device_names, str):
            device_names = [device_names]
        elif hasattr(device_names, "tolist"):
            device_names = device_names.tolist()

        # Convert param_value to flat 1D array
        param_array = np.atleast_1d(param_value).flatten()

        # Undo power unit scaling to get back to original units (MW)
        param_array_scaled = param_array * power_unit

        # Determine which PyPSA component and attribute to update
        from zap.devices.injector import Generator
        from zap.devices.transporter import DCLine, ACLine
        from zap.devices.storage_unit import StorageUnit

        if isinstance(device, Generator):
            # Update generators p_nom_opt
            for i, name in enumerate(device_names):
                if name in network.generators.index:
                    value = float(param_array_scaled[i])
                    network.generators.loc[name, "p_nom_opt"] = value
                    logger.info(f"  Generator '{name}': p_nom_opt = {value:.2f} MW")

        elif isinstance(device, StorageUnit):
            # Update storage_units p_nom_opt
            for i, name in enumerate(device_names):
                if name in network.storage_units.index:
                    value = float(param_array_scaled[i])
                    network.storage_units.loc[name, "p_nom_opt"] = value
                    logger.info(f"  StorageUnit '{name}': p_nom_opt = {value:.2f} MW")

        elif isinstance(device, DCLine):
            # Update links p_nom_opt
            for i, name in enumerate(device_names):
                if name in network.links.index:
                    value = float(param_array_scaled[i])
                    network.links.loc[name, "p_nom_opt"] = value
                    logger.info(f"  Link '{name}': p_nom_opt = {value:.2f} MW")

        elif isinstance(device, ACLine):
            # Update lines s_nom_opt
            for i, name in enumerate(device_names):
                if name in network.lines.index:
                    value = float(param_array_scaled[i])
                    network.lines.loc[name, "s_nom_opt"] = value
                    logger.info(f"  Line '{name}': s_nom_opt = {value:.2f} MVA")

    logger.info("=" * 60)
    return network


def _finalize_lp_only_result(
    *,
    config,
    method,
    sampler,
    parameter_names,
    initial_parameters,
    carrier_labels,
    relaxation_result,
    network_files,
    pypsa_args,
    timings,
    use_wandb,
    wandb_logger,
):
    """Build the result dict (and export network) for LP-only methods.

    Used by monolithic / relaxed / stochastic, which finish after Step 6 with
    no gradient-descent run.
    """
    optimal_params = relaxation_result["relaxed_parameters"]
    final_cost = float(relaxation_result["lower_bound"])

    logger.info("=" * 60)
    logger.info(f"RESULTS ({method})")
    logger.info("=" * 60)
    logger.info(f"Single-level optimum: {final_cost:.2f}")
    logger.info("Optimal capacities:")
    for name, value in optimal_params.items():
        logger.info(f"  {name}: {value}")

    name = config.get("name", "experiment")
    run_dir = datadir(name)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Optional network export
    export_config = config.get("export", {})
    export_path = None
    export_time = None
    if export_config.get("should_export", False):
        base_network_file = network_files[0] if network_files else None
        if base_network_file is None:
            logger.warning("No network files available for export. Skipping export.")
        else:
            logger.info("Exporting optimized capacities to PyPSA network...")
            t0 = time.time()
            exported_network = export_solved_network(
                pypsa_network_path=EXPERIMENT_DATA_PATH / base_network_file,
                devices=sampler.base_devices,
                parameter_names=parameter_names,
                optimal_params=optimal_params,
                pypsa_args=pypsa_args,
            )
            export_path = run_dir / "optimized_network.nc"
            exported_network.export_to_netcdf(str(export_path))
            export_time = time.time() - t0
            logger.info(f"Optimized network saved to {export_path} ({export_time:.2f}s)")

    if use_wandb and wandb_logger is not None:
        wandb.finish()

    result = {
        "config": config,
        "method": method,
        "final_cost": final_cost,
        "lower_bound": final_cost,  # LP-only: optimum is its own lower bound
        "initial_parameters": initial_parameters,
        "carrier_labels": carrier_labels,
        "optimal_parameters": {
            k: v.tolist() if hasattr(v, "tolist") else v for k, v in optimal_params.items()
        },
        "timing": timings,
        "sampler_summary": sampler.summary(),
    }
    if export_path is not None:
        result["export_path"] = str(export_path)
        result["timing"]["export"] = export_time

    return result


def _serialize_history(history: dict) -> dict:
    """Serialize optimization history for JSON output.

    Handles scalar trackers (loss, grad_norm, …) as well as the ``"param"``
    tracker whose entries are dicts mapping parameter names to numpy arrays.
    """
    serialized = {}
    for key, values in history.items():
        if key == "param":
            # Each entry is a dict {param_name: np.ndarray}
            serialized[key] = [
                {
                    pname: pval.tolist() if hasattr(pval, "tolist") else pval
                    for pname, pval in snapshot.items()
                }
                for snapshot in values
            ]
        else:
            serialized[key] = [
                float(x) if isinstance(x, (int, float, np.floating)) else x
                for x in values
            ]
    return serialized


def update_operation_objectives(problem, base_network, emissions_weight):
    """Update operation objectives on all subproblems with the given emissions weight.

    Swaps the ``operation_objective`` attribute on each subproblem in-place.
    This is cheap — it just creates Python wrapper objects without touching
    the CVXPY dispatch layer.

    Parameters
    ----------
    problem : StochasticPlanningProblem
        The stochastic planning problem whose subproblems will be updated.
    base_network : PowerNetwork
        The base zap PowerNetwork (needed by DispatchCostObjective).
    emissions_weight : float
        The current Lagrangian multiplier (carbon price) for emissions.
    """
    for sub in problem.subproblems:
        f_cost = DispatchCostObjective(base_network, sub.layer.devices)
        if emissions_weight > 0:
            f_emissions = emissions_weight * EmissionsObjective(sub.layer.devices)
            sub.operation_objective = f_cost + f_emissions
        else:
            sub.operation_objective = f_cost


def evaluate_emissions(problem, emissions_objectives, params):
    """Evaluate total emissions across all subproblems.

    Dispatches the full problem with the given parameters (no gradients),
    then evaluates the pre-built EmissionsObjective on each subproblem's
    dispatch state.  Emissions are weighted by subproblem weights and
    snapshot weights so the result is consistent with the Lagrangian cost.

    Parameters
    ----------
    problem : StochasticPlanningProblem
        The stochastic planning problem (workers should already be initialized).
    emissions_objectives : list[EmissionsObjective]
        One EmissionsObjective per subproblem (pre-built, reusable).
    params : dict
        Current investment parameters (e.g. from ``problem.solve()``).

    Returns
    -------
    float
        Total weighted emissions.
    """
    problem.forward(requires_grad=False, **params)

    total_emissions = 0.0
    for i, (sub, eo) in enumerate(zip(problem.subproblems, emissions_objectives)):
        emissions = eo(sub.state, parameters=sub.params, la=np)
        total_emissions += problem.weights[i] * sub.snapshot_weight * float(emissions)

    return total_emissions


def get_wandb_trackers(problem, sampler, relaxation_result, config):
    """Build extra wandb tracker functions for multi-year experiments.

    Parameters
    ----------
    problem : StochasticPlanningProblem
    sampler : MultiYearBlockSampler
    relaxation_result : dict or None
    config : dict

    Returns
    -------
    dict[str, callable]
    """
    emissions_objectives = [
        EmissionsObjective(sub.layer.devices) for sub in problem.subproblems
    ]
    cost_objectives = [
        DispatchCostObjective(sampler.base_network, sub.layer.devices)
        for sub in problem.subproblems
    ]

    def emissions_tracker(J, grad, params, last_state, prob):
        # Only evaluate subproblems in the current batch (others may not have state)
        batch = prob.batch
        states = [prob.subproblems[b].state for b in batch]
        parameters = [prob.subproblems[b].layer.setup_parameters(**params) for b in batch]
        return sum(
            emissions_objectives[b](s, parameters=p)
            for b, s, p in zip(batch, states, parameters)
        )

    def cost_tracker(J, grad, params, last_state, prob):
        batch = prob.batch
        states = [prob.subproblems[b].state for b in batch]
        parameters = [prob.subproblems[b].layer.setup_parameters(**params) for b in batch]
        return sum(
            cost_objectives[b](s, parameters=p)
            for b, s, p in zip(batch, states, parameters)
        )

    lower_bound = relaxation_result["lower_bound"] if relaxation_result else None

    trackers = {
        "emissions": emissions_tracker,
        "fuel_costs": cost_tracker,
        "inv_cost": lambda J, grad, params, last_state, prob: sum(
            prob.subproblems[b].inv_cost for b in prob.batch
        ),
        "op_cost": lambda J, grad, params, last_state, prob: sum(
            prob.subproblems[b].op_cost for b in prob.batch
        ),
        "batch": lambda J, grad, params, last_state, prob: prob.batch[0],
        "lower_bound": lambda *args: lower_bound if lower_bound is not None else np.nan,
    }

    # Full loss tracker (evaluated once per epoch)
    optimizer_config = config.get("optimizer", {})
    track_full_loss_every = optimizer_config.get("track_full_loss_every", 0)
    batch_size = optimizer_config.get("batch_size", 0)
    num_subs = problem.num_subproblems

    if batch_size == 0:
        batch_size = num_subs
    if track_full_loss_every == 0:
        track_full_loss_every = max(1, int(num_subs / batch_size))

    logger.info(f"Tracking full loss every {track_full_loss_every} batches.")

    _full_loss_cache = {"value": np.inf}

    def full_loss_tracker(J, grad, params, last_state, prob):
        iteration = prob.iteration
        if iteration > 0 and iteration % track_full_loss_every == 0:
            logger.info(f"Evaluating full loss at iteration {iteration}...")
            _full_loss_cache["value"] = prob(**params)
        return _full_loss_cache["value"]

    trackers["full_loss"] = full_loss_tracker

    # Capacity-by-group trackers (GW)
    pypsa_args = config.get("pypsa_args", {})
    power_unit = pypsa_args.get("power_unit", 1.0)

    gen_device = problem.subproblems[0].layer.devices[0]
    fuel_types = gen_device.fuel_type.reshape(-1)

    wind_mask = np.isin(fuel_types, ["onwind", "offwind", "offwind_floating"])
    solar_mask = fuel_types == "solar"
    thermal_mask = np.isin(fuel_types, ["CCGT", "OCGT", "coal", "oil", "nuclear", "CCGT-95CCS"])

    def _gen_cap_sum(params, mask=None):
        caps = np.asarray(params["generator_capacity"]).flatten() * power_unit / 1e3
        if mask is not None:
            caps = caps[mask]
        return float(np.sum(caps))

    trackers["wind_capacity_gw"] = lambda J, grad, params, *_a, _m=wind_mask: _gen_cap_sum(params, _m)
    trackers["solar_capacity_gw"] = lambda J, grad, params, *_a, _m=solar_mask: _gen_cap_sum(params, _m)
    trackers["thermal_capacity_gw"] = lambda J, grad, params, *_a, _m=thermal_mask: _gen_cap_sum(params, _m)

    return trackers


def run_experiment(config: dict) -> dict:
    """
    Run a multi-year stochastic planning experiment.

    Parameters
    ----------
    config : dict
        Experiment configuration with keys:
        - network_files: List of network filenames (optional, loads all if not specified)
        - hours_per_year: Hours per year (optional, uses all hours if not specified)
        - block_size: Hours per block (default 168 for weekly)
        - sampling_strategy: 'all', 'random', 'stratified', 'uniform'
        - num_blocks: Number of blocks (for random/stratified)
        - num_workers: Parallel workers for StochasticPlanningProblem
        - pypsa_args: Arguments for load_pypsa_network (optional)

    Returns
    -------
    dict
        Experiment results including optimal capacities and costs
    """
    # Method dispatch -- fail fast on unknown / stubbed methods.
    method = config.get("method", "gradient")
    if method not in _KNOWN_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of {sorted(_KNOWN_METHODS)}."
        )
    if method == "admm_gpu":
        raise NotImplementedError(
            "ADMM/GPU path not yet ported into multi_year/runner.py. "
            "See experiments/plan/runner.py:288-316 for the reference implementation."
        )

    # Default the run name to {network}_{method} when not user-overridden.
    if not config.get("name"):
        net_tag = config.get("network", "experiment")
        config["name"] = f"{net_tag}_{method}"

    logger.info("=" * 60)
    logger.info(f"Starting experiment: {config['name']} (method={method})")
    logger.info("=" * 60)

    # Extract config
    network_files = config.get("network_files", None)
    hours_per_year = config.get("hours_per_year", None)
    block_size = config.get("block_size", 168)
    sampling_strategy = config.get("sampling_strategy", "all")
    num_blocks = config.get("num_blocks", None)
    num_workers = config.get("num_workers", 1)
    seed = config.get("seed", 42)
    pypsa_args = config.get("pypsa_args", DEFAULT_PYPSA_ARGS)
    emissions_weight = config.get("emissions_weight", 0.0)
    emissions_target_config = config.get("emissions_target", {})
    adaptive_emissions = emissions_target_config.get("enabled", False)
    if adaptive_emissions and emissions_weight > 0:
        logger.warning(
            "Both emissions_weight > 0 and emissions_target.enabled are set. "
            "Adaptive emissions targeting takes precedence over emissions_weight."
        )

    # Wandb setup
    wandb_config = config.get("wandb", {})
    use_wandb = wandb_config.get("enabled", False)
    log_wandb_every = wandb_config.get("log_every", 1)
    wandb_logger = None

    if use_wandb:
        if wandb is None:
            logger.warning("wandb not installed. Skipping wandb logging.")
            use_wandb = False
        else:
            wandb.init(
                project=wandb_config.get("project", "zap-multiyear"),
                name=wandb_config.get("run_name", config.get("name", None)),
                tags=wandb_config.get("tags", []),
                config=config,
            )
            wandb_logger = wandb
            logger.info(f"Wandb initialized (project={wandb_config.get('project', 'zap-multiyear')})")

    # -------------------------------------------------------------------------
    # Step 1: Load multi-year networks
    # -------------------------------------------------------------------------
    logger.info("Loading multi-year networks...")
    start_time = time.time()

    networks, snapshots_list = load_multi_year_networks(
        network_files=network_files,
        hours_per_year=hours_per_year,
    )

    load_time = time.time() - start_time
    logger.info(f"Network loading took {load_time:.2f}s")

    # -------------------------------------------------------------------------
    # Step 2: Create MultiYearBlockSampler
    # -------------------------------------------------------------------------
    logger.info("Creating MultiYearBlockSampler...")
    start_time = time.time()

    sampler = MultiYearBlockSampler(
        networks,
        snapshots_list,
        **pypsa_args,
    )

    logger.info(f"Sampler summary: {sampler.summary()}")
    sampler_time = time.time() - start_time
    logger.info(f"Sampler creation took {sampler_time:.2f}s")

    # -------------------------------------------------------------------------
    # Step 3: Sample blocks
    # -------------------------------------------------------------------------
    # block_size: null is the "monolithic" sentinel: one block covering the full
    # horizon. Resolve it now that the sampler knows total_hours.
    if block_size is None:
        block_size = sampler.total_hours
        logger.info(f"block_size=null -> using full horizon ({block_size}h, 1 block)")

    logger.info(f"Sampling blocks with strategy='{sampling_strategy}'...")

    blocks = sampler.sample_blocks(
        block_size=block_size,
        num_blocks=num_blocks,
        strategy=sampling_strategy,
        seed=seed,
    )

    logger.info(f"Sampled {len(blocks)} blocks of {block_size} hours each")
    logger.info(f"Total hours covered: {len(blocks) * block_size}")

    # -------------------------------------------------------------------------
    # Step 4: Setup parameter names and bounds for planning
    # -------------------------------------------------------------------------
    parameter_names = setup_parameter_names(sampler.base_devices)
    logger.info(f"Planning parameters: {list(parameter_names.keys())}")

    power_unit = pypsa_args.get("power_unit", 1.0)
    lower_bounds, upper_bounds = setup_bounds(
        sampler.base_devices, parameter_names, power_unit=power_unit
    )
    for p, (device_idx, attr_name) in parameter_names.items():
        logger.info(
            f"  {p}: lower=[{lower_bounds[p].min():.1f}, {lower_bounds[p].max():.1f}], "
            f"upper=[{upper_bounds[p].min():.1f}, {upper_bounds[p].max():.1f}]"
        )
        dev = sampler.base_devices[device_idx]
        names = np.asarray(dev.name).reshape(-1) if hasattr(dev.name, '__len__') else np.array([dev.name])
        existing_mask = np.array(["existing" in str(n) for n in names])
        lb_flat = lower_bounds[p].flatten()
        ub_flat = upper_bounds[p].flatten()
        if hasattr(dev, "fuel_type"):
            fuel_types = np.asarray(dev.fuel_type).reshape(-1)
            for ft in sorted(set(fuel_types)):
                ft_mask = fuel_types == ft
                lb = lb_flat[ft_mask]
                ub = ub_flat[ft_mask]
                logger.info(
                    f"    {ft} ({ft_mask.sum()}): "
                    f"lower=[{lb.min():.1f}, {lb.max():.1f}], "
                    f"upper=[{ub.min():.1f}, {ub.max():.1f}]"
                )
                ft_existing = ft_mask & existing_mask
                if ft_existing.any():
                    lb_ex = lb_flat[ft_existing]
                    ub_ex = ub_flat[ft_existing]
                    logger.info(
                        f"      existing ({ft_existing.sum()}): "
                        f"lower=[{lb_ex.min():.1f}, {lb_ex.max():.1f}], "
                        f"upper=[{ub_ex.min():.1f}, {ub_ex.max():.1f}]"
                    )
        else:
            if existing_mask.any():
                lb_ex = lb_flat[existing_mask]
                ub_ex = ub_flat[existing_mask]
                logger.info(
                    f"    existing ({existing_mask.sum()}): "
                    f"lower=[{lb_ex.min():.1f}, {lb_ex.max():.1f}], "
                    f"upper=[{ub_ex.min():.1f}, {ub_ex.max():.1f}]"
                )

    # Capture initial (pre-optimization) capacities for plotting
    initial_parameters = {}
    for param_name, (device_idx, attr_name) in parameter_names.items():
        cap = getattr(sampler.base_devices[device_idx], attr_name)
        initial_parameters[param_name] = cap.tolist() if hasattr(cap, "tolist") else cap

    # Capture carrier labels for each parameter so plots don't need the network file
    carrier_labels = {}
    for param_name, (device_idx, attr_name) in parameter_names.items():
        dev = sampler.base_devices[device_idx]
        if hasattr(dev, "fuel_type"):
            carrier_labels[param_name] = dev.fuel_type.reshape(-1).tolist()
        elif hasattr(dev, "name") and hasattr(dev.name, "tolist"):
            carrier_labels[param_name] = dev.name.tolist()
    if carrier_labels:
        logger.info(f"Saved carrier labels for: {list(carrier_labels.keys())}")

    # -------------------------------------------------------------------------
    # Step 5: Create StochasticPlanningProblem
    # -------------------------------------------------------------------------
    logger.info("Creating StochasticPlanningProblem...")
    start_time = time.time()

    # Get dispatch solver from config
    dispatch_solver_name = config.get("dispatch_solver", DEFAULT_DISPATCH_SOLVER)
    dispatch_solver = getattr(cp, dispatch_solver_name)
    dispatch_solver_kwargs = config.get("dispatch_solver_kwargs", DEFAULT_DISPATCH_SOLVER_KWARGS)
    logger.info(f"Using dispatch solver: {dispatch_solver_name}")
    if emissions_weight > 0:
        logger.info(f"Emissions penalty: {emissions_weight} $/ton CO2")
    else:
        logger.info("No emissions penalty (emissions_weight=0)")

    def op_objective_fn(devices):
        f_cost = DispatchCostObjective(sampler.base_network, devices)
        if emissions_weight > 0:
            f_emissions = emissions_weight * EmissionsObjective(devices)
            return f_cost + f_emissions
        return f_cost

    def inv_objective_fn(devices, layer):
        return InvestmentObjective(devices, layer)

    problem = sampler.create_stochastic_problem(
        blocks=blocks,
        parameter_names=parameter_names,
        operation_objective_fn=op_objective_fn,
        investment_objective_fn=inv_objective_fn,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        solver=dispatch_solver,
        solver_kwargs=dispatch_solver_kwargs,
    )

    problem_time = time.time() - start_time
    logger.info(f"Problem creation took {problem_time:.2f}s")
    logger.info(f"Created problem with {problem.num_subproblems} subproblems")

    # -------------------------------------------------------------------------
    # Step 6: Solve monolithic single-level LP (for lower bound and initialization)
    # The monolithic value is the exact LP optimum when planning and dispatch
    # objectives match (true here -- both use DispatchCostObjective +
    # emissions_weight * EmissionsObjective). It also serves as a lower bound
    # for the decomposed gradient run.
    # Note: must happen BEFORE initializing parallel workers, because
    # MonolithicPlanningProblem deepcopies the problem and worker pools
    # (which contain SimpleQueue objects) cannot be pickled.
    # -------------------------------------------------------------------------
    relaxation_config = config.get("relaxation", {})
    # The `relaxed` method requires Step 6 regardless of the should_solve flag;
    # otherwise honor the config setting.
    should_solve_relaxation = (
        method in _LP_ONLY_METHODS or relaxation_config.get("should_solve", True)
    )
    # `relaxation.kind` chooses between MonolithicPlanningProblem (default) and
    # the strong-duality RelaxedPlanningProblem. The `relaxed` method pins kind.
    relaxation_kind = (
        "strong_duality" if method == "relaxed"
        else relaxation_config.get("kind", "monolithic")
    )

    relaxation_result = None
    if should_solve_relaxation:
        kind_label = "strong-duality" if relaxation_kind == "strong_duality" else "monolithic"
        logger.info(f"Solving {kind_label} single-level problem...")
        start_time = time.time()

        relaxation_solver_name = relaxation_config.get("solver", DEFAULT_RELAXATION_SOLVER)
        relaxation_solver = getattr(cp, relaxation_solver_name)
        relaxation_solver_kwargs = relaxation_config.get(
            "solver_kwargs", DEFAULT_RELAXATION_SOLVER_KWARGS
        )
        logger.info(f"Using single-level solver: {relaxation_solver_name}")

        # Optionally subsample blocks for a smaller single-level problem
        num_relaxation_subproblems = relaxation_config.get("num_subproblems", None)
        if num_relaxation_subproblems is not None and num_relaxation_subproblems < len(blocks):
            logger.info(
                f"Subsampling single-level problem: {num_relaxation_subproblems} blocks "
                f"(out of {len(blocks)} total)"
            )
            relaxation_blocks = sampler.sample_blocks(
                block_size=block_size,
                num_blocks=num_relaxation_subproblems,
                strategy="uniform",
            )
            relaxation_problem = sampler.create_stochastic_problem(
                blocks=relaxation_blocks,
                parameter_names=parameter_names,
                operation_objective_fn=op_objective_fn,
                investment_objective_fn=inv_objective_fn,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                solver=dispatch_solver,
                solver_kwargs=dispatch_solver_kwargs,
            )
            logger.info(
                f"Solving stochastic single-level problem with "
                f"{relaxation_problem.num_subproblems} scenarios"
            )
        else:
            relaxation_problem = problem

        if relaxation_kind == "strong_duality":
            single_level = RelaxedPlanningProblem(
                relaxation_problem,
                max_price=relaxation_config.get("price_bound", 100.0),
                solver=relaxation_solver,
                solver_kwargs=relaxation_solver_kwargs,
            )
        else:
            single_level = MonolithicPlanningProblem(
                relaxation_problem,
                solver=relaxation_solver,
                solver_kwargs=relaxation_solver_kwargs,
            )
        relaxed_params, relax_solve_data = single_level.solve()

        relax_time = time.time() - start_time
        logger.info(f"Single-level solve took {relax_time:.2f}s")

        lower_bound = relax_solve_data["problem"].value
        if lower_bound is None:
            logger.warning(
                f"Single-level solve failed (status: {relax_solve_data['problem'].status}). "
                "Proceeding without lower bound or warm-start."
            )
            relaxation_result = None
        else:
            logger.info(
                f"Single-level optimum (lower bound for decomposed run): {lower_bound:.2f}"
            )
            relaxation_result = {
                "relaxed_parameters": relaxed_params,
                "lower_bound": lower_bound,
                "solve_time": relax_time,
            }
    else:
        logger.info("Skipping single-level solve...")
        relax_time = 0.0

    # -------------------------------------------------------------------------
    # LP-only short-circuit: monolithic / relaxed / stochastic methods are done
    # after Step 6. Build a minimal result, export the network, return.
    # -------------------------------------------------------------------------
    if method in _LP_ONLY_METHODS:
        if relaxation_result is None:
            raise RuntimeError(
                f"Method '{method}' requires the single-level solve to succeed, "
                f"but it failed. Check solver logs."
            )
        return _finalize_lp_only_result(
            config=config,
            method=method,
            sampler=sampler,
            parameter_names=parameter_names,
            initial_parameters=initial_parameters,
            carrier_labels=carrier_labels,
            relaxation_result=relaxation_result,
            network_files=network_files,
            pypsa_args=pypsa_args,
            timings={
                "network_loading": load_time,
                "sampler_creation": sampler_time,
                "problem_creation": problem_time,
                "single_level_solve": relax_time,
            },
            use_wandb=use_wandb,
            wandb_logger=wandb_logger,
        )

    # -------------------------------------------------------------------------
    # Step 7: Initialize parallel workers (after relaxation to avoid deepcopy issues)
    # -------------------------------------------------------------------------
    if num_workers > 1:
        logger.info(f"Initializing {num_workers} parallel workers...")
        problem.initialize_workers(num_workers)

    # -------------------------------------------------------------------------
    # Step 8: Solve the actual problem with gradient descent
    # -------------------------------------------------------------------------
    optimizer_config = config.get("optimizer", {})
    num_iterations = optimizer_config.get("num_iterations", 100)
    step_size = optimizer_config.get("step_size", 1e-3)
    clip = optimizer_config.get("clip", 1e3)
    initial_state_source = optimizer_config.get("initial_state", "relaxation")
    batch_size = optimizer_config.get("batch_size", 0)
    batch_strategy = optimizer_config.get("batch_strategy", "sequential")
    init_full_loss = optimizer_config.get("init_full_loss", True)
    peak_net_load_k = optimizer_config.get("peak_net_load_k", None)
    peak_net_load_rerank_every = optimizer_config.get("peak_net_load_rerank_every", 1)

    logger.info("Solving with gradient descent...")
    logger.info(f"  num_iterations: {num_iterations}")
    logger.info(f"  step_size: {step_size}")
    logger.info(f"  clip: {clip}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  batch_strategy: {batch_strategy}")
    if peak_net_load_k is not None:
        logger.info(f"  peak_net_load_k: {peak_net_load_k}")
        logger.info(f"  peak_net_load_rerank_every: {peak_net_load_rerank_every}")

    # Initialize from relaxation or from scratch
    if initial_state_source == "relaxation" and relaxation_result is not None:
        logger.info("Initializing with relaxation solution.")
        initial_state = deepcopy(relaxation_result["relaxed_parameters"])
    else:
        logger.info("Initializing with default parameters (no investment).")
        initial_state = None

    # Create algorithm
    algorithm = GradientDescent(step_size=step_size, clip=clip)

    # Build tracker list (always include param trajectory for post-hoc analysis)
    save_param_history = optimizer_config.get("save_param_history", True)
    trackers_list = list(tr.DEFAULT_TRACKERS)
    if save_param_history:
        trackers_list.append(tr.PARAM)
        logger.info("  Saving parameter trajectory (save_param_history=true)")

    # Build wandb trackers if enabled
    extra_wandb_trackers = None
    if use_wandb:
        extra_wandb_trackers = get_wandb_trackers(
            problem, sampler, relaxation_result, config
        )

    emissions_target_result = None
    outer_histories = None

    if not adaptive_emissions:
        # =================================================================
        # Non-adaptive path: single gradient descent solve (existing behavior)
        # =================================================================
        start_time = time.time()

        optimal_params, history = problem.solve(
            num_iterations=num_iterations,
            algorithm=algorithm,
            trackers=trackers_list,
            initial_state=initial_state,
            lower_bound=relaxation_result["lower_bound"] if relaxation_result else None,
            batch_size=batch_size,
            batch_strategy=batch_strategy,
            wandb=wandb_logger,
            log_wandb_every=log_wandb_every,
            extra_wandb_trackers=extra_wandb_trackers,
            verbosity=1,
            init_full_loss=init_full_loss,
            peak_net_load_k=peak_net_load_k,
            peak_net_load_rerank_every=peak_net_load_rerank_every,
        )

        solve_time = time.time() - start_time
        logger.info(f"Gradient descent took {solve_time:.2f}s")

    else:
        # =================================================================
        # Adaptive emissions targeting via dual ascent outer loop
        # =================================================================
        target = emissions_target_config["target"]
        initial_weight = emissions_target_config.get("initial_weight", 0.0)
        dual_step_size = emissions_target_config.get("dual_step_size", 1.0)
        max_weight = emissions_target_config.get("max_weight", 1000.0)
        num_outer_iterations = emissions_target_config.get("num_outer_iterations", 10)
        tolerance = emissions_target_config.get("tolerance", 0.05)

        logger.info("=" * 60)
        logger.info("Adaptive emissions targeting (dual ascent)")
        logger.info(f"  target: {target}")
        logger.info(f"  initial_weight: {initial_weight}")
        logger.info(f"  dual_step_size: {dual_step_size}")
        logger.info(f"  max_weight: {max_weight}")
        logger.info(f"  num_outer_iterations: {num_outer_iterations}")
        logger.info(f"  tolerance: {tolerance}")
        logger.info("=" * 60)

        # Build per-subproblem EmissionsObjective list (one-time, cheap)
        emissions_objectives = [
            EmissionsObjective(sub.layer.devices) for sub in problem.subproblems
        ]

        current_params = initial_state
        current_lambda = initial_weight

        lambda_history = []
        emissions_history = []
        outer_histories = []
        converged = False

        start_time = time.time()

        for outer_iter in range(num_outer_iterations):
            logger.info("=" * 60)
            logger.info(
                f"Outer iteration {outer_iter + 1}/{num_outer_iterations} "
                f"| lambda={current_lambda:.4f}"
            )
            logger.info("=" * 60)

            # 1. Swap objectives on all subproblems with current lambda
            update_operation_objectives(problem, sampler.base_network, current_lambda)

            # 2. Inner solve (warm-started from current_params)
            #    LP relaxation bound is only valid for the initial lambda
            lb = (
                relaxation_result["lower_bound"]
                if (relaxation_result and outer_iter == 0)
                else None
            )

            optimal_params, history = problem.solve(
                num_iterations=num_iterations,
                algorithm=algorithm,
                trackers=trackers_list,
                initial_state=current_params,
                lower_bound=lb,
                batch_size=batch_size,
                batch_strategy=batch_strategy,
                wandb=wandb_logger,
                log_wandb_every=log_wandb_every,
                extra_wandb_trackers=extra_wandb_trackers,
                verbosity=1,
                init_full_loss=init_full_loss,
                peak_net_load_k=peak_net_load_k,
                peak_net_load_fill=peak_net_load_fill,
                peak_net_load_rerank_every=peak_net_load_rerank_every,
            )

            outer_histories.append(_serialize_history(history))

            # 3. Evaluate emissions (full-batch forward + per-subproblem evaluation)
            total_emissions = evaluate_emissions(
                problem, emissions_objectives, optimal_params
            )

            # 4. Log progress
            gap = (
                (total_emissions - target) / abs(target)
                if target != 0
                else float("inf")
            )
            logger.info(f"  Emissions: {total_emissions:.2f}")
            logger.info(f"  Target:    {target:.2f}")
            logger.info(f"  Gap:       {gap * 100:.2f}%")
            logger.info(f"  Lambda:    {current_lambda:.4f}")

            lambda_history.append(float(current_lambda))
            emissions_history.append(float(total_emissions))

            # 5. Check convergence
            if abs(gap) < tolerance:
                logger.info(
                    f"  Converged! |gap| {abs(gap) * 100:.2f}% "
                    f"< tolerance {tolerance * 100:.2f}%"
                )
                converged = True
                break

            # 6. Dual update: subgradient step, clamped to [0, max_weight]
            current_lambda = max(
                0.0, current_lambda + dual_step_size * (total_emissions - target)
            )
            current_lambda = min(current_lambda, max_weight)
            logger.info(f"  Updated lambda: {current_lambda:.4f}")

            # 7. Carry state for warm-start
            current_params = optimal_params

        solve_time = time.time() - start_time
        logger.info(f"Dual ascent took {solve_time:.2f}s")
        logger.info(
            f"Completed {len(emissions_history)} outer iterations, "
            f"converged={converged}"
        )

        emissions_target_result = {
            "target": float(target),
            "final_emissions": emissions_history[-1],
            "final_lambda": float(current_lambda),
            "lambda_history": lambda_history,
            "emissions_history": emissions_history,
            "num_outer_iterations_completed": len(emissions_history),
            "converged": converged,
        }

    # -------------------------------------------------------------------------
    # Step 9: Extract results
    # -------------------------------------------------------------------------
    eval_final_full_loss = config.get("eval_final_full_loss", False)

    if eval_final_full_loss:
        final_cost = float(problem.forward(**optimal_params))
    else:
        final_cost = float(history["loss"][-1])

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Final cost{'' if eval_final_full_loss else ' (last batch)'}: {final_cost:.2f}")
    if relaxation_result:
        logger.info(f"Lower bound: {relaxation_result['lower_bound']:.2f}")
        logger.info(
            f"Optimality gap (approx): {(final_cost - relaxation_result['lower_bound']) / relaxation_result['lower_bound'] * 100:.2f}%"
        )
    logger.info("Optimal capacities:")
    for name, value in optimal_params.items():
        logger.info(f"  {name}: {value}")

    # -------------------------------------------------------------------------
    # Step 10: Export solved network to PyPSA (optional)
    # -------------------------------------------------------------------------
    export_config = config.get("export", {})
    should_export = export_config.get("should_export", False)
    exported_network = None
    export_path = None

    if should_export:
        logger.info("Exporting optimized capacities to PyPSA network...")
        start_time = time.time()

        # Use the first network file as the base
        base_network_file = network_files[0] if network_files else None
        if base_network_file is None:
            logger.warning("No network files available for export. Skipping export.")
        else:
            base_network_path = EXPERIMENT_DATA_PATH / base_network_file

            exported_network = export_solved_network(
                pypsa_network_path=base_network_path,
                devices=sampler.base_devices,
                parameter_names=parameter_names,
                optimal_params=optimal_params,
                pypsa_args=pypsa_args,
            )

            export_time = time.time() - start_time
            logger.info(f"Export took {export_time:.2f}s")

            # Save the exported network under outputs/{name}/.
            experiment_name = config.get("name", "experiment")
            run_dir = datadir(experiment_name)
            run_dir.mkdir(parents=True, exist_ok=True)
            export_path = run_dir / "optimized_network.nc"
            exported_network.export_to_netcdf(str(export_path))
            logger.info(f"Optimized network saved to {export_path}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    if use_wandb and wandb_logger is not None:
        wandb.finish()

    if num_workers > 1:
        problem.shutdown_workers()

    result = {
        "config": config,
        "num_subproblems": problem.num_subproblems,
        "final_cost": float(final_cost),
        "lower_bound": float(relaxation_result["lower_bound"]) if relaxation_result else None,
        "initial_parameters": initial_parameters,
        "carrier_labels": carrier_labels,
        "optimal_parameters": {
            k: v.tolist() if hasattr(v, "tolist") else v for k, v in optimal_params.items()
        },
        "history": _serialize_history(history),
        "timing": {
            "network_loading": load_time,
            "sampler_creation": sampler_time,
            "problem_creation": problem_time,
            "relaxation_solve": relax_time,
            "gradient_descent": solve_time,
        },
        "sampler_summary": sampler.summary(),
    }

    if should_export:
        result["export_path"] = str(export_path) if export_path else None
        result["timing"]["export"] = export_time

    if emissions_target_result is not None:
        result["emissions_target"] = emissions_target_result
        result["outer_histories"] = outer_histories

    return result


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the experiment.

    Usage:
        python runner.py NETWORK.yaml METHOD.yaml [OVERRIDE.yaml ...]

    Each YAML is deep-merged in order (later files override earlier). The typical
    invocation passes one network config + one method config:

        python runner.py configs/networks/normal.yaml configs/methods/gradient.yaml
    """
    import sys

    if len(sys.argv) <= 1:
        raise SystemExit(
            "Usage: python runner.py NETWORK.yaml METHOD.yaml [OVERRIDE.yaml ...]"
        )

    config = load_and_merge_configs(sys.argv[1:])

    # Run experiment
    results = run_experiment(config)

    # Save results under outputs/{name}/results.json
    name = config.get("name", "experiment")
    run_dir = datadir(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
