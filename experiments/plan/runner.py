import torch
import numpy as np
import pandas as pd
import datetime as dt
import cvxpy as cp
import sys
import pypsa
import wandb
import yaml
import json
import platform

from pathlib import Path
from copy import deepcopy
from scipy.stats import rankdata

import zap
import zap.planning.trackers as tr
from zap.admm import ADMMSolver, ADMMLayer


ZAP_PATH = Path(zap.__file__).parent.parent
DATA_PATH = ZAP_PATH / "data"

PARAMETERS = {
    "generator": (zap.Generator, "nominal_capacity"),
    "dc_line": (zap.DCLine, "nominal_capacity"),
    "ac_line": (zap.ACLine, "nominal_capacity"),
    "battery": (zap.Battery, "power_capacity"),
}

ALGORITHMS = {
    "gradient_descent": zap.planning.GradientDescent,
}

MOSEK_THREADS = 32
UTC_TIME_SHIFT = 7
TOTAL_PYPSA_HOUR = 8760 - 24
PYPSA_START_DAY = dt.datetime(2019, 1, 1, UTC_TIME_SHIFT)
PYPSA_DEFAULT_ARGS = {
    "power_unit": 1.0e3,
    "cost_unit": 10.0,
    "marginal_load_value": 500.0,
    "scale_load": 1.5,
    "scale_generator_capacity_factor": 0.8,
    "scale_line_capacity_factor": 0.8,
    "carbon_tax": 0.0,
    "generator_cost_perturbation": 1.0,
    "load_cost_perturbation": 10.0,
    "drop_empty_generators": False,
    "expand_empty_generators": 0.5,
}

CVX_LAYER_ARGS = {
    "solver": cp.MOSEK,
    "solver_kwargs": {
        "verbose": False,
        "accept_unknown": True,
        "mosek_params": {
            "MSK_IPAR_NUM_THREADS": MOSEK_THREADS,
        },
    },
    "add_ground": False,
}

TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
}


def expand_config(config: dict, key="expand") -> list[dict]:
    configs = _expand_config(config, key=key)

    # Tag every config with an index
    for i, c in enumerate(configs):
        c["index"] = i

    return configs


def _expand_config(config: dict, key="expand") -> list[dict]:
    """Expand config with multiple values for a single parameter.

    Lists of the form `param: [expand, arg1, arg2, ..., argn]` will be expanded into
    `n` distinct configurations.
    """
    config = deepcopy(config)

    # Expand sub-dictionaries
    for k, v in config.items():
        if isinstance(v, dict):
            v = _expand_config(v, key=key)
            if len(v) == 1:
                config[k] = v[0]
            else:
                config[k] = [key] + v

    # Expand lists
    for k, v in config.items():
        if isinstance(v, list) and len(v) > 0 and v[0] == key:
            data: list = v[1:]
            assert isinstance(data, list)

            expanded_configs = [deepcopy(config) for _ in range(len(data))]

            for i, d in enumerate(data):
                expanded_configs[i][k] = d

            return sum(
                [_expand_config(e, key=key) for e in expanded_configs], start=[]
            )  # Join lists

    # No expansion happened, just return a singleton list
    return [config]


def load_config(path):
    path = Path(path)

    if not path.exists():
        path = ZAP_PATH / path

    path = path.absolute()
    assert path.exists()
    print(path)

    # Open config
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Tag config from name
    config_short_name = Path(path).name.split(".")[0]
    config_id = config_short_name  # + "_000"

    # # Check if name has already been used
    # while get_results_path(config_id).exists():
    #     config_id = config_id[:-3] + f"{int(config_id[-3:]) + 1:03d}"

    config["name"] = config_short_name
    config["id"] = config_id

    # Hunt for improperly parsed scientific notation (bug in Python yaml parser)
    config = parse_scientific_notation(config)

    return config


def load_dataset(
    name="pypsa",
    battery_cost_scale=1.0,
    generator_cost_scale={},
    dont_expand=[],
    reconductoring_cost=1.0,
    reconductoring_threshold=1.4,
    scale_by_hours=True,
    **kwargs,
):
    print("Loading dataset...")

    print(name)

    if name == "pypsa":
        net, devices = setup_pypysa_dataset(**kwargs)

        if scale_by_hours:
            print("Rescaling costs by number of hours.")
            for d in devices:
                d.scale_costs(kwargs["num_hours"])  # Divide costs by number of hours
        else:
            print("Not rescaling costs by number of hours.")

    else:
        raise ValueError("Unknown dataset")

    # Scale capital costs
    for d in devices:
        if isinstance(d, zap.Battery):
            print(f"Scaling battery capital costs by {100*battery_cost_scale:.2f} %.")
            d.capital_cost *= battery_cost_scale

        if isinstance(d, zap.Generator):
            # Scale by fuel type
            for fuel in generator_cost_scale.keys():
                print(
                    f"Scaling generator capital costs for {fuel} by {100*generator_cost_scale[fuel]:.2f} %."
                )
                d.capital_cost[d.fuel_type == fuel] *= generator_cost_scale[fuel]

            # Set pnom max to pnom
            for fuel in dont_expand:
                print("Setting max power to nominal power for", fuel)
                d.max_nominal_capacity[d.fuel_type == fuel] = d.nominal_capacity[
                    d.fuel_type == fuel
                ]

        if isinstance(d, zap.ACLine):
            if reconductoring_cost != 1.0:
                print("Setting reconductoring cost and threshold.")
                assert 0.0 <= reconductoring_cost <= 1.0
                assert 1.0 <= reconductoring_threshold

                # This is an absolute quantity
                d.reconductoring_cost = reconductoring_cost * d.capital_cost
                # This is a relative quantity
                d.reconductoring_threshold = reconductoring_threshold * np.ones_like(d.capital_cost)

    return {
        "net": net,
        "devices": devices,
    }


def setup_pypysa_dataset(
    add_ground=True,
    use_batteries=True,
    use_extra_components=True,
    num_nodes=100,
    start_hour="peak_load_day",
    num_hours=4,
    args=None,
    case="load_medium",
    **kwargs,
):
    print(args)
    # Load pypsa file
    csv_dir = f"{case}/elec_s_{num_nodes}"
    if use_extra_components:
        csv_dir += "_ec"

    pn = pypsa.Network()
    pn.import_from_csv_folder(DATA_PATH / "pypsa/western/" / csv_dir)

    # Filter out extra components (battery nodes, links, and stores)
    # No longer needed with new pypsa dataset
    # pn.buses = pn.buses[~pn.buses.index.str.contains("battery")]
    # pn.links = pn.links[~pn.links.index.str.contains("battery")]

    # Pick dates
    # Rule 1 - Just a fixed hour of the year
    if isinstance(start_hour, int):
        print("Using a fixed start hour.")
        hours = np.array(range(start_hour, start_hour + num_hours))

    # Rule 2 - Dynamically selected by peak load
    elif start_hour == "peak_load_day":
        print("Finding the peak load day.")
        hours = sort_hours_by_peak(pn, args, by="load", period=24, reducer=np.max)
        hours = hours[:num_hours]

    elif start_hour == "peak_renewable_day":
        print("Finding the peak renewable day.")
        hours = sort_hours_by_peak(pn, args, by="renewable", period=24, reducer=np.max)
        hours = hours[:num_hours]

    elif start_hour == "peak_hybrid_day":
        print("Finding the peak hybrid (mix of load and renewables) day.")
        hours = sort_hours_by_peak(pn, args, by="hybrid", period=24, reducer=np.max)
        hours = hours[:num_hours]

    else:  # Rule 3 - Specify explicit date
        start_hour = dt.datetime.combine(start_hour, dt.time(hour=UTC_TIME_SHIFT))
        hours = pd.date_range(start_hour, periods=num_hours, freq="1h", inclusive="left")

    # Build zap network
    if not isinstance(hours, pd.DatetimeIndex):
        hours = [PYPSA_START_DAY + dt.timedelta(hours=int(h)) for h in hours]
        hours = pd.DatetimeIndex(hours)

    print(f"Solving a case with {len(hours)} hours.")
    print(hours)
    net, devices = zap.importers.load_pypsa_network(pn, hours, **args)

    if (not use_batteries) or (num_hours == 1):
        devices = [d for d in devices if type(d) is not zap.Battery]

    if add_ground:
        ground = zap.Ground(
            num_nodes=net.num_nodes, terminal=np.array([0]), voltage=np.array([0.0])
        )
        devices += [ground]

    return net, devices


def layer_function(
    net,
    parameter_names,
    use_admm,
    adapt_rho,
    adapt_rho_rate,
    torch_dtype,
    linear_solver,
    admm_args: dict,
):
    if use_admm:
        print("Using ADMM layer.")

        def f(devices, time_horizon):
            return ADMMLayer(
                net,
                devices,
                parameter_names,
                time_horizon=time_horizon,
                solver=ADMMSolver(**admm_args, dtype=torch_dtype),
                adapt_rho=adapt_rho,
                adapt_rho_rate=adapt_rho_rate,
            )

    else:
        print("Using CVX layer.")

        def f(devices, time_horizon):
            return zap.DispatchLayer(
                net,
                devices,
                parameter_names,
                time_horizon=time_horizon,
                linear_solver=linear_solver,
                **CVX_LAYER_ARGS,
            )

    return f


def setup_problem(
    net,
    devices: list[zap.devices.AbstractDevice],
    parameters=["generator", "dc_line", "ac_line", "battery"],
    cost_weight=1.0,
    emissions_weight=0.0,
    regularize=0.0,
    stochastic=False,
    hours_per_scenario=1,
    use_admm=False,
    adapt_rho=False,
    adapt_rho_rate=0.1,
    torch_dtype="float32",
    linear_solver="scipy",
    args={},
):
    print("Building planning problem...")
    print("ADMM is enabled." if use_admm else "ADMM is disabled.")
    time_horizon = np.max([d.time_horizon for d in devices])

    torch_dtype = TORCH_DTYPES[torch_dtype]

    # Drop batteries
    if stochastic and hours_per_scenario == 1:
        devices = [d for d in devices if type(d) is not zap.Battery]

    if use_admm:
        devices = [d.torchify(machine=args["machine"], dtype=torch_dtype) for d in devices]

    # Setup parameters
    parameter_names = {}
    for dev in parameters:
        d_type, param_field = PARAMETERS[dev]
        d_index = device_index(devices, d_type)

        if d_index is not None:
            parameter_names[dev] = d_index, param_field
        else:
            print(f"Warning: device {dev} not found in devices. Will not be expanded.")

    # Setup layer
    layer_map = layer_function(
        net, parameter_names, use_admm, adapt_rho, adapt_rho_rate, torch_dtype, linear_solver, args
    )
    layer = layer_map(devices, time_horizon)

    # Build objective
    def make_objective(devs):
        f_cost = cost_weight * zap.planning.DispatchCostObjective(net, devs)
        f_emissions = emissions_weight * zap.planning.EmissionsObjective(devs)
        return f_cost + f_emissions

    op_objective = make_objective(devices)
    inv_objective = zap.planning.InvestmentObjective(devices, layer)

    # Setup planning problem
    problem_args = {"lower_bounds": None, "upper_bounds": None}
    if not use_admm:
        problem_args["regularize"] = regularize

    problem = zap.planning.PlanningProblem(
        operation_objective=op_objective,
        investment_objective=inv_objective,
        layer=deepcopy(layer),
        **problem_args,
    )

    # Setup stochastic problem
    if stochastic:
        assert hours_per_scenario >= 1
        assert time_horizon % hours_per_scenario == 0

        scenarios = [
            range(i, i + hours_per_scenario) for i in range(0, time_horizon, hours_per_scenario)
        ]
        sub_devices = [[d.sample_time(s, time_horizon) for d in devices] for s in scenarios]
        sub_layers = [layer_map(d, hours_per_scenario) for d in sub_devices for d in sub_devices]

        sub_op_objectives = [make_objective(d) for d in sub_devices]
        sub_inv_objectives = [
            zap.planning.InvestmentObjective(d, lay) for d, lay in zip(sub_devices, sub_layers)
        ]

        sub_problems = [
            zap.planning.PlanningProblem(
                operation_objective=o,
                investment_objective=i,
                layer=lay,
                **problem_args,
            )
            for o, i, lay in zip(sub_op_objectives, sub_inv_objectives, sub_layers)
        ]

        stochastic_problem = zap.planning.StochasticPlanningProblem(sub_problems)

    else:
        stochastic_problem = None
        sub_devices = None

    return {
        "problem": problem,
        "layer": layer,
        "stochastic_problem": stochastic_problem,
        "sub_devices": sub_devices,
    }


def solve_relaxed_problem(problem, *, should_solve=True, price_bound=50.0, inf_value=50.0):
    if problem["stochastic_problem"] is not None:
        problem = problem["stochastic_problem"]
    else:
        problem = problem["problem"]

    if not should_solve:
        print("Skipping relaxation...")
        return None

    else:
        print("Solving relaxation...")
        relaxation = zap.planning.RelaxedPlanningProblem(
            problem,
            max_price=price_bound,
            inf_value=inf_value,
            sd_tolerance=1e-3,
            solver=cp.MOSEK,
            solver_kwargs={
                "verbose": True,
                "accept_unknown": True,
                "mosek_params": {
                    "MSK_IPAR_NUM_THREADS": MOSEK_THREADS,
                    "MSK_IPAR_INTPNT_BASIS": 0,
                },
            },
        )

        relaxed_parameters, data = relaxation.solve()
        print(f"Solved relaxation in {data['problem'].solver_stats.solve_time / 60:.2f} minutes.")
        print("Lower bound: ", data["problem"].value)

        return {
            "relaxation": relaxation,
            "relaxed_parameters": relaxed_parameters,
            "data": data,
            "lower_bound": data["problem"].value,
        }


def solve_problem(
    problem_data,
    relaxation,
    config,
    *,
    use_wandb=False,
    log_wandb_every=2,
    name="gradient_descent",
    initial_state="relaxation",
    num_iterations=10,
    args={"step_size": 1e-3, "num_iterations": 100},
    checkpoint_every=100_000,
    parallel=False,
    num_parallel_workers=-1,
    batch_size=0,
    track_full_loss_every=0,
    batch_strategy="sequential",
):
    print("Solving problem...")

    problem: zap.planning.PlanningProblem = problem_data["problem"]
    if problem_data["stochastic_problem"] is not None:
        print("Solving stochastic problem.")
        problem: zap.planning.StochasticPlanningProblem = problem_data["stochastic_problem"]

        if parallel:
            if num_parallel_workers <= 0:
                num_parallel_workers = len(problem.subproblems)

            problem.initialize_workers(num_parallel_workers)
            print(f"Assigned {num_parallel_workers} parallel workers.")

    # Construct algorithm
    alg = ALGORITHMS[name](**args)

    # Setup wandb
    if use_wandb:
        wandb.init(project="zap", config=config)
        logger = wandb
    else:
        logger = None

    # Initialize
    init = initial_state

    if relaxation is not None and init == "relaxation":
        print("Initializing with relaxation solution.")
        initial_state = deepcopy(relaxation["relaxed_parameters"])

    elif init == "initial":
        print("Initializing with initial parameters (no investment).")
        initial_state = None

    else:
        print("Initializing with a previous solution.")

        # Check if file exists
        initial_path = datadir("results", f"{init}.json")
        if not initial_path.exists():
            raise ValueError("Could not find initial parameters.")

        with open(initial_path, "r") as f:
            initial_state = json.load(f)

        ref_init = {k: v for k, v in problem.initialize_parameters(None).items()}
        initial_state = {
            k: np.array(v).reshape(ref_init[k].shape) for k, v in initial_state.items()
        }

    # Convert to torch if needed
    if initial_state is not None and isinstance(problem_data["problem"].layer, ADMMLayer):
        print("Converting initial state to torch.")
        initial_state = {
            k: torch.tensor(v, dtype=ref_init[k].dtype, device=ref_init[k].device)
            for k, v in initial_state.items()
        }

    # Solve
    parameters, history = problem.solve(
        num_iterations=num_iterations,
        algorithm=alg,
        trackers=tr.DEFAULT_TRACKERS + [tr.GRAD, tr.PARAM],
        initial_state=initial_state,
        wandb=logger,
        log_wandb_every=log_wandb_every,
        lower_bound=relaxation["lower_bound"] if relaxation is not None else None,
        extra_wandb_trackers=get_wandb_trackers(problem_data, relaxation, config),
        checkpoint_every=checkpoint_every,
        checkpoint_func=lambda *args: checkpoint_model(*args, config),
        batch_size=batch_size,
        batch_strategy=batch_strategy,
    )

    if parallel:
        problem.shutdown_workers()

    if use_wandb:
        wandb.finish()

    return {
        "initial_state": initial_state,
        "parameters": parameters,
        "history": history,
    }


def solve_baseline(
    problem_data,
    mip_solver="mosek",
    pao_solver="pao.pyomo.FA",
    verbose=True,
):
    from zap.pyomo.bilevel import solve_bilevel_model
    import pyomo.environ as pyo

    print("\n\n\n\nSolving baseline problem.")

    problem = problem_data["problem"]
    layer: zap.DispatchLayer = problem_data["layer"]

    assert type(problem) is not zap.planning.StochasticPlanningProblem

    # Get parameter types
    parameter_types = [ind for key, (ind, pname) in layer.parameter_names.items()]

    # Model and solve problem
    model, solver_data = solve_bilevel_model(
        layer.network,
        layer.devices,
        problem.time_horizon,
        problem.operation_objective,
        param_device_types=parameter_types,
        pao_solver=pao_solver,
        mip_solver=mip_solver,
        verbose=verbose,
    )

    # Print stuff
    print("Solved baseline problem.")
    print("Objective: ", pyo.value(model.objective))
    print("Emissions: ", pyo.value(model.dispatch.device[0].emissions))
    print(
        "Investment Cost: ",
        sum(pyo.value(model.param_blocks[p].investment_cost) for p in parameter_types),
    )

    # Parse results
    params = {
        key: np.array(
            [model.param_blocks[ind].param[k].value for k in range(layer.devices[ind].num_devices)]
        )
        for key, ind in layer.parameter_names.items()
    }

    return {
        "parameters": params,
    }


def save_results(relaxation, results, config):
    # Pick a file name
    results_path = get_results_path(config["id"], config.get("index", None))
    results_path.mkdir(parents=True, exist_ok=True)

    # Save relaxation parameters
    if relaxation is not None:
        relax_params = {k: v.ravel().tolist() for k, v in relaxation["relaxed_parameters"].items()}
        with open(results_path / "relaxed.json", "w") as f:
            json.dump(relax_params, f)

    # Save final parameters
    final_params = {k: v.ravel().tolist() for k, v in results["parameters"].items()}
    with open(results_path / "optimized.json", "w") as f:
        json.dump(final_params, f)

    return None


def run_experiment(config: dict):
    print(platform.architecture())
    print(platform.machine())
    print(platform.platform())
    print(platform.processor())
    print(platform.system())
    print(platform.version())
    print(platform.uname())
    print(platform.python_version())
    print("\n\n\n")

    if "layer" not in config.keys():
        config["layer"] = {}

    if "relaxation" not in config.keys():
        config["relaxation"] = {"should_solve": False}

    # Load data and formulate problem
    data = load_dataset(**config["data"])
    print(config["layer"])
    problem = setup_problem(**data, **config["problem"], **config["layer"])

    # Solve baseline
    if config.get("solve_baseline", False):
        print("Solving baseline problem...")
        baseline = solve_baseline(problem, **config["baseline"])
        save_results(None, baseline, config)
        return None

    # Solve relaxation and original problem
    else:
        relaxation = solve_relaxed_problem(problem, **config["relaxation"])
        results = solve_problem(problem, relaxation, config, **config["optimizer"])

        save_results(relaxation, results, config)

        return None


def get_wandb_trackers(problem_data, relaxation, config: dict):
    problem, layer = problem_data["problem"], problem_data["layer"]
    is_stochastic = problem_data["stochastic_problem"] is not None
    sub_devices = problem_data["sub_devices"]

    # full_stoch_problem = deepcopy(problem_data["stochastic_problem"])

    # TODO - Generalize for multi-objective problems
    if is_stochastic:
        carbon_objective = [zap.planning.EmissionsObjective(d) for d in sub_devices]
        cost_objective = [zap.planning.DispatchCostObjective(layer.network, d) for d in sub_devices]
    else:
        carbon_objective = zap.planning.EmissionsObjective(layer.devices)
        cost_objective = zap.planning.DispatchCostObjective(layer.network, layer.devices)

    if is_stochastic:

        def emissions_tracker(J, grad, params, last_state, problem):
            states = [sub.state for sub in problem.subproblems]
            carbon_costs = [
                c(s, parameters=layer.setup_parameters(**params))
                for c, s in zip(carbon_objective, states)
            ]

            return sum(carbon_costs)

        def cost_tracker(J, grad, params, last_state, problem):
            states = [sub.state for sub in problem.subproblems]

            fuel_costs = [
                c(s, parameters=layer.setup_parameters(**params))
                for c, s in zip(cost_objective, states)
            ]

            return sum(fuel_costs)

    else:

        def emissions_tracker(J, grad, state, last_state, problem):
            return carbon_objective(problem.state, parameters=layer.setup_parameters(**state))

        def cost_tracker(J, grad, state, last_state, problem):
            return cost_objective(problem.state, parameters=layer.setup_parameters(**state))

    lower_bound = relaxation["lower_bound"] if relaxation is not None else 1.0
    true_relax_cost = (
        problem(**relaxation["relaxed_parameters"]) if relaxation is not None else np.inf
    )
    relax_solve_time = (
        relaxation["data"]["problem"].solver_stats.solve_time if relaxation is not None else 0.0
    )

    if is_stochastic:

        def all_op_costs(J, grad, params, last_state, problem):
            return wandb.Histogram(np.array([p.op_cost.item() for p in problem.subproblems]))
    else:

        def all_op_costs(J, grad, params, last_state, problem):
            return wandb.Histogram(np.array([problem.op_cost.item()]))

    # Trackers
    trackers = {
        "emissions": emissions_tracker,
        "fuel_costs": cost_tracker,
        "inv_cost": lambda J, grad, params, last_state, problem: problem.inv_cost.item(),
        "op_cost": lambda J, grad, params, last_state, problem: problem.op_cost.item(),
        "all_op_costs": all_op_costs,
        "batch": lambda J, grad, params, last_state, problem: getattr(problem, "batch", [None])[0],
        "lower_bound": lambda *args: lower_bound,
        "true_relaxation_cost": lambda *args: true_relax_cost,
        "relaxation_solve_time": lambda *args: relax_solve_time,
    }

    if is_stochastic:
        # Add full loss tracker
        track_full_loss_every = config["optimizer"].get("track_full_loss_every", 0)
        batch_size = config["optimizer"].get("batch_size", 0)
        num_problems = problem_data["stochastic_problem"].num_subproblems

        if batch_size == 0:
            batch_size = num_problems

        if track_full_loss_every == 0:  # Track once per batch
            track_full_loss_every = int(num_problems / batch_size)

        if batch_size != num_problems:
            print(f"Tracking full loss every {track_full_loss_every} batches.")

            def full_loss_tracker(J, grad, params, last_state, _stoch_prob):
                iteration = _stoch_prob.iteration

                if iteration % track_full_loss_every == 0:
                    print(f"Updating full problem loss on iteration {iteration}...  ", end="")
                    problem.full_loss = _stoch_prob(**params)
                    print("Done!")
                    return problem.full_loss
                else:
                    return getattr(problem, "full_loss", np.inf)
        else:
            print("Not tracking full loss because batch_size is 0.")

            def full_loss_tracker(J, grad, params, last_state, _stoch_prob):
                return J
    else:

        def full_loss_tracker(J, grad, params, last_state, problem):
            return J

    trackers["full_loss"] = full_loss_tracker

    if isinstance(layer, ADMMLayer):
        trackers["admm_iteration"] = lambda J, _0, _1, _2, problem: sum(
            [p.layer.solver.cumulative_iteration for p in problem.subproblems]
        )
    else:
        trackers["admm_iteration"] = lambda *args: 0

    return trackers


# ====
# Data Processing Utilities
# ====


def device_index(devices, kind):
    if not any(isinstance(d, kind) for d in devices):
        return None

    return next(i for i, d in enumerate(devices) if isinstance(d, kind))


def get_total_load_curve(devices, every=1, reducer=np.sum):
    devs = [d for d in devices if isinstance(d, zap.Load)]
    total_hourly_load = sum([d.load * d.nominal_capacity for d in devs])

    return [
        reducer(total_hourly_load[:, t : t + every])
        for t in range(0, total_hourly_load.shape[1], every)
    ]


def get_total_renewable_curve(devices, every=1, reducer=np.sum, renewables=["solar", "onwind"]):
    devs = [d for d in devices if isinstance(d, zap.Generator)]

    # Filter out non-renewable generators
    is_renewable = [np.isin(d.fuel_type, renewables).reshape(-1, 1) for d in devs]
    capacities = [(d.nominal_capacity * is_renewable) * d.dynamic_capacity for d in devs]

    total_hourly_renewable = sum([c for c in capacities])[0, :, :]

    return np.array(
        [
            reducer(total_hourly_renewable[:, t : t + every])
            for t in range(0, total_hourly_renewable.shape[1], every)
        ]
    )


# daily_peak_load = get_total_load_curve(year_devices, every=24, reducer=np.max)
# peak_day = np.argmax(daily_peak_load).item()

# start_hour = PYPSA_START_DAY + dt.timedelta(hours=peak_day * 24)


def sort_hours_by_peak(pn: pypsa.Network, pn_args, by="load", period=24, reducer=np.max):
    # Load network
    all_dates = pd.date_range(start=PYPSA_START_DAY, periods=TOTAL_PYPSA_HOUR, freq="1h")
    _, year_devices = zap.importers.load_pypsa_network(pn, all_dates, **pn_args)

    # Get peak load
    if by == "load":
        daily_peak = get_total_load_curve(year_devices, every=period, reducer=reducer)

    elif by == "renewable":
        daily_peak = get_total_renewable_curve(year_devices, every=period, reducer=reducer)

    elif by == "hybrid":
        lc = get_total_load_curve(year_devices, every=period, reducer=reducer)
        rc = get_total_renewable_curve(year_devices, every=period, reducer=reducer)

        peak_load = rankdata(lc, method="ordinal") + 0.4
        peak_net_load = rankdata(lc - rc, method="ordinal") + 0.3
        peak_renew = rankdata(rc, method="ordinal") + 0.2
        worst_renew = rankdata(-rc, method="ordinal") + 0.1

        daily_peak = np.max([peak_load, peak_net_load, peak_renew, worst_renew], axis=0)

        # for d in np.argsort(-np.array(daily_peak))[:30]:
        #     d = int(d)
        #     print(lc[d], rc[d], lc[d] - rc[d])
        #     print(peak_load[d], peak_net_load[d], peak_renew[d], worst_renew[d])
        #     print(PYPSA_START_DAY + dt.timedelta(hours=d * period), "\n")

    else:
        raise ValueError("Unknown peak type.")

    # Sort periods in descending order (biggest values first)
    sorted_periods = np.argsort(-np.array(daily_peak))

    # Expand periods and concatenate
    expanded_periods = [np.arange(p * period, (p + 1) * period) for p in sorted_periods]
    return np.concatenate(expanded_periods)


# ====
# File I/O and Config Utilities
# ====


def checkpoint_model(parameters, history, config):
    result_dir = get_results_path(config["id"], config.get("index", None))
    result_dir.mkdir(parents=True, exist_ok=True)

    iteration = len(history[tr.LOSS])

    # Save current model
    with open(result_dir / f"model_{iteration:05d}.json", "w") as f:
        ps = {k: v.ravel().tolist() for k, v in parameters.items()}
        json.dump(ps, f)

    # Save current gradient
    with open(result_dir / f"gradient_{iteration:05d}.json", "w") as f:
        gs = {k: v.ravel().tolist() for k, v in history[tr.GRAD][-1].items()}
        json.dump(gs, f)

    return None


def datadir(*args):
    return Path(DATA_PATH, *args)


def projectdir(*args):
    return Path(ZAP_PATH, *args)


def get_results_path(config_name, index=None):
    if index is None:
        return datadir("results", config_name)
    else:
        return datadir("results", config_name, f"{index:03d}")


def load_model(model):
    # Check if file exists
    initial_path = datadir("results", f"{model}.json")
    if not initial_path.exists():
        raise ValueError(f"Could not find model at {initial_path}")

    with open(initial_path, "r") as f:
        model = json.load(f)

    model = {k: np.array(v).reshape((-1, 1)) for k, v in model.items()}
    return model


def parse_scientific_notation(s):
    if isinstance(s, list):
        return [parse_scientific_notation(v) for v in s]
    if isinstance(s, dict):
        return {k: parse_scientific_notation(v) for k, v in s.items()}
    if isinstance(s, str):
        try:
            return float(s)
        except ValueError:
            return s
    return s


# ====
# MAIN FUNCTION
# ====


if __name__ == "__main__":
    config_path = sys.argv[1]

    if len(sys.argv) > 2:
        config_num = int(sys.argv[2])
    else:
        config_num = 0

    config = expand_config(load_config(config_path))[config_num]

    run_experiment(config)
