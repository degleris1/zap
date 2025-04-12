import argparse
import os
import time
import csv
import yaml
import torch
import cvxpy as cp
import logging
from copy import deepcopy
from benchmarks.maros_benchmark import MarosBenchmarkSet
from benchmarks.netlib_benchmark import NetlibBenchmarkSet
from benchmarks.lasso_benchmark import LassoBenchmarkSet
from zap.admm import ADMMSolver
from zap.conic.cone_bridge import ConeBridge
from zap.conic.cone_utils import get_standard_conic_problem

BENCHMARK_CLASSES = {
    "maros": MarosBenchmarkSet,
    "netlib": NetlibBenchmarkSet,
    "lasso": LassoBenchmarkSet,
}

RESULTS_DIR = "results/benchmark"


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


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def instantiate_benchmark_set(config_entry: dict):
    cfg = dict(config_entry)
    benchmark_type = cfg.pop("type").lower()
    if benchmark_type not in BENCHMARK_CLASSES:
        raise ValueError(f"Unknown benchmark set type: {benchmark_type}")
    return BENCHMARK_CLASSES[benchmark_type](**cfg)


def solve(problem: cp.Problem, solver_name: str, solver_args):
    """
    Solves a CVXPY problem using the specified solver.
    Using this function as a unified interface for solving problems
    so that we can also support ZAP which is not a CVXPY solver
    (and possibly others in the future).
    """
    if solver_name.lower() == "zap":
        cone_params, _, _ = get_standard_conic_problem(problem, solver=cp.CLARABEL)
        cone_bridge = ConeBridge(cone_params)
        machine = solver_args.get("machine", "cpu")
        dtype = torch.float32
        admm_devices = [d.torchify(machine=machine, dtype=dtype) for d in cone_bridge.devices]
        admm = ADMMSolver(**solver_args)
        start_time = time.time()
        solution_admm, _ = admm.solve(cone_bridge.net, admm_devices, cone_bridge.time_horizon)
        end_time = time.time()
        pobj = solution_admm.objective
        solve_time = end_time - start_time

    else:
        problem.solve(solver=solver_name, **solver_args)
        pobj = problem.value
        solve_time = problem.solver_stats.solve_time
    return pobj, solve_time


def run_benchmark_set(benchmark_set, solver_dict):
    """
    Runs a given benchmark set using the specified solver and its args.
    Returns a list of results (as dictionaries).
    """
    results = []
    solver_name = solver_dict.get("name")
    solver_args = solver_dict.get("args", {})
    for idx, problem in enumerate(benchmark_set):
        print(
            f"Running problem {idx + 1} from {benchmark_set.__class__.__name__} with {solver_name}..."
        )
        try:
            pobj, solve_time = solve(problem, solver_name, solver_args)
        except Exception as e:
            pobj = float("nan")
            solve_time = float("nan")
            print(f"  Error: {e}")
        results.append(
            {
                "benchmark_set": benchmark_set.__class__.__name__,
                "problem_index": idx,
                "solver": solver_name,
                "solve_time": solve_time,
                "pobj": pobj,
            }
        )
    return results


def run_benchmarks(config: dict):
    all_results = []
    solver_list = [config.get("solvers")]

    benchmark_sets_config = config.get("benchmark_sets")
    for bs_name, bs_params in benchmark_sets_config.items():
        print(f"Instantiating benchmark set: {bs_name}")
        bench_set = instantiate_benchmark_set(bs_params)
        for solver_dict in solver_list:
            print(f"Running {bs_name} with solver {solver_dict['name']}...")
            res = run_benchmark_set(bench_set, solver_dict=solver_dict)
            all_results.extend(res)
    return all_results


def write_results_to_csv(results, output_file: str):
    fieldnames = ["benchmark_set", "problem_index", "solver", "solve_time", "pobj"]
    output_path = os.path.join(RESULTS_DIR, output_file)
    with open(output_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--config_num", type=int, default=0, help="Index of the configs to run")
    args = parser.parse_args()
    config_path = args.config_path
    config_num = args.config_num
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    output_file = f"{config_name}_{config_num}.csv"

    config = expand_config(load_config(config_path))[config_num]
    results = run_benchmarks(config)
    write_results_to_csv(results, output_file)


if __name__ == "__main__":
    main()
