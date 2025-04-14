import os
import time
import csv
import yaml
import torch
import cvxpy as cp
import sys
import zap
from copy import deepcopy
from benchmarks.maros_benchmark import MarosBenchmarkSet
from benchmarks.netlib_benchmark import NetlibBenchmarkSet
from benchmarks.lasso_benchmark import LassoBenchmarkSet
from pathlib import Path
from benchmarks.sparse_cone_benchmark import SparseConeBenchmarkSet
from zap.admm import ADMMSolver
from zap.conic.cone_bridge import ConeBridge
from zap.conic.cone_utils import get_standard_conic_problem

# Reference from .yaml type to get the right class
BENCHMARK_CLASSES = {
    "maros": MarosBenchmarkSet,
    "netlib": NetlibBenchmarkSet,
    "lasso": LassoBenchmarkSet,
    "sparse_cone": SparseConeBenchmarkSet,
}

RESULTS_DIR = "results/benchmark"
ZAP_PATH = Path(zap.__file__).parent.parent
DATA_PATH = ZAP_PATH / "data"


def datadir(*args):
    return Path(DATA_PATH, *args)


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
    if solver_name.lower() == "admm":
        cone_params, _, _ = get_standard_conic_problem(problem, solver=cp.SCS)
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
        start_time = time.time()
        problem.solve(solver=solver_name.upper(), **solver_args)
        end_time = time.time()
        pobj = problem.value
        if solver_name.lower() == "pdlp":
            solve_time = end_time - start_time
        else:
            solve_time = problem.solver_stats.solve_time
    return pobj, solve_time


def run_benchmark_set(benchmark_set, solver_params, bs_name, solver_name):
    """
    Runs a given benchmark set using the specified solver and its args.
    Returns a list of results (as dictionaries).
    """
    results = []
    for idx, problem in enumerate(benchmark_set):
        print(f"Running problem {idx + 1} from {bs_name} with {solver_name}...")
        try:
            pobj, solve_time = solve(problem, solver_name, solver_params)
        except Exception as e:
            pobj = float("nan")
            solve_time = float("nan")
            print(f"  Error: {e}")
        results.append(
            {
                "benchmark_set": bs_name,
                "problem_index": idx,
                "solver": solver_name,
                "solve_time": solve_time,
                "pobj": pobj,
            }
        )
    return results


def run_benchmarks(config: dict):
    all_results = []
    solver_list = config.get("solver")
    benchmark_list = config.get("benchmarks")
    if type(benchmark_list) is not list:
        benchmark_list = [benchmark_list]
    if type(solver_list) is not list:
        solver_list = [solver_list]

    for bs_name in benchmark_list:
        print(f"Instantiating benchmark set: {bs_name}")
        bs_params = config.get(f"{bs_name}_args", {})
        bench_set = instantiate_benchmark_set(bs_params)
        for solver_name in solver_list:
            print(f"Running {bs_name} with solver {solver_name}...")
            solver_params = config.get(f"{solver_name}_args", {})
            res = run_benchmark_set(
                bench_set, solver_params=solver_params, bs_name=bs_name, solver_name=solver_name
            )
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
    config_path = sys.argv[1]

    if len(sys.argv) > 2:
        config_num = int(sys.argv[2])
    else:
        config_num = 0
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    output_file = f"{config_name}_{config_num:03d}.csv"

    config = expand_config(load_config(config_path))[config_num]
    results = run_benchmarks(config)
    write_results_to_csv(results, output_file)


if __name__ == "__main__":
    main()
