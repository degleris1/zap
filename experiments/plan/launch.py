import sys
import os
import runner
import numpy as np

PERLMUTTER_CORES_PER_GPU = 32
MAX_TIME = 24 * 60
BASELINE_POWER = np.log2(5.0 / 2.0)

config_path = sys.argv[1]
config_list = runner.expand_config(runner.load_config(config_path))


def get_dynamic_runtime(config: dict):
    runtime_setting = config["system"]["runtime"]
    if isinstance(runtime_setting, int):
        return runtime_setting

    dynamic, base_time, offset_time = runtime_setting
    num_hours = config["data"]["num_hours"]
    assert dynamic == "dynamic"

    if config["solve_baseline"]:
        return np.minimum(
            MAX_TIME,
            int(np.ceil(offset_time + np.power(base_time * num_hours, BASELINE_POWER))),
        )
    else:
        return np.minimum(MAX_TIME, int(np.ceil(offset_time + base_time * num_hours)))


print(f"Launching {len(config_list)} jobs...")
for i, config in enumerate(config_list):
    assert i == config["index"]

    system: dict = config["system"]

    # Configure file paths
    output_file = runner.datadir("slurm", f"{config['name']}_{i:03d}.out")
    script_file = runner.datadir("scripts", f"{config['name']}_{i:03d}.sh")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    script_file.parent.mkdir(parents=True, exist_ok=True)

    if "gpu" in system.keys() and system["gpu"] > 0:
        constraint = "gpu&hbm80g"
        threads = PERLMUTTER_CORES_PER_GPU * system["gpu"]
        gpu_line = f"#SBATCH --gpus={system['gpu']}"

    else:
        constraint = "cpu"
        threads = system["threads"]
        gpu_line = ""

    runtime = get_dynamic_runtime(config)

    # Write slurm script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={config["name"]}_{i:03d}
#SBATCH --output={output_file}
#SBATCH --qos=shared
#SBATCH --constraint={constraint}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={threads}
{gpu_line}
#SBATCH --time={runtime}

module load PrgEnv-intel
poetry shell
srun python -u experiments/plan/runner.py {config_path} {i}
"""

    with open(script_file, "w") as f:
        f.write(slurm_script)

    # Launch
    print(f"Launching job {config['name']} (parameter {i}) for {runtime//60} hours...")
    os.system(f"sbatch {script_file}")
