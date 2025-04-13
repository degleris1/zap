import sys
import os
import runner
from pathlib import Path

config_path = sys.argv[1]
config_list = runner.expand_config(runner.load_config(config_path))

print(f"Launching {len(config_list)} jobs...")
for i, config in enumerate(config_list):
    assert i == config["index"]

    system: dict = config["system"]
    job_name = Path(config_path).stem

    # Configure file paths
    output_file = runner.datadir("slurm", f"{job_name}_{i:03d}.out")
    script_file = runner.datadir("scripts", f"{job_name}_{i:03d}.sh")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    script_file.parent.mkdir(parents=True, exist_ok=True)

    if "gpu" in system.keys() and system["gpu"] > 0:
        gpu_line = f"#SBATCH --gpus={system['gpu']}"

    else:
        constraint = "cpu"
        threads = system["threads"]
        gpu_line = ""

    partition_and_resources = (
        f"#SBATCH --partition={system['partition']}\n#SBATCH --cpus-per-task=32\n#SBATCH --mem=64G"
    )

    # Write slurm script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}_{i:03d}
#SBATCH --output={output_file}
{partition_and_resources}
#SBATCH --nodes=1
{gpu_line}
#SBATCH --time={system["runtime"]}

poetry run python3 -u experiments/conic_solve/runner.py {config_path} {i}
"""

    with open(script_file, "w") as f:
        f.write(slurm_script)

    # Launch
    print(f"Launching job {job_name} (parameter {i})...")
    os.system(f"sbatch {script_file}")
