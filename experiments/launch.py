import sys
import os
import runner


config_path = sys.argv[1]
config_list = runner.expand_config(runner.load_config(config_path))

print(f"Launching {len(config_list)} jobs...")
for i, config in enumerate(config_list):
    assert i == config["index"]

    system = config["system"]

    # Configure file paths
    output_file = runner.datadir("slurm", f"{config['name']}_{i:03d}.out")
    script_file = runner.datadir("scripts", f"{config['name']}_{i:03d}.sh")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    script_file.parent.mkdir(parents=True, exist_ok=True)

    if system["gpu"] > 0:
        constraint = "gpu"
    else:
        constraint = "cpu"

    # Write slurm script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={config["name"]}_{i:03d}
#SBATCH --output={output_file}
#SBATCH --qos=shared
#SBATCH --constraint={constraint}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={system["threads"]}
#SBATCH --gpus {system["gpu"]}
#SBATCH --time={system["runtime"]}

module load conda
conda activate $ZAP_ENV
srun python -u experiments/runner.py {config_path} {i}
"""

    with open(script_file, "w") as f:
        f.write(slurm_script)

    # Launch
    print(f"Launching job {config['name']} (parameter {i})...")
    os.system(f"sbatch {script_file}")
