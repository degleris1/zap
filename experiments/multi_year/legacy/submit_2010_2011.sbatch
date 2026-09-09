#!/bin/bash
#SBATCH --job-name=zap_my_2010_2011
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/oak/stanford/groups/iazevedo/kamran/CH3/zap/experiments/multi_year/outputs/slurm_%x_%j.out
#SBATCH --error=/oak/stanford/groups/iazevedo/kamran/CH3/zap/experiments/multi_year/outputs/slurm_%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

ZAP_ROOT=/oak/stanford/groups/iazevedo/kamran/CH3/zap
cd "$ZAP_ROOT"

# Gurobi — picks up GUROBI_HOME, GRB_LICENSE_FILE, etc. from the module
module load gurobi

# Activate the existing poetry venv (order matters: after module load)
source "$ZAP_ROOT/venv/bin/activate"

# Thread caps — prevents BLAS from oversubscribing against num_workers.
# Gurobi's own Threads parameter is set explicitly in the config.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Wandb — set your API key (or run `wandb login` beforehand)
# export WANDB_API_KEY=<your_key>
export WANDB_MODE=online       # or "offline" if compute nodes are firewalled

python -u experiments/multi_year/runner.py \
    experiments/multi_year/config_2010_2011.yaml


#  srun --partition=serc --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --time=06:00:00 --pty bash