#!/bin/bash
#SBATCH --job-name=zap_single
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
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

# Thread caps — prevents BLAS from oversubscribing.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u experiments/multi_year/runner.py \
    experiments/multi_year/config_single.yaml
