#!/bin/bash
#SBATCH --job-name=zap_my_gradient
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/oak/stanford/groups/iazevedo/kamran/CH3/zap/experiments/multi_year/outputs/slurm_%x_%j.out
#SBATCH --error=/oak/stanford/groups/iazevedo/kamran/CH3/zap/experiments/multi_year/outputs/slurm_%x_%j.err
#SBATCH --mail-type=END,FAIL

# Usage: sbatch submit_gradient.sbatch {small|normal|large}
set -euo pipefail

NETWORK="${1:?Usage: sbatch submit_gradient.sbatch NETWORK}"
ZAP_ROOT=/oak/stanford/groups/iazevedo/kamran/CH3/zap
cd "$ZAP_ROOT"

module load gurobi
source /home/groups/iazevedo/miniforge3/etc/profile.d/conda.sh
conda activate zap

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=online

python -u experiments/multi_year/runner.py \
    "experiments/multi_year/configs/networks/${NETWORK}.yaml" \
    experiments/multi_year/configs/methods/gradient.yaml
