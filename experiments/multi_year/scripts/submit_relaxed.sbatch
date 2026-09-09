#!/bin/bash
#SBATCH --job-name=zap_my_relaxed
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/oak/stanford/groups/iazevedo/kamran/CH3/zap/experiments/multi_year/outputs/slurm_%x_%j.out
#SBATCH --error=/oak/stanford/groups/iazevedo/kamran/CH3/zap/experiments/multi_year/outputs/slurm_%x_%j.err
#SBATCH --mail-type=END,FAIL

# Usage: sbatch submit_relaxed.sbatch {small|normal|large}
set -euo pipefail

NETWORK="${1:?Usage: sbatch submit_relaxed.sbatch NETWORK}"
ZAP_ROOT=/oak/stanford/groups/iazevedo/kamran/CH3/zap
cd "$ZAP_ROOT"

module load gurobi
source /home/groups/iazevedo/miniforge3/etc/profile.d/conda.sh
conda activate zap

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u experiments/multi_year/runner.py \
    "experiments/multi_year/configs/networks/${NETWORK}.yaml" \
    experiments/multi_year/configs/methods/relaxed.yaml
