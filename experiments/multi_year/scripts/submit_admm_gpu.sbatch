#!/bin/bash
#SBATCH --job-name=zap_my_admm_gpu
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/oak/stanford/groups/iazevedo/kamran/CH3/zap/experiments/multi_year/outputs/slurm_%x_%j.out
#SBATCH --error=/oak/stanford/groups/iazevedo/kamran/CH3/zap/experiments/multi_year/outputs/slurm_%x_%j.err
#SBATCH --mail-type=END,FAIL

# STUB: ADMM/GPU path is not yet ported into multi_year/runner.py.
# The runner will raise NotImplementedError when this method is selected.
# Follow-up: port experiments/plan/runner.py:288-316 (ADMMLayer + ADMMSolver).

# Usage: sbatch submit_admm_gpu.sbatch {small|normal|large}
set -euo pipefail

NETWORK="${1:?Usage: sbatch submit_admm_gpu.sbatch NETWORK}"
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
    experiments/multi_year/configs/methods/admm_gpu.yaml
