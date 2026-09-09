#!/bin/bash
#SBATCH --job-name=ra_run
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL

# RA harness as a SLURM array: each array task claims one shard of the task list.
#
# Usage:
#   N=16 sbatch --array=1-16 ra_array.sbatch experiments/ra/configs/experiments/op_benchmark_z4_2020.yaml
#
# Then aggregate once every shard has finished (or submit it as a dependent job):
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> --wrap "..."
#
# ZAP_ROOT defaults to the Sherlock checkout but is parameterised (see LESSONS.md):
#   ZAP_ROOT=/path/to/zap sbatch ...

set -euo pipefail

CONFIG="${1:?Usage: sbatch ra_array.sbatch CONFIG [EXTRA ARGS...]}"
shift || true

ZAP_ROOT="${ZAP_ROOT:-/oak/stanford/groups/iazevedo/kamran/CH3/zap}"
cd "$ZAP_ROOT"

module load gurobi
source /home/groups/iazevedo/miniforge3/etc/profile.d/conda.sh
conda activate zap

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

N="${N:-${SLURM_ARRAY_TASK_COUNT:-1}}"
K="${SLURM_ARRAY_TASK_ID:-1}"

python -u -m experiments.ra.cli run --config "$CONFIG" --shard "${K}/${N}" "$@"
