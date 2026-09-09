#!/bin/bash
#SBATCH --job-name=ra_aggregate
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Aggregation job for an RA run; submit with
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> ra_aggregate.sbatch CONFIG
set -euo pipefail

CONFIG="${1:?Usage: sbatch ra_aggregate.sbatch CONFIG}"
shift || true

ZAP_ROOT="${ZAP_ROOT:-/oak/stanford/groups/iazevedo/kamran/CH3/zap}"
cd "$ZAP_ROOT"

source /home/groups/iazevedo/miniforge3/etc/profile.d/conda.sh
conda activate zap

python -u -m experiments.ra.cli aggregate --config "$CONFIG" "$@"
