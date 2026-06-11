#!/usr/bin/env bash
# Step 1: install micromamba (no conda needed) + clone pypsa-usa + create its env.
# Logs to $WORK/setup.log. Designed to be run in the background.
set -euo pipefail
WORK="${1:-$HOME/pypsa-usa-2023}"
mkdir -p "$WORK/bin"
cd "$WORK"

echo "=== [1] download micromamba (osx-arm64) ==="
if [ ! -x "$WORK/bin/micromamba" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj -C "$WORK" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$WORK/mamba"
MM="$WORK/bin/micromamba"
"$MM" --version

echo "=== [2] clone pypsa-usa ==="
[ -d pypsa-usa ] || git clone --depth 1 https://github.com/PyPSA/pypsa-usa.git
echo "env file:"; head -1 pypsa-usa/workflow/envs/environment.yaml

echo "=== [3] create env 'pypsa-usa' (this is the slow step) ==="
"$MM" create -y -r "$MAMBA_ROOT_PREFIX" -n pypsa-usa -f pypsa-usa/workflow/envs/environment.yaml

echo "=== [4] sanity: key packages ==="
"$MM" run -r "$MAMBA_ROOT_PREFIX" -n pypsa-usa python -c "import pypsa, atlite, snakemake, geopandas, atlite; print('pypsa', pypsa.__version__, '| atlite', atlite.__version__, '| snakemake', snakemake.__version__)"
echo "=== SETUP DONE ==="
