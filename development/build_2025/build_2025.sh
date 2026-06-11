#!/usr/bin/env bash
# Build a higher-resolution, 2025-weather-year PyPSA-USA western network.
#
# Requirements (cannot run in the zap sandbox; run on a workstation/cluster):
#   - conda/mamba
#   - a Copernicus CDS API key (free): https://cds.climate.copernicus.eu  (see step 2)
#   - ~30 GB free disk, several hours (the ERA5-2025 cutout build is the long pole)
#
# Usage:
#   bash build_2025.sh /path/to/workdir
# After it finishes, the network is at:
#   $WORKDIR/pypsa-usa/workflow/resources/western/elec_s1000_c1000.nc
set -euo pipefail

WORKDIR="${1:-$HOME/pypsa-usa-2025}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTERS=1000          # edit to taste (e.g. 2000); must match the config file's scenario
SIMPL=1000
JOBS="${JOBS:-4}"

echo ">>> [1/5] clone PyPSA-USA into $WORKDIR"
mkdir -p "$WORKDIR" && cd "$WORKDIR"
[ -d pypsa-usa ] || git clone https://github.com/PyPSA/pypsa-usa.git
cd pypsa-usa

echo ">>> [2/5] CDS API key check (~/.cdsapirc)"
if [ ! -f "$HOME/.cdsapirc" ]; then
  cat <<'EOF'
  !! Missing ~/.cdsapirc. Create it (the 2025 cutout download needs it):

     1) Register/log in at https://cds.climate.copernicus.eu
     2) Accept the "ERA5 hourly data on single levels" licence (Datasets -> search ERA5 -> Download -> Terms)
     3) Copy your API key from your CDS profile, then write ~/.cdsapirc:

        url: https://cds.climate.copernicus.eu/api
        key: <YOUR-UID>:<YOUR-API-KEY>

  Re-run this script once that file exists.
EOF
  exit 1
fi

echo ">>> [3/5] create conda env 'pypsa-usa' (pypsa 0.30.2, atlite 0.3.0, snakemake 7.32.4)"
if ! conda env list | grep -q '^pypsa-usa\b'; then
  (mamba env create -f workflow/envs/environment.yaml || conda env create -f workflow/envs/environment.yaml)
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pypsa-usa
pip install --quiet cdsapi || true   # ensure the CDS client is present for atlite

echo ">>> [4/5] install the 2025 high-res config override"
mkdir -p workflow/config
cp "$THIS_DIR/config.western_2025_highres.yaml" workflow/config/config.yaml
echo "    using workflow/config/config.yaml (clusters=$CLUSTERS, year=2025)"

echo ">>> [5/5] run snakemake (builds: 2025 ERA5 cutout -> renewable profiles -> clustered network)"
cd workflow
TARGET="resources/western/elec_s${SIMPL}_c${CLUSTERS}.nc"
echo "    target: $TARGET"
# Build the cutout first (isolates the slow CDS step), then the network.
snakemake -j"$JOBS" --use-conda -- "$TARGET"

echo ""
echo ">>> DONE. Network: $WORKDIR/pypsa-usa/workflow/$TARGET"
echo ">>> Verify it:    python $THIS_DIR/verify_2025_network.py $WORKDIR/pypsa-usa/workflow/$TARGET"
echo ">>> Then rerun the study from the zap repo:"
echo "    python development/dc_placement_study.py --network $WORKDIR/pypsa-usa/workflow/$TARGET --year 2025 --budget 3.0"
