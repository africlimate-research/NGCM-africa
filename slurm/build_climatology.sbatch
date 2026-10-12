#!/bin/bash
# Builds the SST / sea-ice day-of-year climatology that
# config.FORCING_MODE="anomaly_persistence" needs. Run ONCE, and let it
# finish, before submitting any run_month jobs.
#
# CPU only -- no GPU. It reads config.CLIMATOLOGY_YEARS (30 years) of daily
# 0.25deg SST + sea ice from ERA5 one year at a time (~3GB per year at
# native resolution before regridding, hence --mem), so expect it to take
# an hour or more, dominated by the GCS reads. --time is a guess: check
# the log's per-year "Climatology: reading <year>" timestamps and tighten
# or raise it accordingly.
#
# Usage (from ngcm/slurm/, with logs/ existing):  sbatch build_climatology.sbatch
#SBATCH --job-name=ngcm-climatology
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/climatology_%j.out
#SBATCH --error=logs/climatology_%j.err

set -euo pipefail

module load Python/3.11.5-GCCcore-13.2.0
source "$DATA/envs/ngcm-env/bin/activate"

export PYTHONPATH="$HOME:${PYTHONPATH:-}"

OUTPUT_DIR="$DATA/neuralgcm_reforecast"
mkdir -p "$OUTPUT_DIR"

python -m ngcm.build_climatology --output-dir "$OUTPUT_DIR"
