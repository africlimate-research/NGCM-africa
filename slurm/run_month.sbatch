#!/bin/bash
# Runs one (year, month) of the NeuralGCM reforecast archive. Not meant to
# be sbatch'd directly -- submitted by submit_all.sh, which sets YEAR (via
# --export) and MONTH (either via --export for the lone seed submission, or
# implicitly via --array for the rest of that year's months).
#
# GPU choice: the 2.8deg checkpoint is small; an A100 (40GB, 5 nodes / 20
# GPUs on htc-g[015-019]) comfortably fits it with less queueing risk than
# the H100/GH200 nodes, which are scarcer and better reserved for jobs that
# actually need them. If A100 queue times are long, htc-g[061-084] (L40S,
# 24 nodes / 96 GPUs) is a fine fallback -- swap the --gres line below.
#
# --time/--mem below are sized from a real benchmark (sacct job 8788737):
# one init with all 10 ensemble members (config.ENSEMBLE_SIZE) took
# 16m11s elapsed. GPU memory wasn't captured (this cluster's accounting
# doesn't record per-step MaxRSS for a bare `srun bash -c` invocation),
# but the run completed cleanly with no --mem specified and no CUDA
# RESOURCE_EXHAUSTED, so 10-member vmap batching clearly fits on one A100
# (40GB). Since a month now runs up to 31 inits sequentially
# (config.INIT_DAYS_OF_MONTH covers every day), budget: 16m11s x 31 =
# ~8.4h, +~50% margin for slower months/nodes -> 14h, comfortably inside
# `medium`'s 48h ceiling. --mem is bumped from the single-member 2.9GB
# baseline for the 10x more xarray Dataset objects the per-member Python
# loop in reforecast.py now holds per chunk (host-side only -- the GPU
# compute itself is what vmap batches, not host RAM).
#
#SBATCH --job-name=neuralgcm-reforecast
#SBATCH --partition=medium
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --output=logs/reforecast_%A_%a.out
#SBATCH --error=logs/reforecast_%A_%a.err

set -euo pipefail

# --- one-time environment setup (run once, interactively, before first sbatch) ---
# No conda on htc -- use a plain venv instead.
#   module load Python/3.11.5-GCCcore-13.2.0
#   python3 -m venv $DATA/envs/ngcm-env
#   source $DATA/envs/ngcm-env/bin/activate
#   pip install -r requirements.txt
# ---------------------------------------------------------------------------------

module load Python/3.11.5-GCCcore-13.2.0
source "$DATA/envs/ngcm-env/bin/activate"

# ngcm isn't pip-installed (it's the source tree at $HOME/ngcm), so
# `python -m ngcm.run_month` only resolves if $HOME is on PYTHONPATH --
# the job's working directory is wherever it was sbatch'd from
# (ngcm/slurm/, per submit_all.sh), not $HOME itself.
export PYTHONPATH="$HOME:${PYTHONPATH:-}"

: "${YEAR:?YEAR must be set via sbatch --export}"
MONTH="${SLURM_ARRAY_TASK_ID:-${MONTH:?Set MONTH via --export, or submit as an --array}}"

OUTPUT_DIR="$DATA/neuralgcm_reforecast"
mkdir -p "$OUTPUT_DIR"

python -m ngcm.run_month \
  --year "$YEAR" --month "$MONTH" --output-dir "$OUTPUT_DIR"
