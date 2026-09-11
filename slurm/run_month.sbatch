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
# --time/--mem/--partition below are STALE placeholders, not yet
# re-benchmarked. The prior numbers (sacct job 8569257: ~14m52s, ~2.9GB
# MaxRSS) were for ONE init with ONE ensemble member -- since
# config.INIT_DAYS_OF_MONTH now covers every day of the month (~30
# inits/month, processed sequentially in one job) and config.ENSEMBLE_SIZE
# batches 10 stochastic realizations per init via jax.vmap, both the
# per-job wall time and the per-init GPU memory footprint have changed in
# ways that benchmark no longer reflects. Before relying on these values:
#   srun --partition=short --gres=gpu:a100:1 --time=00:30:00 \
#     python -m ngcm.run_month --year 2023 --month 1 \
#     --output-dir $OUTPUT_DIR --limit-inits 1
# then check `nvidia-smi` (or sacct MaxRSS for host memory) during/after
# for the actual 10-member GPU memory peak, and multiply the elapsed time
# by ~30 (inits/month) with margin for --time. The values below
# (`medium`/`12G`/`8h`) are a conservative guess pending that, not a
# validated number.
#
#SBATCH --job-name=neuralgcm-reforecast
#SBATCH --partition=medium
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
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
