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
# --time and --partition below are a conservative placeholder (this repo
# has no way to benchmark actual per-init runtime on ARC). Before running
# submit_all.sh for real, benchmark one init on the "devel" or "short"
# partition:
#   srun --partition=short --gres=gpu:a100:1 --time=00:30:00 \
#     python -m ngcm.run_month --year 2023 --month 1 \
#     --output-dir $OUTPUT_DIR --limit-inits 1
# then tighten --time/--partition here accordingly (one month is at most 8
# inits).
#
#SBATCH --job-name=neuralgcm-reforecast
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/reforecast_%A_%a.out
#SBATCH --error=logs/reforecast_%A_%a.err

set -euo pipefail

# --- one-time environment setup (run once, interactively, before first sbatch) ---
# No conda on htc -- use a plain venv instead. Confirm the exact Python
# module name/version first (neuralgcm needs >=3.10):
#   module avail python       # or: module spider Python
#   module load <Python module reported above>
#   python3 -m venv $DATA/envs/ngcm-env
#   source $DATA/envs/ngcm-env/bin/activate
#   pip install -r requirements.txt
# ---------------------------------------------------------------------------------

module load Python/3.11  # TODO: confirm this module name via `module avail python` on ARC
source "$DATA/envs/ngcm-env/bin/activate"

: "${YEAR:?YEAR must be set via sbatch --export}"
MONTH="${SLURM_ARRAY_TASK_ID:-${MONTH:?Set MONTH via --export, or submit as an --array}}"

OUTPUT_DIR="$DATA/neuralgcm_reforecast"
mkdir -p "$OUTPUT_DIR" logs

python -m ngcm.run_month \
  --year "$YEAR" --month "$MONTH" --output-dir "$OUTPUT_DIR"
