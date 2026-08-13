#!/bin/bash
# Submits the full NeuralGCM reforecast archive: years 2023-2026, all 12
# months each.
#
# For each year, month 1 is submitted alone first -- it's the one that
# creates that year's zarr skeleton on its first write (see
# run_month.py's docstring). Months 2-12 are then submitted as a single
# array job with an `afterok` dependency on month 1, so they only start
# once the store exists and never race to create it themselves.
#
# Usage: bash submit_all.sh
set -euo pipefail
cd "$(dirname "$0")"

for YEAR in 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
  SEED_JOB=$(sbatch --parsable --export=ALL,YEAR="$YEAR",MONTH=1 run_month.sbatch)
  echo "Year $YEAR: month 1 seed job = $SEED_JOB"

  REST_JOB=$(sbatch --parsable --array=9-12 \
    --dependency=afterok:"$SEED_JOB" \
    --export=ALL,YEAR="$YEAR" \
    run_month.sbatch)
  echo "Year $YEAR: months 9-12 array job = $REST_JOB (depends on $SEED_JOB)"
done
