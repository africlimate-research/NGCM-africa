#!/bin/bash
# Uploads each year's .zarr reforecast archive to GCS, one year at a time,
# for years START_YEAR through the current year.
#
# Run as a real sbatch job -- not raw srun, and never directly on the login
# node -- so a many-hour transfer gets a proper memory allocation (zarr's
# many-small-chunk-file layout means `gcloud storage` has to build a
# transfer manifest across a huge file count, which is exactly the kind of
# thing that hit the login node's tight per-user memory cgroup before) and
# survives independently of your SSH session staying connected.
#
# Uses `gcloud storage rsync`, not `cp`, so a partial/failed run is cheap to
# retry: rsync only transfers files that are new or changed, rather than
# re-uploading everything from scratch. Looping per-year (rather than one
# glob) bounds the blast radius of any single failure to one year's ~32GB
# instead of the whole archive, makes progress legible in the log (one
# line per year), and is what lets START_YEAR restrict the range at all.
#
# Usage: sbatch upload_to_gcs.sbatch
#SBATCH --job-name=ngcm-gcs-upload
#SBATCH --partition=interactive
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gcs_upload_%j.out
#SBATCH --error=logs/gcs_upload_%j.err

set -euo pipefail

SRC_DIR="$DATA/neuralgcm_reforecast"
DEST="gs://africlimate-ai-fcst-training/ngcm-1990-2026"
START_YEAR=2005
END_YEAR=$(date +%Y)

for YEAR in $(seq "$START_YEAR" "$END_YEAR"); do
  store="$SRC_DIR/neuralgcm_reforecast_${YEAR}.zarr"
  if [ ! -d "$store" ]; then
    echo "=== Skipping $YEAR: no store at $store ==="
    continue
  fi
  echo "=== Syncing $YEAR ==="
  gcloud storage rsync -r "$store" "$DEST/neuralgcm_reforecast_${YEAR}.zarr"
  echo "=== Done: $YEAR ==="
done
