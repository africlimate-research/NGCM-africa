"""CLI: builds the SST / sea-ice climatology used by FORCING_MODE="anomaly_persistence".

Usage:
  python -m ngcm.build_climatology --output-dir /path/to/archive

Run once, before any run_month jobs. It reads CLIMATOLOGY_YEARS of ERA5
(~11k daily fields per variable), so it takes a while -- see
slurm/build_climatology.sbatch. Does nothing if the file already exists.
"""

import argparse
import logging
from pathlib import Path

from ngcm import climatology
from ngcm import model_io

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()

  path = climatology.climatology_path(args.output_dir)
  if path.exists():
    logger.info("%s already exists; nothing to do", path)
    return

  model = model_io.load_model()
  full_era5 = model_io.open_era5()
  regridder = model_io.build_regridder(model, full_era5)
  climatology.build(model, full_era5, regridder, path)


if __name__ == "__main__":
  main()
