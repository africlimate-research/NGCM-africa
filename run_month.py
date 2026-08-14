"""CLI entry point: builds/extends one (year, month) slice of the NeuralGCM
reforecast Zarr archive for that year.

Usage:
  python -m ngcm.run_month --year 2023 --month 3 \
      --output-dir /path/to/archive

The store `neuralgcm_reforecast_<year>.zarr` is shared across all 12
months of a year: it's NaN-filled and sized for the whole year on its
first write (see zarr_io.write_skeleton), and each month after that just
fills in its own region. Computing `full_year_init_times` below is pure
calendar arithmetic (no I/O, no model calls) -- it only exists to size
that skeleton correctly; only the current month's inits are ever
actually run.

Because the skeleton write only happens once per year, whichever month
runs first "wins" it -- see slurm/submit_all.sh, which always submits
month 1 alone and waits for it to finish before submitting the rest of
that year's months as an array, so there's no race between months
trying to create the store concurrently.

Safe to re-run after a crash or SLURM requeue: already-completed
initializations (tracked in a `<store>.progress.json` sidecar file next
to the zarr store) are skipped.
"""

import argparse
import json
import logging
from pathlib import Path

import jax
import numpy as np

from ngcm import config
from ngcm import init_times as init_times_lib
from ngcm import model_io
from ngcm import reforecast
from ngcm import zarr_io

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_progress(progress_path: Path) -> set[str]:
  if progress_path.exists():
    return set(json.loads(progress_path.read_text()))
  return set()


def _mark_done(progress_path: Path, done: set[str], init_time: np.datetime64) -> None:
  done.add(str(init_time))
  progress_path.write_text(json.dumps(sorted(done)))


def run_month(
    year: int,
    month: int,
    output_dir: Path,
    limit_inits: int | None = None,
) -> None:
  store_path = zarr_io.year_store_path(output_dir, year)
  progress_path = store_path.with_suffix(".progress.json")
  done = _load_progress(progress_path)

  full_year_init_times = init_times_lib.init_times_for_year(year)
  month_init_times = init_times_lib.init_times_for_year(year, months=[month])
  if limit_inits is not None:
    month_init_times = month_init_times[:limit_inits]
  logger.info(
      "Year %d month %d: %d scheduled initializations",
      year, month, len(month_init_times),
  )

  model = model_io.load_model()
  full_era5 = model_io.open_era5()
  regridder = model_io.build_regridder(model, full_era5)
  era5_max_time = full_era5.time.max().values

  skeleton_written = store_path.exists()

  for init_time in month_init_times:
    if str(init_time) in done:
      logger.info("Skipping %s (already done)", init_time)
      continue

    required_end = init_time + np.timedelta64(config.LEAD_DAYS, "D")
    if required_end > era5_max_time:
      logger.warning(
          "Stopping at %s: ERA5 archive only extends to %s, not far enough "
          "to cover a %d-day rollout.",
          init_time, era5_max_time, config.LEAD_DAYS,
      )
      break

    ic = model_io.load_initial_conditions(full_era5, model, regridder, init_time)
    forcing = model_io.load_forcing_window(
        full_era5, model, regridder, init_time, config.LEAD_DAYS
    )
    seed = abs(hash((year, str(init_time)))) % (2**32)
    rng_key = jax.random.key(seed)

    for chunk_idx, chunk_ds in enumerate(
        reforecast.run_chunked_rollout(model, ic, forcing, rng_key)
    ):
      chunk_ds = chunk_ds.expand_dims(init_time=[init_time])
      if not skeleton_written:
        zarr_io.write_skeleton(store_path, chunk_ds, full_year_init_times)
        skeleton_written = True
      zarr_io.write_region(store_path, chunk_ds)
      logger.info(
          "Wrote %s chunk %d/%d", init_time, chunk_idx + 1, reforecast._N_CHUNKS
      )

    _mark_done(progress_path, done, init_time)
    logger.info("Completed init %s", init_time)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--year", type=int, required=True)
  parser.add_argument("--month", type=int, required=True, choices=range(1, 13))
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--limit-inits",
      type=int,
      default=None,
      help="Only process the first N scheduled inits (for a quick test run).",
  )
  args = parser.parse_args()
  run_month(args.year, args.month, args.output_dir, args.limit_inits)


if __name__ == "__main__":
  main()
