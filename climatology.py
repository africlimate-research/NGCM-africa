"""Day-of-year SST / sea-ice climatology on the model grid, for anomaly persistence."""

import logging
from pathlib import Path

import xarray
from dinosaur import xarray_utils

from ngcm import config

logger = logging.getLogger(__name__)


def climatology_path(output_dir: Path) -> Path:
  start, end = config.CLIMATOLOGY_YEARS
  return Path(output_dir) / f"forcing_climatology_{start}_{end}.zarr"


def build(model, full_era5: xarray.Dataset, regridder, path: Path) -> None:
  """Averages daily 00Z forcing fields by day of year over CLIMATOLOGY_YEARS.

  Each year is read at native 0.25deg and regridded to the model grid before
  averaging, so only the small (dayofyear, lon, lat) result is kept. NaNs
  (land) are skipped in the mean and left to
  xarray_utils.fill_nan_with_nearest at use time, same as the observed path.
  """
  start, end = config.CLIMATOLOGY_YEARS
  regridded_years = []
  for year in range(start, end + 1):
    logger.info("Climatology: reading %d", year)
    ds = (
        full_era5[model.forcing_variables]
        .sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        .thin(time=24)
        .compute()
    )
    regridded_years.append(xarray_utils.regrid(ds, regridder))
  clim = (
      xarray.concat(regridded_years, "time")
      .groupby("time.dayofyear")
      .mean("time", skipna=True)
  )
  clim.to_zarr(path, mode="w")
  logger.info("Climatology written to %s", path)


def load(path: Path) -> xarray.Dataset:
  if not path.exists():
    raise FileNotFoundError(
        f"{path} not found. FORCING_MODE='anomaly_persistence' needs the "
        "climatology built once first: run `python -m ngcm.build_climatology "
        "--output-dir ...` (see slurm/build_climatology.sbatch)."
    )
  return xarray.open_zarr(path).compute()
