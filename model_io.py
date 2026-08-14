"""Loading the NeuralGCM checkpoint and its ERA5 initial conditions/forcing."""

import pickle

import gcsfs
import neuralgcm
import numpy as np
import xarray
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils

from ngcm import config


def load_model() -> neuralgcm.PressureLevelModel:
  gcs = gcsfs.GCSFileSystem(token="anon")
  path = f"{config.CHECKPOINT_BUCKET}/{config.CHECKPOINT}"
  with gcs.open(path, "rb") as f:
    ckpt = pickle.load(f)
  return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)


def open_era5() -> xarray.Dataset:
  return xarray.open_zarr(
      config.ERA5_PATH, chunks=None, storage_options=dict(token="anon")
  )


def build_regridder(
    model: neuralgcm.PressureLevelModel, full_era5: xarray.Dataset
) -> horizontal_interpolation.ConservativeRegridder:
  era5_grid = spherical_harmonic.Grid(
      latitude_nodes=full_era5.sizes["latitude"],
      longitude_nodes=full_era5.sizes["longitude"],
      latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
      longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
  )
  return horizontal_interpolation.ConservativeRegridder(
      era5_grid, model.data_coords.horizontal, skipna=True
  )


def load_initial_conditions(
    full_era5: xarray.Dataset,
    model: neuralgcm.PressureLevelModel,
    regridder: horizontal_interpolation.ConservativeRegridder,
    init_time: np.datetime64,
) -> xarray.Dataset:
  """Loads the single timestep of pressure-level data needed to encode state.

  Only `model.input_variables` need this -- they're read once, at
  `init_time`, unlike the forcing variables which span the whole rollout
  (see load_forcing_window). Keeping them separate avoids pulling ERA5's
  full 37-level, 0.25deg pressure-level fields across every timestep of a
  45-day window, which is tens of GB for variables that are only ever used
  at t=0.
  """
  ic = full_era5[model.input_variables].sel(time=init_time).compute()
  ic = xarray_utils.regrid(ic, regridder)
  ic = xarray_utils.fill_nan_with_nearest(ic)
  return ic


def load_forcing_window(
    full_era5: xarray.Dataset,
    model: neuralgcm.PressureLevelModel,
    regridder: horizontal_interpolation.ConservativeRegridder,
    init_time: np.datetime64,
    lead_days: int,
) -> xarray.Dataset:
  """Loads, regrids, and daily-thins the ERA5 forcing window an init needs.

  Covers `init_time` (the initial condition) through `init_time + lead_days`,
  at daily (00Z) resolution, matching config.INIT_HOUR and
  config.OUTPUT_TIMESTEP_HOURS == 24. Only `model.forcing_variables` (single
  -level fields like SST/sea ice) are loaded here -- see
  load_initial_conditions for the pressure-level input variables.
  """
  end_time = init_time + np.timedelta64(lead_days, "D")
  window = (
      full_era5[model.forcing_variables]
      .pipe(
          xarray_utils.selective_temporal_shift,
          variables=model.forcing_variables,
          time_shift="24 hours",
      )
      .sel(time=slice(init_time, end_time))
      .thin(time=24)
      .compute()
  )
  window = xarray_utils.regrid(window, regridder)
  window = xarray_utils.fill_nan_with_nearest(window)
  return window
