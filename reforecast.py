"""Runs one 45-day NeuralGCM rollout in chunks, selecting the archive's variables."""

from collections.abc import Iterator

import jax
import neuralgcm
import numpy as np
import xarray

from ngcm import config

_TOTAL_STEPS = config.LEAD_DAYS * 24 // config.OUTPUT_TIMESTEP_HOURS
_STEPS_PER_CHUNK = config.CHUNK_DAYS * 24 // config.OUTPUT_TIMESTEP_HOURS

if _TOTAL_STEPS % _STEPS_PER_CHUNK:
  raise ValueError(
      f"LEAD_DAYS ({config.LEAD_DAYS}) must be an exact multiple of "
      f"CHUNK_DAYS ({config.CHUNK_DAYS})."
  )

_N_CHUNKS = _TOTAL_STEPS // _STEPS_PER_CHUNK


def _select_output_variables(ds: xarray.Dataset) -> xarray.Dataset:
  wanted = config.PRESSURE_LEVEL_VARIABLES + config.SURFACE_VARIABLES
  missing = [v for v in wanted if v not in ds.data_vars]
  if missing:
    raise ValueError(
        f"Checkpoint {config.CHECKPOINT!r} decode() output is missing "
        f"{missing}. Available outputs: {sorted(ds.data_vars)}. Update "
        "config.SURFACE_VARIABLES / config.PRESSURE_LEVEL_VARIABLES to match."
    )
  ds = ds[wanted]
  level_vars = [v for v in config.PRESSURE_LEVEL_VARIABLES if "level" in ds[v].dims]
  ds = ds.sel(level=config.PRESSURE_LEVELS_HPA) if level_vars else ds
  return ds


def run_chunked_rollout(
    model: neuralgcm.PressureLevelModel,
    ic_ds: xarray.Dataset,
    forcing_ds: xarray.Dataset,
    rng_key: jax.Array,
) -> Iterator[xarray.Dataset]:
  """Yields one xarray.Dataset per CHUNK_DAYS-long piece of a LEAD_DAYS rollout.

  `ic_ds` is the single-timestep initial condition (see
  model_io.load_initial_conditions); `forcing_ds` must span
  [init_time, init_time + LEAD_DAYS] at daily resolution (see
  model_io.load_forcing_window), with its first timestep matching `ic_ds`'s.
  """
  inputs = model.inputs_from_xarray(ic_ds)
  forcings0 = model.forcings_from_xarray(forcing_ds.isel(time=0))
  state = model.encode(inputs, forcings0, rng_key)
  all_forcings = model.forcings_from_xarray(forcing_ds)

  timedelta = np.timedelta64(config.OUTPUT_TIMESTEP_HOURS, "h")

  for chunk_idx in range(_N_CHUNKS):
    state, outputs = model.unroll(
        state,
        all_forcings,
        steps=_STEPS_PER_CHUNK,
        timedelta=timedelta,
        start_with_input=False,
    )
    lead_hours = (
        np.arange(_STEPS_PER_CHUNK) + 1 + chunk_idx * _STEPS_PER_CHUNK
    ) * config.OUTPUT_TIMESTEP_HOURS
    chunk_ds = model.data_to_xarray(outputs, times=lead_hours)
    chunk_ds = chunk_ds.rename(time="lead_time")
    chunk_ds["lead_time"] = chunk_ds["lead_time"].astype("timedelta64[h]")
    yield _select_output_variables(chunk_ds)
