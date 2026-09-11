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

  Runs config.ENSEMBLE_SIZE independent stochastic realizations together via
  jax.vmap (one rng_key per member, split from `rng_key`; `inputs`/
  `forcing_ds` are the same observed ERA5 data for every member -- only the
  injected noise differs). Each yielded chunk stacks all members along a
  "realization" dimension. model.data_to_xarray does host-side Dataset
  construction, not a pure array op, so it isn't itself vmapped -- it's
  called once per member on the already-computed (vmapped, then concrete)
  outputs, and the per-member Datasets are concatenated afterwards.
  """
  inputs = model.inputs_from_xarray(ic_ds)
  forcings0 = model.forcings_from_xarray(forcing_ds.isel(time=0))
  all_forcings = model.forcings_from_xarray(forcing_ds)

  member_keys = jax.random.split(rng_key, config.ENSEMBLE_SIZE)
  state = jax.vmap(lambda key: model.encode(inputs, forcings0, key))(member_keys)

  timedelta = np.timedelta64(config.OUTPUT_TIMESTEP_HOURS, "h")

  def unroll_member(member_state):
    return model.unroll(
        member_state,
        all_forcings,
        steps=_STEPS_PER_CHUNK,
        timedelta=timedelta,
        start_with_input=False,
    )

  for chunk_idx in range(_N_CHUNKS):
    state, outputs = jax.vmap(unroll_member)(state)
    lead_hours = (
        np.arange(_STEPS_PER_CHUNK) + 1 + chunk_idx * _STEPS_PER_CHUNK
    ) * config.OUTPUT_TIMESTEP_HOURS

    member_datasets = [
        model.data_to_xarray(jax.tree.map(lambda x: x[m], outputs), times=lead_hours)
        for m in range(config.ENSEMBLE_SIZE)
    ]
    chunk_ds = xarray.concat(member_datasets, dim="realization")
    chunk_ds = chunk_ds.assign_coords(realization=np.arange(config.ENSEMBLE_SIZE))
    chunk_ds = chunk_ds.rename(time="lead_time")
    chunk_ds["lead_time"] = chunk_ds["lead_time"].astype("timedelta64[h]")
    yield _select_output_variables(chunk_ds)
