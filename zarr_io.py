"""Creates and incrementally fills one Zarr store per reforecast year.

The store is sized upfront for every scheduled init_time x the full
lead_time range (NaN-filled, metadata-only write), so that each
(init_time, lead_time-chunk) result can be written independently as a
region write -- this is what gives the pipeline per-chunk flushing and
init-level resumability across a long-running SLURM job.
"""

from pathlib import Path

import dask.array as da
import numpy as np
import xarray

from ngcm import config

_TOTAL_STEPS = config.LEAD_DAYS * 24 // config.OUTPUT_TIMESTEP_HOURS
_STEPS_PER_CHUNK = config.CHUNK_DAYS * 24 // config.OUTPUT_TIMESTEP_HOURS


def year_store_path(output_dir: Path, year: int) -> Path:
  return Path(output_dir) / f"neuralgcm_reforecast_{year}.zarr"


def _full_lead_time() -> np.ndarray:
  hours = (np.arange(_TOTAL_STEPS) + 1) * config.OUTPUT_TIMESTEP_HOURS
  return hours.astype("timedelta64[h]")


def write_skeleton(
    store_path: Path,
    sample_chunk_ds: xarray.Dataset,
    all_init_times: list[np.datetime64],
) -> None:
  """Writes an empty, correctly-shaped/chunked zarr store (data left as NaN)."""
  lead_time = _full_lead_time()
  n_init = len(all_init_times)
  n_lead = len(lead_time)

  coords = {
      "init_time": np.array(all_init_times, dtype="datetime64[ns]"),
      "lead_time": lead_time,
  }
  for name in ("level", "latitude", "longitude"):
    if name in sample_chunk_ds.coords:
      coords[name] = sample_chunk_ds.coords[name].values

  data_vars = {}
  for name, var in sample_chunk_ds.data_vars.items():
    extra_dims = [d for d in var.dims if d not in ("init_time", "lead_time")]
    extra_sizes = tuple(sample_chunk_ds.sizes[d] for d in extra_dims)
    shape = (n_init, n_lead) + extra_sizes
    chunks = (1, _STEPS_PER_CHUNK) + extra_sizes
    dims = ("init_time", "lead_time") + tuple(extra_dims)
    data_vars[name] = (
        dims,
        da.full(shape, np.nan, dtype=np.float32, chunks=chunks),
    )

  skeleton = xarray.Dataset(data_vars, coords=coords)
  skeleton.to_zarr(store_path, mode="w", compute=False)


def write_region(store_path: Path, chunk_ds: xarray.Dataset) -> None:
  """Writes one (init_time, lead_time-chunk) slice into its pre-allocated region."""
  chunk_ds.to_zarr(store_path, region="auto")
