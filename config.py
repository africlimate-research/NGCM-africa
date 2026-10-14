"""Configuration for the NeuralGCM reforecast archive pipeline.

Produces repeated 45-day, ENSEMBLE_SIZE-member rollouts of the
stochastic-precipitation NeuralGCM checkpoint, initialized from ERA5 every
day of every month, and stores a fixed set of variables/pressure-levels
into one Zarr store per initialization year.
"""

CHECKPOINT_BUCKET = "gs://neuralgcm/models"
CHECKPOINT = "v1_precip/stochastic_precip_2_8_deg.pkl"

ERA5_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Initializations are at 00 UTC on these days of every month. A day that
# doesn't exist in a given month is clamped down to that month's last day
# instead (init_times.py dedupes the result), so a short month just gets
# fewer, not duplicate, inits.
INIT_HOUR = 0
INIT_DAYS_OF_MONTH = list(range(1, 32))

# Each init runs this many ensemble members (independent stochastic
# realizations, batched together via jax.vmap -- see reforecast.py), stored
# along a "realization" dimension. This multiplies both GPU memory per init
# and total storage by ENSEMBLE_SIZE compared to a single member: verify
# with a fresh --limit-inits 1 dry run that this still fits the GPU/--mem
# budget in slurm/run_month.sbatch before relying on the configured
# defaults there, which predate this setting.
ENSEMBLE_SIZE = 10

# How the SST / sea-ice forcing for each rollout is built:
#   "observed": read ERA5 at every valid date. Needs ERA5 through
#     init + LEAD_DAYS, and feeds the model the true future ocean state.
#   "anomaly_persistence": forcing(valid) = clim(valid) + [ERA5(init) -
#     clim(init)], i.e. the init-time anomaly is held fixed on top of a
#     day-of-year climatology (sea ice clipped to [0, 1]). Only needs ERA5
#     at init and uses no future ocean information. Requires the climatology
#     to be built first: `python -m ngcm.build_climatology` (see
#     slurm/build_climatology.sbatch).
# A zarr store records the mode it was created with, and run_month refuses to
# write into a store made with a different one.
FORCING_MODE = "anomaly_persistence"
CLIMATOLOGY_YEARS = (1991, 2020)

LEAD_DAYS = 45
OUTPUT_TIMESTEP_HOURS = 24
CHUNK_DAYS = 9  # rollout is computed and flushed to zarr in chunks this long

PRESSURE_LEVELS_HPA = [1000]

# Variables carrying a `level` dimension, selected down to PRESSURE_LEVELS_HPA.
PRESSURE_LEVEL_VARIABLES = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]

# Single-level diagnostic variables (no `level` dimension).
# NOTE: verify these are the exact names this checkpoint's decode() produces
# on your first test run (`run_month.py ... --limit-inits 1`) -- neuralgcm
# does not document decoder output names outside of the model's gin config,
# and the pipeline will raise a clear error listing what's actually
# available if either name is wrong.
SURFACE_VARIABLES = [
    "precipitation_cumulative_mean",
    "evaporation",
]

RNG_SEED = 42
