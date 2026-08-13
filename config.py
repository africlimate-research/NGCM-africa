"""Configuration for the NeuralGCM reforecast archive pipeline.

Produces repeated 45-day rollouts of the stochastic-precipitation NeuralGCM
checkpoint, initialized from ERA5 on a fixed set of calendar days each month,
and stores a fixed set of variables/pressure-levels into one Zarr store per
initialization year.
"""

CHECKPOINT_BUCKET = "gs://neuralgcm/models"
CHECKPOINT = "v1_precip/stochastic_precip_2_8_deg.pkl"

ERA5_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Initializations are at 00 UTC on these days of every month. A day that
# doesn't exist in a given month (e.g. day 31 in April, or day 31/29/28 in
# February) is clamped down to that month's last day instead, so every
# month always contributes exactly len(INIT_DAYS_OF_MONTH) inits.
INIT_HOUR = 0
INIT_DAYS_OF_MONTH = [1, 7, 9, 13, 17, 19, 25, 31]

LEAD_DAYS = 45
OUTPUT_TIMESTEP_HOURS = 24
CHUNK_DAYS = 9  # rollout is computed and flushed to zarr in chunks this long

PRESSURE_LEVELS_HPA = [1000, 850, 500, 200, 50]

# Variables carrying a `level` dimension, selected down to PRESSURE_LEVELS_HPA.
PRESSURE_LEVEL_VARIABLES = [
    "geopotential",
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
    "precipitation",
    "evaporation",
]

RNG_SEED = 42
