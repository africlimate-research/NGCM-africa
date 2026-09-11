"""Generates reforecast initialization times for a given year."""

import calendar

import numpy as np

from ngcm import config


def init_times_for_year(
    year: int, months: list[int] | None = None
) -> list[np.datetime64]:
  """Returns sorted 00Z init datetimes for `year`, per config.INIT_DAYS_OF_MONTH.

  Args:
    year: calendar year.
    months: which months (1-12) to include; defaults to all 12.
  """
  times = []
  for month in months if months is not None else range(1, 13):
    days_in_month = calendar.monthrange(year, month)[1]
    # Clamping day 31 (or 30/29 in February) down to days_in_month can
    # collide with a day already in the list -- e.g. with every day 1-31
    # configured, April's 31 clamps to 30, which is already scheduled. The
    # set() dedupes that down to one init instead of writing the same
    # calendar day twice.
    actual_days = {min(day, days_in_month) for day in config.INIT_DAYS_OF_MONTH}
    for actual_day in sorted(actual_days):
      times.append(
          np.datetime64(
              f"{year:04d}-{month:02d}-{actual_day:02d}T{config.INIT_HOUR:02d}:00"
          )
      )
  return sorted(times)
