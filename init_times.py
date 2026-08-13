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
    # config.INIT_DAYS_OF_MONTH's largest entry below 31 is 25, which is
    # always < the shortest possible days_in_month (28, for February), so
    # clamping day 31 down can never collide with an earlier scheduled day.
    for day in config.INIT_DAYS_OF_MONTH:
      actual_day = min(day, days_in_month)
      times.append(
          np.datetime64(
              f"{year:04d}-{month:02d}-{actual_day:02d}T{config.INIT_HOUR:02d}:00"
          )
      )
  return sorted(times)
