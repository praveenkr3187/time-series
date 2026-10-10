"""Minimal nightly scheduler for the Docker stack (stand-in for Airflow / cron / a cloud scheduler).

    python -m forecastlab.serving.scheduler --at 02:00          # run the nightly job every day at 02:00
    python -m forecastlab.serving.scheduler --demo-seconds 60   # demo: advance one simulated night per minute

The job is idempotent per run date (a re-run replaces that night's outputs), so a crashed or repeated trigger
is safe. In demo mode the scheduler also "receives" the next POS file into the landing zone.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time

import pandas as pd

from forecastlab.data.lake import Curated, day_batch, land
from forecastlab.data.load import load_dataset
from forecastlab.serving.nightly import NightlyJob

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default="02:00")
    ap.add_argument("--demo-seconds", type=int, default=0)
    a = ap.parse_args()
    ds = load_dataset()
    job = NightlyJob(ds)
    while True:
        if a.demo_seconds:
            cur = Curated.load()
            nxt = cur.last_day + 1
            if nxt + 1 >= len(ds.dates):
                log.info("end of data reached")
                return
            land(day_batch(ds, nxt), ds.dates[nxt])
            job.run(ds.dates[nxt + 1])
            time.sleep(a.demo_seconds)
        else:
            now = dt.datetime.now()
            target = dt.datetime.combine(now.date(), dt.time.fromisoformat(a.at))
            if target <= now:
                target += dt.timedelta(days=1)
            time.sleep((target - now).total_seconds())
            job.run(pd.Timestamp(dt.date.today()))


if __name__ == "__main__":
    main()
