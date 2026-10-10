"""A tiny file-based data lake that stands in for POS ingestion (Kafka / CDC / ETL in production).

    artifacts/lake/landing/sales/date=YYYY-MM-DD/pos.parquet     raw daily files as they "arrive"
    artifacts/lake/curated/sales.npy, prices.npy                 validated history the models read

`simulate_arrivals` replays real M5 days into the landing zone one file per day, which is exactly what the
nightly job (serving/nightly.py) consumes: land -> validate -> append to curated -> forecast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecastlab.config import artifacts_dir
from forecastlab.data.load import Dataset


def lake_dir() -> Path:
    d = artifacts_dir() / "lake"
    d.mkdir(parents=True, exist_ok=True)
    return d


def landing_path(date: pd.Timestamp) -> Path:
    return lake_dir() / "landing" / "sales" / f"date={date.date()}" / "pos.parquet"


def day_batch(ds: Dataset, day: int) -> pd.DataFrame:
    """One day of POS data in the shape a store system would send it (on-shelf series only)."""
    on = ~np.isnan(ds.P[:, day])
    s = ds.series.loc[on, ["store_id", "item_id"]]
    return pd.DataFrame({"date": ds.dates[day], "store_id": s["store_id"].to_numpy(), "item_id": s["item_id"].to_numpy(),
                         "units": ds.Y[on, day].astype(int), "sell_price": ds.P[on, day]})


def land(batch: pd.DataFrame, date: pd.Timestamp) -> Path:
    p = landing_path(date)
    p.parent.mkdir(parents=True, exist_ok=True)
    batch.to_parquet(p, index=False)
    return p


def read_landing(date: pd.Timestamp) -> pd.DataFrame | None:
    p = landing_path(date)
    return pd.read_parquet(p) if p.exists() else None


def simulate_arrivals(ds: Dataset, start: int, end: int) -> list[Path]:
    """Write days start..end (inclusive) into the landing zone."""
    return [land(day_batch(ds, d), ds.dates[d]) for d in range(start, end + 1)]


class Curated:
    """Validated sales history the nightly job forecasts from. Columns after `last_day` are unknown (zeros).

    Prices / calendar are master data (the planned price schedule is known in advance) and come from the dataset.
    """

    def __init__(self, Y: np.ndarray, last_day: int):
        self.Y, self.last_day = Y, last_day

    @staticmethod
    def path() -> Path:
        d = lake_dir() / "curated"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def initialise(cls, ds: Dataset, last_day: int) -> "Curated":
        Y = ds.Y.copy()
        Y[:, last_day + 1:] = 0
        c = cls(Y, last_day)
        c.save()
        return c

    @classmethod
    def load(cls) -> "Curated":
        p = cls.path()
        return cls(np.load(p / "sales.npy"), int((p / "last_day.txt").read_text()))

    def save(self) -> None:
        p = self.path()
        np.save(p / "sales.npy", self.Y)
        (p / "last_day.txt").write_text(str(self.last_day))

    def append(self, day: int, batch: pd.DataFrame, series: pd.DataFrame) -> None:
        if day != self.last_day + 1:
            raise ValueError(f"curated history ends at day {self.last_day}; cannot append day {day}")
        key = series[["item_id", "store_id"]].reset_index().rename(columns={"index": "row"})
        m = batch.merge(key, on=["item_id", "store_id"]).drop_duplicates(["item_id", "store_id"])
        self.Y[:, day] = 0
        self.Y[m["row"].to_numpy(), day] = m["units"].to_numpy(dtype=np.float32)
        self.last_day = day
        self.save()

    def view(self, ds: Dataset) -> Dataset:
        """A Dataset whose sales stop at last_day - what the forecaster is allowed to see."""
        return Dataset(self.Y, ds.P, ds.series, ds.cal)
