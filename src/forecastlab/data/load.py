"""Raw M5 CSVs -> compact arrays used by every stage.

    python -m forecastlab.data.load

M5 ships sales "wide" (one row per item x store, one column per day) and prices weekly.
We keep that wide layout as numpy matrices because almost every forecasting operation
(lags, rolling windows, naive forecasts, scaling) is a vectorised slice along the time axis:

    sales.npy   int16   [n_series, n_days]   units sold
    prices.npy  float32 [n_series, n_days]   sell price; NaN = item not on the shelf yet
    series.parquet      one row per series (id, item, dept, cat, store, state)
    calendar.parquet    one row per day (events, SNAP days, Walmart week)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from forecastlab.config import artifacts_dir, load_config, raw_dir

log = logging.getLogger(__name__)
LEVEL_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def data_dir():
    d = artifacts_dir() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build() -> None:
    raw = raw_dir()
    cal = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])
    cal["d"] = np.arange(len(cal))
    for c in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        cal[c] = cal[c].fillna("")
    cal = cal.drop(columns=["weekday"])

    tr = pd.read_csv(raw / "sales_train_evaluation.csv")           # d_1 .. d_1941
    te = pd.read_csv(raw / "sales_test_evaluation.csv")            # d_1942 .. d_1969 (released after the competition)
    assert (tr[["item_id", "store_id"]].to_numpy() == te[["item_id", "store_id"]].to_numpy()).all()
    day_cols = [c for c in tr.columns if c.startswith("d_")] + [c for c in te.columns if c.startswith("d_")]
    Y = np.concatenate([tr.filter(like="d_").to_numpy(), te.filter(like="d_").to_numpy()], axis=1)
    assert Y.shape[1] == len(day_cols) and Y.max() < np.iinfo(np.int16).max
    cal = cal.iloc[: Y.shape[1]].reset_index(drop=True)

    series = tr[LEVEL_COLS].copy()
    series.insert(0, "id", series["item_id"] + "_" + series["store_id"])

    # weekly prices -> daily matrix
    pr = pd.read_csv(raw / "sell_prices.csv")
    key = series[["item_id", "store_id"]].reset_index().rename(columns={"index": "row"})
    pr = pr.merge(key, on=["item_id", "store_id"])
    weeks = cal["wm_yr_wk"].to_numpy()
    week_codes, week_of_day = np.unique(weeks, return_inverse=True)
    W = np.full((len(series), len(week_codes)), np.nan, dtype=np.float32)
    W[pr["row"].to_numpy(), np.searchsorted(week_codes, pr["wm_yr_wk"].to_numpy())] = pr["sell_price"].to_numpy()
    P = W[:, week_of_day]

    out = data_dir()
    np.save(out / "sales.npy", Y.astype(np.int16))
    np.save(out / "prices.npy", P)
    series.to_parquet(out / "series.parquet", index=False)
    cal.to_parquet(out / "calendar.parquet", index=False)
    log.info("saved %d series x %d days (%s .. %s)", *Y.shape, cal["date"].min().date(), cal["date"].max().date())


@dataclass
class Dataset:
    Y: np.ndarray           # [n_series, n_days] float32 units sold
    P: np.ndarray           # [n_series, n_days] float32 price, NaN = not on shelf
    series: pd.DataFrame    # metadata, row i <-> Y[i]
    cal: pd.DataFrame       # calendar, row t <-> Y[:, t]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.cal["date"])

    def day(self, date: str | pd.Timestamp) -> int:
        """Column index of a calendar date."""
        return int(self.dates.get_loc(pd.Timestamp(date)))

    @property
    def on_shelf(self) -> np.ndarray:
        return ~np.isnan(self.P)


@lru_cache(maxsize=1)
def load_dataset() -> Dataset:
    d = data_dir()
    if not (d / "sales.npy").exists():
        build()
    return Dataset(np.load(d / "sales.npy").astype(np.float32), np.load(d / "prices.npy"),
                   pd.read_parquet(d / "series.parquet"), pd.read_parquet(d / "calendar.parquet"))


def first_sale_day(Y: np.ndarray) -> np.ndarray:
    """Index of each series' first non-zero sale (M5 treats leading zeros as 'not launched yet')."""
    nz = Y > 0
    return np.where(nz.any(axis=1), nz.argmax(axis=1), Y.shape[1])


def model_start_day(ds: Dataset) -> int:
    return ds.day(load_config()["data"]["model_start"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()


if __name__ == "__main__":
    main()
