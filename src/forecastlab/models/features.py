"""Horizon-aware features for global ML forecasters, built straight from the wide sales matrix.

The single most important rule: a feature for target day t may only use sales up to day t - L, where L
(`min_lag`) is the smallest gap between the last observed day and the day being forecast.

    direct model, L = 28   one model forecasts every day 1..28 ahead; it never needs its own predictions
    recursive,    L = 1    lag-1 features; days 2..28 ahead are produced by feeding predictions back in

Known-future covariates (calendar, events, SNAP, planned prices) are taken AT day t: the retailer knows
them in advance. Sales-derived features are taken at t - L or earlier.

All windows are computed with cumulative sums, so a window sum is two lookups:
    sum(Y[:, e-k+1 .. e]) = C[:, e+1] - C[:, e+1-k]      where C[:, j] = sum(Y[:, :j])
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecastlab.data.load import LEVEL_COLS, Dataset

CATEGORICAL = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1",
               "event_name_2"]
ROLL_MEAN = [7, 14, 28, 56, 112]
ROLL_STD = [7, 28]


def lags_for(min_lag: int) -> list[int]:
    """Recent lags L..L+6 plus four same-weekday lags at or beyond L."""
    first_weekly = int(np.ceil(min_lag / 7) * 7)
    return sorted({min_lag + k for k in range(7)} | {first_weekly + 7 * j for j in range(4)})


def _calendar_frame(ds: Dataset) -> pd.DataFrame:
    cal = ds.cal
    c = pd.DataFrame({
        "wday": cal["wday"], "month": cal["month"], "mday": cal["date"].dt.day,
        "week": cal["date"].dt.isocalendar().week.astype(int).to_numpy(),
    })
    for col in ["event_name_1", "event_type_1", "event_name_2"]:
        c[col] = pd.Categorical(cal[col]).codes  # "" -> its own code
    is_event = (cal["event_name_1"] != "").to_numpy()
    idx = np.arange(len(cal))
    ev_idx = np.flatnonzero(is_event)
    nxt = np.searchsorted(ev_idx, idx)
    c["days_to_event"] = np.where(nxt < len(ev_idx), ev_idx[np.minimum(nxt, len(ev_idx) - 1)] - idx, 99).clip(0, 30)
    prv = np.searchsorted(ev_idx, idx, side="right") - 1
    c["days_since_event"] = np.where(prv >= 0, idx - ev_idx[np.maximum(prv, 0)], 99).clip(0, 30)
    return c.astype(np.int16)


_ITEM_PRICE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _item_price_mean(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Each item's average price across the stores that stock it, per day (cached per dataset)."""
    if id(ds) not in _ITEM_PRICE:
        codes = pd.Categorical(ds.series["item_id"]).codes
        on = ~np.isnan(ds.P)
        tot = pd.DataFrame(np.nan_to_num(ds.P)).groupby(codes).sum().to_numpy()
        cnt = pd.DataFrame(on.astype(np.float32)).groupby(codes).sum().to_numpy()
        _ITEM_PRICE[id(ds)] = (codes, (tot / np.maximum(cnt, 1)).astype(np.float32))
    return _ITEM_PRICE[id(ds)]


class FeatureBuilder:
    """Features for a subset of bottom series (`rows`, e.g. one store). Y can be replaced for recursion."""

    def __init__(self, ds: Dataset, rows: np.ndarray, Y: np.ndarray | None = None):
        self.ds, self.rows = ds, np.asarray(rows)
        self.cal = _calendar_frame(ds)
        meta = ds.series.iloc[self.rows]
        full = ds.series
        self.codes = {c: pd.Categorical(meta[c], categories=sorted(full[c].unique())).codes.astype(np.int16)
                      for c in LEVEL_COLS}
        snap = ds.cal[[f"snap_{s}" for s in ["CA", "TX", "WI"]]].to_numpy().T.astype(np.int8)  # [3, days]
        self.snap = snap[pd.Categorical(meta["state_id"], categories=["CA", "TX", "WI"]).codes]
        P = ds.P[self.rows].astype(np.float64)
        self.P = P
        self.release = np.where(~np.isnan(P).all(axis=1), np.argmax(~np.isnan(P), axis=1), P.shape[1])
        Pf = pd.DataFrame(P.T).ffill().to_numpy().T
        self.price_max = np.fmax.accumulate(np.nan_to_num(Pf, nan=-np.inf), axis=1)
        self.price_mean365 = pd.DataFrame(Pf.T).rolling(365, min_periods=1).mean().to_numpy().T
        item_codes, item_mean = _item_price_mean(ds)
        self.item_price_mean = item_mean[item_codes[self.rows]]
        self.group = pd.factorize(meta["store_id"] + "/" + meta["dept_id"])[0]
        self.set_sales(ds.Y[self.rows] if Y is None else Y)

    def set_sales(self, Y: np.ndarray) -> None:
        self.Y = np.asarray(Y, dtype=np.float64)
        n = len(self.Y)
        z = np.zeros((n, 1))
        self.C = np.concatenate([z, np.cumsum(self.Y, axis=1)], axis=1)
        self.C2 = np.concatenate([z, np.cumsum(self.Y ** 2, axis=1)], axis=1)
        self.CZ = np.concatenate([z, np.cumsum(self.Y == 0, axis=1)], axis=1)
        pos = np.where(self.Y > 0, np.arange(self.Y.shape[1]), -10_000)
        self.last_sale = np.maximum.accumulate(pos, axis=1)
        G = np.zeros((self.group.max() + 1, self.C.shape[1]))
        np.add.at(G, self.group, self.C)
        self.GC = G / np.bincount(self.group)[:, None]

    def _win(self, C: np.ndarray, end: np.ndarray, k: int) -> np.ndarray:
        return (C[:, end + 1] - C[:, np.maximum(end + 1 - k, 0)]) / k

    def frame(self, days: np.ndarray, min_lag: int) -> pd.DataFrame:
        """Feature rows for every (series, day) pair, series-major: row = i * len(days) + j."""
        days = np.asarray(days)
        n, nd = len(self.rows), len(days)
        end = days - min_lag                       # last sales day the feature may see
        f: dict[str, np.ndarray] = {}
        for c in LEVEL_COLS:
            f[c] = np.repeat(self.codes[c], nd)
        cal = self.cal.iloc[days]
        for c in cal.columns:
            f[c] = np.tile(cal[c].to_numpy(), n)
        f["snap"] = self.snap[:, days].ravel()

        p = self.P[:, days]
        f["price"] = p.ravel()
        f["price_vs_max"] = (p / self.price_max[:, days]).ravel()
        f["price_vs_365d_mean"] = (p / self.price_mean365[:, days]).ravel()
        f["price_change_7d"] = (p / self.P[:, np.maximum(days - 7, 0)]).ravel()
        f["price_vs_item_mean"] = (p / self.item_price_mean[:, days]).ravel()
        f["days_on_shelf"] = (days[None, :] - self.release[:, None]).ravel()

        Y = self.Y
        for lag in lags_for(min_lag):
            f[f"lag_{lag}"] = Y[:, days - lag].ravel()
        for k in ROLL_MEAN:
            f[f"mean_{k}"] = self._win(self.C, end, k).ravel()
        for k in ROLL_STD:
            m = self._win(self.C, end, k)
            f[f"std_{k}"] = np.sqrt(np.maximum(self._win(self.C2, end, k) - m ** 2, 0)).ravel()
        for k in [28, 112]:
            f[f"zero_share_{k}"] = self._win(self.CZ, end, k).ravel()
        f["days_since_sale"] = np.minimum(end[None, :] - self.last_sale[:, end], 365).ravel()
        first_weekly = int(np.ceil(min_lag / 7) * 7)
        for weeks in [4, 8]:
            acc = sum(Y[:, days - first_weekly - 7 * j] for j in range(weeks))
            f[f"same_wday_mean_{weeks}w"] = (acc / weeks).ravel()
        f["store_dept_mean_28"] = self._win(self.GC, end, 28)[self.group].ravel()
        f["mean_28_vs_store_dept"] = f["mean_28"] / np.maximum(f["store_dept_mean_28"], 1e-3)
        f["mean_7_vs_28"] = f["mean_7"] / np.maximum(f["mean_28"], 1e-3)
        return pd.DataFrame({k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in f.items()})

    def target(self, days: np.ndarray) -> np.ndarray:
        return self.Y[:, days].ravel().astype(np.float32)

    def on_shelf(self, days: np.ndarray) -> np.ndarray:
        return (~np.isnan(self.P[:, days])).ravel()


def assert_no_lookahead(fb: FeatureBuilder, days: np.ndarray, min_lag: int, seed: int = 0) -> None:
    """Leakage guard: scrambling every sales value after day t - min_lag must not change any feature of day t.

    Runs on the earliest target day; anything that changes is reading sales the forecaster cannot have seen.
    """
    days = np.asarray(days)
    t = int(days.min())
    before = fb.frame(np.array([t]), min_lag)
    Y0 = fb.Y.copy()
    Y1 = Y0.copy()
    rng = np.random.default_rng(seed)
    Y1[:, t - min_lag + 1:] = rng.poisson(5, size=Y1[:, t - min_lag + 1:].shape)
    fb.set_sales(Y1)
    try:
        after = fb.frame(np.array([t]), min_lag)
    finally:
        fb.set_sales(Y0)
    changed = [c for c in before.columns if not np.allclose(before[c].to_numpy(np.float64),
                                                            after[c].to_numpy(np.float64), equal_nan=True)]
    if changed:
        raise AssertionError(f"features read sales after t-{min_lag}: {changed}")


FEATURE_GROUPS = {
    "identity": LEVEL_COLS,
    "calendar & events": ["wday", "month", "mday", "week", "event_name_1", "event_type_1", "event_name_2",
                          "days_to_event", "days_since_event", "snap"],
    "price": ["price", "price_vs_max", "price_vs_365d_mean", "price_change_7d", "price_vs_item_mean", "days_on_shelf"],
}


def feature_group(name: str) -> str:
    for g, cols in FEATURE_GROUPS.items():
        if name in cols:
            return g
    return "sales history (lags, windows)"
