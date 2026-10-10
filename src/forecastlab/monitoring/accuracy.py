"""Forecast accuracy as actuals arrive: every past forecast that targeted yesterday is scored.

One night produces rows for each run origin in the last 28 days (horizon = days between origin and target), at
three scopes: all items (summed absolute item errors), per store (store totals), and company total.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecastlab.serving.db import ForecastDB


def score_target_day(db: ForecastDB, Y: np.ndarray, series: pd.DataFrame, dates: pd.DatetimeIndex,
                     target: int, horizon: int = 28) -> pd.DataFrame:
    runs = db.query("SELECT run_id, origin_date FROM runs WHERE status = 'SUCCESS'")
    actual = Y[:, target]
    stores = series["store_id"].to_numpy()
    out = []
    for run in runs.itertuples():
        o = int(dates.get_loc(pd.Timestamp(run.origin_date)))
        k = target - o
        if not 1 <= k <= horizon:
            continue
        items = db.read_items(run.run_id)
        if items is None:
            continue
        f, lo, hi = items["mean"][:, k - 1], items["p10"][:, k - 1], items["p90"][:, k - 1]
        base = {"target_date": str(dates[target].date()), "origin_date": run.origin_date, "horizon": k}
        out.append({**base, "scope": "items", "key": "all", "abs_error": float(np.abs(f - actual).sum()),
                    "actual": float(actual.sum()), "forecast": float(f.sum()),
                    "in_interval": float(np.mean((actual >= lo) & (actual <= hi))), "n": len(actual)})
        st = pd.DataFrame({"s": stores, "a": actual, "f": f}).groupby("s")[["a", "f"]].sum()
        for s, row in st.iterrows():
            out.append({**base, "scope": "store", "key": s, "abs_error": abs(row.f - row.a), "actual": row.a,
                        "forecast": row.f, "in_interval": np.nan, "n": 1})
        out.append({**base, "scope": "total", "key": "total", "abs_error": abs(f.sum() - actual.sum()),
                    "actual": float(actual.sum()), "forecast": float(f.sum()), "in_interval": np.nan, "n": 1})
    return pd.DataFrame(out)


def weekly_summary(db: ForecastDB, scope: str = "store") -> pd.DataFrame:
    df = db.query("SELECT * FROM accuracy WHERE scope = ?", (scope,))
    if df.empty:
        return df
    df["week"] = pd.to_datetime(df["target_date"]).dt.to_period("W-SUN").dt.start_time
    df["horizon_bucket"] = pd.cut(df["horizon"], [0, 7, 14, 21, 28], labels=["1-7", "8-14", "15-21", "22-28"])
    g = df.groupby(["week", "horizon_bucket"], observed=True)[["abs_error", "actual", "forecast"]].sum()
    g["wape"] = g["abs_error"] / g["actual"]
    g["bias"] = g["forecast"] / g["actual"] - 1
    return g.reset_index()
